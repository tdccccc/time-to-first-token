#!/usr/bin/env python3
"""Finite-workload benchmark for an OpenAI-compatible streaming endpoint.

Standard library only; the client needs no GPU. Start mock_sse.py for a
complete loopback-only smoke path, then run:

    python3 bench_openai.py --base-url http://127.0.0.1:8400 \
        --model mock --max-tokens 16 --concurrency 1 4 --requests 20

Formal finite scans need many more requests than the requested concurrency.
N < any C is rejected before HTTP. C is a worker/inflight cap, not a promise
of constant concurrency: startup and drain are included in the wall window.
Do not substitute C for a measured steady-state population in Little's law.

TTFT ends at the first nonempty delta.content chunk; empty role/usage chunks
never start it. ITL samples are observed content-chunk intervals, NOT invented
per-token timestamps. TPOT uses the first/last content span and reported token
count; one token or only one content chunk leaves it undefined (JSON null).
Missing usage uses an explicitly marked len(text)//4 estimate, not a tokenizer.
The fixed prompt and max_tokens cap do not guarantee 512 input tokens or a
fixed actual output length. HTTP, timeout and incomplete/error SSE fail.

JSON retains every request, its token source, timing and failure, in addition
to the existing summary fields. Default: 128 requests per level, C=1/8/32.
"""

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
from pathlib import Path
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

PROMPT = "Explain why the sky is blue in one short paragraph."
DEFAULT_MAX_TOKENS = 128
DEFAULT_REQUESTS = 128
MEASUREMENT_CONTRACT = {
    "ttft": "request start to first nonempty delta.content chunk (client observed)",
    "itl": "intervals between content chunks; chunk-level approximation, not token timestamps",
    "tpot": "(last_content_s-first_content_s)/(completion_tokens-1); null for single token or insufficient chunks",
    "total": "request start to complete [DONE] stream or failure detection, including HTTP and parsing",
    "throughput": "successful completion tokens / entire finite-workload wall window, including failures",
    "concurrency": "requested worker cap; observed client inflight includes startup/drain, not server batch size",
}


def percentile(values, pct):
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def dist(values, pcts=(50, 95, 99)):
    """Summary for a list of samples; None when there are no samples."""
    if not values:
        return {"mean": None, "max": None, "samples": 0,
                **{f"p{p}": None for p in pcts}}
    out = {
        "mean": round(statistics.mean(values), 4),
        "max": round(max(values), 4),
        "samples": len(values),
    }
    for p in pcts:
        out[f"p{p}"] = round(percentile(values, p), 4)
    return out


class StreamFailure(Exception):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


def sse_payloads(response):
    """Read complete SSE events; comments/metadata do not become chunks."""
    data = []
    for raw in response:
        try:
            line = raw.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError as exc:
            raise StreamFailure("invalid_sse", "invalid UTF-8 in SSE stream") from exc
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if field == "data":
            data.append(value[1:] if separator and value.startswith(" ") else value)
        # event/id/retry and unknown SSE fields are transport metadata.
    if data:
        raise StreamFailure("truncated_stream", "SSE event ended without its blank-line delimiter")


def checked_token_count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StreamFailure("invalid_usage", f"usage.{name} must be a nonnegative integer")
    return value


def validate_level(concurrency, requests, max_tokens, timeout):
    for name, value in (("concurrency", concurrency), ("requests", requests),
                        ("max_tokens", max_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if requests < concurrency:
        raise ValueError(f"requests (N={requests}) must be at least concurrency (C={concurrency})")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number")


def stream_request(base_url, model, max_tokens, timeout, *, prompt=PROMPT, request_id=None):
    """Record a complete request or a structured failure, retaining partial timing."""
    request_id = request_id or f"request-{uuid.uuid4().hex}"
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "X-Request-ID": request_id},
        method="POST",
    )
    start = time.perf_counter()
    content_times = []
    text_parts = []
    usage_tokens = prompt_tokens = None
    error = error_type = server_request_id = response_id = finish_reason = None
    http_status = None
    complete = False
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            http_status = getattr(resp, "status", 200)
            server_request_id = resp.getheader("X-Request-ID")
            content_type = resp.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "text/event-stream":
                raise StreamFailure("invalid_sse", "response Content-Type is not text/event-stream")
            for payload in sse_payloads(resp):
                if payload.strip() == "[DONE]":
                    complete = True
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise StreamFailure("invalid_sse", f"malformed SSE JSON: {exc}") from exc
                if not isinstance(chunk, dict):
                    raise StreamFailure("invalid_sse", "SSE JSON must be an object")
                if chunk.get("error") is not None:
                    raise StreamFailure("server_error", f"server error event: {chunk['error']}")
                if "choices" not in chunk and "usage" not in chunk:
                    raise StreamFailure("invalid_sse", "SSE JSON has neither choices nor usage")
                if isinstance(chunk.get("id"), str):
                    response_id = chunk["id"]
                choices = chunk.get("choices", [])
                if not isinstance(choices, list) or len(choices) > 1:
                    raise StreamFailure("invalid_sse", "choices must be a list with at most one choice (n=1)")
                for choice in choices:
                    if not isinstance(choice, dict) or not isinstance(choice.get("delta"), dict):
                        raise StreamFailure("invalid_sse", "streaming choice.delta must be an object")
                    text = choice["delta"].get("content")
                    if text is not None and not isinstance(text, str):
                        raise StreamFailure("invalid_sse", "delta.content must be a string or null")
                    if text:
                        content_times.append(time.perf_counter() - start)
                        text_parts.append(text)
                    if choice.get("finish_reason") is not None:
                        if not isinstance(choice["finish_reason"], str):
                            raise StreamFailure("invalid_sse", "finish_reason must be a string or null")
                        finish_reason = choice["finish_reason"]
                        if finish_reason == "error":
                            raise StreamFailure("server_error", "finish_reason=error")
                usage = chunk.get("usage")
                if usage is not None:
                    if not isinstance(usage, dict):
                        raise StreamFailure("invalid_usage", "usage must be an object or null")
                    if usage.get("completion_tokens") is not None:
                        usage_tokens = checked_token_count(usage["completion_tokens"], "completion_tokens")
                    if usage.get("prompt_tokens") is not None:
                        prompt_tokens = checked_token_count(usage["prompt_tokens"], "prompt_tokens")
    except StreamFailure as exc:
        error_type, error = exc.kind, str(exc)
    except urllib.error.HTTPError as exc:
        http_status = exc.code
        error_type, error = "http_error", f"HTTP {exc.code}: {exc.reason}"
        exc.close()
    except urllib.error.URLError as exc:
        error_type = "timeout" if isinstance(exc.reason, TimeoutError) else "network_error"
        error = str(exc)
    except TimeoutError as exc:
        error_type, error = "timeout", str(exc) or "request timed out"
    except http.client.IncompleteRead as exc:
        error_type, error = "truncated_stream", str(exc)
    except (OSError, http.client.HTTPException, ValueError) as exc:
        error_type, error = "network_error", str(exc)
    total = time.perf_counter() - start
    if error is None:
        if not complete:
            error_type, error = "truncated_stream", "stream ended before [DONE]"
        elif not content_times:
            error_type, error = "empty_response", "no nonempty delta.content received"
        elif usage_tokens == 0:
            error_type, error = "invalid_usage", "nonempty content cannot have zero completion_tokens"
    text = "".join(text_parts)
    estimated = usage_tokens is None and bool(text_parts)
    tokens = usage_tokens if usage_tokens is not None else (max(1, len(text) // 4) if text_parts else None)
    token_source = "usage.completion_tokens" if usage_tokens is not None else (
        "estimated_characters_div4" if estimated else "unavailable")
    first_content = content_times[0] if content_times else None
    last_content = content_times[-1] if content_times else None
    intervals = [right - left for left, right in zip(content_times, content_times[1:])]
    tpot = None
    tpot_reason = "failed_request" if error is not None else None
    if error is None:
        if tokens <= 1:
            tpot_reason = "single_token"
        elif len(content_times) < 2:
            tpot_reason = "insufficient_content_chunks"
        else:
            tpot = (last_content - first_content) / (tokens - 1)
    return {
        "request_id": request_id, "server_request_id": server_request_id,
        "response_id": response_id, "http_status": http_status,
        "ok": error is None, "error": error, "error_type": error_type,
        "stream_complete": complete, "finish_reason": finish_reason,
        "ttft_s": first_content, "first_content_s": first_content,
        "last_content_s": last_content, "total_s": total,
        "content_times_s": content_times, "content_chunks": len(content_times),
        "itl_samples": intervals, "tpot_s": tpot,
        "tpot_unavailable_reason": tpot_reason,
        "completion_tokens": tokens, "completion_tokens_source": token_source,
        "tokens_estimated": estimated, "prompt_tokens": prompt_tokens,
        "output_chars": len(text),
    }


def sweep_level(base_url, model, max_tokens, concurrency, requests, timeout, *,
                prompt=PROMPT, test_id=None, conditions_id=None):
    """Finite workload with a requested cap, measured inflight and raw records."""
    validate_level(concurrency, requests, max_tokens, timeout)
    test_id = test_id or f"bench-{uuid.uuid4().hex[:12]}-c{concurrency}"
    conditions_id = conditions_id or "client-" + hashlib.sha256(json.dumps(
        [base_url, model, max_tokens, prompt, concurrency, requests, timeout],
        ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    wall_start = time.perf_counter()
    lock = threading.Lock()
    active = peak = 0
    inflight_events = []

    def execute(index):
        nonlocal active, peak
        request_id = f"{test_id}-q{index + 1:06d}"
        with lock:
            started = time.perf_counter() - wall_start
            active += 1
            peak = max(peak, active)
            inflight_events.append({"at_s": started, "inflight": active})
        try:
            row = stream_request(base_url, model, max_tokens, timeout,
                                 prompt=prompt, request_id=request_id)
        except Exception as exc:
            row = {"ok": False, "error": f"client exception: {exc}",
                   "error_type": "client_exception", "stream_complete": False,
                   "total_s": time.perf_counter() - wall_start - started,
                   "ttft_s": None, "first_content_s": None, "last_content_s": None,
                   "content_times_s": [], "content_chunks": 0, "itl_samples": [],
                   "tpot_s": None, "tpot_unavailable_reason": "failed_request",
                   "completion_tokens": None, "completion_tokens_source": "unavailable",
                   "prompt_tokens": None, "tokens_estimated": False}
        finally:
            with lock:
                finished = time.perf_counter() - wall_start
                active -= 1
                inflight_events.append({"at_s": finished, "inflight": active})
        row.update(request_id=request_id, request_index=index + 1,
                   test_id=test_id, conditions_id=conditions_id,
                   started_at_s=started, finished_at_s=finished)
        return row

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(execute, index) for index in range(requests)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    wall = time.perf_counter() - wall_start
    results.sort(key=lambda row: row["request_index"])
    area = previous = 0.0
    previous_active = 0
    for event in inflight_events:
        area += previous_active * (event["at_s"] - previous)
        previous, previous_active = event["at_s"], event["inflight"]
    ok = [row for row in results if row["ok"]]
    total_tokens = sum(row["completion_tokens"] for row in ok)
    undefined_tpot = sum(row["tpot_s"] is None for row in ok)
    result = {
        "test_id": test_id, "conditions_id": conditions_id,
        "concurrency": concurrency, "requested_concurrency": concurrency,
        "max_possible_inflight": min(concurrency, requests),
        "observed_peak_inflight": peak,
        "observed_mean_inflight": round(area / wall, 6),
        "inflight_events": inflight_events,
        "load_model": "finite_workload", "steady_state_verified": False,
        "ok": len(ok), "requests": requests, "failed": requests - len(ok),
        "timeouts": sum(row.get("error_type") == "timeout" for row in results),
        "wall_s": round(wall, 6),
        "ttft_s": dist([row["ttft_s"] for row in ok]),
        "itl_s": dist([value for row in ok for value in row["itl_samples"]]),
        "tpot_s": dist([row["tpot_s"] for row in ok if row["tpot_s"] is not None], (50, 95)),
        "tpot_undefined_requests": undefined_tpot,
        "total_s": dist([row["total_s"] for row in ok]),
        "all_request_total_s": dist([row["total_s"] for row in results]),
        "completion_tokens": dist([row["completion_tokens"] for row in ok]),
        "prompt_tokens": dist([row["prompt_tokens"] for row in ok if row.get("prompt_tokens") is not None]),
        "total_completion_tokens": total_tokens,
        "throughput_tok_per_s": round(total_tokens / wall, 1),
        "tokens_estimated": any(row["tokens_estimated"] for row in ok),
        "estimated_requests": sum(row["tokens_estimated"] for row in ok),
        "request_records": results,
        "warnings": ["finite_workload includes startup/drain; requested C is not a steady-state population"],
    }
    if result["tokens_estimated"]:
        result["warnings"].append("completion tokens estimated from characters; throughput/TPOT are approximate")
    if not ok:
        result["error"] = results[0]["error"] or "all requests failed"
    return result



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True, help="server root, e.g. http://127.0.0.1:8400 (without /v1)")
    parser.add_argument("--model", required=True, help="model name as served")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="output cap, not guaranteed actual length (default: 128)")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32], help="requested inflight caps (default: 1 8 32)")
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS, help="per level; must be >= every concurrency (default: 128)")
    parser.add_argument("--timeout", type=float, default=300, help="socket-operation timeout in seconds, not total request deadline (default: 300)")
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt", help="fixed literal prompt; default is the short sky-is-blue prompt")
    prompts.add_argument("--prompt-file", type=Path, help="UTF-8 fixed prompt file; exact text is saved in JSON")
    parser.add_argument("--test-id", help="run ID; auto-generated if omitted (ASCII without whitespace)")
    parser.add_argument("--conditions-id", help="ID of your environment/conditions record; default hashes client settings only")
    parser.add_argument("--output", default="bench-results.json", help="JSON containing summaries AND request_records")
    args = parser.parse_args(argv)
    try:
        for level in args.concurrency:
            validate_level(level, args.requests, args.max_tokens, args.timeout)
    except ValueError as exc:
        parser.error(str(exc))
    url = urllib.parse.urlsplit(args.base_url)
    if url.scheme not in ("http", "https") or not url.hostname:
        parser.error("--base-url must be an HTTP(S) server root")
    for name in ("test_id", "conditions_id"):
        value = getattr(args, name)
        if value is not None and (not value or any(not c.isascii() or c.isspace() or ord(c) < 33 for c in value)):
            parser.error(f"--{name.replace('_', '-')} must be a nonempty ASCII ID without whitespace")
    try:
        prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else (
            args.prompt if args.prompt is not None else PROMPT)
    except (OSError, UnicodeError) as exc:
        parser.error(f"cannot read --prompt-file as UTF-8: {exc}")
    if not prompt.strip():
        parser.error("prompt must contain non-whitespace text")
    prompt_source = "file" if args.prompt_file else ("inline" if args.prompt is not None else "default")
    created_at = datetime.now(timezone.utc).isoformat()
    test_id = args.test_id or f"bench-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    conditions = {
        "base_url": args.base_url, "model": args.model, "max_tokens": args.max_tokens,
        "prompt_sha256": prompt_sha256, "requested_concurrency": args.concurrency,
        "requests_per_level": args.requests, "timeout_s": args.timeout,
        "temperature": 0, "stream_options": {"include_usage": True},
        "script_sha256": script_sha256,
    }
    conditions_id = args.conditions_id or "client-" + hashlib.sha256(json.dumps(
        conditions, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    print("finite workload: requested C is a cap, startup/drain are included; not a steady-state test.", file=sys.stderr)
    levels = []
    for index, level in enumerate(args.concurrency, 1):
        print(f"requested_concurrency={level}, requests={args.requests} ...", file=sys.stderr)
        levels.append(sweep_level(
            args.base_url, args.model, args.max_tokens, level, args.requests, args.timeout,
            prompt=prompt, test_id=f"{test_id}-c{level}-l{index}", conditions_id=conditions_id,
        ))
    document = {
        "schema_version": 2, "created_at": created_at,
        "test_id": test_id, "conditions_id": conditions_id,
        "conditions_scope": "client settings; separately attach hardware/engine/version evidence",
        "conditions": conditions, "raw_path": args.output,
        "base_url": args.base_url, "model": args.model, "max_tokens": args.max_tokens,
        "prompt": prompt, "prompt_source": prompt_source,
        "prompt_file": str(args.prompt_file) if args.prompt_file else None,
        "prompt_sha256": prompt_sha256, "script_sha256": script_sha256,
        "requested_concurrency": args.concurrency, "requests_per_level": args.requests,
        "timeout_s": args.timeout, "load_model": "finite_workload",
        "measurement_contract": MEASUREMENT_CONTRACT,
        "argv": list(argv) if argv is not None else sys.argv[1:], "levels": levels,
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2, ensure_ascii=False, allow_nan=False)
        fh.write("\n")

    def fmt(value):
        return "n/a" if value is None else value

    print("\n| 请求并发 / 观测峰值 | 成功 | TTFT p50 / p95 (s) | ITL chunk p50 / p95 (s) | TPOT p50 (s) | 吞吐 (tok/s) |")
    print("|---|---|---|---|---|---|")
    for level in levels:
        cap = f"{level['requested_concurrency']} / {level['observed_peak_inflight']}"
        if not level["ok"]:
            print(f"| {cap} | 0/{level['requests']} | 失败：{level.get('error')} | | | |")
            continue
        print(
            f"| {cap} | {level['ok']}/{level['requests']} | "
            f"{fmt(level['ttft_s']['p50'])} / {fmt(level['ttft_s']['p95'])} | "
            f"{fmt(level['itl_s']['p50'])} / {fmt(level['itl_s']['p95'])} | "
            f"{fmt(level['tpot_s']['p50'])} | {level['throughput_tok_per_s']} |"
        )
        if level["tokens_estimated"]:
            print("WARNING: estimated completion tokens; throughput and TPOT are approximate.", file=sys.stderr)
    print(f"\n完整结果（含逐请求记录）见 {args.output}。")
    return 1 if any(level["ok"] != level["requests"] for level in levels) else 0


if __name__ == "__main__":
    raise SystemExit(main())
