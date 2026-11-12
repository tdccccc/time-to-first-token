#!/usr/bin/env python3
"""Minimal reproducible benchmark harness for an OpenAI-compatible endpoint.

Usage (against a running vLLM / SGLang / llama.cpp server):

    python3 bench_openai.py --base-url http://localhost:8000 \
        --model Qwen2.5-7B-Instruct --concurrency 1 8 32 --requests 20

What it measures per request (streaming, ``/v1/chat/completions``):
  * TTFT — time to first streamed chunk (first token time).
  * total latency — time to the final streamed chunk.
  * ITL — inter-token latency, (total - TTFT) / (completion_tokens - 1).
  * completion tokens — read from the final chunk ``usage`` field when the
    server supports ``stream_options.include_usage`` (vLLM does); otherwise
    estimated as ``len(text) / 4`` characters and marked as an estimate.

Output: one JSON file with every request plus a Markdown summary table
printed to stdout. Only the Python standard library is used, so the script
runs anywhere Python 3.10+ exists. It needs no GPU itself — the GPU lives
on the server side.

The script deliberately fixes the prompt and the output length: without
fixed lengths, KV-cache size (and therefore throughput) changes between
runs and the numbers stop being comparable. See M0 session 2 for why.
"""

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

PROMPT = "Explain why the sky is blue in one short paragraph."
DEFAULT_MAX_TOKENS = 128


def percentile(values, pct):
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def stream_request(base_url, model, max_tokens, timeout):
    """One streamed chat completion. Returns a dict of measurements."""
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    ttft = None
    text_parts = []
    usage_tokens = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - start
                for choice in chunk.get("choices", []):
                    text_parts.append(choice.get("delta", {}).get("content") or "")
                usage = chunk.get("usage") or {}
                if usage.get("completion_tokens"):
                    usage_tokens = usage["completion_tokens"]
    except urllib.error.URLError as exc:
        return {"ok": False, "error": str(exc)}
    if ttft is None:
        # 200 OK but no streamed chunk at all (e.g. only "[DONE]"): this is
        # not a measurable request. Report it as failed instead of letting
        # a None TTFT crash the statistics below.
        return {"ok": False, "error": "no streamed content received (empty response)"}
    total = time.perf_counter() - start
    text = "".join(text_parts)
    estimated = usage_tokens is None
    tokens = usage_tokens if usage_tokens else max(1, len(text) // 4)
    itl = (total - ttft) / max(tokens - 1, 1) if ttft is not None else float("nan")
    return {
        "ok": True,
        "ttft_s": ttft,
        "total_s": total,
        "itl_s": itl,
        "completion_tokens": tokens,
        "tokens_estimated": estimated,
    }


def sweep_level(base_url, model, max_tokens, concurrency, requests, timeout):
    """Run `requests` completions at a fixed concurrency level."""
    wall_start = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(stream_request, base_url, model, max_tokens, timeout)
            for _ in range(requests)
        ]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
    wall = time.perf_counter() - wall_start
    ok = [r for r in results if r["ok"]]
    if not ok:
        return {
            "concurrency": concurrency,
            "ok": 0,
            "requests": requests,
            "error": results[0].get("error", "all requests failed")
            if results
            else "no requests were run",
        }
    ttfts = [r["ttft_s"] for r in ok]
    totals = [r["total_s"] for r in ok]
    itls = [r["itl_s"] for r in ok]
    total_tokens = sum(r["completion_tokens"] for r in ok)
    return {
        "concurrency": concurrency,
        "ok": len(ok),
        "requests": requests,
        "wall_s": round(wall, 2),
        "ttft_s": {
            "mean": round(statistics.mean(ttfts), 4),
            "p50": round(percentile(ttfts, 50), 4),
            "p95": round(percentile(ttfts, 95), 4),
            "p99": round(percentile(ttfts, 99), 4),
        },
        "itl_s": {
            "mean": round(statistics.mean(itls), 4),
            "p50": round(percentile(itls, 50), 4),
            "p95": round(percentile(itls, 95), 4),
        },
        "throughput_tok_per_s": round(total_tokens / wall, 1),
        "tokens_estimated": any(r["tokens_estimated"] for r in ok),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="e.g. http://localhost:8000")
    parser.add_argument("--model", required=True, help="model name as served")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--requests", type=int, default=20, help="per level")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", default="bench-results.json")
    args = parser.parse_args()
    if args.requests <= 0:
        parser.error("--requests must be greater than 0")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be greater than 0")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")
    if any(level <= 0 for level in args.concurrency):
        parser.error("--concurrency values must be greater than 0")

    levels = []
    for level in args.concurrency:
        print(f"concurrency={level} ...", file=sys.stderr)
        levels.append(
            sweep_level(
                args.base_url, args.model, args.max_tokens,
                level, args.requests, args.timeout,
            )
        )
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "base_url": args.base_url,
                "model": args.model,
                "max_tokens": args.max_tokens,
                "prompt": PROMPT,
                "levels": levels,
            },
            fh, indent=2, ensure_ascii=False,
        )

    print("\n| 并发 | 成功 | TTFT p50 / p95 (s) | ITL p50 (s) | 吞吐 (tok/s) |")
    print("|---|---|---|---|---|")
    for lv in levels:
        if "ttft_s" not in lv:
            print(f"| {lv['concurrency']} | 0/{lv['requests']} | 失败：{lv.get('error')} | | |")
            continue
        print(
            f"| {lv['concurrency']} | {lv['ok']}/{lv['requests']} | "
            f"{lv['ttft_s']['p50']} / {lv['ttft_s']['p95']} | "
            f"{lv['itl_s']['p50']} | {lv['throughput_tok_per_s']} |"
        )
    print(f"\n完整结果见 {args.output}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
