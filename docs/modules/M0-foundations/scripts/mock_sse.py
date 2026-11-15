#!/usr/bin/env python3
"""Loopback-only, standard-library SSE fixture, NOT an LLM performance server.

Run: python3 mock_sse.py --port 8400
Then: python3 bench_openai.py --base-url http://127.0.0.1:8400 --model mock \
          --max-tokens 16 --concurrency 1 4 --requests 20

Each content chunk is one synthetic "x" token. Role/finish/usage chunks are
not content. --prompt-tokens is an optional synthetic fixture count, not a
measurement by a real tokenizer. Port 0 asks the OS for a temporary port.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import time
import uuid


def make_server(port=8400, *, tokens=128, interval=0.010,
                first_content_delay=0.020, prompt_tokens=None, include_usage=True):
    """Create (but do not start) the loopback fixture; useful for unittest."""
    if not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    if not isinstance(tokens, int) or tokens <= 0:
        raise ValueError("tokens must be a positive integer")
    for name, value in (("interval", interval), ("first_content_delay", first_content_delay)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if prompt_tokens is not None and (isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens < 0):
        raise ValueError("prompt_tokens must be a nonnegative synthetic count")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path != "/health":
                self.send_error(404, "use /health or POST /v1/chat/completions")
                return
            body = json.dumps({"status": "mock", "llm": False, "gpu": False}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404, "only /v1/chat/completions is supported")
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(size).decode("utf-8"))
                if not isinstance(request, dict) or request.get("stream") is not True:
                    raise ValueError("stream=true is required")
                cap = request.get("max_tokens", tokens)
                if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
                    raise ValueError("max_tokens must be positive")
                if not isinstance(request.get("messages"), list):
                    raise ValueError("messages must be a list")
            except (ValueError, UnicodeError) as exc:
                self.send_error(400, str(exc))
                return
            count = min(tokens, cap)
            response_id = f"chatcmpl-mock-{uuid.uuid4().hex}"
            request_id = self.headers.get("X-Request-ID", f"mock-{uuid.uuid4().hex}")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Connection", "close")
            self.send_header("X-Request-ID", request_id)
            self.send_header("X-Mock-SSE", "stdlib-no-llm")
            self.end_headers()
            self.close_connection = True

            def event(payload):
                data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
                self.wfile.write(("data: " + data + "\n\n").encode("utf-8"))
                self.wfile.flush()

            def choice(delta, finish_reason=None):
                return {"id": response_id, "object": "chat.completion.chunk",
                        "model": request.get("model", "mock"),
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}

            try:
                event(choice({"role": "assistant", "content": ""}))
                if first_content_delay:
                    time.sleep(first_content_delay)
                for index in range(count):
                    if index and interval:
                        time.sleep(interval)
                    event(choice({"content": "x"}))
                event(choice({}, "length" if count == cap else "stop"))
                if include_usage:
                    usage = {"completion_tokens": count}
                    if prompt_tokens is not None:
                        usage["prompt_tokens"] = prompt_tokens
                        usage["total_tokens"] = prompt_tokens + count
                    event({"id": response_id, "choices": [], "usage": usage})
                event("[DONE]")
            except (BrokenPipeError, ConnectionResetError):
                # A timed-out client may close during the fixture's delay.
                pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8400, help="loopback port; 0 chooses a temporary port (default: 8400)")
    parser.add_argument("--tokens", type=int, default=128, help="maximum synthetic output tokens, also capped by max_tokens")
    parser.add_argument("--interval", type=float, default=0.010, help="seconds between content chunks (default: 0.010)")
    parser.add_argument("--first-content-delay", type=float, default=0.020, help="delay after the empty role chunk, in seconds (default: 0.020)")
    parser.add_argument("--prompt-tokens", type=int, help="optional synthetic prompt count; not a real tokenizer measurement")
    parser.add_argument("--no-usage", action="store_true", help="omit usage to exercise the client's explicit token-estimation path")
    args = parser.parse_args(argv)
    try:
        server = make_server(args.port, tokens=args.tokens, interval=args.interval,
                             first_content_delay=args.first_content_delay,
                             prompt_tokens=args.prompt_tokens, include_usage=not args.no_usage)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Mock SSE listening on http://127.0.0.1:{server.server_port} (synthetic fixture; no LLM/GPU)", flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
