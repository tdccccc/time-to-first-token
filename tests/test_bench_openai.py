"""Standard-library regressions for the finite OpenAI/SSE benchmark."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "docs/modules/M0-foundations/scripts"
BENCH_PATH = SCRIPTS / "bench_openai.py"
MOCK_PATH = SCRIPTS / "mock_sse.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load_module("bench_openai_test", BENCH_PATH)


def sse(payload):
    if not isinstance(payload, bytes):
        payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return b"data: " + payload + b"\n\n"


def content(text, **extra):
    return {"id": "chatcmpl-test", "choices": [{"index": 0,
            "delta": {"content": text}, "finish_reason": None}], **extra}


class FakeResponse:
    """A file-like SSE boundary with a deterministic request-relative clock."""
    status = 200
    headers = {"Content-Type": "text/event-stream; charset=utf-8",
               "X-Request-ID": "server-request-test"}

    def __init__(self, clock, events, end_time=0.10):
        self.clock = clock
        self.events = events
        self.end_time = end_time

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.clock[0] = self.end_time

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def __iter__(self):
        for at, event in self.events:
            self.clock[0] = at
            if isinstance(event, BaseException):
                raise event
            yield from event.splitlines(keepends=True)


def raw_record():
    return {"ok": True, "error": None, "error_type": None,
            "ttft_s": 0.01, "first_content_s": 0.01,
            "last_content_s": 0.03, "total_s": 0.04,
            "content_times_s": [0.01, 0.03], "itl_samples": [0.02],
            "tpot_s": 0.01, "completion_tokens": 3,
            "completion_tokens_source": "usage.completion_tokens",
            "prompt_tokens": 11, "tokens_estimated": False,
            "content_chunks": 2, "stream_complete": True}


class BenchmarkValidationTests(unittest.TestCase):
    def test_sweep_rejects_insufficient_requests_before_network(self):
        with mock.patch.object(bench, "stream_request", return_value=raw_record()) as request:
            with self.assertRaisesRegex(ValueError, "requests.*concurrency"):
                bench.sweep_level("http://127.0.0.1:1", "mock", 3, 4, 2, 1)
            request.assert_not_called()

    def test_sweep_retains_request_records_and_totals(self):
        with mock.patch.object(bench, "stream_request", side_effect=lambda *a, **k: raw_record()):
            result = bench.sweep_level("unused", "mock", 3, 2, 4, 1)
        self.assertIn("request_records", result)
        rows = result["request_records"]
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({row["request_id"] for row in rows}), 4)
        self.assertEqual(result["total_completion_tokens"], 12)
        self.assertEqual(result["completion_tokens"]["samples"], 4)
        self.assertEqual(result["completion_tokens"]["mean"], 3)
        self.assertEqual(result["total_s"]["samples"], 4)
        self.assertEqual(result["total_s"]["mean"], 0.04)
        self.assertEqual(result["prompt_tokens"]["mean"], 11)
        self.assertEqual(result["requested_concurrency"], 2)
        self.assertEqual(result["concurrency"], 2)  # legacy summary field
        self.assertEqual(result["max_possible_inflight"], 2)
        self.assertEqual(result["load_model"], "finite_workload")
        self.assertFalse(result["steady_state_verified"])
        for row in rows:
            self.assertLessEqual(row["started_at_s"], row["finished_at_s"])
            self.assertEqual(row["completion_tokens_source"], "usage.completion_tokens")
        json.dumps(result, allow_nan=False)

    def test_observed_inflight_is_bounded_and_measured(self):
        gate = threading.Barrier(2, timeout=3)
        def request(*args, **kwargs):
            gate.wait()
            return raw_record()
        with mock.patch.object(bench, "stream_request", side_effect=request):
            result = bench.sweep_level("unused", "mock", 3, 2, 4, 1)
        self.assertEqual(result["observed_peak_inflight"], 2)
        self.assertGreater(result["observed_mean_inflight"], 0)
        self.assertLessEqual(result["observed_mean_inflight"], 2)

    def test_all_failed_requests_are_saved_with_finite_summaries(self):
        failed = {**raw_record(), "ok": False, "error": "read timed out",
                  "error_type": "timeout", "stream_complete": False}
        with mock.patch.object(bench, "stream_request", side_effect=lambda *a, **k: dict(failed)):
            result = bench.sweep_level("unused", "mock", 3, 2, 2, 1)
        self.assertEqual(result["ok"], 0)
        self.assertEqual(result["failed"], 2)
        self.assertEqual(result["timeouts"], 2)
        self.assertEqual(len(result["request_records"]), 2)
        self.assertEqual(result["total_completion_tokens"], 0)
        self.assertEqual(result["throughput_tok_per_s"], 0)
        self.assertEqual(result["total_s"]["samples"], 0)
        self.assertEqual(result["all_request_total_s"]["samples"], 2)
        self.assertIsNone(result["ttft_s"]["p99"])
        self.assertIn("timed out", result["error"])
        json.dumps(result, allow_nan=False)

    def test_undefined_tpot_is_excluded_not_reported_as_zero(self):
        one = {**raw_record(), "completion_tokens": 1, "tpot_s": None,
               "tpot_unavailable_reason": "single_token", "content_chunks": 1,
               "content_times_s": [0.01], "itl_samples": []}
        with mock.patch.object(bench, "stream_request", side_effect=lambda *a, **k: dict(one)):
            result = bench.sweep_level("unused", "mock", 1, 1, 2, 1)
        self.assertEqual(result["tpot_s"]["samples"], 0)
        self.assertIsNone(result["tpot_s"]["mean"])
        self.assertIsNone(result["itl_s"]["p50"])
        self.assertEqual(result["tpot_undefined_requests"], 2)
        json.dumps(result, allow_nan=False)

    def test_cli_rejects_every_invalid_level_before_output(self):
        with tempfile.TemporaryDirectory(prefix="ttft-bench-tests-") as tmp:
            output = Path(tmp) / "invalid.json"
            run = subprocess.run([sys.executable, "-B", str(BENCH_PATH),
                "--base-url", "http://127.0.0.1:1", "--model", "mock",
                "--concurrency", "1", "4", "--requests", "2",
                "--output", str(output)], capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 2, run.stderr)
            self.assertIn("requests", run.stderr)
            self.assertIn("concurrency", run.stderr)
            self.assertFalse(output.exists())

    def test_default_request_count_is_128_and_covers_default_levels(self):
        captured = []
        def sweep(*args, **kwargs):
            c, n = args[3:5]
            captured.append((c, n))
            return {"concurrency": c, "requested_concurrency": c,
                "observed_peak_inflight": 1, "ok": 0, "requests": n,
                "failed": n, "error": "controlled default-argument check"}
        with tempfile.TemporaryDirectory(prefix="ttft-bench-tests-") as tmp:
            argv = [str(BENCH_PATH), "--base-url", "http://127.0.0.1:1",
                    "--model", "mock", "--output", str(Path(tmp) / "default.json")]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(bench, "sweep_level", side_effect=sweep), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                bench.main()
        self.assertEqual([c for c, _ in captured], [1, 8, 32])
        self.assertTrue(all(n == 128 and n >= c for c, n in captured), captured)


class StreamingBoundaryTests(unittest.TestCase):
    def measure(self, events, end_time=0.10):
        clock = [0.0]
        response = FakeResponse(clock, events, end_time)
        with mock.patch.object(bench.time, "perf_counter", side_effect=lambda: clock[0]), mock.patch.object(bench.urllib.request, "urlopen", return_value=response) as opened:
            result = bench.stream_request("http://127.0.0.1:1", "mock", 7, 1)
        return result, opened.call_args.args[0]

    def test_empty_role_and_usage_do_not_start_ttft(self):
        events = [(0.001, sse({"choices": [{"delta": {"role": "assistant", "content": ""}}]})),
                  (0.020, sse({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7}})),
                  (0.040, sse(content("several tokens"))),
                  (0.080, sse(content("more tokens"))),
                  (0.090, sse(b"[DONE]"))]
        result, request = self.measure(events)
        self.assertTrue(result["ok"], result)
        self.assertAlmostEqual(result["ttft_s"], 0.040)
        self.assertEqual(result["first_content_s"], result["ttft_s"])
        self.assertAlmostEqual(result["last_content_s"], 0.080)
        self.assertAlmostEqual(result["total_s"], 0.100)
        self.assertEqual(result["content_times_s"], [0.040, 0.080])
        self.assertEqual(result["content_chunks"], 2)
        self.assertEqual(result["completion_tokens"], 7)
        self.assertEqual(result["prompt_tokens"], 11)
        self.assertEqual(result["completion_tokens_source"], "usage.completion_tokens")
        self.assertEqual(len(result["itl_samples"]), 1)  # no invented per-token timestamps
        self.assertAlmostEqual(result["itl_samples"][0], 0.040)
        self.assertAlmostEqual(result["tpot_s"], 0.040 / 6)
        self.assertTrue(result["stream_complete"])
        self.assertTrue(result["request_id"])
        self.assertEqual(request.get_header("X-request-id"), result["request_id"])
        self.assertEqual(result["server_request_id"], "server-request-test")

    def test_missing_usage_keeps_estimation_explicit(self):
        result, _ = self.measure([(0.01, sse(content("abcdefgh"))),
                                  (0.03, sse(content("ijklmnop"))),
                                  (0.05, sse(b"[DONE]"))])
        self.assertTrue(result["ok"])
        self.assertEqual(result["completion_tokens"], 4)
        self.assertTrue(result["tokens_estimated"])
        self.assertEqual(result["completion_tokens_source"], "estimated_characters_div4")
        self.assertIsNone(result["prompt_tokens"])
        self.assertEqual(len(result["itl_samples"]), 1)

    def test_one_output_token_has_undefined_tpot(self):
        result, _ = self.measure([(0.01, sse(content("x"))),
            (0.03, sse({"choices": [], "usage": {"completion_tokens": 1}})),
            (0.05, sse(b"[DONE]"))])
        self.assertTrue(result["ok"])
        self.assertIsNone(result["tpot_s"])
        self.assertEqual(result["tpot_unavailable_reason"], "single_token")
        self.assertEqual(result["itl_samples"], [])

    def test_multiple_tokens_in_one_chunk_do_not_imply_zero_tpot(self):
        result, _ = self.measure([(0.01, sse(content("many tokens"))),
            (0.03, sse({"choices": [], "usage": {"completion_tokens": 7}})),
            (0.05, sse(b"[DONE]"))])
        self.assertTrue(result["ok"])
        self.assertIsNone(result["tpot_s"])
        self.assertEqual(result["tpot_unavailable_reason"], "insufficient_content_chunks")
        self.assertEqual(result["itl_samples"], [])

    def test_empty_stream_is_not_success(self):
        result, _ = self.measure([(0.01, sse({"choices": [{"delta": {"role": "assistant"}}]})),
                                  (0.03, sse(b"[DONE]"))])
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "empty_response")
        self.assertIsNone(result["ttft_s"])
        self.assertIsNone(result["tpot_s"])
        self.assertTrue(result["request_id"])

    def test_incomplete_and_error_sse_preserve_partial_record_but_fail(self):
        cases = {
            "missing_done": ([], "truncated_stream"),
            "malformed_json": ([(0.03, sse(b"{broken")), (0.04, sse(b"[DONE]"))], "invalid_sse"),
            "server_error": ([(0.03, sse({"error": {"message": "backend stopped"}})), (0.04, sse(b"[DONE]"))], "server_error"),
            "non_object": ([(0.03, sse(b"[]")), (0.04, sse(b"[DONE]"))], "invalid_sse"),
            "invalid_choices": ([(0.03, sse({"choices": "bad"})), (0.04, sse(b"[DONE]"))], "invalid_sse"),
            "unknown_event": ([(0.03, sse({"unexpected": "packet"})), (0.04, sse(b"[DONE]"))], "invalid_sse"),
            "invalid_finish_reason": ([(0.03, sse({"choices": [{"delta": {}, "finish_reason": []}]})), (0.04, sse(b"[DONE]"))], "invalid_sse"),
            "invalid_utf8": ([(0.03, b'data: {"choices": [{"delta": {"content": "\xff"}}]}\n\n')], "invalid_sse"),
            "unclosed_event": ([(0.03, b'data: {"choices": []}')], "truncated_stream"),
            "timeout_after_content": ([(0.03, TimeoutError("read timed out"))], "timeout"),
        }
        for name, (tail, expected) in cases.items():
            with self.subTest(name=name):
                result, _ = self.measure([(0.01, sse(content("partial"))), *tail])
                self.assertFalse(result["ok"], result)
                self.assertEqual(result["error_type"], expected)
                self.assertTrue(result["error"])
                self.assertEqual(result["content_chunks"], 1)
                self.assertAlmostEqual(result["first_content_s"], 0.01)
                self.assertTrue(result["request_id"])
                json.dumps(result, allow_nan=False)

    def test_invalid_usage_cannot_be_a_successful_token_count(self):
        for value in (-1, 0, True, "seven"):
            with self.subTest(value=value):
                result, _ = self.measure([(0.01, sse(content("x"))),
                    (0.03, sse({"choices": [], "usage": {"completion_tokens": value}})),
                    (0.05, sse(b"[DONE]"))])
                self.assertFalse(result["ok"])
                self.assertEqual(result["error_type"], "invalid_usage")

    def test_multiline_sse_data_and_comments_are_supported(self):
        event = b': ping\n\nevent: message\ndata: {"choices": [\ndata: {"delta": {"content": "abcdefgh"}}]}\n\n'
        result, _ = self.measure([(0.01, event), (0.03, sse(b"[DONE]"))])
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["content_chunks"], 1)
        self.assertTrue(result["tokens_estimated"])

    def test_http_failure_and_timeout_return_failure_records(self):
        failures = [(urllib.error.HTTPError("http://127.0.0.1:1", 503, "unavailable", {}, io.BytesIO(b"down")), "http_error"),
                    (urllib.error.URLError(TimeoutError("connect timed out")), "timeout"),
                    (TimeoutError("read timed out"), "timeout")]
        for error, expected in failures:
            with self.subTest(expected=expected, error=str(error)):
                with mock.patch.object(bench.urllib.request, "urlopen", side_effect=error):
                    result = bench.stream_request("http://127.0.0.1:1", "mock", 3, 1)
                self.assertFalse(result["ok"])
                self.assertEqual(result["error_type"], expected)
                self.assertTrue(result["request_id"])
                self.assertGreaterEqual(result["total_s"], 0)
                self.assertIsNone(result["ttft_s"])


class MockGoldenPathTests(unittest.TestCase):
    @contextlib.contextmanager
    def server(self, **kwargs):
        service = load_module("mock_sse_test", MOCK_PATH)
        server = service.make_server(port=0, interval=0.001, first_content_delay=0, **kwargs)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())

    def test_complete_mock_and_benchmark_cli_on_temporary_port(self):
        with self.server(tokens=3, prompt_tokens=11) as url, tempfile.TemporaryDirectory(prefix="ttft-bench-tests-") as tmp:
            output = Path(tmp) / "smoke.json"
            run = subprocess.run([sys.executable, "-B", str(BENCH_PATH),
                "--base-url", url, "--model", "mock", "--max-tokens", "3",
                "--concurrency", "1", "4", "--requests", "20", "--timeout", "2",
                "--test-id", "mock-E1-r1", "--conditions-id", "mock-v1",
                "--output", str(output)], capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stderr)
            document = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(document["test_id"], "mock-E1-r1")
            self.assertEqual(document["conditions_id"], "mock-v1")
            self.assertEqual(document["requests_per_level"], 20)
            self.assertEqual(document["requested_concurrency"], [1, 4])
            self.assertEqual(document["prompt"], bench.PROMPT)
            self.assertIn("chunk", document["measurement_contract"]["itl"])
            for level in document["levels"]:
                self.assertEqual(level["ok"], 20)
                self.assertEqual(level["failed"], 0)
                self.assertEqual(level["total_completion_tokens"], 60)
                self.assertEqual(level["total_s"]["samples"], 20)
                self.assertEqual(len(level["request_records"]), 20)
                self.assertFalse(level["tokens_estimated"])
                self.assertEqual(level["prompt_tokens"]["mean"], 11)
                self.assertGreater(level["observed_peak_inflight"], 0)
                self.assertLessEqual(level["observed_peak_inflight"], level["requested_concurrency"])
                for row in level["request_records"]:
                    self.assertEqual(row["server_request_id"], row["request_id"])
                    self.assertEqual(row["content_chunks"], 3)
                    self.assertEqual(len(row["itl_samples"]), 2)
            self.assertIn("finite", run.stderr.lower())
            self.assertIn("chunk", run.stdout.lower())

    def test_prompt_file_is_preserved_and_prompt_flags_are_exclusive(self):
        with self.server(tokens=2) as url, tempfile.TemporaryDirectory(prefix="ttft-bench-tests-") as tmp:
            prompt_file = Path(tmp) / "prompt.txt"
            prompt_file.write_text("中文固定输入\nsecond line\n", encoding="utf-8")
            output = Path(tmp) / "prompt.json"
            args = [sys.executable, "-B", str(BENCH_PATH), "--base-url", url,
                    "--model", "mock", "--concurrency", "1", "--requests", "2",
                    "--max-tokens", "2", "--prompt-file", str(prompt_file), "--output", str(output)]
            run = subprocess.run(args, capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
            doc = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(doc["prompt"], prompt_file.read_text(encoding="utf-8"))
            self.assertEqual(doc["prompt_source"], "file")
            self.assertEqual(doc["levels"][0]["request_records"][0]["prompt_tokens"], None)
            conflict = subprocess.run([*args, "--prompt", "other"], capture_output=True, text=True, timeout=10)
            self.assertEqual(conflict.returncode, 2)
            self.assertIn("not allowed", conflict.stderr)

    def test_mock_without_usage_marks_estimated_counts(self):
        with self.server(tokens=8, include_usage=False) as url:
            result = bench.sweep_level(url, "mock", 8, 1, 2, 2)
        self.assertEqual(result["ok"], 2)
        self.assertTrue(result["tokens_estimated"])
        self.assertEqual(result["total_completion_tokens"], 4)
        self.assertTrue(all(row["tokens_estimated"] for row in result["request_records"]))

    def test_help_for_both_entrypoints(self):
        for path in (BENCH_PATH, MOCK_PATH):
            with self.subTest(path=path.name):
                result = subprocess.run([sys.executable, "-B", str(path), "--help"], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
