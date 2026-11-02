import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from taintmem.llm.backends import AnthropicLLM, CachedLLM, FunctionLLM, LLMError, OpenAICompatLLM

SEEN = []


class H(BaseHTTPRequestHandler):
    fail_first = {"n": 0}

    def log_message(self, *a): pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        SEEN.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        if self.path.endswith("/flaky") or (self.path.startswith("/flaky") and H.fail_first["n"] < 1):
            H.fail_first["n"] += 1
            self.send_response(503); self.end_headers(); self.wfile.write(b"busy"); return
        if self.path.endswith("/v1/messages"):
            out = {"content": [{"type": "text", "text": "anthropic-reply"}], "usage": {"input_tokens": 7, "output_tokens": 3}}
        else:
            out = {"choices": [{"message": {"content": "openai-reply"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
        raw = json.dumps(out).encode()
        self.send_response(200); self.send_header("content-length", str(len(raw))); self.end_headers(); self.wfile.write(raw)


class TestBackends(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_anthropic_request_shape_and_usage(self):
        SEEN.clear()
        llm = AnthropicLLM("m1", api_key="k", base_url=self.base)
        self.assertEqual(llm.complete("SYS", "USER"), "anthropic-reply")
        path, hdr, body = SEEN[0]
        self.assertEqual(path, "/v1/messages")
        self.assertEqual(hdr["x-api-key"], "k"); self.assertEqual(hdr["anthropic-version"], "2023-06-01")
        self.assertEqual((body["system"], body["messages"][0]["content"], body["model"]), ("SYS", "USER", "m1"))
        self.assertEqual((llm.usage.input_tokens, llm.usage.output_tokens), (7, 3))

    def test_openai_compat(self):
        SEEN.clear()
        llm = OpenAICompatLLM("open-m", self.base, api_key="tok")
        self.assertEqual(llm.complete("S", "U"), "openai-reply")
        path, hdr, body = SEEN[0]
        self.assertEqual(path, "/chat/completions"); self.assertEqual(hdr["authorization"], "Bearer tok")
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])

    def test_retries_on_503_then_succeeds(self):
        H.fail_first["n"] = 0
        llm = OpenAICompatLLM("m", self.base + "/flaky", retries=2)
        self.assertEqual(llm.complete("s", "u"), "openai-reply")

    def test_missing_key_raises_and_errors_wrapped(self):
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with self.assertRaises(LLMError):
                AnthropicLLM("m")
        finally:
            if old: os.environ["ANTHROPIC_API_KEY"] = old
        with self.assertRaises(LLMError):
            OpenAICompatLLM("m", self.base + "/flaky", retries=0).complete("s", "u")

    def test_cache_hits_and_persists(self):
        calls = []
        inner = FunctionLLM(lambda s, u: calls.append(u) or f"echo:{u}", "fn-model")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.jsonl")
            c = CachedLLM(inner, p)
            self.assertEqual(c.complete("s", "a"), "echo:a"); self.assertEqual(c.complete("s", "a"), "echo:a")
            self.assertEqual((len(calls), c.usage.cache_hits), (1, 1))
            c2 = CachedLLM(inner, p)                       # fresh process: still cached
            self.assertEqual(c2.complete("s", "a"), "echo:a"); self.assertEqual(len(calls), 1)
            c2.complete("s", "b"); self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
