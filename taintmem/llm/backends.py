"""Real LLM backends (stdlib HTTP only) + a disk cache. Keys come from the environment, never from code.

AnthropicLLM       POST {base}/v1/messages           (ANTHROPIC_API_KEY)
OpenAICompatLLM    POST {base}/chat/completions      (OPENAI_API_KEY or key=...) works with vLLM / hosted open models
CachedLLM          wraps any LLM; JSONL cache keyed by (model, system, user, temperature)
FunctionLLM        wraps a python callable (tests, deterministic agents)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


class LLMError(RuntimeError):
    pass


def _post(url: str, headers: dict, body: dict, timeout: float, retries: int) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers={"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < retries:
                time.sleep(min(2 ** attempt, 8)); continue
            raise LLMError(f"HTTP {e.code}: {e.read()[:300]!r}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8)); continue
            raise LLMError(f"network error: {e}") from None
    raise LLMError("unreachable")


class _Base:
    model_id: str

    def __init__(self):
        self.usage = Usage()

    def _account(self, t0, i, o):
        u = self.usage
        u.calls += 1; u.input_tokens += i; u.output_tokens += o; u.seconds += time.time() - t0


class AnthropicLLM(_Base):
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0, timeout: float = 120, retries: int = 3):
        super().__init__()
        self.model_id, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base = (base_url or "https://api.anthropic.com").rstrip("/")
        self.timeout, self.retries = timeout, retries
        if not self.key:
            raise LLMError("ANTHROPIC_API_KEY not set")

    def complete(self, system: str, user: str) -> str:
        t0 = time.time()
        r = _post(f"{self.base}/v1/messages", {"x-api-key": self.key, "anthropic-version": "2023-06-01"},
                  {"model": self.model_id, "max_tokens": self.max_tokens, "temperature": self.temperature,
                   "system": system, "messages": [{"role": "user", "content": user}]}, self.timeout, self.retries)
        text = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
        u = r.get("usage", {})
        self._account(t0, u.get("input_tokens", 0), u.get("output_tokens", 0))
        return text


class OpenAICompatLLM(_Base):
    def __init__(self, model: str, base_url: str, api_key: str | None = None, max_tokens: int = 1024,
                 temperature: float = 0.0, timeout: float = 120, retries: int = 3):
        super().__init__()
        self.model_id, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.base = base_url.rstrip("/")
        self.key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout, self.retries = timeout, retries

    def complete(self, system: str, user: str) -> str:
        t0 = time.time()
        h = {"authorization": f"Bearer {self.key}"} if self.key else {}
        r = _post(f"{self.base}/chat/completions", h,
                  {"model": self.model_id, "max_tokens": self.max_tokens, "temperature": self.temperature,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                  self.timeout, self.retries)
        try:
            text = r["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            raise LLMError(f"malformed response: {str(r)[:200]}") from None
        u = r.get("usage", {})
        self._account(t0, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        return text


class FunctionLLM(_Base):
    def __init__(self, fn, model_id: str = "function"):
        super().__init__()
        self.fn, self.model_id = fn, model_id

    def complete(self, system: str, user: str) -> str:
        t0 = time.time()
        out = self.fn(system, user)
        self._account(t0, len(system + user) // 4, len(out) // 4)
        return out


class CachedLLM(_Base):
    """Makes reruns free and byte-reproducible. Cache is append-only JSONL."""
    def __init__(self, inner, path: str):
        super().__init__()
        self.inner, self.path = inner, path
        self.model_id = getattr(inner, "model_id", "unknown")
        self.cache: dict[str, str] = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line); self.cache[d["k"]] = d["v"]

    def _key(self, system: str, user: str) -> str:
        meta = [self.model_id, getattr(self.inner, "temperature", None), system, user]
        return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()

    def complete(self, system: str, user: str) -> str:
        k = self._key(system, user)
        if k in self.cache:
            self.usage.cache_hits += 1
            return self.cache[k]
        out = self.inner.complete(system, user)
        self.cache[k] = out
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps({"k": k, "v": out}) + "\n")
        iu = getattr(self.inner, "usage", None)
        if iu:
            self.usage = Usage(iu.calls, self.usage.cache_hits, iu.input_tokens, iu.output_tokens, iu.seconds)
        return out
