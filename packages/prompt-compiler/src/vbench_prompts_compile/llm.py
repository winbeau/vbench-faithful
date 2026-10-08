"""Provider-agnostic chat clients (text + vision) with a shared budget ledger.

Two providers are configured by the project token file:

* ``deepseek`` -> ``https://api.deepseek.com`` (``deepseek-flash``), text only;
* ``chiyi``    -> ``https://api.chiyi.cc`` (``gpt-5.6-luna``), text + image input,
  used as the visual annotator/arbiter for scene data.

The token file may contain either ``name=value`` lines or bare lines; bare lines
are assigned in the documented order ``deepseek``, ``chiyi``. Tokens are never
printed, logged or written into reports; only lengths and prefixes are reported.

Budget accounting is per purpose (``output/<root>/BUDGET-<purpose>.json``) and is
charged *before* a request is sent, so a crash cannot silently under-count.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import base64
from contextlib import contextmanager
import fcntl
import http.client
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKEN_PATH = REPO_ROOT / "token.txt"
BARE_LINE_ORDER = ("deepseek", "chiyi")

PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-flash"},
    "chiyi": {"base_url": "https://api.chiyi.cc", "model": "gpt-5.6-luna"},
}


class LLMError(RuntimeError):
    """Transport/HTTP/parse failure that is not a modelling error."""


class BudgetExceeded(LLMError):
    """The authorised request budget for this purpose is exhausted."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class TokenSet:
    tokens: Mapping[str, str]
    source: str

    def get(self, provider: str) -> str:
        if provider not in self.tokens:
            raise LLMError(f"token file has no entry for provider {provider!r}; found: {sorted(self.tokens)}")
        return self.tokens[provider]

    def describe(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "providers": {name: {"length": len(value), "prefix": value[:3] + "***"} for name, value in self.tokens.items()},
        }


def load_tokens(token_path: Path | str | None = None) -> TokenSet:
    """Parse ``name=value`` lines, else positional bare lines."""
    path = Path(token_path) if token_path else DEFAULT_TOKEN_PATH
    env_map = {
        "deepseek": os.environ.get("DEEPSEEK_API_KEY", "").strip(),
        "chiyi": os.environ.get("CHIYI_API_KEY", "").strip(),
    }
    env_map = {k: v for k, v in env_map.items() if v}
    if env_map and not path.exists():
        return TokenSet(env_map, "environment")
    if not path.exists():
        raise LLMError(f"missing token file: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    parsed: dict[str, str] = {}
    bare: list[str] = []
    for line in lines:
        if "=" in line and not line.startswith("sk-"):
            name, _, value = line.partition("=")
            parsed[name.strip().lower()] = value.strip()
        else:
            bare.append(line)
    for index, value in enumerate(bare):
        if index < len(BARE_LINE_ORDER):
            parsed.setdefault(BARE_LINE_ORDER[index], value)
    parsed.update(env_map)
    if not parsed:
        raise LLMError(f"no tokens parsed from {path}")
    return TokenSet(parsed, str(path))


def encode_image(path: Path | str) -> str:
    data = Path(path).read_bytes()
    suffix = Path(path).suffix.lower().lstrip(".") or "png"
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif"}.get(suffix, "png")
    return f"data:image/{mime};base64,{base64.b64encode(data).decode()}"


@dataclass
class ChatResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    latency_ms: int
    http_status: int

    def as_metadata(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "http_status": self.http_status,
        }


class ChatClient:
    """OpenAI-compatible chat client with injectable transport (for offline tests)."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 180.0,
        transport: Callable[[str, bytes, Mapping[str, str], float], tuple[int, bytes]] | None = None,
        json_mode: bool = True,
        extra_payload: Mapping[str, Any] | None = None,
    ) -> None:
        defaults = PROVIDER_DEFAULTS.get(provider, {})
        self.provider = provider
        self.base_url = (base_url or defaults.get("base_url", "")).rstrip("/")
        self.model = model or defaults.get("model", "")
        if not self.base_url or not self.model:
            raise LLMError(f"unknown provider {provider!r}: configure base_url and model")
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._transport = transport or self._urllib_transport
        self.json_mode = json_mode
        self.extra_payload = dict(extra_payload or {})

    @staticmethod
    def _urllib_transport(url: str, payload: bytes, headers: Mapping[str, str], timeout: float) -> tuple[int, bytes]:
        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102 - stdlib signature
                raise LLMError(f"refusing redirect to {newurl!r}")

        request = urllib.request.Request(url, data=payload, headers=dict(headers), method="POST")
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            with opener.open(request, timeout=timeout) as response:
                return int(response.status), response.read()
        except urllib.error.HTTPError as error:
            return int(error.code), b""
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, http.client.HTTPException) as error:
            # Transient network failures (SSL EOF, reset, DNS) must not kill a long
            # annotation batch; surface them as a retryable transport status.
            raise LLMError(f"transport failure: {type(error).__name__}") from error

    def chat(
        self,
        *,
        user: str,
        system: str,
        images: Sequence[Path | str] = (),
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool | None = None,
    ) -> ChatResponse:
        if images:
            content: list[dict[str, Any]] = [{"type": "text", "text": user}]
            for image in images:
                content.append({"type": "image_url", "image_url": {"url": encode_image(image)}})
        else:
            content = user  # type: ignore[assignment]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": int(max_tokens),
            "stream": False,
            **self.extra_payload,
        }
        use_json = self.json_mode if json_mode is None else json_mode
        if use_json:
            payload["response_format"] = {"type": "json_object"}
        if self.provider == "deepseek":
            payload["thinking"] = {"type": "disabled"}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "User-Agent": "vbench-prompts-compile/0.2",
        }
        started = time.monotonic()
        status, raw = self._transport(f"{self.base_url}/v1/chat/completions" if self.provider != "deepseek" else f"{self.base_url}/chat/completions", body, headers, self.timeout_seconds)
        latency_ms = int((time.monotonic() - started) * 1000)
        if status != 200:
            raise LLMError(f"HTTP {status} from {self.provider}")
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LLMError("unparseable response envelope") from error
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError("unexpected response shape") from error
        usage = data.get("usage") or {}
        return ChatResponse(
            text=text,
            model=str(data.get("model", self.model)),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=str(choice.get("finish_reason", "")),
            latency_ms=latency_ms,
            http_status=status,
        )


class RequestBudget:
    """Append-only, per-purpose request accounting shared by all providers."""

    def __init__(self, root: Path | str, purpose: str, *, authorized_total: int) -> None:
        self.dir = Path(root)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.purpose = purpose
        self.path = self.dir / f"BUDGET-{purpose}.json"
        self.ledger_path = self.dir / f"ledger-{purpose}.jsonl"
        self.authorized_total = int(authorized_total)
        if self.authorized_total < 0:
            raise ValueError("authorized_total must be nonnegative")
        self.lock_path = self.dir / f"BUDGET-{purpose}.lock"
        with self._locked():
            if not self.path.exists():
                self._write({"authorized_total": self.authorized_total, "used_requests": 0, "by_provider": {}})
            state = self._read()
            # The explicit cap applies to this invocation; an older, larger cap
            # must never silently authorise more requests. Charges and history stay.
            state["authorized_total"] = self.authorized_total
            self._write(state)

    @contextmanager
    def _locked(self):
        with self.lock_path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _write(self, payload: Mapping[str, Any]) -> None:
        payload = dict(payload)
        payload["updated_utc"] = utc_now()
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def _read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    @property
    def used(self) -> int:
        return int(self._read()["used_requests"])

    def remaining(self) -> int:
        return self.authorized_total - self.used

    def charge(self, entry: Mapping[str, Any], *, provider: str) -> int:
        with self._locked():
            state = self._read()
            used = int(state["used_requests"])
            cap = min(self.authorized_total, int(state["authorized_total"]))
            if used >= cap:
                raise BudgetExceeded(f"{self.purpose}: budget exhausted {used}/{cap}")
            used += 1
            state["used_requests"] = used
            by_provider = dict(state.get("by_provider", {}))
            by_provider[provider] = by_provider.get(provider, 0) + 1
            state["by_provider"] = by_provider
            self._write(state)
            self.append({**dict(entry), "kind": "charge", "charge_index": used, "provider": provider, "utc": utc_now()})
        return used

    def append(self, entry: Mapping[str, Any]) -> None:
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def summary(self) -> dict[str, Any]:
        state = self._read()
        return {
            "purpose": self.purpose,
            "authorized_total": state["authorized_total"],
            "used_requests": state["used_requests"],
            "by_provider": state.get("by_provider", {}),
        }
