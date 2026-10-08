"""Teacher (DeepSeek) request/response handling: budget, prompts, validation.

Design rules enforced here:

* every HTTP POST is charged to an append-only ledger before it is sent;
* the API key is read from a file or the environment and never logged;
* teacher output is *candidate* material: it is parsed strictly and validated
  against the same contract as every other record, and failures are quarantined
  with a reason instead of being repaired silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.error
import urllib.request

from .records import (
    RecordError,
    SCENE_LABELS,
    TEACHER_QUALITY,
    add_length_meta,
    canonical_target,
    make_record,
    normalize_phrase,
    normalize_space,
    sha256_text,
    span_support,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKEN_PATH = REPO_ROOT / "token.txt"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "teacher"

FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
SPATIAL_RELATION_WORDS = (
    "left, right, above, below, on, under, beneath, underneath, near, next to, "
    "beside, behind, in front of, over, inside, outside, between"
)


class TeacherError(RuntimeError):
    """Transport/HTTP failure that is not a modelling error."""


class BudgetExceeded(TeacherError):
    """The authorised request budget is exhausted."""


@dataclass(frozen=True)
class TeacherLimits:
    max_requests: int
    max_output_tokens: int
    timeout_seconds: float = 180.0

    def check(self, requested_output_tokens: int) -> None:
        if requested_output_tokens > self.max_output_tokens:
            raise BudgetExceeded(
                f"requested {requested_output_tokens} output tokens > authorised {self.max_output_tokens}"
            )


@dataclass
class TeacherResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    latency_ms: int
    http_status: int
    system_fingerprint: str | None = None

    def as_metadata(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "http_status": self.http_status,
            "system_fingerprint": self.system_fingerprint,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def load_api_key(token_path: Path | str | None = None, *, provider: str = "deepseek") -> str:
    """Read one provider's key from the (possibly multi-token) token file or env.

    The token file may hold several tokens (``name=value`` lines, or bare lines
    in the documented order deepseek, chiyi); see :mod:`vbench_prompts_compile.llm`.
    """
    from .llm import load_tokens

    try:
        return load_tokens(token_path).get(provider)
    except Exception as error:  # re-raise with the historical error type
        raise TeacherError(str(error)) from error


def _no_redirect_opener() -> urllib.request.OpenerDirector:
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102 - stdlib signature
            raise TeacherError(f"refusing redirect to {newurl!r}")

    return urllib.request.build_opener(_NoRedirect)


class DeepSeekClient:
    """Minimal chat-completions client with injectable transport for tests."""

    def __init__(
        self,
        *,
        token_path: Path | str | None = None,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = 180.0,
        transport: Callable[[str, bytes, Mapping[str, str], float], tuple[int, bytes]] | None = None,
    ) -> None:
        self._api_key = api_key or load_api_key(token_path)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._transport = transport or self._urllib_transport

    @staticmethod
    def _urllib_transport(url: str, payload: bytes, headers: Mapping[str, str], timeout: float) -> tuple[int, bytes]:
        request = urllib.request.Request(url, data=payload, headers=dict(headers), method="POST")
        opener = _no_redirect_opener()
        try:
            with opener.open(request, timeout=timeout) as response:
                return int(response.status), response.read()
        except urllib.error.HTTPError as error:  # status is preserved, body is not echoed
            return int(error.code), b""

    def chat(
        self,
        *,
        user: str,
        system: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> TeacherResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "stream": False,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "User-Agent": "vbench-prompts-compile/0.1",
        }
        started = time.monotonic()
        status, raw = self._transport(f"{self.base_url}/chat/completions", body, headers, self.timeout_seconds)
        latency_ms = int((time.monotonic() - started) * 1000)
        if status != 200:
            raise TeacherError(f"HTTP {status} from {self.base_url}/chat/completions")
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TeacherError("unparseable API response envelope") from error
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as error:
            raise TeacherError("unexpected API response shape") from error
        usage = data.get("usage") or {}
        return TeacherResponse(
            text=content,
            model=str(data.get("model", self.model)),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=str(choice.get("finish_reason", "")),
            latency_ms=latency_ms,
            http_status=status,
            system_fingerprint=data.get("system_fingerprint"),
        )


SYSTEM_PARSE = (
    "You label one text-to-video prompt. Use only what the prompt states explicitly. "
    "Never guess, never add objects, actions or relations that are not written, never explain, "
    "never translate. Answer with a single JSON object and nothing else."
)
SYSTEM_SCENE = (
    "You compare a video prompt with one automatic image caption. Judge only the scene requirement "
    "of the prompt against the caption. Never explain, never use outside knowledge. "
    "Answer with a single JSON object and nothing else."
)

TASK_INSTRUCTIONS: Mapping[str, str] = {
    "spatial": (
        "Extract every explicit spatial relation between two named entities. "
        f"Allowed relation words: {SPATIAL_RELATION_WORDS}. "
        'Answer exactly {"relationships":[{"subject":"...","relation":"...","object":"..."}]}. '
        "Use the entity words from the prompt. If the prompt states no spatial relation, answer "
        '{"relationships":[]}.'
    ),
    "action": (
        "Extract the actions performed by a person or animal. Each action is a short everyday English "
        'verb phrase, for example "playing guitar" or "riding a bike". '
        'Answer exactly {"actions":["..."]}. If the prompt states no action, answer {"actions":[]}. '
        "Do not list objects or scene descriptions as actions."
    ),
    "objects": (
        "Extract the concrete physical things explicitly mentioned: people, animals, objects and visible "
        "scene elements. Use the shortest noun phrase that names each one, and keep the wording of the "
        'prompt. Answer exactly {"entities":["..."]}. Do not add things that are only implied.'
    ),
    "scene": (
        'Answer exactly {"label":"supported"} or {"label":"contradicted"} or {"label":"insufficient"}. '
        "supported: the caption describes the scene the prompt requires. "
        "contradicted: the caption clearly shows a different scene. "
        "insufficient: the prompt requires no scene, or the caption gives no usable evidence about it."
    ),
}


def build_request(task: str, prompt: str, caption: str | None = None) -> tuple[str, str]:
    """Return ``(system, user)`` for one task; the user text carries the input."""
    if task not in TASK_INSTRUCTIONS:
        raise KeyError(f"unknown task {task!r}")
    prompt = normalize_space(prompt)
    if task == "scene":
        if caption is None:
            raise ValueError("scene requests need a caption")
        user = (
            f"{TASK_INSTRUCTIONS['scene']}\n\n"
            f"Prompt: {prompt}\n"
            f"Caption: {normalize_space(caption)}"
        )
        return SYSTEM_SCENE, user
    return SYSTEM_PARSE, f"{TASK_INSTRUCTIONS[task]}\n\nPrompt: {prompt}"


def strip_code_fence(text: str) -> str:
    return FENCE_RE.sub("", text).strip()


def parse_teacher_text(task: str, text: str) -> tuple[Any | None, list[str]]:
    """Parse the raw completion into a target-shaped value (no span checks yet)."""
    cleaned = strip_code_fence(text)
    if task == "scene":
        if normalize_phrase(cleaned) in SCENE_LABELS:
            return normalize_phrase(cleaned), []
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None, ["invalid_json"]
        if isinstance(data, Mapping) and set(data) == {"label"}:
            return data["label"], []
        return None, ["schema_keys"]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, ["invalid_json"]
    if not isinstance(data, Mapping):
        return None, ["schema_types"]
    return dict(data), []


@dataclass
class IngestResult:
    target: Any | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    record: dict[str, Any] | None = None


def ingest_teacher_output(
    *,
    task: str,
    prompt: str,
    caption: str | None,
    response: TeacherResponse,
    source: str,
    source_id: str,
    group_id: str,
    max_output_tokens: int,
    action_resolver: Callable[[str], str | None] | None = None,
) -> IngestResult:
    """Validate one teacher response and, when valid, build a candidate record."""
    errors: list[str] = []
    warnings: list[str] = []
    if response.finish_reason != "stop":
        errors.append("truncated")
    if response.completion_tokens > max_output_tokens:
        errors.append("over_token_budget")
    value, parse_errors = parse_teacher_text(task, response.text)
    errors.extend(parse_errors)
    if value is None:
        return IngestResult(None, tuple(errors), tuple(warnings))
    target, target_errors, target_warnings = canonical_target(task, value, action_resolver=action_resolver)
    errors.extend(target_errors)
    warnings.extend(target_warnings)
    if target is None:
        return IngestResult(None, tuple(errors), tuple(warnings))
    if task == "spatial":
        for row in target["relationships"]:
            for field in ("subject", "object"):
                support = span_support(prompt, row[field])
                if support == "none":
                    errors.append(f"unsupported_span:{field}")
                elif support != "exact":
                    warnings.append(f"span_{support}")
    elif task == "action":
        for name in target["actions"]:
            support = span_support(prompt, name)
            if support == "none":
                errors.append("unsupported_span:action")
            elif support != "exact":
                warnings.append(f"span_{support}")
    elif task == "objects":
        for name in target["entities"]:
            support = span_support(prompt, name)
            if support == "none":
                errors.append("unsupported_span:entity")
            elif support != "exact":
                warnings.append(f"span_{support}")
    if errors:
        return IngestResult(target, tuple(errors), tuple(warnings))
    sample_input = {"prompt": normalize_space(prompt)}
    if task == "scene":
        sample_input["caption"] = normalize_space(caption or "")
    try:
        record = make_record(
            task=task,
            sample_input=sample_input,
            target=target,
            source=source,
            source_id=source_id,
            group_id=group_id,
            quality=TEACHER_QUALITY,
            meta={
                "teacher_model": response.model,
                "teacher_finish_reason": response.finish_reason,
                "teacher_completion_tokens": response.completion_tokens,
                "teacher_prompt_sha256": sha256_text(prompt),
                "teacher_raw_response_sha256": sha256_text(response.text),
                "review_status": "unreviewed",
            },
            action_resolver=action_resolver,
            check_spans=False,  # span checks already ran above with explicit reasons
        )
    except RecordError as error:
        return IngestResult(target, tuple(errors) + error.errors, tuple(warnings))
    return IngestResult(add_length_meta(record), (), tuple(warnings), record)


class BudgetLedger:
    """Append-only request ledger plus an aggregated authorised-budget file."""

    def __init__(
        self,
        run_dir: Path | str,
        *,
        authorized_total: int,
        max_output_tokens: int,
        budget_path: Path | str | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.run_dir / "ledger.jsonl"
        # The authorised total is global across runs; only the ledger is per run.
        self.budget_path = Path(budget_path) if budget_path else DEFAULT_OUTPUT_ROOT / "BUDGET.json"
        self.budget_path.parent.mkdir(parents=True, exist_ok=True)
        self.authorized_total = int(authorized_total)
        self.max_output_tokens = int(max_output_tokens)
        if not self.budget_path.exists():
            self._write_budget({"authorized_total": self.authorized_total, "used_requests": 0, "runs": {}})

    def _write_budget(self, payload: Mapping[str, Any]) -> None:
        payload = dict(payload)
        payload["updated_utc"] = utc_now()
        tmp = self.budget_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.budget_path)

    def _read_budget(self) -> dict[str, Any]:
        return json.loads(self.budget_path.read_text(encoding="utf-8"))

    @property
    def used(self) -> int:
        return int(self._read_budget()["used_requests"])

    def remaining(self) -> int:
        return self.authorized_total - self.used

    def charge(self, entry: Mapping[str, Any]) -> int:
        """Charge one request *before* it is sent; returns the new used count."""
        budget = self._read_budget()
        used = int(budget["used_requests"])
        if used >= self.authorized_total:
            raise BudgetExceeded(f"budget exhausted: {used}/{self.authorized_total}")
        if int(entry.get("max_tokens", 0)) > self.max_output_tokens:
            raise BudgetExceeded("request max_tokens above authorised per-request limit")
        used += 1
        budget["used_requests"] = used
        runs = dict(budget.get("runs", {}))
        runs[self.run_dir.name] = runs.get(self.run_dir.name, 0) + 1
        budget["runs"] = runs
        self._write_budget(budget)
        self._append({"kind": "charge", "charge_index": used, "utc": utc_now(), **dict(entry)})
        return used

    def append_result(self, entry: Mapping[str, Any]) -> None:
        self._append({"kind": "result", "utc": utc_now(), **dict(entry)})

    def _append(self, entry: Mapping[str, Any]) -> None:
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
