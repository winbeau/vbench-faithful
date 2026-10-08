"""Generate text-only DeepSeek silver labels; reuse the existing teacher client.

Requires the sibling vbench-prompts-compile source explicitly. Nothing in that
repository is modified. Human review is reserved before any teacher call and
is never replaced by teacher agreement. This command does not train a model.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from .common import ROOT, sha256_file
from .subject_artifacts import canonical_json, new_output, object_sha256, read_jsonl, write_json, write_jsonl
from .subject_semantics import REWRITE_TEMPLATE, TEMPLATE, text_request, validate_label


class TeacherIdentityDrift(ValueError):
    pass


def check_teacher_identity(response, config: dict) -> None:
    if response.model != config["expected_response_model"] or response.system_fingerprint != config["expected_system_fingerprint"]:
        raise TeacherIdentityDrift("DeepSeek response model/fingerprint drifted from the frozen run declaration")
    if response.finish_reason != "stop":
        raise ValueError(f"incomplete teacher response: {response.finish_reason}")


def annotate(rows: list[dict], *, client, config: dict, vocabulary: set[str], output: Path, ledger,
             workers: int = 4, operation: str = "label") -> list[dict]:
    tasks = []
    for row in rows:
        system, user = (text_request(row["prompt"], vocabulary) if operation == "label" else
                        (REWRITE_TEMPLATE, json.dumps({"prompt": row["prompt"]}, ensure_ascii=False)))
        request_id = f"{operation}-{row['sample_id']}"
        request = {"request_id": request_id, "model": config["api_model"], "system": system, "user": user,
                   "temperature": config["temperature"], "max_tokens": config["max_tokens"]}
        # Charge serially before any HTTP work; the reused ledger isn't a
        # concurrency primitive. Workers only issue independent HTTP calls.
        ledger.charge({"request_id": request_id, "max_tokens": config["max_tokens"], "request_sha256": object_sha256(request)})
        write_json(output / "requests" / f"{request_id}.json", request)
        tasks.append((row, request))

    def call(task):
        row, request = task
        try:
            response = client.chat(system=request["system"], user=request["user"],
                                   max_tokens=config["max_tokens"], temperature=config["temperature"])
            payload = {"request_id": request["request_id"], "raw": response.text, "response": response.as_metadata()}
            # Save even a drifting/invalid response; it is quarantined, not
            # silently repaired or substituted with a guessed label.
            write_json(output / "responses" / f"{request['request_id']}.json", payload)
            check_teacher_identity(response, config)
            value = json.loads(response.text)
            if operation == "label":
                validate_label(value, row["prompt"], vocabulary, original_prompt=row.get("original_prompt"))
                return {**row, "label": value, "quality": "silver", "request_id": request["request_id"],
                        "status": "supported" if value["status"] == "ok" else "unsupported"}
            if not isinstance(value, dict) or set(value) != {"prompt"} or not isinstance(value["prompt"], str) or not value["prompt"].strip():
                raise ValueError("rewrite schema must contain only nonempty prompt text")
            if value["prompt"].strip() == row["prompt"].strip():
                raise ValueError("rewrite duplicated the original prompt")
            return {**row, "sample_id": hashlib.sha256(f"{row['sample_id']}:rewrite:{value['prompt']}".encode()).hexdigest(),
                    "original_prompt": row["prompt"], "prompt": value["prompt"], "quality": "unlabeled_rewrite",
                    "rewrite_request_id": request["request_id"], "status": "rewritten"}
        except TeacherIdentityDrift:
            raise
        except Exception as exc:
            return {**row, "quality": "quarantined", "status": "unsupported", "request_id": request["request_id"],
                    "error": f"{type(exc).__name__}: {exc}"}

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for row in executor.map(call, tasks):
            results.append(row)
            ledger.append_result({"sample_id": row["sample_id"], "quality": row["quality"], "status": row["status"]})
            write_jsonl(output / f"{operation}.checkpoint.jsonl", results)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--prompt-compile-root", type=Path, default=ROOT / "packages/prompt-compiler")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/subject-repair/deepseek.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    today = datetime.now(timezone.utc).date().isoformat()
    if today != config["run_date_utc"]:
        raise ValueError("refresh and freeze the model/version/date declaration before a new-day API run")
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    compile_root = args.prompt_compile_root.resolve()
    sys.path.insert(0, str(compile_root / "src"))
    from vbench_prompts_compile.teacher import BudgetLedger, DeepSeekClient

    vocabulary_path = ROOT / "configs/subject-repair/vocabulary.json"
    vocabulary = set(json.loads(vocabulary_path.read_text())["subjects"])
    pool = read_jsonl(args.pool / "prompts.jsonl")
    review = read_jsonl(args.pool / "human_review.pending.jsonl")
    if len(review) != config["human_review_originals"]:
        raise ValueError("human review reservation differs from frozen configuration")
    train = sorted((r for r in pool if r["split"] == "train"),
                   key=lambda r: hashlib.sha256(f"20260920:{r['sample_id']}".encode()).hexdigest())[:config["train_originals"]]
    if len(train) != config["train_originals"]:
        raise ValueError("insufficient training prompt pool")
    output = new_output(args.output)
    client = DeepSeekClient(token_path=compile_root / "token.txt", base_url=config["base_url"], model=config["api_model"], timeout_seconds=55)
    ledger = BudgetLedger(output, authorized_total=config["max_requests"], max_output_tokens=config["max_tokens"], budget_path=output / "budget.json")
    # API rows contain text only; pending human review fields aren't targets.
    originals = [{k: v for k, v in row.items() if k not in {"label", "reviewed", "reviewer"}} for row in train + review]
    manifest = {"config": config, "config_sha256": sha256_file(args.config),
                "vocabulary_sha256": sha256_file(vocabulary_path), "pool_manifest_sha256": sha256_file(args.pool / "manifest.json"),
                "template": TEMPLATE, "rewrite_template": REWRITE_TEMPLATE,
                "template_sha256": hashlib.sha256(TEMPLATE.encode()).hexdigest(),
                "client_source_sha256": sha256_file(compile_root / "src/vbench_prompts_compile/teacher.py"),
                "human_review_reserved": len(review), "human_review_completed": 0,
                "llm_human_agreement": "NOT RUN", "head_accuracy": "NOT RUN", "training": "NOT RUN"}
    write_json(output / "manifest.json", manifest)
    labeled = annotate(originals, client=client, config=config, vocabulary=vocabulary, output=output, ledger=ledger, workers=args.workers)
    rewrite_inputs = train[:config["train_rewrites"]]
    rewrites = annotate(rewrite_inputs, client=client, config=config, vocabulary=vocabulary, output=output, ledger=ledger,
                        workers=args.workers, operation="rewrite")
    valid_rewrites = [row for row in rewrites if row["quality"] == "unlabeled_rewrite"]
    if valid_rewrites:
        labeled.extend(annotate(valid_rewrites, client=client, config=config, vocabulary=vocabulary, output=output,
                                ledger=ledger, workers=args.workers))
    write_jsonl(output / "silver.jsonl", labeled)
    write_jsonl(output / "rewrites.jsonl", rewrites)
    manifest.update({"requests_charged": ledger.used, "silver_records": sum(row["quality"] == "silver" for row in labeled),
                     "quarantined_records": sum(row["quality"] == "quarantined" for row in labeled),
                     "unsupported_labels": sum(row["quality"] == "silver" and row["status"] == "unsupported" for row in labeled),
                     "files_sha256": {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*"))
                                      if p.is_file() and p.name != "manifest.json"}})
    write_json(output / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("requests_charged", "silver_records", "quarantined_records", "human_review_completed")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
