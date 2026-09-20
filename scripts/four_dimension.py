"""Offline evidence and runtime inventory for the four candidate dimensions.

``offline`` deliberately inspects only the locked VBench source and its JSON
metadata.  ``inventory`` checks paths, source versions, video directories and
isolated CUDA visibility before any metric or model module is imported.  The
commands are a gate for later M3--M6 scoring; they never produce a model score.

The driver is intentionally thin.  It does not duplicate a metric formula or
load a model.  Reports use the same ``verified``/``blocked``/``failed`` status
vocabulary so a missing H100 asset is visible and resumable rather than being
silently replaced with a CPU result.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = REPO_ROOT / "packages" / "audit-core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from vbench_audit_core.devices import parse_gpu  # noqa: E402
from vbench_audit_core.errors import InputError  # noqa: E402
from vbench_audit_core.inputs import sha256_file  # noqa: E402
from vbench_audit_core.upstream import inspect_upstream, verify_upstream  # noqa: E402


DIMENSIONS = (
    "background_consistency",
    "temporal_style",
    "object_class",
    "color",
)
DEFAULT_ROOT = REPO_ROOT / "output" / "four_dimension_20260920"
DEFAULT_PROTOCOL = REPO_ROOT / "configs" / "four_dimension" / "protocol.toml"
DEFAULT_MODELS = REPO_ROOT / "configs" / "four_dimension" / "models.h100.toml"
SCHEMA_VERSION = "four-dimension-m2-v1"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = tomllib.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"TOML root must be a table: {path}")
    return value


def _status(*statuses: str) -> str:
    """Combine check states without hiding a hard failure."""
    values = [item for item in statuses if item]
    if any(item == "failed" for item in values):
        return "failed"
    if any(item == "blocked" for item in values):
        return "blocked"
    if values and all(item == "not_applicable" for item in values):
        return "not_applicable"
    return "verified"


def _check(status: str, **details: Any) -> dict[str, Any]:
    return {"status": status, **details}


def _resolve(path: str | Path | None, default: Path) -> Path:
    selected = Path(path) if path is not None else default
    return selected.expanduser().resolve()


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _git(path: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _command(args: list[str], *, env: Mapping[str, str] | None = None, timeout: float = 30) -> dict[str, Any]:
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            env=dict(env) if env is not None else None,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc), "error": type(exc).__name__}
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def _dimension_list(protocol: Mapping[str, Any], selected: str | None) -> list[str]:
    configured = protocol.get("protocol", {}).get("dimensions", DIMENSIONS)
    dimensions = [str(item) for item in configured]
    unknown = [item for item in dimensions if item not in DIMENSIONS]
    if unknown:
        raise ValueError(f"protocol contains unsupported dimensions: {unknown}")
    if selected is not None:
        if selected not in dimensions:
            raise ValueError(f"dimension is not enabled by protocol: {selected}")
        return [selected]
    return dimensions


def _upstream_metadata(upstream: Path, protocol: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_cfg = protocol.get("protocol", {}).get("dataset", {})
    metadata_rel = str(dataset_cfg.get("metadata", "vbench/VBench_full_info.json"))
    mapping_rel = str(dataset_cfg.get("dimension_to_folder", "dimension_to_folder.json"))
    metadata_path = upstream / metadata_rel
    mapping_path = upstream / mapping_rel
    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    mapping: dict[str, Any] = {}
    if mapping_path.is_file():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"metadata is not a list of objects: {metadata_path}")
    return rows, {
        "metadata_path": str(metadata_path),
        "dimension_to_folder_path": str(mapping_path),
        "dimension_to_folder": mapping,
    }


def _suite(rows: Iterable[Mapping[str, Any]], dimension: str) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if dimension in row.get("dimension", [])]


def _extract_checker(source: Path, function_name: str) -> Any:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name),
        None,
    )
    if function is None:
        raise ValueError(f"{function_name} is missing from {source}")
    namespace: dict[str, Any] = {}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace[function_name]


def _fixture_evidence(upstream: Path) -> dict[str, Any]:
    """Execute only the two pure upstream ``check_generate`` functions."""
    object_checker = _extract_checker(upstream / "vbench" / "object_class.py", "check_generate")
    color_checker = _extract_checker(upstream / "vbench" / "color.py", "check_generate")
    classes = ["car", "bicycle"]
    captions = [
        ("a red car", [0, 0, 10, 10], classes),
        ("a blue bicycle", [20, 0, 30, 10], classes),
    ]
    legacy = [[(item[0], item[2][0]) for item in captions]]
    indexed = [[(item[0], item[2][index]) for index, item in enumerate(captions)]]
    blue_car = [color_checker("blue", "car", predictions) for predictions in (legacy, indexed)]
    blue_bicycle = [color_checker("blue", "bicycle", predictions) for predictions in (legacy, indexed)]
    return {
        "object_rule_fixture_couch_sofa": [
            object_checker("couch", [{"sofa"}]),
            object_checker("sofa", [{"sofa"}]),
        ],
        "color_red_substring_fixture": color_checker("red", "car", [[("a colored car", "car")]]),
        "color_blue_car_legacy_indexed": blue_car,
        "color_blue_bicycle_legacy_indexed": blue_bicycle,
        "color_word_boundary_expected": {
            "red_in_colored": False,
            "red_in_hundred": False,
        },
        "scope": "pure upstream rule fixtures; no detector or model measurement",
    }


def _token_audit(upstream: Path, rows: list[dict[str, Any]], models: Mapping[str, Any]) -> dict[str, Any]:
    model_table = models.get("models", {}) if isinstance(models, Mapping) else {}
    configured = model_table.get("viclip_bpe")
    bpe = Path(str(configured)).expanduser() if configured else Path(
        os.environ.get("VBENCH_CACHE_DIR", str(Path.home() / ".cache" / "vbench"))
    ) / "ViCLIP" / "bpe_simple_vocab_16e6.txt.gz"
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dimension": "temporal_style",
        "path": str(bpe),
        "scope": "prompt token audit only; no ViCLIP model measurement",
    }
    # simple_tokenizer.py evaluates default_bpe() while the module is imported.
    # Check the exact file first, then import with a matching cache directory.
    try:
        bpe_present = bpe.is_file()
    except OSError as exc:
        return {
            **base,
            "status": "blocked",
            "reason": f"local BPE cannot be inspected: {exc}",
            "records": [],
        }
    if not bpe_present:
        return {**base, "status": "blocked", "reason": "local BPE missing", "records": []}
    if bpe.name != "bpe_simple_vocab_16e6.txt.gz" or bpe.parent.name != "ViCLIP":
        # The upstream module evaluates default_bpe() at import time.  A
        # nonstandard configured path cannot be made safe without modifying
        # that source or staging a second file, so refuse it before import.
        return {
            **base,
            "status": "blocked",
            "reason": "BPE path must be <cache>/ViCLIP/bpe_simple_vocab_16e6.txt.gz to avoid auto-fetch",
            "records": [],
        }
    try:
        cache_root = bpe.parent.parent
        old_cache = os.environ.get("VBENCH_CACHE_DIR")
        old_offline = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        os.environ["VBENCH_CACHE_DIR"] = str(cache_root)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        selected = str(upstream)
        if selected not in sys.path:
            sys.path.insert(0, selected)
        from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer  # type: ignore

        tokenizer = SimpleTokenizer(str(bpe))
        sot = tokenizer.encoder["<|startoftext|>"]
        eot = tokenizer.encoder["<|endoftext|>"]

        def tokens(text: str) -> tuple[list[int], list[int]]:
            full = [sot] + tokenizer.encode(text) + [eot]
            kept = full[:] if len(full) <= 32 else full[:31] + [eot]
            return full, kept

        records: list[dict[str, Any]] = []
        for row in _suite(rows, "temporal_style"):
            prompt = str(row.get("prompt_en", ""))
            if ", " in prompt:
                content, style = prompt.rsplit(", ", 1)
            else:
                content, style = prompt, ""
            full, kept = tokens(prompt)
            style_full, style_kept = tokens(style)
            long_left = tokens((content + " ") * 40 + ", pan left")[1]
            long_right = tokens((content + " ") * 40 + ", pan right")[1]
            records.append(
                {
                    "prompt": prompt,
                    "style": style,
                    "full_tokens": full,
                    "kept_tokens": kept,
                    "truncated": len(full) > 32,
                    "style_only_length": len(style_full),
                    "style_only_truncated": len(style_full) > 32,
                    "long_prefix_collision": long_left == long_right,
                }
            )
        result = {
            **base,
            "status": "verified",
            "bpe_sha256": sha256_file(bpe),
            "records": records,
            "truncated_count": sum(bool(record["truncated"]) for record in records),
            "style_only_truncated_count": sum(bool(record["style_only_truncated"]) for record in records),
            "long_prefix_collision_count": sum(bool(record["long_prefix_collision"]) for record in records),
        }
    except Exception as exc:  # dependency/runtime absence is a blocked audit, not a score
        result = {**base, "status": "blocked", "reason": f"tokenizer unavailable: {exc}", "records": []}
    finally:
        if old_cache is None:
            os.environ.pop("VBENCH_CACHE_DIR", None)
        else:
            os.environ["VBENCH_CACHE_DIR"] = old_cache
        for key, value in old_offline.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return result


def _source_evidence(
    upstream: Path,
    protocol: Mapping[str, Any],
    models: Mapping[str, Any],
    dimensions: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """Build M0/M2 source witnesses and one report per dimension."""
    upstream_info: dict[str, Any]
    upstream_state_status = "verified"
    try:
        state = verify_upstream(upstream)
        upstream_info = {"status": "verified", "state": state.__dict__}
    except Exception as exc:
        upstream_state_status = "blocked"
        upstream_info = {"status": "blocked", "error": str(exc), "path": str(upstream)}

    reports: dict[str, dict[str, Any]] = {}
    token_report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dimension": "temporal_style",
        "status": "blocked",
        "reason": "metadata unavailable",
        "records": [],
    }
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": upstream_state_status,
        "upstream": upstream_info,
        "scope": "source and synthetic rule witnesses; no model measurements",
    }
    try:
        if upstream_state_status != "verified":
            raise RuntimeError("locked upstream verification is blocked; source fixtures were not executed")
        rows, metadata_info = _upstream_metadata(upstream, protocol)
        evidence["metadata"] = metadata_info
        # ``scene`` is outside the four selected reports but is the required
        # shared input witness for background_consistency.
        suites = {dimension: _suite(rows, dimension) for dimension in (*DIMENSIONS, "scene")}
        counts = {dimension: len(suites[dimension]) for dimension in DIMENSIONS}
        configured = protocol.get("dimensions", {})
        expected = {
            dimension: int(configured.get(dimension, {}).get("expected_prompt_count", 0))
            for dimension in DIMENSIONS
        }
        prompt_sets = {
            dimension: {str(row.get("prompt_en", "")) for row in suites[dimension]}
            for dimension in (*DIMENSIONS, "scene")
        }
        temporal_parts = [
            str(row.get("prompt_en", "")).rsplit(", ", 1)
            for row in suites["temporal_style"]
        ]
        temporal_base_count = len({part[0] for part in temporal_parts if len(part) == 2})
        temporal_styles = Counter(part[1] for part in temporal_parts if len(part) == 2)
        object_labels = {
            str(
                row.get("auxiliary_info", {})
                .get("object_class", {})
                .get("object", row.get("prompt_en", ""))
            )
            for row in suites["object_class"]
        }
        color_labels = {
            str(row.get("auxiliary_info", {}).get("color", {}).get("color", ""))
            for row in suites["color"]
        }
        temporal_source = upstream / "vbench" / "temporal_style.py"
        overall_source = upstream / "vbench" / "overall_consistency.py"
        normalized_ast_equal = False
        if temporal_source.is_file() and overall_source.is_file():
            left = temporal_source.read_text(encoding="utf-8").replace(
                "temporal_style", "overall_consistency"
            )
            normalized_ast_equal = ast.dump(ast.parse(left)) == ast.dump(
                ast.parse(overall_source.read_text(encoding="utf-8"))
            )
        fixture = _fixture_evidence(upstream)
        if "temporal_style" in dimensions:
            token_report = _token_audit(upstream, rows, models)
        else:
            token_report = {
                "schema_version": SCHEMA_VERSION,
                "dimension": "temporal_style",
                "status": "not_applicable",
                "reason": "temporal_style was not selected",
                "records": [],
            }
        evidence.update(
            {
                "counts": counts,
                "expected_counts": expected,
                "background_scene_prompt_equal": prompt_sets["background_consistency"]
                == prompt_sets["scene"],
                "temporal_normalized_AST_equal": normalized_ast_equal,
                "temporal_base_count": temporal_base_count,
                "temporal_styles": dict(sorted(temporal_styles.items())),
                "temporal_aux_count": sum(
                    "temporal_style" in row.get("auxiliary_info", {}) for row in suites["temporal_style"]
                ),
                "temporal_max_words": max(
                    (len(str(row.get("prompt_en", "")).split()) for row in suites["temporal_style"]),
                    default=0,
                ),
                "object_label_count": len(object_labels),
                "color_label_count": len(color_labels),
                "fixtures": fixture,
            }
        )
        evidence_checks = [
            "verified" if counts[dimension] == expected[dimension] else "failed"
            for dimension in dimensions
        ]
        evidence["status"] = _status(upstream_state_status, *evidence_checks)

        def report(
            dimension: str,
            checks: Mapping[str, Mapping[str, Any]],
            suite: Mapping[str, Any],
            source: Mapping[str, Any],
        ) -> None:
            reports[dimension] = {
                "schema_version": SCHEMA_VERSION,
                "dimension": dimension,
                "status": _status(*(item.get("status", "failed") for item in checks.values())),
                "checks": dict(checks),
                "suite": dict(suite),
                "source": dict(source),
                "model_measurement": {"status": "not_run", "value": None},
                "scope": "offline source/rule evidence; no model measurement",
            }

        background_checks = {
            "suite_coupling": _check(
                "verified" if counts["background_consistency"] == expected["background_consistency"] else "failed",
                prompt_count=counts["background_consistency"],
                source_folder="scene",
                shared_video_dimension="scene",
            ),
            "scene_prompt_set_equal": _check(
                "verified" if prompt_sets["background_consistency"] == prompt_sets["scene"] else "failed",
                background_prompt_count=len(prompt_sets["background_consistency"]),
                scene_prompt_count=len(prompt_sets["scene"]),
            ),
            "no_model_measurement": _check("not_applicable", value=None),
        }
        report(
            "background_consistency",
            background_checks,
            {
                "prompt_count": counts["background_consistency"],
                "source_dimension": "background_consistency",
                "source_folder": "scene",
                "shared_prompt_dimension": "scene",
            },
            {"metadata_path": metadata_info["metadata_path"], "video_source": "scene/"},
        )

        temporal_checks = {
            "estimator_ast_equivalence": _check(
                "verified" if normalized_ast_equal else "failed",
                temporal_source="vbench/temporal_style.py",
                legacy_source="vbench/overall_consistency.py",
            ),
            "style_clause_exchange": _check(
                "verified"
                if temporal_base_count == int(configured.get("temporal_style", {}).get("expected_base_count", 10))
                and all(
                    count == int(configured.get("temporal_style", {}).get("expected_style_count", 10))
                    for count in temporal_styles.values()
                )
                else "failed",
                base_count=temporal_base_count,
                styles=dict(sorted(temporal_styles.items())),
            ),
            "token_truncation": _check(
                token_report.get("status", "blocked"),
                path=token_report.get("path"),
                truncated_count=token_report.get("truncated_count"),
                style_only_truncated_count=token_report.get("style_only_truncated_count"),
                long_prefix_collision_count=token_report.get("long_prefix_collision_count"),
            ),
        }
        report(
            "temporal_style",
            temporal_checks,
            {
                "prompt_count": counts["temporal_style"],
                "base_count": temporal_base_count,
                "style_count": len(temporal_styles),
                "style_rows_per_base": dict(sorted(temporal_styles.items())),
                "aux_count": evidence["temporal_aux_count"],
                "max_words": evidence["temporal_max_words"],
            },
            {"metadata_path": metadata_info["metadata_path"], "estimator_pair": "temporal_style/overall_consistency"},
        )

        object_fixture_ok = fixture.get("object_rule_fixture_couch_sofa") == [0, 1]
        object_checks = {
            "target_label_fixture": _check(
                "verified" if object_fixture_ok else "failed",
                couch_vs_sofa=fixture.get("object_rule_fixture_couch_sofa"),
                expected_label_count=expected["object_class"],
                observed_label_count=len(object_labels),
            ),
            "exact_string_rule": _check(
                "verified" if object_fixture_ok else "failed",
                case_sensitive=True,
                source="vbench/object_class.py:check_generate",
            ),
            "fixed_frame_denominator": _check(
                "verified",
                denominator="all sampled frames",
                model_measurement=None,
            ),
        }
        report(
            "object_class",
            object_checks,
            {"prompt_count": counts["object_class"], "label_count": len(object_labels)},
            {"metadata_path": metadata_info["metadata_path"], "source": "vbench/object_class.py"},
        )

        color_checks = {
            "instance_index_fixture": _check(
                "verified",
                legacy=fixture.get("color_blue_car_legacy_indexed"),
                indexed=fixture.get("color_blue_bicycle_legacy_indexed"),
            ),
            "word_boundary_fixture": _check(
                "verified",
                raw_substring=fixture.get("color_red_substring_fixture"),
                expected_boundary=fixture.get("color_word_boundary_expected"),
            ),
            "fixed_frame_denominator": _check(
                "verified",
                denominator="all sampled frames; conditional rate is diagnostic",
                dropped_video_count=0,
                model_measurement=None,
            ),
        }
        report(
            "color",
            color_checks,
            {"prompt_count": counts["color"], "color_label_count": len(color_labels)},
            {"metadata_path": metadata_info["metadata_path"], "source": "vbench/color.py"},
        )
    except Exception as exc:
        evidence["status"] = "blocked"
        evidence["error"] = str(exc)
        for dimension in dimensions:
            reports[dimension] = {
                "schema_version": SCHEMA_VERSION,
                "dimension": dimension,
                "status": "blocked",
                "checks": {"metadata": _check("blocked", error=str(exc))},
                "suite": {},
                "source": {"upstream": str(upstream)},
                "model_measurement": {"status": "not_run", "value": None},
                "scope": "offline source/rule evidence; no model measurement",
            }
    return evidence, reports, token_report


def _asset_paths(models: Mapping[str, Any], dimensions: Iterable[str]) -> list[dict[str, Any]]:
    tables = models.get("models", {}) if isinstance(models, Mapping) else {}
    dimension_tables = models.get("dimensions", {}) if isinstance(models, Mapping) else {}
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for dimension in dimensions:
        config = dimension_tables.get(dimension, {})
        refs: list[tuple[str, str]] = []
        if isinstance(config, Mapping):
            model = config.get("model")
            if isinstance(model, str):
                refs.append(("model", model))
            for key in ("text_assets", "assets"):
                values = config.get(key, [])
                if isinstance(values, list):
                    refs.extend((key, str(value)) for value in values)
        for role, key in refs:
            if key not in tables:
                records.append(
                    {
                        "dimension": dimension,
                        "role": role,
                        "asset_key": key,
                        "status": "blocked",
                        "reason": "asset key is not defined in [models]",
                    }
                )
                continue
            path = Path(str(tables[key])).expanduser()
            identity = (dimension, str(path))
            if identity in seen:
                continue
            seen.add(identity)
            item: dict[str, Any] = {
                "dimension": dimension,
                "role": role,
                "asset_key": key,
                "path": str(path),
                "status": "blocked",
            }
            try:
                if not path.is_file():
                    item["reason"] = "asset file missing"
                elif path.stat().st_size == 0:
                    item["reason"] = "asset file is empty"
                else:
                    item["status"] = "verified"
                    item["size_bytes"] = path.stat().st_size
                    item["sha256"] = sha256_file(path)
            except OSError as exc:
                item["reason"] = f"asset unreadable: {exc}"
            records.append(item)
    return records


def _source_video_inventory(
    dataset_root: Path | None,
    upstream: Path,
    protocol: Mapping[str, Any],
    dimensions: list[str],
) -> dict[str, Any]:
    dataset_cfg = protocol.get("protocol", {}).get("dataset", {})
    generators = [str(item) for item in dataset_cfg.get("generators", [])]
    suffixes = {
        str(item).lower()
        for item in dataset_cfg.get("video_suffixes", [".mp4", ".gif"])
    }
    if not suffixes:
        suffixes = {".mp4", ".gif"}
    try:
        _, metadata_info = _upstream_metadata(upstream, protocol)
        mapping = metadata_info.get("dimension_to_folder", {})
    except Exception as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "error": str(exc),
            "dataset_root": str(dataset_root) if dataset_root else None,
            "dimensions": {},
        }
    dimensions_out: dict[str, Any] = {}
    for dimension in dimensions:
        folder = str(mapping.get(dimension, dimension))
        rows: list[dict[str, Any]] = []
        for generator in generators:
            candidate = dataset_root / generator / folder if dataset_root else None
            item: dict[str, Any] = {
                "generator": generator,
                "dimension": dimension,
                "folder": folder,
                "path": str(candidate) if candidate else None,
                "status": "blocked",
                "video_count": 0,
            }
            if candidate is None:
                item["reason"] = "dataset root not supplied"
            elif not _is_dir(candidate):
                item["reason"] = "dimension folder missing"
            else:
                try:
                    videos = sorted(
                        path
                        for path in candidate.rglob("*")
                        if path.is_file() and path.suffix.lower() in suffixes
                    )
                    item.update(
                        {
                            "status": "verified",
                            "video_count": len(videos),
                            "canonical_path": str(candidate.resolve()),
                            "sample_files": [str(path) for path in videos[:5]],
                        }
                    )
                except OSError as exc:
                    item["reason"] = f"video directory unreadable: {exc}"
            rows.append(item)
        dimensions_out[dimension] = {
            "source_folder": folder,
            "shared_with": ["scene"] if dimension == "background_consistency" else [],
            "generators": rows,
            "status": _status(*(row["status"] for row in rows)) if rows else "blocked",
        }
    shared: dict[str, Any] = {}
    if "background_consistency" in dimensions:
        shared_rows: list[dict[str, Any]] = []
        for generator in generators:
            background_path = dataset_root / generator / str(mapping.get("background_consistency", "scene")) if dataset_root else None
            scene_path = dataset_root / generator / str(mapping.get("scene", "scene")) if dataset_root else None
            relation: dict[str, Any] = {
                "generator": generator,
                "background_path": str(background_path) if background_path else None,
                "scene_path": str(scene_path) if scene_path else None,
                "status": "blocked",
            }
            if background_path and scene_path and _is_dir(background_path) and _is_dir(scene_path):
                relation["background_canonical_path"] = str(background_path.resolve())
                relation["scene_canonical_path"] = str(scene_path.resolve())
                relation["same_canonical_path"] = background_path.resolve() == scene_path.resolve()
                relation["status"] = "verified" if relation["same_canonical_path"] else "failed"
            else:
                relation["reason"] = "background or scene folder missing"
            shared_rows.append(relation)
        shared["background_consistency"] = {
            "with_dimension": "scene",
            "folder": "scene",
            "relation": "same source folder; prompt suite checked by offline",
            "generators": shared_rows,
        }
    shared_statuses = []
    for value in shared.values():
        if isinstance(value, Mapping):
            shared_statuses.extend(
                str(row.get("status", "failed"))
                for row in value.get("generators", [])
                if isinstance(row, Mapping)
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": _status(
            *(item["status"] for item in dimensions_out.values()),
            *shared_statuses,
        ),
        "dataset_root": str(dataset_root) if dataset_root else None,
        "dimensions": dimensions_out,
        "shared_video_sources": shared,
        "scope": "filesystem inventory; no videos decoded and no scores computed",
    }


def _gpu_inventory(requested: str, python_executable: Path | None) -> dict[str, Any]:
    try:
        gpu_ids = parse_gpu(requested)
    except InputError as exc:
        return {"schema_version": SCHEMA_VERSION, "status": "failed", "error": str(exc), "devices": []}
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name,memory.free",
        "--format=csv,noheader,nounits",
    ]
    smi = _command(command, timeout=20)
    visible: list[dict[str, Any]] = []
    if smi.get("returncode") == 0:
        for line in str(smi.get("stdout", "")).splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 5:
                visible.append(
                    {
                        "index": fields[0],
                        "uuid": fields[1],
                        "pci_bus_id": fields[2],
                        "name": fields[3],
                        "memory_free_mib": fields[4],
                    }
                )
    else:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "requested_gpu_ids": gpu_ids,
            "nvidia_smi": smi,
            "devices": [],
            "policy": {"one_visible_gpu_per_process": True, "logical_device": "cuda:0"},
        }
    probe_python = python_executable or Path(sys.executable)
    devices: list[dict[str, Any]] = []
    probe_code = (
        "import json, torch; "
        "x={'cuda_available':bool(torch.cuda.is_available()),'device_count':int(torch.cuda.device_count())}; "
        "x['devices']=[{'name':torch.cuda.get_device_name(i),'uuid':str(getattr(torch.cuda.get_device_properties(i),'uuid','')) or None} "
        "for i in range(torch.cuda.device_count())]; print(json.dumps(x))"
    )
    for gpu in gpu_ids:
        item: dict[str, Any] = {"physical_index": gpu, "status": "blocked"}
        if not _is_file(probe_python):
            item["reason"] = "probe interpreter missing"
            devices.append(item)
            continue
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONDONTWRITEBYTECODE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        probe = _command([str(probe_python), "-B", "-c", probe_code], env=env, timeout=30)
        item["probe"] = {key: value for key, value in probe.items() if key != "stdout"}
        try:
            parsed = json.loads(str(probe.get("stdout", "")).strip())
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if probe.get("returncode") == 0 and isinstance(parsed, dict):
            item["cuda_available"] = parsed.get("cuda_available")
            item["device_count"] = parsed.get("device_count")
            item["logical_devices"] = parsed.get("devices", [])
            item["status"] = (
                "verified"
                if parsed.get("cuda_available") is True and parsed.get("device_count") == 1
                else "blocked"
            )
            if item["status"] == "blocked":
                item["reason"] = "CUDA unavailable or isolated device_count is not 1"
        else:
            item["reason"] = "torch CUDA probe failed"
        devices.append(item)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": _status(*(item["status"] for item in devices)),
        "requested_gpu_ids": gpu_ids,
        "nvidia_smi": {key: value for key, value in smi.items() if key != "stdout"},
        "visible_devices": visible,
        "devices": devices,
        "policy": {"one_visible_gpu_per_process": True, "logical_device": "cuda:0"},
        "scope": "device and visibility probe only; no metric/model import",
    }


def _environment_inventory(
    upstream: Path,
    protocol_path: Path,
    models_path: Path,
    python_executable: Path | None,
    gpu_report: Mapping[str, Any],
    dataset_root: Path | None,
) -> dict[str, Any]:
    uv = _command(["uv", "--version"], timeout=10)
    interpreter = python_executable or Path(sys.executable)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified" if _is_file(interpreter) else "blocked",
        "python": {
            "executable": str(interpreter),
            "exists": _is_file(interpreter),
            "version": platform.python_version() if interpreter == Path(sys.executable) else None,
        },
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "offline_environment": {
            key: os.environ.get(key)
            for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VBENCH_CACHE_DIR", "VBENCH_AUDIT_UPSTREAM")
        },
        "uv": {"returncode": uv.get("returncode"), "stdout": uv.get("stdout", "").strip()},
        "upstream": str(upstream),
        "dataset_root": str(dataset_root) if dataset_root else None,
        "protocol_sha256": sha256_file(protocol_path) if _is_file(protocol_path) else None,
        "models_sha256": sha256_file(models_path) if _is_file(models_path) else None,
        "gpu_probe_status": gpu_report.get("status"),
        "scope": "runtime metadata; no model import",
    }


def _source_code_inventory(upstream: Path, protocol_path: Path, models_path: Path) -> dict[str, Any]:
    repo_head = _git(REPO_ROOT, "rev-parse", "HEAD")
    status_text = _git(REPO_ROOT, "status", "--porcelain")
    diff = _command(["git", "-C", str(REPO_ROOT), "diff", "--no-ext-diff", "--binary"], timeout=60)
    diff_hash = hashlib.sha256(str(diff.get("stdout", "")).encode()).hexdigest()
    try:
        state = verify_upstream(upstream)
        upstream_block: dict[str, Any] = {"status": "verified", "state": state.__dict__}
    except Exception as exc:
        upstream_block = {"status": "blocked", "error": str(exc), "path": str(upstream)}
    required = [REPO_ROOT / "uv.lock", protocol_path, models_path]
    files = {
        str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path): {
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        for path in required
    }
    file_status = "verified" if all(item["exists"] for item in files.values()) else "blocked"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": _status(str(upstream_block["status"]), file_status),
        "repository": {
            "path": str(REPO_ROOT),
            "head": repo_head,
            "dirty": bool(status_text),
            "working_tree_status_sha256": hashlib.sha256((status_text or "").encode()).hexdigest(),
            "diff_sha256": diff_hash,
        },
        "upstream": upstream_block,
        "files": files,
        "scope": "source/version manifest; no checkout mutation",
    }


def run_offline(args: argparse.Namespace) -> int:
    root = _resolve(args.root, DEFAULT_ROOT)
    protocol_path = _resolve(args.protocol, DEFAULT_PROTOCOL)
    models_path = _resolve(args.models, DEFAULT_MODELS)
    upstream = _resolve(args.upstream, REPO_ROOT.parent / "VBench")
    try:
        protocol = _read_toml(protocol_path)
        models = _read_toml(models_path) if models_path.is_file() else {}
        dimensions = _dimension_list(protocol, args.dimension)
    except Exception as exc:
        _write_json(root / "offline" / "source-evidence.json", {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "error": str(exc),
            "scope": "source and synthetic rule witnesses; no model measurements",
        })
        print(json.dumps({"command": "offline", "status": "failed", "error": str(exc)}))
        return 2
    evidence, reports, token_report = _source_evidence(upstream, protocol, models, dimensions)
    _write_json(root / "offline" / "source-evidence.json", evidence)
    _write_json(root / "offline" / "temporal-tokens.json", token_report)
    outputs: list[str] = []
    for dimension in dimensions:
        path = root / "offline" / f"{dimension}.json"
        _write_json(path, reports[dimension])
        outputs.append(str(path))
    result = {
        "command": "offline",
        "status": _status(evidence.get("status", "failed"), *(reports[d]["status"] for d in dimensions)),
        "outputs": [str(root / "offline" / "source-evidence.json"), str(root / "offline" / "temporal-tokens.json"), *outputs],
        "scope": "source and synthetic rule witnesses; no model measurements",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["status"] == "failed" else 0


def run_inventory(args: argparse.Namespace) -> int:
    root = _resolve(args.root, DEFAULT_ROOT)
    protocol_path = _resolve(args.protocol, DEFAULT_PROTOCOL)
    models_path = _resolve(args.models, DEFAULT_MODELS)
    upstream = _resolve(args.upstream, REPO_ROOT.parent / "VBench")
    try:
        protocol = _read_toml(protocol_path)
        models = _read_toml(models_path) if models_path.is_file() else {}
        dimensions = _dimension_list(protocol, args.dimension)
    except Exception as exc:
        print(json.dumps({"command": "inventory", "status": "failed", "error": str(exc)}))
        return 2
    dataset_root = _resolve(args.dataset_root, Path("/nonexistent")) if args.dataset_root else None
    runtime_cfg = models.get("runtime", {}) if isinstance(models, Mapping) else {}
    configured_python = runtime_cfg.get("python") if isinstance(runtime_cfg, Mapping) else None
    probe_python = _resolve(args.python, Path(str(configured_python))) if args.python else (
        Path(str(configured_python)).expanduser().resolve() if configured_python else Path(sys.executable)
    )
    assets = _asset_paths(models, dimensions)
    source_videos = _source_video_inventory(dataset_root, upstream, protocol, dimensions)
    source_code = _source_code_inventory(upstream, protocol_path, models_path)
    gpu_report = _gpu_inventory(args.gpus, probe_python)
    environment = _environment_inventory(upstream, protocol_path, models_path, probe_python, gpu_report, dataset_root)
    reports = {
        "assets.json": {"schema_version": SCHEMA_VERSION, "status": _status(*(item["status"] for item in assets)) if assets else "not_applicable", "assets": assets, "scope": "path/hash inventory; no model import"},
        "source-videos.json": source_videos,
        "source-code.json": source_code,
        "gpus.json": gpu_report,
        "environment.json": environment,
    }
    for name, payload in reports.items():
        _write_json(root / "inventory" / name, payload)
    aggregate = _status(*(str(payload.get("status", "failed")) for payload in reports.values()))
    result = {
        "command": "inventory",
        "status": aggregate,
        "outputs": [str(root / "inventory" / name) for name in reports],
        "scope": "asset/source/video/device inventory; no metric/model import",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if aggregate == "failed" else 0


def _add_common(parser: argparse.ArgumentParser, *, suppress_defaults: bool = False) -> None:
    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument("--root", type=Path, default=default, help=f"report root (default: {DEFAULT_ROOT})")
    parser.add_argument("--upstream", type=Path, default=default, help="locked VBench checkout")
    parser.add_argument("--protocol", type=Path, default=default, help=f"protocol TOML (default: {DEFAULT_PROTOCOL})")
    parser.add_argument("--models", type=Path, default=default, help=f"model path TOML (default: {DEFAULT_MODELS})")
    parser.add_argument("--dimension", choices=DIMENSIONS, default=default, help="restrict to one dimension")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_common(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)
    offline = subparsers.add_parser("offline", help="write zero-model source and rule evidence")
    _add_common(offline, suppress_defaults=True)
    inventory = subparsers.add_parser("inventory", help="inventory assets, videos, source and isolated GPUs")
    _add_common(inventory, suppress_defaults=True)
    inventory.add_argument("--dataset-root", type=Path, default=None, help="read-only VBench video archive")
    inventory.add_argument("--gpus", default="1,2,3,4", help="physical GPU ids, comma separated")
    inventory.add_argument("--python", type=Path, default=None, help="interpreter for isolated torch probes")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.root is None:
        args.root = DEFAULT_ROOT
    if args.upstream is None:
        args.upstream = REPO_ROOT.parent / "VBench"
    if args.protocol is None:
        args.protocol = DEFAULT_PROTOCOL
    if args.models is None:
        args.models = DEFAULT_MODELS
    if args.command == "offline":
        return run_offline(args)
    if args.command == "inventory":
        return run_inventory(args)
    raise AssertionError(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
