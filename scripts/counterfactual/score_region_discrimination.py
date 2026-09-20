"""Score the frozen PNG dataset with Official, shipped repair, and masked repair.

Every column sees the same RGB uint8 tensors and DINO transform. The dataset's
construction masks are used only by the integrity verifier, never a scorer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace

from .build_region_discrimination import verify
from .common import ROOT, sha256_file
from .region_discrimination import LEVELS, POSITIONS
from .subject_artifacts import (artifact_path, new_output, read_jsonl, read_png_sequence,
                                write_json, write_jsonl, write_npz)
from .subject_region_statistics import BACKENDS, analyze


class ClipExtractor:
    """Reuse the metric batch API for already-decoded lossless PNG frames."""
    def __init__(self, extractor, frames):
        self.extractor = extractor
        self.module = SimpleNamespace(load_video=lambda _: frames)
        self.patches = None
        self.transformed_size = None

    def patches_from_frames(self, frames):
        if self.patches is None:
            self.patches = self.extractor.patches_from_frames(frames)
            self.transformed_size = self.extractor.transformed_size
        return self.patches


class CachedScoringMasks:
    """Cache only between zero/exclude for this exact variant, never across clips."""
    def __init__(self, provider):
        self.provider = provider
        self.value = None

    def masks_for(self, video, frames, phrase):
        if self.value is None:
            self.value = self.provider.masks_for(video, frames, phrase)
        return self.value


def score_dataset(dataset: Path, output: Path, extractor, provider, prompts: dict, *, device: str) -> list[dict]:
    import torch
    from subject_consistency.backends.vbench import official_diagnostics
    from subject_consistency.metric import evaluate_masked_batch, subject_consistency_diagnostics

    rows = []
    for entry in read_jsonl(dataset / "index.jsonl"):
        manifest = json.loads(artifact_path(dataset, entry["manifest"]).read_text())
        base = manifest["base"]
        uid = base["video_uid"]
        prompt = prompts.get(uid)
        if prompt and manifest["status"] == "accepted":
            if prompt["source_frame_sha256"] != manifest["variants"]["clean"][0]["sha256"]:
                raise ValueError(f"prompt confirmation is not for this clean source: {uid}")
        # The clean measurement is shared across all temporal positions, not
        # rerun to create three apparently independent clean observations.
        clean_results = None
        for position in POSITIONS:
            for level in LEVELS:
                identity = {"base_id": base["base_id"], "video_uid": uid,
                            "source_prompt_id": base.get("prompt_id", base["prompt_en"]),
                            "position": position, "level": level}
                if level == "clean" and clean_results is not None:
                    rows.extend({**row, **identity} for row in clean_results)
                    continue
                results = []
                if manifest["status"] != "accepted":
                    for backend in BACKENDS:
                        results.append({**identity, "backend": backend, "status": "unsupported", "score": None,
                            "failure_reason": ",".join(manifest["rejection_reasons"]), "diagnostics": None})
                else:
                    variant = "clean" if level == "clean" else f"{position}/{level}"
                    frames = torch.from_numpy(read_png_sequence(dataset, manifest["variants"][variant])).permute(0, 3, 1, 2)
                    name = f"{base['base_id']}__{variant.replace('/', '__')}"
                    video = Path(name)
                    try:
                        features = extractor.features_from_frames(frames)
                        for backend, fn in (("official", official_diagnostics), ("aggregation", subject_consistency_diagnostics)):
                            diagnostics = fn(features)
                            results.append({**identity, "backend": backend, "status": "succeeded",
                                            "score": diagnostics.final_score, "diagnostics": diagnostics.to_dict(), "failure_reason": None})
                    except Exception as exc:
                        for backend in ("official", "aggregation"):
                            results.append({**identity, "backend": backend, "status": "failed", "score": None,
                                            "failure_reason": f"{type(exc).__name__}: {exc}", "diagnostics": None})
                    if provider is None or prompt is None:
                        for policy in ("zero", "exclude"):
                            results.append({**identity, "backend": f"masked_{policy}", "status": "not_run", "score": None,
                                            "failure_reason": "missing_human_confirmed_localizer_prompt", "diagnostics": None})
                    else:
                        provider.clip_to_video_uid[name] = uid
                        cached = CachedScoringMasks(provider)
                        adapter = ClipExtractor(extractor, frames)
                        metadata = {name: {"prompt": base["prompt_en"], "subject_phrase": prompt["phrase"]}}
                        zero_diagnostics = None
                        for policy in ("zero", "exclude"):
                            value = evaluate_masked_batch([video], metadata, device, {}, cached, extractor=adapter,
                                phrase_field="subject_phrase", max_frames=None, missing_policy=policy)[0]
                            if policy == "zero":
                                zero_diagnostics = value["diagnostics"]
                            elif value["status"] != "succeeded" and zero_diagnostics is not None:
                                value["diagnostics"] = {key: zero_diagnostics[key] for key in (
                                    "num_frames", "num_present_frames", "num_missing_frames", "missing_fraction", "coverage")}
                                value["diagnostics"].update(missing_policy="exclude", score_status="undefined")
                            results.append({**identity, "backend": f"masked_{policy}", "status": value["status"],
                                            "score": value["score"], "failure_reason": value["failure_reason"],
                                            "diagnostics": value["diagnostics"]})
                        if cached.value is not None:
                            masks = cached.value
                            path = output / "scoring_masks" / f"{name}.npz"
                            write_npz(path, masks=masks.instance_masks.cpu().numpy()[:, 0].astype("uint8"),
                                      present=masks.instance_present.cpu().numpy(),
                                      metadata_json=json.dumps({**provider.provenance, "prompt_sha256": prompt["sha256"]}, sort_keys=True))
                            for value in results:
                                if value["backend"].startswith("masked_"):
                                    value["scoring_mask_file"] = {"path": str(path.relative_to(output)), "sha256": sha256_file(path)}
                rows.extend(results)
                if level == "clean":
                    clean_results = results
        # Keep an observable checkpoint without mutating any prior run. An
        # incomplete grid cannot be analyzed as a complete experiment.
        write_jsonl(output / "scores.jsonl", rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--localizer", type=Path)
    parser.add_argument("--mobilesam-checkpoint", type=Path)
    parser.add_argument("--whole-frame-only", action="store_true", help="Record masked columns as NOT RUN; never infer human prompts.")
    parser.add_argument("--dino-repo")
    parser.add_argument("--dino-weight")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--analyze-scores", type=Path, help="Analyze an existing complete score grid without models.")
    args = parser.parse_args()
    integrity = verify(args.dataset)
    output = new_output(args.output)
    if args.analyze_scores:
        rows = read_jsonl(args.analyze_scores)
        run = {"scores_sha256": sha256_file(args.analyze_scores)}
    else:
        from subject_consistency.localizer import MobileSamSubjectMaskProvider, load_prompts
        from subject_consistency.metric import build_dino_config, upstream_path
        from subject_consistency.models import OfficialDinoPatchExtractor
        from .generate_subject_masks import deterministic_setup

        if not args.whole_frame_only and (not args.localizer or not args.mobilesam_checkpoint):
            parser.error("masked experiment requires --localizer and --mobilesam-checkpoint")
        deterministic_setup()
        config = build_dino_config(args.dino_repo, args.dino_weight)
        extractor = OfficialDinoPatchExtractor(args.device, config, upstream_path())
        prompts = load_prompts(args.localizer) if args.localizer else {}
        if not args.whole_frame_only:
            assets = json.loads((ROOT / "configs/subject-repair/assets.lock.json").read_text())
            if sha256_file(args.mobilesam_checkpoint) != assets["mobilesam"]["checkpoint_sha256"]:
                raise ValueError("MobileSAM checkpoint differs from assets.lock.json")
        provider = None if args.whole_frame_only else MobileSamSubjectMaskProvider(
            prompts, clip_to_video_uid={}, checkpoint=args.mobilesam_checkpoint, device=args.device)
        rows = score_dataset(args.dataset, output, extractor, provider, prompts, device=args.device)
        run = {"device": args.device, "python": platform.python_version(),
               "dino_weight_sha256": sha256_file(config["path"]), "upstream": vars(extractor.upstream_state),
               "localizer_sha256": sha256_file(args.localizer) if args.localizer else None,
               "localizer": provider.provenance if provider else None, "real_model_parity": "NOT RUN"}
    try:
        code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        code_sha = None  # A transported source snapshot is identified by file hashes.
    run.update({"dataset_index_sha256": sha256_file(args.dataset / "index.jsonl"), "integrity": integrity,
                "code_sha": code_sha,
                "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p)
                    for directory in (ROOT / "scripts/counterfactual", ROOT / "metrics/subject-consistency/src")
                    for p in sorted(directory.rglob("*.py"))}})
    write_json(output / "run.json", run)
    report = analyze(read_jsonl(args.dataset / "index.jsonl"), rows)
    write_json(output / "statistics.json", report)
    print(json.dumps(report["primary"], indent=2))
    return 1 if any(row["status"] == "failed" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
