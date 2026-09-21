"""Native background cohort: foreground blur and background scene switches.

SegFormer construction masks never go to the scoring localizer. Rough semantic
union is intentional; tiny boundary errors do not require manual approval.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from .common import ROOT, sha256_file
from .generate_subject_masks import SegFormerConstructionLocalizer, deterministic_setup
from .region_discrimination import clean_components, corrupt_image, inward_alpha, window_indices
from .subject_artifacts import new_output, read_jsonl, write_json, write_npz, write_png_sequence


def foreground_and_background_edits(frames, masks, donor, *, sigma=18):
    if frames.dtype != np.uint8 or donor.dtype != np.uint8 or frames.ndim != 4 or donor.ndim != 4:
        raise ValueError("source and donor must be uint8 RGB sequences")
    if masks.shape != frames.shape[:3] or not np.isin(masks, (0, 1)).all():
        raise ValueError("construction mask shape or binary values invalid")
    indices = np.rint(np.linspace(0, len(donor)-1, len(frames))).astype(int)
    foreground, background = [], []
    for frame, mask, index in zip(frames, masks, indices):
        foreground.append(corrupt_image(frame, mask, operator="gaussian", scale=sigma))
        target = cv2.resize(donor[index], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        alpha = inward_alpha(1-mask)[..., None].astype(np.float64)
        edited = np.clip(np.rint(frame.astype(np.float64)*(1-alpha) + target.astype(np.float64)*alpha), 0, 255).astype(np.uint8)
        if not np.array_equal(edited[mask == 1], frame[mask == 1]):
            raise AssertionError("background scene switch changed foreground pixels")
        background.append(edited)
    return np.stack(foreground), np.stack(background), indices.tolist()


def donor_map(entries, *, strategy="next_prompt_cycle"):
    prompts = sorted({r["prompt_id"] for r in entries})
    if len(prompts) < 2:
        raise ValueError("background scene switch needs distinct source prompts")
    if len({r.get('split') for r in entries}) != 1:
        raise ValueError("donors must stay within one split")
    if strategy == 'disjoint_sorted_pairs':
        if len(prompts) % 2:
            raise ValueError('disjoint pairing requires an even number of prompts')
        targets = {p: prompts[i ^ 1] for i, p in enumerate(prompts)}
    elif strategy == 'next_prompt_cycle':
        targets = {p: prompts[(i+1) % len(prompts)] for i, p in enumerate(prompts)}
    else:
        raise ValueError('unknown donor strategy')
    lookup = {(r["prompt_id"], r["generator"], r["seed"]): r for r in entries}
    if len(lookup) != len(entries) or len({r['video_uid'] for r in entries}) != len(entries):
        raise ValueError('duplicate donor identity')
    return {r["video_uid"]: lookup[(targets[r['prompt_id']], r["generator"], r["seed"])] for r in entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=Path("/root/wenbiao_zhao/datasets/vbench-1.0-human-preference/videos"))
    parser.add_argument("--segformer-dir", type=Path, default=Path("/root/wenbiao_zhao/models/subject-repair/segformer-b0-ade20k"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--protocol", type=Path, default=ROOT/"configs/background-repair/construction_dev_v1.json")
    args = parser.parse_args()
    config = ROOT/"configs/background-repair"
    protocol_path = args.protocol
    protocol = json.loads(protocol_path.read_text())
    manifest = config/"natural1720_manifest_v2.jsonl"
    if sha256_file(manifest) != protocol["manifest_sha256"]:
        raise ValueError("native background manifest changed")
    if protocol['split'] not in ('dev', 'test'):
        raise ValueError('invalid construction split')
    entries = [r for r in read_jsonl(manifest) if r["split"] == protocol['split']]
    if len(entries) != protocol['candidate_count']:
        raise ValueError("expected the entire declared input population")
    donors = donor_map(entries, strategy=protocol.get('donor_strategy', 'next_prompt_cycle'))
    deterministic_setup()
    import torch
    from vbench_audit_core.upstream import import_official_module
    torch.set_num_threads(2)
    module, state = import_official_module("background_consistency")
    localizer = SegFormerConstructionLocalizer(args.segformer_dir, device=args.device)
    expected_assets = json.loads((ROOT/"configs/subject-repair/assets.lock.json").read_text())["segformer"]["files"]
    if localizer.provenance["files_sha256"] != expected_assets:
        raise ValueError("SegFormer weight or configuration pin changed")
    ids = [k for k, v in localizer.id2label.items() if v in protocol["foreground_labels"]]
    if len(ids) != len(protocol["foreground_labels"]):
        raise ValueError("ADE20K foreground vocabulary mismatch")
    output = new_output(args.output)
    start = time.time()
    run = {"protocol_sha256": sha256_file(protocol_path), "started_unix": start, "total": len(entries),
           "segformer": {**localizer.provenance, "grabcut": "not used: coarse semantic union",
                         "components": {"minimum_area": 20, "largest_instance_only": False}},
           "upstream": vars(state), "source_sha256": sha256_file(Path(__file__)), "device": args.device,
           "physical_gpu": os.environ.get('CUDA_VISIBLE_DEVICES'), "torch_version": torch.__version__,
           "source_files_sha256": {name: sha256_file(ROOT/'scripts/counterfactual'/name) for name in
               ('build_background_interventions.py', 'generate_subject_masks.py', 'region_discrimination.py', 'subject_artifacts.py')}}
    write_json(output/"run.json", run); write_json(output/"protocol.json", protocol)
    counts = Counter()
    def decode(path):
        return module.load_video(str(path)).permute(0, 2, 3, 1).numpy().astype(np.uint8)
    for index, entry in enumerate(entries):
        uid = entry["video_uid"]
        row = {"base": entry, "base_id": uid, "rejection_reasons": [], "variants": {}}
        try:
            source = args.video_root/entry["relative_video_path"]
            row["video_sha256"] = sha256_file(source)
            frames = decode(source)
            masks = np.stack([clean_components(np.isin(localizer.labels_for(f), ids).astype(np.uint8), person=False) for f in frames])
            area = masks.mean((1, 2)); row["foreground_fraction"] = area.tolist()
            mask_path = output/"construction_masks"/f"{uid}.npz"
            write_npz(mask_path, masks=masks, metadata_json=json.dumps({"role": "construction", "family": "segformer", "source_video_sha256": row["video_sha256"]}))
            row["construction_mask"] = {"path": str(mask_path.relative_to(output)), "sha256": sha256_file(mask_path)}
            if len(frames) < 4:
                row["rejection_reasons"].append("too_few_frames")
            if area.mean() < protocol["mean_foreground_min"]:
                row["rejection_reasons"].append("no_sufficient_foreground")
            if area.mean() > protocol["mean_foreground_max"] or area.max() > protocol["frame_foreground_max"]:
                row["rejection_reasons"].append("insufficient_background")
            if (area >= protocol["frame_presence_min_area"]).mean() < protocol["foreground_frame_presence_min"]:
                row["rejection_reasons"].append("foreground_too_infrequent")
            if row["rejection_reasons"]:
                row["status"] = "construction_rejected"
            else:
                donor = donors[uid]
                donor_path = args.video_root/donor["relative_video_path"]
                row["donor"] = {**donor, "video_sha256": sha256_file(donor_path)}
                fg, bg, donor_indices = foreground_and_background_edits(frames, masks, decode(donor_path), sigma=protocol["sigma"])
                clean_files = write_png_sequence(output, f"clips/{uid}/clean", frames)
                fg_files = write_png_sequence(output, f"clips/{uid}/subject_blur", fg)
                bg_files = write_png_sequence(output, f"clips/{uid}/background_switch", bg)
                row["variants"] = {"clean": clean_files, "full/subject_blur": fg_files}
                row["windows"] = {}
                for position in ("start", "middle", "end"):
                    indices = window_indices(len(frames), position)
                    row["windows"][position] = list(indices)
                    row["variants"][position+"/subject_blur"] = [fg_files[t] if t in indices else clean_files[t] for t in range(len(frames))]
                    row["variants"][position+"/background_switch"] = [bg_files[t] if t in indices else clean_files[t] for t in range(len(frames))]
                if not np.array_equal(fg[masks == 0], frames[masks == 0]) or not np.array_equal(bg[masks == 1], frames[masks == 1]):
                    raise AssertionError("intervention leaked outside its designated region")
                row.update(status="accepted", shape=list(frames.shape), donor_frame_indices=donor_indices,
                    proofs={"subject_blur_background_changed_pixels": 0, "background_switch_subject_changed_pixels": 0,
                            "donor_same_split": donor["split"] == entry["split"], "donor_different_prompt": donor["prompt_id"] != entry["prompt_id"]})
        except Exception as exc:
            row.update(status="failed", failure_reason=f"{type(exc).__name__}: {exc}")
        path = output/"manifests"/f"{uid}.json"
        write_json(path, row)
        summary = {"video_uid": uid, "prompt_id": entry["prompt_id"], "status": row["status"],
                   "manifest": str(path.relative_to(output)), "manifest_sha256": sha256_file(path),
                   "rejection_reasons": row["rejection_reasons"]}
        with (output/"index.jsonl").open("a") as handle:
            handle.write(json.dumps(summary, sort_keys=True)+"\n")
        counts[row["status"]] += 1
        progress = {"processed": index+1, "total": len(entries), "counts": dict(counts), "elapsed_seconds": time.time()-start}
        write_json(output/"progress.json", progress); print(json.dumps(progress), flush=True)
    run.update(completed=True, finished_unix=time.time(), counts=dict(counts), index_sha256=sha256_file(output/"index.jsonl"))
    write_json(output/"run.json", run)
    return int(counts["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
