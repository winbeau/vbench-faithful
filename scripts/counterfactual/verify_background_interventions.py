"""Replay native background interventions and check every saved PNG byte value."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np

from .build_background_interventions import donor_map, foreground_and_background_edits
from .common import ROOT, sha256_file
from .region_discrimination import window_indices
from .subject_artifacts import artifact_path, new_output, read_jsonl, read_png_sequence, write_json

_DECODER = None


def verify_one(task):
    global _DECODER
    dataset, video_root, entry, protocol, expected_donor = task
    source_path = artifact_path(dataset, entry["manifest"])
    if sha256_file(source_path) != entry["manifest_sha256"]:
        raise ValueError("manifest hash mismatch")
    source = json.loads(source_path.read_text())
    if source["status"] != entry["status"] or source["base"]["video_uid"] != entry["video_uid"]:
        raise ValueError("index/manifest identity mismatch")
    if source["status"] == "failed":
        return {"video_uid": entry["video_uid"], "status": "construction_failed", "reason": source.get("failure_reason")}
    ref = source["construction_mask"]
    mask_path = artifact_path(dataset, ref["path"])
    if sha256_file(mask_path) != ref["sha256"]:
        raise ValueError("construction mask hash mismatch")
    with np.load(mask_path, allow_pickle=False) as data:
        masks = data["masks"]
        meta = json.loads(str(data["metadata_json"]))
        semantic_labels = data['semantic_labels'] if 'semantic_labels' in data else None
    if meta["role"] != "construction" or not np.isin(masks, (0, 1)).all():
        raise ValueError("invalid construction mask role or values")
    area = masks.mean((1, 2))
    refined = protocol.get('stage') in ('official_background_dev_segformer_grabcut_v2', 'official_background_dev_segformer_grabcut_v3')
    if refined:
        from .refine_background_construction import choose_target, eligibility
        label_map = json.loads((ROOT/'configs/subject-repair/ade20k_labels.json').read_text())['id2label']
        if semantic_labels is None or semantic_labels.shape != masks.shape:
            raise ValueError('refinement semantic evidence missing or inconsistent')
        target, target_id, fractions = choose_target(semantic_labels, {int(k): v for k, v in label_map.items()}, protocol['foreground_labels'],
                                                    rule=protocol.get('target_selection_rule', 'legacy_median_lexical_v2'))
        if target != source['construction_target'] or fractions != source['semantic_class_fractions']:
            raise ValueError('target was changed after semantic inference')
        if source['construction_target_source'] != protocol['target_source'] or source['base']['dimension'] != 'background_consistency':
            raise ValueError('target or official dimension provenance mismatch')
        if not np.allclose(area, source['foreground_fraction'], rtol=0, atol=1e-12):
            raise ValueError('reported area differs from actual construction mask')
        semantic_area = np.array(fractions[target]) if target is not None else np.zeros(len(masks))
        reasons = eligibility(semantic_area, area, source['refinement_failures'], protocol)
        if target is None:
            if masks.any() or source['refinement_failures']:
                raise ValueError('absent target must not produce a mask or fabricated GrabCut attempt')
            reasons.insert(0, 'no_observed_foreground_class')
        if len(masks) < 4:
            reasons.append('too_few_frames')
    else:
        reasons = []
        if len(masks) < 4:
            reasons.append("too_few_frames")
        if area.mean() < protocol["mean_foreground_min"]:
            reasons.append("no_sufficient_foreground")
        if area.mean() > protocol["mean_foreground_max"] or area.max() > protocol["frame_foreground_max"]:
            reasons.append("insufficient_background")
        if (area >= protocol["frame_presence_min_area"]).mean() < protocol["foreground_frame_presence_min"]:
            reasons.append("foreground_too_infrequent")
    if reasons != source["rejection_reasons"]:
        raise ValueError("qualification was changed after construction")
    if reasons:
        if source["status"] != "construction_rejected":
            raise ValueError("invalid accepted status")
        return {"video_uid": entry["video_uid"], "status": "rejection_verified", "reasons": reasons}
    if source["status"] != "accepted":
        raise ValueError("qualified candidate was dropped")
    if source["donor"]["video_uid"] != expected_donor["video_uid"]:
        raise ValueError("background donor was selected differently from protocol")
    if source["base"]["split"] != source["donor"]["split"] or source["base"]["prompt_id"] == source["donor"]["prompt_id"]:
        raise ValueError("donor violates split or scene identity contract")
    if _DECODER is None:
        import cv2
        import torch
        from vbench_audit_core.upstream import import_official_module
        cv2.setNumThreads(1); torch.set_num_threads(1)
        _DECODER = import_official_module("background_consistency")[0].load_video
    def decode(record):
        path = video_root/record["relative_video_path"]
        return _DECODER(str(path)).permute(0, 2, 3, 1).numpy().astype(np.uint8)
    if sha256_file(video_root/source["base"]["relative_video_path"]) != source["video_sha256"]:
        raise ValueError("original video changed")
    if sha256_file(video_root/source["donor"]["relative_video_path"]) != source["donor"]["video_sha256"]:
        raise ValueError("donor video changed")
    clean = read_png_sequence(dataset, source["variants"]["clean"])
    if not np.array_equal(clean, decode(source["base"])):
        raise ValueError("clean PNGs differ from official video decoding")
    if refined:
        from .refine_background_construction import refine_selected
        replay_masks, failures = refine_selected(clean, semantic_labels, target, target_id)
        if failures != source['refinement_failures'] or not np.array_equal(replay_masks, masks):
            raise ValueError('subject GrabCut refinement failed exact replay')
        sigma = protocol['sigma_person'] if target == 'person' else protocol['sigma_other']
        if source['gaussian_sigma'] != sigma:
            raise ValueError('blur differs from the subject construction rule')
    else:
        sigma = protocol['sigma']
    foreground, background, indices = foreground_and_background_edits(clean, masks, decode(source["donor"]), sigma=sigma)
    if indices != source["donor_frame_indices"]:
        raise ValueError("donor frame mapping changed")
    expected_keys = {"clean", "full/subject_blur"} | {p+"/"+k for p in ("start", "middle", "end") for k in ("subject_blur", "background_switch")}
    if set(source["variants"]) != expected_keys:
        raise ValueError("missing or unexpected intervention version")
    for variant, files in source["variants"].items():
        actual = read_png_sequence(dataset, files)
        if variant == "clean":
            expected = clean
        else:
            position, kind = variant.split("/")
            selected = range(len(clean)) if position == "full" else window_indices(len(clean), position)
            expected = clean.copy()
            edited = foreground if kind == "subject_blur" else background
            expected[list(selected)] = edited[list(selected)]
        if not np.array_equal(actual, expected):
            raise ValueError(f"deterministic pixel replay failed: {variant}")
    return {"video_uid": entry["video_uid"], "status": "accepted_replay_verified", "frames_per_variant": len(clean),
            "variants": len(expected_keys), "subject_blur_background_changed_pixels": 0,
            "background_switch_subject_changed_pixels": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--protocol", type=Path, default=ROOT/"configs/background-repair/construction_dev_v1.json")
    parser.add_argument("--video-root", type=Path, default=Path("/root/wenbiao_zhao/datasets/vbench-1.0-human-preference/videos"))
    args = parser.parse_args()
    config = ROOT/"configs/background-repair"
    run = json.loads((args.dataset/"run.json").read_text())
    if not run.get("completed") or run["index_sha256"] != sha256_file(args.dataset/"index.jsonl"):
        raise ValueError("dataset is incomplete or modified")
    if run["protocol_sha256"] != sha256_file(args.protocol):
        raise ValueError("construction protocol changed")
    protocol = json.loads(args.protocol.read_text())
    manifest = config/'natural1720_manifest_v2.jsonl'
    if sha256_file(manifest) != protocol['manifest_sha256']:
        raise ValueError('construction input manifest changed')
    expected = [r for r in read_jsonl(manifest) if r["split"] == protocol['split']]
    entries = read_jsonl(args.dataset/"index.jsonl")
    if len(entries) != protocol['candidate_count'] or {r["video_uid"] for r in entries} != {r["video_uid"] for r in expected}:
        raise ValueError("construction coverage mismatch")
    donors = donor_map(expected, strategy=protocol.get('donor_strategy', 'next_prompt_cycle'))
    tasks = [(args.dataset, args.video_root, r, protocol, donors[r["video_uid"]]) for r in entries]
    output = new_output(args.output)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(verify_one, tasks))
    report = {"candidate_count": len(entries), "counts": dict(Counter(r["status"] for r in results)),
              "dataset_index_sha256": run["index_sha256"], "protocol_sha256": run["protocol_sha256"],
              "verification_source_sha256": sha256_file(Path(__file__)), "results": results}
    write_json(output/"verification.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}))


if __name__ == "__main__":
    main()
