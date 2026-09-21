#!/usr/bin/env python3
"""Prepare frozen source groups, real GRiT observations and label vocabularies.

Scoring is delegated to metric packages. Existing archives are read only;
all extracted media and observation traces live under an explicit output root.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

from vbench_audit_core.inputs import sha256_file
from vbench_audit_core.upstream import verify_upstream

DIMENSIONS = ("object_class", "color")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != body:
            raise FileExistsError(f"refusing to replace frozen artifact: {path}")
    else:
        path.write_text(body)


def group_id(prompt):
    return hashlib.sha256(" ".join(prompt.casefold().split()).encode()).hexdigest()[:20]


def prepare(args):
    state = verify_upstream(args.upstream)
    suite_path = Path(state.path) / "vbench/VBench_full_info.json"
    suite = json.loads(suite_path.read_text())
    rows = []
    objects = [r["auxiliary_info"]["object_class"]["object"] for r in suite if "object_class" in r["dimension"]]
    for dimension in DIMENSIONS:
        seeds = [r for r in suite if dimension in r["dimension"]]
        assert len(seeds) == (79 if dimension == "object_class" else 85)
        ordered = sorted(seeds, key=lambda r: group_id(r["prompt_en"]))
        for index, row in enumerate(ordered):
            prompt = row["prompt_en"]
            if dimension == "object_class":
                target = row["auxiliary_info"][dimension]
            else:
                color = row["auxiliary_info"][dimension]["color"]
                match = re.fullmatch(r"an? " + re.escape(color) + r" (.+)", prompt)
                if not match or match.group(1) not in objects:
                    raise ValueError(f"official seed object not unambiguous: {prompt}")
                target = {"object": match.group(1), "color": color}
            rows.append({"source_group": group_id(prompt), "dimension": dimension,
                         "prompt": prompt, "gold": target,
                         "split": "test" if index < 20 else "dev" if index < 25 else "train",
                         "label_source": "official_auxiliary_info" if dimension == "object_class"
                         else "official_color_aux_plus_unambiguous_template_object"})
    splits = {}
    for row in rows:
        if row["source_group"] in splits and splits[row["source_group"]] != row["split"]:
            raise ValueError("shared source prompt crosses splits")
        splits[row["source_group"]] = row["split"]
    save(args.config / "object_color_seeds.json", {"upstream_sha": state.sha,
         "suite_sha256": sha256_file(suite_path), "split_rule": "sha256(normalized source prompt), first20 test,next5 dev,rest train per dimension BEFORE expansion",
         "rows": rows})
    # Explicit English count forms, not a suffix stripper applied to detections.
    special = {"person": "people", "knife": "knives", "sheep": "sheep", "skis": "skis",
               "scissors": "scissors", "broccoli": "broccoli", "mouse": "mice"}
    aliases = {}
    for obj in objects:
        plural = special.get(obj)
        if plural is None:
            plural = obj[:-1]+"ies" if obj.endswith("y") and obj[-2] not in "aeiou" else obj+"es" if obj.endswith(("s", "ch", "sh", "x")) else obj+"s"
        aliases[obj] = [] if plural == obj else [plural]
    for canonical, forms in {"bicycle": ["bike", "bikes"], "motorcycle": ["motorbike", "motorbikes"],
                             "airplane": ["aeroplane", "aeroplanes", "plane", "planes"],
                             "couch": ["sofa", "sofas"], "tv": ["television", "televisions", "television set"],
                             "cell phone": ["cellphone", "cellphones", "mobile phone", "mobile phones"],
                             "hair drier": ["hair dryer", "hair dryers"],
                             "donut": ["doughnut", "doughnuts"], "tennis racket": ["tennis racquet", "tennis racquets"]}.items():
        aliases[canonical] += forms
    save(args.config / "object_color_aliases.json", {"version": "premeasurement-v1",
         "provenance": {"kind": "declared_true_synonym_and_plural_controls_before_any_scores",
                        "observed_surface_validation": "pending independent training-only GRiT pass",
                        "no_hypernym_rules": True},
         "objects": sorted(objects), "object_aliases": aliases,
         "colors": ["black", "blue", "brown", "crimson", "cyan", "gold", "gray", "green", "magenta", "maroon", "navy", "orange", "pink", "purple", "red", "silver", "teal", "violet", "white", "yellow"],
         "color_aliases": {"gray": ["grey"]}})
    protocol = {"version": "object-color-preregistered-v1", "upstream_sha": state.sha,
         "families": {
           "object_label_invariance": {"dimension": "object_class", "base_count": {"dev": 5, "test": 20},
             "transformation": "same bytes and raw prompt; metadata object replaced by uppercase canonical and first frozen true alias (when available)",
             "primary_pair": "canonical vs uppercase", "secondary_pairs": "canonical vs first frozen alias; no replacement of zero effects",
             "official_relation": "change", "repair_relation": "invariance"},
           "object_absent_control": {"dimension": "object_class", "base_count": {"dev": 5, "test": 20},
             "transformation": "first frozen vocabulary class by source hash not located by independent Mask R-CNN; unconfirmed absence explicitly unverified",
             "relation": "both_low", "low_threshold": .1},
           "color_visibility_denominator": {"dimension": "color", "base_count": {"dev": 5, "test": 20},
             "visible_fraction": [1, .75, .5, .25, 0], "official_relation": "invariance_where_defined",
             "repair_relation": "ordered_drop", "construction": "independent Mask R-CNN locator plus SegFormer-B0 target mask; MobileSAM point fallback; inward feather only; scoring never consumes construction masks"},
           "color_synonym_control": {"dimension": "color", "base_count": {"dev": 5, "test": 20},
             "transformation": "gray/grey where applicable, otherwise frozen object true synonym in raw prompt; no image changes",
             "relation": "invariance", "scope": "available aliases only; eligible counts explicit"}},
         "sampling": {"frames": 16, "locator": "maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
                      "locator_score_min": .7, "target_presence_min_fraction": .75,
                      "eligibility": "independent target presence, no GRiT score gate, retained rejected ledger, no backfill",
                      "generator_selection": "sha256(source prompt) modulo3 over lavie/modelscope/videocrafter; lowest available seed",
                      "observation": "first24 train object_class source groups;16 official sampled frames;not score-selected"},
         "gates": {"change_median_abs_min": .2, "change_ci_lower_exclusive": .1,
                   "invariance_median_abs_max": .05, "invariance_signed_ci_contains": 0,
                   "color_spearman_min": .8, "color_strict_monotonic_base_fraction_min": .7,
                   "bootstrap_unit": "base", "bootstrap_replicates": 10000, "bootstrap_seed": 20260920,
                   "zero_effect_keep": True, "missing": "retain null rows and coverage, no imputation as observed score"},
         "training": {"backbone": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218",
                      "input_fields": ["prompt"], "adapters": list(DIMENSIONS), "stacking": False,
                      "silver_per_source": 8, "human_review_slots": 200,
                      "rank": 16, "alpha": 32, "dropout": .05, "learning_rate": .0001,
                      "max_steps": 300, "batch_size": 1, "gradient_accumulation": 4,
                      "max_length": 2048, "seed": 20260920}}
    save(args.config / "object_color_protocol.json", protocol)
    hashes = {name: sha256_file(args.config / name) for name in
              ("object_color_aliases.json", "object_color_protocol.json", "object_color_seeds.json")}
    save(args.config / "object_color_freeze.json", {"files": hashes, "stage": "before visual observation or scores"})
    print(json.dumps({"seeds": dict(Counter(r["dimension"] for r in rows)), "frozen": hashes}))


def extract(args):
    rows = json.loads((args.config / "object_color_seeds.json").read_text())["rows"]
    by_dimension = {dim: [r for r in rows if r["dimension"] == dim] for dim in DIMENSIONS}
    generators = ["lavie", "modelscope", "videocrafter"]
    archive_names = ["lavie.zip", "modelscope.zip", "videocrafter-09.zip"]
    archives = {g: zipfile.ZipFile(args.archives / name) for g, name in zip(generators, archive_names)}
    names = {g: z.namelist() for g, z in archives.items()}
    manifest = []
    try:
        for dimension, records in by_dimension.items():
            for row in records:
                group = row["source_group"]
                g = generators[int(group, 16) % len(generators)]
                pattern = re.compile(r"(?:^|/)" + dimension + "/" + re.escape(row["prompt"]) + r"-(\d+)\.mp4$")
                matches = [(int(m.group(1)), name) for name in names[g] if (m := pattern.search(name))]
                if not matches:
                    manifest.append({**row, "status": "missing_archive_member", "generator": g})
                    continue
                seed, name = min(matches)
                payload = archives[g].read(name)
                digest = hashlib.sha256(payload).hexdigest()
                destination = args.output / "sources" / dimension / f"{group}.mp4"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and sha256_file(destination) != digest:
                    raise ValueError(f"existing source differs: {destination}")
                if not destination.exists():
                    destination.write_bytes(payload)
                manifest.append({**row, "status": "available", "generator": g,
                                 "seed": seed, "archive": str(args.archives / archive_names[generators.index(g)]),
                                 "archive_member": name, "video": str(destination.resolve()), "sha256": digest})
    finally:
        for z in archives.values():
            z.close()
    save(args.output / "source_manifest.json", {"seeds_sha256": sha256_file(args.config / "object_color_seeds.json"), "rows": manifest})
    print(json.dumps(dict(Counter(r["status"] for r in manifest))))


def observe(args):
    from vbench_audit_models.grit import GritEvidenceModel
    rows = json.loads((args.output / "source_manifest.json").read_text())["rows"]
    selected = sorted([r for r in rows if r["dimension"] == "object_class" and r["split"] == "train"], key=lambda r: r["source_group"])[:24]
    save(args.output / "observation_selection.json", {"rows": selected, "score_selection": False})
    model = GritEvidenceModel("object_class", args.checkpoint, device="cuda:0", upstream=args.upstream)
    records, counts = [], Counter()
    for row in selected:
        if row["status"] != "available":
            records.append({"source_group": row["source_group"], "status": row["status"]})
            continue
        path = args.output / "observation" / (row["source_group"] + ".json")
        if path.exists():
            recorded = json.loads(path.read_text())
            if recorded["video_sha256"] != row["sha256"] or recorded["provenance"] != model.provenance:
                raise ValueError("observation resume provenance mismatch")
        else:
            frames = model.detect_video(Path(row["video"]))
            recorded = {"source_group": row["source_group"], "video_sha256": row["sha256"],
                        "provenance": model.provenance, "frames": frames}
            save(path, recorded)
        for frame in recorded["frames"]:
            counts.update(item["text"] for item in frame.get("objects", []))
        records.append({"source_group": row["source_group"], "path": str(path.resolve()),
                        "sha256": sha256_file(path), "frame_statuses": dict(Counter(f["status"] for f in recorded["frames"]))})
        print(json.dumps(records[-1]), flush=True)
    save(args.output / "grit_observation.json", {"provenance": model.provenance, "records": records,
         "labels": dict(sorted(counts.items())), "selection_sha256": sha256_file(args.output / "observation_selection.json"),
         "scope": "one frozen training-only pass; observed labels are not a capability inventory"})


def freeze_vocabulary(args):
    from vbench_audit_models.labels import LabelVocabulary, normalize
    aliases = json.loads((args.config / "object_color_aliases.json").read_text())
    observation = json.loads((args.output / "grit_observation.json").read_text())
    if not observation["records"] or not observation["labels"]:
        raise ValueError("real observation is empty; refusing a synthetic vocabulary")
    raw_labels = list(observation["labels"])
    vocabulary = {**aliases, "objects": sorted(set(aliases["objects"]) | {normalize(x) for x in raw_labels}),
                  "provenance": {"official_and_observed_union": True,
                                 "aliases_sha256": sha256_file(args.config / "object_color_aliases.json"),
                                 "observation_sha256": sha256_file(args.output / "grit_observation.json"),
                                 "observed_raw_labels": raw_labels,
                                 "observation_model": observation["provenance"],
                                 "support_scope": "frozen output vocabulary; absence from observation is not detector unsupported"}}
    LabelVocabulary(vocabulary)
    save(args.config / "object_color_vocabulary.json", vocabulary)
    print(json.dumps({"objects": len(vocabulary["objects"]), "observed_labels": len(raw_labels),
                      "sha256": sha256_file(args.config / "object_color_vocabulary.json")}))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["prepare", "extract", "observe", "freeze-vocabulary"])
    parser.add_argument("--config", type=Path, default=Path("configs/four_dimension"))
    parser.add_argument("--upstream", type=Path, default=Path("../VBench"))
    parser.add_argument("--output", type=Path, default=Path("output/object_color_20260920"))
    parser.add_argument("--archives", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    {"prepare": prepare, "extract": extract, "observe": observe,
     "freeze-vocabulary": freeze_vocabulary}[args.command](args)


if __name__ == "__main__":
    main()
