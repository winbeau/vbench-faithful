"""Compose single-frame interventions from already replayed full-blur pixels."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

from .common import sha256_file
from .subject_artifacts import artifact_path, new_output, read_jsonl, write_json, write_jsonl


POSITIONS = ("first_frame", "middle_frame", "last_frame", "full")
LEVELS = ("background_corrupt", "subject_corrupt")


def selected_frames(total, position):
    if total < 3:
        raise ValueError("single-frame controls need at least three actual frames")
    if position == "full":
        return tuple(range(total))
    return ({"first_frame": 0, "middle_frame": (total - 1) // 2, "last_frame": total - 1}[position],)


def parent_rows(parent, protocol):
    if sha256_file(parent / "index.jsonl") != protocol["parent_index_sha256"]:
        raise ValueError("parent population changed")
    if sha256_file(parent / "integrity.json") != protocol["parent_integrity_sha256"]:
        raise ValueError("parent replay receipt changed")
    proof = json.loads((parent / "integrity.json").read_text())
    rows = read_jsonl(parent / "index.jsonl")
    if (proof["index_sha256"] != protocol["parent_index_sha256"]
            or proof["outside_mask_changed_pixels"] != 0
            or proof["verified_corrupted_frames"] != protocol["parent_verified_frames"]
            or proof["bases"] != len(rows) or len(rows) != protocol["cohort_candidates"]
            or len({r["base_id"] for r in rows}) != len(rows)
            or sum(r["status"] == "accepted" for r in rows) != protocol["expected_constructed"]
            or protocol["positions"] != list(POSITIONS)):
        raise ValueError("parent replay does not cover the declared complete population")
    return rows


def read_manifest(root, row):
    path = artifact_path(root, row["manifest"])
    if sha256_file(path) != row["manifest_sha256"]:
        raise ValueError("manifest hash mismatch")
    value = json.loads(path.read_text())
    if value["status"] != row["status"] or value["base"]["base_id"] != row["base_id"]:
        raise ValueError("index and manifest disagree")
    if value["protocol_sha256"] != sha256_file(root / "protocol.json"):
        raise ValueError("construction protocol changed")
    return value


def composed_manifest(source, source_row, protocol_sha):
    manifest = copy.deepcopy(source)
    manifest.update(family="subject_single_frame_probe", protocol_sha256=protocol_sha,
                    parent_manifest_sha256=source_row["manifest_sha256"], variants={}, positions={})
    if source["status"] != "accepted":
        return manifest
    clean = source["variants"]["clean"]
    manifest["variants"]["clean"] = copy.deepcopy(clean)
    for position in POSITIONS:
        indices = selected_frames(len(clean), position)
        params = {**source["positions"]["full"]["parameters"], "position": position,
                  "window_indices": list(indices), "window_frames": len(indices),
                  "window_fraction": len(indices) / len(clean)}
        manifest["positions"][position] = {"parameters": params,
            "proofs": [p for p in source["positions"]["full"]["proofs"] if p["frame"] in indices]}
        for level in LEVELS:
            key = position + "/" + level
            parent_full = source["variants"]["full/" + level]
            manifest["variants"][key] = [
                {"path": f"clips/{source_row['base_id']}/{key}/{i:06d}.png",
                 "sha256": (parent_full[i] if i in indices else original)["sha256"]}
                for i, original in enumerate(clean)]
    return manifest


def copy_verified(parent, original, output, target):
    path = artifact_path(output, target["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("refusing to overwrite an existing artifact")
    shutil.copyfile(artifact_path(parent, original["path"]), path)
    if sha256_file(path) != original["sha256"] or original["sha256"] != target["sha256"]:
        raise ValueError("parent pixels or copied artifact changed")


def build(parent, output, protocol_path):
    protocol = json.loads(protocol_path.read_text())
    rows = parent_rows(parent, protocol)
    results = []
    for row in rows:
        source = read_manifest(parent, row)
        manifest = composed_manifest(source, row, sha256_file(protocol_path))
        if source["status"] == "accepted":
            copy_verified(parent, source["construction_masks"], output, manifest["construction_masks"])
            clean = source["variants"]["clean"]
            for name, targets in manifest["variants"].items():
                if name == "clean":
                    originals = clean
                else:
                    position, level = name.split("/")
                    indices = selected_frames(len(clean), position)
                    full = source["variants"]["full/" + level]
                    originals = [full[i] if i in indices else f for i, f in enumerate(clean)]
                for original, target in zip(originals, targets):
                    copy_verified(parent, original, output, target)
        target_manifest = artifact_path(output, row["manifest"])
        write_json(target_manifest, manifest)
        results.append({**row, "manifest_sha256": sha256_file(target_manifest)})
    write_jsonl(output / "index.jsonl", results)
    (output / "protocol.json").write_bytes(protocol_path.read_bytes())
    summary = {"total_bases": len(rows), "accepted": sum(r["status"] == "accepted" for r in rows),
               "rejected": sum(r["status"] != "accepted" for r in rows),
               "positions": list(POSITIONS), "parent_index_sha256": protocol["parent_index_sha256"],
               "parent_integrity_sha256": protocol["parent_integrity_sha256"],
               "scoring_masks_reused": False}
    write_json(output / "summary.json", summary)
    return summary


def verify(parent, dataset):
    protocol = json.loads((dataset / "protocol.json").read_text())
    originals = {r["base_id"]: r for r in parent_rows(parent, protocol)}
    rows = read_jsonl(dataset / "index.jsonl")
    if len(rows) != len(originals) or {r["base_id"] for r in rows} != set(originals):
        raise ValueError("composition population changed")
    checked = 0
    for row in rows:
        old = originals[row["base_id"]]
        if {k: v for k, v in row.items() if k != "manifest_sha256"} != {k: v for k, v in old.items() if k != "manifest_sha256"}:
            raise ValueError("composition identity or qualification changed")
        source = read_manifest(parent, old)
        manifest = read_manifest(dataset, row)
        if manifest != composed_manifest(source, old, sha256_file(dataset / "protocol.json")):
            raise ValueError("composition differs from the declared frame positions")
        if manifest["status"] != "accepted":
            continue
        for descriptor in [manifest["construction_masks"], *[f for files in manifest["variants"].values() for f in files]]:
            if sha256_file(artifact_path(dataset, descriptor["path"])) != descriptor["sha256"]:
                raise ValueError("composed artifact hash mismatch")
        # Each current output equals either protected clean pixels or the
        # full intervention previously replayed from those exact clean pixels.
        checked += sum(len(files) for key, files in manifest["variants"].items() if key != "clean")
    return {"bases": len(rows), "verified_corrupted_frames": checked, "outside_mask_changed_pixels": 0,
            "index_sha256": sha256_file(dataset / "index.jsonl"),
            "verification_mode": "exact_frame_hash_composition_from_independently_replayed_parent",
            "parent_integrity_sha256": protocol["parent_integrity_sha256"],
            "parent_index_sha256": protocol["parent_index_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.parent, args.verify)))
    else:
        if not args.protocol or not args.output:
            parser.error("construction requires --protocol and --output")
        print(json.dumps(build(args.parent, new_output(args.output), args.protocol)))


if __name__ == "__main__":
    main()
