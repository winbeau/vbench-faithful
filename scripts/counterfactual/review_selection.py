"""Runner-only, hash-bound user review; never imported by a metric evaluator."""
from __future__ import annotations

import json
from pathlib import Path

from .static_jitter import digest


def select_reviewed_candidates(rows, review_path: Path, manifest_path: Path, root: Path):
    review = json.loads(review_path.read_text())
    if review.get("schema") != "counterfactual-user-review-v1" or review.get("decision") != "accepted_as_development_counterfactual":
        raise ValueError("unsupported review schema or decision")
    if review["manifest_sha256"] != digest(manifest_path):
        raise ValueError("review does not bind this construction manifest")
    for field in ("construction_config", "sources"):
        relative = Path(review[field])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "configs":
            raise ValueError("review source/config must be repository config paths")
        if digest(root / relative) != review[f"{field}_sha256"]:
            raise ValueError(f"review {field} identity changed")
    by_id = {row["candidate_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate construction candidate")
    accepted = set()
    sources = set()
    for item in review["candidates"]:
        key = item["candidate_id"]
        if key in accepted or key not in by_id:
            raise ValueError("duplicate or missing reviewed candidate")
        row = by_id[key]
        for field in ("video_uid", "seed", "sha256"):
            if row[field] != item[field]:
                raise ValueError(f"reviewed candidate {field} identity changed")
        if (row["status"] != item["automatic_status_at_build"] or row["split"] != "dev"
                or row["protocol"] != "official-video-local-texture-jitter-v1"
                or row["family"] != review["family"] or row["amplitude"] != review["amplitude"]
                or review["amplitude_units"] != "native_pixels"):
            raise ValueError("review scope/status mismatch")
        if not (row.get("pixel_exact_to_intended") and row.get("native_timeline_preserved")
                and row.get("sampling_coordinates_in_bounds") and row.get("minimum_warp_jacobian", 0) > 0
                and row.get("intensity_noise_added") is False):
            raise ValueError("review cannot bypass verified media or no-fold constraints")
        accepted.add(key)
        sources.add(row["video_uid"])
    if not accepted:
        raise ValueError("empty review cohort")
    selected = [r for r in rows if r["candidate_id"] in accepted or
                (r["video_uid"] in sources and r["family"] in {"original", "encoding_control"})]
    for uid in sources:
        for family in ("original", "encoding_control"):
            controls = [r for r in selected if r["video_uid"] == uid and r["family"] == family]
            if len(controls) != 1 or controls[0]["status"] != "qualified":
                raise ValueError("review requires one qualified original and encoding control per source")
    return selected
