"""Paper win ratios and paired human agreement on the frozen natural cohort."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import importlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from scripts.evaluate_pairwise_statistics import calibrate
from scripts.object_color_natural import cohort
from scripts.object_color_natural_media import save
from vbench_audit_core.inputs import sha256_file
from vbench_audit_models.labels import LabelVocabulary
from vbench_audit_models.prompt_compiler import PromptCompiler

METHODS = ("official", "repair_deterministic", "repair_base", "repair_lora")


def finite_score(record):
    value = record.get("score")
    return (float(value) if record.get("status") == "succeeded" and value is not None
            and np.isfinite(value) else None)


def outcome(gap, margin=0.):
    return .5 if abs(gap) <= margin else float(gap > 0)


def interval(samples, point, n_groups):
    return np.quantile(samples, [.025, .975]).tolist() if n_groups > 1 else None


def paper_win_ratios(pairs, videos, method, eligibility=None):
    stats = defaultdict(lambda: {"expected": 0, "defined": 0, "metric_wins": 0.,
                                 "human_wins_all": 0., "human_wins_defined": 0.})
    for index, pair in enumerate(pairs):
        a, b = (finite_score(videos[pair[k]]["scores"][method]) for k in ("video_a_uid", "video_b_uid"))
        prediction = outcome(a-b) if a is not None and b is not None else None
        if eligibility is not None and not eligibility[index]:
            prediction = None
        human = float(pair["human_label"])
        for model, metric_win, human_win in ((pair["model_a"], prediction, human),
                (pair["model_b"], 1-prediction if prediction is not None else None, 1-human)):
            s = stats[model]
            s["expected"] += 1
            s["human_wins_all"] += human_win
            if metric_win is not None:
                s["defined"] += 1
                s["metric_wins"] += metric_win
                s["human_wins_defined"] += human_win
    rows = []
    for model, s in sorted(stats.items()):
        n, valid = s["expected"], s["defined"]
        rows.append({"generator": model, "expected_comparisons": n, "defined_comparisons": valid,
            "coverage": valid/n, "human_win_ratio_all": s["human_wins_all"]/n,
            "human_win_ratio_same_defined_pairs": s["human_wins_defined"]/valid if valid else None,
            "metric_win_ratio_defined_pairs": s["metric_wins"]/valid if valid else None,
            "metric_win_ratio_full_population": s["metric_wins"]/n if valid == n else None,
            "full_population_lower_bound": s["metric_wins"]/n,
            "full_population_upper_bound": (s["metric_wins"]+n-valid)/n})
    x = [r["metric_win_ratio_defined_pairs"] for r in rows]
    y = [r["human_win_ratio_same_defined_pairs"] for r in rows]
    correlation = (float(pearsonr(x, y).statistic) if len(rows) == 4
        and all(v is not None for v in x+y) and np.ptp(x) > 0 and np.ptp(y) > 0 else None)
    return {"models": rows, "pearson_n4_defined_pairs": correlation,
            "full_pair_population_covered": all(r["coverage"] == 1 for r in rows),
            "scope": "paper-style zero-margin win ratios; undefined pairs are disclosed, never assigned fabricated wins"}


def population(pairs, videos, margins, *, resamples=10000, seed=20260920):
    if not pairs:
        return {"expected_pairs": 0, "methods": {}}
    labels = np.asarray([float(p["human_label"]) for p in pairs])
    if not np.isin(labels, [0., .5, 1.]).all():
        raise ValueError("unexpected official preference label")
    groups = sorted({p["prompt_id"] for p in pairs})
    human_tie = labels == .5
    group_index = {g: i for i, g in enumerate(groups)}
    gi = np.asarray([group_index[p["prompt_id"]] for p in pairs])
    group_counts = np.bincount(gi, minlength=len(groups))
    draws = np.random.default_rng(seed).integers(0, len(groups), (resamples, len(groups)))
    denominators = group_counts[draws].sum(1)
    valid, gaps, strict, calibrated, boots, boot0 = {}, {}, {}, {}, {}, {}
    for method in METHODS:
        aa = [finite_score(videos[p["video_a_uid"]]["scores"][method]) for p in pairs]
        bb = [finite_score(videos[p["video_b_uid"]]["scores"][method]) for p in pairs]
        valid[method] = np.asarray([a is not None and b is not None for a,b in zip(aa,bb)])
        gaps[method] = np.asarray([a-b if a is not None and b is not None else 0. for a,b in zip(aa,bb)])
        pred0 = np.where(gaps[method] == 0., .5, (gaps[method] > 0).astype(float))
        pred = np.where(abs(gaps[method]) <= margins[method], .5, (gaps[method] > 0).astype(float))
        strict[method] = (pred0 == labels) & valid[method]
        calibrated[method] = (pred == labels) & valid[method]
        sums = np.bincount(gi, weights=calibrated[method], minlength=len(groups))
        sums0 = np.bincount(gi, weights=strict[method], minlength=len(groups))
        boots[method] = sums[draws].sum(1)/denominators
        boot0[method] = sums0[draws].sum(1)/denominators
    result = {}
    common_primary = valid["official"] & valid["repair_deterministic"]
    for method in METHODS:
        mask = valid[method]
        common = mask & valid["official"]
        counts = np.bincount(gi, weights=common, minlength=len(groups))
        differences = calibrated[method].astype(float)-calibrated["official"].astype(float)
        sums = np.bincount(gi, weights=differences*common, minlength=len(groups))
        sampled_den = counts[draws].sum(1)
        available = sampled_den > 0
        common_boot = sums[draws].sum(1)[available]/sampled_den[available]
        n = int(mask.sum())
        point = float(calibrated[method].mean())
        result[method] = {"expected_pairs": len(pairs), "defined_pairs": n,
            "undefined_pairs": len(pairs)-n, "coverage": n/len(pairs),
            "tie_margin_dev_only": margins[method],
            "zero_margin_agreement_full_denominator": float(strict[method].mean()),
            "zero_margin_ci95_prompt_cluster": interval(boot0[method], float(strict[method].mean()), len(groups)),
            "tie_aware_agreement_full_denominator": point,
            "tie_aware_agreement_defined_pairs": float(calibrated[method][mask].mean()) if n else None,
            "tie_aware_ci95_prompt_cluster": interval(boots[method], point, len(groups)),
            "paired_delta_vs_official_full_denominator": point-float(calibrated["official"].mean()),
            "paired_delta_ci95_prompt_cluster": interval(boots[method]-boots["official"], 0., len(groups)),
            "zero_margin_delta_vs_official": float(strict[method].mean()-strict["official"].mean()),
            "zero_margin_delta_ci95_prompt_cluster": interval(boot0[method]-boot0["official"], 0., len(groups)),
            "common_defined_pairs_with_official": int(common.sum()),
            "paired_delta_common_defined_pairs": float(differences[common].mean()) if common.any() else None,
            "paired_delta_common_ci95": interval(common_boot, 0., int((counts>0).sum())) if common.any() else None,
            "human_ordered_pairs": int((~human_tie).sum()),
            "human_tie_pairs": int(human_tie.sum()),
            "ordered_agreement_full_denominator": float(calibrated[method][~human_tie].mean()) if (~human_tie).any() else None,
            "human_tie_agreement_full_denominator": float(calibrated[method][human_tie].mean()) if human_tie.any() else None,
            "zero_margin_ordered_agreement": float(strict[method][~human_tie].mean()) if (~human_tie).any() else None,
            "zero_margin_human_tie_agreement": float(strict[method][human_tie].mean()) if human_tie.any() else None,
            "predicted_tie_fraction_defined_pairs": float((abs(gaps[method][mask]) <= margins[method]).mean()) if n else None,
            "paper_alignment": paper_win_ratios(pairs, videos, method),
            "paper_alignment_common_official_deterministic": paper_win_ratios(pairs, videos, method, common_primary)}
    return {"expected_pairs": len(pairs), "source_prompt_clusters": len(groups),
            "human_label_counts": dict(Counter(str(v) for v in labels)),
            "all_tie_prediction_baseline": float(human_tie.mean()),
            "methods": result,
            "denominator_policy": "undefined comparisons count as unsuccessful in full-denominator agreement; conditional agreement and common-pair deltas are separate diagnostics"}


def replay(root, dimension, vocabulary):
    manifest = cohort(root, dimension)
    raw = {}
    for method in METHODS[:2]:
        path = root / "scores" / dimension / method / "results.json"
        records = json.loads(path.read_text())
        by_uid = {r["video_uid"]: r for r in records}
        if len(by_uid) != len(records) or set(by_uid) != {r["video_uid"] for r in manifest}:
            raise ValueError("score UID coverage differs from the frozen full cohort")
        raw[method] = by_uid
    algorithms = importlib.import_module(dimension + ".algorithms")
    compilers = {mode: PromptCompiler(dimension, vocabulary,
        {"kind": mode, "path": str(root / "compiled" / (dimension+"-"+mode+".json"))}) for mode in ("base", "lora")}
    videos, differences, null_agree, deterministic_errors, mismatches = {}, [], [], [], []
    for source in manifest:
        uid = source["video_uid"]
        rows = {method: raw[method][uid] for method in METHODS[:2]}
        scores = {method: {"score": rows[method]["score"], "status": rows[method]["status"]}
                  for method in METHODS[:2]}
        audit = rows["repair_deterministic"]
        artifact = audit.get("evidence")
        if artifact is None or sha256_file(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError("missing or corrupt raw GRiT evidence")
        trace = json.loads(Path(artifact["path"]).read_text())
        frames = trace["raw_frames"]

        def calculate(query):
            if dimension == "object_class":
                return algorithms.score_frames(frames, query.get("object"), vocabulary)
            return algorithms.score_frames(frames, query.get("object"), query.get("color"), vocabulary)

        replayed = calculate(trace["query"])
        aa, bb = finite_score(audit), finite_score(replayed)
        if (aa is None) != (bb is None):
            raise ValueError("deterministic replay changed score availability")
        if aa is not None:
            deterministic_errors.append(abs(aa-bb))
        aux = source["official_auxiliary_info"]
        legacy = (algorithms.score_frames(frames, aux["object"], vocabulary, lexical=False)
                  if dimension == "object_class" else algorithms.legacy_trace_score(frames, source["prompt"], aux["color"]))
        aa, bb = finite_score(rows["official"]), finite_score(legacy)
        null_agree.append((aa is None) == (bb is None))
        if aa is not None and bb is not None:
            differences.append(abs(aa-bb))
        if (aa is None) != (bb is None) or (aa is not None and bb is not None and abs(aa-bb)>1e-6):
            mismatches.append({"video_uid": uid, "official": aa, "legacy_trace": bb,
                               "official_status": rows["official"]["status"], "trace_status": legacy["status"]})
        for mode, compiler in compilers.items():
            try:
                query = compiler(source["prompt"])
                scored = calculate(query)
                scores["repair_"+mode] = {"score": scored["score"], "status": scored["status"],
                    "query": query, "reason_counts": scored.get("reason_counts", {}),
                    "measurement": "frozen prompt compilation and exact replay of real GRiT evidence"}
            except Exception as exc:
                scores["repair_"+mode] = {"score": None, "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}"}
        videos[uid] = {**source, "scores": scores, "raw_evidence": artifact,
                       "frame_count": len(frames), "reason_counts": trace["scoring"].get("reason_counts", {})}
    parity = {"true_official_inputs": len(manifest), "defined_value_pairs": len(differences),
              "availability_agreement": sum(null_agree), "max_abs_error": max(differences, default=None),
              "deterministic_replay_max_abs_error": max(deterministic_errors, default=None),
              "mismatches": mismatches,
              "scope": "Official independently calls pinned compute_*; trace replay is a diagnostic, not an independent Official call"}
    save(root / "analysis" / dimension / "parity.json", parity)
    if not all(null_agree) or max(differences, default=0.) > 1e-6 or max(deterministic_errors, default=0.) > 1e-12:
        raise ValueError(f"GRiT replay/Official parity failed: {parity}")
    return videos, parity, {mode: compiler.provenance for mode, compiler in compilers.items()}


def analyze(args):
    root, dimension = args.root.resolve(), args.dimension
    vocabulary = LabelVocabulary.from_file(args.vocabulary)
    videos, parity, compilers = replay(root, dimension, vocabulary)
    pairs = json.loads((root / (dimension + "-pairs.json")).read_text())["rows"]
    for pair in pairs:
        for field in ("video_a_uid", "video_b_uid"):
            row = videos[pair[field]]
            if row["prompt_id"] != pair["prompt_id"] or row["split"] != pair["split"]:
                raise ValueError("frozen pair and video identity mismatch")
    dev = [p for p in pairs if p["split"] == "dev"]
    test = [p for p in pairs if p["split"] == "test"]
    margins = {}
    for method in METHODS:
        selected = []
        for p in dev:
            aa, bb = (finite_score(videos[p[k]]["scores"][method]) for k in ("video_a_uid", "video_b_uid"))
            if aa is not None and bb is not None:
                selected.append((p, aa-bb))
        if not selected:
            raise ValueError("no defined dev pairs; cannot calibrate a tie margin")
        margins[method] = calibrate(selected)
    kwargs = {"resamples": args.resamples, "seed": 20260920}
    strict_test = [p for p in test if videos[p["video_a_uid"]]["semantic_training_split"] == "test"]
    report = {"dimension": dimension, "n_videos": len(videos), "n_pairs": len(pairs),
        "source_prompt_count": len({r["prompt"] for r in videos.values()}),
        "parity": parity, "compilers": compilers, "bootstrap": kwargs,
        "all": population(pairs, videos, margins, **kwargs),
        "dev": population(dev, videos, margins, **kwargs),
        "test": population(test, videos, margins, **kwargs),
        "test_intersect_semantic_test": population(strict_test, videos, margins, **kwargs),
        "video_status_counts": {m: dict(Counter(r["scores"][m]["status"] for r in videos.values())) for m in METHODS},
        "training_overlap_by_natural_split": {split: dict(Counter(r["semantic_training_split"]
            for r in videos.values() if r["split"] == split)) for split in ("dev", "test")},
        "generator_scores": {}, "source_files": {},
        "scope": "deterministic repair is primary and fixed before scoring; LoRA whole-cohort numbers include trained prompts and are descriptive; semantic-test intersection is reported separately"}
    for generator in sorted({r["generator"] for r in videos.values()}):
        selected = [r for r in videos.values() if r["generator"] == generator]
        report["generator_scores"][generator] = {}
        for method in METHODS:
            values = [finite_score(r["scores"][method]) for r in selected]
            finite = [v for v in values if v is not None]
            report["generator_scores"][generator][method] = {"expected_videos": len(values),
                "defined_videos": len(finite), "undefined_videos": len(values)-len(finite),
                "mean_defined_videos": float(np.mean(finite)) if finite else None,
                "mean_full_cohort": float(np.mean(finite)) if len(finite) == len(values) else None}
    for method in METHODS[1:]:
        common = [(finite_score(r["scores"]["official"]), finite_score(r["scores"][method])) for r in videos.values()]
        delta = np.asarray([b-a for a,b in common if a is not None and b is not None])
        report.setdefault("paired_video_score_changes", {})[method] = {"common_videos": len(delta),
            "mean_delta": float(delta.mean()) if len(delta) else None,
            "median_delta": float(np.median(delta)) if len(delta) else None,
            "changed_videos": int((abs(delta)>1e-12).sum()),
            "scope": "raw support-score changes, not human agreement or accuracy"}
    for path in [root/"protocol.json", root/"code-snapshot.json", root/"media-ledger.json",
                 root/(dimension+"-manifest.json"), root/(dimension+"-pairs.json"), args.vocabulary,
                 *[root/"compiled"/(dimension+"-"+mode+".json") for mode in ("base", "lora")],
                 *[root/"scores"/dimension/m/"results.json" for m in METHODS[:2]],
                 *[root/"scores"/dimension/m/"run.json" for m in METHODS[:2]]]:
        report["source_files"][str(path)] = sha256_file(path)
    output = root / "analysis" / dimension
    save(output / "report.json", report)
    save(output / "per_video.json", {"rows": list(videos.values())})
    rows = [{"dimension": dimension, "population": name, "method": method,
             **{k: v for k,v in pop["methods"][method].items() if not isinstance(v, dict)}}
            for name in ("all", "dev", "test", "test_intersect_semantic_test")
            for pop in [report[name]] for method in METHODS]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k,v in r.items()} for r in rows)
    print(json.dumps({"dimension": dimension, "parity": parity,
        "test": {m: {k: v for k,v in report["test"]["methods"][m].items() if not isinstance(v, dict)} for m in METHODS}}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dimension", choices=("object_class", "color"), required=True)
    parser.add_argument("--vocabulary", type=Path, default=Path("configs/four_dimension/object_color_vocabulary.json"))
    parser.add_argument("--resamples", type=int, default=10000)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
