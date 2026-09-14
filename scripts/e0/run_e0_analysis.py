#!/usr/bin/env python3
"""Reproduce the frozen E0 model-level and clip-pair agreement analysis."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/processed/e0_scoring_manifest.csv"
MASTER = ROOT / "data/processed/pairwise_master_split.csv"
SPLIT = ROOT / "splits/e0_prompt_split.csv"
RAW = ROOT / "results/e0/raw_official_scores"
OUT = ROOT / "results/e0"
SCORED = ROOT / "data/processed/e0_pairs_scored.csv"

SEED = 20260911
BOOTSTRAP_SEED = 20260911
N_BOOTSTRAP = 2000
DIMS = ("dynamics_degree", "subject_consistency", "human_action", "spatial_relationship")
GENERATORS = ("cogvideo", "lavie", "modelscope", "videocraft")
EXPECTED_RECORDS = {"dynamics_degree": 1440, "subject_consistency": 1440, "human_action": 2000, "spatial_relationship": 2160}
EXPECTED_PAIRS = {"dynamics_degree": 2160, "subject_consistency": 2160, "human_action": 3000, "spatial_relationship": 3240}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def pair_id(row: dict[str, str]) -> str:
    payload = "\0".join((row["dimension"], row["instance_id"], row["model_a"], row["model_b"]))
    return "p_" + hashlib.sha256(payload.encode()).hexdigest()[:20]


def prediction(diff: float, delta: float) -> float:
    if diff > delta:
        return 1.0
    if diff < -delta:
        return 0.0
    return 0.5


def accuracy(rows: list[dict], delta: float) -> float:
    return sum(prediction(row["score_diff"], delta) == row["human"] for row in rows) / len(rows)


def pearson(x: list[float], y: list[float]) -> float:
    xm, ym = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - xm) * (b - ym) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - xm) ** 2 for a in x) * sum((b - ym) ** 2 for b in y))
    return numerator / denominator if denominator else float("nan")


def choose2(n: int) -> int:
    return n * (n - 1) // 2


def kendall_tau_b(rows: list[dict]) -> float:
    """Tau-b over metric score difference and human ordering, retaining ties."""
    observations = [(row["score_diff"], 2 * row["human"] - 1) for row in rows]
    n0 = choose2(len(observations))
    x_counts, y_counts, joint_counts = Counter(x for x, _ in observations), Counter(y for _, y in observations), Counter(observations)
    n1 = sum(choose2(n) for n in x_counts.values())
    n2 = sum(choose2(n) for n in y_counts.values())
    n3 = sum(choose2(n) for n in joint_counts.values())
    discordant = 0
    prior = Counter()
    for x_value, group in _group_sorted_x(observations):
        for _, y_value in group:
            discordant += sum(n for y, n in prior.items() if y > y_value)
        for _, y_value in group:
            prior[y_value] += 1
    comparable = n0 - n1 - n2 + n3
    numerator = comparable - 2 * discordant
    denominator = math.sqrt((n0 - n1) * (n0 - n2))
    return numerator / denominator if denominator else float("nan")


def _group_sorted_x(observations):
    ordered = sorted(observations)
    index = 0
    while index < len(ordered):
        stop = index + 1
        while stop < len(ordered) and ordered[stop][0] == ordered[index][0]:
            stop += 1
        yield ordered[index][0], ordered[index:stop]
        index = stop


def percentile(values: list[float], q: float) -> float:
    values = sorted(value for value in values if math.isfinite(value))
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] if low == high else values[low] * (high - position) + values[high] * (position - low)


def validate_and_join() -> tuple[list[dict], dict[str, dict]]:
    manifest = [row for row in read_csv(MANIFEST) if row["dimension"] in DIMS]
    master = [row for row in read_csv(MASTER) if row["dimension"] in DIMS]
    split_rows = [row for row in read_csv(SPLIT) if row["dimension"] in DIMS]
    assert len(manifest) == 7040
    assert len({row["video_uid"] for row in manifest}) == 7040, "cross-dimension or within-dimension video_uid collision"
    manifest_by_uid = {row["video_uid"]: row for row in manifest}
    split_map = {(row["dimension"], row["prompt_id"]): row for row in split_rows}
    assert all(int(row["seed"]) == SEED for row in split_rows)
    for dim in DIMS:
        records = [row for row in manifest if row["dimension"] == dim]
        assert len(records) == EXPECTED_RECORDS[dim]
        dev = {row["prompt_id"] for row in records if row["split"] == "dev"}
        test = {row["prompt_id"] for row in records if row["split"] == "test"}
        assert not dev.intersection(test), f"prompt leakage: {dim}"
        assert all(split_map[(dim, row["prompt_id"])]["split"] == row["split"] for row in records)

    scores: dict[str, dict[str, dict[str, str]]] = {}
    for dim in DIMS:
        path = RAW / dim / "results.csv"
        rows = read_csv(path)
        assert len(rows) == EXPECTED_RECORDS[dim], (dim, len(rows))
        assert len({row["video_uid"] for row in rows}) == len(rows), f"duplicate score video_uid: {dim}"
        scores[dim] = {row["video_uid"]: row for row in rows}
        for row in rows:
            source = manifest_by_uid.get(row["video_uid"])
            assert source is not None and source["dimension"] == dim
            for key in ("generator", "relative_video_path", "split", "prompt_id"):
                assert row[key] == source[key], f"score/manifest {key} mismatch: {row['video_uid']}"
        errors = [row for row in rows if row["status"] != "success"]
        if dim != "spatial_relationship":
            assert not errors, f"unexpected missing official scores: {dim}"
        else:
            assert len(errors) == 480
            assert len({row["prompt_id"] for row in errors}) == 24
            assert all(" inside of " in row["prompt_id"] for row in errors)
            assert all("MissingOfficialSpatialMetadata" in row["error"] for row in errors)

    assert len(master) == sum(EXPECTED_PAIRS.values())
    output, analysis = [], {}
    seen_pair_ids = set()
    for row in master:
        dim = row["dimension"]
        assert row["video_a_uid"] in manifest_by_uid and row["video_b_uid"] in manifest_by_uid
        ma, mb = manifest_by_uid[row["video_a_uid"]], manifest_by_uid[row["video_b_uid"]]
        assert ma["relative_video_path"] == row["video_a_path"] and mb["relative_video_path"] == row["video_b_path"]
        assert ma["generator"] == row["model_a"] and mb["generator"] == row["model_b"]
        assert ma["dimension"] == dim and mb["dimension"] == dim
        assert ma["prompt_id"] == row["prompt_id"] == mb["prompt_id"]
        assert ma["split"] == row["split"] == mb["split"]
        sa, sb = scores[dim][row["video_a_uid"]], scores[dim][row["video_b_uid"]]
        supported = sa["status"] == "success" and sb["status"] == "success"
        pid = pair_id(row)
        assert pid not in seen_pair_ids
        seen_pair_ids.add(pid)
        output.append({
            "pair_id": pid, "dimension": dim, "split": row["split"], "prompt_id": row["prompt_id"],
            "group_id": row["instance_id"], "generator_a": row["model_a"], "video_uid_a": row["video_a_uid"],
            "relative_video_path_a": row["video_a_path"], "score_a": sa["score"], "status_a": sa["status"],
            "generator_b": row["model_b"], "video_uid_b": row["video_b_uid"], "relative_video_path_b": row["video_b_path"],
            "score_b": sb["score"], "status_b": sb["status"], "human_label": row["human_label"],
            "supported_by_official": str(supported).lower(),
        })
    write_csv(SCORED, output)

    for dim in DIMS:
        full = [row for row in output if row["dimension"] == dim]
        assert len(full) == EXPECTED_PAIRS[dim]
        supported = []
        for row in full:
            if row["supported_by_official"] == "true":
                supported.append({**row, "human": float(row["human_label"]), "score_diff": float(row["score_a"]) - float(row["score_b"])})
        analysis[dim] = {"all": full, "supported": supported}
    assert len(analysis["spatial_relationship"]["supported"]) == 2520
    assert len(analysis["spatial_relationship"]["all"]) - len(analysis["spatial_relationship"]["supported"]) == 720
    return output, analysis


def model_statistics(analysis: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    rates, correlations = [], []
    for dim in DIMS:
        all_rows, supported = analysis[dim]["all"], analysis[dim]["supported"]
        for generator in GENERATORS:
            full_participation = sum(generator in (row["generator_a"], row["generator_b"]) for row in all_rows)
            relevant = [row for row in supported if generator in (row["generator_a"], row["generator_b"])]
            human_total = metric_total = 0.0
            for row in relevant:
                is_a = row["generator_a"] == generator
                human_total += row["human"] if is_a else 1 - row["human"]
                metric_a = prediction(row["score_diff"], 0.0)
                metric_total += metric_a if is_a else 1 - metric_a
            rates.append({"dimension": dim, "generator": generator, "human_win_rate": human_total / len(relevant),
                          "metric_win_rate": metric_total / len(relevant), "num_pairs": full_participation,
                          "num_supported_pairs": len(relevant)})
        current = [row for row in rates if row["dimension"] == dim]
        correlations.append({"dimension": dim, "pearson_r": pearson([row["metric_win_rate"] for row in current], [row["human_win_rate"] for row in current]),
                             "n_models": 4, "supported_prompt_count": len({row["prompt_id"] for row in supported}),
                             "supported_pair_count": len(supported)})
    write_csv(OUT / "model_win_rates.csv", rates)
    write_csv(OUT / "model_level_pearson.csv", correlations)
    return rates, correlations


def calibrate_and_score(analysis: dict[str, dict]):
    calibration, pair_metrics, kendall = [], [], []
    deltas = {}
    for dim in DIMS:
        supported = analysis[dim]["supported"]
        dev = [row for row in supported if row["split"] == "dev"]
        candidates = sorted({0.0, *(abs(row["score_diff"]) for row in dev)})
        scored = [(accuracy(dev, delta), delta) for delta in candidates]
        best_accuracy = max(value for value, _ in scored)
        delta = min(delta for value, delta in scored if value == best_accuracy)
        deltas[dim] = delta
        calibration.append({"dimension": dim, "delta": delta, "dev_accuracy": best_accuracy, "dev_pair_count": len(dev),
                            "dev_prompt_count": len({row["prompt_id"] for row in dev}),
                            "selection_rule": "maximize dev tie-aware accuracy over unique dev absolute score differences; smallest delta on tie",
                            "seed": SEED})
        for scope, rows in (("dev", dev), ("test", [row for row in supported if row["split"] == "test"]), ("all_supported", supported)):
            pair_metrics.append({"dimension": dim, "scope": scope, "zero_margin_accuracy": accuracy(rows, 0.0),
                                 "tie_aware_accuracy": accuracy(rows, delta), "frozen_delta": delta,
                                 "pair_count": len(rows), "prompt_count": len({row["prompt_id"] for row in rows})})
            kendall.append({"dimension": dim, "scope": scope, "kendall_tau_b": kendall_tau_b(rows), "pair_count": len(rows),
                            "prompt_count": len({row["prompt_id"] for row in rows}),
                            "inputs": "metric score_a-score_b vs human ordering {-1,0,1}; ties retained"})
    write_csv(OUT / "tie_margin_calibration.csv", calibration)
    write_csv(OUT / "test_pair_metrics.csv", pair_metrics)
    write_csv(OUT / "kendall_tau_b.csv", kendall)
    return deltas, calibration, pair_metrics, kendall


def bootstrap(analysis: dict[str, dict], deltas: dict[str, float]) -> list[dict]:
    output = []
    for dim_index, dim in enumerate(DIMS):
        test = [row for row in analysis[dim]["supported"] if row["split"] == "test"]
        by_prompt = defaultdict(list)
        for row in test:
            by_prompt[row["prompt_id"]].append(row)
        prompts = sorted(by_prompt)
        rng = random.Random(BOOTSTRAP_SEED)
        samples = {"zero_margin_pair_accuracy": [], "frozen_test_tie_aware_accuracy": [], "kendall_tau_b": []}
        for _ in range(N_BOOTSTRAP):
            sampled = []
            for _ in prompts:
                sampled.extend(by_prompt[rng.choice(prompts)])
            samples["zero_margin_pair_accuracy"].append(accuracy(sampled, 0.0))
            samples["frozen_test_tie_aware_accuracy"].append(accuracy(sampled, deltas[dim]))
            samples["kendall_tau_b"].append(kendall_tau_b(sampled))
        estimates = {"zero_margin_pair_accuracy": accuracy(test, 0.0),
                     "frozen_test_tie_aware_accuracy": accuracy(test, deltas[dim]),
                     "kendall_tau_b": kendall_tau_b(test)}
        for metric, values in samples.items():
            output.append({"dimension": dim, "metric": metric, "estimate": estimates[metric],
                           "ci_lower": percentile(values, 0.025), "ci_upper": percentile(values, 0.975),
                           "n_bootstrap": N_BOOTSTRAP, "cluster_unit": "prompt_id", "seed": BOOTSTRAP_SEED})
    write_csv(OUT / "bootstrap_ci.csv", output)
    return output


def spatial_coverage(analysis: dict[str, dict]) -> list[dict]:
    all_rows, supported = analysis["spatial_relationship"]["all"], analysis["spatial_relationship"]["supported"]
    values = [{
        "total_records": 2160, "supported_records": 1680, "unsupported_records": 480, "record_coverage": 1680/2160,
        "total_prompts": 108, "supported_prompts": 84, "unsupported_prompts": 24, "prompt_coverage": 84/108,
        "unsupported_relation": "inside of", "dev_total": 860, "dev_unsupported": 160, "dev_supported": 700,
        "test_total": 1300, "test_unsupported": 320, "test_supported": 980,
        "total_pairs": 3240, "supported_pairs": 2520, "unsupported_pairs": 720, "pair_coverage": 2520/3240,
        "dev_total_pairs": sum(row["split"] == "dev" for row in all_rows),
        "dev_supported_pairs": sum(row["split"] == "dev" for row in supported),
        "test_total_pairs": sum(row["split"] == "test" for row in all_rows),
        "test_supported_pairs": sum(row["split"] == "test" for row in supported),
    }]
    assert sum(row["split"] == "dev" for row in all_rows) == 1290
    assert sum(row["split"] == "test" for row in all_rows) == 1950
    assert sum(row["split"] == "dev" for row in supported) == 1050
    assert sum(row["split"] == "test" for row in supported) == 1470
    write_csv(OUT / "spatial_coverage.csv", values)
    return values


def lookup(rows, dimension, **conditions):
    return next(row for row in rows if row["dimension"] == dimension and all(row[key] == value for key, value in conditions.items()))


def summaries(correlations, calibration, pair_metrics, kendall, bootstrap_rows):
    rows = []
    total_prompts = {dim: len({row["prompt_id"] for row in read_csv(MASTER) if row["dimension"] == dim}) for dim in DIMS}
    for dim in DIMS:
        corr = lookup(correlations, dim)
        cal = lookup(calibration, dim)
        dev = lookup(pair_metrics, dim, scope="dev")
        test = lookup(pair_metrics, dim, scope="test")
        tau = lookup(kendall, dim, scope="test")
        ci_pair = lookup(bootstrap_rows, dim, metric="zero_margin_pair_accuracy")
        ci_tie = lookup(bootstrap_rows, dim, metric="frozen_test_tie_aware_accuracy")
        rows.append({"dimension": dim, "model_level_pearson": corr["pearson_r"], "n_models": 4,
                     "dev_zero_margin_accuracy": dev["zero_margin_accuracy"], "test_zero_margin_accuracy": test["zero_margin_accuracy"],
                     "selected_tie_margin": cal["delta"], "test_tie_aware_accuracy": test["tie_aware_accuracy"],
                     "kendall_tau_b": tau["kendall_tau_b"], "pair_accuracy_ci_low": ci_pair["ci_lower"],
                     "pair_accuracy_ci_high": ci_pair["ci_upper"], "tie_accuracy_ci_low": ci_tie["ci_lower"],
                     "tie_accuracy_ci_high": ci_tie["ci_upper"], "supported_prompts": corr["supported_prompt_count"],
                     "total_prompts": total_prompts[dim], "supported_pairs": corr["supported_pair_count"],
                     "total_pairs": EXPECTED_PAIRS[dim], "coverage": corr["supported_pair_count"] / EXPECTED_PAIRS[dim]})
    write_csv(OUT / "e0_metrics_summary.csv", rows)
    make_markdown(rows)
    make_latex(rows)
    return rows


def display_name(dim):
    return {"dynamics_degree":"Dynamic Degree", "subject_consistency":"Subject Consistency", "human_action":"Human Action", "spatial_relationship":"Spatial Relationship"}[dim]


def make_markdown(rows):
    lines = ["| Dimension | Pearson (n=4) | Test clip-pair accuracy | Tie-aware test accuracy | Kendall tau-b | Pair accuracy 95% CI | Coverage |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {display_name(r['dimension'])} | {r['model_level_pearson']:.3f} | {r['test_zero_margin_accuracy']:.3f} | {r['test_tie_aware_accuracy']:.3f} | {r['kendall_tau_b']:.3f} | [{r['pair_accuracy_ci_low']:.3f}, {r['pair_accuracy_ci_high']:.3f}] | {r['coverage']:.1%} |")
    (OUT / "e0_metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_latex(rows):
    lines = [r"\begin{tabular}{lrrrrrl}", r"\toprule", r"Dimension & Pearson ($n=4$) & Clip Acc. & Tie Acc. & Kendall $\tau_b$ & 95\% CI & Coverage \\", r"\midrule"]
    for r in rows:
        lines.append(f"{display_name(r['dimension'])} & {r['model_level_pearson']:.3f} & {r['test_zero_margin_accuracy']:.3f} & {r['test_tie_aware_accuracy']:.3f} & {r['kendall_tau_b']:.3f} & [{r['pair_accuracy_ci_low']:.3f}, {r['pair_accuracy_ci_high']:.3f}] & {100*r['coverage']:.1f}\\% " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "e0_metrics_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(rows):
    width, height, margin = 1100, 650, 90
    colors = ("#315c8c", "#d4783c")
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#fbfaf7"/>', '<text x="550" y="42" text-anchor="middle" font-family="sans-serif" font-size="25" fill="#20242a">E0: model-level versus clip-level agreement</text>']
    chart_h = height - 2 * margin
    for tick in range(0, 11, 2):
        y = margin + chart_h * (1 - tick / 10)
        parts += [f'<line x1="{margin}" y1="{y}" x2="{width-margin}" y2="{y}" stroke="#d9d6cf"/>', f'<text x="{margin-12}" y="{y+5}" text-anchor="end" font-family="sans-serif" font-size="14">{tick/10:.1f}</text>']
    group_w=(width-2*margin)/4
    for i,r in enumerate(rows):
        center=margin+group_w*(i+.5)
        values=(max(0.0,min(1.0,float(r['model_level_pearson']))),float(r['test_zero_margin_accuracy']))
        for j,value in enumerate(values):
            x=center-43+j*48; y=margin+chart_h*(1-value)
            parts.append(f'<rect x="{x}" y="{y}" width="40" height="{margin+chart_h-y}" fill="{colors[j]}" rx="3"/>')
            parts.append(f'<text x="{x+20}" y="{y-8}" text-anchor="middle" font-family="sans-serif" font-size="13">{value:.3f}</text>')
        name=display_name(r['dimension']).replace(' ','&#10;')
        parts.append(f'<text x="{center}" y="{height-margin+30}" text-anchor="middle" font-family="sans-serif" font-size="15">{name}</text>')
    parts += [f'<rect x="{width-360}" y="55" width="15" height="15" fill="{colors[0]}"/><text x="{width-338}" y="68" font-family="sans-serif" font-size="14">Model-level Pearson</text>',
              f'<rect x="{width-190}" y="55" width="15" height="15" fill="{colors[1]}"/><text x="{width-168}" y="68" font-family="sans-serif" font-size="14">Test pair accuracy</text>', '</svg>']
    svg=OUT/'model_vs_pair_agreement.svg'; svg.write_text('\n'.join(parts),encoding='utf-8')
    subprocess.run(['convert',str(svg),str(OUT/'model_vs_pair_agreement.png')],check=True)
    make_pdf_chart(rows)


def make_pdf_chart(rows):
    ps = ["%!PS-Adobe-3.0", "%%BoundingBox: 0 0 792 468", "/Helvetica findfont 12 scalefont setfont", "0.98 0.97 0.94 setrgbcolor 0 0 792 468 rectfill", "0 0 0 setrgbcolor 260 440 moveto (E0: model-level versus clip-level agreement) show"]
    for i,r in enumerate(rows):
        center=115+i*170
        for j,(key,color) in enumerate((("model_level_pearson",(0.19,0.36,0.55)),("test_zero_margin_accuracy",(0.83,0.47,0.24)))):
            value=max(0,min(1,float(r[key]))); x=center+j*38; h=300*value
            ps.append(f"{color[0]} {color[1]} {color[2]} setrgbcolor {x} 80 {30} {h} rectfill")
            ps.append(f"0 0 0 setrgbcolor {x} 65 moveto ({value:.3f}) show")
        ps.append(f"0 0 0 setrgbcolor {center-10} 40 moveto ({display_name(r['dimension'])}) show")
    ps += ["showpage", "%%EOF"]
    ps_path=OUT/'model_vs_pair_agreement.ps'; ps_path.write_text('\n'.join(ps)+'\n',encoding='ascii')
    subprocess.run(['gs','-q','-dBATCH','-dNOPAUSE','-sDEVICE=pdfwrite',f'-sOutputFile={OUT / "model_vs_pair_agreement.pdf"}',str(ps_path)],check=True)
    ps_path.unlink()


def make_report(summary_rows, rates, spatial):
    metadata = json.loads((RAW / "run_metadata.txt").read_text(encoding="utf-8"))
    counts=[]
    master=read_csv(MASTER)
    for dim in DIMS:
        rows=[r for r in master if r['dimension']==dim]
        counts.append(f"- {dim}: prompts={len({r['prompt_id'] for r in rows})}, groups={len({r['instance_id'] for r in rows})}, pairs={len(rows)}, labels={dict(Counter(r['human_label'] for r in rows))}")
    metric_lines=[]
    for r in summary_rows:
        metric_lines.append(f"- {r['dimension']}: Pearson={r['model_level_pearson']:.6f} (n=4), test zero-margin={r['test_zero_margin_accuracy']:.6f} CI=[{r['pair_accuracy_ci_low']:.6f}, {r['pair_accuracy_ci_high']:.6f}], delta={r['selected_tie_margin']:.9g}, test tie-aware={r['test_tie_aware_accuracy']:.6f} CI=[{r['tie_accuracy_ci_low']:.6f}, {r['tie_accuracy_ci_high']:.6f}], test tau-b={r['kendall_tau_b']:.6f}, coverage={r['coverage']:.6f}")
    text=f"""# E0 Analysis Report

## Reproducibility

- Local repository Git SHA: `{git_sha(ROOT)}`
- Server evaluator Git SHA: `{metadata.get('vbench_audit_sha','unavailable')}`
- Locked VBench 1.0 Git SHA: `{metadata.get('vbench1_sha','unavailable')}`
- Inputs: `{MASTER.relative_to(ROOT)}`, `{MANIFEST.relative_to(ROOT)}`, `{SPLIT.relative_to(ROOT)}`, `results/e0/raw_official_scores/*/results.csv`
- Input SHA256: master `{sha256(MASTER)}`, manifest `{sha256(MANIFEST)}`, split `{sha256(SPLIT)}`
- Split: frozen prompt-disjoint `(dimension, prompt_id)`, seed={SEED}, nominal dev/test=40%/60%; no re-splitting.
- Human label: 1=model_a wins, 0=model_b wins, 0.5=tie.
- Win ratio: win=1, loss=0, tie=0.5, divided by comparisons participated; Spatial human and metric rates use the same supported subset.
- Tie-margin calibration: dev only; candidates are all unique absolute dev score differences plus zero; maximize dev tie-aware accuracy; smallest delta wins ties.
- Test protocol: freeze each dimension's selected delta without test tuning.
- Kendall tau-b: metric `(score_a-score_b)` versus human ordering `{{-1,0,1}}`; ties retained. CSV reports dev, test, and all-supported; summary uses test.
- Bootstrap: {N_BOOTSTRAP} repetitions, clustered by test `prompt_id`, seed base={BOOTSTRAP_SEED}; percentile 95% intervals.

## Data counts

{chr(10).join(counts)}

## Final metrics

{chr(10).join(metric_lines)}

## Spatial coverage and abstention

Spatial official VBench 1.0 supports only left/right/top/bottom in the locked evaluator path.

24 inside-of prompts from the public preference set are therefore reported as unsupported/abstained, not assigned fabricated scores.

- Records: 1680/2160 supported ({spatial['record_coverage']:.2%}); 480 unsupported.
- Prompts: 84/108 supported ({spatial['prompt_coverage']:.2%}); 24 unsupported.
- Pairs: 2520/3240 supported ({spatial['pair_coverage']:.2%}); 720 unsupported.

## Warnings and limitations

- Model-level Pearson uses only four generators (`n=4`); it is descriptive and no strong model-level confidence interval is claimed.
- Pair-level uncertainty uses prompt-cluster bootstrap; pair rows within a prompt are not treated as independent.
- Binary Human Action and Dynamic Degree scores create many metric ties; tie-aware results depend on dev-calibrated margins.
- Spatial results apply only to the official-supported relation subset and must not be generalized to `inside of`.
"""
    (OUT/'E0_ANALYSIS_REPORT.md').write_text(text,encoding='utf-8')


def main():
    required=[MANIFEST,MASTER,SPLIT,RAW/'run_metadata.txt',*(RAW/d/'results.csv' for d in DIMS)]
    missing=[str(path) for path in required if not path.is_file()]
    if missing: raise SystemExit('missing inputs: '+', '.join(missing))
    _,analysis=validate_and_join()
    rates,correlations=model_statistics(analysis)
    deltas,calibration,pair_metrics,kendall=calibrate_and_score(analysis)
    bootstrap_rows=bootstrap(analysis,deltas)
    spatial=spatial_coverage(analysis)[0]
    summary_rows=summaries(correlations,calibration,pair_metrics,kendall,bootstrap_rows)
    make_figures(summary_rows)
    make_report(summary_rows,rates,spatial)
    print('E0 ANALYSIS = PASS')
    for row in summary_rows: print(row)


if __name__ == '__main__':
    main()
