import numpy as np
import pytest

from scripts.counterfactual.build_background_interventions import donor_map, foreground_and_background_edits
from scripts.counterfactual.analyze_background_development import region_cases


def test_native_edits_preserve_the_opposite_region_and_replay_exactly():
    rng = np.random.default_rng(31)
    frames = rng.integers(0, 256, (4, 24, 32, 3), dtype=np.uint8)
    donor = rng.integers(0, 256, (3, 32, 24, 3), dtype=np.uint8)
    mask = np.zeros(frames.shape[:3], np.uint8)
    mask[:, 4:20, 8:24] = 1
    fg, bg, indices = foreground_and_background_edits(frames, mask, donor)
    assert np.array_equal(fg[mask == 0], frames[mask == 0])
    assert np.array_equal(bg[mask == 1], frames[mask == 1])
    assert not np.array_equal(fg[mask == 1], frames[mask == 1])
    assert not np.array_equal(bg[mask == 0], frames[mask == 0])
    again = foreground_and_background_edits(frames, mask, donor)
    assert all(np.array_equal(a, b) for a, b in zip((fg, bg, indices), again))


def test_donors_change_prompt_within_generator_and_seed():
    rows = [{"video_uid": f"{p}{g}{s}", "prompt_id": p, "generator": g, "seed": s}
            for p in ("beach", "forest", "river") for g in ("a", "b") for s in (0, 1)]
    mapping = donor_map(rows)
    for row in rows:
        donor = mapping[row["video_uid"]]
        assert donor["prompt_id"] != row["prompt_id"]
        assert donor["generator"] == row["generator"] and donor["seed"] == row["seed"]


def test_disjoint_donors_are_reciprocal_and_cannot_cross_split():
    rows = [{"video_uid": f"{p}{g}", "prompt_id": p, "generator": g, "seed": 0, "split": "test"}
            for p in ('a', 'b', 'c', 'd') for g in ('x', 'y')]
    mapping = donor_map(rows, strategy='disjoint_sorted_pairs')
    for row in rows:
        donor = mapping[row['video_uid']]
        assert mapping[donor['video_uid']] == row
        assert donor['generator'] == row['generator']
    assert mapping['ax']['prompt_id'] == 'b'
    assert mapping['cx']['prompt_id'] == 'd'
    with pytest.raises(ValueError, match='even'):
        donor_map([r for r in rows if r['prompt_id'] != 'd'], strategy='disjoint_sorted_pairs')
    with pytest.raises(ValueError, match='split'):
        donor_map([{**rows[0], 'split': 'dev'}, *rows[1:]])


def test_background_analysis_treats_subject_blur_as_nuisance():
    def score(v): return {"scores": {"official": {"status": "succeeded", "score": v}}}
    variants = {"clean": score(.9)}
    for pos in ("full", "start", "middle", "end"):
        variants[pos+"/subject_corrupt"] = score(.85)
        variants[pos+"/background_corrupt"] = score(.5)
    rows = [{"construction_status": "accepted", "base": {"base_id": "a", "prompt_id": "b"}, "variants": variants}]
    case = next(r for r in region_cases(rows) if r["method"] == "official")
    assert case["foreground_signed_change"] == pytest.approx(-.05)
    assert case["foreground_absolute_change"] == pytest.approx(.05)
    assert case["background_blur_drop"] == pytest.approx(.4)
    assert case["background_drop_gt_foreground_abs"] == 1
