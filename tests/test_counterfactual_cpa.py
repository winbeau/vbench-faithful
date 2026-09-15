"""Contract-level tests for the counterfactual CPA instrument.

`temporal_relocation` (Subject Consistency) is a mixed-rank family: `clean`
ranks above three corrupted positions that all share one rank.  Its composite
CPA therefore mixes a sensitivity contract (`clean > corrupted`) with a tie
contract (the positions are interchangeable).  These tests pin the pair census
and the per-contract statistics so the report cannot silently go back to
quoting a single composite number for a conjunction of two contracts.
"""
from __future__ import annotations

import unittest

from scripts.counterfactual.cpa import (
    _base_pairs,
    family_pairs,
    group_stats,
    rank_gap_groups,
)
from scripts.counterfactual.run_dimension import (
    invariance_stats,
    render_report,
    tied_level_groups,
)

LEVELS = ("clean", "corrupt_start", "corrupt_middle", "corrupt_end")


def rows(n_bases: int, split: str = "test") -> list[dict]:
    out = []
    for i in range(n_bases):
        for level in LEVELS:
            out.append(
                {
                    "derived_id": f"{split}{i:02d}::{level}",
                    "base_id": f"{split}{i:02d}",
                    "family": "temporal_relocation",
                    "level": level,
                    "expected_rank": 1 if level == "clean" else 0,
                    "split": split,
                }
            )
    return out


def scores(n_bases: int, split: str = "test", position_gap: float = 0.0) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for i in range(n_bases):
        base = f"{split}{i:02d}"
        out[f"{base}::clean"] = 0.90
        out[f"{base}::corrupt_start"] = 0.85
        out[f"{base}::corrupt_middle"] = 0.85 - position_gap / 2
        out[f"{base}::corrupt_end"] = 0.85 - position_gap
    return out


class FamilyPairCensusTests(unittest.TestCase):
    def test_mixed_rank_family_splits_into_two_contracts(self):
        levels = {
            "clean": {"score": 0.9, "expected_rank": 1},
            "corrupt_start": {"score": 0.8, "expected_rank": 0},
            "corrupt_middle": {"score": 0.8, "expected_rank": 0},
            "corrupt_end": {"score": 0.8, "expected_rank": 0},
        }
        pairs = family_pairs(levels)
        self.assertEqual(len(pairs), 6)
        self.assertEqual(sum(1 for expected, _gap, _d in pairs if expected == 1), 3)
        self.assertEqual(sum(1 for expected, _gap, _d in pairs if expected == 0), 3)

    def test_rank_gap_groups_separate_sensitivity_from_invariance(self):
        test_rows = rows(2)
        groups = rank_gap_groups(_base_pairs(test_rows, scores(2)))
        self.assertEqual(sorted(groups), [0, 1])
        self.assertEqual(len(groups[1]), 6)  # 2 bases x 3 clean-vs-corrupt pairs
        self.assertEqual(len(groups[0]), 6)  # 2 bases x 3 corrupted-position pairs

    def test_clean_must_outrank_every_corrupted_position(self):
        test_rows = rows(2)
        groups = rank_gap_groups(_base_pairs(test_rows, scores(2)))
        stats = group_stats(groups, margin=0.0)
        self.assertEqual(stats[1]["match_rate"], 1.0)

    def test_tie_contract_is_met_only_when_positions_are_equal(self):
        flat = group_stats(rank_gap_groups(_base_pairs(rows(2), scores(2))), margin=0.0)
        spread = group_stats(
            rank_gap_groups(_base_pairs(rows(2), scores(2, position_gap=0.05))), margin=0.0
        )
        self.assertEqual(flat[0]["tie_rate"], 1.0)
        self.assertLess(spread[0]["tie_rate"], flat[0]["tie_rate"])


class DeclaredEqualSubgroupTests(unittest.TestCase):
    def test_tied_level_groups_collect_each_rank(self):
        groups = tied_level_groups(rows(1))
        self.assertEqual(groups[0], ["corrupt_end", "corrupt_middle", "corrupt_start"])
        self.assertEqual(groups[1], ["clean"])

    def test_dispersion_only_sees_the_declared_equal_subset(self):
        test_rows = rows(4)
        flat = invariance_stats(test_rows, scores(4), {"corrupt_start", "corrupt_middle", "corrupt_end"})
        spread = invariance_stats(
            test_rows,
            scores(4, position_gap=0.05),
            {"corrupt_start", "corrupt_middle", "corrupt_end"},
        )
        self.assertEqual(flat["n_bases"], 4)
        self.assertEqual(flat["mean_cv"], 0.0)
        self.assertGreater(spread["mean_cv"], 0.0)

    def test_all_levels_would_dilute_the_position_spread(self):
        test_rows = rows(4)
        both = invariance_stats(test_rows, scores(4, position_gap=0.05))
        subset = invariance_stats(
            test_rows,
            scores(4, position_gap=0.05),
            {"corrupt_start", "corrupt_middle", "corrupt_end"},
        )
        self.assertIsNotNone(both["mean_cv"])
        self.assertGreater(both["mean_cv"], subset["mean_cv"])


class MixedRankReportTests(unittest.TestCase):
    def _report(self, position_gap: float = 0.05) -> str:
        test_rows = rows(4)
        coverage = [{"backend": "official", "scored_clips": 16, "expected_clips": 16,
                     "incomplete_shards": []}]
        pairs = _base_pairs(test_rows, scores(4, position_gap=position_gap))
        cpa = {
            "coverage": [{"backend": "official", "split": "test", "scored": 16, "total": 16}],
            "profiles": {"official": {"clean": {"n": 4, "mean": 0.9, "std": 0.0, "min": 0.9,
                                                "max": 0.9, "distinct": 1}}},
            "contracts": {
                "official": {
                    "test_zero_margin": group_stats(rank_gap_groups(pairs), 0.0),
                    "test_tie_aware": group_stats(rank_gap_groups(pairs), 0.0),
                }
            },
            "official": {"dev_margin": 0.0},
        }
        return render_report(
            "subject_consistency", "temporal_relocation", test_rows, coverage, cpa,
            "deadbeef", {r["derived_id"]: {"official": 0.9} for r in test_rows},
        )

    def test_mixed_rank_report_shows_both_contracts(self):
        report = self._report()
        self.assertIn("## Contract decomposition", report)
        self.assertIn("Declared-equal subgroups", report)
        self.assertIn("Rank-gap-0 pairs are the family's actual target", report)

    def test_position_spread_shows_up_in_the_decomposition(self):
        flat = self._report(position_gap=0.0)
        spread = self._report(position_gap=0.05)
        self.assertIn("| official | test | rank 0:", flat)
        self.assertNotEqual(flat, spread)

    def test_same_rank_family_keeps_the_invariance_block(self):
        test_rows = [{**row, "expected_rank": 1} for row in rows(4)]
        coverage = [{"backend": "official", "scored_clips": 16, "expected_clips": 16,
                     "incomplete_shards": []}]
        cpa = {
            "coverage": [{"backend": "official", "split": "test", "scored": 16, "total": 16}],
            "profiles": {},
            "contracts": {},
            "official": {"dev_margin": 0.0},
        }
        report = render_report(
            "dynamics_degree", "fps_resampling", test_rows, coverage, cpa,
            "deadbeef", {r["derived_id"]: {"official": 0.9} for r in test_rows},
        )
        self.assertIn("## Invariance statistics", report)
        self.assertNotIn("## Contract decomposition", report)


if __name__ == "__main__":
    unittest.main()
