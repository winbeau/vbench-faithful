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
    order_statistics,
    paired_bootstrap_ci,
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


JERK_LEVELS = (
    ("jerk_0_original", 4),
    ("jerk_1_duplicate", 3),
    ("jerk_2_duplicate_skip", 2),
    ("jerk_3_local_reverse", 1),
    ("jerk_4_multiple", 0),
)


def jerk_rows(n_bases: int, split: str = "test") -> list[dict]:
    out = []
    for i in range(n_bases):
        for level, rank in JERK_LEVELS:
            out.append(
                {
                    "derived_id": f"{split}{i:02d}::{level}",
                    "base_id": f"{split}{i:02d}",
                    "family": "temporal_jerk",
                    "level": level,
                    "expected_rank": rank,
                    "split": split,
                }
            )
    return out


class OrderedFamilyReportTests(unittest.TestCase):
    """A pure ordered family must not be described as a contract mixture.

    `temporal_jerk` gives every level its own rank, so it has no rank-gap-0
    pairs.  The generator used to emit the mixed-contract template for any
    family with more than one rank, which told the reader the composite was a
    mixture of a sensitivity and an invariance contract and that "rank-gap-0
    pairs are the family's actual target" - neither of which held.
    """

    def _report(self) -> str:
        test_rows = jerk_rows(4)
        scores = {
            row["derived_id"]: {"official": 0.5 + 0.01 * row["expected_rank"]}
            for row in test_rows
        }
        flat = {row["derived_id"]: 0.5 + 0.01 * row["expected_rank"] for row in test_rows}
        coverage = [{"backend": "official", "scored_clips": 20, "expected_clips": 20,
                     "incomplete_shards": []}]
        cpa = {
            "coverage": [{"backend": "official", "split": "test", "scored": 20, "total": 20}],
            "profiles": {"official": {"jerk_0_original": {
                "n": 4, "mean": 0.9, "std": 0.0, "min": 0.9, "max": 0.9, "distinct": 1}}},
            "contracts": {"official": {
                "test_zero_margin": group_stats(rank_gap_groups(_base_pairs(test_rows, flat)), 0.0),
                "test_tie_aware": group_stats(rank_gap_groups(_base_pairs(test_rows, flat)), 0.0),
            }},
            "contract_split": {"official": {
                "sensitivity": {"n_pairs": 40, "cpa": 1.0, "cpa_zero_margin": 1.0},
                "invariance": {"n_pairs": 0, "cpa": None},
            }},
            "order": {
                "official": {"dev": order_statistics(test_rows[:10], flat),
                             "test": order_statistics(test_rows, flat)},
                "repair": {"dev": order_statistics(test_rows[:10], flat),
                           "test": order_statistics(test_rows, flat)},
            },
            "paired": {
                "zero_margin": {"delta": -0.1, "ci_low": -0.2, "ci_high": -0.01, "n_bases": 4},
                "tie_aware": {"delta": -0.1, "ci_low": -0.2, "ci_high": -0.01, "n_bases": 4},
            },
            "discontinuity": {"repair": {"jerk_0_original": {
                "n": 4, "mean_discontinuity": 0.1, "tail_discontinuity": 0.2, "std_mean": 0.01,
            }}},
            "official": {
                "dev_margin": 0.0,
                "test_tie_aware": {"temporal_jerk": {
                    "cpa": 1.0, "n_pairs": 40, "ci_low": 1.0, "ci_high": 1.0}},
            },
            "repair": {
                "dev_margin": 0.0,
                "test_tie_aware": {"temporal_jerk": {
                    "cpa": 0.9, "n_pairs": 40, "ci_low": 0.8, "ci_high": 0.95}},
            },
        }
        return render_report(
            "motion_smoothness", "temporal_jerk", test_rows, coverage, cpa,
            "deadbeef", scores,
        )

    def test_ordered_family_omits_the_mixed_contract_template(self):
        report = self._report()
        self.assertNotIn("## Contract decomposition", report)
        self.assertNotIn("## CPA by contract half", report)
        self.assertNotIn("mixture of two contracts", report)

    def test_ordered_family_gets_sequence_level_and_paired_statistics(self):
        report = self._report()
        self.assertIn("## Sequence-level order statistics", report)
        self.assertIn("## Official vs Repair (test, tie-aware)", report)
        self.assertIn("paired 95% CI", report)
        self.assertIn("[-0.2000, -0.0100]", report)
        self.assertIn("## Repair continuity components (dev + test)", report)
        self.assertNotIn("## Frame evidence", report)

    def test_ordered_family_still_reports_the_composite(self):
        report = self._report()
        self.assertIn("## CPA", report)
        self.assertIn("## Score sensitivity", report)


class OrderStatisticTests(unittest.TestCase):
    def test_perfect_ladder_has_unit_spearman_and_full_strict_order(self):
        test_rows = jerk_rows(3)
        scores = {
            row["derived_id"]: 0.5 + 0.1 * row["expected_rank"] for row in test_rows
        }
        stats = order_statistics(test_rows, scores)
        self.assertEqual(stats["n_bases"], 3)
        self.assertAlmostEqual(stats["mean_spearman"], 1.0, places=9)
        self.assertEqual(stats["strict_order_bases"], 3)
        self.assertEqual(stats["strict_order_rate"], 1.0)

    def test_one_inverted_level_breaks_the_strict_order(self):
        test_rows = jerk_rows(1)
        scores = {row["derived_id"]: 0.5 + 0.1 * row["expected_rank"] for row in test_rows}
        scores["test00::jerk_2_duplicate_skip"] = -1.0
        stats = order_statistics(test_rows, scores)
        self.assertEqual(stats["strict_order_bases"], 0)
        self.assertEqual(stats["strict_order_rate"], 0.0)
        self.assertLess(stats["mean_spearman"], 1.0)

    def test_missing_level_is_skipped_not_scored_as_a_violation(self):
        test_rows = jerk_rows(2)
        scores = {row["derived_id"]: 0.5 + 0.1 * row["expected_rank"] for row in test_rows}
        scores["test00::jerk_3_local_reverse"] = None
        stats = order_statistics(test_rows, scores)
        # The base still has four scored levels, so it is evaluated on those and
        # is not penalised for the missing one.
        self.assertEqual(stats["n_bases"], 2)
        self.assertEqual(stats["strict_order_bases"], 2)

    def test_base_with_fewer_than_two_scores_is_excluded(self):
        test_rows = jerk_rows(1)
        scores = {row["derived_id"]: None for row in test_rows}
        scores["test00::jerk_0_original"] = 0.9
        stats = order_statistics(test_rows, scores)
        self.assertEqual(stats["n_bases"], 0)
        self.assertIsNone(stats["strict_order_rate"])

    def test_declared_equal_levels_are_not_required_to_be_strictly_ordered(self):
        test_rows = rows(1)  # mixed family: clean rank 1, three corrupted rank 0
        scores = {
            "test00::clean": 0.9,
            "test00::corrupt_start": 0.8,
            "test00::corrupt_middle": 0.7,
            "test00::corrupt_end": 0.6,
        }
        stats = order_statistics(test_rows, scores)
        self.assertEqual(stats["strict_order_bases"], 1)
        self.assertEqual(stats["strict_order_rate"], 1.0)


class PairedDeltaTests(unittest.TestCase):
    def _pairs(self, repair_correct: bool) -> dict[str, list[tuple[int, int, float]]]:
        out = {}
        for i in range(6):
            base = f"test{i:02d}"
            second = (1, 1, 0.5 if repair_correct else -0.5)
            out[base] = [(1, 1, 0.5), second]
        return out

    def test_paired_delta_uses_the_same_bases_for_both_backends(self):
        official = self._pairs(repair_correct=False)
        repair = self._pairs(repair_correct=True)
        stats = paired_bootstrap_ci(official, repair, 0.0, 0.0, 200, 2026)
        # official matches 1/2 pairs, repair matches 2/2 on every base, so the
        # paired delta is exactly +0.5 and the bootstrap cannot move it.
        self.assertAlmostEqual(stats["delta"], 0.5, places=9)
        self.assertAlmostEqual(stats["ci_low"], 0.5, places=9)
        self.assertAlmostEqual(stats["ci_high"], 0.5, places=9)
        self.assertEqual(stats["n_bases"], 6)

    def test_disjoint_bases_produce_no_estimate(self):
        stats = paired_bootstrap_ci({"a": [(1, 1, 1.0)]}, {"b": [(1, 1, 1.0)]}, 0.0, 0.0, 50, 1)
        self.assertIsNone(stats["delta"])
        self.assertEqual(stats["n_bases"], 0)


if __name__ == "__main__":
    unittest.main()