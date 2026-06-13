"""Unit tests for the guarded-ranker frontier evaluator + the literal
gate-coverage check (Task 16 — offline analysis tooling, no DB).

The pure functions under test take PRECOMPUTED scores (the LODO model is
bypassed entirely): synthetic fixture = 2 runs x 2 passes, 12 rows per
(run, pass) pool, hand-placed positives, deterministic scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.check_gate_coverage import GateColumnsMissing, check_gate_coverage
from scripts.eval_guarded_ranker import (
    frontier_table,
    guarded_keep_mask,
    guarded_keep_mask_all,
    parse_quantile_grid,
    recall_and_kept,
    topk_keep_mask_all,
)


# ===========================================================================
# Fixtures
# ===========================================================================
def _one_pool(n: int = 12) -> tuple[pd.DataFrame, np.ndarray]:
    """One (run, pass) pool: 12 rows, deterministic scores 0.0, 0.1 .. 1.1."""
    df = pd.DataFrame({
        "run_id": ["r1"] * n,
        "pass_name": ["pA"] * n,
        "chunk_index": list(range(n)),
    })
    scores = np.arange(n) * 0.1
    return df, scores


def _fixture_df() -> pd.DataFrame:
    """2 runs x 2 passes, 12 rows/pool (48 rows), hand-placed positives.

    Per pool: deterministic scores 0.0..1.1; the TOP row (score 1.1) is a
    positive in every pool. Pool (r1, pA) carries ONE extra positive at score
    0.2 protected only by unit_gate=1 (the gate-union recall case).
    Totals: 5 positives, 1 gate-covered row.
    """
    frames = []
    for run in ("r1", "r2"):
        for pas in ("pA", "pB"):
            n = 12
            d = pd.DataFrame({
                "run_id": [run] * n,
                "pass_name": [pas] * n,
                "doc_filename": [f"{run}.pdf"] * n,
                "chunk_index": list(range(n)),
                "score": np.arange(n) * 0.1,
                "final_score": np.arange(n) * 0.1,
                "used": [0] * n,
                "unit_gate": [0.0] * n,
                "table_gate": [0.0] * n,
            })
            d.loc[n - 1, "used"] = 1  # top-scored row is a positive everywhere
            if (run, pas) == ("r1", "pA"):
                d.loc[2, "used"] = 1        # low-scored positive (score 0.2) ...
                d.loc[2, "unit_gate"] = 1.0  # ... protected ONLY by the gate
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


# ===========================================================================
# guarded_keep_mask — exact keep-sets on a hand-computed pool
# ===========================================================================
class TestGuardedKeepMask:
    def test_q50_gated_member_below_threshold_stays(self):
        """q=0.5 on scores 0.0..1.1 -> median 0.55 -> ranker keeps the 6 rows
        scoring 0.6..1.1; the gated row at score 0.2 stays via the union."""
        df, scores = _one_pool()
        gate = np.zeros(12, dtype=bool)
        gate[2] = True  # score 0.2 — well below the q=0.5 threshold
        keep = guarded_keep_mask(df, scores, q=0.5, k_min=3, k_max=0, gate_mask=gate)
        assert set(np.flatnonzero(keep)) == {2, 6, 7, 8, 9, 10, 11}

    def test_k_min_honored_when_threshold_keeps_fewer(self):
        """q=0.95 -> threshold 1.045 -> only 1 row clears it, but k_min=3 lifts
        the ranker keep to the top-3 by rank (plus the gated row)."""
        df, scores = _one_pool()
        gate = np.zeros(12, dtype=bool)
        gate[2] = True
        keep = guarded_keep_mask(df, scores, q=0.95, k_min=3, k_max=0, gate_mask=gate)
        assert set(np.flatnonzero(keep)) == {2, 9, 10, 11}

    def test_k_max_smaller_than_gate_count_gates_all_stay(self):
        """k_max=2 with 3 gated members: the cap binds the RANKER only — all 3
        gates stay via the union; the ranker adds exactly 2, none beyond."""
        df, scores = _one_pool()
        gate = np.zeros(12, dtype=bool)
        gate[[0, 1, 2]] = True  # 3 gated rows, all low-scored
        keep = guarded_keep_mask(df, scores, q=0.5, k_min=3, k_max=2, gate_mask=gate)
        assert set(np.flatnonzero(keep)) == {0, 1, 2, 10, 11}
        assert int(keep.sum()) == 5  # 3 gates + 2 ranker (cap), nothing more

    def test_no_gates_pure_ranker(self):
        df, scores = _one_pool()
        keep = guarded_keep_mask(df, scores, q=0.5, k_min=3, k_max=0,
                                 gate_mask=np.zeros(12, dtype=bool))
        assert set(np.flatnonzero(keep)) == {6, 7, 8, 9, 10, 11}

    def test_length_mismatch_raises(self):
        df, scores = _one_pool()
        with pytest.raises(ValueError):
            guarded_keep_mask(df, scores[:5], q=0.5, k_min=3, k_max=0,
                              gate_mask=np.zeros(12, dtype=bool))


# ===========================================================================
# recall / kept computation
# ===========================================================================
class TestRecallAndKept:
    def test_basic(self):
        recall, kept = recall_and_kept([1, 1, 0, 0], [True, False, True, False])
        assert recall == pytest.approx(0.5)
        assert kept == pytest.approx(0.5)

    def test_no_positives_vacuous_recall_one(self):
        recall, kept = recall_and_kept([0, 0, 0], [True, False, False])
        assert recall == 1.0
        assert kept == pytest.approx(1 / 3)

    def test_fixture_guarded_vs_calibrated_only(self):
        """On the 4-pool fixture at q=0.5: each pool's ranker keeps its top-6;
        the gate union rescues the low-scored positive in (r1, pA) -> guarded
        recall 1.0 (25/48 kept) vs calibrated-only recall 4/5 (24/48 kept)."""
        df = _fixture_df()
        scores = df["score"].to_numpy()
        gate = df["unit_gate"].to_numpy() == 1.0

        keep_g = guarded_keep_mask_all(df, scores, q=0.5, k_min=3, k_max=0, gate_mask=gate)
        recall_g, kept_g = recall_and_kept(df["used"], keep_g)
        assert recall_g == 1.0
        assert kept_g == pytest.approx(25 / 48)

        keep_c = guarded_keep_mask_all(df, scores, q=0.5, k_min=3, k_max=0,
                                       gate_mask=np.zeros(len(df), dtype=bool))
        recall_c, kept_c = recall_and_kept(df["used"], keep_c)
        assert recall_c == pytest.approx(4 / 5)
        assert kept_c == pytest.approx(24 / 48)

    def test_frontier_table_rows(self):
        df = _fixture_df()
        scores = df["score"].to_numpy()
        gate = df["unit_gate"].to_numpy() == 1.0
        rows = frontier_table(df, scores, gate, qs=[0.5, 0.95], k_min=3, k_max=0)
        assert [r["q"] for r in rows] == [0.5, 0.95]
        assert rows[0]["recall"] == 1.0
        assert rows[0]["kept_frac"] == pytest.approx(25 / 48)
        assert rows[0]["gate_only_kept_frac"] == pytest.approx(1 / 48)
        # q=0.95: k_min=3 keeps each pool's top-3 (positives are top-scored)
        # and the gate rescues the low positive -> recall stays 1.0, 13/48 kept
        assert rows[1]["recall"] == 1.0
        assert rows[1]["kept_frac"] == pytest.approx(13 / 48)

    def test_topk_keep_mask_all_per_pool(self):
        df = _fixture_df()
        keep = topk_keep_mask_all(df, df["final_score"].to_numpy(), k=1)
        # exactly one row kept per pool — the top-scored one (a positive)
        assert int(keep.sum()) == 4
        recall, kept = recall_and_kept(df["used"], keep)
        assert recall == pytest.approx(4 / 5)  # gate-only positive not in top-1
        assert kept == pytest.approx(4 / 48)


# ===========================================================================
# check_gate_coverage — pass / fail / missing-column
# ===========================================================================
class TestCheckGateCoverage:
    def _df(self, *, unit_gate, table_gate, used=(1, 1, 0)):
        return pd.DataFrame({
            "doc_filename": ["a.pdf", "a.pdf", "b.pdf"],
            "pass_name": ["p1", "p2", "p1"],
            "chunk_index": [0, 1, 2],
            "used": list(used),
            "unit_gate": list(unit_gate),
            "table_gate": list(table_gate),
        })

    def test_pass_all_positives_covered(self):
        df = self._df(unit_gate=[1.0, 0.0, 0.0], table_gate=[0.0, 1.0, 0.0])
        assert check_gate_coverage(df) == []

    def test_fail_uncovered_positive_reported(self):
        df = self._df(unit_gate=[1.0, 0.0, 0.0], table_gate=[0.0, 0.0, 0.0])
        misses = check_gate_coverage(df)
        assert len(misses) == 1
        doc, pas, ci, ug, tg = misses[0]
        assert (doc, pas, ci) == ("a.pdf", "p2", 1)
        assert float(ug) == 0.0 and float(tg) == 0.0

    def test_unused_rows_never_flagged(self):
        # the used==0 row has no gates — must NOT be a miss
        df = self._df(unit_gate=[1.0, 1.0, 0.0], table_gate=[0.0, 0.0, 0.0])
        assert check_gate_coverage(df) == []

    def test_missing_gate_columns_raises_distinct_error(self):
        df = self._df(unit_gate=[1.0, 1.0, 0.0], table_gate=[0.0, 0.0, 0.0])
        df = df.drop(columns=["unit_gate", "table_gate"])
        with pytest.raises(GateColumnsMissing):
            check_gate_coverage(df)

    # --- Task 18 — offline gate recompute (construction-faithful) ----------
    def _df_text(self, *, unit_gate, table_gate, chunk_text, used=(1, 1, 0),
                 pass_name=("p1", "p2", "p1"), is_table=None):
        cols = {
            "doc_filename": ["a.pdf", "a.pdf", "b.pdf"],
            "pass_name": list(pass_name),
            "chunk_index": [0, 1, 2],
            "used": list(used),
            "unit_gate": list(unit_gate),
            "table_gate": list(table_gate),
            "chunk_text": list(chunk_text),
        }
        if is_table is not None:
            cols["is_table"] = list(is_table)
        return pd.DataFrame(cols)

    def test_offline_recompute_rescues_fallback_path_positive(self):
        """A used==1 row whose captured unit_gate=0 (fallback-path doc) but whose
        chunk_text contains '50 km' for a km-signature pass PASSES coverage via
        the offline G1 recompute — no false-fail on the fallback gap."""
        df = self._df_text(
            unit_gate=[0.0, 1.0, 0.0],
            table_gate=[0.0, 0.0, 0.0],
            chunk_text=["max range 50 km", "irrelevant", "n/a"],
        )
        # p1 carries a km signature; recompute fires on row 0's "50 km".
        signatures = {"p1": ("km",), "p2": ("kw",)}
        assert check_gate_coverage(df, signatures=signatures) == []

    def test_offline_recompute_still_fails_when_predicate_does_not_fire(self):
        """If neither captured flag NOR the recomputed predicate fires, the
        positive is still reported as an uncovered miss (no silent pass)."""
        df = self._df_text(
            unit_gate=[0.0, 1.0, 0.0],
            table_gate=[0.0, 0.0, 0.0],
            chunk_text=["no numbers no units here", "irrelevant", "n/a"],
        )
        signatures = {"p1": ("km",), "p2": ("kw",)}
        misses = check_gate_coverage(df, signatures=signatures)
        assert len(misses) == 1
        assert (misses[0][0], misses[0][1], misses[0][2]) == ("a.pdf", "p1", 0)

    def test_offline_recompute_g2_table_path(self):
        """G2 recompute: a table row (is_table=1) with a unit token but NO digit
        passes via the offline table-gate predicate."""
        df = self._df_text(
            unit_gate=[0.0, 1.0, 0.0],
            table_gate=[0.0, 0.0, 0.0],
            chunk_text=["power rating in kw", "irrelevant", "n/a"],
            is_table=[1, 0, 0],
        )
        signatures = {"p1": ("kw",), "p2": ("kw",)}
        assert check_gate_coverage(df, signatures=signatures) == []

    def test_offline_recompute_none_signatures_is_captured_only(self):
        """signatures=None (default) → behaves exactly like the captured-flag
        check (back-compat): the same uncovered positive is reported."""
        df = self._df_text(
            unit_gate=[0.0, 1.0, 0.0],
            table_gate=[0.0, 0.0, 0.0],
            chunk_text=["max range 50 km", "irrelevant", "n/a"],
        )
        misses = check_gate_coverage(df)  # no signatures → no recompute
        assert len(misses) == 1
        assert (misses[0][0], misses[0][1], misses[0][2]) == ("a.pdf", "p1", 0)


# ===========================================================================
# quantile-grid parser
# ===========================================================================
class TestParseQuantileGrid:
    def test_default_grid_inclusive_stop(self):
        qs = parse_quantile_grid("0.5:0.95:0.05")
        assert qs == pytest.approx(
            [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        assert qs[-1] == pytest.approx(0.95)  # FP-safe inclusive stop

    def test_single_point(self):
        assert parse_quantile_grid("0.5:0.5:0.05") == [0.5]

    @pytest.mark.parametrize("bad", [
        "0.5:0.95",          # not 3 parts
        "0.5:0.4:0.05",      # stop < start
        "0.5:0.95:0",        # step <= 0
        "0.5:1.5:0.5",       # leaves [0, 1]
        "-0.1:0.5:0.1",      # leaves [0, 1]
    ])
    def test_bad_specs_raise(self, bad):
        with pytest.raises(ValueError):
            parse_quantile_grid(bad)
