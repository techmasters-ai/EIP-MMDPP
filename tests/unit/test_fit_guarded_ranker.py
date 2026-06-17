"""Unit tests for the guarded-ranker calibration script (Task 19).

scripts.fit_guarded_ranker turns a bake-off dataset into deployable
``ranker_weights`` + ``quantile_q`` with a finite-sample margin and honest LODO
reporting. These tests drive a SYNTHETIC, separable fixture (no DB, no live
dataset) through the public surface:

  * ``fit_and_select`` — the end-to-end fit/selection (sign-constrained LogReg →
    LODO scores → smallest-q-with-recall-1.0 → margin step-down), returning the
    result dict the JSON/manifest are built from.
  * ``build_manifest_snippet`` — the ready-to-paste YAML.
  * ``main`` — the CLI, including the gate-coverage REFUSAL path.

The fixture is built so the sign pattern, the chosen q (smallest with LODO
recall 1.0, minus the margin), and the gate-coverage refusal are all
deterministic.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.fit_guarded_ranker import (
    DEFAULT_FEATURES,
    apply_margin,
    build_manifest_snippet,
    choose_quantile,
    fit_and_select,
    main,
    parse_c_grid,
)


# ===========================================================================
# Fixtures
# ===========================================================================
def _separable_df(*, n_docs: int = 4, n_per_pool: int = 12, seed: int = 0,
                  inject_bad_sign: bool = False, gate_hole: bool = False) -> pd.DataFrame:
    """Synthetic, SEPARABLE bake-off dataset.

    ``n_docs`` runs, one (run, pass) pool each, ``n_per_pool`` rows. In every
    pool the top-scoring row is the positive (``used==1``); ``cosine`` is the
    clean separating signal (positives ~0.9, negatives ~0.2). Gate columns are
    set so EVERY positive is gate-covered (unit_gate=1) — so the recall floor is
    satisfiable and the gate union alone holds recall 1.0.

    ``inject_bad_sign``: give ``negative_norm`` a strong POSITIVE correlation
    with the label (it must be dropped — its expected sign is <= 0).

    ``gate_hole``: clear the gate flags on one positive so the recall-floor
    precondition (check_gate_coverage) FAILS — drives the refusal path.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for d in range(n_docs):
        n = n_per_pool
        cosine = np.clip(rng.normal(0.2, 0.05, n), 0.0, 1.0)
        cosine[-1] = 0.92  # the positive sits clearly on top
        used = np.zeros(n, dtype=int)
        used[-1] = 1
        rec = {
            "run_id": [f"run{d}"] * n,
            "document_id": [f"doc{d}"] * n,
            "doc_filename": [f"doc{d}.pdf"] * n,
            "pass_name": ["pA"] * n,
            "chunk_index": list(range(n)),
            "used": used,
            "final_score": cosine,
            "cosine": cosine,
            # field-cosine signals track cosine (positive correlation → kept)
            "max_field_cosine": cosine,
            "mean_top3_field_cosine": cosine * 0.9,
            "rerank_norm": cosine,
            "is_table": np.zeros(n),
            "unit_token_count": (used * 3).astype(float),
            "digit_density": (used * 0.4).astype(float),
            "label_value_lines": (used * 2).astype(float),
            # negative_norm: by default anti-correlated (its expected sign <= 0,
            # so it survives). inject_bad_sign flips it positive → must be dropped.
            "negative_norm": (cosine if inject_bad_sign else (1.0 - cosine)),
            "unit_gate": np.zeros(n),
            "table_gate": np.zeros(n),
            "chunk_text": ["text"] * n,
        }
        df = pd.DataFrame(rec)
        df.loc[n - 1, "unit_gate"] = 1.0  # the positive is gate-covered
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if gate_hole:
        # punch a hole: a positive covered by NEITHER gate
        hole = out.index[out["used"] == 1][0]
        out.loc[hole, "unit_gate"] = 0.0
        out.loc[hole, "table_gate"] = 0.0
    return out


# ===========================================================================
# C-grid parser
# ===========================================================================
class TestParseCGrid:
    def test_basic(self):
        assert parse_c_grid("0.01,0.1,1,10") == pytest.approx([0.01, 0.1, 1.0, 10.0])

    def test_single(self):
        assert parse_c_grid("1.0") == [1.0]

    @pytest.mark.parametrize("bad", ["", "0,1", "-1,1", "abc"])
    def test_bad_raises(self, bad):
        with pytest.raises(ValueError):
            parse_c_grid(bad)


# ===========================================================================
# fit_and_select — sign pattern, chosen q, margin
# ===========================================================================
class TestFitAndSelect:
    def test_sign_pattern_clean_keeps_all(self):
        """Clean fixture: every feature carries its expected sign → none dropped
        (negative_norm anti-correlated, the rest non-negative)."""
        df = _separable_df()
        res = fit_and_select(df, features=list(DEFAULT_FEATURES))
        # No feature should be dropped on the clean, well-signed fixture.
        assert res["features_dropped_by_sign"] == []
        assert set(res["features_kept"]) == set(DEFAULT_FEATURES)
        # negative_norm retains a non-positive coefficient.
        w = res["ranker_weights"]
        assert w["negative_norm"] <= 1e-9
        # the field-cosine signals retain non-negative coefficients.
        for f in ("cosine", "max_field_cosine", "mean_top3_field_cosine", "rerank_norm"):
            assert w[f] >= -1e-9

    def test_wrong_signed_feature_dropped(self):
        """negative_norm given a POSITIVE correlation violates its expected
        sign (<= 0) → it is dropped and logged."""
        df = _separable_df(inject_bad_sign=True)
        res = fit_and_select(df, features=list(DEFAULT_FEATURES))
        assert "negative_norm" not in res["features_kept"]
        dropped = [d["feature"] for d in res["features_dropped_by_sign"]]
        assert "negative_norm" in dropped

    def test_chosen_q_is_smallest_recall_one_minus_margin(self):
        """The gate union alone holds recall 1.0 on every fold (all positives
        gate-covered), so the SMALLEST grid q already gives recall 1.0; the
        margin then steps q DOWN by `margin_steps` grid steps (clamped to the
        grid floor)."""
        df = _separable_df()
        grid = [0.5, 0.55, 0.6, 0.65, 0.7]
        res = fit_and_select(df, features=list(DEFAULT_FEATURES),
                             quantile_grid=grid, margin_steps=1)
        # smallest q with recall 1.0 is the grid floor here; margin clamps to it.
        assert res["quantile_q_before_margin"] == pytest.approx(0.5)
        assert res["quantile_q"] == pytest.approx(0.5)
        # every fold holds recall 1.0
        assert all(f["recall"] == pytest.approx(1.0) for f in res["per_fold"])

    def test_margin_clamps_at_grid_floor_through_fit(self):
        """End-to-end: the gate union holds recall 1.0 at the grid floor, so the
        pre-margin q IS the floor and the margin can only clamp (never below the
        floor) regardless of margin_steps. (The downward step-down arithmetic
        itself is asserted directly in TestApplyMargin — `choose_quantile` cannot
        return a non-floor q because the guarded keep-set is monotone in q, so a
        passing floor is always the smallest passing q.)"""
        df = _separable_df()
        grid = [0.5, 0.55, 0.6, 0.65, 0.7]
        res = fit_and_select(df, features=list(DEFAULT_FEATURES),
                             quantile_grid=grid, margin_steps=2)
        assert res["quantile_q_before_margin"] == pytest.approx(0.5)
        assert res["quantile_q"] == pytest.approx(0.5)  # clamped at the floor

    def test_lodo_both_conventions_present(self):
        df = _separable_df()
        res = fit_and_select(df, features=list(DEFAULT_FEATURES))
        assert "pooled_oof_auroc" in res["lodo"]
        assert "mean_per_fold_auroc" in res["lodo"]
        # separable fixture → strong AUROC
        assert res["lodo"]["pooled_oof_auroc"] >= 0.9

    def test_does_not_mutate_caller_dataframe(self):
        """fit_and_select fills missing feature columns with 0.0 internally; that
        must NOT leak back to the caller's DataFrame (it copies at entry)."""
        df = _separable_df().drop(columns=["is_table"])  # one DEFAULT feature absent
        cols_before = set(df.columns)
        fit_and_select(df, features=list(DEFAULT_FEATURES))
        assert set(df.columns) == cols_before  # no 'is_table' added to the caller's df
        assert "is_table" not in df.columns


# ===========================================================================
# apply_margin — the downward step-down arithmetic (the real margin)
# ===========================================================================
class TestApplyMargin:
    GRID = [0.5, 0.55, 0.6, 0.65, 0.7]

    def test_steps_down_one(self):
        """pre-margin q=0.6, one step → 0.55 (a genuine downward move)."""
        assert apply_margin(self.GRID, 0.6, margin_steps=1) == pytest.approx(0.55)

    def test_steps_down_two(self):
        """pre-margin q=0.7, two steps → 0.6."""
        assert apply_margin(self.GRID, 0.7, margin_steps=2) == pytest.approx(0.6)

    def test_clamps_at_floor(self):
        """q already at the grid floor → clamps, never below 0.5."""
        assert apply_margin(self.GRID, 0.5, margin_steps=2) == pytest.approx(0.5)

    def test_zero_steps_is_identity(self):
        assert apply_margin(self.GRID, 0.65, margin_steps=0) == pytest.approx(0.65)

    def test_step_below_floor_clamps(self):
        """Stepping further than there is room clamps to the floor (q=0.55, 5
        steps → 0.5, not an index error)."""
        assert apply_margin(self.GRID, 0.55, margin_steps=5) == pytest.approx(0.5)

    def test_negative_steps_treated_as_zero(self):
        """margin_steps<0 is floored to 0 (no upward move)."""
        assert apply_margin(self.GRID, 0.6, margin_steps=-3) == pytest.approx(0.6)

    def test_off_grid_q_snaps_to_nearest_then_steps(self):
        """A pre-margin q not exactly on the grid snaps to the nearest grid point
        (FP-safe), then steps down: 0.62 → nearest 0.6 → one step → 0.55."""
        assert apply_margin(self.GRID, 0.62, margin_steps=1) == pytest.approx(0.55)


# ===========================================================================
# choose_quantile — ranker-driven recall path + the None (no-safe-q) branch
# ===========================================================================
def _ranker_pool(run, *, pos_index, n=12, gate=False):
    """One (run, pass) pool: deterministic scores 0.0..(n-1)*0.1, the positive at
    rank ``pos_index`` (so the RANKER, not the gate, must keep it unless gate)."""
    scores = np.arange(n) * 0.1
    used = np.zeros(n, dtype=int)
    used[pos_index] = 1
    ug = np.zeros(n)
    if gate:
        ug[pos_index] = 1.0
    return pd.DataFrame({
        "run_id": [run] * n,
        "pass_name": ["pA"] * n,
        "chunk_index": list(range(n)),
        "used": used,
        "unit_gate": ug,
    }), scores


class TestChooseQuantile:
    """The guarded keep-set is monotone in q (lower q keeps MORE rows), so the
    SMALLEST q that holds recall 1.0 is the grid floor whenever ANY q does; a
    non-floor q can never be the *smallest* passing q. These tests pin that
    contract: the ranker-driven (no-gate) floor pick, and the None branch when no
    q reaches recall 1.0."""

    def _df_scores(self, pools):
        frames, sc = [], []
        for d, s in pools:
            frames.append(d)
            sc.append(s)
        df = pd.concat(frames, ignore_index=True)
        return df, np.concatenate(sc)

    def test_ranker_drives_recall_floor_chosen(self):
        """No gate coverage at all: the RANKER must keep the positives. Each
        positive is the top-scored row of its pool, so the ranker keeps it at
        every q → recall 1.0 everywhere → the smallest q (the floor) is chosen."""
        df, scores = self._df_scores([
            _ranker_pool("r0", pos_index=11),  # top-scored positive, no gate
            _ranker_pool("r1", pos_index=11),
        ])
        gate = df["unit_gate"].to_numpy() == 1.0  # all-False (no gate)
        groups = df["run_id"].to_numpy()
        grid = [0.5, 0.55, 0.6, 0.65, 0.7]
        chosen, summary = choose_quantile(df, scores, gate, groups, grid,
                                          k_min=3, k_max=0)
        assert chosen == pytest.approx(0.5)  # ranker-driven, floor is chosen
        assert all(s["recall_one_every_fold"] for s in summary)
        # and the margin then steps DOWN from a non-floor pre-margin q only when
        # one exists; here it clamps at the floor.
        assert apply_margin(grid, chosen, margin_steps=1) == pytest.approx(0.5)

    def test_returns_none_when_no_q_reaches_recall_one(self):
        """A low-ranked positive with NO gate and k_min too small for even the
        lowest q to keep it → no q in the grid holds recall 1.0 → chosen is
        None (the SystemExit guard in fit_and_select fires on this)."""
        df, scores = self._df_scores([
            _ranker_pool("r0", pos_index=0),  # bottom-ranked positive, no gate
            _ranker_pool("r1", pos_index=0),
        ])
        gate = df["unit_gate"].to_numpy() == 1.0  # all-False
        groups = df["run_id"].to_numpy()
        grid = [0.5, 0.55, 0.6, 0.65, 0.7]
        chosen, summary = choose_quantile(df, scores, gate, groups, grid,
                                          k_min=1, k_max=0)
        assert chosen is None
        assert all(not s["recall_one_every_fold"] for s in summary)


# ===========================================================================
# manifest snippet
# ===========================================================================
class TestManifestSnippet:
    def test_snippet_is_valid_yaml_with_required_keys(self):
        df = _separable_df()
        res = fit_and_select(df, features=list(DEFAULT_FEATURES))
        snippet = build_manifest_snippet(res, k_min=3, k_max=0)
        parsed = yaml.safe_load(snippet)
        assert parsed["selection_mode"] == "guarded_quantile"
        assert parsed["quantile_q"] == pytest.approx(res["quantile_q"])
        assert parsed["k_min"] == 3
        assert "k_max" in parsed
        # ranker_weights keys are the dataset column names exactly
        rw = parsed["ranker_weights"]
        assert "max_field_cosine" in rw
        assert "mean_top3_field_cosine" in rw
        assert "rerank_norm" in rw
        # only kept features appear
        assert set(rw) == set(res["features_kept"])


# ===========================================================================
# main — JSON emission + gate-coverage refusal
# ===========================================================================
class TestMain:
    def test_emits_json_with_expected_keys(self, tmp_path):
        csv = tmp_path / "data.csv"
        _separable_df().to_csv(csv, index=False)
        out = tmp_path / "fit.json"
        rc = main([
            "--csv", str(csv),
            "--out", str(out),
            "--quantile-grid", "0.5:0.7:0.05",
            "--margin-steps", "1",
        ])
        assert rc == 0
        payload = json.loads(out.read_text())
        for key in ("ranker_weights", "quantile_q", "quantile_q_before_margin",
                    "features_kept", "features_dropped_by_sign", "per_fold",
                    "lodo", "manifest_snippet"):
            assert key in payload, f"missing JSON key {key!r}"
        assert payload["lodo"]["pooled_oof_auroc"] is not None
        assert payload["lodo"]["mean_per_fold_auroc"] is not None
        # the embedded manifest snippet round-trips
        parsed = yaml.safe_load(payload["manifest_snippet"])
        assert parsed["selection_mode"] == "guarded_quantile"

    def test_refuses_when_gate_coverage_fails(self, tmp_path, capsys):
        """A positive covered by neither gate fails the recall-floor
        precondition → non-zero exit, no JSON written."""
        csv = tmp_path / "hole.csv"
        _separable_df(gate_hole=True).to_csv(csv, index=False)
        out = tmp_path / "fit.json"
        rc = main(["--csv", str(csv), "--out", str(out)])
        assert rc != 0
        assert not out.exists()
        err = capsys.readouterr().out + capsys.readouterr().err
        assert "GATE COVERAGE" in err.upper() or "gate" in err.lower()
