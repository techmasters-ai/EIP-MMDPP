"""Recompute selection performance (recall/precision/frac-sel/F1 + per-signal)
for the SHIPPED absolute_union vs guarded_quantile, over the bake-off ground truth.

OFFLINE note: image arm omitted (source_refs absent from parquet);
image-signal coverage is validated via the live Task 7 sweep instead.
Cosine threshold τ=0.55.  Baseline = guarded_quantile q=0.5 median cut.

Datasets:
  reports/dataset/bakeoff_dataset.parquet               — SA-2 corpus v0
  reports/dataset_generalization/bakeoff_dataset.parquet — generalization corpus (has max_field_cosine)
  reports/dataset_v1_relabel/bakeoff_dataset.parquet     — SA-2 corpus v1 (relabelled)
  reports/dataset_v2/bakeoff_dataset.parquet             — SA-2 corpus v2 (has max_field_cosine)

Columns max_field_cosine is absent in dataset/ and dataset_v1_relabel/;
'cosine' (query-chunk embedding similarity) is used as fallback for those rows.
"""
from __future__ import annotations

import glob
import os
import sys
import numpy as np
import pandas as pd

# --- shipped production imports ---
from app.services.extraction_pass_signal_config import derive_pass_signal_config
from app.services.extraction_signal_detectors import measurement_present, categorical_present

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
TAU = 0.55          # cosine gate threshold (shipped value)
QUANTILE = 0.5      # guarded_quantile median cut (deployed baseline)
REPORT_PATH = os.path.join(os.path.dirname(__file__), "absolute_union_metrics_report.txt")
SPEC_RECALL_REF = 0.896   # offline-prototype reference from spec (~89.6%)
SPEC_FRAC_REF = 0.22      # ~22% selected
SPEC_PREC_REF = 0.077     # ~7.7% precision
TOLERANCE_PTS = 0.03      # ±3 pp = within tolerance

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load + concatenate all 4 bake-off parquets
# ─────────────────────────────────────────────────────────────────────────────
parquet_paths = sorted(glob.glob("reports/*/bakeoff_dataset.parquet"))
if not parquet_paths:
    sys.exit("ERROR: no reports/*/bakeoff_dataset.parquet files found; run from repo root.")

frames = []
for p in parquet_paths:
    df_src = pd.read_parquet(p)
    df_src["_source"] = os.path.basename(os.path.dirname(p))
    frames.append(df_src)

raw = pd.concat(frames, ignore_index=True)
print(f"Loaded {len(raw):,} rows from {len(parquet_paths)} parquets before dedup.")

# Dedup on (run_id, pass_name, chunk_index); keep last (most recent dataset wins)
raw = raw.drop_duplicates(subset=["run_id", "pass_name", "chunk_index"], keep="last")
print(f"After dedup: {len(raw):,} rows.")

# Verify expected columns
required_cols = {"run_id", "pass_name", "chunk_index", "used", "chunk_text"}
missing = required_cols - set(raw.columns)
if missing:
    sys.exit(f"ERROR: missing required columns: {missing}")

# Coerce types
raw["used"] = pd.to_numeric(raw["used"], errors="coerce").fillna(0).astype(int)
raw["chunk_text"] = raw["chunk_text"].astype(str)

# Build 'mfc' = max_field_cosine where available, else fall back to 'cosine'
if "max_field_cosine" in raw.columns:
    raw["mfc"] = pd.to_numeric(raw["max_field_cosine"], errors="coerce")
else:
    raw["mfc"] = np.nan

if "cosine" in raw.columns:
    raw["cosine_fallback"] = pd.to_numeric(raw["cosine"], errors="coerce")
else:
    raw["cosine_fallback"] = np.nan

fallback_mask = raw["mfc"].isna() & raw["cosine_fallback"].notna()
raw["mfc"] = raw["mfc"].where(~fallback_mask, raw["cosine_fallback"])
raw["mfc"] = raw["mfc"].fillna(0.0)

n_fallback = int(fallback_mask.sum())
n_mfc = int((~fallback_mask).sum())
print(f"Cosine source: max_field_cosine={n_mfc:,}  fallback-to-cosine={n_fallback:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Restrict to the 9 routable passes
# ─────────────────────────────────────────────────────────────────────────────
CFG = derive_pass_signal_config("air_defense_v3")
routable = set(CFG.keys())
pre_filter = len(raw)
df = raw[raw["pass_name"].isin(routable)].copy()
post_filter = len(df)
dropped = pre_filter - post_filter
if dropped:
    print(f"Dropped {dropped:,} rows for non-routable passes.")
print(f"Working dataset: {post_filter:,} rows, {df['used'].sum()} positives, "
      f"{df['pass_name'].nunique()} passes, {df['run_id'].nunique()} runs.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Compute per-row signals
# ─────────────────────────────────────────────────────────────────────────────
def _apply_signals(row) -> tuple[bool, bool, bool]:
    """Returns (measurement_keep, categorical_keep, cosine_keep)."""
    c = CFG.get(row["pass_name"])
    if c is None:
        return False, False, False
    m = measurement_present(c.dimensions, row["chunk_text"])
    cat = categorical_present(c.categorical_fields, row["chunk_text"])
    k = bool(row["mfc"] >= TAU)
    return m, cat, k

sig = df.apply(_apply_signals, axis=1, result_type="expand")
df["sig_measurement"] = sig[0].astype(bool)
df["sig_categorical"] = sig[1].astype(bool)
df["sig_cosine"] = sig[2].astype(bool)
# Image arm: omitted offline (source_refs not in parquet); counted as False here
df["sig_image"] = False

# absolute_union keep = any arm fires
df["au_keep"] = df["sig_measurement"] | df["sig_categorical"] | df["sig_cosine"] | df["sig_image"]

# guarded_quantile keep: per (run_id, pass_name) group median of mfc
df["gq_keep"] = df.groupby(["run_id", "pass_name"])["mfc"].transform(
    lambda s: s >= s.quantile(QUANTILE)
).astype(bool)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────
def metrics(keep_mask: pd.Series, used_mask: pd.Series) -> dict:
    sel = int(keep_mask.sum())
    pos = int(used_mask.sum())
    tp = int((keep_mask & used_mask).sum())
    total = len(keep_mask)
    rec = tp / pos if pos else float("nan")
    prec = tp / sel if sel else float("nan")
    frac = sel / total if total else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"recall": rec, "precision": prec, "frac_sel": frac, "f1": f1,
            "tp": tp, "kept": sel, "positives": pos, "total": total}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Aggregate metrics
# ─────────────────────────────────────────────────────────────────────────────
used_mask = df["used"] == 1
agg_au = metrics(df["au_keep"], used_mask)
agg_gq = metrics(df["gq_keep"], used_mask)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Per-pass metrics
# ─────────────────────────────────────────────────────────────────────────────
per_pass_rows = []
for pn in sorted(routable):
    pmask = df["pass_name"] == pn
    sub = df[pmask]
    if len(sub) == 0:
        continue
    u = sub["used"] == 1
    au_m = metrics(sub["au_keep"], u)
    gq_m = metrics(sub["gq_keep"], u)
    per_pass_rows.append({
        "pass": pn,
        "total": len(sub),
        "positives": int(u.sum()),
        # au
        "au_recall": au_m["recall"],
        "au_precision": au_m["precision"],
        "au_frac_sel": au_m["frac_sel"],
        "au_f1": au_m["f1"],
        "au_kept": au_m["kept"],
        "au_tp": au_m["tp"],
        # gq
        "gq_recall": gq_m["recall"],
        "gq_precision": gq_m["precision"],
        "gq_frac_sel": gq_m["frac_sel"],
        "gq_f1": gq_m["f1"],
        "gq_kept": gq_m["kept"],
        "gq_tp": gq_m["tp"],
    })

# ─────────────────────────────────────────────────────────────────────────────
# 7. Per-signal keep counts (absolute_union signal attribution)
# ─────────────────────────────────────────────────────────────────────────────
sig_rows = []
for pn in sorted(routable):
    pmask = df["pass_name"] == pn
    sub = df[pmask]
    if len(sub) == 0:
        continue
    u = sub["used"] == 1
    # exclusive attribution for clear breakdown (also show overlap counts)
    m_only = sub["sig_measurement"] & ~sub["sig_categorical"] & ~sub["sig_cosine"]
    c_only = ~sub["sig_measurement"] & sub["sig_categorical"] & ~sub["sig_cosine"]
    k_only = ~sub["sig_measurement"] & ~sub["sig_categorical"] & sub["sig_cosine"]
    sig_rows.append({
        "pass": pn,
        # raw keep counts per signal (a row can appear in multiple)
        "meas_keeps": int(sub["sig_measurement"].sum()),
        "meas_tp": int((sub["sig_measurement"] & u).sum()),
        "cat_keeps": int(sub["sig_categorical"].sum()),
        "cat_tp": int((sub["sig_categorical"] & u).sum()),
        "cos_keeps": int(sub["sig_cosine"].sum()),
        "cos_tp": int((sub["sig_cosine"] & u).sum()),
        "img_keeps": 0,  # omitted offline
        "img_tp": 0,
        # exclusive (any-arm attribution)
        "meas_only_keeps": int(m_only.sum()),
        "meas_only_tp": int((m_only & u).sum()),
        "cat_only_keeps": int(c_only.sum()),
        "cat_only_tp": int((c_only & u).sum()),
        "cos_only_keeps": int(k_only.sum()),
        "cos_only_tp": int((k_only & u).sum()),
    })

# ─────────────────────────────────────────────────────────────────────────────
# 8. Format report
# ─────────────────────────────────────────────────────────────────────────────
SEP = "=" * 100
sep2 = "-" * 100

lines = []
def w(*args):
    lines.append(" ".join(str(a) for a in args))

w(SEP)
w("ABSOLUTE_UNION VS GUARDED_QUANTILE — OFFLINE BAKE-OFF METRICS RECOMPUTE")
w(f"  Offline: image arm OMITTED (source_refs absent from parquet).")
w(f"           Image-signal coverage validated via live Task 7 sweep.")
w(f"  Cosine threshold τ={TAU}  |  Baseline = guarded_quantile q={QUANTILE} (median per group)")
w(f"  Cosine source: max_field_cosine where available; fallback to 'cosine' for older datasets.")
w(f"  Rows with max_field_cosine: {n_mfc:,}   Fallback-to-cosine: {n_fallback:,}")
w(f"  Datasets loaded: {', '.join(os.path.basename(os.path.dirname(p)) for p in parquet_paths)}")
w(f"  Total rows (post-dedup, routable passes): {post_filter:,}")
w(f"  Total positives (used==1): {int(used_mask.sum())}")
w(SEP)
w()

# Aggregate table
w("AGGREGATE PERFORMANCE")
w(sep2)
w(f"{'Selector':<22} {'Recall':>8} {'Precision':>10} {'Frac-Sel':>10} {'F1':>8} "
  f"{'TP':>6} {'Kept':>7} {'Positives':>10} {'Total':>8}")
w(sep2)
for label, m in [("absolute_union", agg_au), ("guarded_quantile", agg_gq)]:
    w(f"{label:<22} {m['recall']:>8.3f} {m['precision']:>10.3f} {m['frac_sel']:>10.3f} "
      f"{m['f1']:>8.3f} {m['tp']:>6} {m['kept']:>7} {m['positives']:>10} {m['total']:>8}")
w(sep2)
w()

# Per-pass table
w("PER-PASS PERFORMANCE")
w(sep2)
hdr = (f"{'Pass':<25} {'Tot':>5} {'Pos':>4} "
       f"| {'AU_Rec':>7} {'AU_Prec':>8} {'AU_Fsel':>8} {'AU_F1':>7} {'AU_Kpt':>7} "
       f"| {'GQ_Rec':>7} {'GQ_Prec':>8} {'GQ_Fsel':>8} {'GQ_F1':>7} {'GQ_Kpt':>7}")
w(hdr)
w(sep2)
for r in per_pass_rows:
    def _fmt(v):
        if np.isnan(v): return "  n/a  "
        return f"{v:.3f}"
    w(f"{r['pass']:<25} {r['total']:>5} {r['positives']:>4} "
      f"| {_fmt(r['au_recall']):>7} {_fmt(r['au_precision']):>8} "
      f"{_fmt(r['au_frac_sel']):>8} {_fmt(r['au_f1']):>7} {r['au_kept']:>7} "
      f"| {_fmt(r['gq_recall']):>7} {_fmt(r['gq_precision']):>8} "
      f"{_fmt(r['gq_frac_sel']):>8} {_fmt(r['gq_f1']):>7} {r['gq_kept']:>7}")
w(sep2)
w()

# Per-signal keep counts table
w("PER-SIGNAL KEEP COUNTS (absolute_union, measurement/categorical/cosine arms)")
w("  image arm: 0 keeps offline — covered by live Task 7 sweep")
w("  'keeps' = rows where arm fires; 'tp' = positive rows captured by that arm")
w("  'excl_*' = keeps where ONLY that arm fires (exclusive attribution)")
w(sep2)
hdr2 = (f"{'Pass':<25} "
        f"{'M_keeps':>8} {'M_tp':>6} "
        f"{'C_keeps':>8} {'C_tp':>6} "
        f"{'K_keeps':>8} {'K_tp':>6} "
        f"{'M_excl':>7} {'M_ex_tp':>8} "
        f"{'C_excl':>7} {'C_ex_tp':>8} "
        f"{'K_excl':>7} {'K_ex_tp':>8}")
w(hdr2)
w(sep2)
sig_totals = {k: 0 for k in ["meas_keeps","meas_tp","cat_keeps","cat_tp",
                               "cos_keeps","cos_tp",
                               "meas_only_keeps","meas_only_tp",
                               "cat_only_keeps","cat_only_tp",
                               "cos_only_keeps","cos_only_tp"]}
for r in sig_rows:
    for k in sig_totals:
        sig_totals[k] += r[k]
    w(f"{r['pass']:<25} "
      f"{r['meas_keeps']:>8} {r['meas_tp']:>6} "
      f"{r['cat_keeps']:>8} {r['cat_tp']:>6} "
      f"{r['cos_keeps']:>8} {r['cos_tp']:>6} "
      f"{r['meas_only_keeps']:>7} {r['meas_only_tp']:>8} "
      f"{r['cat_only_keeps']:>7} {r['cat_only_tp']:>8} "
      f"{r['cos_only_keeps']:>7} {r['cos_only_tp']:>8}")
w(sep2)
w(f"{'TOTAL':<25} "
  f"{sig_totals['meas_keeps']:>8} {sig_totals['meas_tp']:>6} "
  f"{sig_totals['cat_keeps']:>8} {sig_totals['cat_tp']:>6} "
  f"{sig_totals['cos_keeps']:>8} {sig_totals['cos_tp']:>6} "
  f"{sig_totals['meas_only_keeps']:>7} {sig_totals['meas_only_tp']:>8} "
  f"{sig_totals['cat_only_keeps']:>7} {sig_totals['cat_only_tp']:>8} "
  f"{sig_totals['cos_only_keeps']:>7} {sig_totals['cos_only_tp']:>8}")
w(sep2)
w()

# ─────────────────────────────────────────────────────────────────────────────
# 9. Spec-reference comparison (sanity gate)
# ─────────────────────────────────────────────────────────────────────────────
w("SPEC REFERENCE COMPARISON (sanity gate)")
w(sep2)
au_rec = agg_au["recall"]
au_prec = agg_au["precision"]
au_frac = agg_au["frac_sel"]

w(f"  Spec offline-prototype: recall≈{SPEC_RECALL_REF:.1%}  frac-sel≈{SPEC_FRAC_REF:.1%}"
  f"  precision≈{SPEC_PREC_REF:.1%}")
w(f"  Shipped absolute_union: recall= {au_rec:.1%}  frac-sel= {au_frac:.1%}"
  f"  precision= {au_prec:.1%}")
recall_delta = au_rec - SPEC_RECALL_REF   # signed: positive = above spec
recall_delta_abs = abs(recall_delta)
if recall_delta >= 0:
    direction = "ABOVE"
    regression = False
elif recall_delta_abs <= TOLERANCE_PTS:
    direction = "BELOW (within tolerance)"
    regression = False
else:
    direction = "BELOW (OUT OF TOLERANCE — REGRESSION)"
    regression = True

w(f"  Recall delta: {recall_delta:+.1%}  ({direction})")
if recall_delta_abs <= TOLERANCE_PTS:
    w(f"  RESULT: WITHIN TOLERANCE — |delta|={recall_delta_abs:.1%} (<= {TOLERANCE_PTS:.0%})")
elif not regression:
    w(f"  RESULT: ABOVE SPEC — shipped code exceeds prototype (not a regression).")
else:
    w(f"  RESULT: REGRESSION — recall dropped {recall_delta_abs:.1%} below spec reference.")
    w(f"  WARNING: Investigate before finalising — likely a dataset or implementation gap.")
w(sep2)
w()

# ─────────────────────────────────────────────────────────────────────────────
# 10. Flag passes with notably low recall
# ─────────────────────────────────────────────────────────────────────────────
LOW_RECALL_THRESHOLD = 0.75
flagged = [(r["pass"], r["au_recall"], r["positives"])
           for r in per_pass_rows
           if not np.isnan(r["au_recall"]) and r["au_recall"] < LOW_RECALL_THRESHOLD
           and r["positives"] > 0]

w("SIGNAL-COVERAGE GAPS (passes where absolute_union recall < 75%, positives > 0)")
w(sep2)
if flagged:
    for pn, rec, pos in flagged:
        w(f"  {pn:<25} recall={rec:.3f}  positives={pos}")
    w()
    w("  ACTION: inspect which signal arms fire for missed positives in these passes.")
else:
    w("  None — all passes with positives meet the 75% recall threshold (or have 0 positives).")
w(sep2)
w()
w(SEP)
w("END OF REPORT")
w(SEP)

report_text = "\n".join(lines)

# Print to stdout
print()
print(report_text)

# Write to file
with open(REPORT_PATH, "w") as fh:
    fh.write(report_text)
    fh.write("\n")

print(f"\nReport written to: {REPORT_PATH}")
