#!/usr/bin/env python3
"""A0 captured-component separation analysis (corrected, un-confounded).

Reads the LIVE-captured decomposed router components
(``diagnostics_json['router']['score_components_all']``) from SHADOW full-doc
run(s) plus the lineage USED labels, and tests whether a score THRESHOLD /
dynamic-k / learned re-weighting over the DECOMPOSED features separates the
chunks a fact was actually traced TO (USED, via lineage) from the rest — well
enough that the rule preserves recall while cutting chunks sent to the LLM.

Why this script and not phase1/phase1b
--------------------------------------
phase1 + phase1b RECOMPUTE scores offline. That recompute does NOT carry the
live C8 committed-entity anchors or the projected section signal, so the
decomposed lexical features it would need (field_label / pass_keyword /
anchor_text / anchor_section) come out inert — exactly the signals feature (b)
just wired. This script instead reads the EXACT components the live router
computed and persisted (vertex_id keying fix + C8 anchors + section signal all
in effect), so the measurement is honest. It also requires a SHADOW full-doc
run so the USED labels are not confounded by narrowing (a chunk the LLM never
saw cannot be USED even if relevant).

Modes
-----
single-run (default)::

    python -m scripts.a0_captured_separation --run-id <RUN> [--doc-id <DOC>]

  Extract <RUN>'s (pass, chunk, decomposed-components, used) table; report
  per-feature AUROC, the production final_score baseline AUROC, a learned
  re-weighting (L2 logistic + GBM ceiling, in-sample + stratified k-fold CV),
  and recall-vs-chunks-kept sweeps on both final_score and the learned score.

fit / validate (cross-doc; the user's plan: fit on SA-2 + 1-2 more, validate
on the rest)::

    python -m scripts.a0_captured_separation \
        --fit-runs <RUN_A>,<RUN_B> --validate-runs <RUN_C>,<RUN_D>,...

  Learn the re-weighting on the POOLED fit runs, pick the threshold that holds
  recall >= --recall-floor on the fit pool, then apply it HELD-OUT to each
  validation run and report per-doc recall + chunk-reduction (the real
  generalization test).

Reuse / isolation
-----------------
Reuses ONLY phase1's import-safe pure helpers: ``build_used_table`` (the
validated lineage USED join) and ``auroc`` (exact-tie, matches
sklearn.roc_auc_score). Databases are read DIRECTLY — postgres via SQLAlchemy,
ArcadeDB via HTTP — so the script runs host-side WITHOUT importing the ``app``
package. Connection settings come from env (host defaults baked in):

    A0_DATABASE_URL   (default postgresql+psycopg2://eip:eip_secret@localhost:5437/eip)
    A0_ARCADEDB_URL   (default http://localhost:2480)
    A0_ARCADEDB_DB    (default eip_knowledge_graph)
    A0_ARCADEDB_USER  / A0_ARCADEDB_PASS  (default root / eip_arcadedb_secret)

``--self-test`` exercises the join + AUROC + fit on a synthetic fixture (no DB).
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

# phase1's pure, import-safe helpers (no app import at module load; its app
# imports are lazy, inside I/O wrappers we never call here).
from scripts import phase1_score_used_separation as p1

# ---------------------------------------------------------------------------
# Feature set — the REAL live components feeding final_score, plus cosine.
# Order is the canonical feature-matrix column order.
# ---------------------------------------------------------------------------
FEATURES: tuple[str, ...] = (
    "cosine",
    "rerank_norm",
    "field_label_norm",
    "pass_keyword_norm",
    "anchor_text_norm",
    "anchor_section_norm",
    "section_norm",
    "is_table",
    "pattern_norm",
    "negative_norm",
)
AUROC_FLOOR = 0.75  # a separator must clear this to count as "signal exists"


# ---------------------------------------------------------------------------
# DB config (host-side direct reads; no app package required)
# ---------------------------------------------------------------------------
def _pg_url() -> str:
    return os.environ.get(
        "A0_DATABASE_URL",
        "postgresql+psycopg2://eip:eip_secret@localhost:5437/eip",
    )


def _arcadedb_cfg() -> tuple[str, str, str, str]:
    return (
        os.environ.get("A0_ARCADEDB_URL", "http://localhost:2480").rstrip("/"),
        os.environ.get("A0_ARCADEDB_DB", "eip_knowledge_graph"),
        os.environ.get("A0_ARCADEDB_USER", "root"),
        os.environ.get("A0_ARCADEDB_PASS", "eip_arcadedb_secret"),
    )


# ===========================================================================
# Row model
# ===========================================================================
@dataclass
class Row:
    """One captured (run, pass, chunk) candidate: decomposed features + label."""
    run_id: str
    pass_name: str
    chunk_index: int | None
    candidate_key: str
    used: bool | None
    used_field_level: bool
    used_source: str
    join_source: str           # vertex_id | self_ref | UNMATCHED
    self_ref: str = ""
    page_number: int | None = None
    token_count: int = 0
    final_score: float = 0.0
    # decomposed features
    cosine: float = 0.0
    rerank_norm: float = 0.0
    field_label_norm: float = 0.0
    pass_keyword_norm: float = 0.0
    anchor_text_norm: float = 0.0
    anchor_section_norm: float = 0.0
    section_norm: float = 0.0
    is_table: float = 0.0
    pattern_norm: float = 0.0
    negative_norm: float = 0.0


def _fnum(d: Mapping[str, Any], k: str) -> float:
    try:
        v = d.get(k)
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


# ===========================================================================
# DB readers (direct — no app import)
# ===========================================================================
def fetch_captured_components(run_id: str) -> dict[str, list[dict]]:
    """{pass_name: [component dict, ...]} from diagnostics_json['router']
    ['score_components_all']. Latest attempt wins per pass."""
    from sqlalchemy import create_engine, text

    engine = create_engine(_pg_url())
    out: dict[str, list[dict]] = {}
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT pass_name, attempt, "
                    "  diagnostics_json->'router'->'score_components_all' AS sca "
                    "FROM ingest.pipeline_pass_outputs "
                    "WHERE pipeline_run_id = :r "
                    "ORDER BY pass_name, attempt"
                ),
                {"r": run_id},
            ).fetchall()
        for pass_name, _attempt, sca in rows:
            comps = p1._as_list(sca)
            if comps:  # latest non-empty attempt wins (ORDER BY attempt)
                out[pass_name] = comps
    finally:
        engine.dispose()
    return out


def fetch_pass_outputs(run_id: str) -> dict[str, dict[str, Any]]:
    """{pass_name: {provenance, field_provenance}} — provenance blocks for the
    USED join. Direct reimplementation of p1._fetch_pass_outputs on our engine
    (p1's version resolves the DSN via app.config, which we avoid host-side)."""
    from sqlalchemy import create_engine, text

    engine = create_engine(_pg_url())
    out: dict[str, dict[str, Any]] = {}
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT pass_name, "
                    "  COALESCE(extract_pass_response_json->'provenance','[]'::jsonb) AS prov, "
                    "  COALESCE(extract_pass_response_json->'field_provenance','[]'::jsonb) AS fprov, "
                    "  COALESCE(extract_pass_response_json->'pass_output','{}'::jsonb) AS pout "
                    "FROM ingest.pipeline_pass_outputs "
                    "WHERE pipeline_run_id = :r "
                    "ORDER BY pass_name, attempt"
                ),
                {"r": run_id},
            ).fetchall()
        for pass_name, prov, fprov, pout in rows:
            out[pass_name] = {
                "provenance": p1._as_list(prov),
                "field_provenance": p1._as_list(fprov),
                "pass_output": json.loads(pout) if isinstance(pout, str) else (pout or {}),
            }
    finally:
        engine.dispose()
    return out


def fetch_chunks(run_id: str) -> list[dict[str, Any]]:
    """ExtractionChunk index rows (ArcadeDB HTTP) carrying the join keys +
    USED-join metadata: vertex_id, self_ref, chunk_index, source_refs,
    token_count, page_number."""
    base, db, user, pw = _arcadedb_cfg()
    sql = (
        "SELECT vertex_id, self_ref, chunk_index, source_refs, token_count, "
        "page_number FROM ExtractionChunk WHERE pipeline_run_id = '"
        + run_id.replace("'", "")
        + "' ORDER BY chunk_index ASC"
    )
    body = json.dumps({"command": sql, "language": "sql"}).encode()
    auth = base64.b64encode(f"{user}:{pw}".encode()).decode()
    req = urllib.request.Request(
        f"{base}/api/v1/command/{db}",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.load(resp)
    return list(payload.get("result") or [])


# ===========================================================================
# build_run_table — join captured components → chunk_index → USED  (critical)
# ===========================================================================
def build_extracted_from_target(pass_outputs: dict[str, Any]) -> dict[tuple[str, int], bool]:
    """Clean per-(pass, chunk) 'extracted-from' target from the FIXED field
    lineage: a pass extracted from chunk C iff any of that pass's
    ``field_provenance`` rows attributes a field to C (``chunk_indexes`` /
    ``chunk_index``). This replaces the broken USED labels — with the per-field
    ``__property_provenance`` lineage fix, ``field_provenance.chunk_indexes`` is
    the exact chunk a field's value was extracted FROM, so (pass, chunk) is a
    true positive iff the pass pulled a field value out of that chunk."""
    ef: dict[str, set[int]] = {}
    for pass_name, blocks in pass_outputs.items():
        chunks: set[int] = set()
        for fp in blocks.get("field_provenance") or []:
            if not isinstance(fp, dict):
                continue
            for c in fp.get("chunk_indexes") or []:
                if isinstance(c, int):
                    chunks.add(c)
            ci = fp.get("chunk_index")
            if isinstance(ci, int):
                chunks.add(ci)
        ef[pass_name] = chunks
    return {(p, c): True for p, cs in ef.items() for c in cs}


def _fetch_chunk_text(run_id: str) -> dict[int, str]:
    """chunk_index -> chunk_text (ArcadeDB HTTP) for value-grounding the target."""
    base, db, user, pw = _arcadedb_cfg()
    sql = ("SELECT chunk_index, chunk_text FROM ExtractionChunk WHERE pipeline_run_id = '"
           + run_id.replace("'", "") + "'")
    auth = base64.b64encode(f"{user}:{pw}".encode()).decode()
    req = urllib.request.Request(
        f"{base}/api/v1/command/{db}",
        data=json.dumps({"command": sql, "language": "sql"}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        rows = json.load(resp).get("result") or []
    return {int(r["chunk_index"]): (r.get("chunk_text") or "")
            for r in rows if r.get("chunk_index") is not None}


def build_extracted_from_target_grounded(
    pass_outputs: dict[str, Any], chunk_text: dict[int, str]
) -> dict[tuple[str, int], bool]:
    """VALUE-GROUNDED target: (pass, chunk) is positive iff a field the pass
    extracted has its VALUE literally present in that chunk's text — recovering
    the TRUE extracted-from chunk and discarding LLM over-emission (the model
    emits salient values for prose chunks that don't contain them; see #67).
    Non-text / null / image fields (no groundable value) contribute no positive.
    Uses the shared field_value_grounding matcher so it matches the worker /
    ground-truth tooling exactly: ``value_in_chunk`` is the SINGLE label
    authority; no fallback tiers."""
    import re as _re
    from app.services.field_value_grounding import (
        nfc as _g_nfc, num_variants as _g_nums, units_for as _g_units,
        value_in_chunk as _g_inchunk,
    )

    # (b) image/chart-caption chunks: the picture-description enrichment text.
    # Values "extracted" from these are chart-axis scales / image descriptions
    # (e.g. "Y-axis scaled 0 to 30 km" → max_altitude_km=30), not stated specs —
    # target NOISE. Match the boilerplate PREFIX, not bare "axis", so prose that
    # merely mentions a chart (real spec stated in words) is NOT excluded.
    _img_re = _re.compile(
        r"this image is classified|image consists of|image depicts|"
        r"photographic capture|rendering of|block[_ ]?diagram|schematic shows",
        _re.I,
    )

    def _grounded(value, fname, txt: str) -> bool:
        # NUMERIC + unit-bearing ONLY (strings over-match). Grounding is
        # delegated ENTIRELY to the shared matcher's two tiers — ADJACENT
        # (number next to its unit, prose) and SAME_CHUNK (token-bounded unit
        # + digit-boundary-guarded >=2-digit number elsewhere in the chunk,
        # which already covers FLATTENED TABLES like 'Weight,1=kg ...
        # Weight,3=2283'). This builder adds NO extra tiers: value_in_chunk is
        # the SINGLE label authority (guarded-ranker spec §2/§3). A raw
        # substring co-occurrence fallback that used to live here (no token
        # boundaries, no digit guards) minted junk positives once 1-2 char
        # unit synonyms ("s", "us") existed — removed in Task 4b.
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        units = _g_units(fname) or []
        if not units:
            return False
        return bool(_g_inchunk(_g_nums(value), units, txt))

    norm_text = {ci: _g_nfc(t) for ci, t in chunk_text.items()}
    img_chunks = {ci for ci, t in chunk_text.items() if _img_re.search(t or "")}  # (b)
    out: dict[tuple[str, int], bool] = {}
    for pass_name, blocks in pass_outputs.items():
        # (c) scope to the pass's OWN fields. pass_output is the MERGED entity
        # (carries every pass's fields → cross-pass contamination), but
        # field_provenance is per-pass clean — use its field_names as the filter.
        own_fields = {
            fp.get("field_name")
            for fp in (blocks.get("field_provenance") or [])
            if isinstance(fp, dict) and fp.get("field_name")
        }
        fvals: dict[str, list] = {}
        for v in (blocks.get("pass_output") or {}).values():
            if isinstance(v, list):
                for rec in v:
                    if isinstance(rec, dict):
                        for f, val in rec.items():
                            if f not in own_fields:
                                continue  # (c) drop merged cross-pass fields
                            if val is not None and val != "" and not isinstance(val, (list, dict)):
                                fvals.setdefault(f, []).append(val)
        # TRUE source = ANY non-image chunk whose text contains the extracted
        # value (search the whole doc, not the LLM's emission chunks).
        for fname, vals in fvals.items():
            for v in vals:
                for ci, txt in norm_text.items():
                    if ci in img_chunks:          # (b) skip image/chart captions
                        continue
                    if _grounded(v, fname, txt):
                        out[(pass_name, ci)] = True
    return out


def build_run_table(run_id: str, target_mode: str = "used") -> tuple[list[Row], dict[str, Any]]:
    """Per (pass, candidate) Row with decomposed features + a label.
    candidate_key (vertex_id preferred, self_ref fallback) is mapped to
    chunk_index via the ExtractionChunk index.

    ``target_mode``:
      * ``used``    — phase1's validated build_used_table (the legacy lineage
        USED labels; confounded by entity-mention pollution).
      * ``lineage`` — the FIXED per-field extracted-from target
        (build_extracted_from_target): a chunk is positive for a pass iff that
        pass extracted a field VALUE from it. This is the honest calibration
        target the __property_provenance fix unlocked."""
    comps = fetch_captured_components(run_id)
    stats: dict[str, Any] = {
        "run_id": run_id,
        "passes_with_components": sorted(comps),
        "n_candidates": sum(len(v) for v in comps.values()),
        "matched": 0,
        "unmatched": 0,
        "by_vertex_id": 0,
        "by_self_ref": 0,
        "used_label_missing": 0,
    }
    if not comps:
        stats["status"] = "no captured components yet (run incomplete or shadow off)"
        return [], stats

    pass_outputs = fetch_pass_outputs(run_id)
    chunks = fetch_chunks(run_id)
    vid2ci: dict[str, int] = {}
    sref2ci: dict[str, int] = {}
    for c in chunks:
        try:
            ci = int(c["chunk_index"])
        except (KeyError, TypeError, ValueError):
            continue
        vid = c.get("vertex_id")
        if vid is not None:
            vid2ci[str(vid)] = ci
        sref = c.get("self_ref")
        if sref:
            sref2ci[str(sref)] = ci

    # USED labels via the validated phase1 join (same doc-global index per pass).
    chunks_by_pass = {p: chunks for p in comps}
    used_rows = p1.build_used_table(pass_outputs, chunks_by_pass)
    used_by: dict[tuple[str, int], Any] = {(r.pass_name, r.chunk_index): r for r in used_rows}
    # FIXED-lineage extracted-from target. "lineage" = emission chunks (LLM
    # may over-emit); "lineage_grounded" = only chunks whose text contains the
    # field value (recovers the true source, drops LLM carry-over — see #67).
    if target_mode == "lineage":
        ef_target = build_extracted_from_target(pass_outputs)
    elif target_mode == "lineage_grounded":
        ef_target = build_extracted_from_target_grounded(pass_outputs, _fetch_chunk_text(run_id))
    else:
        ef_target = {}
    stats["target_mode"] = target_mode

    rows: list[Row] = []
    for pass_name, clist in comps.items():
        for cd in clist:
            ck = str(cd.get("candidate_key", ""))
            ci = vid2ci.get(ck)
            jsrc = "vertex_id"
            if ci is None:
                ci = sref2ci.get(ck)
                jsrc = "self_ref"
            if ci is None:
                jsrc = "UNMATCHED"
                stats["unmatched"] += 1
                ur = None
            else:
                stats["matched"] += 1
                stats["by_vertex_id" if jsrc == "vertex_id" else "by_self_ref"] += 1
                ur = used_by.get((pass_name, ci))
            if ci is not None and ur is None:
                stats["used_label_missing"] += 1
            if target_mode in ("lineage", "lineage_grounded"):
                # Every matched candidate gets a definite 0/1 from the fixed
                # lineage (no missing labels); unmatched stays None.
                used_val = ef_target.get((pass_name, ci), False) if ci is not None else None
                usedfl_val = bool(used_val) if used_val is not None else False
            else:
                used_val = (bool(ur.used) if ur is not None else None)
                usedfl_val = (bool(ur.used_field_level) if ur is not None else False)
            rows.append(Row(
                run_id=run_id,
                pass_name=pass_name,
                chunk_index=ci,
                candidate_key=ck,
                used=used_val,
                used_field_level=usedfl_val,
                used_source=(ur.used_source if ur is not None else ""),
                join_source=jsrc,
                self_ref=(ur.self_ref if ur is not None else ""),
                page_number=(ur.page_number if ur is not None else None),
                token_count=(ur.token_count if ur is not None else 0),
                final_score=_fnum(cd, "final_score"),
                cosine=_fnum(cd, "cosine"),
                rerank_norm=_fnum(cd, "rerank_norm"),
                field_label_norm=_fnum(cd, "field_label_norm"),
                pass_keyword_norm=_fnum(cd, "pass_keyword_norm"),
                anchor_text_norm=_fnum(cd, "anchor_text_norm"),
                anchor_section_norm=_fnum(cd, "anchor_section_norm"),
                section_norm=_fnum(cd, "section_norm"),
                is_table=_fnum(cd, "is_table"),
                pattern_norm=_fnum(cd, "pattern_norm"),
                negative_norm=_fnum(cd, "negative_norm"),
            ))
    return rows, stats


# ===========================================================================
# Analysis primitives
# ===========================================================================
def _labeled(rows: Sequence[Row]) -> list[Row]:
    return [r for r in rows if r.used is not None and r.chunk_index is not None]


def per_feature_auroc(rows: Sequence[Row]) -> dict[str, Any]:
    """AUROC(used, feature) for each feature + the production final_score,
    pooled and per-pass (reuses phase1's exact-tie auroc)."""
    used = [bool(r.used) for r in rows]
    cols = list(FEATURES) + ["final_score"]
    pooled = {f: p1.auroc([getattr(r, f) for r in rows], used) for f in cols}
    by_pass: dict[str, list[Row]] = {}
    for r in rows:
        by_pass.setdefault(r.pass_name, []).append(r)
    per_pass: dict[str, dict[str, float]] = {}
    support: dict[str, dict[str, int]] = {}
    for pn, prs in by_pass.items():
        u = [bool(r.used) for r in prs]
        per_pass[pn] = {f: p1.auroc([getattr(r, f) for r in prs], u) for f in cols}
        support[pn] = {"n": len(prs), "n_used": sum(1 for x in u if x)}
    return {
        "pooled": pooled,
        "per_pass": per_pass,
        "support": {
            "pooled": {"n": len(rows), "n_used": sum(1 for x in used if x)},
            "per_pass": support,
        },
    }


def _cv_auroc(estimator, X, y, *, n_splits: int, seed: int) -> float:
    """Stratified k-fold CV AUROC (honest within-doc estimate). NaN-safe; 0.5
    when CV cannot be formed."""
    import numpy as np
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos < 2 or n_neg < 2:
        return 0.5
    k = min(n_splits, n_pos, n_neg)
    if k < 2:
        return 0.5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    aucs: list[float] = []
    for tr, te in skf.split(X, y):
        if len(set(y[te].tolist())) < 2:
            continue
        est = clone(estimator)
        est.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], est.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else 0.5


def _build_logistic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def _build_gbm(seed: int):
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(random_state=seed, max_depth=2, n_estimators=100)


def _matrix(rows: Sequence[Row], features: Sequence[str]):
    import numpy as np
    X = np.asarray([[getattr(r, f) for f in features] for r in rows], dtype=np.float64)
    y = np.asarray([1 if r.used else 0 for r in rows], dtype=np.int64)
    return X, y


def fit_separator(
    rows: Sequence[Row], *, features: Sequence[str] = FEATURES, n_splits: int = 5, seed: int = 0
) -> dict[str, Any]:
    """L2 logistic (the re-weighting; learned coefficients reported) + GBM
    (nonlinear ceiling), each in-sample and stratified-k-fold CV. CV is the
    honest 'does the signal exist' number; in-sample overfits a small/wide
    sample and is only a ceiling."""
    from sklearn.metrics import roc_auc_score

    out: dict[str, Any] = {
        "n": len(rows),
        "n_used": sum(1 for r in rows if r.used),
        "features": list(features),
    }
    X, y = _matrix(rows, features)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos < 2 or n_neg < 2:
        out["status"] = f"degenerate (n_used={n_pos}, n_unused={n_neg}; need >=2 each)"
        return out

    logistic = _build_logistic()
    logistic.fit(X, y)
    lr = logistic.named_steps["lr"]
    out["logistic"] = {
        "in_sample_auroc": float(roc_auc_score(y, logistic.predict_proba(X)[:, 1])),
        "cv_auroc": _cv_auroc(_build_logistic(), X, y, n_splits=n_splits, seed=seed),
        "weights": {f: float(w) for f, w in zip(features, lr.coef_[0])},
        "intercept": float(lr.intercept_[0]),
    }
    gbm = _build_gbm(seed)
    gbm.fit(X, y)
    out["gbm"] = {
        "in_sample_auroc": float(roc_auc_score(y, gbm.predict_proba(X)[:, 1])),
        "cv_auroc": _cv_auroc(_build_gbm(seed), X, y, n_splits=n_splits, seed=seed),
        "feature_importance": {f: float(i) for f, i in zip(features, gbm.feature_importances_)},
    }
    out["_logistic_scores"] = logistic.predict_proba(X)[:, 1].tolist()
    return out


def recall_threshold_sweep(
    scores: Sequence[float], used: Sequence[bool], *, n: int = 21
) -> list[dict[str, float]]:
    """Recall vs fraction-of-chunks-kept as the score threshold sweeps from
    max→min. The core 'threshold that maintains recall while cutting chunks'
    curve. recall=1.0 when there are no USED chunks (vacuous)."""
    import numpy as np

    if not scores:
        return []
    total = len(scores)
    total_used = sum(1 for u in used if u)
    pairs = list(zip([float(s) for s in scores], [bool(u) for u in used]))
    lo, hi = min(s for s, _ in pairs), max(s for s, _ in pairs)
    out: list[dict[str, float]] = []
    for t in np.linspace(hi, lo, n):
        kept = [(s, u) for s, u in pairs if s >= t]
        ku = sum(1 for _, u in kept if u)
        out.append({
            "threshold": float(t),
            "kept": len(kept),
            "kept_frac": len(kept) / total,
            "recall": (ku / total_used) if total_used else 1.0,
        })
    return out


def min_kept_at_recall(curve: Sequence[Mapping[str, float]], floor: float) -> dict[str, float] | None:
    """The threshold point with the FEWEST kept chunks that still holds
    recall >= floor — i.e. the most aggressive recall-preserving cut."""
    ok = [p for p in curve if p["recall"] >= floor]
    return min(ok, key=lambda p: p["kept_frac"]) if ok else None


# ===========================================================================
# Single-run report
# ===========================================================================
def analyze_single_run(rows: Sequence[Row], *, recall_floor: float, seed: int) -> dict[str, Any]:
    lab = _labeled(rows)
    rep: dict[str, Any] = {
        "n_rows": len(rows),
        "n_labeled": len(lab),
        "n_used": sum(1 for r in lab if r.used),
    }
    if len({bool(r.used) for r in lab}) < 2:
        rep["status"] = "single-class (no separation question) — need both USED and UNUSED"
        return rep
    rep["per_feature_auroc"] = per_feature_auroc(lab)
    rep["fit"] = fit_separator(lab, seed=seed)
    # recall-vs-kept on the production baseline and the learned score
    base_curve = recall_threshold_sweep([r.final_score for r in lab], [bool(r.used) for r in lab])
    rep["final_score_recall_curve"] = base_curve
    rep["final_score_min_kept_at_floor"] = min_kept_at_recall(base_curve, recall_floor)
    learned = rep["fit"].get("_logistic_scores")
    if learned:
        learned_curve = recall_threshold_sweep(learned, [bool(r.used) for r in lab])
        rep["learned_recall_curve"] = learned_curve
        rep["learned_min_kept_at_floor"] = min_kept_at_recall(learned_curve, recall_floor)
    return rep


# ===========================================================================
# Cross-doc fit / validate (fit on few, validate held-out on the rest)
# ===========================================================================
def fit_validate(
    fit_rows: Sequence[Row],
    validate_rows_by_run: Mapping[str, Sequence[Row]],
    *,
    features: Sequence[str] = FEATURES,
    recall_floor: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Learn the re-weighting on the POOLED fit rows, pick the recall-preserving
    threshold on the fit pool, then apply it HELD-OUT to each validation run."""
    import numpy as np

    fit_lab = _labeled(fit_rows)
    rep: dict[str, Any] = {
        "fit_n": len(fit_lab),
        "fit_n_used": sum(1 for r in fit_lab if r.used),
        "features": list(features),
        "recall_floor": recall_floor,
    }
    if len({bool(r.used) for r in fit_lab}) < 2:
        rep["status"] = "fit pool single-class — cannot learn"
        return rep

    model = _build_logistic()
    Xf, yf = _matrix(fit_lab, features)
    model.fit(Xf, yf)
    rep["fit_cv_auroc"] = _cv_auroc(_build_logistic(), Xf, yf, n_splits=5, seed=seed)
    lr = model.named_steps["lr"]
    rep["learned_weights"] = {f: float(w) for f, w in zip(features, lr.coef_[0])}

    fit_scores = model.predict_proba(Xf)[:, 1].tolist()
    fit_curve = recall_threshold_sweep(fit_scores, [bool(r.used) for r in fit_lab], n=101)
    pick = min_kept_at_recall(fit_curve, recall_floor)
    rep["fit_threshold_point"] = pick
    if pick is None:
        rep["status"] = "no fit threshold holds the recall floor"
        return rep
    thr = pick["threshold"]

    per_run: dict[str, Any] = {}
    for run_id, vrows in validate_rows_by_run.items():
        vlab = _labeled(vrows)
        if not vlab:
            per_run[run_id] = {"status": "no labeled rows"}
            continue
        Xv, yv = _matrix(vlab, features)
        sv = model.predict_proba(Xv)[:, 1]
        kept = int((sv >= thr).sum())
        total = len(vlab)
        total_used = int(yv.sum())
        kept_used = int(((sv >= thr) & (yv == 1)).sum())
        per_run[run_id] = {
            "n": total,
            "n_used": total_used,
            "kept": kept,
            "kept_frac": kept / total if total else 0.0,
            "held_out_recall": (kept_used / total_used) if total_used else 1.0,
            "chunk_reduction": 1.0 - (kept / total) if total else 0.0,
        }
    rep["validation"] = per_run
    if per_run:
        recs = [v["held_out_recall"] for v in per_run.values() if "held_out_recall" in v]
        keeps = [v["kept_frac"] for v in per_run.values() if "kept_frac" in v]
        rep["validation_summary"] = {
            "mean_held_out_recall": float(np.mean(recs)) if recs else None,
            "min_held_out_recall": float(np.min(recs)) if recs else None,
            "mean_kept_frac": float(np.mean(keeps)) if keeps else None,
        }
    return rep


# ===========================================================================
# Multi-model bake-off — the 5 sklearn classifiers, scored two ways:
#   pooled stratified k-fold CV  (within-pool separability)
#   leave-one-document-out       (GroupKFold by run_id; honest cross-doc number)
# ===========================================================================
def _build_models(seed: int) -> dict[str, Any]:
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        # standardized — coefficients/activations are scale-sensitive
        "LogisticRegression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]),
        # trees are scale-invariant — no standardizer
        "RandomForestClassifier": RandomForestClassifier(n_estimators=300, random_state=seed),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=seed),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(random_state=seed),
        "MLPClassifier": Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1500, random_state=seed)),
        ]),
    }


def _lodo_auroc(estimator, X, y, groups, *, seed: int) -> float | None:
    """Leave-one-document-out AUROC (GroupKFold by run_id). Each fold trains on
    all-but-one doc and scores the held-out doc — the honest 'does it generalize
    to an unseen document' estimate. None if <2 usable docs."""
    import numpy as np
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    uniq = sorted(set(groups))
    if len(uniq) < 2:
        return None
    gkf = GroupKFold(n_splits=len(uniq))
    aucs: list[float] = []
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
            continue
        est = clone(estimator)
        est.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], est.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else None


def model_bakeoff(
    rows: Sequence[Row], *, features: Sequence[str] = FEATURES, seed: int = 0, n_splits: int = 5
) -> dict[str, Any]:
    """Score all 5 classifiers by pooled stratified-CV AUROC + leave-one-doc-out
    AUROC, plus a recall-vs-kept curve on the best leave-one-doc-out model."""
    lab = _labeled(rows)
    X, y = _matrix(lab, features)
    groups = [r.run_id for r in lab]
    out: dict[str, Any] = {
        "n": len(lab),
        "n_used": int(y.sum()),
        "n_docs": len(set(groups)),
        "docs": sorted(set(groups)),
        "features": list(features),
        "models": {},
    }
    if len({int(v) for v in y}) < 2:
        out["status"] = "single-class — cannot fit"
        return out
    for name, est in _build_models(seed).items():
        out["models"][name] = {
            "pooled_cv_auroc": _cv_auroc(est, X, y, n_splits=n_splits, seed=seed),
            "leave_one_doc_out_auroc": _lodo_auroc(est, X, y, groups, seed=seed),
        }
    ranked = sorted(
        out["models"].items(),
        key=lambda kv: (kv[1]["leave_one_doc_out_auroc"] or -1, kv[1]["pooled_cv_auroc"] or -1),
        reverse=True,
    )
    out["best_by_lodo"] = ranked[0][0] if ranked else None
    # recall-vs-kept on the best model's LEAVE-ONE-DOC-OUT out-of-fold scores —
    # the honest cross-document cut curve (same methodology as the frontier in
    # plot_recall_vs_savings). The prior version fit on all data and scored the
    # same rows in-sample, which overstated savings (the model had seen every
    # test row): e.g. it reported "keep 2% @ recall 1.0" where the honest LODO
    # number is ~24%.
    if out["best_by_lodo"]:
        from sklearn.model_selection import GroupKFold, cross_val_predict
        best = _build_models(seed)[out["best_by_lodo"]]
        n_groups = len(set(groups))
        if n_groups >= 2:
            scores = cross_val_predict(
                best, X, y, groups=groups,
                cv=GroupKFold(n_splits=n_groups), method="predict_proba",
            )[:, 1].tolist()
        else:  # single doc — no cross-doc split possible; fall back to in-sample
            best.fit(X, y)
            scores = best.predict_proba(X)[:, 1].tolist()
        out["best_recall_curve"] = recall_threshold_sweep(scores, [bool(v) for v in y], n=41)
    return out


def _md_bakeoff(rep: Mapping[str, Any], recall_floor: float) -> str:
    L = [f"# A0 model bake-off — {rep.get('n')} chunks, {rep.get('n_used')} USED, "
         f"{rep.get('n_docs')} docs"]
    if rep.get("status"):
        L.append(f"\n**status:** {rep['status']}")
        return "\n".join(L)
    L.append("")
    L.append("| model | pooled CV AUROC | leave-one-doc-out AUROC |")
    L.append("|---|---|---|")
    for name, m in rep["models"].items():
        p = m["pooled_cv_auroc"]
        lo = m["leave_one_doc_out_auroc"]
        star = " ⬅" if name == rep.get("best_by_lodo") else ""
        L.append(f"| {name} | {p:.3f} | {('%.3f' % lo) if lo is not None else '—'}{star} |")
    L.append("")
    L.append(f"**best (leave-one-doc-out): {rep.get('best_by_lodo')}** — LODO is the honest "
             "cross-document number (each doc held out once); pooled CV overstates because "
             "same-doc chunks can land in both train and test folds.")
    pt = min_kept_at_recall(rep.get("best_recall_curve", []), recall_floor)
    if pt:
        L.append(f"- best model recall≥{recall_floor:.2f} (leave-one-doc-out): "
                 f"keep {pt['kept_frac']*100:.0f}% of chunks "
                 f"(recall {pt['recall']:.2f}, threshold {pt['threshold']:.3f})")
    return "\n".join(L)


# ===========================================================================
# Output
# ===========================================================================
def _write_csv(path: str, rows: Sequence[Row]) -> None:
    cols = list(Row.__dataclass_fields__.keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _fmt_auroc_table(pfa: Mapping[str, Any]) -> list[str]:
    pooled = pfa.get("pooled", {})
    lines = ["| feature | pooled AUROC |", "|---|---|"]
    for f in list(FEATURES) + ["final_score"]:
        v = pooled.get(f)
        flag = " ⬅" if (v is not None and v >= AUROC_FLOOR) else ""
        lines.append(f"| {f} | {v:.3f}{flag} |" if v is not None else f"| {f} | — |")
    return lines


def _md_single(run_id: str, stats: Mapping[str, Any], rep: Mapping[str, Any], recall_floor: float) -> str:
    L: list[str] = []
    L.append(f"# A0 captured-component separation — run {run_id[:8]}")
    L.append("")
    L.append(f"- candidates: {stats.get('n_candidates')}  matched: {stats.get('matched')} "
             f"(vid {stats.get('by_vertex_id')} / self_ref {stats.get('by_self_ref')})  "
             f"UNMATCHED: {stats.get('unmatched')}  used-label-missing: {stats.get('used_label_missing')}")
    L.append(f"- passes with captured components: {', '.join(stats.get('passes_with_components', []))}")
    if rep.get("status"):
        L.append(f"\n**status:** {rep['status']}")
        return "\n".join(L)
    L.append(f"- labeled rows: {rep['n_labeled']}  USED: {rep['n_used']}")
    L.append("")
    L.append("## Per-feature separation (pooled AUROC; ⬅ clears the 0.75 floor)")
    L += _fmt_auroc_table(rep["per_feature_auroc"])
    fit = rep.get("fit", {})
    if "logistic" in fit:
        lo, gb = fit["logistic"], fit["gbm"]
        L.append("")
        L.append("## Learned re-weighting (does ANY weighting separate?)")
        L.append(f"- logistic  CV AUROC **{lo['cv_auroc']:.3f}**  (in-sample {lo['in_sample_auroc']:.3f})")
        L.append(f"- GBM       CV AUROC **{gb['cv_auroc']:.3f}**  (in-sample {gb['in_sample_auroc']:.3f})  ← nonlinear ceiling")
        verdict = "GO — a learned threshold/dynamic-k is viable" if max(lo["cv_auroc"], gb["cv_auroc"]) >= AUROC_FLOOR \
            else "NO-GO — signal not separable within this doc; fall back to tuned per-pass top_k"
        L.append(f"- **verdict (CV): {verdict}**")
        ws = sorted(lo["weights"].items(), key=lambda kv: -abs(kv[1]))
        L.append("- learned weights (|coef| desc): " + ", ".join(f"{k}={v:+.2f}" for k, v in ws))
    for label, ckey, mkey in (
        ("final_score (production baseline)", "final_score_recall_curve", "final_score_min_kept_at_floor"),
        ("learned logistic score", "learned_recall_curve", "learned_min_kept_at_floor"),
    ):
        pt = rep.get(mkey)
        if pt:
            L.append(f"- recall≥{recall_floor:.2f} on {label}: keep {pt['kept_frac']*100:.0f}% of chunks "
                     f"(recall {pt['recall']:.2f}, threshold {pt['threshold']:.3f})")
        elif ckey in rep:
            L.append(f"- recall≥{recall_floor:.2f} on {label}: NO threshold holds the floor")
    return "\n".join(L)


def write_single_run(out_dir: str, run_id: str, rows: Sequence[Row],
                     stats: Mapping[str, Any], rep: Mapping[str, Any], recall_floor: float) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    r8 = run_id[:8]
    paths = {
        "csv": os.path.join(out_dir, f"a0_components_{r8}.csv"),
        "json": os.path.join(out_dir, f"a0_separation_{r8}.json"),
        "md": os.path.join(out_dir, f"a0_separation_{r8}.md"),
    }
    _write_csv(paths["csv"], rows)
    clean = {k: v for k, v in rep.items()}
    if "fit" in clean and isinstance(clean["fit"], dict):
        clean["fit"] = {k: v for k, v in clean["fit"].items() if not k.startswith("_")}
    with open(paths["json"], "w") as fh:
        json.dump({"run_id": run_id, "join_stats": stats, "report": clean}, fh, indent=2, default=str)
    md = _md_single(run_id, stats, rep, recall_floor)
    with open(paths["md"], "w") as fh:
        fh.write(md + "\n")
    return paths


# ===========================================================================
# Self-test (no DB) — exercises the join, AUROC and fit on a synthetic fixture
# ===========================================================================
def self_test() -> int:
    # Build a clearly-separable synthetic table: USED chunks have high anchor +
    # rerank, UNUSED have low. The learned fit should clear the AUROC floor.
    rows: list[Row] = []
    for i in range(40):
        used = i < 16
        rows.append(Row(
            run_id="test", pass_name="p1", chunk_index=i, candidate_key=f"#1:{i}",
            used=used, used_field_level=used, used_source="direct", join_source="vertex_id",
            final_score=(0.8 if used else 0.2),
            cosine=(0.7 if used else 0.3), rerank_norm=(0.9 if used else 0.1),
            anchor_text_norm=(1.0 if used else 0.0), field_label_norm=(0.5 if used else 0.0),
        ))
    lab = _labeled(rows)
    assert len(lab) == 40, lab
    pfa = per_feature_auroc(lab)
    assert pfa["pooled"]["rerank_norm"] > 0.9, pfa["pooled"]
    assert pfa["pooled"]["anchor_text_norm"] > 0.9, pfa["pooled"]
    fit = fit_separator(lab, seed=0)
    assert fit["logistic"]["cv_auroc"] >= AUROC_FLOOR, fit["logistic"]
    curve = recall_threshold_sweep([r.final_score for r in lab], [bool(r.used) for r in lab])
    pt = min_kept_at_recall(curve, 1.0)
    assert pt and pt["kept_frac"] < 0.6, pt  # separable → recall held while cutting >40%
    # inseparable fixture → CV ~ chance
    flat = [Row(run_id="t", pass_name="p", chunk_index=i, candidate_key=str(i),
                used=(i % 2 == 0), used_field_level=False, used_source="direct",
                join_source="vertex_id", rerank_norm=0.5, cosine=0.5) for i in range(40)]
    fit2 = fit_separator(_labeled(flat), seed=0)
    assert fit2["logistic"]["cv_auroc"] < AUROC_FLOOR, fit2["logistic"]
    print("self-test OK: separable→GO, inseparable→NO-GO, join/AUROC/fit/recall-sweep all sane")
    return 0


# ===========================================================================
# main
# ===========================================================================
def _split(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="", help="single-run mode: analyze this run")
    ap.add_argument("--doc-id", default="", help="optional, for reporting only")
    ap.add_argument("--fit-runs", default="", help="comma-sep run ids to LEARN on")
    ap.add_argument("--validate-runs", default="", help="comma-sep run ids to validate HELD-OUT")
    ap.add_argument("--recall-floor", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="reports/collection")
    ap.add_argument("--target", choices=("used", "lineage", "lineage_grounded"), default="used",
                    help="label source: 'used' (legacy USED labels), 'lineage' "
                         "(per-field emission chunks), or 'lineage_grounded' "
                         "(only chunks whose text contains the value — drops LLM carry-over)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    fit_runs = _split(args.fit_runs)
    val_runs = _split(args.validate_runs)

    if fit_runs:
        # cross-doc fit/validate mode
        fit_rows: list[Row] = []
        all_rows: list[Row] = []
        for rid in fit_runs:
            rr, st = build_run_table(rid, target_mode=args.target)
            print(f"[fit] {rid[:8]}: {st.get('matched')} matched, "
                  f"{sum(1 for r in rr if r.used)} used", file=sys.stderr)
            fit_rows += rr
            all_rows += rr
        val_by_run: dict[str, list[Row]] = {}
        for rid in val_runs:
            rr, st = build_run_table(rid, target_mode=args.target)
            print(f"[val] {rid[:8]}: {st.get('matched')} matched, "
                  f"{sum(1 for r in rr if r.used)} used", file=sys.stderr)
            val_by_run[rid] = rr
            all_rows += rr
        rep = fit_validate(fit_rows, val_by_run, recall_floor=args.recall_floor, seed=args.seed)
        os.makedirs(args.out_dir, exist_ok=True)
        tag = fit_runs[0][:8]
        with open(os.path.join(args.out_dir, f"a0_fitvalidate_{tag}.json"), "w") as fh:
            json.dump(rep, fh, indent=2, default=str)
        _write_csv(os.path.join(args.out_dir, f"a0_fitvalidate_rows_{tag}.csv"), all_rows)
        print(json.dumps(rep.get("validation_summary", rep.get("status", {})), indent=2))
        # 5-model bake-off on the pooled fit rows (pooled CV + leave-one-doc-out)
        bake = model_bakeoff(fit_rows, seed=args.seed)
        with open(os.path.join(args.out_dir, f"a0_bakeoff_{tag}.json"), "w") as fh:
            json.dump(bake, fh, indent=2, default=str)
        bake_md = _md_bakeoff(bake, args.recall_floor)
        with open(os.path.join(args.out_dir, f"a0_bakeoff_{tag}.md"), "w") as fh:
            fh.write(bake_md + "\n")
        print("\n" + bake_md)
        return 0

    if not args.run_id and not fit_runs:
        ap.error("provide --run-id (single-run) or --fit-runs/--validate-runs (cross-doc)")
    rows, stats = build_run_table(args.run_id, target_mode=args.target)
    rep = analyze_single_run(rows, recall_floor=args.recall_floor, seed=args.seed) if rows else {
        "status": stats.get("status", "no rows")}
    paths = write_single_run(args.out_dir, args.run_id, rows, stats, rep, args.recall_floor)
    print(_md_single(args.run_id, stats, rep, args.recall_floor))
    print("\nwrote:", json.dumps(paths, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
