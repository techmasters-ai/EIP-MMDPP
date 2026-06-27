#!/usr/bin/env python3
"""Literal gate-coverage acceptance check (guarded-ranker spec §3).

For EVERY positive row (``used == 1``) in a bake-off dataset CSV, the safety
gates must cover the chunk: ``unit_gate == 1`` OR ``table_gate == 1``. A
positive that neither gate covers is a chunk the guarded ranker could drop
while it holds a true field value — i.e. a recall hole the gate union cannot
protect.

OFFLINE GATE RECOMPUTE (Task 18)
--------------------------------
The CAPTURED ``unit_gate`` / ``table_gate`` flags are 0 on the E2 fallback path
(small docs that trip ``build_pool_from_multi_channel_state``) for runs whose
capture predates the Task-18 fallback gate-union wiring — even though the gate
predicate WOULD fire on the chunk text.  The gate predicate is a pure function
of (chunk_text, pass unit-signature) and is a strict superset of the
value-grounded label BY CONSTRUCTION, so we can recompute it offline and rescue
those positives without a re-collection.

When a ``signatures`` map (``{pass_name: unit_signature_tuple}``) is supplied,
each used==1 row is ALSO checked against the recomputed gate predicate on its
``chunk_text`` (G1: digit + signature unit token; G2: ``is_table`` + signature
unit token, digit not required).  A positive PASSES coverage if EITHER the
captured flag is 1 OR the recomputed predicate fires.  The summary reports both
(captured vs recomputed) so the fallback-path gap stays visible without
false-failing.  ``signatures=None`` (the default / no ``--bundle``) preserves
the captured-flag-only behaviour exactly.

Usage::

    python3 -m scripts.check_gate_coverage <csv-path> \
        [--bundle air_defense_v3] [--allow-missing-columns]

Exit codes (distinct on purpose):
  0  — every used==1 row is covered (captured flag OR recomputed predicate)
  1  — coverage FAILURE: at least one uncovered positive (misses printed)
  2  — gate columns ABSENT from the CSV (old dataset, capture predates the
       gates) AND ``--allow-missing-columns`` was passed; without the flag a
       missing column is treated as a hard failure (exit 1).
"""
from __future__ import annotations

import argparse
import sys

GATE_COLS: tuple[str, ...] = ("unit_gate", "table_gate")
MISS_COLS: tuple[str, ...] = ("doc_filename", "pass_name", "chunk_index", "unit_gate", "table_gate")


class GateColumnsMissing(ValueError):
    """The CSV has no gate columns at all (old capture) — coverage is UNCHECKABLE,
    which is a different condition from a coverage failure."""

    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        super().__init__(
            "gate columns absent from dataset: "
            + ", ".join(missing)
            + " (old capture predates Tasks 7/10 — re-export after re-collection)"
        )


def _recomputed_gate_fires(
    chunk_text: str,
    signature: "tuple[str, ...]",
    *,
    is_table: bool,
) -> bool:
    """Offline gate predicate (construction-faithful — SAME matcher as the
    runtime gate / value-grounding label).

    Fires when EITHER:
      G1 — digit present AND a signature unit token present, OR
      G2 — the row is a table AND a signature unit token present (digit not
           required; serialized table cells may carry units without co-located
           digits).
    """
    # Imported lazily so the pure captured-flag path (signatures=None) has no
    # dependency on the app package (keeps the core check import-light).
    from app.services.extraction_unit_gate import chunk_passes_unit_gate
    from app.services.field_value_grounding import has_unit_token, nfc

    if not signature:
        return False
    text_nfc = nfc(chunk_text or "")
    if not text_nfc:
        return False
    # G1: digit + unit token.
    if chunk_passes_unit_gate(text_nfc, signature):
        return True
    # G2: table row + unit token (digit-free).
    if is_table and has_unit_token(text_nfc, signature):
        return True
    return False


def check_gate_coverage(df, *, signatures: "dict[str, tuple[str, ...]] | None" = None) -> list[tuple]:
    """Core check (pure; pandas DataFrame in, miss rows out).

    Returns one tuple per uncovered positive:
    ``(doc_filename, pass_name, chunk_index, unit_gate, table_gate)``.
    Empty list == full coverage. Raises :class:`GateColumnsMissing` when the
    gate columns are absent entirely.

    Parameters
    ----------
    signatures:
        Optional ``{pass_name: unit_signature_tuple}`` map.  When supplied, each
        used==1 row not covered by a CAPTURED flag is additionally checked
        against the recomputed gate predicate on its ``chunk_text`` (and
        ``is_table`` column if present).  ``None`` (default) → captured-flag-only
        behaviour, byte-identical to the pre-Task-18 check.
    """
    missing = [c for c in GATE_COLS if c not in df.columns]
    if missing:
        raise GateColumnsMissing(missing)
    if "used" not in df.columns:
        raise ValueError("dataset has no 'used' label column — not a bake-off export")

    pos = df[df["used"].astype(int) == 1]
    captured = (pos["unit_gate"].astype(float) == 1.0) | (pos["table_gate"].astype(float) == 1.0)
    uncovered = pos[~captured]

    misses: list[tuple] = []
    for _, r in uncovered.iterrows():
        # Offline rescue: recompute the gate predicate from chunk_text using the
        # pass's unit signature.  Covered iff the predicate fires.
        if signatures is not None and "chunk_text" in df.columns:
            sig = signatures.get(r.get("pass_name"), ())
            is_table = bool(float(r.get("is_table", 0) or 0)) if "is_table" in df.columns else False
            if _recomputed_gate_fires(r.get("chunk_text", ""), sig, is_table=is_table):
                continue  # rescued by offline recompute — not a miss
        misses.append(tuple(r.get(c) for c in MISS_COLS))
    return misses


def _resolve_signatures(bundle_key: str) -> "dict[str, tuple[str, ...]]":
    """Build ``{pass_name: unit_signature}`` for every field-group (routable)
    pass in the bundle — the SAME loader path the audit / diagnose scripts use
    (``load_bundle_manifest`` → ``build_retrieval_profile`` → ``unit_signature``).
    """
    import importlib

    from app.services.extraction_query_builder import build_retrieval_profile
    from app.services.ontology_bundles import load_bundle_manifest

    manifest = load_bundle_manifest(bundle_key)
    out: dict[str, tuple[str, ...]] = {}
    for pass_def in manifest.passes:
        if pass_def.retrieval is None:
            continue  # identity / relationship passes — not routable
        mod = importlib.import_module(
            f"ontology_bundles.{bundle_key}.{pass_def.module}"
        )
        template_cls = getattr(mod, pass_def.template_class)
        signals = build_retrieval_profile(pass_def, template_cls)
        out[pass_def.name] = tuple(getattr(signals, "unit_signature", ()) or ())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path", help="bake-off dataset CSV (export_bakeoff_dataset output)")
    ap.add_argument(
        "--bundle",
        default=None,
        help="ontology bundle key (e.g. air_defense_v3) — resolves the pass→"
        "unit-signature map for the offline gate recompute that rescues "
        "fallback-path positives whose captured flag is 0. Omit → captured-"
        "flag-only check (pre-Task-18 behaviour).",
    )
    ap.add_argument(
        "--allow-missing-columns",
        action="store_true",
        help="if the gate columns are absent entirely (old datasets), exit 2 "
        "with a clear message instead of failing",
    )
    args = ap.parse_args(argv)

    import pandas as pd

    df = pd.read_csv(args.csv_path)

    signatures = None
    if args.bundle:
        signatures = _resolve_signatures(args.bundle)

    try:
        misses = check_gate_coverage(df, signatures=signatures)
    except GateColumnsMissing as e:
        print(f"GATE COLUMNS MISSING — coverage UNCHECKABLE: {e}")
        if args.allow_missing_columns:
            print("(--allow-missing-columns: exiting 2 — not a coverage verdict)")
            return 2
        print("(pass --allow-missing-columns to exit 2 for old datasets; failing)")
        return 1

    n_pos = int((df["used"].astype(int) == 1).sum())

    # Visibility: how many positives the CAPTURED flag covers vs how many the
    # offline recompute had to rescue (the fallback-path gap).  Reported even on
    # the green path so the gap stays observable without false-failing.
    if "unit_gate" in df.columns and "table_gate" in df.columns:
        pos = df[df["used"].astype(int) == 1]
        n_captured = int(
            ((pos["unit_gate"].astype(float) == 1.0)
             | (pos["table_gate"].astype(float) == 1.0)).sum()
        )
        n_rescued = (n_pos - n_captured) - len(misses)
        print(
            f"coverage breakdown: {n_captured}/{n_pos} covered by CAPTURED flag; "
            f"{max(n_rescued, 0)}/{n_pos} rescued by OFFLINE recompute"
            + ("" if signatures is not None else " (recompute OFF — pass --bundle)")
        )

    if misses:
        print(
            f"GATE COVERAGE FAIL: {len(misses)}/{n_pos} used==1 rows covered by "
            "NEITHER captured unit_gate/table_gate NOR offline recompute:"
        )
        print("(doc_filename, pass_name, chunk_index, unit_gate, table_gate)")
        for m in misses:
            print(m)
        return 1

    print(
        f"GATE COVERAGE OK: {n_pos}/{n_pos} used==1 rows covered by "
        f"unit_gate|table_gate (captured or recomputed) ({len(df)} rows total)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
