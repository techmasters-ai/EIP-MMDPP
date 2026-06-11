#!/usr/bin/env python3
"""Literal gate-coverage acceptance check (guarded-ranker spec §3).

For EVERY positive row (``used == 1``) in a bake-off dataset CSV, the safety
gates must have fired: ``unit_gate == 1`` OR ``table_gate == 1``. A positive
that neither gate covers is a chunk the guarded ranker could drop while it
holds a true field value — i.e. a recall hole the gate union cannot protect.

Usage::

    python3 -m scripts.check_gate_coverage <csv-path> [--allow-missing-columns]

Exit codes (distinct on purpose):
  0  — every used==1 row is covered by unit_gate|table_gate
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


def check_gate_coverage(df) -> list[tuple]:
    """Core check (pure; pandas DataFrame in, miss rows out).

    Returns one tuple per uncovered positive:
    ``(doc_filename, pass_name, chunk_index, unit_gate, table_gate)``.
    Empty list == full coverage. Raises :class:`GateColumnsMissing` when the
    gate columns are absent entirely.
    """
    missing = [c for c in GATE_COLS if c not in df.columns]
    if missing:
        raise GateColumnsMissing(missing)
    if "used" not in df.columns:
        raise ValueError("dataset has no 'used' label column — not a bake-off export")

    pos = df[df["used"].astype(int) == 1]
    uncovered = pos[
        ~((pos["unit_gate"].astype(float) == 1.0) | (pos["table_gate"].astype(float) == 1.0))
    ]
    misses: list[tuple] = []
    for _, r in uncovered.iterrows():
        misses.append(tuple(r.get(c) for c in MISS_COLS))
    return misses


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path", help="bake-off dataset CSV (export_bakeoff_dataset output)")
    ap.add_argument(
        "--allow-missing-columns",
        action="store_true",
        help="if the gate columns are absent entirely (old datasets), exit 2 "
        "with a clear message instead of failing",
    )
    args = ap.parse_args(argv)

    import pandas as pd

    df = pd.read_csv(args.csv_path)
    try:
        misses = check_gate_coverage(df)
    except GateColumnsMissing as e:
        print(f"GATE COLUMNS MISSING — coverage UNCHECKABLE: {e}")
        if args.allow_missing_columns:
            print("(--allow-missing-columns: exiting 2 — not a coverage verdict)")
            return 2
        print("(pass --allow-missing-columns to exit 2 for old datasets; failing)")
        return 1

    n_pos = int((df["used"].astype(int) == 1).sum())
    if misses:
        print(
            f"GATE COVERAGE FAIL: {len(misses)}/{n_pos} used==1 rows covered by "
            "NEITHER unit_gate NOR table_gate:"
        )
        print("(doc_filename, pass_name, chunk_index, unit_gate, table_gate)")
        for m in misses:
            print(m)
        return 1

    print(
        f"GATE COVERAGE OK: {n_pos}/{n_pos} used==1 rows covered by "
        f"unit_gate|table_gate ({len(df)} rows total)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
