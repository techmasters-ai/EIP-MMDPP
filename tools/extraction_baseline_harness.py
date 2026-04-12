#!/usr/bin/env python3
"""Baseline harness for PR 2 comparison. Spec §8.3.

Collects per-entity-type and per-edge-type extraction stats from the
configured pipeline, and compares two collection runs (legacy vs
bundle_passes) to determine if the new path meets baseline criteria.

Usage:
    # Collect legacy baseline (with GRAPH_EXTRACTION_ENGINE=legacy in env)
    python tools/extraction_baseline_harness.py --corpus corpus/ --collect legacy > legacy.json

    # Collect new baseline (with GRAPH_EXTRACTION_ENGINE=bundle_passes)
    python tools/extraction_baseline_harness.py --corpus corpus/ --collect bundle_passes > bundle.json

    # Compare
    python tools/extraction_baseline_harness.py --compare \
        --legacy-results legacy.json --bundle-results bundle.json

Baseline criteria (spec §8.3):
- Per-entity-type extraction count: bundle within 10% of legacy (or better)
- Per-edge-type extraction count: bundle within 10% of legacy (or better)
- Overall rejection ratio: bundle within 5 percentage points of legacy
- No pass yield regression: no pass goes from HIT in legacy to EMPTY/DEGRADED in bundle

Exit code 0 = baseline met; 1 = baseline NOT met.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def collect_from_corpus(corpus_dir: Path, label: str) -> dict:
    """Ingest each document in corpus_dir and collect extraction stats.

    This is a placeholder implementation. The real collection would:
    1. Iterate PDF/doc files in corpus_dir
    2. Upload each via POST /documents/upload
    3. Poll for pipeline completion
    4. Read PipelineRun.metrics from the DB
    5. Aggregate into the return dict

    For the initial harness, return a structure that the comparison
    can consume. Actual collection requires the full compose stack
    running — the user runs this manually during soak.
    """
    documents = []
    for path in sorted(corpus_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in (".pdf", ".docx", ".txt"):
            documents.append(str(path.name))

    # Placeholder: actual collection would populate these from PipelineRun.metrics
    return {
        "label": label,
        "document_count": len(documents),
        "documents": documents,
        "aggregate": {
            "entity_counts_by_type": {},
            "edge_counts_by_type": {},
            "yield_distribution": {},  # {pass_name: {HIT: n, EMPTY: n, ...}}
            "overall_rejection_ratio": 0.0,
        },
        "per_document": [],  # list of per-doc metrics dicts
    }


def compare(legacy: dict, bundle: dict) -> dict:
    """Compare two collection results. Returns a diff dict with baseline_met."""
    entity_diffs = {}
    all_entity_types = set(
        list(legacy["aggregate"]["entity_counts_by_type"].keys()) +
        list(bundle["aggregate"]["entity_counts_by_type"].keys())
    )
    entity_baseline_ok = True
    for etype in sorted(all_entity_types):
        legacy_count = legacy["aggregate"]["entity_counts_by_type"].get(etype, 0)
        bundle_count = bundle["aggregate"]["entity_counts_by_type"].get(etype, 0)
        delta = bundle_count - legacy_count
        pct = (delta / legacy_count * 100) if legacy_count > 0 else (0.0 if bundle_count == 0 else float("inf"))
        entity_diffs[etype] = {
            "legacy": legacy_count,
            "bundle": bundle_count,
            "delta": delta,
            "pct_change": round(pct, 1),
        }
        # Baseline: bundle within 10% of legacy (or better = positive delta)
        if legacy_count > 0 and pct < -10.0:
            entity_baseline_ok = False

    edge_diffs = {}
    all_edge_types = set(
        list(legacy["aggregate"]["edge_counts_by_type"].keys()) +
        list(bundle["aggregate"]["edge_counts_by_type"].keys())
    )
    edge_baseline_ok = True
    for etype in sorted(all_edge_types):
        legacy_count = legacy["aggregate"]["edge_counts_by_type"].get(etype, 0)
        bundle_count = bundle["aggregate"]["edge_counts_by_type"].get(etype, 0)
        delta = bundle_count - legacy_count
        pct = (delta / legacy_count * 100) if legacy_count > 0 else (0.0 if bundle_count == 0 else float("inf"))
        edge_diffs[etype] = {
            "legacy": legacy_count,
            "bundle": bundle_count,
            "delta": delta,
            "pct_change": round(pct, 1),
        }
        if legacy_count > 0 and pct < -10.0:
            edge_baseline_ok = False

    # Rejection ratio comparison (within 5 percentage points)
    legacy_ratio = legacy["aggregate"].get("overall_rejection_ratio", 0.0)
    bundle_ratio = bundle["aggregate"].get("overall_rejection_ratio", 0.0)
    ratio_delta = bundle_ratio - legacy_ratio
    ratio_ok = abs(ratio_delta) <= 0.05  # 5 percentage points

    # Yield regression: no pass goes from HIT → EMPTY/DEGRADED
    yield_regressions = []
    legacy_yields = legacy["aggregate"].get("yield_distribution", {})
    bundle_yields = bundle["aggregate"].get("yield_distribution", {})
    for pass_name in set(list(legacy_yields.keys()) + list(bundle_yields.keys())):
        legacy_dominant = _dominant_yield(legacy_yields.get(pass_name, {}))
        bundle_dominant = _dominant_yield(bundle_yields.get(pass_name, {}))
        if legacy_dominant == "HIT" and bundle_dominant in ("EMPTY", "DEGRADED"):
            yield_regressions.append({
                "pass_name": pass_name,
                "legacy_yield": legacy_dominant,
                "bundle_yield": bundle_dominant,
            })
    yield_ok = len(yield_regressions) == 0

    baseline_met = entity_baseline_ok and edge_baseline_ok and ratio_ok and yield_ok

    return {
        "baseline_met": baseline_met,
        "entity_type_diffs": entity_diffs,
        "edge_type_diffs": edge_diffs,
        "rejection_ratio": {
            "legacy": round(legacy_ratio, 4),
            "bundle": round(bundle_ratio, 4),
            "delta_pp": round(ratio_delta * 100, 2),
            "within_5pp": ratio_ok,
        },
        "yield_regressions": yield_regressions,
        "criteria": {
            "entity_extraction_within_10pct": entity_baseline_ok,
            "edge_extraction_within_10pct": edge_baseline_ok,
            "rejection_ratio_within_5pp": ratio_ok,
            "no_yield_regressions": yield_ok,
        },
    }


def _dominant_yield(yield_counts: dict) -> str:
    """Return the most common yield status for a pass, or 'UNKNOWN'."""
    if not yield_counts:
        return "UNKNOWN"
    return max(yield_counts, key=yield_counts.get)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Baseline harness for extraction refactor comparison (spec §8.3)",
    )
    parser.add_argument("--corpus", type=Path, help="Directory of documents to ingest")
    parser.add_argument("--collect", choices=["legacy", "bundle_passes"],
                        help="Collect extraction stats under this label")
    parser.add_argument("--compare", action="store_true",
                        help="Compare two previously collected result files")
    parser.add_argument("--legacy-results", type=Path,
                        help="Path to legacy collection JSON (for --compare)")
    parser.add_argument("--bundle-results", type=Path,
                        help="Path to bundle_passes collection JSON (for --compare)")
    args = parser.parse_args()

    if args.collect:
        if not args.corpus:
            parser.error("--collect requires --corpus")
        result = collect_from_corpus(args.corpus, args.collect)
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.compare:
        if not args.legacy_results or not args.bundle_results:
            parser.error("--compare requires --legacy-results and --bundle-results")
        legacy = json.loads(args.legacy_results.read_text())
        bundle = json.loads(args.bundle_results.read_text())
        diff = compare(legacy, bundle)
        print(json.dumps(diff, indent=2, default=str))
        return 0 if diff["baseline_met"] else 1

    parser.error("provide either --collect or --compare")


if __name__ == "__main__":
    sys.exit(main())
