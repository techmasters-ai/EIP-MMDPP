#!/usr/bin/env python3
"""Task 14 STEP 1 — table-haystack vocabulary diagnostic (header projection).

Answers ONE question empirically, producing a GO/NO-GO evidence report for a
possible ``match_text`` haystack feature (guarded-ranker spec §5.2):

    Do table-derived chunks' MATCHING haystacks (``chunk_text``) lack the
    header/caption/section vocabulary the lexical channels need — or is that
    vocabulary ALREADY in the haystack, making a separate ``match_text``
    field (headers + caption + section heading) YAGNI?

For each target chunk it reports:

  1. STRUCTURAL summary — is_table flag, text length, first 300 chars,
     "- label: value" line count (the REAL compiled ``_LABEL_VALUE_RE`` from
     ``extraction_candidate_scoring``), whether a rendered "TABLE: {caption}"
     line exists, whether a contextualize/heading prefix exists (lines before
     the TABLE:/first label-value line), legacy flattened-table markers
     (", N = " — the pre-normalization Docling table serialization), and
     the projected ``headings`` / ``section_path`` columns.

  2. CHANNEL-VOCABULARY audit vs the passes the chunk was a bake-off positive
     for, using the REAL production matchers (zero recompute drift):
       (a) effective per-pass ``lexical_keywords`` — manifest list UNIONed with
           schema-derived units via ``inject_pass_keywords`` (post-Task-11
           semantics) → ``keyword_hit_counts`` (post-Task-12 word-boundary
           semantics);
       (b) field aliases — ``signals.field_queries`` → ``lexical_hit_counts``;
       (c) unit-signature tokens — ``signals.unit_signature`` →
           ``has_unit_token`` / ``count_unit_tokens``.

  3. HEADER-DELTA test — the decisive evidence: rebuild the candidate
     ``match_text`` (projected headings + section_path + any TABLE: caption
     line) and re-run every channel on IT; any needle that fires on the
     header text but NOT on ``chunk_text`` is vocabulary ``match_text`` would
     ADD. Empty delta everywhere = match_text adds nothing = NO-GO.

READ-ONLY: ArcadeDB is queried over HTTP with SELECTs only (same ``_arc``
pattern + env config as scripts/a0_captured_separation.py):

    A0_ARCADEDB_URL   (default http://localhost:2480)
    A0_ARCADEDB_DB    (default eip_knowledge_graph)
    A0_ARCADEDB_USER  / A0_ARCADEDB_PASS  (default root / eip_arcadedb_secret)

Usage::

    python3 -m scripts.diagnose_table_haystack \
        --run 1329caf5-d57b-403d-96ea-1399a7d3d67f --chunks 93,94,124

    python3 -m scripts.diagnose_table_haystack --all-tables   # scan is_table rows

Writes docs/operational/table-haystack-diagnostic-2026-06.md and prints the
same markdown to stdout.
"""
from __future__ import annotations

import argparse
import base64
import importlib
import json
import os
import re
import sys
import urllib.request
from dataclasses import replace
from typing import Any, Sequence

# Production matchers / resolvers — the REAL code paths, no reimplementation.
from app.api.v1.extraction_routing import inject_pass_keywords
from app.services.extraction_candidate_scoring import _LABEL_VALUE_RE
from app.services.extraction_lexical_search import (
    keyword_hit_counts,
    lexical_hit_counts,
)
from app.services.extraction_query_builder import build_retrieval_profile
from app.services.extraction_unit_gate import count_unit_tokens
from app.services.field_value_grounding import has_unit_token, nfc
from app.services.ontology_bundles import load_bundle_manifest

DEFAULT_RUN = "1329caf5-d57b-403d-96ea-1399a7d3d67f"  # Engagement doc
DEFAULT_BUNDLE = "air_defense_v3"
DEFAULT_OUT = "docs/operational/table-haystack-diagnostic-2026-06.md"

# Known table-derived positives from the bake-off (per-chunk pass lists).
BAKEOFF_CHUNK_PASSES: dict[int, list[str]] = {
    93: ["missile_kinematics", "radar_antenna", "radar_power_rf"],
    94: ["radar_antenna"],
    124: ["missile_kinematics", "radar_antenna"],
}

# Legacy (pre-table-normalization) Docling flattened-table serialization
# marker: "Azimuth, 1 = ±45°" → ", 1 = ".
_LEGACY_FLAT_RE = re.compile(r",\s*\d+\s*=")


# ---------------------------------------------------------------------------
# ArcadeDB read-only helper (a0_captured_separation pattern)
# ---------------------------------------------------------------------------
def _arcadedb_cfg() -> tuple[str, str, str, str]:
    return (
        os.environ.get("A0_ARCADEDB_URL", "http://localhost:2480").rstrip("/"),
        os.environ.get("A0_ARCADEDB_DB", "eip_knowledge_graph"),
        os.environ.get("A0_ARCADEDB_USER", "root"),
        os.environ.get("A0_ARCADEDB_PASS", "eip_arcadedb_secret"),
    )


def _arc(sql: str) -> list[dict]:
    base, db, user, pw = _arcadedb_cfg()
    auth = base64.b64encode(f"{user}:{pw}".encode()).decode()
    req = urllib.request.Request(
        f"{base}/api/v1/command/{db}",
        data=json.dumps({"command": sql, "language": "sql"}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return list(json.load(resp).get("result") or [])


def fetch_chunks(run_id: str) -> list[dict]:
    """All ExtractionChunk rows of the run (SELECT only — read-only)."""
    rid = run_id.replace("'", "")
    return _arc(
        "SELECT chunk_index, chunk_text, is_table, source_refs, vertex_id, "
        "self_ref, headings, section_path FROM ExtractionChunk "
        f"WHERE pipeline_run_id = '{rid}' ORDER BY chunk_index ASC"
    )


# ---------------------------------------------------------------------------
# Pass resolution (audit_groundable_fields / extraction_routing pattern)
# ---------------------------------------------------------------------------
def _resolve_template_class(bundle_key: str, pass_def) -> type:
    mod = importlib.import_module(f"ontology_bundles.{bundle_key}.{pass_def.module}")
    return getattr(mod, pass_def.template_class)


def resolve_pass_channels(bundle_key: str) -> dict[str, dict[str, Any]]:
    """{pass_name: {signals, effective_keywords, keyword_weights}} for every
    field-group pass (pass_def.retrieval is not None) in the bundle."""
    manifest = load_bundle_manifest(bundle_key)
    out: dict[str, dict[str, Any]] = {}
    for pass_def in manifest.passes:
        if pass_def.retrieval is None:
            continue  # identity / relationship passes — not routable
        template_cls = _resolve_template_class(bundle_key, pass_def)
        signals = build_retrieval_profile(pass_def, template_cls)
        effective = inject_pass_keywords(pass_def.retrieval, signals)
        out[pass_def.name] = {
            "signals": signals,
            "effective_keywords": list(effective.lexical_keywords),
            "keyword_weights": dict(effective.lexical_keyword_weights or {}),
        }
    return out


# ---------------------------------------------------------------------------
# Per-chunk channel audit (real matchers on single-row lists)
# ---------------------------------------------------------------------------
def _row_for_matchers(text: str) -> dict:
    return {"self_ref": "diag", "vertex_id": None, "chunk_text": text}


def _matched_keywords(text: str, keywords: Sequence[str]) -> list[str]:
    """Which effective keywords fire on *text* (real keyword_hit_counts,
    one needle at a time so we can NAME the matches)."""
    row = [_row_for_matchers(text)]
    return [
        k for k in keywords
        if keyword_hit_counts(row, [k])["diag"]["keyword_hits"] > 0
    ]


def _matched_aliases(text: str, field_queries) -> list[tuple[str, str]]:
    """[(field_name, alias)] pairs that fire (real lexical_hit_counts, one
    single-alias FieldRetrievalQuery at a time)."""
    row = [_row_for_matchers(text)]
    hits: list[tuple[str, str]] = []
    for fq in field_queries:
        for alias in fq.aliases:
            probe = replace(fq, aliases=(alias,), negative_terms=())
            if lexical_hit_counts(row, [probe])["diag"]["alias_hits"] > 0:
                hits.append((fq.field_name, alias))
    return hits


def _matched_units(text_nfc: str, signature: Sequence[str]) -> list[str]:
    return [u for u in signature if has_unit_token(text_nfc, [u])]


def audit_chunk_pass(chunk_text: str, channels: dict[str, Any]) -> dict[str, Any]:
    """All three channel verdicts for one (chunk, pass) using the production
    matchers on a single-row list."""
    signals = channels["signals"]
    keywords = channels["effective_keywords"]
    weights = channels["keyword_weights"]
    row = [_row_for_matchers(chunk_text)]
    text_nfc = nfc(chunk_text)

    kw_total = keyword_hit_counts(row, keywords, weights)["diag"]["keyword_hits"]
    lex = lexical_hit_counts(row, signals.field_queries)["diag"]
    unit_count = count_unit_tokens(text_nfc, signals.unit_signature)

    return {
        "keyword_hits": float(kw_total),
        "keywords_matched": _matched_keywords(chunk_text, keywords),
        "n_keywords_effective": len(keywords),
        "alias_hits": int(lex["alias_hits"]),
        "aliases_matched": _matched_aliases(chunk_text, signals.field_queries),
        "supported_fields": sorted(lex["supported_fields"]),
        "unit_tokens": int(unit_count),
        "units_matched": _matched_units(text_nfc, signals.unit_signature),
        "unit_fires": bool(has_unit_token(text_nfc, signals.unit_signature)),
    }


# ---------------------------------------------------------------------------
# Structural summary
# ---------------------------------------------------------------------------
def structural_summary(row: dict) -> dict[str, Any]:
    text = row.get("chunk_text") or ""
    lines = text.splitlines()
    text_nfc = nfc(text)
    label_value_lines = len(_LABEL_VALUE_RE.findall(text_nfc))
    table_caption_line = next(
        (ln.strip() for ln in lines if ln.strip().startswith("TABLE:")), None
    )

    # Contextualize/heading prefix = lines BEFORE the TABLE: line or the first
    # label-value line (whichever comes first).
    first_struct = len(lines)
    for i, ln in enumerate(lines):
        if ln.strip().startswith("TABLE:") or _LABEL_VALUE_RE.match(ln):
            first_struct = i
            break
    prefix_lines = [ln for ln in lines[:first_struct] if ln.strip()]

    headings = [h for h in (row.get("headings") or []) if h]
    section_path = row.get("section_path") or ""
    # Is the projected heading already IN the chunk_text haystack?
    folded = text_nfc.casefold()
    headings_in_text = all(nfc(h).casefold() in folded for h in headings) if headings else False

    return {
        "is_table": row.get("is_table"),
        "text_len": len(text),
        "first_300": text[:300],
        "label_value_lines": label_value_lines,
        "table_caption_line": table_caption_line,
        "prefix_lines": prefix_lines,
        "legacy_flat_markers": len(_LEGACY_FLAT_RE.findall(text)),
        "table_source_refs": [s for s in (row.get("source_refs") or []) if "/tables/" in s],
        "headings": headings,
        "section_path": section_path,
        "headings_in_text": headings_in_text,
    }


def build_match_text_candidate(row: dict, struct: dict[str, Any]) -> str:
    """What the hypothesized ``match_text`` field would contain: projected
    headings + section_path + rendered TABLE: caption (when present)."""
    parts: list[str] = list(struct["headings"])
    if struct["section_path"] and struct["section_path"] not in parts:
        parts.append(struct["section_path"])
    if struct["table_caption_line"]:
        parts.append(struct["table_caption_line"].removeprefix("TABLE:").strip())
    return "\n".join(p for p in parts if p)


def header_delta(chunk_text: str, match_text: str, channels: dict[str, Any]) -> dict[str, list]:
    """Needles that fire on the candidate match_text but NOT on chunk_text —
    the vocabulary match_text would ADD. Empty everywhere = YAGNI evidence."""
    signals = channels["signals"]
    keywords = channels["effective_keywords"]
    if not match_text.strip():
        return {"keywords": [], "aliases": [], "units": []}
    kw_body = set(_matched_keywords(chunk_text, keywords))
    kw_head = set(_matched_keywords(match_text, keywords))
    al_body = set(_matched_aliases(chunk_text, signals.field_queries))
    al_head = set(_matched_aliases(match_text, signals.field_queries))
    un_body = set(_matched_units(nfc(chunk_text), signals.unit_signature))
    un_head = set(_matched_units(nfc(match_text), signals.unit_signature))
    return {
        "keywords": sorted(kw_head - kw_body),
        "aliases": sorted(al_head - al_body),
        "units": sorted(un_head - un_body),
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _fence(text: str) -> list[str]:
    return ["```text", *text.splitlines(), "```"]


def _fmt_list(items: Sequence, cap: int = 12) -> str:
    items = list(items)
    if not items:
        return "—"
    shown = ", ".join(
        (f"{i[0]}:{i[1]!r}" if isinstance(i, tuple) else str(i)) for i in items[:cap]
    )
    extra = f" (+{len(items) - cap} more)" if len(items) > cap else ""
    return shown + extra


def corpus_shape_stats(rows: list[dict]) -> dict[str, Any]:
    """Run-level structural shape: does the render_graph.py 'TABLE:' shape
    exist ANYWHERE in this run's haystacks? (Central to the §5.2 premise.)"""
    n = len(rows)
    with_table_line = sorted(
        int(r["chunk_index"]) for r in rows
        if "TABLE:" in (r.get("chunk_text") or "") and r.get("chunk_index") is not None
    )
    with_tables_ref = sorted(
        int(r["chunk_index"]) for r in rows
        if any("/tables/" in s for s in (r.get("source_refs") or []))
        and r.get("chunk_index") is not None
    )
    with_legacy_flat = sorted(
        int(r["chunk_index"]) for r in rows
        if _LEGACY_FLAT_RE.search(r.get("chunk_text") or "")
        and r.get("chunk_index") is not None
    )
    return {
        "n_chunks": n,
        "with_table_line": with_table_line,
        "with_tables_ref": with_tables_ref,
        "with_legacy_flat": with_legacy_flat,
    }


def render_report(
    run_id: str,
    bundle_key: str,
    targets: list[dict[str, Any]],
    is_table_all_null: bool,
    all_tables_mode: bool,
    all_tables_fallback: bool,
    corpus: dict[str, Any] | None = None,
) -> str:
    L: list[str] = []
    L.append("# Table-haystack vocabulary diagnostic — guarded-ranker Task 14 STEP 1")
    L.append("")
    L.append(f"- run: `{run_id}` (Engagement doc)" if run_id == DEFAULT_RUN else f"- run: `{run_id}`")
    L.append(f"- bundle: `{bundle_key}`")
    L.append(f"- generated by: `scripts/diagnose_table_haystack.py` (read-only ArcadeDB)")
    L.append("")
    if is_table_all_null:
        L.append(
            "> **NOTE (prominent):** every `ExtractionChunk.is_table` of this run is "
            "`null` — the rows were indexed BEFORE Task 9; `is_table` arrives only "
            "after Phase-3 re-collection. The structural `is_table` column therefore "
            "cannot confirm table identity here, but the vocabulary question is "
            "answered from `chunk_text` regardless."
        )
        L.append("")
    if all_tables_mode and all_tables_fallback:
        L.append(
            "> **NOTE:** `--all-tables` found ZERO `is_table=true` rows (see above); "
            "fell back to chunks whose `source_refs` include a `#/tables/` element — "
            "the only structural table identity available pre-Task-9."
        )
        L.append("")

    if corpus is not None:
        L.append("## Run-level structural shape")
        L.append("")
        n_tl = len(corpus["with_table_line"])
        L.append(
            f"- chunks in run: {corpus['n_chunks']}  |  chunks containing a rendered "
            f"`TABLE:` line: **{n_tl}**"
            + (f" ({corpus['with_table_line']})" if n_tl else "")
        )
        if n_tl == 0:
            L.append(
                "- **The `render_graph.py` table-normalization shape (contextualize "
                "prefix + `TABLE: {caption}` + `ENTITY:` block + `- label: value` rows) "
                "is ABSENT from this entire run** — the §5.2 structural premise "
                "(rendered table chunks whose haystack lacks header vocabulary) does "
                "not describe this corpus. Table-derived chunks here are merged "
                "text-fragment chunks and legacy `, N = ` flattened Docling tables "
                "whose row labels and bracketed unit headers arrive INLINE in "
                "`chunk_text`."
            )
        L.append(
            f"- chunks with a `#/tables/` source_ref: {len(corpus['with_tables_ref'])} "
            f"({corpus['with_tables_ref']})"
        )
        L.append(
            f"- chunks with legacy `, N = ` flattened-table markers: "
            f"{len(corpus['with_legacy_flat'])} ({corpus['with_legacy_flat']})"
        )
        L.append("")

    # ----- per-chunk evidence -----
    verdict_rows: list[tuple] = []
    any_unreachable = False
    any_header_delta = False
    for t in targets:
        ci = t["chunk_index"]
        s = t["struct"]
        L.append(f"## Chunk {ci}")
        L.append("")
        is_table_disp = s["is_table"] if s["is_table"] is not None else "null (pre-Task-9 index)"
        L.append(f"- `is_table`: {is_table_disp}  |  text length: {s['text_len']} chars")
        L.append(f"- `- label: value` lines (`_LABEL_VALUE_RE`): **{s['label_value_lines']}**")
        L.append(
            f"- rendered `TABLE:` caption line: "
            f"{'**yes** — ' + repr(s['table_caption_line']) if s['table_caption_line'] else '**no** (NOT the render_graph.py shape)'}"
        )
        L.append(
            f"- contextualize/heading prefix lines (before TABLE:/first label line): "
            f"{len(s['prefix_lines'])}"
            + (f" — {s['prefix_lines'][:3]!r}" if s["prefix_lines"] else "")
        )
        L.append(
            f"- legacy flattened-table markers (`, N = `): {s['legacy_flat_markers']}"
            f"  |  `#/tables/` source_refs: {len(s['table_source_refs'])}"
        )
        L.append(
            f"- projected headings: {s['headings']!r}  |  section_path: {s['section_path']!r}"
            f"  |  headings already in chunk_text: **{'yes' if s['headings_in_text'] else 'no'}**"
        )
        L.append("")
        L.append("First 300 chars:")
        L += _fence(s["first_300"])
        L.append("")

        L.append("### Channel audit (real production matchers, post-Tasks-11/12)")
        L.append("")
        L.append("| pass | keyword_hits | alias_hits | unit_tokens | reachable today? |")
        L.append("|---|---|---|---|---|")
        for pn, a in t["audits"].items():
            reachable = a["keyword_hits"] > 0 or a["alias_hits"] > 0 or a["unit_tokens"] > 0
            if not reachable:
                any_unreachable = True
            verdict_rows.append((
                ci, pn, a["keyword_hits"], a["alias_hits"], a["unit_tokens"], reachable
            ))
            L.append(
                f"| {pn} | {a['keyword_hits']:g} | {a['alias_hits']} | "
                f"{a['unit_tokens']} | {'YES' if reachable else '**NO**'} |"
            )
        L.append("")
        for pn, a in t["audits"].items():
            L.append(f"**{pn}** — matched needles:")
            L.append(f"- keywords ({len(a['keywords_matched'])}/{a['n_keywords_effective']} effective): "
                     f"{_fmt_list(a['keywords_matched'])}")
            L.append(f"- aliases: {_fmt_list(a['aliases_matched'])}")
            L.append(f"- unit-signature tokens: {_fmt_list(a['units_matched'])}")
            d = t["deltas"][pn]
            delta_bits = d["keywords"] + d["aliases"] + d["units"]
            if delta_bits:
                any_header_delta = True
            L.append(
                f"- **header-delta** (fires on headings/caption but NOT chunk_text): "
                f"keywords {_fmt_list(d['keywords'])} | aliases {_fmt_list(d['aliases'])} "
                f"| units {_fmt_list(d['units'])}"
            )
            L.append("")

    # ----- verdict table -----
    L.append("## Verdict table")
    L.append("")
    L.append("| chunk | pass | keyword_hits>0 | alias_hits>0 | unit_tokens>0 | reachable |")
    L.append("|---|---|---|---|---|---|")
    for ci, pn, kw, al, un, reach in verdict_rows:
        L.append(
            f"| {ci} | {pn} | {'YES' if kw > 0 else 'no'} | {'YES' if al > 0 else 'no'} | "
            f"{'YES' if un > 0 else 'no'} | {'YES' if reach else '**NO**'} |"
        )
    L.append("")

    # ----- recommendation -----
    n_rows = len(verdict_rows)
    n_reach = sum(1 for r in verdict_rows if r[5])
    L.append("## Recommendation")
    L.append("")
    if not any_unreachable and not any_header_delta:
        rec = "NO-GO"
        rationale = (
            f"All {n_reach}/{n_rows} (chunk, pass) cells are already reachable by at "
            "least one lexical channel on today's `chunk_text` haystack — the "
            "Task 11 keyword-union (`inject_pass_keywords`) and Task 12 boundary "
            "fixes revived the channels without any haystack change. The "
            "header-delta test found ZERO needles (keyword, alias, or unit) that "
            "fire on the projected headings/caption but not on `chunk_text` — the "
            "heading vocabulary is already embedded in the haystack, so a separate "
            "`match_text` field would add no new lexical reachability."
        )
    elif any_unreachable and any_header_delta:
        rec = "GO"
        rationale = (
            f"{n_rows - n_reach}/{n_rows} (chunk, pass) cells are NOT reachable by "
            "any lexical channel today, and the header-delta test shows needles "
            "that fire on header/caption vocabulary absent from `chunk_text` — "
            "`match_text` would demonstrably add that missing vocabulary class."
        )
    else:
        rec = "NO-GO"
        rationale = (
            f"{n_rows - n_reach}/{n_rows} (chunk, pass) cells lack lexical hits, but "
            "the header-delta test found no needle that the headings/caption would "
            "add — the missing vocabulary would NOT come from headers either, so "
            "`match_text` cannot close the gap. The remaining reach must come from "
            "the dense/rerank channels (or the unit gate), not a new haystack field."
        )
    L.append(f"**{rec}** on `match_text`.")
    L.append("")
    L.append(rationale)
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=DEFAULT_RUN, help="pipeline_run_id")
    ap.add_argument("--chunks", default="93,94,124",
                    help="comma-sep chunk_index targets (bake-off table positives)")
    ap.add_argument("--all-tables", action="store_true",
                    help="scan every is_table row of the run instead of --chunks "
                         "(falls back to '#/tables/' source_refs when is_table is "
                         "all-null, i.e. pre-Task-9 runs)")
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    rows = fetch_chunks(args.run)
    if not rows:
        print(f"BLOCKED: ArcadeDB has no ExtractionChunk rows for run {args.run!r} "
              "— cannot run the diagnostic.", file=sys.stderr)
        return 2

    by_index = {int(r["chunk_index"]): r for r in rows if r.get("chunk_index") is not None}
    is_table_all_null = all(r.get("is_table") is None for r in rows)

    channels = resolve_pass_channels(args.bundle)
    field_group_passes = list(channels)  # the 9 routable field-group passes

    all_tables_fallback = False
    if args.all_tables:
        targets_idx = [ci for ci, r in sorted(by_index.items()) if r.get("is_table")]
        if not targets_idx:
            all_tables_fallback = True
            targets_idx = [
                ci for ci, r in sorted(by_index.items())
                if any("/tables/" in s for s in (r.get("source_refs") or []))
            ]
        chunk_passes = {ci: field_group_passes for ci in targets_idx}
    else:
        targets_idx = []
        for tok in args.chunks.split(","):
            tok = tok.strip()
            if tok:
                targets_idx.append(int(tok))
        chunk_passes = {
            ci: BAKEOFF_CHUNK_PASSES.get(ci, field_group_passes) for ci in targets_idx
        }

    missing = [ci for ci in targets_idx if ci not in by_index]
    if missing:
        print(f"WARNING: chunk_index {missing} not found in run {args.run!r}; skipping.",
              file=sys.stderr)
        targets_idx = [ci for ci in targets_idx if ci in by_index]
    if not targets_idx:
        print("BLOCKED: no target chunks resolved.", file=sys.stderr)
        return 2

    targets: list[dict[str, Any]] = []
    for ci in targets_idx:
        row = by_index[ci]
        text = row.get("chunk_text") or ""
        struct = structural_summary(row)
        match_text = build_match_text_candidate(row, struct)
        audits = {pn: audit_chunk_pass(text, channels[pn]) for pn in chunk_passes[ci]}
        deltas = {pn: header_delta(text, match_text, channels[pn]) for pn in chunk_passes[ci]}
        targets.append({
            "chunk_index": ci,
            "struct": struct,
            "audits": audits,
            "deltas": deltas,
        })

    report = render_report(
        args.run, args.bundle, targets,
        is_table_all_null, args.all_tables, all_tables_fallback,
        corpus=corpus_shape_stats(rows),
    )
    print(report)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(report + "\n")
    print(f"\nwrote: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
