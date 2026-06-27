#!/usr/bin/env python3
"""Data-driven mining of prose-friendly, GENERALIZABLE per-pass keywords.

For each field-group pass, find unigrams/bigrams that appear in the pass's
POSITIVE chunks (lineage target) materially more than its NEGATIVE chunks
(within-pass discrimination), AND that recur across MULTIPLE documents (so they
generalize rather than overfit one doc). Excludes numbers/units, stopwords,
short tokens, and instance-designation patterns (digit-bearing, letter-dash-digit)
to honor the no-instance-names guardrail. Output is a REVIEW list — not
auto-committed.

    python3 -m scripts.mine_pass_keywords --runs <r1,r2,...>

Positives source (two modes):
  * Default: per-pass positive chunk indexes are read from ``field_provenance``
    rows in the pipeline's own ``extract_pass_response_json`` (circular: mined
    from what the pipeline already found).
  * ``--positives-csv <path>``: a bakeoff/relabel CSV with columns including
    ``run_id``, ``pass_name``, ``chunk_index``, and ``used``; rows with
    ``used==1`` supply the positive set for each (run, pass) pair. Runs in
    ``--runs`` that have no CSV positives for a pass are treated as all-negative
    (same as the default when field_provenance is absent). Use this mode for
    value-grounded labels that are independent of the pipeline's own retrieval.

Unit-token exception (``--allow-units``):
  When set, tokens that are unit synonyms — membership in the union of all
  ``app.services.field_value_grounding.SUFFIX_UNITS`` values, after NFC +
  casefold — bypass both the DESIG drop and the WORD regex length floor.  This
  lets short tokens like "kw", "µs", "m/s", or "mach" survive as keyword
  candidates. The DESIG guardrail (no instance/designation names) still drops
  digit-bearing tokens that are NOT in the unit set; the no-instance-names rule
  is non-negotiable.

NFC parity: text is normalised with NFC+casefold — identical to the runtime
``keyword_hit_counts`` function — so lift/pos-fire/neg-fire statistics predict
runtime matching behaviour exactly. (Prior versions used NFKD+strip-combining,
which differs from the runtime path.)
"""
from __future__ import annotations
import argparse, base64, csv, json, os, re, unicodedata, urllib.request
from collections import defaultdict
from pathlib import Path

_PG = os.environ.get("A0_DATABASE_URL", "postgresql+psycopg2://eip:eip_secret@localhost:5437/eip")
_ARC = os.environ.get("A0_ARCADEDB_URL", "http://localhost:2480").rstrip("/")
PASSES = ["radar_power_rf","radar_antenna","radar_modulation","radar_timing",
          "missile_kinematics","missile_airframe","missile_propulsion",
          "missile_guidance","missile_speed_timing"]

STOP = set("the a an and or of to in for with on at by is are be as it its this that "
           "from into over under per each both all any can may which when where while "
           "such not no only also more most than then they them their these those has "
           "have had was were will would could should one two three first second other "
           "between within about above below up down out off was been being but if".split())
WORD = re.compile(r"[a-z][a-z\-]{2,}")          # alpha tokens, len>=3, allows hyphen
DESIG = re.compile(r"\d|^[a-z]{1,3}-?\d")        # digit-bearing or letter-dash-digit → instance-ish


def _nfc(t: str) -> str:
    """NFC-normalise + casefold — matches runtime ``keyword_hit_counts`` exactly."""
    return unicodedata.normalize("NFC", t).casefold()


def _build_unit_set() -> frozenset[str]:
    """Return the NFC+casefolded union of all SUFFIX_UNITS synonym lists."""
    from app.services.field_value_grounding import SUFFIX_UNITS
    return frozenset(_nfc(u) for syns in SUFFIX_UNITS.values() for u in syns)


def _arc(sql: str):
    auth = base64.b64encode(b"root:eip_arcadedb_secret").decode()
    req = urllib.request.Request(f"{_ARC}/api/v1/command/eip_knowledge_graph",
        data=json.dumps({"command": sql, "language": "sql"}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"})
    return json.load(urllib.request.urlopen(req, timeout=40)).get("result", [])


def _terms(text: str, unit_set: frozenset[str] | None = None) -> set[str]:
    """Tokenise *text* into unigram/bigram candidates.

    When *unit_set* is provided (``--allow-units`` mode), tokens that are unit
    synonyms bypass the DESIG filter and the WORD length floor so short
    unit strings like "kw", "µs", "m/s" appear as candidates.  All other
    guardrails (stopwords, DESIG for non-unit tokens) remain in force.
    """
    # Tokenise: all non-whitespace runs, then also the WORD pattern for the
    # alpha-only path.  We collect two pools:
    #   • prose_toks: WORD-matched alpha tokens (existing behaviour)
    #   • unit_toks: unit-set members when allow-units is active
    raw_toks_for_units: list[str] = []
    if unit_set is not None:
        # Split on whitespace/punctuation except '/' and 'µ' to preserve "m/s" and "µs"
        for tok in re.split(r"[\s,;:()\[\]{}<>\"']+", text):
            if tok:
                raw_toks_for_units.append(_nfc(tok))

    prose_toks = [w for w in WORD.findall(text) if w not in STOP and not DESIG.search(w)]

    if unit_set is None:
        toks = prose_toks
    else:
        # Merge: prose_toks first, then insert unit tokens that aren't already present
        # We need to preserve ordering for bigram construction, so build a merged list
        # that keeps prose tokens in order and appends unit-only tokens at their
        # natural positions (approximate: use raw_toks_for_units as the ordering base).
        unit_only = [t for t in raw_toks_for_units if t in unit_set and t not in STOP]
        toks_set = set(prose_toks)
        extra_unit_toks = [t for t in unit_only if t not in toks_set]
        toks = prose_toks  # bigrams over prose sequence

    out = set(toks)
    for a, b in zip(toks, toks[1:]):             # bigrams over prose sequence
        out.add(f"{a} {b}")

    if unit_set is not None:
        # Add unit-only tokens as unigrams (not bigrams with prose — no ordering)
        for t in raw_toks_for_units:
            if t in unit_set and t not in STOP:
                out.add(t)

    return out


def load_csv_possets(csv_path: str) -> dict[tuple[str, str], set[int]]:
    """Load per-(run_id, pass_name) positive chunk_index sets from a bakeoff CSV.

    Only rows with ``used == '1'`` (or 1) are included. Rows with ``used == 0``
    are treated as negative (excluded from the positive set). Returns a dict
    mapping ``(run_id, pass_name)`` → ``{chunk_index, ...}``.

    Args:
        csv_path: Path to a CSV file with columns ``run_id``, ``pass_name``,
            ``chunk_index``, and ``used``.

    Returns:
        Dict of ``(run_id, pass_name)`` → set of positive chunk indexes.
    """
    possets: dict[tuple[str, str], set[int]] = defaultdict(set)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("used", "0")).strip() == "1":
                possets[(row["run_id"], row["pass_name"])].add(int(row["chunk_index"]))
    return dict(possets)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--min-lift", type=float, default=0.10)
    ap.add_argument("--min-docs", type=int, default=2)
    ap.add_argument("--min-posfire", type=float, default=0.15)
    ap.add_argument("--positives-csv", default=None,
                    help="Path to a bakeoff/relabel CSV; used==1 rows supply the "
                         "positive set per (run, pass). Overrides field_provenance.")
    ap.add_argument("--allow-units", action="store_true",
                    help="Allow unit-synonym tokens (e.g. 'kw', 'µs', 'm/s') to "
                         "bypass DESIG/WORD guards and appear as keyword candidates.")
    args = ap.parse_args(argv)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]

    unit_set: frozenset[str] | None = None
    if args.allow_units:
        unit_set = _build_unit_set()

    csv_possets: dict[tuple[str, str], set[int]] | None = None
    if args.positives_csv:
        csv_possets = load_csv_possets(args.positives_csv)

    from sqlalchemy import create_engine, text as _t
    eng = create_engine(_PG)
    # per pass: pos[doc] = list of chunk-term-sets ; neg[doc] = list
    pos: dict[str, dict[str, list[set]]] = {p: defaultdict(list) for p in PASSES}
    neg: dict[str, dict[str, list[set]]] = {p: defaultdict(list) for p in PASSES}
    for run in runs:
        chunks = {int(c["chunk_index"]): _terms(_nfc(c.get("chunk_text") or ""), unit_set)
                  for c in _arc(f"SELECT chunk_index, chunk_text FROM ExtractionChunk WHERE pipeline_run_id='{run}'")}
        with eng.connect() as c:
            rows = c.execute(_t("SELECT pass_name, extract_pass_response_json FROM ingest.pipeline_pass_outputs "
                                "WHERE pipeline_run_id=:r ORDER BY pass_name, attempt"), {"r": run}).fetchall()
        latest = {pn: (json.loads(j) if isinstance(j, str) else j) for pn, j in rows}
        for p in PASSES:
            if csv_possets is not None:
                # Value-grounded labels: use CSV positive set for this (run, pass)
                posset: set[int] = csv_possets.get((run, p), set())
            else:
                # Default: derive positive set from field_provenance
                j = latest.get(p); posset = set()
                if j:
                    for fp in (j.get("field_provenance") or []):
                        posset.update(x for x in (fp.get("chunk_indexes") or []) if isinstance(x, int))
                        if isinstance(fp.get("chunk_index"), int):
                            posset.add(fp["chunk_index"])
            for ci, terms in chunks.items():
                (pos[p] if ci in posset else neg[p])[run].append(terms)

    for p in PASSES:
        all_pos = [s for docs in pos[p].values() for s in docs]
        all_neg = [s for docs in neg[p].values() for s in docs]
        nP, nN = len(all_pos), len(all_neg)
        if nP == 0:
            print(f"## {p}: 0 positives — skip\n"); continue
        # candidate terms = those in any positive
        cand = set().union(*all_pos) if all_pos else set()
        scored = []
        for term in cand:
            pf = sum(1 for s in all_pos if term in s) / nP
            nf = (sum(1 for s in all_neg if term in s) / nN) if nN else 0.0
            docspread = sum(1 for run, docs in pos[p].items() if any(term in s for s in docs))
            if pf >= args.min_posfire and (pf - nf) >= args.min_lift and docspread >= args.min_docs:
                scored.append((pf - nf, pf, nf, docspread, term))
        scored.sort(reverse=True)
        print(f"## {p}  ({nP} pos / {nN} neg)  candidates: {len(scored)}")
        print(f"   {'lift':>5s} {'posf':>5s} {'negf':>5s} {'docs':>4s}  term")
        for lift, pf, nf, ds, term in scored[:18]:
            print(f"   {lift:+5.2f} {pf:5.2f} {nf:5.2f} {ds:4d}  {term}")
        print()
    mode = f"csv:{Path(args.positives_csv).name}" if args.positives_csv else "field_provenance"
    units_note = " unit-tokens=ALLOWED" if args.allow_units else ""
    print(f"filters: pos-fire≥{args.min_posfire}, lift≥{args.min_lift}, docs≥{args.min_docs} "
          f"(generalizable). Numbers/units/stopwords/designations excluded (non-unit).{units_note} "
          f"positives-source={mode}. REVIEW before committing.")
    print()
    print("REVIEW ONLY — hand-curate into manifests via inject/union; see guarded-ranker spec §5.4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
