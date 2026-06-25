#!/usr/bin/env python3
"""Per-document + cumulative LIVE-run-vs-GT selection metrics for absolute_union.

RE-RUNNABLE / IDEMPOTENT: reads the current set of COMPLETE sweep runs from
/home/josh/.guardrank_eval_state/t7_results.txt on every invocation, so it can
be run repeatedly as the Task-7 sweep finishes more docs.

For each COMPLETED live sweep run it RECONSTRUCTS the shipped absolute_union
keep-rule from the run's ACTUAL data:
    live_selected(pass, chunk) =
        measurement_present(dims, chunk_text)
        OR categorical_present(cats, chunk_text)
        OR (has_image AND image_present(source_refs))
        OR (live_max_field_cosine >= TAU)
and joins that selection to the bake-off ground truth (`used` 0/1) to report
recall / precision / frac_selected / F1 per doc and cumulatively.

DATA SOURCES
  1. Completed runs:  t7_results.txt  lines `sweep|<label>|status=COMPLETE|...|run=<8char>`.
  2. Run -> doc:      postgres ingest.pipeline_runs + ingest.documents (prefix match).
  3. Live chunks:     ArcadeDB ExtractionChunk (chunk_index, chunk_text, source_refs)
                      filtered by full pipeline_run_id.
  4. Live cosine:     postgres ingest.pipeline_pass_outputs
                      diagnostics_json->router->score_components_all  (per pass;
                      candidate_key ends in chunk_<idx>, has max_field_cosine).
  5. GT:              concat reports/*/bakeoff_dataset.parquet, dedup
                      (run_id, pass_name, chunk_index); doc_filename carried in parquet.

ALIGNMENT (critical): the live graph_only re-runs re-chunk image content, so a
GT run's chunk_index does NOT generally line up with the live run's chunk_index.
We therefore JOIN BY NORMALIZED chunk_text (pooled across all GT runs for the
doc) and report the text-match rate. GT positive cells whose exact text is NOT
present in the live run cannot be scored against the live selection and are
reported separately (they reflect re-chunking drift, not a selector miss).

τ = 0.55.  Baseline of comparison = bake-off GT.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys

import pandas as pd

from app.services.extraction_pass_signal_config import derive_pass_signal_config
from app.services.extraction_signal_detectors import (
    measurement_present, categorical_present, image_present,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
TAU = 0.55
STATE_FILE = "/home/josh/.guardrank_eval_state/t7_results.txt"
PG_CONTAINER = "eip-mmdpp-postgres-1"
ARCADE_CONTAINER = "eip-mmdpp-arcadedb-1"
ARCADE_DB = "eip_knowledge_graph"
ARCADE_PW = "eip_arcadedb_secret"
REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "absolute_union_live_metrics_report.txt")

CFG = derive_pass_signal_config("air_defense_v3")
ROUTABLE = set(CFG.keys())

# Known GT-positive coverage per doc (coordinator-supplied reference, for the header note)
GT_REF = {
    "Engagement and Fire Control Radars (S-Band, X-band).pdf": 68,
    "S-75 Dvina _ Military Wiki _ Fandom.pdf": 20,
    "SA-2 Guideline _ Зенитный Ракетный Комплекс С-75 Двина_Десна_Волхов.pdf": 10,
    "SA-2_and_SR-71_17_Apr_2020.pdf": 6,
    "V-75 SA-2 GUIDELINE.pdf": 3,
    "S-75 Dvina.pdf": 3,
    "Images_Demo_Doc.pdf": 2,
    "SA-2 Surface-to-Air Missile _ National Museum of the United States Air Force™ _ Display.pdf": 2,
    "SNR-75 - Wikipedia.pdf": 1,
}


def norm(t) -> str:
    return re.sub(r"\s+", " ", str(t)).strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────
def pg(sql: str) -> str:
    """Run a read-only psql query, return raw tab-/pipe-separated text."""
    out = subprocess.run(
        ["docker", "exec", PG_CONTAINER, "psql", "-U", "eip", "-d", "eip",
         "-t", "-A", "-F", "\t", "-c", sql],
        capture_output=True, text=True, timeout=120,
    )
    if out.returncode != 0:
        raise RuntimeError(f"psql failed: {out.stderr.strip()}")
    return out.stdout


def arcade(sql: str) -> list[dict]:
    """Run a read-only ArcadeDB SQL command, return result list."""
    payload = json.dumps({"language": "sql", "command": sql})
    curl = (
        f"curl -s -u root:{ARCADE_PW} -X POST "
        f"http://localhost:2480/api/v1/command/{ARCADE_DB} "
        f"-H 'Content-Type: application/json' -d {json.dumps(payload)}"
    )
    out = subprocess.run(
        ["docker", "exec", ARCADE_CONTAINER, "sh", "-c", curl],
        capture_output=True, text=True, timeout=120,
    )
    if out.returncode != 0:
        raise RuntimeError(f"arcade curl failed: {out.stderr.strip()}")
    try:
        return json.loads(out.stdout).get("result", [])
    except json.JSONDecodeError:
        raise RuntimeError(f"arcade bad JSON: {out.stdout[:300]}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Parse completed sweep runs (idempotent: read file fresh)
# ─────────────────────────────────────────────────────────────────────────────
def parse_completed_runs() -> list[dict]:
    """Returns [{label, prefix, run_id, doc_filename}] for status=COMPLETE rows,
    in sweep order, dedup by label keeping the LAST occurrence."""
    if not os.path.exists(STATE_FILE):
        sys.exit(f"ERROR: state file not found: {STATE_FILE}")
    rows = []
    with open(STATE_FILE) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln.startswith("sweep|") or "status=COMPLETE" not in ln:
                continue
            parts = ln.split("|")
            label = parts[1]
            m = re.search(r"run=([0-9a-f]+)", ln)
            if not m:
                continue
            rows.append({"label": label, "prefix": m.group(1)})
    # dedup by label, keep last (most recent re-run)
    seen = {}
    for r in rows:
        seen[r["label"]] = r
    ordered = list(seen.values())  # insertion order preserved => sweep order
    # resolve prefix -> full run_id + filename
    for r in ordered:
        res = pg(
            "SELECT pr.id, d.filename FROM ingest.pipeline_runs pr "
            "JOIN ingest.documents d ON d.id = pr.document_id "
            f"WHERE pr.id::text LIKE '{r['prefix']}%' "
            "ORDER BY pr.started_at DESC LIMIT 1;"
        ).strip()
        if not res:
            r["run_id"] = None
            r["doc_filename"] = None
            continue
        rid, fname = res.split("\t", 1)
        r["run_id"] = rid
        r["doc_filename"] = fname
    return ordered


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load + dedup ground truth
# ─────────────────────────────────────────────────────────────────────────────
def load_gt() -> pd.DataFrame:
    paths = sorted(glob.glob("reports/*/bakeoff_dataset.parquet"))
    if not paths:
        sys.exit("ERROR: no reports/*/bakeoff_dataset.parquet found (run from repo root).")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df = df.drop_duplicates(subset=["run_id", "pass_name", "chunk_index"], keep="last")
    df["used"] = pd.to_numeric(df["used"], errors="coerce").fillna(0).astype(int)
    df["chunk_text"] = df["chunk_text"].astype(str)
    df["nt"] = df["chunk_text"].map(norm)
    return df[df["pass_name"].isin(ROUTABLE)].copy()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Live selection for one run
# ─────────────────────────────────────────────────────────────────────────────
def live_selection(run_id: str) -> tuple[dict, dict, dict]:
    """Returns:
      chunks:   {chunk_index: {"text": str, "source_refs": list}}
      cosine:   {(pass_name, chunk_index): max_field_cosine}
      keep:     {(pass_name, chunk_index): bool}  -- absolute_union keep
    """
    rows = arcade(
        "SELECT chunk_index, chunk_text, source_refs FROM ExtractionChunk "
        f'WHERE pipeline_run_id = "{run_id}"'
    )
    chunks = {}
    for r in rows:
        srefs = r.get("source_refs")
        if isinstance(srefs, str):
            srefs = [srefs]
        chunks[int(r["chunk_index"])] = {
            "text": str(r.get("chunk_text") or ""),
            "source_refs": list(srefs or []),
        }

    # live per-(pass,chunk) cosine from router score_components_all
    cosine: dict[tuple[str, int], float] = {}
    res = pg(
        "SELECT pass_name, diagnostics_json->'router'->'score_components_all' "
        "FROM ingest.pipeline_pass_outputs "
        f"WHERE pipeline_run_id = '{run_id}' "
        "AND jsonb_typeof(diagnostics_json->'router'->'score_components_all') = 'array';"
    )
    for ln in res.splitlines():
        if "\t" not in ln:
            continue
        pass_name, blob = ln.split("\t", 1)
        if pass_name not in ROUTABLE or not blob.strip():
            continue
        try:
            comps = json.loads(blob)
        except json.JSONDecodeError:
            continue
        for e in comps:
            ck = e.get("candidate_key", "")
            m = re.search(r"chunk_(\d+)$", ck)
            if not m:
                continue
            idx = int(m.group(1))
            mfc = e.get("max_field_cosine")
            if mfc is not None:
                cosine[(pass_name, idx)] = float(mfc)

    # reconstruct keep-rule for every (routable pass, live chunk)
    keep: dict[tuple[str, int], bool] = {}
    for pass_name in ROUTABLE:
        c = CFG[pass_name]
        for idx, ch in chunks.items():
            txt = ch["text"]
            m = measurement_present(c.dimensions, txt)
            cat = categorical_present(c.categorical_fields, txt)
            img = bool(c.has_image_field and image_present(ch["source_refs"]))
            cos = cosine.get((pass_name, idx), 0.0) >= TAU
            keep[(pass_name, idx)] = bool(m or cat or img or cos)
    return chunks, cosine, keep


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-doc metrics: join GT (by normalized text) to live selection
# ─────────────────────────────────────────────────────────────────────────────
def doc_metrics(gt_all: pd.DataFrame, run: dict) -> dict:
    fname = run["doc_filename"]
    gt = gt_all[gt_all["doc_filename"] == fname].copy()

    result = {
        "label": run["label"],
        "doc": fname,
        "run_id": run["run_id"],
        "note": "",
        "gt_cells": 0, "used_cells": 0, "gt_positives_pooled": 0,
        "text_match_cells": 0, "text_match_rate": float("nan"),
        "recall": float("nan"), "precision": float("nan"),
        "frac_sel": float("nan"), "f1": float("nan"),
        "selected": 0, "tp": 0,
        "unmatched_positives": 0,
        "cosine_pairs": 0,
    }

    if len(gt) == 0:
        result["note"] = "no GT"
        return result

    chunks, cosine, keep = live_selection(run["run_id"])
    result["cosine_pairs"] = len(cosine)
    if not chunks:
        result["note"] = "no live chunks"
        return result

    # Map normalized live text -> set of live chunk_index (text may repeat across idx)
    live_text_to_idx: dict[str, list[int]] = {}
    for idx, ch in chunks.items():
        live_text_to_idx.setdefault(norm(ch["text"]), []).append(idx)

    # Pool GT cells across all GT runs for this doc, dedup on (pass_name, normalized text).
    # 'used' = max over duplicates (a cell is positive if ANY GT run marked it used).
    gt["used"] = gt["used"].astype(int)
    pooled = (gt.groupby(["pass_name", "nt"], as_index=False)["used"].max())

    result["gt_cells"] = len(pooled)
    result["gt_positives_pooled"] = int((pooled["used"] == 1).sum())

    # join each pooled GT cell to a live chunk by normalized text
    joined = []        # list of (pass_name, used, live_keep)
    matched = 0
    unmatched_pos = 0
    for _, row in pooled.iterrows():
        pass_name, nt, used = row["pass_name"], row["nt"], int(row["used"])
        idxs = live_text_to_idx.get(nt)
        if not idxs:
            if used == 1:
                unmatched_pos += 1
            continue
        matched += 1
        # if text maps to multiple live chunks, a cell is 'selected' if kept for ANY
        sel = any(keep.get((pass_name, i), False) for i in idxs)
        joined.append((pass_name, used, sel))

    result["text_match_cells"] = matched
    result["text_match_rate"] = matched / len(pooled) if len(pooled) else float("nan")
    result["unmatched_positives"] = unmatched_pos

    if not joined:
        result["note"] = "0 GT cells text-aligned to live run"
        return result

    jdf = pd.DataFrame(joined, columns=["pass_name", "used", "sel"])
    sel_mask = jdf["sel"]
    used_mask = jdf["used"] == 1
    selected = int(sel_mask.sum())
    pos = int(used_mask.sum())
    tp = int((sel_mask & used_mask).sum())
    total = len(jdf)

    result["selected"] = selected
    result["tp"] = tp
    # used_cells in the main table = SCOREABLE (text-aligned) positives, so it
    # lines up with recall. Drift positives are surfaced separately in the note.
    result["used_cells"] = pos
    result["recall"] = (tp / pos) if pos else float("nan")
    result["precision"] = (tp / selected) if selected else float("nan")
    result["frac_sel"] = (selected / total) if total else float("nan")
    rec, prec = result["recall"], result["precision"]
    if pos and selected and (prec + rec) > 0:
        result["f1"] = 2 * prec * rec / (prec + rec)
    else:
        result["f1"] = float("nan")

    # build the note
    notes = []
    if result["text_match_rate"] < 0.999:
        notes.append(f"text-align {result['text_match_rate']:.0%}")
    if unmatched_pos:
        notes.append(f"{unmatched_pos} pos not in live (re-chunk drift)")
    if pos == 0:
        notes.append("0 used-cells (recall N/A)")
    result["note"] = "; ".join(notes)
    # carry joined frame for cumulative pooling
    result["_joined"] = jdf
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Format + write report
# ─────────────────────────────────────────────────────────────────────────────
def fmt(v) -> str:
    if v is None or (isinstance(v, float) and v != v):  # NaN
        return "  N/A  "
    return f"{v:.3f}"


def main() -> None:
    runs = parse_completed_runs()
    if not runs:
        sys.exit("No COMPLETE sweep runs found in state file yet.")
    gt_all = load_gt()

    per_doc = []
    cumulative_joined = []
    for run in runs:
        if not run["run_id"]:
            per_doc.append({"label": run["label"], "doc": f"<unresolved prefix {run['prefix']}>",
                            "note": "run prefix not in postgres",
                            "gt_cells": 0, "used_cells": 0, "text_match_rate": float("nan"),
                            "recall": float("nan"), "precision": float("nan"),
                            "frac_sel": float("nan"), "f1": float("nan")})
            continue
        m = doc_metrics(gt_all, run)
        per_doc.append(m)
        if "_joined" in m:
            cumulative_joined.append(m["_joined"])

    # cumulative over all text-aligned GT cells across completed docs
    cum = {"used_cells": 0, "recall": float("nan"), "precision": float("nan"),
           "frac_sel": float("nan"), "f1": float("nan"), "selected": 0, "tp": 0, "total": 0}
    if cumulative_joined:
        allj = pd.concat(cumulative_joined, ignore_index=True)
        sel_mask = allj["sel"]
        used_mask = allj["used"] == 1
        sel = int(sel_mask.sum()); pos = int(used_mask.sum()); tp = int((sel_mask & used_mask).sum())
        tot = len(allj)
        cum["used_cells"] = pos
        cum["selected"] = sel
        cum["tp"] = tp
        cum["total"] = tot
        cum["recall"] = (tp / pos) if pos else float("nan")
        cum["precision"] = (tp / sel) if sel else float("nan")
        cum["frac_sel"] = (sel / tot) if tot else float("nan")
        if pos and sel and (cum["precision"] + cum["recall"]) > 0:
            cum["f1"] = 2 * cum["precision"] * cum["recall"] / (cum["precision"] + cum["recall"])

    # ── render ──
    lines = []
    def w(*a): lines.append(" ".join(str(x) for x in a))

    SEP = "=" * 118
    sub = "-" * 118
    w(SEP)
    w("ABSOLUTE_UNION — PER-DOC + CUMULATIVE LIVE-RUN-vs-GT SELECTION METRICS")
    w(f"  source = live-run reconstructed selection vs bake-off GT  |  τ = {TAU}")
    w(f"  generated-from = {sum(1 for r in per_doc if r.get('run_id'))} completed sweep runs "
      f"(state file: {STATE_FILE})")
    w("  selection = measurement OR categorical OR (image AND has_image_field) OR (live max_field_cosine >= τ)")
    w("  JOIN = normalized chunk_text (pooled across GT runs); live re-chunks images, so chunk_index is NOT stable.")
    w("  used_cells = GT-positive cells that text-align to a live chunk (scoreable). "
      "Positives present only in stale GT chunkings are reported as 're-chunk drift'.")
    w(SEP)
    w()
    hdr = (f"{'#':>2} {'doc':<42} {'used':>5} {'recall':>7} {'prec':>7} "
           f"{'fsel':>7} {'F1':>7}  note")
    w(hdr)
    w(sub)
    for i, m in enumerate(per_doc, 1):
        doc = m["doc"] or "?"
        doc_disp = (doc[:39] + "...") if len(doc) > 42 else doc
        w(f"{i:>2} {doc_disp:<42} {m['used_cells']:>5} "
          f"{fmt(m['recall']):>7} {fmt(m['precision']):>7} "
          f"{fmt(m['frac_sel']):>7} {fmt(m['f1']):>7}  {m['note']}")
    w(sub)
    w(f"{'':>2} {'CUMULATIVE (all text-aligned GT cells)':<42} {cum['used_cells']:>5} "
      f"{fmt(cum['recall']):>7} {fmt(cum['precision']):>7} "
      f"{fmt(cum['frac_sel']):>7} {fmt(cum['f1']):>7}  "
      f"tp={cum['tp']} selected={cum['selected']} cells={cum['total']}")
    w(SEP)
    w()

    # alignment-detail block
    w("ALIGNMENT DETAIL (chunk_text join; per completed doc)")
    w(sub)
    w(f"{'doc':<42} {'gt_cells':>9} {'aligned':>8} {'align%':>7} "
      f"{'gt_pos':>7} {'pos_align':>10} {'pos_drift':>10} {'cos_pairs':>10}")
    w(sub)
    for m in per_doc:
        if not m.get("run_id"):
            continue
        doc = m["doc"]; doc_disp = (doc[:39] + "...") if len(doc) > 42 else doc
        ar = m.get("text_match_rate")
        w(f"{doc_disp:<42} {m.get('gt_cells',0):>9} {m.get('text_match_cells',0):>8} "
          f"{(f'{ar:.0%}' if ar==ar else 'N/A'):>7} {m.get('gt_positives_pooled',0):>7} "
          f"{m.get('used_cells',0):>10} "
          f"{m.get('unmatched_positives',0):>10} {m.get('cosine_pairs',0):>10}")
    w(sub)
    w()

    # low-recall flags
    w("SIGNAL-COVERAGE FLAGS (recall < 0.75 over used_cells >= 3)")
    w(sub)
    flagged = [m for m in per_doc
               if m.get("recall") == m.get("recall")  # not NaN
               and m.get("used_cells", 0) >= 3
               and m["recall"] < 0.75]
    if flagged:
        for m in flagged:
            w(f"  {m['doc']:<55} recall={m['recall']:.3f} used_cells={m['used_cells']}")
    else:
        w("  None over a non-trivial denominator (used_cells >= 3). "
          "Most completed docs so far are sparse/zero-GT; dense docs pending.")
    w(sub)
    w()
    w(SEP)
    w("END OF REPORT")
    w(SEP)

    text = "\n".join(lines)
    print(text)
    with open(REPORT_PATH, "w") as fh:
        fh.write(text + "\n")
    print(f"\nReport written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
