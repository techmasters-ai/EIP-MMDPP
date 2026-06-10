#!/usr/bin/env bash
# Session-proof AUTO-RE-BASELINE: waits for the notebooks ingest to finish, then
# re-derives the value-grounded bake-off over the EXPANDED corpus (existing 7
# curated SA-2-family runs + the newly-ingested notebooks docs that captured
# score_components) and writes the refreshed frontier.
#
#   setsid bash scripts/run_rebaseline_on_ingest.sh </dev/null >/dev/null 2>&1 &
set -u
WT=/home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
export A0_DATABASE_URL="postgresql+psycopg2://eip:eip_secret@localhost:5437/eip"
ING_LOG="$WT/reports/collection/NOTEBOOKS_ingest_driver.log"
ING_PID=1130604
OUT="$WT/reports/collection/REBASELINE_result.txt"
LOG="$WT/reports/collection/REBASELINE_driver.log"
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

# existing curated value-grounded bake-off runs (SA-2 family).
EXISTING="e864ba84-6b3c-41d8-8da3-688cb3034524 28a58eb9-f0f3-4b6b-8b07-f55df6efd5ba 58767f3f-5112-4816-9427-e61d5d8c068b e79a4866-750b-41e3-a4b4-521f3e31ff26 ff35b0e2-6a8a-4444-8a07-ee8e85a49011 295aea8e-bc3f-4003-81da-7713705c5daa de6f44d9-69b8-4d3c-bcb3-f0eaa487af19"

log "=== rebaseline driver start pid=$$ (waiting for ingest done) ==="
for i in $(seq 1 288); do        # up to ~24h at 300s
  grep -q "=== notebooks ingest done ===" "$ING_LOG" 2>/dev/null && { log "ingest done detected"; break; }
  kill -0 "$ING_PID" 2>/dev/null || { log "ingest pid gone"; sleep 30; break; }
  sleep 300
done

# Assemble the expanded run list: existing + each notebooks doc's latest run that
# captured score_components (bake-off-eligible).
JOINED=$(python3 - "$EXISTING" <<'PY'
import sys
import scripts.a0_captured_separation as a0
from sqlalchemy import create_engine, text
existing = sys.argv[1].split()
runs = list(existing)
# notebooks docs ingested this round
try:
    nb = [ln.split()[-1] for ln in open(
        "/home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry/reports/collection/NOTEBOOKS_runs.txt")
        if ln.strip()]
except FileNotFoundError:
    nb = []
eng = create_engine(a0._pg_url())
for doc in nb:
    with eng.connect() as c:
        rid = c.execute(text("SELECT id FROM ingest.pipeline_runs WHERE document_id=:d "
                             "ORDER BY started_at DESC LIMIT 1"), {"d": doc}).scalar()
    if not rid:
        continue
    rid = str(rid)
    try:
        comps = a0.fetch_captured_components(rid)
    except Exception:
        comps = {}
    if comps:                      # only runs with captured features can contribute
        runs.append(rid)
print(",".join(dict.fromkeys(runs)))   # dedup, preserve order
PY
)
log "expanded run set: $JOINED"

{
  echo "=== REBASELINE over expanded corpus ($(date -u +%FT%TZ)) ==="
  echo "runs: $JOINED"
  echo ""
  echo "### value-grounded bake-off (LODO) ###"
  python3 -m scripts.a0_captured_separation --fit-runs "$JOINED" --target lineage_grounded \
    --out-dir "$WT/reports/collection" 2>&1 | grep -vE "Convergence|warnings.warn" | grep -E "USED|AUROC|Logistic|RandomForest|HistGrad|Gradient|MLP|best model"
  echo ""
  echo "### frontier, ALL features ###"
  python3 -m scripts.plot_recall_vs_savings --runs "$JOINED" --target lineage_grounded \
    --out "$WT/reports/collection/recall_vs_savings_expanded.png" 2>&1 | grep -vE "Convergence|warnings.warn" | grep -E "positives|recall ≥"
  echo ""
  echo "### frontier, negative_norm dropped ###"
  python3 -m scripts.plot_recall_vs_savings --runs "$JOINED" --target lineage_grounded --drop-features negative_norm \
    --out "$WT/reports/collection/recall_vs_savings_expanded_noneg.png" 2>&1 | grep -vE "Convergence|warnings.warn" | grep -E "positives|recall ≥"
} > "$OUT" 2>&1
log "=== rebaseline done → $OUT ==="
