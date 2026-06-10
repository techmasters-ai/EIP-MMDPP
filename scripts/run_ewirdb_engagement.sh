#!/usr/bin/env bash
# Session-proof: retry EWIRDB (graph_only on cached parse, idle pool) → ingest
# Engagement and Fire Control Radars (full, 29MB) → auto re-baseline the
# value-grounded bake-off over the expanded corpus. Skips chinese_research_paper.
#
#   setsid bash scripts/run_ewirdb_engagement.sh </dev/null >/dev/null 2>&1 &
set -u
REPO=/home/josh/development/EIP-MMDPP
WT=$REPO/.worktrees/walltime-c0-telemetry
API=http://localhost:8005
SOURCE=ddac23e2-b821-4399-b14a-f2197757fabd
BUNDLE=air_defense_v3
EWIRDB_DOC=a98fdaab-3290-4689-b74c-e938cbdbcb6f
ENG_FILE="$REPO/notebooks/Engagement and Fire Control Radars (S-Band, X-band).pdf"
export A0_DATABASE_URL="postgresql+psycopg2://eip:eip_secret@localhost:5437/eip"
LOG="$WT/reports/collection/EWIRDB_ENG_driver.log"
STATUS="$WT/reports/collection/EWIRDB_ENG_STATUS.txt"
OUT="$WT/reports/collection/REBASELINE_result.txt"
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
n_proc() { docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -tA -c "SELECT count(*) FROM ingest.pipeline_runs WHERE status='PROCESSING';" 2>/dev/null | tr -d '[:space:]'; }
doc_status() { docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -tA -c "SELECT pipeline_status FROM ingest.documents WHERE id='$1';" 2>/dev/null | tr -d '[:space:]'; }
run_status() { docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -tA -c "SELECT status FROM ingest.pipeline_runs WHERE id='$1';" 2>/dev/null | tr -d '[:space:]'; }
wait_idle() { local i; for i in $(seq 1 600); do [ "$(n_proc)" = "0" ] && { sleep 60; return 0; }; sleep 60; done; return 1; }

EXISTING="e864ba84-6b3c-41d8-8da3-688cb3034524 28a58eb9-f0f3-4b6b-8b07-f55df6efd5ba 58767f3f-5112-4816-9427-e61d5d8c068b e79a4866-750b-41e3-a4b4-521f3e31ff26 ff35b0e2-6a8a-4444-8a07-ee8e85a49011 295aea8e-bc3f-4003-81da-7713705c5daa de6f44d9-69b8-4d3c-bcb3-f0eaa487af19"
NEW_DOCS=""

log "=== Engagement ingest + rebaseline start pid=$$ ==="

# --- 1. EWIRDB graph_only retry — SKIPPED ---
# Diagnosis (2026-06-08): EWIRDB_Production.pdf is an EW coordinate-system
# standards paper, NOT a radar/missile spec sheet. radar passes captured only
# the name "AN/SPY-1" (from a page-15 photo caption) with EVERY quantitative
# field null; missile passes legitimately found no content and hard-FAILED.
# The "8.6M" was embedded base64 page/picture images in the JSON, not a
# markdown explosion (real markdown = 84K chars). NOT a timeout, NOT an ingest
# bug. A retry would reproduce the same empty result. Skip it entirely.
log "EWIRDB graph_only retry SKIPPED — confirmed content-empty (EW standards paper), not a pipeline bug"

# --- 2. Engagement and Fire Control Radars (full ingest, 29MB) ---
echo "ingest Engagement radars ($(date -u +%FT%TZ))" > "$STATUS"
# Reuse an already-uploaded Engagement doc for this source (idempotent — avoids
# a duplicate upload if the driver is re-launched).
edoc=$(docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -tA -c "SELECT id FROM ingest.documents WHERE source_id='$SOURCE' AND filename ILIKE '%Engagement%' ORDER BY created_at DESC LIMIT 1;" 2>/dev/null | tr -d '[:space:]')
if [ -n "$edoc" ]; then
  log "Engagement doc already present → reuse doc=$edoc (no re-upload)"
elif [ -f "$ENG_FILE" ]; then
  wait_idle
  # NOTE: the filename contains spaces/comma/parens — curl -F needs the path
  # double-quoted INSIDE the form value or it errors curl:(26) cannot read file.
  resp=$(curl -s --max-time 900 -X POST "$API/v1/sources/$SOURCE/documents" -F "file=@\"$ENG_FILE\";type=application/pdf")
  edoc=$(printf '%s' "$resp" | python3 -c "import sys,json;print(json.load(sys.stdin).get('id') or '')" 2>/dev/null || true)
  log "Engagement upload → doc=${edoc:-NONE} resp=${resp:0:160}"
else
  log "Engagement file MISSING: $ENG_FILE"
fi
if [ -n "$edoc" ]; then
  for i in $(seq 1 720); do sleep 60; st=$(doc_status "$edoc"); case "$st" in COMPLETE|FAILED|ERROR|PARTIAL_COMPLETE) break;; esac; done
  log "Engagement doc $edoc reached ${st:-TIMEOUT}"
  NEW_DOCS="$NEW_DOCS $edoc"
fi

# --- 3. re-baseline over expanded corpus ---
log "assembling expanded run set + bake-off"
JOINED=$(python3 - "$EXISTING" "$NEW_DOCS" <<'PY'
import sys
import scripts.a0_captured_separation as a0
from sqlalchemy import create_engine, text
runs=list(sys.argv[1].split())
eng=create_engine(a0._pg_url())
for doc in sys.argv[2].split():
    with eng.connect() as c:
        rid=c.execute(text("SELECT id FROM ingest.pipeline_runs WHERE document_id=:d ORDER BY started_at DESC LIMIT 1"),{"d":doc}).scalar()
    if not rid: continue
    rid=str(rid)
    try: comps=a0.fetch_captured_components(rid)
    except Exception: comps={}
    if comps: runs.append(rid)
print(",".join(dict.fromkeys(runs)))
PY
)
log "expanded run set: $JOINED"
{
  echo "=== REBASELINE (EWIRDB retry + Engagement) $(date -u +%FT%TZ) ==="
  echo "runs: $JOINED"; echo ""
  echo "### value-grounded bake-off ###"
  python3 -m scripts.a0_captured_separation --fit-runs "$JOINED" --target lineage_grounded --out-dir "$WT/reports/collection" 2>&1 | grep -vE "Convergence|warnings.warn" | grep -E "USED|Logistic|RandomForest|HistGrad|Gradient|MLP|best model"
  echo ""; echo "### frontier (all features) ###"
  python3 -m scripts.plot_recall_vs_savings --runs "$JOINED" --target lineage_grounded --out "$WT/reports/collection/recall_vs_savings_expanded.png" 2>&1 | grep -vE "Convergence|warnings.warn" | grep -E "positives|recall ≥"
  echo ""; echo "### frontier (negative_norm dropped) ###"
  python3 -m scripts.plot_recall_vs_savings --runs "$JOINED" --target lineage_grounded --drop-features negative_norm --out "$WT/reports/collection/recall_vs_savings_expanded_noneg.png" 2>&1 | grep -vE "Convergence|warnings.warn" | grep -E "positives|recall ≥"
} > "$OUT" 2>&1
echo "done $(date -u +%FT%TZ)" > "$STATUS"
log "=== ewirdb_engagement done ==="
