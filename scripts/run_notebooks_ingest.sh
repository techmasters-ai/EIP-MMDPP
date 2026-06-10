#!/usr/bin/env bash
# Session-proof FULL-INGEST of the useful new notebooks/ documents.
#
#   setsid bash scripts/run_notebooks_ingest.sh </dev/null >/dev/null 2>&1 &
#
# Uploads each file to the notebooks_collection source (default bundle
# air_defense_v3) — upload AUTO-DISPATCHES a full ingest (parse→OCR→chunk→embed→
# extract on the DEPLOYED per-field value-grounding builder + shadow capture).
# These add the variety the SA-2-family corpus lacks: a multi-system DB
# (EWIRDB), radar textbooks/spec sheets (where the signature keywords finally
# fire + value-grounding exercises table values), and non-English. Sequential /
# idle-pool. Survives session close (setsid). Data durable in postgres/ArcadeDB.
set -u
REPO=/home/josh/development/EIP-MMDPP
WT=$REPO/.worktrees/walltime-c0-telemetry
API=http://localhost:8005
SOURCE=ddac23e2-b821-4399-b14a-f2197757fabd   # notebooks_collection → air_defense_v3
LOG="$WT/reports/collection/NOTEBOOKS_ingest_driver.log"
STATUS="$WT/reports/collection/NOTEBOOKS_INGEST_STATUS.txt"
RUNS="$WT/reports/collection/NOTEBOOKS_runs.txt"
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

# small → large; 29MB radar spec LAST (heaviest parse/OCR).
FILES=(
  "$REPO/notebooks/Radar Basics.pdf"
  "$REPO/notebooks/radar_textbook_chapter7.pdf"
  "$REPO/notebooks/EWIRDB_Production.pdf"
  "$REPO/notebooks/chinese_research_paper.pdf"
  "$REPO/notebooks/Engagement and Fire Control Radars (S-Band, X-band).pdf"
)

n_processing() { docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -c \
  "SELECT count(*) FROM ingest.pipeline_runs WHERE status='PROCESSING';" 2>/dev/null | tr -d '[:space:]'; }
doc_latest_status() { docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -c \
  "SELECT pipeline_status FROM ingest.documents WHERE id='$1';" 2>/dev/null | tr -d '[:space:]'; }
wait_idle() { local i np; for i in $(seq 1 600); do np=$(n_processing);
  if [ "$np" = "0" ]; then sleep 60; return 0; fi; sleep 60; done; return 1; }
poll_doc() { local doc="$1" st="" i; for i in $(seq 1 720); do sleep 60; st=$(doc_latest_status "$doc");
  case "$st" in COMPLETE|FAILED|ERROR) echo "$st"; return 0;; esac; done; echo "${st:-TIMEOUT}"; return 1; }
upload() { local path="$1" resp doc;
  resp=$(curl -s --max-time 600 -X POST "$API/v1/sources/$SOURCE/documents" -F "file=@$path");
  doc=$(printf '%s' "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id') or '')" 2>/dev/null || true);
  log "upload '$path' → doc=${doc:-NONE} resp=${resp:0:160}"; printf '%s' "$doc"; }

log "=== NOTEBOOKS ingest start pid=$$ source=$SOURCE files=${#FILES[@]} ==="
: > "$RUNS"
i=0
for f in "${FILES[@]}"; do
  i=$((i+1)); base=$(basename "$f")
  log "--- [$i/${#FILES[@]}] $base ---"
  echo "ingesting [$i/${#FILES[@]}] $base ($(date -u +%FT%TZ))" > "$STATUS"
  if [ ! -f "$f" ]; then log "MISSING $f; skip"; continue; fi
  if ! wait_idle; then log "pool never idle; abort"; break; fi
  doc=$(upload "$f"); [ -z "$doc" ] && { log "upload FAILED $base"; continue; }
  echo "$base $doc" >> "$RUNS"
  st=$(poll_doc "$doc"); log "doc $base ($doc) reached $st"
done
{ echo "NOTEBOOKS ingest finished $(date -u +%FT%TZ)"; echo "docs (name id):"; cat "$RUNS";
  echo ""; echo "VERIFY builder value-grounding on a table-bearing doc; then add good ones to the bake-off corpus."; } > "$STATUS"
log "=== notebooks ingest done ==="
