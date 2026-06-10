#!/usr/bin/env bash
# Session-proof PROBE + COLLECT (concurrency 8, no service restart).
#
#   setsid bash scripts/run_probe_collect.sh </dev/null >/dev/null 2>&1 &
#
# Runs the 3 unsurveyable docs (PROBE: museum, Images_Demo, SR-71 PDF) + the
# value-dense docs (Dvina, SA-2) graph_only on the DEPLOYED lineage fix + shadow.
# Per doc: --target lineage separation (USED = field-group positive cells) +
# value-grounding health. Does NOT auto-run the final bake-off — the FINAL doc
# set is curated by hand from the confirmed per-doc yields (drop zero-yield).
# Already clean at conc-8 and reused in the final bake-off: SNR-75 e79a4866,
# V-75 295aea8e. Sequential / idle-pool. Survives session close.
set -u
WT=/home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
API=http://localhost:8005
BUNDLE=air_defense_v3
A0_DATABASE_URL="postgresql+psycopg2://eip:eip_secret@localhost:5437/eip"; export A0_DATABASE_URL

# probe-small-first, SA-2 (the ~5h prize) LAST so probes confirm early.
DOCS=(
  1915cd62-dc8e-45b9-9a11-2a3b8ff898af  # SA-2 museum display   (PROBE, 14ch)
  29fed6ff-9659-4e76-a4c2-79589b97bbcd  # Images_Demo_Doc       (PROBE, 25ch)
  128e48f9-06f9-459a-b2f0-6d42bf62c42d  # SA-2_and_SR-71 PDF    (PROBE + SR-71 rep, 64ch)
  9c8e09c7-e39f-4359-92c0-46330158c73c  # S-75 Dvina            (value-dense, 32ch)
  ddaa9e36-2854-47c3-bc94-ff38d531dafd  # SA-2 Guideline RU     (value-dense PRIZE, 102ch, ~5h)
)
LOG="$WT/reports/collection/PROBE_collect_driver.log"
STATUS="$WT/reports/collection/PROBE_COLLECT_STATUS.txt"
YIELD="$WT/reports/collection/PROBE_yields.txt"
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

run_status() { docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -c \
  "SELECT status FROM ingest.pipeline_runs WHERE id='$1';" 2>/dev/null | tr -d '[:space:]'; }
n_processing() { docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -c \
  "SELECT count(*) FROM ingest.pipeline_runs WHERE status='PROCESSING';" 2>/dev/null | tr -d '[:space:]'; }
poll_terminal() { local run="$1" st="" i; for i in $(seq 1 480); do sleep 180; st=$(run_status "$run");
  case "$st" in COMPLETE|FAILED) echo "$st"; return 0;; esac; done; echo "${st:-TIMEOUT}"; return 1; }
wait_idle() { local i np; for i in $(seq 1 480); do np=$(n_processing);
  if [ "$np" = "0" ]; then sleep 120; return 0; fi; sleep 180; done; return 1; }
launch() { local doc="$1" resp run;
  resp=$(curl -s --max-time 25 -X POST "$API/v1/documents/$doc/reingest" -H 'Content-Type: application/json' \
    -d "{\"mode\":\"graph_only\",\"ontology_bundle_key\":\"$BUNDLE\"}");
  run=$(printf '%s' "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('pipeline_run_id') or '')" 2>/dev/null || true);
  log "launch doc=$doc → run=${run:-NONE}"; printf '%s' "$run"; }
analyze() {  # $1=run $2=doc → logs field-group positive count (USED) + grounding
  python3 -m scripts.a0_captured_separation --run-id "$1" --target lineage --out-dir "$WT/reports/collection" >> "$LOG" 2>&1 || true
  used=$(grep -m1 "labeled rows:" "$WT/reports/collection/a0_separation_${1:0:8}.md" 2>/dev/null || echo "?")
  python3 -m scripts.verify_field_chunk_lineage --run "$1" >> "$LOG" 2>&1 || true
  echo "doc=$2 run=${1:0:8}  $used" >> "$YIELD"
  log "yield doc=$2 run=$1 :: $used"
}

log "=== PROBE+COLLECT start pid=$$ docs=${#DOCS[@]} (conc 8) ==="
: > "$YIELD"
i=0
for doc in "${DOCS[@]}"; do
  i=$((i+1)); log "--- [$i/${#DOCS[@]}] doc $doc ---"
  echo "running [$i/${#DOCS[@]}] doc=$doc ($(date -u +%FT%TZ))" > "$STATUS"
  if ! wait_idle; then log "pool never idle; abort"; break; fi
  run=$(launch "$doc"); [ -z "$run" ] && { log "launch FAILED $doc"; continue; }
  st=$(poll_terminal "$run"); log "doc $doc run $run reached $st"
  [ "$st" = "COMPLETE" ] && analyze "$run" "$doc" || log "skip analyze (status=$st)"
done
{ echo "PROBE+COLLECT finished $(date -u +%FT%TZ)"; echo "per-doc field-group yields:"; cat "$YIELD";
  echo ""; echo "reused clean conc-8 runs: SNR-75 e79a4866, V-75 295aea8e";
  echo "NEXT: curate final bake-off set from yields (drop zero/low), run a0_captured_separation --fit-runs <set> --target lineage"; } > "$STATUS"
log "=== probe+collect done ==="
