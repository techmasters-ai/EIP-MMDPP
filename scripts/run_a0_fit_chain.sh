#!/usr/bin/env bash
# Unattended, session-proof A0 fit-chain driver.
#
#   setsid bash scripts/run_a0_fit_chain.sh <SA2_RUN_ID> <FIT_DOC_1> <FIT_DOC_2> ... </dev/null >/dev/null 2>&1 &
#
# Owns the whole calibration data-collection + fit pipeline:
#   1. wait for the in-flight SA-2 run to reach terminal, run its analysis
#   2. for each fit doc (SEQUENTIAL, idle pool only — avoids the LLM-timeout
#      contention stall failure mode): wait for an idle pool, POST a graph_only
#      reingest (shadow mode is live in .env → full-doc + captured components),
#      poll to terminal, run its per-run analysis
#   3. FIT the model on the pooled fit runs (SA-2 + the fit docs)
#
# All component + USED data is durable in postgres/arcadedb as each run
# commits, independent of this driver. Survives session close (setsid).
set -u
WT=/home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
API=http://localhost:8005
SA2_RUN="${1:?SA-2 run id required}"
shift
FIT_DOCS=("$@")
LOG="$WT/reports/collection/A0_chain_driver.log"
STATUS="$WT/reports/collection/A0_CHAIN_STATUS.txt"
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

run_status() {
  docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -c \
    "SELECT status FROM ingest.pipeline_runs WHERE id='$1';" 2>/dev/null | tr -d '[:space:]'
}
n_processing() {
  docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -c \
    "SELECT count(*) FROM ingest.pipeline_runs WHERE status='PROCESSING';" 2>/dev/null | tr -d '[:space:]'
}
poll_terminal() {  # $1=run; echoes terminal status (COMPLETE/FAILED/TIMEOUT)
  local run="$1" st="" i
  for i in $(seq 1 340); do          # up to ~17h
    sleep 180
    st=$(run_status "$run")
    case "$st" in COMPLETE | FAILED) echo "$st"; return 0 ;; esac
  done
  echo "${st:-TIMEOUT}"; return 1
}
analyze() {  # $1=run
  if python3 -m scripts.a0_captured_separation --run-id "$1" --out-dir "$WT/reports/collection" >> "$LOG" 2>&1; then
    log "analyze OK run=$1 → a0_separation_${1:0:8}.md"
  else
    log "analyze FAILED run=$1 (data still durable in db)"
  fi
}
wait_idle() {  # block until no PROCESSING run, then settle
  local i np
  for i in $(seq 1 340); do
    np=$(n_processing)
    if [ "$np" = "0" ]; then log "pool idle (settling 120s before launch)"; sleep 120; return 0; fi
    sleep 180
  done
  return 1
}
launch() {  # $1=doc; echoes new run id ('' on failure)
  local doc="$1" resp run
  resp=$(curl -s --max-time 25 -X POST "$API/v1/documents/$doc/reingest" \
    -H 'Content-Type: application/json' -d '{"mode":"graph_only"}')
  run=$(printf '%s' "$resp" | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print(d.get('pipeline_run_id') or '')" 2>/dev/null || true)
  log "launch doc=$doc → run=${run:-NONE} resp=${resp:0:200}"
  printf '%s' "$run"
}

log "=== A0 fit-chain start pid=$$ sa2=$SA2_RUN docs=${FIT_DOCS[*]} ==="
ALL_RUNS=("$SA2_RUN")

# 1) SA-2
st=$(run_status "$SA2_RUN")
case "$st" in
  COMPLETE | FAILED) log "SA-2 already $st" ;;
  *) log "waiting on SA-2 ($SA2_RUN) ..."; st=$(poll_terminal "$SA2_RUN"); log "SA-2 reached $st" ;;
esac
analyze "$SA2_RUN"

# 2) fit docs, sequential, idle-pool gated
for doc in "${FIT_DOCS[@]}"; do
  log "--- fit doc $doc ---"
  if ! wait_idle; then log "pool never went idle; aborting remaining chain"; break; fi
  run=$(launch "$doc")
  if [ -z "$run" ]; then log "launch FAILED for $doc; skipping"; continue; fi
  ALL_RUNS+=("$run")
  st=$(poll_terminal "$run"); log "doc $doc run $run reached $st"
  analyze "$run"
done

# 3) fit the model on the pooled fit runs
JOINED=$(IFS=,; echo "${ALL_RUNS[*]}")
log "=== FIT model on runs: $JOINED ==="
if python3 -m scripts.a0_captured_separation --fit-runs "$JOINED" --out-dir "$WT/reports/collection" >> "$LOG" 2>&1; then
  log "FIT OK"
else
  log "FIT FAILED"
fi

{
  echo "A0 fit-chain finished $(date -u +%FT%TZ)"
  echo "fit runs: $JOINED"
  echo "per-run reports : reports/collection/a0_separation_<run8>.md"
  echo "fitted model    : reports/collection/a0_fitvalidate_${SA2_RUN:0:8}.json (+ _rows_*.csv)"
  echo "NOTE: VECTOR_ROUTER_MODE is still 'shadow' in .env — revert to narrow_only after calibration."
} > "$STATUS"
log "=== chain done ==="
