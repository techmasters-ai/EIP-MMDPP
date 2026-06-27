#!/usr/bin/env bash
# Phase-3 re-collection SUPERVISOR (guarded-ranker Task 17).
#
# Session-independent safety net for the 8-doc re-collection driver
# (/tmp/recollect_v2.sh, launched detached). Designed to run from cron every
# 10 minutes. It:
#   1. syncs the driver's /tmp state to a persistent dir (survives /tmp wipes),
#   2. exits quietly while the driver is alive or a pipeline run is PROCESSING,
#   3. relaunches a RESUME driver for the remaining docs if the driver died,
#   4. self-quiesces once all 8 docs reach a terminal state.
#
# Remove the crontab entry after Task 17b (evaluation) completes.
set -u

STATE=/home/josh/.recollect_v2_state
TMP_RUNS=/tmp/recollect_v2_runs.txt
TMP_LOG=/tmp/recollect_v2.log
API=http://localhost:8005
BUNDLE=air_defense_v3
SLOG="$STATE/supervisor.log"

mkdir -p "$STATE"
slog() { echo "$(date -u +%FT%TZ) $*" >> "$SLOG"; }

# --- 1. sync /tmp state to persistent dir (newest wins, never truncate) -----
for f in "$TMP_RUNS" "$TMP_LOG"; do
  base=$(basename "$f")
  if [ -s "$f" ] && [ "$(wc -c < "$f")" -ge "$( (wc -c < "$STATE/$base") 2>/dev/null || echo 0)" ]; then
    cp "$f" "$STATE/$base"
  fi
done
RUNS="$STATE/recollect_v2_runs.txt"
touch "$RUNS"

# --- 2. all done? ------------------------------------------------------------
DOCS=(
  1915cd62-dc8e-45b9-9a11-2a3b8ff898af
  10b39886-5322-4191-bb7c-96a57589dd60
  f365f1cc-f717-4d29-906c-c7fc00165d83
  29fed6ff-9659-4e76-a4c2-79589b97bbcd
  9c8e09c7-e39f-4359-92c0-46330158c73c
  128e48f9-06f9-459a-b2f0-6d42bf62c42d
  dbb4edee-5f33-493e-8372-8c8890b9ff7e
  ddaa9e36-2854-47c3-bc94-ff38d531dafd
)
remaining=()
for doc in "${DOCS[@]}"; do
  grep -q "doc=$doc run=.* terminal=" "$RUNS" || remaining+=("$doc")
done
if [ "${#remaining[@]}" -eq 0 ]; then
  # quiesce: nothing to do (leave the cron entry for manual removal)
  exit 0
fi

# --- 3. driver alive? --------------------------------------------------------
if pgrep -f "bash /tmp/recollect_v2(_resume)?.sh" > /dev/null 2>&1; then
  exit 0
fi

# --- 4. a run still PROCESSING? (driver may have died mid-poll; the run
#         itself lives in the workers — wait for it, don't double-launch) -----
np=$(docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -c \
  "SELECT count(*) FROM ingest.pipeline_runs WHERE status='PROCESSING';" 2>/dev/null | tr -d '[:space:]')
if [ -z "$np" ]; then slog "postgres unreachable; will retry next tick"; exit 0; fi
if [ "$np" != "0" ]; then
  # Record terminal state for any launched-but-unrecorded run before waiting.
  slog "driver dead but $np run(s) PROCESSING; waiting"
  exit 0
fi

# Reconcile: a doc may have been launched and even completed while untracked.
for doc in "${remaining[@]}"; do
  line=$(grep "doc=$doc run=" "$RUNS" | tail -1 || true)
  run=$(printf '%s' "$line" | sed -n 's/.*run=\([0-9a-f-]*\).*/\1/p')
  if [ -n "$run" ] && [ "$run" != "LAUNCH_FAILED" ]; then
    st=$(docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -c \
      "SELECT status FROM ingest.pipeline_runs WHERE id='$run';" 2>/dev/null | tr -d '[:space:]')
    if [ "$st" = "COMPLETE" ] || [ "$st" = "FAILED" ]; then
      echo "doc=$doc run=$run terminal=$st at=$(date -u +%FT%TZ) (reconciled)" >> "$RUNS"
      cp "$RUNS" "$TMP_RUNS" 2>/dev/null || true
      slog "reconciled doc=$doc run=$run terminal=$st"
    fi
  fi
done
# Re-derive remaining after reconciliation.
remaining=()
for doc in "${DOCS[@]}"; do
  grep -q "doc=$doc run=.* terminal=" "$RUNS" || remaining+=("$doc")
done
[ "${#remaining[@]}" -eq 0 ] && exit 0

# --- 5. relaunch a resume driver for the remaining docs ----------------------
slog "driver dead, pool idle, ${#remaining[@]} doc(s) remaining — relaunching: ${remaining[*]}"
RESUME=/tmp/recollect_v2_resume.sh
{
  echo '#!/usr/bin/env bash'
  echo '# AUTO-GENERATED resume driver (recollect_v2_supervisor). Same protocol as /tmp/recollect_v2.sh.'
  echo 'set -u'
  echo "API=$API; BUNDLE=$BUNDLE"
  echo "LOG=$TMP_LOG; RUNS=$TMP_RUNS; PRUNS=$RUNS"
  echo 'log() { echo "$(date -u +%FT%TZ) [resume] $*" >> "$LOG"; }'
  echo 'run_status() { docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -c "SELECT status FROM ingest.pipeline_runs WHERE id='"'"'$1'"'"';" 2>/dev/null | tr -d "[:space:]"; }'
  echo 'n_processing() { docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -c "SELECT count(*) FROM ingest.pipeline_runs WHERE status='"'"'PROCESSING'"'"';" 2>/dev/null | tr -d "[:space:]"; }'
  echo 'poll_terminal() { local run="$1" st="" i; for i in $(seq 1 480); do sleep 180; st=$(run_status "$run"); case "$st" in COMPLETE|FAILED) echo "$st"; return 0;; esac; done; echo "${st:-TIMEOUT}"; return 1; }'
  echo 'wait_idle() { local i np; for i in $(seq 1 480); do np=$(n_processing); if [ "$np" = "0" ]; then sleep 120; return 0; fi; sleep 180; done; return 1; }'
  echo "DOCS=(${remaining[*]})"
  echo 'log "=== RESUME start pid=$$ docs=${#DOCS[@]} ==="'
  echo 'for doc in "${DOCS[@]}"; do'
  echo '  log "--- resume doc $doc ---"'
  echo '  wait_idle || { log "pool never idle; ABORT"; break; }'
  echo '  resp=$(curl -s --max-time 25 -X POST "$API/v1/documents/$doc/reingest" -H "Content-Type: application/json" -d "{\"mode\":\"graph_only\",\"ontology_bundle_key\":\"$BUNDLE\"}")'
  echo '  run=$(printf "%s" "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get(\"pipeline_run_id\") or \"\")" 2>/dev/null || true)'
  echo '  log "launch doc=$doc -> run=${run:-NONE}"'
  echo '  if [ -z "$run" ]; then echo "doc=$doc run=LAUNCH_FAILED" | tee -a "$RUNS" >> "$PRUNS"; continue; fi'
  echo '  echo "doc=$doc run=$run launched=$(date -u +%FT%TZ)" | tee -a "$RUNS" >> "$PRUNS"'
  echo '  st=$(poll_terminal "$run"); log "doc=$doc run=$run terminal=$st"'
  echo '  echo "doc=$doc run=$run terminal=$st at=$(date -u +%FT%TZ)" | tee -a "$RUNS" >> "$PRUNS"'
  echo 'done'
  echo 'log "=== RESUME done ==="'
} > "$RESUME"
chmod +x "$RESUME"
setsid nohup bash "$RESUME" >> "$STATE/resume.out" 2>&1 < /dev/null &
slog "resume driver launched pid=$!"
