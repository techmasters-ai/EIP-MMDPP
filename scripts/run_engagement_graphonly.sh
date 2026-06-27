#!/usr/bin/env bash
# Session-proof: graph_only re-run of the Engagement & Fire Control Radars doc
# (cached parse) on the post-timeout-bump build (PASS_SOFT_TIME_LIMIT=80h), then
# auto re-baseline the value-grounded bake-off over the expanded corpus.
#
# Triple duty: (1) actually complete Engagement's extraction now that the 8h
# soft-timeout that killed radar_identity is lifted to 80h; (2) E2E-validate the
# timeout bump; (3) E2E-validate the empty-pass fix (off-domain pass -> EMPTY).
#
#   setsid bash scripts/run_engagement_graphonly.sh </dev/null >/dev/null 2>&1 &
set -u
REPO=/home/josh/development/EIP-MMDPP
WT=$REPO/.worktrees/walltime-c0-telemetry
API=http://localhost:8005
BUNDLE=air_defense_v3
ENG_DOC=dbb4edee-5f33-493e-8372-8c8890b9ff7e
export A0_DATABASE_URL="postgresql+psycopg2://eip:eip_secret@localhost:5437/eip"
LOG="$WT/reports/collection/ENG_graphonly_driver.log"
STATUS="$WT/reports/collection/ENG_graphonly_STATUS.txt"
OUT="$WT/reports/collection/ENG_REBASELINE_result.txt"
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
n_proc() { docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -tA -c "SELECT count(*) FROM ingest.pipeline_runs WHERE status='PROCESSING';" 2>/dev/null | tr -d '[:space:]'; }
run_status() { docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -tA -c "SELECT status FROM ingest.pipeline_runs WHERE id='$1';" 2>/dev/null | tr -d '[:space:]'; }
wait_idle() { local i; for i in $(seq 1 600); do [ "$(n_proc)" = "0" ] && { sleep 30; return 0; }; sleep 60; done; return 1; }

EXISTING="e864ba84-6b3c-41d8-8da3-688cb3034524 28a58eb9-f0f3-4b6b-8b07-f55df6efd5ba 58767f3f-5112-4816-9427-e61d5d8c068b e79a4866-750b-41e3-a4b4-521f3e31ff26 ff35b0e2-6a8a-4444-8a07-ee8e85a49011 295aea8e-bc3f-4003-81da-7713705c5daa de6f44d9-69b8-4d3c-bcb3-f0eaa487af19"

log "=== Engagement graph_only re-run start pid=$$ ==="
echo "graph_only Engagement dispatched ($(date -u +%FT%TZ))" > "$STATUS"
wait_idle
resp=$(curl -s --max-time 30 -X POST "$API/v1/documents/$ENG_DOC/reingest" \
  -H 'Content-Type: application/json' \
  -d "{\"mode\":\"graph_only\",\"ontology_bundle_key\":\"$BUNDLE\"}")
run=$(printf '%s' "$resp" | python3 -c "import sys,json;print(json.load(sys.stdin).get('pipeline_run_id') or '')" 2>/dev/null || true)
log "Engagement graph_only -> run=${run:-NONE} resp=${resp:0:200}"
if [ -z "$run" ]; then
  log "FAILED to start graph_only run"; echo "FAILED to start ($(date -u +%FT%TZ))" > "$STATUS"; exit 1
fi
echo "run $run PROCESSING ($(date -u +%FT%TZ))" > "$STATUS"

# monitor up to ~30h (600 * 180s); passes now have an 80h soft ceiling but on an
# idle pool Engagement should land in a handful of hours.
for i in $(seq 1 600); do
  sleep 180
  st=$(run_status "$run")
  case "$st" in COMPLETE|FAILED) break;; esac
done
log "Engagement run $run reached ${st:-TIMEOUT}"
echo "run $run = ${st:-TIMEOUT} ($(date -u +%FT%TZ))" > "$STATUS"

# --- re-baseline over the expanded corpus (existing 7 + Engagement if it captured features) ---
log "assembling expanded run set + bake-off"
JOINED=$(python3 - "$EXISTING" "$ENG_DOC" <<'PY'
import sys
import scripts.a0_captured_separation as a0
from sqlalchemy import create_engine, text
runs = list(sys.argv[1].split())
eng = create_engine(a0._pg_url())
for doc in sys.argv[2].split():
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
    if comps:                       # only add if the run captured score_components
        runs.append(rid)
print(",".join(dict.fromkeys(runs)))
PY
)
log "expanded run set: $JOINED"
{
  echo "=== ENGAGEMENT REBASELINE (graph_only re-run) $(date -u +%FT%TZ) ==="
  echo "runs: $JOINED"; echo ""
  echo "### value-grounded bake-off (LODO) ###"
  python3 -m scripts.a0_captured_separation --fit-runs "$JOINED" --target lineage_grounded \
    --out-dir "$WT/reports/collection" 2>&1 | grep -vE "Convergence|warnings.warn" | grep -E "USED|AUROC|Logistic|RandomForest|HistGrad|Gradient|MLP|best model"
  echo ""; echo "### per-metric univariate signal (expanded corpus) ###"
  python3 -m scripts.per_metric_signal --runs "$JOINED" --target lineage_grounded 2>&1 | grep -vE "Convergence|warnings.warn" | sed -n '1,40p'
  echo ""; echo "### frontier (all features) ###"
  python3 -m scripts.plot_recall_vs_savings --runs "$JOINED" --target lineage_grounded \
    --out "$WT/reports/collection/recall_vs_savings_expanded.png" 2>&1 | grep -vE "Convergence|warnings.warn" | grep -E "positives|recall ≥"
} > "$OUT" 2>&1
echo "done ($(date -u +%FT%TZ))" > "$STATUS"
log "=== Engagement graph_only + rebaseline done -> $OUT ==="
