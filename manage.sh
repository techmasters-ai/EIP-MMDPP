#!/usr/bin/env bash
# ============================================================
# EIP-MMDPP — Project management CLI
# ============================================================
# Usage:
#   ./manage.sh --start          Build and start all services (single worker)
#   ./manage.sh --start-split    Build and start with split workers (ingest/embed/graph)
#   ./manage.sh --stop           Stop all services
#   ./manage.sh --restart        Restart without rebuild
#   ./manage.sh --status         Show status and health checks
#   ./manage.sh --logs [service] Stream logs
#   ./manage.sh --blow-away      Destroy everything
#   ./manage.sh --migrate        Run database migrations
#   ./manage.sh --seed           Seed ontology data
#   ./manage.sh --db-shell       Open psql shell
#   ./manage.sh --worker-status  Show Celery worker status
#   ./manage.sh --test [mode]    Run tests (delegates to scripts/run_tests.sh)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"
ENV_FILE="${ROOT_DIR}/.env"

HEALTH_TIMEOUT=120
HEALTH_INTERVAL=3

# ---------------------------------------------------------------------------
# Colors / output helpers  (matches scripts/run_tests.sh conventions)
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
header()  { echo -e "\n${CYAN}${BOLD}=== $* ===${NC}"; }
divider() { echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"; }

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------
require_cmd() {
  command -v "$1" &>/dev/null || error "'$1' is required but not found in PATH."
}

check_prerequisites() {
  require_cmd docker
  require_cmd curl

  if docker compose version &>/dev/null; then
    COMPOSE="docker compose"
  elif command -v docker-compose &>/dev/null; then
    COMPOSE="docker-compose"
  else
    error "Neither 'docker compose' (plugin) nor 'docker-compose' found."
  fi
}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
validate_env() {
  if [[ ! -f "${ENV_FILE}" ]]; then
    error ".env not found at ${ENV_FILE}. Copy .env.example and configure it first."
  fi

  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
safe_rm_rf() {
  local target="$1"
  if [[ -z "${target}" ]] || [[ "${target}" == "/" ]]; then
    error "safe_rm_rf: refusing to remove '${target}'."
  fi
  if [[ "${target}" != "${ROOT_DIR}"/* ]]; then
    warn "Skipping removal outside repo root: ${target}"
    return 0
  fi
  rm -rf -- "${target}"
}

dc() {
  ${COMPOSE} -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" "$@"
}

wait_for_healthy() {
  local url="$1"
  local label="${2:-service}"
  local timeout="${3:-${HEALTH_TIMEOUT}}"
  local elapsed=0

  echo -n "Waiting for ${label}"
  while [[ ${elapsed} -lt ${timeout} ]]; do
    if curl -sf --max-time 5 "${url}" &>/dev/null; then
      echo -e " ${GREEN}ready${NC}"
      return 0
    fi
    echo -n "."
    sleep "${HEALTH_INTERVAL}"
    elapsed=$((elapsed + HEALTH_INTERVAL))
  done
  echo
  error "Timed out after ${timeout}s waiting for ${label}."
}

# ---------------------------------------------------------------------------
# Service lifecycle commands
# ---------------------------------------------------------------------------
cmd_start() {
  local profile="${1:-}"
  local profile_args=()

  if [[ "${profile}" == "split" ]]; then
    header "Starting EIP-MMDPP stack (split workers)"
    profile_args=(--profile split)
    # Stop single-mode worker to prevent overlap
    info "Stopping single-mode worker (if running)..."
    dc stop worker 2>/dev/null || true
    dc rm -f worker 2>/dev/null || true
    info "Building and starting with split workers (ingest/embed/graph)..."
  else
    header "Starting EIP-MMDPP stack"
    # Stop split-mode workers to prevent overlap
    info "Stopping split-mode workers (if running)..."
    dc --profile split stop worker-ingest worker-embed worker-graph 2>/dev/null || true
    dc --profile split rm -f worker-ingest worker-embed worker-graph 2>/dev/null || true
    info "Building and starting all services..."
  fi

  dc "${profile_args[@]}" up -d --build

  local api_port="${API_PORT:-8000}"
  divider
  wait_for_healthy "http://localhost:${api_port}/v1/health" "API liveness" "${HEALTH_TIMEOUT}"
  wait_for_healthy "http://localhost:${api_port}/v1/health/ready" "API readiness" "${HEALTH_TIMEOUT}"

  # Pre-download ML models (air-gapped: must be available before first query)
  header "Pre-downloading ML models"

  # Download sentence-transformers models inside the API container (skip if cached)
  local text_model_dir
  text_model_dir="models--$(echo "${TEXT_EMBEDDING_MODEL}" | tr '/' '--')"
  local reranker_model_dir
  reranker_model_dir="models--$(echo "${RERANKER_MODEL}" | tr '/' '--')"

  if dc exec -T api python -c "
import os
cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
model_dir = os.path.join(cache_dir, '${text_model_dir}')
exit(0 if os.path.isdir(model_dir) else 1)
" 2>/dev/null; then
    info "Text embedding model already cached — skipping download"
  else
    info "Downloading text embedding model (${TEXT_EMBEDDING_MODEL})..."
    dc exec -T api python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('${TEXT_EMBEDDING_MODEL}')
print('Text embedding model ready')
" 2>/dev/null && info "Text embedding model ready" || warn "Text embedding model download failed (will retry on first use)"
  fi

  if dc exec -T api python -c "
import os
cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
model_dir = os.path.join(cache_dir, '${reranker_model_dir}')
exit(0 if os.path.isdir(model_dir) else 1)
" 2>/dev/null; then
    info "Reranker model already cached — skipping download"
  else
    info "Downloading reranker model (${RERANKER_MODEL})..."
    dc exec -T api python -c "
from sentence_transformers import CrossEncoder
CrossEncoder('${RERANKER_MODEL}')
print('Reranker model ready')
" 2>/dev/null && info "Reranker model ready" || warn "Reranker model download failed (will retry on first use)"
  fi

  # Pull Ollama model via the Ollama service on the Docker network
  local ollama_url="${OLLAMA_BASE_URL:-http://ollama:11434}"
  if dc exec -T api curl -sf "${ollama_url}/api/tags" 2>/dev/null | grep -q "${DOCLING_GRAPH_MODEL}"; then
    info "Ollama model ${DOCLING_GRAPH_MODEL} already present — skipping pull"
  else
    info "Pulling Ollama model (${DOCLING_GRAPH_MODEL}) via ${ollama_url}..."
    dc exec -T api curl -sf "${ollama_url}/api/pull" -d "{\"name\":\"${DOCLING_GRAPH_MODEL}\"}" >/dev/null 2>&1 \
      && info "Ollama model ready" || warn "Ollama pull failed (ensure Ollama service is on the Docker network)"
  fi

  divider
  info "All services started."
  info "  UI:            http://localhost:${api_port}"
  info "  API:           http://localhost:${api_port}/v1"
  info "  API docs:      http://localhost:${api_port}/docs"
  info "  MinIO console: http://localhost:${MINIO_CONSOLE_PORT:-9001}"
  if [[ "${profile}" == "split" ]]; then
    info "  Workers:       ingest(${WORKER_INGEST_CONCURRENCY:-4}), embed(${WORKER_EMBED_CONCURRENCY:-2}), graph(${WORKER_GRAPH_CONCURRENCY:-2})"
  fi
}

cmd_stop() {
  header "Stopping EIP-MMDPP stack"
  # Stop split-profile workers first (not included in default profile)
  dc --profile split down --remove-orphans
  info "All services stopped."
}

cmd_restart() {
  header "Restarting EIP-MMDPP stack (no rebuild)"
  dc restart

  local api_port="${API_PORT:-8000}"
  divider
  wait_for_healthy "http://localhost:${api_port}/v1/health" "API liveness" "${HEALTH_TIMEOUT}"
  wait_for_healthy "http://localhost:${api_port}/v1/health/ready" "API readiness" "${HEALTH_TIMEOUT}"

  info "All services restarted."
}

cmd_status() {
  header "Container Status"
  dc ps

  local api_port="${API_PORT:-8000}"

  header "Health Checks"

  echo -n "  Liveness  (/v1/health)      : "
  if curl -sf --max-time 5 "http://localhost:${api_port}/v1/health" &>/dev/null; then
    echo -e "${GREEN}OK${NC}"
  else
    echo -e "${RED}UNREACHABLE${NC}"
  fi

  echo -n "  Readiness (/v1/health/ready) : "
  local ready_response
  if ready_response=$(curl -sf --max-time 5 "http://localhost:${api_port}/v1/health/ready" 2>/dev/null); then
    echo -e "${GREEN}OK${NC}"
    echo "${ready_response}" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for svc, status in data.get('checks', {}).items():
        color = '\033[0;32m' if status == 'ok' else '\033[0;31m'
        print(f'    {svc:12s}: {color}{status}\033[0m')
except Exception:
    pass
" 2>/dev/null || true
  else
    echo -e "${RED}UNREACHABLE${NC}"
  fi
}

cmd_logs() {
  local service="${1:-}"
  if [[ -n "${service}" ]]; then
    info "Streaming logs for '${service}' (Ctrl+C to stop)..."
    dc logs -f "${service}"
  else
    info "Streaming all service logs (Ctrl+C to stop)..."
    dc logs -f
  fi
}

cmd_blow_away() {
  header "DESTROY all EIP-MMDPP services, volumes, and data"
  warn "This will delete ALL Docker volumes (postgres, redis, minio, docling_model_cache, watch_dirs, celerybeat)."
  warn "This action is IRREVERSIBLE."
  echo

  echo -n -e "${RED}Type 'yes' to confirm: ${NC}"
  read -r confirm
  if [[ "${confirm}" != "yes" ]]; then
    info "Aborted."
    return 0
  fi

  info "Stopping and removing all containers, networks, and volumes..."
  dc down -v --remove-orphans

  safe_rm_rf "${ROOT_DIR}/reports"

  divider
  info "Everything destroyed. Run './manage.sh --start' to rebuild from scratch."
}

# ---------------------------------------------------------------------------
# Database commands
# ---------------------------------------------------------------------------
cmd_migrate() {
  header "Running database migrations"
  dc exec api alembic upgrade head
  info "Migrations complete."
}

cmd_seed() {
  header "Seeding ontology"
  dc exec api python scripts/seed_ontology.py
  info "Ontology seeded."
}

cmd_db_shell() {
  header "Opening PostgreSQL shell"
  dc exec postgres psql -U "${POSTGRES_USER:-eip}" -d "${POSTGRES_DB:-eip}"
}

# ---------------------------------------------------------------------------
# Worker commands
# ---------------------------------------------------------------------------
cmd_worker_status() {
  header "Celery Worker Status"

  info "Active tasks:"
  dc exec worker celery -A app.workers.celery_app inspect active 2>/dev/null || warn "No workers responded."

  divider
  info "Reserved (prefetched) tasks:"
  dc exec worker celery -A app.workers.celery_app inspect reserved 2>/dev/null || warn "No workers responded."

  divider
  info "Scheduled tasks:"
  dc exec worker celery -A app.workers.celery_app inspect scheduled 2>/dev/null || warn "No workers responded."

  divider
  info "Registered task types:"
  dc exec worker celery -A app.workers.celery_app inspect registered 2>/dev/null || warn "No workers responded."

  divider
  info "Worker stats:"
  dc exec worker celery -A app.workers.celery_app inspect stats 2>/dev/null || warn "No workers responded."

  header "Beat Container"
  dc ps beat
}

# ---------------------------------------------------------------------------
# Test delegation
# ---------------------------------------------------------------------------
cmd_test() {
  local mode="${1:-}"
  local test_script="${ROOT_DIR}/scripts/run_tests.sh"

  if [[ ! -x "${test_script}" ]]; then
    error "Test script not found or not executable: ${test_script}"
  fi

  if [[ -n "${mode}" ]]; then
    info "Delegating to scripts/run_tests.sh ${mode}"
    exec "${test_script}" "${mode}"
  else
    info "Delegating to scripts/run_tests.sh (full suite)"
    exec "${test_script}"
  fi
}

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
  cat <<EOF
${BOLD}EIP-MMDPP Management CLI${NC}

${CYAN}Usage:${NC}  ./manage.sh <command> [args]

${CYAN}Service Lifecycle:${NC}
  --start              Build images and start all services (single mixed worker)
  --start-split        Build and start with split workers (ingest/embed/graph)
  --stop               Stop all services gracefully (preserves data)
  --restart            Restart without rebuilding images
  --status             Show service status and health checks
  --logs [service]     Stream logs (optionally: api, worker, beat, postgres, redis, minio, docling)
  --blow-away          Destroy everything: containers, volumes, data (confirms first)

${CYAN}Database:${NC}
  --migrate            Run alembic upgrade head in the api container
  --seed               Run ontology seeder (scripts/seed_ontology.py)
  --db-shell           Open interactive psql shell

${CYAN}Workers:${NC}
  --worker-status      Show Celery worker/beat status and task info

${CYAN}Testing:${NC}
  --test [mode]        Run tests via scripts/run_tests.sh
                       Modes: all (default), unit, integration, contract, e2e

${CYAN}Examples:${NC}
  ./manage.sh --start                  # Build & start (single worker)
  ./manage.sh --start-split            # Build & start (split workers)
  ./manage.sh --logs worker            # Follow worker logs
  ./manage.sh --db-shell               # Open psql
  ./manage.sh --test unit              # Run unit tests only
  ./manage.sh --worker-status          # Inspect Celery workers
  ./manage.sh --blow-away              # Nuclear reset
EOF
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
main() {
  check_prerequisites

  case "${1:-}" in
    --help|-h|"")
      usage
      exit 0
      ;;
  esac

  validate_env

  case "${1}" in
    --start)          cmd_start ;;
    --start-split)    cmd_start split ;;
    --stop)           cmd_stop ;;
    --restart)        cmd_restart ;;
    --status)         cmd_status ;;
    --logs)           cmd_logs "${2:-}" ;;
    --blow-away)      cmd_blow_away ;;
    --migrate)        cmd_migrate ;;
    --seed)           cmd_seed ;;
    --db-shell)       cmd_db_shell ;;
    --worker-status)  cmd_worker_status ;;
    --test)           cmd_test "${2:-}" ;;
    *)
      error "Unknown command: ${1}\nRun './manage.sh --help' for usage."
      ;;
  esac
}

main "$@"
