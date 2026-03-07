#!/usr/bin/env bash
# ============================================================
# EIP-MMDPP — Single test execution entrypoint
# ============================================================
# Usage:
#   ./scripts/run_tests.sh           — full suite (unit → integration → e2e)
#   ./scripts/run_tests.sh unit      — unit tests only
#   ./scripts/run_tests.sh integration
#   ./scripts/run_tests.sh e2e
#
# Environment:
#   KEEP_STACK=1    — preserve Docker Compose stack after tests
#   SKIP_COV=1      — skip coverage instrumentation (faster, lower RAM)
#   TEST_SEED=42    — deterministic random seed for fixtures
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
REPORTS_DIR="${PROJECT_ROOT}/reports"
ENV_FILE="${PROJECT_ROOT}/.env.test"

MODE="${1:-all}"
KEEP_STACK="${KEEP_STACK:-0}"
SKIP_COV="${SKIP_COV:-0}"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
divider() { echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"; }

# ---------------------------------------------------------------------------
# Load .env.test
# ---------------------------------------------------------------------------
if [[ ! -f "${ENV_FILE}" ]]; then
  error ".env.test not found at ${ENV_FILE}. Copy .env.example and configure test values."
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

mkdir -p "${REPORTS_DIR}"

# ---------------------------------------------------------------------------
# Docker Compose stack management
# ---------------------------------------------------------------------------
COMPOSE_CMD="docker compose -p eip-mmdpp-test -f ${PROJECT_ROOT}/docker-compose.yml"
if [[ -f "${PROJECT_ROOT}/docker-compose.test.yml" ]]; then
  COMPOSE_CMD="${COMPOSE_CMD} -f ${PROJECT_ROOT}/docker-compose.test.yml"
fi

start_stack() {
  divider
  info "Starting test stack..."
  ${COMPOSE_CMD} up -d postgres redis minio

  info "Waiting for services to be healthy..."
  local attempts=0
  local max_attempts=30

  while [[ ${attempts} -lt ${max_attempts} ]]; do
    if ${COMPOSE_CMD} ps --format json 2>/dev/null | \
       python3 -c "
import sys, json
services = [json.loads(l) for l in sys.stdin if l.strip()]
required = {'postgres', 'redis', 'minio'}
healthy = {s['Service'] for s in services if s.get('Health') == 'healthy'}
missing = required - healthy
if missing:
    sys.exit(1)
" 2>/dev/null; then
      info "All services healthy."
      break
    fi
    attempts=$((attempts + 1))
    echo -n "."
    sleep 3
  done
  echo

  if [[ ${attempts} -ge ${max_attempts} ]]; then
    error "Timed out waiting for services to become healthy."
  fi
}

stop_stack() {
  if [[ "${KEEP_STACK}" == "1" ]]; then
    warn "KEEP_STACK=1: leaving Docker Compose stack running."
    return
  fi
  divider
  info "Stopping test stack..."
  ${COMPOSE_CMD} down -v --remove-orphans || true
}

trap stop_stack EXIT

# ---------------------------------------------------------------------------
# Migrations and seed
# ---------------------------------------------------------------------------
run_migrations() {
  info "Running database migrations..."
  cd "${PROJECT_ROOT}"
  DATABASE_URL="${DATABASE_URL_SYNC:-postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT:-5435}/${POSTGRES_DB}}"
  DATABASE_URL="${DATABASE_URL}" alembic upgrade head
  info "Seeding ontology..."
  python3 scripts/seed_ontology.py
}

# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------
run_unit() {
  divider
  info "Running unit tests..."
  cd "${PROJECT_ROOT}"
  local cov_args=()
  if [[ "${SKIP_COV}" != "1" ]]; then
    cov_args=(--cov=app --cov-report=xml:"${REPORTS_DIR}/coverage_unit.xml")
  fi
  pytest tests/unit \
    -m "unit" \
    --tb=short \
    "${cov_args[@]}" \
    --junitxml="${REPORTS_DIR}/junit_unit.xml" \
    -q
  info "Unit tests passed."
}

run_integration() {
  divider
  info "Running integration tests..."
  cd "${PROJECT_ROOT}"
  local cov_args=()
  if [[ "${SKIP_COV}" != "1" ]]; then
    cov_args=(--cov=app --cov-append --cov-report=xml:"${REPORTS_DIR}/coverage_integration.xml")
  fi
  pytest tests/integration \
    -m "integration" \
    --tb=short \
    "${cov_args[@]}" \
    --junitxml="${REPORTS_DIR}/junit_integration.xml" \
    -q
  info "Integration tests passed."
}

run_e2e() {
  divider
  info "Running E2E tests..."
  cd "${PROJECT_ROOT}"
  pytest tests/e2e \
    -m "e2e" \
    --tb=short \
    --timeout=300 \
    --junitxml="${REPORTS_DIR}/junit_e2e.xml" \
    -q
  info "E2E tests passed."
}

write_summary() {
  divider
  info "Writing test summary..."
  python3 - <<'PYEOF'
import glob, xml.etree.ElementTree as ET, sys, os

reports = glob.glob(os.path.join(os.environ.get('REPORTS_DIR', 'reports'), 'junit_*.xml'))
total_tests = 0
total_failures = 0
total_errors = 0

for report in reports:
    try:
        tree = ET.parse(report)
        root = tree.getroot()
        suite = root if root.tag == 'testsuite' else root.find('testsuite')
        if suite is not None:
            total_tests   += int(suite.get('tests', 0))
            total_failures += int(suite.get('failures', 0))
            total_errors   += int(suite.get('errors', 0))
    except Exception:
        pass

summary = f"""
EIP-MMDPP Test Summary
======================
Total Tests  : {total_tests}
Failures     : {total_failures}
Errors       : {total_errors}
Result       : {'PASSED' if total_failures == 0 and total_errors == 0 else 'FAILED'}
"""
print(summary)

summary_path = os.path.join(os.environ.get('REPORTS_DIR', 'reports'), 'summary.txt')
with open(summary_path, 'w') as f:
    f.write(summary)

if total_failures > 0 or total_errors > 0:
    sys.exit(1)
PYEOF
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
FAILED=0

case "${MODE}" in
  unit)
    run_unit || FAILED=1
    ;;
  integration)
    start_stack
    run_migrations || FAILED=1
    run_integration || FAILED=1
    ;;
  e2e)
    start_stack
    run_migrations || FAILED=1
    run_e2e || FAILED=1
    ;;
  all)
    run_unit || FAILED=1
    start_stack
    run_migrations || FAILED=1
    run_integration || FAILED=1
    run_e2e         || FAILED=1
    write_summary   || FAILED=1
    ;;
  *)
    error "Unknown mode: ${MODE}. Use: all | unit | integration | e2e"
    ;;
esac

if [[ ${FAILED} -ne 0 ]]; then
  error "Test suite FAILED. See reports/ for details."
fi

divider
info "Test suite PASSED."
