#!/usr/bin/env bash
# CI lints enforcing the PR 3 deletions stay deleted.
# Spec §7.5 CI additions. Each lint exits nonzero on any hit.
set -euo pipefail

FAILED=0

check() {
    local description="$1"
    local pattern="$2"
    local scope="${3:-app/}"
    if grep -rn --exclude="ci_lints.sh" --exclude-dir=".venv" --exclude-dir="venv" --exclude-dir="repo" --exclude-dir="__pycache__" --include="*.py" --include="*.sh" --include="*.yaml" --include="*.yml" --include="*.txt" --include="*.toml" --include="Dockerfile*" "$pattern" $scope > /tmp/lint_hits 2>/dev/null; then
        echo "FAIL: $description"
        cat /tmp/lint_hits
        FAILED=1
    else
        echo "PASS: $description"
    fi
}

# Import scope lint: service-side schemas never imported by worker
check "worker does not import service-side extraction_schemas" \
    "from ontology_bundles\..*\.extraction_schemas" \
    "app/"

# Unsafe confidence defaulting lint — catches bare `confidence or 0.0`
# (variable literally named `confidence`), not `extraction_confidence or 0.0`.
check "no naked 'confidence or 0.0' defaulting" \
    "[^a-z_]confidence or 0\.0" \
    "app/ docker/"

# prefer_active resurrection
check "no resurrected prefer_active kwarg" \
    "prefer_active" \
    "app/ docker/"

# ontology_definition absence — the deleted /extract-all endpoint parameter;
# app/query_profiles legitimately uses this column name for a different purpose.
check "no ontology_definition references in docker service" \
    "ontology_definition" \
    "docker/"

# /app/ontology/ path absence (but allow /app/ontology_bundles/)
check "no /app/ontology/ path references (excluding ontology_bundles)" \
    "/app/ontology[^_]" \
    "app/ docker/"

# graph_extraction_engine absence
check "no graph_extraction_engine references" \
    "graph_extraction_engine" \
    "app/"

# graph_layered_ absence
check "no graph_layered_ references" \
    "graph_layered_" \
    "app/"

# template_count absence (renamed in Task 5.3)
check "no template_count field references" \
    "template_count" \
    "app/ docker/"

if [ $FAILED -ne 0 ]; then
    exit 1
fi
echo "All CI lints passed."
