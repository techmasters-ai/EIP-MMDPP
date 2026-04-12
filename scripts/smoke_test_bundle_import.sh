#!/usr/bin/env bash
# Smoke test: verify ontology_bundles is importable inside both the
# worker and docling-graph images. Run during CI after builds.
#
# Introduced in Task 2.6 of the extraction-refactor plan (spec §2 packaging).
set -euo pipefail

echo "==> Building worker and docling-graph images"
docker compose build worker docling-graph

echo "==> Importing ontology_bundles inside the worker image"
docker compose run --rm --no-deps --entrypoint python worker -c "
from ontology_bundles.air_defense_v3.extraction_schemas.radar_domain import RadarDomainPass
print('worker OK:', RadarDomainPass)
"

echo "==> Importing ontology_bundles inside the docling-graph image"
docker compose run --rm --no-deps --entrypoint python docling-graph -c "
from ontology_bundles.air_defense_v3.extraction_schemas.radar_domain import RadarDomainPass
print('docling-graph OK:', RadarDomainPass)
"

echo "==> Smoke test passed"
