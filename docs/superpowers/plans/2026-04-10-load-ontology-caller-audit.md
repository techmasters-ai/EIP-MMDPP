# load_ontology caller audit (PR 1, Task 2.1)

Generated as part of Task 2.1 of the extraction-refactor plan. Source:
`grep -rn "load_ontology\|get_ontology_cache_signature\|prefer_active" app/ tests/ docker/`

Buckets:
- **extraction**: extraction hot path → deferred to Chunk 4 (orchestrator rewrite)
- **registry**: version-pinned consumers → `load_registry_ontology(version_id=...)`
- **default**: no-arg callers safe with system-default bundle → drop `prefer_active=True` if present
- **unaffected**: grep false-positive (local helpers, patches, etc.)

## Caller table

| File:line | Snippet | Bucket | Action |
|---|---|---|---|
| app/workers/pipeline.py:2564 | `from app.services.ontology_templates import load_ontology` | default | no change (legacy path, no-arg call) |
| app/workers/pipeline.py:2565 | `ontology = load_ontology()` | default | no change (no args, system default bundle fine) |
| app/api/v1/_retrieval_helpers.py:12 | `get_ontology_cache_signature,` | default | no change — `get_ontology_cache_signature()` still exists with same no-arg semantics |
| app/api/v1/_retrieval_helpers.py:13 | `load_ontology,` | default | no change — import stays valid |
| app/api/v1/_retrieval_helpers.py:50 | `data = load_ontology()` | default | no change (no args) |
| app/api/v1/_retrieval_helpers.py:58 | `return _load_scoring_weights(get_ontology_cache_signature())` | default | no change (no-arg call, signature function kept) |
| app/services/ontology_templates.py:143 | `def get_ontology_cache_signature(` | registry | REFACTORED: drop `path` + `prefer_active` params, new sig `get_ontology_cache_signature(bundle_key=None)` |
| app/services/ontology_templates.py:146 | `prefer_active: bool = True,` | registry | DELETED: `prefer_active` param removed in refactor |
| app/services/ontology_templates.py:149 | `if path is not None or not prefer_active:` | registry | DELETED: branch removed in refactor |
| app/services/ontology_templates.py:155 | `def load_ontology(` | default | REFACTORED: new sig `load_ontology(*, bundle_key=None, path=None)` |
| app/services/ontology_templates.py:158 | `prefer_active: bool = True,` | registry | DELETED: `prefer_active` param dropped |
| app/services/ontology_templates.py:161 | `if path is not None or not prefer_active:` | registry | DELETED: old path/prefer_active branch removed |
| app/services/ontology_templates.py:170 | `def load_validation_matrix(` | default | REFACTORED: new sig `load_validation_matrix(*, bundle_key=None, path=None)` |
| app/services/ontology_templates.py:176 | `ontology = load_ontology(path, prefer_active=prefer_active)` | registry | UPDATED: updated to `load_ontology(bundle_key=bundle_key, path=path)` |
| app/services/ontology_templates.py:201 | `ontology = load_ontology()` | default | no change (internal helper, no args) |
| app/services/ontology_templates.py:208 | `ontology = load_ontology()` | default | no change (internal helper, no args) |
| app/services/arcadedb_graph.py:433 | `from app.services.ontology_templates import load_ontology` | default | no change (import stays valid) |
| app/services/arcadedb_graph.py:434 | `ontology = load_ontology()` | default | no change (no-arg call) |
| app/services/arcadedb_graph.py:879 | `from app.services.ontology_templates import load_ontology` | default | no change (import stays valid) |
| app/services/arcadedb_graph.py:881 | `ont = load_ontology()` | default | no change (no-arg call) |
| app/services/arcadedb_graph.py:2060 | `from app.services.ontology_templates import load_ontology` | default | no change (import stays valid) |
| app/services/arcadedb_graph.py:2062 | `ont = load_ontology()` | default | no change (no-arg call) |
| app/services/docling_graph_service.py:17 | `from app.services.ontology_templates import load_ontology` | default | no change (import stays valid) |
| app/services/docling_graph_service.py:39 | `ontology = ontology_definition or load_ontology()` | default | no change (no-arg fallback, legacy service param `ontology_definition` phased out in PR 3) |
| app/services/query_profiles.py:30 | `from app.services.ontology_templates import load_ontology, load_repository_ontology` | default | no change (both functions remain; `load_ontology` import stays valid) |
| app/services/query_profiles.py:147 | `else load_ontology()` | default | no change — `_ontology_subset()` calls `load_ontology()` with no args which returns system default bundle; acceptable for query-profile rendering (see Bucket D note below) |
| app/main.py:52 | `from app.services.ontology_templates import load_ontology` | default | no change (import stays valid) |
| app/main.py:54 | `ontology = load_ontology()` | default | no change (startup bootstrap, no-arg call) |
| tests/unit/test_query_profiles.py:241 | `def test_load_ontology_prefer_active_false_falls_back_to_yaml(self):` | registry | DELETED: test exercises behavior that no longer exists (`prefer_active=False` path dropped) |
| tests/unit/test_query_profiles.py:242 | `"""With prefer_active=False the loader must skip the DB...` | registry | DELETED: as above |
| tests/unit/test_query_profiles.py:244 | `from app.services.ontology_templates import load_ontology` | registry | DELETED: as above |
| tests/unit/test_query_profiles.py:246 | `ontology = load_ontology(prefer_active=False)` | registry | DELETED: as above — `prefer_active` no longer a valid kwarg |
| tests/unit/test_ontology_templates.py:15 | `from app.services.ontology_templates import load_ontology` | default | no change (import stays valid) |
| tests/unit/test_ontology_templates.py:17 | `ontology = load_ontology()` | default | no change (no-arg call) |
| tests/unit/test_ontology_templates.py:23 | `from app.services.ontology_templates import load_ontology` | default | no change |
| tests/unit/test_ontology_templates.py:25 | `ontology = load_ontology()` | default | no change |
| tests/unit/test_ontology_templates.py:32 | `from app.services.ontology_templates import load_ontology` | default | no change |
| tests/unit/test_ontology_templates.py:34 | `ontology = load_ontology()` | default | no change |
| tests/unit/test_ontology_templates.py:45 | `from app.services.ontology_templates import load_ontology` | default | no change |
| tests/unit/test_ontology_templates.py:47 | `ontology = load_ontology()` | default | no change |
| tests/unit/test_ontology_templates.py:57 | `from app.services.ontology_templates import load_ontology` | default | no change |
| tests/unit/test_ontology_templates.py:59 | `ontology = load_ontology()` | default | no change |
| tests/unit/test_arcadedb_schema.py:11 | `def _load_ontology():` | unaffected | local helper in test file — reads file directly, not `app.services.ontology_templates.load_ontology` |
| tests/unit/test_arcadedb_schema.py:26 | `ontology = _load_ontology()` | unaffected | calls local helper, not the worker API |
| tests/unit/test_arcadedb_schema.py:39 | `ontology = _load_ontology()` | unaffected | calls local helper |
| tests/unit/test_arcadedb_schema.py:49 | `ontology = _load_ontology()` | unaffected | calls local helper |
| tests/unit/test_arcadedb_schema.py:60 | `ontology = _load_ontology()` | unaffected | calls local helper |
| tests/unit/test_arcadedb_schema.py:70 | `ontology = _load_ontology()` | unaffected | calls local helper |
| tests/unit/test_arcadedb_schema.py:83 | `ontology = _load_ontology()` | unaffected | calls local helper |
| tests/unit/test_arcadedb_schema.py:92 | `ontology = _load_ontology()` | unaffected | calls local helper |
| tests/unit/test_arcadedb_schema.py:102 | `ontology = _load_ontology()` | unaffected | calls local helper |
| tests/unit/test_arcadedb_schema.py:115 | `ontology = _load_ontology()` | unaffected | calls local helper |
| tests/unit/test_arcadedb_schema.py:126 | `ontology = _load_ontology()` | unaffected | calls local helper |
| tests/unit/test_arcadedb_schema.py:136 | `ontology = _load_ontology()` | unaffected | calls local helper |
| tests/unit/test_arcadedb_schema.py:145 | `ontology = _load_ontology()` | unaffected | calls local helper |
| tests/unit/test_arcadedb_schema.py:153 | `ontology = _load_ontology()` | unaffected | calls local helper |
| tests/unit/test_arcadedb_schema.py:172 | `ontology = _load_ontology()` | unaffected | calls local helper |
| tests/unit/test_startup_bootstrap.py:21 | `@patch("app.services.ontology_templates.load_ontology", return_value=None)` | unaffected | unittest.mock patch — target path stays valid after refactor, no code change needed |
| tests/unit/test_startup_bootstrap.py:23 | `self, mock_load_ontology, mock_get_graph_store,` | unaffected | mock parameter name; no change |
| docker/docling-graph/tests/test_template_builder.py:9 | `def _load_ontology():` | unaffected | local helper in docker test file — reads bundle YAML directly, entirely separate from the app service |
| docker/docling-graph/tests/test_template_builder.py:67 | `ontology = _load_ontology()` | unaffected | calls local docker-test helper |
| docker/docling-graph/tests/test_template_builder.py:74 | `ontology = _load_ontology()` | unaffected | calls local docker-test helper |
| docker/docling-graph/tests/test_template_builder.py:81 | `ontology = _load_ontology()` | unaffected | calls local docker-test helper |
| docker/docling-graph/tests/test_template_builder.py:91 | `ontology = _load_ontology()` | unaffected | calls local docker-test helper |
| docker/docling-graph/tests/test_template_builder.py:102 | `ontology = _load_ontology()` | unaffected | calls local docker-test helper |
| docker/docling-graph/tests/test_template_builder.py:110 | `ontology = _load_ontology()` | unaffected | calls local docker-test helper |
| docker/docling-graph/tests/test_template_builder.py:124 | `ontology = _load_ontology()` | unaffected | calls local docker-test helper |
| docker/docling-graph/tests/test_template_builder.py:131 | `ontology = _load_ontology()` | unaffected | calls local docker-test helper |
| docker/docling-graph/tests/test_template_builder.py:141 | `ontology = _load_ontology()` | unaffected | calls local docker-test helper |
| docker/docling-graph/tests/test_template_builder.py:150 | `ontology = _load_ontology()` | unaffected | calls local docker-test helper |
| docker/docling-graph/tests/test_template_builder.py:161 | `ontology = _load_ontology()` | unaffected | calls local docker-test helper |
| docker/docling-graph/tests/test_template_builder.py:171 | `ontology = _load_ontology()` | unaffected | calls local docker-test helper |
| docker/docling-graph/tests/test_template_builder.py:181 | `ontology = _load_ontology()` | unaffected | calls local docker-test helper |
| docker/docling-graph/tests/test_template_builder.py:189 | `ontology = _load_ontology()` | unaffected | calls local docker-test helper |

## Bucket D — TBD / flagged for human review

### `app/services/query_profiles.py:147` — `else load_ontology()`

Context: `_ontology_subset(*, repository_only: bool = False)` uses `load_ontology()` (no args) as the non-repository branch. This is used when building the query-profile registry template — it currently means "active registry ontology or fallback to file." After the refactor the no-arg call becomes "system default bundle" which is semantically equivalent in the current deployment (one bundle = air_defense_v3 = the file the symlink points to).

**Assessment: Acceptable as-is for now.** The behavior is equivalent because:
1. `load_ontology()` (new) → `air_defense_v3/ontology.yaml` (via system default bundle)
2. `load_ontology()` (old) → tries active registry row → falls back to `ontology/ontology.yaml` (which symlinks to `air_defense_v3/ontology.yaml`)

If in the future a caller in `query_profiles.py` genuinely needs to load the *current active registry row* (not just any bundle's YAML), it should be changed to call `load_registry_ontology(version_id=<active_row_id>)`. That requires a separate wrapper to look up the active row's ID first — deferred to a future task.

## Summary of actions taken

| Bucket | Count | Action |
|---|---|---|
| extraction | 0 | deferred to Chunk 4 |
| registry | 8 | `prefer_active` param and registry-path branches deleted from `ontology_templates.py`; test covering old behavior deleted |
| default | 30 | no code changes needed; all use no-arg `load_ontology()` or no-arg `get_ontology_cache_signature()` |
| unaffected | 36 | grep false-positives (local helpers, mock patches) — zero changes |
