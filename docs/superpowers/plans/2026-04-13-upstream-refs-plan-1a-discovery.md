# Task 5a Discovery: docling_graph Prompt Injection Seam

**Date:** 2026-04-14
**Branch:** feature/upstream-refs
**Library version:** 1.4.4 (vendored at `/home/josh/development/EIP-MMDPP/docker/docling-graph/repo/docling_graph/`)

---

## Decision

**Path B** — No dedicated prompt/preamble kwarg exists anywhere in `PipelineConfig` or the pipeline call chain. The document body is inlined verbatim as a plain Python `str` named `markdown` before it reaches any LLM prompt. The injection point for Task 5b is: **prepend the preamble to the `markdown` string passed into `extract_from_markdown` / `extract_from_chunk_batches` with copy-on-write, inside `build_pipeline_config` or the service handler**.

The exact target field/attribute: the `markdown` positional argument (a plain Python `str`) at `LlmBackend.extract_from_markdown(markdown=..., ...)` and `LlmBackend.extract_from_chunk_batches(chunks=[...], ...)`.

---

## Evidence

### 1. PipelineConfig — no prompt/preamble fields

`repo/docling_graph/config.py:102-368` (`class PipelineConfig`) defines **57 fields**. None accept a prompt, preamble, or free-form text that feeds the LLM. The full field list is in the audit section below.

Confirmation: `PipelineConfig.to_dict()` at lines 408–491 enumerates every field forwarded to `run_pipeline`. No prompt-related key appears among them.

### 2. `run_pipeline` signature — no prompt kwarg

`repo/docling_graph/pipeline.py:21-85`:
```python
def run_pipeline(config: Union[PipelineConfig, Dict[str, Any]]) -> PipelineContext:
    return _run_pipeline(config, mode="api")
```
`repo/docling_graph/pipeline/orchestrator.py:309-367`:
```python
def run_pipeline(config: Union[PipelineConfig, Dict[str, Any]], mode: Literal["cli", "api"] = "api") -> PipelineContext:
    if isinstance(config, dict):
        config = PipelineConfig(**config)
    orchestrator = PipelineOrchestrator(config, mode=mode)
    return orchestrator.run()
```
No extra kwargs accepted.

### 3. Prompt assembly — direct contract

`repo/docling_graph/core/extractors/contracts/direct/prompts.py:83-144`

`get_extraction_prompt(markdown_content, schema_json, ...)` accepts only:
- `markdown_content: str` — the document body, inlined verbatim
- `schema_json: str` — template schema (no preamble slot)
- `structured_output: bool`, `schema_dict`, `force_legacy_prompt_schema` — control flags

The document body is embedded at lines 55–64 / 66–75 inside:
```
=== COMPLETE DOCUMENT ===
{markdown_content}
=== END COMPLETE DOCUMENT ===
```
There is no slot for additional instructions injected from `PipelineConfig`.

Call site: `repo/docling_graph/core/extractors/backends/llm_backend.py:704-711`:
```python
prompt = direct.get_extraction_prompt(
    markdown_content=markdown,
    schema_json=schema_json,
    ...
)
```

### 4. Prompt assembly — delta contract

`repo/docling_graph/core/extractors/contracts/delta/prompts.py:8-79`

`get_delta_batch_prompt(batch_markdown, ..., global_context=None, already_found=None)` accepts:
- `batch_markdown: str` — one or more chunk payloads concatenated, inlined at lines 54–57
- `global_context: str | None` — but this is **derived internally** from `chunks[0][:600]` at `orchestrator.py:500-507`, not injectable from outside

The document body is inlined at lines 54–58:
```
=== BATCH DOCUMENT ===
{batch_markdown}
=== END BATCH DOCUMENT ===
```
No external preamble slot.

Call site: `repo/docling_graph/core/extractors/contracts/delta/orchestrator.py:199-207`:
```python
prompt = get_delta_batch_prompt(
    batch_markdown=batch_markdown,
    schema_semantic_guide=semantic_guide,
    path_catalog_block=catalog_block,
    batch_index=batch_index,
    total_batches=total_batches,
    global_context=global_context,
    already_found=already_found,
)
```

### 5. Prompt assembly — staged contract

`repo/docling_graph/core/extractors/contracts/staged/prompts.py` delegates entirely to `catalog.get_discovery_prompt(markdown_content, catalog, ...)` — same pattern, only `markdown_content` is the document text, no preamble slot.

### 6. The `markdown` string — where it originates

Strategy layer (`repo/docling_graph/core/extractors/strategies/many_to_one.py:305`):
```python
full_markdown = self.doc_processor.extract_full_markdown(document)
```
This plain `str` is passed directly to `backend.extract_from_markdown(markdown=full_markdown, ...)` at line 428 or to `extract_delta_from_document(...)` at line 309. For the delta path, the document is rechunked from `full_markdown` inside `extract_delta_from_document`. The string is a regular Python `str` — trivially prependable.

### 7. No preamble/extra_context anywhere in the library

Exhaustive search for `preamble`, `extra_prompt`, `prompt_override`, `additional_context`, `extra_context`, `system_prompt_override`, `user_prompt_override`, `instructions` in the library source found zero occurrences of any such field or parameter (only the internal `_EXTRACTION_INSTRUCTIONS` constant which is hardcoded).

---

## Fields scanned (for audit)

Every `PipelineConfig` field from `config.py:102-368`:

| Field | Type | Prompt-related? |
|---|---|---|
| `source` | `Union[str, Path]` | No |
| `template` | `Union[str, type[BaseModel]]` | No (schema only) |
| `backend` | `Literal["llm", "vlm"]` | No |
| `inference` | `Literal["local", "remote"]` | No |
| `processing_mode` | `Literal["one-to-one", "many-to-one"]` | No |
| `extraction_contract` | `Literal["direct", "staged", "delta"]` | No |
| `docling_config` | `Literal["ocr", "vision"]` | No |
| `model_override` | `str | None` | No |
| `provider_override` | `str | None` | No |
| `llm_client` | `Any | None` | No |
| `models` | `ModelsConfig` | No |
| `llm_overrides` | `LlmRuntimeOverrides` | No |
| `structured_output` | `bool` | No |
| `structured_sparse_check` | `bool` | No |
| `use_chunking` | `bool` | No |
| `chunk_max_tokens` | `int | None` | No |
| `llm_batch_token_size` | `int` | No |
| `debug` | `bool` | No |
| `max_batch_size` | `int` | No |
| `staged_tuning_preset` | `Literal[...]` | No |
| `staged_pass_retries` | `int | None` | No |
| `parallel_workers` | `int | None` | No |
| `delta_normalizer_validate_paths` | `bool` | No |
| `delta_normalizer_canonicalize_ids` | `bool` | No |
| `delta_normalizer_strip_nested_properties` | `bool` | No |
| `delta_normalizer_attach_provenance` | `bool` | No |
| `delta_resolvers_enabled` | `bool` | No |
| `delta_resolvers_mode` | `Literal[...]` | No |
| `delta_resolver_fuzzy_threshold` | `float` | No |
| `delta_resolver_semantic_threshold` | `float` | No |
| `delta_resolver_properties` | `list[str] | None` | No |
| `delta_resolver_paths` | `list[str] | None` | No |
| `delta_resolver_allow_merge_different_ids` | `bool` | No |
| `quality_max_unknown_path_drops` | `int` | No |
| `quality_max_id_mismatch` | `int` | No |
| `quality_max_nested_property_drops` | `int` | No |
| `delta_quality_require_root` | `bool` | No |
| `delta_quality_min_instances` | `int` | No |
| `delta_quality_max_parent_lookup_miss` | `int` | No |
| `delta_quality_adaptive_parent_lookup` | `bool` | No |
| `delta_quality_min_non_empty_properties` | `int` | No |
| `delta_quality_min_root_non_empty_fields` | `int` | No |
| `delta_quality_min_non_empty_by_path` | `dict[str, int] | None` | No |
| `delta_quality_max_orphan_ratio` | `float` | No |
| `delta_quality_max_canonical_duplicates` | `int` | No |
| `delta_batch_split_max_retries` | `int` | No |
| `delta_identity_filter_enabled` | `bool` | No |
| `delta_identity_filter_strict` | `bool` | No |
| `gleaning_enabled` | `bool` | No |
| `gleaning_max_passes` | `int` | No |
| `staged_nodes_fill_cap` | `int | None` | No |
| `staged_id_shard_size` | `int | None` | No |
| `staged_id_identity_only` | `bool` | No |
| `staged_id_compact_prompt` | `bool` | No |
| `staged_id_auto_shard_threshold` | `int` | No |
| `staged_id_shard_min_size` | `int` | No |
| `staged_quality_require_root` | `bool` | No |
| `staged_quality_min_instances` | `int` | No |
| `staged_quality_max_parent_lookup_miss` | `int` | No |
| `staged_id_max_tokens` | `int | None` | No |
| `staged_fill_max_tokens` | `int | None` | No |
| `export_format` | `Literal[...]` | No |
| `export_docling` | `bool` | No |
| `export_docling_json` | `bool` | No |
| `export_markdown` | `bool` | No |
| `export_per_page_markdown` | `bool` | No |
| `reverse_edges` | `bool` | No |
| `output_dir` | `Union[str, Path]` | No |
| `dump_to_disk` | `bool | None` | No |

---

## Notes for Task 5b

### Injection strategy

Since there is no `PipelineConfig` kwarg to forward a preamble to, Task 5b must inject the preamble by **modifying the document content string** before it enters the pipeline. There are two viable sub-approaches:

**Option B1 (recommended) — Prepend to the DoclingDocument's markdown at the service handler level:**

In `docker/docling-graph/app/main.py`, before calling `run_pipeline`, the service handler already has access to the `DoclingDocument` (loaded from the input). The preamble can be prepended to the document's text at the `DoclingDocument` level so all three contracts (direct, delta, staged) receive it naturally. The `doc_processor.extract_full_markdown(document)` call in the strategy layer will naturally include it.

**Option B2 — Widen `build_pipeline_config` with `extra_prompt_preamble=None` and store it, then prepend in the service handler:**

`build_pipeline_config` at `docker/docling-graph/app/config_builder.py:81` can accept `extra_prompt_preamble=None`, stash it in the returned config object (as a custom attribute or alongside), and the `/extract-pass` handler applies it to the document text before calling `run_pipeline`.

**Option B3 — Monkey-patch `get_extraction_prompt` / `get_delta_batch_prompt`:**
Do not do this. It is fragile against library upgrades and would need to intercept all three contracts independently.

### Contract coverage caveat

- **Direct contract:** preamble in `markdown_content` → appears inside `=== COMPLETE DOCUMENT ===` block — fully visible to LLM.
- **Delta contract:** preamble prepended to chunks; it will appear in `batch_markdown` of the first batch (and only that batch if the document is long). This may be sufficient since `global_context` already truncates to 600 chars from the first chunk, but the preamble will be more prominent.
- **Staged contract:** preamble in `markdown_content` → passed to `get_discovery_prompt(markdown_content, ...)` — fully visible.

For all three contracts, prepending to the full markdown string (before chunking) is the correct, uniform approach.

### `build_pipeline_config` does not need a new `PipelineConfig` field

Because `PipelineConfig` has no prompt-related fields and the library provides no hook, Task 5b should **not** attempt to add a field to `PipelineConfig`. Instead, `build_pipeline_config` returns a `(config, extra_prompt_preamble)` tuple or the handler stashes it separately, and the handler applies the prepend to the document text before calling `run_pipeline`.
