"""Docling-Graph entity/relationship extraction service.

FastAPI wrapper around the docling-graph package.  Generates Pydantic
templates from a volume-mounted ontology YAML and routes LLM calls via
LiteLLM to Ollama or an OpenAI-compatible endpoint.

Extraction modes:
  - /extract (entities|relationships) — single group, backward-compatible
  - /extract-all — parallel extraction across all 5 groups + relationships
"""

from __future__ import annotations

import json as json_mod
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from typing import Any

import litellm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.schemas import (
    ExtractedEntityResponse,
    ExtractedRelationshipResponse,
    ExtractionRequest,
    ExtractionResponse,
)
from app.templates import GROUP_MAP, build_templates, get_ontology_version, load_ontology

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
ONTOLOGY_PATH = os.environ.get("ONTOLOGY_PATH", "/ontology/ontology.yaml")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
LLM_MODEL = os.environ.get("LLM_MODEL", "granite3-dense:8b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))
OLLAMA_THINK = os.environ.get("OLLAMA_THINK", "")  # e.g. "low", "medium", "high" for gpt-oss
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
GRAPH_EXTRACTION_CHUNK_SIZE = int(os.environ.get("GRAPH_EXTRACTION_CHUNK_SIZE", "100000"))
GRAPH_EXTRACTION_CHUNK_OVERLAP = int(os.environ.get("GRAPH_EXTRACTION_CHUNK_OVERLAP", "2000"))

# Pre-compiled regex patterns for JSON extraction from LLM output
_RE_THINK_TAGS = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)
_RE_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_RE_ANSWER_MARKERS = [
    re.compile(r"Final [Aa]nswer:\s*"),
    re.compile(r"Answer:\s*"),
    re.compile(r"In conclusion[,:]\s*"),
]

# ---------------------------------------------------------------------------
# Module-level state (populated at startup)
# ---------------------------------------------------------------------------
_templates: dict[str, dict[str, type[BaseModel]]] | None = None
_ontology_version: str | None = None
# Pre-built schema descriptions per group (built from ontology YAML at startup)
_group_schema_prompts: dict[str, str] = {}
# Auto-generated relationship type context from ontology (for relationship extraction prompt)
_relationship_prompt_context: str = ""


def _patch_node_id_registry() -> None:
    """Fix docling-graph bug where class names with underscores cause collisions."""
    try:
        from docling_graph.core.converters.node_id_registry import NodeIDRegistry

        def _patched_get_node_id(self, model_instance, auto_register=True):
            fingerprint = self._generate_fingerprint(model_instance)
            class_name = model_instance.__class__.__name__

            if fingerprint in self.fingerprint_to_id:
                existing_id = self.fingerprint_to_id[fingerprint]
                existing_class = existing_id.rsplit("_", 1)[0] if "_" in existing_id else existing_id
                if existing_class != class_name:
                    raise ValueError(
                        f"Node ID collision: fingerprint {fingerprint} maps to both "
                        f"{existing_id} (class: {existing_class}) and {class_name}_... (new class)"
                    )
                return existing_id

            if class_name not in self.seen_classes:
                self.seen_classes[class_name] = set()

            node_id = f"{class_name}_{fingerprint}"

            if auto_register:
                self.fingerprint_to_id[fingerprint] = node_id
                self.id_to_fingerprint[node_id] = fingerprint
                self.seen_classes[class_name].add(fingerprint)

            return node_id

        NodeIDRegistry.get_node_id = _patched_get_node_id
        logger.info("Patched NodeIDRegistry.get_node_id (rsplit fix for underscore class names)")
    except Exception:
        logger.warning("Could not patch NodeIDRegistry — docling-graph version may differ")


def _configure_ollama_provider() -> None:
    """Set Ollama-specific defaults on LiteLLM's OllamaConfig class."""
    if LLM_PROVIDER != "ollama":
        return
    try:
        from litellm.llms.ollama.completion.transformation import (
            OllamaConfig as _OllamaConfig,
        )

        _OllamaConfig.num_ctx = OLLAMA_NUM_CTX
        if OLLAMA_THINK:
            _OllamaConfig.think = OLLAMA_THINK
        logger.info(
            "Set LiteLLM OllamaConfig: num_ctx=%d, think=%s",
            OLLAMA_NUM_CTX, OLLAMA_THINK or "(default)",
        )
    except Exception:
        logger.warning("Could not set OllamaConfig.num_ctx — LiteLLM version may differ")


def _build_group_schema_prompts(ontology: dict[str, Any]) -> dict[str, str]:
    """Build entity schema description strings from the ontology for each group.

    These are embedded in the LLM prompt so the model knows exactly what
    entity types and properties to extract per group.
    """
    # Index entity type definitions by name
    et_index: dict[str, dict] = {}
    for et in ontology.get("entity_types", []):
        et_index[et["name"]] = et

    prompts: dict[str, str] = {}
    for group_name, member_names in GROUP_MAP.items():
        lines: list[str] = []
        for entity_name in member_names:
            et = et_index.get(entity_name)
            if not et:
                continue
            desc = et.get("description", "")
            parent = et.get("parent", "")
            label = et.get("label", "")
            header = f"### {entity_name}"
            if parent:
                header += f" ({parent})"
            lines.append(header)
            if label:
                lines.append(f"Label: {label}")
            if desc:
                lines.append(f"Description: {desc}")

            props = et.get("properties", {}).get("properties", {})
            if props:
                lines.append("Properties:")
                for prop_name, prop_spec in props.items():
                    ptype = prop_spec.get("type", "string")
                    pdesc = prop_spec.get("description", "")
                    example = prop_spec.get("example", "")
                    enum_vals = prop_spec.get("enum", [])
                    pattern = prop_spec.get("pattern", "")
                    parts = [f"  - {prop_name} ({ptype})"]
                    if pdesc:
                        parts.append(f": {pdesc}")
                    if example:
                        parts.append(f" [example: {example}]")
                    if enum_vals:
                        parts.append(f" [enum: {', '.join(str(v) for v in enum_vals)}]")
                    if pattern:
                        parts.append(f" [format: {pattern}]")
                    lines.append("".join(parts))
            lines.append("")

        prompts[group_name] = "\n".join(lines)

    return prompts


def _build_relationship_prompt_context(ontology: dict[str, Any]) -> str:
    """Build relationship type descriptions and validation rules from ontology."""
    rel_types = ontology.get("relationship_types", [])
    validation = ontology.get("validation_matrix", [])

    lines = ["Available relationship types:"]
    for rt in rel_types:
        name = rt["name"]
        desc = rt.get("description", "")
        src = rt.get("source_type") or "any"
        tgt = rt.get("target_type") or "any"
        card = rt.get("cardinality", "")
        line = f"  - {name}: {desc}"
        if src != "any" or tgt != "any":
            line += f" (from: {src}, to: {tgt})"
        if card:
            line += f" [{card}]"
        lines.append(line)

    if validation:
        lines.append("\nAllowed (source → relationship → target) triples:")
        for v in validation:
            lines.append(f"  {v['source']} → {v['relationship']} → {v['target']}")

    return "\n".join(lines)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ontology and build templates at startup."""
    global _templates, _ontology_version, _group_schema_prompts, _relationship_prompt_context

    _patch_node_id_registry()
    _configure_ollama_provider()
    logger.info("Loading ontology from %s", ONTOLOGY_PATH)
    try:
        ontology = load_ontology(ONTOLOGY_PATH)
        _ontology_version = get_ontology_version(ontology)
        _templates = build_templates(ontology)
        _group_schema_prompts = _build_group_schema_prompts(ontology)
        _relationship_prompt_context = _build_relationship_prompt_context(ontology)
        logger.info(
            "Docling-Graph service ready — ontology v%s, %d templates, %d group prompts, relationship context %d chars",
            _ontology_version,
            sum(len(g) for g in _templates.values()),
            len(_group_schema_prompts),
            len(_relationship_prompt_context),
        )
    except Exception:
        logger.exception("Failed to load ontology — service will report unhealthy")

    yield
    logger.info("Docling-Graph service shutting down.")


app = FastAPI(
    title="EIP-MMDPP Docling-Graph Service",
    description="Entity/relationship extraction via docling-graph + LiteLLM",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    if _templates is None:
        raise HTTPException(status_code=503, detail="Templates not loaded")
    total_entity_types = sum(len(group) for group in _templates.values())
    return {
        "status": "ok",
        "ontology_version": _ontology_version,
        "groups": list(_templates.keys()),
        "template_count": total_entity_types,
    }


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def _litellm_provider() -> str:
    """Return the LiteLLM provider string."""
    if LLM_PROVIDER == "ollama":
        return "ollama_chat"
    return LLM_PROVIDER


def _build_litellm_model_string() -> str:
    """Construct the full ``provider/model`` string LiteLLM expects."""
    provider = _litellm_provider()
    if provider == "openai":
        return LLM_MODEL
    return f"{provider}/{LLM_MODEL}"


def _extract_llm_content(msg: Any) -> str:
    """Extract usable text from an LLM response message.

    Thinking models (Ollama with think=true, OpenAI o-series, etc.) may place
    the final answer in ``content``, or may leave ``content`` empty and put
    everything into a provider-specific reasoning field.

    Fallback chain (per Ollama docs + LiteLLM mapping):
      1. content              — standard answer field
      2. reasoning_content    — LiteLLM maps Ollama's thinking here
      3. reasoning            — Ollama OpenAI-compat path may use this
      4. thinking             — Ollama native field
      5. thinking_blocks      — list of thinking block dicts (stringify)
    """
    content = getattr(msg, "content", None) or ""
    if content.strip():
        return content

    for attr in ("reasoning_content", "reasoning", "thinking"):
        val = getattr(msg, attr, None) or ""
        if isinstance(val, str) and val.strip():
            logger.info("content empty — falling back to %s (%d chars)", attr, len(val))
            return val

    blocks = getattr(msg, "thinking_blocks", None)
    if blocks:
        logger.info("content empty — falling back to thinking_blocks")
        return str(blocks)

    logger.warning("LLM returned no usable text in any field")
    return ""


def _llm_call(system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str | None:
    """Make a single LLM call and return the content string. Returns None on failure."""
    model_string = _build_litellm_model_string()
    llm_kwargs: dict[str, Any] = {
        "model": model_string,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    if LLM_PROVIDER == "ollama":
        llm_kwargs["api_base"] = OLLAMA_BASE_URL
        llm_kwargs["num_ctx"] = OLLAMA_NUM_CTX
        # Use reasoning_effort only — OllamaConfig.think is set globally at
        # startup via _configure_ollama_provider(), so we do NOT also pass it
        # per-request to avoid double-setting via two different LiteLLM paths.

    response = litellm.completion(**llm_kwargs)
    return _extract_llm_content(response.choices[0].message)


def _parse_json_from_llm(raw: str | None) -> dict | list | None:
    """Extract and parse JSON from LLM output.

    Tries multiple strategies in order of reliability:
      1. Whole string as JSON (clean responses)
      2. Strip <think> tags, retry whole string
      3. Fenced ```json code block (regex-based)
      4. Last complete JSON object/array via bracket-matching
         (handles reasoning content where JSON is at the end)
      5. json_repair as last resort for truncated output
    """
    if not raw:
        return None
    raw = raw.strip()

    # Strategy 1: whole string as JSON (cheapest, handles clean output)
    try:
        return json_mod.loads(raw)
    except (json_mod.JSONDecodeError, ValueError):
        pass

    # Strategy 2: strip reasoning markers and retry
    cleaned = _RE_THINK_TAGS.sub("", raw).strip()
    for pattern in _RE_ANSWER_MARKERS:
        m = pattern.search(cleaned)
        if m:
            after = cleaned[m.end():].strip()
            if after and after[0] in ("{", "["):
                cleaned = after
                break
    if cleaned != raw:
        try:
            return json_mod.loads(cleaned)
        except (json_mod.JSONDecodeError, ValueError):
            pass
    raw = cleaned

    # Strategy 3: fenced ```json code block
    m = _RE_FENCED_JSON.search(raw)
    if m:
        try:
            return json_mod.loads(m.group(1).strip())
        except (json_mod.JSONDecodeError, ValueError):
            pass

    # Strategy 4: find the last complete JSON object/array via bracket-matching
    # (reasoning models put JSON after pages of thinking with stray brackets)
    candidates: list[tuple[int, int]] = []
    for close_ch, open_ch in [("}", "{"), ("]", "[")]:
        end_pos = raw.rfind(close_ch)
        if end_pos < 0:
            continue
        depth = 0
        for i in range(end_pos, -1, -1):
            if raw[i] == close_ch:
                depth += 1
            elif raw[i] == open_ch:
                depth -= 1
                if depth == 0:
                    candidates.append((i, end_pos))
                    break

    if not candidates:
        return None

    # Pick the outermost (earliest start) candidate
    json_start, json_end = min(candidates, key=lambda c: c[0])
    snippet = raw[json_start:json_end + 1]

    try:
        return json_mod.loads(snippet)
    except json_mod.JSONDecodeError:
        # Strategy 5: json_repair for truncated/malformed output
        try:
            import json_repair
            return json_repair.loads(snippet)
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Entity extraction — direct LLM (1 call per group, returns array of entities)
# ---------------------------------------------------------------------------
def _extract_entities_for_group(text: str, group_name: str) -> list[dict]:
    """Extract all entities for one ontology group via a single LLM call.

    Returns a list of entity dicts with keys: name, entity_type, confidence, properties.
    """
    from app.prompts import GROUP_PROMPTS, GROUP_FEW_SHOT_EXAMPLES

    schema_desc = _group_schema_prompts.get(group_name, "")
    system_prompt = GROUP_PROMPTS.get(group_name, "You are an entity extraction assistant.")
    entity_type_names = GROUP_MAP.get(group_name, [])
    example = GROUP_FEW_SHOT_EXAMPLES.get(group_name, "")

    user_prompt = (
        f"Extract ALL instances of the following entity types from the text below.\n"
        f"Entity types to extract: {', '.join(entity_type_names)}\n\n"
        f"=== ENTITY TYPE SCHEMAS ===\n{schema_desc}\n"
        f"=== END SCHEMAS ===\n\n"
    )
    if example:
        user_prompt += f"=== EXAMPLE ===\n{example}=== END EXAMPLE ===\n\n"
    user_prompt += (
        f"=== TEXT ===\n{text}\n=== END TEXT ===\n\n"
        f"Return a JSON object with a single key \"entities\" containing an array.\n"
        f"Each entity object MUST have:\n"
        f"  - \"name\": string (the entity's primary name or designation)\n"
        f"  - \"entity_type\": string (MUST be one of: {', '.join(entity_type_names)})\n"
        f"  - \"confidence\": number (0.0-1.0, your confidence in the extraction)\n"
        f"  - \"properties\": object (type-specific properties as defined in the schema above)\n\n"
        f"Extract EVERY instance mentioned in the text — there may be multiple entities of the same type.\n"
        f"Return ONLY valid JSON. Do not include entities not explicitly mentioned in the text."
    )

    t0 = time.monotonic()
    try:
        raw = _llm_call(system_prompt, user_prompt, max_tokens=4096)
        parsed = _parse_json_from_llm(raw)
        if parsed is None:
            raw_preview = (raw or "")[:500]
            logger.warning(
                "Entity extraction for group %s: failed to parse JSON. Raw preview: %s",
                group_name, raw_preview,
            )
            return []

        entities = parsed.get("entities", []) if isinstance(parsed, dict) else parsed
        if not isinstance(entities, list):
            entities = []

        elapsed = time.monotonic() - t0
        logger.info(
            "Entity extraction group=%s: %d entities in %.1fs",
            group_name, len(entities), elapsed,
        )
        return entities
    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.warning(
            "Entity extraction group=%s failed after %.1fs: %s",
            group_name, elapsed, exc,
        )
        return []


# ---------------------------------------------------------------------------
# Relationship extraction — direct LLM
# ---------------------------------------------------------------------------
def _extract_relationships(text: str, entities_context: list[dict]) -> list[dict]:
    """Extract relationships between known entities via a single LLM call."""
    from app.prompts import get_relationship_prompt

    system_prompt = get_relationship_prompt(entities_context, _relationship_prompt_context)
    user_prompt = (
        f"Analyze this text and extract relationships between the known entities:\n\n"
        f"=== TEXT ===\n{text}\n=== END TEXT ===\n\n"
        "Return a JSON object with a single key 'relationships' containing an array. "
        "Each relationship object must have: from_name, from_type, rel_type, to_name, to_type, confidence (0.0-1.0).\n"
        "Example: {\"relationships\": [{\"from_name\": \"AN/MPQ-53\", \"from_type\": \"RADAR_SYSTEM\", "
        "\"rel_type\": \"INSTALLED_ON\", \"to_name\": \"Patriot\", \"to_type\": \"PLATFORM\", \"confidence\": 0.9}]}\n"
        "Return ONLY valid JSON."
    )

    t0 = time.monotonic()
    try:
        raw = _llm_call(system_prompt, user_prompt, max_tokens=4096)
        parsed = _parse_json_from_llm(raw)
        if parsed is None:
            raw_preview = (raw or "")[:500]
            logger.warning(
                "Relationship extraction: failed to parse JSON. Raw preview: %s",
                raw_preview,
            )
            return []

        relationships = parsed.get("relationships", []) if isinstance(parsed, dict) else []
        elapsed = time.monotonic() - t0
        logger.info("Relationship extraction: %d relationships in %.1fs", len(relationships), elapsed)
        return relationships
    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.warning("Relationship extraction failed after %.1fs: %s", elapsed, exc)
        return []


# ---------------------------------------------------------------------------
# Text chunking and entity deduplication
# ---------------------------------------------------------------------------
def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks. Returns at least one chunk."""
    if len(text) <= chunk_size:
        return [text]
    overlap = min(overlap, chunk_size - 1)  # prevent infinite loop
    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += step
    return chunks


def _dedup_entities(entities: list[dict]) -> list[dict]:
    """Deduplicate entities by (name, entity_type), keeping highest confidence."""
    best: dict[tuple[str, str], dict] = {}
    for e in entities:
        key = (e.get("name", ""), e.get("entity_type", ""))
        existing = best.get(key)
        if existing is None or e.get("confidence", 0) > existing.get("confidence", 0):
            best[key] = e
    return list(best.values())


# ---------------------------------------------------------------------------
# Full extraction — chunked entities + global relationships
# ---------------------------------------------------------------------------
def _run_full_extraction(text: str) -> tuple[list[dict], list[dict]]:
    """Extract entities from text chunks, then relationships from the full text.

    Entity extraction is chunked to stay within context window limits.
    Entities are deduped by (name, entity_type) keeping highest confidence.
    Relationship extraction runs once on the full text with all entities as context.

    Returns (entities, relationships) as raw dicts.
    """
    chunks = _chunk_text(text, GRAPH_EXTRACTION_CHUNK_SIZE, GRAPH_EXTRACTION_CHUNK_OVERLAP)
    group_names = list(GROUP_MAP.keys())

    # Phase 1: Entity extraction — per chunk, all groups in parallel per chunk
    all_entities: list[dict] = []
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=len(group_names)) as pool:
        for chunk_idx, chunk in enumerate(chunks):
            futures = {
                pool.submit(_extract_entities_for_group, chunk, group): group
                for group in group_names
            }
            for future in as_completed(futures):
                group = futures[future]
                try:
                    all_entities.extend(future.result())
                except Exception as exc:
                    logger.warning("Entity extraction group=%s chunk=%d raised: %s", group, chunk_idx, exc)

    # Dedup across chunks
    raw_count = len(all_entities)
    all_entities = _dedup_entities(all_entities)

    entity_elapsed = time.monotonic() - t0
    logger.info(
        "Phase 1 complete: %d unique entities (from %d raw) across %d chunks × %d groups in %.1fs",
        len(all_entities), raw_count, len(chunks), len(group_names), entity_elapsed,
    )

    # Phase 2: Relationship extraction on FULL text with all entities as context
    t1 = time.monotonic()
    entities_context = [
        {"name": e.get("name", ""), "entity_type": e.get("entity_type", "UNKNOWN")}
        for e in all_entities
        if e.get("name")
    ]
    all_relationships = _extract_relationships(text, entities_context)
    rel_elapsed = time.monotonic() - t1

    logger.info(
        "Phase 2 complete: %d relationships in %.1fs (total: %.1fs)",
        len(all_relationships), rel_elapsed, time.monotonic() - t0,
    )

    return all_entities, all_relationships


# ---------------------------------------------------------------------------
# Mock responses
# ---------------------------------------------------------------------------
def _mock_extraction_response(
    mode: str = "entities", template_group: str | None = None
) -> ExtractionResponse:
    """Return a canned response for testing (LLM_PROVIDER=mock)."""
    mock_entities = [
        ExtractedEntityResponse(
            name="Mock Radar System", entity_type="RADAR_SYSTEM",
            confidence=0.95, properties={"designation": "AN/APG-00"},
        ),
        ExtractedEntityResponse(
            name="Mock Platform", entity_type="PLATFORM",
            confidence=0.90, properties={"platform_type": "aircraft"},
        ),
    ]
    mock_relationships = [
        ExtractedRelationshipResponse(
            from_name="Mock Platform", from_type="PLATFORM",
            rel_type="HAS_COMPONENT", to_name="Mock Radar System",
            to_type="RADAR_SYSTEM", confidence=0.85,
        ),
    ]

    if mode == "relationships":
        return ExtractionResponse(
            entities=[], relationships=mock_relationships,
            ontology_version=_ontology_version, model="mock", provider="mock",
        )
    return ExtractionResponse(
        entities=mock_entities, relationships=[],
        ontology_version=_ontology_version, model="mock", provider="mock",
    )


# ---------------------------------------------------------------------------
# /extract — single group (backward-compatible)
# ---------------------------------------------------------------------------
_VALID_MODES = {"entities", "relationships"}


@app.post("/extract", response_model=ExtractionResponse)
async def extract(request: ExtractionRequest):
    """Extract entities or relationships from document text (single group)."""
    if _templates is None:
        raise HTTPException(status_code=503, detail="Service not ready — templates not loaded")

    if request.mode not in _VALID_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown mode '{request.mode}'; valid modes: {sorted(_VALID_MODES)}",
        )

    if request.template_group is not None and request.template_group not in (_templates or {}):
        raise HTTPException(
            status_code=422,
            detail=f"Unknown template_group '{request.template_group}'; valid groups: {sorted((_templates or {}).keys())}",
        )

    if request.ontology_version and request.ontology_version != _ontology_version:
        logger.warning("Ontology version mismatch: request=%s, loaded=%s", request.ontology_version, _ontology_version)

    logger.info(
        "Extracting from document %s (%d chars, group=%s, mode=%s)",
        request.document_id, len(request.text),
        request.template_group or "(default)", request.mode,
    )

    if LLM_PROVIDER == "mock":
        return _mock_extraction_response(mode=request.mode, template_group=request.template_group)

    if request.mode == "relationships":
        rel_dicts = await run_in_threadpool(
            _extract_relationships, request.text, request.entities_context or [],
        )
        relationships = [
            ExtractedRelationshipResponse(
                from_name=r.get("from_name", ""), from_type=r.get("from_type", "UNKNOWN"),
                rel_type=r.get("rel_type", "RELATED_TO"),
                to_name=r.get("to_name", ""), to_type=r.get("to_type", "UNKNOWN"),
                confidence=float(r.get("confidence", 0.5)),
            )
            for r in rel_dicts if r.get("from_name") and r.get("to_name")
        ]
        return ExtractionResponse(
            entities=[], relationships=relationships,
            ontology_version=_ontology_version, model=LLM_MODEL, provider=LLM_PROVIDER,
        )

    # Entity extraction — direct LLM call for the group
    group = request.template_group or next(iter(GROUP_MAP))
    entity_dicts = await run_in_threadpool(_extract_entities_for_group, request.text, group)
    entities = [
        ExtractedEntityResponse(
            name=e.get("name", ""), entity_type=e.get("entity_type", "UNKNOWN"),
            confidence=float(e.get("confidence", 0.5)),
            properties=e.get("properties", {}),
        )
        for e in entity_dicts if e.get("name")
    ]

    logger.info(
        "Extracted %d entities from document %s (group=%s)",
        len(entities), request.document_id, group,
    )
    return ExtractionResponse(
        entities=entities, relationships=[],
        ontology_version=_ontology_version, model=LLM_MODEL, provider=LLM_PROVIDER,
    )


# ---------------------------------------------------------------------------
# /extract-all — all groups in parallel + relationships
# ---------------------------------------------------------------------------
class FullExtractionRequest(BaseModel):
    """Request body for the /extract-all endpoint."""
    document_id: str
    text: str


@app.post("/extract-all", response_model=ExtractionResponse)
async def extract_all(request: FullExtractionRequest):
    """Extract all entities (5 groups in parallel) and relationships in one call.

    This is ~10x faster than calling /extract for each group sequentially.
    """
    if _templates is None:
        raise HTTPException(status_code=503, detail="Service not ready — templates not loaded")

    logger.info(
        "extract-all: document %s (%d chars, %d groups)",
        request.document_id, len(request.text), len(GROUP_MAP),
    )

    if LLM_PROVIDER == "mock":
        return _mock_extraction_response()

    entity_dicts, rel_dicts = await run_in_threadpool(_run_full_extraction, request.text)

    entities = [
        ExtractedEntityResponse(
            name=e.get("name", ""), entity_type=e.get("entity_type", "UNKNOWN"),
            confidence=float(e.get("confidence", 0.5)),
            properties=e.get("properties", {}),
        )
        for e in entity_dicts if e.get("name")
    ]
    relationships = [
        ExtractedRelationshipResponse(
            from_name=r.get("from_name", ""), from_type=r.get("from_type", "UNKNOWN"),
            rel_type=r.get("rel_type", "RELATED_TO"),
            to_name=r.get("to_name", ""), to_type=r.get("to_type", "UNKNOWN"),
            confidence=float(r.get("confidence", 0.5)),
        )
        for r in rel_dicts if r.get("from_name") and r.get("to_name")
    ]

    logger.info(
        "extract-all complete: document %s — %d entities, %d relationships",
        request.document_id, len(entities), len(relationships),
    )

    return ExtractionResponse(
        entities=entities, relationships=relationships,
        ontology_version=_ontology_version, model=LLM_MODEL, provider=LLM_PROVIDER,
    )
