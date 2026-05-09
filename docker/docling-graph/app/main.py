"""Docling-Graph extraction service -- thin wrapper around run_pipeline()."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request

try:
    import networkx as nx
except ModuleNotFoundError:  # pragma: no cover - test-host fallback only
    class _FallbackDiGraph:
        def number_of_nodes(self) -> int:
            return 0

        def number_of_edges(self) -> int:
            return 0

        def has_node(self, _node: Any) -> bool:
            return False

        def has_edge(self, _source: Any, _target: Any) -> bool:
            return False

        def add_edge(self, _source: Any, _target: Any, **_kwargs: Any) -> None:
            return None

    class _FallbackNetworkX:
        DiGraph = _FallbackDiGraph

        @staticmethod
        def node_link_data(_graph: Any, edges: str = "links") -> dict[str, Any]:
            return {"nodes": [], edges: []}

    nx = _FallbackNetworkX()

# ---------------------------------------------------------------------------
# We bypass LiteLLM entirely via PipelineConfig(llm_client=OllamaChatClient(...))
# (see app.config_builder.build_pipeline_config + app.ollama_clients).
# OllamaChatClient also absorbs the legacy-fallback schema-strip behavior
# in-process via _maybe_strip_legacy_schema, so no upstream LlmBackend
# patches are required for that either.
#
# Only one upstream patch remains:
#   - NodeIDRegistry: fixes a class-name parsing bug in the library's
#     collision detection (TABLE_REF_<fingerprint> being misread as TABLE).
#     Unrelated to LLM call paths.
# ---------------------------------------------------------------------------

_logger = logging.getLogger("docling_graph.patches")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    _logger.addHandler(_h)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


# Fix TABLE_REF node ID collision: split("_")[0] yields "TABLE" for
# "TABLE_REF_<fingerprint>", causing false collision. Use rsplit so the
# class name is parsed correctly regardless of how many underscores it has.
try:
    from docling_graph.core.converters.node_id_registry import NodeIDRegistry

    _original_get_node_id = NodeIDRegistry.get_node_id

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
    _logger.info("NodeIDRegistry patched for underscore class name collision fix")
except ImportError:
    _logger.warning("Could not patch NodeIDRegistry")

# Delta system-prompt rewrite. Replaces Rules 2-4 in get_delta_batch_prompt
# so the LLM accepts BOTH structure-backed evidence AND explicit named
# mentions in prose, while still rejecting section titles + unnamed
# descriptions. See docker/docling-graph/app/prompt_rules.py and
# ontology_bundles/_shared/prompt_rules.py for the rule text and rationale.
# Section-title slippage is handled by the library's own post-extraction
# filter_entity_nodes_by_identity (delta_identity_filter_enabled=True).
from app.prompt_rules import install as _install_prompt_rules
from app.resolver_patch import install as _install_resolver_patch
from app.gleaning_patch import install as _install_gleaning_patch

_install_prompt_rules()
_install_resolver_patch()
_install_gleaning_patch()

from app.bundles import load_bundle_manifest, load_pass_template, preload_all_templates
from app._field_provenance_helpers import _primary_list_field_name
from app.config_builder import build_pipeline_config, DoclingGraphSettings
from app.evidence_gate import (
    apply_bundle_postprocessing as _apply_bundle_postprocessing,
    collect_entity_identity_examples as _collect_entity_identity_examples,
    collect_batch_evidence_text as _collect_batch_evidence_text,
    filter_pass_output_by_batch_text as _filter_pass_output_by_batch_text,
    filter_provenance_rows_by_allowed_identities as _filter_provenance_rows_by_allowed_identities,
    summarize_pass_output as _summarize_pass_output,
)
from app.provenance import (
    build_auto_field_evidence,
    build_provenance_from_context,
    build_relationship_provenance_from_delta_trace,
    synthesize_provenance_from_pass_output,
)
from app.schemas import (
    ExtractionMetadata,
    ExtractionProvenance,
    ExtractionRelationshipProvenance,
    HealthResponse,
    ExtractPassRequest,
    ExtractionFieldProvenance,
    ExtractPassResponse,
    EntityRef,  # for typing only; runtime uses attribute access
)

logger = logging.getLogger(__name__)
# uvicorn doesn't configure handlers for app-level loggers, so logger.info /
# .warning / .error from this module would be silently dropped (matches the
# pattern in app/ollama_pool_client.py and docling_graph/patches.py). Attach
# our own StreamHandler at INFO so all our loud-ERROR tags
# (GRAPH_EXTRACTION_FAILED, _PASS_DEGRADED, _LIBRARY_WARNING, _SANITIZED,
# EXTRACT_PASS_PIPELINE_ERROR, _ZERO_YIELD, "extract-pass: START/END", etc.)
# actually reach docker stdout. Idempotent — only adds the handler once.
if not any(
    isinstance(h, logging.StreamHandler) and getattr(h, "_eip_main_handler", False)
    for h in logger.handlers
):
    _main_handler = logging.StreamHandler()
    _main_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    setattr(_main_handler, "_eip_main_handler", True)
    logger.addHandler(_main_handler)
    logger.setLevel(logging.INFO)

MAX_CONCURRENT = int(os.environ.get("DOCLING_GRAPH_MAX_CONCURRENT_EXTRACTIONS", "2"))


def _table_overlay_enabled_parser() -> bool:
    """Parser-side kill switch for the deterministic table overlay
    (Mechanism A1, spec §4.3 + §5.5). Defaults to enabled. Set
    DOCLING_GRAPH_TABLE_OVERLAY_ENABLED=false to short-circuit
    extract_table_overlay() and ship table_overlay=None on every response.
    """
    return os.environ.get(
        "DOCLING_GRAPH_TABLE_OVERLAY_ENABLED", "true",
    ).lower() != "false"


def _json_for_log(value: Any, *, max_chars: int = 2000) -> str:
    try:
        text = json.dumps(value, sort_keys=True, default=str)
    except Exception:
        text = str(value)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _validate_library_surface() -> str:
    """Validate docling-graph library API. Returns version string."""
    try:
        from docling_graph import run_pipeline, PipelineConfig  # noqa: F401
        import docling_graph
        return getattr(docling_graph, "__version__", "unknown")
    except ImportError as e:
        logger.warning("docling-graph library not fully available: %s", e)
        return "unavailable"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-import every bundle's extraction schemas so per-request dispatch via
    # load_pass_template is constant-time.
    try:
        preload_all_templates()
        logger.info("Preloaded all bundle extraction schemas")
    except Exception as exc:
        logger.warning("preload_all_templates failed: %s", exc)

    app.state.pipeline_version = _validate_library_surface()
    logger.info("docling-graph library version: %s", app.state.pipeline_version)

    app.state.extraction_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    yield
    logger.info("Shutting down")


app = FastAPI(title="Docling-Graph Extraction Service", version="2.0.0", lifespan=lifespan)


from app.ollama_clients import get_docling_graph_client


@app.get("/debug/routing-metrics", tags=["diagnostics"])
def debug_routing_metrics():
    """Return per-URL request counts for the in-process LLM pool.

    Gated behind DOCLING_GRAPH_DEBUG_ENDPOINTS=true (default off) — this
    endpoint exposes backend Ollama URLs in its response, which is a small
    leak on a port that's published in compose. Enable only when running
    Gate 5 validation (Chunk 4).

    Used by Chunk 4's Gate 5 to verify fan-out across all configured Ollama
    URLs. Returns {} for the LLM role only in v1; VLM/embedding pools
    aren't used inside docling-graph.

    NOTE: pool URL list is cached per-process (see
    app.ollama_clients.get_docling_graph_client). Restart docling-graph to
    refresh after rotating OLLAMA_LLM_BASE_URLS — only the gate (DEBUG_ENDPOINTS
    env) is read per request; the URLs themselves are frozen at first call.
    """
    if os.environ.get("DOCLING_GRAPH_DEBUG_ENDPOINTS", "false").lower() not in (
        "true", "1", "yes", "on"
    ):
        raise HTTPException(status_code=404, detail="Not Found")
    try:
        client = get_docling_graph_client()
        return {"llm": client.pool.routing_metrics}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


_AD_TRACKING_DOMAINS = (
    "adroll.com",
    "adrta.com",
    "doubleclick.net",
    "googletagmanager.com",
    "googletagservices.com",
    "googleadservices.com",
    "google-analytics.com",
    "googlesyndication.com",
    "facebook.com/tr",
    "amazon-adsystem.com",
    "adservice.google",
    "scorecardresearch.com",
)

# Standalone URL or markdown link with no surrounding prose. Captures patterns
# like "[Foo](https://...)" or just "https://example.com/path" with optional
# surrounding whitespace. Leading list markers ("- ", "* ") are tolerated so
# entire navigation/sidebar lists collapse cleanly.
_PURE_LINK_LINE = __import__("re").compile(
    r"^\s*[-*]?\s*"
    r"(?:\[[^\]]*\]\([^)]+\)|https?://\S+|<https?://\S+>)"
    r"\s*$",
    flags=__import__("re").IGNORECASE | __import__("re").MULTILINE,
)

# Long unbroken token in the base64 alphabet (incl. URL-safe variants).
# Matched on whitespace-delimited tokens so prose with embedded short tokens
# (UUIDs, hex hashes, semvers) is never accidentally classified as a blob.
_BASE64_TOKEN = __import__("re").compile(r"[A-Za-z0-9+/_-]{64,}={0,2}")

# Single %XX triplet. Counted per whitespace-delimited token so a sentence
# with an occasional encoded URL doesn't false-trigger.
_PERCENT_TRIPLET = __import__("re").compile(r"%[0-9A-Fa-f]{2}")


def _contains_encoded_blob(text: str) -> bool:
    """True if any whitespace-delimited token looks like an obfuscated blob.

    Two sub-rules, both gated on a length floor of 64 chars to avoid
    misclassifying short identifiers:

    (a) **base64** — a long unbroken token in the base64 alphabet that has
        either explicit padding (``+``, ``/``, ``=``) OR mixed-case + digit
        composition. Catches ad-tracker payloads (``adroll_ad_payload=...``)
        and inline ``data:image/...;base64,...`` embeds. Excludes hex
        hashes (no uppercase, no padding), UUIDs (too short, dashed), and
        all-lowercase identifiers.

    (b) **percent-encoded URL fragment** — a long token containing six or
        more ``%XX`` triplets. Catches the residue of a tracker URL after
        docling has split it on a line break (the leading hostname, which
        Rule 1 would have matched, is now in a sibling text element while
        the continuation is just bare percent-encoded params).
    """
    if not isinstance(text, str) or not text:
        return False
    # (a) base64 candidates anywhere in the text.
    for m in _BASE64_TOKEN.finditer(text):
        tok = m.group(0)
        has_padding = ("=" in tok) or ("+" in tok) or ("/" in tok)
        has_mixed = (
            any(c.isupper() for c in tok)
            and any(c.islower() for c in tok)
            and any(c.isdigit() for c in tok)
        )
        if has_padding or has_mixed:
            return True
    # (b) percent-encoded fragment per whitespace-delimited token.
    for tok in text.split():
        if len(tok) < 64:
            continue
        if len(_PERCENT_TRIPLET.findall(tok)) >= 6:
            return True
    return False


def _looks_like_nav_or_tracking(text: str) -> bool:
    """Return True if the entire text is web cruft we should drop pre-extraction.

    Heuristics (conservative — false negatives preferred over false positives):
      1. Text contains an ad-tracking domain (high-confidence drop).
      2. Text is ONLY one or more pure markdown links / bare URLs, with no
         meaningful prose interspersed (sidebar nav, link lists, "Related"
         columns, "Share on X" rows, etc.).
      3. Text contains an obfuscated/encoded blob — either a base64 payload
         (ad payloads, embedded data URIs) or a long percent-encoded URL
         fragment (tracker-URL continuations that lost their hostname when
         docling split the URL on a line break).

    Image-description prose (label='caption' on the element, or descriptive
    sentences) does NOT match these patterns and is kept — the user explicitly
    wants those preserved.
    """
    if not isinstance(text, str) or not text.strip():
        return False

    # Rule 1: ad/tracking domain anywhere in the text — drop unconditionally.
    lowered = text.lower()
    for dom in _AD_TRACKING_DOMAINS:
        if dom in lowered:
            return True

    # Rule 2: every non-blank line is a pure link / bare URL (nav list).
    nonblank = [ln for ln in text.splitlines() if ln.strip()]
    if not nonblank:
        return False
    if all(_PURE_LINK_LINE.match(ln) for ln in nonblank):
        return True

    # Rule 3: contains an encoded blob (base64 payload OR long percent-encoded
    # URL fragment). Catches the residue that slipped past Rule 1's hostname
    # check after docling fragmented a tracker URL across line breaks.
    if _contains_encoded_blob(text):
        return True

    return False


def _sanitize_docling_document(doc: dict, stats: dict) -> dict:
    """Blank the text content of ad-tracking / navigation-list elements.

    **Key design decision (2026-04-30):** does NOT remove elements from
    texts[]. Removing an element shifts all subsequent indices, but the
    DoclingDocument format has many cross-references to text indices —
    `body.children` (handled), but ALSO `pictures[].children`,
    `tables[].children`, `groups[].children`, `furniture.children`, and
    `parent` back-references on every containee. The library's hierarchy
    validator (Pipeline stage 'Input Normalization') walks every $ref and
    fails if a parent and a child disagree about who their counterpart is.

    Instead, this implementation REPLACES the noisy text element's
    `text` and `orig` fields with empty strings AND clears the
    `hyperlink` annotation, leaving the element + all its $refs in
    place. The HybridChunker treats empty/whitespace texts as
    zero-token contributions, so they vanish from the markdown fed to
    the LLM without disturbing the document hierarchy.

    **Hyperlink note (2026-05-01):** docling stores markdown link URLs
    in a separate ``hyperlink`` annotation, not in ``text``. The chunker
    re-renders ``[text](hyperlink)`` from both fields, so a tracker URL
    survives in the chunked output even after we blank the visible text.
    To catch this we run the rule predicates against the **rendered**
    form ``[text](hyperlink)`` — same shape the chunker emits — instead
    of just the bare text. That makes Rule 2 (pure-link-line) match
    nav items where docling split the markdown link into
    text="FIFB-22" + hyperlink="https://...", and Rule 1 / Rule 3 see
    the URL even when the visible text is innocent ("Ready to win
    bigger…"). When blanking we clear ``hyperlink`` alongside
    ``text`` / ``orig`` so the chunker has nothing to render.

    KEEPS image captions (label='caption') unconditionally — user wants
    image-description prose preserved.
    """
    texts_in = list(doc.get("texts") or [])
    stats["texts_in"] = len(texts_in)

    new_texts: list = []
    blanked = 0
    for t in texts_in:
        if not isinstance(t, dict):
            new_texts.append(t)
            continue
        label = (t.get("label") or "").lower()
        if label == "caption":
            # Protected — image-description prose stays.
            new_texts.append(t)
            continue
        text_str = t.get("text") or t.get("orig") or ""
        hyperlink = t.get("hyperlink") or ""
        # Render the item the way the chunker's format_batch_markdown
        # will: `[text](hyperlink)` when hyperlinked, bare text
        # otherwise. Running rules against the rendered form catches
        # Rule 2 cases where docling split the markdown link into
        # text="FIFB-22" + hyperlink="https://..." separate fields —
        # joined by newline they fail the pure-link-line check, but in
        # rendered form they're a clean nav link.
        if hyperlink:
            rendered = f"[{text_str}]({hyperlink})"
        else:
            rendered = text_str
        if _looks_like_nav_or_tracking(rendered):
            # Element stays in texts[]; its content is blanked AND its
            # hyperlink annotation cleared. All $refs from
            # body.children / pictures[].children / etc. remain valid
            # and point to a now-empty text element with no rendered URL.
            blanked_t = dict(t)
            blanked_t["text"] = ""
            blanked_t["orig"] = ""
            if "hyperlink" in blanked_t:
                blanked_t["hyperlink"] = None
            new_texts.append(blanked_t)
            blanked += 1
            continue
        new_texts.append(t)

    stats["texts_dropped"] = blanked  # field name kept for compatibility;
                                      # semantically "blanked", not removed.

    if blanked == 0:
        return doc  # nothing changed; avoid the copy

    new_doc = dict(doc)
    new_doc["texts"] = new_texts
    return new_doc


def _render_upstream_entities_preamble(upstream_entities: list | None) -> str:
    """Render a plain-text preamble listing upstream entity refs for the LLM.

    Returns an empty string when:
    - upstream_entities is None or empty
    - DOCLING_GRAPH_UPSTREAM_PREAMBLE env var is "false" / "0" / "no"

    The returned string is designed to be prepended to the document body so all
    three docling-graph extraction contracts (direct, delta, staged) see it.
    """
    flag = os.environ.get("DOCLING_GRAPH_UPSTREAM_PREAMBLE", "true")
    if flag.lower() in ("false", "0", "no"):
        return ""

    if not upstream_entities:
        return ""

    lines = ["Upstream entities:"]
    for entity in upstream_entities:
        ref_id = getattr(entity, "ref_id", None) or entity.get("ref_id", "")
        entity_type = getattr(entity, "entity_type", None) or entity.get("entity_type", "")
        display_label = getattr(entity, "display_label", None)
        if display_label is None and isinstance(entity, dict):
            display_label = entity.get("display_label")

        if display_label:
            lines.append(f"  [{ref_id}] {entity_type} \u2014 {display_label}")
        else:
            lines.append(f"  [{ref_id}] {entity_type}")

    lines.append(
        "Only emit from_ref_id and to_ref_id values from the list above "
        "when referencing these upstream entities."
    )
    return "\n".join(lines)


def run_extraction_pass(
    docling_document_json: dict[str, Any],
    template_cls: type,
    upstream_entities: list | None = None,
    pass_name: str | None = None,
    temperature: float | None = None,
    llm_batch_token_size: int | None = None,
    model: str | None = None,
) -> Any:
    """Run docling-graph pipeline for a SINGLE fixed-template pass.

    Mirrors the deleted run_extraction_pipeline() exactly but takes the template
    class directly from the bundle loader instead of resolving it from a dynamic
    definition blob.

    ``pass_name`` is threaded through to :func:`build_pipeline_config` so
    per-pass quality-gate overrides (spec §4.6) apply — e.g. system_links
    drops min_instances to 1.

    Path B preamble injection: if DOCLING_GRAPH_UPSTREAM_PREAMBLE is enabled and
    upstream_entities is non-empty, the preamble is appended to the document's
    texts array and prepended to body.children so that export_to_markdown()
    includes it at the top of the document body for all three extraction contracts.
    """
    import shutil
    import tempfile
    from docling_graph import run_pipeline

    # --- Empty-source short-circuit --------------------------------------
    # If the DoclingDocument has no body content AND no registered items
    # (texts / pictures / tables), the library's LlmBackend fails fast with
    # "Markdown is empty for DoclingDocument. Cannot proceed." and bubbles a
    # 500. That thrashes the worker's retry loop for true-negative cases
    # (image-only inputs Docling didn't decompose — see the Docling service's
    # synthetic-picture fallback). Return a clean empty response instead.
    def _is_empty(doc: dict) -> bool:
        body_children = (doc.get("body") or {}).get("children") or []
        return (
            not body_children
            and not doc.get("texts")
            and not doc.get("pictures")
            and not doc.get("tables")
        )

    # Pre-extraction sanitizer: blank web cruft (ad-tracking URLs, navigation
    # link lists, encoded blobs — see _looks_like_nav_or_tracking for the
    # full rule set) from the DoclingDocument's texts[] array before chunking.
    # KEEPS image-description prose (label="caption") and all real document
    # content. Opt out via DOCLING_GRAPH_SANITIZE_INPUT=false. The blanked
    # count is recorded in diagnostics so an operator can verify the
    # heuristic isn't too aggressive (surfaced in the notebook outcome
    # tracker's `sanit` column).
    sanitize_stats: dict[str, int] = {"texts_in": 0, "texts_dropped": 0}
    _settings_for_sanitize = DoclingGraphSettings()
    if getattr(_settings_for_sanitize, "docling_graph_sanitize_input", True):
        docling_document_json = _sanitize_docling_document(
            docling_document_json, sanitize_stats
        )
        if sanitize_stats["texts_dropped"] > 0:
            logger.info(
                "GRAPH_EXTRACTION_SANITIZED pass=%s texts_in=%d texts_dropped=%d "
                "(filtered web-cruft texts before chunking; image captions preserved)",
                pass_name, sanitize_stats["texts_in"], sanitize_stats["texts_dropped"],
            )

    # Section-aware table-fact synthesis (table_facts.py + alias_map.py) was
    # built and validated in the 2026-05-06 plan, then reverted here after
    # cross-pass measurement showed the cost (+10-30% wall on docs with
    # variants tables, +output truncation pressure) outweighed the benefit
    # (+2 ✓ exact on airframe for 1 of 21 corpus docs; no improvement on
    # kinematics/speed_timing; propulsion fix landed but unverified). Modules
    # remain on disk in app/_table_facts.py + app/_alias_map.py with full
    # tests; re-enable when the corpus has more variants-table documents to
    # amortize the maintenance cost. See TODO #84.

    if _is_empty(docling_document_json):
        logger.warning(
            "extract-pass short-circuit: DoclingDocument has no extractable content "
            "(body.children/texts/pictures/tables all empty). Returning empty pass_output "
            "with diagnostic marker instead of calling run_pipeline."
        )
        import networkx as _nx

        class _EmptySourceContext:
            template_instance = None
            extracted_models: list = []
            knowledge_graph = _nx.DiGraph()

            class _Meta:
                node_count = 0
                edge_count = 0
                node_types: dict = {}
                edge_types: dict = {}

            graph_metadata = _Meta()

        ctx = _EmptySourceContext()
        ctx._upstream_preamble_applied = False  # no chance to apply — source was empty
        ctx._chunk_to_self_refs = None  # no doc, nothing to map
        ctx._chunk_to_evidence_units = {}  # no doc, nothing to map
        ctx._delta_trace = {
            "empty_source": True,
            "reason": "docling_document_has_no_extractable_content",
            "body_children": 0,
            "texts": 0,
            "pictures": 0,
            "tables": 0,
            "library_log": "",
            "suggestion": (
                "Upstream Docling conversion produced an empty body. Image-only "
                "inputs that the PDF pipeline didn't decompose are the typical "
                "cause; see docker/docling/app/converter.py synthetic-picture "
                "fallback and TODO #29 for VLM routing."
            ),
        }
        return ctx
    # ----------------------------------------------------------------------

    # --- Path B injection -------------------------------------------------
    preamble = _render_upstream_entities_preamble(upstream_entities)
    preamble_applied = False

    if preamble:
        new_index = len(docling_document_json.get("texts", []))
        preamble_item = {
            "self_ref": f"#/texts/{new_index}",
            "parent": {"$ref": "#/body"},
            "children": [],
            "content_layer": "body",
            "label": "text",
            "prov": [],
            "orig": preamble,
            "text": preamble,
        }
        # Append to texts (preserves existing indices so RefItem lookups don't break)
        new_texts = list(docling_document_json.get("texts", [])) + [preamble_item]
        # Prepend a body-child ref so the preamble appears first in markdown
        existing_body = docling_document_json.get("body", {})
        new_body = dict(existing_body)
        new_body["children"] = [
            {"$ref": f"#/texts/{new_index}"}
        ] + list(new_body.get("children", []))
        # Copy-on-write: never mutate the caller's dict
        docling_document_json = dict(docling_document_json)
        docling_document_json["texts"] = new_texts
        docling_document_json["body"] = new_body
        preamble_applied = True
    # ----------------------------------------------------------------------

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(docling_document_json, tmp, ensure_ascii=False, default=str)
        tmp_path = tmp.name

    # Per-call debug dir; read back delta_trace.json afterwards and stash
    # on context so extract_pass can surface it as response.diagnostics.
    debug_dir = tempfile.mkdtemp(prefix="docgraph-debug-")

    try:
        config = build_pipeline_config(
            source=tmp_path,
            template_class=template_cls,
            pass_name=pass_name,
            debug_dir=debug_dir,
            temperature_override=temperature,
            llm_batch_token_size_override=llm_batch_token_size,
            model_override=model,
        )

        # Capture the library's print() + logging output to stdout/stderr during
        # run_pipeline so the response can surface the exact fallback chain
        # ([DeltaExtraction] → "Warning: ...produced no JSON" → "falling back
        # to direct" → "LiteLLMClient returned empty or all-null JSON" → etc.)
        # to the notebook and worker. A Tee keeps the original streams flowing
        # to the container logs so uvicorn's log aggregation doesn't go dark.
        library_log_buf = io.StringIO()

        class _Tee:
            def __init__(self, *streams):
                self._streams = streams
            def write(self, data):
                for s in self._streams:
                    try:
                        s.write(data)
                    except Exception:
                        pass
                return len(data) if isinstance(data, str) else 0
            def flush(self):
                for s in self._streams:
                    try:
                        s.flush()
                    except Exception:
                        pass

        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(original_stdout, library_log_buf)
        sys.stderr = _Tee(original_stderr, library_log_buf)
        pipeline_error: Exception | None = None
        try:
            context = run_pipeline(config)
        except Exception as exc:
            # Library raised PipelineError (or anything else). Build a stub
            # context so the service returns a clean 200 with diagnostics
            # instead of a 500 that tanks the worker's retry loop. The
            # notebook + worker see why the library bailed via the captured
            # log + pipeline_error marker in _delta_trace.
            pipeline_error = exc
            logger.error(
                "GRAPH_EXTRACTION_FAILED pass=%s exc_type=%s exc_msg=%s "
                "markdown_chars=%d library_log_tail=%r — soft-failing to stub "
                "context; downstream pass_output will be empty.",
                pass_name,
                type(exc).__name__,
                str(exc),
                len(json.dumps(docling_document_json)) if isinstance(docling_document_json, dict) else 0,
                library_log_buf.getvalue()[-500:],
            )
            import networkx as _nx

            class _PipelineFailureContext:
                template_instance = None
                extracted_models: list = []
                knowledge_graph = _nx.DiGraph()
                docling_document = None

                class _Meta:
                    node_count = 0
                    edge_count = 0
                    node_types: dict = {}
                    edge_types: dict = {}

                graph_metadata = _Meta()

            context = _PipelineFailureContext()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

        # docling-graph's stages don't set ``context.template_instance``
        # — they populate ``extracted_models``. Promote the single
        # extracted pass-root to ``template_instance`` so the response
        # builder serializes the populated Pydantic object (not the
        # node_link_data fallback).
        extracted = getattr(context, "extracted_models", None)
        if isinstance(extracted, list) and extracted:
            context.template_instance = extracted[0]

        try:
            context._upstream_preamble_applied = preamble_applied
        except AttributeError:
            pass

        # Build chunk_index → [doc_item.self_ref, ...] mapping so the
        # provenance resolver can turn chunk indexes (the only location
        # identity the delta-IR normalizer attaches) into real
        # DoclingDocument self_refs.
        #
        # PRIMARY: doc_processor.last_chunk_metadata (debug-independent —
        # always set by Task 4's strategy_ops update). Walk:
        # context.extractor.doc_processor.
        extractor = getattr(context, "extractor", None)
        doc_processor = getattr(extractor, "doc_processor", None)
        chunk_metadata = getattr(doc_processor, "last_chunk_metadata", None) or []

        chunk_to_self_refs: dict[int, list[str]] = {}
        chunk_to_evidence_units: dict[int, list[dict]] = {}
        for cmeta in chunk_metadata:
            cid = cmeta.get("chunk_id")
            if cid is None:
                continue
            refs = cmeta.get("self_refs") or []
            units = cmeta.get("evidence_units") or []
            chunk_to_self_refs[int(cid)] = [r for r in refs if isinstance(r, str)]
            chunk_to_evidence_units[int(cid)] = list(units)

        # FALLBACK: trace events (debug-only, but cross-check / diagnostic).
        if not chunk_to_self_refs:
            trace_data = getattr(context, "trace_data", None)
            trace_events = getattr(trace_data, "events", None) or []
            chunk_to_self_refs, chunk_to_evidence_units = _chunk_maps_from_trace(trace_events)
            if chunk_to_self_refs:
                logger.info(
                    "provenance source: trace fallback (debug mode). "
                    "doc_processor.last_chunk_metadata was empty — verify Task 4 wiring."
                )

        if not chunk_to_self_refs:
            logger.warning(
                "no chunk metadata available from doc_processor or trace — "
                "provenance will be empty. Check that strategy_ops set "
                "doc_processor.last_chunk_metadata after chunking."
            )
        try:
            context._chunk_to_self_refs = chunk_to_self_refs
            context._chunk_to_evidence_units = chunk_to_evidence_units
        except AttributeError:
            pass

        # Read library-level trace (batch_errors, quality_gate, identity_filter,
        # path_counts, merge_stats, diagnostics) for response surfacing.
        trace: dict | None = None
        for candidate in (
            os.path.join(debug_dir, "debug", "delta_trace.json"),
            os.path.join(debug_dir, "delta_trace.json"),
            os.path.join(debug_dir, "debug", "trace_data.json"),
        ):
            if os.path.exists(candidate):
                try:
                    with open(candidate, encoding="utf-8") as f:
                        trace = json.load(f)
                    break
                except Exception as exc:
                    logger.warning("Failed to load debug trace %s: %s", candidate, exc)
        if trace is None:
            trace = {}

        # Load delta_merged_graph.json for relationship provenance. This file
        # carries the normalized IR's {"nodes": [...], "relationships": [...]}
        # with per-relationship provenance dicts (evidence_ids, self_refs).
        # delta_trace.json has stats/diagnostics only — NOT relationship data.
        merged_graph: dict | None = None
        for mg_candidate in (
            os.path.join(debug_dir, "debug", "delta_merged_graph.json"),
            os.path.join(debug_dir, "delta_merged_graph.json"),
        ):
            if os.path.exists(mg_candidate):
                try:
                    with open(mg_candidate, encoding="utf-8") as f:
                        merged_graph = json.load(f)
                    break
                except Exception as exc:
                    logger.warning(
                        "Failed to load delta_merged_graph %s: %s", mg_candidate, exc
                    )
        try:
            context._delta_merged_graph = merged_graph
        except AttributeError:
            pass

        captured_log = library_log_buf.getvalue()
        trace["library_log"] = captured_log
        # Sanitize stats are exposed in the response diagnostics so the
        # notebook outcome tracker and operator log lines can show how
        # aggressive the cruft filter was on each pass.
        trace["input_sanitize"] = {
            "texts_in": sanitize_stats.get("texts_in", 0),
            "texts_dropped": sanitize_stats.get("texts_dropped", 0),
        }
        if pipeline_error is not None:
            trace["pipeline_error"] = {
                "type": type(pipeline_error).__name__,
                "message": str(pipeline_error),
            }

        # Promote silent library-level warnings to ERROR so the service log
        # surfaces every failure mode the upstream library prints to stdout
        # (these otherwise stay buried inside captured_log and never reach
        # uvicorn's structured logger). Counts emitted alongside the message
        # let an operator grep for `GRAPH_EXTRACTION_LIBRARY_WARNING`
        # without re-deriving the underlying soft-fail pattern.
        library_warning_signatures = (
            ("Quality gate failed", "quality_gate_failed"),
            ("No valid JSON returned from LLM", "no_valid_json"),
            ("Warning: Structured output appears sparse", "structured_output_sparse"),
            ("Warning: Structured output failed", "structured_output_failed"),
            ("falling back to direct", "falling_back_to_direct"),
            ("LiteLLMClient returned empty", "litellm_returned_empty"),
            ("retrying legacy", "retrying_legacy"),
            # Patched by us at orchestrator.py:518 — surfaces silent
            # deadlocks in the parallel-batch dispatcher. Only fires when
            # parallel_workers > 1 AND a future hangs past the per-batch
            # ceiling (DOCLING_GRAPH_BATCH_HARD_TIMEOUT_SECONDS, default 1h).
            ("BATCH_HARD_TIMEOUT", "batch_hard_timeout"),
        )
        any_library_warning = False
        for signature, tag in library_warning_signatures:
            occurrences = captured_log.count(signature)
            if occurrences > 0:
                any_library_warning = True
                logger.error(
                    "GRAPH_EXTRACTION_LIBRARY_WARNING pass=%s tag=%s count=%d "
                    "signature=%r — library soft-failed silently; pass_output "
                    "will be empty or partial.",
                    pass_name, tag, occurrences, signature,
                )

        # When the pass exhibited any failure indicator (run_pipeline raised,
        # OR a library warning fired), embed the library's per-batch trace
        # files in the response so an operator can inspect the EXACT prompt
        # + LLM output that caused the failure. Without this, the trace
        # files get cleaned up by the finally block below and there's no
        # forensic surface for debugging failed batches. Only embeds on
        # failures to keep response size bounded for the success path.
        had_failure = pipeline_error is not None or any_library_warning
        if had_failure:
            batch_traces: dict[str, dict] = {}
            debug_subdir = os.path.join(debug_dir, "debug")
            search_dirs = [debug_subdir, debug_dir]
            for d in search_dirs:
                if not os.path.isdir(d):
                    continue
                try:
                    for fname in sorted(os.listdir(d)):
                        if not fname.startswith("delta_batch_"):
                            continue
                        if not fname.endswith(".json"):
                            continue
                        fpath = os.path.join(d, fname)
                        try:
                            with open(fpath, encoding="utf-8") as bf:
                                batch_traces[fname] = json.load(bf)
                        except Exception as bexc:
                            batch_traces[fname] = {"_load_error": str(bexc)}
                except Exception as exc:
                    logger.warning(
                        "extract-pass: failed to enumerate batch traces in %s: %s",
                        d, exc,
                    )
            if batch_traces:
                trace["failed_batch_traces"] = batch_traces
                logger.info(
                    "extract-pass: embedded %d batch trace(s) in diagnostics "
                    "for forensic inspection (pass=%s, document_id=%s)",
                    len(batch_traces), pass_name,
                    getattr(context, "_document_id_for_logging", "?"),
                )

        try:
            context._delta_trace = trace
        except AttributeError:
            pass
        return context
    finally:
        os.unlink(tmp_path)
        shutil.rmtree(debug_dir, ignore_errors=True)


def _trace_event_payload(evt):
    """Return (event_type, payload) from a trace event, supporting the
    TraceEvent dataclass shape used by docling_graph.pipeline.trace and
    the dict/tuple fallbacks some test harnesses emit. Returns
    (None, None) for shapes we don't recognize."""
    if hasattr(evt, "event_type") and hasattr(evt, "payload"):
        return evt.event_type, evt.payload
    if isinstance(evt, tuple) and len(evt) >= 3:
        return evt[0], evt[2]
    if isinstance(evt, dict):
        return evt.get("event") or evt.get("name"), evt.get("payload") or evt
    return None, None


def _chunk_maps_from_trace(trace_events) -> tuple[dict[int, list[str]], dict[int, list[dict]]]:
    """Build (chunk_to_self_refs, chunk_to_evidence_units) from chunk_created
    trace events in a single pass.

    Authoritative provenance source — uses the EXACT chunks the LLM saw.
    Replaces re-chunking. Both maps share the same iteration / filtering
    so they cannot drift out of sync."""
    refs_map: dict[int, list[str]] = {}
    units_map: dict[int, list[dict]] = {}
    for evt in trace_events or []:
        name, payload = _trace_event_payload(evt)
        if name != "chunk_created" or not isinstance(payload, dict):
            continue
        cid = payload.get("chunk_id")
        if cid is None:
            continue
        cid_int = int(cid)
        refs = payload.get("self_refs")
        if isinstance(refs, list):
            refs_map[cid_int] = [r for r in refs if isinstance(r, str)]
        units = payload.get("evidence_units")
        if isinstance(units, list):
            units_map[cid_int] = list(units)
    return refs_map, units_map


def _build_chunk_to_self_refs_map(docling_document: Any) -> dict[int, list[str]] | None:
    """DIAGNOSTIC-ONLY fallback. Re-chunks the document with a default
    HybridChunker.

    DO NOT use as a normal provenance source — the re-chunked boundaries
    do NOT match the extraction-time chunks (different default max_tokens,
    independent merge_peers state). Trace-event-derived
    `_chunk_to_self_refs_from_trace` is authoritative.
    """
    if docling_document is None:
        return None
    try:
        from docling.chunking import HybridChunker
    except ImportError as exc:
        logger.warning("HybridChunker import failed; provenance self_refs unavailable: %s", exc)
        return None

    try:
        chunker = HybridChunker()  # diagnostic-only: re-chunk for fallback only
        out: dict[int, list[str]] = {}
        for i, chunk in enumerate(chunker.chunk(dl_doc=docling_document)):
            meta = getattr(chunk, "meta", None)
            doc_items = getattr(meta, "doc_items", None) if meta is not None else None
            refs = []
            for item in doc_items or []:
                ref = getattr(item, "self_ref", None)
                if isinstance(ref, str) and ref:
                    refs.append(ref)
            out[i] = refs
        return out
    except Exception as exc:
        logger.warning("Re-chunking for provenance self_refs failed: %s", exc)
        return None


def _should_run_validation_pass() -> bool:
    val = os.environ.get("DOCLING_GRAPH_VALIDATION_PASS_ENABLED", "true")
    return val.lower() in ("true", "1", "yes")


def _apply_validation_edges(graph: nx.DiGraph, new_edges: list[dict[str, Any]]) -> int:
    added = 0
    for edge in new_edges:
        source, target = edge.get("source"), edge.get("target")
        if source and target and graph.has_node(source) and graph.has_node(target) and not graph.has_edge(source, target):
            graph.add_edge(source, target, label=edge.get("label", "RELATED_TO"),
                          confidence=edge.get("confidence", 0.5), _source="validation_pass")
            added += 1
    return added


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    from app.bundles import _template_cache  # noqa: PLC0415 — local import to avoid circular
    schema_count = len(_template_cache)
    return HealthResponse(
        schema_count=schema_count,
        extraction_contract=os.environ.get("DOCLING_GRAPH_EXTRACTION_CONTRACT", "delta"),
        pipeline_version=request.app.state.pipeline_version,
    )


@app.post("/extract-pass", response_model=ExtractPassResponse)
async def extract_pass(request: Request, body: ExtractPassRequest):
    """Fixed-template extraction for ONE pass from a bundle. Spec §5.9."""
    semaphore = request.app.state.extraction_semaphore
    if semaphore is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # 1. Resolve bundle + pass
    try:
        manifest = load_bundle_manifest(body.bundle_key)
    except (KeyError, FileNotFoundError):
        raise HTTPException(status_code=404, detail=f"Unknown bundle_key: {body.bundle_key}")

    pass_def = None
    for p in manifest.get("passes", []):
        if p.get("name") == body.pass_name:
            pass_def = p
            break
    if pass_def is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown pass_name: {body.pass_name} in bundle {body.bundle_key}",
        )

    # 2. Validate input_mode compatibility
    input_mode = pass_def.get("input_mode")
    if input_mode == "document_only" and body.upstream_entities:
        raise HTTPException(
            status_code=400,
            detail=f"document_only pass {body.pass_name} received upstream_entities",
        )
    if input_mode == "document_plus_entity_refs" and not body.upstream_entities:
        raise HTTPException(
            status_code=400,
            detail=f"document_plus_entity_refs pass {body.pass_name} missing upstream_entities",
        )

    # 3. Load the pre-imported fixed template
    try:
        template_cls = load_pass_template(body.bundle_key, body.pass_name)
    except (ImportError, AttributeError, KeyError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load template for {body.bundle_key}/{body.pass_name}: {exc}",
        )

    # 4. Run the pipeline
    #
    # The docling-graph library emits validation-salvage warnings
    # (e.g. `specifications -> N -> value: Input should be a valid string`)
    # without naming the pass. Bracket the call with bundle/pass context
    # so operators can correlate those warnings to the offending pass.
    upstream_ref_count = len(body.upstream_entities) if body.upstream_entities else 0
    logger.info(
        "extract-pass: START bundle=%s pass=%s input_mode=%s document_id=%s upstream_ref_count=%d",
        body.bundle_key, body.pass_name, pass_def.get("input_mode"),
        body.document_id, upstream_ref_count,
    )
    # Log the effective extraction-relevant config at pass start so each
    # pass's behavior is reproducible from the logs alone (no need to
    # cross-reference docker-compose / env / config_builder defaults to
    # work out which knob was active for a given run). One line per pass;
    # operators can grep `extract-pass: CONFIG` to audit drift.
    try:
        from app.config import settings as _service_settings
        _dg_settings = DoclingGraphSettings()
        logger.info(
            "extract-pass: CONFIG pass=%s model=%s force_json_mode=%s "
            "temperature=%s max_tokens=%s truncation_retry_max_tokens=%s "
            "max_output_tokens=%s context_limit=%s batch_token_size=%s "
            "parallel_workers=%s gleaning_enabled=%s "
            "gleaning_max_passes=%s sanitize_input=%s "
            "structured_output_threshold_chars=%s llm_urls=%s",
            body.pass_name,
            body.model if body.model is not None else _dg_settings.docling_graph_llm_model,
            getattr(_service_settings, "force_json_mode", "?"),
            (
                body.temperature
                if body.temperature is not None
                else _dg_settings.docling_graph_llm_temperature
            ),
            _dg_settings.docling_graph_llm_max_tokens,
            _dg_settings.docling_graph_llm_truncation_retry_max_tokens,
            _dg_settings.docling_graph_llm_max_output_tokens,
            _dg_settings.docling_graph_llm_context_limit,
            (
                body.llm_batch_token_size
                if body.llm_batch_token_size is not None
                else _dg_settings.docling_graph_llm_batch_token_size
            ),
            _dg_settings.docling_graph_parallel_workers,
            _dg_settings.docling_graph_gleaning_enabled,
            _dg_settings.docling_graph_gleaning_max_passes,
            getattr(_dg_settings, "docling_graph_sanitize_input", "?"),
            getattr(_service_settings, "structured_output_threshold_chars", "?"),
            _dg_settings.get_ollama_llm_urls(),
        )
    except Exception as _cfg_log_exc:
        logger.warning(
            "extract-pass: failed to log effective config (%s); "
            "continuing pass without config snapshot.",
            _cfg_log_exc,
        )
    # Sentinels: -1 means extraction did not complete (failure path).
    node_count_for_log = -1
    edge_count_for_log = -1
    raw_node_count_for_log = -1
    raw_edge_count_for_log = -1
    service_dropped_for_log = -1

    # Mechanism A1 (spec §4.3 + §5.5): deterministic table overlay parse.
    # Runs on the raw DoclingDocument BEFORE the LLM extraction. Sanitize
    # only blanks texts[]; tables[] are untouched, so parsing the overlay
    # on body.docling_document_json is equivalent to parsing the
    # post-sanitize doc seen by the LLM. Catch-and-continue: a parser
    # failure leaves table_overlay_obj=None and records repr(exc) into
    # diagnostics — the LLM extraction still runs.
    table_overlay_obj = None
    overlay_stats: dict[str, Any] = {
        "kill_switch_active_parser": not _table_overlay_enabled_parser(),
        "tables_processed": 0,
    }
    if _table_overlay_enabled_parser():
        try:
            from app._table_facts import extract_table_overlay
            import typing as _typing
            # Resolve the EXACT TableOverlay class that
            # ExtractPassResponse references in its annotation. In
            # production this is the single class from sys.modules
            # ['app.schemas']; under the test conftest's importlib
            # swap, ExtractPassResponse may have been loaded against a
            # now-stale schemas module while a fresh `from app.schemas
            # import TableOverlay` would resolve to a different class
            # object. Pulling the class off the model_fields annotation
            # itself sidesteps that gap — the JSON round-trip rebinds
            # the parser's empty-class-mirror instance to the response
            # class. Spec §5.4 mandates JSON travels parser↔worker so
            # this round-trip is faithful.
            _resp_overlay_args = _typing.get_args(
                ExtractPassResponse.model_fields["table_overlay"].annotation
            )
            _ResponseTableOverlay = next(
                (a for a in _resp_overlay_args if a is not type(None)),
                None,
            )
            parsed_overlay, parser_stats = extract_table_overlay(
                body.docling_document_json,
            )
            if (
                _ResponseTableOverlay is not None
                and not isinstance(parsed_overlay, _ResponseTableOverlay)
            ):
                parsed_overlay = _ResponseTableOverlay.model_validate(
                    parsed_overlay.model_dump(mode="json"),
                )
            overlay_stats.update(parser_stats)
            # Spec §5.4: response MUST carry table_overlay=None (NOT an
            # empty TableOverlay object) when no qualifying table found.
            # The parser internally returns an empty TableOverlay() so the
            # call site signature is uniform; we collapse "empty" → None
            # at the response boundary so downstream worker logic and
            # diagnostics treat "no overlay" identically regardless of
            # whether parsing succeeded with no data or kill switch was
            # set.
            is_empty = (
                not parsed_overlay.alias_map_by_entity_type
                and not parsed_overlay.facts
                and not parsed_overlay.cross_entity_hints
            )
            table_overlay_obj = None if is_empty else parsed_overlay
            if table_overlay_obj is not None:
                overlay_stats["alias_map_size"] = sum(
                    len(m) for m in table_overlay_obj.alias_map_by_entity_type.values()
                )
                overlay_stats["facts_count"] = len(table_overlay_obj.facts)
                overlay_stats["cross_entity_hints_count"] = len(
                    table_overlay_obj.cross_entity_hints,
                )
            else:
                overlay_stats["alias_map_size"] = 0
                overlay_stats["facts_count"] = 0
                overlay_stats["cross_entity_hints_count"] = 0
        except Exception as exc:
            logger.warning(
                "extract_table_overlay failed: %s — continuing with table_overlay=None",
                exc,
            )
            table_overlay_obj = None
            overlay_stats["extract_failure"] = repr(exc)

    try:
        async with semaphore:
            try:
                context = await asyncio.to_thread(
                    run_extraction_pass,
                    body.docling_document_json,
                    template_cls,
                    body.upstream_entities,
                    body.pass_name,
                    body.temperature,
                    body.llm_batch_token_size,
                    body.model,
                )
            except Exception as exc:
                logger.exception(
                    "extract-pass pipeline failed for document_id=%s bundle=%s pass=%s",
                    body.document_id, body.bundle_key, body.pass_name,
                )
                raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")
        # 5. Build response — mirror the /extract-all metadata shape
        graph = context.knowledge_graph
        meta = context.graph_metadata
        metadata = ExtractionMetadata(
            node_count=getattr(meta, "node_count", graph.number_of_nodes()),
            edge_count=getattr(meta, "edge_count", graph.number_of_edges()),
            node_types=getattr(meta, "node_types", {}),
            edge_types=getattr(meta, "edge_types", {}),
            extraction_contract=os.environ.get("DOCLING_GRAPH_EXTRACTION_CONTRACT", "delta"),
            # --- Plan 1 — appended below. -----------------------------------
            upstream_ref_count=upstream_ref_count,
            upstream_preamble_applied=getattr(context, "_upstream_preamble_applied", False),
        )
        node_count_for_log = metadata.node_count
        edge_count_for_log = metadata.edge_count

        # pass_output is the dumped template instance so the worker can re-parse it
        pass_output: dict[str, Any] = {}
        template_instance = getattr(context, "template_instance", None)
        if template_instance is not None and hasattr(template_instance, "model_dump"):
            pass_output = template_instance.model_dump(mode="json")
        else:
            # Fallback: serialize the graph as the pass_output until Chunk 3
            # wires template_instance through the docling-graph pipeline
            import networkx as nx
            pass_output = {"graph": nx.node_link_data(graph, edges="links")}
        raw_pass_counts = _summarize_pass_output(pass_output, template_cls)
        raw_node_count_for_log = raw_pass_counts["node_count"]
        raw_edge_count_for_log = raw_pass_counts["edge_count"]
        raw_identity_examples = _collect_entity_identity_examples(pass_output, template_cls)

        # Phase 8 Task 51: per-entity-instance provenance payload. Nodes
        # whose element_uid cannot be resolved are dropped with WARNING
        # inside the helper — the response only carries chunk-linkable
        # rows.
        try:
            provenance_rows = build_provenance_from_context(
                context, ExtractionProvenance,
                chunk_to_self_refs=getattr(context, "_chunk_to_self_refs", None),
                chunk_to_evidence_units=getattr(context, "_chunk_to_evidence_units", None),
            )
        except Exception as exc:
            logger.warning(
                "extract-pass: provenance builder failed for document_id=%s bundle=%s pass=%s: %s",
                body.document_id, body.bundle_key, body.pass_name, exc,
            )
            provenance_rows = []

        # Fallback: when the library's salvage path (missing_root_instance
        # / empty_output → legacy prompt-schema retry → direct mode) drops
        # chunk-tracking attributes from the knowledge_graph, the primary
        # path returns []. Synthesize one row per pass_output entity so
        # downstream MENTIONED_IN / EXTRACTED_FROM edges can still fire.
        if not provenance_rows:
            provenance_rows = synthesize_provenance_from_pass_output(
                pass_output,
                template_cls,
                chunk_to_self_refs=getattr(context, "_chunk_to_self_refs", None),
                provenance_cls=ExtractionProvenance,
            )
            if provenance_rows:
                logger.info(
                    "extract-pass: synthesized %d provenance rows from "
                    "pass_output for document_id=%s bundle=%s pass=%s "
                    "(library knowledge_graph path yielded none)",
                    len(provenance_rows), body.document_id, body.bundle_key, body.pass_name,
                )

        evidence_text = _collect_batch_evidence_text(body.docling_document_json)
        filtered_pass_output, service_filter_stats, allowed_identities = _filter_pass_output_by_batch_text(
            pass_output,
            template_cls,
            evidence_text,
        )
        pass_output = filtered_pass_output
        pass_output, postprocess_stats = _apply_bundle_postprocessing(
            body.bundle_key,
            body.pass_name,
            pass_output,
            evidence_text,
            body.upstream_entities,
        )
        filtered_counts = _summarize_pass_output(pass_output, template_cls)
        filtered_identity_examples = _collect_entity_identity_examples(pass_output, template_cls)
        dropped_by_field = (service_filter_stats or {}).get("dropped_entities_by_field", {})
        service_dropped_for_log = (
            sum(dropped_by_field.values()) if isinstance(dropped_by_field, dict) else 0
        )
        metadata.node_count = filtered_counts["node_count"]
        metadata.edge_count = filtered_counts["edge_count"]
        metadata.node_types = filtered_counts["node_types"]
        metadata.edge_types = filtered_counts["edge_types"]
        node_count_for_log = metadata.node_count
        edge_count_for_log = metadata.edge_count
        provenance_rows = _filter_provenance_rows_by_allowed_identities(
            provenance_rows,
            allowed_identities,
        )
        diagnostics = getattr(context, "_delta_trace", None)
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        diagnostics["service_identity_gate"] = service_filter_stats or {
            "dropped_entities_by_field": {},
            "dropped_entity_examples": {},
        }
        diagnostics["service_identity_gate"]["evidence_text_nonempty"] = bool(evidence_text)
        diagnostics["service_pre_filter_counts"] = raw_pass_counts
        diagnostics["service_pre_filter_identity_examples"] = raw_identity_examples
        diagnostics["service_postprocess"] = postprocess_stats or {}
        diagnostics["service_post_filter_counts"] = filtered_counts
        diagnostics["service_post_filter_identity_examples"] = filtered_identity_examples
        # Mechanism A1: surface the parser-side overlay stats so an
        # operator (or downstream worker) can verify per-pass whether
        # the kill switch fired, how many tables qualified, alias_map
        # / facts / hints sizes, and any extract-time failure repr.
        diagnostics["service_table_overlay"] = overlay_stats
        if "path_counts" in diagnostics:
            diagnostics["raw_path_counts"] = diagnostics.get("path_counts", {})
        diagnostics["path_counts"] = filtered_counts["path_counts"]
        context._delta_trace = diagnostics
        if service_dropped_for_log > 0:
            logger.info(
                "extract-pass: IDENTITY_FILTER bundle=%s pass=%s document_id=%s "
                "raw_node_count=%d filtered_node_count=%d dropped=%d "
                "raw_examples=%s kept_examples=%s dropped_examples=%s",
                body.bundle_key, body.pass_name, body.document_id,
                raw_node_count_for_log, node_count_for_log, service_dropped_for_log,
                _json_for_log(raw_identity_examples),
                _json_for_log(filtered_identity_examples),
                _json_for_log(
                    (service_filter_stats or {}).get("dropped_entity_examples", {})
                ),
            )

        # Phase 3 (post-LLM-quote refactor): deterministic per-field
        # provenance. For each non-None populated field on each primary
        # entity, substring-match the value against the DoclingDocument's
        # text elements and attach matching chunks as evidence. No LLM
        # involvement — robust against extraction salvage / model
        # compliance issues. Falls back to batch-level attribution
        # when no chunk literally contains the value's string form.
        field_provenance_rows: list[ExtractionFieldProvenance] = []
        primary_types = pass_def.get("primary_entity_types", []) or []
        primary_type = primary_types[0] if primary_types else None
        list_field_name: str | None = None
        if primary_type:
            try:
                list_field_name = _primary_list_field_name(template_cls, primary_type)
            except ValueError:
                list_field_name = None
        if list_field_name:
            # Re-validate the FILTERED pass_output back into a template
            # instance so auto-evidence sees the same field values that
            # will be persisted on the vertex (post identity-gate +
            # bundle postprocess). Falls back to the unfiltered
            # template_instance if re-validation fails.
            try:
                filtered_template = template_cls.model_validate(pass_output)
                primary_entities = getattr(filtered_template, list_field_name, []) or []
            except Exception:
                primary_entities = getattr(template_instance, list_field_name, []) or []
            if primary_entities:
                # Build per-entity instance_id list aligned with primary_entities.
                identity_to_instance: dict[tuple[str, str], str] = {}
                for prow in provenance_rows:
                    iv = getattr(prow, "identity_values", {}) or {}
                    for k, v in iv.items():
                        if v:
                            identity_to_instance[(str(k), str(v))] = getattr(prow, "instance_id", "")
                instance_ids: list[str] = []
                for entity in primary_entities:
                    iv_dump = entity.model_dump() if hasattr(entity, "model_dump") else {}
                    found = ""
                    for k, v in iv_dump.items():
                        if v and (str(k), str(v)) in identity_to_instance:
                            found = identity_to_instance[(str(k), str(v))]
                            break
                    instance_ids.append(found)

                # Skip system fields, edges, and identity adjuncts —
                # these are bookkeeping or already part of the entity
                # header, not parametric properties.
                skip_fields: set[str] = {"confidence"}
                if primary_entities:
                    sample = primary_entities[0]
                    for fname, finfo in sample.__class__.model_fields.items():
                        extra = finfo.json_schema_extra or {}
                        if not isinstance(extra, dict):
                            continue
                        if extra.get("system_field") is True:
                            skip_fields.add(fname)
                        if extra.get("edge_label"):
                            skip_fields.add(fname)
                        if extra.get("identity_field") is True:
                            skip_fields.add(fname)
                graph_id_fields = (
                    primary_entities[0].__class__.model_config.get("graph_id_fields", []) or []
                )
                skip_fields.update(graph_id_fields)

                # Collect input chunks (element_uid, text) once.
                input_chunks_for_resolver: list[tuple[str, str]] = []
                doc = getattr(context, "docling_document", None)
                if doc is not None:
                    for text_elem in (getattr(doc, "texts", []) or []):
                        self_ref = getattr(text_elem, "self_ref", None)
                        txt = getattr(text_elem, "text", None) or getattr(text_elem, "orig", None)
                        if self_ref and txt:
                            input_chunks_for_resolver.append((str(self_ref), str(txt)))

                if input_chunks_for_resolver:
                    evidence_units_by_chunk = getattr(context, "_chunk_to_evidence_units", None) or {}
                    all_evidence_units: list[dict] = []
                    for units in evidence_units_by_chunk.values():
                        for u in units:
                            u_copy = dict(u)
                            u_copy.setdefault("document_id", body.document_id)
                            all_evidence_units.append(u_copy)
                    field_provenance_rows = build_auto_field_evidence(
                        primary_entities=primary_entities,
                        instance_ids=instance_ids,
                        input_chunks=all_evidence_units or input_chunks_for_resolver,
                        skip_fields=skip_fields,
                        provenance_cls=ExtractionFieldProvenance,
                    )

        relationship_provenance_rows = build_relationship_provenance_from_delta_trace(
            context, ExtractionRelationshipProvenance,
        )
        if not relationship_provenance_rows and getattr(context, "_delta_merged_graph", None):
            logger.warning(
                "delta_merged_graph present but yielded 0 relationship_provenance rows — "
                "verify delta_merged_graph.json shape against "
                "build_relationship_provenance_from_delta_trace."
            )

        return ExtractPassResponse(
            bundle_key=body.bundle_key,
            pass_name=body.pass_name,
            pass_output=pass_output,
            metadata=metadata,
            model=os.environ.get("DOCLING_GRAPH_LLM_MODEL", "granite3-dense:8b"),
            provider=os.environ.get("DOCLING_GRAPH_LLM_PROVIDER", "ollama"),
            provenance=provenance_rows,
            field_provenance=field_provenance_rows,
            relationship_provenance=relationship_provenance_rows,
            diagnostics=getattr(context, "_delta_trace", None),
            table_overlay=table_overlay_obj,
        )
    finally:
        # Pull chunk/batch counts off the response's diagnostics so an operator
        # tailing the docker log can see per-pass extraction shape (how the
        # doc was chunked, how those chunks were batched into LLM calls)
        # without round-tripping back to the notebook.
        _diag = locals().get("context", None)
        _diag = getattr(_diag, "_delta_trace", None) if _diag is not None else None
        _chunks = (_diag or {}).get("chunk_count", -1) if isinstance(_diag, dict) else -1
        _batches = (_diag or {}).get("batch_count", -1) if isinstance(_diag, dict) else -1
        _sanitize = (_diag or {}).get("input_sanitize", {}) if isinstance(_diag, dict) else {}
        _texts_dropped = (_sanitize or {}).get("texts_dropped", -1) if isinstance(_sanitize, dict) else -1
        logger.info(
            "extract-pass: END bundle=%s pass=%s document_id=%s "
            "chunks=%s batches=%s node_count=%d edge_count=%d "
            "raw_node_count=%d raw_edge_count=%d service_identity_dropped=%d "
            "sanitize_dropped=%s",
            body.bundle_key, body.pass_name, body.document_id,
            _chunks, _batches, node_count_for_log, edge_count_for_log,
            raw_node_count_for_log, raw_edge_count_for_log,
            service_dropped_for_log, _texts_dropped,
        )
        # Loud surface for soft-failed extractions: when the run_pipeline
        # path was caught and stubbed, OR when extraction completed but
        # produced zero nodes and zero edges, emit a single ERROR line per
        # pass so an operator grepping logs sees one entry per bad pass
        # (the per-failure-mode WARNING_LIBRARY emits add the *why*).
        try:
            ctx_diag = locals().get("context", None)
            diag = getattr(ctx_diag, "_delta_trace", None) if ctx_diag is not None else None
            pipeline_err = (diag or {}).get("pipeline_error") if isinstance(diag, dict) else None
            zero_yield = (
                node_count_for_log == 0
                and edge_count_for_log == 0
                and node_count_for_log != -1
            )
            if pipeline_err or zero_yield:
                logger.error(
                    "GRAPH_EXTRACTION_PASS_DEGRADED bundle=%s pass=%s document_id=%s "
                    "node_count=%d edge_count=%d pipeline_error=%s — pass returned "
                    "200 but yielded no usable extraction.",
                    body.bundle_key, body.pass_name, body.document_id,
                    node_count_for_log, edge_count_for_log,
                    bool(pipeline_err),
                )
        except Exception as _diag_exc:
            logger.warning("post-pass diagnostics emit failed: %s", _diag_exc)
