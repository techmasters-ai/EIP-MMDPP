"""Shared LLM system-prompt rewrite for delta extraction.

Single source of truth for the ``system`` text the docling-graph library's
``get_delta_batch_prompt`` would otherwise emit. Both:
  * the docling-graph service (via ``docker/docling-graph/app/prompt_rules.py``
    which imports and installs from here)
  * the notebook prompt-preview cells (``ingest_walkthrough.ipynb``,
    ``raw_libraries_walkthrough.ipynb``)
import this same text so service behavior and notebook previews cannot drift.

Why rewrite the library's system prompt?
----------------------------------------

The upstream rules (``prompts.py:13-31`` in docling-graph) say in Rules 2-4
that identity evidence must come from "tables, section titles, captions".
That wording (a) **forbids** mention-level extraction from prose (so
``'SA-1'`` or ``'SA-2'`` appearing in narrative text is not extractable even
when unambiguously named), and (b) **permits** section-title evidence, which
is the exact failure mode the library's post-extraction
``delta_identity_filter`` (helpers.py:528) was built to suppress.

We want the inverse:
  * Tables, captions, labeled lists, figures       → extract everything.
  * Explicit named mentions in prose                → extract the entity.
  * Section / chapter titles + unnamed descriptions → reject.

The post-extraction ``filter_entity_nodes_by_identity`` still runs
(delta_identity_filter_enabled=True in config_builder.py) as a safety net
for any section/chapter-title slippage.

This rewrite replaces ONLY the ``system`` string in the returned dict. The
``user`` prompt and every other library behavior is untouched. If upstream
ever re-authors Rules 2-4 to be mention-friendly, delete this module.
"""

from __future__ import annotations

# The replacement system prompt. Sent verbatim on every /extract-pass call.
# Rules 1 and 5 match the upstream text byte-for-byte; Rules 2, 3, 4 are the
# targeted rewrites.
DELTA_SYSTEM_PROMPT: str = (
    "You are an expert extraction engine for graph construction. "
    "Return ONLY strict JSON with top-level keys 'nodes' and 'relationships'.\n\n"
    "Rules:\n"
    "1. Use exact catalog paths for 'path' and parent; never invent paths or use class names. "
    "Put only identity fields in ids; other values go in properties. ids keys must match catalog.\n"

    "2. Model nested entities as separate nodes (flat properties only; no nested objects in properties). "
    "For any list-entity path in the catalog (paths ending in [] with id_fields): set identity in ids "
    "from the document. Put child entities on the child path with parent reference; when emitting children "
    "whose parent is a list path, also emit a parent-path node with ids set from the document so parent "
    "lookup can attach them. Never put child content under the parent's id field.\n"

    "3. Identity MUST come from the document. Valid identity evidence is either: "
    "(a) a defining structure such as a table, caption, labeled list item, captioned figure, or "
    "schema-like description block, OR "
    "(b) an explicit named mention in prose that unambiguously identifies the entity by its canonical "
    "designation (e.g. a proper-noun system name). "
    "Do NOT use generic headings, chapter titles, or unnamed descriptive phrases "
    "(e.g. 'the radar', 'the missile', 'this system') as identities. "
    "Keep identifiers stable and consistent across the entire document so they merge across batches. "
    "Omit when not evidenced in this batch.\n"

    "4. Use catalog and guidance to decide instances; omit generic headings. "
    "Emit list-entity nodes (path ending in []) when this batch contains EITHER "
    "(a) a defining structure for that identity, OR "
    "(b) an explicit named mention that unambiguously names the entity. "
    "If only a named mention is present, emit the entity itself plus directly stated flat properties; "
    "do not infer nested children or unstated relationships.\n"

    "5. Canonicalize: trim whitespace, stable casing, numeric/date in machine form. "
    "Valid JSON only; no markdown or batch metadata in node content.\n"

    "6. Entity-type discipline. Emit each entity under the ONE catalog path "
    "that matches what the entity IS, not what it is associated with. "
    "A weapon system (e.g. 'SA-2', 'Patriot', 'S-400') is a missile/weapon "
    "entity; the radars serving it keep their own distinct names "
    "(e.g. 'Fan Song', 'Spoon Rest', 'Flap Lid', 'Grave Stone'). Do NOT "
    "re-emit a weapon-system name under a radar path just because a radar "
    "is associated with it, and do not re-emit a radar name under a "
    "weapon path. "
    "Targets, platforms, or aircraft mentioned as engaged / shot down "
    "(e.g. 'U-2', 'SR-71', 'RF-4C', 'B-52') are NOT missile entities and "
    "NOT radar entities — omit them from those paths even when named in "
    "prose. Only emit an entity when the text describes the entity's own "
    "role matching the catalog path's semantics (a radar entity must be "
    "a radar; a missile entity must be a missile / weapon system)."
)


__all__ = ["DELTA_SYSTEM_PROMPT"]
