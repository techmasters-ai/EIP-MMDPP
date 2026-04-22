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
    "Valid JSON only; no markdown or batch metadata in node content. "
    "For unknown, absent, or unstated OPTIONAL fields, emit the JSON null "
    "literal — NEVER the string \"None\", \"N/A\", \"Unknown\", \"null\", "
    "or an empty string. For unknown OPTIONAL booleans, use null; never "
    "invent false as a default placeholder. If you have no evidence for a "
    "field, the field's value is null.\n"

    "6. Entity-type discipline. Emit each entity under the ONE catalog path "
    "that matches what the entity IS, not what it is associated with.\n"
    "   - Weapon / missile systems — e.g. SA-2, Patriot, S-300, THAAD — "
    "are MISSILE entities. NEVER emit them under a radar catalog path.\n"
    "   - Radar systems — e.g. Fan Song, Spoon Rest, Tombstone, AN/MPQ-65 "
    "— are RADAR entities. NEVER emit them under a missile / weapon catalog "
    "path.\n"
    "   - Aircraft / platforms / targets — e.g. U-2, RF-4C, F-16, B-52 — "
    "are NEITHER radars NOR missiles. NEVER emit them under either catalog "
    "path even when named in prose as engaged or shot down.\n"
    "   Do NOT re-emit a weapon-system name under a radar path just "
    "because a radar is associated with it. Do NOT re-emit a radar name "
    "under a weapon path just because it serves a weapon. Only emit an "
    "entity when the text describes the entity's own role matching the "
    "catalog path's semantics.\n"

    "7. Cross-entity relationships. For a relationships-only pass that "
    "receives an upstream entity catalog (ref ids like E001, E002): when "
    "the document explicitly describes multiple named systems together — "
    "especially a search / early-warning radar, a fire-control / tracking "
    "radar, and a missile / weapon system mentioned as part of the same "
    "engagement kill chain — emit the corresponding cross-pass "
    "relationships. For a search radar that hands off to a fire-control "
    "radar, emit CUES. For a fire-control radar paired with the weapon it "
    "guides, emit ASSOCIATED_WITH (or CUES when the text emphasizes "
    "target-handoff). Silence is only correct when the document contains "
    "no narrative link between the named systems. Do NOT invent edges "
    "between systems the document does not jointly describe.\n"

    "8. Evidence-scope discipline. Extract ONLY from the current batch's "
    "direct document content. The DOCUMENT CONTEXT block and prior-batch "
    "context exist only to keep names stable across batches; they do NOT by "
    "themselves justify emitting an entity, property, or relationship in "
    "this batch.\n"

    "9. Treat preprocessing scaffolding, analyst summaries, and website / "
    "viewer chrome as NON-EVIDENCE unless they directly quote the document. "
    "Non-evidence examples include blocks labeled Classification, Why This "
    "Category, General Description, OCR wrappers, Category-Specific Details, "
    "Extracted Technical Takeaways, Uncertainty, Analyst Notes, provenance / "
    "source-identification notes, page counters, navigation arrows, download "
    "buttons, related-links sections, gallery-return links, footer text, and "
    "recommendations such as 'next analytical step'. These may mention system "
    "names from other pages or analyst context; do NOT extract from them "
    "unless the same entity/property is directly evidenced in quoted verbatim "
    "document text or a defining structure in this batch.\n"

    "10. Conservative default. If a mention could plausibly be a missile, "
    "radar, aircraft, photo subject, provenance marker, generic explainer, "
    "or off-page summary, prefer omission over guessing. Empty output for a "
    "domain pass is correct when the batch lacks direct evidence for an "
    "entity of that domain.\n"

    "11. Prompt-content non-evidence rule. Names or values that appear in "
    "this prompt, the catalog, schema guidance, field descriptions, or "
    "examples are NEVER evidence by themselves. Treat them only as "
    "instructions about type semantics and output shape. Do NOT emit a "
    "system, property, or relationship merely because it appears in prompt "
    "examples or allowed-value lists; emit it only if the current batch "
    "document itself states it.\n"

    "12. Field-semantics and unit discipline. Match each source measurement "
    "to the field that has the same meaning and unit. Convert units into the "
    "field's declared unit suffix (for example miles or feet into _km). If "
    "the document states multiple related values such as minimum range, "
    "maximum effective range, slant range, and ceiling, map each only to the "
    "matching field. Do NOT copy a raw source number into a differently "
    "unitized field, and do NOT place slant range into effective-intercept "
    "range. If no exact field match exists, omit the value.\n"

    "13. Status and role inference discipline. Do NOT infer system_status "
    "from historical narrative, museum display context, or generic knowledge. "
    "Populate system_status only when the document explicitly states a status "
    "such as operational, retired, or developmental. For radar role fields, "
    "a guidance / illumination / missile-command radar is FIRE_CONTROL, not "
    "mere TRACKING.\n"

    "14. Relationships-only recall rule. When the document explicitly names a "
    "search or acquisition radar, a guidance or fire-control radar, and a "
    "missile or weapon system as part of the same site, battery, or kill "
    "chain, do NOT return an empty relationship list. Emit SEARCH_RADAR CUES "
    "GUIDANCE_RADAR and GUIDANCE_RADAR ASSOCIATED_WITH MISSILE_SYSTEM unless "
    "the text clearly indicates a different relation."
)


__all__ = ["DELTA_SYSTEM_PROMPT"]
