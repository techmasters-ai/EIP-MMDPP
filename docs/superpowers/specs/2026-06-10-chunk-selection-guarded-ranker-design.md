# Chunk Selection: Guarded Ranker — Design Spec

Date: 2026-06-10
Status: approved (user, 2026-06-10)
Worktree: `.worktrees/walltime-c0-telemetry` (branch `walltime/c0-telemetry`)
Predecessor context: `docs/chunk-selection-handoff.md`, `reports/dataset/README.md`

## 0. Decisions locked with the user

1. **Recall floor:** hard recall 1.0 on value-grounded positives. Never drop a
   true-source chunk; pruning comes second.
2. **Label fix first:** repair the ground-truth label and re-export the dataset
   before any model/selection tuning.
3. **Keywords scope:** all four directions — unit/label vocabulary + OR-gate,
   header projection into table-derived chunks, matching mechanics + runtime
   per-keyword weights, and de-emphasizing keywords in the model until they
   re-validate (cosine-led ranker meanwhile).
4. **Sequencing:** features first on the 8-doc corpus, re-collect those 8 docs,
   validate — then run the planned 21-doc collection once, with improved capture.
5. **Architecture:** A "guarded ranker" — recall floor produced *by construction*
   via label-aligned OR-gates; pruning produced by a per-(doc,pass) quantile cut
   over a small learned score. (Alternatives considered: B calibrated-score-only
   — rejected as the architecture that already failed, kept as eval baseline;
   C haystack-first — folded into this design as feature work, not an
   architecture.)
6. **Metric convention:** pooled-OOF LODO is the official gate metric;
   mean-per-fold is always reported alongside. State the convention on every
   number.

## 1. Problem statement

For each extraction pass, send the LLM only the chunks that actually contain
that pass's content: maximize recall of true-source chunks, drop as many
irrelevant (pass, chunk) candidates as possible (the dominant wall-time lever),
generalize across document shapes and air-defense systems, and preserve
complete data lineage.

### Review findings this design responds to (verified 2026-06-10)

- **Keywords are anti-predictive pooled and inert in production.**
  `pass_keyword_norm` pooled AUROC 0.459 (fires on 61% of negatives vs 46% of
  positives). In production `final_score` the curated keywords contribute
  exactly zero: `lexical_decomposed` defaults False
  (`app/services/ontology_bundles.py:266-275`), nothing sets it, and all four
  decomposed weights default 0.0. The keyword channel is also precision-only:
  `merge_candidates` drops lexical hits on chunks absent from the dense pool
  (`app/services/extraction_candidate_scoring.py:163-191`), and the
  lexical_table fallback ignores `keyword_hits`
  (`app/api/v1/extraction_routing.py:200-205`) — a keyword can never ADD a
  chunk.
- **The structural table miss is mechanical.** Table rows render as
  `- <label>: <value> <unit>` (labels + units vocabulary); keywords are prose
  nouns, and the miner explicitly bans digit/unit tokens
  (`scripts/mine_pass_keywords.py:29-30`). Schema-derived unit keywords exist
  (`extraction_query_builder.derive_pass_keywords`) but are shadowed whenever a
  manifest list is non-empty (`extraction_routing.py:106-113`) — always, in
  production.
- **The ground-truth label has holes (verified empirically).**
  `units_for()` matches field-name suffixes against `SUFFIX_UNITS`
  (`app/services/field_value_grounding.py:33-61`); `_sec`, `_usec`, `_dbi` are
  missing, so radar_timing (3 numeric fields), six `*_time_sec` fields in
  missile_speed_timing, two in missile_propulsion, and `gain_dbi` can never
  ground. Net today: only 7 of 12 passes / 22 of ~77 fields can produce a
  positive; 3 zero-positive retrieval passes account for 33% of dataset rows.
  One suspect positive exists (SA2-SR71 `radar_antenna` chunk 16 — missile
  prose grounding a radar pass, cosine rank 30/42).
- **Cosine alone cannot reach the goal.** Pure cosine-led selection at recall
  1.0 saves only ~18-20%; even oracle per-doc k keeps 45.2% of rows. The
  binding constraint is deep-ranked Engagement *prose* positives (cosine rank
  40-41/50) — tables defeat keywords, prose depth defeats cosine.
- **The baseline is weak.** Hand-weighted `final_score` is near-chance pooled
  (AUROC 0.584), worse than random on 2 of 7 docs, and ranks at least one true
  positive dead last in its 50-candidate pool.
- **Free signal unused:** per-field dense cosines are computed and stored in
  `field_scores` (`extraction_candidate_scoring.py:148-159`) but the captured
  `cosine` feature reads only the entity-level `vector_score` (`:460`).
- **Dead/duplicate features:** `is_table` constant 0.0 (`table_meta={}` at
  `extraction_chunk_search.py:879,1061,1148`); `section_norm` ===
  `anchor_section_norm` (r=1.000). Effective feature count is 8, not 10.

## 2. Phase 0 — Ground truth (label fix + re-baseline)

- Add to `SUFFIX_UNITS`: `"sec"` → same synonym list as `"s"`, `"usec"` → same
  list as `"us"`, `"dbi"` → `["dbi"]`.
- New audit script: enumerate every field of every pass in `air_defense_v3`,
  report numeric fields where `units_for(field) == []`. Wire it into the
  acceptance harness so suffix gaps cannot silently recur. (Known intentional
  residue: unitless ints like `num_bits_in_code`; `*_thrust` fields are
  `Optional[str]` by schema design and stay un-groundable.)
- Audit the suspect SA2-SR71 positive. If the grounding is coincidental
  cross-content matching, fix with a principled rule (e.g. match tier required
  by chunk modality), never a one-off label deletion. Document the outcome.
- Re-export the 8-doc dataset (`scripts/export_bakeoff_dataset.py`) and
  re-baseline per-feature AUROC, frontier, and per-doc breakdowns. All
  subsequent comparisons use the fixed label. Expect new (likely
  table-derived) positives in the activated passes.

## 3. Phase 1 — Recall floor: label-aligned OR-gates

The label is "a numeric value + unit of the pass's OWN fields appears in the
chunk text" with two tiers: ADJACENT (prose, `50 km`) and SAME_CHUNK (table,
number and unit anywhere in the chunk). The gates mirror the label's *form*
without knowing values:

- **G1 (unit gate):** force-keep chunk for pass P iff chunk contains a
  digit-bearing token AND a unit token from P's **unit signature** — the union
  of `units_for(field)` over P's fields, derived automatically at
  RetrievalProfile build time. Pure schema-SSoT: no config file, no curation,
  no equipment names. Mirroring SAME_CHUNK (not just ADJACENT) is required so
  the floor covers table-serialized text (`Peak Power [kW] 180.0`).
- **G2 (table gate):** force-keep `is_table` chunks that are unit-bearing
  (available once is_table is wired, Phase 2).
- Gates only ever ADD keeps. Failure mode is "gate didn't fire" → ranker +
  `k_min` floor; `fallback_to_full` remains the last resort. Gates never
  remove anything.
- **Acceptance test (literal):** gate coverage = 100% of positives in the
  re-exported dataset. Any miss is a unit-lexicon bug by construction.
- Unit-match semantics must guard against substring collisions (`m` inside
  words, `s` plurals): unit tokens match with word-ish boundaries, mirroring
  `value_in_chunk`'s existing token discipline (reuse that module — native
  reuse, not a parallel regex dialect).

Selectivity comes from per-pass signatures (GHz does not fire the kinematics
gate). Unit collisions across passes (km) cost pruning, never recall.

## 4. Phase 1 — Pruning lever: features + per-(doc,pass) quantile ranker

Non-gated chunks are kept iff they rank above a quantile cut of their own
(doc, pass) score-distribution, with `k_min` floor and `k_max` cap. Quantile
`q` (one global hyperparameter) calibrated offline pooled-OOF with a
finite-sample margin (conformal-style discipline over per-positive
nonconformity ranks — used as margin-setting, not as a certificate).

Score = sign-constrained, L2-regularized logistic regression. Weights ship as
config numbers (a dot product at the endpoint — no model-artifact
infrastructure). Feature set (all zero-label except where noted):

| feature | source | status |
|---|---|---|
| cosine | existing | proven generalizable (LODO 0.857) |
| max_field_cosine (+ mean-top-3) | `field_scores`, already computed | new capture, zero cost |
| rerank_norm | existing | keep (watch S75-Dvina inversion) |
| negative_norm | existing | keep (held up 0.718) |
| is_table | Phase 2 wiring | new |
| digit_density | chunk_text stat | new |
| label_value_lines | spec_overlay-style regex reuse | new |
| unit_token_count (pass signature) | G1 machinery | new |
| pass_keyword_norm, field_label_norm, anchor_text_norm | existing | **excluded** until §5 re-validates |
| section_norm | duplicate of anchor_section_norm | dropped |

New features are appended to `COMPONENT_KEYS` /
`score_components_all` (additive, diagnostics-only) so they appear in capture
without touching `final_score`.

## 5. Phase 2 — Keyword channel revival (feature work, never recall-bearing)

1. **Unit-vocab union:** `inject_pass_keywords` merges
   `derive_pass_keywords(signals)` units INTO the manifest list (union, not
   override-when-empty). Safe: feeds `pass_keyword_hits`, which is
   diagnostics-only today.
2. **Header projection:** table-derived chunks carry column/row headers +
   caption + nearest section heading in a SEPARATE matching/embedding haystack
   field. Raw `chunk_text`, `source_refs`, page lineage stay verbatim (hard
   lineage requirement; the value-grounding haystack is unchanged). First step
   is diagnostic: inspect `_render_table_chunk` output and merged-mode chunk
   text for one known table positive to determine whether this is a rendering
   fix or a chunk-splitting fix. Requires re-embed → lands before the Phase 3
   re-collection.
3. **Matching mechanics:** word-boundary matching for single-token needles
   (substring kept for multi-word phrases); unify normalization between mining
   scripts (NFKD+strip-accents) and runtime (NFC) so offline stats predict
   runtime behavior; per-keyword mined-lift weights at runtime
   (`lexical_keywords` entries gain optional weights).
4. **Re-mine** keywords against the fixed `used` label with Engagement-style
   docs included and the digit/unit exclusion relaxed for the unit-token
   class. Output remains a human-review list (never auto-committed).

Re-entry rule: the keyword trio re-enters the ranker only if it shows
generalizable signal post-revival — positive contribution under BOTH LODO
conventions with per-doc breakdown, no doc worse than chance.

## 6. Phase 2 — is_table wiring (task #70)

Index-time persistence (robust to table-normalization ref suppression):
`hybrid_chunking.py` marks merged chunks from `#/tables/` doc_items or
synth-ref membership (~10 LOC); `extraction_chunk_index.py` persists the
column (~15 LOC, ArcadeDB schemaless — no migration);
`extraction_chunk_search.py` projects it in both SELECTs and builds
`table_meta` at the 3 merge call sites (~25 LOC);
`extraction_routing.py:244` sets `content_type` from the row (~3 LOC).
**Score-neutrality:** flip `table_boost` default 0.08 → 0.0 (the
`section_weight` precedent) so wiring the feature leaves production
`final_score` byte-identical. Old captured runs stay 0 — the feature appears
in data at the Phase 3 re-collection.

## 7. Phase 3 — Re-collect + evaluate

- Re-run the 8-doc collection on an idle inference pool (wall-time comparisons
  and capture quality both require it; see contention memory).
- Re-export dataset v2 with new features; evaluate:
  - per-feature AUROC (pooled + both LODO conventions, per-doc breakdown);
  - guarded-ranker frontier: savings at recall 1.0 (gates ∪ quantile-cut
    LogReg) vs architecture-B baseline (calibrated score only) vs production
    `final_score` baseline;
  - gate-coverage acceptance (100% of positives);
  - worst-case analysis: per-doc minimum positive rank, gate vs ranker keeps.
- Go/no-go gate for Phase 4 plumbing rollout discussion.

## 8. Phase 4 — Selection plumbing (flag-gated, byte-identical default)

- `RetrievalProfile.selection_mode: topk | guarded_quantile` (default `topk`),
  plus `quantile_q`, `k_min`, `k_max`, and ranker weights as profile fields.
- One shared cut helper (natural home: `extraction_candidate_scoring.py`)
  replaces the four `[:profile.top_k]` slice sites
  (`extraction_routing.py:650,766,833,891`); per-element legacy path
  unchanged.
- Diagnostics record per pool: gate keeps (G1/G2 counts), chosen k, threshold,
  score-distribution stats — enabling offline A/B from capture alone.
- Manifest/schema edits land in `air_defense_v3` first, then the 3 sibling
  bundles. New env vars mirrored into `.env` and `.env.example` with comments.
- Production `narrow_only` flip is OUT of this design; it follows a separate
  validation against the active baseline's quality gates
  (`project_extraction_baseline_75bd0f3`).

## 9. Phase 5 — 21-doc collection

Runs after Phase 3 validates, using the improved capture (new features,
fixed label) so the corpus is collected once, well. Config decisions for that
collection are a separate discussion (existing memory:
`project_large_scale_extraction_collection`).

## 10. Testing

- Unit tests: `units_for` additions + audit script; G1 signature derivation and
  unit-boundary matching; cut helper (topk default byte-identical, quantile
  mode, k_min/k_max, gate union); is_table threading; word-boundary keyword
  matcher; header-projection haystack separation (lineage text untouched).
- Dataset-level acceptance: gate coverage 100% of positives; re-export
  determinism.
- Byte-identical default: existing legacy-default tests extended to
  `selection_mode=topk` + `table_boost=0.0` flip.
- Post-change workflow per standing rule: simplify, full test suite,
  VERIFICATION_CHECKLIST.md, README.

## 11. Out of scope

- Wholesale BM25 (only per-keyword weights); revisit after header projection.
- Pass-level skip decisions for zero-positive passes (no signal until the
  larger corpus).
- Production data migration / `narrow_only` flip.
- docling-graph upstream refresh; HNSW post-filter fix (tracked separately).

## 12. Risks

- Gate selectivity on spec-dense docs may bound savings (~30-60% kept there);
  measured at Phase 3, mitigated by per-pass signatures and the ranker
  handling the rest.
- Header projection requires re-embed; effect only measurable post
  re-collection — keep its diff minimal and flag-gated.
- 35→N positives after label fix may shift all baselines; that is the point of
  Phase 0 ordering.
- Each phase begins with a discussion gate (standing rule) — no phase is
  implemented without explicit go-ahead.
