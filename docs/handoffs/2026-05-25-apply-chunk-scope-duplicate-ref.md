# Handoff: apply_chunk_scope `Duplicate ref` bug

**Worktree:** `/home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry`
**Branch:** `walltime/c0-telemetry`
**File to fix:** `app/services/scoped_docling_document.py`
**Test file:** `tests/unit/test_scoped_docling_document.py`

## The bug

When `apply_chunk_scope` builds a scoped DoclingDocument from SA-2 (11MB radar/missile document, 317 ExtractionChunks), the resulting doc fails Pydantic validation in docling-graph's `ChunkingDocSerializer`:

```
1 validation error for ChunkingDocSerializer
doc
  Value error, Duplicate ref: #/texts/46 [type=value_error]
```

- **Failed pipeline_run_id:** `1f427080-3d7b-4ed5-846d-1b4bce936e19`
- **Failed pass:** `missile_kinematics`
- Reproduces 4× per pass (the C1.5 dispatcher retries) — same `#/texts/46` every time
- **Does NOT reproduce on Dvina** (180 chunks, simpler structure)
- **Does NOT reproduce on full doc** (only when narrowing produces certain self_ref combinations)

## What's already in place

Commit `1896e54` added a hierarchy mutation pass at `scoped_docling_document.py` ~lines 384-458 that:

1. Rewrites retained elements' `parent` field to `{"cref": "#/body"}` for any ref in `unique_scoped_crefs`.
2. Strips reparented refs from `groups[N].children` lists.

That fix solved a DIFFERENT bug:
```
Document hierarchy is inconsistent.
#/body has child #/texts/N with parent #/groups/M
```
And works correctly on Dvina C.7. **27 unit tests pass.**

The new `Duplicate ref` bug is a SEPARATE problem the hierarchy fix did not address.

## Suspected root causes (investigate all)

### 1. String-form group children (most likely)
The strip loop at ~lines 443-450 only filters dicts:
```python
filtered = [
    c for c in children
    if not (
        isinstance(c, dict)
        and (c.get("cref") or c.get("$ref") or c.get("$cref"))
        in direct_body_children_set
    )
]
```
If a group's `children` list contains bare strings (`"#/texts/46"`) instead of dicts, they pass through unfiltered. Real docling may emit string-form children for some payload types.

### 2. Group ref accidentally in body.children
Docstring at ~lines 390-394 says body.children is FLAT (no groups), but if VR ever returns a self_ref pointing at a group AND that group contains an already-selected text, both could end up referenced.

### 3. Nested groups
If `groups[A]` is in body.children and `groups[B]` is a descendant of `groups[A]` and contains `#/texts/46`, AND `#/texts/46` is ALSO directly in body.children, the strip loop only checks one level deep.

### 4. Heading-context duplication
Heading-context pass (~line 218) prepends headings to `scoped_crefs`. Dedup at lines 225-231 collapses identical strings — but verify the order: are heading crefs added BEFORE or AFTER all other potential duplicate-introducers?

## Constraints

- **DO NOT** restart any Docker containers
- **DO NOT** trigger any extractions
- **DO NOT** touch any files outside the two listed above
- **DO NOT** touch the running pipeline_run `4c30b3a3-5ac4-415f-a0e2-c0f44b7ca342` (Dvina C.7b, in flight)
- **MAINTAIN** the no-mutation invariant on input `doc_json` (existing tests verify this)

## Recommended approach

1. **Write failing tests first** that reproduce the `Duplicate ref` failure shape via `DoclingDocument(**result)` round-trip. Cover each suspected cause (string children, nested groups, etc.) with a dedicated fixture.
2. **Then fix the bug(s)** in `apply_chunk_scope`.
3. **Verify** all 27 existing unit tests still pass. Add the new tests to the file.
4. Commit each logical change separately (failing test → fix → cleanup). Stay on branch `walltime/c0-telemetry`. Co-author commits to Claude (existing convention).

## Verification command

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
python -m pytest tests/unit/test_scoped_docling_document.py -v
```

## Expected output

- Root cause(s) identified (vs. the four hypotheses above)
- New test fixtures that reproduce the failure
- Fix landed
- Test result (X/X passing)
- Commit SHAs
