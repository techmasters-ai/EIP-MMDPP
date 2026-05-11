"""Process-local map: text_idx (docling #/texts/N) -> list of cell_refs.

Populated by _text_item_from_chunk at TextItem-creation time. Read by
the field-provenance enrichment wrapper that fills
ExtractionFieldProvenance.cell_refs after extraction.

Per-pass reset() prevents cross-pass leakage. Module-level state is
safe in the single-process docling-graph FastAPI worker; multi-process
deployments maintain per-process maps.
"""
from __future__ import annotations

_TEXT_IDX_TO_CELL_REFS: dict[int, list[str]] = {}


def record_text_idx_cell_refs(text_idx: int, cell_refs: list[str]) -> None:
    """Record cell_refs at TextItem-creation time.

    Empty/None lists are not stored (saves memory and makes
    cell_refs_for_text_idx() return [] cleanly).
    """
    if cell_refs:
        _TEXT_IDX_TO_CELL_REFS[int(text_idx)] = list(cell_refs)


def cell_refs_for_text_idx(text_idx: int) -> list[str]:
    """Return a COPY of the cell_refs for text_idx; [] if not recorded."""
    return list(_TEXT_IDX_TO_CELL_REFS.get(int(text_idx), ()))


def reset() -> None:
    """Clear the bridge map. Called at the start of each
    run_extraction_pass to prevent cross-pass leakage."""
    _TEXT_IDX_TO_CELL_REFS.clear()
