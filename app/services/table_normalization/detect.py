"""Shape detection — pure heuristic over Docling table_cells.

Implements the rules per §7 of the design spec. No LLM, no external state.
Returns Shape; emits diagnostics dict separately for caller logging."""
from __future__ import annotations

import logging
from typing import Any
from app.services.table_normalization.models import Shape

logger = logging.getLogger(__name__)


SPEC_ROW_KEYWORDS: frozenset[str] = frozenset({
    "max range", "min range", "range", "max altitude", "min altitude", "altitude",
    "max speed", "min speed", "speed", "velocity", "vmax", "vmin",
    "weight", "mass", "total weight", "warhead weight",
    "length", "width", "diameter", "span", "height",
    "max alt", "min alt",
    "missile type", "missile variant",
    "frequency", "wavelength", "power",
    "thrust", "burn time", "stage",
})

SECTION_KEYWORDS: frozenset[str] = frozenset({
    "missile", "1st stage", "2nd stage", "first stage", "second stage",
    "booster", "sustainer", "propulsion",
    "radar", "launcher", "guidance",
    "warhead", "fuze",
    "system performance", "performance",
})

IDENTITY_LABEL_KEYWORDS: frozenset[str] = frozenset({
    "designation", "variant", "type", "name",
    "industry designation", "military designation", "nato designation",
    "fan song variant", "radar variant",
    "system name", "system designation",
})


_MIN_DIM = 4


def _safe_get_int(d: dict, key: str, default: int = -1) -> int:
    v = d.get(key, default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _cells_at_col_zero(cells: list[dict]) -> list[dict]:
    return [c for c in cells if _safe_get_int(c, "start_col_offset_idx") == 0 and (c.get("text") or "").strip()]


def _cells_at_row_zero(cells: list[dict]) -> list[dict]:
    return [c for c in cells if _safe_get_int(c, "start_row_offset_idx") == 0 and (c.get("text") or "").strip()]


def _num_rows(cells: list[dict]) -> int:
    if not cells:
        return 0
    return max(_safe_get_int(c, "end_row_offset_idx", 0) for c in cells) + 1


def _num_cols(cells: list[dict]) -> int:
    if not cells:
        return 0
    return max(_safe_get_int(c, "end_col_offset_idx", 0) for c in cells) + 1


def _has_spec_keyword(texts: list[str]) -> bool:
    for t in texts:
        if any(kw in t.lower() for kw in SPEC_ROW_KEYWORDS):
            return True
    return False


def _is_identity_shaped(text: str) -> bool:
    text = text.strip()
    if not text or len(text) >= 40:
        return False
    try:
        float(text.replace(",", "").replace(" ", ""))
        return False
    except ValueError:
        return True


def detect_shape(table_cells: list[dict], table_data: dict) -> Shape:
    """Return the Shape classification of a Docling table.

    See §7 of the design spec for the decision rules.
    """
    try:
        if not table_cells:
            return Shape.OTHER
        if _num_rows(table_cells) < _MIN_DIM or _num_cols(table_cells) < _MIN_DIM:
            return Shape.OTHER

        # Test 2: COLUMN_MAJOR
        col0 = _cells_at_col_zero(table_cells)
        if col0:
            row_header_share = sum(1 for c in col0 if c.get("row_header")) / len(col0)
            if row_header_share >= 0.5 and _has_spec_keyword([c.get("text") or "" for c in col0]):
                # Test 3: HYBRID upgrade — count identity rows at top.
                # An identity row is one where the col-0 label matches an
                # IDENTITY_LABEL_KEYWORD AND the data cells are identity-shaped
                # (short, non-numeric strings like model numbers).
                col0_by_row = {
                    _safe_get_int(c, "start_row_offset_idx"): (c.get("text") or "")
                    for c in col0
                }
                identity_row_count = 0
                for row_idx in range(_num_rows(table_cells)):
                    label = col0_by_row.get(row_idx, "").strip().lower()
                    if not label:
                        break
                    if not any(kw in label for kw in IDENTITY_LABEL_KEYWORDS):
                        break
                    data_cells_in_row = [
                        c for c in table_cells
                        if _safe_get_int(c, "start_row_offset_idx") == row_idx
                        and _safe_get_int(c, "start_col_offset_idx") > 0
                    ]
                    if not data_cells_in_row:
                        break
                    if all(_is_identity_shaped(c.get("text") or "") for c in data_cells_in_row):
                        identity_row_count += 1
                    else:
                        break
                if identity_row_count >= 2:
                    return Shape.HYBRID
                return Shape.COLUMN_MAJOR

        # Test 4: ROW_MAJOR
        row0 = _cells_at_row_zero(table_cells)
        if row0:
            col_header_share = sum(1 for c in row0 if c.get("column_header")) / len(row0)
            if col_header_share >= 0.5 and _has_spec_keyword([c.get("text") or "" for c in row0]):
                return Shape.ROW_MAJOR

        rows, cols = _num_rows(table_cells), _num_cols(table_cells)
        if rows >= _MIN_DIM and cols >= _MIN_DIM:
            logger.warning(
                "table_normalization.detect: %dx%d table fell to OTHER. "
                "Row-0 headers=%d, col-0 row_headers=%d. Consider adding row labels "
                "to SPEC_ROW_KEYWORDS or column headers to row-major detection.",
                rows, cols, len(row0) if row0 else 0,
                sum(1 for c in col0 if c.get("row_header")) if col0 else 0,
            )
        return Shape.OTHER

    except Exception as exc:
        logger.warning("table_normalization.detect: exception during shape detection: %s; returning OTHER", exc)
        return Shape.OTHER
