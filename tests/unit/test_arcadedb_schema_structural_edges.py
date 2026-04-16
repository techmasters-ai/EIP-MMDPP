from app.services.arcadedb_schema import _STRUCTURAL_EDGE_TYPES


def test_anchor_structural_edges_declared():
    """HAS_SECTION/HAS_FIGURE/HAS_TABLE/CHILD_OF must be declared."""
    for label in ("HAS_SECTION", "HAS_FIGURE", "HAS_TABLE", "CHILD_OF"):
        assert label in _STRUCTURAL_EDGE_TYPES, f"{label} missing from _STRUCTURAL_EDGE_TYPES"
