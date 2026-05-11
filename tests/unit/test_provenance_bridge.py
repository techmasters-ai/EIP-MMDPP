from app.services.table_normalization import _provenance_bridge as bridge


def test_record_and_lookup():
    bridge.reset()
    bridge.record_text_idx_cell_refs(142, ["#/tables/3/data/table_cells/42"])
    assert bridge.cell_refs_for_text_idx(142) == ["#/tables/3/data/table_cells/42"]


def test_lookup_unknown_returns_empty():
    bridge.reset()
    assert bridge.cell_refs_for_text_idx(999) == []


def test_reset_clears_state():
    bridge.record_text_idx_cell_refs(1, ["#/tables/0/data/table_cells/0"])
    bridge.reset()
    assert bridge.cell_refs_for_text_idx(1) == []


def test_empty_list_not_recorded():
    bridge.reset()
    bridge.record_text_idx_cell_refs(5, [])
    assert bridge.cell_refs_for_text_idx(5) == []


def test_returned_list_is_a_copy():
    """Mutating the returned list must not corrupt the stored value."""
    bridge.reset()
    bridge.record_text_idx_cell_refs(7, ["#/tables/0/data/table_cells/1"])
    got = bridge.cell_refs_for_text_idx(7)
    got.append("MALICIOUS")
    assert bridge.cell_refs_for_text_idx(7) == ["#/tables/0/data/table_cells/1"]
