"""Part B: _resolve_element_uid must prefer the per-node evidence_ids self_ref
(precise Docling self_ref) over the batch-wide self_refs[0] (identical across a
batch). Verified empirically: real gemma4 nodes carry distinct per-node
evidence_ids (#/texts/59, #/texts/61, ...) while self_refs is batch-wide.
"""
import importlib.util
import sys
from pathlib import Path

# provenance.py does `from app._numeric_evidence import ...` at module load, so
# the docling-graph SERVICE dir must be FIRST on sys.path BEFORE exec_module —
# else `app` resolves to the repo-root worker package (no _numeric_evidence) and
# the import errors at collection. The dir's conftest already APPENDS the service
# root (index ~9, behind the repo-root at index 0), so a guarded insert is a
# no-op; force it to the FRONT and evict any cached repo-root `app.*` so the
# fresh provenance import binds against the service-root `app` package.
_SERVICE_ROOT = Path(__file__).resolve().parent.parent  # docker/docling-graph
while str(_SERVICE_ROOT) in sys.path:
    sys.path.remove(str(_SERVICE_ROOT))
sys.path.insert(0, str(_SERVICE_ROOT))
for _stale in [k for k in list(sys.modules) if k == "app" or k.startswith("app.")]:
    del sys.modules[_stale]

_PROV = _SERVICE_ROOT / "app" / "provenance.py"
_spec = importlib.util.spec_from_file_location("dg_provenance_under_test", _PROV)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_resolve_element_uid = _mod._resolve_element_uid


def test_prefers_per_node_evidence_id_over_batch_self_refs():
    node = {"provenance": {
        "self_refs": ["#/texts/10"],
        "evidence_ids": ["#/texts/42"],
    }}
    assert _resolve_element_uid(node, None) == "#/texts/42"


def test_falls_back_to_self_refs_when_no_evidence_ids():
    node = {"provenance": {"self_refs": ["#/texts/10"], "evidence_ids": []}}
    assert _resolve_element_uid(node, None) == "#/texts/10"


def test_non_selfref_evidence_id_does_not_win():
    node = {"provenance": {"self_refs": ["#/texts/10"], "evidence_ids": ["e1", "e2"]}}
    assert _resolve_element_uid(node, None) == "#/texts/10"


def test_direct_element_uid_still_wins():
    node = {"element_uid": "p1-2-text-abcd",
            "provenance": {"evidence_ids": ["#/texts/42"], "self_refs": ["#/texts/10"]}}
    assert _resolve_element_uid(node, None) == "p1-2-text-abcd"


def test_chunk_indexes_fallback_unchanged():
    node = {"provenance": {"chunk_indexes": [0]}}
    assert _resolve_element_uid(node, {0: ["#/texts/7"]}) == "#/texts/7"


def test_evidence_id_choice_is_deterministic_regardless_of_order():
    a = {"provenance": {"self_refs": ["#/texts/1"], "evidence_ids": ["#/texts/42", "#/texts/9"]}}
    b = {"provenance": {"self_refs": ["#/texts/1"], "evidence_ids": ["#/texts/9", "#/texts/42"]}}
    assert _resolve_element_uid(a, None) == _resolve_element_uid(b, None) == "#/texts/42"
