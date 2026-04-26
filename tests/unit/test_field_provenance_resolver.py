"""Phase 3 task 31 — snippet → element_uid resolver."""
import sys
import pathlib

# The docling-graph service has its own `app/` package alongside the
# repo's main `app/` package. Both can't be on sys.path for normal
# pytest collection, so we add the service path explicitly here so
# `from docking_graph.app.provenance import resolve_field_provenance_uids`
# would conflict. Instead, import via importlib to bypass package shadowing.
import importlib.util

_SERVICE_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent / "docker" / "docling-graph" / "app"


def _load(modname: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(modname, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_provenance = _load("_dgp_provenance", _SERVICE_ROOT / "provenance.py")
_schemas = _load("_dgp_schemas", _SERVICE_ROOT / "schemas.py")

resolve_field_provenance_uids = _provenance.resolve_field_provenance_uids
ExtractionFieldProvenance = _schemas.ExtractionFieldProvenance


def test_resolver_matches_single_chunk():
    chunks = [
        ("uid-1", "The radar gain is 35 dBi nominal."),
        ("uid-2", "Other unrelated text."),
    ]
    rows = [ExtractionFieldProvenance(
        instance_id="i1", field_name="gain_dbi", value=35.0,
        supporting_snippet="gain is 35 dBi",
    )]
    resolve_field_provenance_uids(rows, chunks)
    assert rows[0].element_uid == "uid-1"


def test_resolver_handles_no_match():
    rows = [ExtractionFieldProvenance(
        instance_id="i1", field_name="x", value=1,
        supporting_snippet="not in any chunk",
    )]
    resolve_field_provenance_uids(rows, [("uid-1", "different text")])
    assert rows[0].element_uid is None


def test_resolver_picks_longest_chunk_on_ambiguity():
    chunks = [
        ("uid-1", "the antenna gain"),
        ("uid-2", "the antenna gain is 35 dBi nominal"),
    ]
    rows = [ExtractionFieldProvenance(
        instance_id="i1", field_name="gain_dbi", value=35.0,
        supporting_snippet="antenna gain",
    )]
    resolve_field_provenance_uids(rows, chunks)
    # Tiebreaker: longest chunk text wins.
    assert rows[0].element_uid == "uid-2"


def test_resolver_ignores_whitespace_differences():
    chunks = [("uid-1", "The   radar  gain   is  35 dBi.")]
    rows = [ExtractionFieldProvenance(
        instance_id="i1", field_name="gain_dbi", value=35.0,
        supporting_snippet="radar gain is 35 dBi",
    )]
    resolve_field_provenance_uids(rows, chunks)
    assert rows[0].element_uid == "uid-1"
