"""Patch-0006 regression: the delta-IR normalizer must stamp the batch's
AUTHORITATIVE POSITIONAL lineage (self_refs / chunk_indexes / page_numbers) on
EVERY normalized node and relationship, and keep the LLM's cited evidence_ids
separate as `cited_refs` / `evidence_ids` (explicit only).

The old code did `out["evidence_ids"] = valid or list(batch_evidence_ids)`,
which CLOBBERED a node's lineage with the whole batch evidence pool whenever the
LLM cited nothing — coarsening per-entity lineage. The fix makes positional
self_refs authoritative and `evidence_ids`/`cited_refs` hold ONLY explicit
citations (possibly empty).

The fix ships as a library PATCH applied at Docker build, so this test applies
the full patch stack (0001-0006) to a temp repo copy and imports the PATCHED
module in a clean subprocess. Mirrors test_chunked_batches_stores_chunk_metadata.py.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SERVICE_ROOT = _HERE.parent
_REPO = _SERVICE_ROOT / "repo"
_PATCHES = _SERVICE_ROOT / "patches"


def _apply_patches(dst_repo: Path) -> None:
    for p in sorted(_PATCHES.glob("*.patch")):
        subprocess.run(
            ["patch", "-p1", "--fuzz=0", "-i", str(p)],
            cwd=str(dst_repo), check=True, capture_output=True, text=True,
        )


_DRIVER = textwrap.dedent(
    '''
    import sys
    sys.path.insert(0, sys.argv[1])

    from docling_graph.core.extractors.contracts.delta.ir_normalizer import (
        DeltaIrNormalizerConfig,
        normalize_delta_ir_batch_results,
    )
    from docling_graph.core.extractors.contracts.delta.catalog import (
        DeltaNodeCatalog,
        DeltaNodeSpec,
    )
    from docling_graph.core.extractors.contracts.delta.helpers import DedupPolicy

    PATH = "#/systems"

    catalog = DeltaNodeCatalog(
        nodes=[
            DeltaNodeSpec(
                path=PATH,
                node_type="System",
                id_fields=[],          # no identity -> trivial, every node kept
                kind="entity",
                parent_path="",        # top-level -> no parent normalization issues
                property_fields=["name"],
            )
        ],
        field_aliases={},
    )

    dedup_policy = {
        PATH: DedupPolicy(
            path=PATH,
            node_type="System",
            identity_fields=(),
            fallback_text_fields=(),
            allowed_match_fields=(),
            is_entity=True,
        )
    }

    config = DeltaIrNormalizerConfig(attach_provenance=True, validate_paths=False)

    # Two chunks, indices 0 and 1, each with its own self_ref + page.
    chunk_metadata = [
        {"chunk_id": 0, "self_refs": ["#/texts/3"], "page_numbers": [1],
         "evidence_ids": ["#/texts/3"], "evidence_units": [], "token_count": 10},
        {"chunk_id": 1, "self_refs": ["#/texts/4"], "page_numbers": [2],
         "evidence_ids": ["#/texts/4"], "evidence_units": [], "token_count": 12},
    ]

    # ONE batch spanning both chunks: (chunk_index, self_ref, token_count) tuples.
    batch_plan = [[(0, "#/texts/3", 10), (1, "#/texts/4", 12)]]

    # One graph for the single batch: a cited node, an uncited node, and a rel.
    batch_results = [
        {
            "nodes": [
                {
                    "path": PATH,
                    "node_type": "System",
                    "ids": {},
                    "properties": {"name": "Cited System"},
                    "evidence_ids": ["#/texts/3"],   # LLM cited chunk 0 only
                },
                {
                    "path": PATH,
                    "node_type": "System",
                    "ids": {},
                    "properties": {"name": "Uncited System"},
                    # NO evidence_ids -> LLM cited nothing
                },
            ],
            "relationships": [
                {
                    "edge_label": "RELATED_TO",
                    "source_path": PATH,
                    "source_ids": {},
                    "target_path": PATH,
                    "target_ids": {},
                    "properties": {},
                    # NO evidence_ids on the rel either
                }
            ],
        }
    ]

    normalized_results, stats = normalize_delta_ir_batch_results(
        batch_results=batch_results,
        batch_plan=batch_plan,
        chunk_metadata=chunk_metadata,
        catalog=catalog,
        dedup_policy=dedup_policy,
        config=config,
    )

    assert len(normalized_results) == 1, normalized_results
    nodes = normalized_results[0]["nodes"]
    rels = normalized_results[0]["relationships"]
    assert len(nodes) == 2, f"expected 2 nodes, got {nodes!r}"
    assert len(rels) == 1, f"expected 1 rel, got {rels!r}"

    def _by_name(name):
        for n in nodes:
            if n.get("properties", {}).get("name") == name:
                return n
        raise AssertionError(f"node {name!r} not found in {nodes!r}")

    cited = _by_name("Cited System")
    uncited = _by_name("Uncited System")

    BATCH_REFS = ["#/texts/3", "#/texts/4"]

    # --- Cited node: explicit citation preserved, positional refs authoritative.
    cp = cited["provenance"]
    assert cp.get("cited_refs") == ["#/texts/3"], (
        f"cited node cited_refs should equal explicit LLM citation, got {cp!r}"
    )
    assert cp.get("evidence_ids") == ["#/texts/3"], (
        f"cited node evidence_ids should equal explicit citation only, got {cp!r}"
    )
    assert sorted(cp.get("self_refs") or []) == sorted(BATCH_REFS), (
        f"cited node must carry BOTH batch self_refs, got {cp!r}"
    )
    assert sorted(cp.get("chunk_indexes") or []) == [0, 1], (
        f"cited node must carry batch chunk_indexes, got {cp!r}"
    )
    assert sorted(cp.get("page_numbers") or []) == [1, 2], (
        f"cited node must carry batch page_numbers, got {cp!r}"
    )
    assert "property_evidence" in cp, f"property_evidence must be present: {cp!r}"

    # --- Uncited node: cited_refs empty, but full positional refs (NOT the old
    #     batch-pool clobber where evidence_ids == whole batch pool).
    up = uncited["provenance"]
    assert up.get("cited_refs") == [], (
        f"uncited node cited_refs must be empty, got {up!r}"
    )
    assert up.get("evidence_ids") == [], (
        "REGRESSION: uncited node evidence_ids must be EXPLICIT-ONLY (empty), "
        f"not the whole batch pool, got {up!r}"
    )
    assert sorted(up.get("self_refs") or []) == sorted(BATCH_REFS), (
        f"uncited node must STILL carry BOTH batch self_refs (positional), got {up!r}"
    )
    assert sorted(up.get("chunk_indexes") or []) == [0, 1], (
        f"uncited node must carry batch chunk_indexes, got {up!r}"
    )
    assert sorted(up.get("page_numbers") or []) == [1, 2], (
        f"uncited node must carry batch page_numbers, got {up!r}"
    )

    # --- Relationship: same positional lineage, empty cited_refs.
    rp = rels[0]["provenance"]
    assert rp.get("cited_refs") == [], f"uncited rel cited_refs must be empty, got {rp!r}"
    assert rp.get("evidence_ids") == [], (
        f"uncited rel evidence_ids must be explicit-only (empty), got {rp!r}"
    )
    assert sorted(rp.get("self_refs") or []) == sorted(BATCH_REFS), (
        f"rel must carry BOTH batch self_refs (positional), got {rp!r}"
    )
    assert sorted(rp.get("chunk_indexes") or []) == [0, 1], (
        f"rel must carry batch chunk_indexes, got {rp!r}"
    )
    assert sorted(rp.get("page_numbers") or []) == [1, 2], (
        f"rel must carry batch page_numbers, got {rp!r}"
    )

    print("DRIVER_OK")
    '''
)


def test_positional_provenance_stamp(tmp_path):
    dst = tmp_path / "repo"
    subprocess.run(["cp", "-a", str(_REPO), str(dst)], check=True)
    _apply_patches(dst)
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER)
    proc = subprocess.run([sys.executable, str(driver), str(dst)],
                          capture_output=True, text=True)
    assert "DRIVER_OK" in proc.stdout, (
        f"patched-artifact check failed.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
