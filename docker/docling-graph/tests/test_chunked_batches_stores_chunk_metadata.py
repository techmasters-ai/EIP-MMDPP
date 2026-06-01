"""Patch-0002 regression: the CHUNKED-BATCHES branch of
ExtractionStage._extract_from_docling_document MUST publish chunk_metadata to
doc_processor.last_chunk_metadata. app/main.py builds the chunk_to_self_refs /
chunk_to_page_numbers provenance maps from it; empty -> element_uid="" -> rows
dropped -> no EXTRACTED_FROM. The fix is in a library PATCH applied at Docker
build, so this test applies the patch stack to a temp repo copy and imports the
PATCHED module in a clean subprocess.
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
    from docling_graph.pipeline.stages import ExtractionStage

    CHUNK_META = [
        {"chunk_id": 0, "self_refs": ["#/texts/3"], "page_numbers": [1],
         "evidence_ids": ["#/texts/3"], "evidence_units": [], "token_count": 10},
        {"chunk_id": 1, "self_refs": ["#/texts/4"], "page_numbers": [2],
         "evidence_ids": ["#/texts/4"], "evidence_units": [], "token_count": 12},
    ]

    class FakeDocProcessor:
        chunker = object()
        last_chunk_metadata = []
        def extract_chunks_with_metadata(self, document):
            return (["chunk one text", "chunk two text"], CHUNK_META)

    class FakeBackend:
        def extract_from_chunk_batches(self, *, chunks, chunk_metadata, template, context):
            return {"ok": True}
        def extract_from_markdown(self, **kw):
            raise AssertionError("must take CHUNKED-BATCHES path, not single-blob")

    class FakeExtractor:
        _extraction_contract = "delta"
        def __init__(self):
            self.doc_processor = FakeDocProcessor()
            self.backend = FakeBackend()

    class FakeDoc:
        def export_to_markdown(self):
            raise AssertionError("single-blob path should not run")

    class FakeContext:
        def __init__(self):
            self.extractor = FakeExtractor()
            self.docling_document = FakeDoc()
            self.template = type("T", (), {})
            self.trace_data = None

    stage = ExtractionStage()
    ctx = FakeContext()
    models = stage._extract_from_docling_document(ctx)
    dp = ctx.extractor.doc_processor
    assert models, "expected an extracted model from the chunked path"
    assert dp.last_chunk_metadata, (
        "REGRESSION: chunked-batches path did not store chunk_metadata on "
        "doc_processor.last_chunk_metadata (empty => no EXTRACTED_FROM lineage)"
    )
    refs = [r for row in dp.last_chunk_metadata for r in (row.get("self_refs") or [])]
    pages = [p for row in dp.last_chunk_metadata for p in (row.get("page_numbers") or [])]
    assert "#/texts/3" in refs, f"self_refs not preserved: {refs!r}"
    assert 1 in pages and 2 in pages, f"page_numbers not preserved: {pages!r}"
    print("DRIVER_OK")
    '''
)


def test_chunked_batches_stores_last_chunk_metadata(tmp_path):
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
