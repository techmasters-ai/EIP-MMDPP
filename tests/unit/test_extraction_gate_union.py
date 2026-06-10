"""Task 7 — G1 gate union into the candidate pool (cap-exempt) + diagnostics.

TDD tests written BEFORE implementation per TDD contract.

Contract under test (guarded-ranker spec §3):
  - The G1 gate scans ALL rows of the run (not just the dense pool).
  - Gated chunks (digit + pass-unit-signature token) join the candidate pool
    EXEMPT from the ``top_n_candidates`` cap so they flow through rerank +
    C5 scoring and the ``score_components_all`` capture.
  - Default off (``RetrievalProfile.unit_gate=False``) → byte-identical pool.
  - Pool order: dense-capped first, then beyond-cap gated (merged_pool order),
    then row-built gated (candidate_key order).
  - Diagnostics: ``unit_gate_total`` / ``unit_gate_added``.

Run:
    python3 -m pytest tests/unit/test_extraction_gate_union.py -v
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared helpers (mirrors tests/unit/test_extraction_chunk_search_dense_multi.py)
# ---------------------------------------------------------------------------

def _vec(*xs: float) -> list[float]:
    """1024-dim vector with the supplied first values, rest zero."""
    v = [0.0] * 1024
    for i, x in enumerate(xs):
        v[i] = float(x)
    return v


def _norm(v: list[float]) -> list[float]:
    """L2-normalise a vector (same as embed_texts does on output)."""
    arr = np.array(v, dtype=np.float32)
    n = np.linalg.norm(arr)
    if n > 0:
        arr = arr / n
    return arr.tolist()


def _row(self_ref: str, embedding: list[float] | None, **extra) -> dict:
    """Build a fake ArcadeDB ExtractionChunk row (fetch SELECT projection)."""
    return {
        "node_id": extra.pop("node_id", f"#170:{self_ref}"),
        "vertex_id": extra.pop("vertex_id", f"run-G:{self_ref}"),
        "self_ref": self_ref,
        "chunk_text": extra.pop("chunk_text", f"filler text for {self_ref}"),
        "embedding": embedding,
        "page_number": extra.pop("page_number", None),
        "modality": extra.pop("modality", "text"),
        "pipeline_run_id": extra.pop("pipeline_run_id", "run-G"),
        "chunk_index": extra.pop("chunk_index", -1),
        "source_refs": extra.pop("source_refs", []),
        "token_count": extra.pop("token_count", 0),
        **extra,
    }


def _fake_store(rows: list[dict]) -> Any:
    """SimpleNamespace mimicking ArcadeDBGraphStore (only _database +
    _client.query are touched by fetch_extraction_chunks_for_run)."""
    client = SimpleNamespace(query=AsyncMock(return_value=rows))
    return SimpleNamespace(_database="eip_knowledge_graph", _client=client)


def _make_signals(
    entity_query: str = "entity query text",
    field_queries: tuple = (),
    unit_signature: tuple[str, ...] = (),
):
    from app.services.extraction_query_builder import PassRetrievalSignals
    return PassRetrievalSignals(
        pass_name="test_pass",
        entity_doc="entity doc",
        entity_query=entity_query,
        field_queries=field_queries,
        lexical_terms=(),
        negative_terms=(),
        likely_sections=(),
        evidence_patterns=(),
        unit_signature=unit_signature,
    )


def _make_field_query(field_name: str, query_text: str):
    from app.services.extraction_query_builder import FieldRetrievalQuery
    return FieldRetrievalQuery(
        field_name=field_name,
        query_text=query_text,
        aliases=(),
        negative_terms=(),
        evidence_patterns=(),
        likely_sections=(),
        units=(),
    )


def _gate_rows(n: int = 12, gated_index: int = 11) -> list[dict]:
    """n rows with strictly descending entity cosine vs query (1,0,...).

    Row ``gated_index`` carries gate-firing text (digit + 'kw' unit token);
    every other row's text has digits but NO signature unit token.
    """
    rows = []
    for i in range(n):
        text = (
            "spec block 180 kw table"
            if i == gated_index
            else f"filler text {i} about the radar system"
        )
        rows.append(
            _row(
                f"c{i:02d}",
                _vec(1.0 - i * 0.05, 0.5),
                chunk_text=text,
            )
        )
    return rows


def _entity_cosine(i: float) -> float:
    """Hand-computed entity cosine for _gate_rows row i vs query (1,0,...)."""
    d0 = 1.0 - i * 0.05
    return d0 / math.sqrt(d0 * d0 + 0.25)


async def _run_full(rows, signals, cfg, *, n_queries: int = 1):
    """Drive search_extraction_chunks_multi_channel_full with mocked
    embed_texts (entity-query-aligned vectors) + a fake store."""
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_multi_channel_full,
    )

    store = _fake_store(rows)
    q_entity = _norm(_vec(1.0))
    q_field = _norm(_vec(0.0, 0.0, 1.0))
    vectors = [q_entity] + [q_field] * (n_queries - 1)

    with patch(
        "app.services.extraction_chunk_search.embed_texts",
        MagicMock(return_value=vectors),
    ):
        pool, diag, state = await search_extraction_chunks_multi_channel_full(
            signals, "run-G", cfg, store=store,
        )
    return pool, diag, state


# ---------------------------------------------------------------------------
# 1. Gate union — row outside the dense pool joins cap-exempt
# ---------------------------------------------------------------------------


class TestGateUnion:

    @pytest.mark.asyncio
    async def test_gated_row_beyond_cap_joins_pool(self):
        """12 rows / top_n_candidates=10; the dense-ranked-LAST row carries
        gate text ('spec block 180 kw table') → returned pool has 11 entries
        with the gated row appended cap-exempt."""
        from app.services.ontology_bundles import RetrievalProfile

        rows = _gate_rows(n=12, gated_index=11)
        signals = _make_signals(unit_signature=("kw",))
        cfg = RetrievalProfile(top_n_candidates=10, unit_gate=True)

        pool, diag, _state = await _run_full(rows, signals, cfg)

        keys = [mc.candidate_key for mc in pool]
        assert len(pool) == 11, f"expected 10 capped + 1 gated; got {keys}"
        # Dense-capped first, in dense order.
        assert keys[:10] == [f"run-G:c{i:02d}" for i in range(10)]
        # Gated row appended last.
        assert keys[10] == "run-G:c11"

        gated = pool[10]
        assert gated.gate_flags == {"unit"}
        assert "unit_gate" in gated.retrieval_sources
        # Row-built extra: vector_score = entity cosine from row_cosines.
        assert gated.vector_score == pytest.approx(_entity_cosine(11), abs=1e-5)
        assert gated.field_scores == {}
        assert gated.chunk_text == "spec block 180 kw table"

        # Diagnostics.
        assert diag.unit_gate_total == 1
        assert diag.unit_gate_added == 1
        assert diag.pool_size == 11

    @pytest.mark.asyncio
    async def test_gate_scans_all_rows_not_just_pool(self):
        """A gated row that is ALSO outside the entity-dense channel entirely
        (rank 12 of 12, cap 10) must still be admitted — the scan runs over
        the raw per-run rows, not over the merged pool."""
        from app.services.ontology_bundles import RetrievalProfile

        rows = _gate_rows(n=12, gated_index=11)
        signals = _make_signals(unit_signature=("kw",))
        cfg = RetrievalProfile(top_n_candidates=10, unit_gate=True)

        pool, _diag, state = await _run_full(rows, signals, cfg)

        # Sanity: the gated key is NOT in the dense channel results.
        dense_keys = {
            r.properties.get("vertex_id") for r in state.entity_dense
        }
        assert "run-G:c11" not in dense_keys
        assert "run-G:c11" in {mc.candidate_key for mc in pool}


# ---------------------------------------------------------------------------
# 2. Byte-identical default — flag off / signature empty
# ---------------------------------------------------------------------------


class TestGateUnionDefaultOff:

    @pytest.mark.asyncio
    async def test_flag_off_pool_byte_identical(self):
        """unit_gate=False (default) → pool is EXACTLY the first 10 keys in
        dense order; no gate_flags anywhere; diagnostics zeros."""
        from app.services.ontology_bundles import RetrievalProfile

        rows = _gate_rows(n=12, gated_index=11)
        signals = _make_signals(unit_signature=("kw",))
        cfg = RetrievalProfile(top_n_candidates=10)  # unit_gate default False

        pool, diag, _state = await _run_full(rows, signals, cfg)

        keys = [mc.candidate_key for mc in pool]
        assert keys == [f"run-G:c{i:02d}" for i in range(10)]
        for mc in pool:
            assert mc.gate_flags == set(), f"{mc.candidate_key} has gate_flags"
            assert "unit_gate" not in mc.retrieval_sources
        assert diag.unit_gate_total == 0
        assert diag.unit_gate_added == 0
        assert diag.pool_size == 10

    @pytest.mark.asyncio
    async def test_flag_on_empty_signature_byte_identical(self):
        """unit_gate=True but unit_signature=() → gate cannot fire; pool is
        byte-identical to the default path."""
        from app.services.ontology_bundles import RetrievalProfile

        rows = _gate_rows(n=12, gated_index=11)
        signals = _make_signals(unit_signature=())
        cfg = RetrievalProfile(top_n_candidates=10, unit_gate=True)

        pool, diag, _state = await _run_full(rows, signals, cfg)

        assert [mc.candidate_key for mc in pool] == [
            f"run-G:c{i:02d}" for i in range(10)
        ]
        assert all(mc.gate_flags == set() for mc in pool)
        assert diag.unit_gate_total == 0
        assert diag.unit_gate_added == 0


# ---------------------------------------------------------------------------
# 3. In-pool marking — gated member inside the cap is flagged, NOT duplicated
# ---------------------------------------------------------------------------


class TestGateUnionInPoolMarking:

    @pytest.mark.asyncio
    async def test_in_pool_gated_member_flagged_not_duplicated(self):
        """A gated chunk already inside the cap gets gate_flags={'unit'} +
        'unit_gate' source but the pool length is unchanged (no dup)."""
        from app.services.ontology_bundles import RetrievalProfile

        rows = _gate_rows(n=5, gated_index=2)  # row 2 well inside cap 10
        signals = _make_signals(unit_signature=("kw",))
        cfg = RetrievalProfile(top_n_candidates=10, unit_gate=True)

        pool, diag, _state = await _run_full(rows, signals, cfg)

        keys = [mc.candidate_key for mc in pool]
        assert len(pool) == 5, f"no duplication expected; got {keys}"
        assert len(set(keys)) == 5
        by_key = {mc.candidate_key: mc for mc in pool}
        gated = by_key["run-G:c02"]
        assert gated.gate_flags == {"unit"}
        assert "unit_gate" in gated.retrieval_sources
        assert "dense" in gated.retrieval_sources  # still the dense candidate
        # The other members stay unflagged.
        for k, mc in by_key.items():
            if k != "run-G:c02":
                assert mc.gate_flags == set()
        assert diag.unit_gate_total == 1
        assert diag.unit_gate_added == 0  # in-pool: marked, nothing added


# ---------------------------------------------------------------------------
# 4. Beyond-cap merged_pool member — re-admitted as the EXISTING object
# ---------------------------------------------------------------------------


class TestGateUnionBeyondCapReadmission:

    @pytest.mark.asyncio
    async def test_gated_merged_pool_member_beyond_cap_readmitted(self):
        """A gated chunk that IS in merged_pool (via the field-dense channel)
        but fell beyond the top_n cap is re-admitted as the SAME object —
        field channel evidence (field_scores / 'field:<name>' source) is
        preserved, proving it was not rebuilt from the raw row."""
        from app.services.ontology_bundles import RetrievalProfile

        # a,b,c align with the entity query (dim 0); d,e align with the field
        # query (dim 2) so they enter merged_pool ONLY via field_dense and sort
        # behind the cap (vector_score=None → after dense-scored candidates).
        rows = [
            _row("a", _vec(1.0, 0.0), chunk_text="alpha filler"),
            _row("b", _vec(0.9, 0.1), chunk_text="bravo filler"),
            _row("c", _vec(0.8, 0.2), chunk_text="charlie filler"),
            _row("d", _vec(0.0, 0.0, 1.0), chunk_text="pump output 200 kw rating"),
            _row("e", _vec(0.0, 0.0, 0.9), chunk_text="echo filler"),
        ]
        signals = _make_signals(
            field_queries=(_make_field_query("tx_peak_power_kw", "peak power"),),
            unit_signature=("kw",),
        )
        cfg = RetrievalProfile(
            top_n_candidates=3, field_query_top_k=2, unit_gate=True
        )

        pool, diag, _state = await _run_full(rows, signals, cfg, n_queries=2)

        keys = [mc.candidate_key for mc in pool]
        assert keys[:3] == ["run-G:a", "run-G:b", "run-G:c"], keys
        assert "run-G:d" in keys, f"gated beyond-cap member must be re-admitted; {keys}"
        assert "run-G:e" not in keys, "ungated beyond-cap member stays out"
        assert len(keys) == len(set(keys)) == 4

        d = pool[keys.index("run-G:d")]
        assert d.gate_flags == {"unit"}
        assert "unit_gate" in d.retrieval_sources
        # Existing merged object reused — field channel evidence preserved.
        assert "field:tx_peak_power_kw" in d.retrieval_sources
        assert d.field_scores, "field_scores from the field-dense channel must survive"
        assert diag.unit_gate_total == 1
        assert diag.unit_gate_added == 1


# ---------------------------------------------------------------------------
# 5. Diagnostics dataclass — new fields default 0 (old constructors fine)
# ---------------------------------------------------------------------------


class TestMultiChannelDiagnosticsDefaults:

    def test_unit_gate_fields_default_zero(self):
        from app.services.extraction_chunk_search import MultiChannelDiagnostics

        diag = MultiChannelDiagnostics(
            raw_row_count=0,
            entity_dense_count=0,
            field_dense_total_count=0,
            lexical_hit_count=0,
            pattern_hit_count=0,
            pool_size=0,
            per_field_dense_counts={},
        )
        assert diag.unit_gate_total == 0
        assert diag.unit_gate_added == 0
