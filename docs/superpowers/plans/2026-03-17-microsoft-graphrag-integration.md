# Microsoft GraphRAG Integration Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the custom GraphRAG implementation with Microsoft's GraphRAG library, using the Neo4j ontology graph built by Docling-Graph as input, with all four search methods (Local, Global, DRIFT, Basic), scheduled indexing + auto-tuning, and flexible LLM provider support (Ollama/OpenAI).

**Architecture:** Neo4j entities/relationships (populated by Docling → Docling-Graph) are exported via a bridge layer into GraphRAG's Parquet format. GraphRAG handles community detection, report generation, embeddings, and search. Celery Beat schedules indexing (hourly) and prompt auto-tuning (daily). The existing retrieval API contract is preserved with two new search strategies added.

**Tech Stack:** Microsoft GraphRAG (`graphrag>=1.0.0`), pandas, pyarrow, LanceDB (bundled with GraphRAG), Neo4j, Celery, FastAPI

---

## File Structure

### New Files
- `app/services/graphrag_bridge.py` — Neo4j/Postgres → Parquet export (ontology-driven)
- `app/services/graphrag_config.py` — GraphRAG configuration builder (settings.yaml generation)
- `app/services/graphrag_prompts.py` — Manual community report prompt + search prompts
- `tests/unit/test_graphrag_bridge.py` — Bridge layer tests
- `tests/unit/test_graphrag_config.py` — Config builder tests
- `tests/unit/test_graphrag_prompts.py` — Prompt generation tests
- `alembic/versions/0008_drop_custom_graphrag_tables.py` — Drop old tables

### Modified Files
- `app/services/graphrag_service.py` — Complete rewrite: GraphRAGIndexer + GraphRAGSearcher
- `app/workers/graphrag_tasks.py` — Rewrite: indexing + auto-tuning Celery tasks
- `app/workers/celery_app.py` — Add auto-tuning beat schedule
- `app/schemas/retrieval.py` — Add `graphrag_drift` and `graphrag_basic` strategies
- `app/api/v1/retrieval.py` — Wire new strategies + tune endpoint
- `app/config.py` — Add new GraphRAG config fields
- `docker-compose.yml` — Add `graphrag_data` volume
- `pyproject.toml` — Add `pyarrow` dependency, remove `graspologic`
- `tests/unit/test_graphrag_service.py` — Complete rewrite for new service
- `tests/conftest.py` — Add GraphRAG-specific fixtures

### Removed Files
- None removed entirely, but `graspologic` dependency dropped from pyproject.toml

---

## Chunk 1: Configuration, Prompts, and Bridge Layer

### Task 0: Add Configuration Fields

**Files:**
- Modify: `app/config.py:124-128`
- Test: `tests/unit/test_graphrag_config.py` (new)

- [ ] **Step 1: Write failing test for new config fields**

```python
# tests/unit/test_graphrag_config.py
"""Unit tests for GraphRAG configuration builder."""

import pytest

pytestmark = pytest.mark.unit


class TestGraphRAGSettings:
    def test_default_graphrag_settings(self):
        """New GraphRAG settings have correct defaults."""
        from app.config import Settings

        s = Settings(
            _env_file=None,
            postgres_password="test",
            neo4j_password="test",
        )
        assert s.graphrag_llm_provider == "ollama"
        assert s.graphrag_llm_model == "llama3.2"
        assert s.graphrag_llm_api_base == "http://localhost:11434/v1"
        assert s.graphrag_api_key == ""
        assert s.graphrag_embedding_model == "nomic-embed-text"
        assert s.graphrag_data_dir == "/app/graphrag_data"
        assert s.graphrag_community_level == 2
        assert s.graphrag_tune_interval_minutes == 1440
        assert s.graphrag_indexing_enabled is True
        assert s.graphrag_indexing_interval_minutes == 60
        assert s.graphrag_max_cluster_size == 10

    def test_openai_provider_config(self):
        """Settings accept OpenAI provider configuration."""
        from app.config import Settings

        s = Settings(
            _env_file=None,
            postgres_password="test",
            neo4j_password="test",
            graphrag_llm_provider="openai",
            graphrag_llm_model="gpt-4o",
            graphrag_api_key="sk-test",
            graphrag_embedding_model="text-embedding-3-small",
        )
        assert s.graphrag_llm_provider == "openai"
        assert s.graphrag_api_key == "sk-test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_graphrag_config.py -v`
Expected: FAIL — `graphrag_llm_provider` attribute not found

- [ ] **Step 3: Add new settings fields to config.py**

Replace the existing GraphRAG settings block (`app/config.py:124-128`) with:

```python
    # GraphRAG (Microsoft GraphRAG — community detection + reports + search)
    graphrag_indexing_enabled: bool = True
    graphrag_indexing_interval_minutes: int = 60
    graphrag_max_cluster_size: int = 10
    graphrag_community_level: int = 2
    graphrag_data_dir: str = "/app/graphrag_data"
    # LLM provider for GraphRAG (ollama | openai)
    graphrag_llm_provider: str = "ollama"
    graphrag_llm_model: str = "llama3.2"
    graphrag_llm_api_base: str = "http://localhost:11434/v1"
    graphrag_api_key: str = ""
    # Embedding model for GraphRAG's LanceDB store
    graphrag_embedding_model: str = "nomic-embed-text"
    # Auto-tuning schedule (minutes, default 24h)
    graphrag_tune_interval_minutes: int = 1440
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_graphrag_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/unit/test_graphrag_config.py
git commit -m "feat(graphrag): add Microsoft GraphRAG configuration fields"
```

---

### Task 1: Create GraphRAG Configuration Builder

**Files:**
- Create: `app/services/graphrag_config.py`
- Test: `tests/unit/test_graphrag_config.py` (append)

- [ ] **Step 1: Write failing test for config builder**

Append to `tests/unit/test_graphrag_config.py`:

```python
class TestBuildGraphRAGConfig:
    def test_builds_config_with_ollama(self, tmp_path):
        """Config builder creates valid GraphRAG config for Ollama."""
        from app.services.graphrag_config import build_graphrag_config
        from unittest.mock import MagicMock

        settings = MagicMock()
        settings.graphrag_llm_provider = "ollama"
        settings.graphrag_llm_model = "llama3.2"
        settings.graphrag_llm_api_base = "http://ollama:11434/v1"
        settings.graphrag_api_key = ""
        settings.graphrag_embedding_model = "nomic-embed-text"
        settings.graphrag_data_dir = str(tmp_path / "graphrag_data")
        settings.graphrag_community_level = 2
        settings.graphrag_max_cluster_size = 10

        config = build_graphrag_config(settings)

        # Config should be a GraphRagConfig object
        assert config is not None
        assert config.storage.base_dir == str(tmp_path / "graphrag_data" / "output")

    def test_builds_config_with_openai(self, tmp_path):
        """Config builder creates valid GraphRAG config for OpenAI."""
        from app.services.graphrag_config import build_graphrag_config
        from unittest.mock import MagicMock

        settings = MagicMock()
        settings.graphrag_llm_provider = "openai"
        settings.graphrag_llm_model = "gpt-4o"
        settings.graphrag_llm_api_base = ""
        settings.graphrag_api_key = "sk-test"
        settings.graphrag_embedding_model = "text-embedding-3-small"
        settings.graphrag_data_dir = str(tmp_path / "graphrag_data")
        settings.graphrag_community_level = 2
        settings.graphrag_max_cluster_size = 10

        config = build_graphrag_config(settings)
        assert config is not None

    def test_prompts_dir_created(self, tmp_path):
        """Config builder creates prompts directory."""
        from app.services.graphrag_config import build_graphrag_config
        from unittest.mock import MagicMock

        settings = MagicMock()
        settings.graphrag_llm_provider = "ollama"
        settings.graphrag_llm_model = "llama3.2"
        settings.graphrag_llm_api_base = "http://ollama:11434/v1"
        settings.graphrag_api_key = ""
        settings.graphrag_embedding_model = "nomic-embed-text"
        settings.graphrag_data_dir = str(tmp_path / "graphrag_data")
        settings.graphrag_community_level = 2
        settings.graphrag_max_cluster_size = 10

        build_graphrag_config(settings)
        assert (tmp_path / "graphrag_data" / "prompts").is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_graphrag_config.py::TestBuildGraphRAGConfig -v`
Expected: FAIL — `graphrag_config` module not found

- [ ] **Step 3: Implement config builder**

```python
# app/services/graphrag_config.py
"""Build Microsoft GraphRAG configuration from application settings."""

import logging
from pathlib import Path

from graphrag.config.create_graphrag_config import create_graphrag_config

logger = logging.getLogger(__name__)


def build_graphrag_config(settings):
    """Build a GraphRagConfig from application settings.

    Supports both Ollama (via OpenAI-compatible API) and native OpenAI.
    """
    data_dir = Path(settings.graphrag_data_dir)
    output_dir = data_dir / "output"
    prompts_dir = data_dir / "prompts"
    cache_dir = data_dir / "cache"

    for d in (output_dir, prompts_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Model configuration — Ollama uses openai_chat type with custom api_base
    if settings.graphrag_llm_provider == "ollama":
        chat_config = {
            "type": "openai_chat",
            "model": settings.graphrag_llm_model,
            "api_key": settings.graphrag_api_key or "ollama",
            "api_base": settings.graphrag_llm_api_base,
        }
        embedding_config = {
            "type": "openai_embedding",
            "model": settings.graphrag_embedding_model,
            "api_key": settings.graphrag_api_key or "ollama",
            "api_base": settings.graphrag_llm_api_base,
        }
    else:
        chat_config = {
            "type": "openai_chat",
            "model": settings.graphrag_llm_model,
            "api_key": settings.graphrag_api_key,
        }
        embedding_config = {
            "type": "openai_embedding",
            "model": settings.graphrag_embedding_model,
            "api_key": settings.graphrag_api_key,
        }

    config_dict = {
        "models": {
            "default_chat_model": chat_config,
            "default_embedding_model": embedding_config,
        },
        "storage": {
            "type": "file",
            "base_dir": str(output_dir),
        },
        "cache": {
            "type": "file",
            "base_dir": str(cache_dir),
        },
        "reporting": {
            "type": "file",
            "base_dir": str(data_dir / "logs"),
        },
        "community_reports": {
            "max_length": 3000,
            "max_input_length": 16000,
        },
        "vector_store": {
            "default_entity_description": {
                "type": "lancedb",
                "db_uri": str(output_dir / "lancedb"),
            },
            "default_community_full_content": {
                "type": "lancedb",
                "db_uri": str(output_dir / "lancedb"),
            },
        },
        "local_search": {
            "prompt": str(prompts_dir / "local_search_system_prompt.txt"),
        },
        "global_search": {
            "map_prompt": str(prompts_dir / "global_search_map_system_prompt.txt"),
            "reduce_prompt": str(prompts_dir / "global_search_reduce_system_prompt.txt"),
        },
        "drift_search": {
            "prompt": str(prompts_dir / "drift_search_system_prompt.txt"),
        },
        "basic_search": {
            "prompt": str(prompts_dir / "basic_search_system_prompt.txt"),
        },
    }

    config = create_graphrag_config(config_dict, str(data_dir))
    logger.info("GraphRAG config built: provider=%s, model=%s, data_dir=%s",
                settings.graphrag_llm_provider, settings.graphrag_llm_model, data_dir)
    return config
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_graphrag_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/graphrag_config.py tests/unit/test_graphrag_config.py
git commit -m "feat(graphrag): add GraphRAG configuration builder with Ollama/OpenAI support"
```

---

### Task 2: Create Military Ontology Prompts

**Files:**
- Create: `app/services/graphrag_prompts.py`
- Test: `tests/unit/test_graphrag_prompts.py` (new)

- [ ] **Step 1: Write failing test for prompt generation**

```python
# tests/unit/test_graphrag_prompts.py
"""Unit tests for GraphRAG military ontology prompts."""

import pytest

pytestmark = pytest.mark.unit


class TestCommunityReportPrompt:
    def test_prompt_contains_ontology_layers(self):
        from app.services.graphrag_prompts import get_community_report_prompt

        prompt = get_community_report_prompt()
        assert "LAYER 1" in prompt
        assert "LAYER 2" in prompt
        assert "LAYER 3" in prompt
        assert "LAYER 4" in prompt
        assert "LAYER 5" in prompt

    def test_prompt_contains_key_entity_types(self):
        from app.services.graphrag_prompts import get_community_report_prompt

        prompt = get_community_report_prompt()
        for entity_type in ["PLATFORM", "RADAR_SYSTEM", "MISSILE_SYSTEM",
                            "FREQUENCY_BAND", "SEEKER", "CAPABILITY"]:
            assert entity_type in prompt

    def test_prompt_contains_key_relationships(self):
        from app.services.graphrag_prompts import get_community_report_prompt

        prompt = get_community_report_prompt()
        for rel in ["PART_OF", "CONTAINS", "OPERATES_IN_BAND", "CUES",
                     "TRACKS", "ENGAGES", "LAUNCHES"]:
            assert rel in prompt

    def test_prompt_contains_scoring_weights(self):
        from app.services.graphrag_prompts import get_community_report_prompt

        prompt = get_community_report_prompt()
        assert "0.95" in prompt
        assert "0.90" in prompt
        assert "0.85" in prompt


class TestSearchPrompts:
    def test_local_search_prompt_exists(self):
        from app.services.graphrag_prompts import get_local_search_prompt

        prompt = get_local_search_prompt()
        assert len(prompt) > 100
        assert "military" in prompt.lower()

    def test_global_search_map_prompt_exists(self):
        from app.services.graphrag_prompts import get_global_search_map_prompt

        prompt = get_global_search_map_prompt()
        assert len(prompt) > 100

    def test_global_search_reduce_prompt_exists(self):
        from app.services.graphrag_prompts import get_global_search_reduce_prompt

        prompt = get_global_search_reduce_prompt()
        assert len(prompt) > 100

    def test_drift_search_prompt_exists(self):
        from app.services.graphrag_prompts import get_drift_search_prompt

        prompt = get_drift_search_prompt()
        assert len(prompt) > 100

    def test_basic_search_prompt_exists(self):
        from app.services.graphrag_prompts import get_basic_search_prompt

        prompt = get_basic_search_prompt()
        assert len(prompt) > 100


class TestWritePromptFiles:
    def test_writes_all_prompt_files(self, tmp_path):
        from app.services.graphrag_prompts import write_prompt_files

        write_prompt_files(tmp_path)
        expected_files = [
            "community_report.txt",
            "local_search_system_prompt.txt",
            "global_search_map_system_prompt.txt",
            "global_search_reduce_system_prompt.txt",
            "drift_search_system_prompt.txt",
            "basic_search_system_prompt.txt",
        ]
        for fname in expected_files:
            assert (tmp_path / fname).exists(), f"Missing prompt file: {fname}"
            content = (tmp_path / fname).read_text()
            assert len(content) > 50, f"Prompt file too short: {fname}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_graphrag_prompts.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement prompt module**

Create `app/services/graphrag_prompts.py` with the full military ontology community report prompt (from the brainstorming design), plus search prompts for Local, Global (map+reduce), DRIFT, and Basic — all grounded in the military ontology domain. Include a `write_prompt_files(prompts_dir: Path)` function that writes all prompts to disk for GraphRAG to read.

The community report prompt should be the full text from Design Section 4 of our brainstorming session. The search prompts should instruct the LLM to interpret results through a military systems analysis lens, referencing the 5-layer ontology structure.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_graphrag_prompts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/graphrag_prompts.py tests/unit/test_graphrag_prompts.py
git commit -m "feat(graphrag): add military ontology prompts for community reports and search"
```

---

### Task 3: Create Neo4j → Parquet Bridge Layer

**Files:**
- Create: `app/services/graphrag_bridge.py`
- Test: `tests/unit/test_graphrag_bridge.py` (new)

This is the most critical custom component. It exports entities, relationships, text units, and documents from Neo4j and Postgres into pandas DataFrames matching GraphRAG's expected Parquet schema.

- [ ] **Step 1: Write failing test for entity export**

```python
# tests/unit/test_graphrag_bridge.py
"""Unit tests for the Neo4j → GraphRAG Parquet bridge layer."""

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


def _mock_neo4j_records(data: list[dict]):
    """Create a mock Neo4j result that iterates over records."""
    result = MagicMock()
    records = []
    for row in data:
        rec = MagicMock()
        rec.data.return_value = row
        rec.__getitem__ = lambda self, key, r=row: r[key]
        records.append(rec)
    result.__iter__ = MagicMock(return_value=iter(records))
    return result


class TestExportEntities:
    def test_exports_entity_dataframe(self, mock_neo4j_driver):
        from app.services.graphrag_bridge import export_entities

        driver, session = mock_neo4j_driver
        session.run.return_value = _mock_neo4j_records([
            {
                "name": "S-400 Triumf",
                "entity_type": "MISSILE_SYSTEM",
                "description": "Russian long-range SAM system",
                "id": "s400-1",
            },
            {
                "name": "91N6E",
                "entity_type": "RADAR_SYSTEM",
                "description": "Battle management radar for S-400",
                "id": "91n6e-1",
            },
        ])

        df = export_entities(driver)
        assert len(df) == 2
        assert "id" in df.columns
        assert "title" in df.columns
        assert "type" in df.columns
        assert "description" in df.columns
        assert "human_readable_id" in df.columns
        assert df.iloc[0]["title"] == "S-400 Triumf"
        assert df.iloc[0]["type"] == "MISSILE_SYSTEM"

    def test_empty_graph_returns_empty_df(self, mock_neo4j_driver):
        from app.services.graphrag_bridge import export_entities

        driver, session = mock_neo4j_driver
        session.run.return_value = _mock_neo4j_records([])

        df = export_entities(driver)
        assert len(df) == 0
        assert "id" in df.columns

    def test_exception_returns_empty_df(self, mock_neo4j_driver):
        from app.services.graphrag_bridge import export_entities

        driver, session = mock_neo4j_driver
        session.run.side_effect = Exception("connection failed")

        df = export_entities(driver)
        assert len(df) == 0


class TestExportRelationships:
    def test_exports_relationship_dataframe(self, mock_neo4j_driver):
        from app.services.graphrag_bridge import export_relationships

        driver, session = mock_neo4j_driver
        session.run.return_value = _mock_neo4j_records([
            {
                "source": "S-400 Triumf",
                "target": "91N6E",
                "relationship": "CONTAINS",
                "description": "S-400 contains 91N6E radar",
            },
        ])

        df = export_relationships(driver)
        assert len(df) == 1
        assert "id" in df.columns
        assert "source" in df.columns
        assert "target" in df.columns
        assert "weight" in df.columns
        assert "description" in df.columns
        assert df.iloc[0]["weight"] == 0.90  # CONTAINS weight from ontology

    def test_default_weight_for_unknown_relationship(self, mock_neo4j_driver):
        from app.services.graphrag_bridge import export_relationships

        driver, session = mock_neo4j_driver
        session.run.return_value = _mock_neo4j_records([
            {
                "source": "A",
                "target": "B",
                "relationship": "UNKNOWN_REL",
                "description": "",
            },
        ])

        df = export_relationships(driver)
        assert df.iloc[0]["weight"] == 0.70  # default weight


class TestExportTextUnits:
    def test_exports_text_units_with_entity_links(self):
        from app.services.graphrag_bridge import export_text_units

        db = MagicMock()
        # Mock the SQL query result: chunk_id, content_text, document_id, entity_names
        db.execute.return_value.fetchall.return_value = [
            ("chunk-1", "The S-400 system uses...", "doc-1", ["S-400 Triumf", "91N6E"]),
            ("chunk-2", "Radar characteristics...", "doc-1", ["91N6E"]),
        ]

        df = export_text_units(db)
        assert len(df) == 2
        assert "id" in df.columns
        assert "text" in df.columns
        assert "document_ids" in df.columns
        assert "entity_ids" in df.columns


class TestExportDocuments:
    def test_exports_documents(self):
        from app.services.graphrag_bridge import export_documents

        db = MagicMock()
        db.execute.return_value.fetchall.return_value = [
            ("doc-1", "Technical Manual TM-123", "Full document text..."),
        ]

        df = export_documents(db)
        assert len(df) == 1
        assert "id" in df.columns
        assert "title" in df.columns


class TestExportAll:
    def test_writes_parquet_files(self, tmp_path, mock_neo4j_driver):
        from app.services.graphrag_bridge import export_all

        driver, session = mock_neo4j_driver

        # Mock entity query
        session.run.side_effect = [
            _mock_neo4j_records([
                {"name": "A", "entity_type": "PLATFORM", "description": "test", "id": "a1"},
            ]),
            _mock_neo4j_records([
                {"source": "A", "target": "B", "relationship": "CONTAINS", "description": ""},
            ]),
        ]

        db = MagicMock()
        # text_units query
        db.execute.return_value.fetchall.side_effect = [
            [("c1", "text", "d1", ["A"])],  # text_units
            [("d1", "Doc 1", "full text")],  # documents
        ]

        export_all(driver, db, tmp_path)

        assert (tmp_path / "entities.parquet").exists()
        assert (tmp_path / "relationships.parquet").exists()
        assert (tmp_path / "text_units.parquet").exists()
        assert (tmp_path / "documents.parquet").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_graphrag_bridge.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the bridge layer**

```python
# app/services/graphrag_bridge.py
"""Neo4j/Postgres → GraphRAG Parquet bridge layer.

Exports entities, relationships, text units, and documents from the
ontology-populated Neo4j graph and Postgres chunk store into pandas
DataFrames matching Microsoft GraphRAG's expected Parquet schema.
"""

import logging
import uuid
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from app.services.ontology_templates import load_ontology

logger = logging.getLogger(__name__)

# Load scoring weights from ontology
_ontology = load_ontology()
_scoring_weights: dict[str, float] = _ontology.get("scoring_weights", {})
_DEFAULT_WEIGHT = _scoring_weights.get("default", 0.70)

# GraphRAG expected column schemas
_ENTITY_COLUMNS = ["id", "title", "type", "description", "human_readable_id"]
_RELATIONSHIP_COLUMNS = ["id", "source", "target", "description", "weight",
                          "human_readable_id"]
_TEXT_UNIT_COLUMNS = ["id", "text", "n_tokens", "document_ids", "entity_ids",
                       "relationship_ids"]
_DOCUMENT_COLUMNS = ["id", "title", "raw_content", "text_unit_ids"]


def export_entities(neo4j_driver) -> pd.DataFrame:
    """Export all Entity nodes from Neo4j into a GraphRAG entities DataFrame."""
    try:
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (n:Entity) "
                "RETURN n.name AS name, n.entity_type AS entity_type, "
                "n.description AS description, n.id AS id"
            )
            rows = []
            for i, record in enumerate(result):
                data = record.data()
                rows.append({
                    "id": data.get("id") or str(uuid.uuid4()),
                    "title": data.get("name", ""),
                    "type": data.get("entity_type", "UNKNOWN"),
                    "description": data.get("description", ""),
                    "human_readable_id": i,
                })
    except Exception:
        logger.exception("Failed to export entities from Neo4j")
        rows = []

    return pd.DataFrame(rows, columns=_ENTITY_COLUMNS)


def export_relationships(neo4j_driver) -> pd.DataFrame:
    """Export all relationships from Neo4j into a GraphRAG relationships DataFrame."""
    try:
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (a:Entity)-[r]->(b:Entity) "
                "RETURN a.name AS source, b.name AS target, "
                "type(r) AS relationship, r.description AS description"
            )
            rows = []
            for i, record in enumerate(result):
                data = record.data()
                rel_type = data.get("relationship", "")
                weight = _scoring_weights.get(rel_type, _DEFAULT_WEIGHT)
                rows.append({
                    "id": str(uuid.uuid4()),
                    "source": data.get("source", ""),
                    "target": data.get("target", ""),
                    "description": data.get("description") or rel_type,
                    "weight": weight,
                    "human_readable_id": i,
                })
    except Exception:
        logger.exception("Failed to export relationships from Neo4j")
        rows = []

    return pd.DataFrame(rows, columns=_RELATIONSHIP_COLUMNS)


def export_text_units(db_session) -> pd.DataFrame:
    """Export text chunks with entity linkages from Postgres + Neo4j provenance."""
    try:
        result = db_session.execute(text("""
            SELECT
                tc.chunk_id::text,
                tc.content_text,
                tc.document_id::text,
                COALESCE(tc.entity_names, ARRAY[]::text[])
            FROM retrieval.text_chunks tc
            WHERE tc.content_text IS NOT NULL
        """))
        rows = []
        for i, row in enumerate(result.fetchall()):
            chunk_id, content, doc_id, entity_names = row
            rows.append({
                "id": chunk_id,
                "text": content or "",
                "n_tokens": len((content or "").split()),
                "document_ids": [doc_id] if doc_id else [],
                "entity_ids": list(entity_names) if entity_names else [],
                "relationship_ids": [],
            })
    except Exception:
        logger.exception("Failed to export text units from Postgres")
        rows = []

    return pd.DataFrame(rows, columns=_TEXT_UNIT_COLUMNS)


def export_documents(db_session) -> pd.DataFrame:
    """Export document metadata from Postgres."""
    try:
        result = db_session.execute(text("""
            SELECT
                d.id::text,
                COALESCE(d.title, sf.original_filename, 'untitled'),
                ''
            FROM ingest.documents d
            LEFT JOIN ingest.source_files sf ON sf.id = d.source_file_id
        """))
        rows = []
        for row in result.fetchall():
            doc_id, title, raw_content = row
            rows.append({
                "id": doc_id,
                "title": title,
                "raw_content": raw_content,
                "text_unit_ids": [],
            })
    except Exception:
        logger.exception("Failed to export documents from Postgres")
        rows = []

    return pd.DataFrame(rows, columns=_DOCUMENT_COLUMNS)


def export_all(neo4j_driver, db_session, output_dir: Path) -> dict:
    """Export all data and write Parquet files to output_dir.

    Returns dict with counts of exported entities, relationships, etc.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    entities_df = export_entities(neo4j_driver)
    relationships_df = export_relationships(neo4j_driver)
    text_units_df = export_text_units(db_session)
    documents_df = export_documents(db_session)

    # Link text_unit_ids back to documents
    if not text_units_df.empty and not documents_df.empty:
        doc_chunks = text_units_df.explode("document_ids")
        for doc_id in documents_df["id"]:
            chunk_ids = doc_chunks[doc_chunks["document_ids"] == doc_id]["id"].tolist()
            documents_df.loc[documents_df["id"] == doc_id, "text_unit_ids"] = [chunk_ids]

    entities_df.to_parquet(output_dir / "entities.parquet", index=False)
    relationships_df.to_parquet(output_dir / "relationships.parquet", index=False)
    text_units_df.to_parquet(output_dir / "text_units.parquet", index=False)
    documents_df.to_parquet(output_dir / "documents.parquet", index=False)

    stats = {
        "entities": len(entities_df),
        "relationships": len(relationships_df),
        "text_units": len(text_units_df),
        "documents": len(documents_df),
    }
    logger.info("GraphRAG bridge export complete: %s", stats)
    return stats
```

**NOTE:** The exact SQL queries for `export_text_units` and `export_documents` may need adjustment based on the actual Postgres schema columns. The implementer should verify column names by reading `alembic/versions/0001_initial_schemas.py` and `0004_ingest_v2_and_chunk_links.py`. The `entity_names` column on `text_chunks` may not exist — if so, use a subquery joining through Neo4j's `EXTRACTED_FROM` edges or add a separate Neo4j query to resolve chunk→entity mappings.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_graphrag_bridge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/graphrag_bridge.py tests/unit/test_graphrag_bridge.py
git commit -m "feat(graphrag): add Neo4j/Postgres to GraphRAG Parquet bridge layer"
```

---

## Chunk 2: GraphRAG Service Rewrite + Celery Tasks

### Task 4: Rewrite GraphRAG Service — Indexer

**Files:**
- Modify: `app/services/graphrag_service.py` (full rewrite)
- Test: `tests/unit/test_graphrag_service.py` (full rewrite)

- [ ] **Step 1: Write failing test for GraphRAG indexer**

Replace `tests/unit/test_graphrag_service.py` entirely:

```python
# tests/unit/test_graphrag_service.py
"""Unit tests for Microsoft GraphRAG service (indexer + searcher)."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


def _mock_settings(**overrides):
    s = MagicMock()
    s.graphrag_indexing_enabled = True
    s.graphrag_llm_provider = "ollama"
    s.graphrag_llm_model = "llama3.2"
    s.graphrag_llm_api_base = "http://ollama:11434/v1"
    s.graphrag_api_key = ""
    s.graphrag_embedding_model = "nomic-embed-text"
    s.graphrag_data_dir = "/tmp/test_graphrag"
    s.graphrag_community_level = 2
    s.graphrag_max_cluster_size = 10
    s.graphrag_tune_interval_minutes = 1440
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


# ---------------------------------------------------------------------------
# run_graphrag_indexing
# ---------------------------------------------------------------------------

class TestRunGraphragIndexing:
    @patch("app.services.graphrag_service.get_settings")
    def test_disabled_returns_zeros(self, mock_gs):
        from app.services.graphrag_service import run_graphrag_indexing

        mock_gs.return_value = _mock_settings(graphrag_indexing_enabled=False)
        result = run_graphrag_indexing(MagicMock(), MagicMock())
        assert result == {"communities_created": 0, "reports_generated": 0}

    @patch("app.services.graphrag_service.export_all", return_value={"entities": 0})
    @patch("app.services.graphrag_service.get_settings")
    def test_empty_graph_returns_zeros(self, mock_gs, mock_export):
        from app.services.graphrag_service import run_graphrag_indexing

        mock_gs.return_value = _mock_settings()
        result = run_graphrag_indexing(MagicMock(), MagicMock())
        assert result["communities_created"] == 0

    @patch("app.services.graphrag_service._run_graphrag_pipeline")
    @patch("app.services.graphrag_service.export_all")
    @patch("app.services.graphrag_service.get_settings")
    def test_full_pipeline_returns_stats(self, mock_gs, mock_export, mock_pipeline):
        from app.services.graphrag_service import run_graphrag_indexing

        mock_gs.return_value = _mock_settings()
        mock_export.return_value = {"entities": 5, "relationships": 3}
        mock_pipeline.return_value = {"communities_created": 2, "reports_generated": 2}

        result = run_graphrag_indexing(MagicMock(), MagicMock())
        assert result["communities_created"] == 2
        assert result["reports_generated"] == 2

    @patch("app.services.graphrag_service.export_all", side_effect=Exception("fail"))
    @patch("app.services.graphrag_service.get_settings")
    def test_exception_returns_zeros(self, mock_gs, mock_export):
        from app.services.graphrag_service import run_graphrag_indexing

        mock_gs.return_value = _mock_settings()
        result = run_graphrag_indexing(MagicMock(), MagicMock())
        assert result["communities_created"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_graphrag_service.py::TestRunGraphragIndexing -v`
Expected: FAIL — new function signatures don't match

- [ ] **Step 3: Implement the indexer portion of graphrag_service.py**

Rewrite `app/services/graphrag_service.py`:

```python
# app/services/graphrag_service.py
"""Microsoft GraphRAG integration — indexing, search, and prompt tuning.

Replaces the custom community detection/report implementation with
Microsoft's GraphRAG library. Uses the Neo4j ontology graph built by
Docling-Graph as input via the bridge layer.
"""

import asyncio
import logging
from pathlib import Path

import pandas as pd

from app.config import get_settings
from app.services.graphrag_bridge import export_all
from app.services.graphrag_config import build_graphrag_config
from app.services.graphrag_prompts import write_prompt_files

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def run_graphrag_indexing(neo4j_driver, db_session) -> dict:
    """Run the full GraphRAG indexing pipeline.

    1. Export Neo4j entities/relationships + Postgres text units to Parquet
    2. Run GraphRAG community detection + report generation + embeddings
    3. Returns stats dict with communities_created and reports_generated
    """
    settings = get_settings()
    if not settings.graphrag_indexing_enabled:
        return {"communities_created": 0, "reports_generated": 0}

    try:
        data_dir = Path(settings.graphrag_data_dir)
        output_dir = data_dir / "output"

        # Step 1: Bridge — export ontology graph to Parquet
        stats = export_all(neo4j_driver, db_session, output_dir)
        if stats.get("entities", 0) == 0:
            logger.info("No entities in graph — skipping GraphRAG indexing")
            return {"communities_created": 0, "reports_generated": 0}

        # Ensure prompts are written
        prompts_dir = data_dir / "prompts"
        write_prompt_files(prompts_dir)

        # Step 2: Run GraphRAG pipeline (community detection + reports + embeddings)
        result = _run_graphrag_pipeline(settings, data_dir, output_dir)
        return result

    except Exception:
        logger.exception("GraphRAG indexing failed")
        return {"communities_created": 0, "reports_generated": 0}


def _run_graphrag_pipeline(settings, data_dir: Path, output_dir: Path) -> dict:
    """Run Microsoft GraphRAG's indexing pipeline on pre-exported Parquet data.

    Uses the build_index API with pre-populated input documents.
    Since our entities/relationships are already extracted (by Docling-Graph),
    we provide them as pre-built Parquet and run only the community detection,
    report generation, and embedding workflows.
    """
    from graphrag.api import build_index
    from graphrag.config.enums import IndexingMethod

    config = build_graphrag_config(settings)

    # Load pre-exported text as input documents for GraphRAG
    text_units_path = output_dir / "text_units.parquet"
    if text_units_path.exists():
        text_df = pd.read_parquet(text_units_path)
        input_docs = pd.DataFrame({
            "id": text_df["id"],
            "text": text_df["text"],
            "title": text_df["id"],
        })
    else:
        input_docs = None

    # Run indexing
    results = asyncio.run(build_index(
        config=config,
        method=IndexingMethod.Standard,
        is_update_run=False,
        memory_profile=False,
        verbose=True,
        input_documents=input_docs,
    ))

    communities_created = 0
    reports_generated = 0

    for result in results:
        if result.errors:
            logger.warning("GraphRAG workflow %s errors: %s",
                           result.workflow, result.errors)
        else:
            logger.info("GraphRAG workflow %s completed", result.workflow)

    # Count outputs
    communities_path = output_dir / "communities.parquet"
    reports_path = output_dir / "community_reports.parquet"
    if communities_path.exists():
        communities_created = len(pd.read_parquet(communities_path))
    if reports_path.exists():
        reports_generated = len(pd.read_parquet(reports_path))

    return {
        "communities_created": communities_created,
        "reports_generated": reports_generated,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_graphrag_service.py::TestRunGraphragIndexing -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/graphrag_service.py tests/unit/test_graphrag_service.py
git commit -m "feat(graphrag): rewrite indexer with Microsoft GraphRAG build_index API"
```

---

### Task 5: Rewrite GraphRAG Service — Searcher

**Files:**
- Modify: `app/services/graphrag_service.py` (append search functions)
- Test: `tests/unit/test_graphrag_service.py` (append search tests)

- [ ] **Step 1: Write failing tests for all four search methods**

Append to `tests/unit/test_graphrag_service.py`:

```python
# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------

class TestLocalSearch:
    @patch("app.services.graphrag_service._load_search_data")
    @patch("app.services.graphrag_service.get_settings")
    def test_returns_results(self, mock_gs, mock_load):
        from app.services.graphrag_service import local_search

        mock_gs.return_value = _mock_settings()
        mock_load.return_value = _make_mock_search_data()

        with patch("app.services.graphrag_service._run_local_search") as mock_run:
            mock_run.return_value = ("Answer text", {"entities": []})
            result = local_search("S-400 capabilities")

        assert result["response"] == "Answer text"

    @patch("app.services.graphrag_service._load_search_data", side_effect=Exception("no data"))
    @patch("app.services.graphrag_service.get_settings")
    def test_exception_returns_empty(self, mock_gs, mock_load):
        from app.services.graphrag_service import local_search

        mock_gs.return_value = _mock_settings()
        result = local_search("query")
        assert result["response"] == ""
        assert result["context"] == {}


class TestGlobalSearch:
    @patch("app.services.graphrag_service._load_search_data")
    @patch("app.services.graphrag_service.get_settings")
    def test_returns_results(self, mock_gs, mock_load):
        from app.services.graphrag_service import global_search

        mock_gs.return_value = _mock_settings()
        mock_load.return_value = _make_mock_search_data()

        with patch("app.services.graphrag_service._run_global_search") as mock_run:
            mock_run.return_value = ("Global answer", {"reports": []})
            result = global_search("broad question")

        assert result["response"] == "Global answer"


class TestDriftSearch:
    @patch("app.services.graphrag_service._load_search_data")
    @patch("app.services.graphrag_service.get_settings")
    def test_returns_results(self, mock_gs, mock_load):
        from app.services.graphrag_service import drift_search

        mock_gs.return_value = _mock_settings()
        mock_load.return_value = _make_mock_search_data()

        with patch("app.services.graphrag_service._run_drift_search") as mock_run:
            mock_run.return_value = ("DRIFT answer", {"entities": []})
            result = drift_search("guidance methods")

        assert result["response"] == "DRIFT answer"


class TestBasicSearch:
    @patch("app.services.graphrag_service._load_search_data")
    @patch("app.services.graphrag_service.get_settings")
    def test_returns_results(self, mock_gs, mock_load):
        from app.services.graphrag_service import basic_search

        mock_gs.return_value = _mock_settings()
        mock_load.return_value = _make_mock_search_data()

        with patch("app.services.graphrag_service._run_basic_search") as mock_run:
            mock_run.return_value = ("Basic answer", {"chunks": []})
            result = basic_search("simple query")

        assert result["response"] == "Basic answer"


def _make_mock_search_data():
    """Create minimal mock search data dict."""
    return {
        "entities": pd.DataFrame(columns=["id", "title", "type", "description"]),
        "communities": pd.DataFrame(columns=["id", "title", "level"]),
        "community_reports": pd.DataFrame(columns=["id", "community_id", "full_content"]),
        "text_units": pd.DataFrame(columns=["id", "text"]),
        "relationships": pd.DataFrame(columns=["id", "source", "target"]),
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_graphrag_service.py::TestLocalSearch -v`
Expected: FAIL — function not found

- [ ] **Step 3: Implement search functions**

Append to `app/services/graphrag_service.py`:

```python
# ---------------------------------------------------------------------------
# Search — all four methods
# ---------------------------------------------------------------------------

def _load_search_data(settings) -> dict:
    """Load indexed Parquet data for search queries."""
    output_dir = Path(settings.graphrag_data_dir) / "output"

    data = {}
    for name in ("entities", "communities", "community_reports",
                 "text_units", "relationships"):
        path = output_dir / f"{name}.parquet"
        if path.exists():
            data[name] = pd.read_parquet(path)
        else:
            data[name] = pd.DataFrame()

    return data


def _run_local_search(config, data: dict, query: str, community_level: int):
    """Run GraphRAG local search."""
    from graphrag.api import local_search as graphrag_local

    return asyncio.run(graphrag_local(
        config=config,
        entities=data["entities"],
        communities=data["communities"],
        community_reports=data["community_reports"],
        text_units=data["text_units"],
        relationships=data["relationships"],
        community_level=community_level,
        response_type="Detailed explanation",
        query=query,
    ))


def _run_global_search(config, data: dict, query: str, community_level: int):
    """Run GraphRAG global search."""
    from graphrag.api import global_search as graphrag_global

    return asyncio.run(graphrag_global(
        config=config,
        entities=data["entities"],
        communities=data["communities"],
        community_reports=data["community_reports"],
        community_level=community_level,
        dynamic_community_selection=False,
        response_type="Multiple Paragraphs",
        query=query,
    ))


def _run_drift_search(config, data: dict, query: str, community_level: int):
    """Run GraphRAG DRIFT search."""
    from graphrag.api import drift_search as graphrag_drift

    return asyncio.run(graphrag_drift(
        config=config,
        entities=data["entities"],
        communities=data["communities"],
        community_reports=data["community_reports"],
        text_units=data["text_units"],
        relationships=data["relationships"],
        community_level=community_level,
        response_type="In-depth analysis",
        query=query,
    ))


def _run_basic_search(config, data: dict, query: str):
    """Run GraphRAG basic search."""
    from graphrag.api import basic_search as graphrag_basic

    return asyncio.run(graphrag_basic(
        config=config,
        text_units=data["text_units"],
        query=query,
    ))


def local_search(query: str) -> dict:
    """Entity-centric search with community context."""
    try:
        settings = get_settings()
        config = build_graphrag_config(settings)
        data = _load_search_data(settings)
        response, context = _run_local_search(
            config, data, query, settings.graphrag_community_level
        )
        return {"response": response, "context": context}
    except Exception:
        logger.exception("GraphRAG local search failed")
        return {"response": "", "context": {}}


def global_search(query: str) -> dict:
    """Cross-community summarization for broad questions."""
    try:
        settings = get_settings()
        config = build_graphrag_config(settings)
        data = _load_search_data(settings)
        response, context = _run_global_search(
            config, data, query, settings.graphrag_community_level
        )
        return {"response": response, "context": context}
    except Exception:
        logger.exception("GraphRAG global search failed")
        return {"response": "", "context": {}}


def drift_search(query: str) -> dict:
    """Community-informed expansion search."""
    try:
        settings = get_settings()
        config = build_graphrag_config(settings)
        data = _load_search_data(settings)
        response, context = _run_drift_search(
            config, data, query, settings.graphrag_community_level
        )
        return {"response": response, "context": context}
    except Exception:
        logger.exception("GraphRAG DRIFT search failed")
        return {"response": "", "context": {}}


def basic_search(query: str) -> dict:
    """Vector search over text units."""
    try:
        settings = get_settings()
        config = build_graphrag_config(settings)
        data = _load_search_data(settings)
        response, context = _run_basic_search(config, data, query)
        return {"response": response, "context": context}
    except Exception:
        logger.exception("GraphRAG basic search failed")
        return {"response": "", "context": {}}


# ---------------------------------------------------------------------------
# Prompt tuning
# ---------------------------------------------------------------------------

def run_auto_tune() -> dict:
    """Run GraphRAG auto prompt tuning against the current corpus."""
    try:
        settings = get_settings()
        data_dir = Path(settings.graphrag_data_dir)
        config = build_graphrag_config(settings)

        from graphrag.api import prompt_tune

        results = asyncio.run(prompt_tune(config=config))

        # Write tuned prompts to prompts dir
        prompts_dir = data_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        for name, content in results.items():
            (prompts_dir / f"{name}.txt").write_text(content)

        logger.info("GraphRAG auto-tuning complete: %d prompts updated", len(results))
        return {"prompts_updated": len(results)}
    except Exception:
        logger.exception("GraphRAG auto-tuning failed")
        return {"prompts_updated": 0, "error": True}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_graphrag_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/graphrag_service.py tests/unit/test_graphrag_service.py
git commit -m "feat(graphrag): add Local/Global/DRIFT/Basic search + auto-tuning"
```

---

### Task 6: Rewrite Celery Tasks

**Files:**
- Modify: `app/workers/graphrag_tasks.py` (rewrite)
- Modify: `app/workers/celery_app.py` (add auto-tune schedule)

- [ ] **Step 1: Rewrite graphrag_tasks.py**

```python
# app/workers/graphrag_tasks.py
"""GraphRAG Celery tasks — indexing and prompt auto-tuning.

Indexing: scheduled hourly (configurable), exports Neo4j graph to Parquet,
runs Microsoft GraphRAG community detection + report generation.

Auto-tuning: scheduled daily (configurable), refines prompts based on corpus.

Both tasks use Redis locks to prevent overlapping runs.
"""

import logging

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, soft_time_limit=3600, time_limit=3660)
def run_graphrag_indexing_task(self) -> dict:
    """Run GraphRAG indexing pipeline as a Celery task."""
    from app.config import get_settings
    from app.db.session import get_neo4j_driver, get_sync_session
    from app.services.graphrag_service import run_graphrag_indexing

    settings = get_settings()
    if not settings.graphrag_indexing_enabled:
        return {"skipped": True}

    import redis

    r = redis.from_url(settings.celery_broker_url)
    lock = r.lock("graphrag:indexing:lock", timeout=3600, blocking=False)

    if not lock.acquire(blocking=False):
        logger.info("GraphRAG indexing already in progress — skipping")
        return {"skipped": True, "reason": "locked"}

    try:
        neo4j_driver = get_neo4j_driver()
        db = get_sync_session()
        try:
            stats = run_graphrag_indexing(neo4j_driver, db)
            logger.info("GraphRAG indexing complete: %s", stats)
            return stats
        finally:
            db.close()
    finally:
        try:
            lock.release()
        except Exception:
            pass


@celery_app.task(bind=True, soft_time_limit=3600, time_limit=3660)
def run_graphrag_auto_tune_task(self) -> dict:
    """Run GraphRAG prompt auto-tuning as a Celery task."""
    from app.config import get_settings
    from app.services.graphrag_service import run_auto_tune

    settings = get_settings()
    if not settings.graphrag_indexing_enabled:
        return {"skipped": True}

    import redis

    r = redis.from_url(settings.celery_broker_url)
    lock = r.lock("graphrag:tuning:lock", timeout=3600, blocking=False)

    if not lock.acquire(blocking=False):
        logger.info("GraphRAG auto-tuning already in progress — skipping")
        return {"skipped": True, "reason": "locked"}

    try:
        result = run_auto_tune()
        logger.info("GraphRAG auto-tuning complete: %s", result)
        return result
    finally:
        try:
            lock.release()
        except Exception:
            pass
```

- [ ] **Step 2: Update celery_app.py beat schedule**

In `app/workers/celery_app.py`, update the `beat_schedule` to add the auto-tuning task:

```python
    beat_schedule={
        "scan-watch-directories": {
            "task": "app.workers.watcher.scan_watch_directories",
            "schedule": settings.watch_dir_poll_interval_seconds,
        },
        **(
            {
                "graphrag-indexing": {
                    "task": "app.workers.graphrag_tasks.run_graphrag_indexing_task",
                    "schedule": settings.graphrag_indexing_interval_minutes * 60,
                },
                "graphrag-auto-tune": {
                    "task": "app.workers.graphrag_tasks.run_graphrag_auto_tune_task",
                    "schedule": settings.graphrag_tune_interval_minutes * 60,
                },
            }
            if settings.graphrag_indexing_enabled
            else {}
        ),
    },
```

Also add the auto-tune task to the `task_routes`:

```python
        "app.workers.graphrag_tasks.run_graphrag_auto_tune_task": {"queue": "graph"},
```

- [ ] **Step 3: Run existing tests to verify nothing broke**

Run: `pytest tests/ -v -k "not integration and not e2e"`
Expected: PASS (or only pre-existing failures)

- [ ] **Step 4: Commit**

```bash
git add app/workers/graphrag_tasks.py app/workers/celery_app.py
git commit -m "feat(graphrag): add indexing + auto-tuning Celery tasks with Redis locks"
```

---

## Chunk 3: API Integration, Schema Updates, and Deployment

### Task 7: Add New Query Strategies to Schema

**Files:**
- Modify: `app/schemas/retrieval.py:12-16`

- [ ] **Step 1: Add graphrag_drift and graphrag_basic to QueryStrategy enum**

```python
class QueryStrategy(str, Enum):
    basic = "basic"
    hybrid = "hybrid"
    graphrag_local = "graphrag_local"
    graphrag_global = "graphrag_global"
    graphrag_drift = "graphrag_drift"
    graphrag_basic = "graphrag_basic"
```

Also add to `_MODE_MAP`:

```python
    "graphrag_drift": (QueryStrategy.graphrag_drift, ModalityFilter.all),
    "graphrag_basic": (QueryStrategy.graphrag_basic, ModalityFilter.all),
```

- [ ] **Step 2: Run existing retrieval schema tests**

Run: `pytest tests/unit/test_retrieval_pipeline.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add app/schemas/retrieval.py
git commit -m "feat(graphrag): add graphrag_drift and graphrag_basic query strategies"
```

---

### Task 8: Wire New Search Methods into Retrieval Endpoint

**Files:**
- Modify: `app/api/v1/retrieval.py`

- [ ] **Step 1: Rewrite GraphRAG query functions in retrieval.py**

Replace the existing `_graphrag_local_query` and `_graphrag_global_query` functions and add `_graphrag_drift_query` and `_graphrag_basic_query`. All four follow the same pattern:

1. Call the corresponding `graphrag_service` function in an executor (sync → async bridge)
2. Convert the response to `QueryResultItem` format
3. Return results

Key changes to `_graphrag_local_query`:
- Remove: `from app.services.graphrag_service import local_search` (old sync signature)
- Add: Import new `local_search` that returns `{"response": str, "context": dict}`
- The result is now a single `QueryResultItem` with the LLM-generated response as `content_text` and the context dict

Key changes to `_graphrag_global_query`:
- Remove: SQL-based report search
- Add: Import new `global_search` that returns `{"response": str, "context": dict}`

New functions `_graphrag_drift_query` and `_graphrag_basic_query` follow same pattern.

Update the `unified_query` router to route the new strategies:

```python
        elif body.strategy == QueryStrategy.graphrag_drift:
            results = await _graphrag_drift_query(db, body)
        elif body.strategy == QueryStrategy.graphrag_basic:
            results = await _graphrag_basic_query(db, body)
```

Add the auto-tune manual trigger endpoint:

```python
@router.post("/graphrag/tune")
async def trigger_graphrag_tuning():
    """Dispatch GraphRAG prompt auto-tuning as a Celery task."""
    from app.workers.graphrag_tasks import run_graphrag_auto_tune_task

    task = run_graphrag_auto_tune_task.delay()
    return {"status": "tuning_started", "task_id": str(task.id)}
```

- [ ] **Step 2: Run retrieval tests**

Run: `pytest tests/unit/test_retrieval_pipeline.py tests/integration/test_retrieval_api.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add app/api/v1/retrieval.py
git commit -m "feat(graphrag): wire Local/Global/DRIFT/Basic search + tune endpoint into API"
```

---

### Task 9: Database Migration — Drop Old Tables

**Files:**
- Create: `alembic/versions/0008_drop_custom_graphrag_tables.py`

- [ ] **Step 1: Create migration**

```python
# alembic/versions/0008_drop_custom_graphrag_tables.py
"""Drop custom GraphRAG Postgres tables (replaced by Microsoft GraphRAG Parquet/LanceDB).

Revision ID: 0008
Revises: 0007
Create Date: 2026-03-17
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("graphrag_community_reports", schema="retrieval")
    op.drop_table("graphrag_communities", schema="retrieval")


def downgrade() -> None:
    # Recreate tables for rollback
    op.create_table(
        "graphrag_communities",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("community_id", sa.Text(), nullable=False, unique=True),
        sa.Column("level", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("entity_ids", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        schema="retrieval",
    )
    op.create_table(
        "graphrag_community_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "community_id",
            sa.Text(),
            sa.ForeignKey("retrieval.graphrag_communities.community_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("report_text", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("rank", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        schema="retrieval",
    )
    op.create_unique_constraint(
        "uq_graphrag_reports_community_id",
        "graphrag_community_reports",
        ["community_id"],
        schema="retrieval",
    )
    op.create_index(
        "ix_graphrag_reports_community_id",
        "graphrag_community_reports",
        ["community_id"],
        schema="retrieval",
    )
```

- [ ] **Step 2: Commit**

```bash
git add alembic/versions/0008_drop_custom_graphrag_tables.py
git commit -m "migrate: drop custom GraphRAG Postgres tables (replaced by Parquet/LanceDB)"
```

---

### Task 10: Docker & Dependency Updates

**Files:**
- Modify: `docker-compose.yml`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add graphrag_data volume to docker-compose.yml**

Add to the `volumes` section at bottom:

```yaml
  graphrag_data:
```

Add the volume mount to `api`, `worker`, `worker-graph` services:

```yaml
      - graphrag_data:/app/graphrag_data
```

Also add to the `beat` service so auto-tuning can write prompts:

```yaml
      - graphrag_data:/app/graphrag_data
```

- [ ] **Step 2: Update pyproject.toml dependencies**

Add `pyarrow` (required for Parquet I/O):

```
    "pyarrow>=14.0.0",
```

The `graphrag>=1.0.0` dependency is already present — it will now actually be used.

Remove `graspologic` if it was listed (it's not in the current pyproject.toml but verify).

- [ ] **Step 3: Add GraphRAG env vars to .env.example or document them**

Add to `.env` or document:

```bash
# Microsoft GraphRAG
GRAPHRAG_LLM_PROVIDER=ollama
GRAPHRAG_LLM_MODEL=llama3.2
GRAPHRAG_LLM_API_BASE=http://ollama:11434/v1
GRAPHRAG_API_KEY=
GRAPHRAG_EMBEDDING_MODEL=nomic-embed-text
GRAPHRAG_DATA_DIR=/app/graphrag_data
GRAPHRAG_COMMUNITY_LEVEL=2
GRAPHRAG_TUNE_INTERVAL_MINUTES=1440
```

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml pyproject.toml
git commit -m "feat(graphrag): add graphrag_data volume and pyarrow dependency"
```

---

### Task 11: Update Test Fixtures

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Remove old GraphRAG mock fixtures that reference Postgres tables**

The existing `mock_neo4j_driver` fixture stays as-is. No new GraphRAG-specific fixtures are needed at the conftest level — the service tests mock at the function level.

If any conftest fixtures reference `graphrag_communities` or `graphrag_community_reports` tables, remove those references.

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v -k "not integration and not e2e" --timeout=60`
Expected: All unit tests PASS

- [ ] **Step 3: Commit if changes were needed**

```bash
git add tests/conftest.py
git commit -m "test: update fixtures for Microsoft GraphRAG integration"
```

---

### Task 12: Integration Smoke Test

- [ ] **Step 1: Verify Docker build succeeds**

Run: `docker compose build api worker`
Expected: Build succeeds, graphrag package installed

- [ ] **Step 2: Verify migration runs**

Run: `docker compose run --rm api alembic upgrade head`
Expected: Migration 0008 applies, old tables dropped

- [ ] **Step 3: Verify API starts and new endpoints respond**

Run: `docker compose up -d api`
Then test:
```bash
curl -X POST http://localhost:8000/v1/retrieval/query \
  -H "Content-Type: application/json" \
  -d '{"query_text": "test", "strategy": "graphrag_drift"}'
```
Expected: 200 response (empty results if no data indexed) or appropriate 409 error

- [ ] **Step 4: Verify manual indexing trigger works**

```bash
curl -X POST http://localhost:8000/v1/graphrag/index
curl -X POST http://localhost:8000/v1/graphrag/tune
```
Expected: Both return `{"status": "...", "task_id": "..."}`

- [ ] **Step 5: Final commit**

```bash
git commit -m "feat(graphrag): Microsoft GraphRAG integration complete"
```

---

## Implementation Notes

### Critical Paths to Verify During Implementation

1. **Parquet schema compatibility**: GraphRAG's `build_index` expects specific column names and types. The bridge layer must match exactly. Read GraphRAG's source code for `read_indexer_entities`, `read_indexer_relationships`, etc. to verify column expectations.

2. **Async/sync bridge**: GraphRAG's API functions are async (`build_index`, `local_search`, etc.) but Celery tasks are sync. The service uses `asyncio.run()` to bridge. Verify there are no event loop conflicts with Celery's existing loop.

3. **LanceDB initialization**: DRIFT and Local search require LanceDB vector stores to be populated during indexing. Verify the `build_index` pipeline creates the LanceDB tables that search expects.

4. **Text unit → entity linkage**: The bridge's `export_text_units` must correctly link text chunks to entities. If the `text_chunks` table doesn't have entity names, implement a Neo4j query through `EXTRACTED_FROM` edges to resolve the mapping.

5. **Ollama OpenAI-compatible API**: GraphRAG expects an OpenAI-compatible endpoint. Ollama serves this at `/v1/chat/completions`. Verify the model can handle GraphRAG's structured prompts (JSON mode may be required — check if Ollama supports it for the configured model).

### Rollback Plan

If Microsoft GraphRAG integration fails:
1. Revert the `graphrag_service.py` rewrite (git revert)
2. Do NOT run migration 0008 (keep Postgres tables)
3. The old custom implementation continues working unchanged
