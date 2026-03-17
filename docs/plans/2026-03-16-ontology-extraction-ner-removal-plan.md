# Full Ontology Extraction + NER Removal Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Fix Docling-Graph to extract entities using all 45 ontology types (in 5 focused batches + 1 relationship pass) and remove the dead regex NER service.

**Architecture:** The docling-graph service builds 5 combined Pydantic templates (one per ontology layer) at startup. The pipeline task calls `/extract` 6 times per document — 5 entity passes with group-specific LLM prompts, then 1 relationship pass with all discovered entities as context. Confidence filtering and Neo4j import remain unchanged.

**Tech Stack:** FastAPI, Pydantic v2 `create_model()`, LiteLLM, Celery, Neo4j, PostgreSQL

**Design doc:** `docs/plans/2026-03-16-ontology-extraction-ner-removal-design.md`

---

### Task 0: Delete NER service and tests

**Files:**
- Delete: `app/services/ner.py`
- Delete: `tests/unit/test_ner.py`

**Step 1: Delete the files**

```bash
git rm app/services/ner.py tests/unit/test_ner.py
```

**Step 2: Verify no remaining imports**

```bash
grep -r "from app.services.ner" app/ tests/ --include="*.py"
grep -r "import ner" app/ tests/ --include="*.py"
```

Expected: No matches (only plan docs reference it).

**Step 3: Run tests to confirm nothing breaks**

```bash
./scripts/run_tests.sh
```

Expected: All tests pass (NER was not imported anywhere in production code).

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove dead regex NER service and tests"
```

---

### Task 1: Add GROUP_MAP and build combined templates in `templates.py`

**Files:**
- Modify: `docker/docling-graph/app/templates.py`
- Create: `docker/docling-graph/tests/test_templates.py`

**Step 1: Write the failing test**

Create `docker/docling-graph/tests/test_templates.py`:

```python
"""Tests for grouped template building."""

import pytest
from app.templates import build_templates, GROUP_MAP, load_ontology

ONTOLOGY_PATH = "/ontology/ontology.yaml"


class TestGroupMap:
    def test_group_map_has_five_groups(self):
        assert len(GROUP_MAP) == 5
        assert set(GROUP_MAP.keys()) == {"reference", "equipment", "rf_signal", "weapon", "operational"}

    def test_all_ontology_types_covered(self):
        """Every entity type in the ontology must appear in exactly one group."""
        ontology = load_ontology(ONTOLOGY_PATH)
        ontology_names = {et["name"] for et in ontology.get("entity_types", [])}
        grouped_names = set()
        for names in GROUP_MAP.values():
            for name in names:
                assert name not in grouped_names, f"{name} appears in multiple groups"
                grouped_names.add(name)
        assert grouped_names == ontology_names, f"Missing: {ontology_names - grouped_names}, Extra: {grouped_names - ontology_names}"


class TestBuildTemplates:
    def test_returns_dict_keyed_by_group(self):
        ontology = load_ontology(ONTOLOGY_PATH)
        templates = build_templates(ontology)
        assert isinstance(templates, dict)
        assert set(templates.keys()) == set(GROUP_MAP.keys())

    def test_each_template_is_pydantic_model(self):
        from pydantic import BaseModel
        ontology = load_ontology(ONTOLOGY_PATH)
        templates = build_templates(ontology)
        for group_name, model_cls in templates.items():
            assert issubclass(model_cls, BaseModel), f"{group_name} is not a Pydantic model"

    def test_equipment_template_has_entity_list_fields(self):
        """The equipment combined model should have optional list fields for each entity type."""
        ontology = load_ontology(ONTOLOGY_PATH)
        templates = build_templates(ontology)
        model_cls = templates["equipment"]
        fields = model_cls.model_fields
        # Should have a field for each entity type in the equipment group
        for entity_name in GROUP_MAP["equipment"]:
            field_key = entity_name.lower()
            assert field_key in fields, f"Missing field {field_key} in equipment template"

    def test_combined_model_fields_are_optional_lists(self):
        """Each field in a combined model should be Optional[list[...]]."""
        ontology = load_ontology(ONTOLOGY_PATH)
        templates = build_templates(ontology)
        model_cls = templates["reference"]
        # Instantiate with no args — all fields should default to None
        instance = model_cls()
        for field_name in model_cls.model_fields:
            assert getattr(instance, field_name) is None

    def test_combined_model_has_graph_metadata(self):
        """Each combined model should have is_entity=True in model_config."""
        ontology = load_ontology(ONTOLOGY_PATH)
        templates = build_templates(ontology)
        for group_name, model_cls in templates.items():
            assert model_cls.model_config.get("is_entity") is True, f"{group_name} missing is_entity"
```

**Step 2: Run test to verify it fails**

```bash
cd docker/docling-graph && python -m pytest tests/test_templates.py -v 2>&1 | head -30
```

Expected: FAIL — `GROUP_MAP` not defined, `build_templates` returns wrong shape.

**Step 3: Implement GROUP_MAP and refactor `build_templates()`**

Modify `docker/docling-graph/app/templates.py`:

```python
"""YAML-to-Pydantic template generator for docling-graph.

Reads the unified ontology YAML and builds dynamic Pydantic model classes that
docling-graph uses to type entity extraction results.  Templates are grouped
by ontology layer for focused LLM extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)

# Mapping from ontology YAML type strings to Python types
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

# Hardcoded grouping of entity types by ontology layer.
# Every entity type in the ontology MUST appear in exactly one group.
GROUP_MAP: dict[str, list[str]] = {
    "reference": [
        "DOCUMENT", "SECTION", "FIGURE", "TABLE", "SPREADSHEET", "ASSERTION",
    ],
    "equipment": [
        "PLATFORM", "RADAR_SYSTEM", "MISSILE_SYSTEM",
        "AIR_DEFENSE_ARTILLERY_SYSTEM", "ELECTRONIC_WARFARE_SYSTEM",
        "FIRE_CONTROL_SYSTEM", "INTEGRATED_AIR_DEFENSE_SYSTEM",
        "LAUNCHER_SYSTEM", "WEAPON_SYSTEM",
    ],
    "rf_signal": [
        "FREQUENCY_BAND", "RF_EMISSION", "WAVEFORM", "MODULATION",
        "RF_SIGNATURE", "SCAN_PATTERN", "ANTENNA", "TRANSMITTER",
        "RECEIVER", "IF_AMPLIFIER", "SIGNAL_PROCESSING_CHAIN", "SEEKER",
    ],
    "weapon": [
        "GUIDANCE_METHOD", "MISSILE_PERFORMANCE",
        "MISSILE_PHYSICAL_CHARACTERISTICS", "PROPULSION_STACK",
        "PROPULSION_STAGE", "SUBSYSTEM", "COMPONENT",
    ],
    "operational": [
        "CAPABILITY", "RADAR_PERFORMANCE", "ENGAGEMENT_TIMELINE",
        "FORCE_STRUCTURE", "EQUIPMENT_SYSTEM", "ASSEMBLY",
        "SPECIFICATION", "STANDARD", "PROCEDURE", "FAILURE_MODE",
        "TEST_EVENT",
    ],
}


def load_ontology(path: str | Path) -> dict[str, Any]:
    """Load and return the parsed ontology YAML."""
    with open(path) as fh:
        return yaml.safe_load(fh)


def get_ontology_version(ontology: dict[str, Any]) -> str:
    """Return the version string from the ontology dict."""
    return ontology.get("version", "unknown")


def _build_entity_model(name: str, props_schema: dict[str, Any]) -> type[BaseModel]:
    """Build a single Pydantic model for one entity type."""
    prop_fields = props_schema.get("properties", {})
    field_defs: dict[str, Any] = {}
    id_fields: list[str] = []

    for field_name, field_spec in prop_fields.items():
        py_type = _TYPE_MAP.get(field_spec.get("type", "string"), str)
        field_defs[field_name] = (py_type | None, None)

    if prop_fields:
        id_fields = [next(iter(prop_fields))]

    model = create_model(name, __config__=None, **field_defs)
    model.model_config["is_entity"] = True
    model.model_config["graph_id_fields"] = id_fields
    return model


def build_templates(ontology: dict[str, Any]) -> dict[str, type[BaseModel]]:
    """Generate combined Pydantic model classes grouped by ontology layer.

    Returns a dict mapping group name -> combined Pydantic model class.
    Each combined model has optional list fields for every entity type
    in that group (e.g., ``radar_system: list[RadarSystemModel] | None``).
    """
    # First, build individual entity models keyed by name
    entity_models: dict[str, type[BaseModel]] = {}
    entity_types = ontology.get("entity_types", [])
    for et in entity_types:
        name: str = et["name"]
        props_schema = et.get("properties", {})
        entity_models[name] = _build_entity_model(name, props_schema)

    # Build a reverse lookup: entity_name -> group_name
    entity_to_group: dict[str, str] = {}
    for group_name, type_names in GROUP_MAP.items():
        for type_name in type_names:
            entity_to_group[type_name] = group_name

    # Build combined models per group
    templates: dict[str, type[BaseModel]] = {}
    for group_name, type_names in GROUP_MAP.items():
        combined_fields: dict[str, Any] = {}
        for type_name in type_names:
            if type_name not in entity_models:
                logger.warning(
                    "Entity type %s in GROUP_MAP[%s] not found in ontology — skipping",
                    type_name, group_name,
                )
                continue
            field_key = type_name.lower()
            entity_model = entity_models[type_name]
            combined_fields[field_key] = (list[entity_model] | None, None)

        combined_model = create_model(
            f"{group_name.title().replace('_', '')}Entities",
            __config__=None,
            **combined_fields,
        )
        combined_model.model_config["is_entity"] = True
        combined_model.model_config["graph_id_fields"] = []
        templates[group_name] = combined_model

    logger.info(
        "Built %d group templates (%d entity types) from ontology v%s",
        len(templates),
        len(entity_models),
        get_ontology_version(ontology),
    )

    _register_edges(ontology)
    return templates


def _register_edges(ontology: dict[str, Any]) -> None:
    """Try to register relationship types with docling_graph.utils.edge."""
    relationship_types = ontology.get("relationship_types", [])
    if not relationship_types:
        return

    try:
        from docling_graph.utils.edge import register_edge_type  # type: ignore[import-untyped]
    except (ImportError, ModuleNotFoundError):
        logger.warning(
            "docling_graph.utils.edge not available — skipping %d edge declarations",
            len(relationship_types),
        )
        return

    registered = 0
    for rt in relationship_types:
        try:
            register_edge_type(
                name=rt["name"],
                from_types=rt.get("from_types", []),
                to_types=rt.get("to_types", []),
            )
            registered += 1
        except Exception:
            logger.warning("Failed to register edge type %s", rt.get("name"), exc_info=True)

    logger.info("Registered %d / %d edge types with docling-graph", registered, len(relationship_types))
```

**Step 4: Run tests to verify they pass**

```bash
cd docker/docling-graph && python -m pytest tests/test_templates.py -v
```

Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add docker/docling-graph/app/templates.py docker/docling-graph/tests/test_templates.py
git commit -m "feat(docling-graph): group ontology templates into 5 extraction batches"
```

---

### Task 2: Create group-specific LLM prompts in `prompts.py`

**Files:**
- Create: `docker/docling-graph/app/prompts.py`
- Create: `docker/docling-graph/tests/test_prompts.py`

**Step 1: Write the failing test**

Create `docker/docling-graph/tests/test_prompts.py`:

```python
"""Tests for group-specific LLM prompts."""

from app.prompts import get_entity_prompt, get_relationship_prompt, GROUP_PROMPTS
from app.templates import GROUP_MAP


class TestGroupPrompts:
    def test_every_group_has_prompt(self):
        for group_name in GROUP_MAP:
            assert group_name in GROUP_PROMPTS, f"Missing prompt for group {group_name}"

    def test_entity_prompt_returns_string(self):
        prompt = get_entity_prompt("equipment")
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_entity_prompt_unknown_group_raises(self):
        import pytest
        with pytest.raises(KeyError):
            get_entity_prompt("nonexistent_group")

    def test_relationship_prompt_includes_entity_context(self):
        entities = [
            {"name": "Patriot PAC-3", "entity_type": "MISSILE_SYSTEM"},
            {"name": "AN/MPQ-53", "entity_type": "RADAR_SYSTEM"},
        ]
        prompt = get_relationship_prompt(entities)
        assert "Patriot PAC-3" in prompt
        assert "AN/MPQ-53" in prompt
        assert "MISSILE_SYSTEM" in prompt

    def test_relationship_prompt_empty_entities(self):
        prompt = get_relationship_prompt([])
        assert isinstance(prompt, str)
        assert len(prompt) > 50
```

**Step 2: Run test to verify it fails**

```bash
cd docker/docling-graph && python -m pytest tests/test_prompts.py -v
```

Expected: FAIL — `app.prompts` module not found.

**Step 3: Implement `prompts.py`**

Create `docker/docling-graph/app/prompts.py`:

```python
"""Group-specific LLM system prompts for ontology-driven extraction.

Each extraction group gets a tailored prompt that provides domain context
to the LLM, improving extraction quality for 8B-parameter models by
narrowing the entity search space.
"""

from __future__ import annotations

GROUP_PROMPTS: dict[str, str] = {
    "reference": (
        "You are a military document analyst. Extract document structure elements "
        "from the following text: sections, figures, tables, spreadsheets, and assertions. "
        "For each element, capture its title, document number, classification marking, "
        "publication date, and any other identifying metadata. "
        "Return only elements explicitly mentioned in the text."
    ),
    "equipment": (
        "You are a military equipment analyst specializing in weapons systems identification. "
        "Extract military equipment systems from the following text: platforms (vehicles, "
        "aircraft, vessels), radar systems, missile systems, air defense artillery systems, "
        "electronic warfare systems, fire control systems, integrated air defense systems (IADS), "
        "launcher systems, and weapon systems. "
        "Look for system designations (AN/XXX-YY format), NATO reporting names, nomenclature, "
        "manufacturer details, and operational characteristics. "
        "Return only systems explicitly mentioned in the text."
    ),
    "rf_signal": (
        "You are an RF/electromagnetic signal analyst specializing in radar and electronic warfare. "
        "Extract electromagnetic and RF signal characteristics from the following text: "
        "frequency bands (S-band, X-band, C-band, L-band, etc.), RF emissions with specific "
        "frequency values, waveforms (LFM, NLFM, Barker, Frank code, Costas, frequency hopping, chirp), "
        "modulation types (pulse, Doppler, FMCW, pulse-Doppler), RF signatures, "
        "scan patterns (search, track, TWS, STT, pulse doppler, MTI, ECCM modes), "
        "antennas (type, gain, beamwidth), transmitters (power, ERP), receivers (sensitivity, bandwidth), "
        "IF amplifiers, signal processing chains, and seekers (terminal guidance). "
        "Pay close attention to specific numeric values: frequencies in GHz/MHz, "
        "PRF/PRI values, pulse durations, ERP in dBW/watts, antenna gain in dBi. "
        "Return only characteristics explicitly mentioned in the text."
    ),
    "weapon": (
        "You are a weapons systems analyst specializing in missile and munitions technology. "
        "Extract weapon and missile subsystem details from the following text: "
        "guidance methods (SARH, ARH, IIR, command guidance, GPS/INS, MMW, dual-mode), "
        "missile performance parameters (max/min range, altitude envelope, max speed, "
        "time-of-flight, single-shot probability of kill), "
        "missile physical characteristics (body diameter, length, wingspan, launch mass, warhead mass), "
        "propulsion details (ejector, booster, sustainer stages, fuel type, burn time), "
        "subsystems, and components with part numbers or NSN identifiers. "
        "Return only details explicitly mentioned in the text."
    ),
    "operational": (
        "You are a military operations analyst specializing in capability assessment. "
        "Extract operational and capability information from the following text: "
        "functional capabilities (detection, tracking, engagement, surveillance), "
        "radar performance metrics (detection range, velocity resolution, range resolution, "
        "ambiguity limits, clutter rejection), "
        "engagement timelines (detection-to-designate, designation-to-launch, "
        "time-to-intercept), force structure elements (units, echelons, battalions, batteries), "
        "equipment system top-level designations, assemblies, "
        "specifications with measurable parameters (max range, frequency, power, weight), "
        "standards (MIL-STD, MIL-DTL, MIL-PRF references), "
        "procedures (maintenance, operational, test), "
        "failure modes (FMECA severity, detection methods, MTBF), "
        "and test events (DT, OT, IOT results). "
        "Return only information explicitly mentioned in the text."
    ),
}


def get_entity_prompt(group_name: str) -> str:
    """Return the system prompt for entity extraction in the given group.

    Raises KeyError if group_name is not in GROUP_PROMPTS.
    """
    return GROUP_PROMPTS[group_name]


def get_relationship_prompt(entities_context: list[dict]) -> str:
    """Return the system prompt for relationship extraction.

    Args:
        entities_context: List of dicts with 'name' and 'entity_type' keys
            from prior entity extraction passes.
    """
    entity_lines = "\n".join(
        f"  - {e['name']} ({e['entity_type']})"
        for e in entities_context
    ) if entities_context else "  (no entities extracted)"

    return (
        "You are a military systems analyst specializing in relationships between "
        "equipment, capabilities, and organizational elements. "
        "Given the following entities extracted from a military technical document, "
        "identify relationships between them.\n\n"
        f"Known entities:\n{entity_lines}\n\n"
        "Focus on these relationship types:\n"
        "- Hierarchy: PART_OF, CONTAINS, HAS_SUBSYSTEM, HAS_COMPONENT, HAS_STAGE\n"
        "- Installation: INSTALLED_ON, DEPLOYED_ON (target is a PLATFORM)\n"
        "- Association: ASSOCIATED_WITH, OPERATED_BY, MANUFACTURED_BY\n"
        "- Functional: OPERATES_IN_BAND, USES_WAVEFORM, USES_MODULATION, EMITS, "
        "RADIATES, RECEIVES, PROCESSES\n"
        "- RF chain: HAS_ANTENNA, HAS_TRANSMITTER, HAS_RECEIVER, HAS_PROCESSING_CHAIN, "
        "HAS_SIGNATURE, HAS_SCAN, HAS_PERFORMANCE\n"
        "- Tactical: CUES, GUIDES, TRACKS, ENGAGES, DEFENDS, DETECTS, DESIGNATES\n"
        "- Standards: SPECIFIED_BY, COMPLIES_WITH (target is a STANDARD)\n"
        "- Signal chain: FEEDS_INTO, RECEIVES_FROM\n"
        "- Type: IS_A, INSTANCE_OF, ALIAS_OF\n\n"
        "Return only relationships supported by the text. "
        "Each relationship must connect two of the known entities listed above."
    )
```

**Step 4: Run tests to verify they pass**

```bash
cd docker/docling-graph && python -m pytest tests/test_prompts.py -v
```

Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add docker/docling-graph/app/prompts.py docker/docling-graph/tests/test_prompts.py
git commit -m "feat(docling-graph): add group-specific LLM prompts for extraction"
```

---

### Task 3: Update `schemas.py` with new request fields

**Files:**
- Modify: `docker/docling-graph/app/schemas.py`

**Step 1: Write the failing test**

Add to `docker/docling-graph/tests/test_prompts.py` (or a new test file — keep it simple, add inline):

No separate test needed — the schema changes are structural and will be validated by Task 4's integration test. Just update the file.

**Step 2: Update `schemas.py`**

Replace `ExtractionRequest` in `docker/docling-graph/app/schemas.py`:

```python
class ExtractionRequest(BaseModel):
    """Request body for the /extract endpoint."""

    document_id: str = Field(..., description="Internal document identifier")
    text: str = Field(..., description="Plain text to extract entities/relationships from")
    ontology_version: str | None = Field(
        None,
        description="Expected ontology version; logged as warning on mismatch",
    )
    template_group: str | None = Field(
        None,
        description="Ontology layer group to extract (reference, equipment, rf_signal, weapon, operational). "
        "If None, uses legacy single-template behavior.",
    )
    mode: str = Field(
        "entities",
        description="Extraction mode: 'entities' for entity extraction, 'relationships' for relationship-only pass.",
    )
    entities_context: list[dict] | None = Field(
        None,
        description="For mode='relationships': list of {name, entity_type} dicts from prior entity passes.",
    )
```

**Step 3: Verify schema loads**

```bash
cd docker/docling-graph && python -c "from app.schemas import ExtractionRequest; print(ExtractionRequest.model_json_schema())"
```

Expected: Prints JSON schema with new fields.

**Step 4: Commit**

```bash
git add docker/docling-graph/app/schemas.py
git commit -m "feat(docling-graph): add template_group, mode, entities_context to ExtractionRequest"
```

---

### Task 4: Update `main.py` to route by group/mode and inject prompts

**Files:**
- Modify: `docker/docling-graph/app/main.py`
- Create: `docker/docling-graph/tests/test_extraction.py`

**Step 1: Write the failing test**

Create `docker/docling-graph/tests/test_extraction.py`:

```python
"""Tests for the extraction endpoint with grouped templates."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_templates():
    """Patch startup to load templates."""
    from app.templates import load_ontology, build_templates
    ontology = load_ontology("/ontology/ontology.yaml")
    return build_templates(ontology)


@pytest.fixture
def client(mock_templates):
    """Create a test client with templates pre-loaded."""
    from app import main
    main._templates = mock_templates
    main._ontology_version = "3.0.0"
    return TestClient(main.app)


class TestExtractEndpointGrouped:
    def test_mock_mode_with_group(self, client):
        """Mock mode should return entities matching the requested group."""
        import app.main as main_module
        original_provider = main_module.LLM_PROVIDER
        main_module.LLM_PROVIDER = "mock"
        try:
            response = client.post("/extract", json={
                "document_id": "test-123",
                "text": "The AN/MPQ-53 radar operates in C-band.",
                "template_group": "equipment",
                "mode": "entities",
            })
            assert response.status_code == 200
            data = response.json()
            assert "entities" in data
            assert data["provider"] == "mock"
        finally:
            main_module.LLM_PROVIDER = original_provider

    def test_mock_mode_relationships(self, client):
        """Mock mode relationship pass should return relationships."""
        import app.main as main_module
        original_provider = main_module.LLM_PROVIDER
        main_module.LLM_PROVIDER = "mock"
        try:
            response = client.post("/extract", json={
                "document_id": "test-123",
                "text": "The AN/MPQ-53 radar is installed on the Patriot system.",
                "mode": "relationships",
                "entities_context": [
                    {"name": "AN/MPQ-53", "entity_type": "RADAR_SYSTEM"},
                    {"name": "Patriot", "entity_type": "MISSILE_SYSTEM"},
                ],
            })
            assert response.status_code == 200
            data = response.json()
            assert "relationships" in data
        finally:
            main_module.LLM_PROVIDER = original_provider

    def test_legacy_no_group(self, client):
        """No template_group should use legacy behavior (backward compat)."""
        import app.main as main_module
        original_provider = main_module.LLM_PROVIDER
        main_module.LLM_PROVIDER = "mock"
        try:
            response = client.post("/extract", json={
                "document_id": "test-123",
                "text": "Some text.",
            })
            assert response.status_code == 200
        finally:
            main_module.LLM_PROVIDER = original_provider

    def test_invalid_group_returns_422(self, client):
        """Unknown template_group should return 422."""
        response = client.post("/extract", json={
            "document_id": "test-123",
            "text": "Some text.",
            "template_group": "nonexistent",
            "mode": "entities",
        })
        assert response.status_code == 422
```

**Step 2: Run test to verify it fails**

```bash
cd docker/docling-graph && python -m pytest tests/test_extraction.py -v
```

Expected: FAIL — current `main.py` doesn't handle `template_group`.

**Step 3: Update `main.py`**

Full replacement of `docker/docling-graph/app/main.py`:

```python
"""Docling-Graph entity/relationship extraction service.

FastAPI wrapper around the docling-graph package.  Generates Pydantic
templates from a volume-mounted ontology YAML and routes LLM calls via
LiteLLM to Ollama or an OpenAI-compatible endpoint.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.prompts import get_entity_prompt, get_relationship_prompt
from app.schemas import (
    ExtractedEntityResponse,
    ExtractedRelationshipResponse,
    ExtractionRequest,
    ExtractionResponse,
)
from app.templates import GROUP_MAP, build_templates, get_ontology_version, load_ontology

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
ONTOLOGY_PATH = os.environ.get("ONTOLOGY_PATH", "/ontology/ontology.yaml")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
LLM_MODEL = os.environ.get("LLM_MODEL", "granite3-dense:8b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")

# ---------------------------------------------------------------------------
# Module-level state (populated at startup)
# ---------------------------------------------------------------------------
_templates: dict[str, type[BaseModel]] | None = None
_ontology_version: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ontology and build templates at startup."""
    global _templates, _ontology_version

    logger.info("Loading ontology from %s", ONTOLOGY_PATH)
    try:
        ontology = load_ontology(ONTOLOGY_PATH)
        _ontology_version = get_ontology_version(ontology)
        _templates = build_templates(ontology)
        logger.info(
            "Docling-Graph service ready — ontology v%s, %d group templates",
            _ontology_version,
            len(_templates),
        )
    except Exception:
        logger.exception("Failed to load ontology — service will report unhealthy")

    yield
    logger.info("Docling-Graph service shutting down.")


app = FastAPI(
    title="EIP-MMDPP Docling-Graph Service",
    description="Entity/relationship extraction via docling-graph + LiteLLM",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    if _templates is None:
        raise HTTPException(status_code=503, detail="Templates not loaded")
    return {
        "status": "ok",
        "ontology_version": _ontology_version,
        "template_count": len(_templates),
        "groups": list(_templates.keys()),
    }


# ---------------------------------------------------------------------------
# Extraction endpoint
# ---------------------------------------------------------------------------
def _build_litellm_model_string() -> str:
    """Construct the model string LiteLLM expects for the configured provider."""
    if LLM_PROVIDER == "ollama":
        return f"ollama/{LLM_MODEL}"
    if LLM_PROVIDER == "openai":
        return LLM_MODEL
    return f"{LLM_PROVIDER}/{LLM_MODEL}"


def _mock_extraction_response(
    mode: str = "entities",
    template_group: str | None = None,
) -> ExtractionResponse:
    """Return a canned response for testing (LLM_PROVIDER=mock)."""
    if mode == "relationships":
        return ExtractionResponse(
            entities=[],
            relationships=[
                ExtractedRelationshipResponse(
                    from_name="Mock Platform",
                    from_type="PLATFORM",
                    rel_type="HAS_COMPONENT",
                    to_name="Mock Radar System",
                    to_type="RADAR_SYSTEM",
                    confidence=0.85,
                ),
            ],
            ontology_version=_ontology_version,
            model="mock",
            provider="mock",
        )
    return ExtractionResponse(
        entities=[
            ExtractedEntityResponse(
                name="Mock Radar System",
                entity_type="RADAR_SYSTEM",
                confidence=0.95,
                properties={"designation": "AN/APG-00"},
            ),
            ExtractedEntityResponse(
                name="Mock Platform",
                entity_type="PLATFORM",
                confidence=0.90,
                properties={"platform_type": "aircraft"},
            ),
        ],
        relationships=[],
        ontology_version=_ontology_version,
        model="mock",
        provider="mock",
    )


def _run_extraction(text: str, template_group: str | None = None, mode: str = "entities",
                     entities_context: list[dict] | None = None) -> Any:
    """Run docling-graph pipeline synchronously (called in threadpool).

    The ``docling_graph`` import is deferred because the package is only
    available inside the Docker container.
    """
    import tempfile
    from pathlib import Path

    from docling_graph import run_pipeline  # type: ignore[import-untyped]

    model_string = _build_litellm_model_string()

    # Select template based on group or fall back to first available
    if template_group and _templates:
        template_cls = _templates.get(template_group)
    elif _templates:
        template_cls = next(iter(_templates.values()))
    else:
        template_cls = None

    # Build system prompt based on mode
    if mode == "relationships" and entities_context is not None:
        system_prompt = get_relationship_prompt(entities_context)
    elif template_group:
        system_prompt = get_entity_prompt(template_group)
    else:
        system_prompt = None

    # Write text to a temp .md file — docling-graph's input handler tries
    # Path(source).exists() first, which raises ENAMETOOLONG for long text
    tmp = tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8")
    try:
        tmp.write(text)
        tmp.close()

        config: dict[str, Any] = {
            "source": tmp.name,
            "template": template_cls,
            "backend": "llm",
            "inference": "remote",
            "model_override": model_string,
            "provider_override": LLM_PROVIDER,
            "dump_to_disk": False,
            "llm_overrides": {
                "connection": {"base_url": OLLAMA_BASE_URL},
            },
        }
        if system_prompt:
            config["llm_overrides"]["system_prompt"] = system_prompt

        context = run_pipeline(config=config, mode="api")
        return context.knowledge_graph
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _graph_to_response(graph: Any) -> tuple[list[ExtractedEntityResponse], list[ExtractedRelationshipResponse]]:
    """Convert a NetworkX DiGraph to lists of entity/relationship responses."""
    entities: list[ExtractedEntityResponse] = []
    relationships: list[ExtractedRelationshipResponse] = []

    for node_id, data in graph.nodes(data=True):
        entities.append(
            ExtractedEntityResponse(
                name=data.get("name", str(node_id)),
                entity_type=data.get("entity_type", data.get("type", "UNKNOWN")),
                confidence=float(data.get("confidence", 1.0)),
                properties={
                    k: v
                    for k, v in data.items()
                    if k not in {"name", "entity_type", "type", "confidence"}
                },
            )
        )

    for src, dst, data in graph.edges(data=True):
        src_data = graph.nodes[src]
        dst_data = graph.nodes[dst]
        relationships.append(
            ExtractedRelationshipResponse(
                from_name=src_data.get("name", str(src)),
                from_type=src_data.get("entity_type", src_data.get("type", "UNKNOWN")),
                rel_type=data.get("rel_type", data.get("type", "RELATED_TO")),
                to_name=dst_data.get("name", str(dst)),
                to_type=dst_data.get("entity_type", dst_data.get("type", "UNKNOWN")),
                confidence=float(data.get("confidence", 1.0)),
            )
        )

    return entities, relationships


@app.post("/extract", response_model=ExtractionResponse)
async def extract(request: ExtractionRequest):
    """Extract entities and relationships from document text."""
    if _templates is None:
        raise HTTPException(status_code=503, detail="Service not ready — templates not loaded")

    # Validate template_group if provided
    if request.template_group and request.template_group not in _templates:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown template_group '{request.template_group}'. "
            f"Valid groups: {list(_templates.keys())}",
        )

    # Validate mode
    if request.mode not in ("entities", "relationships"):
        raise HTTPException(
            status_code=422,
            detail=f"Unknown mode '{request.mode}'. Valid modes: entities, relationships",
        )

    # Warn on ontology version mismatch
    if request.ontology_version and request.ontology_version != _ontology_version:
        logger.warning(
            "Ontology version mismatch: request=%s, loaded=%s",
            request.ontology_version,
            _ontology_version,
        )

    logger.info(
        "Extracting from document %s (%d chars, group=%s, mode=%s)",
        request.document_id,
        len(request.text),
        request.template_group or "legacy",
        request.mode,
    )

    # Mock mode
    if LLM_PROVIDER == "mock":
        logger.info("Mock mode — returning canned extraction for document %s", request.document_id)
        return _mock_extraction_response(mode=request.mode, template_group=request.template_group)

    try:
        graph = await run_in_threadpool(
            _run_extraction,
            request.text,
            request.template_group,
            request.mode,
            request.entities_context,
        )
    except Exception:
        logger.exception("Extraction failed for document %s", request.document_id)
        raise HTTPException(status_code=503, detail="Extraction pipeline failed")

    entities, relationships = _graph_to_response(graph)

    logger.info(
        "Extracted %d entities, %d relationships from document %s (group=%s, mode=%s)",
        len(entities),
        len(relationships),
        request.document_id,
        request.template_group or "legacy",
        request.mode,
    )

    return ExtractionResponse(
        entities=entities,
        relationships=relationships,
        ontology_version=_ontology_version,
        model=LLM_MODEL,
        provider=LLM_PROVIDER,
    )
```

**Step 4: Run tests to verify they pass**

```bash
cd docker/docling-graph && python -m pytest tests/test_extraction.py -v
```

Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add docker/docling-graph/app/main.py docker/docling-graph/tests/test_extraction.py
git commit -m "feat(docling-graph): route extraction by template_group and mode with prompts"
```

---

### Task 5: Update pipeline client `docling_graph_service.py`

**Files:**
- Modify: `app/services/docling_graph_service.py`

**Step 1: Update `extract_graph()` signature and payload**

Add `template_group`, `mode`, and `entities_context` parameters to `extract_graph()` in `app/services/docling_graph_service.py`:

```python
def extract_graph(
    text: str,
    document_id: str,
    *,
    ontology_version: str | None = None,
    template_group: str | None = None,
    mode: str = "entities",
    entities_context: list[dict] | None = None,
) -> dict[str, Any]:
    """Extract entities and relationships via the Docling-Graph service.

    Returns a dict with keys: entities, relationships, ontology_version, model, provider.
    Raises httpx.HTTPStatusError on service errors (caller should retry).
    Raises DoclingGraphCapacityError when all concurrency permits are in use.
    """
    settings = get_settings()
    url = f"{settings.docling_graph_base_url}/extract"
    timeout = settings.docling_graph_timeout

    payload: dict[str, Any] = {
        "document_id": document_id,
        "text": text,
        "mode": mode,
    }
    if ontology_version:
        payload["ontology_version"] = ontology_version
    if template_group:
        payload["template_group"] = template_group
    if entities_context is not None:
        payload["entities_context"] = entities_context

    # --- Redis concurrency gate (unchanged) ---
    r = _get_redis()
    concurrency = settings.docling_graph_concurrency
    lock_timeout = timeout + 60
    permit_lock = None

    for permit_i in range(concurrency):
        candidate = r.lock(
            f"docling-graph:permit:{permit_i}",
            timeout=lock_timeout,
            blocking=False,
        )
        if candidate.acquire(blocking=False):
            permit_lock = candidate
            break

    if permit_lock is None:
        logger.warning(
            "Docling-Graph at capacity (%d/%d) for document %s — raising for retry",
            concurrency,
            concurrency,
            document_id,
        )
        raise DoclingGraphCapacityError(
            f"All {concurrency} Docling-Graph permits in use"
        )

    logger.info(
        "Calling Docling-Graph service for document %s (%d chars, group=%s, mode=%s, permit acquired)",
        document_id,
        len(text),
        template_group or "legacy",
        mode,
    )

    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    finally:
        try:
            permit_lock.release()
        except redis_lib.exceptions.LockNotOwnedError:
            logger.warning(
                "Docling-Graph permit lock expired before release for document %s",
                document_id,
            )

    result = response.json()

    entity_count = len(result.get("entities", []))
    rel_count = len(result.get("relationships", []))
    logger.info(
        "Docling-Graph returned %d entities, %d relationships for document %s (group=%s, mode=%s, model=%s)",
        entity_count,
        rel_count,
        document_id,
        template_group or "legacy",
        mode,
        result.get("model", "unknown"),
    )

    return result
```

**Step 2: Verify import still works**

```bash
python -c "from app.services.docling_graph_service import extract_graph; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add app/services/docling_graph_service.py
git commit -m "feat: add template_group/mode/entities_context to docling_graph_service client"
```

---

### Task 6: Update pipeline task `derive_ontology_graph` for 6-call extraction

**Files:**
- Modify: `app/workers/pipeline.py` (lines 1437-1652, the `derive_ontology_graph` task)

**Step 1: Write the failing test**

Add a new test to `tests/integration/test_pipeline_graph.py`:

```python
class TestBatchedExtraction:
    """Tests for the 6-call batched extraction (5 entity groups + 1 relationship pass)."""

    def test_calls_extract_graph_six_times(
        self, db_session, sample_document_id, sample_document_element
    ):
        """Pipeline should call extract_graph 6 times: 5 entity + 1 relationship."""
        from app.workers.pipeline import derive_ontology_graph

        # Entity pass returns different entities per group
        entity_results = {
            "reference": {"entities": [{"name": "TM-001", "entity_type": "DOCUMENT", "confidence": 0.9, "properties": {}}], "relationships": [], "ontology_version": "3.0.0", "model": "test", "provider": "ollama"},
            "equipment": {"entities": [{"name": "Patriot PAC-3", "entity_type": "MISSILE_SYSTEM", "confidence": 0.95, "properties": {}}], "relationships": [], "ontology_version": "3.0.0", "model": "test", "provider": "ollama"},
            "rf_signal": {"entities": [{"name": "X-band", "entity_type": "FREQUENCY_BAND", "confidence": 0.88, "properties": {}}], "relationships": [], "ontology_version": "3.0.0", "model": "test", "provider": "ollama"},
            "weapon": {"entities": [{"name": "SARH", "entity_type": "GUIDANCE_METHOD", "confidence": 0.85, "properties": {}}], "relationships": [], "ontology_version": "3.0.0", "model": "test", "provider": "ollama"},
            "operational": {"entities": [{"name": "MIL-STD-1553B", "entity_type": "STANDARD", "confidence": 0.92, "properties": {}}], "relationships": [], "ontology_version": "3.0.0", "model": "test", "provider": "ollama"},
        }
        rel_result = {
            "entities": [],
            "relationships": [
                {"from_name": "Patriot PAC-3", "from_type": "MISSILE_SYSTEM", "rel_type": "OPERATES_IN_BAND", "to_name": "X-band", "to_type": "FREQUENCY_BAND", "confidence": 0.8},
            ],
            "ontology_version": "3.0.0", "model": "test", "provider": "ollama",
        }

        call_count = {"n": 0}

        def mock_extract(text, doc_id, *, ontology_version=None, template_group=None, mode="entities", entities_context=None):
            call_count["n"] += 1
            if mode == "relationships":
                # Verify entities_context was passed
                assert entities_context is not None
                assert len(entities_context) == 5  # one from each group
                return rel_result
            return entity_results[template_group]

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.workers.pipeline._get_pipeline_run_id", return_value=None),
            patch("app.db.session.get_neo4j_driver", return_value=MagicMock()),
            patch("app.services.neo4j_graph.upsert_nodes_batch", return_value=5),
            patch("app.services.neo4j_graph.upsert_relationships_batch", return_value=1),
            patch("app.services.docling_graph_service.extract_graph", side_effect=mock_extract),
        ):
            result = derive_ontology_graph.run(sample_document_id)

        assert call_count["n"] == 6
        assert result["status"] == "ok"
        assert result["nodes"] == 5

    def test_partial_group_failure_continues(
        self, db_session, sample_document_id, sample_document_element
    ):
        """If one entity group fails, others should still be processed."""
        from app.workers.pipeline import derive_ontology_graph
        import httpx

        call_count = {"n": 0}
        ok_result = {"entities": [{"name": "Test", "entity_type": "PLATFORM", "confidence": 0.9, "properties": {}}], "relationships": [], "ontology_version": "3.0.0", "model": "test", "provider": "ollama"}
        rel_result = {"entities": [], "relationships": [], "ontology_version": "3.0.0", "model": "test", "provider": "ollama"}

        def mock_extract(text, doc_id, *, ontology_version=None, template_group=None, mode="entities", entities_context=None):
            call_count["n"] += 1
            if mode == "relationships":
                return rel_result
            if template_group == "rf_signal":
                mock_resp = MagicMock()
                mock_resp.status_code = 503
                mock_resp.request = MagicMock()
                raise httpx.HTTPStatusError("fail", request=mock_resp.request, response=mock_resp)
            return ok_result

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.workers.pipeline._get_pipeline_run_id", return_value=None),
            patch("app.db.session.get_neo4j_driver", return_value=MagicMock()),
            patch("app.services.neo4j_graph.upsert_nodes_batch", return_value=4),
            patch("app.services.neo4j_graph.upsert_relationships_batch", return_value=0),
            patch("app.services.docling_graph_service.extract_graph", side_effect=mock_extract),
        ):
            result = derive_ontology_graph.run(sample_document_id)

        # Should still succeed with partial results
        assert result["status"] == "ok"
```

**Step 2: Run test to verify it fails**

```bash
./scripts/run_tests.sh tests/integration/test_pipeline_graph.py::TestBatchedExtraction -v
```

Expected: FAIL — `extract_graph` called only once with old signature.

**Step 3: Update `derive_ontology_graph` in `app/workers/pipeline.py`**

Replace the extraction call block (lines ~1486-1499) with the batched extraction loop. The key change is in the section between text assembly and Neo4j import:

```python
        # ---- Batched entity extraction (5 groups + 1 relationship pass) ----
        from app.services.docling_graph_service import extract_graph

        EXTRACTION_GROUPS = ["reference", "equipment", "rf_signal", "weapon", "operational"]

        all_entities: list[dict] = []
        all_relationships: list[dict] = []
        provider = "docling-graph"
        model_name = "unknown"
        group_errors: list[str] = []

        for group in EXTRACTION_GROUPS:
            try:
                result = extract_graph(
                    full_text, document_id,
                    template_group=group, mode="entities",
                )
                provider = result.get("provider", provider)
                model_name = result.get("model", model_name)
                all_entities.extend(result.get("entities", []))
                logger.info(
                    "derive_ontology_graph: group=%s entities=%d for %s",
                    group, len(result.get("entities", [])), document_id,
                )
            except Exception as exc:
                logger.warning(
                    "derive_ontology_graph: group=%s failed for %s: %s — continuing",
                    group, document_id, exc,
                )
                group_errors.append(f"{group}: {exc}")

        # Relationship pass with all discovered entities as context
        entities_context = [
            {"name": e["name"], "entity_type": e["entity_type"]}
            for e in all_entities
        ]
        try:
            rel_result = extract_graph(
                full_text, document_id,
                mode="relationships",
                entities_context=entities_context,
            )
            all_relationships = rel_result.get("relationships", [])
            logger.info(
                "derive_ontology_graph: relationship pass returned %d for %s",
                len(all_relationships), document_id,
            )
        except Exception as exc:
            logger.warning(
                "derive_ontology_graph: relationship pass failed for %s: %s",
                document_id, exc,
            )
            group_errors.append(f"relationships: {exc}")

        graph_data = {
            "nodes": all_entities,
            "edges": all_relationships,
        }
        graph_data["mentions"] = _build_entity_mentions(
            graph_data["nodes"], elements, provider,
        )
        if group_errors:
            graph_data["_group_errors"] = group_errors
```

Everything below this (Neo4j import, confidence filtering, DocumentGraphExtraction upsert) remains unchanged.

**Step 4: Run tests to verify they pass**

```bash
./scripts/run_tests.sh tests/integration/test_pipeline_graph.py -v
```

Expected: All tests PASS (both old and new).

**Step 5: Commit**

```bash
git add app/workers/pipeline.py tests/integration/test_pipeline_graph.py
git commit -m "feat: batched 6-call extraction in derive_ontology_graph (5 entity groups + relationships)"
```

---

### Task 7: Update existing pipeline tests for new call signature

**Files:**
- Modify: `tests/integration/test_pipeline_graph.py`

**Step 1: Update MOCK_EXTRACTION_RESULT usage**

The existing tests mock `extract_graph` with `return_value=MOCK_EXTRACTION_RESULT`. With the batched approach, `extract_graph` is called 6 times. Update mocks to use `side_effect` that returns appropriate results based on `template_group`/`mode` kwargs:

```python
def _mock_extract_graph_factory(entity_result=None, rel_result=None):
    """Build a side_effect callable for mocking batched extract_graph calls."""
    if entity_result is None:
        entity_result = MOCK_EXTRACTION_RESULT
    if rel_result is None:
        rel_result = {
            "entities": [],
            "relationships": MOCK_EXTRACTION_RESULT["relationships"],
            "ontology_version": "3.0.0",
            "model": "llama3.2",
            "provider": "ollama",
        }

    def _mock(text, doc_id, *, ontology_version=None, template_group=None, mode="entities", entities_context=None):
        if mode == "relationships":
            return rel_result
        return entity_result

    return _mock
```

Then replace all `patch("app.services.docling_graph_service.extract_graph", return_value=MOCK_EXTRACTION_RESULT)` with `patch("app.services.docling_graph_service.extract_graph", side_effect=_mock_extract_graph_factory())`.

Adjust assertions:
- `result["nodes"]` will be `3 * 5 = 15` (since each entity pass returns 3 entities) unless `upsert_nodes_batch` mock return value is updated. Keep mock return values consistent.
- Update `upsert_nodes_batch` mock to return `15` and `upsert_relationships_batch` to return `2`.
- Update `graph_json["nodes"]` length assertions from `3` to `15`.

**Step 2: Run tests**

```bash
./scripts/run_tests.sh tests/integration/test_pipeline_graph.py -v
```

Expected: All tests PASS.

**Step 3: Commit**

```bash
git add tests/integration/test_pipeline_graph.py
git commit -m "test: update pipeline graph tests for batched extraction signature"
```

---

### Task 8: Run full test suite and update README

**Files:**
- Run: `./scripts/run_tests.sh` or `./manage.sh --test`
- Modify: `README.md` (if extraction architecture section needs updating)

**Step 1: Run full test suite**

```bash
./scripts/run_tests.sh
```

Expected: All tests pass.

**Step 2: Fix any failures**

Address test failures if any arise from the changes.

**Step 3: Update README if needed**

If the README mentions NER, single-template extraction, or the old architecture, update those sections to reflect the batched extraction approach.

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: update README for batched ontology extraction"
```

---

### Task 9: Rebuild docling-graph Docker image and smoke test

**Files:**
- No code changes — deployment verification

**Step 1: Rebuild the docling-graph image**

```bash
./manage.sh --build docling-graph
```

**Step 2: Start the service**

```bash
./manage.sh --up docling-graph
```

**Step 3: Verify health endpoint**

```bash
curl -s http://localhost:8002/health | python -m json.tool
```

Expected: `{"status": "ok", "ontology_version": "3.0.0", "template_count": 5, "groups": ["reference", "equipment", "rf_signal", "weapon", "operational"]}`

**Step 4: Smoke test extraction with mock mode**

```bash
curl -s -X POST http://localhost:8002/extract \
  -H "Content-Type: application/json" \
  -d '{"document_id": "test", "text": "AN/MPQ-53 radar", "template_group": "equipment", "mode": "entities"}' | python -m json.tool
```

Expected: 200 with entities in response.

**Step 5: Reingest a test document**

Upload or reingest a document through the API/frontend to verify the full pipeline works end-to-end with the 6-call extraction.

**Step 6: Commit any fixes**

If smoke testing reveals issues, fix and commit.
