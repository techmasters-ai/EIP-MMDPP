"""Ontology loading and type-name helpers for GraphRAG and retrieval.

Extraction prompt building and Pydantic extraction models have moved to
the Docling-Graph Docker service.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_ONTOLOGY_PATH = Path(__file__).resolve().parent.parent.parent / "ontology" / "ontology.yaml"


def load_ontology(path: Path | None = None) -> dict[str, Any]:
    """Load and return the ontology YAML as a dict."""
    p = path or _ONTOLOGY_PATH
    with open(p) as f:
        return yaml.safe_load(f)


def load_validation_matrix(
    path: Path | None = None,
) -> set[tuple[str, str, str]]:
    """Load the ontology validation matrix as a set of (source, rel, target) triples.

    Returns an empty set if the ontology doesn't define a validation_matrix.
    """
    ontology = load_ontology(path)
    matrix: set[tuple[str, str, str]] = set()
    for entry in ontology.get("validation_matrix", []):
        source = entry.get("source", "") or entry.get("source_type", "")
        rel = entry.get("relationship", "")
        target = entry.get("target", "") or entry.get("target_type", "")
        if source and rel and target:
            matrix.add((source, rel, target))
    return matrix


def build_entity_type_names(ontology: dict[str, Any] | None = None) -> list[str]:
    """Return a list of all entity type names from the ontology."""
    if ontology is None:
        ontology = load_ontology()
    return [et["name"] for et in ontology.get("entity_types", [])]


def build_relationship_type_names(ontology: dict[str, Any] | None = None) -> list[str]:
    """Return a list of all relationship type names from the ontology."""
    if ontology is None:
        ontology = load_ontology()
    return [rt["name"] for rt in ontology.get("relationship_types", [])]
