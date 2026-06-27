"""Regression test for TABLE_REF node ID collision fix.

The bug: node_id_registry.py used split("_")[0] to extract class name,
which yields "TABLE" for "TABLE_REF_<fingerprint>", causing false collision.
Fix: use rsplit("_", 1)[0].
"""

import pytest
from pydantic import BaseModel

# The `docling_graph` package is only installed inside the docling-graph
# service image (it ships there, not in the monorepo's host/CI env). These
# tests patch and exercise docling_graph.core.converters.node_id_registry, so
# they are in-container/CI-only. Skip (not fail) when the package is absent.
pytest.importorskip("docling_graph")


class TABLE_REF(BaseModel):
    """Model with underscore in class name (ArcadeDB reserved word mapping)."""
    model_config = {"graph_id_fields": ["title"]}
    title: str


class TABLE(BaseModel):
    """Model without underscore."""
    model_config = {"graph_id_fields": ["title"]}
    title: str


def _apply_node_id_patch():
    """Apply the same monkey-patch that docker/docling-graph/app/main.py
    applies at service startup. The upstream NodeIDRegistry.get_node_id
    at line 128 uses split('_')[0] which extracts 'TABLE' from
    'TABLE_REF_<hash>', causing false collisions. The patched version
    uses rsplit('_', 1)[0] to preserve the full class name including
    underscores.

    This mirrors the patch at docker/docling-graph/app/main.py:243.
    """
    from docling_graph.core.converters.node_id_registry import NodeIDRegistry

    _original = NodeIDRegistry.get_node_id

    def _patched_get_node_id(self, model_instance, auto_register=True):
        fingerprint = self._generate_fingerprint(model_instance)
        class_name = model_instance.__class__.__name__

        if fingerprint in self.fingerprint_to_id:
            existing_id = self.fingerprint_to_id[fingerprint]
            # FIX: use rsplit to preserve underscored class names
            existing_class = existing_id.rsplit("_", 1)[0] if "_" in existing_id else existing_id
            if existing_class != class_name:
                raise ValueError(
                    f"Node ID collision: fingerprint {fingerprint} maps to both "
                    f"{existing_id} (class: {existing_class}) and {class_name}_... (new class)"
                )
            return existing_id

        if class_name not in self.seen_classes:
            self.seen_classes[class_name] = set()

        node_id = f"{class_name}_{fingerprint}"

        if auto_register:
            self.fingerprint_to_id[fingerprint] = node_id
            self.id_to_fingerprint[node_id] = fingerprint
            self.seen_classes[class_name].add(fingerprint)

        return node_id

    NodeIDRegistry.get_node_id = _patched_get_node_id
    return _original


def test_no_false_collision_underscore_class():
    """Two identical TABLE_REF models should NOT collide.

    The upstream NodeIDRegistry has a bug where split('_')[0] extracts
    'TABLE' from 'TABLE_REF_<hash>', causing a false collision with any
    TABLE model. The patched version in docker/docling-graph/app/main.py
    fixes this. This test applies the same patch inline.
    """
    from docling_graph.core.converters.node_id_registry import NodeIDRegistry

    original = _apply_node_id_patch()
    try:
        registry = NodeIDRegistry()
        model1 = TABLE_REF(title="Table 1")
        model2 = TABLE_REF(title="Table 1")

        id1 = registry.get_node_id(model1)
        id2 = registry.get_node_id(model2)

        assert id1 == id2  # Same entity, same ID
        assert id1.startswith("TABLE_REF_")
    finally:
        NodeIDRegistry.get_node_id = original


def test_true_collision_different_classes():
    """TABLE and TABLE_REF with different class names get different node IDs."""
    from docling_graph.core.converters.node_id_registry import NodeIDRegistry

    original = _apply_node_id_patch()
    try:
        registry = NodeIDRegistry()
        model_ref = TABLE_REF(title="Test")
        model_plain = TABLE(title="Test")

        id_ref = registry.get_node_id(model_ref)
        id_plain = registry.get_node_id(model_plain)
        assert id_ref != id_plain
    finally:
        NodeIDRegistry.get_node_id = original


def test_rsplit_extracts_full_class_name():
    """Verify rsplit("_", 1) correctly extracts class with underscores."""
    node_id = "TABLE_REF_6c872d968f4908a9"
    class_name = node_id.rsplit("_", 1)[0]
    assert class_name == "TABLE_REF"

    # Old broken behavior
    old_class = node_id.split("_")[0]
    assert old_class == "TABLE"  # This was the bug
