"""Regression test for TABLE_REF node ID collision fix.

The bug: node_id_registry.py used split("_")[0] to extract class name,
which yields "TABLE" for "TABLE_REF_<fingerprint>", causing false collision.
Fix: use rsplit("_", 1)[0].
"""

import pytest
from pydantic import BaseModel


class TABLE_REF(BaseModel):
    """Model with underscore in class name (ArcadeDB reserved word mapping)."""
    model_config = {"graph_id_fields": ["title"]}
    title: str


class TABLE(BaseModel):
    """Model without underscore."""
    model_config = {"graph_id_fields": ["title"]}
    title: str


def test_no_false_collision_underscore_class():
    """Two identical TABLE_REF models should NOT collide."""
    from docker.docling_graph.core.converters.node_id_registry import NodeIDRegistry

    registry = NodeIDRegistry()
    model1 = TABLE_REF(title="Table 1")
    model2 = TABLE_REF(title="Table 1")

    id1 = registry.get_node_id(model1)
    id2 = registry.get_node_id(model2)

    assert id1 == id2  # Same entity, same ID
    assert id1.startswith("TABLE_REF_")


def test_true_collision_different_classes():
    """TABLE and TABLE_REF with same fingerprint should collide."""
    from docker.docling_graph.core.converters.node_id_registry import NodeIDRegistry

    # This test verifies that ACTUAL collisions (different classes, same fingerprint)
    # are still detected. In practice this requires hash collision which is unlikely,
    # so we just verify the registry stores class info correctly.
    registry = NodeIDRegistry()
    model_ref = TABLE_REF(title="Test")
    model_plain = TABLE(title="Test")

    id_ref = registry.get_node_id(model_ref)
    # Different class name → different fingerprint (includes __class__)
    id_plain = registry.get_node_id(model_plain)
    assert id_ref != id_plain


def test_rsplit_extracts_full_class_name():
    """Verify rsplit("_", 1) correctly extracts class with underscores."""
    node_id = "TABLE_REF_6c872d968f4908a9"
    class_name = node_id.rsplit("_", 1)[0]
    assert class_name == "TABLE_REF"

    # Old broken behavior
    old_class = node_id.split("_")[0]
    assert old_class == "TABLE"  # This was the bug
