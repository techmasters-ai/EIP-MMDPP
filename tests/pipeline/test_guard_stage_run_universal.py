"""Assert every pipeline stage task is wrapped by guard_stage_run with the
CANONICAL stage name used elsewhere in the pipeline.

Locks in two invariants:
  1. The decorator is applied (detects missing application).
  2. The stage_name argument matches the canonical value written to
     ingest.stage_runs by the task body / finalize_document. This is
     critical because derive_text_chunks_and_embeddings (function name)
     writes stage_runs under "derive_text_embeddings" (stage name);
     mismatched bookkeeping silently breaks retrieval.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

# (task_function_name, expected_stage_name_passed_to_guard_stage_run)
# Function name and stage_name intentionally differ for one task — see plan.
PIPELINE_TASKS = [
    ("prepare_document", "prepare_document"),
    ("derive_document_metadata", "derive_document_metadata"),
    ("detect_and_translate", "detect_and_translate"),
    ("derive_picture_descriptions", "derive_picture_descriptions"),
    ("purge_document_derivations", "purge_document_derivations"),
    ("derive_text_chunks_and_embeddings", "derive_text_embeddings"),
    ("derive_image_embeddings", "derive_image_embeddings"),
    ("derive_document_anchors", "derive_document_anchors"),
    ("derive_ontology_graph", "derive_ontology_graph"),
    ("derive_ontology_graph_pass", "derive_ontology_graph_pass"),
    ("derive_structure_links", "derive_structure_links"),
    ("collect_derivations", "collect_derivations"),
    ("derive_canonicalization", "derive_canonicalization"),
    ("finalize_document", "finalize_document"),
]


@pytest.mark.parametrize("task_name,expected_stage", PIPELINE_TASKS)
def test_pipeline_task_has_guard_stage_run(task_name, expected_stage):
    """Task is wrapped AND uses the correct canonical stage_name."""
    import app.workers.pipeline as pipeline
    task = getattr(pipeline, task_name)
    assert hasattr(task.run, "__wrapped__"), (
        f"{task_name} is not wrapped by guard_stage_run — its failures will "
        "leave stage_run rows at RUNNING if an exception escapes the body."
    )
    assert getattr(task.run, "stage_name", None) == expected_stage, (
        f"{task_name} is wrapped with stage_name="
        f"{getattr(task.run, 'stage_name', None)!r} but should use "
        f"{expected_stage!r} to match the canonical _update_stage_run calls."
    )
