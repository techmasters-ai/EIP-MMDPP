"""Unit tests for docling-raw annotation injection.

Tests the _inject_image_description_annotations helper that enriches
Docling JSON with image description tooltips from the database.
"""

import pytest

pytestmark = pytest.mark.unit


class TestInjectImageDescriptionAnnotations:
    """Test the annotation injection helper used by get_docling_raw_json."""

    def test_injects_description_into_picture_annotations(self):
        from app.api.v1.sources import _inject_image_description_annotations

        doc_json = {
            "pictures": [
                {"prov": [{"page_no": 1}], "annotations": []},
            ],
            "texts": [{"text": "hello"}],
        }
        descriptions = [
            {"page_number": 1, "content_text": "A radar system photo."},
        ]

        _inject_image_description_annotations(doc_json, descriptions)

        annos = doc_json["pictures"][0]["annotations"]
        assert len(annos) == 1
        assert annos[0]["kind"] == "description"
        assert annos[0]["text"] == "A radar system photo."
        assert annos[0]["provenance"] == "llm-generated"

    def test_matches_by_page_and_order(self):
        from app.api.v1.sources import _inject_image_description_annotations

        doc_json = {
            "pictures": [
                {"prov": [{"page_no": 1}], "annotations": []},
                {"prov": [{"page_no": 1}], "annotations": []},
                {"prov": [{"page_no": 2}], "annotations": []},
            ],
            "texts": [],
        }
        descriptions = [
            {"page_number": 1, "content_text": "First image on page 1."},
            {"page_number": 1, "content_text": "Second image on page 1."},
            {"page_number": 2, "content_text": "Image on page 2."},
        ]

        _inject_image_description_annotations(doc_json, descriptions)

        assert doc_json["pictures"][0]["annotations"][0]["text"] == "First image on page 1."
        assert doc_json["pictures"][1]["annotations"][0]["text"] == "Second image on page 1."
        assert doc_json["pictures"][2]["annotations"][0]["text"] == "Image on page 2."

    def test_skips_pictures_with_empty_prov(self):
        from app.api.v1.sources import _inject_image_description_annotations

        doc_json = {
            "pictures": [
                {"prov": [], "annotations": []},
                {"prov": [{"page_no": 1}], "annotations": []},
            ],
            "texts": [],
        }
        descriptions = [
            {"page_number": 1, "content_text": "Page 1 image."},
        ]

        _inject_image_description_annotations(doc_json, descriptions)

        # First picture (no prov) should have no annotations
        assert len(doc_json["pictures"][0]["annotations"]) == 0
        # Second picture should get the description
        assert len(doc_json["pictures"][1]["annotations"]) == 1

    def test_handles_mismatched_counts_gracefully(self):
        """More pictures than descriptions on a page — extra pictures get nothing."""
        from app.api.v1.sources import _inject_image_description_annotations

        doc_json = {
            "pictures": [
                {"prov": [{"page_no": 1}], "annotations": []},
                {"prov": [{"page_no": 1}], "annotations": []},
                {"prov": [{"page_no": 1}], "annotations": []},
            ],
            "texts": [],
        }
        descriptions = [
            {"page_number": 1, "content_text": "Only one description."},
        ]

        _inject_image_description_annotations(doc_json, descriptions)

        assert len(doc_json["pictures"][0]["annotations"]) == 1
        assert len(doc_json["pictures"][1]["annotations"]) == 0
        assert len(doc_json["pictures"][2]["annotations"]) == 0

    def test_no_descriptions_leaves_annotations_empty(self):
        from app.api.v1.sources import _inject_image_description_annotations

        doc_json = {
            "pictures": [
                {"prov": [{"page_no": 1}], "annotations": []},
            ],
            "texts": [],
        }

        _inject_image_description_annotations(doc_json, [])

        assert len(doc_json["pictures"][0]["annotations"]) == 0

    def test_no_pictures_is_noop(self):
        from app.api.v1.sources import _inject_image_description_annotations

        doc_json = {"pictures": [], "texts": [{"text": "hi"}]}
        descriptions = [
            {"page_number": 1, "content_text": "Orphan description."},
        ]

        _inject_image_description_annotations(doc_json, descriptions)
        assert doc_json["pictures"] == []

    def test_preserves_existing_annotations(self):
        from app.api.v1.sources import _inject_image_description_annotations

        doc_json = {
            "pictures": [
                {"prov": [{"page_no": 1}], "annotations": [
                    {"kind": "classification", "predicted_classes": []}
                ]},
            ],
            "texts": [],
        }
        descriptions = [
            {"page_number": 1, "content_text": "New description."},
        ]

        _inject_image_description_annotations(doc_json, descriptions)

        annos = doc_json["pictures"][0]["annotations"]
        assert len(annos) == 2
        assert annos[0]["kind"] == "classification"
        assert annos[1]["kind"] == "description"
