"""Tests for Office document image extraction and injection."""

import base64
import io
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# extract_docx_images
# ---------------------------------------------------------------------------

class TestExtractDocxImages:
    @patch("docx.Document")
    def test_extracts_images_from_rels(self, MockDocument):
        """Should extract images from DOCX relationship parts."""
        mock_doc = MagicMock()
        mock_image_part = MagicMock()
        mock_image_part.blob = b"\x89PNG fake image data"
        mock_image_part.content_type = "image/png"

        mock_rel = MagicMock()
        mock_rel.reltype = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
        mock_rel.target_part = mock_image_part

        mock_other_rel = MagicMock()
        mock_other_rel.reltype = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"

        mock_doc.part.rels.values.return_value = [mock_rel, mock_other_rel]
        MockDocument.return_value = mock_doc

        from app.services.office_image_extractor import extract_docx_images
        images = extract_docx_images(b"fake docx bytes")

        assert len(images) == 1
        assert images[0]["content_type"] == "image/png"
        assert images[0]["b64"]  # should be non-empty base64

    @patch("docx.Document")
    def test_no_images(self, MockDocument):
        """DOCX with no image rels should return empty list."""
        mock_doc = MagicMock()
        mock_doc.part.rels.values.return_value = []
        MockDocument.return_value = mock_doc

        from app.services.office_image_extractor import extract_docx_images
        images = extract_docx_images(b"fake docx bytes")
        assert images == []

    @patch("docx.Document")
    def test_corrupt_docx_returns_empty(self, MockDocument):
        """Corrupt DOCX should log warning and return empty list."""
        MockDocument.side_effect = Exception("Bad ZIP file")

        from app.services.office_image_extractor import extract_docx_images
        images = extract_docx_images(b"not a docx")
        assert images == []


# ---------------------------------------------------------------------------
# extract_pptx_images
# ---------------------------------------------------------------------------

class TestExtractPptxImages:
    @patch("pptx.enum.shapes.MSO_SHAPE_TYPE")
    @patch("pptx.Presentation")
    def test_extracts_slide_images(self, MockPresentation, MockShapeType):
        """Should extract images from PPTX slide shapes."""
        MockShapeType.PICTURE = 13
        MockShapeType.GROUP = 6

        mock_image = MagicMock()
        mock_image.blob = b"\x89PNG slide image"
        mock_image.content_type = "image/jpeg"

        mock_shape = MagicMock()
        mock_shape.shape_type = 13  # PICTURE
        mock_shape.image = mock_image

        mock_slide = MagicMock()
        mock_slide.shapes = [mock_shape]

        mock_prs = MagicMock()
        mock_prs.slides = [mock_slide]
        MockPresentation.return_value = mock_prs

        from app.services.office_image_extractor import extract_pptx_images
        images = extract_pptx_images(b"fake pptx bytes")

        assert len(images) == 1
        assert images[0]["content_type"] == "image/jpeg"
        assert images[0]["slide"] == 1
        assert images[0]["index"] == 0

    @patch("pptx.Presentation")
    def test_corrupt_pptx_returns_empty(self, MockPresentation):
        MockPresentation.side_effect = Exception("Bad file")

        from app.services.office_image_extractor import extract_pptx_images
        images = extract_pptx_images(b"not a pptx")
        assert images == []


# ---------------------------------------------------------------------------
# inject_images_into_docling_json
# ---------------------------------------------------------------------------

class TestInjectImagesIntoDoclingJson:
    def test_injects_into_empty_slots(self):
        """Should inject image data URIs into PictureItems missing image data."""
        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100).decode()

        docling_json = {
            "pictures": [
                {"image": {"uri": ""}},
                {"image": {"uri": "data:image/png;base64,existing"}},  # already has data
                {"image": {}},  # missing uri entirely
            ]
        }
        images = [
            {"index": 0, "b64": b64, "content_type": "image/png"},
            {"index": 1, "b64": b64, "content_type": "image/png"},
        ]

        from app.services.office_image_extractor import inject_images_into_docling_json
        injected = inject_images_into_docling_json(docling_json, images)

        assert injected == 2
        # First empty slot gets first image
        assert docling_json["pictures"][0]["image"]["uri"].startswith("data:image/png;base64,")
        # Second slot already had data, skipped
        assert docling_json["pictures"][1]["image"]["uri"] == "data:image/png;base64,existing"
        # Third slot (empty uri) gets second image
        assert docling_json["pictures"][2]["image"]["uri"].startswith("data:image/png;base64,")

    def test_no_empty_slots(self):
        """If all pictures have data, should inject 0."""
        docling_json = {
            "pictures": [
                {"image": {"uri": "data:image/png;base64,abc"}},
            ]
        }

        from app.services.office_image_extractor import inject_images_into_docling_json
        injected = inject_images_into_docling_json(docling_json, [{"b64": "x", "content_type": "image/png"}])
        assert injected == 0

    def test_no_images_available(self):
        """If no extracted images, should inject 0."""
        docling_json = {
            "pictures": [
                {"image": {"uri": ""}},
            ]
        }

        from app.services.office_image_extractor import inject_images_into_docling_json
        injected = inject_images_into_docling_json(docling_json, [])
        assert injected == 0

    def test_no_pictures_key(self):
        """If docling_json has no pictures, should return 0."""
        from app.services.office_image_extractor import inject_images_into_docling_json
        injected = inject_images_into_docling_json({}, [{"b64": "x", "content_type": "image/png"}])
        assert injected == 0

    def test_pictures_not_list(self):
        """If pictures is not a list, should return 0."""
        from app.services.office_image_extractor import inject_images_into_docling_json
        injected = inject_images_into_docling_json({"pictures": "invalid"}, [])
        assert injected == 0
