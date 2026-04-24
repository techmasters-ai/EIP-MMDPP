"""Verify detect_and_translate skips translation when langdetect couldn't
classify ANY regex-flagged element. Without this guard, the non-Latin regex
falsely firing on OCR noise sends garbage to the translation LLM and wastes
a full timeout per element (observed: 31-min pipeline stall on a single doc)."""

import uuid
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


def _mk_element(text: str, order: int = 0):
    m = MagicMock()
    m.content_text = text
    m.element_type = "text"
    m.element_order = order
    m.translated_text = None
    return m


class TestLangdetectNoMatchGuard:
    def test_skips_when_all_flagged_elements_unknown(self):
        """When every non-Latin-flagged element fails langdetect classification
        (language_confidences == {"unknown": N}), translate_elements MUST NOT be
        called — the regex false-fired on OCR noise."""
        from app.workers.pipeline import detect_and_translate

        doc_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())

        mock_db = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = [
            _mk_element("some text 你", 0),
            _mk_element("other text Я", 1),
        ]

        fake_detection = {
            "document_language": "unknown",
            "non_english_indices": [0, 1],
            "language_confidences": {"unknown": 2},
        }

        with patch("app.workers.pipeline._get_db", return_value=mock_db), \
             patch("app.workers.pipeline._update_document_status"), \
             patch("app.workers.pipeline._update_stage_run"), \
             patch("app.services.translation.detect_element_languages", return_value=fake_detection), \
             patch("app.services.translation.translate_elements") as mock_translate:
            result = detect_and_translate.run(doc_id, run_id)

        mock_translate.assert_not_called()
        assert result["status"] == "skipped"
        assert result["reason"] == "langdetect_no_match"
        assert result["flagged"] == 2

    def test_proceeds_when_any_element_classified(self):
        """If AT LEAST ONE flagged element was classified by langdetect
        (e.g. 2 ru + 4 unknown), we still attempt translation — the document
        contains real non-English content worth translating."""
        from app.workers.pipeline import detect_and_translate

        doc_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())

        elements = [_mk_element(f"elem {i}", i) for i in range(6)]
        mock_db = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = elements

        fake_detection = {
            "document_language": "unknown",
            "non_english_indices": [0, 1, 2, 3, 4, 5],
            "language_confidences": {"ru": 2, "unknown": 4},
        }

        with patch("app.workers.pipeline._get_db", return_value=mock_db), \
             patch("app.workers.pipeline._update_document_status"), \
             patch("app.workers.pipeline._update_stage_run"), \
             patch("app.services.storage.download_bytes_sync", side_effect=Exception("no artifact")), \
             patch("app.services.storage.upload_bytes_sync"), \
             patch("app.services.translation.detect_element_languages", return_value=fake_detection), \
             patch("app.services.translation.translate_elements", return_value=[f"elem {i}" for i in range(6)]) as mock_translate:
            detect_and_translate.run(doc_id, run_id)

        mock_translate.assert_called_once()
