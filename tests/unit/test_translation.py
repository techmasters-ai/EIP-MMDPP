import pytest
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


class TestDetectElementLanguages:
    def test_english_text_detected(self):
        from app.services.translation import detect_element_languages
        elements = [{"content_text": "This is a long enough English sentence for reliable language detection by the library.", "element_type": "text"}]
        result = detect_element_languages(elements)
        assert result["document_language"] == "en"
        assert result["non_english_indices"] == []

    def test_russian_text_detected(self):
        from app.services.translation import detect_element_languages
        elements = [{"content_text": "Зенитная ракетная система С-75 Двина предназначена для поражения воздушных целей на средних и больших высотах.", "element_type": "text"}]
        result = detect_element_languages(elements)
        assert result["document_language"] == "ru"
        assert result["non_english_indices"] == [0]

    def test_short_text_skipped(self):
        from app.services.translation import detect_element_languages
        elements = [{"content_text": "Hi", "element_type": "text"}]
        result = detect_element_languages(elements)
        assert result["non_english_indices"] == []

    def test_short_cyrillic_text_detected(self):
        """Short Cyrillic headings should be detected (min_detect_length=5)."""
        from app.services.translation import detect_element_languages
        elements = [{"content_text": "Зенитный Ракетный Комплекс", "element_type": "heading"}]
        result = detect_element_languages(elements)
        assert result["non_english_indices"] == [0]
        assert result["document_language"] == "ru"

    def test_mixed_language_element_per_line_detection(self):
        """Russian lines inside a mostly-English element should be detected."""
        from app.services.translation import detect_element_languages
        elements = [{
            "content_text": (
                "Almaz S-75 Dvina/Desna/Volkhov\n"
                "Air Defence System / HQ-2A/B / CSA-1 / SA-2 Guideline\n"
                "Зенитный Ракетный Комплекс С-75 Двина/Десна/ Волхов"
            ),
            "element_type": "heading",
        }]
        result = detect_element_languages(elements)
        assert result["non_english_indices"] == [0]

    def test_mixed_language_document(self):
        from app.services.translation import detect_element_languages
        elements = [
            {"content_text": "This is English text that is long enough for detection by the langdetect library.", "element_type": "text"},
            {"content_text": "Зенитная ракетная система С-75 Двина предназначена для поражения воздушных целей на средних и больших высотах.", "element_type": "text"},
            {"content_text": "Another English paragraph that should be detected as English by the library.", "element_type": "text"},
        ]
        result = detect_element_languages(elements)
        assert result["non_english_indices"] == [1]
        assert result["document_language"] == "ru"

    def test_short_cjk_text_detected(self):
        from app.services.translation import detect_element_languages
        elements = [{"content_text": "参考文献与结束语", "element_type": "heading"}]
        result = detect_element_languages(elements)
        assert result["non_english_indices"] == [0]
        assert result["document_language"] == "zh-cn"

    def test_empty_elements(self):
        from app.services.translation import detect_element_languages
        result = detect_element_languages([])
        assert result["document_language"] == "en"
        assert result["non_english_indices"] == []


class TestTranslateElements:
    @patch("app.services.translation._ollama_translate")
    def test_translates_non_english_elements(self, mock_translate):
        from app.services.translation import translate_elements
        mock_translate.return_value = "Translated text"
        elements = [
            {"content_text": "Оригинальный текст", "element_order": 0},
            {"content_text": "English text", "element_order": 1},
        ]
        non_english_indices = [0]
        translated = translate_elements(elements, non_english_indices)
        assert translated[0] == "Translated text"
        assert translated[1] == "English text"
        mock_translate.assert_called_once()

    @patch("app.services.translation._ollama_translate")
    def test_batches_elements(self, mock_translate):
        from app.services.translation import translate_elements
        mock_translate.return_value = "Trans A\n---ELEMENT_BOUNDARY---\nTrans B"
        elements = [
            {"content_text": "Текст А", "element_order": 0},
            {"content_text": "Текст Б", "element_order": 1},
        ]
        non_english_indices = [0, 1]
        translated = translate_elements(elements, non_english_indices)
        assert translated[0] == "Trans A"
        assert translated[1] == "Trans B"

    @patch("app.services.translation._ollama_translate")
    def test_fallback_on_boundary_parse_failure(self, mock_translate):
        from app.services.translation import translate_elements
        mock_translate.side_effect = ["No boundaries here", "Trans A", "Trans B"]
        elements = [
            {"content_text": "Текст А", "element_order": 0},
            {"content_text": "Текст Б", "element_order": 1},
        ]
        non_english_indices = [0, 1]
        translated = translate_elements(elements, non_english_indices)
        assert translated[0] == "Trans A"
        assert translated[1] == "Trans B"

    @patch("app.services.translation._ollama_translate")
    def test_no_translation_needed(self, mock_translate):
        from app.services.translation import translate_elements
        elements = [{"content_text": "English", "element_order": 0}]
        translated = translate_elements(elements, [])
        assert translated == ["English"]
        mock_translate.assert_not_called()
