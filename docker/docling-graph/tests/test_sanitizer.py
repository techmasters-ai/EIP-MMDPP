"""Tests for the pre-extraction DoclingDocument sanitizer.

Covers `_looks_like_nav_or_tracking`, `_contains_encoded_blob`, and the
`_sanitize_docling_document` blanking pass. Each test asserts a specific
rule (1, 2, or 3) so a regression points directly at the failing predicate.
"""


def test_rule1_ad_tracking_domain_dropped(dg_app_module):
    m = dg_app_module
    text = (
        "[Ready to win bigger;\nfaster and smarter with\nAI?]"
        "(http://d.adroll.com/click/?adroll_insertion_id=48760b03)"
    )
    assert m._looks_like_nav_or_tracking(text) is True


def test_rule2_pure_link_lines_dropped(dg_app_module):
    m = dg_app_module
    text = (
        "[FIFB-22](https://www.ausairpower.net/raptor.html)\n"
        "[PACRIM WEPS](https://www.ausairpower.net/region.html)"
    )
    assert m._looks_like_nav_or_tracking(text) is True


def test_rule3a_base64_blob_with_mixed_case_and_digits(dg_app_module):
    m = dg_app_module
    text = (
        "adroll_ad_payload=__HIA9QBkwHFA8HIA70AAZ1TXYjcVBSeZNb6"
        "UFHclQF9WlBkHzbZ5Oa_Wkp2dudvd5OZHeeXpfEmuTMTZzLJziQ7uoK0qA9CW"
    )
    assert m._contains_encoded_blob(text) is True
    assert m._looks_like_nav_or_tracking(text) is True


def test_rule3b_long_percent_encoded_fragment_dropped(dg_app_module):
    m = dg_app_module
    # Continuation of a tracker URL that lost its hostname when docling
    # split the URL on a line break. Has many %XX triplets in a single
    # whitespace-delimited token; only Rule 3b catches this.
    text = (
        "0%26kv7%3DBA%26kv10%3D%5BISP%5D%26kv11%3D831042526613476378943"
        "0582991558063169%26kv18%3D%26kv19%3D%5BDevice_ID%5D%26kv24%3DDesktop&"
    )
    assert m._contains_encoded_blob(text) is True
    assert m._looks_like_nav_or_tracking(text) is True


def test_rule3a_data_uri_base64_dropped(dg_app_module):
    m = dg_app_module
    text = (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        "AAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    assert m._contains_encoded_blob(text) is True


def test_rule3_preserves_hex_hash(dg_app_module):
    m = dg_app_module
    # SHA-256 hex: 64 chars but no uppercase, no padding -> Rule 3 must NOT match.
    text = "1debdccc062ab7af3be05d10d9f6513b1234567890abcdef1234567890abcdef"
    assert m._contains_encoded_blob(text) is False


def test_rule3_preserves_uuid(dg_app_module):
    m = dg_app_module
    text = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
    assert m._contains_encoded_blob(text) is False


def test_rule3_preserves_short_serial_number(dg_app_module):
    m = dg_app_module
    text = "Catalog reference RP-12345-A6B7C8-D9E0F1 — see annex 4."
    assert m._contains_encoded_blob(text) is False
    assert m._looks_like_nav_or_tracking(text) is False


def test_rule3_preserves_decimal_run(dg_app_module):
    m = dg_app_module
    # All-digits, no letters: fails has_mixed; no padding chars.
    text = "8310425266134763789430582991558063169834567890123456789012345"
    assert m._contains_encoded_blob(text) is False


def test_rule3_preserves_prose_with_long_word(dg_app_module):
    m = dg_app_module
    # No single token >= 64 chars.
    text = (
        "The Tombstone radar transmits at 9300 MHz with a peak power of "
        "600 kW and a pulse width of 1.5 microseconds."
    )
    assert m._contains_encoded_blob(text) is False
    assert m._looks_like_nav_or_tracking(text) is False


def test_caption_is_preserved_unconditionally(dg_app_module):
    m = dg_app_module
    doc = {
        "texts": [
            {
                "label": "caption",
                "text": (
                    "Figure 1: adroll.com banner placement on the SAM-3 page "
                    "(this text is a caption and must be kept)."
                ),
                "orig": "...",
            },
        ],
    }
    stats: dict = {}
    out = m._sanitize_docling_document(doc, stats)
    assert stats["texts_dropped"] == 0
    assert out["texts"][0]["text"].startswith("Figure 1:")


def test_sanitize_blanks_in_place_with_base64(dg_app_module):
    m = dg_app_module
    doc = {
        "texts": [
            {"label": "text", "text": "Real document content about SA-2 Volkhov radar.", "orig": "..."},
            {
                "label": "text",
                "text": (
                    "adroll_ad_payload=__HIA9QBkwHFA8HIA70AAZ1TXYjcVBSeZNb6"
                    "UFHclQF9WlBkHzbZ5Oa_Wkp2dudvd5OZHeeXpfEmuTMTZzLJziQ7uoK0qA9CW"
                ),
                "orig": "...",
            },
        ],
    }
    stats: dict = {}
    out = m._sanitize_docling_document(doc, stats)
    assert stats["texts_in"] == 2
    assert stats["texts_dropped"] == 1
    # First element preserved (no rule matches), second blanked in place.
    assert out["texts"][0]["text"].startswith("Real document content")
    assert out["texts"][1]["text"] == ""
    assert out["texts"][1]["orig"] == ""
