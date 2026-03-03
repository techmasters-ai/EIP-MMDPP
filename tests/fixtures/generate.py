"""Generate binary test fixture files.

Run from the repo root:
    python tests/fixtures/generate.py

Requires: reportlab (PDF generation), Pillow (PNG), python-docx (DOCX).
These are separate from production dependencies — install with:
    pip install reportlab python-docx Pillow
"""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "documents"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Minimal valid PNG (16×16 white + grey checkerboard)
# ---------------------------------------------------------------------------

def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def _make_png(width: int = 16, height: int = 16) -> bytes:
    """Create a minimal valid grayscale PNG."""
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    ihdr = _png_chunk(b"IHDR", ihdr_data)

    # Build raw pixel data: alternating 0x00 / 0xFF for checkerboard
    raw_rows = []
    for y in range(height):
        row = b"\x00"  # filter type none
        for x in range(width):
            row += b"\xff" if (x + y) % 2 == 0 else b"\x00"
        raw_rows.append(row)
    compressed = zlib.compress(b"".join(raw_rows), 9)
    idat = _png_chunk(b"IDAT", compressed)
    iend = _png_chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


# ---------------------------------------------------------------------------
# Military-content PDF (hand-crafted valid PDF 1.4)
# Each PDF object is offset-tracked for the cross-reference table.
# ---------------------------------------------------------------------------

MANUAL_TEXT = """\
TECHNICAL MANUAL

GUIDED MISSILE SYSTEM MK-4 SUBSYSTEM MAINTENANCE

NSN: 1410-01-234-5678

CHAPTER 1 — GUIDANCE COMPUTER SUBSYSTEM

1.1 COMPONENT OVERVIEW
The MK-4 guidance computer (P/N GC-4521-A) is the primary navigation
processor for the Patriot PAC-3 interceptor missile system. It integrates
inertial measurement data with terminal seeker inputs to compute intercept
geometry in real time.

Compliance: MIL-STD-1553B (data bus), MIL-DTL-38999 (connectors).
CAGE Code: 12345

1.2 SUBSYSTEM COMPONENTS
The guidance subsystem contains the following assemblies:
 - Inertial Measurement Unit (P/N IMU-7700): accelerometers and gyroscopes
 - Terminal Seeker Assembly (P/N TSA-2201): radar seeker head
 - Power Conditioning Unit (P/N PCU-0510): 28VDC regulated supply

The IMU-7700 is a subsystem of the MK-4 guidance computer.
The PCU-0510 provides power to the IMU-7700.

1.3 SPECIFICATIONS
 - Operating temperature range: -40°C to +85°C
 - Input voltage: 28 VDC ± 2V
 - Processing speed: 1000 MHz
 - Memory: 256 MB SDRAM
 - MTBF: 5000 hours
 - Weight: 2.3 kg
 - Dimensions: 200 mm × 150 mm × 75 mm

1.4 APPLICABLE STANDARDS
This subsystem meets the requirements of:
 - MIL-STD-461G (electromagnetic interference)
 - MIL-STD-810H (environmental engineering)
 - MIL-PRF-38535 (integrated circuits)

1.5 KNOWN FAILURE MODES
 - Crystal oscillator drift above +70°C operating temperature
 - Connector corrosion on TSA-2201 in salt-fog environments
"""


def _make_pdf_with_text(text: str) -> bytes:
    """Build a minimal but valid PDF 1.4 document containing the given text.

    The text is split into lines and rendered in a simple 10pt Courier font.
    No graphics — purely text for pdfplumber extraction testing.
    """
    lines = text.split("\n")
    # Each page holds ~50 lines at 14pt leading, starting at y=750
    pages_content: list[bytes] = []
    page_size = 50
    for i in range(0, max(len(lines), 1), page_size):
        chunk = lines[i : i + page_size]
        ops: list[str] = ["BT", "/F1 10 Tf", "40 750 Td", "14 TL"]
        for line in chunk:
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({safe}) Tj T*")
        ops.append("ET")
        pages_content.append("\n".join(ops).encode())

    # Build PDF objects
    objects: list[bytes] = []
    offsets: list[int] = []

    def add_obj(content: bytes) -> int:
        obj_num = len(objects) + 1
        objects.append(content)
        return obj_num

    # 1: Catalog (placeholder — updated at the end)
    # 2: Pages (placeholder — updated at the end)
    # 3: Font
    font_obj = add_obj(b"")  # will be replaced
    pages_obj_num = add_obj(b"")  # will be replaced
    catalog_obj_num = add_obj(b"")  # will be replaced

    page_obj_nums: list[int] = []
    stream_obj_nums: list[int] = []

    for content in pages_content:
        stream_num = add_obj(b"")
        page_num = add_obj(b"")
        stream_obj_nums.append(stream_num)
        page_obj_nums.append(page_num)

    # Build actual PDF bytes
    body = b"%PDF-1.4\n"

    obj_byte_offsets: list[int] = []

    def write_obj(obj_num: int, content: bytes) -> bytes:
        return f"{obj_num} 0 obj\n".encode() + content + b"\nendobj\n"

    # Collect all object definitions in order
    obj_defs: dict[int, bytes] = {}

    # Font (obj 1)
    obj_defs[font_obj] = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"
    )

    # Streams and Pages
    kids_refs = " ".join(f"{n} 0 R" for n in page_obj_nums)
    obj_defs[pages_obj_num] = (
        f"<< /Type /Pages /Kids [{kids_refs}] /Count {len(page_obj_nums)} >>".encode()
    )
    obj_defs[catalog_obj_num] = (
        f"<< /Type /Catalog /Pages {pages_obj_num} 0 R >>".encode()
    )

    for i, (stream_num, page_num) in enumerate(zip(stream_obj_nums, page_obj_nums)):
        content = pages_content[i]
        stream_def = (
            f"<< /Length {len(content)} >>\nstream\n".encode()
            + content
            + b"\nendstream"
        )
        obj_defs[stream_num] = stream_def
        obj_defs[page_num] = (
            f"<< /Type /Page /Parent {pages_obj_num} 0 R "
            f"/MediaBox [0 0 612 792] "
            f"/Contents {stream_num} 0 R "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> >>".encode()
        )

    # Write body
    all_nums = sorted(obj_defs.keys())
    byte_offsets: dict[int, int] = {}
    buf = body
    for num in all_nums:
        byte_offsets[num] = len(buf)
        buf += f"{num} 0 obj\n".encode() + obj_defs[num] + b"\nendobj\n"

    # Cross-reference table
    xref_offset = len(buf)
    total_objs = max(all_nums) + 1
    buf += f"xref\n0 {total_objs}\n".encode()
    buf += b"0000000000 65535 f \n"
    for i in range(1, total_objs):
        off = byte_offsets.get(i, 0)
        buf += f"{off:010d} 00000 n \n".encode()

    buf += (
        f"trailer\n<< /Size {total_objs} /Root {catalog_obj_num} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    return buf


# ---------------------------------------------------------------------------
# DOCX (minimal Open XML — no external dependency required)
# ---------------------------------------------------------------------------

def _make_minimal_docx() -> bytes:
    """Return a minimal valid DOCX file with military-content paragraphs."""
    import io
    import zipfile

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )

    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )

    paragraphs = [
        "MAINTENANCE PROCEDURE: MK-4 Guidance Computer",
        "NSN: 1410-01-234-5678",
        "Part Number: GC-4521-A  CAGE: 12345",
        "Reference: MIL-STD-1553B data bus specification",
        "The Patriot PAC-3 system integrates the THAAD terminal defense capability.",
        "Applicable standard: MIL-PRF-38535 (microcircuits).",
        "Specifications: Operating voltage 28 VDC, weight 2.3 kg.",
    ]

    def para(text: str) -> str:
        safe = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        return (
            '<w:p><w:r><w:t xml:space="preserve">' + safe + "</w:t></w:r></w:p>"
        )

    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(para(p) for p in paragraphs)
        + "</w:body></w:document>"
    )

    word_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", doc_xml)
        z.writestr("word/_rels/document.xml.rels", word_rels)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    target = FIXTURES_DIR

    # Technical manual PDF
    pdf_path = target / "sample_technical_manual.pdf"
    pdf_path.write_bytes(_make_pdf_with_text(MANUAL_TEXT))
    print(f"Wrote {pdf_path} ({pdf_path.stat().st_size} bytes)")

    # Schematic PNG
    png_path = target / "sample_schematic.png"
    png_path.write_bytes(_make_png(64, 64))
    print(f"Wrote {png_path} ({png_path.stat().st_size} bytes)")

    # DOCX
    docx_path = target / "sample_docx.docx"
    docx_path.write_bytes(_make_minimal_docx())
    print(f"Wrote {docx_path} ({docx_path.stat().st_size} bytes)")

    # Oversized PDF (>10 MB) for streaming upload test.
    # Build a valid PDF, then append a large PDF comment to exceed 10 MB.
    oversized_path = target / "sample_oversized.pdf"
    base_pdf = _make_pdf_with_text(MANUAL_TEXT)
    # Pad with a PDF comment block (valid to appear after %%EOF in PDF 1.4)
    padding_needed = max(0, 10 * 1024 * 1024 + 1024 - len(base_pdf))
    padding = b"% " + b"x" * 78 + b"\n"
    padded_pdf = base_pdf + padding * (padding_needed // len(padding) + 1)
    oversized_path.write_bytes(padded_pdf)
    print(f"Wrote {oversized_path} ({oversized_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
