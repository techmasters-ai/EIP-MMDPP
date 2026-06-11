"""Tests for the load_csv_possets loader in scripts/mine_pass_keywords.py.

Pure-function test — no DB required.
"""
from __future__ import annotations
import csv
from pathlib import Path

from scripts.mine_pass_keywords import load_csv_possets


def _write_fixture(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a minimal bakeoff CSV with the expected columns."""
    path = tmp_path / "fixture.csv"
    fieldnames = ["run_id", "pass_name", "chunk_index", "used"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def test_load_csv_possets_basic(tmp_path):
    """Positive rows (used==1) are collected; used==0 rows are excluded."""
    run_a = "aaaaaaaa-0000-0000-0000-000000000001"
    run_b = "bbbbbbbb-0000-0000-0000-000000000002"
    csv_path = _write_fixture(tmp_path, [
        {"run_id": run_a, "pass_name": "radar_power_rf", "chunk_index": 5, "used": 1},
        {"run_id": run_a, "pass_name": "radar_power_rf", "chunk_index": 12, "used": 0},
        {"run_id": run_b, "pass_name": "missile_kinematics", "chunk_index": 3, "used": 1},
    ])
    result = load_csv_possets(str(csv_path))

    assert result[(run_a, "radar_power_rf")] == {5}
    assert (run_a, "radar_power_rf") in result
    assert result[(run_b, "missile_kinematics")] == {3}
    # used==0 row must NOT appear as a key
    assert (run_a, "radar_power_rf") in result
    # chunk 12 must not be in the positive set
    assert 12 not in result[(run_a, "radar_power_rf")]


def test_load_csv_possets_empty_when_all_negative(tmp_path):
    """A CSV with all used==0 rows returns an empty dict."""
    run_a = "aaaaaaaa-0000-0000-0000-000000000001"
    csv_path = _write_fixture(tmp_path, [
        {"run_id": run_a, "pass_name": "radar_timing", "chunk_index": 7, "used": 0},
    ])
    result = load_csv_possets(str(csv_path))
    assert result == {}


def test_load_csv_possets_multiple_chunks_same_pass(tmp_path):
    """Multiple positive rows for the same (run, pass) are merged into one set."""
    run_a = "aaaaaaaa-0000-0000-0000-000000000001"
    csv_path = _write_fixture(tmp_path, [
        {"run_id": run_a, "pass_name": "missile_kinematics", "chunk_index": 9,  "used": 1},
        {"run_id": run_a, "pass_name": "missile_kinematics", "chunk_index": 54, "used": 1},
        {"run_id": run_a, "pass_name": "missile_kinematics", "chunk_index": 92, "used": 1},
    ])
    result = load_csv_possets(str(csv_path))
    assert result[(run_a, "missile_kinematics")] == {9, 54, 92}
