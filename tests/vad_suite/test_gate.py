# -*- coding: utf-8 -*-
"""Test kiểm tra cổng cùng ngày (GAP-4)."""
from __future__ import annotations

from pathlib import Path

import pytest

from vad.train.gate import gate_is_go

REPO = Path(__file__).resolve().parents[2]
REAL_GATE_NOTE = REPO / "notes" / "gb10-gate.md"


def test_real_gate_note_returns_false_for_stale_dates():
    assert not gate_is_go(REAL_GATE_NOTE, "2026-11-20")
    assert not gate_is_go(REAL_GATE_NOTE, "2026-11-21")
    assert gate_is_go(REAL_GATE_NOTE, "2026-11-25")


def test_synthetic_go_note_returns_true(tmp_path: Path):
    note = tmp_path / "gate.md"
    note.write_text(
        "# Gate check\n\n"
        "## 2026-11-21 go\n\n"
        "| Field | Value |\n"
        "|---|---|\n"
        "| Timestamp | 2026-11-21T09:00:00+07:00 |\n"
        "| Verdict | **go** |\n",
        encoding="utf-8",
    )
    assert gate_is_go(note, "2026-11-21")
    # Khác ngày -> false
    assert not gate_is_go(note, "2026-11-22")


def test_synthetic_newest_no_go_returns_false(tmp_path: Path):
    note = tmp_path / "gate.md"
    note.write_text(
        "# Gate check\n\n"
        "## 2026-11-21 go\n\n"
        "| Field | Value |\n"
        "|---|---|\n"
        "| Timestamp | 2026-11-21T09:00:00+07:00 |\n"
        "| Verdict | **go** |\n\n"
        "## 2026-11-21 no-go\n\n"
        "| Field | Value |\n"
        "|---|---|\n"
        "| Timestamp | 2026-11-21T10:00:00+07:00 |\n"
        "| Verdict | **no-go** |\n",
        encoding="utf-8",
    )
    assert not gate_is_go(note, "2026-11-21")


def test_missing_path_and_no_date_return_false(tmp_path: Path):
    missing = tmp_path / "missing.md"
    assert not gate_is_go(missing, "2026-11-21")

    no_date_note = tmp_path / "nodate.md"
    no_date_note.write_text("## go without date\nVerdict: **go**\n", encoding="utf-8")
    assert not gate_is_go(no_date_note, "2026-11-21")
