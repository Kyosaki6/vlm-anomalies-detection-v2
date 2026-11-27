# -*- coding: utf-8 -*-
"""H2 gate: eval_allowed chỉ mở khi holdout vượt class prior hơn 0.01."""
from __future__ import annotations

from pathlib import Path

from vad.train.verdict import eval_allowed

REPO = Path(__file__).resolve().parents[2]
REAL_HOLDOUT = REPO / "notes" / "holdout.md"


def _note(tmp_path: Path, accuracy: str, prior: str) -> Path:
    path = tmp_path / "holdout.md"
    path.write_text(
        "# Holdout\n\n"
        "| Field | Value |\n"
        "|---|---|\n"
        f"| citation_accuracy | {accuracy} |\n"
        f"| class_prior | {prior} |\n",
        encoding="utf-8",
    )
    return path


def test_committed_holdout_note_refuses_eval():
    # 0.0211 so với 0.2632: quy tắc dừng đã bắn, eval không được mở.
    assert not eval_allowed(REAL_HOLDOUT)


def test_clearing_gap_allows_eval(tmp_path: Path):
    assert eval_allowed(_note(tmp_path, "0.40", "0.26"))


def test_gap_inside_one_point_refuses_eval(tmp_path: Path):
    # 0.002 không vượt ngưỡng 0.01 mà notes.py ghi.
    assert not eval_allowed(_note(tmp_path, "0.265", "0.263"))


def test_missing_file_refuses_eval(tmp_path: Path):
    assert not eval_allowed(tmp_path / "absent.md")


def test_missing_row_refuses_eval(tmp_path: Path):
    path = tmp_path / "holdout.md"
    path.write_text("| citation_accuracy | 0.40 |\n", encoding="utf-8")
    assert not eval_allowed(path)


def test_unmeasured_cell_refuses_eval(tmp_path: Path):
    assert not eval_allowed(_note(tmp_path, "not measured", "0.26"))
