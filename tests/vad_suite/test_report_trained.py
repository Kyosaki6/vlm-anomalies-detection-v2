# -*- coding: utf-8 -*-
"""H4: mục báo cáo sinh từ JSON, không gõ số, không gọi lần train là zero-shot."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vad.report.trained import ARXIV_ID, render_section
from vad.train.verdict import holdout_numbers

REPO = Path(__file__).resolve().parents[2]
SECTION = REPO / "report" / "04b-trained.md"
HOLDOUT_NOTE = REPO / "notes" / "holdout.md"


def _load(name: str) -> dict:
    return json.loads((REPO / "results" / name).read_text(encoding="utf-8-sig"))


def _holdout() -> dict[str, float]:
    numbers = holdout_numbers(HOLDOUT_NOTE)
    assert numbers is not None
    accuracy, prior = numbers
    return {"citation_accuracy": accuracy, "class_prior": prior}


def test_real_a6_cell_matches_the_file():
    text = render_section(_load("ablation_A4c.json"), _load("ablation_A6.json"), _holdout())
    # 0.6790206378435205 -> 67.90, CI [58.39, 77.49]. Khóa ô A6 vào file, không vào hằng.
    assert "| A6 | 67.90 [58.39, 77.49] |" in text
    assert "| A4c | 73.00 [64.45, 81.15] |" in text


def test_section_names_every_row_and_the_arxiv_id():
    text = render_section(_load("ablation_A4c.json"), _load("ablation_A6.json"), _holdout())
    for label in ("A4c", "A6", "R0", "R1", "R2", "R3"):
        assert f"| {label} |" in text
    first_paragraph = text.split("\n\n")[1]
    assert ARXIV_ID in first_paragraph
    assert text.count("not measured") == 3
    assert "not run" in text
    assert "zero-shot" not in text


def test_missing_citation_accuracy_raises():
    with pytest.raises(ValueError, match="citation_accuracy"):
        render_section(_load("ablation_A4c.json"), _load("ablation_A6.json"), {"class_prior": 0.26})


def test_written_section_equals_the_renderer():
    """File đã sinh phải trùng byte với render_section trên đúng các nguồn."""
    if not SECTION.is_file():
        pytest.skip("report/04b-trained.md chưa được sinh")
    expected = render_section(_load("ablation_A4c.json"), _load("ablation_A6.json"), _holdout())
    assert SECTION.read_text(encoding="utf-8") == expected
