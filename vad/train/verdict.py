# -*- coding: utf-8 -*-
"""Cổng eval sau holdout (H2). Đọc notes/holdout.md, không tính lại accuracy.

Cùng ngưỡng với vad/train/notes.py: citation_accuracy phải cao hơn class_prior hơn 0.01.
File thiếu, hàng thiếu, số không đọc được -> False, không raise.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Final

# Khớp notes.py: diff > 0.01 mới cho chạy. 0.265 so với 0.263 không đủ.
CLEAR_MARGIN: Final[float] = 0.01
_ROW: Final[re.Pattern[str]] = re.compile(
    r"^\|\s*(citation_accuracy|class_prior)\s*\|\s*([^|]+?)\s*\|",
    re.MULTILINE,
)


def _cell(text: str, field: str) -> float | None:
    """Ô đầu tiên của field. 'not measured' và chữ khác số -> None."""
    for match in _ROW.finditer(text):
        if match.group(1) != field:
            continue
        try:
            value = float(match.group(2))
        except ValueError:
            return None
        if not math.isfinite(value):
            return None
        return value
    return None


def holdout_numbers(note: str | Path) -> tuple[float, float] | None:
    """(citation_accuracy, class_prior) khi cả hai ô là số hữu hạn. Thiếu -> None."""
    path = Path(note)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return None
    accuracy = _cell(text, "citation_accuracy")
    prior = _cell(text, "class_prior")
    if accuracy is None or prior is None:
        return None
    return accuracy, prior


def eval_allowed(note: str | Path) -> bool:
    """True khi note có cả hai số và citation_accuracy - class_prior > 0.01.

    Không đọc đồng hồ, không tính lại accuracy. File thiếu -> False.
    """
    numbers = holdout_numbers(note)
    if numbers is None:
        return False
    accuracy, prior = numbers
    return accuracy - prior > CLEAR_MARGIN
