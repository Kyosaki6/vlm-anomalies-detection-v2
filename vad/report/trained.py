# -*- coding: utf-8 -*-
"""Mục báo cáo cho lần chạy đã huấn luyện (H4). Hàm thuần, không đọc file.

A4c và A6 lấy auc_micro, ci_lo, ci_hi từ dict nguồn. R0-R2 ghi 'not measured'.
R3 ghi 'not run' kèm citation accuracy và class prior của holdout. Thiếu một số
holdout -> ValueError, không bịa CI.
"""
from __future__ import annotations

from typing import Final, Mapping

ARXIV_ID: Final[str] = "arXiv:2403.10131"
NOT_MEASURED: Final[str] = "not measured"
_ROWS: Final[tuple[str, ...]] = ("A4c", "A6", "R0", "R1", "R2", "R3")
_REQUIRED: Final[tuple[str, ...]] = ("auc_micro", "ci_lo", "ci_hi")


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}"


def _measured_cell(row: Mapping[str, float], label: str) -> str:
    missing = [key for key in _REQUIRED if key not in row]
    if missing:
        raise ValueError(f"{label} thiếu {missing[0]}")
    auc, lo, hi = (float(row[key]) for key in _REQUIRED)
    return f"{_percent(auc)} [{_percent(lo)}, {_percent(hi)}]"


def _holdout_number(holdout: Mapping[str, float], key: str) -> float:
    if key not in holdout:
        raise ValueError(f"holdout thiếu {key}")
    return float(holdout[key])


def render_section(
    a4c: Mapping[str, float],
    a6: Mapping[str, float],
    holdout: Mapping[str, float],
) -> str:
    """Mục tiếng Việt. Đoạn đầu nói fork đã huấn luyện và nêu Berkeley RAFT.

    Không ghi 'zero-shot': lần chạy này đã train. Không bịa CI cho R3.
    """
    accuracy = _holdout_number(holdout, "citation_accuracy")
    prior = _holdout_number(holdout, "class_prior")
    cells = {
        "A4c": _measured_cell(a4c, "A4c"),
        "A6": _measured_cell(a6, "A6"),
        "R0": NOT_MEASURED,
        "R1": NOT_MEASURED,
        "R2": NOT_MEASURED,
        "R3": (
            f"not run (citation accuracy {_percent(accuracy)}%, "
            f"class prior {_percent(prior)}%)"
        ),
    }
    body = [
        "# Lần chạy đã huấn luyện",
        "",
        "Fork này đã được huấn luyện bằng Berkeley RAFT "
        f"({ARXIV_ID}): mô hình trích các clip láng giềng rồi buộc phải trích dẫn "
        "đúng clip trước khi gọi tên lớp. Lượt holdout không vượt class prior, nên "
        "đánh giá 290 video không được chạy.",
        "",
        "| Mã | ROC-AUC (%) [95% CI] |",
        "|---|---|",
    ]
    body.extend(f"| {label} | {cells[label]} |" for label in _ROWS)
    body.append("")
    return "\n".join(body)
