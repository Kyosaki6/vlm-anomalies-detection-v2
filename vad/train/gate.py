# -*- coding: utf-8 -*-
"""Kiểm tra cùng ngày với GPU gate note (GAP-4).

Chỉ cho phép khởi chạy huấn luyện GPU khi `notes/gb10-gate.md` có mục ghi `**go**`
vào đúng ngày hôm nay (`today`). Một mục no-go mới hơn, một go quá hạn,
hoặc file không tồn tại đều trả về False.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path


def gate_is_go(note: str | Path, today: str | date) -> bool:
    """Trả về True khi và chỉ khi mục mới nhất trong note là go và cùng ngày với today.

    Không shell out ra nvidia-smi, không đọc đồng hồ hệ thống (nhận today làm tham số).
    File thiếu -> False, không raise exception.
    """
    path = Path(note)
    if not path.is_file():
        return False

    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return False

    today_str = today.isoformat() if hasattr(today, "isoformat") else str(today).strip()

    # Tách thành các section theo tiêu đề markdown (## )
    sections = [s.strip() for s in re.split(r"\n(?=## )", text) if s.strip().startswith("## ")]
    if not sections:
        if "**go**" in text or "**no-go" in text.lower():
            sections = [text.strip()]
        else:
            return False

    latest_sec = sections[-1]

    # Phải chứa chuỗi literal "**go**"
    go_matches = list(re.finditer(r"\*\*go\*\*", latest_sec))
    if not go_matches:
        return False

    # Kiểm tra xem có no-go mới hơn trong cùng section không
    nogo_matches = list(re.finditer(r"\*\*no-go[^*]*\*\*", latest_sec, re.IGNORECASE))
    if nogo_matches:
        last_go = max(m.start() for m in go_matches)
        last_nogo = max(m.start() for m in nogo_matches)
        if last_nogo > last_go:
            return False

    # Kiểm tra dòng timestamp (YYYY-MM-DD)
    ts_match = re.search(r"Timestamp.*?(\d{4}-\d{2}-\d{2})", latest_sec, re.IGNORECASE)
    if not ts_match:
        ts_match = re.search(r"(\d{4}-\d{2}-\d{2})", latest_sec)
        if not ts_match:
            return False

    date_str = ts_match.group(1)
    if date_str != today_str:
        return False

    return True
