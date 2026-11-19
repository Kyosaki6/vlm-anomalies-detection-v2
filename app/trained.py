# -*- coding: utf-8 -*-
"""Nhãn 'trained' cho một lần chạy (H3). Đọc `RunResult.metadata`, không import streamlit.

`schema_version` không đổi: các trường mới nằm trong `metadata`. Dashboard hiện `TRAINED` khi
`trained` đúng, để một lần chạy có adapter không thể bị nhìn nhầm thành `results/run_A4c/`.
"""
from __future__ import annotations

from vad.results import RunResult

TRAINED_BADGE = "TRAINED"


def is_trained(run: RunResult) -> bool:
    """True khi metadata khai báo lần chạy này đã qua LoRA. Mọi giá trị khác -> False."""
    return run.metadata.get("trained") is True


def trained_badge(run: RunResult) -> str:
    """`TRAINED` hoặc chuỗi rỗng. UI chỉ vẽ khi chuỗi khác rỗng."""
    return TRAINED_BADGE if is_trained(run) else ""
