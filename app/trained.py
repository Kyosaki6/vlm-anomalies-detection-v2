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


def _hit_id(hit: object) -> str:
    """Một id clip. `..`, dấu phân cách đường dẫn, hoặc kiểu khác str -> ValueError."""
    if not isinstance(hit, str) or not hit or hit in {".", ".."} or "/" in hit or "\\" in hit or ".." in hit:
        raise ValueError(f"raft_hits chứa id không hợp lệ: {hit!r}")
    return hit


def cited_clips(run: RunResult) -> tuple[str, ...]:
    """`metadata.raft_hits` khi lần chạy đã train. Lần chưa train -> rỗng, kể cả khi có hits."""
    if not is_trained(run):
        return ()
    hits = run.metadata.get("raft_hits")
    if hits is None:
        return ()
    if isinstance(hits, str) or not isinstance(hits, (list, tuple)):
        raise ValueError(f"raft_hits phải là danh sách id, nhận {type(hits).__name__}")
    return tuple(_hit_id(hit) for hit in hits)


def cited_caption(run: RunResult) -> str:
    """`Cited clips: a, b`, hoặc rỗng khi không có clip để hiện."""
    clips = cited_clips(run)
    if not clips:
        return ""
    return "Cited clips: " + ", ".join(clips)
