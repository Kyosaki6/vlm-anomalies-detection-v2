# -*- coding: utf-8 -*-
"""Dựng một ví dụ RAFT: cửa sổ truy vấn, vài clip train, và lời giải là trích dẫn hoặc abstain.

P phần ví dụ có clip đúng lớp (golden) kèm clip nhiễu, lời giải trích đúng clip đó. Phần còn lại chỉ
có clip nhiễu và lời giải là abstain, tức `Normal`, không trích gì. P=0.6 là điểm xuất phát của
kế hoạch, không phải kết quả đã đo.

Clip nhiễu lấy từ đỉnh chỉ mục nhưng khác lớp, không lấy ngẫu nhiên: clip ngẫu nhiên làm bài toán
quá dễ. Không có clip cùng lớp để trích thì abstain, kể cả khi P=1 — không bịa trích dẫn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from vad.index.siglip import Hit, SiglipIndex

SCHEMA_VERSION = 1
ABSTAIN_TEXT = "Normal"
TOP_K = 4


@dataclass(frozen=True, slots=True)
class Query:
    """Một cửa sổ mang nhãn mức video. `embedding` là vector cửa sổ đã chuẩn hoá L2."""

    video_id: str
    start_s: float
    end_s: float
    frame_paths: tuple[str, ...]
    event_class: str
    embedding: np.ndarray


@dataclass(frozen=True, slots=True)
class Clip:
    """Một video trong chỉ mục và lớp mức video của nó. Lớp lấy từ danh sách train, không từ tên file."""

    video_id: str
    event_class: str


def _class_of(video_id: str, catalog: dict[str, Clip]) -> str:
    clip = catalog.get(video_id)
    if clip is None:
        raise KeyError(video_id)
    return clip.event_class


def _golden(query: Query, hits: Sequence[Hit], catalog: dict[str, Clip]) -> Hit | None:
    """Hit cùng lớp gần nhất, khác chính video truy vấn. Không có -> None."""
    same = [h for h in hits if h.video_id != query.video_id
            and _class_of(h.video_id, catalog) == query.event_class]
    return same[0] if same else None


def _distractors(query: Query, hits: Sequence[Hit], catalog: dict[str, Clip],
                 skip: str | None, k: int) -> tuple[Hit, ...]:
    """Hit khác lớp, bỏ video truy vấn và clip đã chọn làm golden, giữ thứ tự cosine."""
    out = [h for h in hits if h.video_id not in (query.video_id, skip)
           and _class_of(h.video_id, catalog) != query.event_class]
    return tuple(out[:k])


def _row(hit: Hit, role: str, rank: int) -> dict[str, object]:
    return {"video_id": hit.video_id, "rank": rank, "role": role, "cosine": hit.cosine}


def build_example(query: Query, index: SiglipIndex, catalog: dict[str, Clip],
                  p: float, rng: np.random.Generator, k: int = TOP_K) -> dict[str, object]:
    """Một ví dụ. `rng.random() < p` và có clip cùng lớp -> golden, ngược lại abstain."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p phải nằm trong [0, 1], nhận {p}")
    if k < 1:
        raise ValueError(f"k phải >= 1, nhận {k}")
    if not query.frame_paths:
        raise ValueError(f"query {query.video_id} không có frame nào")
    hits = index.search(query.embedding, k=len(index.video_ids), exclude_video_id=query.video_id)
    want_golden = bool(rng.random() < p)
    golden = _golden(query, hits, catalog) if want_golden else None
    if golden is None:
        chosen = _distractors(query, hits, catalog, skip=None, k=k)
        return _assemble(query, chosen, golden_present=False, target=ABSTAIN_TEXT, abstain=True)
    rest = _distractors(query, hits, catalog, skip=golden.video_id, k=k - 1)
    return _assemble(query, (golden, *rest), golden_present=True,
                     target=f"{query.event_class}, cited from {golden.video_id}", abstain=False)


def _assemble(query: Query, hits: tuple[Hit, ...], golden_present: bool,
              target: str, abstain: bool) -> dict[str, object]:
    role0 = "golden" if golden_present else "distractor"
    retrieved = [_row(h, role0 if i == 0 else "distractor", i + 1) for i, h in enumerate(hits)]
    return {
        "schema_version": SCHEMA_VERSION,
        "query": {"video_id": query.video_id, "start_s": query.start_s, "end_s": query.end_s,
                  "frame_paths": list(query.frame_paths)},
        "retrieved": retrieved,
        "golden_present": golden_present,
        "target_text": target,
        "abstain": abstain,
    }


def build_dataset(queries: Sequence[Query], index: SiglipIndex, catalog: dict[str, Clip],
                  p: float, seed: int, k: int = TOP_K) -> tuple[dict[str, object], ...]:
    """Cả tập, một seed cho mọi ví dụ theo đúng thứ tự `queries`. Cùng seed -> cùng kết quả."""
    rng = np.random.default_rng(seed)
    return tuple(build_example(q, index, catalog, p, rng, k) for q in queries)
