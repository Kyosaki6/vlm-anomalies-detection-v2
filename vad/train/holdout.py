# -*- coding: utf-8 -*-
"""Holdout split cho tập train UCF-Crime.

Chia tập train thành 10% holdout phân tầng theo lớp (class-stratified).
Holdout này dùng để đo T4 (độ chính xác trích dẫn trước khi test) và không bao giờ
được đưa vào adapter LoRA hay RAFT index.
"""
from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from vad.train.raft import Clip


def read_train_catalog(path: str | Path = "manifests/ucf_train.csv") -> dict[str, Clip]:
    """Đọc manifest train -> {video_id: Clip(video_id, event_class)}."""
    catalog: dict[str, Clip] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video_id"]
            p = row["path"].replace("\\", "/").split("/")
            event_class = p[-2] if len(p) >= 2 else "Unknown"
            catalog[vid] = Clip(vid, event_class)
    return catalog


def holdout_ids(
    catalog: Mapping[str, Clip] | Iterable[Clip],
    fraction: float = 0.1,
    seed: int = 20261119,
) -> tuple[str, ...]:
    """Tách holdout phân tầng theo event_class với tỷ lệ fraction.

    Cùng seed -> cùng holdout_ids.
    fraction phải nằm trong (0, 1), catalog không được rỗng.
    """
    if not (0.0 < fraction < 1.0):
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")

    clips = list(catalog.values()) if isinstance(catalog, Mapping) else list(catalog)
    if not clips:
        raise ValueError("catalog cannot be empty")

    by_class: dict[str, list[str]] = {}
    for clip in clips:
        by_class.setdefault(clip.event_class, []).append(clip.video_id)

    rng = np.random.default_rng(seed)
    holdout: list[str] = []
    for cls in sorted(by_class.keys()):
        group = sorted(by_class[cls])
        rng.shuffle(group)
        k = round(fraction * len(group))
        holdout.extend(group[:k])

    return tuple(sorted(holdout))
