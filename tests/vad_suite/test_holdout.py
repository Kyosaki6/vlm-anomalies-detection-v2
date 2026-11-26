# -*- coding: utf-8 -*-
"""Test holdout split phân tầng theo lớp (GAP-2)."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from vad.train.holdout import holdout_ids, read_train_catalog
from vad.train.raft import Clip

REPO = Path(__file__).resolve().parents[2]
TRAIN_MANIFEST = REPO / "manifests" / "ucf_train.csv"
TEST_MANIFEST = REPO / "manifests" / "ucf_test.csv"


def test_same_seed_gives_identical_ids():
    catalog = {
        f"v{i}": Clip(f"v{i}", "Abuse" if i < 10 else "Normal")
        for i in range(20)
    }
    ids1 = holdout_ids(catalog, fraction=0.1, seed=20261119)
    ids2 = holdout_ids(catalog, fraction=0.1, seed=20261119)
    assert ids1 == ids2
    assert isinstance(ids1, tuple)


def test_different_seed_gives_different_set():
    catalog = {
        f"v{i}": Clip(f"v{i}", "Abuse" if i < 10 else "Normal")
        for i in range(20)
    }
    ids1 = holdout_ids(catalog, fraction=0.2, seed=20261119)
    ids2 = holdout_ids(catalog, fraction=0.2, seed=42)
    assert ids1 != ids2


def test_real_catalog_holdout_disjoint_and_stratified():
    catalog = read_train_catalog(TRAIN_MANIFEST)
    assert len(catalog) == 1610

    h_ids = holdout_ids(catalog, fraction=0.1, seed=20261119)
    assert len(h_ids) == 161

    # Đọc test manifest để kiểm tra không trùng (disjoint)
    with open(TEST_MANIFEST, encoding="utf-8-sig", newline="") as f:
        test_ids = {row["video_id"] for row in csv.DictReader(f) if row.get("video_id")}

    overlap = set(h_ids) & test_ids
    assert not overlap, f"Holdout trùng test ids: {overlap}"

    # Kiểm tra tỷ lệ từng lớp nằm trong khoảng [8%, 12%]
    by_class: dict[str, list[str]] = {}
    for clip in catalog.values():
        by_class.setdefault(clip.event_class, []).append(clip.video_id)

    holdout_set = set(h_ids)
    for cls_name, vids in by_class.items():
        n_total = len(vids)
        n_holdout = sum(1 for v in vids if v in holdout_set)
        ratio = n_holdout / n_total
        assert 0.08 <= ratio <= 0.12, f"Lớp {cls_name}: ratio {ratio:.3f} không nằm trong [0.08, 0.12]"


def test_invalid_fraction_and_empty_catalog_raise_value_error():
    catalog = {"v1": Clip("v1", "Abuse")}

    with pytest.raises(ValueError, match="fraction"):
        holdout_ids(catalog, fraction=0.0)

    with pytest.raises(ValueError, match="fraction"):
        holdout_ids(catalog, fraction=1.5)

    with pytest.raises(ValueError, match="fraction"):
        holdout_ids(catalog, fraction=1.0)

    with pytest.raises(ValueError, match="empty"):
        holdout_ids({}, fraction=0.1)
