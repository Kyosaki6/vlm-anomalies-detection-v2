# -*- coding: utf-8 -*-
"""H1: video test không được có mặt trong train hay trong index. Không tải model, không cần GPU."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from vad.leakcheck import LeakError, check, read_ids

REPO = Path(__file__).resolve().parents[2]
PLANTED = "Abuse028_x264"   # id thật của tập test, manifest ucf_test.csv dòng 2


def _write_csv(path: Path, ids: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id"])
        w.writeheader()
        w.writerows({"video_id": v} for v in ids)


def test_planted_test_id_in_train_fails_and_names_it():
    with pytest.raises(LeakError, match=PLANTED):
        check(test_ids={PLANTED, "Normal_Videos001_x264"}, train_ids={PLANTED})


def test_planted_test_id_in_index_fails_and_names_it():
    with pytest.raises(LeakError, match=PLANTED):
        check(test_ids={PLANTED}, train_ids=set(), index_ids={PLANTED, "Abuse001_x264"})


def test_disjoint_passes():
    check(test_ids={PLANTED}, train_ids={"Abuse001_x264"}, index_ids={"Abuse002_x264"})


def test_index_none_skips_index_check():
    """Không truyền index -> chỉ kiểm train. Tập test sạch với train là đủ."""
    check(test_ids={PLANTED}, train_ids={"Abuse001_x264"})


def test_real_manifests_are_disjoint():
    """1610 train và 290 test đã commit không giao nhau. Đây là nửa xanh của H1."""
    test_ids = read_ids(REPO / "manifests" / "ucf_test.csv")
    train_ids = read_ids(REPO / "manifests" / "ucf_train.csv")
    assert len(test_ids) == 290
    assert len(train_ids) == 1610
    check(test_ids, train_ids)


def test_real_manifest_plus_planted_index_id_fails(tmp_path: Path):
    """Index giả chứa một id test -> fail, và thông báo nêu đúng id đó."""
    test_ids = read_ids(REPO / "manifests" / "ucf_test.csv")
    train_ids = read_ids(REPO / "manifests" / "ucf_train.csv")
    index = tmp_path / "index.csv"
    _write_csv(index, [PLANTED, "Abuse001_x264"])
    with pytest.raises(LeakError, match=PLANTED):
        check(test_ids, train_ids, read_ids(index))


def test_read_ids_rejects_missing_column(tmp_path: Path):
    bad = tmp_path / "no_id.csv"
    bad.write_text("name\nAbuse001_x264\n", encoding="utf-8")
    with pytest.raises(ValueError, match="video_id"):
        read_ids(bad)
