# -*- coding: utf-8 -*-
"""N1: build_train_manifest từ chối id test và id trùng, không ghi file khi sai."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.build_train_manifest import build_rows, main, read_train_entries

PLANTED = "Abuse028_x264"


def _train_list(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "Anomaly_Train.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _test_manifest(tmp_path: Path, ids: list[str]) -> Path:
    p = tmp_path / "ucf_test.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "path", "dataset", "split"])
        w.writeheader()
        w.writerows({"video_id": v, "path": f"datasets/{v}.mp4", "dataset": "ucf_crime", "split": "test"} for v in ids)
    return p


def test_planted_test_id_exits_nonzero_and_writes_nothing(tmp_path: Path, capsys):
    train = _train_list(tmp_path, ["Abuse/Abuse001_x264.mp4", f"Abuse/{PLANTED}.mp4"])
    test = _test_manifest(tmp_path, [PLANTED])
    out = tmp_path / "ucf_train.csv"
    rc = main(["--train-list", str(train), "--test-manifest", str(test), "--out", str(out), "--expect", "2"])
    assert rc == 1
    assert PLANTED in capsys.readouterr().err
    assert not out.exists()


def test_duplicate_train_id_exits_nonzero(tmp_path: Path, capsys):
    train = _train_list(tmp_path, ["Abuse/Abuse001_x264.mp4", "Abuse/Abuse001_x264.mp4"])
    test = _test_manifest(tmp_path, [PLANTED])
    out = tmp_path / "ucf_train.csv"
    rc = main(["--train-list", str(train), "--test-manifest", str(test), "--out", str(out), "--expect", "1"])
    assert rc == 1
    assert "Abuse001_x264" in capsys.readouterr().err
    assert not out.exists()


def test_wrong_count_writes_nothing(tmp_path: Path):
    train = _train_list(tmp_path, ["Abuse/Abuse001_x264.mp4"])
    test = _test_manifest(tmp_path, [PLANTED])
    out = tmp_path / "ucf_train.csv"
    rc = main(["--train-list", str(train), "--test-manifest", str(test), "--out", str(out), "--expect", "1610"])
    assert rc == 1
    assert not out.exists()


def test_clean_list_writes_train_split(tmp_path: Path):
    train = _train_list(tmp_path, ["Abuse/Abuse001_x264.mp4", "Robbery/Robbery045_x264.mp4"])
    test = _test_manifest(tmp_path, [PLANTED])
    out = tmp_path / "ucf_train.csv"
    rc = main(["--train-list", str(train), "--test-manifest", str(test), "--out", str(out), "--expect", "2"])
    assert rc == 0
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert [r["video_id"] for r in rows] == ["Abuse001_x264", "Robbery045_x264"]
    assert {r["split"] for r in rows} == {"train"}
    assert rows[0]["path"] == "datasets/ucf_crime/train/Abuse/Abuse001_x264.mp4"
    assert rows[0]["n_frames"] == ""


def test_unsafe_entry_raises(tmp_path: Path):
    train = _train_list(tmp_path, ["../etc/passwd"])
    with pytest.raises(ValueError, match="không an toàn"):
        read_train_entries(train)


def test_build_rows_names_the_leaked_id():
    with pytest.raises(ValueError, match=PLANTED):
        build_rows([(PLANTED, "datasets/ucf_crime/train/Abuse/x.mp4")], test_ids={PLANTED})
