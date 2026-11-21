# -*- coding: utf-8 -*-
"""relink_train_manifest: path đoán từ Anomaly_Train.txt được thay bằng file thật, hoặc không ghi gì."""
from __future__ import annotations

import csv
from pathlib import Path

from scripts.relink_train_manifest import main, relink

HEADER = ["video_id", "path", "dataset", "split", "n_frames", "fps"]


def _manifest(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    p = tmp_path / "ucf_train.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, lineterminator="\n")
        w.writeheader()
        for vid, path in rows:
            w.writerow({"video_id": vid, "path": path, "dataset": "ucf_crime", "split": "train",
                        "n_frames": "", "fps": ""})
    return p


def _video(root: Path, rel: str) -> None:
    dest = root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"")


def test_relink_points_at_the_real_file_and_keeps_identity(tmp_path: Path):
    root = tmp_path / "train"
    _video(root, "Anomaly-Videos-Part-1/Abuse/Abuse001_x264.mp4")
    _video(root, "Training-Normal-Videos-Part-2/Normal_Videos500_x264.mp4")
    manifest = _manifest(tmp_path, [
        ("Abuse001_x264", "datasets/ucf_crime/train/Abuse/Abuse001_x264.mp4"),
        ("Normal_Videos500_x264", "datasets/ucf_crime/train/Training_Normal/Normal_Videos500_x264.mp4"),
    ])
    rc = main(["--manifest", str(manifest), "--root", str(root)])
    assert rc == 0
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    assert [r["video_id"] for r in rows] == ["Abuse001_x264", "Normal_Videos500_x264"]
    assert rows[0]["path"].endswith("Anomaly-Videos-Part-1/Abuse/Abuse001_x264.mp4")
    assert rows[1]["path"].endswith("Training-Normal-Videos-Part-2/Normal_Videos500_x264.mp4")
    assert {r["split"] for r in rows} == {"train"}


def test_missing_file_writes_nothing(tmp_path: Path, capsys):
    root = tmp_path / "train"
    root.mkdir()
    manifest = _manifest(tmp_path, [("Abuse001_x264", "datasets/ucf_crime/train/Abuse/Abuse001_x264.mp4")])
    before = manifest.read_text(encoding="utf-8")
    rc = main(["--manifest", str(manifest), "--root", str(root)])
    assert rc == 1
    assert "Abuse001_x264" in capsys.readouterr().err
    assert manifest.read_text(encoding="utf-8") == before


def test_two_files_for_one_id_raises():
    rows = [{"video_id": "Abuse001_x264", "path": "x", "dataset": "ucf_crime", "split": "train",
             "n_frames": "", "fps": ""}]
    found = {"Abuse001_x264": [Path("a.mp4"), Path("b.mp4")]}
    try:
        relink(rows, found, Path("train"))
    except ValueError as e:
        assert "Abuse001_x264" in str(e) and "2" in str(e)
    else:
        raise AssertionError("hai file cho một id phải raise")
