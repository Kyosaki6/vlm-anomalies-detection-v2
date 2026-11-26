# -*- coding: utf-8 -*-
"""Test scripts/train_lora.py với cờ --run và --launch (GAP-1, GAP-4)."""
from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

import pytest

from scripts.train_lora import main
from vad.train.holdout import holdout_ids, read_train_catalog

REPO = Path(__file__).resolve().parents[2]
REAL_SMOKE_NOTE = REPO / "notes" / "qwen3-smoke.md"


def test_train_lora_run_with_fake_steps_and_holdout_exclusion(tmp_path: Path):
    manifest = tmp_path / "train_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "path", "dataset", "split"])
        w.writeheader()
        for i in range(20):
            cls_dir = "Abuse" if i < 10 else "Training_Normal_Videos_Anomaly"
            vid = f"vid_{i:03d}"
            w.writerow({
                "video_id": vid,
                "path": f"datasets/ucf_crime/train/{cls_dir}/{vid}.mp4",
                "dataset": "ucf_crime",
                "split": "train",
            })

    catalog = read_train_catalog(manifest)
    expected_holdout = set(holdout_ids(catalog, fraction=0.1, seed=20261119))
    assert len(expected_holdout) > 0

    note_out = tmp_path / "lora-train.md"
    recorded_ids = []

    def mock_step(ex):
        recorded_ids.append(ex["query"]["video_id"])
        return 0.42

    code = main(
        [
            "--run",
            "--train-manifest", str(manifest),
            "--train-note", str(note_out),
            "--smoke-note", str(REAL_SMOKE_NOTE),
            "--seed", "20261119",
        ],
        steps_fn=mock_step,
    )
    assert code == 0
    assert note_out.is_file()

    content = note_out.read_text(encoding="utf-8")
    assert "steps" in content
    assert "| loss | pending-launch |" in content
    assert "abstain_rate" in content
    assert "0.42" not in content

    # Holdout ids không bao giờ được xuất hiện trong training steps
    assert set(recorded_ids).isdisjoint(expected_holdout)
    assert len(recorded_ids) == len(catalog) - len(expected_holdout)


def test_run_stops_when_citation_matches_prior(tmp_path: Path):
    manifest = tmp_path / "train_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "path", "dataset", "split"])
        w.writeheader()
        for i in range(20):
            cls_dir = "Abuse" if i < 10 else "Training_Normal_Videos_Anomaly"
            vid = f"vid_{i:03d}"
            w.writerow({
                "video_id": vid,
                "path": f"datasets/ucf_crime/train/{cls_dir}/{vid}.mp4",
                "dataset": "ucf_crime",
                "split": "train",
            })

    called = []
    code = main(
        [
            "--run",
            "--train-manifest", str(manifest),
            "--test-manifest", str(tmp_path / "missing_test.csv"),
            "--train-note", str(tmp_path / "lora-train.md"),
            "--holdout-note", str(tmp_path / "holdout.md"),
            "--smoke-note", str(REAL_SMOKE_NOTE),
            "--citation-accuracy", "0.50",
            "--class-prior", "0.505",
        ],
        steps_fn=lambda ex: 0.1,
        eval_fn=lambda: called.append(True),
    )
    assert code == 4
    assert called == []
    assert "The adapter predicts the class prior" in (tmp_path / "holdout.md").read_text(encoding="utf-8")


def test_launch_saves_injected_adapter(tmp_path: Path):
    manifest = tmp_path / "train_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "path", "dataset", "split"])
        w.writeheader()
        for i in range(20):
            cls_dir = "Abuse" if i < 10 else "Training_Normal_Videos_Anomaly"
            vid = f"vid_{i:03d}"
            w.writerow({
                "video_id": vid,
                "path": f"datasets/ucf_crime/train/{cls_dir}/{vid}.mp4",
                "dataset": "ucf_crime",
                "split": "train",
            })

    gate = tmp_path / "gate.md"
    gate.write_text(
        "## today go\n\n"
        f"| Timestamp | {date.today().isoformat()}T22:06:11+07:00 |\n"
        "| Verdict | **go** |\n",
        encoding="utf-8",
    )

    def mock_step(ex):
        return 0.2

    mock_step.adapter = {"lora_A.weight": [0.1, 0.2], "lora_B.weight": [0.3]}

    out_dir = tmp_path / "adapter"
    code = main(
        [
            "--launch",
            "--train-manifest", str(manifest),
            "--test-manifest", str(tmp_path / "missing_test.csv"),
            "--train-note", str(tmp_path / "lora-train.md"),
            "--holdout-note", str(tmp_path / "holdout.md"),
            "--gate-note", str(gate),
            "--smoke-note", str(REAL_SMOKE_NOTE),
            "--out-dir", str(out_dir),
            "--seed", "20261119",
        ],
        steps_fn=mock_step,
    )
    assert code == 0
    saved = json.loads((out_dir / "adapter_model.json").read_text(encoding="utf-8"))
    assert saved == mock_step.adapter
    assert "peft" not in sys.modules


def test_launch_without_adapter_raises(tmp_path: Path):
    manifest = tmp_path / "train_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "path", "dataset", "split"])
        w.writeheader()
        for i in range(20):
            cls_dir = "Abuse" if i < 10 else "Training_Normal_Videos_Anomaly"
            vid = f"vid_{i:03d}"
            w.writerow({
                "video_id": vid,
                "path": f"datasets/ucf_crime/train/{cls_dir}/{vid}.mp4",
                "dataset": "ucf_crime",
                "split": "train",
            })

    gate = tmp_path / "gate.md"
    gate.write_text(
        "## today go\n\n"
        f"| Timestamp | {date.today().isoformat()}T22:06:11+07:00 |\n"
        "| Verdict | **go** |\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        main(
            [
                "--launch",
                "--train-manifest", str(manifest),
                "--gate-note", str(gate),
                "--smoke-note", str(REAL_SMOKE_NOTE),
                "--out-dir", str(tmp_path / "adapter"),
            ],
            steps_fn=lambda ex: 0.2,
        )


def test_train_lora_launch_refuses_with_exit_3(tmp_path: Path):
    note = tmp_path / "gate.md"
    note.write_text(
        "## 2026-11-21 no-go\n\n"
        "| Timestamp | 2026-11-21T10:00:00+07:00 |\n"
        "| Verdict | **no-go** |\n",
        encoding="utf-8",
    )
    code = main([
        "--launch",
        "--gate-note", str(note),
        "--smoke-note", str(REAL_SMOKE_NOTE),
    ])
    assert code == 3
    assert "peft" not in sys.modules
