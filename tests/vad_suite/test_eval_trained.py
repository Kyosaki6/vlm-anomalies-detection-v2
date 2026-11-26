# -*- coding: utf-8 -*-
"""Test scripts/eval_trained.py (GAP-3, GAP-5)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from scripts.eval_trained import main

REPO = Path(__file__).resolve().parents[2]
REAL_TEST_MANIFEST = REPO / "manifests" / "ucf_test.csv"
REAL_TRAIN_MANIFEST = REPO / "manifests" / "ucf_train.csv"


def test_eval_trained_happy(tmp_path: Path):
    # Tạo file A6 tạm
    a6_file = tmp_path / "ablation_A6.json"
    a6_payload = {
        "ablation": "A6",
        "threshold": 1.1685914439041403,
        "auc_micro": 0.6790206378435205,
        "ci_lo": 0.5839401630377785,
        "ci_hi": 0.7749175179170449,
        "operating_after": {
            "window_precision": 0.49612403100775193,
            "fa_per_hour": 0.1664342238077993,
            "incident_recall": 0.3974358974358974,
        },
    }
    a6_file.write_text(json.dumps(a6_payload), encoding="utf-8")

    out_file = tmp_path / "trained" / "R3.json"

    def mock_scorer(test_ids, tau):
        assert round(tau, 4) == 1.1686
        return {
            "scores_per_video": [[0.0, 1.0], [0.0, 1.0]],
            "labels_per_video": [[False, True], [False, True]],
            "n_false_alarms": 2,
            "normal_seconds": 3600.0,
            "true_positive_windows": 3,
            "n_windows": 4,
            "incidents_detected": 1,
            "incidents": 2,
        }

    rc = main(
        [
            "--test-manifest", str(REAL_TEST_MANIFEST),
            "--train-manifest", str(REAL_TRAIN_MANIFEST),
            "--a6-file", str(a6_file),
            "--out", str(out_file),
            "--n-bootstraps", "20",
        ],
        scorer_fn=mock_scorer,
    )
    assert rc == 0
    assert out_file.is_file()

    r3_data = json.loads(out_file.read_text(encoding="utf-8"))

    # 6 chỉ số ra từ hàm đo, không phải số scorer đưa sẵn.
    assert r3_data["auc_micro"] == pytest.approx(1.0)
    assert r3_data["ci_lo"] == pytest.approx(1.0)
    assert r3_data["ci_hi"] == pytest.approx(1.0)
    assert r3_data["fa_per_hour"] == pytest.approx(2.0)
    assert r3_data["window_precision"] == pytest.approx(0.75)
    assert r3_data["incident_recall"] == pytest.approx(0.5)
    assert r3_data["delta_auc"] == pytest.approx(1.0 - a6_payload["auc_micro"])

    # Số liệu A6 bằng chính xác file tạm
    assert r3_data["a6"] == a6_payload

    # tau không đổi
    assert r3_data["threshold"] == a6_payload["threshold"]

    # R0, R1, R2 có status: not measured
    assert r3_data["R0"] == {"status": "not measured"}
    assert r3_data["R1"] == {"status": "not measured"}
    assert r3_data["R2"] == {"status": "not measured"}

    # Không import peft ở module level
    assert "peft" not in sys.modules


def test_planted_test_id_aborts_eval_and_writes_nothing(tmp_path: Path):
    # Plant a test id into train manifest
    test_manifest = tmp_path / "test.csv"
    with open(test_manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id"])
        w.writeheader()
        w.writerow({"video_id": "Abuse028_x264"})

    train_manifest = tmp_path / "train.csv"
    with open(train_manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id"])
        w.writeheader()
        w.writerow({"video_id": "Abuse028_x264"})  # LEAK!

    out_file = tmp_path / "trained" / "R3.json"

    rc = main(
        [
            "--test-manifest", str(test_manifest),
            "--train-manifest", str(train_manifest),
            "--out", str(out_file),
        ],
        scorer_fn=lambda ids, tau: {},
    )
    assert rc != 0
    assert not out_file.exists()
