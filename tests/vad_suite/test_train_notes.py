# -*- coding: utf-8 -*-
"""Test tạo ghi chú huấn luyện và kiểm tra holdout (GAP-2, T3, T4)."""
from __future__ import annotations

from pathlib import Path
import pytest

from vad.leakcheck import read_ids
from vad.train.holdout import holdout_ids, read_train_catalog
from vad.train.notes import (
    STOP_SENTENCE,
    render_holdout_note,
    render_lora_train_note,
    write_notes,
)

REPO = Path(__file__).resolve().parents[2]
TRAIN_MANIFEST = REPO / "manifests" / "ucf_train.csv"
TEST_MANIFEST = REPO / "manifests" / "ucf_test.csv"


def test_render_notes_happy(tmp_path: Path):
    run_dict = {
        "commit": "58dab21",
        "config": "r=16, alpha=32, dropout=0.05",
        "steps": 1449,
        "loss": 0.1000,
        "peak_memory": "pending-launch",
        "wall_time": "pending-launch",
        "abstain_rate": 0.3920,
    }

    catalog = read_train_catalog(TRAIN_MANIFEST)
    h_ids = holdout_ids(catalog, fraction=0.1, seed=20261119)
    test_ids = read_ids(TEST_MANIFEST)

    train_note = tmp_path / "lora-train.md"
    holdout_note = tmp_path / "holdout.md"

    not_measured = write_notes(
        run_info=run_dict,
        holdout_ids_seq=h_ids,
        test_ids=test_ids,
        citation_accuracy=None,
        class_prior=None,
        train_note_path=train_note,
        holdout_note_path=holdout_note,
    )
    assert not_measured is False

    # Citation accuracy cao hơn prior -> can_eval is True
    can_eval = write_notes(
        run_info=run_dict,
        holdout_ids_seq=h_ids,
        test_ids=test_ids,
        citation_accuracy=0.6500,
        class_prior=0.4969,
        train_note_path=train_note,
        holdout_note_path=holdout_note,
    )

    assert can_eval is True
    assert train_note.is_file()
    assert holdout_note.is_file()

    # Kiểm tra các trường trong lora-train.md
    t_content = train_note.read_text(encoding="utf-8")
    for key in ("commit", "config", "steps", "loss", "peak_memory", "wall_time", "abstain_rate"):
        assert key in t_content

    # Kiểm tra holdout.md
    h_content = holdout_note.read_text(encoding="utf-8")
    assert "PASS" in h_content
    assert str(len(h_ids)) in h_content
    for vid in h_ids:
        assert vid in h_content

    # Disjoint với test_ids
    assert set(h_ids).isdisjoint(test_ids)


def test_citation_accuracy_equal_prior_stops_eval(tmp_path: Path):
    run_dict = {
        "commit": "58dab21",
        "config": "r=16, alpha=32, dropout=0.05",
        "steps": 10,
        "loss": 0.2,
        "peak_memory": "pending-launch",
        "wall_time": "pending-launch",
        "abstain_rate": 0.4,
    }

    catalog = read_train_catalog(TRAIN_MANIFEST)
    h_ids = holdout_ids(catalog, fraction=0.1, seed=20261119)
    test_ids = read_ids(TEST_MANIFEST)

    train_note = tmp_path / "lora-train.md"
    holdout_note = tmp_path / "holdout.md"

    # Citation accuracy gần hoặc bằng class prior (trong vòng 1%)
    prior = 0.5000
    citation_acc = 0.5050  # diff = 0.0050 <= 0.01

    eval_called = []

    def mock_eval():
        eval_called.append(True)

    can_eval = write_notes(
        run_info=run_dict,
        holdout_ids_seq=h_ids,
        test_ids=test_ids,
        citation_accuracy=citation_acc,
        class_prior=prior,
        train_note_path=train_note,
        holdout_note_path=holdout_note,
    )

    # Note chứa câu dừng
    h_content = holdout_note.read_text(encoding="utf-8")
    assert STOP_SENTENCE in h_content
    assert can_eval is False

    # Eval command KHÔNG được gọi
    if can_eval:
        mock_eval()

    assert len(eval_called) == 0
