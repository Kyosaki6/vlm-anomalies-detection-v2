# -*- coding: utf-8 -*-
"""Test vòng lặp huấn luyện train() (GAP-1)."""
from __future__ import annotations

import sys
import pytest

from vad.train.holdout import holdout_ids
from vad.train.loop import train
from vad.train.lora import LoRASpec
from vad.train.raft import Clip


def test_train_loop_happy_three_examples():
    spec = LoRASpec()
    examples = [
        {"query": {"video_id": "v1"}, "abstain": True},
        {"query": {"video_id": "v2"}, "abstain": False},
        {"query": {"video_id": "v3"}, "abstain": False},
    ]

    recorded = []

    def mock_step(ex):
        recorded.append(ex["query"]["video_id"])
        return 0.5

    result = train(examples, spec, mock_step)
    assert result["steps"] == 3
    assert result["loss"] == pytest.approx(0.5)
    assert result["abstain_rate"] == pytest.approx(1 / 3)
    assert result["r"] == spec.r
    assert result["alpha"] == spec.alpha
    assert result["dropout"] == spec.dropout
    assert result["target_modules"] == list(spec.target_modules)
    assert recorded == ["v1", "v2", "v3"]


def test_train_with_holdout_exclusion():
    catalog = {
        f"v{i}": Clip(f"v{i}", "Abuse" if i < 10 else "Normal")
        for i in range(20)
    }
    h_ids = set(holdout_ids(catalog, fraction=0.2, seed=20261119))
    train_clips = [c for c in catalog.values() if c.video_id not in h_ids]

    examples = [
        {"query": {"video_id": c.video_id}, "abstain": (i % 2 == 0)}
        for i, c in enumerate(train_clips)
    ]

    seen_ids = []

    def record_step(ex):
        seen_ids.append(ex["query"]["video_id"])
        return 1.2

    spec = LoRASpec()
    res = train(examples, spec, record_step)

    expected_ids = [c.video_id for c in train_clips]
    assert seen_ids == expected_ids
    assert res["steps"] == len(train_clips)
    assert set(seen_ids).isdisjoint(h_ids)

    expected_abstains = sum(1 for ex in examples if ex["abstain"])
    assert res["abstain_rate"] == pytest.approx(expected_abstains / len(examples))
    assert res["loss"] == pytest.approx(1.2)


def test_empty_examples_raises_value_error():
    spec = LoRASpec()
    with pytest.raises(ValueError, match="empty"):
        train([], spec, lambda ex: 0.0)


def test_no_peft_and_no_bitsandbytes_in_sys_modules():
    assert "peft" not in sys.modules
    assert "bitsandbytes" not in sys.modules
