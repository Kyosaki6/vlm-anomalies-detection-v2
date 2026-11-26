# -*- coding: utf-8 -*-
"""Vòng lặp huấn luyện một pass cho LoRA + RAFT (GAP-1).

Hàm train() nhận callable `steps_fn` được inject để thực thi từng bước huấn luyện.
Không import peft, bitsandbytes, hay torch; không mở/ghi file.
Phần này chạy độc lập trong CI không có GPU.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from vad.train.lora import LoRASpec


def train(
    examples: Sequence[dict[str, Any]],
    spec: LoRASpec,
    steps_fn: Callable[[dict[str, Any]], float],
) -> dict[str, Any]:
    """Chạy một pass huấn luyện qua danh sách examples.

    Trả về dict gồm loss trung bình, số bước (steps), và tỷ lệ abstain (abstain_rate).
    Raise ValueError nếu examples rỗng.
    """
    if not examples:
        raise ValueError("examples cannot be empty")
    if not isinstance(spec, LoRASpec):
        raise TypeError(f"spec phải là LoRASpec, nhận {type(spec).__name__}")

    total_loss = 0.0
    abstain_count = 0
    n = len(examples)

    for ex in examples:
        loss = float(steps_fn(ex))
        total_loss += loss
        if ex.get("abstain", False):
            abstain_count += 1

    return {
        "loss": total_loss / n,
        "steps": n,
        "abstain_rate": abstain_count / n,
        "r": spec.r,
        "alpha": spec.alpha,
        "dropout": spec.dropout,
        "target_modules": list(spec.target_modules),
    }
