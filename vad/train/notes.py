# -*- coding: utf-8 -*-
"""Định dạng và ghi log ghi chú huấn luyện LoRA và kiểm tra holdout (T3, T4).

Bao gồm:
- notes/lora-train.md: commit, config, steps, loss, peak_memory, wall_time, abstain_rate.
- notes/holdout.md: danh sách video_id holdout, kết quả kiểm tra disjointness với ucf_test.csv,
  so sánh citation accuracy với class prior, và stop sentence nếu chênh lệch <= 1%.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


STOP_SENTENCE = "The adapter predicts the class prior and the eval does not start."
PROCEED_SENTENCE = "Citation accuracy exceeds class prior. Evaluation may proceed."


def render_lora_train_note(run_info: dict[str, Any]) -> str:
    """Định dạng ghi chú huấn luyện notes/lora-train.md."""
    commit = run_info.get("commit", "unknown")
    config = run_info.get("config", "r=16, alpha=32, dropout=0.05")
    steps = run_info.get("steps", 0)
    loss = run_info.get("loss", 0.0)
    peak_memory = run_info.get("peak_memory", "pending-launch")
    wall_time = run_info.get("wall_time", "pending-launch")
    abstain_rate = run_info.get("abstain_rate", 0.0)

    loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
    abstain_str = f"{abstain_rate:.4f}" if isinstance(abstain_rate, (int, float)) else str(abstain_rate)

    lines = [
        "# LoRA Train Log (T3)",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| commit | {commit} |",
        f"| config | {config} |",
        f"| steps | {steps} |",
        f"| loss | {loss_str} |",
        f"| peak_memory | {peak_memory} |",
        f"| wall_time | {wall_time} |",
        f"| abstain_rate | {abstain_str} |",
        "",
    ]
    return "\n".join(lines)


NOT_MEASURED_SENTENCE = "Citation accuracy not measured. Eval does not start."


def render_holdout_note(
    holdout_ids_seq: Sequence[str],
    test_ids: set[str],
    citation_accuracy: float | None,
    class_prior: float | None,
) -> tuple[str, bool]:
    """Định dạng ghi chú holdout notes/holdout.md và kiểm tra quy tắc dừng.

    Trả về (nội dung_markdown, is_at_prior).
    is_at_prior là True nếu cả hai số đều có và chênh lệch <= 0.01.
    Thiếu một trong hai số -> không dừng theo prior, ghi 'not measured'.
    """
    holdout_set = set(holdout_ids_seq)
    overlap = sorted(holdout_set & test_ids)
    disjoint = (len(overlap) == 0)

    if citation_accuracy is None or class_prior is None:
        diff = None
        is_at_prior = False
        verdict_sentence = NOT_MEASURED_SENTENCE
        acc_cell = "not measured"
        prior_cell = "not measured" if class_prior is None else f"{class_prior:.4f}"
        diff_cell = "not measured"
    else:
        diff = abs(citation_accuracy - class_prior)
        is_at_prior = diff <= 0.01
        verdict_sentence = STOP_SENTENCE if is_at_prior else PROCEED_SENTENCE
        acc_cell = f"{citation_accuracy:.4f}"
        prior_cell = f"{class_prior:.4f}"
        diff_cell = f"{diff:.4f}"

    lines = [
        "# Holdout Check Log (T4)",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| holdout_count | {len(holdout_ids_seq)} |",
        f"| disjointness_check | {'PASS (0 overlap)' if disjoint else f'FAIL ({len(overlap)} overlap)'} |",
        f"| citation_accuracy | {acc_cell} |",
        f"| class_prior | {prior_cell} |",
        f"| abs_diff | {diff_cell} |",
        f"| verdict | {verdict_sentence} |",
        "",
        "## Disjointness with manifests/ucf_test.csv",
        "",
        f"Disjointness check result: {'PASS' if disjoint else 'FAIL'}.",
        f"Number of overlapping test videos: {len(overlap)}.",
        "",
        "## Holdout Video IDs",
        "",
    ]
    for vid in sorted(holdout_ids_seq):
        lines.append(f"- {vid}")
    lines.append("")

    return "\n".join(lines), is_at_prior


def write_notes(
    run_info: dict[str, Any],
    holdout_ids_seq: Sequence[str],
    test_ids: set[str],
    citation_accuracy: float | None,
    class_prior: float | None,
    train_note_path: str | Path = "notes/lora-train.md",
    holdout_note_path: str | Path = "notes/holdout.md",
) -> bool:
    """Ghi cả 2 file ghi chú. Trả về True chỉ khi đã đo và citation accuracy vượt class prior.

    Thiếu một trong hai số thì eval không được chạy: chưa đo không phải là đã vượt prior.
    """
    train_text = render_lora_train_note(run_info)
    Path(train_note_path).parent.mkdir(parents=True, exist_ok=True)
    Path(train_note_path).write_text(train_text, encoding="utf-8")

    holdout_text, is_at_prior = render_holdout_note(
        holdout_ids_seq, test_ids, citation_accuracy, class_prior
    )
    Path(holdout_note_path).parent.mkdir(parents=True, exist_ok=True)
    Path(holdout_note_path).write_text(holdout_text, encoding="utf-8")

    measured = citation_accuracy is not None and class_prior is not None
    return measured and not is_at_prior
