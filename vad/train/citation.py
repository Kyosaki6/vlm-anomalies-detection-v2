# -*- coding: utf-8 -*-
"""Đo citation accuracy trên holdout (T4). Không tải model, không import peft.

`score_holdout` nhận một hàm sinh câu. Hàm đó là chỗ duy nhất đụng GPU, và nó nằm
ngoài module này để CI chạy được. Câu đúng là câu có đúng `video_id` của clip golden.
Class prior là tỷ lệ lớp đông nhất trong các ví dụ có golden: adapter bỏ qua clip
trích dẫn và đoán lớp đông nhất sẽ ra đúng bằng số này.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

from vad.train.raft import ABSTAIN_TEXT, Clip, Query, build_dataset
from vad.index.siglip import SiglipIndex


def prompt_text(example: dict[str, object]) -> str:
    """Câu user đưa vào model. Liệt kê clip đã truy xuất, không ghi vai trò.

    Vai trò `golden` trong prompt biến bài trích dẫn thành bài đọc nhãn. Train và
    chấm điểm dùng cùng hàm này, nếu không số đo không phải là việc model đã học.
    Adapter lưu ngày 2026-11-25 không thấy câu này, nên không dùng để chấm.
    """
    rows = example.get("retrieved") or []
    lines = [f"{row.get('rank', i + 1)}. {row.get('video_id', '')}" for i, row in enumerate(rows)]
    listing = "\n".join(lines) if lines else "(none)"
    return "Name the event and cite the clip, or say Normal.\nRetrieved clips:\n" + listing


def class_prior(examples: Sequence[dict[str, object]]) -> float:
    """Tỷ lệ lớp đông nhất trên các ví dụ có golden. Không có ví dụ nào -> ValueError."""
    counts: Counter[str] = Counter()
    for ex in examples:
        if not ex.get("golden_present"):
            continue
        target = str(ex.get("target_text", ""))
        cls = target.split(",", 1)[0].strip()
        if cls and cls != ABSTAIN_TEXT:
            counts[cls] += 1
    total = sum(counts.values())
    if total == 0:
        raise ValueError("không có ví dụ golden nào để tính class prior")
    return counts.most_common(1)[0][1] / total


def score_holdout(
    examples: Sequence[dict[str, object]],
    generate_fn: Callable[[dict[str, object]], str],
) -> dict[str, float]:
    """Chấm citation accuracy. Đúng khi câu sinh chứa đúng video_id của clip golden.

    Chỉ ví dụ `golden_present` vào mẫu số. Ví dụ abstain tính riêng ở `abstain_rate`:
    câu sinh chứa `Normal`. Không có ví dụ golden -> ValueError, không bịa tỷ lệ.
    """
    if not examples:
        raise ValueError("examples rỗng")
    golden = [ex for ex in examples if ex.get("golden_present")]
    if not golden:
        raise ValueError("không có ví dụ golden nào để chấm citation accuracy")

    hits = 0
    abstain_hits = 0
    n_abstain = 0
    for ex in examples:
        text = generate_fn(ex)
        if ex.get("golden_present"):
            cited = str(ex["retrieved"][0]["video_id"])
            if cited and cited in text:
                hits += 1
        if ex.get("abstain"):
            n_abstain += 1
            if ABSTAIN_TEXT in text:
                abstain_hits += 1

    return {
        "citation_accuracy": hits / len(golden),
        "abstain_rate": (abstain_hits / n_abstain) if n_abstain else 0.0,
        "n_golden": float(len(golden)),
        "n_abstain": float(n_abstain),
        "class_prior": class_prior(examples),
    }


def holdout_examples(
    held: Sequence[str],
    full_catalog: dict[str, Clip],
    index_catalog: dict[str, Clip],
    index: SiglipIndex,
    paths: dict[str, str],
    embeddings: dict[str, object],
    p: float = 0.6,
    seed: int = 20261119,
) -> tuple[dict[str, object], ...]:
    """Dựng ví dụ RAFT cho holdout.

    `index_catalog` và `index` là phần train đã bỏ holdout, đúng như vòng train dùng.
    `full_catalog` giữ lớp của chính các id holdout, vì query cần lớp của nó.
    Holdout lọt vào index hoặc index_catalog -> ValueError. Thiếu embedding -> KeyError.
    """
    held_set = set(held)
    leaked = held_set & set(index.video_ids)
    if leaked:
        raise ValueError(f"holdout lọt vào index: {sorted(leaked)[:3]}")
    leaked = held_set & set(index_catalog)
    if leaked:
        raise ValueError(f"holdout lọt vào catalog train: {sorted(leaked)[:3]}")

    queries = tuple(
        Query(vid, 0.0, 8.0, (paths[vid],), full_catalog[vid].event_class, embeddings[vid])
        for vid in held
    )
    return build_dataset(queries, index, index_catalog, p=p, seed=seed)
