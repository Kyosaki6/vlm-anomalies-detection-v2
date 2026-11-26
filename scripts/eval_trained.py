# -*- coding: utf-8 -*-
"""Đánh giá mô hình đã huấn luyện (R3) và so sánh với A6 (GAP-3, GAP-5).

    python scripts/eval_trained.py --a6-file results/ablation_A6.json --out results/trained/R3.json

Quy tắc:
1. Chạy leak check H1 trước: nếu video test có trong train manifest hoặc index -> thoát lỗi ngay.
2. Đọc A6 từ results/ablation_A6.json (không gõ tay số liệu).
3. Đọc tau từ A6, khẳng định round(tau, 4) == 1.1686, không nhận tau từ tham số đầu vào.
4. Ghi kết quả gồm 6 chỉ số: micro AUC, video bootstrap CI, FA/hour, window precision,
   incident recall, và paired delta so với A6.
5. R0, R1, R2 ghi rõ {"status": "not measured"}.
6. Không import peft ở module level.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from vad.eval.metrics import (  # noqa: E402
    false_alarms_per_hour,
    frame_level_auc,
    video_bootstrap_auc_ci,
)
from vad.leakcheck import LeakError, check, read_ids  # noqa: E402
from vad.train.verdict import eval_allowed  # noqa: E402

HOLDOUT_REFUSED = 4


def _metrics_from_scorer(raw: dict[str, Any], a6_auc: float, n_bootstraps: int, seed: int) -> dict[str, float]:
    """Scorer trả điểm thô. AUC, CI và FA/hour tính ở đây, không nhận số đã tính sẵn."""
    scores = [np.asarray(s, dtype=np.float64) for s in raw["scores_per_video"]]
    labels = [np.asarray(y, dtype=bool) for y in raw["labels_per_video"]]
    if len(scores) != len(labels):
        raise ValueError("scores_per_video và labels_per_video lệch số video")
    auc = frame_level_auc(scores, labels)
    ci_lo, ci_hi = video_bootstrap_auc_ci(scores, labels, n_bootstraps=n_bootstraps, seed=seed)
    fa = false_alarms_per_hour(int(raw["n_false_alarms"]), float(raw["normal_seconds"]))
    n_windows = int(raw["n_windows"])
    incidents = int(raw["incidents"])
    if n_windows < 1 or incidents < 1:
        raise ValueError(f"n_windows và incidents phải >= 1, nhận {n_windows}, {incidents}")
    return {
        "auc_micro": float(auc),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "fa_per_hour": float(fa),
        "window_precision": int(raw["true_positive_windows"]) / n_windows,
        "incident_recall": int(raw["incidents_detected"]) / incidents,
        "delta_auc": float(auc) - a6_auc,
    }


def main(argv=None, scorer_fn: Callable[[list[str], float], dict[str, Any]] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-manifest", default="manifests/ucf_test.csv")
    p.add_argument("--train-manifest", default="manifests/ucf_train.csv")
    p.add_argument("--index-manifest", default=None)
    p.add_argument("--a6-file", default="results/ablation_A6.json")
    p.add_argument("--out", default="results/trained/R3.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bootstraps", type=int, default=1000)
    p.add_argument("--holdout-note", default="notes/holdout.md")
    args = p.parse_args(argv)

    # 1. Leak check H1
    test_ids = read_ids(args.test_manifest)
    train_ids = read_ids(args.train_manifest)
    index_ids = read_ids(args.index_manifest) if args.index_manifest and Path(args.index_manifest).is_file() else None

    try:
        check(test_ids=test_ids, train_ids=train_ids, index_ids=index_ids)
    except LeakError as e:
        print(f"LỖI RÒ RỈ DỮ LIỆU (LeakError): {e}", file=sys.stderr)
        return 1

    # 2. Cổng holdout. False -> thoát trước khi đọc A6 hay gọi scorer, không ghi file.
    if not eval_allowed(args.holdout_note):
        print(
            f"HOLDOUT REFUSED: {args.holdout_note} không vượt class prior hơn 0.01. "
            "Eval không chạy.",
            file=sys.stderr,
        )
        return HOLDOUT_REFUSED

    # 3. Đọc file A6
    a6_path = Path(args.a6_file)
    if not a6_path.is_file():
        print(f"LỖI: không thấy A6 file {a6_path}", file=sys.stderr)
        return 1

    a6_data = json.loads(a6_path.read_text(encoding="utf-8-sig"))

    tau = a6_data.get("threshold")
    if tau is None:
        print("LỖI: file A6 không có trường 'threshold'", file=sys.stderr)
        return 1

    assert round(float(tau), 4) == 1.1686, f"tau phải bằng 1.1686, nhận {tau}"

    # 3. Chấm điểm qua scorer_fn được inject
    if scorer_fn is None:
        raise NotImplementedError("Mô hình thật cần GPU, hãy inject scorer_fn hoặc chạy có model")

    raw = scorer_fn(list(test_ids), float(tau))
    if not isinstance(raw, dict):
        raise ValueError("scorer_fn phải trả về dict điểm thô")
    metrics = _metrics_from_scorer(raw, float(a6_data["auc_micro"]), args.n_bootstraps, args.seed)

    result = {
        "ablation": "R3",
        "threshold": tau,
        "auc_micro": metrics["auc_micro"],
        "ci_lo": metrics["ci_lo"],
        "ci_hi": metrics["ci_hi"],
        "fa_per_hour": metrics["fa_per_hour"],
        "window_precision": metrics["window_precision"],
        "incident_recall": metrics["incident_recall"],
        "delta_auc": metrics["delta_auc"],
        "a6": a6_data,
        "R0": {"status": "not measured"},
        "R1": {"status": "not measured"},
        "R2": {"status": "not measured"},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote R3 eval results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
