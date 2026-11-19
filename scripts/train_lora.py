# -*- coding: utf-8 -*-
"""Dry run LoRA trên 2 ví dụ RAFT. Không tải model, không cần GPU, không import peft.

    python scripts/train_lora.py --dry-run --out-dir /tmp/lora-dry

Vòng thật (`--run`) từ chối khởi động khi thiếu `notes/qwen3-smoke.md`. File đó là log P2,
chưa có cho tới khi smoke 8-bit thoát 0. Dry run không đọc file đó.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vad.index.siglip import build  # noqa: E402
from vad.train.lora import DRY_RUN_SHAPES, LoRASpec, attach, dry_run  # noqa: E402
from vad.train.raft import Clip, Query, build_dataset  # noqa: E402

import numpy as np  # noqa: E402


def two_examples(seed: int) -> list[dict[str, object]]:
    """Đúng 2 ví dụ đi qua builder T1. Một lớp, hai video, để P=1 có golden."""
    catalog = {"A0": Clip("A0", "Abuse"), "A1": Clip("A1", "Abuse")}
    vecs = {
        "A0": np.array([1.0, 0.0], dtype=np.float32),
        "A1": np.array([0.0, 1.0], dtype=np.float32),
    }
    index = build(vecs, "google/siglip2-base-patch16-384", 1.0, test_ids=set())
    queries = tuple(
        Query(vid, 0.0, 8.0, (f"frames/{vid}.jpg",), "Abuse", vecs[vid]) for vid in ("A0", "A1")
    )
    return list(build_dataset(queries, index, catalog, p=1.0, seed=seed))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="2 ví dụ, ghi adapter_config.json, không tải model")
    mode.add_argument("--run", action="store_true", help="vòng thật; từ chối nếu thiếu notes/qwen3-smoke.md")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=20261119)
    p.add_argument("--smoke-note", default="notes/qwen3-smoke.md")
    args = p.parse_args(argv)

    spec = LoRASpec()
    if args.dry_run:
        examples = two_examples(args.seed)
        result = dry_run(examples, spec, Path(args.out_dir), DRY_RUN_SHAPES)
        print(f"trainable_parameters={result['trainable_parameters']}")
        print(f"adapter_config={result['adapter_config']}")
        print("attached=false")
        return 0

    # --run: kiểm smoke trước, rồi mới import peft bên trong attach.
    try:
        attach(spec, Path(args.smoke_note))
    except FileNotFoundError as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    print("LỖI: --run chưa được phép ở Phase 2 (dừng trước 1610)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
