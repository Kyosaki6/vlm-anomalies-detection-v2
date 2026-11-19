# -*- coding: utf-8 -*-
"""Dựng chỉ mục RAFT từ cache SigLIP của tập train. Không tải model, không cần GPU.

    python scripts/build_raft_index.py \
        --train-manifest manifests/ucf_train.csv \
        --test-manifest manifests/ucf_test.csv \
        --cache-dir cache/embeddings \
        --out cache/raft_index

Mỗi video train -> một vector: trung bình các khung đã cache rồi chuẩn hoá L2. Truy vấn sau này loại
chính `video_id` của nó (`vad.index.siglip.SiglipIndex.search`).

Luật chống rò: một `video_id` test có mặt trong manifest train, hoặc có file cache riêng, thì
`LeakError`, in ĐÚNG id đó, và không ghi `--out`. Thiếu cache của một video train cũng không ghi.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vad.embed.cache import load_embeddings  # noqa: E402
from vad.embed.siglip import DEFAULT_MODEL_ID  # noqa: E402
from vad.index.siglip import build, save, window_embedding  # noqa: E402
from vad.leakcheck import LeakError, read_ids  # noqa: E402


def collect(train_ids: set[str], model_id: str, sample_fps: float, cache_dir: Path) -> dict[str, np.ndarray]:
    """Đọc cache từng video train. Thiếu file -> FileNotFoundError nêu tên video."""
    out: dict[str, np.ndarray] = {}
    for vid in sorted(train_ids):
        frames = load_embeddings(model_id, vid, sample_fps, cache_dir)
        out[vid] = window_embedding(frames)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--fps", type=float, default=1.0)
    args = p.parse_args(argv)

    test_ids = read_ids(Path(args.test_manifest))
    train_ids = read_ids(Path(args.train_manifest))
    try:
        vectors = collect(train_ids, args.model, args.fps, Path(args.cache_dir))
        index = build(vectors, args.model, args.fps, test_ids)
    except (LeakError, FileNotFoundError, ValueError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    save(index, Path(args.out))
    print(f"ghi {len(index.video_ids)} video -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
