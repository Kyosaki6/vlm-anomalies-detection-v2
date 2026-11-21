# -*- coding: utf-8 -*-
"""Kiểm cache N4: tập train nạp được bằng đúng loader, model, fps của cache test.

    python scripts/check_train_cache.py \
        --train-manifest manifests/ucf_train.csv \
        --test-cache cache/embeddings --train-cache cache/embeddings \
        --model google/siglip2-base-patch16-384 --fps 1

N4 chưa chạy thì cache train trống và lệnh này thoát khác 0. Đó là trạng thái đúng, không
phải lỗi cần vá. Đếm theo shard đã xong: shard nào còn thiếu video thì bị loại khỏi số đếm
và in tên. Một shard thiếu một phần mà vẫn được cộng vào sẽ báo "xong" khi mới chạy được nửa.

Cache ghi float16 ở mọi dtype tính toán (`save_embeddings` ép kiểu trước khi ghi), nên dtype
trên đĩa là một giá trị duy nhất. Lệnh này đọc dtype đó từ file test đầu tiên và đòi mọi
file train trùng nó.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.precompute_embeddings import read_manifest  # noqa: E402
from vad.datasets.sharding import parse_shard_spec, select_shard  # noqa: E402
from vad.embed.cache import cache_path, load_embeddings  # noqa: E402
from vad.embed.siglip import DEFAULT_MODEL_ID  # noqa: E402

DEFAULT_FPS = 1.0


@dataclass(frozen=True)
class CacheCheck:
    """Một lần đối chiếu cache train với cache test. Mọi thứ quyết định đường dẫn file."""

    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    model_id: str
    sample_fps: float
    train_cache: Path
    test_cache: Path
    n_shards: int


def on_disk_dtype(spec: CacheCheck, video_id: str, cache_dir: Path) -> str:
    """Dtype thật trên đĩa. `load_embeddings` tự cast float32 nên không dùng được ở đây."""
    path = cache_path(spec.model_id, video_id, spec.sample_fps, cache_dir)
    return str(np.load(path, mmap_mode="r").dtype)


def finished_shards(spec: CacheCheck) -> tuple[list[str], list[str]]:
    """-> (shard đã đủ cache, shard còn thiếu). Shard thiếu được nêu tên, không đếm."""
    done: list[str] = []
    partial: list[str] = []
    for index in range(spec.n_shards):
        mine = select_shard(list(spec.train_ids), index, spec.n_shards)
        missing = [
            v for v in mine
            if not cache_path(spec.model_id, v, spec.sample_fps, spec.train_cache).is_file()
        ]
        label = f"{index + 1}/{spec.n_shards}"
        if missing:
            partial.append(f"{label}: thiếu {len(missing)}/{len(mine)} (ví dụ {missing[0]})")
        elif mine:
            done.append(label)
    return done, partial


def check(spec: CacheCheck) -> dict[str, object]:
    """Đối chiếu cache train với cache test. Raise khi không có gì để đối chiếu."""
    if not spec.test_ids:
        raise ValueError("manifest test rỗng, không có cache để đối chiếu")
    test_dtype = on_disk_dtype(spec, spec.test_ids[0], spec.test_cache)
    done, partial = finished_shards(spec)
    if not done:
        detail = "; ".join(partial) or "không có shard nào"
        raise ValueError(f"chưa có shard train nào chạy xong: {detail}")

    finished_ids = [
        v for index in range(spec.n_shards)
        if f"{index + 1}/{spec.n_shards}" in done
        for v in select_shard(list(spec.train_ids), index, spec.n_shards)
    ]
    mismatched = [
        v for v in finished_ids
        if on_disk_dtype(spec, v, spec.train_cache) != test_dtype
    ]
    if mismatched:
        raise ValueError(
            f"dtype trên đĩa khác cache test ({test_dtype}): {mismatched[0]} và {len(mismatched) - 1} video nữa"
        )
    for video_id in finished_ids:
        load_embeddings(spec.model_id, video_id, spec.sample_fps, spec.train_cache)

    return {
        "model_id": spec.model_id,
        "fps": spec.sample_fps,
        "dtype": test_dtype,
        "finished_shards": done,
        "partial_shards": partial,
        "count": len(finished_ids),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--test-cache", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p.add_argument("--shards", default="1/1", help="N trong i/N; shard nào chưa đủ thì bị loại khỏi số đếm")
    args = p.parse_args(argv)

    _, n_shards = parse_shard_spec(args.shards if "/" in args.shards else f"1/{args.shards}")
    spec = CacheCheck(
        train_ids=tuple(r.video_id for r in read_manifest(args.train_manifest)),
        test_ids=tuple(r.video_id for r in read_manifest(args.test_manifest)),
        model_id=args.model, sample_fps=args.fps,
        train_cache=Path(args.train_cache), test_cache=Path(args.test_cache),
        n_shards=n_shards,
    )
    try:
        report = check(spec)
    except (OSError, ValueError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1

    print(
        f"model={report['model_id']} fps={report['fps']} dtype={report['dtype']} "
        f"count={report['count']} shards={','.join(report['finished_shards'])}"
    )
    for item in report["partial_shards"]:
        print(f"bỏ qua: {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
