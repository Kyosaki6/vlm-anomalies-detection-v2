# -*- coding: utf-8 -*-
"""N4: cache train nạp bằng đúng loader, model, fps, dtype của cache test.

Chưa có cache train thật. Test dựng file .npy nhỏ trong tmp và kiểm cái mà N4 có thể làm
sai mà không cần GPU: shard chạy dở bị tính là xong, hoặc dtype trên đĩa lệch cache test.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from check_train_cache import CacheCheck, check, main  # noqa: E402
from vad.embed.cache import save_embeddings  # noqa: E402

MODEL = "google/siglip2-base-patch16-384"
FPS = 1.0


def _spec(train_ids: list[str], train_cache: Path, test_cache: Path, n_shards: int) -> CacheCheck:
    return CacheCheck(tuple(train_ids), ("ref",), MODEL, FPS, train_cache, test_cache, n_shards)


def _write(cache: Path, video_id: str, dtype: str = "float16") -> None:
    embs = np.ones((2, 4), dtype=np.float32)
    save_embeddings(embs, MODEL, video_id, FPS, cache)
    if dtype != "float16":
        from vad.embed.cache import cache_path
        path = cache_path(MODEL, video_id, FPS, cache)
        np.save(path, np.ones((2, 4), dtype=np.float32))


def _manifest(path: Path, video_ids: list[str]) -> None:
    rows = ["video_id,path,dataset,split,n_frames,fps\n"]
    rows += [f"{v},{v}.mp4,ucf_crime,train,,\n" for v in video_ids]
    path.write_text("".join(rows), encoding="utf-8")


def test_finished_shard_loads_with_test_dtype(tmp_path):
    """Shard đủ cache thì đếm đúng số video và nạp được. Shard thiếu thì bị loại, không đếm."""
    test_cache, train_cache = tmp_path / "test", tmp_path / "train"
    _write(test_cache, "ref")
    for vid in ("a", "b"):
        _write(train_cache, vid)
    # "c" thuộc shard còn lại và cố tình không có cache.

    # SHA-256 đặt a,b vào shard 2/3 và c vào shard 3/3. Cố định ở đây vì test dựa vào nó.
    report = check(_spec(["a", "b", "c"], train_cache, test_cache, n_shards=3))
    assert report["count"] == 2
    assert report["dtype"] == "float16"
    assert report["model_id"] == MODEL
    assert report["fps"] == FPS
    assert len(report["partial_shards"]) == 1


def test_no_finished_shard_raises(tmp_path):
    """Cache train trống không được báo count=0. Nó là chưa chạy."""
    test_cache, train_cache = tmp_path / "test", tmp_path / "train"
    _write(test_cache, "ref")
    with pytest.raises(ValueError, match="shard"):
        check(_spec(["a"], train_cache, test_cache, n_shards=1))


def test_dtype_mismatch_with_test_cache_raises(tmp_path):
    """File train ghi float32 trong khi cache test là float16 thì raise, nêu đúng video."""
    test_cache, train_cache = tmp_path / "test", tmp_path / "train"
    _write(test_cache, "ref")
    _write(train_cache, "a", dtype="float32")
    with pytest.raises(ValueError, match="a"):
        check(_spec(["a"], train_cache, test_cache, n_shards=1))


def test_main_reports_count_for_finished_shard(tmp_path, capsys):
    """Đi từ manifest CSV, không gọi check() trực tiếp, để khoá cả phần đọc file."""
    train_manifest, test_manifest = tmp_path / "train.csv", tmp_path / "test.csv"
    _manifest(train_manifest, ["a", "b"])
    _manifest(test_manifest, ["ref"])
    test_cache, train_cache = tmp_path / "test", tmp_path / "train"
    _write(test_cache, "ref")
    _write(train_cache, "a")
    _write(train_cache, "b")

    code = main([
        "--train-manifest", str(train_manifest),
        "--test-manifest", str(test_manifest),
        "--train-cache", str(train_cache),
        "--test-cache", str(test_cache),
        "--model", MODEL,
        "--fps", str(FPS),
    ])
    assert code == 0
    assert "count=2" in capsys.readouterr().out
