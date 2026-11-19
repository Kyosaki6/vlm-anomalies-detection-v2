# -*- coding: utf-8 -*-
"""N2: truy vấn không thấy chính video của nó, và id test cấy vào chỉ mục thì raise nêu tên."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.build_raft_index import main
from vad.embed.cache import save_embeddings
from vad.index.siglip import build, load, save
from vad.leakcheck import LeakError

PLANTED = "Abuse028_x264"
MODEL = "google/siglip2-base-patch16-384"
FPS = 1.0
DIM = 8


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(DIM).astype(np.float32)


def _frames(seed: int, n: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, DIM)).astype(np.float32)


def _manifest(path, ids: list[str], split: str) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "path", "dataset", "split"])
        w.writeheader()
        w.writerows({"video_id": v, "path": f"datasets/{v}.mp4", "dataset": "ucf_crime", "split": split}
                    for v in ids)


def test_query_video_is_absent_from_its_own_hits():
    ids = ["Abuse001_x264", "Abuse002_x264", "Robbery045_x264"]
    index = build({v: _vec(i) for i, v in enumerate(ids)}, MODEL, FPS, test_ids=set())
    query = index.vectors[index.video_ids.index("Abuse001_x264")]
    hits = index.search(query, k=3, exclude_video_id="Abuse001_x264")
    assert {h.video_id for h in hits} == {"Abuse002_x264", "Robbery045_x264"}


def test_search_ranks_by_cosine_not_by_id():
    """Id sắp xếp alphabet khác thứ tự gần-xa, nên test này chết nếu search sort theo tên."""
    index = build(
        {"m_far": np.array([1.0, 0.0]), "a_near": np.array([0.0, 1.0])},
        MODEL, FPS, test_ids=set(),
    )
    hits = index.search(np.array([0.0, 1.0]), k=2, exclude_video_id="absent")
    assert [h.video_id for h in hits] == ["a_near", "m_far"]
    assert hits[0].cosine > hits[1].cosine


def test_planted_test_id_raises_and_names_it():
    with pytest.raises(LeakError, match=PLANTED):
        build({PLANTED: _vec(0), "Abuse001_x264": _vec(1)}, MODEL, FPS, test_ids={PLANTED})


def test_save_load_round_trip_keeps_model_fps_and_order(tmp_path):
    index = build({"b": _vec(1), "a": _vec(2)}, MODEL, FPS, test_ids=set())
    save(index, tmp_path)
    back = load(tmp_path)
    assert back.model_id == MODEL
    assert back.sample_fps == FPS
    assert back.video_ids == ("a", "b")
    assert np.allclose(back.vectors, index.vectors)


def test_builder_with_planted_cache_exits_nonzero_and_writes_nothing(tmp_path, capsys):
    cache = tmp_path / "cache"
    save_embeddings(_frames(0), MODEL, "Abuse001_x264", FPS, cache)
    save_embeddings(_frames(1), MODEL, PLANTED, FPS, cache)
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    _manifest(train, ["Abuse001_x264", PLANTED], "train")
    _manifest(test, [PLANTED], "test")
    out = tmp_path / "index"
    rc = main(["--train-manifest", str(train), "--test-manifest", str(test),
               "--cache-dir", str(cache), "--out", str(out), "--model", MODEL, "--fps", str(FPS)])
    assert rc == 1
    assert PLANTED in capsys.readouterr().err
    assert not out.exists()


def test_builder_writes_index_a_later_search_can_read(tmp_path):
    cache = tmp_path / "cache"
    save_embeddings(_frames(0), MODEL, "Abuse001_x264", FPS, cache)
    save_embeddings(_frames(1), MODEL, "Robbery045_x264", FPS, cache)
    train, test = tmp_path / "train.csv", tmp_path / "test.csv"
    _manifest(train, ["Abuse001_x264", "Robbery045_x264"], "train")
    _manifest(test, [PLANTED], "test")
    out = tmp_path / "index"
    rc = main(["--train-manifest", str(train), "--test-manifest", str(test),
               "--cache-dir", str(cache), "--out", str(out), "--model", MODEL, "--fps", str(FPS)])
    assert rc == 0
    index = load(out)
    assert index.model_id == MODEL
    assert index.sample_fps == FPS
    assert set(index.video_ids) == {"Abuse001_x264", "Robbery045_x264"}
