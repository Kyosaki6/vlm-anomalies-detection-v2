# -*- coding: utf-8 -*-
"""T1: P=0 toàn abstain, P=1 toàn golden, P=0.6 trên seed cố định khớp số lần golden đếm riêng."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vad.index.siglip import build
from vad.train.raft import TOP_K, Clip, Query, build_dataset

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "raft_example.json"
REQUIRED = {"schema_version", "query", "retrieved", "golden_present", "target_text", "abstain"}
MODEL = "google/siglip2-base-patch16-384"
SEED = 20261119
N = 200

CLASSES = ("Abuse", "Robbery", "Burglary", "Normal")


def _catalog() -> dict[str, Clip]:
    """Mỗi lớp 4 video, vector riêng, để lớp nào cũng có láng giềng cùng lớp."""
    return {f"{c}{i}": Clip(f"{c}{i}", c) for c in CLASSES for i in range(4)}


def _index(catalog: dict[str, Clip]):
    vecs = {vid: np.eye(len(catalog), dtype=np.float32)[n] for n, vid in enumerate(sorted(catalog))}
    return build(vecs, MODEL, 1.0, test_ids=set())


def _queries(catalog: dict[str, Clip], index) -> tuple[Query, ...]:
    out = []
    for vid in sorted(catalog):
        row = index.vectors[index.video_ids.index(vid)]
        out.append(Query(vid, 0.0, 8.0, (f"frames/{vid}.jpg",), catalog[vid].event_class, row))
    return tuple(out)


def _expected_golden(p: float, n: int, seed: int) -> int:
    """Đếm riêng bằng cùng seed, không đọc từ kết quả builder."""
    rng = np.random.default_rng(seed)
    return sum(bool(rng.random() < p) for _ in range(n))


def test_p0_abstains_everywhere():
    catalog = _catalog()
    index = _index(catalog)
    examples = build_dataset(_queries(catalog, index), index, catalog, p=0.0, seed=SEED)
    assert len(examples) == len(catalog)
    assert all(e["abstain"] is True and e["golden_present"] is False for e in examples)
    assert {e["target_text"] for e in examples} == {"Normal"}


def test_p1_is_all_golden_and_cites_a_same_class_clip():
    catalog = _catalog()
    index = _index(catalog)
    examples = build_dataset(_queries(catalog, index), index, catalog, p=1.0, seed=SEED)
    assert all(e["golden_present"] is True and e["abstain"] is False for e in examples)
    for e in examples:
        golden = [h for h in e["retrieved"] if h["role"] == "golden"]
        assert len(golden) == 1
        assert golden[0]["video_id"] != e["query"]["video_id"]
        assert catalog[golden[0]["video_id"]].event_class == catalog[e["query"]["video_id"]].event_class


def test_p06_golden_count_matches_an_independent_draw():
    """N query, mỗi query đúng một video cùng lớp khác id. Lặp catalog 16 clip làm builder
    abstain vì không còn láng giềng mới, nên số golden lệch khỏi số lần tung đồng xu."""
    catalog = {f"C{c}_{i}": Clip(f"C{c}_{i}", f"C{c}") for c in range(N // 2) for i in range(2)}
    index = _index(catalog)
    queries = _queries(catalog, index)
    assert len(queries) == N
    examples = build_dataset(queries, index, catalog, p=0.6, seed=SEED)
    got = sum(e["golden_present"] is True for e in examples)
    assert got == _expected_golden(0.6, len(queries), SEED)


def test_example_validates_against_the_frozen_fixture_fields():
    catalog = _catalog()
    index = _index(catalog)
    example = build_dataset(_queries(catalog, index), index, catalog, p=1.0, seed=SEED)[0]
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert set(example) == set(fixture) == REQUIRED
    assert example["schema_version"] == fixture["schema_version"] == 1
    assert set(example["query"]) == set(fixture["query"])
    assert set(example["retrieved"][0]) == set(fixture["retrieved"][0])


def test_query_is_absent_from_its_own_retrieved_list():
    catalog = _catalog()
    index = _index(catalog)
    for e in build_dataset(_queries(catalog, index), index, catalog, p=0.6, seed=SEED):
        assert e["query"]["video_id"] not in [h["video_id"] for h in e["retrieved"]]


def test_no_same_class_neighbour_abstains_even_at_p1():
    """Chỉ một video trong lớp -> không có gì để trích -> abstain, không bịa golden."""
    catalog = {"Abuse0": Clip("Abuse0", "Abuse"), "Robbery0": Clip("Robbery0", "Robbery")}
    index = _index(catalog)
    q = _queries(catalog, index)
    examples = build_dataset(q, index, catalog, p=1.0, seed=SEED)
    assert all(e["abstain"] is True for e in examples)


def test_k_other_than_default_limits_the_list():
    catalog = _catalog()
    index = _index(catalog)
    examples = build_dataset(_queries(catalog, index), index, catalog, p=0.0, seed=SEED, k=2)
    assert TOP_K != 2
    assert {len(e["retrieved"]) for e in examples} == {2}


def test_p_outside_unit_interval_raises():
    catalog = _catalog()
    index = _index(catalog)
    with pytest.raises(ValueError, match="p"):
        build_dataset(_queries(catalog, index), index, catalog, p=1.5, seed=SEED)
