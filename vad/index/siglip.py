# -*- coding: utf-8 -*-
"""Chỉ mục SigLIP cho RAFT: chỉ video train, truy vấn loại chính video của nó.

Một clip truy ra chính nó thì cosine = 1 và citation trở nên vô nghĩa. Một video test nằm trong
chỉ mục làm AUC nhìn cao hơn thực và không phát hiện được sau khi số đã tính. `build` gọi
`vad.leakcheck.check` và raise trước khi ghi; `search` bỏ `exclude_video_id` khỏi kết quả.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vad.leakcheck import LeakError, check

INDEX_NAME = "index.json"
VECTORS_NAME = "vectors.npy"


@dataclass(frozen=True, slots=True)
class Hit:
    """Một video láng giềng. `cosine` là trung bình cosine trên các khung cửa sổ."""

    video_id: str
    cosine: float


@dataclass(frozen=True, slots=True)
class SiglipIndex:
    """Vector đã chuẩn hoá L2, một hàng một video, cùng thứ tự với `video_ids`."""

    model_id: str
    sample_fps: float
    video_ids: tuple[str, ...]
    vectors: np.ndarray

    def search(self, embedding: np.ndarray, k: int, exclude_video_id: str) -> tuple[Hit, ...]:
        """Top-k video gần `embedding` nhất, bỏ `exclude_video_id`. k < 1 hoặc rỗng -> ()."""
        if k < 1 or not self.video_ids:
            return ()
        query = np.asarray(embedding, dtype=np.float32).ravel()
        query = query / (np.linalg.norm(query) + 1e-12)
        scores = self.vectors @ query
        order = np.argsort(-scores, kind="stable")
        hits: list[Hit] = []
        for i in order:
            vid = self.video_ids[int(i)]
            if vid == exclude_video_id:
                continue
            hits.append(Hit(video_id=vid, cosine=float(scores[int(i)])))
            if len(hits) == k:
                break
        return tuple(hits)


def window_embedding(frame_embs: np.ndarray) -> np.ndarray:
    """(T, D) -> (D,) trung bình rồi chuẩn hoá L2. T = 0 thì raise."""
    embs = np.asarray(frame_embs, dtype=np.float32)
    if embs.ndim != 2 or embs.shape[0] == 0:
        raise ValueError(f"frame_embs phải là (T, D) với T > 0, nhận shape {embs.shape}")
    mean = embs.mean(axis=0)
    return mean / (np.linalg.norm(mean) + 1e-12)


def build(vectors_by_id: dict[str, np.ndarray], model_id: str, sample_fps: float,
          test_ids: set[str]) -> SiglipIndex:
    """Dựng chỉ mục. `vectors_by_id` rỗng, hoặc chứa một id test, thì raise và nêu tên id."""
    if not vectors_by_id:
        raise ValueError("chỉ mục rỗng: không có video nào để ghi")
    ids = tuple(sorted(vectors_by_id))
    check(test_ids, set(), set(ids))
    rows = [np.asarray(vectors_by_id[v], dtype=np.float32).ravel() for v in ids]
    dim = rows[0].shape[0]
    bad = [ids[i] for i, r in enumerate(rows) if r.shape[0] != dim]
    if bad:
        raise ValueError(f"vector lệch chiều {dim}, ví dụ: {bad[0]}")
    mat = np.stack(rows)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return SiglipIndex(model_id=model_id, sample_fps=sample_fps, video_ids=ids, vectors=mat)


def save(index: SiglipIndex, directory: str | Path) -> Path:
    """Ghi `index.json` (id, model, fps) + `vectors.npy` (float32). Trả về thư mục."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / VECTORS_NAME, index.vectors.astype(np.float32))
    payload = {
        "model_id": index.model_id,
        "sample_fps": index.sample_fps,
        "video_ids": list(index.video_ids),
    }
    (out / INDEX_NAME).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load(directory: str | Path) -> SiglipIndex:
    """Đọc lại đúng cặp file `save` đã ghi. Thiếu file hoặc lệch số dòng -> raise."""
    src = Path(directory)
    meta_path, vec_path = src / INDEX_NAME, src / VECTORS_NAME
    if not meta_path.is_file() or not vec_path.is_file():
        raise FileNotFoundError(f"chỉ mục thiếu {INDEX_NAME} hoặc {VECTORS_NAME}: {src}")
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    ids = tuple(meta["video_ids"])
    vectors = np.load(vec_path).astype(np.float32)
    if vectors.shape[0] != len(ids):
        raise ValueError(f"{src}: {len(ids)} video_id nhưng vectors có {vectors.shape[0]} hàng")
    return SiglipIndex(model_id=str(meta["model_id"]), sample_fps=float(meta["sample_fps"]),
                       video_ids=ids, vectors=vectors)
