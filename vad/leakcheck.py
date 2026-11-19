# -*- coding: utf-8 -*-
"""Chặn rò video test vào tập train hoặc vào index RAFT.

Một video test nằm trong train, hoặc trong index retrieval, làm AUC nhìn cao hơn thực tế và
không có cách phát hiện sau khi số đã tính. Hàm này chạy TRƯỚC mọi bước tính điểm.

`check` raise LeakError nêu tên đúng video_id vi phạm. Không sửa file, không bỏ qua.
"""
from __future__ import annotations

import csv
from pathlib import Path


class LeakError(ValueError):
    """Một video_id của tập test xuất hiện ở nơi nó không được phép."""


def read_ids(path: str | Path) -> set[str]:
    """Đọc cột `video_id` từ một CSV manifest. Thiếu cột -> ValueError."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "video_id" not in reader.fieldnames:
            raise ValueError(f"{path}: thiếu cột 'video_id'")
        return {row["video_id"] for row in reader if row.get("video_id")}


def check(test_ids: set[str], train_ids: set[str], index_ids: set[str] | None = None) -> None:
    """Raise LeakError nếu test giao với train, hoặc giao với index. Thông báo nêu tên id."""
    leaked_train = sorted(test_ids & train_ids)
    if leaked_train:
        raise LeakError(
            f"{len(leaked_train)} video test nằm trong train, ví dụ: {leaked_train[0]}"
        )
    if index_ids is not None:
        leaked_index = sorted(test_ids & index_ids)
        if leaked_index:
            raise LeakError(
                f"{len(leaked_index)} video test nằm trong index, ví dụ: {leaked_index[0]}"
            )
