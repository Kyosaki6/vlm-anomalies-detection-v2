# -*- coding: utf-8 -*-
"""N3: fixture RAFT đúng hình dạng đã chốt. Đổi schema mà không sửa fixture -> test này đỏ."""
from __future__ import annotations

import json
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "raft_example.json"

REQUIRED = {"schema_version", "query", "retrieved", "golden_present", "target_text", "abstain"}
QUERY_REQUIRED = {"video_id", "start_s", "end_s", "frame_paths"}
HIT_REQUIRED = {"video_id", "rank", "role", "cosine"}


def test_fixture_has_the_frozen_fields():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert REQUIRED <= set(data)
    assert data["schema_version"] == 1
    assert QUERY_REQUIRED <= set(data["query"])
    assert data["query"]["frame_paths"], "query phải có ít nhất một frame"
    assert data["retrieved"], "retrieved không được rỗng trong fixture mẫu"
    for hit in data["retrieved"]:
        assert HIT_REQUIRED <= set(hit)
        assert hit["role"] in ("golden", "distractor")
    roles = [h["role"] for h in data["retrieved"]]
    if data["golden_present"]:
        assert "golden" in roles
        assert data["abstain"] is False
    else:
        assert "golden" not in roles
        assert data["abstain"] is True
    ids = [h["video_id"] for h in data["retrieved"]]
    assert data["query"]["video_id"] not in ids
