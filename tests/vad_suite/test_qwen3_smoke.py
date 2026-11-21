# -*- coding: utf-8 -*-
"""P2: smoke chỉ nạp model khi mục gate mới nhất nói go đúng ngày hôm nay.

Không tải model, không đụng GPU. Cửa sổ thật nằm trên GB10; ở đây chỉ khoá thứ mà một
máy CPU có thể làm sai: đọc nhầm gate, hoặc nhận quantization khác 8-bit.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from qwen3_smoke import gate_allows, main, read_gate  # noqa: E402

TODAY = "2026-11-19"


def _gate(tmp_path: Path, sections: list[tuple[str, str, str]]) -> Path:
    body = ["# gate\n"]
    for day, verdict, timestamp in sections:
        body.append(
            f"\n## {day} {verdict}\n\n"
            f"| Field | Value |\n|---|---|\n"
            f"| Timestamp | {timestamp} |\n"
            f"| Verdict | **{verdict}** |\n"
        )
    path = tmp_path / "gate.md"
    path.write_text("".join(body), encoding="utf-8")
    return path


def test_gate_allows_only_same_day_go(tmp_path):
    """Go của ngày khác, hoặc no-go mới hơn một go cùng ngày, đều không cho chạy."""
    stale = _gate(tmp_path, [("2026-11-18", "go", "2026-11-18T21:00:00+07:00")])
    assert gate_allows(stale, TODAY) is False

    overridden = _gate(tmp_path, [
        ("2026-11-19", "go", "2026-11-19T09:00:00+07:00"),
        ("2026-11-19", "no-go", "2026-11-19T21:25:19+07:00"),
    ])
    assert gate_allows(overridden, TODAY) is False

    fresh = _gate(tmp_path, [("2026-11-19", "go", "2026-11-19T22:00:00+07:00")])
    assert gate_allows(fresh, TODAY) is True


def test_gate_without_verdict_raises(tmp_path):
    """Gate hỏng phải raise. Coi như no-go sẽ nuốt lỗi và smoke im lặng không chạy."""
    path = tmp_path / "gate.md"
    path.write_text("# gate\n\n## 2026-11-19\n\nkhông có bảng\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Verdict"):
        read_gate(path, TODAY)


def test_main_refuses_to_load_when_gate_says_no(tmp_path, monkeypatch):
    """No-go: thoát khác 0 và không ghi note. Gọi _load_model lúc này là sai."""
    gate = _gate(tmp_path, [("2026-11-19", "no-go", "2026-11-19T21:25:19+07:00")])
    out = tmp_path / "smoke.md"

    def _boom(self):
        raise AssertionError("no-go mà vẫn nạp model")

    monkeypatch.setattr("vad.verify.vlm.VLMVerifier._load_model", _boom)
    code = main(["--gate", str(gate), "--out", str(out), "--video", "x.mp4"])
    assert code != 0
    assert not out.exists()


def test_main_rejects_quantization_other_than_8bit(tmp_path):
    """4-bit không phải lối thoát khi OOM. Lệnh phải từ chối trước khi đọc gate."""
    gate = _gate(tmp_path, [("2026-11-19", "go", "2026-11-19T22:00:00+07:00")])
    code = main(["--quantization", "4bit", "--gate", str(gate), "--out", str(tmp_path / "s.md")])
    assert code != 0


def test_dry_run_exits_zero_without_reading_gate(tmp_path):
    """Dry run không có gate, không có note. Nó chỉ in kế hoạch."""
    missing = tmp_path / "khong-ton-tai.md"
    code = main(["--dry-run", "--gate", str(missing), "--out", str(tmp_path / "s.md")])
    assert code == 0
    assert not (tmp_path / "s.md").exists()
