# -*- coding: utf-8 -*-
"""Smoke P2: nạp Qwen3-VL-8B-Instruct 8-bit, verify đúng một cửa sổ, ghi log.

    python scripts/qwen3_smoke.py --quantization 8bit --out notes/qwen3-smoke.md

Chưa có GPU ở đây, và `notes/gb10-gate.md` nói no-go. Script này không được tải model
trong lúc đó: nó đọc gate trước, và chỉ gọi `_load_model` khi mục mới nhất là `go`
cùng ngày. Không có nhánh 4-bit. OOM được ghi vào note và thoát khác 0, vì T2 chỉ
kiểm file tồn tại: một note ghi OOM mà thoát 0 sẽ cho vòng train chạy trên model chết.

`--dry-run` in kế hoạch và thoát 0. Không đọc gate, không import torch, không ghi note.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vad.verify.vlm import VLMVerifier  # noqa: E402

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_QUANTIZATION = "8bit"
DEFAULT_GATE = Path("notes/gb10-gate.md")
DEFAULT_K = 4
LOCAL_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

_HEADING = re.compile(r"^## (.+)$", re.MULTILINE)
_FIELD = re.compile(r"^\| (Timestamp|Verdict) \| (.+) \|$", re.MULTILINE)


def read_gate(gate_path: Path, today: str) -> tuple[str, str]:
    """Mục mới nhất của gate -> (ngày ISO, verdict viết thường).

    Gate không có mục, hoặc mục mới nhất không ghi verdict, là lỗi: im lặng coi như
    no-go sẽ nuốt một file gate hỏng và cho smoke chạy.
    """
    text = gate_path.read_text(encoding="utf-8-sig")
    headings = list(_HEADING.finditer(text))
    if not headings:
        raise ValueError(f"{gate_path} không có mục '## ...'")
    last = headings[-1]
    section = text[last.end():]
    found = {name: value.strip() for name, value in _FIELD.findall(section)}
    if "Verdict" not in found or "Timestamp" not in found:
        raise ValueError(f"mục mới nhất của {gate_path} thiếu Timestamp hoặc Verdict")
    verdict = found["Verdict"].strip("*").strip().lower()
    if verdict not in ("go", "no-go"):
        raise ValueError(f"verdict không đọc được: {found['Verdict']!r}")
    return found["Timestamp"][:10], verdict


def gate_allows(gate_path: Path, today: str) -> bool:
    """True chỉ khi mục mới nhất nói `go` và timestamp đúng ngày chạy."""
    day, verdict = read_gate(gate_path, today)
    return verdict == "go" and day == today


def git_commit() -> str:
    """Commit HEAD. Không phải repo git -> chuỗi rỗng, không raise.

    `SMOKE_COMMIT` cho máy chạy smoke không có `.git`: người chạy điền commit
    thật của repo nguồn, thay vì để note ghi rỗng như thể không có commit.
    """
    override = os.environ.get("SMOKE_COMMIT", "").strip()
    if override:
        return override
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.stdout.strip()


def driver_version() -> str:
    """Dòng `Driver Version` của nvidia-smi. Không có card -> chuỗi rỗng."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.stdout.strip().splitlines()[0].strip()


def package_version(name: str) -> str:
    """Version một package đã cài. Chưa cài -> 'missing'."""
    try:
        from importlib.metadata import version
        return version(name)
    except importlib_metadata_error():
        return "missing"


def importlib_metadata_error() -> tuple[type[Exception], ...]:
    from importlib.metadata import PackageNotFoundError
    return (PackageNotFoundError,)


def render_note(fields: dict[str, str]) -> str:
    """Log P2. Các khoá là hợp đồng mà T2 và Phase 3 trích, không phải văn xuôi."""
    rows = "\n".join(f"| {key} | {fields[key]} |" for key in (
        "timestamp", "model", "quantization", "peak_memory_mib", "seconds",
        "bitsandbytes", "transformers", "driver", "commit", "forced", "exit",
    ))
    detail = fields.get("detail", "")
    body = f"\n\n{detail}\n" if detail else "\n"
    return f"# Qwen3-VL smoke (P2)\n\n| Field | Value |\n|---|---|\n{rows}\n{body}"


def write_note(path: Path, fields: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(render_note(fields), encoding="utf-8")
    tmp.replace(path)


def run_smoke(verifier: VLMVerifier, frames: list, context: dict) -> tuple[float, float]:
    """Nạp model, verify một cửa sổ. -> (giây, peak MiB). Không bắt lỗi: caller ghi OOM.

    Giây đếm riêng phần verify. Lần chạy đầu còn tải weight, và cộng nó vào sẽ
    biến thời gian model thành thời gian mạng.
    """
    import torch

    verifier._load_model()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    verifier.verify(frames, context)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, torch.cuda.max_memory_allocated() / (1024 * 1024)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quantization", default=DEFAULT_QUANTIZATION)
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--out", default="notes/qwen3-smoke.md")
    p.add_argument("--gate", default=str(DEFAULT_GATE))
    p.add_argument("--video", default=None, help="một video; bỏ trống thì không có cửa sổ để verify")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="bỏ qua gate. Chỉ khi người giữ máy chủ động ra lệnh; note ghi lại")
    args = p.parse_args(argv)

    if args.quantization != DEFAULT_QUANTIZATION:
        print(
            f"LỖI: P2 chỉ chạy {DEFAULT_QUANTIZATION}. Nhận {args.quantization!r}. "
            "OOM thì dừng phase, không xuống 4-bit.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        print(f"dry-run: {args.model} {args.quantization}, k={args.k}, out={args.out}")
        return 0

    today = datetime.now(LOCAL_TZ).date().isoformat()
    try:
        allowed = gate_allows(Path(args.gate), today)
    except (OSError, ValueError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    if not allowed and not args.force:
        print(
            f"LỖI: {args.gate} chưa cho go trong ngày {today}. P2 không tải model.",
            file=sys.stderr,
        )
        return 1
    if args.force and not allowed:
        print(
            f"CẢNH BÁO: {args.gate} chưa cho go, chạy vì --force. GPU đang có tiến trình khác.",
            file=sys.stderr,
        )

    if args.video is None:
        print("LỖI: gate đã cho go nhưng thiếu --video, không có cửa sổ để verify.", file=sys.stderr)
        return 1

    from scripts.qwen3_smoke_frames import load_window

    try:
        frames, context = load_window(Path(args.video), args.k)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1

    verifier = VLMVerifier(model_id=args.model, quantization=args.quantization)
    fields = {
        "timestamp": datetime.now(LOCAL_TZ).isoformat(timespec="seconds"),
        "model": args.model,
        "quantization": args.quantization,
        "bitsandbytes": package_version("bitsandbytes"),
        "transformers": package_version("transformers"),
        "driver": driver_version(),
        "commit": git_commit(),
        "forced": "yes" if args.force else "no",
    }
    try:
        seconds, peak = run_smoke(verifier, frames, context)
    except torch_out_of_memory() as e:
        fields.update(peak_memory_mib="OOM", seconds="", exit="oom", detail=f"OOM: {e}")
        write_note(Path(args.out), fields)
        print(f"LỖI: hết bộ nhớ, ghi {args.out}. Phase 2 dừng, không chuyển 4-bit.", file=sys.stderr)
        return 1

    fields.update(peak_memory_mib=f"{peak:.0f}", seconds=f"{seconds:.1f}", exit="0")
    write_note(Path(args.out), fields)
    print(f"xong: {seconds:.1f}s, peak {peak:.0f} MiB -> {args.out}")
    return 0


def torch_out_of_memory() -> tuple[type[Exception], ...]:
    """Import torch chỉ khi thật sự chạy. Dry run và gate no-go không đi tới đây."""
    import torch
    return (torch.cuda.OutOfMemoryError,)


if __name__ == "__main__":
    raise SystemExit(main())
