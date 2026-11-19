# -*- coding: utf-8 -*-
"""Dựng manifests/ucf_train.csv từ Anomaly_Train.txt (1610 video: 800 normal + 810 anomalous).

    python scripts/build_train_manifest.py \
        --train-list datasets/ucf_crime/Anomaly_Train.txt \
        --test-manifest manifests/ucf_test.csv \
        --out manifests/ucf_train.csv

Khác scripts/build_manifest.py: file này KHÔNG probe video. Tập train chưa nằm trên mọi máy, còn
danh sách chính thức thì có. `n_frames`/`fps` để trống và được điền khi video thật có mặt.

Luật chống rò:
- `video_id` nào đã có trong manifest test -> SystemExit, in ĐÚNG id đó, không ghi file.
- Trùng id trong chính file train -> SystemExit.
- Số dòng khác --expect (mặc định 1610) -> SystemExit, không ghi file.
- `path` suy từ entry (`Abuse/Abuse001_x264.mp4` -> `datasets/ucf_crime/train/Abuse/Abuse001_x264.mp4`),
  tương đối theo gốc repo. Không đường dẫn tuyệt đối, không `..`.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FIELDS = ["video_id", "path", "dataset", "split", "n_frames", "fps"]
TRAIN_PREFIX = "datasets/ucf_crime/train"


def read_train_entries(path: Path) -> list[tuple[str, str]]:
    """Mỗi dòng 'Abuse/Abuse001_x264.mp4' -> (video_id, path tương đối). Bỏ dòng trống."""
    out: list[tuple[str, str]] = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            tok = line.strip().replace("\\", "/").split()
            if not tok:
                continue
            entry = tok[0]
            if entry.startswith("/") or ".." in entry.split("/"):
                raise ValueError(f"entry không an toàn (tuyệt đối hoặc '..'): {entry!r}")
            vid = Path(entry).stem
            if not vid:
                raise ValueError(f"entry không có tên video: {entry!r}")
            out.append((vid, f"{TRAIN_PREFIX}/{entry}"))
    return out


def read_manifest_ids(path: Path) -> set[str]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {row["video_id"] for row in csv.DictReader(f)}


def build_rows(entries: list[tuple[str, str]], test_ids: set[str]) -> list[dict]:
    """Thành dòng manifest, hoặc raise ValueError nêu tên id vi phạm (trùng, hoặc nằm trong test)."""
    seen: set[str] = set()
    rows: list[dict] = []
    for vid, rel in entries:
        if vid in seen:
            raise ValueError(f"video_id trùng trong danh sách train: {vid}")
        if vid in test_ids:
            raise ValueError(f"video_id của tập test lọt vào train: {vid}")
        seen.add(vid)
        rows.append({
            "video_id": vid,
            "path": rel,
            "dataset": "ucf_crime",
            "split": "train",
            "n_frames": "",
            "fps": "",
        })
    return rows


def write_manifest(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-list", required=True, help="Anomaly_Train.txt, mỗi dòng '<lớp>/<tên>.mp4'")
    p.add_argument("--test-manifest", required=True, help="manifest test; mọi id ở đây bị cấm trong train")
    p.add_argument("--out", required=True)
    p.add_argument("--expect", type=int, default=1610)
    args = p.parse_args(argv)

    try:
        rows = build_rows(read_train_entries(Path(args.train_list)), read_manifest_ids(Path(args.test_manifest)))
    except (ValueError, OSError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    if len(rows) != args.expect:
        print(f"LỖI: mong đợi {args.expect} dòng, thực có {len(rows)}. Không ghi file.", file=sys.stderr)
        return 1
    write_manifest(rows, Path(args.out))
    print(f"ghi {len(rows)} dòng -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
