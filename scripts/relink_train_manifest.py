# -*- coding: utf-8 -*-
"""Sửa cột `path` của manifests/ucf_train.csv cho khớp file thật trên đĩa.

`build_train_manifest.py` đoán path từ entry của Anomaly_Train.txt
(`datasets/ucf_crime/train/Abuse/Abuse001_x264.mp4`). Mirror HF không giữ cấu trúc đó:
video nằm ở `datasets/ucf_crime/train/Anomaly-Videos-Part-N/<lớp>/<tên>.mp4`.
`precompute_embeddings.py` mở đúng cột `path`, nên manifest để nguyên sẽ không thấy file nào.

    python scripts/relink_train_manifest.py \
        --manifest manifests/ucf_train.csv --root datasets/ucf_crime/train

Chỉ sửa `path`. `video_id`, `split`, thứ tự dòng giữ nguyên. Mỗi video_id phải khớp ĐÚNG MỘT
file `<video_id>.mp4` dưới --root; 0 hoặc >1 khớp thì dừng và không ghi gì.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

FIELDS = ["video_id", "path", "dataset", "split", "n_frames", "fps"]


def index_videos(root: Path) -> dict[str, list[Path]]:
    """video_id -> mọi file `<video_id>.mp4` dưới root. Không chọn hộ."""
    found: dict[str, list[Path]] = {}
    for p in sorted(root.rglob("*.mp4")):
        found.setdefault(p.stem, []).append(p)
    return found


def relink(rows: list[dict], found: dict[str, list[Path]], root: Path) -> list[dict]:
    """Trả dòng mới với `path` trỏ tới file thật. Raise ValueError nêu tên id không khớp đúng 1 file."""
    out: list[dict] = []
    for row in rows:
        vid = row["video_id"]
        hits = found.get(vid, [])
        if len(hits) != 1:
            raise ValueError(f"{vid}: {len(hits)} file khớp dưới {root} (cần đúng 1)")
        fixed = dict(row)
        fixed["path"] = hits[0].as_posix()
        out.append(fixed)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", required=True, help="thư mục chứa video train đã tải")
    p.add_argument("--out", default=None, help="mặc định: ghi đè --manifest")
    args = p.parse_args(argv)

    manifest = Path(args.manifest)
    root = Path(args.root)
    with open(manifest, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    try:
        fixed = relink(rows, index_videos(root), root)
    except ValueError as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1

    dest = Path(args.out) if args.out else manifest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(fixed)
    tmp.replace(dest)
    changed = sum(a["path"] != b["path"] for a, b in zip(rows, fixed))
    print(f"sửa {changed}/{len(fixed)} path -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
