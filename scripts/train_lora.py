# -*- coding: utf-8 -*-
"""Dry run LoRA trên 2 ví dụ RAFT hoặc chạy huấn luyện qua loop.

    python scripts/train_lora.py --dry-run --out-dir /tmp/lora-dry
    python scripts/train_lora.py --run
    python scripts/train_lora.py --launch

Vòng thật (`--launch`) từ chối khởi động khi thiếu `notes/qwen3-smoke.md` hoặc gate hôm nay
chưa go. Vòng `--run` chạy qua injected loop, không tải model, ghi `notes/lora-train.md`.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vad.index.siglip import build  # noqa: E402
from vad.leakcheck import read_ids  # noqa: E402
from vad.train.gate import gate_is_go  # noqa: E402
from vad.train.holdout import holdout_ids, read_train_catalog  # noqa: E402
from vad.train.loop import train  # noqa: E402
from vad.train.lora import DRY_RUN_SHAPES, LoRASpec, attach, dry_run, require_smoke  # noqa: E402
from vad.train.notes import write_notes  # noqa: E402
from vad.train.raft import Clip, Query, build_dataset  # noqa: E402

import numpy as np  # noqa: E402


def two_examples(seed: int) -> list[dict[str, object]]:
    """Đúng 2 ví dụ đi qua builder T1. Một lớp, hai video, để P=1 có golden."""
    catalog = {"A0": Clip("A0", "Abuse"), "A1": Clip("A1", "Abuse")}
    vecs = {
        "A0": np.array([1.0, 0.0], dtype=np.float32),
        "A1": np.array([0.0, 1.0], dtype=np.float32),
    }
    index = build(vecs, "google/siglip2-base-patch16-384", 1.0, test_ids=set())
    queries = tuple(
        Query(vid, 0.0, 8.0, (f"frames/{vid}.jpg",), "Abuse", vecs[vid]) for vid in ("A0", "A1")
    )
    return list(build_dataset(queries, index, catalog, p=1.0, seed=seed))


def _video_paths(manifest: str) -> dict[str, str]:
    import csv
    out: dict[str, str] = {}
    with open(manifest, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("video_id") and row.get("path"):
                out[row["video_id"]] = row["path"]
    return out


def training_examples(manifest: str, seed: int):
    """Catalog trừ holdout, ví dụ RAFT ở p=0.6. Trả về (examples, holdout ids)."""
    catalog = read_train_catalog(manifest)
    paths = _video_paths(manifest)
    held = holdout_ids(catalog, fraction=0.1, seed=seed)
    train_catalog = {vid: clip for vid, clip in catalog.items() if vid not in set(held)}
    dim = min(len(train_catalog), 64)
    rng = np.random.default_rng(seed)
    vecs = {vid: rng.standard_normal(dim).astype(np.float32) for vid in train_catalog}
    index = build(vecs, "google/siglip2-base-patch16-384", 1.0, test_ids=set())
    queries = tuple(
        Query(vid, 0.0, 8.0, (paths.get(vid, f"frames/{vid}.jpg"),), clip.event_class, vecs[vid])
        for vid, clip in train_catalog.items()
    )
    examples = list(build_dataset(queries, index, train_catalog, p=0.6, seed=seed))
    return examples, held


def get_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _record(res: dict, spec: LoRASpec, held, args, citation_accuracy, class_prior) -> bool:
    """Ghi cả hai note bằng một hàm. Trả về True khi prior không chặn eval."""
    test_ids = read_ids(args.test_manifest) if Path(args.test_manifest).is_file() else set()
    return write_notes(
        run_info={
            "commit": get_commit_hash(),
            "config": f"r={spec.r}, alpha={spec.alpha}, dropout={spec.dropout}",
            "steps": res["steps"],
            "loss": res["loss"],
            "peak_memory": res.get("peak_memory", "pending-launch"),
            "wall_time": res.get("wall_time", "pending-launch"),
            "abstain_rate": res["abstain_rate"],
        },
        holdout_ids_seq=held,
        test_ids=test_ids,
        citation_accuracy=citation_accuracy,
        class_prior=class_prior,
        train_note_path=args.train_note,
        holdout_note_path=args.holdout_note,
    )


def main(argv=None, steps_fn=None, eval_fn=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="2 ví dụ, ghi adapter_config.json, không tải model")
    mode.add_argument("--run", action="store_true", help="vòng huấn luyện qua loop đã inject, ghi notes/lora-train.md")
    mode.add_argument("--launch", action="store_true", help="vòng thật trên GPU, yêu cầu gate hôm nay là go")
    p.add_argument("--out-dir", default="/tmp/lora-dry")
    p.add_argument("--seed", type=int, default=20261119)
    p.add_argument("--smoke-note", default="notes/qwen3-smoke.md")
    p.add_argument("--gate-note", default="notes/gb10-gate.md")
    p.add_argument("--train-manifest", default="manifests/ucf_train.csv")
    p.add_argument("--test-manifest", default="manifests/ucf_test.csv")
    p.add_argument("--train-note", default="notes/lora-train.md")
    p.add_argument("--holdout-note", default="notes/holdout.md")
    p.add_argument("--citation-accuracy", type=float, default=None)
    p.add_argument("--class-prior", type=float, default=None)
    args = p.parse_args(argv)

    spec = LoRASpec()
    if args.dry_run:
        examples = two_examples(args.seed)
        result = dry_run(examples, spec, Path(args.out_dir), DRY_RUN_SHAPES)
        print(f"trainable_parameters={result['trainable_parameters']}")
        print(f"adapter_config={result['adapter_config']}")
        print("attached=false")
        return 0

    if args.launch:
        try:
            require_smoke(Path(args.smoke_note))
        except FileNotFoundError as e:
            print(f"LỖI: {e}", file=sys.stderr)
            return 1

        if not gate_is_go(Path(args.gate_note), date.today()):
            print("LỖI: gate_is_go từ chối khởi chạy (cổng hôm nay chưa go)", file=sys.stderr)
            return 3

        examples, held = training_examples(args.train_manifest, args.seed)
        if steps_fn is not None:
            fn = steps_fn
        else:
            lora_config = attach(spec, Path(args.smoke_note))
            fn = _gpu_step(spec, lora_config, Path(args.out_dir))
        t0 = time.perf_counter()
        res = train(examples, spec, fn)
        save = getattr(fn, "save", None)
        if save is not None:
            save()
        res["wall_time"] = f"{time.perf_counter() - t0:.1f}s"
        peak = getattr(fn, "peak_mib", None)
        if peak:
            res["peak_memory"] = f"{peak:.0f} MiB"
        if not getattr(fn, "saved", False):
            _save_injected(fn, Path(args.out_dir))
        return _finish(res, spec, held, args, eval_fn)

    # --run
    try:
        require_smoke(Path(args.smoke_note))
    except FileNotFoundError as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1

    examples, held = training_examples(args.train_manifest, args.seed)
    fn = steps_fn if steps_fn is not None else (lambda ex: 0.1)
    res = train(examples, spec, fn)
    res["loss"] = "pending-launch"
    return _finish(res, spec, held, args, eval_fn)


def _finish(res, spec, held, args, eval_fn) -> int:
    measured = args.citation_accuracy is not None and args.class_prior is not None
    can_eval = _record(res, spec, held, args, args.citation_accuracy, args.class_prior)
    if measured and can_eval and eval_fn is not None:
        eval_fn()
        return 0
    if measured and not can_eval:
        print("DỪNG: adapter ở mức class prior, eval không chạy", file=sys.stderr)
        return 4
    return 0


def _save_injected(fn, out_dir: Path) -> None:
    """Ghi adapter khi bước tối ưu là hàm inject, không phải `_gpu_step`.

    Hàm inject không giữ model. Nó ghi trọng số bằng `fn.adapter`, một state_dict
    dựng sẵn cho test. Thiếu thì raise: im lặng là mất adapter lần nữa.
    """
    adapter = getattr(fn, "adapter", None)
    if not isinstance(adapter, dict) or not adapter:
        raise RuntimeError("launch xong mà không có adapter để ghi: fn.adapter rỗng")
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "adapter_model.json").write_text(
        json.dumps(adapter, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _gpu_step(spec: LoRASpec, lora_config, out_dir: Path):
    """Một bước tối ưu thật. Chỉ chạy sau khi gate go. Thiếu frame -> raise, không bịa loss.

    Nạp model cùng đường với `VLMVerifier._load_model` (8-bit, đúng class Qwen3-VL),
    gắn LoRA, rồi AdamW một bước trên ảnh thật của ví dụ. Không có file ảnh thì dừng.
    Hết vòng thì ghi adapter vào `out_dir`. Trọng số không được chỉ sống trong RAM.
    """
    import torch
    import peft
    from PIL import Image

    from vad.verify.vlm import VLMVerifier

    verifier = VLMVerifier(model_id=spec.base_model, quantization=spec.quantization)
    verifier._load_model()
    model = peft.get_peft_model(verifier._model, lora_config)
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    processor = verifier._processor

    def step(example: dict) -> float:
        from vad.io.video import iter_sampled_frames

        paths = example.get("query", {}).get("frame_paths") or []
        if not paths or not Path(paths[0]).is_file():
            raise FileNotFoundError(f"không có video để train: {paths[:1]}")
        frame = next(iter_sampled_frames(paths[0], sample_fps=1.0), None)
        if frame is None:
            raise RuntimeError(f"không lấy được khung hình từ {paths[0]}")
        image = Image.fromarray(frame[1])
        target = str(example.get("target_text", ""))
        user = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Name the event and cite the clip, or say Normal."},
            ],
        }
        full = processor.apply_chat_template(
            [user, {"role": "assistant", "content": target}],
            tokenize=False, add_generation_prompt=False,
        )
        prefix = processor.apply_chat_template(
            [user], tokenize=False, add_generation_prompt=True,
        )
        batch = processor(text=[full], images=[image], return_tensors="pt")
        prefix_ids = processor(text=[prefix], images=[image], return_tensors="pt")
        batch = {k: v.to(model.device) for k, v in batch.items()}
        n_prefix = prefix_ids["input_ids"].shape[1]
        if n_prefix >= batch["input_ids"].shape[1]:
            raise RuntimeError("prefix dài hơn cả chuỗi: loss sẽ không có token nào để học")
        labels = batch["input_ids"].clone()
        labels[:, :n_prefix] = -100
        batch["labels"] = labels
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step.peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
        value = float(loss.detach().cpu())
        print(f"step loss={value:.4f} peak_mib={step.peak_mib:.0f}", flush=True)
        return value

    def save() -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        step.saved = True
        print(f"saved adapter to {out_dir}", flush=True)

    step.save = save
    return step


if __name__ == "__main__":
    raise SystemExit(main())
