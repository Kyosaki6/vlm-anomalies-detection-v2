# -*- coding: utf-8 -*-
"""Cấu hình LoRA cho base Qwen3-VL 8-bit đóng băng. Dry run không tải model, không import peft.

Vòng thật (`attach`) mới import `peft` và `bitsandbytes`. CI không cài hai gói đó
(`requirements-vlm.txt`, không nằm trong `requirements-dev.txt`), nên import module này
phải rẻ và không được kéo chúng theo.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# T2: q, k, v, o của attention và gate, up, down của MLP. Vision tower không nằm trong danh sách.
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
QUANTIZATION = "8bit"
BATCH_SIZE = 1
SMOKE_NOTE = Path("notes/qwen3-smoke.md")

# Qwen3-VL-8B: 36 lớp ngôn ngữ, hidden 4096, kv 1024, MLP intermediate 12288. Dry run không tải model.
LANGUAGE_LAYERS = 36
DRY_RUN_SHAPES: dict[str, tuple[int, int]] = {
    "q_proj": (4096, 4096),
    "k_proj": (4096, 1024),
    "v_proj": (4096, 1024),
    "o_proj": (4096, 4096),
    "gate_proj": (4096, 12288),
    "up_proj": (4096, 12288),
    "down_proj": (12288, 4096),
}


@dataclass(frozen=True, slots=True)
class LoRASpec:
    """Thông số đóng băng. Đổi một số là đổi thí nghiệm, không phải đổi mặc định trong code gọi."""

    base_model: str = BASE_MODEL_ID
    quantization: str = QUANTIZATION
    r: int = LORA_R
    alpha: int = LORA_ALPHA
    dropout: float = LORA_DROPOUT
    target_modules: tuple[str, ...] = TARGET_MODULES
    vision_tower_frozen: bool = True
    batch_size: int = BATCH_SIZE

    def __post_init__(self) -> None:
        if self.quantization != "8bit":
            raise ValueError(f"quantization phải là '8bit', nhận {self.quantization!r}")
        if self.r < 1 or self.alpha < 1:
            raise ValueError(f"r và alpha phải >= 1, nhận r={self.r} alpha={self.alpha}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout phải nằm trong [0, 1), nhận {self.dropout}")
        if self.batch_size != 1:
            raise ValueError(f"batch_size phải là 1, nhận {self.batch_size}")
        if not self.vision_tower_frozen:
            raise ValueError("vision tower phải đóng băng")
        if not self.target_modules:
            raise ValueError("target_modules rỗng")


def adapter_config(spec: LoRASpec) -> dict[str, object]:
    """Nội dung `adapter_config.json`. Không có trọng số, không có state_dict."""
    return {
        "base_model_name_or_path": spec.base_model,
        "quantization": spec.quantization,
        "peft_type": "LORA",
        "r": spec.r,
        "lora_alpha": spec.alpha,
        "lora_dropout": spec.dropout,
        "target_modules": list(spec.target_modules),
        "modules_to_save": [],
        "vision_tower_frozen": spec.vision_tower_frozen,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
    }


def trainable_parameter_count(spec: LoRASpec, module_shapes: dict[str, tuple[int, int]],
                              layers: int = LANGUAGE_LAYERS) -> int:
    """Đếm tham số LoRA sẽ train. Không đụng base, không import peft.

    `module_shapes` map tên module -> (in_features, out_features). Mỗi module: A là (r, in),
    B là (out, r). Module không nằm trong `target_modules` không được đếm. Thiếu một module
    đích -> KeyError, không bịa shape.
    """
    if set(module_shapes) != set(spec.target_modules):
        missing = [m for m in spec.target_modules if m not in module_shapes]
        extra = [m for m in module_shapes if m not in spec.target_modules]
        raise KeyError(f"module_shapes lệch target: thiếu {missing}, thừa {extra}")
    if layers < 1:
        raise ValueError(f"layers phải >= 1, nhận {layers}")
    total = 0
    for name in spec.target_modules:
        in_f, out_f = module_shapes[name]
        if in_f < 1 or out_f < 1:
            raise ValueError(f"{name}: shape phải dương, nhận {(in_f, out_f)}")
        total += spec.r * in_f + out_f * spec.r
    return total * layers


def write_adapter_config(spec: LoRASpec, out_dir: Path) -> Path:
    """Ghi `adapter_config.json`. Không ghi trọng số, không tải model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "adapter_config.json"
    path.write_text(
        json.dumps(adapter_config(spec), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def dry_run(examples: list[dict[str, object]], spec: LoRASpec, out_dir: Path,
            module_shapes: dict[str, tuple[int, int]]) -> dict[str, object]:
    """Hai ví dụ RAFT, không model. Trả về số tham số train được và đường dẫn config.

    `module_shapes` là shape base đã biết (dry run không tải model nên người gọi đưa vào).
    Không import `peft`. Không ghi gì ngoài `adapter_config.json`.
    """
    if len(examples) != 2:
        raise ValueError(f"dry run nhận đúng 2 ví dụ, nhận {len(examples)}")
    for i, ex in enumerate(examples):
        if ex.get("schema_version") != 1 or "abstain" not in ex or "query" not in ex:
            raise ValueError(f"ví dụ {i} không phải RAFT schema 1")
    path = write_adapter_config(spec, out_dir)
    written = sorted(out_dir.iterdir())
    if written != [path]:
        raise RuntimeError(f"dry run chỉ được ghi adapter_config.json, thấy {written}")
    count = trainable_parameter_count(spec, module_shapes)
    return {
        "trainable_parameters": count,
        "adapter_config": str(path),
        "attached": False,
        "examples": len(examples),
        "base_frozen": True,
        "quantization": spec.quantization,
    }


def require_smoke(note: Path = SMOKE_NOTE) -> None:
    """Vòng thật không được bắt đầu khi chưa có log smoke P2."""
    if not note.is_file():
        raise FileNotFoundError(
            f"thiếu {note}: vòng train thật cần smoke 8-bit exit 0 (P2). Dry run không cần file này."
        )


def attach(spec: LoRASpec, note: Path = SMOKE_NOTE):
    """Gắn LoRA lên base 8-bit. Import peft chỉ ở đây. Chưa có smoke -> không tải gì."""
    require_smoke(note)
    import peft  # noqa: WPS433  — chỉ vòng thật, CI không đi tới dòng này

    return peft.LoraConfig(
        r=spec.r,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        target_modules=list(spec.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
