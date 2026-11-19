# -*- coding: utf-8 -*-
"""T2: dry run 2 ví dụ in số tham số và ghi adapter_config.json. Không import peft, không gắn weight."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.train_lora import main, two_examples
from vad.train.lora import (
    DRY_RUN_SHAPES,
    LANGUAGE_LAYERS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    TARGET_MODULES,
    LoRASpec,
    adapter_config,
    attach,
    dry_run,
    trainable_parameter_count,
)

ROOT = Path(__file__).resolve().parents[2]


def _module_level_imports(path: Path) -> set[str]:
    """Import ở thân module. Import trong hàm là lazy và không tính."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_import_does_not_pull_peft():
    assert "peft" not in sys.modules
    assert "vad.train.lora" in sys.modules
    assert "peft" not in _module_level_imports(ROOT / "vad" / "train" / "lora.py")
    assert "peft" not in _module_level_imports(ROOT / "scripts" / "train_lora.py")


def test_dry_run_writes_config_and_attaches_nothing(tmp_path: Path):
    spec = LoRASpec()
    out = tmp_path / "adapter"
    result = dry_run(two_examples(20261119), spec, out, DRY_RUN_SHAPES)
    assert result["attached"] is False
    assert result["base_frozen"] is True
    assert result["quantization"] == "8bit"
    assert result["examples"] == 2
    written = list(out.iterdir())
    assert [p.name for p in written] == ["adapter_config.json"]
    saved = json.loads(written[0].read_text(encoding="utf-8"))
    assert saved == adapter_config(spec)
    assert saved["r"] == LORA_R == 16
    assert saved["lora_alpha"] == LORA_ALPHA == 32
    assert saved["lora_dropout"] == LORA_DROPOUT == 0.05
    assert saved["target_modules"] == list(TARGET_MODULES)
    assert "vision" not in "".join(saved["target_modules"])
    assert result["trainable_parameters"] == trainable_parameter_count(spec, DRY_RUN_SHAPES)
    assert result["trainable_parameters"] > 0


def test_trainable_count_scales_with_rank_and_layers():
    shapes = dict(DRY_RUN_SHAPES)
    full = trainable_parameter_count(LoRASpec(r=16), shapes)
    assert full == 2 * trainable_parameter_count(LoRASpec(r=8), shapes)
    assert full == LANGUAGE_LAYERS * trainable_parameter_count(LoRASpec(), shapes, layers=1)
    assert LANGUAGE_LAYERS == 36


def test_real_run_refuses_without_smoke_note(tmp_path: Path):
    missing = tmp_path / "qwen3-smoke.md"
    with pytest.raises(FileNotFoundError, match="qwen3-smoke.md"):
        attach(LoRASpec(), missing)
    assert "peft" not in sys.modules


def test_cli_dry_run_prints_count_and_refuses_real_run(tmp_path: Path):
    out = tmp_path / "adapter"
    assert main(["--dry-run", "--out-dir", str(out)]) == 0
    assert (out / "adapter_config.json").is_file()
    code = main(["--run", "--out-dir", str(out), "--smoke-note", str(tmp_path / "absent.md")])
    assert code == 1
    assert "peft" not in sys.modules
