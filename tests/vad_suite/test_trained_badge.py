# -*- coding: utf-8 -*-
"""H3: Replay hiện TRAINED đúng khi metadata.trained là true, và không hiện khi không phải."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))

from trained import TRAINED_BADGE, is_trained, trained_badge  # noqa: E402

from vad.results import RunResult, make_metadata  # noqa: E402


def _run(**meta) -> RunResult:
    return RunResult(video_id="v1", sample_fps=1.0, anomaly_scores=(0.1,),
                     metadata=make_metadata(**meta))


def test_trained_run_badges_trained():
    run = _run(trained=True, adapter_id="lora-r16")
    assert is_trained(run)
    assert trained_badge(run) == TRAINED_BADGE


def test_untrained_run_badges_nothing():
    assert trained_badge(_run()) == ""
    assert not is_trained(_run(trained=False))
    assert not is_trained(_run(trained="true"))
