# -*- coding: utf-8 -*-
"""score_holdout chấm citation accuracy. Mỗi ví dụ chỉ sinh một lần."""
from __future__ import annotations

import pytest

from vad.train.citation import class_prior, prompt_text, score_holdout
from vad.train.raft import ABSTAIN_TEXT


def _ex(vid: str, retrieved: str, golden: bool, abstain: bool, target: str) -> dict:
    return {
        "query": {"video_id": vid, "frame_paths": [f"{vid}.mp4"]},
        "retrieved": [{"video_id": retrieved, "rank": 1, "role": "golden" if golden else "distractor"}],
        "golden_present": golden,
        "abstain": abstain,
        "target_text": target,
    }


def test_score_holdout_generates_once_per_example():
    examples = (
        _ex("q1", "g1", golden=True, abstain=False, target="Abuse, cited from g1"),
        _ex("q2", "d2", golden=False, abstain=True, target=ABSTAIN_TEXT),
        _ex("q3", "g3", golden=True, abstain=False, target="Arrest, cited from g3"),
    )
    seen: list[str] = []

    def generate(ex: dict) -> str:
        seen.append(ex["query"]["video_id"])
        cited = ex["retrieved"][0]["video_id"]
        if ex["abstain"]:
            return ABSTAIN_TEXT
        if ex["query"]["video_id"] == "q3":
            return "no citation here"
        return f"Abuse, cited from {cited}"

    scored = score_holdout(examples, generate)

    assert seen == ["q1", "q2", "q3"]
    assert scored["citation_accuracy"] == pytest.approx(0.5)
    assert scored["n_golden"] == 2
    assert scored["n_abstain"] == 1
    assert scored["abstain_rate"] == pytest.approx(1.0)
    assert scored["class_prior"] == pytest.approx(0.5)


def test_score_holdout_empty_and_no_golden_raise():
    with pytest.raises(ValueError):
        score_holdout((), lambda ex: "")
    abstain_only = (_ex("q", "d", golden=False, abstain=True, target=ABSTAIN_TEXT),)
    with pytest.raises(ValueError):
        score_holdout(abstain_only, lambda ex: ABSTAIN_TEXT)


def test_class_prior_uses_majority_golden_class():
    examples = (
        _ex("a", "g", True, False, "Abuse, cited from g"),
        _ex("b", "g", True, False, "Abuse, cited from g"),
        _ex("c", "g", True, False, "Arrest, cited from g"),
        _ex("d", "d", False, True, ABSTAIN_TEXT),
    )
    assert class_prior(examples) == pytest.approx(2 / 3)


def test_prompt_text_lists_clips_without_role():
    ex = {
        "retrieved": [
            {"rank": 1, "video_id": "g1", "role": "golden"},
            {"rank": 2, "video_id": "d2", "role": "distractor"},
        ]
    }
    text = prompt_text(ex)
    assert "g1" in text and "d2" in text
    assert "golden" not in text
    assert "role" not in text
