# vad/

**Parent:** `../AGENTS.md`. Library only. No CLI, no Streamlit, no model download at import.

## OVERVIEW
Training-free tier-1 scoring plus a real tier-2 VLM verifier. Subpackages do not import each other; `pipeline.py` is the only composer. `types.py` is the import hub.

## STRUCTURE
```
types.py          EventType, VideoRef, Window, VerifyResult, Alert
interfaces.py     Embedder, Verifier, AlertPolicy (Protocols)
fakes.py          FakeEmbedder, FakeVerifier — tests and --fake only
results.py        RunResult <-> results/<video_id>.json + windows/<id>.json
pipeline.py       PipelineConfig + ABLATIONS (A0..A5). CPU. No GPU.
alerting.py       re-export only (graph.alerts + merge_alerts); no logic here
datasets/         ucf_crime.py (GT), sharding.py (SHA-256 shard)
io/               video.py — sequential grab, sample_indices, probe_video
embed/            cache.py (fp16 .npy, atomic) + siglip.py (one class, two models)
score/            text.py (prompt + memory, numpy) + typing.py (13-class cosine)
temporal/         ops.py — robust_normalize -> top-k -> ema|median -> windows
verify/           vlm.py — parse_verifier_json, VLMVerifier (4-bit, raises)
graph/            policy.py (YAML -> NetworkX) + alerts.py (cooldown, feed)
eval/             metrics.py — frame AUC, video bootstrap CI, FA/hour, latency
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Change an ablation switch | `pipeline.py` `PipelineConfig` / `ABLATIONS` | A0..A5 differ only in config, not code paths |
| Detection score math | `score/text.py` | numpy only; logit scale is a real parameter |
| Window cutting | `temporal/ops.py` `candidate_windows` | gap 5 s; do not retune inside a script |
| Verdict JSON | `verify/vlm.py` `parse_verifier_json` | strict; garbage in raises, never a guessed bool |
| Alert cooldown | `graph/alerts.py` | keyed by `Alert.camera`, falls back to video_id |
| Cache key | `embed/cache.py` `cache_path` | model + video_id + sample_fps |

## CONVENTIONS
- `from __future__ import annotations`. No `StrEnum`. `EventType(str, Enum)` — Python 3.10.
- Public surface is the submodule. `vad/__init__.py` exports nothing.
- GPU backends (`SigLIPEmbedder`, `VLMVerifier`) lazy-load weights. Import must stay cheap.
- Missing backend raises `NotImplementedError` or `RuntimeError`. Never a synthetic score.

## ANTI-PATTERNS
- Do not import `streamlit`, `argparse`, or a script from here.
- Do not put class names in detection prompts. `score/typing.py` is the only place class prompts live.
- Do not return a verdict when JSON parse fails. `VLMVerifier` retries once, then raises.
- `use_memory=True` requires `per_video_norm=True` (`pipeline.py`). Text probability and memory cosine distance are not on one scale.
- Do not build a memory bank that contains the video being scored. `build_memory_bank` does not exclude it; the caller must. Distance collapses to 0.
- Do not read `vad/README.md` module status. It still says the dashboard is fake and `run_benchmark.py` does not exist.
- Do not add alert logic to `alerting.py`. It re-exports `graph.alerts`; `test_alerting.py` locks both import paths to the same functions.

## NOTES
- Temporal order is fixed: `robust_normalize` -> `window_topk_pool` -> `ema` or `median_smooth`. Same input + same `PipelineConfig` = same scores.
- `embed/siglip.py`: `padding="max_length", max_length=64` is SigLIP-only. CLIP B/32 uses 77 tokens and breaks on that padding.
- `results.py` drops unknown JSON fields and refuses a `video_id` that is a path.
