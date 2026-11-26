# Phase 2 report

Status: done, with one recorded deviation. All eight tasks landed. `python -m pytest -q` was 786 passed when the phase closed (`notes/2026-11-19-phase2-tasks.md`). Entry was Phase 1 at `8c004c3`, 749 passed.

GPU rule held for the gate readings: one process at a time, no resident job killed, no polling loop. The deviation is P2, below.

## What landed

| ID | Owner | Where | Result |
|---|---|---|---|
| P0 | Thịnh | `notes/gb10-gate.md`, `b2a964c` | Read-only gate. First two readings no-go: host timeout, then util 95%, 71 C, PIDs 548193 / 572109 / 572949. A no-go counts as a finished task. |
| N2 | Nam | `d6fa85b` | `vad/index/siglip.py`, `scripts/build_raft_index.py`. Search drops the query `video_id`. A test id in the index raises and names it. |
| T1 | Thanh | `5edd275` | `vad/train/raft.py`. P=0.6 golden, else distractors and `abstain=true`. Fixed seed. Output validates against the frozen fixture. |
| H3 | Thịnh | `4101a8b` | Dashboard metadata: `trained`, `adapter_id`, `base_model`, `quantization`, `raft_hits`. `schema_version` stays 1. Replay shows `TRAINED`. |
| P3 | Phước | `b2a964c` | `notes/gb10-access.md`. Host, port, key holder, how a job dies with the ssh session. |
| T2 | Thanh | `9d5f6cb` | `vad/train/lora.py`, `scripts/train_lora.py`. Frozen 8-bit base, LoRA r=16, alpha=32, dropout 0.05. Dry run on 2 examples writes `adapter_config.json` and prints the trainable count. `--run` refused without the smoke note. |
| P2 | Phước, launched by Thịnh | `notes/qwen3-smoke.md`, recorded in `13bb87b` | `Qwen/Qwen3-VL-8B-Instruct`, 8-bit. Peak 10065 MiB, 4.7 s, bitsandbytes 0.50.2, transformers 5.22.2, driver 580.178.04, commit `5666838`, exit 0. |
| N4 | Nam, launched by Thịnh | `notes/n4-embed.md`, recorded in `13bb87b` | `google/siglip2-base-patch16-384`, fps 1, float16. 1550 embedded this run (417137 frames, 97.1 min, 0 failures), 60 already cached. `scripts/check_train_cache.py` exit 0, `count=1610 shards=1/1 dtype=float16`. |

## The deviation

P2 ran on 2026-11-20 12:02+07:00 with `--force` while the latest gate section was still the 2026-11-19 **no-go**. The note says `forced=yes`. Binding B7 says a go is valid only on the day it was written, and there was no go that day yet. The 2026-11-20 **go** (17:57+07:00, util 0%, 48 C, no compute PIDs) was written after the smoke, and it covers the download only. A later reading the same evening was **no-go for the N4 embedding run** (util 96%, PIDs 1043889 / 1044115 / 1044550). N4 then ran 2026-11-21, 13:55 to 15:32+07:00, and exited 0.

No switch to 4-bit. No OOM.

## Bindings that closed

| Binding | State reached |
|---|---|
| B7 | Broken by P2 (`forced=yes` against a no-go). Met by N4 only in the sense that the run happened the next day and exited 0. The same-day gate check was not in the launch path yet. Phase 3 adds it. |
| B8 | `search(emb, k, exclude_video_id)` merged. The query id is absent from its own hits. |
| B9 | `tests/vad_suite/fixtures/raft_example.json` unchanged. T1 validates against it. |
| B10 | `vad/train/raft.py` importable, seed test green. T2's dry run builds two examples through it. |
| B11 | Launch command is in `notes/gb10-access.md`. |
| B12 | `"8bit"` unit test green before the smoke. |
| B13 | `notes/qwen3-smoke.md` has peak memory and exit 0. The real train run still needs it. The dry run does not. |
| B14 | Train cache loads with the same loader, model id, fps, and dtype as the test cache. |
| B15 | A trained sample JSON round-trips and renders `TRAINED`. `schema_version` 2 is still rejected. |

## What Phase 2 did not do

T3, T4, H2, H4, F3. Training on the 1610 is Phase 3. F3 is a README paragraph and rides with H4 in Phase 4.
