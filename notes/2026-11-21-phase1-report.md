# Phase 1 report

Status: done. Merged on `origin/main` at `8c004c3`. No GPU, no model download.

Source of the task list: `notes/2026-11-19-phase1-tasks.md`. Phase 2 records the entry condition as true at that commit, with `python -m pytest -q` at 749 passed.

## What landed

| ID | Owner | Commit | Result |
|---|---|---|---|
| F1 | Phước | `8c004c3` | Fork plan, task assignment, and the Phase 1 sheet committed. Source repo was not pushed to. |
| F2 | Phước | `e563bca` | `requirements-vlm.txt` has `transformers>=4.57.0`, `peft>=0.13.0`, `trl>=0.12.0`. `requirements-dev.txt` stays GPU-free. |
| P1 | Phước | `fc16640` | `vad/verify/vlm.py` accepts `quantization="8bit"` and the Qwen3-VL class. `"4bit"` and `"none"` unchanged. Anything else raises. |
| N1 | Nam | `abc985e` | `scripts/build_train_manifest.py` writes `manifests/ucf_train.csv`. Recounted 2026-11-21: 1610 rows, 0 overlap with `manifests/ucf_test.csv` (290 rows). |
| N3 | Nam | `52384b2` | RAFT example schema frozen at `tests/vad_suite/fixtures/raft_example.json`. Not edited since. |
| H1 | Thịnh | `59447cc` | `vad/leakcheck.py`. A test `video_id` in the train manifest or the index fails the run and names the id. |

## Bindings that closed

| Binding | State reached |
|---|---|
| B1 | F1 committed. F2, P1, N1, N3, H1 are commits after it. |
| B2 | `transformers>=4.57` landed in `e563bca`, before the Qwen3-VL class in `fc16640`. |
| B3 | H1 reads the real manifests. The planted-id half uses a fixture. |
| B4 | Schema JSON merged. T1 (Phase 2, `5edd275`) builds against it. |
| B5 | `"8bit"` unit test green with the model mocked. P2 calls that branch. |
| B6 | Leak check merged. Phase 3 eval imports it and exits before writing `R3.json` when a test id is planted. |

## What Phase 1 did not do

F3, P0, P2, P3, N2, N4, T1, T2, T3, T4, H2, H3, H4. Those start in Phase 2 or later.

## Open item

The sheet's F1 acceptance line asks for the fork URL in the PR and for `results/run_A4c/` hashes unchanged. The commits are on this repo. The URL itself is not in the notes.
