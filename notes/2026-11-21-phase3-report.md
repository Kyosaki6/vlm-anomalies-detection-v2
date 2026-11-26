# Phase 3 report

Status: code done, and the GPU run happened. The six implementation todos in `notes/2026-11-21-phase3-lora-train-plan.md` are checked. The final verification wave (F1-F4) is still open. The numbers below are the rerun, the one that kept the weights.

`python -m pytest -q` : **807 passed** on 2026-11-25. Baseline before this phase was 786. The new tests are the Phase 3 suite plus the adapter-save test and the empty-adapter test.

## What landed

All of this is uncommitted, on top of `58dab21`.

| Todo | Files | What it does |
|---|---|---|
| 1 | `vad/train/holdout.py`, `tests/vad_suite/test_holdout.py` | `holdout_ids(catalog, fraction=0.1, seed=20261119)`. Class-stratified. Same seed, same ids. Real catalog: 161 of 1610, each class between 8% and 12%, zero overlap with `manifests/ucf_test.csv`. |
| 2 | `vad/train/gate.py`, `tests/vad_suite/test_gate.py` | `gate_is_go(note, today)`. True only when the latest section contains literal `**go**` on `today` and no later `**no-go**`. No `nvidia-smi`, no clock read. Missing file returns false. |
| 3 | `vad/train/loop.py`, `tests/vad_suite/test_loop.py` | `train(examples, spec, steps_fn)`. One pass. Returns loss, steps, abstain rate. No peft, bitsandbytes, or torch. No file IO. |
| 4 | `scripts/train_lora.py`, `tests/vad_suite/test_train_lora_run.py` | `--run` trains through the injected loop and writes `notes/lora-train.md`. `--launch` calls `gate_is_go` and exits 3 before importing peft when the gate is not a same-day go. `--dry-run` unchanged. |
| 5 | `scripts/eval_trained.py`, `tests/vad_suite/test_eval_trained.py` | Leak check first. Six A6 metrics, tau read and asserted `round(tau, 4) == 1.1686`, A6 numbers copied from the file. R0, R1, R2 are `{"status": "not measured"}`. A planted test id exits non-zero and writes no `R3.json`. |
| 6 | `vad/train/notes.py`, `tests/vad_suite/test_train_notes.py`, `notes/lora-train.md`, `notes/holdout.md` | Both notes rendered from the run. Eval starts only when citation accuracy and class prior were both measured and differ by more than one percentage point. |

## Notes on disk

`notes/lora-train.md`

| Field | Value |
|---|---|
| commit | `58dab21`, training code uncommitted on top of this HEAD |
| config | r=16, alpha=32, dropout=0.05 |
| steps | 1449 (1610 minus the 161 holdout) |
| loss | 0.3162 |
| peak_memory | 12016 MiB |
| wall_time | 1086.7s |
| abstain_rate | 0.3920 |

The first launch also finished 1449 steps and exited 0, then freed the weights, because nothing called `save_pretrained`. The rerun writes the adapter to `~/vlm-adapter` on the GB10: `adapter_config.json` and `adapter_model.safetensors` (174663096 bytes). That copy is not in the repo. The commit field says `58dab21` because the GB10 checkout has no `.git`, so `rev-parse` returns nothing and the hash is written by hand. `--run` still stamps `pending-launch`, since no weights load on that path.

`notes/holdout.md`

| Field | Value |
|---|---|
| holdout_count | 161 |
| disjointness_check | PASS, 0 overlap with `manifests/ucf_test.csv` |
| citation_accuracy | not measured |
| class_prior | not measured |
| verdict | Citation accuracy not measured. Eval does not start. |

The 161 ids are listed in the note. An earlier copy of this file carried `vid_003` and `vid_016` from the 20-row test fixture. That copy was wrong. The file was regenerated from `holdout_ids` on the real catalog.

## Gate

`notes/gb10-gate.md` carries three go sections. `## 2026-11-21 go` (20:59, 0%, 38 C) was never used. `## 2026-11-25 go` (22:06, 0%, 61 C) covered the first launch. `## 2026-11-25 go, rerun` (23:19, 0%, 57 C) covered the rerun that saved the adapter. The gate test follows the note: false for `2026-11-20` and `2026-11-21`, true for `2026-11-25`.

## Two defects fixed during review

1. `write_notes` returned true when citation accuracy was `None`. Missing was treated as "not at the class prior", so the caller was told the eval may start while the note said it must not. The return is now true only when both numbers exist and differ by more than one point.
2. `--run` recorded the fake step's 0.1 as `loss = 0.1000`. No weights load on `--run`, so the field is `pending-launch`.

## What Phase 3 did not do

- No `results/trained/R3.json`. The eval script exists and is tested with an injected scorer. Its real scorer raises `NotImplementedError`.
- R0, R1, R2 not run. Recorded as not measured.
- Tau not recomputed. A6 numbers not retyped.
- Citation accuracy on the holdout not measured, so the stop rule cannot be evaluated yet. The eval does not start. Nothing in the code computes that number.
- The adapter weights are on the GB10 only, at `~/vlm-adapter`. Not in the repo.
- Final verification wave F1-F4 in the plan is still unchecked.
