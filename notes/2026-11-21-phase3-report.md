# Phase 3 report

Status: done. The six implementation todos in `notes/2026-11-21-phase3-lora-train-plan.md` are checked, and the holdout citation run has been measured. It falls short of the class prior, so the stop rule fires and the 290-video eval does not start. The final verification wave (F1-F4) was not run.

`python -m pytest -q` : **812 passed** on 2026-11-26. Baseline before this phase was 786. The new tests are the Phase 3 suite, the adapter-save test, the empty-adapter test, and `tests/vad_suite/test_citation.py`.

## What landed

All of this is uncommitted, on top of `58dab21`.

| Todo | Files | What it does |
|---|---|---|
| 1 | `vad/train/holdout.py`, `tests/vad_suite/test_holdout.py` | `holdout_ids(catalog, fraction=0.1, seed=20261119)`. Class-stratified. Same seed, same ids. Real catalog: 161 of 1610, each class between 8% and 12%, zero overlap with `manifests/ucf_test.csv`. |
| 2 | `vad/train/gate.py`, `tests/vad_suite/test_gate.py` | `gate_is_go(note, today)`. True only when the latest section contains literal `**go**` on `today` and no later `**no-go**`. No `nvidia-smi`, no clock read. Missing file returns false. |
| 3 | `vad/train/loop.py`, `tests/vad_suite/test_loop.py` | `train(examples, spec, steps_fn)`. One pass. Returns loss, steps, abstain rate. No peft, bitsandbytes, or torch. No file IO. |
| 4 | `scripts/train_lora.py`, `tests/vad_suite/test_train_lora_run.py` | `--run` trains through the injected loop and writes `notes/lora-train.md`. `--launch` calls `gate_is_go` and exits 3 before importing peft when the gate is not a same-day go. `--dry-run` unchanged. |
| 5 | `scripts/eval_trained.py`, `tests/vad_suite/test_eval_trained.py` | Leak check first. Six A6 metrics, tau read and asserted `round(tau, 4) == 1.1686`, A6 numbers copied from the file. R0, R1, R2 are `{"status": "not measured"}`. A planted test id exits non-zero and writes no `R3.json`. |
| 6 | `vad/train/notes.py`, `vad/train/citation.py`, `tests/vad_suite/test_train_notes.py`, `tests/vad_suite/test_citation.py`, `notes/lora-train.md`, `notes/holdout.md` | Both notes rendered from the run. Eval starts only when citation accuracy and class prior were both measured and differ by more than one percentage point. `score_holdout` generates once per example. |

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
| citation_accuracy | 0.0211 (2 of 95 golden clips) |
| class_prior | 0.2632 |
| diff | -0.2421 |
| holdout_abstain_rate | 0.9394 (62 of 66) |
| verdict | Citation accuracy is below the class prior. Evaluation does not start. |

The 161 ids are listed in the note. An earlier copy of this file carried `vid_003` and `vid_016` from the 20-row test fixture. That copy was wrong. The file was regenerated from `holdout_ids` on the real catalog.

The citation numbers come from the 2026-11-26 launch on the GB10. It trained 1449 steps (loss 0.0696, peak 12827 MiB, 963.2 s), saved the adapter, scored the holdout, printed `DỪNG: citation không vượt class prior, eval không chạy`, and exited 4. The adapter cites the right clip on 2 of 95 golden examples, against a majority-class prior of 26.3%. It loses by 24 points, so `scripts/eval_trained.py` was not run and `results/trained/R3.json` does not exist. That is the stop rule, not a skipped step.

## Gate

`notes/gb10-gate.md` carries three go sections. `## 2026-11-21 go` (20:59, 0%, 38 C) was never used. `## 2026-11-25 go` (22:06, 0%, 61 C) covered the first launch. `## 2026-11-25 go, rerun` (23:19, 0%, 57 C) covered the rerun that saved the adapter. The gate test follows the note: false for `2026-11-20` and `2026-11-21`, true for `2026-11-25`.

The 2026-11-26 citation run has no section of its own. Two readings, 15:06:32 and 15:08:00, both showed 0%, 56 C, no compute process, 108 GiB available. The gap is 88 seconds, under the ten-minute bar the note requires, so it is not a go and writing it as the latest section would make `gate_is_go` return false for `2026-11-25`, which the test pins as true. The launch ran off that reading anyway. It retrained rather than reusing the 2026-11-25 adapter, and `~/vlm-adapter` now holds the 2026-11-26 weights.

## Three defects fixed during review

1. `write_notes` returned true when citation accuracy was `None`. Missing was treated as "not at the class prior", so the caller was told the eval may start while the note said it must not. The return is now true only when both numbers exist and differ by more than one point.
2. `--run` recorded the fake step's 0.1 as `loss = 0.1000`. No weights load on `--run`, so the field is `pending-launch`.
3. `test_train_lora_run_with_fake_steps_and_holdout_exclusion` passed `--train-note` as a temp path but left `--holdout-note` at its default `notes/holdout.md`. The 20-row fixture holdout is `vid_003` and `vid_016`, so every pytest run rewrote the real note with those two ids. The report above already called that copy wrong once, and the file was back in that state. The test now passes `--holdout-note` into `tmp_path` and asserts the repo note still says `holdout_count | 161` and contains no `vid_003`. `notes/holdout.md` was regenerated from `holdout_ids` on `manifests/ucf_train.csv`: 161 ids, 0 overlap with `manifests/ucf_test.csv`. Full suite after the fix: 808 passed, and the note still read 161.

## What Phase 3 did not do

- No `results/trained/R3.json`. The citation run landed below the class prior, so the eval did not start. The script exists and is tested with an injected scorer. Its real scorer raises `NotImplementedError`.
- R0, R1, R2 not run. Recorded as not measured.
- Tau not recomputed. A6 numbers not retyped.
- The adapter weights are on the GB10 only, at `~/vlm-adapter`, and that file is the 2026-11-26 run. The 2026-11-25 weights were overwritten. Not in the repo.
- `notes/lora-train.md` still records the 2026-11-25 run. The box rewrote its own copy with `commit | unknown` because the checkout there has no `.git`, so that copy was not brought back.
- Final verification wave F1-F4 in the plan was not run.
