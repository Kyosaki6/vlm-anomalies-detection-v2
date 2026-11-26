# phase3-lora-train - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**Who this is for and what changes for them:** Thanh can start a training run and have it either train or refuse before it touches the GPU. Thịnh gets one trained result to report, with the three unmeasured rows labeled as unmeasured instead of left blank.

**What you'll get:** A training command that holds out a fixed slice of the data, trains on the rest, and writes down what it did. A results file for the one trained configuration. A launch command that will not start unless the GPU gate was opened the same day.

**Why this approach:** The training code that exists today stops before loading the model, so the plan adds the loop rather than flipping a flag. The GPU is shared and currently busy, so everything except the launch is built and tested on CPU, and the launch checks the gate itself.

**What it will NOT do:**
- It will not run the three comparison configurations. They are recorded as not measured.
- It will not change the frozen threshold or the frozen baseline numbers.
- It will not launch the GPU run. That waits on a same-day go.

**Effort:** Large
**Risk:** Medium - the real training run can still show the adapter predicting the class prior, and the plan stops there on purpose.
**Decisions to sanity-check:** the holdout is 10% and class-stratified, training is a single pass, and only one of the four eval rows actually runs.

Your next move: run it with `$ulw-execute`, or ask for a high-accuracy review first. Full detail follows.

---

> TL;DR (machine): Large, medium risk. Six todos. Training loop, seeded holdout, R3-only eval, same-day gate guard. No GPU launch.

## Scope
### Affected user and ideal state
**Affected user:** Thanh, who launches the training and today gets exit code 2 from `scripts/train_lora.py --run`. Thịnh, who reports the result and today has no trained number to report. The next reader, who needs a launch command that will not start on a stale go, because the GPU box is shared.

| Row | Statement | Reason |
| --- | --- | --- |
| IS-1 | `--run` trains, or refuses before touching the GPU | the Phase 2 stub exits 2 after the smoke check, so the loop replaces that line rather than bypassing it |
| IS-2 | holdout ids are fixed by a seed, written down before training, and never seen by the adapter | T4 catches an adapter that predicts the class prior, and cannot if the holdout was trained on |
| IS-3 | the test set enters neither the adapter, the index, nor the holdout | H1 already guards the index; the new split has to pass the same check |
| IS-4 | `results/trained/R3.json` carries micro AUC, video bootstrap CI, FA/hour, window precision, incident recall, and the paired delta vs A6 | those are the A6 metrics the trained run is compared against |
| IS-5 | the results table names R0, R1, R2 as not measured | a blank cell reads as a forgotten run |
| IS-6 | tau stays 1.1686 and the A6 numbers are read from `results/ablation_A6.json` | both are frozen, and retyping is how a digit drifts |
| IS-7 | a launch with a stale or missing go exits before loading the model | a go is valid only on the day it was written |
| IS-8 | everything except the launch runs in CI with no GPU and no model download | CI has no GPU, and peft and bitsandbytes are not in `requirements-dev.txt` |
| GAP-1 | no training loop, optimizer, or data loader | `attach()` returns a `LoraConfig` and stops |
| GAP-2 | no holdout split | `build_dataset` trains on every query it is given |
| GAP-3 | no trained eval | `scripts/eval_trained.py` does not exist |
| GAP-4 | the same-day gate rule lives in a note, not in the launch path | a stale go would start a multi-hour job |
| GAP-5 | R0, R1, R2 have no recorded disposition | three empty cells |

### Must have
- A training loop behind `scripts/train_lora.py --run` that writes `notes/lora-train.md`.
- A seeded, class-stratified 10% holdout, consumed by both the trainer and the checker.
- `scripts/eval_trained.py` producing `results/trained/R3.json` with the six A6 metrics.
- R0, R1, R2 recorded as `{"status": "not measured"}`.
- A gate check that refuses `--launch` unless `notes/gb10-gate.md` says go on the same day.
- A test per todo that fails when its guard is removed.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- No R0, R1, or R2 runs.
- No change to tau, to the A4c or A6 numbers, or to `schema_version`.
- No 4-bit fallback.
- No second GPU job, no polling loop, no killing of a resident process.
- No edit to `tests/vad_suite/fixtures/raft_example.json`.
- No peft or bitsandbytes import in `requirements-dev.txt` or at module level.
- No push, and no GPU launch. The launch waits on a same-day go.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: tests-after, pytest. Each todo lands its test in the same change. The GPU launch has no CI test. Its evidence is the note it writes, and its guard is tested by asserting exit code 3.
- Evidence: `.omo/evidence/task-<N>-phase3-lora-train.txt`, the pytest output for that todo. Outside the ulw loop the same path is used.
- Full suite gate: `python -m pytest -q` stays green. Baseline before this work is 786 passed.

## Execution strategy
### Parallel execution waves
> Two todos share a wave unless one reads what the other produces or both write the same file.

- Wave 1: todos 1 and 2. Neither reads the other. Different new files.
- Wave 2: todos 3 and 5. Todo 3 reads the holdout from todo 1. Todo 5 reads only existing metrics and the leak check, so it does not wait on the loop.
- Wave 3: todo 4. Reads the holdout, the gate, and the loop, and writes `scripts/train_lora.py`, which all three touch only here.
- Wave 4: todo 6. Reads the run output and the eval script to name the stop rule.

### Dependency matrix
| Todo | Writes (files, shared state) | Depends on (reason) | Blocks |
| --- | --- | --- | --- |
| 1 | `vad/train/holdout.py`, `tests/vad_suite/test_holdout.py` | none | 3, 4 |
| 2 | `vad/train/gate.py`, `tests/vad_suite/test_gate.py` | none | 4 |
| 3 | `vad/train/loop.py`, `tests/vad_suite/test_loop.py` | 1 (examples are the post-holdout set) | 4 |
| 4 | `scripts/train_lora.py`, `tests/vad_suite/test_train_lora_run.py` | 1, 2, 3 (calls all three) | 6 |
| 5 | `scripts/eval_trained.py`, `tests/vad_suite/test_eval_trained.py` | none for the script; real numbers wait on 6 | 6 |
| 6 | `notes/lora-train.md`, `notes/holdout.md` | 4, 5 (records what they produce) | none |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [x] 1. `vad/train/holdout.py`: add `holdout_ids(catalog, fraction, seed)` that returns a class-stratified 10% split - expect the same ids for the same seed, and zero overlap with `manifests/ucf_test.csv`
  What to do: new module. Read `manifests/ucf_train.csv` into the `Clip` catalog (`vad/train/raft.py:37`). Group `video_id` by `event_class`, shuffle each group with `numpy.random.default_rng(seed)`, take `round(fraction * len(group))` from each, and return the sorted tuple. Default `fraction=0.1`, `seed=20261119`. Reject `fraction` outside (0, 1) and an empty catalog with `ValueError`.
  Must NOT do: do not read the test manifest here. Disjointness is the test's job and the note's job, not this function's, so the function stays usable on a fake catalog. Do not call `build_dataset`.
  Closes: GAP-2
  Parallelization: Wave 1 | Blocked by: none | Blocks: 3, 4
  References: `vad/train/raft.py:37` (`Clip`), `vad/train/raft.py:106` (`build_dataset` consumes every query, so the split must happen before it), `scripts/train_lora.py:44` (seed `20261119`), `manifests/ucf_train.csv`, `manifests/ucf_test.csv`, `tests/vad_suite/test_leakcheck.py` (the existing disjointness pattern to copy).
  Acceptance criteria: `python -m pytest -q tests/vad_suite/test_holdout.py` exits 0. The test asserts three things: same seed gives identical ids, a different seed gives a different set, and the real catalog's holdout is disjoint from `manifests/ucf_test.csv` and lands between 8% and 12% per class.
  QA scenarios: happy - call `holdout_ids` twice with one seed and assert equal output. failure - pass `fraction=0` and `fraction=1.5`, assert `ValueError`, and assert no file was written. Evidence: `.omo/evidence/task-1-phase3-lora-train.txt` (pytest output).
  Commit: N | bundled into commit 1 with todo 2
  Recommended task executor category: quick - one new module plus one test, no design choice left open.

- [x] 2. `vad/train/gate.py`: add `gate_is_go(note, today)` that reads `notes/gb10-gate.md` and returns true only for a same-day go - expect false for the current note
  What to do: parse the note as a sequence of sections. A section counts as a go only when it contains the literal `**go**` and its timestamp line (`YYYY-MM-DD`) equals `today`. The latest such section wins. Any other state, including a newer no-go, a go with no date, or a missing file, returns false. `today` is an argument, never `date.today()` inside the function, so the test controls it.
  Must NOT do: do not shell out to `nvidia-smi`. Do not read the clock. Do not raise on a missing file. Return false.
  Closes: GAP-4
  Parallelization: Wave 1 | Blocked by: none | Blocks: 4
  References: `notes/gb10-gate.md:38-63` (a go section followed by a newer no-go the same day; the function must return false for `today=2026-11-20`), binding B7 in `notes/2026-11-19-phase2-tasks.md:45`.
  Acceptance criteria: `python -m pytest -q tests/vad_suite/test_gate.py` exits 0. Cases: the real note returns false for `2026-11-20` and for `2026-11-21`; a synthetic note with one go dated today returns true; a synthetic note whose newest section is a no-go returns false; a missing path returns false.
  QA scenarios: happy - synthetic go note, assert true. failure - the committed `notes/gb10-gate.md`, assert false, so a stale go cannot pass. Evidence: `.omo/evidence/task-2-phase3-lora-train.txt`.
  Commit: Y | feat(train): add the holdout split and the same-day gate check
  Recommended task executor category: quick - pure function over text, one test file.

- [x] 3. `vad/train/loop.py`: add `train(examples, spec, steps_fn)` that runs one pass and returns loss, steps, and abstain rate - expect no peft import and no file written
  What to do: the loop body, with the model behind an injected callable. `steps_fn(example) -> float` returns the loss for one example. `train` iterates the examples once, sums the loss, counts the abstains from `example["abstain"]`, and returns `{"loss": mean, "steps": n, "abstain_rate": k/n}`. An empty example list raises `ValueError`.
  Must NOT do: do not import peft, bitsandbytes, or torch. Do not open a file. Do not call `attach`. The real optimizer stays in todo 4. This function is the part CI can run.
  Closes: GAP-1
  Parallelization: Wave 2 | Blocked by: 1 (the examples it trains on are the post-holdout set, and the test builds them through `holdout_ids`) | Blocks: 4
  References: `vad/train/lora.py:117` (`dry_run` shape to mirror: take data in, return a dict, write nothing), `vad/train/raft.py:91` (the example dict, `abstain` key), `tests/vad_suite/fixtures/raft_example.json` (frozen, do not edit).
  Acceptance criteria: `python -m pytest -q tests/vad_suite/test_loop.py` exits 0. The test passes a `steps_fn` that records the `video_id`s it saw and returns a constant loss, then asserts the recorded ids equal the input ids minus the holdout, the step count equals that length, and the abstain rate matches a hand count.
  QA scenarios: happy - three examples, one abstain, assert `abstain_rate == 1/3` and `steps == 3`. failure - empty list raises `ValueError`, and `sys.modules` contains no `peft` and no `bitsandbytes` after the call. Evidence: `.omo/evidence/task-3-phase3-lora-train.txt`.
  Commit: Y | feat(train): add the one-pass training loop behind an injected step
  Recommended task executor category: quick - one function, one test, the model is injected so nothing here is hard.

- [x] 4. `scripts/train_lora.py`: replace the exit-2 stub so `--run` trains through the injected loop, and add `--launch` that refuses without a same-day go - expect `--run` to write `notes/lora-train.md` from the fake step and `--launch` to exit 3 today
  What to do: delete lines 63-64 (the Phase 2 refusal). `--run` builds the catalog from `manifests/ucf_train.csv`, removes the holdout from todo 1, builds examples with `build_dataset` at `p=0.6`, calls `train` from todo 3 with a `steps_fn` argument, and writes `notes/lora-train.md` with commit, config, steps, final loss, and abstain rate. The peak memory and wall time fields are written as `pending-launch` because no weights were loaded. `--launch` is a new flag. It calls `gate_is_go(note, date.today())` first and exits 3 with the reason when that returns false. Only when it returns true does it import peft and load the model. That branch is not unit-tested.
  Must NOT do: do not load a model on `--run`. Do not let `--launch` reach the peft import when the gate returns false. Do not change `--dry-run`. Do not write the holdout ids into the training set.
  Closes: GAP-1, GAP-4
  Parallelization: Wave 3 | Blocked by: 1, 2, 3 (it calls all three) | Blocks: 6
  References: `scripts/train_lora.py:38-64` (the stub being replaced), `vad/train/lora.py:144` (`require_smoke`, still called first), `vad/train/gate.py` (todo 2), `vad/train/loop.py` (todo 3), `notes/2026-11-19-task-assignment-qwen3-vl-lora-raft.md:86` (the fields `notes/lora-train.md` must carry).
  Acceptance criteria: `python -m pytest -q tests/vad_suite/test_train_lora_run.py` exits 0. The test runs `--run` against a temp manifest and a fake `steps_fn`, asserts `notes/lora-train.md` contains `steps`, `loss`, and `abstain_rate`, and asserts the holdout ids are absent from the ids the fake step recorded. A second test runs `--launch` with `notes/gb10-gate.md` and asserts exit code 3.
  QA scenarios: happy - `--run` on the temp manifest, note written, holdout excluded. failure - `--launch` with the real gate note, exit 3, and `peft` not imported. Evidence: `.omo/evidence/task-4-phase3-lora-train.txt`.
  Commit: Y | feat(train): replace the Phase 2 stub with the real run and a gated launch
  Recommended task executor category: deep-low - it threads three modules and the failure that matters is a holdout id leaking into training, which the test has to pin down.

- [x] 5. `scripts/eval_trained.py`: add the R3 eval that reads A6 from `results/ablation_A6.json` and writes `results/trained/R3.json`, with R0, R1, R2 recorded as not measured - expect the not-measured rows present and tau unchanged
  What to do: new script. It runs the H1 leak check first by importing the check `tests/vad_suite/test_leakcheck.py` already exercises, and exits non-zero if a test id is in the train manifest or the index. Then it scores the 290 with the trained adapter and writes `results/trained/R3.json` containing micro AUC, the video bootstrap CI from `video_bootstrap_auc_ci`, FA/hour from `false_alarms_per_hour`, window precision, incident recall, and the paired delta against A6. The A6 numbers are loaded from `results/ablation_A6.json`, not literals. tau is read from the same place the A6 run used, asserted equal to 1.1686, and never written as an input. R0, R1, R2 appear in the output as `{"status": "not measured"}`.
  Must NOT do: do not retype an A6 number. Do not recompute tau. Do not run R0, R1, or R2. Do not import peft at module level. The scoring call takes an injected scorer so the test needs no model.
  Closes: GAP-3, GAP-5
  Parallelization: Wave 2 | Blocked by: none for the script and its test; the real numbers wait on todo 6 | Blocks: 6
  References: `vad/eval/metrics.py:66` (`video_bootstrap_auc_ci`), `vad/eval/metrics.py:166` (`false_alarms_per_hour`), `scripts/eval_tier2.py` (the existing eval to mirror for CLI shape), `results/ablation_A6.json` (source of the A6 numbers), `notes/2026-11-19-task-assignment-qwen3-vl-lora-raft.md:96` (the metric list and the tau rule), `tests/vad_suite/test_leakcheck.py` (the leak check to run first).
  Acceptance criteria: `python -m pytest -q tests/vad_suite/test_eval_trained.py` exits 0. The test points the script at a fake scorer and a temp A6 file, asserts `R3.json` carries all six metrics, asserts the A6 numbers in the output equal the temp file byte for byte in their field, asserts tau was not modified, and asserts R0, R1, R2 each equal `{"status": "not measured"}`.
  QA scenarios: happy - fake scorer, all six metrics present, not-measured rows present. failure - plant a test id in the train manifest, assert non-zero exit and no `R3.json` written. Evidence: `.omo/evidence/task-5-phase3-lora-train.txt`.
  Commit: Y | feat(eval): add the R3 trained eval, mark R0 R1 R2 not measured
  Recommended task executor category: unspecified-high - one script but it has to reuse the metrics and the leak check correctly, and the test has to prove the A6 numbers were read rather than typed.

- [x] 6. `notes/lora-train.md` and `notes/holdout.md`: write both notes from the run, and stop if the holdout citation accuracy equals the class prior - expect the notes to name the commit, the holdout ids, and the disjointness check
  What to do: `--launch` already refuses without a go (todo 4), so this todo is the note format and the stop rule. `notes/lora-train.md` carries commit, config, steps, final loss, peak memory, wall time, and the holdout abstain rate, which is the list in the assignment at line 86. `notes/holdout.md` lists every holdout `video_id` and states the result of the disjointness check against `manifests/ucf_test.csv`. Citation accuracy on the holdout is compared to the class prior. If it equals the prior within one percentage point, the note says the adapter predicts the class prior and the eval does not start.
  Must NOT do: do not launch the GPU run in this todo. Do not start `eval_trained.py` for real numbers when the note says the adapter is at the prior. Do not add a holdout id that is also in `ucf_test.csv`.
  Closes: GAP-2
  Parallelization: Wave 4 | Blocked by: 4 and 5 | Blocks: nothing
  References: `notes/2026-11-19-task-assignment-qwen3-vl-lora-raft.md:86-89` (the required fields and the stop rule), `notes/qwen3-smoke.md` (the note format to follow: one table, a verdict line), `manifests/ucf_test.csv`.
  Acceptance criteria: a test renders both notes from a fixed run dict and asserts every required field is present, the holdout list matches `holdout_ids`, and none of those ids are in `ucf_test.csv`. A second case feeds a citation accuracy equal to the prior and asserts the rendered note contains the stop sentence and that the eval command is not invoked.
  QA scenarios: happy - note rendered with all fields and a disjoint holdout. failure - citation accuracy equal to the prior, stop sentence present, eval not called. Evidence: `.omo/evidence/task-6-phase3-lora-train.txt`.
  Commit: Y | docs(train): record the training run and the holdout check
  Recommended task executor category: writing - the output is two notes, and the test checks their content rather than their prose.

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit
- [ ] F2. Code quality review
- [ ] F3. Real manual QA
- [ ] F4. Ideal-state fidelity

## Commit strategy

Four commits, one per wave pair, never one commit per todo where two todos share a file.

1. `feat(train): add the holdout split and the same-day gate check` — todos 1 and 2.
2. `feat(train): add the one-pass training loop behind an injected step` — todo 3.
3. `feat(train): replace the Phase 2 stub with the real run and a gated launch` — todo 4.
4. `feat(eval): add the R3 trained eval, mark R0 R1 R2 not measured` — todo 5.
5. `docs(train): record the training run and the holdout check` — todo 6.

Each commit must leave `python -m pytest -q` green. No commit touches `raft_example.json`, `results/ablation_A6.json`, or `schema_version`.

## Success criteria
> One row per IS row. F4 checks the delivered behavior against these rows, and a shortfall becomes new `- [ ] N.` rows, never a note.

| IS | Delivering todo(s) | Proving QA scenario | Evidence |
| --- | --- | --- | --- |
| IS-1 | 4 | `--run` writes the note, `--launch` exits 3 | `.omo/evidence/task-4-phase3-lora-train.txt` |
| IS-2 | 1, 6 | same seed gives the same ids, and the note lists them | `.omo/evidence/task-1-phase3-lora-train.txt` |
| IS-3 | 1, 5 | holdout disjoint from `ucf_test.csv`, planted test id aborts the eval | `.omo/evidence/task-5-phase3-lora-train.txt` |
| IS-4 | 5 | `R3.json` carries all six metrics | `.omo/evidence/task-5-phase3-lora-train.txt` |
| IS-5 | 5 | R0, R1, R2 each equal `{"status": "not measured"}` | `.omo/evidence/task-5-phase3-lora-train.txt` |
| IS-6 | 5 | A6 numbers match the source file, tau unchanged | `.omo/evidence/task-5-phase3-lora-train.txt` |
| IS-7 | 2, 4 | real gate note returns false, `--launch` exits 3 | `.omo/evidence/task-2-phase3-lora-train.txt` |
| IS-8 | 3 | `peft` and `bitsandbytes` absent from `sys.modules` after the loop | `.omo/evidence/task-3-phase3-lora-train.txt` |
