# Task assignment: Qwen3-VL LoRA + Berkeley RAFT fork

Companion to `notes/2026-11-19-plan-qwen3-vl-lora-raft-fork.md` (the WHAT/WHY). This file is the WHO/WHEN.

> Written 2026-11-19 against the submitted repo (`main`, results frozen). No tasks are done. Status column starts at `[ ]`.

**SCOPE:**
- New project, no deadline. The 2026-11-04 submission stays on source `main` and is not edited.
- Team: 4 people (Phước, Nam, Thanh, Thịnh). Huy is out of this revamp.
- Train host: `gx10@aitopatom-d4ac` (NVIDIA GB10, CUDA 13.0, 121 GiB unified), Tailscale only, behind the gate in section 6. Thịnh holds the account for now (settled 2026-11-19). GPU launches go through him until that changes.

No day numbers. The submitted assignment used D1..D6 because of a hard cutoff. This one uses phases P0..P4, and a phase starts when its handoff is true, not when a date arrives.

ID scheme: F1..F3 fork, P-series is Phước, N-series is Nam, T-series is Thanh, H-series is Thịnh. `P0`..`P4` mean phases.

## 0. Assumptions (change the plan if any is wrong)

1. Thịnh = leak test + eval table + dashboard metadata + Vietnamese report. Same person as the submitted report owner.
2. Thịnh holds the GB10 for now. It is not idle. Snapshot 2026-11-19 20:10: GPU-util 96%, 81 GiB available of 121 GiB. Resident compute: PID 401001 `python` 17024 MiB, PID 535418 and PID 537304 `MATE-2026/.venv/bin/python` 2428 MiB each. Holding the account does not license killing them. Those PIDs will change. The rule does not.
3. VRAM bounds in the plan (8-bit LoRA ~20-24 GB) are planning numbers. Phước's smoke log replaces them. Nothing else is scheduled on the GB10 before that log exists.
4. Nobody has been asked whether they accept these roles. Swap anything; keep the interfaces.
5. Thanh has no track in `notes/2026-10-29-task-assignment.md`. This file is the first assignment that names him.

## 1. Summary table

| Person | Hardware | Main track | Also | Blocked by |
|---|---|---|---|---|
| Phước | code only; GB10 launches via Thịnh | **P. 8-bit Qwen3-VL loader + smoke** | keeps the serve path documented | the util gate, and Thịnh's go |
| Nam | CPU (index build is numpy) | **N. Train manifest + SigLIP index + RAFT schema** | none | train split download |
| Thanh | GB10 via Thịnh, after Phước's smoke | **T. LoRA loop + RAFT dataset + holdout** | none | Phước smoke + Nam schema + Thịnh's go |
| Thịnh | GB10 account, for now | **H. Leak test, eval, dashboard, report** | launches every GPU job, reviews every PR | Thanh adapter + Nam index |

GPU time is the scarce thing, and it is serial: Phước smoke, then Thanh train, then Thịnh eval. Nam's index and Thịnh's tests run on CPU in parallel with all of it.

## 2. Gaps (each is a task below)

| # | Gap | Where | Task |
|---|---|---|---|
| G1 | No fork remote. LoRA is named only as future work in the Vietnamese report | repo | F1 |
| G2 | `quantization` accepts only `"4bit"` and `"none"`. Default model is `Qwen/Qwen2.5-VL-3B-Instruct`. Loader maps `qwen2.5-vl` only | `vad/verify/vlm.py:34`, `:135`, `:197` | P1 |
| G3 | `requirements-vlm.txt` pins `transformers>=4.49.0`, no `peft`, no `trl`. bitsandbytes is inference-only | `requirements-vlm.txt` | F2 |
| G4 | No train manifest. `manifests/` has `ucf_test.csv` (290) and `ucf_calib_normal.csv` (60) only | `manifests/` | N1 |
| G5 | No retrieval index, no golden/distractor builder | absence | N2, T1 |
| G6 | No leak check. A test `video_id` in the index would silently inflate AUC | eval path | H1 |
| G7 | `RunResult.metadata` has no adapter id, no RAFT hits, no trained flag. `schema_version` is 1 and must stay 1 | `vad/results.py:31` | H3 |
| G8 | GB10 is busy. 8-bit LoRA fits the memory and does not fit the current compute | snapshot 2026-11-19 20:10 | P0 |

## 3. Per-person task lists

Format: ID | phase | task | ACCEPTANCE.

### F. Fork (Phước opens it, everyone consumes it)

| ID | Phase | Task | Acceptance | Status |
|---|---|---|---|---|
| F1 | P0 | Create the fork remote from this repo. Source `main` is not pushed to. README first paragraph says the fork is trained and points back at the submitted AUC | Fork URL posted. `git log` on the source repo is unchanged. Submitted `results/run_A4c/` still hashes the same | `[ ]` |
| F2 | P0 | Fork `requirements-vlm.txt`: add `peft`, `trl`, raise `transformers` to `>=4.57`. Leave `requirements-dev.txt` GPU-free | `pip install -r requirements-dev.txt && python -m pytest -q` still green, no model download. VLM requirements install on the GB10 only | `[ ]` |
| F3 | P0 | Copy the frozen baselines into the fork README: A4c 73.00 [64.45, 81.15], A6 67.90 [58.39, 77.49], paired delta -5.10 [-11.42, -0.25] | Numbers match `results/ablation_A4c.json` and `results/ablation_A6.json` byte for byte, not from memory | `[ ]` |

### P. Phước: model load + GB10

| ID | Phase | Task | Acceptance | Status |
|---|---|---|---|---|
| P0 | P0 | Gate check, no training. SSH over Tailscale, record `nvidia-smi` and `free -h`. If GPU-util is not under 10% for 10 continuous minutes, stop and post the reading. Do not kill PID 401001, 535418, 537304, or any later resident / MATE-2026 process | `notes/gb10-gate.md` with timestamp, util, temperature, resident PIDs, and a go / no-go line | `[ ]` |
| P1 | P1 | `vad/verify/vlm.py`: accept `quantization="8bit"` (`load_in_8bit=True`). Add the Qwen3-VL class next to `Qwen2_5_VLForConditionalGeneration`. `"4bit"` and `"none"` unchanged. Anything else still raises `ValueError` | Unit test: `"8bit"` is accepted, `"4bit"` still builds a 4-bit config, `"2bit"` raises. Mocked model, no download | `[ ]` |
| P2 | P1 | Smoke, only after P0 is go. Load `Qwen/Qwen3-VL-8B-Instruct` 8-bit, one window, k frames. Write `notes/qwen3-smoke.md`: peak memory, seconds, bitsandbytes version, transformers version, driver, commit | Log exists and the process exited 0. This log replaces the 20-24 GB planning bound. If it OOMs, stop and report; do not switch to 4-bit | `[ ]` |
| P3 | P1 | Document the Tailscale serve path: host, port, who has a key, how a job is launched so it dies with the ssh session (`tmux` or nohup named in the note) | A second person (Thịnh) reproduces the smoke command from the note | `[ ]` |

Risk owner for: 8-bit load does not fit, or bitsandbytes misbehaves on CUDA 13.0. Decision rule: P2 is the decision. No full run starts around it.

### N. Nam: train split, index, schema

| ID | Phase | Task | Acceptance | Status |
|---|---|---|---|---|
| N1 | P1 | `scripts/build_train_manifest.py` + `manifests/ucf_train.csv`. 1610 rows (800 normal + 810 anomalous), `split=train`, `video_id` unique, zero overlap with `manifests/ucf_test.csv`. `calib_normal` ids may appear; test ids may not | Test plants one test id and the script exits non-zero. Row counts asserted, not printed | `[ ]` |
| N2 | P1-P2 | `vad/index/siglip.py` + `scripts/build_raft_index.py`. Index = train window embeddings only. `search(emb, k, exclude_video_id)` drops that id. Building the index with a test id present raises | Test: query video is absent from its own hits. Test: a planted test id aborts the build | `[ ]` |
| N3 | P1 | RAFT example schema, agreed with Thanh before either writes code: query video id, frame paths, retrieved ids, `golden_present: bool`, target text, abstain flag. One JSON example committed under `tests/vad_suite/fixtures/` | Both Thanh and Thịnh sign off in the PR. Schema is frozen after merge; later changes need both | `[ ]` |
| N4 | P2 | Embed the train split with the existing SigLIP 2 cache (`scripts/precompute_embeddings.py`), same model id, fps, dtype as the test cache. CPU-side manifest first; GPU run waits on P0 | Cache key matches the test cache. A train file and a test file of the same settings load with one loader | `[ ]` |

### T. Thanh: LoRA + RAFT

| ID | Phase | Task | Acceptance | Status |
|---|---|---|---|---|
| T1 | P2 | `vad/train/raft.py`. For each query, with probability P=0.6 include the golden clip (same video-level class, different `video_id`) plus distractors; else distractors only and `abstain=true`. Distractors are top index hits of a different class. Seed fixed | Test: P=0 gives all abstain, P=1 gives all golden, P=0.6 over a fixed seed matches a stored count. No GPU | `[ ]` |
| T2 | P2 | `vad/train/lora.py` + `scripts/train_lora.py`. Frozen 8-bit base, LoRA on `q,k,v,o,gate,up,down`, r=16, alpha=32, dropout 0.05, vision tower frozen. Config written beside the adapter. Batch 1 | Dry run on 2 examples prints the trainable-parameter count and saves `adapter_config.json`. Refuses to start if `notes/qwen3-smoke.md` is absent | `[ ]` |
| T3 | P3 | Real train on the 1610, RAFT mix, GB10 only, after P0 go and P2 smoke. One process. No second job | `notes/lora-train.md`: commit, config, steps, final loss, peak memory, wall time, abstain rate on a train holdout that the adapter did not see | `[ ]` |
| T4 | P3 | Holdout check before test. A slice of train held out of the adapter. Report citation accuracy and abstain rate there. If citation accuracy is at the class prior, stop and say so | Note states the holdout ids and confirms none of them are in `ucf_test.csv` | `[ ]` |

Risk owner for: the adapter ignores the retrieved clips and predicts the class prior. T4 is the check. Do not wait for the test set to discover it.

### H. Thịnh: eval, dashboard, report

| ID | Phase | Task | Acceptance | Status |
|---|---|---|---|---|
| H1 | P1 | Leak test, written before any trained number exists. Loads train manifest, test manifest, index id list. One test id in either train or index -> non-zero exit, message names the id | Test plants `Abuse028_x264` into a fake index and asserts failure. Test with the real disjoint manifests asserts success | `[ ]` |
| H2 | P3-P4 | `scripts/eval_trained.py`. Runs H1 first. Then R0 (8-bit, no LoRA, no retrieval), R1 (retrieval, no LoRA), R2 (LoRA, no retrieval), R3 (LoRA + RAFT). Same metrics as A6: micro AUC, video bootstrap CI, FA/hour, window precision, incident recall, paired delta vs A6. tau = 1.1686 untouched | JSON per row under `results/trained/`. A6 numbers in the table copied from `results/ablation_A6.json`, not retyped | `[ ]` |
| H3 | P2 | Dashboard metadata, no schema bump. `metadata.trained`, `metadata.adapter_id`, `metadata.base_model`, `metadata.quantization`, `metadata.raft_hits`. `app/data.py` still drops `schema_version != 1`. UI shows a "TRAINED" tag and the cited clip ids, English | Test: a trained sample JSON round-trips through `RunResult`. Test: `schema_version` 2 is still rejected. Replay tab shows the tag on the sample | `[ ]` |
| H4 | P4 | Report section, Vietnamese. Method names Berkeley RAFT (arXiv:2403.10131) in the first paragraph. Table is A4c, A6, R0, R1, R2, R3. A trained run that loses to A6 is reported as a loss | Numbers in the tex match the JSON. Grep shows no "zero-shot" applied to the trained run | `[ ]` |

## 4. Phases and handoffs

```
F1 fork + F2 requirements
        |
   P0 gate ──no-go──> stop, post the reading, do not queue
        |
   P1 loader (CPU tests) ──> P2 smoke on GB10 ──> notes/qwen3-smoke.md
        |
   N1 manifest ──> N2 index ──> N3 schema freeze
        |                          |
        |                          v
        |                    T1 RAFT builder (CPU) ──> T2 train script
        |                                                     |
        v                                                     v
   H1 leak test (CPU, before any trained number)       T3 train ──> T4 holdout
                                                              |
                                                              v
                                                    H2 eval R0..R3 ──> H3 dashboard ──> H4 report
```

| Handoff | From -> To | Must be true before the next starts | Status |
|---|---|---|---|
| HF1 | Phước F1 -> everyone | Fork URL exists, source `main` untouched | `[ ]` |
| HF2 | Phước P0 -> Phước P2, Thanh T3 | `notes/gb10-gate.md` says go, dated the same day as the run | `[ ]` |
| HF3 | Phước P2 -> Thanh T2/T3 | `notes/qwen3-smoke.md` has peak memory and exit 0 | `[ ]` |
| HF4 | Nam N3 -> Thanh T1 | Schema JSON merged, both signed off | `[ ]` |
| HF5 | Nam N1 + N2 -> Thịnh H1, Thanh T3 | Manifest and index both refuse a planted test id | `[ ]` |
| HF6 | Thanh T4 -> Thịnh H2 | Holdout note written. Test eval does not start on a prior-only adapter without that note saying so | `[ ]` |

## 5. Order if the GB10 stays busy

CPU work does not wait on the gate: F1, F2, F3, P1, N1, N2, N3, T1, H1, H3.

GPU work waits, in this order, one job at a time: P2 smoke, N4 train embeddings, T3 train, H2 eval (R0..R3).

If the gate stays shut for days, do not rent another GPU and do not drop to 4-bit. Post the reading and stop that branch.

Cut order if scope has to shrink (top first, bottom never):

1. R0 row (model-swap only). Say it was not measured.
2. Second rank ablation, projector LoRA. Already stretch.
3. Train-embedding shard shared across people. Nam runs it alone.
4. NEVER cut: the leak test (H1), the smoke log (P2), `schema_version` staying 1, the "do not kill resident PIDs" rule, the frozen A4c/A6 numbers.

## 6. GB10 rules (everyone)

- Reach it over Tailscale only.
- Before any compute: `nvidia-smi` and `free -h`. GPU-util under 10% for 10 continuous minutes, or do not start.
- Never kill a process you did not start. That includes PID 401001, 535418, 537304, and whatever replaces them.
- One GPU process per machine. Batch size 1. No background eval while training runs.
- Write the log note before declaring the task done. A run with no note did not happen.
- Snapshot to beat, 2026-11-19 20:10: util 96%, 69 C, 36 W, 81 GiB available, three resident Python processes. P0's first reading will differ; record the new one instead of reusing these numbers.

## 7. Working agreements

- One branch per task: `feat/<owner>-<id>`, PRs into the **fork**, never into the submitted repo.
- `vad/types.py`, `vad/results.py` `SCHEMA_VERSION`, and the RAFT example schema are frozen once N3 merges. A change needs a yes from every consumer.
- Commit style unchanged: `type(scope): message`, English.
- A task is done when its acceptance line is true and `python -m pytest -q` is green. GPU tasks are done when the note exists and names the commit.
- Tests never download a model and never need a GPU. The smoke log is the GPU proof.
- Stubs raise `NotImplementedError`. Nobody returns a fake verdict, a fake AUC, or a fake memory number.

## 8. First-day checklist

| Who | Do | Report |
|---|---|---|
| Phước | F1 fork URL. P0 gate reading, even if the answer is no-go | URL + `notes/gb10-gate.md` |
| Nam | Confirm `manifests/ucf_test.csv` has 290 data rows. Draft the N3 schema JSON, one example, no code yet | row count + the JSON in the PR description |
| Thanh | Read arXiv:2403.10131 sections on the golden-or-absent mix. Write five lines in the PR: what P=0.6 means, what abstain maps to | the five lines |
| Thịnh | Add the H1 test with a planted test id against a fake index, before any of Nam's or Thanh's code exists | failing test on `main` of the fork, red for the right reason |

## 9. Open questions

1. SETTLED 2026-11-19. Thịnh has the GB10 for now. Who else gets a key, and who holds it long-term, stay open.
2. Is there a report date for this revamp? None assumed.
3. Does Thanh accept the train loop? GPU launches go through Thịnh while the box is his, not through Phước alone.
