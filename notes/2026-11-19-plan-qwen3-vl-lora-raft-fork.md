# Plan: Qwen3-VL-8B LoRA + Berkeley RAFT (fork)

Status: DRAFT v1 (2026-11-19). Source: `.omo/drafts/qwen3-vl-lora-raft-fork.md`.
No hard deadline. This is a new project. The 2026-11-04 submission already landed on `main` and stays there.
Team: 4 people (Thịnh, Nam, Phước, Thanh). Huy is out of this revamp.
Language: report + defense in Vietnamese; dashboard UI in English; code comments bilingual (Vietnamese prose, technical terms verbatim).
Claim change: the submitted system is training-free. This fork is trained. The report must say so. Zero-shot numbers stay the frozen baseline, never overwritten.

Hardware. Thịnh holds the GB10 for now (settled 2026-11-19; "for now", not a permanent assignment). Measured 2026-11-19 20:10 local on `gx10@aitopatom-d4ac`, reached over Tailscale:

| Item | Value |
|---|---|
| GPU | NVIDIA GB10, persistence on, bus `0000000F:01:00.0` |
| Driver / CUDA | 580.178.04 / CUDA 13.0 |
| Unified memory | 121 GiB total, 81 GiB available, 55 GiB buff/cache, 15 GiB swap almost idle (98 MiB used) |
| GPU util at snapshot | 96%, 69 C, 36 W, P0 |
| Resident compute | PID 401001 `python` 17024 MiB; PID 535418 and PID 537304 `MATE-2026/.venv/bin/python` 2428 MiB each |
| Display | Xorg 43 MiB, gnome-shell 16 MiB. Ignore. |

8-bit LoRA of the 8B fits the memory (planning bound ~20-24 GB, not a lab measurement). It does not fit the current compute. Gate in section 9.

---

## 0. One-paragraph pitch (what the defense says)

A trained verifier on top of the submitted training-free pipeline. Tier 1 is unchanged: SigLIP 2 frame embeddings, generic-prompt plus normal-memory scoring, temporal smoothing, candidate windows, tau from `calib_normal`. Tier 2 is replaced. `Qwen/Qwen3-VL-8B-Instruct` loads in 8-bit with a LoRA adapter trained in this fork by Berkeley RAFT (Zhang et al., arXiv:2403.10131): each training example retrieves SigLIP neighbours from the UCF-Crime TRAIN split only, sometimes including the golden clip, sometimes only distractors, and the model learns to cite the evidence or abstain. The frozen 290-video test is eval-only. The same metrics sit next to the submitted baseline: micro frame AUC 73.00 tier-1 / 67.90 tier-2, video-bootstrap CI, FA/hour, window precision, incident recall. The dashboard still reads `RunResult` JSON `schema_version` 1 and labels the run trained, not zero-shot.

## 1. Honest framing (decide this up front, it protects the defense)

- The submitted result is the baseline, not a number to beat by rewriting history. Tier-1 A4c: micro AUC **73.00** [64.45, 81.15], 307 windows in 114 of 290 videos, tau = 1.1686 from `calib_normal`. Tier-2 A6 (Qwen2.5-VL-3B 4-bit): 129/307 confirmed, AUC **67.90** [58.39, 77.49], paired delta **-5.10** [-11.42, -0.25], FA/hour 20.64 -> 0.17, window precision 28.0% -> 49.6%, incident recall 51.9% -> 39.7%.
- A0 missed its pre-registered band (48.77 [38.30, 59.40] vs LAVAD 53.16). That miss stays in the report. Do not re-run it hoping for a different number.
- This fork is NOT zero-shot and NOT training-free. LoRA is a gradient step on UCF-Crime train. Say "trained verifier, frozen tier-1". Do not say "zero-shot Qwen3-VL".
- RAFT here is Berkeley retrieval-augmented fine-tuning (arXiv:2403.10131). It is not Teed & Deng optical flow (arXiv:2003.12039). No published VAD paper was found that fine-tunes a VLM with this curriculum, so the method section specifies the recipe instead of citing one.
- Typing numbers from the submission (VLM 83.9% on 62 windows) are conditional and class-skewed. Do not quote them as "VLM types 84% of UCF events". A new typing number, if measured, gets the same caveat.
- ShanghaiTech stays out of scope (D4-1 NO-GO). Do not reopen it.
- Published reference points, cited not rerun: LAVAD (CVPR 2024) ZS-CLIP 53.16, LLaVA-1.5 72.84, LAVAD 80.28. Weakly-supervised SOTA ~85-87. A trained 8B verifier that fails to beat 67.90 is still a result. Report it.

## 2. Scope

IN:
- New git remote forked from this repo. `main` of the source repo is read-only.
- 8-bit loader for `Qwen/Qwen3-VL-8B-Instruct` and a LoRA train loop (frozen base, trainable adapter).
- UCF-Crime train split materialised: 1610 videos (800 normal + 810 anomalous), video-level labels, manifest, SigLIP index built from train only.
- RAFT dataset builder: query window + retrieved train clips, golden present with probability P, else distractors only and the target is abstain.
- Leak test: a test `video_id` in the train manifest or in the index fails the run.
- Eval table beside A4c/A6, same metrics, tau still from `calib_normal`.
- Dashboard metadata: adapter id, RAFT evidence ids, explicit trained label. `schema_version` stays 1.

OUT:
- Any edit that changes the submitted numbers or rewrites `results/run_A4c/`.
- Training, retrieval, or threshold fitting on the 290 test videos.
- Dropping to 4-bit QLoRA to dodge the VRAM question. That is a different plan and needs its own yes.
- Optical-flow RAFT, unless the user overrides the default in section 12.
- ShanghaiTech.
- A new `schema_version`. New fields go in `metadata`.
- Killing or nicening the resident GB10 jobs. See section 9.

STRETCH (only after the eval table exists): a second rank (r=8 or r=32) as an ablation row; vision-projector LoRA if the smoke log shows it fits; Vietnamese report chapter.

## 3. Data

| Split | Use | Notes |
|---|---|---|
| UCF-Crime train, 1610 videos (800 normal + 810 anomalous) | LoRA supervision + RAFT index | Video-level labels only (Sultani, CVPR 2018, arXiv:1801.04264). Not in this repo yet. |
| UCF-Crime test, 290 videos (150 normal + 140 anomalous) | eval only | Already materialised: `manifests/ucf_test.csv`. Frozen. |
| `calib_normal`, 60 train-normal videos | tau only, unchanged | `manifests/ucf_calib_normal.csv`. tau = 1.1686 stays. These 60 are inside the 1610; the index may contain them; the test set must not. |
| Own demo clips | out | Not benchmark. Do not train on them. |

Disk: the test split plus derivatives already lives in the submitted repo. The train split is the new download. Size is not measured here; check `df` on the GB10 before starting. Embeddings cached to `.npy`, same key as today (model + video_id + sample_fps).

Label rule, unchanged: detection prompts stay generic. Class names ("fighting", "road accident") appear only in the typing stage and in the RAFT target text. Putting a class name in the detection prompt is a label leak.

## 4. System architecture

```
                FORK (new remote). Source main is read-only.
                submitted tier-1 is FROZEN and reused, not retrained

video -> sampler (1 fps) -> SigLIP 2 -> emb[t]          (existing cache, test split)
                                  |
        Tier 1 (unchanged A4c): generic prompt + normal memory
                                robust stats from calib_normal
                                top-k + EMA + median
                                candidate windows, tau = 1.1686
                                  |
        RAFT index (NEW, train split only):
            SigLIP window embeddings of the 1610
            query = candidate window; top-4 neighbours by cosine
            the query video_id is excluded
            a test video_id in this index is a hard failure
                                  |
        Tier 2 (REPLACED):
            Qwen3-VL-8B-Instruct, 8-bit load, LoRA adapter
            input = k frames of the window + retrieved train clips
            output = cite (which clip, what event) OR abstain
            rejected windows never become alerts (rule kept)
                                  |
        Event typer -> Alert Policy Graph (unchanged) -> RunResult JSON
                                  |
        Dashboard: same loader. metadata.trained = true,
                   metadata.adapter_id, metadata.raft_hits
```

What is reused as-is from the submitted repo:

```
vad/types.py            EventType, Window, VerifyResult, Alert
vad/results.py          RunResult, SCHEMA_VERSION = 1, make_metadata(**extra)
vad/pipeline.py         PipelineConfig, ABLATIONS, run_tier1
vad/score/              text prompts, normal memory
vad/temporal/ops.py     candidate windows
vad/eval/metrics.py     micro AUC, video bootstrap, FA/hour
vad/graph/policy.py     YAML -> NetworkX, cooldown, escalation
vad/datasets/ucf_crime.py   GT parser, inclusive frame ranges
app/data.py             loads schema_version 1, drops other versions
```

What is new:

```
vad/verify/vlm.py       + "8bit" branch, + Qwen3-VL class next to Qwen2.5-VL. 4bit/none stay.
vad/train/lora.py       frozen 8-bit base, LoRA on q,k,v,o,gate,up,down
vad/train/raft.py       example builder: golden+distractors, or distractors only -> abstain
vad/index/siglip.py     train-only index, query excludes its own video_id
scripts/build_train_manifest.py     1610 rows, split=train, no id in ucf_test.csv
scripts/build_raft_index.py         writes the index, refuses a test id
scripts/train_lora.py               our loop, not a vendored LlamaFactory
scripts/eval_trained.py             same metrics, writes next to A4c/A6, leak check first
tests                       leak test, 8bit-rejected-on-bad-input test, RAFT mix test
```

`vad/__init__.py` stays empty. Import from the submodule.

## 5. Technical design decisions (with reasons)

1. **Fork, not a branch on the submitted repo.** The graded AUC has to stay reproducible from `main`. A fork that can reset to the submitted commit is the whole point. New remote, source `main` untouched.
2. **Model.** `Qwen/Qwen3-VL-8B-Instruct` (HF, 2025-10-15, Apache-2.0). Official weights are BF16, plus an FP8 repo and GGUF. There is no official bitsandbytes checkpoint. We quantize on load. Needs `transformers>=4.57`. Video defaults from the model card: fps 2, min 4 frames, max 768. We do not adopt max 768; the window budget stays the current k frames.
3. **8-bit, not 4-bit.** The submitted verifier is 4-bit Qwen2.5-VL-3B and that result is already measured. This plan asks for 8-bit LoRA of the 8B. Dropping to 4-bit QLoRA to make a 16 GB card work is a different plan. `vad/verify/vlm.py:135` currently rejects anything except `"4bit"` and `"none"`; `"8bit"` is a new branch, the old two stay.
4. **What is frozen.** 8-bit base frozen. LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. Vision tower frozen. The projector is the OOM risk and stays frozen unless the smoke log says it fits. First config: r=16, alpha=32, dropout 0.05. That is a starting point, not a tuned result.
5. **RAFT mix.** For a fraction P=0.6 of examples the retrieved set contains the golden train clip plus distractors, and the target cites it. For the rest the set is distractors only and the target is abstain (Normal, no citation). This is the paper's golden-or-absent curriculum. P is config, not a result.
6. **Retrieval.** Top-4 train clips by SigLIP window embedding. k=4 matches the current frame budget. Index = train only. The query's own `video_id` is excluded even inside train, so a clip never retrieves itself.
7. **Supervision.** Weakly supervised on all 1610, video-level labels only. Normal-only would not teach the 13-class verifier. Frame-level GT exists for the test split only; do not invent frame labels for train.
8. **Threshold.** tau = 1.1686, source `calib_normal`, unchanged. No threshold is fit on the 290. A trained verifier that wants its own tau gets a new calibration on train-normal held out from the adapter's data, written up as a separate row, never as a replacement.
9. **Metrics.** Micro frame AUC, video-bootstrap 95% CI, FA/hour, window precision, incident recall, paired delta against A6. Frame bootstrap stays forbidden. The leak check runs before the metric code and fails the process.
10. **Schema.** `SCHEMA_VERSION` stays 1. New facts ride in `metadata`: `trained: true`, `baseline: false`, `adapter_id`, `base_model`, `quantization: "8bit"`, `raft_hits` (list of train video ids), `lora_config`. `app/data.py` drops other schema versions, so a bump would hide the run. A trained run must be impossible to mistake for `results/run_A4c/`.
11. **Stack.** `transformers>=4.57`, `peft`, `bitsandbytes`, `trl`. Training entry is `scripts/train_lora.py`. No vendored LlamaFactory: the leak check has to live in our code. `pytest` never downloads a model and never needs a GPU. GPU proof is a smoke log, same rule as the submitted repo.
12. **VRAM bounds are planning numbers, not measurements.** 8-bit inference ~10.4-11.5 GB, 4-bit inference ~5.8-6.5 GB, 4-bit QLoRA ~14-18 GB, 8-bit LoRA ~20-24 GB (librarian fact sheet: StudioTV table, bitsandbytes issue 1780). The first real number replaces them. A 16 GB GPU cannot host this plan; the GB10 can.

## 6. RAFT, specified because nobody has published this recipe for VAD

Per training example:

- Query: one candidate window from a train video, k frames, SigLIP embedding already cached.
- Retrieved: top-4 from the train index, query `video_id` removed.
- With probability P=0.6, force the golden clip in (a train clip of the same video-level class, different `video_id`) and fill the rest with distractors. Target: cite that clip's id and the class.
- Otherwise drop the golden, keep distractors only. Target: abstain. Abstain maps to `Normal` and an empty citation. It must not map to a guessed class.
- Distractors are top-index hits of a different class, not random clips. Random distractors make the task too easy and the citation meaningless.
- At eval time the same retriever runs, but the index is still the train index. The test video is the query and is absent from the index by construction. The leak test checks the construction instead of trusting it.

What the adapter is graded on: does citation-or-abstain change the A6 operating point (FA/hour, window precision, incident recall) without a silent AUC drop being waved through. A gain that comes from a test id inside the index is discarded.

## 7. Evaluation protocol

Unchanged from the submission, on purpose:

- Primary: frame-level ROC-AUC, micro over concatenated frames.
- CI: bootstrap 95% over **videos**, reuse `vad/eval/metrics.py`. Never over frames.
- Operating point: FA/hour on the 150 normal test videos, window precision, incident recall, at tau = 1.1686.
- Paired delta vs A6, with its own CI. The A6 delta was -5.10 [-11.42, -0.25]; a trained run has to publish the same shape of number.
- Typing, if measured: 13-way accuracy on windows that are both confirmed and overlap GT, reported next to the chance line and labelled class-skewed.
- Cost: peak memory, seconds per window, fraction of windows that abstain, fraction that cite a clip which does not support the class.

Rows:

| # | Variant | Role |
|---|---|---|
| A4c | frozen tier-1 | baseline, already measured, 73.00 |
| A6 | Qwen2.5-VL-3B 4-bit, no training | baseline, already measured, 67.90 |
| R0 | Qwen3-VL-8B 8-bit, no LoRA, no retrieval | model swap only |
| R1 | R0 + train-index retrieval, no training | retrieval without gradient |
| R2 | R0 + LoRA, no retrieval | gradient without retrieval |
| R3 | LoRA + RAFT (the system) | the claim |

R0-R2 exist so a gain cannot be attributed to "we used a bigger model" or "we retrieved something" without evidence. If the GB10 gate slips, R0 is the first cut, not R3's leak test.

## 8. Team split

No deadline, so this is ownership, not a day count. Order is the dependency order. Details and acceptance lines live in `notes/2026-11-19-task-assignment-qwen3-vl-lora-raft.md`.

| Owner | Track | Interface |
|---|---|---|
| Phước | Model load, 8-bit smoke. Code owner; GB10 launches go through Thịnh while the box is his | `load_qwen3(quantization="8bit") -> model`, smoke log with peak memory |
| Nam | Train manifest, SigLIP index, RAFT example schema | `index.search(emb, exclude_video_id) -> list[hit]`, manifest with zero test overlap |
| Thanh | LoRA loop, RAFT dataset, held-out train eval | `train_lora(config) -> adapter_dir`, abstain rate on a train holdout |
| Thịnh | Leak test, eval table beside A4c/A6, dashboard metadata, report. Also the GB10 account, for now | `eval_trained` refuses a leaked id; dashboard shows `trained` |

Huy has no track. Do not assign him residual work unless the user adds him back.

Interfaces first: the RAFT example schema and the metadata keys are agreed before Phước's loader and Thanh's loop diverge.

## 9. GB10 gate (load-bearing)

Thịnh holds the box for now. It is not empty: at 2026-11-19 20:10 three resident Python jobs had it at 96% util. Holding the account does not license killing them.

- No GB10 compute until `nvidia-smi` GPU-util stays under 10% for 10 continuous minutes.
- Never kill PID 401001, PID 535418, PID 537304, or any later MATE-2026 / resident job. Never `kill -9` a process you do not own. The draft named older PIDs (356247, 364861, 365064); those are already gone. The rule is about the class of process, not the number.
- If the gate is shut, the worker stops and reports util, temperature, and the resident PIDs. It does not share the GPU, and it does not poll in a loop that keeps the GPU awake.
- One job at a time. Training batch 1. No second smoke "while we wait".
- `free -h` before the run and after. 81 GiB available is enough for an 8-bit 8B plus the cache; the check is there so a runaway resident job is noticed before ours starts.
- First GB10 action is the smoke log (load 8-bit, one window, peak memory). That log replaces the planning VRAM bounds. No full run before it exists.

## 10. Risks

| Risk | P | Mitigation |
|---|---|---|
| Resident jobs never drop under 10% | high | Gate stops the worker. Report, do not queue. No rental budget, so waiting is the only move. |
| 8-bit LoRA OOM despite the 20-24 GB bound | med | Smoke log first. Cut the projector (already frozen) before touching rank. Do not silently switch to 4-bit. |
| Test video leaks into the index | high impact, low if tested | Failing test, run before metrics. A leaked run is discarded, not footnoted. |
| RAFT citation learns the class prior, ignores the clips | med | R1/R2/R3 split. Abstain-only slice reported separately. |
| Bigger model beats A6 with no training, so the LoRA claim is empty | med | R0 is mandatory. If R0 already wins, say so. |
| `transformers>=4.57` breaks the Qwen2.5-VL loader the baseline needs | med | Fork only. Source repo pins stay. Fork CI runs the old verifier tests with the model mocked. |
| bitsandbytes on CUDA 13.0 / GB10 | med | Smoke log is also the compatibility test. Pinned version goes in the note. |
| Examiner: "you trained, so the old claim is void" | high | Both claims live. Baseline table unchanged, new table labelled trained. |
| Examiner: "RAFT is optical flow" | med | Named as Berkeley arXiv:2403.10131 in the first paragraph of the method. |

## 11. Definition of done

- [ ] Fork remote exists. Source `main` still reproduces 73.00 and 67.90 from `results/`.
- [ ] `quantization="8bit"` loads `Qwen/Qwen3-VL-8B-Instruct`. `"4bit"` and `"none"` still work. Anything else raises.
- [ ] Smoke log: one window, peak memory, driver, bitsandbytes version, commit.
- [ ] `manifests/ucf_train.csv` has 1610 rows and zero `video_id` overlap with `manifests/ucf_test.csv`.
- [ ] Index built from train only. Query excludes its own `video_id`. Leak test fails the process on a planted test id.
- [ ] LoRA adapter trained with the RAFT mix (P=0.6). Config recorded next to the adapter.
- [ ] Eval table with R0, R1, R2, R3 beside A4c and A6, video-bootstrap CIs, paired delta vs A6.
- [ ] Dashboard loads the trained `RunResult`, `schema_version` 1, and shows a trained label plus the cited clip ids.
- [ ] `python -m pytest -q` green with no model download and no GPU.

## 12. Decisions log

| # | Question | Answer |
|---|---|---|
| 1 | Where does the revamp live? | New remote forked from this repo. Source `main` read-only. |
| 2 | Which RAFT? | Berkeley retrieval-augmented fine-tuning, arXiv:2403.10131. Not optical flow. Reversible if the user says otherwise. |
| 3 | Which model, which precision? | `Qwen/Qwen3-VL-8B-Instruct`, 8-bit load, LoRA. Not 4-bit QLoRA. |
| 4 | Train host? | `gx10@aitopatom-d4ac`, NVIDIA GB10, CUDA 13.0, 121 GiB unified. Tailscale only. Thịnh holds it for now (2026-11-19). |
| 5 | May we use the GPU while it is busy? | No. Under 10% util for 10 minutes, else stop. Never kill resident PIDs. |
| 6 | Who? | Thịnh, Nam, Phước, Thanh. Huy out. |
| 7 | Schema? | `schema_version` stays 1. New fields in `metadata`. |
| 8 | Deadline? | None. |

## 13. Open questions

1. SETTLED 2026-11-19. Thịnh has the GB10 for now. Long-term owner and who else gets a key stay open; until then GPU launches go through him.
2. Does a report deadline exist for this revamp? None assumed. Ask only if a date exists.
3. RAFT = Berkeley paper unless the user says optical flow. Default stands.
