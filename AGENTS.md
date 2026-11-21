# PROJECT KNOWLEDGE BASE

**Generated:** 2026-11-19 · **Commit:** `eadf000` · **Branch:** main · Plan: `notes/2026-10-28-plan-zero-shot-video-vad.md`

## OVERVIEW
One codebase, `vad/`:

- **`vad/`** — Zero-Shot Video Anomaly & Surveillance Alerting. Video -> SigLIP 2 frame embeddings
   (cached) -> generic-prompt + normal-memory scoring -> temporal smoothing -> candidate windows -> generative VLM
   verification (tier 2, mandatory: the topic was registered as VLM) -> alert policy graph -> Streamlit dashboard.
   Benchmark: UCF-Crime test 290 videos, frame-level AUC with bootstrap CI over **videos**. ShanghaiTech is out of
   scope (time cut, D4-1 NO-GO) — do not claim results for it.

Submitted 2026-11-04 (historical). Team = 4 (Phước, Huy, Nam, Thịnh). Dashboard owner: Thịnh.
Report + defense in Vietnamese, dashboard in English. Scope cuts: `notes/2026-10-29-task-assignment.md` §6.
Child docs: `vad/AGENTS.md`, `scripts/AGENTS.md`, `tests/vad_suite/AGENTS.md`. `vad/README.md` module table is stale (predates tier-1/tier-2 runs) — trust code + `results/`, not that table.

## STRUCTURE
```
vad/            types.py, interfaces.py (Protocols), fakes.py  — contracts shared by all tracks
  results.py    RunResult JSON schema -> results/<video_id>.json (this is what the dashboard reads)
  datasets/     ucf_crime.py (GT parser), sharding.py (SHA-256 shard split)
  io/           video.py (sequential-grab sampler)
  embed/        cache.py (fp16 .npy, atomic), siglip.py (real, model-generic: SigLIP 2 + CLIP)
  score/        text.py (prompt + memory scores, numpy)
  temporal/     ops.py (normalize/EMA/median/top-k/candidate windows/frame alignment)
  verify/       vlm.py (real VLMVerifier: Qwen2.5-VL, 4-bit, chunk <= 30 s; plus the strict JSON parser)
  graph/        policy.py (YAML -> NetworkX alert policy) ; configs/policy.yaml
  eval/         metrics.py (frame-level AUC, video-level bootstrap, FA/hour, latency)
scripts/        20 CLIs, not installed as console_scripts. Order: precompute_embeddings ->
                run_benchmark -> calibrate_threshold -> export_windows -> run_verify ->
                build_results -> eval_tier2 / eval_vlm_typing / make_figures.
                Also: download_ucf_crime, build_manifest, fetch_cache, fetch_demo, bench_vlm,
                audit_report_numbers, a0_stride_check, draw_a6_sample, eval_typer.
app/            data.py (loads RunResult, NO streamlit import) + streamlit_app.py
                (real results/run_A4c/, Replay + Overview). `default_results_dir()` prefers run_A4c/.
configs/        policy.yaml, prompts.yaml — no pyproject.toml, no pytest.ini, no ruff/mypy config
tests/vad_suite/ 35 modules. Root conftest.py only prepends repo root to sys.path.
notes/          experiment logs + the plan. report/ Vietnamese chapters.
results/        committed JSON (fake=false). Not a code package.
```

## WHERE TO LOOK
| Task | Location |
|---|---|
| Module map, public symbols | `vad/AGENTS.md` (README module table is stale) |
| Full plan, timeline, risks, decisions log | `notes/2026-10-28-plan-zero-shot-video-vad.md` |
| Detection / typing / verifier prompts | `configs/prompts.yaml` |
| Alert rules (zones, time bands, escalation) | `configs/policy.yaml` |
| UCF-Crime annotation parsing | `vad/datasets/ucf_crime.py` |
| ROC-AUC with ties, video bootstrap, FA/hour | `vad/eval/metrics.py` |
| Tier-1 ablation switches A0..A5 | `vad/pipeline.py` `ABLATIONS` / `PipelineConfig` |
| RunResult schema the dashboard reads | `vad/results.py` |
| Dashboard loader (no streamlit) | `app/data.py` |
| Which CLI writes which path | `scripts/AGENTS.md` |

## CODE MAP

Python LSP not installed. Inventory from `ast` on `vad/**/*.py` (73 public top-level symbols). Ref counts are import-sites from `scripts/`, `app/`, `tests/` — not LSP references.

| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| `RunResult` | class | `vad/results.py:131` | 17 | JSON contract the dashboard and build_results share |
| `EventType` | class | `vad/types.py:13` | 18 | 13-class label enum; strings, not StrEnum |
| `Window` | class | `vad/types.py:56` | 17 | candidate interval tier-1 emits, tier-2 verifies |
| `VerifyResult` | class | `vad/types.py:69` | 15 | VLM verdict; rejected windows never become alerts |
| `Alert` | class | `vad/types.py:78` | 7 | policy output; cooldown keyed by camera |
| `load_annotations` | func | `vad/datasets/ucf_crime.py:59` | 14 | UCF GT parser; inclusive frame ranges |
| `PipelineConfig` | class | `vad/pipeline.py:42` | 6 | every ablation switch; A0..A5 differ only here |
| `SigLIPEmbedder` | class | `vad/embed/siglip.py:55` | 7 | real embedder, SigLIP 2 and CLIP-B/32, one class |
| `VLMVerifier` | class | `vad/verify/vlm.py:104` | — | real tier-2; raises, never invents a verdict |
| `GraphPolicy` | class | `vad/graph/policy.py:72` | 6 | YAML to NetworkX alert policy |
| `candidate_windows` | func | `vad/temporal/ops.py:67` | 6 | thresholded samples to windows, gap 5 s |
| `ZeroShotEventTyper` | class | `vad/score/typing.py:64` | 6 | 13-class cosine typer; typing stage only |
| `FakeEmbedder` / `FakeVerifier` | class | `vad/fakes.py` | 5 / 8 | GPU stand-ins; tests and `--fake` only |

Import hub: `vad/types.py`. Subpackages barely import each other; `vad/pipeline.py` is the composer. `vad/__init__.py` is empty — import from the submodule, not `vad`.

## NO STUBS LEFT — what is already measured (as of 2026-11-02, main `04cf958`)

Everything in the pipeline is real: `SigLIPEmbedder` (model-generic SigLIP2 + CLIP), `VLMVerifier`
(Qwen2.5-VL-3B-Instruct, 4-bit), `scripts/run_benchmark.py`, `build_results.py`, and the dashboard reading real
`results/`. Keep `NotImplementedError` for any *future* missing backend; never return fake output.

Committed results (all `fake=false`, reproducible from `results/`):

- Tier 1 A4c, τ = 1.1686 (calib_normal): micro AUC **73.00** [64.45, 81.15], 307 candidate windows in 114 of 290 videos.
- Tier 2 A6: 307/307 windows verified, 129 confirmed (42.0%), AUC **67.90** [58.39, 77.49], paired delta **-5.10**
  [-11.42, -0.25]. FA/hour 20.64 -> 0.17; window precision 28.0% -> 49.6%; incident recall 51.9% -> 39.7%.
- `results/run_A4c/`: 290 RunResult JSON + 290 window files, 101 alerts in 72 videos, `metadata.tier2` = "json" for
  114 videos and "none" for 176.
- Typing (C7e, 62 windows confirmed AND overlapping GT): VLM 83.9% vs zero-shot typer 24.2% vs chance 7.7%.
  Conditional and class-skewed sample — never quote as "VLM types 84% of UCF events".
- A0 validity gate: 48.77 [38.30, 59.40], below the pre-registered 50.2-56.2 band around LAVAD's 53.16. Reported
  as a miss, not papered over (`notes/a0-gate.md`).


## CONVENTIONS
- Python 3.10 compatible (CI runs 3.10 + 3.11 on Ubuntu + Windows). No `StrEnum`; use `from __future__ import annotations`.
- Read JSON/text with `utf-8-sig` (Windows BOM tolerance). Event ids are strings.
- Code comments/docstrings are bilingual: Vietnamese prose, technical terms verbatim.
- Tests never download models or need a GPU; anything GPU-bound has a `Fake*` in `vad/fakes.py`.
- Test dir is `tests/vad_suite/` — NOT `tests/vad/` (that name shadows the `vad` package under pytest).
- Embedding cache keyed by model + video_id + sample_fps; every shard must use the same model, fps, preprocessing.

## ANTI-PATTERNS (THIS PROJECT)
- DO NOT put class names ("fighting", "road accident") in detection prompts: label leak. Typing stage only.
- DO NOT calibrate thresholds on the test split; use `split=calib_normal`.
- DO NOT report CIs resampled over frames; resample over videos (frames are autocorrelated).
- DO NOT use top-1 gallery similarity as anomaly probability (legacy finding; AUC ~0.64).
- DO NOT measure retrieval with the query inside the gallery (false 100%).
- DO NOT claim zero-shot for ShanghaiTech: label it "training-free + normal reference".
- DO NOT pretend a stub works. A missing backend must raise, so the dashboard can never show fabricated results.
- DO NOT use bare ignore patterns like `cache/`: they swallow `vad/embed/cache.py`. Anchor to root (`/cache/`).
- DO NOT implement from `notes/2026-10-28-plan-*.md` alone. `notes/2026-10-29-day1-tasks.md` is newer and
  carries requirements the older plan lacks (e.g. the embedder must be model-generic: SigLIP2 AND CLIP-B/32,
  one decode pass). Read the newest task sheet before writing code, not after.
- DO NOT test only the default option of a config. If a parameter claims to support N choices, one test must
  use a non-default one. `padding="max_length", max_length=64` is SigLIP-only; CLIP breaks on it.
  A test asserting the constant back to itself is decoration, not a test.
- DO NOT validate only sane-but-wrong input. Also reject hostile input: NaN/Inf scores, `video_id` that is a
  path (`../../x`), empty arrays. These produce corrupt or misplaced files, not exceptions.

## COMMANDS
```bash
pip install -r requirements-dev.txt        # tests / CI
pip install -r requirements-app.txt        # + streamlit
python -m pytest -q                        # 725 passed, no GPU, no model download
streamlit run app/streamlit_app.py         # dashboard on real results/run_A4c/ (290 videos)
python scripts/precompute_embeddings.py --manifest manifests/<f>.csv --shard 2/4 --dry-run
```

## HARDWARE (shard planning)
Phước RTX 3050 6 GB (fp16, also VLM) · Huy RTX 3050 4 GB (fp16) · Nam GTX 1060 3 GB (fp32; check `sm_61` in torch) ·
Thịnh no GPU (CPU + Colab T4). Heavy runs = embedding cache + tier-2 batch only; everything downstream is CPU.

## NOTES
- Published zero-shot frame-level AUC on UCF-Crime (LAVAD, CVPR 2024): ZS-CLIP 53.2, LLaVA-1.5 72.8, LAVAD 80.3.
  Naive per-frame prompting is near chance; the contribution is the pipeline (normalisation, temporal, memory, VLM).
- Day-4 validity gate: RAN (A0 = 48.77, see `notes/a0-gate.md`). It missed the pre-registered band; that is the
  recorded outcome, do not re-run it hoping for a different number.
- UCF-Crime annotation segments are INCLUSIVE `[start, end]`, 0-indexed, 30 fps. Convert to seconds as the
  half-open interval `[start/30, (end+1)/30)`; `end/30` drops the last frame and silently changes which windows
  count as overlapping (`notes/ucf-gt-convention.md`, C7e).
- Do not demo with `streamlit run app/streamlit_app.py -- --demo`: `app/data.py::demo_run` is synthetic scores
  plus a fake verifier, labelled "DEMO DATA" by the app. Present the real `results/run_A4c/` instead.


