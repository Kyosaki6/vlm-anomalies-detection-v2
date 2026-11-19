# scripts/

**Parent:** `../AGENTS.md`. 20 CLIs. Not installed as console_scripts — run `python scripts/<name>.py`. `scripts/__init__.py` exists only so tests can import helpers.

## OVERVIEW
Each script is one stage. They share a manifest CSV and the embedding cache. They do not call each other, except three that import `run_benchmark` (see CONVENTIONS).

## WHERE TO LOOK
| Script | Writes | Do not |
|--------|--------|--------|
| `download_ucf_crime.py` | UCF test + calib-normal videos | treat as a test dependency |
| `build_manifest.py` | `manifests/*.csv` | emit absolute paths unless `--allow-absolute` |
| `precompute_embeddings.py` | `cache/` `.npy` per `--model` | mix fps/model across shards |
| `run_benchmark.py` | `results/ablation_A*.json` | calibrate τ here; it only scores |
| `calibrate_threshold.py` | threshold JSON from `calib_normal` | fit τ on the test split |
| `export_windows.py` | `results/windows/<id>.json` + `_index.json` | hand-edit windows after export |
| `run_verify.py` | `results/verify/<id>/window_<i>.json` | point `--fake` at `results/verify` |
| `build_results.py` | `results/run_A4c/<id>.json` + alerts | let a rejected window become an alert |
| `eval_tier2.py` | tier-2 metrics JSON | drop `--no-paired` and still quote a delta |
| `eval_vlm_typing.py` | typing accuracy on confirmed∩GT windows | quote 83.9% as all-event accuracy |
| `eval_typer.py` | zero-shot typer accuracy | compare against a gallery that contains the query |
| `make_figures.py` | `results/figures/` | pick frames by eye; selection is rule-based |
| `audit_report_numbers.py` | stdout audit of `report/` numbers | "fix" a mismatch by editing the report to match a stale JSON |
| `bench_vlm.py` | VLM timing JSONL | run inside CI; needs a GPU and real windows |
| `a0_stride_check.py` | stride-16 A0 check | re-roll the recorded A0 miss |
| `draw_a6_sample.py` | blind hand-check CSV | commit the answer key |
| `fetch_cache.py` / `fetch_demo.py` | `cache/`, `datasets/own/` | skip the SHA-256 check |
| `make_sample_result.py` | `results/sample_run.json` | present it as a measured run |
| `make_bench_windows.py` | 16-window VLM bench set | mix it into the 290-video AUC |

## CONVENTIONS
- Read manifests and JSON with `utf-8-sig` (`make_sample_result.py` carries a BOM).
- `--dry-run` must not write cache, results, or download weights.
- `--shard i/N` is SHA-256 stable (`vad/datasets/sharding.py`). Same spec on every machine.
- Resume is by fingerprint (window + k + model + quantization + prompt hash), not by mtime.
- Stages talk through files (`cache/*.npy` -> `results/windows/` -> `results/verify/` -> `results/run_A4c/`). Three exceptions import a sibling: `calibrate_threshold.py`, `export_windows.py`, `eval_tier2.py` do `import run_benchmark as rb`.

## ANTI-PATTERNS
- Do not add a script that both chooses τ and reports test AUC. Split stays `calibrate_threshold` then `run_benchmark` / `eval_tier2`.
- Do not default `--out-dir` to a path that already holds `fake=false` results.
- `run_verify.py --fake` refuses `--out-dir results/verify`. `eval_tier2.py --allow-fake` writes `ablation: "A6-fake"` and refuses the default `results/ablation_A6.json`. Keep both guards.
- `calibrate_threshold.py` rejects `norm_ref == "video"` (A3/A4/A5). A per-video median of 0 makes one global τ meaningless.
- `eval_tier2.py` follows `notes/a6-protocol.md`. Do not change the metric after reading verdicts.
- `run_benchmark.py --label-shift` / `--logit-scale` require `--tag`, so a diagnostic run cannot overwrite `ablation_A*.json`.
- Do not document a flag that the argparse block does not have. The flag list is the contract tests lock.

## NOTES
- Large files: `build_results.py` (738), `eval_vlm_typing.py` (663), `run_verify.py` (659), `bench_vlm.py` (650). Contract tests: `tests/vad_suite/test_build_results.py`, `test_run_verify.py`.
- Pipeline order for a full rerun is the table top to bottom through `build_results.py`. Eval scripts are read-only on `results/`.
