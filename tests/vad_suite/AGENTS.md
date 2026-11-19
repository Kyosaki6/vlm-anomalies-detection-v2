# tests/vad_suite/

**Parent:** `../../AGENTS.md`. 35 modules. CI runs `python -m pytest tests/vad_suite -q` on Ubuntu + Windows, Python 3.10 and 3.11. No pytest.ini, no markers, no GPU job.

## OVERVIEW
One test module per `vad/` unit or per `scripts/` CLI. Root `conftest.py` only inserts the repo root on `sys.path` so `import vad` and `import scripts` both work.

## WHERE TO LOOK
| Change | Test module |
|--------|-------------|
| `vad/results.py` schema | `test_results.py`, `test_build_results.py` |
| `scripts/run_verify.py` | `test_run_verify.py`, `test_run_verify_clamp.py` |
| `vad/pipeline.py` ablations | `test_pipeline.py`, `test_run_benchmark.py` |
| `vad/verify/vlm.py` parser | `test_vlm_verifier.py` (mock runner, no weights) |
| `vad/embed/siglip.py` | `test_siglip_embedder.py` |
| `app/data.py` | `test_app_data.py` |
| contracts (`Window`, `VerifyResult`) | `test_contracts.py` |
| AUC / bootstrap | `test_metrics_roc.py` |

## CONVENTIONS
- Directory name is `vad_suite`, not `vad`. `tests/vad/` would shadow the package under pytest.
- GPU stand-ins are `FakeEmbedder` and `FakeVerifier` from `vad/fakes.py`. Construct them in the test. Do not hit the network.
- A config knob with N values needs one test on a non-default value.
- Hostile inputs are in scope: NaN/Inf scores, `video_id` containing `..`, empty arrays.

## ANTI-PATTERNS
- Do not download weights, call Hugging Face, or require CUDA. CI installs CPU torch on Linux only.
- Do not assert a constant equal to itself (`assert DEFAULT == DEFAULT`). That is not a test.
- Do not read `results/run_A4c/` as an oracle inside a unit test. Build the fixture in `tmp_path`.
- Do not add `pytest.mark.gpu` or skip-if-no-cuda as a way to land a real-model test. Those tests do not exist and CI will not grow a GPU job for them.

## NOTES
- Largest modules are contract tests, not unit tests: `test_build_results.py` (957 lines), `test_run_verify.py` (866). Edit the script and the test together.
- `test_app_data.py` covers the loader only. The Streamlit UI is not under pytest.
- Two skips already exist. Leave them: `test_siglip_embedder.py` skips the CPU-fallback case when CUDA is present, and `test_video_io.py` does `importorskip("cv2")`. Neither is a license to download weights.
