# phase4 - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**Who this is for and what changes for them:** All four. The trained model was tested on data it had not seen and scored 2% where guessing the most common answer scores 26%, so the big evaluation is not allowed to start. Each person owns the piece of that failure that sits in their earlier work, plus the report and the dashboard, which were always the last phase.

**What you'll get:** Phước finds out whether the model is being shown the answer during training. Nam checks the search index returns clips from the right class. Thanh owns the decision to retrain, and writes down why the last run failed. Thịnh writes the report section and finishes the dashboard, and his evaluation command refuses to run until the retest passes.

**Why this approach:** The assignment gave this phase to one person because it assumed the training had worked. It did not. Sending the failure back to the person who owns each piece is faster than having one person guess at all of them, and the report can be written now because "it lost, here is why" is a real result.

**What it will NOT do:**
- It will not run the 290-video evaluation.
- It will not retrain the model. That needs its own go-ahead once the diagnosis is in.
- It will not change the frozen baseline numbers or the submitted report chapters.

**Effort:** Large
**Risk:** Medium - the diagnosis may come back "nothing is wrong, the recipe just loses", and that is an acceptable answer this plan records rather than papers over.
**Decisions to sanity-check:** the work is split four ways, nobody retrains in this phase, and the report says the run lost instead of leaving the cell blank.

Your next move: run it with `$ulw-execute`, or ask for a high-accuracy review first. Full detail follows.

---

> TL;DR (machine): Large, medium risk. Eight todos, four owners. Diagnosis of the 0.0211-vs-0.2632 holdout miss split across Phước (prompt), Nam (index), Thanh (retrain decision), Thịnh (report, dashboard, gated eval). No retrain, no R3 number. Baseline pytest count: 812 passed.

## Why this phase is split four ways

`notes/2026-11-19-task-assignment-qwen3-vl-lora-raft.md` puts only H2 and H4 in P4, both Thịnh's. That split assumed T4 passed. It did not: citation accuracy 0.0211 against a class prior of 0.2632, and the stop rule fired. The failure crosses three owners, so the phase returns to them.

| Person | Already shipped | Open now | Why it is theirs |
| --- | --- | --- | --- |
| Phước | P1 loader, P2 smoke, P3 serve path | the prompt the adapter trained on | `prompt_text` carries no class names, and the adapter that was scored never saw it. His loader, his prompt |
| Nam | N1 manifest, N2 index, N4 embeddings | whether the index returns same-class neighbours | a golden clip of the wrong class makes the citation task unsolvable. His index |
| Thanh | T1 RAFT mix, T2 loop, T3 train, T4 holdout | the retrain decision and the failure note | he owns the loop, and the assignment names him the risk owner for "the adapter predicts the class prior" |
| Thịnh | H1 leak test, H3 badge | H4 report, H3 cited clips, H2 gate | the report and dashboard were his regardless, and the eval must stay shut |

GPU time stays serial and stays with Thịnh. Nothing in this phase launches a job.

## Scope
### Affected user and ideal state
**Affected user:** the four owners above. The reader of the report, who needs to see that the trained run lost and why.

| Row | Statement | Reason |
| --- | --- | --- |
| IS-1 | each owner has a written diagnosis of their own piece, keyed to the 0.0211 result | a single shared "look into it" produces one vague note |
| IS-2 | the retrain happens only after those notes exist, and not inside this phase | retraining blind repeats the miss |
| IS-3 | the report section names Berkeley RAFT (arXiv:2403.10131) in its first paragraph and carries rows A4c, A6, R0, R1, R2, R3 | H4's acceptance line |
| IS-4 | A4c and A6 equal their JSON, R0-R2 say not measured, R3 says not run with the holdout numbers | frozen baselines, and a blank cell reads as a forgotten run |
| IS-5 | the new section never says zero-shot | H4 forbids it for the trained run |
| IS-6 | the dashboard shows the cited clip ids on a trained run and nothing on an untrained one | H3's second half |
| IS-7 | the eval command exits without scoring while the holdout note says it lost | the stop rule has to live in the script |
| IS-8 | no model download, no peft import at module level, no `results/trained/R3.json` | CI has no GPU and the gate is shut |
| GAP-1 | no per-owner diagnosis of the miss | the phase 3 report states the number and stops |
| GAP-2 | no report text for the trained run | `report/` ends at chapter 6 |
| GAP-3 | cited clip ids never reach the screen | `trained_badge` returns the word and drops the list |
| GAP-4 | `eval_trained.py` never reads the holdout note | it leak-checks, then raises |

### Must have
- One diagnosis note per owner, each pointing at a file they own and at the holdout number.
- `report/04b-trained.md`, Vietnamese, six-row table, generated rather than typed.
- `cited_clips(run)` shown in the Replay tab.
- `eval_trained.py` exiting 4 on the current `notes/holdout.md`.
- A test per todo that fails when its guard is removed.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- No retrain, no GPU launch, no second job on the GB10.
- No edit to `report/01` through `report/06` or `report/refs.md`.
- No `results/trained/R3.json` and no invented metric for any R row.
- No change to tau, `results/ablation_A4c.json`, `results/ablation_A6.json`, or `SCHEMA_VERSION`.
- No "zero-shot" in the new section. No class name added to a detection prompt.
- No peft, bitsandbytes, or torch import at module level.
- No push, no commit unless the user asks.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: tests-after, pytest, for the four code todos. The three diagnosis notes are checked by a test that asserts the required fields are present, not by reading them for style.
- Evidence: `.omo/evidence/task-<N>-phase4.txt`.
- Full suite gate: `python -m pytest -q` stays green. Baseline 812 passed.

## Execution strategy
### Parallel execution waves
> Todos in one wave do not read each other and do not write the same file. This is also the team split: a wave is work that can happen on the same day.

- Wave 1: todos 1, 2, 3, 4. Phước, Nam, Thanh's inputs, Thịnh. Four files, no shared write.
- Wave 2: todos 5, 6, 7. Todo 5 and todo 6 are both Thịnh's but different files. Todo 7 is Thanh's and only reads the two diagnosis notes.
- Wave 3: todo 8. Renders the report file from todo 5. One writer.

### Dependency matrix
| Todo | Owner | Writes | Depends on | Blocks |
| --- | --- | --- | --- | --- |
| 1 | Phước | `notes/prompt-gap.md`, `tests/vad_suite/test_prompt_gap.py` | none | 7 |
| 2 | Nam | `notes/index-neighbours.md`, `tests/vad_suite/test_index_neighbours.py` | none | 7 |
| 3 | Thịnh | `vad/train/verdict.py`, `tests/vad_suite/test_verdict.py` | none | 6 |
| 4 | Thịnh | `app/trained.py`, `app/streamlit_app.py`, `tests/vad_suite/test_trained_badge.py` | none | none |
| 5 | Thịnh | `vad/report/trained.py`, `tests/vad_suite/test_report_trained.py` | none | 8 |
| 6 | Thịnh | `scripts/eval_trained.py`, `tests/vad_suite/test_eval_trained.py` | 3 | none |
| 7 | Thanh | `notes/retrain-decision.md` | 1, 2 | none |
| 8 | Thịnh | `report/04b-trained.md` | 5 | none |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [ ] 1. Phước, `notes/prompt-gap.md`: state whether `prompt_text` shows the adapter the class it must cite - expect the note to quote the prompt and say the 2026-11-25 adapter never saw it
  What to do: read `vad/train/citation.py` `prompt_text` and the train path, and write a note with three fields: the prompt text, whether a class name appears in it, and whether the adapter scored on 2026-11-26 was trained on it. The answer to the third is no, and the note says so. Recommend one change, not three.
  Must NOT do: do not edit `prompt_text` in this todo. Do not retrain. Do not add a class name to a detection prompt.
  Closes: GAP-1
  Parallelization: Wave 1 | Blocked by: none | Blocks: 7
  References: `vad/train/citation.py:18` (`prompt_text`), `vad/train/citation.py:23` (the comment that the 2026-11-25 adapter never saw this prompt), `notes/2026-11-21-phase3-report.md:48` (the run that scored 0.0211).
  Acceptance criteria: `python -m pytest -q tests/vad_suite/test_prompt_gap.py` exits 0. The test reads the note and asserts it contains the prompt string, the literal `2026-11-25`, and exactly one recommended change.
  QA scenarios: happy - note names the prompt and the one change. failure - a note that recommends nothing fails the test. Evidence: `.omo/evidence/task-1-phase4.txt`.
  Commit: Y | docs(train): record the prompt gap behind the holdout miss
  Recommended task executor category: writing - the output is a diagnosis, and the test checks the fields.

- [ ] 2. Nam, `notes/index-neighbours.md`: measure how often a query's top neighbour shares its class - expect a fraction computed from the index, not estimated
  What to do: over the train catalog, for each video take its top index hit after removing its own id, and count the share whose `event_class` matches the query. Write the fraction, the count, and the class with the worst share. Use `vad/index/siglip.py` and the catalog `vad/train/raft.py` already builds.
  Must NOT do: do not rebuild the index. Do not touch the test split. Do not change retrieval.
  Closes: GAP-1
  Parallelization: Wave 1 | Blocked by: none | Blocks: 7
  References: `vad/index/siglip.py` (`search`, `exclude_video_id`), `vad/train/raft.py:37` (`Clip.event_class`), `scripts/build_raft_index.py` (how the index was built).
  Acceptance criteria: `python -m pytest -q tests/vad_suite/test_index_neighbours.py` exits 0. The test builds a four-clip fake index where one query's neighbour is same-class and one's is not, and asserts the computed fraction is exact. A second assertion checks the note contains a fraction and a class name.
  QA scenarios: happy - fake index, exact fraction. failure - a query that retrieves itself fails the test. Evidence: `.omo/evidence/task-2-phase4.txt`.
  Commit: Y | docs(index): measure same-class neighbour rate
  Recommended task executor category: quick - one measurement over an existing index, one test on a fake one.

- [ ] 3. Thịnh, `vad/train/verdict.py`: add `eval_allowed(note)` that returns false unless the holdout beat the class prior - expect false for `notes/holdout.md`
  What to do: parse `citation_accuracy` and `class_prior` from the note. True only when both are present and differ by more than 0.01. Missing file, missing row, or a smaller gap returns false. Same threshold `vad/train/notes.py` writes with.
  Must NOT do: do not recompute the accuracy. Do not raise on a missing file.
  Closes: GAP-4
  Parallelization: Wave 1 | Blocked by: none | Blocks: 6
  References: `notes/holdout.md` (`citation_accuracy | 0.0211`, `class_prior | 0.2632`), `vad/train/notes.py` (the writer and the one-point rule), `vad/train/gate.py` (the function shape to copy).
  Acceptance criteria: `python -m pytest -q tests/vad_suite/test_verdict.py` exits 0. Real note returns false. Synthetic 0.40 vs 0.26 returns true. Synthetic 0.265 vs 0.263 returns false. Missing path returns false.
  QA scenarios: happy - a note that clears the bar returns true. failure - the committed note returns false. Evidence: `.omo/evidence/task-3-phase4.txt`.
  Commit: N | bundled with todo 4
  Recommended task executor category: quick - one parser, four cases.

- [ ] 4. Thịnh, `app/trained.py` + `app/streamlit_app.py`: show the cited clip ids under the TRAINED badge - expect the line only when the run is trained and has hits
  What to do: add `cited_clips(run)` returning `metadata.raft_hits` as a tuple of strings, and only when `is_trained(run)` is true. A hit containing `..` or a path separator raises `ValueError`. Add `cited_caption(run)` returning `Cited clips: a, b` or `""`. Call it in `_replay_tab` after the caption at line 82, rendering a caption only when the string is non-empty.
  Must NOT do: do not import streamlit into `app/trained.py`. Do not change `is_trained` or `trained_badge`. Do not show the line for an untrained run.
  Closes: GAP-3
  Parallelization: Wave 1 | Blocked by: none | Blocks: none
  References: `app/trained.py:14` (`is_trained`), `app/streamlit_app.py:82` (the caption), `vad/results.py:74` (`make_metadata(**extra)` already carries the key, no schema bump), `tests/vad_suite/test_trained_badge.py` (extend it).
  Acceptance criteria: `python -m pytest -q tests/vad_suite/test_trained_badge.py` exits 0. Trained run with two hits captions both. Trained run without hits captions nothing. Untrained run with hits captions nothing. A hit of `../x` raises `ValueError`.
  QA scenarios: happy - two ids, both named. failure - `raft_hits` as a bare string raises. Evidence: `.omo/evidence/task-4-phase4.txt`.
  Commit: Y | feat(app): show cited clips and gate the eval on the holdout note
  Recommended task executor category: quick - one helper, one call site.

- [ ] 5. Thịnh, `vad/report/trained.py`: add `render_section(a4c, a6, holdout)` returning the Vietnamese section - expect six rows and the arXiv id in the first paragraph
  What to do: pure function, no IO. First paragraph says the fork is trained, names Berkeley RAFT, and contains `arXiv:2403.10131`. The table rows are A4c, A6, R0, R1, R2, R3. A4c and A6 take `auc_micro`, `ci_lo`, `ci_hi` from the dicts, as percentages to two decimals. R0, R1, R2 render `not measured`. R3 renders `not run` with the citation accuracy and class prior from the holdout dict. A missing holdout number raises `ValueError`.
  Must NOT do: do not read a file. Do not write `zero-shot`. Do not invent a CI for R3.
  Closes: GAP-2
  Parallelization: Wave 2 | Blocked by: none | Blocks: 8
  References: `notes/2026-11-19-task-assignment-qwen3-vl-lora-raft.md:98` (H4), `results/ablation_A6.json:21` (field names), `report/04-thuc-nghiem.md:44` (table shape to follow, not edit).
  Acceptance criteria: `python -m pytest -q tests/vad_suite/test_report_trained.py` exits 0. Real A4c and A6 files in, the A6 cell matches the file, all six labels present, `arXiv:2403.10131` present, `zero-shot` absent. Missing `citation_accuracy` raises.
  QA scenarios: happy - real A6 file, correct cell. failure - holdout dict without the accuracy raises. Evidence: `.omo/evidence/task-5-phase4.txt`.
  Commit: Y | feat(report): render the trained-run section from the sources
  Recommended task executor category: writing - prose output, test locks values and the banned word.

- [ ] 6. Thịnh, `scripts/eval_trained.py`: exit 4 when `eval_allowed` is false, before any scorer runs - expect no `R3.json` on the current note
  What to do: add `--holdout-note`, default `notes/holdout.md`. After the leak check, call `eval_allowed`. False exits 4 with the reason and writes nothing. The injected `scorer_fn` still wins when the gate is open. With no injected scorer and an open gate, raise `NotImplementedError` as today.
  Must NOT do: do not import peft. Do not catch the `NotImplementedError`. Do not write the output file on the exit-4 path.
  Closes: GAP-4
  Parallelization: Wave 2 | Blocked by: 3 | Blocks: none
  References: `scripts/eval_trained.py:70` (leak check stays first), `vad/train/verdict.py` (todo 3), `tests/vad_suite/test_eval_trained.py` (existing cases must keep passing).
  Acceptance criteria: `python -m pytest -q tests/vad_suite/test_eval_trained.py` exits 0. Real note, no scorer: exit 4, no file, `peft` not imported. Synthetic clearing note, no scorer: `NotImplementedError`. Clearing note plus injected scorer: file written. Planted test id: still non-zero, no file.
  QA scenarios: happy - injected scorer and an open gate writes the file. failure - the real note writes nothing. Evidence: `.omo/evidence/task-6-phase4.txt`.
  Commit: N | lands with todo 4 if ready, otherwise its own `feat(eval): gate the trained eval on the holdout note`
  Recommended task executor category: deep-low - the failure that matters is a result file written after the note said no.

- [ ] 7. Thanh, `notes/retrain-decision.md`: decide whether to retrain, from the two diagnosis notes - expect a yes or a no with exactly one reason
  What to do: read `notes/prompt-gap.md` and `notes/index-neighbours.md`. Write one line that is `retrain` or `do not retrain`, one sentence naming which note decided it, and the holdout number it is responding to. If both notes say the pipeline is doing what it was asked, the answer is `do not retrain` and the reason is that the recipe loses.
  Must NOT do: do not retrain here. Do not propose more than one change. Do not launch a GPU job.
  Closes: GAP-1
  Parallelization: Wave 2 | Blocked by: 1, 2 | Blocks: none
  References: `notes/2026-11-19-task-assignment-qwen3-vl-lora-raft.md:89` (Thanh owns this risk), `notes/lora-train.md` (the run being judged), `notes/holdout.md` (the number).
  Acceptance criteria: a test reads the note and asserts it contains either `retrain` or `do not retrain`, contains `0.0211`, and names exactly one of the two source notes.
  QA scenarios: happy - one decision, one reason, the number. failure - a note that says both retrain and do not retrain fails. Evidence: `.omo/evidence/task-7-phase4.txt`.
  Commit: Y | docs(train): record the retrain decision
  Recommended task executor category: writing - one decision, and the test checks it is one.

- [ ] 8. Thịnh, `report/04b-trained.md`: render the section from the real sources - expect the file to equal `render_section` byte for byte
  What to do: read `results/ablation_A4c.json`, `results/ablation_A6.json`, and `notes/holdout.md`, call `render_section`, write the result. The test does the same and compares. Add one row to the status table in `report/README.md`. Do not add the file to the pandoc command; the submitted document is built from the six chapters.
  Must NOT do: do not edit a submitted chapter. Do not type a number by hand. Do not point `scripts/audit_report_numbers.py` at the new file.
  Closes: GAP-2
  Parallelization: Wave 3 | Blocked by: 5 | Blocks: none
  References: `vad/report/trained.py` (todo 5), `report/README.md:79` (the table), `notes/holdout.md`.
  Acceptance criteria: `python -m pytest -q tests/vad_suite/test_report_trained.py` exits 0. File equals the renderer output, contains `arXiv:2403.10131`, contains `not measured` three times, does not contain `zero-shot`.
  QA scenarios: happy - file matches the renderer. failure - appending a line makes the equality fail. Evidence: `.omo/evidence/task-8-phase4.txt`.
  Commit: Y | docs(report): add the trained-run section
  Recommended task executor category: writing - a generated chapter, test checks it was generated.

## Final verification wave
> Run after all todos close. Not part of implementation.
- [ ] F1. Plan compliance audit
- [ ] F2. Code quality review
- [ ] F3. Real manual QA: read the section, confirm the table against the two JSON files and the holdout note, confirm the three diagnosis notes agree with the code
- [ ] F4. Ideal-state fidelity against the IS rows

## Commit strategy

One commit per owner where the work lands together, otherwise one per todo.

1. `docs(train): record the prompt gap behind the holdout miss` — todo 1, Phước.
2. `docs(index): measure same-class neighbour rate` — todo 2, Nam.
3. `feat(app): show cited clips and gate the eval on the holdout note` — todos 3, 4, 6, Thịnh.
4. `feat(report): render the trained-run section from the sources` — todo 5, Thịnh.
5. `docs(train): record the retrain decision` — todo 7, Thanh.
6. `docs(report): add the trained-run section` — todo 8, Thịnh.

Each commit leaves `python -m pytest -q` green. None touches `report/04-thuc-nghiem.md`, the ablation JSON, or `SCHEMA_VERSION`.

## Success criteria
> One row per IS row. F4 checks the delivered behavior against these rows, and a shortfall becomes new `- [ ] N.` rows, never a note.

| IS | Delivering todo(s) | Proving QA scenario | Evidence |
| --- | --- | --- | --- |
| IS-1 | 1, 2, 7 | each note carries its required fields and the holdout number | `.omo/evidence/task-1-phase4.txt` |
| IS-2 | 7 | the decision is one line and no training was launched | `.omo/evidence/task-7-phase4.txt` |
| IS-3 | 5, 8 | six rows, `arXiv:2403.10131` in the first paragraph | `.omo/evidence/task-5-phase4.txt` |
| IS-4 | 5 | A6 cell matches the file, R0-R2 say not measured, R3 says not run | `.omo/evidence/task-5-phase4.txt` |
| IS-5 | 8 | `zero-shot` absent from the new file | `.omo/evidence/task-8-phase4.txt` |
| IS-6 | 4 | trained run captions the ids, untrained run captions nothing | `.omo/evidence/task-4-phase4.txt` |
| IS-7 | 3, 6 | real note gives exit 4 and no output file | `.omo/evidence/task-6-phase4.txt` |
| IS-8 | 6 | `peft` absent from `sys.modules`, no `R3.json` | `.omo/evidence/task-6-phase4.txt` |
