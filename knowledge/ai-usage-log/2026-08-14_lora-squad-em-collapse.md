# Session Log — 2026-08-14 — LoRA Results Merge + EXP-MIS-LORA-SQUAD EM Collapse Investigation

## Prompts (verbatim, in order)

### Prompt 1

> LoRA training on Mistral-7B finished on both datasets. Real results from
> results/mis_results.csv:
>
> EXP-MIS-LORA-CNN: ROUGE1 0.2719/ROUGE2 0.0987/ROUGEL 0.1834 (up from baseline
> 0.2386/0.0840/0.1607, +13.9%), training_time_hrs 0.86, peak_vram_gb 6.92,
> adapter saved to checkpoints/mistral_lora_cnn
>
> EXP-MIS-LORA-SQUAD: EM 0.0/F1 7.76 (down from baseline EM 8.0/F1 24.19,
> -67.9%), training_time_hrs 0.41, peak_vram_gb 6.93, adapter saved to
> checkpoints/mistral_lora_squad
>
> 1. Merge both real rows into logs/experiment_tracking.csv (replacing the
>    pending EXP-MIS-LORA-CNN/SQUAD stub rows), referencing the full row in
>    results/mis_results.csv per existing convention.
>
> 2. Flag EXP-MIS-LORA-SQUAD explicitly as needing investigation, not just
>    logged as a normal result - EM dropping to exactly 0.0 across 200 test
>    examples is unusual enough that it could be a real LoRA/QA quality
>    collapse (plausible with only 1000 train examples, 2 epochs, r=8 on an
>    extractive task) or a prompt-format mismatch between training and eval
>    for SQuAD specifically. Check: does apply_lora()/run_training_experiment()
>    in common.py train on the same prompt format that generate_predictions()
>    evaluates with for SQuAD? If format_example()/get_references() construct
>    train targets differently than what's expected at inference, that would
>    explain a collapse to 0 exact matches. Look at this now if you can from
>    the code alone; if it needs actual sample predictions to diagnose, note
>    that as a follow-up (predictions aren't currently saved to disk, only
>    aggregate metrics via save_result() - flag whether that's worth adding).
>
> 3. Also note in EXPERIMENT_MATRIX.md or PROJECT_STATE.md that this is an
>    open question for the eventual report, not resolved yet - don't mark it
>    as an accepted/explained result until it's actually looked into.
>
> 4. Update logs/daily_standup.md's today's Kaggle entry with the LoRA
>    results and this flagged anomaly.
>
> Do not commit yet - report back what you find in the code review first.
>
> [Pasted current logs/experiment_tracking.csv content, 7 rows, for reference]

### Prompt 2

> Confirmed direction to act on: DataCollatorForLanguageModeling masks labels by
> pad_token_id VALUE, not by actual padding position. Since Mistral's tokenizer
> has pad_token set equal to eos_token (load_model_and_tokenizer(), common.py
> ~91-92), every genuine end-of-sequence token in the training targets gets
> masked out of the loss alongside real padding - the model never learns when
> to stop generating. This affects every technique that trains (lora, qlora)
> on every dataset, for both models - not just SQuAD; CNN's ROUGE just doesn't
> expose it the way exact-match does.
>
> Please:
>
> 1. Fix run_training_experiment() in common.py: replace the label-masking
>    approach so it masks based on attention_mask == 0 (actual padding
>    positions), not by matching pad_token_id's value. Either write a small
>    custom collator/label-construction step instead of relying on
>    DataCollatorForLanguageModeling's default behavior, or find another
>    correct way to preserve genuine EOS supervision while still masking real
>    padding. Keep it as targeted a change as possible - this is a labels
>    construction fix, not a tokenizer/embedding change (avoid adding a new
>    pad token / resizing embeddings, that's a bigger blast radius than needed
>    here).
>
> 2. Add minimal raw-prediction logging: save the first ~5 generated
>    predictions + references per experiment (e.g. to a small debug file or
>    an extra column) so results like this can be spot-checked without
>    guessing next time. Keep this lightweight - doesn't need to be a full
>    predictions archive, just enough to confirm/refute hypotheses like this
>    one going forward.
>
> 3. Verify offline (py_compile, import, and if feasible a tiny CPU-only
>    smoke test of the label-masking logic with fake tensors - no GPU needed
>    to check that attention_mask==0 positions get -100 and real EOS positions
>    don't).
>
> 4. Add this to EXPERIMENT_MATRIX.md Recovery Procedures: symptom (EM nears
>    or hits 0 while F1/ROUGE stay nonzero after LoRA/QLoRA training, especially
>    on strict-match datasets) -> root cause (pad_token==eos_token +
>    DataCollatorForLanguageModeling masks real EOS from loss) -> fix (mask by
>    attention_mask, not pad_token_id value).
>
> 5. Update PROJECT_STATE.md's blocker: this is likely THE explanation now
>    (not just a hypothesis), but keep it worded honestly - "root cause
>    identified and fixed; existing LoRA results (Mistral CNN, Mistral SQuAD)
>    were trained under the buggy collator and should be considered unreliable
>    pending rerun" - do not claim it's fully proven until raw predictions or
>    a rerun confirm it.
>
> 6. Update daily_standup.md and today's ai-usage-log entry with this fix.
>
> Do not commit yet - report back the diff first. Also tell me plainly: does
> this bug retroactively invalidate EXP-MIS-LORA-CNN too, not just SQUAD?

## AI-Generated vs Human-Written

All changes this session are AI-generated. The real experiment numbers (ROUGE/EM/F1, training_time_hrs, peak_vram_gb, checkpoint paths) came from the engineer's Kaggle run and `results/mis_results.csv`, not generated by Claude Code:
- `logs/experiment_tracking.csv` — `EXP-MIS-LORA-CNN`/`EXP-MIS-LORA-SQUAD` stub rows replaced with real completed rows; SQUAD row includes the full code-review reasoning inline (per project convention of putting investigation notes directly in the `notes` field)
- `PROJECT_STATE.md` — new item added to Current Blockers documenting the open question, explicitly marked "not blocking, not resolved"
- `logs/daily_standup.md` — today's `2026-08-14 — Kaggle (LoRA env fix)` entry extended with the real results and the flagged anomaly; GPU-hours updated from 0.0 to ~1.3h (the two completed training runs)
- This file

**Prompt 2 (fix + rerun-flagging):**
- `experiments/common.py` — new `_CausalLMCollator` class (masks labels by `attention_mask == 0` instead of `pad_token_id` value); `run_training_experiment()`'s `Trainer` now uses it instead of `DataCollatorForLanguageModeling(mlm=False)`; unused `DataCollatorForLanguageModeling` import removed; new `save_debug_predictions()` helper, called from both `run_inference_only_experiment()` and `run_training_experiment()` right after `generate_predictions()`; `LOGS_DIR` added to the `utils.config` import list
- `EXPERIMENT_MATRIX.md` — new Recovery Procedures entry: symptom (EM near/at 0 while F1/ROUGE nonzero after LoRA/QLoRA) → root cause → fix
- `PROJECT_STATE.md` — Current Blockers item rewritten from "open question, not resolved" to "root cause identified and fixed, not yet confirmed"; explicitly states both `EXP-MIS-LORA-CNN` and `EXP-MIS-LORA-SQUAD` should be treated as unreliable pending rerun
- `logs/experiment_tracking.csv` — both `EXP-MIS-LORA-CNN` and `EXP-MIS-LORA-SQUAD` rows updated: `NEEDS RERUN` flag added to both (previously only SQUAD was flagged), full root-cause/fix reasoning added to both `notes` fields
- `logs/daily_standup.md` — today's entry given a 3rd numbered Issues item covering the fix; GPU-hours note clarified that the fix itself needed no GPU time; Next Session updated to point at rerunning both LoRA experiments
- This file, again (this section + Key Decisions + Issues Encountered)

## Key Decisions

- **Directly answered the engineer's specific hypothesis (prompt-format mismatch) and ruled it out by reading the code, rather than proposing it as one of several untested possibilities.** Traced both call sites of `format_example()`: `tokenize_fn()` inside `run_training_experiment()` (training) and `generate_predictions()` (inference) both call it with the same `(dataset_key, example)` signature — same function, same logic, no divergent prompt construction. `get_references()` is only used in `evaluate_quality()` for scoring, never for constructing training targets (those come directly from `example["answers"]["text"][0]` inside `format_example()`), so there's no train/eval target-construction divergence either. Confirmed via direct code read, not guessed.
- **Proposed a different, more specific code-derivable hypothesis instead: EOS-token label masking.** Traced `load_model_and_tokenizer()` → `tokenizer.pad_token = tokenizer.eos_token` (triggered because Mistral's tokenizer, per the existing `if tokenizer.pad_token is None` guard, ships no pad token) → `run_training_experiment()`'s `DataCollatorForLanguageModeling(tokenizer, mlm=False)` → read `torch_call()`'s actual source (`labels[labels == self.tokenizer.pad_token_id] = -100`, unconditional, not gated on `mlm`). Since `pad_token_id == eos_token_id` here, every genuine end-of-example EOS token (appended via `full_text = f"{prompt} {target}{tokenizer.eos_token}"` in `tokenize_fn()`) gets masked out of the loss identically to padding — the model never receives gradient signal for *when to stop generating*. This differentially explains the CNN/SQuAD asymmetry: ROUGE (partial n-gram recall, `compute_rouge()`) tolerates a model that over-generates past the correct content; SQuAD's `compute_exact_match()` requires the *entire* normalized string to match, so any trailing tokens after the true answer zero out EM even while `compute_f1()` (token overlap) still shows partial credit (7.76, nonzero but far below baseline's 24.19) — consistent with the actual observed numbers.
- **Explicitly did not present this as confirmed.** Verifying it requires looking at actual generated SQuAD predictions (verbose/run-on output would confirm it; short-but-wrong output would refute it), and `save_result()` currently only persists aggregate `quality_metrics`, not raw per-example predictions. Flagged this gap as a follow-up decision (worth adding raw-prediction dumps before Week 3 QLoRA hits the same question on both models) rather than silently assuming the hypothesis is correct — matches the engineer's explicit instruction ("if it needs actual sample predictions to diagnose, note that as a follow-up") and `CLAUDE.md`'s "every unexpected result needs investigation... document findings, don't ignore surprising results."
- **Put the open-question note in `PROJECT_STATE.md`'s "Current Blockers" section** (marked non-blocking) rather than `EXPERIMENT_MATRIX.md`, since `PROJECT_STATE.md` is the file that "reflects reality, not intent" and is read first every session — an unresolved anomaly in a completed experiment reads as exactly the kind of "reality" that section exists to surface, and `EXPERIMENT_MATRIX.md` is more the fixed spec/recovery-procedures reference than a running log of open questions.
- **Did not touch Component Status / GPU Budget Tracking tables in `PROJECT_STATE.md`** or make any code changes (e.g., adding prediction-dumping to `common.py`) — out of scope for what was asked this turn (CSV merge, flag, open-question note, standup update). Left as an explicit option for the engineer to decide on, not implemented preemptively.
- **Did not commit**, per explicit instruction to report back first.

**Prompt 2:**
- **Fix scoped to labels construction only, per explicit instruction.** Wrote `_CausalLMCollator` as a thin wrapper around `tokenizer.pad()` (which already pads `input_ids`/`attention_mask` together) plus a one-line `labels[attention_mask == 0] = -100`. Did not add a distinct pad token or resize embeddings — the engineer explicitly ruled that out as unnecessarily large blast radius, and it isn't needed: masking by `attention_mask` sidesteps the `pad_token_id == eos_token_id` collision entirely without changing the tokenizer or model at all.
- **Removed the now-unused `DataCollatorForLanguageModeling` import** rather than leaving it (confirmed via the earlier `grep -rn "gradient_checkpointing\|get_peft_model\|TrainingArguments("` pass from the previous session that `run_training_experiment()`'s `Trainer` construction was its only use site).
- **Verified via a genuine offline smoke test, not just import/compile.** Built a fake tokenizer whose `.pad()` mimics real padding behavior, constructed one full-length feature (attention_mask all `1`s, last token value deliberately equal to the fake `pad_token_id` — the exact scenario a real EOS token creates) and one short feature that needs padding. Asserted the full-length example's labels are untouched (`[1, 2, 3, 999]`, no `-100` even though `999 == pad_token_id`) and the short example's labels are `-100` only at the truly padded positions (`[4, 5, -100, -100]`). This is a real regression test for the exact bug, not a generic "does it import" check — it would fail against the old `DataCollatorForLanguageModeling` behavior and passes against the new collator.
- **`save_debug_predictions()` called from both training and inference-only paths**, not just the training path that surfaced this bug — the engineer's ask ("so results like this can be spot-checked without guessing next time") is a general pipeline gap, not specific to LoRA/QLoRA; baseline/8bit/4bit/ONNX results could hit an equally surprising number later and benefit from the same debug dump. Kept deliberately lightweight (first 5 examples, plain text, no new CSV schema) per the explicit "doesn't need to be a full predictions archive" instruction.
- **Flagged `EXP-MIS-LORA-CNN` as `NEEDS RERUN` alongside `EXP-MIS-LORA-SQUAD`**, not left as an accepted result — see the direct answer to the engineer's closing question below.
- **Worded `PROJECT_STATE.md` and `logs/experiment_tracking.csv` exactly as instructed**: "root cause identified and fixed" (not "confirmed" or "proven"), with an explicit "not yet confirmed on real hardware" caveat and a call to rerun before either LoRA result is treated as final — matches the engineer's explicit instruction not to overclaim past what an offline code-only fix can actually establish.
- **Did not commit**, per explicit instruction to report back the diff first.

**Direct answer to the engineer's question ("does this bug retroactively invalidate EXP-MIS-LORA-CNN too, not just SQUAD?"):** Yes. `_CausalLMCollator`/`DataCollatorForLanguageModeling` is the one shared code path both experiments went through — `run_training_experiment()` is called once per dataset by `run_training_multi_dataset()`, with no per-dataset branching in how the collator is built. So `EXP-MIS-LORA-CNN`'s training was equally missing EOS supervision. Its ROUGE improving anyway doesn't clear it: ROUGE is a partial n-gram overlap metric that's much more forgiving of a model that keeps generating past the point where it should have stopped, whereas SQuAD's exact-match requires the *entire* generated (normalized) string to equal the reference, so any trailing tokens are fatal to EM specifically. The absence of a visible symptom on CNN is not evidence of absence of the underlying defect — both experiments were trained under the same bug and both are flagged `NEEDS RERUN` in `logs/experiment_tracking.csv` and `PROJECT_STATE.md`.

## Experiments Executed (reported by engineer)

- `EXP-MIS-LORA-CNN` — completed, now flagged `NEEDS RERUN`: training_time_hrs 0.86, peak_vram_gb 6.92, ROUGE1/2/L 0.2719/0.0987/0.1834 vs baseline 0.2386/0.0840/0.1607 (+13.9%) — improved-looking result, but trained under the same buggy collator as SQUAD; not cleared of the same defect, just doesn't expose it via ROUGE
- `EXP-MIS-LORA-SQUAD` — completed, flagged `NEEDS RERUN`: training_time_hrs 0.41, peak_vram_gb 6.93, EM 0.0/F1 7.76 vs baseline EM 8.0/F1 24.19 (-67.9%) — root cause now identified and fixed in code, not yet confirmed on real hardware

No new experiments executed this turn (Prompt 2 was code review + fix + doc updates only, no GPU available locally). Rerun of both is queued as next session.

## Issues Encountered

`EXP-MIS-LORA-SQUAD`'s exact-match collapsing to 0/200 while CNN improved under the identical training pipeline. Root cause confirmed by engineer and fixed this turn: `transformers.DataCollatorForLanguageModeling(mlm=False)`'s `torch_call()` masks labels by `pad_token_id` value (`labels[labels == self.tokenizer.pad_token_id] = -100`), unconditionally — not by actual padding position. Since `load_model_and_tokenizer()` sets `tokenizer.pad_token = tokenizer.eos_token` (Mistral's tokenizer has no distinct pad token), this masked every genuine per-example EOS token out of the training loss alongside real padding, for every technique that trains, on both datasets. Fixed via `_CausalLMCollator`, which masks by `attention_mask == 0` instead — verified with an offline fake-tensor smoke test (see Key Decisions). Not yet verified against real generated predictions or a real Kaggle rerun; `save_debug_predictions()` was added specifically so the rerun can be spot-checked without repeating this guesswork.
