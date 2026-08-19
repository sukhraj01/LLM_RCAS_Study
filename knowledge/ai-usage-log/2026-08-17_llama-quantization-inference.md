# Session Log — 2026-08-17 → 2026-08-19 — Llama-2-13B 8-bit + 4-bit Inference

## Prompts (verbatim, in order)

### Prompt 1

> Before starting: read PROJECT_STATE.md for current status (12/22
> experiments done - all Mistral techniques complete except ONNX
> [deferred], Llama has only baseline done). Read EXPERIMENT_MATRIX.md's
> Llama 8-bit and 4-bit inference specs specifically - don't re-read the
> whole file. This is a continuation of ongoing work, not a new project -
> do not re-derive project history or re-plan from scratch, just confirm
> this specific task's preconditions.
>
> Task: Run Llama-2-13B's 8-bit and 4-bit inference experiments on both
> datasets (EXP-LLAMA-8BIT-CNN, EXP-LLAMA-8BIT-SQUAD, EXP-LLAMA-4BIT-CNN,
> EXP-LLAMA-4BIT-SQUAD). Inference-only, no training - same pattern
> already validated on Mistral's 8-bit/4-bit runs.
>
> Before running, confirm:
> - Config matches EXPERIMENT_MATRIX.md spec (quantization settings, no
>   typos)
> - VRAM projection fits 16GB with headroom (Mistral's 8-bit peaked
>   ~3.25GB / 4-bit ~1.81GB - Llama-13B will be higher, project it, don't
>   assume it scales identically)
> - Baseline already exists in results/llama_results.csv so quality
>   degradation computes against the right reference
>
> Report results after each experiment before moving to the next - don't
> chain all 4 without confirmation between them.
>
> Do not start Llama QLoRA training - separate time-projection
> conversation needed first given the parameter-count scale-up from
> Mistral's ~7-8x LoRA-to-QLoRA slowdown.

### Clarifying question and answer

Claude Code asked how to proceed given the local session has no GPU (this project's `KAGGLE_SYNC.md` protocol: "Local does not run GPU experiments"). Engineer selected: **"I'll run on Kaggle myself"** — take the scripts to a Kaggle notebook session, run them there, paste results back for logging/validation once done.

### Prompt 2

> exp_id,model,technique,dataset,hardware,training_time_hrs,peak_vram_gb,inference_latency_ms,quality_metrics,status,notes,quality_degradation_percent,speedup_factor,validation_issues
> EXP-LLAMA-BASE-CNN,LLAMA,baseline,CNN,T4,,12.39,15065.6,"{'rouge1': 0.25078525030817, 'rouge2': 0.08700461741796214, 'rougeL': 0.16905006023025135}",completed,,0.0,1.0,
> EXP-LLAMA-BASE-CNN,LLAMA,baseline,CNN,T4,,12.39,14846.0,"{'rouge1': 0.25124785501148034, 'rouge2': 0.08671980409752476, 'rougeL': 0.16893922230470665}",completed,,0.0,1.0,
> EXP-LLAMA-BASE-SQUAD,LLAMA,baseline,SQUAD,T4,,12.38,13456.2,"{'exact_match': 3.0, 'f1': 13.193744841666572}",completed,,0.0,1.0,
> EXP-LLAMA-8BIT-CNN,LLAMA,8bit,CNN,,5.89,26203.1,"{'rouge1': 0.22280625964271758, 'rouge2': 0.07220386766364356, 'rougeL': 0.1511249004439114}",completed,,-11.32,0.57,
> EXP-LLAMA-8BIT-SQUAD,LLAMA,8bit,SQUAD,,5.89,25854.0,"{'exact_match': 0.5, 'f1': 9.713558388516617}",completed,,-26.38,0.52,
>
> Two Llama-2-13B 8-bit inference results just came back from Kaggle
> (EXP-LLAMA-8BIT-CNN, EXP-LLAMA-8BIT-SQUAD) - I've validated the numbers
> against the baseline (quality degradation formula, VRAM projection) and
> they're correct. Do the following:
>
> 1. In results/llama_results.csv, backfill the `hardware` column to `T4`
>    for both new rows - it's currently blank, which breaks the
>    convention just established for tracking which hardware every result
>    ran on.
>
> 2. Append matching rows to logs/experiment_tracking.csv (status:
>    completed, GPU hours used, peak VRAM, quality metrics) following the
>    same format as the existing Mistral 8-bit/4-bit rows.
>
> 3. In EXPERIMENT_MATRIX.md's "Project-Level Finding: T4/Turing
>    Quantization Slowdown" section, add this as a 5th data point and
>    note explicitly that it's the first cross-model replication (Mistral
>    showed the pattern on 4 techniques, this confirms it's not
>    Mistral-specific).
>
> 4. Add a separate note (don't merge with the T4 finding) that Llama's
>    SQuAD quality is weak across both baseline (F1 13.19) and 8-bit (F1
>    9.71) - flag this as a model-specific quality characteristic to
>    mention in qualitative notes for the report, distinct from the
>    hardware-latency finding.
>
> 5. Update PROJECT_STATE.md: EXP-LLAMA-8BIT-CNN and EXP-LLAMA-8BIT-SQUAD
>    move from pending to completed in the Week 4 checklist. Do NOT mark
>    the 8-bit technique fully done yet - only 2 of the 4 remaining Llama
>    quantization experiments (8-bit CNN/SQuAD) are back; 4-bit CNN/SQuAD
>    are still running on Kaggle.
>
> Do not write a phase summary or touch quality gates yet - wait until
> 4-bit results are back too, since 8-bit and 4-bit were scheduled as one
> batch.

### Prompt 3

> exp_id,model,technique,dataset,hardware,training_time_hrs,peak_vram_gb,inference_latency_ms,quality_metrics,status,notes,quality_degradation_percent,speedup_factor,validation_issues
> EXP-LLAMA-BASE-CNN,LLAMA,baseline,CNN,T4,,12.39,15065.6,"{'rouge1': 0.25078525030817, 'rouge2': 0.08700461741796214, 'rougeL': 0.16905006023025135}",completed,,0.0,1.0,
> EXP-LLAMA-BASE-CNN,LLAMA,baseline,CNN,T4,,12.39,14846.0,"{'rouge1': 0.25124785501148034, 'rouge2': 0.08671980409752476, 'rougeL': 0.16893922230470665}",completed,,0.0,1.0,
> EXP-LLAMA-BASE-SQUAD,LLAMA,baseline,SQUAD,T4,,12.38,13456.2,"{'exact_match': 3.0, 'f1': 13.193744841666572}",completed,,0.0,1.0,
> EXP-LLAMA-4BIT-CNN,LLAMA,4bit,CNN,,3.27,24641.4,"{'rouge1': 0.23594123504940756, 'rouge2': 0.0800676130396801, 'rougeL': 0.1596670653573456}",completed,,-6.09,0.6,
> EXP-LLAMA-4BIT-SQUAD,LLAMA,4bit,SQUAD,,3.27,21022.8,"{'exact_match': 1.0, 'f1': 10.602603915788329}",completed,,-19.64,0.64,
>
> Two more Llama-2-13B results came back from Kaggle (EXP-LLAMA-4BIT-CNN,
> EXP-LLAMA-4BIT-SQUAD) - validated (VRAM, quality degradation, speedup
> formula all check out). This completes the full 8-bit+4-bit batch (4
> experiments total). Do the following:
>
> 1. Backfill `hardware` = `T4` for both new rows in
>    results/llama_results.csv (same gap as last time - check the 8-bit
>    rows from the previous commit actually got backfilled too, don't
>    assume).
>
> 2. Append matching rows to logs/experiment_tracking.csv, all 4 Llama
>    quantization experiments now completed.
>
> 3. In EXPERIMENT_MATRIX.md's T4/Turing finding section, add the 4-bit
>    data point as the 6th data point and note the 4-bit-faster-than-
>    8-bit-but-both-slower-than-baseline ordering replicates Mistral's
>    pattern exactly - now 2 full cross-model replications (8-bit, 4-bit)
>    confirm this isn't Mistral-specific.
>
> 4. Add a note (wherever GPU budget estimates live - EXPERIMENT_MATRIX.md
>    or PROJECT_STATE.md) that the original ~0.3h/experiment estimate for
>    Llama inference was wrong by ~4.5x - actual measured was ~2.9 GPU-hrs
>    for the 8-bit batch (2 experiments) and ~2.54 GPU-hrs for the 4-bit
>    batch (2 experiments), computed from the s/sample logs. Correct any
>    remaining GPU-hour projections in the matrix that used the old
>    per-experiment estimate.
>
> 5. Update PROJECT_STATE.md: mark Llama 8-bit and 4-bit techniques fully
>    complete (both datasets). Update overall progress to 16/22
>    experiments complete. Update GPU budget tracking with the real ~5.4
>    GPU-hrs spent on this batch, and flag remaining weekly budget given
>    this was ~4.5x the planned spend.
>
> 6. Do NOT touch Llama QLoRA projections yet - that needs a fresh
>    calculation using this batch's actual measured throughput as the
>    calibration point instead of the old parameter-count heuristic. Flag
>    in PROJECT_STATE.md that the QLoRA time estimate is now considered
>    stale and needs recalculation before scheduling.
>
> Commit when done, tell me if it's safe to push.

## AI-Generated vs Human-Written

All CSV/doc edits and this log are AI-generated. All experiment execution happened on the engineer's own Kaggle session — Claude Code has no GPU access in this local environment (confirmed via `torch` import failure and `KAGGLE_SYNC.md`'s "Local does not run GPU experiments" rule) and never ran any of the 4 experiments directly. Every reported number (latency, VRAM, quality metrics, GPU-hours) originated from the engineer's Kaggle output, not generated or estimated by Claude Code — Claude Code's role was precondition-checking (config match, VRAM projection), independent verification of the reported formulas, and documentation/tracking updates.

## Key Decisions

- **Did not attempt to run the experiments locally or fabricate results.** Verified this local machine has no GPU (`torch` not installed, `nvidia-smi` unavailable) before doing anything else, and deferred to the engineer to run the 4 experiments on Kaggle directly rather than simulating or guessing at outcomes.
- **Independently re-derived the VRAM and quality-formula projections rather than trusting the task prompt's framing at face value.** Projected Llama-2-13B's 8-bit/4-bit peak VRAM two ways (scaling from Mistral's actual quantized/baseline VRAM ratio, and separately from the 13B/7.2B parameter-count ratio) before the runs — both methods agreed within ~0.1GB (~5.8-5.9GB for 8-bit, ~3.25-3.3GB for 4-bit). Both projections were later confirmed almost exactly by the real Kaggle numbers (5.89GB, 3.27GB).
- **Recomputed every reported metric from the raw quality/latency numbers against the project's established formulas** (`quality_degradation_percent = (optimized-baseline)/baseline * 100` using rouge1 for summarization and F1 for QA; `speedup_factor = baseline_latency/optimized_latency`) rather than accepting the engineer's "validated" claim uncritically — all values checked out both turns.
- **Caught and reverted a self-introduced error before committing:** initially wrote fabricated per-experiment `gpu_hours_used` values (1.45h / 1.09h) into the two new 4-bit tracking rows, inferring a split of the engineer's batch-level total (~2.54 GPU-hrs for 2 experiments) that was never actually measured per-dataset. Recognized this violated `CLAUDE.md`'s "never write a result that hasn't actually been measured" principle, reverted to leaving `gpu_hours_used` blank (matching the existing convention already used for every other inference-only row, including the 8-bit rows added the previous turn) and moved the real batch-level figure into the notes/`PROJECT_STATE.md` narrative instead.
- **Went beyond the explicit per-item instructions to fix several counts and claims directly contradicted by the new data** (12/22→16/22 and 8/22→12/22 progress counts, "four-technique"/"five consistent data points" language left over from the prior turn's partial update, the now-false "<20 min per run" quantized-inference assumption in Project Assumptions) — judged these as within the spirit of "PROJECT_STATE.md must reflect reality, not intent" even though not separately requested, since leaving them unfixed would have left the document self-contradictory against the very numbers just logged in the same edit.
- **Did not recalculate Llama QLoRA's GPU-hour estimate**, per explicit instruction — flagged it as stale in `PROJECT_STATE.md` (Current Phase, GPU Budget Tracking, Week 3/4 checklists) with the reasoning for why (same discredited estimation heuristic, now shown wrong twice), but left the actual number untouched pending a dedicated recalculation.
- **Backfilled this log file itself**, noticing partway through that `CLAUDE.md`'s "Session Logging" requirement (a new dated file at the start of every session, not optional) had been skipped for the precondition-check turn that opened this multi-turn session — created it now covering all three prompts rather than leaving the gap.

## Experiments Executed

None by Claude Code directly (no local GPU). Per the engineer's real Kaggle runs, reflected in `results/llama_results.csv` and `logs/experiment_tracking.csv`:
- `EXP-LLAMA-8BIT-CNN`, `EXP-LLAMA-8BIT-SQUAD` (2026-08-18) — both `completed`, `CONFIRMED FINAL`
- `EXP-LLAMA-4BIT-CNN`, `EXP-LLAMA-4BIT-SQUAD` (2026-08-19) — both `completed`, `CONFIRMED FINAL`

## Issues Encountered

- No GPU available in the local Claude Code environment — expected and already documented in `KAGGLE_SYNC.md`, handled by deferring execution to the engineer's own Kaggle session rather than treating it as a blocker to work around.
- Llama's 8-bit+4-bit inference batch (4 experiments) measured ~5.44 GPU-hrs against a planned 1.2h — a ~4.5x miss, the same systematic per-experiment-underestimate pattern already seen with Mistral QLoRA (3.3x miss in an earlier session). Documented as a project-level GPU-hour estimation-method concern (not a one-off anomaly) in `PROJECT_STATE.md` GPU Budget Tracking and `EXPERIMENT_MATRIX.md`'s Full Matrix correction note, with the project's plan-level TOTAL now shown over budget.
- Self-caught data-fabrication near-miss (see Key Decisions above) — corrected before committing, no incorrect data reached a commit.
