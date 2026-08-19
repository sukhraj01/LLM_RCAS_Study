# Session Log — 2026-08-19 — Llama-2-13B QLoRA Time Projection (Recalculated, Not Run)

## Prompts (verbatim, in order)

### Prompt 1

> Before scheduling Llama-2-13B QLoRA training, update the stale time
> projection using real methodology instead of the round-number guess
> that caused the 4.5x miss on the 8-bit/4-bit batch. Document in
> EXPERIMENT_MATRIX.md:
>
> - Mistral's own LoRA-to-QLoRA training slowdown ratio (measured):
>   CNN 0.89h -> 6.44h (~7.2x), SQuAD 0.42h -> 3.42h (~8.1x)
> - Llama/Mistral scaling observed from inference generation latency and
>   VRAM (both consistently ~1.8x, matching the 13B/7.2B param ratio)
> - State explicitly that no direct Llama training benchmark exists yet,
>   so this projection stacks two separate ratios and should be treated
>   as a rough bound, not a confident estimate - flag this as a real
>   epistemic gap, not paper over it
>
> Recommend running SQuAD QLoRA training first as the calibration run
> (projected ~6-8.5h, fits the 12h cap with buffer), not CNN (projected
> ~13-19h, likely over the cap). Confirm checkpoint-every-200-steps is
> wired in per the existing recovery procedure before this starts.
>
> Do not start the actual training run - just get the projection and
> recommendation written down and confirmed against EXPERIMENT_MATRIX.md
> before I decide whether to schedule it.

## AI-Generated vs Human-Written

All doc edits and this log are AI-generated. No training was run (per explicit instruction) — Claude Code has no GPU access in this local environment regardless. The three input data points cited in the prompt (Mistral's measured LoRA→QLoRA ratios, the observed cross-model scaling factor) were restated from real measured values already in `results/mis_results.csv` / `results/llama_results.csv` from prior sessions, not newly generated.

## Key Decisions

- **Did not adopt the prompt's "~1.8x, consistent" characterization of the Llama/Mistral latency-scaling evidence without checking it first.** Recomputed all 6 actual cross-model latency ratios (baseline/8-bit/4-bit × CNN/SQuAD) from the CSVs rather than taking the summary at face value: they range 1.02x-2.05x (average ~1.58x), not tightly consistent. VRAM, by contrast, *is* tightly consistent (1.79-1.81x across all three techniques, matching the 13B/7.2B param ratio almost exactly). Documented this correction explicitly in `EXPERIMENT_MATRIX.md` rather than silently using "1.8x, consistent" as an uncontested premise — used VRAM's tighter signal as the actual justification for the 1.8x central multiplier, with the noisier latency range folded into the uncertainty band instead of the central estimate. Flagged this as a correction to the task's own framing, not just extra detail.
- **Derived a projection with an explicit central estimate and a separate, wider "rough bound"** rather than a single number, per the instruction to avoid a false sense of confidence: central (CNN ~11.6h, SQuAD ~6.2h, from `Mistral_QLoRA_measured × 1.8`) vs. bound (CNN ~9.7-13.5h, SQuAD ~5.1-7.2h, from a 1.5x-2.1x scaling range that trims the 1.02x outlier as likely quantization-specific noise). Did not force the bound to match the prompt's own ballpark figures (~13-19h CNN, ~6-8.5h SQuAD) — derived independently from the three given inputs and reported the result as computed, noting it supports the same qualitative conclusion (SQuAD safer, CNN risks the 12h cap) even though the exact numbers differ from what the engineer sketched in the prompt. Did not attempt to reverse-engineer or silently match the engineer's numbers.
- **Checked the checkpointing precondition by computing actual step counts, not just grepping for `save_steps=200` and calling it confirmed.** Llama QLoRA's hyperparameters (`batch_size=1`, `gradient_accumulation_steps=16`, 1000 train examples, 2 epochs) give only ~126 total optimizer steps per dataset — below the 200-step threshold, so the checkpoint mechanism as configured would never fire during a real run. This is the opposite of what a "confirm it's wired in" check was supposed to find, and is exactly the kind of thing that check exists to catch. Also checked whether this affected Mistral's already-completed QLoRA runs (it does — same effective-batch-16 math) to establish this isn't Llama-specific, just newly consequential now that CNN's projected time risks the 12h cap. Flagged with a concrete fix (`save_steps` ≈ 25) but did not apply it — out of scope for a projection-only session per the instruction not to start training, and changing training hyperparameters deserves the same confirm-first treatment as everything else in this project.
- **Updated every "stale, not yet recalculated" pointer left in `PROJECT_STATE.md` from the prior session** (Current Phase, Component Status, GPU Budget Tracking QLoRA row, Week 3/4 checklist) so none of them contradict the new projection that now exists — rather than only editing `EXPERIMENT_MATRIX.md` as literally instructed and leaving `PROJECT_STATE.md` internally inconsistent.
- **Did not schedule or start Llama QLoRA training**, per explicit instruction — this was strictly a projection/documentation/precondition-check exercise. Recommended SQuAD as the calibration run if the engineer chooses to proceed, but left that decision, and the `save_steps` fix, for the engineer.

## Experiments Executed

None. No Llama QLoRA training (or any other experiment) was run this session, per explicit instruction.

## Issues Encountered

- Found a real, previously-unflagged gap: the project's "checkpoint every 200 steps" recovery procedure (`CLAUDE.md`) is not actually functional for Llama's QLoRA hyperparameters as currently configured, because the total step count per dataset (~126) never reaches the 200-step save threshold. Not an issue introduced this session — pre-existing since the hyperparameters were set, and already latent (unexercised) in Mistral's completed QLoRA runs — but only surfaced now because this session's explicit instruction was to check it before scheduling Llama QLoRA, and because CNN's newly-computed projected range (~9.7-13.5h) makes an uncheckpointed 12h-cap interruption a real risk rather than a theoretical one. Documented, not fixed, per scope.
