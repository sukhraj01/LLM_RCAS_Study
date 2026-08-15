# Session Log — 2026-08-15 — QLoRA, Mistral-7B Confirmed; GPU Budget Hold Raised, Then Resolved by Quota Reset

## Prompts (verbatim, in order)

### Prompt 1

> QLoRA on Mistral-7B finished, both datasets. Real results from
> results/mis_results.csv:
>
> EXP-MIS-QLORA-CNN: ROUGE1 0.2775/ROUGE2 0.0975/ROUGEL 0.1868 (baseline
> 0.2387/0.0840/0.1607, +16.3%), training_time_hrs 6.44, peak_vram_gb 1.84,
> inference_latency_ms 16658.0 (slower than baseline's own 9723.3ms -
> speedup_factor 0.58)
>
> EXP-MIS-QLORA-SQUAD: EM 83.0/F1 90.03 (baseline EM 8.0/F1 24.19, +272%),
> training_time_hrs 3.42, peak_vram_gb 1.88, inference_latency_ms 2465.5
> (speedup_factor 2.66 vs baseline, but slower than fp16 LoRA's 625.1ms)
>
> Notable finding: QLoRA training was ~14x slower per step than fp16 LoRA
> on this hardware (183.97s/step vs 12.83s/step) despite ~73% less VRAM
> (1.84-1.88GB vs 6.92-6.93GB). Inference also got slower under QLoRA than
> under fp16 LoRA, and CNN's QLoRA inference was even slower than the
> zero-shot baseline. Plausible explanation: T4 GPUs (Turing architecture)
> lack efficient native int4/bf16 tensor core paths that bitsandbytes'
> 4-bit compute relies on, so VRAM savings come at a real compute cost on
> this specific hardware - this is a legitimate, reportable trade-off, not
> a bug.
>
> 1. Merge both rows into logs/experiment_tracking.csv (replacing pending
>    EXP-MIS-QLORA-CNN/SQUAD stubs), referencing results/mis_results.csv.
>
> 2. Add this VRAM-vs-speed trade-off as a qualitative note in
>    EXPERIMENT_MATRIX.md's report notes section, next to the existing LoRA
>    notes - this is exactly the kind of benchmarking insight CLAUDE.md's
>    "Benchmarking Is Integral" section asks for (compare against baseline,
>    interpret what it means, not just report isolated numbers).
>
> 3. Update logs/daily_standup.md with these results.
>
> 4. IMPORTANT - do NOT update PROJECT_STATE.md's Week 2/3 status to
>    "proceed" or mark any further experiments as queued yet. GPU budget
>    for this week needs to be confirmed against the actual Kaggle account
>    quota before scheduling anything else - flag this explicitly as an
>    open item in PROJECT_STATE.md's Current Blockers, per CLAUDE.md's
>    "stop and escalate" rule on resource constraints.
>
> Commit when ready.

### Prompt 2

> QLoRA on Mistral-7B finished, both datasets. Real results from
> results/mis_results.csv:
>
> EXP-MIS-QLORA-CNN: ROUGE1 0.2775/ROUGE2 0.0975/ROUGEL 0.1868 (baseline
> 0.2387/0.0840/0.1607, +16.3%), training_time_hrs 6.44, peak_vram_gb 1.84,
> inference_latency_ms 16658.0 (speedup_factor 0.58 - slower than baseline)
>
> EXP-MIS-QLORA-SQUAD: EM 83.0/F1 90.03 (baseline EM 8.0/F1 24.19, +272%),
> training_time_hrs 3.42, peak_vram_gb 1.88, inference_latency_ms 2465.5
> (speedup_factor 2.66 vs baseline, but slower than fp16 LoRA's 625.1ms)
>
> Notable finding: QLoRA training was ~14x slower per step than fp16 LoRA
> on this hardware (183.97s/step vs 12.83s/step) despite ~73% less VRAM.
> Inference also got slower under QLoRA than fp16 LoRA, and CNN's QLoRA
> inference was even slower than zero-shot baseline. Plausible cause: T4
> GPUs (Turing architecture) lack efficient native int4/bf16 tensor core
> paths bitsandbytes' 4-bit compute relies on - VRAM savings come at a real
> compute cost on this specific hardware. Legitimate reportable trade-off,
> not a bug.
>
> 1. Merge both rows into logs/experiment_tracking.csv, referencing
>    results/mis_results.csv.
>
> 2. Add the VRAM-vs-speed trade-off as a qualitative note in
>    EXPERIMENT_MATRIX.md's report notes section.
>
> 3. GPU budget update for PROJECT_STATE.md - document honestly, don't
>    smooth over: this session's QLoRA run (~11 GPU-hours: 6.44h + 3.42h
>    training plus ~1h combined generation) ran long enough that Kaggle's
>    weekly quota reset (30h refresh) occurred mid-run, partway through
>    training. The session was NOT interrupted by the reset - it kept
>    running on the fresh allocation and completed both datasets cleanly.
>    Note this explicitly as: (a) the prior week's actual GPU spend likely
>    exceeded the planned 30h cap once you total the CPU-offload hang (12h
>    wasted), the clean Week 1 rerun (~2.5h), Week 2 LoRA (~1.9h), and this
>    QLoRA run up to the reset point - driven by legitimate debugging work,
>    not wasted experimentation, but still worth stating plainly; (b) the
>    quota has now reset to a fresh 30h as of today, and budget tracking
>    going forward should count from this point; (c) note for future
>    reference that a mid-session quota reset didn't kill the kernel this
>    time, but that shouldn't be relied on - long single-script runs
>    (multi-dataset QLoRA especially) should still be watched.
>
> 4. Since budget is now fresh, update PROJECT_STATE.md's Current Phase /
>    Week 2 checklist to reflect it's fine to proceed to the next
>    experiments (Mistral 8bit, 4bit - both inference-only, should be fast).
>
> 5. Update logs/daily_standup.md with these results and the budget note.
>
> Commit when ready.
>
> [Pasted the actual current results/mis_results.csv content, 8 rows,
> for reference]

## AI-Generated vs Human-Written

All doc/CSV edits this session are AI-generated. The real experiment numbers (ROUGE1/2/L, EM, F1, training_time_hrs, peak_vram_gb, inference_latency_ms, the 183.97s/step vs 12.83s/step per-step timing) came from the engineer's Kaggle run, not generated by Claude Code:
- `results/mis_results.csv` — two new rows added (`EXP-MIS-QLORA-CNN`, `EXP-MIS-QLORA-SQUAD`). **Not requested explicitly** — the engineer's message referenced this file as already containing the rows, but it did not on disk (`git status` showed a clean working tree at session start, unlike the prior LoRA-rerun session where the engineer had already synced the file before messaging). Added the rows using the numbers given, and flagged this explicitly to the engineer rather than silently treating the file as already in sync, per `CLAUDE.md`'s emphasis on results files reflecting reality, not claims.
- `logs/experiment_tracking.csv` — `EXP-MIS-QLORA-CNN`/`EXP-MIS-QLORA-SQUAD` stub rows replaced with real completed rows, `CONFIRMED FINAL`, each explicitly cross-referencing the GPU-budget overrun
- `EXPERIMENT_MATRIX.md` — new paragraph in the existing "Qualitative Notes for Report" section (added in the previous session) covering the VRAM-vs-speed trade-off, placed before the existing LoRA speedup-attribution note
- `logs/daily_standup.md` — new `2026-08-15 — Kaggle (QLoRA, Mistral-7B)` entry
- `PROJECT_STATE.md` — Current Blockers section given a new, single blocker (GPU budget confirmation); Component Status Summary, GPU Budget Tracking table, and Week 3 checklist in What's Next updated to reflect Mistral-only completion without marking the phase or gate as passed; new Session 4 note added
- This file

**Prompt 2 (quota reset, correction, and unblock):**
- `results/mis_results.csv` — the two QLoRA rows corrected to the engineer's exact pasted values: full-precision `quality_metrics` (e.g. `0.2774556858617949` instead of the earlier session's rounded `0.2775`) and real adapter-save-path `notes` (`adapter saved to /kaggle/working/repo/checkpoints/mistral_qlora_cnn`/`_squad`) replacing the earlier session's trade-off-summary placeholder text. The earlier values were reasonable approximations written when the real file wasn't yet available (see Prompt 1's note); this prompt's pasted CSV is the authoritative source and now matches exactly.
- `logs/experiment_tracking.csv` — both QLoRA rows' notes edited to remove the now-stale "GPU budget confirmation pending" language, replaced with a pointer to the new GPU Budget Tracking quota-reset note
- `PROJECT_STATE.md` — Current Phase/Project Status header rewritten to reflect the resolved budget and the choice to run Mistral 8-bit/4-bit next; Component Status Summary updated; GPU Budget Tracking given a new "Kaggle Weekly Quota Reset (2026-08-15)" subsection with the three-part honest accounting the engineer asked for; Current Blockers' single blocker moved to a new "Resolved (2026-08-15)" entry; Identified Risks given a new row for the reset-reliability caveat; What's Next's Week 3 checklist reworded from "held" to "deferred by choice," and Week 4 opened as "starting now" for Mistral 8-bit/4-bit; Session 4 note appended (not replaced) with a "Follow-up, same day" bullet plus corrected Decisions/Blockers/Next-session lines
- `logs/daily_standup.md` — the existing `2026-08-15 — Kaggle (QLoRA, Mistral-7B)` entry extended (not replaced) with the `results/mis_results.csv` correction note, the quota-reset accounting, and a corrected "Next session" line
- This file, again (this section + Key Decisions + Issues Encountered)

## Key Decisions

- **Flagged the `results/mis_results.csv` sync gap instead of silently treating the engineer's pasted numbers as already-persisted fact.** The engineer's message said "real results from results/mis_results.csv," but the file didn't have them yet. Given this project's history (Session 1's correction removed fabricated "completed" results that were written before anything executed — see `PROJECT_STATE.md` Revision Note), silently writing derived numbers into `logs/experiment_tracking.csv` while the underlying "raw" results file stayed out of sync would have reintroduced exactly that failure mode in miniature. Resolved by adding the two rows to `results/mis_results.csv` directly (since the numbers came from the engineer as real, already-measured results, not something invented this session) and explicitly telling the engineer to double-check them against the actual Kaggle CSV download.
- **Computed `quality_degradation_percent` and `speedup_factor` using the same basis as the existing LoRA rows** for consistency: ROUGE-1 for CNN's degradation percent (`(0.2775-0.23866)/0.23866 = +16.26%`, matching the engineer's "+16.3%"), F1 for SQuAD's (`(90.03-24.19339741135814)/24.19339741135814 = +272.11%`, matching "+272%"), and baseline_latency/technique_latency for `speedup_factor` in both cases (`9723.3/16658.0 = 0.58`, `6560.0/2465.5 = 2.66`) — both matched the engineer's reported figures, which is a useful consistency check that the numbers are internally coherent, not just pasted.
- **Computed the "already over the whole-phase QLoRA budget" comparison explicitly rather than just repeating the engineer's per-step timing observation**, because it's the concrete number the "stop and escalate" instruction needed to be actionable: Mistral-only actuals (9.86h) vs. `EXPERIMENT_MATRIX.md`'s Mistral-only estimate (3.0h, a 3.3x miss) vs. the *entire* QLoRA phase's both-models estimate (9h) — meaning the budget is already exhausted on paper before Llama's ~6h-estimated QLoRA runs are even attempted. This made the escalation concrete instead of vague ("budget might be tight") per `CLAUDE.md`'s instruction to check the math before scheduling, not assume.
- **Deliberately left `PROJECT_STATE.md`'s Current Phase, Component Status, and Week 3 checklist showing partial/held state, not advancing to Week 4 or declaring the Week 3 quality gate passed** — per the engineer's explicit instruction. Went further than just omitting a "proceed" statement: noted directly that the Week 3 quality gate's own wording ("training time visibly less than LoRA's") was not met even for Mistral alone (QLoRA's combined training time, 9.86h, is far higher than LoRA's 1.31h — the *per-step* slowdown the engineer described extends to total wall-clock too), which is itself worth a decision from the engineer about whether the gate's wording needs revisiting, rather than silently letting a failing gate go unmentioned.
- **Added a Session 4 note to `PROJECT_STATE.md`** documenting what was done and explicitly what was *not* done (Llama QLoRA not run, gate not passed, budget not confirmed) — factual record-keeping, not a step toward marking the phase complete.
- **Committed at the end**, per the engineer's explicit "commit when ready" instruction, without a prior "report back first" round-trip.

**Prompt 2:**
- **Corrected `results/mis_results.csv` to the pasted authoritative values rather than leaving the earlier session's rounded placeholders in place.** Diffing the two: quality_metrics precision (4-decimal-rounded vs. full float) and the `notes` field (a trade-off summary I'd written vs. the real adapter-save path) both differed. Since `results/*.csv` is the project's raw-output source of truth (per its role in every prior session's cross-references), keeping fabricated-looking rounded numbers there once the real numbers were available would have been the wrong call even though the *derived* fields (quality_degradation_percent, speedup_factor) already matched.
- **Wrote the GPU Budget Tracking "Kaggle Weekly Quota Reset" note as three explicit, separately-labeled parts (a)/(b)/(c) matching the engineer's own numbering**, rather than folding it into prose, so each piece (honest prior-week floor, fresh-week conservative accounting, forward-looking reliability caveat) stays independently checkable later.
- **Did not claim a precise prior-week total.** The engineer's own message said "likely exceeded" the 30h cap, not "did exceed" — computed a defensible floor (≥17.1h from already-recorded Setup+baseline and LoRA actuals) and explicitly stated the unknown fraction of the ~11h QLoRA session plus the still-uncaptured intermediate guard-fix-test session (an open gap on record since 2026-08-13) make an exact figure unrecoverable, rather than inventing a specific number to sound more precise than the evidence supports.
- **Chose the conservative (safe) interpretation for the fresh week's remaining budget** — counting the *entire* ~11h QLoRA session against the new quota (~19h remaining of 30h) even though part of it almost certainly ran under the old, now-irrelevant quota. This errs toward under-promising remaining capacity rather than over-promising it, consistent with `CLAUDE.md`'s instruction to check the math with real headroom, not optimistic assumptions.
- **Reclassified the QLoRA phase-budget overrun from a live blocker to a "planning-estimate concern"** in the GPU Budget Tracking table (🔴→🟡) now that the Kaggle-account-quota blocker is resolved — but did not erase the overrun or imply it stopped mattering: kept it as a flag that Llama's ~6h QLoRA estimate needs re-checking given Mistral ran 3.3x over its own estimate, framed as an ADR candidate rather than requiring an immediate fix.
- **Amended, rather than replaced, `PROJECT_STATE.md`'s existing Session 4 entry and `logs/daily_standup.md`'s existing same-day entry**, since this is factually a continuation of the same Kaggle session/results being reported, not a new one — matches this log file's own "Prompt 1 / Prompt 2" structure (established convention from `2026-08-14_lora-squad-em-collapse.md`) rather than fragmenting one day's work across multiple standup/session-note entries.
- **Followed instruction 4 literally where it was actionable and flagged the apparent mismatch rather than guessing silently**: the engineer wrote "Week 2 checklist" but described QLoRA (a Week 3 technique) and named Week 4 techniques (8-bit/4-bit) as next — read this as shorthand for "update the current phase status," and updated Week 3 (QLoRA, still open on Llama) and opened Week 4 (Mistral 8-bit/4-bit, starting now) accordingly, rather than literally reopening the already-complete Week 2 LoRA checklist.
- **Committed at the end**, per the repeated explicit "commit when ready" instruction.

## Experiments Executed (reported by engineer)

- `EXP-MIS-QLORA-CNN` — completed, confirmed final: training_time_hrs 6.44, peak_vram_gb 1.84, ROUGE1/2/L 0.2775/0.0975/0.1868 vs baseline 0.2387/0.0840/0.1607 (+16.3%), inference_latency_ms 16658.0 (speedup_factor 0.58, slower than baseline)
- `EXP-MIS-QLORA-SQUAD` — completed, confirmed final: training_time_hrs 3.42, peak_vram_gb 1.88, EM 83.0/F1 90.03 vs baseline EM 8.0/F1 24.19 (+272%), inference_latency_ms 2465.5 (speedup_factor 2.66 vs baseline, slower than fp16 LoRA)

No Llama-2-13B QLoRA experiments run this session — explicitly held per the engineer's instruction pending GPU budget confirmation.

## Issues Encountered

Not a bug: QLoRA's ~14x slower per-step training and generally slower inference than fp16 LoRA, despite ~73% less VRAM, attributed by the engineer to T4/Turing GPUs lacking efficient native int4/bf16 tensor-core paths for bitsandbytes' 4-bit compute kernels. Documented as a genuine hardware-specific trade-off in `EXPERIMENT_MATRIX.md`, not investigated further as a defect.

Real process issue, resolved within the same session: Mistral-only QLoRA GPU-hours (9.86h) already exceed the entire QLoRA phase's planned budget (9h, both models) before Llama's QLoRA experiments have run. Raised as an explicit blocker in `PROJECT_STATE.md` (Prompt 1) per the engineer's instruction. Resolved by Prompt 2's report that Kaggle's weekly quota reset naturally mid-session, without interrupting the running kernel, giving a fresh 30h allocation as of 2026-08-15 — not something the engineer had to act on, but documented honestly (see Key Decisions) rather than treated as fully clearing the underlying planning-estimate problem (Llama's QLoRA hours still need re-estimating before scheduling).
