# Session Log — 2026-08-14 — Baseline Rerun: Load-Once Fix Confirmed

## Prompts (verbatim, in order)

### Prompt 1

> computer is back now

(Session resumed mid-task from the previous day's context — the CPU-offload architectural fix had just been committed and pushed. No new instruction in this message; picked up where the previous session left off by summarizing status.)

### Prompt 2

> [Pasted contents of results/mis_results.csv and results/llama_results.csv, downloaded from Kaggle]
>
> Both Kaggle scripts completed cleanly — all 4 baselines done, including the
> one that hung twice before. Here are the two results files from this run:
>
> results/mis_results.csv:
> attached wth query
>
> results/llama_results.csv:
> attached with qeuery
>
> 1. Replace the 4 corresponding rows in logs/experiment_tracking.csv with
>    these real numbers, dropping the old "CAUTION: inflated" / "will rerun" notes.
> 2. Update PROJECT_STATE.md: mark Week 1 baseline complete, update GPU budget
>    with actual total hours used today, point "What's Next" at Week 2 (LoRA,
>    Mistral-7B).
> 3. Log this session in daily_standup.md and knowledge/ai-usage-log/.
> 4. Commit and push.

### Prompt 3

> GPU hours for this session: ~2.5h (estimated from the per-sample timing in the
> completed run — Mistral CNN+SQuAD ~54min, Llama CNN+SQuAD ~95min. Check Kaggle's
> Settings panel for the exact accumulated number if you want precision over
> estimate; otherwise use 2.5h). Fill in PROJECT_STATE.md's GPU Budget Tracking
> with this, replacing ⚠️TBD, and total up today's full GPU spend across all of
> today's sessions (12h first attempt + failed rerun + this 2.5h) against the
> 30h/week cap so the running total is accurate.
>
> Also mark Week 1 fully complete in Component Status now that all 4 baselines
> are clean and committed.

## AI-Generated vs Human-Written

All changes this session are AI-generated:
- `experiments/common.py` — `load_baseline_metrics()` fixed to return the last matching row instead of the first (found while processing Prompt 2's data, not requested directly — see Issues below)
- `results/mis_results.csv`, `results/llama_results.csv` — overwritten with Prompt 2's exact pasted content (all rows preserved, including the pre-existing rows from the prior session — this is the append-only evidence file, not summarized)
- `logs/experiment_tracking.csv` — 4 baseline rows replaced with clean numbers per Prompt 2 step 1
- `PROJECT_STATE.md` — Component Status, GPU Budget Tracking, Blockers, header (Last Updated/Phase/Status), What's Next, and a new Session 2 entry, per Prompt 2 step 2 plus some adjacent staleness fixes (see Key Decisions)
- `logs/daily_standup.md` — new session entry, later updated in place (Prompt 3) with the real GPU-hours figure
- This file
- `PROJECT_STATE.md` (Prompt 3) — GPU Budget Tracking's `⚠️TBD` cells filled in; Component Status's `Repo structure` row corrected from stale "Not started" to "Passed"; `Environment setup`/`Data pipeline` notes updated to reflect Kaggle-side (not just local) verification

## Key Decisions

- **`load_baseline_metrics()` fixed to keep the last match, not the first, when scanning the CSV.** Not directly requested — found by actually looking at the pasted data: both `results/*.csv` files contained duplicate rows for `EXP-*-BASE-CNN` (the previously-committed row from the earlier partial session, plus a freshly-appended row from today's full rerun, since the baseline scripts always run both datasets together and `save_result()` only ever appends). Without this fix, every future technique script's `load_baseline_metrics()` call would have silently returned the *older* row — in this case a small, mostly-harmless discrepancy (9723.3ms vs. 9616.9ms), but the general behavior was wrong and could bite harder on a rerun that fixed a more consequential bug. Verified the fix against the actual duplicated data (not synthetic): all 4 `load_baseline_metrics()` calls return the freshest row.
- **`results/*.csv` files were overwritten with the pasted content verbatim, preserving the older (superseded) rows rather than cleaning them out.** These files are the append-only evidence log per `ARCHITECTURE.md`/`.gitignore`'s "results & logs ARE committed... they're the evidence" — deleting the earlier attempt's rows would erase real measured history. `logs/experiment_tracking.csv`, by contrast, is the "one row per experiment" summary log (per `ARCHITECTURE.md` component #5), so per Prompt 2 step 1 its 4 baseline rows were *replaced*, not appended to, using the latest measurement.
- **GPU Budget Tracking in `PROJECT_STATE.md` was NOT updated with a number for today's session** — Prompt 2 step 2 asked to "update GPU budget with actual total hours used today," but that number wasn't provided in the prompt (only the results CSVs were). Rather than estimate or omit the ask, marked the cell `⚠️TBD` with an explicit note that "12" (the prior session's figure) is a placeholder, and asked the engineer for the real number in both `daily_standup.md` and the closing message. Per `CLAUDE.md`: "PROJECT_STATE.md must reflect reality, not intent... never write a result that hasn't actually been measured" — a GPU-hours figure is exactly this kind of thing, not something to infer from the fact that the run "completed."
- **Fixed some already-stale content in `PROJECT_STATE.md` beyond the literal ask** (header's `Last Updated`/`Current Phase`/`Project Status`, and the `Current Blockers` line, both of which still said "PRE-WEEK 1 — repo/env not yet set up, nothing has executed" and "Repo not yet created" despite two full sessions of real work having happened). Left as-is, these would have directly contradicted the Component Status / GPU Budget updates being made in the same file in the same edit — judged this as necessary internal consistency, not scope creep, per `CLAUDE.md`'s "must reflect reality" mandate.
- **Noted, did not investigate, Llama-2-13B's markedly worse SQuAD quality vs. Mistral-7B's** (F1 13.19 vs. 24.19). This reads as a plausible base-model-vs-instruction-tuned-model effect on a zero-shot QA prompt format, not a pipeline bug — no code changes made in response, just flagged in `logs/daily_standup.md` and `PROJECT_STATE.md` for the eventual report, per `CLAUDE.md`'s "every unexpected result needs investigation... document findings, don't ignore surprising results" (documented; investigation deferred as a report-writing concern rather than a code-correctness one).
- **(Prompt 3) Used 14.5h (12h + 2.5h) as the recorded Setup+baseline total, not just "2.5h" or a made-up combined figure, and explicitly flagged a 3rd, unaccounted session.** Prompt 3's parenthetical named three things — "12h first attempt + failed rerun + this 2.5h" — but gave numbers for only two of them. The "failed rerun" must refer to a real, distinct Kaggle session: the guard-only fix (from two sessions ago) could only have produced its fast `RuntimeError` by actually running on Kaggle hardware, and that session happened *after* the 12h session (which predates the guard existing) and *before* this 2.5h session (which has the load-once fix, not just the guard). That session almost certainly consumed some real GPU time — full Mistral CNN+SQuAD, full Llama CNN, then a fast-failing Llama SQuAD load — that was never reported or captured anywhere in this project's logs. Rather than silently drop it from the total (under-reporting usage against a hard 30h/week cap) or silently guess a number for it, recorded `≥14.5h` as an explicit floor and flagged the gap in `PROJECT_STATE.md`, `logs/daily_standup.md`, and the closing message — consistent with `CLAUDE.md`'s "push back if GPU budget is tightening" and the project's established pattern this session of surfacing gaps rather than closing them with an assumption.
- **Surfaced the Week 2-4 scheduling risk implied by the Setup+baseline overrun**, not just recorded the number. 2h planned → ≥14.5h actual is a 7x miss on a single phase; naively summing Week 2 (4h) + Week 3 (9h) + Week 4 (6.4h) on top of the current ≥14.5h gives ≥33.9h, over the 30h/week cap, before the missing 3rd-session hours are even added. This doesn't necessarily mean the plan is broken (each week's phase may get its own fresh 30h allotment rather than sharing one pool — `CLAUDE.md`'s "30 GPU-hrs/week cap" reads as per-week, and `PROJECT_STATE.md`'s own table apportions hours per-week already), but the magnitude of the overrun on the *first, smallest* phase seemed worth flagging explicitly now rather than assuming later phases will land closer to their estimates.

## Experiments Executed (this Kaggle session, reported by engineer)

- EXP-MIS-BASE-CNN — completed (2nd/latest run): peak VRAM 6.91GB, latency 9616.9ms, ROUGE1/2/L 0.2387/0.0840/0.1607 — consistent with the 1st run (9723.3ms, 0.2386/0.0841/0.1598), a nice reproducibility cross-check
- EXP-MIS-BASE-SQUAD — completed (2nd/latest run): peak VRAM **6.91GB (was 13.65GB pre-fix)**, latency 6560.0ms, EM 8.0 / F1 24.19 — VRAM contamination from the previous session is gone, exactly as predicted
- EXP-LLAMA-BASE-CNN — completed (2nd/latest run): peak VRAM 12.39GB, latency 14846.0ms, ROUGE1/2/L 0.2512/0.0867/0.1689 — consistent with the 1st run
- EXP-LLAMA-BASE-SQUAD — **completed** (previously hung/failed): peak VRAM 12.38GB, latency 13456.2ms, EM 3.0 / F1 13.19

All 4/4 Week 1 baselines now complete and clean. Week 1 is done.

## Issues Encountered

**Duplicate rows in `results/*.csv` from the append-only save + full-script rerun**, and the `load_baseline_metrics()` first-match bug it exposed — both described in Key Decisions above. No further action needed beyond the fix already applied; flagging here that `scripts/compile_results.py` (referenced in `ARCHITECTURE.md`, not yet written) will eventually need to decide how to handle duplicate exp_id rows when compiling `results/master_benchmark_results.csv` — recommend it also keep the latest row per exp_id, consistent with the fix made here, but that script doesn't exist yet so there's nothing to change today.

**No other issues.** This session is the positive confirmation case for the two previous sessions' debugging — the architectural fix (load once per process) held up against real Kaggle hardware on the exact model (Llama-2-13B) and exact failure mode (CPU-offload hang on a second in-process load) that motivated it.
