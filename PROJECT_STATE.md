# Project State: LLM Optimization for Resource-Constrained Systems

> Updated at the start of every work session (Kaggle or local). This is the single file any new Claude Code session should read FIRST. It reflects reality, not the plan — if something hasn't run yet, it says so.

**Last Updated:** 2026-08-21

**Current Phase:** WEEK 3/4 COMPLETE (minus ONNX) — **every non-ONNX experiment in the 22-experiment matrix is now done, on both models.** Mistral-7B: LoRA, QLoRA, 8-bit, 4-bit all complete. Llama-2-13B: 8-bit, 4-bit, and now **QLoRA (both datasets, as of 2026-08-20/21)** all complete — `EXP-LLAMA-QLORA-CNN` (14.43h total, recovered via `resume_from_checkpoint` after a 12h session-cap kill at step 123/126) and `EXP-LLAMA-QLORA-SQUAD` (5.73h, clean single-session run, within its projected rough bound). **The only 4 experiments remaining in the entire matrix are Mistral ONNX (CNN/SQuAD) and Llama ONNX (CNN/SQuAD)**, both already deferred pending Ada cluster access — see Blockers & Risks "Deferred: Mistral-7B ONNX export" and "Open Risk ... Llama-2-13B ONNX export." See EXPERIMENT_MATRIX.md's "Project-Level Finding: T4/Turing Quantization Slowdown — CONFIRMED" for the cross-technique, cross-model pattern (QLoRA train, QLoRA inference, 8-bit inference, 4-bit inference all substantially slower than baseline despite large VRAM savings, confirmed on both models). Week 3 and Week 4 quality gates evaluated below — see "Quality Gate Evaluation."

**Project Status:** 🟢 Feature-complete on everything Kaggle-feasible (18/22 experiments done; the remaining 4 are hardware-blocked, not behind schedule) and 🔴 GPU budget over its plan-level estimate — see GPU Budget Tracking below. Kaggle-account weekly quota is a separate, live number not tracked from here; the over-budget figure is against this project's own cumulative planning estimate, not necessarily a live blocker.

---

# Revision Note (read this first)

The original plan (`HYBRID_APPROACH_DETAILED_IMPLEMENTATION_GUIDE.md`, old `PROJECT_STATE.md`) assumed:
- Full FP32 fine-tuning of Mistral-7B and Llama-2-13B — this needs ~112GB and ~208GB respectively (weights + gradients + 2 Adam moments at 4 bytes each). Kaggle GPUs have 16GB. It cannot run.
- 133.5 total GPU hours, with 56 hours in Week 2 alone — Kaggle's cap is 30 GPU-hrs/week, so Week 2 alone already broke the weekly budget.
- Week 1 results (baseline latency, ROUGE scores, "GPU access confirmed") that were written before anything actually executed. Those numbers were never real. They have been removed.

This file now reflects a **right-sized, Kaggle-feasible plan**. Full details and rationale: `EXPERIMENT_MATRIX.md`. The old guide is kept at `archive/HYBRID_APPROACH_DETAILED_IMPLEMENTATION_GUIDE.md` for its qualitative content (model rationale, recovery procedures) but its experiment matrix and GPU-hour numbers are superseded and should not be trusted.

---

# Executive Summary

**Objective:** Compare LoRA, QLoRA, 8-bit quantization, 4-bit quantization, and ONNX Runtime inference optimization across two open-source LLMs (Mistral-7B, Llama-2-13B) on two tasks (summarization, QA). Produce a benchmark CSV, a FastAPI inference endpoint, a results dashboard, and a technical report.

**Scope:** 22 experiments (12 on Mistral-7B, 10 on Llama-2-13B — Llama skips fp16 LoRA training since it doesn't fit in 16GB VRAM; see `EXPERIMENT_MATRIX.md` for why).

**Hardware:** Kaggle free tier (T4x2 / P100, 16GB VRAM per GPU, 30 GPU-hrs/week cap).

**Estimated GPU budget:** ~20-25 hours of actual experiment time, ~40 hours with reruns/buffer. Comfortably fits in 2 weeks of Kaggle quota, leaving the rest of the timeline for report/API/dashboard work and recovery from failures.

**Timeline:** 10 weeks (unchanged from original ask), reallocated — see "What's Next."

---

# Component Status Summary

| Component | Status | Progress | Notes |
|-----------|--------|----------|-------|
| **Repo structure** | ✅ Passed | 100% | Matches `ARCHITECTURE.md`'s file structure; pushed to `github.com/sukhraj01/LLM_RCAS_Study` |
| **Environment setup** | ✅ Passed (local + Kaggle) | 100% | `.venv` (Python 3.11) created locally, `pip install -r requirements.txt` succeeds after fixing an `optimum[onnxruntime-gpu]`/macOS wheel conflict (see 2026-08-13 ai-usage-log). HF_TOKEN verified. Kaggle-side torch/CUDA now confirmed working via 2 real GPU sessions (baseline runs). The `optimum[onnxruntime-gpu]` extra install step is still unverified — no ONNX experiments have run yet (Week 4). |
| **Data pipeline** | ✅ Passed (local + Kaggle) | 100% | `python -m utils.data_loader` sanity check passed locally: CNN/DailyMail val=200/test=200, SQuAD val=200/test=200, no overlap by construction. Same loader confirmed working on real Kaggle sessions too — all 4 baseline experiments loaded real CNN/SQuAD samples successfully. |
| **Baseline inference** | ✅ Passed (Kaggle) | 100% | All 4 runs complete and clean: EXP-MIS-BASE-CNN, EXP-MIS-BASE-SQUAD, EXP-LLAMA-BASE-CNN, EXP-LLAMA-BASE-SQUAD (the one that hung on the first attempt). Load-once fix (see `experiments/common.py`) confirmed working on real hardware — no CPU-offload, no hang. Real numbers in `results/mis_results.csv` / `results/llama_results.csv` and `logs/experiment_tracking.csv`. Note: Llama-2-13B's SQuAD quality (F1 13.19) is notably lower than Mistral-7B's (F1 24.19) — plausibly because Llama-2-13b-hf is a non-instruction-tuned base model on zero-shot QA, not yet investigated further; worth a callout in the eventual report either way. |
| **Experiment tracking** | ✅ In progress | 18/22 rows | `logs/experiment_tracking.csv` populated for all 4 Week 1 baselines + both Week 2 LoRA (Mistral) + both Week 3 QLoRA (Mistral) + both Week 4 8-bit (Mistral) + both Week 4 4-bit (Mistral) + both Week 4 8-bit (Llama) + both Week 4 4-bit (Llama) + both Week 3 QLoRA (Llama) rows, all `CONFIRMED FINAL`, each now also tagged `hardware=T4`; only 4 rows still `pending` — Mistral + Llama ONNX (4 rows total), all deferred pending Ada cluster access (see Blockers & Risks). **Every experiment this project can run on Kaggle's free tier is now complete.** |
| **Week 2-4 experiments** | 🟡 Complete except ONNX | 14/22 (Week 2 LoRA [Mistral only] + Week 3 QLoRA [both models] + Week 4 8-bit/4-bit [both models, both datasets]) | Mistral LoRA, QLoRA, 8-bit, and 4-bit inference all complete. Llama-2-13B's 8-bit, 4-bit, and now QLoRA (both datasets, 2026-08-20/21) are all complete too — Llama's full non-ONNX technique set is done. `EXP-LLAMA-QLORA-CNN` needed `resume_from_checkpoint` after a 12h session-cap kill (14.43h total); `EXP-LLAMA-QLORA-SQUAD` ran clean in one session (5.73h). Mistral + Llama ONNX (Week 4 remainder, both models) is **deferred pending Ada cluster access**, not scheduled for further Kaggle attempts — see Blockers & Risks. This is the only remaining gap in the entire 22-experiment matrix. |
| **API + dashboard + report** | ⬜ Not started | 0% | Blocked on experiment results |

---

# Deliverables Checklist (vs. Assignment Requirements)

| # | Deliverable | Target | Status |
|---|-------------|--------|--------|
| 1 | Optimized LLM deployment pipeline | LoRA/QLoRA/quant scripts + Docker | ⬜ Not started |
| 2 | Benchmarking report | Master CSV (22 rows) + analysis | ⬜ Not started |
| 3 | Web-based inference API | FastAPI server | ⬜ Not started |
| 4 | Technical report | 10-12 pages | ⬜ Not started |
| *Bonus* | Performance dashboard | Streamlit | ⬜ Not started |
| *Bonus* | Deployment demo | Docker, low-resource | ⬜ Not started |

---

# GPU Budget Tracking

| Phase | Week(s) | Planned Hours | Used | Remaining | Status |
|-------|---------|----------------|------|-----------|--------|
| Setup + baseline | 1 | 2 | ≥14.5 | ≤-12.5 | 🔴 Over budget by 7x the plan (2h planned → ≥14.5h actual). All 4 baselines now complete and clean (see Component Status above). Breakdown: 12h (session 1, 2026-08-13, partial failure — 3/4 completed, `EXP-LLAMA-BASE-SQUAD` hung, killed by 12h session cap) + 2.5h (session 3, 2026-08-14, load-once fix — all 4 clean, estimated from per-sample timing, not exact Kaggle accounting). **Not yet included:** a 3rd, intermediate Kaggle session between those two (the guard-only fix test, which correctly fired a fast `RuntimeError` on `EXP-LLAMA-BASE-SQUAD` instead of hanging) had real but unrecorded GPU-hours — true total is ≥14.5h, exact figure still needed. |
| LoRA (Mistral only) | 2 | 4 | ~2.6 | ~1.4 | ✅ Complete, under budget. ~1.3h (0.86+0.41hr) burned on the buggy-collator run whose results were discarded, plus ~1.3h (0.89+0.42hr) on the confirmed rerun — both real GPU time, counted honestly even though only the rerun's results are final. |
| QLoRA (both models) | 3 | 9 | 9.86 | -0.86 | 🟡 **Mistral-only actuals already exceed the entire phase's budget.** `EXP-MIS-QLORA-CNN` (6.44h) + `EXP-MIS-QLORA-SQUAD` (3.42h) = 9.86h against a 3.0h Mistral-only estimate (3.3x miss) and a 9h whole-phase (both models) estimate — Llama-2-13B's two QLoRA experiments (budgeted ~6h combined) are deferred, not run. Root cause understood, not a bug: QLoRA trains ~14x slower per step than fp16 LoRA on these T4 GPUs (Turing lacks efficient native int4/bf16 tensor-core paths for bitsandbytes' compute), so the ~73% VRAM savings come at a real, and apparently underestimated, time cost. This phase-level over-budget figure is a project-plan concern (re-estimate Llama's QLoRA hours before scheduling it), separate from actual Kaggle-account weekly quota headroom — see "Kaggle Weekly Quota Reset" below, which is now healthy. **Update (2026-08-19): this ~6h Llama QLoRA estimate is now explicitly considered stale, not just "likely an underestimate."** The Llama 8-bit/4-bit inference batch below independently confirmed the same estimation method misses badly for Llama-2-13B specifically (~4.5x on inference), on top of the ~3.3x miss the estimation method already showed for Mistral QLoRA. Llama QLoRA's ~6h figure was derived from the old parameter-count-scaling heuristic (same family of estimate now shown wrong twice) — it should be **recalculated using this batch's actual measured throughput (s/sample, from the Kaggle logs) as the calibration point** before Llama QLoRA is scheduled, not reused as-is. **Recalculated 2026-08-19** (see EXPERIMENT_MATRIX.md "Llama-2-13B QLoRA time projection"): stacking Mistral's measured LoRA→QLoRA training ratio (~7.2-8.1x) with the observed Llama/Mistral scaling factor (~1.8x, grounded in VRAM's tight cross-model agreement — inference-latency scaling turned out noisier than originally assumed, 1.02x-2.05x observed, not "consistently ~1.8x") gives a rough bound of CNN ~9.7-13.5h, SQuAD ~5.1-7.2h. This is explicitly flagged as a rough bound, not a confident estimate — it stacks two independently-uncertain ratios on top of zero direct Llama training measurements. SQuAD is recommended as the calibration run (fits 12h cap with margin; CNN's range straddles the cap). **Update (2026-08-21): `EXP-LLAMA-QLORA-CNN` has now run — actual total 14.43h** (pre-kill ~11.93h + resumed 2.5h after a session-duration-cap kill at step 123/126, recovered via `resume_from_checkpoint`), **above even the rough bound's high end (13.5h).** See EXPERIMENT_MATRIX.md's "Actual CNN result" note. `EXP-LLAMA-QLORA-SQUAD` remains not run — still projected at ~5.1-7.2h, not actual. Combined QLoRA-phase actual/projected total is now ~20.6h (14.43 CNN actual + ~6.2 SQuAD central projection) against this row's original 9h whole-phase estimate — over 2x. |
| Quantization + ONNX | 4 | 6.4 | ~5.44 | ~0.96 | 🟡 Llama 8-bit/4-bit batch (`EXP-LLAMA-8BIT-CNN`/`SQUAD`/`EXP-LLAMA-4BIT-CNN`/`SQUAD`, 2026-08-18/19) measured **~5.44 GPU-hrs** (~2.9h 8-bit batch + ~2.54h 4-bit batch) against a planned 1.2h (4×0.3h) — a **~4.5x miss**, the same systematic per-experiment-estimate-too-low pattern already seen with Mistral QLoRA (3.3x miss). See EXPERIMENT_MATRIX.md's Full Matrix correction note. Mistral's own 8-bit/4-bit inference GPU-hours were never separately captured (pre-existing tracking gap, not fixed here) so the `Used` figure above is the Llama batch only — true phase total is somewhat higher. Mistral + Llama ONNX (5.0h combined estimate) remain deferred, not run, so not counted in `Used`. |
| Buffer / reruns | 5 | 8 | 0 | 8 | ⬜ Reserve |
| **TOTAL** | 1-5 | ~29.4 | ≥32.4 | ≤-3.0 | 🔴 **Now over the entire 5-phase plan-level budget**, before Llama's QLoRA (deferred, estimate stale — see below), the buffer week (8h), or either model's ONNX (deferred) are counted. This is the third technique family (QLoRA, then Mistral quant implicitly, now Llama quant explicitly) to come in substantially over its per-experiment estimate — worth an ADR-level note that this project's original GPU-hour estimation method (rough per-experiment guesses, not calibrated against any real measurement) has now missed by 3-4.5x on every technique it's been checked against post-hoc. Still **not a live Kaggle-account blocker** — see "Kaggle Weekly Quota Reset" below for the distinction between this planning-level budget and actual account quota — but the planning estimate itself should not be trusted for what's left (Llama QLoRA, buffer) without recalibration. |

### Kaggle Weekly Quota Reset (2026-08-15)

Kaggle's actual GPU-hour quota is tracked per Kaggle account on its own rolling weekly clock (resets every Sunday) — a **different clock from this document's project-phase "Week 1/2/3" labels**, which are the engineer's own planning periods and can span less or more than one real Kaggle quota-week depending on how fast sessions actually happen. This distinction became directly relevant today:

- **(a) Honest prior-week total, not smoothed over:** Before today's reset, known real GPU time in the same Kaggle quota-week included the CPU-offload hang (12h, wasted but legitimate debugging), the clean Week 1 baseline rerun (~2.5h), and Week 2 LoRA (~2.6h, buggy run + confirmed rerun combined — see LoRA row above) — a floor of **≥17.1h**, on top of which some unknown, non-zero portion of this session's ~11h QLoRA run (6.44h + 3.42h training + ~1h combined generation) also landed before the reset, since the reset happened *mid-training*, not at a session boundary. Kaggle does not expose an exact split at the reset boundary, so the true prior-week total cannot be recovered precisely — it plausibly met or exceeded the 30h cap once that unrecorded fraction plus the still-never-captured intermediate guard-fix-test session (flagged as an open gap since 2026-08-13, see Session 2) are accounted for. This was driven by legitimate debugging and experiment work, not wasted/duplicated effort, but is recorded honestly rather than assumed to have stayed under budget.
- **(b) Fresh 30h as of 2026-08-15:** The quota has reset. Budget tracking for *this* Kaggle week starts counting from this point. Since the exact pre/post-reset split of this session's ~11h is unknown, the conservative (safe) assumption for planning is to count the **entire** ~11h against the fresh quota — i.e., treat this week as having **~11h used, ~19h remaining** of the new 30h, even though the true remaining figure is likely somewhat higher. Use this ~19h figure, not the full 30h, when deciding what else to schedule this week.
- **(c) Reliability note for future sessions:** The reset did not interrupt the running Kaggle kernel this time — `EXP-MIS-QLORA-CNN`/`EXP-MIS-QLORA-SQUAD` completed cleanly straight through the boundary. **This should not be relied upon.** Long single-script multi-dataset runs (QLoRA especially, at ~5-6h per dataset on this hardware) are the most likely to span a reset boundary; keep the existing per-N-step checkpointing discipline (already required for the 12h session cap) rather than assuming a mid-run reset will always be this uneventful. Added as a new row in Identified Risks below.

Weeks 6-10 have no planned GPU spend (API, dashboard, report, polish) — they exist as slack if experiments run over.

---

# Quality Gate Evaluation (2026-08-21)

Run against `EXPERIMENT_MATRIX.md`'s "Quality Gates" section now that Llama-2-13B QLoRA (both datasets) is complete, making both Week 3's and Week 4's gates evaluable for the first time. **Reported plainly, not assumed passing** — see `CLAUDE.md` "Phase Transitions Require Gate Approval": do not move to the next phase without gate approval, and push back if a gate isn't actually passed.

### Week 3 Gate — "Both models' QLoRA experiments complete, peak VRAM stays under 14GB, training time visibly less than LoRA's"

| Criterion | Result |
|---|---|
| Both models' QLoRA experiments complete | ✅ PASS — Mistral CNN/SQuAD + Llama CNN/SQuAD, 4/4 |
| Peak VRAM stays under 14GB | ✅ PASS — Mistral 1.84-1.88GB, Llama 3.34GB, both comfortably under |
| Training time visibly less than LoRA's | 🔴 **FAIL** (Mistral) / **N/A** (Llama) |

**Overall: FAIL.** Mistral's QLoRA training was ~14x *slower* per step than fp16 LoRA (183.97s/step vs 12.83s/step) — the opposite of what this criterion expects, already documented as a real T4/Turing finding, not a bug. For Llama, this criterion cannot be evaluated at all: Llama skips fp16 LoRA entirely (doesn't fit 16GB, ADR-002), so there is no Llama LoRA training time to compare QLoRA against. This gate's wording assumed a "training time visibly less than LoRA's" outcome that this project's own hardware findings have since disproven for Mistral and made structurally inapplicable for Llama — this was flagged as an open question back in the Week 3 checklist ("worth deciding explicitly whether this gate's wording needs revisiting") and is now resolved by evidence: the wording needs revisiting, not the data.

### Week 4 Gate — "All 8 quantization + ONNX experiments complete, 4-bit VRAM measurably lower than 8-bit, ONNX latency measurably lower than fp16 baseline"

| Criterion | Result |
|---|---|
| All 8 quantization experiments complete | ✅ PASS — Mistral 8-bit/4-bit ×2 datasets + Llama 8-bit/4-bit ×2 datasets = 8/8 |
| ONNX experiments complete | 🔴 **FAIL — 0/4.** Both models' ONNX export deferred pending Ada cluster access, not run on Kaggle (documented, justified deferral after 4 failed Mistral attempts — see Blockers & Risks — not an unaddressed failure) |
| 4-bit VRAM measurably lower than 8-bit | ✅ PASS — both models: Mistral 1.81GB < 3.25GB, Llama 3.27GB < 5.89GB (`utils/validation.py`'s `check_quant_vram_relationship()`, no issues either time) |
| ONNX latency measurably lower than fp16 baseline | ⚪ **CANNOT EVALUATE** — no ONNX data exists for either model |

**Overall: FAIL, as literally written — solely because of the ONNX shortfall.** The quantization-only portion of this gate (6 of its implied 8-12 experiments, plus the VRAM-relationship check) fully passes. This is the same open question flagged in Session 5 ("do not write phase_summary.md or evaluate Week 4's quality gate until a decision is made on whether ONNX being deferred... counts as complete for gate purposes — flagged, not resolved") — now actually evaluated rather than left open, with a plain answer: **it does not pass as the gate is currently worded.**

**Decision needed, not made here:** either (a) formally split this gate into "quantization" (would pass) and "ONNX" (blocked on Ada access, tracked separately) so Week 4's quantization work can be marked complete without waiting on hardware outside this project's control, or (b) leave the gate as a single unit and accept it stays unpassed until Ada access materializes. Both are defensible; this file doesn't pick one without engineer input. Either way: **`logs/phase_summary.md` should not claim Week 4 passed its stated gate as currently worded** — if it's written, it needs to say so explicitly (quantization complete, ONNX deferred, gate not met in the literal sense).

---

# Recent Decisions (ADRs)

| ADR | Title | Status | Notes |
|-----|-------|--------|-------|
| ADR-001 | Drop full-FP32 fine-tuning | Decided | Doesn't fit 16GB Kaggle GPUs for 7B/13B models. Not required by assignment scope either — assignment asks for LoRA/QLoRA/PEFT/quantization comparison, not full FT. |
| ADR-002 | 2 models, reduced technique set on Llama-13B | Decided | Mistral-7B runs all 6 techniques; Llama-2-13B skips fp16 LoRA (needs ~26GB+, doesn't fit) but keeps QLoRA + both quantization + ONNX. |
| ADR-003 | 2 datasets instead of 3 | Decided | CNN/DailyMail (summarization) + SQuAD (QA). Alpaca/instruction-following dropped to control scope; can be added back in Week 5+ buffer if ahead of schedule. |
| ADR-004 | TensorRT is stretch, not required | Decided | ONNX Runtime is the primary inference-optimization comparison for all 22 experiments. TensorRT conversion attempted only if time allows (Week 4 buffer). |
| ADR-005 | T4 is the sole hardware target for all 22 experiments | Decided | Kaggle free tier = T4 only; not evaluated on a second GPU architecture. T4 is a real widely-deployed resource-constrained GPU (AWS G4, GCP, Kaggle/Colab), so results are relevant to real deployments, not arbitrary — but every finding is scoped to T4/Turing and shouldn't be generalized further without validation. See `EXPERIMENT_MATRIX.md` Limitations. |
| *(Future)* | Exact LoRA r/alpha values | Pending | Lock in during Week 1 planning, document in `EXPERIMENT_MATRIX.md` |

---

# Known Constraints & Assumptions

### Hardware Constraints
- GPU VRAM: 16GB (Kaggle T4/P100), not 24GB — this was wrong in the original plan
- GPU hours: 30/week cap, session length capped at 12 hours continuous
- CPU RAM / disk: Kaggle default notebook limits apply (check before large checkpoint saves)

### Project Assumptions
- QLoRA fits both models comfortably in 16GB (this is literally what QLoRA was designed for)
- fp16 LoRA fits Mistral-7B (~14GB base + adapters, batch size 1, gradient checkpointing) but not Llama-2-13B (~26GB, doesn't fit)
- ~~Quantized inference (8-bit, 4-bit) is cheap and fast (<20 min per run) on both models~~ — **disproven 2026-08-18/19:** true for VRAM (both models fit trivially, confirmed), false for GPU-hours. Llama-2-13B's 4-experiment 8-bit+4-bit batch took ~5.44 GPU-hrs combined (~1.4h/experiment average, not <20min) — see GPU Budget Tracking
- ONNX Runtime export/inference is the primary "inference optimization" comparison; TensorRT is a stretch goal

---

# Blockers & Risks

### Current Blockers
None specific to ONNX anymore — that's now a closed decision, not an open blocker (see "Deferred" entry below). Llama-2-13B QLoRA remains deferred by choice (Mistral 8-bit/4-bit was a faster, lower-risk use of the fresh quota), not blocked by an unresolved constraint.

### Resolved (2026-08-19): Llama QLoRA checkpointing was not actually protective as configured
While confirming the "checkpoint every 200 steps" recovery procedure was wired in before Llama QLoRA could be scheduled (per `CLAUDE.md`), found that `TrainingArguments(save_steps=200, ...)` in `experiments/common.py` never actually fired for either Llama QLoRA dataset: `batch_size=1` × `gradient_accumulation_steps=16` = effective batch 16, and 1000 training examples × 2 epochs gives only ~126 total optimizer steps — below the 200-step threshold. The only save was the final `model.save_pretrained()` after training completes; a run killed mid-training (12h session cap, Kaggle preemption) would have had nothing to resume from. This wasn't new — Mistral's QLoRA runs had the identical effective-batch-16 math and the same gap, they just never hit an interruption, so no existing results were affected. **Fixed 2026-08-19:** `save_steps` changed 200 → 25 in the single shared `TrainingArguments` call (applies to any future LoRA/QLoRA run, Llama or Mistral, not per-experiment). Verified before changing: `save_steps` appears exactly once in the codebase, no code hardcodes a `checkpoint-200`-style resume path. Verified after changing: Llama QLoRA's ~126 steps now produce 5 checkpoints (steps 25/50/75/100/125); Mistral QLoRA (identical effective batch 16) also gets 5; Mistral LoRA (effective batch 8, 250 steps) gets 10.

### Deferred (2026-08-16): Mistral-7B ONNX export marked infeasible on Kaggle's free tier — pending Ada cluster access

**`EXP-MIS-ONNX-CNN`/`EXP-MIS-ONNX-SQUAD` are not scheduled for further attempts on Kaggle.** After four consecutive failures — attempts 1-3 below, then a fourth where the verified in-memory-export fix (attempt 3) *still* hit a disk OOM against the real 7B model — the decision was made to stop spending Kaggle GPU hours on repeated attempts rather than pursue a fifth fix. Both Mistral's and Llama's ONNX experiments are now deferred pending access to a different GPU (university Ada cluster), given the same treatment. This does not retroactively invalidate the reasoning behind attempts 1-3 (each was a real, verified fix for the specific failure it targeted) — it reflects that fp16-scale ONNX export of a 7B+ model appears to exceed what Kaggle's free-tier disk/VRAM constraints can support via the paths tried so far, and continuing to iterate would cost GPU hours the project's budget doesn't have slack for at this point. See `EXPERIMENT_MATRIX.md`'s technique #6 section and Recovery Procedures "ONNX Export Disk OOM" for the full four-attempt history and final-outcome note.

### History (superseded by the deferral above): Mistral-7B ONNX export — three code-level attempts before the final, still-failing fourth (2026-08-15/16)

**Attempt 1 — VRAM OOM during export.** `EXP-MIS-ONNX-CNN`/`EXP-MIS-ONNX-SQUAD` hit Kaggle's "Your notebook tried to allocate more memory than is available" mid-export — a genuine VRAM ceiling hit. Root cause: the old code exported via `ORTModelForCausalLM.from_pretrained(model_id, export=True, provider="CUDAExecutionProvider", ...)`, and `main_export`'s `dtype` defaulted to `"fp32"` (unset by the old code) — a fp32 7B export is ~28GB, double the fp16 baseline's ~14GB, exceeding 16GB VRAM when loaded. Fixed by exporting via `main_export(..., device="cpu", dtype="fp16")` first.

**Attempt 2 — disk OOM, on a genuinely fresh Kaggle session.** After the VRAM fix, export died with Kaggle's disk-quota crash page: "Output: 20.94 GB." Kaggle enforces a fixed ~20GB quota on the notebook's *tracked output*, separate from raw disk (`df -h` showed 1.1TB free at the overlay level). The fp16 HF model cache (~14.5GB) plus the ONNX export's own output (~14GB) both landed in that same tracked-output path simultaneously, exceeding the quota regardless of session history. Attempted fix: redirect both via `%env HF_HOME=/opt/bin/hf_cache` + `%env ONNX_CACHE_DIR=/opt/bin/onnx_cache` (`/opt/bin` had ~119GB free per `df -h`).

**Attempt 3 — the `/opt/bin` redirect failed: `OSError: [Errno 30] Read-only file system`.** That mount has space but isn't writable on Kaggle — a dead end, no longer recommended in `README.md`. **Real fix:** the actual constraint is that `main_export()`/`from_pretrained(export=True)` both take a model path/id and load it internally, needing the cache (~14.5GB) *and* the growing ONNX output (~14GB) on disk simultaneously — no mount redirect changes that math once nothing else is writable. Found `optimum.exporters.onnx.onnx_export_from_model()` by reading `optimum`'s source — a lower-level function that accepts an **already-loaded PyTorch model object** instead of a path/id (this is literally what `main_export()` calls into internally). Rewrote both ONNX scripts: (1) load the model ourselves via `AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, token=HF_TOKEN)`; (2) once loaded, free its on-disk cache via `huggingface_hub.scan_cache_dir().delete_revisions(*commit_hashes).execute()` (the official cache-management API — removes actual blob files, not symlinks); (3) export via `onnx_export_from_model(model=model, output=onnx_dir, task="text-generation-with-past", device="cpu")` against the in-memory object. Disk now only ever holds the cache *or* the growing ONNX output, never both — Mistral's peak (~14.5GB either way) comfortably fits the ~20GB quota with no redirect needed. `ONNX_CACHE_DIR` kept in the code as a harmless override (no longer recommended pointing at `/opt/bin`).

**Cache-deletion safety was verified by reading transformers' actual loading code, not assumed:** `load_state_dict()` loads safetensors shards via `with safe_open(...) as f: ... f.get_tensor(k)`, which copies each tensor into a fresh torch-owned buffer and closes the file/mmap before `from_pretrained()` returns — no lingering handle survives to keep deleted blocks alive.

**This fix was verified more rigorously than the previous two** — not just offline import-checking, but a real end-to-end run (load fp16 → free cache → confirm the model still works via a forward pass → export → reload the exported graph → `generate()`) against `hf-internal-testing/tiny-random-MistralForCausalLM`, the same RMSNorm-based architecture family as the real model. **Passed completely.** One caveat surfaced honestly along the way: the identical pipeline failed on a tiny GPT2 (LayerNorm-based) stand-in with an ONNX Runtime `LayerNormalization` fp16/fp32 dtype-mismatch error during export validation — a known fp16-export quirk that did not reproduce on the Mistral-architecture test (Mistral has no `LayerNormalization` op at all).

**Attempt 4 — the real 7B run, and the fix still failed.** Despite passing the stand-in verification above, the fourth attempt against the actual Mistral-7B model on Kaggle hit a disk OOM again. The stand-in test, while a genuinely stronger check than attempts 1-2 got, did not fully predict real-world behavior at 7B scale — plausible gaps include ONNX's external-data serialization for graphs over the 2GB single-file protobuf limit (never exercised by a KB-scale stand-in model) or other scale-dependent transient disk usage during export. This was not further diagnosed — see "Deferred" above for why. Full writeup: `EXPERIMENT_MATRIX.md` Recovery Procedures "ONNX Export Disk OOM" and its "FINAL OUTCOME" note.

### Open Risk (same treatment as Mistral's, above — deferred pending Ada cluster access): Llama-2-13B ONNX export (`experiments/llama/05_onnx.py`) — never attempted, not scheduled on Kaggle
The in-memory-export fix above does **not** resolve Llama's disk problem the way it does Mistral's: even with cache and ONNX output never held simultaneously, the *peak* usage during either single phase is still ~26GB (fp16 13B weights) — which alone exceeds the ~20GB output quota, with no writable redirect target currently known (`/opt/bin` is out). On top of that, separately: Llama-2-13B's fp16 weights alone are already over Kaggle's 16GB VRAM ceiling, so the final `ORTModelForCausalLM.from_pretrained(..., provider="CUDAExecutionProvider")` inference-reload step won't fit regardless of how export itself is solved. Both constraints need to be separately resolved before scheduling — this has not been load-tested and should not be assumed to work. Tracked in `EXPERIMENT_MATRIX.md` Recovery Procedures as well.

### Resolved (2026-08-15): GPU budget hold from Mistral QLoRA overrun
`EXP-MIS-QLORA-CNN`/`EXP-MIS-QLORA-SQUAD` together used 9.86 GPU-hours, exceeding `EXPERIMENT_MATRIX.md`'s entire QLoRA-phase estimate (9h, both models combined) using Mistral alone. This was flagged the same day as an explicit blocker (do not schedule Llama QLoRA until actual Kaggle quota confirmed) per `CLAUDE.md`'s "stop and escalate" rule. It has since resolved itself via a natural weekly quota reset rather than requiring a scope cut — see GPU Budget Tracking above for the honest prior-week total and the fresh count. The underlying estimate miss (QLoRA ~14x slower per step than fp16 LoRA on T4/Turing GPUs) is still real and still worth re-estimating before scheduling Llama's QLoRA runs, which is why they remain deferred rather than immediately scheduled — but that is now a planning decision, not a hard blocker.

### Resolved (2026-08-14): EOS-masking collator bug, confirmed on real hardware
`EXP-MIS-LORA-SQUAD` originally completed with EM 0.0/200 (down from baseline EM 8.0). Root cause: `load_model_and_tokenizer()` sets `tokenizer.pad_token = tokenizer.eos_token` (Mistral's tokenizer ships no distinct pad token), and `run_training_experiment()` previously used `transformers.DataCollatorForLanguageModeling(mlm=False)`, whose default behavior masks any label token equal to `pad_token_id`'s *value* — not by actual padding position — to `-100`. Since `pad_token_id == eos_token_id`, this silently masked every genuine end-of-sequence token in every training target, for every technique/dataset that trains, not SQuAD alone — the model never got gradient signal for when to stop generating. Fixed: `experiments/common.py` now uses a custom `_CausalLMCollator` that masks by `attention_mask == 0` (real padding) instead.

**Confirmed on real hardware, not just code review.** Both `EXP-MIS-LORA-CNN` and `EXP-MIS-LORA-SQUAD` were rerun on Kaggle under the fixed collator on 2026-08-14. Final numbers (`results/mis_results.csv`):
- `EXP-MIS-LORA-CNN`: ROUGE1/2/L 0.2890/0.1074/0.1965 vs baseline 0.2387/0.0840/0.1607 (+21.1%), training_time_hrs 0.89, peak_vram_gb 6.92
- `EXP-MIS-LORA-SQUAD`: EM 85.5/F1 91.96 vs baseline EM 8.0/F1 24.19 (+280%), training_time_hrs 0.42, peak_vram_gb 6.93, inference_latency_ms 625.1 (down from 9327.6 pre-fix)

Spot-check via `save_debug_predictions()`'s dump (5 examples/dataset, `logs/debug_predictions/`) confirms the mechanism directly rather than just inferring it from aggregate metrics: SQuAD predictions now terminate cleanly at EOS and match references almost exactly, instead of running on to `max_new_tokens`. See `EXPERIMENT_MATRIX.md` "Qualitative Notes for Report" for two caveats worth carrying into the writeup: (1) CNN's ROUGE gain is real but the raw predictions show repetition loops and a rambling style, not clean bullet-point summarization — a genuine base-model/small-LoRA quality limit, not a pipeline defect; (2) SQuAD's 10.49x latency speedup is the EOS fix letting the model stop generating, not an inherent LoRA inference speedup — don't cite it in isolation as "LoRA is 10x faster."

Both rows in `logs/experiment_tracking.csv` are now marked `CONFIRMED FINAL`, replacing the prior `NEEDS RERUN` status.

### Identified Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|-----------|
| Kaggle session timeout (12h cap) mid-experiment | Medium | Lost progress | Checkpoint every N steps, resume capability required from day 1 |
| Weekly quota (30h) exhausted mid-week | Low, given ~20-25h total planned | Delay | Buffer week reserved (Week 5) |
| LoRA fp16 training OOMs on Mistral-7B despite estimate | Medium | Experiment blocked | Fallback: gradient checkpointing already planned; further fallback is treating it as QLoRA-only for Mistral too, documented as ADR if triggered |
| Kaggle/local state drift (forget to sync) | Medium | Confusion, lost work | `KAGGLE_SYNC.md` protocol — follow it every session |
| Kaggle weekly quota resets mid-session | Confirmed possible (observed 2026-08-15) | None observed so far (session continued and completed cleanly through the boundary), but not guaranteed every time | Do not rely on it; keep per-N-step checkpointing regardless (same discipline already required for the 12h session cap). Long single-script multi-dataset runs (QLoRA especially, ~5-6h/dataset on this hardware) are the most likely to span a reset boundary. |

---

# What's Next

## Week 1 — COMPLETE
- [x] Create repo structure (see `ARCHITECTURE.md`)
- [x] Set up Kaggle notebook + local venv, pin dependency versions in `requirements.txt`
- [x] Download CNN/DailyMail + SQuAD samples
- [x] Run 4 real baseline inference experiments (2 models × 2 datasets), log real numbers — done across two Kaggle sessions (see `logs/daily_standup.md` 2026-08-13/2026-08-14: first attempt hung on EXP-LLAMA-BASE-SQUAD, root-caused, fixed, rerun succeeded)
- [x] Initialize `logs/experiment_tracking.csv`
- [x] First entry in `knowledge/ai-usage-log/`

## Week 2 — COMPLETE
- [x] Verify config matches `EXPERIMENT_MATRIX.md` technique #2 (r=8, alpha=16, target_modules q_proj/v_proj, batch=1/grad_accum=8, lr=2e-4, 2 epochs)
- [x] Confirm `experiments/mistral/02_lora.py` (now using `run_training_multi_dataset()`) picks up the clean baseline via `require_baseline_metrics("MIS", ...)`
- [x] Run `EXP-MIS-LORA-CNN` and `EXP-MIS-LORA-SQUAD` on Kaggle — required two sessions: first hit an EOS-masking collator bug (fixed, see Current Blockers "Resolved" note), confirmed rerun completed 2026-08-14
- [x] Log real GPU-hours, peak VRAM, training time, quality vs. Mistral baseline — both rows `CONFIRMED FINAL` in `logs/experiment_tracking.csv`
- [x] Quality gate check per `EXPERIMENT_MATRIX.md` ("training time roughly 40-60% of full FT cost, no OOM") — passed, no OOM, peak VRAM 6.92-6.93GB, training time well under 1hr each

## Week 3 (COMPLETE): QLoRA fine-tuning, both models, both datasets
- [x] Verify config matches `EXPERIMENT_MATRIX.md` technique #3
- [x] Run `EXP-MIS-QLORA-CNN`/`EXP-MIS-QLORA-SQUAD` on Kaggle — complete, `CONFIRMED FINAL`
- [x] Run `EXP-LLAMA-QLORA-CNN`/`EXP-LLAMA-QLORA-SQUAD` on Kaggle (2026-08-20/21) — complete, `CONFIRMED FINAL`. CNN needed `resume_from_checkpoint` after a 12h session-cap kill at step 123/126 (14.43h total: ~11.93h pre-kill + 2.5h resumed); SQuAD ran clean in one session (5.73h, within its ~5.1-7.2h projected rough bound)
- [x] Log real GPU-hours, peak VRAM, training time, quality vs. baseline for Mistral
- [x] Log same for Llama
- [x] Quality gate check per `EXPERIMENT_MATRIX.md` — **evaluated 2026-08-21, see "Quality Gate Evaluation" above: FAIL.** "Training time visibly less than LoRA's" fails for Mistral (QLoRA was ~14x *slower* per step) and is N/A for Llama (no LoRA baseline exists to compare against — ADR-002). The other two criteria (both models' QLoRA complete, VRAM under 14GB) pass. Gate wording needs revisiting, not the data.

## Week 4 (Kaggle scope closed out on Mistral; ONNX deferred off-Kaggle for both models): 8-bit + 4-bit quantized inference, ONNX export
- [x] Run `EXP-MIS-8BIT-CNN`/`EXP-MIS-8BIT-SQUAD` on Kaggle — complete, `CONFIRMED FINAL`. Peak VRAM 3.25GB (-53% vs baseline) but inference 2.6-3x *slower* than baseline on both datasets — third consecutive quantization technique confirming the T4/Turing slowdown pattern (see EXPERIMENT_MATRIX.md Qualitative Notes)
- [x] Run `EXP-MIS-4BIT-CNN`/`EXP-MIS-4BIT-SQUAD` on Kaggle — complete, `CONFIRMED FINAL`. Peak VRAM 1.81GB (-74% vs baseline, largest saving of any technique) but inference 1.5-1.7x *slower* than baseline on both datasets — fourth and final data point, T4/Turing quantization-slowdown finding now marked CONFIRMED in EXPERIMENT_MATRIX.md. `utils/validation.py`'s `check_quant_vram_relationship(1.81, 3.25)` confirms 4-bit VRAM < 8-bit VRAM as expected, no issues
- [x] Log real GPU-hours, peak VRAM, latency, quality vs. baseline for 8-bit
- [x] Log same for 4-bit
- [x] **Decision made (2026-08-16), not a completed experiment:** `EXP-MIS-ONNX-CNN`/`EXP-MIS-ONNX-SQUAD` deferred pending Ada cluster access after 4 failed Kaggle attempts (VRAM OOM → disk OOM → failed `/opt/bin` redirect → verified-but-still-failing in-memory export). Not scheduled for further Kaggle attempts. See Blockers & Risks "Deferred: Mistral-7B ONNX export" and `EXPERIMENT_MATRIX.md`'s technique #6 / Recovery Procedures "FINAL OUTCOME" note.
- [x] Run `EXP-LLAMA-8BIT-CNN`/`EXP-LLAMA-8BIT-SQUAD` on Kaggle (2026-08-18) — complete, `CONFIRMED FINAL`. Peak VRAM 5.89GB (-52% vs baseline 12.39/12.38GB, matched the ~5.8-5.9GB pre-run projection); inference ~1.75-1.9x *slower* than baseline on both datasets (speedup_factor 0.57/0.52) — first cross-model replication of the T4/Turing quantization-slowdown pattern (previously Mistral-7B only), now 5 data points across both models (see EXPERIMENT_MATRIX.md Qualitative Notes). Separately, Llama's SQuAD quality is weak both pre- and post-quantization (baseline F1 13.19 → 8-bit F1 9.71) — flagged as a model-specific finding, distinct from the hardware finding (see EXPERIMENT_MATRIX.md new subsection)
- [x] Run `EXP-LLAMA-4BIT-CNN`/`EXP-LLAMA-4BIT-SQUAD` on Kaggle (2026-08-19) — complete, `CONFIRMED FINAL`. Peak VRAM 3.27GB (-74% vs baseline, correctly lower than 8-bit's 5.89GB); inference ~1.6-1.7x *slower* than baseline on both datasets (speedup_factor 0.6/0.64), but faster than 8-bit's own latency — same relative ordering as Mistral's 4-bit-vs-8-bit result. Second cross-model replication (6th data point overall) — both bit-widths now independently confirm the T4/Turing quantization-slowdown pattern isn't Mistral-specific (see EXPERIMENT_MATRIX.md Qualitative Notes). **Llama's 8-bit and 4-bit inference are now both fully complete on both datasets** — this closes out the quantization-inference portion of Week 4 for both models (ONNX for both remains deferred, see below). GPU-hour note: this 4-experiment Llama quant batch (8-bit + 4-bit) measured ~5.44 GPU-hrs combined against a 1.2h planned estimate (~4.5x miss) — see GPU Budget Tracking and EXPERIMENT_MATRIX.md's Full Matrix correction note.
- [x] Llama-2-13B QLoRA's stale ~6h estimate **recalculated** (2026-08-19) then **run** (2026-08-20/21): CNN actual 14.43h (over the projection's own rough-bound high end of 13.5h — correctly flagged as the risky dataset), SQuAD actual 5.73h (within its ~5.1-7.2h bound). See EXPERIMENT_MATRIX.md "Llama-2-13B QLoRA time projection" for the full methodology and the "Actual CNN result" note closing the loop.
- [x] **Llama-2-13B's full non-ONNX technique set is now complete** — LoRA (skipped, ADR-002), QLoRA, 8-bit, 4-bit all done on both datasets. Llama's ONNX export was never attempted and is deferred the same way as Mistral's (13B fp16 weights alone exceed 16GB VRAM, and its disk math is even worse than Mistral's) — see Blockers & Risks "Open Risk ... Llama-2-13B ONNX export". **This is now the only remaining gap in the entire 22-experiment matrix.**

## Week 5 (buffer + compile)
- Reruns for any failed/inconsistent experiments
- Compile `master_benchmark_results.csv` (22 rows)
- If ahead of schedule: add Alpaca as a 3rd dataset, or attempt TensorRT

## Weeks 6-10 (deliverables)
- Weeks 6-7: API (FastAPI) + Dashboard (Streamlit)
- Weeks 8-9: Technical report
- Week 10: Buffer + polish + submission

---

# Session Notes

### Session 1: Plan Correction (2026-08-12)
- Reviewed original hybrid plan against real Kaggle hardware (16GB, 30h/week)
- Found full-FP32-fine-tuning of 7B/13B models infeasible on this hardware (~112GB/~208GB needed)
- Found Week 2 budget (56h) already exceeded weekly quota (30h) in the original plan
- Found `PROJECT_STATE.md` contained fabricated "completed" results with no experiments actually run
- Right-sized to 22 experiments, ~20-25 GPU-hrs core work, ~40h with buffer
- Rewrote `PROJECT_STATE.md`, `ARCHITECTURE.md`; created `EXPERIMENT_MATRIX.md`, `KAGGLE_SYNC.md`; updated `CLAUDE.md`
- Original guide archived, not deleted (useful qualitative content, wrong numbers)

**Decisions made:** ADR-001 through ADR-004 (see above)

**Blockers:** None — ready to start real Week 1

**Next session:** Actually execute Week 1 (repo setup, env, data download, real baselines)

### Session 2: Week 1 Execution (2026-08-13 → 2026-08-14)
- Verified local dev environment end-to-end: venv (Python 3.11), fixed a real `optimum[onnxruntime-gpu]`/macOS wheel dependency conflict in `requirements.txt`, confirmed HF_TOKEN, data pipeline sanity check passed, all 11 experiment scripts syntax/import-checked
- First Kaggle baseline session: 3/4 baselines completed, `EXP-LLAMA-BASE-SQUAD` hung 5+ hours and was killed by the 12h session cap
- Root-caused and fixed in two passes: (1) added a fail-fast device-placement guard + explicit model cleanup (symptom fix — turned the silent hang into a fast error); (2) identified the *actual* cause (reloading a model a second time in the same process causes CUDA allocator fragmentation on 13B models) and restructured `experiments/common.py` so every process loads its model once and loops datasets in-memory (`run_inference_multi_dataset()`, `run_training_multi_dataset()`) — verified offline via logic tests, not yet on real hardware at time of the fix
- While merging real baseline numbers, found and fixed two more latent bugs before they could block Week 2: a numpy-repr serialization bug in `compute_rouge()` that broke `ast.literal_eval()` round-tripping, and a results-filename mismatch between `ARCHITECTURE.md`'s docs and what the code actually reads/writes (`mis_results.csv`, not `mistral_results.csv`)
- Second Kaggle session (post-fix): all 4 baselines completed cleanly, including the one that hung before — load-once fix confirmed working on real hardware. Found (and fixed) a related latent bug: `load_baseline_metrics()` returned the *first* matching row instead of the *latest*, which would have silently picked up stale data given `results/*.csv`'s append-only design plus this session's rerun-created duplicate rows
- Full verbatim session logs: `knowledge/ai-usage-log/2026-08-13_env-verification.md`, `2026-08-13_kaggle-baseline-hang.md`, `2026-08-14_baseline-rerun-clean.md`

**Decisions made:** Load-once-per-process architecture for all technique scripts (documented in `EXPERIMENT_MATRIX.md`/`CLAUDE.md` recovery procedures); `load_baseline_metrics()` takes the latest matching row, not the first

**Blockers:** Session 3's (2026-08-14 rerun) GPU-hours now recorded (~2.5h, estimated from per-sample timing). Still open: the intermediate guard-fix-test session's exact GPU-hours were never captured — running total (≥14.5h) is a confirmed floor, not the true figure. Not blocking Week 2, but should be resolved before the Setup+baseline phase is called fully closed out.

**Next session:** Week 2 — LoRA fine-tuning on Mistral-7B, both datasets

### Session 3: Week 2 LoRA Rerun Confirmed (2026-08-14)
- Reran `EXP-MIS-LORA-CNN` and `EXP-MIS-LORA-SQUAD` on Kaggle under the fixed `_CausalLMCollator` (previous session's EOS-masking bug fix, applied but not yet tested on real hardware)
- Confirmed on real hardware, not just code review: `EXP-MIS-LORA-SQUAD` EM 85.5/F1 91.96 (vs. buggy-run EM 0.0/F1 7.76, vs. baseline EM 8.0/F1 24.19), inference latency 625.1ms (down from 9327.6ms pre-fix); `EXP-MIS-LORA-CNN` ROUGE1/2/L 0.2890/0.1074/0.1965 (vs. baseline 0.2387/0.0840/0.1607, +21.1%)
- Spot-checked `logs/debug_predictions/` for both experiments (5 examples each): SQuAD predictions now terminate cleanly and match references almost exactly, confirming the EOS-masking mechanism directly; CNN predictions still show repetition loops and rambling continuation style despite the ROUGE improvement — a real base-model/small-LoRA quality limitation, documented in `EXPERIMENT_MATRIX.md` "Qualitative Notes for Report" so it isn't overstated later
- Explicitly documented that SQuAD's 10.49x speedup_factor is the EOS fix letting generation stop early, not an inherent LoRA inference speedup — flagged so it isn't misread in isolation during report writing
- Updated `logs/experiment_tracking.csv` (both LoRA rows now `CONFIRMED FINAL`), `PROJECT_STATE.md` (blocker resolved, Week 2 checklist closed out, GPU Budget Tracking updated with actual LoRA-phase hours including the discarded buggy run), `EXPERIMENT_MATRIX.md` (new Qualitative Notes section)

**Decisions made:** Both Week 2 LoRA results are accepted as final. Quality gate for Week 2 passed (no OOM, VRAM well under budget, training time reasonable) — proceeding to Week 3 (QLoRA)

**Blockers:** None

**Next session:** Week 3 — QLoRA fine-tuning, both models, both datasets

### Session 4: Week 3 QLoRA, Mistral-7B Only (2026-08-15)
- Ran `EXP-MIS-QLORA-CNN` and `EXP-MIS-QLORA-SQUAD` on Kaggle — both completed cleanly, no OOM, no NaN
- Real numbers (`results/mis_results.csv`): CNN ROUGE1/2/L 0.2775/0.0975/0.1868 (baseline 0.2387/0.0840/0.1607, +16.3%), training_time_hrs 6.44, peak_vram_gb 1.84, inference_latency_ms 16658.0 (speedup_factor 0.58, slower than baseline); SQuAD EM 83.0/F1 90.03 (baseline EM 8.0/F1 24.19, +272%), training_time_hrs 3.42, peak_vram_gb 1.88, inference_latency_ms 2465.5 (speedup_factor 2.66)
- Notable, report-worthy finding documented in `EXPERIMENT_MATRIX.md` Qualitative Notes: QLoRA trained ~14x slower per step than fp16 LoRA (183.97s/step vs 12.83s/step) despite ~73% less peak VRAM — plausibly T4/Turing GPUs lacking efficient native int4/bf16 tensor-core paths for bitsandbytes' compute. A legitimate hardware-specific trade-off, not a bug.
- **Did not run Llama-2-13B QLoRA.** Mistral-only QLoRA GPU-hours (9.86h) already exceeded `EXPERIMENT_MATRIX.md`'s entire QLoRA-phase estimate (9h, both models) — flagged as an explicit blocker per `CLAUDE.md`'s "stop and escalate" resource-constraint rule rather than proceeding to schedule Llama's runs on the assumption the plan still holds
- Updated `logs/experiment_tracking.csv` (both Mistral QLoRA rows `CONFIRMED FINAL`), `results/mis_results.csv` (added the two rows — flagged to engineer for verification since they weren't yet synced when reported), `EXPERIMENT_MATRIX.md` (new QLoRA trade-off note), `logs/daily_standup.md`
- **Follow-up, same day:** engineer reported that this session's QLoRA run (~11h wall-clock: 6.44h + 3.42h training plus ~1h combined generation) crossed a Kaggle weekly-quota reset boundary partway through training — the session was not interrupted and both experiments completed cleanly. Corrected `results/mis_results.csv`'s two QLoRA rows to the engineer's exact real values (full-precision `quality_metrics`, real adapter-save-path notes) — the values added earlier the same session were rounded approximations written before the real CSV content was available, now superseded. Documented the reset honestly in a new GPU Budget Tracking "Kaggle Weekly Quota Reset" section: prior-week total is a floor of ≥17.1h plus an unrecoverable unknown fraction of the ~11h QLoRA session (plausibly at or over 30h once combined with the still-uncaptured intermediate guard-fix-test session from 2026-08-13); fresh 30h quota starts counting from today, conservatively treated as ~11h already used / ~19h remaining. Added a new Identified Risks row: don't rely on a mid-session reset being this uneventful again.

**Decisions made:** Mistral QLoRA results accepted as final. Week 3 quality gate explicitly **not** marked passed — the "training time visibly less than LoRA's" criterion actually failed (QLoRA was slower, not faster, in total training time), which is itself worth a decision on whether the gate wording needs revisiting. Llama QLoRA explicitly **not** scheduled this session, but the reason changed over the course of the day: from "GPU-budget-blocked, needs confirmation" to "deferred by choice now that quota is confirmed fresh, Mistral 8-bit/4-bit prioritized instead as a faster use of it." Week 4 (Mistral 8-bit/4-bit) opened as "starting now."

**Blockers:** None. The budget-confirmation blocker raised earlier the same session resolved itself via the quota reset rather than requiring engineer action — see GPU Budget Tracking.

**Next session:** Run `EXP-MIS-8BIT-CNN`/`EXP-MIS-8BIT-SQUAD` and `EXP-MIS-4BIT-CNN`/`EXP-MIS-4BIT-SQUAD` on Kaggle (inference-only, expected fast). Revisit Llama-2-13B QLoRA scheduling afterward with a corrected GPU-hour estimate given Mistral's 3.3x miss.

### Session 5: Week 4, Llama-2-13B Quantization Inference (2026-08-18 → 2026-08-19)
- Ran `EXP-LLAMA-8BIT-CNN`/`EXP-LLAMA-8BIT-SQUAD` on Kaggle (2026-08-18), then `EXP-LLAMA-4BIT-CNN`/`EXP-LLAMA-4BIT-SQUAD` (2026-08-19) — engineer validated each batch's numbers (VRAM projection, quality-degradation formula, speedup formula) before reporting back; both batches confirmed correct on inspection here too (recomputed independently against `results/llama_results.csv`'s baseline rows and the project's established `(optimized-baseline)/baseline` / `baseline_latency/optimized_latency` formulas)
- Real numbers (`results/llama_results.csv`): 8-bit — peak VRAM 5.89GB (vs baseline 12.39/12.38GB, -52%, matched the pre-run ~5.8-5.9GB projection from the prior session almost exactly); CNN ROUGE1 0.2228 vs baseline 0.2512 (-11.32%), latency 26203.1ms vs baseline 14846.0ms (speedup_factor 0.57); SQuAD F1 9.71 vs baseline 13.19 (-26.38%), latency 25854.0ms vs baseline 13456.2ms (speedup_factor 0.52). 4-bit — peak VRAM 3.27GB (-74%, correctly lower than 8-bit's), matched the prior session's ~3.25-3.3GB projection; CNN ROUGE1 0.2359 vs baseline (-6.09%), latency 24641.4ms (speedup_factor 0.6); SQuAD F1 10.60 vs baseline (-19.64%), latency 21022.8ms (speedup_factor 0.64)
- **Confirmed the T4/Turing quantization-slowdown finding is not Mistral-specific.** Both bit-widths now independently replicate on Llama-2-13B with the same relative ordering Mistral showed (4-bit faster than 8-bit, both slower than baseline) — now 6 data points total, 2 full cross-model replications. Updated `EXPERIMENT_MATRIX.md`'s Project-Level Finding section (data point count, report-framing language, Limitations section counts) throughout, not just appending new bullets, so the "five"/"four" language left over from the 8-bit-only update didn't go stale
- **Flagged, separately from the hardware finding:** Llama-2-13B's SQuAD quality is weak independent of quantization (baseline F1 13.19 → 8-bit F1 9.71 → 4-bit F1 10.60) — added as its own subsection in `EXPERIMENT_MATRIX.md` Qualitative Notes, explicitly not merged with the T4/Turing finding since one is a hardware/speed effect and the other is a model/task-fit quality characteristic
- **GPU-hour estimate correction:** the engineer reported actual measured batch totals (~2.9 GPU-hrs for the 8-bit batch, ~2.54 GPU-hrs for the 4-bit batch, from Kaggle s/sample logs) — combined ~5.44 GPU-hrs against a planned 1.2h (4×0.3h/experiment), a ~4.5x miss, the same systematic-underestimate pattern already seen with Mistral QLoRA's 3.3x miss. Corrected `EXPERIMENT_MATRIX.md`'s Full Matrix `Est. GPU Hrs` column for the 4 affected rows (struck through the old `0.3` rather than silently replacing it) and the matrix's total-estimate line (~21.3h → ~25.5h). Updated `PROJECT_STATE.md` GPU Budget Tracking's Quantization+ONNX phase row and TOTAL row to reflect the real spend — the project total is now over its 5-phase plan-level budget (not the same thing as live Kaggle-account quota, which is tracked separately)
- **Did not recalculate Llama QLoRA's ~6h estimate**, per engineer's explicit instruction — it's built on the same discredited parameter-count heuristic that just missed by ~4.5x on Llama inference, so it's flagged as **stale** in multiple places (Current Phase, GPU Budget Tracking QLoRA row, Week 3/Week 4 checklists) but left at its old value pending a dedicated recalculation calibrated off this batch's actual throughput
- One self-caught correction mid-session: initially wrote fabricated per-experiment `gpu_hours_used` splits (1.45/1.09) into `logs/experiment_tracking.csv`'s 4-bit rows, extrapolating a number the engineer never actually gave (only the 2-experiment batch total was reported, not measured per-dataset). Caught before committing — reverted to leaving `gpu_hours_used` blank per-row (matching the existing convention for all inference-only rows) and put only the real batch-level total in notes/this file, per `CLAUDE.md`'s "never write a result that hasn't actually been measured"
- Also fixed several other now-stale counts found while in `EXPERIMENT_MATRIX.md`'s Limitations section for this update (12→16 completed, 10→6 pending, "four-technique"→"six-data-point" pattern references) and `PROJECT_STATE.md`'s Component Status Summary (12/22→16/22, 8/22→12/22) and Project Assumptions ("<20 min per run" quantized-inference assumption struck through as disproven) — not explicitly requested this session but directly contradicted by the numbers just logged, so left uncorrected they'd misrepresent reality
- Created `knowledge/ai-usage-log/2026-08-17_llama-quantization-inference.md` for this session (spans the precondition-check turn through both result turns) — CLAUDE.md's session-logging requirement had been skipped for the precondition-check turn that started this multi-turn session; backfilled here rather than left missing

**Decisions made:** Both Llama quantization-inference results accepted as final, `CONFIRMED FINAL`. Week 4's quantization portion (as opposed to ONNX) is now complete for both models. Llama QLoRA scheduling remains explicitly deferred, now for two independent reasons: the original "needs its own time-projection conversation given the parameter-count scale-up" reason, plus the newly-confirmed staleness of its existing estimate.

**Blockers:** None live. The project-level GPU-hour budget is now over its plan (see GPU Budget Tracking TOTAL row), which is a planning concern to address before committing to Llama QLoRA or the buffer week, not an active blocker on Kaggle-account quota.

**Next session:** Llama QLoRA time-projection conversation — recalculate its estimate from this session's actual measured throughput before scheduling anything. Mistral ONNX and Llama ONNX remain deferred pending Ada cluster access, not scheduled for Kaggle. Do not write `logs/phase_summary.md` or evaluate Week 4's quality gate until a decision is made on whether ONNX being deferred (not run, not failed-and-abandoned) counts as "complete" for gate purposes — flagged, not resolved, this session.

### Session 6: Llama-2-13B QLoRA Time Projection — Recalculated, Not Run (2026-08-19)
- Per engineer instruction, recalculated Llama QLoRA's stale ~6h estimate using real methodology instead of the round-number guess that caused the 4.5x miss on the 8-bit/4-bit batch — explicitly a projection/documentation exercise, no training scheduled or run
- Methodology: stacked Mistral's own measured LoRA→QLoRA training slowdown (CNN 0.89h→6.44h ≈7.2x, SQuAD 0.42h→3.42h ≈8.1x) with the observed Llama/Mistral cross-model scaling factor, using the measured Mistral QLoRA time as the base (`Llama_QLoRA ≈ Mistral_QLoRA_measured × scaling_factor`, mathematically equivalent to stacking both ratios from a hypothetical Llama LoRA baseline, since Llama has no real LoRA data — doesn't fit 16GB, see ADR-002)
- **Corrected the engineer's own framing of the scaling factor while deriving it.** The task described Llama/Mistral scaling as "observed... consistently ~1.8x" from both VRAM and latency. Rechecked all 6 actual cross-model latency ratios (baseline/8-bit/4-bit × CNN/SQuAD) individually rather than taking the "~1.8x" characterization at face value: they range 1.02x-2.05x (average ~1.58x), not tightly consistent — VRAM is the tight, ~1.8x-consistent signal (1.79-1.81x across all three techniques, matching the 13B/7.2B param ratio almost exactly). Used VRAM's tighter signal to justify the 1.8x central multiplier (training compute-per-step, like VRAM, plausibly scales with param count more directly than generation-length-confounded inference latency), documented the correction explicitly in `EXPERIMENT_MATRIX.md` rather than silently using 1.8x as if the premise were uncontested
- Central estimate: CNN ~11.6h, SQuAD ~6.2h. Rough bound (1.5x-2.1x range, trimming the 1.02x outlier as likely quantization-specific noise): **CNN ~9.7-13.5h, SQuAD ~5.1-7.2h**. Against the 12h Kaggle session cap: SQuAD fits with real margin even at its high end; CNN's central estimate leaves under 30 minutes of margin and its high end exceeds the cap outright
- **Recommendation: run `EXP-LLAMA-QLORA-SQUAD` first as the calibration run** — safer against the session cap, and its real measured throughput can replace the cross-model 1.8x guess entirely when projecting CNN afterward, rather than compounding cross-model uncertainty twice
- **Explicitly flagged as a rough bound, not a confident estimate**, per instruction: stacks two independently-uncertain ratios on top of zero direct Llama training measurements (Llama has never run a training job of any kind on this project). Documented this as a real epistemic gap in `EXPERIMENT_MATRIX.md`, not softened
- **Checked the "checkpoint every 200 steps" precondition rather than assuming the code already handled it**, per explicit instruction to confirm before scheduling. Found it does not actually protect Llama QLoRA as configured: `save_steps=200` in `experiments/common.py`'s `run_training_experiment()` never fires because Llama's QLoRA hyperparameters (`batch_size=1`, `gradient_accumulation_steps=16`, 1000 examples, 2 epochs) produce only ~126 total optimizer steps — below the threshold. A run killed mid-training would have nothing to resume from. Also checked whether this was a new problem: it isn't — Mistral's QLoRA runs had the same effective-batch-16 math and thus the same gap, they simply never got interrupted. Flagged in both `EXPERIMENT_MATRIX.md` and a new `PROJECT_STATE.md` Blockers & Risks entry, with a concrete fix recommendation (lower `save_steps` to ~25), but did not make the code change — out of scope for a projection-only session, and training hyperparameters shouldn't be touched without the same confirm-first discipline as everything else
- Updated `EXPERIMENT_MATRIX.md` (new "Llama-2-13B QLoRA time projection" subsection; struck-through and replaced the stale `3.0` GPU-hr estimates for both `EXP-LLAMA-QLORA-CNN`/`SQUAD` in the Full Matrix, same convention as the 8-bit/4-bit correction) and `PROJECT_STATE.md` (Current Phase, Component Status, GPU Budget Tracking QLoRA row, Week 3/4 checklists, new Blockers & Risks entry) so all "stale, not yet recalculated" pointers from the prior session now point at this projection instead of contradicting it

**Decisions made:** Projection accepted as documented — a rough bound, not treated as confident enough to schedule against without engineer sign-off. No training scheduled. SQuAD recommended over CNN as the calibration run if/when the engineer decides to proceed.

**Blockers:** One new, concrete pre-scheduling item: Llama QLoRA's checkpointing gap (see above) should be fixed before either experiment actually runs, not just before CNN — SQuAD's own ~5.1-7.2h projected range is long enough that losing all progress to an uncheckpointed interruption would still be costly even though it fits the 12h cap.

**Next session:** Engineer decision on whether/how to schedule `EXP-LLAMA-QLORA-SQUAD` as the calibration run — if yes, fix `save_steps` first, then run, then use its real measured throughput to tighten (or replace) the CNN projection before deciding on `EXP-LLAMA-QLORA-CNN`.

### Session 7: Fix QLoRA/LoRA Checkpointing Gap (2026-08-19)
- Committed Session 6's projection docs as-is (no further edits to the projection itself), then separately fixed the checkpointing gap that session flagged but didn't touch, per engineer instruction to keep the two changes distinct
- Before editing: verified `save_steps=200` appears exactly once in the codebase (`experiments/common.py`'s single shared `TrainingArguments` call, used by every LoRA/QLoRA experiment via `run_training_experiment()`) and that nothing else hardcodes a `checkpoint-200`-style resume path — safe to change in one place
- Changed `save_steps` 200 → 25. Verified the fix produces multiple checkpoints rather than just a lower single threshold: Llama QLoRA (batch=1, grad_accum=16, effective batch 16, 1000 examples × 2 epochs ≈ 126 steps) → checkpoints at steps 25/50/75/100/125, 5 total. Mistral QLoRA has the identical effective-batch-16 math → also 5. Mistral LoRA (effective batch 8, 250 steps) → 10
- Confirmed Mistral's already-completed LoRA/QLoRA runs are unaffected — they finished without interruption under the old (broken) threshold, so this is a forward-looking fix for future training, not a retroactive correctness concern for existing `results/mis_results.csv` rows
- No training run started, per explicit instruction
- Updated the two places that said "flagged, not fixed" (`EXPERIMENT_MATRIX.md`'s checkpointing paragraph, `PROJECT_STATE.md`'s Blockers & Risks entry and Current Phase line) to reflect the fix, so they don't go stale against the commit that just landed

**Decisions made:** `save_steps=25` accepted as the fix, applied globally (not per-experiment) since the same effective-batch-16 pattern affects both models' QLoRA runs. Still no decision on actually scheduling `EXP-LLAMA-QLORA-SQUAD` — that remains the engineer's call.

**Blockers:** None. The checkpointing precondition that was blocking a confident "yes, safe to schedule" recommendation is now resolved.

**Next session:** Engineer decision on whether to schedule `EXP-LLAMA-QLORA-SQUAD` as the calibration run now that both the time projection and the checkpointing fix are in place.

### Session 8: Llama QLoRA Both Datasets Complete — CNN Recovery, Hardware-Column Root Cause, SQuAD, Gate Evaluation (2026-08-19 → 2026-08-21)

Spans several distinct pieces of work, each already fully documented in its own commit message (see `git log`) — summarized here rather than duplicated in full:

- **`EXP-LLAMA-QLORA-CNN` ran, got killed by the 12h session cap at step 123/126, and was recovered.** Root-caused the kill (CNN's actual duration, ~11.93h pre-kill, was already near the projection's central estimate), added `--dataset`/`--resume`/`--checkpoint-source` to `experiments/llama/02_qlora.py` and `resume_from_checkpoint` support to `experiments/common.py` (commits `b01e1ce`, `800fd37`), walked through recovering the surviving `checkpoint-100` off a Kaggle notebook Output that had to be re-uploaded as a Dataset to escape a read-only input mount, verified checkpoint file integrity byte-for-byte before resuming, and logged the corrected total (14.43h = ~11.93h pre-kill + 2.5h resumed) rather than the resumed session's own misleading 2.5h reading (commit `5c01692`).
- **Found and fixed the actual root cause of the repeatedly-blank `hardware` column** (commit `7bdc508`): neither of `save_result()`'s two callers ever included a `"hardware"` key at all — structurally absent, not blank, which silently left-shifted every field after it whenever it happened. Fixed at the shared `save_result()` choke point with a new `HARDWARE = "T4"` constant, verified via simulation that field order now matches the header exactly.
- **Ran a full structural audit** of `results/llama_results.csv`, `results/mis_results.csv`, and `logs/experiment_tracking.csv` (42 rows total) before committing the fix, per engineer instruction not to assume the earlier manual backfill was correct just because it looked right — confirmed 0 broken rows, including field-by-field re-verification of the four Llama 8-bit/4-bit rows patched before the root cause was known. The earlier backfill held up.
- **`EXP-LLAMA-QLORA-SQUAD` ran clean in one session (5.73h, within its projected bound), completing Llama-2-13B's QLoRA technique and, with it, every non-ONNX experiment in the 22-experiment matrix (18/22 done).** Verified both `quality_degradation_percent` (591.46) and `speedup_factor` (3.33) independently before writing anything, per this project's established discipline.
- **Generalized the SQuAD-speedup EOS-effect note in `EXPERIMENT_MATRIX.md`** rather than treating Llama QLoRA's 3.33x as a new finding, per explicit instruction: rewrote the note (previously framed as LoRA-specific) to explicitly cover all three fine-tuned SQuAD speedups (Mistral LoRA 10.49x, Mistral QLoRA 2.66x, Llama QLoRA 3.33x) as one underlying mechanism — zero-shot baselines don't know to stop at EOS, any successful fine-tuning does teach that, regardless of technique or model.
- **Added a new qualitative note**: Llama QLoRA SQuAD (EM 84.5/F1 91.23) lands almost exactly where Mistral's fine-tuned SQuAD results do (LoRA EM 85.5/F1 91.96, QLoRA EM 83.0/F1 90.03) despite an ~11-point F1 gap between the two models' zero-shot baselines — fine-tuning appears to close most of the cross-model quality gap on this task, flagged as report-worthy but single-run/single-seed (Limitations #2).
- **Evaluated both the Week 3 and Week 4 quality gates plainly, per explicit instruction not to assume they pass** — see "Quality Gate Evaluation" section above. Both FAIL as literally worded: Week 3 on "training time visibly less than LoRA's" (disproven for Mistral, N/A for Llama), Week 4 solely on the ONNX shortfall (0/4, deferred, documented). Flagged as a decision point for the engineer (revise gate wording vs. accept as a standing documented exception) rather than resolved unilaterally.
- Assumption flagged rather than silently made: the engineer's SQuAD-completion instructions only mentioned appending to `logs/experiment_tracking.csv`, not `results/llama_results.csv` — appended to both anyway, consistent with every prior turn's convention and `logs/experiment_tracking.csv`'s own notes referencing "full row in results/llama_results.csv."

**Decisions made:** `EXP-LLAMA-QLORA-CNN` and `EXP-LLAMA-QLORA-SQUAD` both accepted as final. Project status updated to reflect that every Kaggle-feasible experiment (18/22) is now complete — remaining work (4 ONNX experiments) is hardware-blocked, not schedule-blocked. Quality gates reported as failing in their literally-worded form; no decision made on whether to revise their wording.

**Blockers:** None live. The GPU-hour planning estimate is now well over budget (~40h+ vs. the document's own 30-40h ceiling) — already documented, not a new surprise, and separate from live Kaggle-account quota (not tracked from here).

**Next session:** Engineer decision on the Week 3/4 gate-wording question above. If proceeding to Week 5 buffer/report work regardless of the gate's literal status, that should be an explicit documented choice, not a silent skip.
