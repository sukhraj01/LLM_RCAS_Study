# Project State: LLM Optimization for Resource-Constrained Systems

> Updated at the start of every work session (Kaggle or local). This is the single file any new Claude Code session should read FIRST. It reflects reality, not the plan — if something hasn't run yet, it says so.

**Last Updated:** 2026-08-15

**Current Phase:** WEEK 3/4 — Mistral-7B QLoRA, 8-bit inference, and 4-bit inference (all datasets) complete and confirmed on real hardware. Llama-2-13B QLoRA is **deferred by choice, not blocked** — Mistral-7B ONNX export is the remaining Week 4 technique. See EXPERIMENT_MATRIX.md's "Project-Level Finding: T4/Turing Quantization Slowdown — CONFIRMED" for the now-closed cross-technique pattern (QLoRA train, QLoRA inference, 8-bit inference, 4-bit inference all substantially slower than baseline despite large VRAM savings).

**Project Status:** 🟢 On track feature-wise (Weeks 1-2 complete, Week 3 half-done on Mistral) and 🟢 GPU budget healthy again after this week's quota reset — see GPU Budget Tracking below.

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
| **Experiment tracking** | ✅ In progress | 12/22 rows | `logs/experiment_tracking.csv` populated for all 4 Week 1 baselines + both Week 2 LoRA (Mistral) + both Week 3 QLoRA (Mistral) + both Week 4 8-bit + both Week 4 4-bit (Mistral) rows, all `CONFIRMED FINAL`; 10 rows still `pending` — Llama-2-13B QLoRA is deferred by choice (see GPU Budget Tracking), Mistral ONNX up next |
| **Week 2-4 experiments** | 🟡 In progress | 8/22 (Week 2 LoRA + Week 3 QLoRA + Week 4 8-bit/4-bit, Mistral-7B only, both datasets) | Mistral LoRA, QLoRA, 8-bit, and 4-bit inference all complete and confirmed on real hardware. Next up is Mistral ONNX export (Week 4 remainder) while Llama-2-13B QLoRA is deferred, not abandoned. |
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
| QLoRA (both models) | 3 | 9 | 9.86 | -0.86 | 🟡 **Mistral-only actuals already exceed the entire phase's budget.** `EXP-MIS-QLORA-CNN` (6.44h) + `EXP-MIS-QLORA-SQUAD` (3.42h) = 9.86h against a 3.0h Mistral-only estimate (3.3x miss) and a 9h whole-phase (both models) estimate — Llama-2-13B's two QLoRA experiments (budgeted ~6h combined) are deferred, not run. Root cause understood, not a bug: QLoRA trains ~14x slower per step than fp16 LoRA on these T4 GPUs (Turing lacks efficient native int4/bf16 tensor-core paths for bitsandbytes' compute), so the ~73% VRAM savings come at a real, and apparently underestimated, time cost. This phase-level over-budget figure is a project-plan concern (re-estimate Llama's QLoRA hours before scheduling it), separate from actual Kaggle-account weekly quota headroom — see "Kaggle Weekly Quota Reset" below, which is now healthy. |
| Quantization + ONNX | 4 | 6.4 | 0 | 6.4 | 🟡 Starting now (Mistral 8-bit/4-bit only — fast, inference-only, good fit for freshly-reset quota) |
| Buffer / reruns | 5 | 8 | 0 | 8 | ⬜ Reserve |
| **TOTAL** | 1-5 | ~29.4 | ≥26.96 | ≤2.44 | 🟡 Setup+baseline + LoRA + Mistral-only QLoRA combined have already used ≥92% of the entire 5-phase 29.4h *plan-level estimate*, with Llama's QLoRA (~6h, likely an underestimate given Mistral's 3.3x miss), all of Week 4 (6.4h), and the buffer week (8h) still ahead. This is a **planning-estimate concern, not a live blocker**: the estimate itself needs revisiting (flag as a future ADR candidate — QLoRA GPU-hour estimates were wrong by >3x), but actual Kaggle-account capacity is fine right now — see next note. |

### Kaggle Weekly Quota Reset (2026-08-15)

Kaggle's actual GPU-hour quota is tracked per Kaggle account on its own rolling weekly clock (resets every Sunday) — a **different clock from this document's project-phase "Week 1/2/3" labels**, which are the engineer's own planning periods and can span less or more than one real Kaggle quota-week depending on how fast sessions actually happen. This distinction became directly relevant today:

- **(a) Honest prior-week total, not smoothed over:** Before today's reset, known real GPU time in the same Kaggle quota-week included the CPU-offload hang (12h, wasted but legitimate debugging), the clean Week 1 baseline rerun (~2.5h), and Week 2 LoRA (~2.6h, buggy run + confirmed rerun combined — see LoRA row above) — a floor of **≥17.1h**, on top of which some unknown, non-zero portion of this session's ~11h QLoRA run (6.44h + 3.42h training + ~1h combined generation) also landed before the reset, since the reset happened *mid-training*, not at a session boundary. Kaggle does not expose an exact split at the reset boundary, so the true prior-week total cannot be recovered precisely — it plausibly met or exceeded the 30h cap once that unrecorded fraction plus the still-never-captured intermediate guard-fix-test session (flagged as an open gap since 2026-08-13, see Session 2) are accounted for. This was driven by legitimate debugging and experiment work, not wasted/duplicated effort, but is recorded honestly rather than assumed to have stayed under budget.
- **(b) Fresh 30h as of 2026-08-15:** The quota has reset. Budget tracking for *this* Kaggle week starts counting from this point. Since the exact pre/post-reset split of this session's ~11h is unknown, the conservative (safe) assumption for planning is to count the **entire** ~11h against the fresh quota — i.e., treat this week as having **~11h used, ~19h remaining** of the new 30h, even though the true remaining figure is likely somewhat higher. Use this ~19h figure, not the full 30h, when deciding what else to schedule this week.
- **(c) Reliability note for future sessions:** The reset did not interrupt the running Kaggle kernel this time — `EXP-MIS-QLORA-CNN`/`EXP-MIS-QLORA-SQUAD` completed cleanly straight through the boundary. **This should not be relied upon.** Long single-script multi-dataset runs (QLoRA especially, at ~5-6h per dataset on this hardware) are the most likely to span a reset boundary; keep the existing per-N-step checkpointing discipline (already required for the 12h session cap) rather than assuming a mid-run reset will always be this uneventful. Added as a new row in Identified Risks below.

Weeks 6-10 have no planned GPU spend (API, dashboard, report, polish) — they exist as slack if experiments run over.

---

# Recent Decisions (ADRs)

| ADR | Title | Status | Notes |
|-----|-------|--------|-------|
| ADR-001 | Drop full-FP32 fine-tuning | Decided | Doesn't fit 16GB Kaggle GPUs for 7B/13B models. Not required by assignment scope either — assignment asks for LoRA/QLoRA/PEFT/quantization comparison, not full FT. |
| ADR-002 | 2 models, reduced technique set on Llama-13B | Decided | Mistral-7B runs all 6 techniques; Llama-2-13B skips fp16 LoRA (needs ~26GB+, doesn't fit) but keeps QLoRA + both quantization + ONNX. |
| ADR-003 | 2 datasets instead of 3 | Decided | CNN/DailyMail (summarization) + SQuAD (QA). Alpaca/instruction-following dropped to control scope; can be added back in Week 5+ buffer if ahead of schedule. |
| ADR-004 | TensorRT is stretch, not required | Decided | ONNX Runtime is the primary inference-optimization comparison for all 22 experiments. TensorRT conversion attempted only if time allows (Week 4 buffer). |
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
- Quantized inference (8-bit, 4-bit) is cheap and fast (<20 min per run) on both models
- ONNX Runtime export/inference is the primary "inference optimization" comparison; TensorRT is a stretch goal

---

# Blockers & Risks

### Current Blockers
`EXP-MIS-ONNX-CNN`/`EXP-MIS-ONNX-SQUAD` have not yet run successfully — two consecutive failures now (VRAM OOM, then disk OOM), see "Resolved (code fixes only, unverified end-to-end)" below. Not a GPU-budget blocker (fresh quota, see GPU Budget Tracking), but do not schedule the ONNX Kaggle session until both code fixes below are reviewed. Llama-2-13B QLoRA remains deferred by choice (Mistral 8-bit/4-bit was a faster, lower-risk use of the fresh quota), not blocked by an unresolved constraint.

### Resolved (code fixes only, unverified end-to-end on hardware): Mistral-7B ONNX export — two consecutive OOMs, both fixed in code (2026-08-15)

**Failure 1 — VRAM OOM during export.** `EXP-MIS-ONNX-CNN`/`EXP-MIS-ONNX-SQUAD` hit Kaggle's "Your notebook tried to allocate more memory than is available" mid-export — confirmed via nvidia-smi/Kaggle's crash banner, a genuine VRAM ceiling hit, not a script bug. `EXPERIMENT_MATRIX.md`'s VRAM math never separately projected the export step, only steady-state inference — that's the gap. Two compounding causes found by reading `experiments/mistral/06_onnx.py` and `optimum.exporters.onnx.main_export`'s source in the pinned version: (1) the old code exported via `ORTModelForCausalLM.from_pretrained(model_id, export=True, provider="CUDAExecutionProvider", ...)`, which does export/trace and CUDA-session creation together rather than keeping the trace off-GPU; (2) more concretely, `main_export`'s `dtype` defaults to `"fp32"` when not set, and the old code never set it — a fp32 7B export is ~28GB, double the fp16 baseline's ~14GB, which alone exceeds 16GB when loaded onto GPU regardless of where tracing happened. Fixed in both `experiments/mistral/06_onnx.py` and `experiments/llama/05_onnx.py`: export via `main_export(..., device="cpu", dtype="fp16")` (CPU, same precision as the fp16 baseline), then load the exported graph via `ORTModelForCausalLM.from_pretrained(onnx_dir, provider="CUDAExecutionProvider")` for the actual benchmarked GPU inference loop. Also turned the "first run only, reuses onnx_dir" caching claim into a real check (previously just a comment, not implemented).

**Failure 2 — disk OOM during export, on a genuinely fresh Kaggle session.** After the VRAM fix let export actually start running, it died with Kaggle's disk-quota crash page showing "Output: 20.94 GB." Confirmed not accumulated-history related (fresh session) and not raw-disk-related (`df -h` showed 1.1TB free at the overlay level) — Kaggle separately enforces a fixed ~20GB quota on the notebook's *tracked output* specifically, and the fp16 HF model cache (~14.5GB) plus the ONNX export's own output (~14GB complete) both land inside that same tracked-output path by default, together exceeding the quota regardless of session freshness. `df -h` also showed `/opt/bin` (122GB, ~119GB free) and `/kaggle/lib` (20GB, nearly empty) as separate mounts presumed outside that tracking. Fixed: both ONNX scripts now read `onnx_dir` from `ONNX_CACHE_DIR` (env var, falls back to the previous `CHECKPOINTS_DIR`-based path when unset — local/default behavior unchanged), and `README.md`'s Kaggle setup section now recommends `%env HF_HOME=/opt/bin/hf_cache` + `%env ONNX_CACHE_DIR=/opt/bin/onnx_cache` before running an ONNX script. Checked whether `CHECKPOINTS_DIR` itself (used for LoRA/QLoRA adapter saves too) needed the same override — no: adapter-only saves are a few tens of MB, never part of the ~20GB pressure, so only the ONNX path needed it.

Full root-cause writeups for both failures: `EXPERIMENT_MATRIX.md` Recovery Procedures ("ONNX Export OOM" and "ONNX Export Disk OOM"). **Both fixes verified offline only** (`py_compile`, full module import with real deps, `inspect.signature` confirms `main_export`'s params, manual trace of the `ONNX_CACHE_DIR` fallback with/without the env var set) — **neither has been confirmed end-to-end on a GPU yet**. Do not treat this as closed until a real Kaggle run gets past both the export step and the benchmarked inference loop cleanly.

### Open Risk: Llama-2-13B ONNX export (`experiments/llama/05_onnx.py`) — not scheduled, needs a decision before it is
Llama-2-13B's fp16 weights alone are ~26GB — already over Kaggle's 16GB VRAM ceiling before any export overhead, so the CPU-export fix above is not a trivial port for Llama: CPU export may be the *only* viable option here (not an optimization), and even then it's unconfirmed whether Kaggle's CPU RAM can hold a model this size for tracing. This has not been load-tested and should not be assumed to work. Do not schedule a Kaggle session for `experiments/llama/05_onnx.py` until this is explicitly checked (CPU RAM headroom on the Kaggle instance, and whether the resulting ONNX graph can even be loaded back for GPU inference at 16GB). Tracked in `EXPERIMENT_MATRIX.md` Recovery Procedures as well.

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

## Week 3 (in progress, Llama deferred): QLoRA fine-tuning, both models, both datasets (~9 GPU-hrs planned, 9.86h already used on Mistral alone)
- [x] Verify config matches `EXPERIMENT_MATRIX.md` technique #3
- [x] Run `EXP-MIS-QLORA-CNN`/`EXP-MIS-QLORA-SQUAD` on Kaggle — complete, `CONFIRMED FINAL`
- [ ] Run `EXP-LLAMA-QLORA-CNN`/`EXP-LLAMA-QLORA-SQUAD` on Kaggle — **deferred by choice** (not GPU-budget-blocked, quota is fresh — see GPU Budget Tracking); Mistral's 3.3x estimate miss means Llama's ~6h budget should be re-checked before scheduling, and Mistral 8-bit/4-bit is a faster, lower-risk use of the quota in the meantime
- [x] Log real GPU-hours, peak VRAM, training time, quality vs. baseline for Mistral
- [ ] Log same for Llama — pending the deferred run above
- [ ] Quality gate check per `EXPERIMENT_MATRIX.md` ("peak VRAM stays under 14GB, training time visibly less than LoRA's") — cannot be evaluated as a whole-phase gate until Llama's rows exist; Mistral's peak VRAM (1.84-1.88GB) is well under 14GB but training time was *not* visibly less than LoRA's (it was ~14x slower per step) — worth deciding explicitly whether this gate's wording needs revisiting before Week 3 is called complete, not silently waved through

## Week 4 (in progress on Mistral): 8-bit + 4-bit quantized inference, ONNX export, both models (6.4 GPU-hrs planned)
- [x] Run `EXP-MIS-8BIT-CNN`/`EXP-MIS-8BIT-SQUAD` on Kaggle — complete, `CONFIRMED FINAL`. Peak VRAM 3.25GB (-53% vs baseline) but inference 2.6-3x *slower* than baseline on both datasets — third consecutive quantization technique confirming the T4/Turing slowdown pattern (see EXPERIMENT_MATRIX.md Qualitative Notes)
- [x] Run `EXP-MIS-4BIT-CNN`/`EXP-MIS-4BIT-SQUAD` on Kaggle — complete, `CONFIRMED FINAL`. Peak VRAM 1.81GB (-74% vs baseline, largest saving of any technique) but inference 1.5-1.7x *slower* than baseline on both datasets — fourth and final data point, T4/Turing quantization-slowdown finding now marked CONFIRMED in EXPERIMENT_MATRIX.md. `utils/validation.py`'s `check_quant_vram_relationship(1.81, 3.25)` confirms 4-bit VRAM < 8-bit VRAM as expected, no issues
- [x] Log real GPU-hours, peak VRAM, latency, quality vs. baseline for 8-bit
- [x] Log same for 4-bit
- [ ] Run Mistral-7B ONNX export/inference (`EXP-MIS-ONNX-CNN`/`EXP-MIS-ONNX-SQUAD`) — last Mistral-7B technique in the matrix. Two consecutive OOMs on Kaggle so far (2026-08-15): first a VRAM OOM mid-export (fixed: CPU export, fp16), then a disk-quota OOM once export actually started running (fixed: `ONNX_CACHE_DIR`/`HF_HOME` redirected off Kaggle's ~20GB tracked-output path). Both fixes are code-only, **not yet reverified end-to-end on real hardware** — see Blockers & Risks — rerun before marking this done
- [ ] Llama-2-13B's 8-bit/4-bit/ONNX and QLoRA remain pending — not scheduled this session. Llama's ONNX export additionally carries an unresolved open risk beyond Mistral's fix (13B fp16 weights alone exceed 16GB VRAM even before export overhead) — see Blockers & Risks "Open Risk: Llama-2-13B ONNX export" before scheduling

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
