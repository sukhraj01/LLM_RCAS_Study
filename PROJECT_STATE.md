# Project State: LLM Optimization for Resource-Constrained Systems

> Updated at the start of every work session (Kaggle or local). This is the single file any new Claude Code session should read FIRST. It reflects reality, not the plan — if something hasn't run yet, it says so.

**Last Updated:** 2026-08-14

**Current Phase:** WEEK 2 — Week 1 (env, data pipeline, baselines) complete; starting LoRA fine-tuning on Mistral-7B

**Project Status:** 🟢 On track feature-wise (all 4 Week 1 baselines complete and clean) but 🔴 over the Setup+baseline GPU-hour budget — see GPU Budget Tracking below.

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
| **Experiment tracking** | ✅ In progress | 4/22 rows | `logs/experiment_tracking.csv` populated for all 4 Week 1 baselines; 18 rows still `pending` for Weeks 2-4 |
| **Week 2-4 experiments** | ⬜ Not started | 0% | 22 experiments per `EXPERIMENT_MATRIX.md` |
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
| LoRA (Mistral only) | 2 | 4 | 0 | 4 | ⬜ Not started |
| QLoRA (both models) | 3 | 9 | 0 | 9 | ⬜ Not started |
| Quantization + ONNX | 4 | 6.4 | 0 | 6.4 | ⬜ Not started |
| Buffer / reruns | 5 | 8 | 0 | 8 | ⬜ Reserve |
| **TOTAL** | 1-5 | ~29.4 | ≥14.5 | ≤14.9 | 🟡 Setup+baseline alone used ≥49% of the entire 5-phase 29.4h *estimate* (planned share was ~7%) — a real miss on the phase-level plan, worth remembering when estimating Weeks 2-4. **Not a cap risk**, though: the 30h figure is a per-week quota that refreshes every Sunday, not a shared multi-phase pool (confirmed by the engineer) — Setup+baseline's ≥14.5h all landed inside a single week and is comfortably under that week's 30h, and each of Weeks 2-4 gets its own fresh 30h regardless of what earlier weeks used. |

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
⚠️ GPU-hours for today's Kaggle rerun session not yet recorded — see GPU Budget Tracking above. Otherwise none: Week 1 is complete and clean, ready to start Week 2 (LoRA).

⚠️ **Root cause identified and fixed (2026-08-14); existing Mistral LoRA results should be considered unreliable pending rerun.** `EXP-MIS-LORA-SQUAD` completed with EM 0.0/200 (down from baseline EM 8.0). Code review confirmed the mechanism: `load_model_and_tokenizer()` sets `tokenizer.pad_token = tokenizer.eos_token` (Mistral's tokenizer ships no distinct pad token), and `run_training_experiment()` previously used `transformers.DataCollatorForLanguageModeling(mlm=False)`, whose default behavior masks any label token equal to `pad_token_id`'s *value* — not by actual padding position — to `-100`. Since `pad_token_id == eos_token_id`, this silently masked every genuine end-of-sequence token in every training target, for every technique/dataset that trains, not SQuAD alone — the model never got gradient signal for when to stop generating. Fixed: `experiments/common.py` now uses a custom `_CausalLMCollator` that masks by `attention_mask == 0` (real padding) instead. Verified with an offline fake-tensor smoke test (no GPU): a token whose value matches `pad_token_id` but whose `attention_mask` is `1` is correctly preserved in labels. `save_debug_predictions()` now also dumps the first 5 predictions + references per experiment to `logs/debug_predictions/` for future spot-checks.

**This is likely THE explanation, not just a hypothesis, but is not yet proven** — no rerun or raw predictions have confirmed it against real hardware yet. **Both `EXP-MIS-LORA-CNN` and `EXP-MIS-LORA-SQUAD` were trained under the buggy collator** (it's the one code path both used), so both should be treated as unreliable, not just the SQuAD row — CNN's ROUGE improvement doesn't clear it, since ROUGE (partial n-gram overlap) is much more tolerant of a model that over-generates past the correct content than SQuAD's strict exact-match is, so the same underlying defect could easily be present but not visible in CNN's headline number. **Recommend rerunning both `EXP-MIS-LORA-CNN` and `EXP-MIS-LORA-SQUAD`** with the fixed collator before treating either as a final result. Full reasoning in `logs/experiment_tracking.csv` and `EXPERIMENT_MATRIX.md` Recovery Procedures. **Do not treat the existing numbers as accepted/final until rerun.**

### Identified Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|-----------|
| Kaggle session timeout (12h cap) mid-experiment | Medium | Lost progress | Checkpoint every N steps, resume capability required from day 1 |
| Weekly quota (30h) exhausted mid-week | Low, given ~20-25h total planned | Delay | Buffer week reserved (Week 5) |
| LoRA fp16 training OOMs on Mistral-7B despite estimate | Medium | Experiment blocked | Fallback: gradient checkpointing already planned; further fallback is treating it as QLoRA-only for Mistral too, documented as ADR if triggered |
| Kaggle/local state drift (forget to sync) | Medium | Confusion, lost work | `KAGGLE_SYNC.md` protocol — follow it every session |

---

# What's Next

## Week 1 — COMPLETE
- [x] Create repo structure (see `ARCHITECTURE.md`)
- [x] Set up Kaggle notebook + local venv, pin dependency versions in `requirements.txt`
- [x] Download CNN/DailyMail + SQuAD samples
- [x] Run 4 real baseline inference experiments (2 models × 2 datasets), log real numbers — done across two Kaggle sessions (see `logs/daily_standup.md` 2026-08-13/2026-08-14: first attempt hung on EXP-LLAMA-BASE-SQUAD, root-caused, fixed, rerun succeeded)
- [x] Initialize `logs/experiment_tracking.csv`
- [x] First entry in `knowledge/ai-usage-log/`

## Week 2 (starting now): LoRA fine-tuning, Mistral-7B, both datasets (~4 GPU-hrs planned)
- [ ] Verify config matches `EXPERIMENT_MATRIX.md` technique #2 (r=8, alpha=16, target_modules q_proj/v_proj, batch=1/grad_accum=8, lr=2e-4, 2 epochs)
- [ ] Confirm `experiments/mistral/02_lora.py` (now using `run_training_multi_dataset()`) picks up the clean baseline via `require_baseline_metrics("MIS", ...)`
- [ ] Run `EXP-MIS-LORA-CNN` and `EXP-MIS-LORA-SQUAD` on Kaggle
- [ ] Log real GPU-hours, peak VRAM, training time, quality vs. Mistral baseline
- [ ] Quality gate check per `EXPERIMENT_MATRIX.md` ("training time roughly 40-60% of full FT cost, no OOM") before moving to Week 3

## Weeks 3-4 (experiments)
- Week 3: QLoRA fine-tuning, both models, both datasets (9 GPU-hrs)
- Week 4: 8-bit + 4-bit quantized inference, ONNX export, both models (6.4 GPU-hrs)

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
