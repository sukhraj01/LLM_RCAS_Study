# Project State: LLM Optimization for Resource-Constrained Systems

> Updated at the start of every work session (Kaggle or local). This is the single file any new Claude Code session should read FIRST. It reflects reality, not the plan — if something hasn't run yet, it says so.

**Last Updated:** 2026-08-12 (project reset — see "Revision Note" below)

**Current Phase:** PRE-WEEK 1 — repo/env not yet set up, nothing has executed

**Project Status:** 🔴 Not started. Planning corrected; ready to begin real Week 1.

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
| **Repo structure** | ⬜ Not started | 0% | Needs creating |
| **Environment setup** | ⬜ Not started | 0% | Kaggle notebook + local venv, pinned versions |
| **Data pipeline** | ⬜ Not started | 0% | CNN/DailyMail, SQuAD — smaller samples than original plan (see below) |
| **Baseline inference** | ⬜ Not started | 0% | 4 runs (2 models × 2 datasets) |
| **Experiment tracking** | ⬜ Not started | 0% | `logs/experiment_tracking.csv` template needed |
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
| Setup + baseline | 1 | 2 | 0 | 2 | ⬜ Not started |
| LoRA (Mistral only) | 2 | 4 | 0 | 4 | ⬜ Not started |
| QLoRA (both models) | 3 | 9 | 0 | 9 | ⬜ Not started |
| Quantization + ONNX | 4 | 6.4 | 0 | 6.4 | ⬜ Not started |
| Buffer / reruns | 5 | 8 | 0 | 8 | ⬜ Reserve |
| **TOTAL** | 1-5 | ~29.4 | 0 | 29.4 | ⬜ Not started |

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
🔴 Repo not yet created. Nothing has executed. This is expected — session 1 starts the real Week 1.

### Identified Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|-----------|
| Kaggle session timeout (12h cap) mid-experiment | Medium | Lost progress | Checkpoint every N steps, resume capability required from day 1 |
| Weekly quota (30h) exhausted mid-week | Low, given ~20-25h total planned | Delay | Buffer week reserved (Week 5) |
| LoRA fp16 training OOMs on Mistral-7B despite estimate | Medium | Experiment blocked | Fallback: gradient checkpointing already planned; further fallback is treating it as QLoRA-only for Mistral too, documented as ADR if triggered |
| Kaggle/local state drift (forget to sync) | Medium | Confusion, lost work | `KAGGLE_SYNC.md` protocol — follow it every session |

---

# What's Next

## Week 1 (starting now)
- [ ] Create repo structure (see `ARCHITECTURE.md`)
- [ ] Set up Kaggle notebook + local venv, pin dependency versions in `requirements.txt`
- [ ] Download CNN/DailyMail + SQuAD samples (start small — 500-1000 rows each — expand later only if GPU budget allows)
- [ ] Run 4 real baseline inference experiments (2 models × 2 datasets), log real numbers
- [ ] Initialize `logs/experiment_tracking.csv`
- [ ] First entry in `knowledge/ai-usage-log/`

## Weeks 2-4 (experiments)
- Week 2: LoRA fine-tuning, Mistral-7B, both datasets (4 GPU-hrs)
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
