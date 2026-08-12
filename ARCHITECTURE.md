# System Architecture: LLM Optimization for Resource-Constrained Systems

**Last Updated:** 2026-08-12
**Status:** Reset — right-sized for Kaggle free-tier hardware (16GB VRAM, 30 GPU-hrs/week). See `PROJECT_STATE.md` "Revision Note" for why.

---

# System Overview

A reproducible benchmarking suite comparing 6 LLM optimization techniques across 2 models and 2 datasets, producing evidence-based recommendations for deployment under resource constraints.

**Core Principle:** Systematic comparison of techniques, not exploration. Every experiment is specified in advance in `EXPERIMENT_MATRIX.md`.

**Scope:**
- 22 total experiments
- Mistral-7B: full technique set (baseline, LoRA, QLoRA, 8-bit, 4-bit, ONNX) — 12 experiments
- Llama-2-13B: reduced technique set (baseline, QLoRA, 8-bit, 4-bit, ONNX — fp16 LoRA skipped, doesn't fit 16GB) — 10 experiments
- 2 datasets (CNN/DailyMail — summarization, SQuAD — QA)
- ~21-25 GPU hours of actual experiment time, ~40 with buffer

**Outputs:**
- Master benchmark CSV (22 rows)
- FastAPI inference server
- Streamlit dashboard
- 10-12 page technical report with decision framework

---

# Design Goals

This architecture is intentionally optimized for:

1. **Reproducibility** — Same code + same config = same results, always
2. **Comparability** — Fair, apples-to-apples comparison across techniques, always against the same baseline
3. **Traceability** — Every result traced to a specific experiment spec in `EXPERIMENT_MATRIX.md`
4. **Fit to hardware** — Every technique is checked against actual VRAM math before being scheduled, not assumed
5. **Modularity** — Each technique can be tested independently
6. **Defensibility** — Every deviation from "textbook" scope (e.g. dropping full FT, dropping fp16 LoRA on the 13B model) is documented as an ADR with the reasoning, not silently done

It is intentionally **not** optimized for:
- Production deployment (this is benchmarking, not serving)
- Massive scale (single 16GB GPU, small datasets)
- Full-parameter fine-tuning of 7B+ models (physically doesn't fit the hardware — see VRAM math in `EXPERIMENT_MATRIX.md`)

---

# Architectural Principles

1. **Check VRAM math before scheduling a technique.** Don't assume something fits — compute bytes/param and compare against 16GB before it goes in the matrix.
2. **Plan before execute** — Understand the experiment before running it.
3. **Track everything** — Every experiment is logged, every result is saved.
4. **Validate results** — Do the numbers make sense? Is quality degradation acceptable?
5. **Document deviations** — If something changes from the plan, record why (ADR).
6. **Gate transitions** — No phase progresses without explicit quality gate approval.
7. **Recover systematically** — Proceduralized recovery for OOM, NaN, timeouts.
8. **Benchmark rigorously** — Every technique compared against the same baseline.
9. **Preserve reproducibility** — Fixed seeds, pinned versions, exact hyperparameters (see `EXPERIMENT_MATRIX.md`).
10. **Keep context small** — This file, `CLAUDE.md`, and `PROJECT_STATE.md` stay short and current. Detailed experiment specs live only in `EXPERIMENT_MATRIX.md`. Nothing should require loading the full archived guide into a session.

---

# System Data Flow

```
┌────────────────────────────────────────────────────────────────┐
│              Experiment Specification (EXPERIMENT_MATRIX.md)    │
│         22 experiments × hyperparams × datasets × models        │
└────────────────┬─────────────────────────────────────────────────┘
                 │
                 ▼
        ┌────────────────────┐
        │  Load Base Model   │
        │  + Tokenizer       │
        └────────┬───────────┘
                 │
        ┌────────▼──────────────────────────────────────┐
        │      Apply Optimization Technique              │
        │  ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
        │  │  LoRA    │ │ QLoRA    │ │ Quant / ONNX  │  │
        │  └──────────┘ └──────────┘ └───────────────┘  │
        └────────┬──────────────────────────────────────┘
                 │
        ┌────────▼───────────┐
        │  Load Dataset      │
        │  (train/val/test)  │
        └────────┬───────────┘
                 │
        ┌────────▼────────────────────────┐
        │  Train / Fine-tune              │
        │  (skip if inference-only tech.) │
        └────────┬─────────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │  Run Inference on Test Set    │
        │  Measure: latency, VRAM, etc. │
        └────────┬──────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │  Compute Quality Metrics      │
        │  (ROUGE / F1+EM)              │
        └────────┬──────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │  Save Results to CSV          │
        └────────┬──────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │  Update Tracking Log          │
        │  (experiment_tracking.csv)    │
        └────────┬──────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │  Validate Results             │
        │  (sanity checks, ranges)      │
        └────────┬──────────────────────┘
                 │
   ╭─────────────▼────────────────────╮
   │ Next Experiment or Phase Gate?   │
   ├──────────────────────────────────┤
   │  • Continue if gate passed       │
   │  • Escalate if validation failed │
   │  • Recovery if experiment failed │
   ╰──────────────────────────────────╯
                 │
        ┌────────▼──────────────────────┐
        │  Compile to Master CSV        │
        │  (Week 5)                     │
        └────────┬──────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │  Analysis & Dashboard         │
        │  (Weeks 6-7)                  │
        └────────┬──────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │  Technical Report             │
        │  (Weeks 8-9)                  │
        └──────────────────────────────┘
```

---

# Components

## 1. Experiment Specification
**Source of Truth:** `EXPERIMENT_MATRIX.md` — exact hyperparameters, datasets, expected outputs, GPU-hour estimates for all 22 experiments.
**Implementation:** `experiments/mistral/*.py`, `experiments/llama/*.py`, one script per technique.

## 2. Data Pipeline
**Purpose:** Download CNN/DailyMail and SQuAD samples, split, validate.
**Implementation:** `utils/data_loader.py`
```python
loader = DataLoader()
train_df = loader.load_dataset("cnn_dailymail", split="train", sample_size=1000)
```
**Guarantees:** Deterministic splits, no leakage (test never seen during training), consistent schema across datasets.

## 3. Experiment Executor
**Implementation:**
- `experiments/mistral/01_baseline.py`, `02_lora.py`, `03_qlora.py`, `04_quant_8bit.py`, `05_quant_4bit.py`, `06_onnx.py`
- `experiments/llama/01_baseline.py`, `02_qlora.py`, `03_quant_8bit.py`, `04_quant_4bit.py`, `05_onnx.py` (no `lora.py` — doesn't fit, see `EXPERIMENT_MATRIX.md`)

**Error Handling:** CUDA OOM → reduce batch → grad accumulation → shrink `max_seq_length` → skip + document. NaN loss → halve LR → retry → skip + document. Kaggle session timeout → resume from checkpoint.

## 4. Metrics Computation
**Implementation:** `utils/metrics.py`
- Summarization (CNN): ROUGE-1/2/L
- QA (SQuAD): Exact Match + F1
- Derived: quality degradation % vs baseline, speedup factor vs baseline, VRAM reduction % vs baseline

## 5. Experiment Tracking
**Implementation:**
- `logs/experiment_tracking.csv` — one row per experiment (22 rows when complete): Exp ID, Model, Technique, Dataset, Status, Start/End time, GPU hrs, Peak VRAM, Quality score, Errors
- `logs/daily_standup.md` — session-by-session log
- `logs/phase_summary.md` — weekly summary + gate check

## 6. Quality Validation
**Implementation:** `utils/validation.py` — sanity checks (latency > 0, quality in valid range), consistency checks (QLoRA should train faster than LoRA did on Mistral), relationship checks (4-bit VRAM < 8-bit VRAM).

## 7. Master Benchmark Compilation
**Implementation:** `scripts/compile_results.py` — merges all per-technique CSVs into `results/master_benchmark_results.csv` (22 rows), runs at end of Week 4/start of Week 5 buffer.

## 8. Analysis & Visualization
**Implementation:** `scripts/analyze_results.py`, `dashboard/dashboard.py` (Streamlit), `TECHNICAL_REPORT.md`.

## 9. API & Deployment
**Implementation:** `api/inference_server.py` (FastAPI: `POST /infer`, `GET /models`, `GET /model/info/{key}`, `GET /benchmark`), `Dockerfile`, `docker-compose.yml`.

---

# Design Decisions (ADRs)

## ADR-001: Drop full-FP32 fine-tuning
**Decision:** No full-parameter fine-tuning of any model.
**Why:** FP32 full FT needs ~16 bytes/param with Adam. 7B ≈ 112GB, 13B ≈ 208GB. Kaggle GPUs have 16GB. Not attemptable. The assignment scope (LoRA/QLoRA/PEFT/quantization comparison) doesn't require it either — it was scope added by an earlier planning pass, not a requirement.

## ADR-002: Llama-2-13B skips fp16 LoRA
**Decision:** Llama-2-13B runs baseline, QLoRA, 8-bit, 4-bit, ONNX — not fp16 LoRA.
**Why:** fp16 LoRA needs the full base model resident at ~2 bytes/param (~26GB for 13B) plus activations. Doesn't fit 16GB. QLoRA (4-bit base, ~6.5GB) does fit and still answers "does fine-tuning scale to a bigger model."

## ADR-003: 2 datasets, not 3
**Decision:** CNN/DailyMail + SQuAD only for the core 22. Alpaca/instruction-following dropped.
**Why:** Controls total GPU hours and experiment count while still covering two distinct task types (generation, extraction). Can be added in the Week 5 buffer if time/budget allow — see `PROJECT_STATE.md`.

## ADR-004: ONNX Runtime is the required inference-optimization technique; TensorRT is stretch
**Decision:** All 22 experiments compare against ONNX Runtime for inference speed. TensorRT conversion is attempted only in Week 5 buffer if time allows.
**Why:** PyTorch → ONNX → TensorRT conversion for LLMs has real operator-support friction; treating it as mandatory across the whole matrix risked blowing the timeline on tooling instead of results.

---

# Known Limitations & Future Work

### Limitations
- 2 models only, both open-weights — no proprietary/API-based comparison
- 2 datasets — covers generation + extraction, not instruction-following (unless added back in buffer)
- Single GPU, free-tier hardware — findings are specific to this VRAM class, stated explicitly in the report
- No full-parameter fine-tuning baseline (see ADR-001) — LoRA serves as the "trainable" upper bound for comparison instead

### Future Extensions (documented for completeness, not this project)
- Add a 3rd/4th model for generalization validation once core 22 are done and budget allows
- Distillation, pruning, speculative decoding
- Multi-GPU / larger VRAM class comparison

---

# Evolution Notes

### 2026-08-12: Reset
- Original architecture assumed 4 models, 72 experiments, 133.5 GPU-hrs, full FP32 fine-tuning of 7B/13B on 24GB — none of which fit the actual hardware (Kaggle, 16GB, 30h/week)
- Right-sized to 2 models, 22 experiments, ~21-25 GPU-hrs core work
- This document, `PROJECT_STATE.md`, `EXPERIMENT_MATRIX.md`, `KAGGLE_SYNC.md` rewritten/created; `CLAUDE.md` updated; original guide archived

### Week 1-4 (Experiments)
- *(To be updated as phases complete)*

### Weeks 5-10 (Deliverables)
- *(To be updated as report/API/dashboard built)*

---

# Appendix: File Structure

```
llm_optimization/
│
├── CLAUDE.md                    [Collaboration contract — stable]
├── PROJECT_STATE.md             [Live progress — read first, every session]
├── ARCHITECTURE.md              [This file — system design]
├── EXPERIMENT_MATRIX.md         [Source of truth for specs/hyperparameters]
├── KAGGLE_SYNC.md                [Protocol for Kaggle <-> local handoff]
│
├── archive/
│   └── HYBRID_APPROACH_DETAILED_IMPLEMENTATION_GUIDE.md  [Deprecated numbers; rationale sections still useful]
│
├── data/
│   ├── cnn_dailymail/
│   └── squad/
│
├── experiments/
│   ├── mistral/
│   │   ├── 01_baseline.py
│   │   ├── 02_lora.py
│   │   ├── 03_qlora.py
│   │   ├── 04_quant_8bit.py
│   │   ├── 05_quant_4bit.py
│   │   └── 06_onnx.py
│   └── llama/
│       ├── 01_baseline.py
│       ├── 02_qlora.py
│       ├── 03_quant_8bit.py
│       ├── 04_quant_4bit.py
│       └── 05_onnx.py
│
├── utils/
│   ├── data_loader.py
│   ├── metrics.py
│   ├── config.py                [Hyperparameters, imported not retyped — matches EXPERIMENT_MATRIX.md]
│   └── validation.py
│
├── results/
│   ├── mistral_results.csv
│   ├── llama_results.csv
│   └── master_benchmark_results.csv   [22 rows, compiled Week 5]
│
├── checkpoints/
├── logs/
│   ├── experiment_tracking.csv
│   ├── daily_standup.md
│   └── phase_summary.md
│
├── api/
│   └── inference_server.py
├── dashboard/
│   └── dashboard.py
├── scripts/
│   ├── compile_results.py
│   └── analyze_results.py
│
├── knowledge/
│   └── ai-usage-log/             [Verbatim session logs, per CLAUDE.md]
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

**This architecture document is the foundation for reproducible, well-tracked experimentation. Keep it and `PROJECT_STATE.md` short — details belong in `EXPERIMENT_MATRIX.md`.**
