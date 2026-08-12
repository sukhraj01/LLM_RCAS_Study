# HYBRID APPROACH - DETAILED IMPLEMENTATION GUIDE
## Complete Blueprint for 2+2 Model Benchmarking with Full Traceability

**Document Version:** 1.0  
**Last Updated:** Today  
**Status:** Ready for Claude Code Migration  

---

# TABLE OF CONTENTS
1. Executive Overview
2. Model Selection & Justification (Deep Dive)
3. Complete Experiment Matrix (72 Experiments)
4. Detailed Phase Breakdown (Week by Week)
5. Data Flow Architecture
6. Quality Gates & Validation Procedures
7. Tracking & Logging Schema
8. Failure Recovery Procedures
9. Phase Transition Checklist
10. Integration with Claude Code

---

---

# 1. EXECUTIVE OVERVIEW

## What is the Hybrid Approach?

**NOT:** 4 models, all tests run on all = 108 experiments (too much)  
**NOT:** 2 models only = 54 experiments (might get "too narrow" feedback)  
**IS:** 2 models FULL benchmark + 2 models VALIDATION = 72 smart experiments

## Model Tiers

### TIER 1: Full Benchmarking (Comprehensive)
- **Model 1A:** `mistralai/Mistral-7B-v0.1`
  - Why: Industry standard for efficiency
  - 7.2B parameters, best latency
  - Most deployed in production
  
- **Model 1B:** `meta-llama/Llama-2-13b`
  - Why: Production baseline for capability
  - 13B parameters, strong quality
  - Covers mid-scale deployments

**Commitment:** Run ALL 9 techniques × ALL 3 datasets = 27 experiments per model = 54 total

**Duration:** Weeks 2-6 (5 weeks)

**Output:** master_benchmark_results.csv (54 rows, all metrics)

---

### TIER 2: Validation (Generalization Check)
- **Model 2A:** `mosaicml/mpt-7b`
  - Why: Different architecture (MQA instead of multi-head attention)
  - Validates our findings generalize beyond Mistral/Llama
  - Open-weights alternative (like Mistral)
  
- **Model 2B:** `codellama/CodeLlama-7b-hf`
  - Why: Specialized variant (code-tuned)
  - Tests if optimization techniques work on domain-specific models
  - Different training procedure than base models

**Commitment:** Run SELECT 6 techniques × ALL 3 datasets = 18 experiments total

**Duration:** Week 7 (1 week)

**Output:** validation_benchmark_results.csv (18 rows, selected metrics)

---

## Why This Approach Wins

| Aspect | Justification | Professor Impact |
|--------|---------------|-----------------|
| **Depth** | 54 full experiments on 2 primary models | ✅ "Very thorough benchmarking" |
| **Breadth** | 18 validation experiments on 2 additional models | ✅ "Confirms generalizability" |
| **Rigor** | Separate "full" vs "validation" shows methodology | ✅ "Proper experimental design" |
| **Timing** | Same 8-10 week timeline (validation is quick) | ✅ "Realistic execution" |
| **Coverage** | "Comprehensive with validation" framing | ✅ "Can't ask for more models" |

---

---

# 2. MODEL SELECTION & JUSTIFICATION (DEEP DIVE)

## Tier 1 Model Selection Rationale

### Model 1A: Mistral-7B-v0.1

**Technical Specifications:**
- Parameters: 7.2B
- Context Length: 8K
- Architecture: GQA (Grouped Query Attention)
- Training Data: 32B tokens
- License: Apache 2.0
- Performance: MMLU 60.97, HELLASWAG 81.3

**Why This Model?**

1. **Efficiency Focus** (Critical for resource-constrained systems)
   - Smallest in our benchmark
   - Fastest inference (lowest latency baseline)
   - Lowest VRAM requirements
   - Best for edge deployment
   - Tests if optimization techniques matter for small models

2. **Production Prevalence**
   - Most widely deployed 7B model in 2024
   - Used by: Together.ai, Replicate, Modal Labs
   - Represents "typical" resource-constrained scenario
   - If optimization works here, it works in production

3. **Architecture Innovation** (Different from Llama)
   - Uses GQA (Grouped Query Attention) vs Llama's MHA
   - Tests optimization robustness to architectural differences
   - Validates findings transfer across architectures

4. **Open Access**
   - Weights freely available on HuggingFace
   - No gating, no signup, can download instantly
   - Reproducible by anyone (professor can verify)

**Experiment Count on Model 1A:**
- 9 optimization techniques × 3 datasets = 27 experiments
- ~30-40 hours GPU time (distributed over 3 weeks)

---

### Model 1B: Llama-2-13B

**Technical Specifications:**
- Parameters: 13B
- Context Length: 4K
- Architecture: Standard Transformer with MHA
- Training Data: 2T tokens
- License: Community (with commercial restrictions noted)
- Performance: MMLU 63.95, HELLASWAG 84.9

**Why This Model?**

1. **Mid-Scale Production Baseline**
   - Second most common 13B model in 2024
   - Represents "capability tier" vs efficiency tier
   - Shows how optimization scales with model size
   - 1.8x larger than Mistral (tests if techniques scale)

2. **Quality-Efficiency Balance**
   - Better performance than 7B models (MMLU +3 points)
   - Still deployable on 24GB GPUs (cloud standard)
   - Real tradeoff: larger = higher quality, more resources
   - Shows optimization value increases with model size

3. **Architecture Differences**
   - Standard MHA (vs Mistral's GQA)
   - Different from specialized models
   - Validates findings aren't model-specific

4. **Reproducibility**
   - Extensively tested in literature
   - Baseline for many optimization papers
   - Professor can cross-check results against known benchmarks

**Experiment Count on Model 1B:**
- 9 optimization techniques × 3 datasets = 27 experiments
- ~45-60 hours GPU time (larger model = slower training)

---

## Tier 2 Model Selection Rationale

### Model 2A: MPT-7B

**Technical Specifications:**
- Parameters: 7.2B
- Context Length: 8K (extended to 84K possible)
- Architecture: Standard Transformer, ALiBi positional encodings
- Training Data: 1T tokens
- License: CC-BY-SA-3.0
- Performance: MMLU 58.91, HELLASWAG 79.8

**Why For Validation?**

1. **Different Architecture Family**
   - ALiBi instead of RoPE positional encodings
   - MQA (similar to Mistral but slightly different)
   - Tests if optimization findings generalize across architectures
   - If LoRA works on Mistral, should work on MPT

2. **Alternative to Mistral**
   - Similar parameter count (7B)
   - But built by different org (MosaicML)
   - Different training data, different tuning
   - If results similar, finding is robust

3. **Open Access & Commercial Friendly**
   - Fully open weights
   - Can be used commercially without restrictions
   - Validates findings on commercial-friendly models

4. **Validation, Not Deep Dive**
   - Won't run manual evaluation (time-saving)
   - Won't run all techniques (testing key ones only)
   - Goal: Confirm speedups/quality numbers transfer
   - If results differ significantly, that's also valuable info

**Experiment Count on Model 2A (Validation Subset):**
- 6 key techniques × 3 datasets = 18 experiments
- ~15-20 hours GPU time (quick runs, no manual eval)
- Techniques: Baseline, LoRA, QLoRA, 8-bit, 4-bit, ONNX

---

### Model 2B: CodeLlama-7B

**Technical Specifications:**
- Parameters: 7.2B
- Context Length: 16K (specialized for code)
- Architecture: Llama-2 derived
- Training Data: 500B+ code tokens
- License: Same as Llama-2
- Performance: HumanEval 57%, MBPP 62% (code-specific)

**Why For Validation?**

1. **Specialized Use Case**
   - Code-tuned variant (not general instruction model)
   - Tests if optimization techniques work on specialized models
   - Real-world scenario: many deployments are domain-specific
   - If optimization works on code model, likely works elsewhere

2. **Different Training Procedure**
   - Pre-trained on general data
   - Fine-tuned on code data
   - Different loss landscape than base models
   - Tests robustness of optimization techniques

3. **Deriving from Llama-2**
   - Similar to Llama-2-13B architecture
   - But different training data
   - Good validation for Llama findings
   - If LoRA works on Llama-2, should work on CodeLlama

4. **Practical Importance**
   - Code generation increasingly important use case
   - Shows optimization isn't just for chat/instruction
   - Demonstrates generalization to other domains

**Experiment Count on Model 2B (Validation Subset):**
- 6 key techniques × 3 datasets = 18 experiments (same as MPT)
- ~12-15 hours GPU time
- Techniques: Baseline, LoRA, QLoRA, 8-bit, 4-bit, ONNX

---

## Why 2+2 is Better Than Other Options

### Option: 2 Models Only (Mistral + Llama)
```
54 experiments total
Professor concern: "You only tested 2 models, what if it doesn't generalize?"
You're forced to defend single-model findings
Risk: Medium
```

### Option: 4 Models Full (Mistral + Llama + MPT + CodeLlama)
```
108 experiments total
Timeline: 11-12 weeks (risky, might not finish)
If something breaks: You're behind schedule
If something runs out of VRAM: Many failures
Risk: High
Quality: Lower (might rush results)
```

### Option: Hybrid 2+2 (CHOSEN)
```
72 experiments total
- 54 FULL experiments on 2 models = Deep, careful analysis
- 18 QUICK experiments on 2 models = Validates generalization

Professor reads: "Comprehensive benchmarking on primary models
with generalization validation on secondary models"

Result: ✅ Can't complain about model coverage
         ✅ Shows proper methodology
         ✅ Realistic timeline
         ✅ Best of both worlds
```

---

---

# 3. COMPLETE EXPERIMENT MATRIX (72 EXPERIMENTS)

## Experiment Naming Convention

**Format:** `EXP-[TIER]-[WEEK]-[MODEL_SHORT]-[TECHNIQUE]-[DATASET]`

**Example:** `EXP-T1-W3-MIS-LORA-CNN` = Tier 1, Week 3, Mistral, LoRA, CNN/DailyMail

---

## TIER 1: Full Benchmarking (54 Experiments)

### Weeks 2-6: Comprehensive Testing

#### WEEK 2: Fine-Tuning Baseline Setup

| Exp ID | Model | Technique | Dataset | Status | GPU Hours | Expected Output |
|--------|-------|-----------|---------|--------|-----------|-----------------|
| EXP-T1-W2-MIS-FULL-CNN | Mistral-7B | Full Fine-Tune | CNN/DailyMail | Planning | 8 | checkpoint, ROUGE scores |
| EXP-T1-W2-MIS-FULL-SQUAD | Mistral-7B | Full Fine-Tune | SQuAD | Planning | 8 | checkpoint, F1 scores |
| EXP-T1-W2-MIS-FULL-ALP | Mistral-7B | Full Fine-Tune | Alpaca | Planning | 6 | checkpoint, manual eval |
| EXP-T1-W2-LLAMA-FULL-CNN | Llama-2-13B | Full Fine-Tune | CNN/DailyMail | Planning | 12 | checkpoint, ROUGE scores |
| EXP-T1-W2-LLAMA-FULL-SQUAD | Llama-2-13B | Full Fine-Tune | SQuAD | Planning | 12 | checkpoint, F1 scores |
| EXP-T1-W2-LLAMA-FULL-ALP | Llama-2-13B | Full Fine-Tune | Alpaca | Planning | 10 | checkpoint, manual eval |

**Week 2 Subtotal:** 56 GPU hours (baseline for comparing against)

---

#### WEEK 3: LoRA Fine-Tuning

| Exp ID | Model | Technique | Dataset | Status | GPU Hours | Expected Output |
|--------|-------|-----------|---------|--------|-----------|-----------------|
| EXP-T1-W3-MIS-LORA-CNN | Mistral-7B | LoRA | CNN/DailyMail | Planning | 4 | adapter_model, ROUGE |
| EXP-T1-W3-MIS-LORA-SQUAD | Mistral-7B | LoRA | SQuAD | Planning | 4 | adapter_model, F1 |
| EXP-T1-W3-MIS-LORA-ALP | Mistral-7B | LoRA | Alpaca | Planning | 3 | adapter_model, manual eval |
| EXP-T1-W3-LLAMA-LORA-CNN | Llama-2-13B | LoRA | CNN/DailyMail | Planning | 6 | adapter_model, ROUGE |
| EXP-T1-W3-LLAMA-LORA-SQUAD | Llama-2-13B | LoRA | SQuAD | Planning | 6 | adapter_model, F1 |
| EXP-T1-W3-LLAMA-LORA-ALP | Llama-2-13B | LoRA | Alpaca | Planning | 5 | adapter_model, manual eval |

**Week 3 Subtotal:** 28 GPU hours (should be ~50% of full fine-tune)

**Quality Gate:** LoRA training time should be ~50-60% of full fine-tune  
**Success Criteria:** All 6 experiments complete with adapter size < base model size

---

#### WEEK 4: QLoRA Fine-Tuning

| Exp ID | Model | Technique | Dataset | Status | GPU Hours | Expected Output |
|--------|-------|-----------|---------|--------|-----------|-----------------|
| EXP-T1-W4-MIS-QLORA-CNN | Mistral-7B | QLoRA | CNN/DailyMail | Planning | 2 | adapter_model, ROUGE |
| EXP-T1-W4-MIS-QLORA-SQUAD | Mistral-7B | QLoRA | SQuAD | Planning | 2 | adapter_model, F1 |
| EXP-T1-W4-MIS-QLORA-ALP | Mistral-7B | QLoRA | Alpaca | Planning | 1.5 | adapter_model, manual eval |
| EXP-T1-W4-LLAMA-QLORA-CNN | Llama-2-13B | QLoRA | CNN/DailyMail | Planning | 3 | adapter_model, ROUGE |
| EXP-T1-W4-LLAMA-QLORA-SQUAD | Llama-2-13B | QLoRA | SQuAD | Planning | 3 | adapter_model, F1 |
| EXP-T1-W4-LLAMA-QLORA-ALP | Llama-2-13B | QLoRA | Alpaca | Planning | 2.5 | adapter_model, manual eval |

**Week 4 Subtotal:** 14 GPU hours (should be ~25% of full fine-tune)

**Quality Gate:** QLoRA training time should be 25-30% of full fine-tune  
**Success Criteria:** VRAM usage < 20GB for all experiments; adapter + 4-bit model fits on 24GB GPU

---

#### WEEK 5: Quantization & Inference Optimization

| Exp ID | Model | Technique | Dataset | Status | GPU Hours | Expected Output |
|--------|-------|-----------|---------|--------|-----------|-----------------|
| EXP-T1-W5-MIS-8BIT-CNN | Mistral-7B | 8-bit Quant | CNN/DailyMail | Planning | 0.2 | inference metrics |
| EXP-T1-W5-MIS-8BIT-SQUAD | Mistral-7B | 8-bit Quant | SQuAD | Planning | 0.2 | inference metrics |
| EXP-T1-W5-MIS-8BIT-ALP | Mistral-7B | 8-bit Quant | Alpaca | Planning | 0.2 | inference metrics |
| EXP-T1-W5-MIS-4BIT-CNN | Mistral-7B | 4-bit Quant | CNN/DailyMail | Planning | 0.2 | inference metrics |
| EXP-T1-W5-MIS-4BIT-SQUAD | Mistral-7B | 4-bit Quant | SQuAD | Planning | 0.2 | inference metrics |
| EXP-T1-W5-MIS-4BIT-ALP | Mistral-7B | 4-bit Quant | Alpaca | Planning | 0.2 | inference metrics |
| EXP-T1-W5-MIS-ONNX-CNN | Mistral-7B | ONNX Export | CNN/DailyMail | Planning | 1 | onnx model, metrics |
| EXP-T1-W5-MIS-ONNX-SQUAD | Mistral-7B | ONNX Export | SQuAD | Planning | 1 | onnx model, metrics |
| EXP-T1-W5-MIS-ONNX-ALP | Mistral-7B | ONNX Export | Alpaca | Planning | 1 | onnx model, metrics |
| EXP-T1-W5-LLAMA-8BIT-CNN | Llama-2-13B | 8-bit Quant | CNN/DailyMail | Planning | 0.3 | inference metrics |
| EXP-T1-W5-LLAMA-8BIT-SQUAD | Llama-2-13B | 8-bit Quant | SQuAD | Planning | 0.3 | inference metrics |
| EXP-T1-W5-LLAMA-8BIT-ALP | Llama-2-13B | 8-bit Quant | Alpaca | Planning | 0.3 | inference metrics |
| EXP-T1-W5-LLAMA-4BIT-CNN | Llama-2-13B | 4-bit Quant | CNN/DailyMail | Planning | 0.3 | inference metrics |
| EXP-T1-W5-LLAMA-4BIT-SQUAD | Llama-2-13B | 4-bit Quant | SQuAD | Planning | 0.3 | inference metrics |
| EXP-T1-W5-LLAMA-4BIT-ALP | Llama-2-13B | 4-bit Quant | Alpaca | Planning | 0.3 | inference metrics |
| EXP-T1-W5-LLAMA-ONNX-CNN | Llama-2-13B | ONNX Export | CNN/DailyMail | Planning | 1.5 | onnx model, metrics |
| EXP-T1-W5-LLAMA-ONNX-SQUAD | Llama-2-13B | ONNX Export | SQuAD | Planning | 1.5 | onnx model, metrics |
| EXP-T1-W5-LLAMA-ONNX-ALP | Llama-2-13B | ONNX Export | Alpaca | Planning | 1.5 | onnx model, metrics |

**Week 5 Subtotal:** 14 GPU hours (quantization/export is fast)

**Quality Gate:** Quantization VRAM should be 25-30% of baseline  
**Success Criteria:** All 18 exports successful; latency measurements recorded

---

#### WEEK 6: Baseline Inference & Final Benchmarking

| Exp ID | Model | Technique | Dataset | Status | GPU Hours | Expected Output |
|--------|-------|-----------|---------|--------|-----------|-----------------|
| EXP-T1-W6-MIS-BASE-CNN | Mistral-7B | Baseline (FP32) | CNN/DailyMail | Planning | 0.5 | latency, VRAM, quality |
| EXP-T1-W6-MIS-BASE-SQUAD | Mistral-7B | Baseline (FP32) | SQuAD | Planning | 0.5 | latency, VRAM, quality |
| EXP-T1-W6-MIS-BASE-ALP | Mistral-7B | Baseline (FP32) | Alpaca | Planning | 0.5 | latency, VRAM, quality |
| EXP-T1-W6-LLAMA-BASE-CNN | Llama-2-13B | Baseline (FP32) | CNN/DailyMail | Planning | 0.7 | latency, VRAM, quality |
| EXP-T1-W6-LLAMA-BASE-SQUAD | Llama-2-13B | Baseline (FP32) | SQuAD | Planning | 0.7 | latency, VRAM, quality |
| EXP-T1-W6-LLAMA-BASE-ALP | Llama-2-13B | Baseline (FP32) | Alpaca | Planning | 0.7 | latency, VRAM, quality |

**Week 6 Subtotal:** 3.4 GPU hours (consolidation and final runs)

**Quality Gate:** Baseline measurements complete for all datasets  
**Success Criteria:** master_benchmark_results.csv created with all 54 rows

---

## TIER 1 SUMMARY: 54 Full Experiments

**Total GPU Hours:** ~115 hours (spread over Weeks 2-6)

**GPU Allocation Strategy:**
- If using Kaggle free tier: 30 hours/week available
  - Week 2: Full fine-tunes (56 hrs → split across 2 weeks or do in parallel if multiple GPUs)
  - Can compress to 6-7 weeks with careful scheduling

**Output Files Expected:**
```
TIER_1_RESULTS/
├── baseline_results.csv (6 rows: 2 models × 3 datasets)
├── full_finetune_results.csv (6 rows: training metrics)
├── lora_results.csv (6 rows: training metrics + adapter size)
├── qlora_results.csv (6 rows: training metrics + 4-bit config)
├── quantization_8bit_results.csv (6 rows: inference metrics)
├── quantization_4bit_results.csv (6 rows: inference metrics)
├── inference_optimization_results.csv (18 rows: ONNX benchmarks)
└── master_benchmark_results.csv (54 rows: all metrics, all experiments)

Checkpoints/Models:
├── checkpoints/mistral_full_finetuned_*
├── checkpoints/mistral_lora_adapters_*
├── checkpoints/llama_full_finetuned_*
├── checkpoints/llama_lora_adapters_*
├── onnx_models/mistral_onnx
└── onnx_models/llama_onnx
```

---

## TIER 2: Validation Benchmarking (18 Experiments)

### Week 7: Quick Validation on 2 Additional Models

**Key Difference from Tier 1:**
- Only 6 techniques (not 9): Baseline, LoRA, QLoRA, 8-bit, 4-bit, ONNX
- All 3 datasets (same as Tier 1)
- No manual evaluation (use automated metrics only)
- No deep analysis (just confirm numbers transfer)
- Duration: 1 week (compact week)

#### Model 2A: MPT-7B Validation

| Exp ID | Model | Technique | Dataset | Status | GPU Hours | Expected Output |
|--------|-------|-----------|---------|--------|-----------|-----------------|
| EXP-T2-W7-MPT-BASE-CNN | MPT-7B | Baseline | CNN/DailyMail | Planning | 0.5 | latency, VRAM |
| EXP-T2-W7-MPT-BASE-SQUAD | MPT-7B | Baseline | SQuAD | Planning | 0.5 | latency, VRAM |
| EXP-T2-W7-MPT-BASE-ALP | MPT-7B | Baseline | Alpaca | Planning | 0.5 | latency, VRAM |
| EXP-T2-W7-MPT-LORA-CNN | MPT-7B | LoRA | CNN/DailyMail | Planning | 2 | training time, quality |
| EXP-T2-W7-MPT-LORA-SQUAD | MPT-7B | LoRA | SQuAD | Planning | 2 | training time, quality |
| EXP-T2-W7-MPT-LORA-ALP | MPT-7B | LoRA | Alpaca | Planning | 1.5 | training time, quality |
| EXP-T2-W7-MPT-QLORA-CNN | MPT-7B | QLoRA | CNN/DailyMail | Planning | 1 | training time, quality |
| EXP-T2-W7-MPT-QLORA-SQUAD | MPT-7B | QLoRA | SQuAD | Planning | 1 | training time, quality |
| EXP-T2-W7-MPT-QLORA-ALP | MPT-7B | QLoRA | Alpaca | Planning | 0.75 | training time, quality |

**MPT-7B Subtotal:** 9.25 GPU hours

#### Model 2B: CodeLlama-7B Validation

| Exp ID | Model | Technique | Dataset | Status | GPU Hours | Expected Output |
|--------|-------|-----------|---------|--------|-----------|-----------------|
| EXP-T2-W7-CODE-BASE-CNN | CodeLlama-7B | Baseline | CNN/DailyMail | Planning | 0.5 | latency, VRAM |
| EXP-T2-W7-CODE-BASE-SQUAD | CodeLlama-7B | Baseline | SQuAD | Planning | 0.5 | latency, VRAM |
| EXP-T2-W7-CODE-BASE-ALP | CodeLlama-7B | Baseline | Alpaca | Planning | 0.5 | latency, VRAM |
| EXP-T2-W7-CODE-LORA-CNN | CodeLlama-7B | LoRA | CNN/DailyMail | Planning | 2 | training time, quality |
| EXP-T2-W7-CODE-LORA-SQUAD | CodeLlama-7B | LoRA | SQuAD | Planning | 2 | training time, quality |
| EXP-T2-W7-CODE-LORA-ALP | CodeLlama-7B | LoRA | Alpaca | Planning | 1.5 | training time, quality |
| EXP-T2-W7-CODE-QLORA-CNN | CodeLlama-7B | QLoRA | CNN/DailyMail | Planning | 1 | training time, quality |
| EXP-T2-W7-CODE-QLORA-SQUAD | CodeLlama-7B | QLoRA | SQuAD | Planning | 1 | training time, quality |
| EXP-T2-W7-CODE-QLORA-ALP | CodeLlama-7B | QLoRA | Alpaca | Planning | 0.75 | training time, quality |

**CodeLlama-7B Subtotal:** 9.25 GPU hours

---

## TIER 2 SUMMARY: 18 Validation Experiments

**Total GPU Hours:** ~18.5 hours (compressed into 1 week)

**Output Files Expected:**
```
TIER_2_RESULTS/
├── mpt_validation_results.csv (9 rows)
├── codellama_validation_results.csv (9 rows)
└── validation_benchmark_results.csv (18 rows: combined)
```

---

## COMPLETE EXPERIMENT MATRIX SUMMARY

**Total Experiments:** 72 (54 Tier 1 + 18 Tier 2)  
**Total GPU Hours:** ~133.5 hours  
**Timeline:** 7 weeks (Weeks 2-8)  

**Breakdown:**
```
Week 2: Full fine-tuning (56 hrs)          → 6 experiments
Week 3: LoRA fine-tuning (28 hrs)          → 6 experiments
Week 4: QLoRA fine-tuning (14 hrs)         → 6 experiments
Week 5: Quantization & ONNX (14 hrs)       → 18 experiments
Week 6: Baselines & compilation (3.4 hrs)  → 6 experiments + CSV merge
Week 7: Validation on 2 models (18.5 hrs)  → 18 experiments
TOTAL:                        133.5 hours  → 72 experiments
```

**What This Gets You:**
- ✅ 54 rigorous experiments on primary models
- ✅ 18 validation experiments on secondary models
- ✅ Covers all 9 optimization techniques
- ✅ Covers all 3 datasets
- ✅ Covers all 4 models
- ✅ Evidence of generalization across architectures
- ✅ Defensible against "why only 2 models?"

---

---

# 4. DETAILED PHASE BREAKDOWN (WEEK BY WEEK)

## WEEK 1: SETUP & PREPARATION (Not Counted in GPU Hours)

### Goals
- [ ] Environment ready
- [ ] All data downloaded
- [ ] All code tested
- [ ] Baseline established

### Day 1-2: Environment Setup

**Task 1.1: Create Project Structure**
```bash
llm_optimization_hybrid/
├── data/
│   ├── cnn_dailymail/
│   ├── squad/
│   └── alpaca/
├── experiments/
│   ├── tier_1/
│   │   ├── 01_baseline.py
│   │   ├── 02_full_finetune.py
│   │   ├── 03_lora_finetune.py
│   │   ├── 04_qlora_finetune.py
│   │   ├── 05_quantization.py
│   │   └── 06_onnx_export.py
│   └── tier_2/
│       └── 01_validation.py
├── utils/
│   ├── data_loader.py
│   ├── metrics.py
│   └── config.py
├── results/
│   ├── tier_1/
│   └── tier_2/
├── checkpoints/
├── logs/
├── api/
│   └── inference_server.py
├── dashboard/
│   └── dashboard.py
├── requirements.txt
├── .env
└── README.md
```

**Task 1.2: Environment Installation**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python verify_setup.py  # Custom script to verify everything
```

**Verification Script Should Check:**
- [ ] CUDA available (nvidia-smi works)
- [ ] PyTorch installed and can detect GPU
- [ ] All required packages importable
- [ ] Can download a test model from HuggingFace
- [ ] Disk space > 500GB available
- [ ] RAM > 32GB available

---

### Day 3: Data Download & Preparation

**Task 1.3: Download Datasets**
```python
# Script: data/download_datasets.py

from datasets import load_dataset
import os

DATASETS = {
    'cnn_dailymail': ('cnn_dailymail', '3.0.0', 'train'),
    'squad': ('squad', 'train'),
    'alpaca': ('tatsu-lab/alpaca', 'train')
}

SAMPLE_SIZES = {
    'cnn_dailymail': 5000,
    'squad': 5000,
    'alpaca': 2000
}

for name, config in DATASETS.items():
    print(f"Downloading {name}...")
    dataset = load_dataset(*config)
    # Sample if needed
    if len(dataset) > SAMPLE_SIZES.get(name, float('inf')):
        dataset = dataset.select(range(SAMPLE_SIZES[name]))
    
    # Save locally
    dataset.save_to_disk(f"data/{name}")
    print(f"✓ {name} saved to data/{name}")
```

**Expected Output:**
```
✓ CNN/DailyMail saved (5000 train, 1250 val, 1250 test samples)
✓ SQuAD saved (5000 train, 1250 val, 1250 test samples)
✓ Alpaca saved (2000 train, 500 val, 500 test samples)
Total disk space used: ~8GB
```

---

### Day 4: Baseline Establishment

**Task 1.4: Run Zero-Shot Baseline**
```python
# Script: experiments/tier_1/00_baseline.py

BASELINE_CONFIG = {
    'models': [
        'mistralai/Mistral-7B-v0.1',
        'meta-llama/Llama-2-13b'
    ],
    'datasets': ['cnn_dailymail', 'squad', 'alpaca'],
    'num_samples': 10,  # Just 10 samples to establish baseline
    'metrics': ['latency_ms', 'vram_gb', 'quality_score']
}

# For each model/dataset combo:
# 1. Load model
# 2. Run inference on 10 samples
# 3. Measure latency (ms), VRAM (GB), quality metrics
# 4. Save to baseline_results.csv

Output: baseline_results.csv
Format:
Model, Dataset, Latency_ms, VRAM_GB, Quality_Score
mistral-7b, cnn_dailymail, 245.3, 15.2, 0.54
mistral-7b, squad, 187.4, 14.8, 0.62
mistral-7b, alpaca, 156.2, 14.5, 0.68
llama-2-13b, cnn_dailymail, 356.7, 24.3, 0.58
llama-2-13b, squad, 287.4, 23.8, 0.65
llama-2-13b, alpaca, 234.5, 23.1, 0.71
```

**Quality Gate:** Baseline complete for all model/dataset combos

---

### Day 5: Code Walkthrough & Documentation

**Task 1.5: Prepare Experiment Scripts**
- [ ] Write Phase 1 script template
- [ ] Test on single experiment
- [ ] Document expected outputs
- [ ] Create logging/tracking system

**Task 1.6: Create Experiment Tracking Template**
```yaml
# logs/experiment_log.yaml

Week2:
  EXP-T1-W2-MIS-FULL-CNN:
    start_time: "YYYY-MM-DD HH:MM:SS"
    end_time: "TBD"
    status: "pending"
    checkpoint_path: ""
    training_time_hours: 0
    peak_vram_gb: 0
    final_loss: 0
    quality_score: 0
    errors: []
```

**Deliverables for Week 1:**
- ✅ Project structure created
- ✅ All dependencies installed & verified
- ✅ All datasets downloaded (8GB)
- ✅ Baseline established (6 experiments done)
- ✅ Code ready for Week 2
- ✅ Experiment tracking system ready

---

## WEEK 2: FULL FINE-TUNING (56 GPU Hours)

### Goals
- [ ] Full fine-tune on Mistral-7B (3 datasets)
- [ ] Full fine-tune on Llama-2-13B (3 datasets)
- [ ] All checkpoints saved
- [ ] Training metrics recorded

### Script Template: Full Fine-Tuning

```python
# experiments/tier_1/02_full_finetune.py

FULL_FINETUNE_CONFIG = {
    'mistralai/Mistral-7B-v0.1': {
        'batch_size': 8,
        'learning_rate': 2e-5,
        'epochs': 2,
        'warmup_ratio': 0.03,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'dtype': 'float32'  # Full precision baseline
    },
    'meta-llama/Llama-2-13b': {
        'batch_size': 4,
        'learning_rate': 1e-5,
        'epochs': 2,
        'warmup_ratio': 0.03,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'dtype': 'float32'  # Full precision baseline
    }
}

# Run 6 experiments:
# For each model in ['mistral', 'llama']:
#     For each dataset in ['cnn', 'squad', 'alpaca']:
#         1. Load model in FP32
#         2. Load dataset
#         3. Train for 2 epochs with specific learning rate
#         4. Save checkpoint
#         5. Log: training_time, peak_vram, final_loss
#         6. Run inference on test set
#         7. Compute quality metrics
#         8. Save results to CSV
```

### Week 2 Schedule

**Days 1-2: Mistral-7B Fine-Tuning**
```
Monday:   EXP-T1-W2-MIS-FULL-CNN   (8 hrs)  [24h wall time: 8h training + overhead]
Tuesday:  EXP-T1-W2-MIS-FULL-SQUAD (8 hrs)  [24h wall time]
Wednesday EXP-T1-W2-MIS-FULL-ALP   (6 hrs)  [24h wall time]
```

**Days 3-4: Llama-2-13B Fine-Tuning**
```
Wednesday: EXP-T1-W2-LLAMA-FULL-CNN   (12 hrs) [48h wall time]
Thursday:  EXP-T1-W2-LLAMA-FULL-SQUAD (12 hrs) [48h wall time]
Friday:    EXP-T1-W2-LLAMA-FULL-ALP   (10 hrs) [48h wall time]
```

(Note: Can run in parallel if multiple GPUs available)

### Expected Output Structure

```
checkpoints/
├── W2/
│   ├── mistral_full_finetune_cnn_dailymail/
│   │   ├── pytorch_model.bin
│   │   ├── training_args.bin
│   │   ├── config.json
│   │   └── trainer_state.json
│   ├── mistral_full_finetune_squad/
│   ├── mistral_full_finetune_alpaca/
│   ├── llama_full_finetune_cnn_dailymail/
│   ├── llama_full_finetune_squad/
│   └── llama_full_finetune_alpaca/

results/tier_1/
├── W2_full_finetune_mistral.csv
│   Columns: dataset, training_time_hours, peak_vram_gb, 
│             final_loss, validation_loss, rouge1, rouge2, rougeL, f1
│   Rows: 3 (one per dataset)
└── W2_full_finetune_llama.csv
    (same format, 3 rows)
```

### Quality Gates for Week 2

| Gate | Criteria | Acceptance |
|------|----------|-----------|
| Training Completion | All 6 experiments finish | ✅ 6/6 complete |
| Peak VRAM | Mistral <20GB, Llama <28GB | ✅ Both within limit |
| Training Time | Mistral 6-10 hrs/dataset, Llama 10-14 hrs/dataset | ✅ Reasonable |
| Loss Convergence | Validation loss decreases by epoch 2 | ✅ Yes |
| Quality Metrics | ROUGE > 0.2, F1 > 0.4 | ✅ Acceptable baseline |
| Checkpoint Size | Model file < 30GB | ✅ Normal |

### If Any Experiment Fails

**Failure Procedure:**
1. Check error log (saved to logs/W2_errors.log)
2. If OOM error:
   - Reduce batch size by 50%
   - Restart experiment
   - Note in logs why OOM happened
3. If NaN loss:
   - Check learning rate (might be too high)
   - Reduce by 50%
   - Restart
4. If timeout:
   - Save intermediate checkpoint
   - Resume from checkpoint
   - Or skip this experiment and move to next

**Always log:** What went wrong, what you tried, what worked

---

## WEEK 3: LORA FINE-TUNING (28 GPU Hours)

### Goals
- [ ] LoRA adapters on Mistral (3 datasets)
- [ ] LoRA adapters on Llama (3 datasets)
- [ ] Training time ~50% of full fine-tune
- [ ] Adapter models saved separately

### Configuration

```python
LORA_CONFIG = {
    'mistralai/Mistral-7B-v0.1': {
        'r': 8,              # Rank
        'lora_alpha': 16,    # Alpha
        'lora_dropout': 0.05,
        'target_modules': ['q_proj', 'v_proj'],
        'bias': 'none',
        'task_type': 'CAUSAL_LM',
        'batch_size': 8,
        'learning_rate': 2e-4
    },
    'meta-llama/Llama-2-13b': {
        'r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'target_modules': ['q_proj', 'v_proj'],
        'bias': 'none',
        'task_type': 'CAUSAL_LM',
        'batch_size': 4,
        'learning_rate': 2e-4
    }
}
```

### Week 3 Schedule

**Monday-Tuesday: Mistral LoRA**
```
EXP-T1-W3-MIS-LORA-CNN   (4 hrs)  Expected: 50% of W2 time
EXP-T1-W3-MIS-LORA-SQUAD (4 hrs)
EXP-T1-W3-MIS-LORA-ALP   (3 hrs)
```

**Wednesday-Friday: Llama LoRA**
```
EXP-T1-W3-LLAMA-LORA-CNN   (6 hrs)
EXP-T1-W3-LLAMA-LORA-SQUAD (6 hrs)
EXP-T1-W3-LLAMA-LORA-ALP   (5 hrs)
```

### Expected Output

```
checkpoints/
└── W3/
    ├── mistral_lora_cnn_dailymail/
    │   ├── adapter_model.bin
    │   ├── adapter_config.json
    │   └── training_args.bin
    ├── mistral_lora_squad/
    ├── mistral_lora_alpaca/
    ├── llama_lora_cnn_dailymail/
    ├── llama_lora_squad/
    └── llama_lora_alpaca/

results/tier_1/
├── W3_lora_mistral.csv
│   Columns: dataset, adapter_size_mb, training_time_hours, 
│             peak_vram_gb, quality_degradation_percent
└── W3_lora_llama.csv
```

### Quality Gate

| Gate | Expected | Actual |
|------|----------|--------|
| Training Time | 50-60% of full fine-tune | ?  |
| Adapter Size | < 50MB each | ? |
| Peak VRAM | 40-50% reduction vs full FT | ? |
| Quality vs Baseline | < 1% degradation | ? |

---

## WEEK 4: QLORA FINE-TUNING (14 GPU Hours)

### Configuration

```python
QLORA_CONFIG = {
    'bnb_4bit_quant_type': 'nf4',
    'bnb_4bit_use_double_quant': True,
    'bnb_4bit_compute_dtype': 'bfloat16',
    'lora_r': 8,  # Mistral / 16 for Llama
    'lora_alpha': 16,  # / 32 for Llama
    'lora_dropout': 0.05,
    'batch_size': 4,
    'learning_rate': 2e-4
}
```

### Week 4 Schedule

```
Monday:   EXP-T1-W4-MIS-QLORA-CNN   (2 hrs) [25% of full fine-tune time]
Tuesday:  EXP-T1-W4-MIS-QLORA-SQUAD (2 hrs)
          EXP-T1-W4-MIS-QLORA-ALP   (1.5 hrs)
Wednesday EXP-T1-W4-LLAMA-QLORA-CNN   (3 hrs)
Thursday: EXP-T1-W4-LLAMA-QLORA-SQUAD (3 hrs)
Friday:   EXP-T1-W4-LLAMA-QLORA-ALP   (2.5 hrs)
```

### Expected Output

```
results/tier_1/
└── W4_qlora_results.csv
    Columns: model, dataset, adapter_size_mb, training_time_hours,
             peak_vram_gb, quality_degradation_percent
```

### Quality Gate

| Gate | Expected |
|------|----------|
| Training Time | 25-30% of full fine-tune |
| Peak VRAM | <20GB for both models |
| Quality | < 3% degradation |

---

## WEEK 5: QUANTIZATION & INFERENCE OPTIMIZATION (14 GPU Hours)

### Quantization (No Training, Just Inference)

**8-bit Quantization:**
```
EXP-T1-W5-MIS-8BIT-CNN   (0.2 hrs)  [load, infer, measure]
EXP-T1-W5-MIS-8BIT-SQUAD (0.2 hrs)
EXP-T1-W5-MIS-8BIT-ALP   (0.2 hrs)
EXP-T1-W5-LLAMA-8BIT-CNN   (0.3 hrs)
EXP-T1-W5-LLAMA-8BIT-SQUAD (0.3 hrs)
EXP-T1-W5-LLAMA-8BIT-ALP   (0.3 hrs)
```

**4-bit Quantization:**
```
EXP-T1-W5-MIS-4BIT-CNN   (0.2 hrs)
EXP-T1-W5-MIS-4BIT-SQUAD (0.2 hrs)
EXP-T1-W5-MIS-4BIT-ALP   (0.2 hrs)
EXP-T1-W5-LLAMA-4BIT-CNN   (0.3 hrs)
EXP-T1-W5-LLAMA-4BIT-SQUAD (0.3 hrs)
EXP-T1-W5-LLAMA-4BIT-ALP   (0.3 hrs)
```

### ONNX Export (Training Time for Export)

```
EXP-T1-W5-MIS-ONNX-CNN   (1 hr)  [export to ONNX format]
EXP-T1-W5-MIS-ONNX-SQUAD (1 hr)
EXP-T1-W5-MIS-ONNX-ALP   (1 hr)
EXP-T1-W5-LLAMA-ONNX-CNN   (1.5 hrs)
EXP-T1-W5-LLAMA-ONNX-SQUAD (1.5 hrs)
EXP-T1-W5-LLAMA-ONNX-ALP   (1.5 hrs)
```

### Week 5 Schedule

```
Monday:    EXP-T1-W5-MIS-8BIT-* (0.6 hrs total)
           EXP-T1-W5-MIS-4BIT-* (0.6 hrs total)
Tuesday:   EXP-T1-W5-LLAMA-8BIT-* (0.9 hrs)
           EXP-T1-W5-LLAMA-4BIT-* (0.9 hrs)
Wed-Fri:   ONNX exports (9 hrs)
```

### Expected Output

```
results/tier_1/
├── W5_quantization_8bit.csv
├── W5_quantization_4bit.csv
└── W5_inference_optimization.csv

models/
├── onnx_models/
│   ├── mistral_onnx/
│   ├── llama_onnx/
│   └── (optimized ONNX format)
```

---

## WEEK 6: BASELINE & COMPILATION (3.4 GPU Hours + Analysis)

### Task: Final Baseline Measurements

Run full baseline inference (not just 10 samples, but 100+ samples) for final reference.

```
EXP-T1-W6-MIS-BASE-CNN   (0.5 hrs)
EXP-T1-W6-MIS-BASE-SQUAD (0.5 hrs)
EXP-T1-W6-MIS-BASE-ALP   (0.5 hrs)
EXP-T1-W6-LLAMA-BASE-CNN   (0.7 hrs)
EXP-T1-W6-LLAMA-BASE-SQUAD (0.7 hrs)
EXP-T1-W6-LLAMA-BASE-ALP   (0.7 hrs)
```

### Task: Compile Master CSV

```python
# Script: compile_results.py

# Read all CSVs from W2-W6
# Merge into single master_benchmark_results.csv
# Columns:
Model, Technique, Dataset, Training_Time_hrs, Peak_VRAM_GB,
Model_Size_MB, Inference_Latency_ms, Throughput_tokens_sec,
Inference_VRAM_GB, Quality_Score, Quality_Degradation_percent,
Speedup_factor, Memory_Reduction_percent

# 54 rows (one per experiment)
# Validate: No NaN values, all metrics present
```

### Week 6 Deliverables

```
results/tier_1/
├── master_benchmark_results.csv (54 rows, 13 columns)
├── efficiency_scores.csv
├── cost_benefit_analysis.csv
└── summary_statistics.txt
```

---

## WEEK 7: VALIDATION ON ADDITIONAL MODELS (18.5 GPU Hours)

### Goal: Quick Validation on 2 More Models

**Model 2A: MPT-7B**
```
EXP-T2-W7-MPT-BASE-CNN     (0.5 hrs)
EXP-T2-W7-MPT-BASE-SQUAD   (0.5 hrs)
EXP-T2-W7-MPT-BASE-ALP     (0.5 hrs)
EXP-T2-W7-MPT-LORA-CNN     (2 hrs)
EXP-T2-W7-MPT-LORA-SQUAD   (2 hrs)
EXP-T2-W7-MPT-LORA-ALP     (1.5 hrs)
EXP-T2-W7-MPT-QLORA-CNN    (1 hr)
EXP-T2-W7-MPT-QLORA-SQUAD  (1 hr)
EXP-T2-W7-MPT-QLORA-ALP    (0.75 hrs)
```

**Model 2B: CodeLlama-7B**
```
EXP-T2-W7-CODE-BASE-CNN    (0.5 hrs)
EXP-T2-W7-CODE-BASE-SQUAD  (0.5 hrs)
EXP-T2-W7-CODE-BASE-ALP    (0.5 hrs)
EXP-T2-W7-CODE-LORA-CNN    (2 hrs)
EXP-T2-W7-CODE-LORA-SQUAD  (2 hrs)
EXP-T2-W7-CODE-LORA-ALP    (1.5 hrs)
EXP-T2-W7-CODE-QLORA-CNN   (1 hr)
EXP-T2-W7-CODE-QLORA-SQUAD (1 hr)
EXP-T2-W7-CODE-QLORA-ALP   (0.75 hrs)
```

### Week 7 Schedule

```
Monday:    MPT baseline + first LoRA runs (4 hrs)
Tuesday:   MPT LoRA completion + CodeLlama baseline (4 hrs)
Wed-Fri:   CodeLlama LoRA + QLoRA for both models (10.5 hrs)
```

### Expected Output

```
results/tier_2/
├── mpt_validation_results.csv (9 rows)
└── codellama_validation_results.csv (9 rows)

Final consolidated file:
results/
└── hybrid_validation_benchmark_results.csv (18 rows)
```

### Validation Comparison Format

```
Model, Technique, Dataset, Inference_Latency_ms, Quality_Score,
Quality_vs_Mistral_percent

(This shows: if Mistral had latency 245ms, does MPT have ~245ms? 
If not, why? Is it architecture, is it training data?)
```

---

## WEEKS 8-9: API, DASHBOARD, REPORT (Non-GPU)

(Covered in separate section)

---

---

# 5. DATA FLOW ARCHITECTURE

## End-to-End Data Pipeline

```
                      ┌─────────────────┐
                      │  Raw Datasets   │
                      │  (HF Hub)       │
                      └────────┬────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │  Download &      │
                    │  Preprocess      │  (Week 1)
                    │  Save Locally    │
                    └────────┬─────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
                ▼            ▼            ▼
        ┌─────────────┐┌──────────┐┌──────────┐
        │ CNN/DailyML││  SQuAD   ││ Alpaca   │
        │ (5K/1.25K) ││(5K/1.25K)││(2K/500)  │
        └─────────────┘└──────────┘└──────────┘
                │            │            │
                └────────────┼────────────┘
                             │
                   ┌─────────▼─────────┐
                   │ Baseline Inference│  (Week 1)
                   │ (Zero-shot)       │
                   └────────┬──────────┘
                            │
                   ┌────────▼──────────┐
                   │ baseline_results  │
                   │ .csv (6 rows)     │
                   └────────┬──────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ W2: Full FT  │ │ W3: LoRA     │ │ W4: QLoRA    │
    │ 6 exp (56h)  │ │ 6 exp (28h)  │ │ 6 exp (14h)  │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │                │                │
           └────────────────┼────────────────┘
                            │
          ┌─────────────────▼─────────────────┐
          │  W5: Quantization & Inference Opt│
          │  (18 exp: 8-bit, 4-bit, ONNX)    │
          │  GPU Time: 14h                   │
          └────────────┬──────────────────────┘
                       │
          ┌────────────▼────────────┐
          │ W6: Baseline & Compile  │
          │ master_benchmark.csv    │
          │ (54 rows, 13 columns)   │
          └────────────┬────────────┘
                       │
                       │  ┌──────────────────────┐
                       │  │ TIER 1 COMPLETE      │
                       │  │ 54 experiments done  │
                       │  └──────────────────────┘
                       │
          ┌────────────▼────────────┐
          │ W7: Validation on 2 More│
          │ Models (MPT, CodeLlama) │
          │ 18 exp (18.5h)          │
          └────────────┬────────────┘
                       │
           ┌───────────▼────────────┐
           │  validation_benchmark  │
           │  .csv (18 rows)        │
           └───────────┬────────────┘
                       │
                       │  ┌──────────────────────┐
                       │  │ TIER 2 COMPLETE      │
                       │  │ 72 experiments done  │
                       │  └──────────────────────┘
                       │
           ┌───────────▼────────────┐
           │  Merge Results         │
           │  Analysis              │
           │  Visualization         │
           └───────────┬────────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │  DELIVERABLES         │
           │  - Report (12 pages)  │
           │  - API Server         │
           │  - Dashboard          │
           │  - All CSVs           │
           └───────────────────────┘
```

---

## Data Structure for Each Experiment

### Input Data Flow

```
Experiment: EXP-T1-W3-MIS-LORA-CNN

Inputs:
  1. Base Model: mistralai/Mistral-7B-v0.1
  2. Training Dataset: CNN/DailyMail (5000 samples)
  3. Validation Dataset: CNN/DailyMail (1250 samples)
  4. Test Dataset: CNN/DailyMail (1250 samples)
  5. Config: LORA_CONFIG['mistralai/Mistral-7B-v0.1']
  6. Hyperparams: batch_size=8, lr=2e-4, epochs=2

Processing:
  1. Load model + tokenizer from HuggingFace
  2. Apply LoRA config
  3. Tokenize datasets
  4. Initialize trainer
  5. Train 2 epochs
  6. Evaluate on validation set
  7. Run inference on test set

Outputs:
  1. Checkpoint: checkpoints/W3/mistral_lora_cnn_dailymail/
  2. Adapter: adapter_model.bin (~50MB)
  3. Metrics: training_loss, val_loss, ROUGE scores
  4. Logs: logs/W3/EXP-T1-W3-MIS-LORA-CNN.log
  5. Results: results/W3_lora_mistral.csv (1 row added)
  6. Inference: Sample outputs for manual evaluation
```

### Output Data Structure

**Per Experiment:**
```csv
Model,Technique,Dataset,Training_Time_hrs,Peak_VRAM_GB,Model_Size_MB,
Inference_Latency_ms,Throughput_tokens_sec,Inference_VRAM_GB,
Quality_Score,Quality_Degradation_percent,Speedup_factor,Memory_Reduction_percent

mistral-7b,lora,cnn_dailymail,4.2,8.3,50,156,1245,8.1,0.52,-3.7,1.57,45.3
```

**Master CSV (After Week 6):**
- 54 rows (one per experiment)
- 13 columns (metrics)
- All values filled
- No NaNs (or documented as "Not Applicable")

**Validation CSV (After Week 7):**
- 18 rows (validation experiments)
- 10 columns (selected metrics only)
- Compared against Tier 1 findings

---

## Logging & Tracking Schema

### Experiment Log Format

```json
{
  "experiment_id": "EXP-T1-W3-MIS-LORA-CNN",
  "tier": 1,
  "week": 3,
  "model": "mistralai/Mistral-7B-v0.1",
  "technique": "LoRA",
  "dataset": "cnn_dailymail",
  "status": "completed",
  "start_time": "2024-01-15T09:00:00Z",
  "end_time": "2024-01-15T13:15:00Z",
  "duration_hours": 4.25,
  "gpu_hours": 4.2,
  "hardware": {
    "gpu_type": "NVIDIA RTX 4090",
    "gpu_memory_gb": 24,
    "cuda_version": "12.1",
    "pytorch_version": "2.0.0"
  },
  "config": {
    "lora_r": 8,
    "lora_alpha": 16,
    "batch_size": 8,
    "learning_rate": 2e-4
  },
  "metrics": {
    "training_loss": 1.234,
    "validation_loss": 1.456,
    "peak_vram_gb": 8.3,
    "inference_latency_ms": 156,
    "quality_score": 0.52,
    "quality_degradation_percent": -3.7
  },
  "outputs": {
    "checkpoint_path": "checkpoints/W3/mistral_lora_cnn_dailymail",
    "adapter_size_mb": 50,
    "results_csv": "results/tier_1/W3_lora_mistral.csv"
  },
  "errors": [],
  "notes": "Training completed successfully. LoRA adapter 50MB. Quality within acceptable range."
}
```

### Tracking File (YAML)

```yaml
Project: LLM_Optimization_Hybrid
StartDate: 2024-01-08
TargetEndDate: 2024-02-25
TotalGPUHours: 133.5
TotalExperiments: 72

Tier1:
  Status: "In Progress"
  TotalExperiments: 54
  CompletedExperiments: 0
  FailedExperiments: 0
  GPUHoursUsed: 0 / 115
  
  Week2:
    Status: "Pending"
    Experiments: 6
    Completed: 0
    GPUHours: 0 / 56
    Experiments:
      - EXP-T1-W2-MIS-FULL-CNN: {"status": "pending", "hours": 8}
      # ... more

  Week3:
    Status: "Pending"
    # ... similar

Tier2:
  Status: "Pending"
  TotalExperiments: 18
  CompletedExperiments: 0
  GPUHoursUsed: 0 / 18.5
  # ...
```

---

---

# 6. QUALITY GATES & VALIDATION PROCEDURES

## Gate 1: Week Completion Gate

**Run After Each Week:**

```python
def validate_week_completion(week_num, tier):
    """Check all experiments for week completed successfully"""
    
    week_experiments = get_experiments_for_week(week_num, tier)
    
    # Check 1: All experiment results exist
    for exp in week_experiments:
        assert os.path.exists(exp['results_file']), f"Missing {exp['id']}"
    
    # Check 2: All CSVs have correct rows
    for exp in week_experiments:
        df = pd.read_csv(exp['results_file'])
        assert len(df) == exp['expected_rows'], f"Wrong row count in {exp['id']}"
    
    # Check 3: All metrics have values (no NaN)
    for exp in week_experiments:
        df = pd.read_csv(exp['results_file'])
        required_cols = ['training_time_hours', 'peak_vram_gb', 'quality_score']
        for col in required_cols:
            assert df[col].notna().all(), f"NaN in {col} for {exp['id']}"
    
    # Check 4: GPU hours used matches expectation (±10%)
    total_gpu_hours = sum_gpu_hours_for_week(week_num)
    expected_hours = week_expectations[week_num]
    assert abs(total_gpu_hours - expected_hours) < (expected_hours * 0.1)
    
    return True  # Week validated
```

**Gate Checklist:**

| Week | Expected Experiments | GPU Hours | Validation |
|------|----------------------|-----------|------------|
| 1 | N/A (Setup only) | N/A | All data downloaded, baseline done |
| 2 | 6 complete | ~56 | No VRAM > 28GB, all losses converged |
| 3 | 6 complete | ~28 | Training time 50-60% of W2 |
| 4 | 6 complete | ~14 | Training time 25-30% of W2, VRAM < 20GB |
| 5 | 18 complete | ~14 | All ONNX exports successful |
| 6 | 6 complete | ~3.4 | Master CSV created, 54 rows complete |
| 7 | 18 complete | ~18.5 | Validation results match Tier 1 patterns |

---

## Gate 2: Quality Metrics Gate

**After Master CSV Complete:**

```python
def validate_quality_metrics(master_csv_path):
    """Ensure metrics are sensible"""
    
    df = pd.read_csv(master_csv_path)
    
    # Gate 1: Latency increases make sense
    for model in df['Model'].unique():
        baseline_lat = df[(df['Model']==model) & (df['Technique']=='Baseline')]['Inference_Latency_ms'].mean()
        
        # Optimized techniques should be <=baseline latency
        opt_lats = df[(df['Model']==model) & (df['Technique'] != 'Baseline')]['Inference_Latency_ms']
        assert (opt_lats <= baseline_lat * 1.1).sum() > 10, "Most optimizations should speed up"
    
    # Gate 2: VRAM reductions make sense
    # Quantization should reduce VRAM by 25-75%
    for model in df['Model'].unique():
        baseline_vram = df[(df['Model']==model) & (df['Technique']=='Baseline')]['Inference_VRAM_GB'].mean()
        
        for quant_type in ['8-bit', '4-bit']:
            quant_vram = df[(df['Model']==model) & (df['Technique']==quant_type)]['Inference_VRAM_GB'].mean()
            reduction = (baseline_vram - quant_vram) / baseline_vram * 100
            assert 25 < reduction < 75, f"VRAM reduction {reduction}% unrealistic for {quant_type}"
    
    # Gate 3: Quality degradation reasonable
    # Quality loss should be <10% for reasonable optimizations
    for model in df['Model'].unique():
        for technique in ['LoRA', 'QLoRA', '8-bit']:
            if technique in df['Technique'].values:
                quality_deg = df[(df['Model']==model) & (df['Technique']==technique)]['Quality_Degradation_percent'].mean()
                assert abs(quality_deg) < 10, f"{technique} quality loss {quality_deg}% too large"
    
    return True  # Metrics validated
```

**Sensibility Checks:**

| Aspect | Expected Range | Red Flag |
|--------|-----------------|----------|
| Latency Speedup | 1.0x - 3.5x | > 5x (suspicious) or < 1x (regression) |
| VRAM Reduction | 25% - 75% | < 10% or > 90% (unrealistic) |
| Quality Degradation | -5% to +5% | > 10% (too much loss) |
| Training Time Ratio | LoRA 50-60% of Full, QLoRA 25-30% | Reversed ratio (bug) |

---

## Gate 3: Consistency Gate

**Check Results are Consistent Across Runs:**

```python
def validate_consistency():
    """If we run same experiment twice, results should be similar"""
    
    # If any experiment ran twice, verify results match ±5%
    for exp_id in get_duplicate_experiments():
        results = get_all_runs_for_experiment(exp_id)
        
        if len(results) >= 2:
            # Compare latency, VRAM, quality
            latencies = [r['Inference_Latency_ms'] for r in results]
            assert max(latencies) - min(latencies) < mean(latencies) * 0.05, \
                f"Inconsistent latency for {exp_id}: {latencies}"
            
            vrams = [r['Inference_VRAM_GB'] for r in results]
            assert max(vrams) - min(vrams) < mean(vrams) * 0.05, \
                f"Inconsistent VRAM for {exp_id}: {vrams}"
    
    return True
```

---

## Gate 4: Master CSV Integrity

```python
def validate_master_csv(csv_path):
    """Final validation before submission"""
    
    df = pd.read_csv(csv_path)
    
    # 1. Correct shape
    assert df.shape[0] == 54, f"Expected 54 rows, got {df.shape[0]}"
    assert df.shape[1] == 13, f"Expected 13 columns, got {df.shape[1]}"
    
    # 2. All required columns
    required_cols = [
        'Model', 'Technique', 'Dataset',
        'Training_Time_hrs', 'Peak_VRAM_GB', 'Model_Size_MB',
        'Inference_Latency_ms', 'Throughput_tokens_sec',
        'Inference_VRAM_GB', 'Quality_Score',
        'Quality_Degradation_percent', 'Speedup_factor',
        'Memory_Reduction_percent'
    ]
    assert all(col in df.columns for col in required_cols)
    
    # 3. All rows have correct combo of model/technique/dataset
    for model in ['Mistral-7B', 'Llama-2-13B']:
        for technique in ['Baseline', 'Full FT', 'LoRA', 'QLoRA', '8-bit', '4-bit', 'ONNX']:
            for dataset in ['CNN/DailyMail', 'SQuAD', 'Alpaca']:
                count = len(df[(df['Model']==model) & (df['Technique']==technique) & (df['Dataset']==dataset)])
                assert count == 1, f"Expected 1 row for {model}/{technique}/{dataset}, got {count}"
    
    # 4. No NaN values
    assert df.notna().all().all(), "CSV contains NaN values"
    
    # 5. All numeric columns are numeric
    numeric_cols = ['Training_Time_hrs', 'Peak_VRAM_GB', 'Inference_Latency_ms', ...]
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} should be numeric"
    
    # 6. Values are in reasonable ranges
    assert (df['Peak_VRAM_GB'] > 0).all() and (df['Peak_VRAM_GB'] < 50).all()
    assert (df['Inference_Latency_ms'] > 0).all() and (df['Inference_Latency_ms'] < 10000).all()
    assert (df['Quality_Score'] > 0).all() and (df['Quality_Score'] < 1).all()
    
    print("✓ Master CSV validation passed")
    return True
```

---

---

# 7. TRACKING & LOGGING SCHEMA

## Experiment Tracking Sheet

Create file: `logs/experiment_tracking.csv`

```csv
Experiment_ID,Tier,Week,Model,Technique,Dataset,Status,Start_DateTime,End_DateTime,
GPU_Hours,Peak_VRAM_GB,Training_Time_hrs,Quality_Score,Errors,Notes

EXP-T1-W2-MIS-FULL-CNN,1,2,Mistral-7B,Full_FT,CNN/DailyMail,completed,2024-01-15T09:00:00Z,
2024-01-15T17:00:00Z,8.0,15.2,8.0,0.54,,Training completed successfully. Loss converged.

EXP-T1-W2-MIS-FULL-SQUAD,1,2,Mistral-7B,Full_FT,SQuAD,completed,2024-01-16T09:00:00Z,
2024-01-16T17:00:00Z,8.0,14.8,8.0,0.62,,F1 score 0.62. Better than baseline.

EXP-T1-W2-MIS-FULL-ALP,1,2,Mistral-7B,Full_FT,Alpaca,completed,2024-01-17T09:00:00Z,
2024-01-17T15:00:00Z,6.0,14.5,6.0,0.68,,Instruction following task. Quality good.

...
```

**Track these fields:**
- [ ] Start & end times (to measure actual vs estimated)
- [ ] GPU hours used
- [ ] All metrics
- [ ] Any errors encountered
- [ ] Recovery steps taken
- [ ] Notes for future reference

## Daily Standup Log

File: `logs/daily_standup.md`

```markdown
# Daily Standup Log

## Week 2 - Day 1 (Monday, Jan 15)

**Experiments Planned:**
- EXP-T1-W2-MIS-FULL-CNN (8 hrs)

**Experiments Completed:**
- ✅ EXP-T1-W2-MIS-FULL-CNN (8 hrs, completed in 8h 15m)

**Metrics:**
- Training loss: 1.234 → 0.987 (converged)
- Validation loss: 1.456 → 1.234 (good)
- ROUGE-1: 0.54 (baseline: 0.50, +8%)

**Issues Encountered:**
- None

**GPU Usage:**
- Used 8 GPU hours
- Remaining Week 2 budget: 48 / 56 hours

**Next Steps:**
- Start EXP-T1-W2-MIS-FULL-SQUAD tomorrow (Tuesday)

---

## Week 2 - Day 2 (Tuesday, Jan 16)

[Continue log format...]
```

This keeps you accountable and provides detailed records for the report.

---

---

# 8. FAILURE RECOVERY PROCEDURES

## Scenario 1: CUDA Out Of Memory (OOM)

**When It Happens:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Recovery Steps:**

1. **First Attempt (Reduce Batch Size):**
   ```python
   # Original config
   batch_size = 8
   
   # Reduced config
   batch_size = 4  # Halve batch size
   
   # Restart experiment with new batch size
   # Expected: 1.5x longer training, but should fit
   ```
   - Log: "OOM recovered by reducing batch size from 8 to 4"
   - Retry: Rerun experiment with new config
   - If successful: Continue, note in results

2. **Second Attempt (Gradient Accumulation):**
   ```python
   # If still OOM:
   gradient_accumulation_steps = 2
   batch_size = 2  # Even smaller
   # Effective batch size = 2 * 2 = 4 (between first two attempts)
   ```
   - Log: "OOM recovered with gradient accumulation"
   - Retry: Rerun
   - If successful: Continue

3. **Third Attempt (CPU Offloading):**
   ```python
   device_map = "cpu"  # Offload to CPU
   # Very slow, but will work
   ```
   - Log: "OOM recovered by offloading to CPU (slow)"
   - This is last resort
   - May take 2-3x longer

4. **Final Attempt (Skip Experiment):**
   - If all else fails, log as "SKIPPED - OOM unrecoverable"
   - Move to next experiment
   - Note in final report: "Experiment X skipped due to hardware constraints"
   - Professor won't complain if you document it

---

## Scenario 2: NaN Loss (Training Diverged)

**When It Happens:**
```
Epoch 1: loss = 1.234
Epoch 1: loss = NaN  # Loss exploded
```

**Recovery Steps:**

1. **Reduce Learning Rate:**
   ```python
   learning_rate = 2e-5  # Original
   learning_rate = 1e-5  # Halved
   
   # Restart training
   ```
   - LR too high causes divergence
   - Halving LR usually fixes it

2. **Add Gradient Clipping:**
   ```python
   max_grad_norm = 0.5  # Tighter clipping
   # Default is 1.0
   ```

3. **Increase Warmup:**
   ```python
   warmup_ratio = 0.1  # Original 0.03
   # Gentler learning rate warmup
   ```

4. **Log & Retry:**
   ```
   LOG: "NaN loss at epoch 1. Reduced LR from 2e-5 to 1e-5. Retrying..."
   ```

---

## Scenario 3: Model Download Failed

**When It Happens:**
```
ConnectionError: Unable to establish connection to Hugging Face Hub
```

**Recovery:**

1. **Check Internet:**
   ```bash
   ping huggingface.co
   ```

2. **Retry with Exponential Backoff:**
   ```python
   import time
   max_retries = 5
   for attempt in range(max_retries):
       try:
           model = AutoModel.from_pretrained(model_name)
           break
       except ConnectionError:
           wait_time = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
           print(f"Retry {attempt+1}/{max_retries} after {wait_time}s")
           time.sleep(wait_time)
   ```

3. **Download Model Locally First:**
   ```bash
   # Use Hugging Face CLI
   huggingface-cli download mistralai/Mistral-7B-v0.1
   
   # Then load from local cache
   model = AutoModel.from_pretrained("~/.cache/huggingface/hub/...")
   ```

---

## Scenario 4: Experiment Timeout (Hung Process)

**When It Happens:**
- Training started 12 hours ago, nothing logged in last 2 hours
- Process is still running but not progressing

**Recovery:**

1. **Save Intermediate Checkpoint:**
   ```bash
   # Check if checkpoint saved during training
   ls -la checkpoints/W3/mistral_lora_cnn_dailymail/
   # Should show checkpoint-500, checkpoint-1000, etc.
   ```

2. **Resume from Checkpoint:**
   ```python
   # In config:
   resume_from_checkpoint = "checkpoints/W3/mistral_lora_cnn_dailymail/checkpoint-1000"
   
   # Trainer will resume from there
   ```

3. **If No Checkpoints, Kill & Restart:**
   ```bash
   kill <process_id>
   
   # Restart with smaller dataset
   num_samples = 100  # Instead of 5000
   
   # If small sample works, scale back up
   ```

---

## Scenario 5: Results Look Wrong/Unrealistic

**When It Happens:**
- Speedup is 10x (too good to be true)
- Quality degradation is -50% (models got worse)
- Latency increased (optimization made it slower)

**Recovery:**

1. **Double-Check Metrics Calculation:**
   ```python
   # Verify metric calculation
   # Example: speedup = baseline_latency / optimized_latency
   baseline_latency = 245.3  # ms
   optimized_latency = 24.5  # ms
   speedup = 245.3 / 24.5 = 10.0
   
   # Is 24.5ms realistic? Or is measurement wrong?
   ```

2. **Rerun Experiment:**
   ```bash
   python experiments/tier_1/XX_experiment.py --experiment_id EXP-T1-W3-MIS-LORA-CNN --rerun
   ```

3. **Compare with Baseline:**
   - If results differ >10% from first run → measurement error
   - If consistent → result is real (surprising but valid)

4. **Check for Common Bugs:**
   - [ ] Model loaded in wrong precision (FP32 vs FP16)?
   - [ ] Quantization config wrong?
   - [ ] Measuring wrong thing (e.g., only model latency, not including tokenization)?
   - [ ] Different dataset used?

5. **Document & Move On:**
   ```
   LOG: "Speedup 10x seems high. Verified calculation 3x. Result stands.
   Possible reason: LoRA weight offset enabled efficient inference."
   ```

---

## Scenario 6: Hyperparameter Completely Wrong

**When It Happens:**
- Training time 10x longer than expected
- Loss not converging
- Quality scores much worse

**Recovery:**

1. **Check Hyperparams vs Plan:**
   ```python
   # From LORA_CONFIG:
   expected_lr = 2e-4
   actual_lr = 1e-6  # Someone changed it wrong!
   ```

2. **Revert to Standard Config:**
   ```python
   # Use exactly what's in Code_Templates file
   LORA_CONFIG = {
       'mistralai/Mistral-7B-v0.1': {
           'r': 8,
           'lora_alpha': 16,
           'lora_dropout': 0.05,
           # ... copy exactly from template
       }
   }
   ```

3. **Restart with Correct Hyperparams:**
   ```bash
   python experiments/tier_1/03_lora_finetune.py --reset_hyperparams
   ```

---

# Recovery Procedure Summary

| Scenario | First Action | Second Action | Last Resort |
|----------|--------------|---------------|------------|
| OOM | Reduce batch size | Gradient accumulation | CPU offload |
| NaN Loss | Reduce LR | Increase warmup | Skip exp |
| Download Failed | Retry with backoff | Download locally | Manual setup |
| Process Hung | Check checkpoints | Resume from checkpoint | Kill & restart |
| Results Wrong | Double-check math | Rerun experiment | Check for bugs |
| Bad Hyperparams | Revert to template | Restart | Investigate |

**Golden Rule:** Always log what went wrong and what you tried. This helps:
1. Debug issues later
2. Explains any missing experiments in final report
3. Shows rigor (you didn't just ignore failures)

---

---

# 9. PHASE TRANSITION CHECKLIST

## Before Moving from Week 2 → Week 3

**Quality Gate Checklist:**
- [ ] All 6 Week 2 experiments completed
- [ ] No VRAM exceeds 28GB
- [ ] All training losses converged (no NaN, no divergence)
- [ ] All 6 results CSV entries complete
- [ ] Checkpoints saved and verified

**Data Integrity:**
- [ ] baseline_results.csv has 6 rows (2 models × 3 datasets)
- [ ] W2_full_finetune_mistral.csv has 3 rows
- [ ] W2_full_finetune_llama.csv has 3 rows
- [ ] Total GPU hours ≈ 56 (±10%)

**Documentation:**
- [ ] Daily standup log complete for all 5 days
- [ ] Experiment tracking spreadsheet updated
- [ ] No errors unfiled

**Approval:**
- [ ] Run validation script: `python validate_week.py 2`
- [ ] Script returns: "✓ Week 2 validation passed"

**Sign-Off:**
```
Week 2 Complete: ___________
Date: _______
All gates passed: YES / NO
Proceed to Week 3: YES / NO
```

---

## Before Moving from Tier 1 (Week 6) → Tier 2 (Week 7)

**Master CSV Creation:**
- [ ] All 54 Week 2-6 experiments completed
- [ ] master_benchmark_results.csv created
- [ ] 54 rows, 13 columns, zero NaN values
- [ ] CSV passes validation script

**Tier 1 Analysis:**
- [ ] Efficiency scores calculated
- [ ] Cost-benefit analysis done
- [ ] Decision matrix created

**Documentation:**
- [ ] Full standup log for Weeks 2-6
- [ ] Experiment tracking complete
- [ ] All errors documented & resolved

**Validation:**
- [ ] Run: `python validate_tier1_complete.py`
- [ ] Returns: "✓ Tier 1 complete and valid"

**Approval:**
```
Tier 1 Complete: ___________
Date: _______
Master CSV created: YES / NO
All 54 experiments valid: YES / NO
Ready for Tier 2: YES / NO
```

---

## Before Final Submission (End of Week 7)

**All Data Compiled:**
- [ ] Master Benchmark CSV (54 rows)
- [ ] Validation Benchmark CSV (18 rows)
- [ ] Combined Results CSV (72 rows)
- [ ] Efficiency scores CSV
- [ ] Cost-benefit CSV

**Code Complete:**
- [ ] All tier_1 scripts working
- [ ] All tier_2 scripts working
- [ ] API server tested
- [ ] Dashboard functional

**Report Ready:**
- [ ] 10-12 page main report (not draft, final)
- [ ] All figures with captions
- [ ] All tables formatted
- [ ] References complete

**Deliverables Package:**
- [ ] `TECHNICAL_REPORT.pdf` (final)
- [ ] `BENCHMARK_VISUALIZATIONS.pdf`
- [ ] `README.md`
- [ ] `REPRODUCIBILITY.md`
- [ ] All CSV files
- [ ] Code directory
- [ ] API directory
- [ ] Dashboard script
- [ ] GitHub repository

**Final Validation:**
- [ ] Run: `python final_validation.py`
- [ ] All gates pass
- [ ] No missing files
- [ ] No broken links

---

---

# 10. INTEGRATION WITH CLAUDE CODE

This document is designed to be your **foundation** for:

1. **claude.md** - Will reference this for:
   - Experiment configurations
   - Hyperparameters
   - Data flows
   - Logging requirements

2. **projectstate.md** - Will track:
   - Which experiments completed
   - GPU hours used
   - Which week we're in
   - Real-time progress vs plan

3. **architecture.md** - Will detail:
   - System design from this data flow section
   - Model/technique matrix from experiment matrix
   - Quality gate logic from section 6
   - Recovery procedures from section 8

---

## Next Steps:

1. **Read this document completely** (you just did!)
2. **Copy exact configs** from section 3 to your code
3. **Create tracking spreadsheet** from section 7
4. **Understand all 72 experiments** before starting
5. **Create claude.md, projectstate.md, architecture.md** based on this
6. **Start Week 1** with full confidence

---

## Quick Reference: Experiment Matrix Summary

**TIER 1 (54 Full Experiments - Weeks 2-6)**
```
2 Models × 9 Techniques × 3 Datasets = 54 experiments
- Mistral-7B: 27 experiments
- Llama-2-13B: 27 experiments

Total GPU Hours: ~115 hours
Timeline: 5 weeks
```

**TIER 2 (18 Validation Experiments - Week 7)**
```
2 Models × 6 Techniques × 3 Datasets = 18 experiments
- MPT-7B: 9 experiments
- CodeLlama-7B: 9 experiments

Total GPU Hours: ~18.5 hours
Timeline: 1 week
Quality: Selected metrics only (no manual eval)
```

**TOTAL: 72 Experiments, 133.5 GPU Hours, 7 Weeks**

---

**This document is your bible. Reference it constantly. It has everything you need to execute flawlessly.**

Good luck! 🚀
