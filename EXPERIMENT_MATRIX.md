# Experiment Matrix — Source of Truth

This is the ONLY place experiment specs and hyperparameters live. Code, tracking sheets, and Claude Code sessions should reference this file, not reproduce its content elsewhere. Supersedes the experiment matrix in `archive/HYBRID_APPROACH_DETAILED_IMPLEMENTATION_GUIDE.md` (that file's model rationale is still fine to read; its numbers are not).

**Why this exists as a separate file:** the old plan was one 2000+ line document. Loading all of it into a Claude Code session to answer "what's the LoRA learning rate again?" burns context for no reason. Keep this file short. If it grows past ~300 lines, split it by technique.

---

## Hardware Reality (why the matrix looks like this)

Kaggle free tier: T4x2 or P100, **16GB VRAM per GPU**, 30 GPU-hrs/week cap, 12h max session length.

Rough VRAM math that drove every decision below (bytes/param for Adam training: weights + grad + 2 moments):
- FP32 full fine-tune: ~16 bytes/param → 7B = ~112GB, 13B = ~208GB. **Does not fit. Not attempted.**
- FP16 LoRA (base frozen in fp16, only adapter trained): base weights ~2 bytes/param → 7B ≈ 14GB (tight but fits with batch=1 + gradient checkpointing), 13B ≈ 26GB (**does not fit**).
- QLoRA (base frozen in 4-bit NF4): base weights ~0.5 bytes/param → 7B ≈ 3.5GB, 13B ≈ 6.5GB. Adapters + activations add a few GB. **Fits both models comfortably** — this is the exact scenario QLoRA was designed for.
- Quantized inference only (no training, no gradients): 8-bit ≈ 1 byte/param, 4-bit ≈ 0.5 byte/param, both trivially fit either model.

**Consequence:** Mistral-7B gets the full technique set. Llama-2-13B skips fp16 LoRA (physically doesn't fit) but keeps everything else, including QLoRA — so we still get a "does fine-tuning scale to a bigger model" data point, just via QLoRA instead of LoRA.

---

## Models

| Key | HF ID | Params | Precision for baseline |
|-----|-------|--------|------------------------|
| MIS | `mistralai/Mistral-7B-v0.1` | 7.2B | fp16 |
| LLAMA | `meta-llama/Llama-2-13b-hf` | 13B | fp16 (baseline/eval only — never trained in fp16) |

## Datasets

| Key | Source | Task | Train / Val / Test rows |
|-----|--------|------|--------------------------|
| CNN | `cnn_dailymail` (v3.0.0) | Summarization | 1000 / 200 / 200 |
| SQUAD | `squad` | Extractive QA | 1000 / 200 / 200 |

Sample sizes are intentionally small relative to the original plan (was 5000/5000/2000 across 3 datasets). Alpaca/instruction-following is dropped for now — add back in Week 5 buffer only if GPU budget allows (see `PROJECT_STATE.md`). Small samples keep every experiment fast and rerunnable, which matters more than dataset size on a 30h/week budget.

---

## Technique Definitions & Hyperparameters

### 1. Baseline (zero-shot inference, no training)
- Load model in fp16, no quantization, no adapters
- Run inference on test split, measure latency (ms/sample), peak VRAM, quality metric
- Models: MIS, LLAMA — both

### 2. LoRA (fp16 base + trainable adapters)
- `r=8, lora_alpha=16, lora_dropout=0.05`
- `target_modules=["q_proj","v_proj"]`
- Base model: fp16, frozen; gradient checkpointing ON
- `batch_size=1, gradient_accumulation_steps=8` (effective batch 8)
- `learning_rate=2e-4`, optimizer `paged_adamw_8bit`, `max_seq_length=512`
- `epochs=2`
- **Models: MIS only** (doesn't fit on LLAMA — see VRAM math above)

### 3. QLoRA (4-bit NF4 base + trainable adapters)
- 4-bit NF4 quantization, double quantization, compute dtype `bf16`
- `r=16, lora_alpha=32, lora_dropout=0.05`
- `target_modules=["q_proj","v_proj","k_proj","o_proj"]`
- `batch_size=2` (MIS) / `batch_size=1` (LLAMA), `gradient_accumulation_steps=8` (MIS) / `16` (LLAMA)
- `learning_rate=2e-4`, optimizer `paged_adamw_8bit`, `max_seq_length=512`
- `epochs=2`
- **Models: MIS, LLAMA — both**

### 4. 8-bit quantized inference (no training)
- `load_in_8bit=True` via bitsandbytes, base pretrained weights (not fine-tuned)
- Inference only: latency, peak VRAM, quality vs baseline
- **Models: MIS, LLAMA — both**

### 5. 4-bit quantized inference (no training)
- `load_in_4bit=True`, NF4, double quantization, base pretrained weights
- Inference only: latency, peak VRAM, quality vs baseline
- **Models: MIS, LLAMA — both**

### 6. ONNX Runtime inference
- Export baseline fp16 model via `optimum.onnxruntime`, run with `CUDAExecutionProvider`
- Inference only: latency, throughput (tokens/sec), quality vs baseline (should be ~identical quality, this measures speed only)
- **Models: MIS, LLAMA — both**
- TensorRT conversion (PyTorch → ONNX → TensorRT) is a stretch goal attempted in the Week 5 buffer, not required for the core 22.

Note on QLoRA vs 4-bit-inference overlap: QLoRA structurally uses 4-bit quantization as part of fine-tuning, which conflates two variables (quantization + adaptation). The 4-bit-inference-only experiments give the isolated quantization signal (no fine-tuning) so the report can separate "what does 4-bit alone cost in quality" from "what does QLoRA fine-tuning buy you."

---

## Full Matrix (22 Experiments)

| Exp ID | Model | Technique | Dataset | Est. GPU Hrs | Output |
|--------|-------|-----------|---------|---------------|--------|
| EXP-MIS-BASE-CNN | Mistral-7B | Baseline | CNN | 0.3 | latency, VRAM, ROUGE |
| EXP-MIS-BASE-SQUAD | Mistral-7B | Baseline | SQuAD | 0.3 | latency, VRAM, F1/EM |
| EXP-MIS-LORA-CNN | Mistral-7B | LoRA | CNN | 2.0 | adapter, ROUGE |
| EXP-MIS-LORA-SQUAD | Mistral-7B | LoRA | SQuAD | 2.0 | adapter, F1/EM |
| EXP-MIS-QLORA-CNN | Mistral-7B | QLoRA | CNN | 1.5 | adapter, ROUGE |
| EXP-MIS-QLORA-SQUAD | Mistral-7B | QLoRA | SQuAD | 1.5 | adapter, F1/EM |
| EXP-MIS-8BIT-CNN | Mistral-7B | 8-bit | CNN | 0.2 | latency, VRAM, ROUGE |
| EXP-MIS-8BIT-SQUAD | Mistral-7B | 8-bit | SQuAD | 0.2 | latency, VRAM, F1/EM |
| EXP-MIS-4BIT-CNN | Mistral-7B | 4-bit | CNN | 0.2 | latency, VRAM, ROUGE |
| EXP-MIS-4BIT-SQUAD | Mistral-7B | 4-bit | SQuAD | 0.2 | latency, VRAM, F1/EM |
| EXP-MIS-ONNX-CNN | Mistral-7B | ONNX | CNN | 1.0 | latency, throughput |
| EXP-MIS-ONNX-SQUAD | Mistral-7B | ONNX | SQuAD | 1.0 | latency, throughput |
| EXP-LLAMA-BASE-CNN | Llama-2-13B | Baseline | CNN | 0.5 | latency, VRAM, ROUGE |
| EXP-LLAMA-BASE-SQUAD | Llama-2-13B | Baseline | SQuAD | 0.5 | latency, VRAM, F1/EM |
| EXP-LLAMA-QLORA-CNN | Llama-2-13B | QLoRA | CNN | 3.0 | adapter, ROUGE |
| EXP-LLAMA-QLORA-SQUAD | Llama-2-13B | QLoRA | SQuAD | 3.0 | adapter, F1/EM |
| EXP-LLAMA-8BIT-CNN | Llama-2-13B | 8-bit | CNN | 0.3 | latency, VRAM, ROUGE |
| EXP-LLAMA-8BIT-SQUAD | Llama-2-13B | 8-bit | SQuAD | 0.3 | latency, VRAM, F1/EM |
| EXP-LLAMA-4BIT-CNN | Llama-2-13B | 4-bit | CNN | 0.3 | latency, VRAM, ROUGE |
| EXP-LLAMA-4BIT-SQUAD | Llama-2-13B | 4-bit | SQuAD | 0.3 | latency, VRAM, F1/EM |
| EXP-LLAMA-ONNX-CNN | Llama-2-13B | ONNX | CNN | 1.5 | latency, throughput |
| EXP-LLAMA-ONNX-SQUAD | Llama-2-13B | ONNX | SQuAD | 1.5 | latency, throughput |

**Total estimated: ~21.3 GPU hours.** Budget 30-40 with reruns and mistakes (see `PROJECT_STATE.md` GPU Budget Tracking — 29.4h planned across weeks 1-4, buffer week reserved separately).

---

## Quality Metrics by Dataset

- CNN/DailyMail (summarization): ROUGE-1 / ROUGE-2 / ROUGE-L
- SQuAD (QA): Exact Match + F1

## Metrics Recorded Per Experiment

`Model, Technique, Dataset, Training_Time_hrs (N/A for inference-only), Peak_VRAM_GB, Inference_Latency_ms, Throughput_tokens_sec, Quality_Score, Quality_Degradation_percent (vs baseline), Speedup_factor (vs baseline)`

Baseline is always the fp16 zero-shot run for that model+dataset. Every other technique is compared against it — never against a different technique's output.

---

## Quality Gates

**After Week 2 (LoRA):** Mistral LoRA experiments complete, training time roughly 40-60% of what full FT would have cost (estimate, since full FT isn't run), no OOM.

**After Week 3 (QLoRA):** Both models' QLoRA experiments complete, peak VRAM stays under 14GB (leaving headroom on the 16GB card), training time visibly less than LoRA's.

**After Week 4 (Quantization + ONNX):** All 8 quantization + ONNX experiments complete, 4-bit VRAM measurably lower than 8-bit, ONNX latency measurably lower than fp16 baseline.

**Before Week 5 buffer closes:** `master_benchmark_results.csv` has all 22 rows, no NaNs, quality-vs-baseline numbers are all within plausible ranges (not >50% degradation, not negative latency, etc).

---

## Recovery Procedures (unchanged from CLAUDE.md, restated here for convenience)

**CUDA OOM:** reduce batch size → increase gradient accumulation → reduce `max_seq_length` to 256 → last resort, skip and document.

**Model silently CPU-offloaded / hangs:** `device_map="auto"` can silently spill layers to CPU (or `meta`) when a model doesn't fully fit in free VRAM instead of raising an error — `.generate()` then keeps running, just slowly enough to burn an entire 12h Kaggle session without producing output or a visible failure. `experiments/common.py` checks parameter device placement immediately after model load (and after `apply_lora()` for training runs) and raises `RuntimeError` right away if anything landed off-GPU, instead of hanging silently — this guard is a safety net, not the fix, so keep it in place even after the actual fix below.

**Actual root cause and fix:** a Llama-2-13B baseline run loaded the model a second time (once per dataset) in the same process, and even with explicit `del model; gc.collect(); torch.cuda.empty_cache()` before the second load, it still failed — the first release wasn't literally leaking the model, it was CUDA allocator fragmentation from the first load/train/inference cycle leaving free VRAM too fragmented for a second ~13B-model-sized contiguous allocation. Mistral-7B survived a second in-process load (with enough margin), Llama-2-13B did not. The del/gc/empty_cache combo was necessary but not sufficient. **The real fix is architectural: never reload a model in the same process.** `experiments/common.py` now exposes `run_inference_multi_dataset()` and `run_training_multi_dataset()`, which load the model **once**, loop all of that technique's datasets **in-memory** against the same model instance, and release **once** at the end — every `experiments/mistral/*.py` and `experiments/llama/*.py` script (except the ONNX scripts, which already followed this pattern) calls one of these instead of looping the old per-dataset functions directly. For training (LoRA/QLoRA), where each dataset needs its own independently-trained adapter, `run_training_multi_dataset()` applies a **fresh** LoRA adapter per dataset via `apply_lora()` and strips it back to the clean base model via `PeftModel.unload()` between datasets — the base weights are loaded once, but no dataset's adapter ever starts from another dataset's trained weights. If the device-placement guard still fires after all of this: reduce batch size / `max_seq_length`, or fall back to splitting model loads across separate script invocations (one `!python` cell per dataset). `generate_predictions()` also prints progress every 20 samples so a stuck run is visible in the Kaggle log instead of going silent for hours.

**NaN loss:** halve learning rate → increase grad clipping (`max_grad_norm=0.5`) → retry once → if still NaN, check data and skip.

**Kaggle session timeout (12h cap):** checkpoint every 200 steps; on restart, resume from last checkpoint rather than restarting the experiment.

**Weekly quota (30h) about to be exceeded:** stop, do not silently shrink the dataset or skip epochs — that invalidates comparability. Log it as a blocker in `PROJECT_STATE.md` and decide explicitly whether to wait for quota reset or move an experiment to next week.
