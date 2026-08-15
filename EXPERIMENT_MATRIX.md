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

## Qualitative Notes for Report

Aggregate metrics (ROUGE, EM/F1) don't always tell the full quality story. Spot-checks against raw generations (`logs/debug_predictions/<exp_id>.txt`, first 5 examples per experiment, added 2026-08-14) surfaced two findings worth carrying into the technical report:

### Project-Level Finding: T4/Turing Quantization Slowdown

**On this project's T4 GPU hardware, every bitsandbytes-based quantization technique tested (8-bit, 4-bit/QLoRA) trades substantial VRAM reduction for substantially slower inference and training compute — the opposite of the commonly assumed "quantization = faster" framing.** Likely cause: T4's Turing architecture lacks efficient native int8/int4 tensor-core paths that bitsandbytes relies on, so its dequantize-on-the-fly kernels do real extra work per step/token that a newer architecture (Ampere+) wouldn't pay as heavily. Three consistent data points so far, all Mistral-7B:

- **QLoRA training** (`EXP-MIS-QLORA-CNN`/`SQUAD`): ~73% less peak VRAM than fp16 LoRA (1.84-1.88GB vs 6.92-6.93GB), but ~14x slower per training step (183.97s/step vs 12.83s/step).
- **QLoRA inference** (same two rows): CNN inference 16658.0ms, *slower than the fp16 zero-shot baseline itself* (9723.3ms, speedup_factor 0.58); SQuAD inference 2465.5ms beat baseline (speedup_factor 2.66) but was still ~4x slower than fp16 LoRA's 625.1ms.
- **8-bit inference, no fine-tuning** (`EXP-MIS-8BIT-CNN`/`SQUAD`, 2026-08-15): peak VRAM 3.25GB vs baseline 6.91GB (-53%), but inference latency 25594.7ms (CNN) and 19562.4ms (SQuAD) vs baseline 9723.3ms/6560.0ms — 2.6x and ~3x *slower*, respectively (speedup_factor 0.38 / 0.34). Quality was roughly flat on CNN (+3.9%, noise-level) and degraded on SQuAD (-12.7%, EM 5.5 vs baseline 8.0) — unlike QLoRA, there's no fine-tuning here to explain a quality shift, so the VRAM-for-speed trade is not buying anything on the quality side either.

**Report framing:** this is now a pattern, not an isolated anomaly per technique — three independent measurements (one training, two inference) all point the same direction on the same hardware. `EXP-MIS-4BIT-CNN`/`EXP-MIS-4BIT-SQUAD` (pure 4-bit inference, not yet run) is the next data point: if it also lands slower than baseline despite lower VRAM, that's a fourth confirmation; if it instead comes in faster, it would be the first disconfirming result and worth investigating why 4-bit inference behaves differently from 8-bit inference and QLoRA on this hardware.

---

**`EXP-MIS-LORA-CNN`: ROUGE improvement is real but not "clean" summarization quality.** LoRA fine-tuning raised ROUGE-1/2/L by +21.1% over baseline (0.2890/0.1074/0.1965 vs 0.2387/0.0840/0.1607), but the raw predictions in `logs/debug_predictions/EXP-MIS-LORA-CNN.txt` show repetition loops (e.g. "I'm not going to say anything about the last few weeks" repeated verbatim 3x in example 0) and a rambling, run-on continuation style, rather than the terse bullet-point style CNN/DailyMail references use. This is consistent with a base (non-instruction-tuned) Mistral-7B model, LoRA r=8, only 1000 training examples, 2 epochs — a genuine quality limitation of this specific setup, not a pipeline defect. **Do not cite the +21.1% ROUGE gain in the report without this caveat** — ROUGE rewards n-gram overlap and is measurably fooled by verbose over-generation here.

**QLoRA's VRAM savings on Mistral-7B come at a real compute cost on this hardware — not a bug, a legitimate trade-off.** `EXP-MIS-QLORA-CNN`/`EXP-MIS-QLORA-SQUAD` used ~73% less peak VRAM than fp16 LoRA (1.84-1.88GB vs 6.92-6.93GB), exactly as expected for 4-bit quantized base weights. But training was ~14x slower *per step* (183.97s/step vs LoRA's 12.83s/step), and inference was slower too — CNN's QLoRA inference (16658.0ms) was even slower than the fp16 zero-shot baseline (9723.3ms, speedup_factor 0.58), and SQuAD's QLoRA inference (2465.5ms) beat baseline (speedup_factor 2.66) but was still ~4x slower than fp16 LoRA's post-EOS-fix 625.1ms. Plausible explanation: the T4 GPUs this project runs on (Turing architecture) lack efficient native int4/bf16 tensor-core paths that bitsandbytes' 4-bit compute kernels rely on, so the dequantize-on-the-fly compute path is doing real extra work per step that a newer architecture (Ampere+) wouldn't pay as heavily. **Report framing:** QLoRA is the right choice when VRAM is the binding constraint (e.g. it's what makes Llama-2-13B trainable at all on 16GB, unlike fp16 LoRA); it is not automatically the faster or cheaper choice in wall-clock/GPU-hour terms on older hardware, and this project's numbers are direct evidence of that trade-off, not an anomaly to explain away.

**`EXP-MIS-LORA-SQUAD`'s 10.49x inference speedup is not an inherent LoRA effect.** Latency dropped from baseline 6560.0ms to 625.1ms after the EOS-masking collator fix — but this is entirely because the fixed model now stops generating at EOS instead of running to `max_new_tokens` every single call (pre-fix latency was 9327.6ms, i.e. *slower* than baseline, because it always hit the generation cap). The `speedup_factor` field for this row measures "buggy-baseline-generation-length vs fixed-model-natural-stopping-length," not "LoRA-adapted inference vs base-model inference" holding generation behavior constant. **When writing up inference-latency comparisons, do not present this number as evidence that LoRA itself makes inference faster** — it doesn't, in general; the effect here is specific to fixing a training bug that had nothing to do with LoRA as a technique. Any real LoRA-vs-baseline latency comparison should control for generation length/stopping behavior on both sides.

---

## Recovery Procedures (unchanged from CLAUDE.md, restated here for convenience)

**CUDA OOM:** reduce batch size → increase gradient accumulation → reduce `max_seq_length` to 256 → last resort, skip and document.

**Model silently CPU-offloaded / hangs:** `device_map="auto"` can silently spill layers to CPU (or `meta`) when a model doesn't fully fit in free VRAM instead of raising an error — `.generate()` then keeps running, just slowly enough to burn an entire 12h Kaggle session without producing output or a visible failure. `experiments/common.py` checks parameter device placement immediately after model load (and after `apply_lora()` for training runs) and raises `RuntimeError` right away if anything landed off-GPU, instead of hanging silently — this guard is a safety net, not the fix, so keep it in place even after the actual fix below.

**Actual root cause and fix:** a Llama-2-13B baseline run loaded the model a second time (once per dataset) in the same process, and even with explicit `del model; gc.collect(); torch.cuda.empty_cache()` before the second load, it still failed — the first release wasn't literally leaking the model, it was CUDA allocator fragmentation from the first load/train/inference cycle leaving free VRAM too fragmented for a second ~13B-model-sized contiguous allocation. Mistral-7B survived a second in-process load (with enough margin), Llama-2-13B did not. The del/gc/empty_cache combo was necessary but not sufficient. **The real fix is architectural: never reload a model in the same process.** `experiments/common.py` now exposes `run_inference_multi_dataset()` and `run_training_multi_dataset()`, which load the model **once**, loop all of that technique's datasets **in-memory** against the same model instance, and release **once** at the end — every `experiments/mistral/*.py` and `experiments/llama/*.py` script (except the ONNX scripts, which already followed this pattern) calls one of these instead of looping the old per-dataset functions directly. For training (LoRA/QLoRA), where each dataset needs its own independently-trained adapter, `run_training_multi_dataset()` applies a **fresh** LoRA adapter per dataset via `apply_lora()` and strips it back to the clean base model via `PeftModel.unload()` between datasets — the base weights are loaded once, but no dataset's adapter ever starts from another dataset's trained weights. If the device-placement guard still fires after all of this: reduce batch size / `max_seq_length`, or fall back to splitting model loads across separate script invocations (one `!python` cell per dataset). `generate_predictions()` also prints progress every 20 samples so a stuck run is visible in the Kaggle log instead of going silent for hours.

**NaN loss:** halve learning rate → increase grad clipping (`max_grad_norm=0.5`) → retry once → if still NaN, check data and skip.

**Kaggle session timeout (12h cap):** checkpoint every 200 steps; on restart, resume from last checkpoint rather than restarting the experiment.

**Weekly quota (30h) about to be exceeded:** stop, do not silently shrink the dataset or skip epochs — that invalidates comparability. Log it as a blocker in `PROJECT_STATE.md` and decide explicitly whether to wait for quota reset or move an experiment to next week.

**`ImportError` mentioning torchao during `get_peft_model()` / LoRA apply:** Kaggle's base image ships `torchao 0.10.0`, but `peft`'s LoRA dispatcher (`get_peft_model` → `dispatch_torchao`) raises `ImportError` unless `torchao>=0.16.0` is installed — even though this project's QLoRA is bitsandbytes-based and never imports torchao directly. Fix: `!pip install -U torchao` before running any LoRA/QLoRA script (see `README.md` Kaggle setup).

**EM (or other strict-match quality metric) collapses to or near 0 while F1/ROUGE stay nonzero, after LoRA/QLoRA training:** Root cause: `load_model_and_tokenizer()` sets `tokenizer.pad_token = tokenizer.eos_token` whenever a model's tokenizer ships no distinct pad token (true for this project's models). Training previously used `transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)`, whose default `torch_call()` masks any label token equal to `pad_token_id`'s *value* to `-100` — unconditionally, not by actual padding position. Since `pad_token_id == eos_token_id` here, that also masked out every genuine end-of-sequence token appended to every training target (`tokenize_fn()`'s `f"{prompt} {target}{tokenizer.eos_token}"`), not just real padding — the model never received gradient signal for when to stop generating. This affects every technique that trains (LoRA, QLoRA) on every dataset, for both models; strict exact-match metrics (SQuAD's EM) expose it because any trailing generated tokens after the true answer zero out the match, while partial-overlap metrics (ROUGE, F1) mostly don't. First observed on `EXP-MIS-LORA-SQUAD` (EM 0.0/200 vs. baseline 8.0; `EXP-MIS-LORA-CNN`, trained under the same buggy collator, is also suspect even though its ROUGE happened to improve). Fix: mask labels by actual padding position (`attention_mask == 0`), not by `pad_token_id` value — `experiments/common.py`'s `_CausalLMCollator` replaces `DataCollatorForLanguageModeling` in `run_training_experiment()`. Verified offline with a fake-tensor smoke test (no GPU needed): a token whose value equals `pad_token_id` but whose `attention_mask` is `1` is preserved in `labels`; only true `attention_mask == 0` padding positions become `-100`. `save_debug_predictions()` now also dumps the first 5 generated predictions + references per experiment to `logs/debug_predictions/<exp_id>.txt`, so a result like this can be spot-checked directly instead of guessing.

**"None of the inputs have requires_grad=True" / `RuntimeError` on backward during LoRA or QLoRA training:** With only adapter params trainable (`get_peft_model()` freezes the base model) and gradient checkpointing on, the frozen base model's input embeddings output `requires_grad=False`, so checkpointing has no tensor to build a backward graph from — the graph never reaches the trainable adapter weights, and the first `trainer.train()` backward call fails. Fix: call `model.enable_input_require_grads()` immediately after `get_peft_model()` returns. Applied inside `apply_lora()` in `experiments/common.py`, so it covers every technique that calls it (LoRA, QLoRA — Mistral and Llama both) uniformly, rather than being set per-script.
