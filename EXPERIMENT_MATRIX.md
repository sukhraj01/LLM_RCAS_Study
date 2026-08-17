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
- **Status (2026-08-16): both models' ONNX export are deferred pending Ada cluster access — infeasible on Kaggle's free tier as currently understood.** Mistral-7B: 4 failed attempts (VRAM OOM, disk OOM, failed mount redirect, verified-but-still-failing in-memory export). Llama-2-13B: never attempted, blocked by the same disk math plus a separate VRAM ceiling (13B fp16 weights exceed 16GB). See Recovery Procedures below for the full history of both.

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

### Project-Level Finding: T4/Turing Quantization Slowdown — CONFIRMED

**On this project's T4 GPU hardware, every bitsandbytes-based quantization technique tested (8-bit, 4-bit/QLoRA) trades substantial VRAM reduction for substantially slower inference and training compute — the opposite of the commonly assumed "quantization = faster" framing.** Plausible cause (a hypothesis consistent with the observed pattern, **not a profiled or kernel-level-verified root cause** — no `nsight-compute`/PyTorch-profiler CUDA-kernel breakdown was run to directly confirm tensor-core utilization or dequantization overhead; see Limitations below): T4's Turing architecture lacks efficient native int8/int4 tensor-core paths that bitsandbytes relies on, so its dequantize-on-the-fly kernels plausibly do real extra work per step/token that a newer architecture (Ampere+) wouldn't pay as heavily. Four consistent data points, all Mistral-7B — the *pattern* itself is **confirmed, no longer pending** (this refers to the four measurements agreeing with each other, not to the architectural explanation being independently verified):

- **QLoRA training** (`EXP-MIS-QLORA-CNN`/`SQUAD`): ~73% less peak VRAM than fp16 LoRA (1.84-1.88GB vs 6.92-6.93GB), but ~14x slower per training step (183.97s/step vs 12.83s/step).
- **QLoRA inference** (same two rows): CNN inference 16658.0ms, *slower than the fp16 zero-shot baseline itself* (9723.3ms, speedup_factor 0.58); SQuAD inference 2465.5ms beat baseline (speedup_factor 2.66) but was still ~4x slower than fp16 LoRA's 625.1ms.
- **8-bit inference, no fine-tuning** (`EXP-MIS-8BIT-CNN`/`SQUAD`, 2026-08-15): peak VRAM 3.25GB vs baseline 6.91GB (-53%), but inference latency 25594.7ms (CNN) and 19562.4ms (SQuAD) vs baseline 9723.3ms/6560.0ms — 2.6x and ~3x *slower*, respectively (speedup_factor 0.38 / 0.34).
- **4-bit inference, no fine-tuning** (`EXP-MIS-4BIT-CNN`/`SQUAD`, 2026-08-15): peak VRAM 1.81GB vs baseline 6.91GB (-74%, the largest VRAM saving of any technique tested), but inference latency 14477.5ms (CNN) and 11292.7ms (SQuAD) vs baseline 9723.3ms/6560.0ms — 1.5x and 1.7x *slower* (speedup_factor 0.66 / 0.58). This was the explicitly-flagged confirming/disconfirming data point for the pattern above, and it confirms it: lower VRAM, still slower than baseline, consistent with the other three.

**Report framing:** four independent measurements (one training, three inference) across two quantization bit-widths all point the same direction on the same hardware — this is a confirmed hardware-specific finding, not an isolated anomaly per technique. **VRAM-savings consistency check:** `utils/validation.py`'s `check_quant_vram_relationship()` confirms 4-bit VRAM < 8-bit VRAM as expected (1.81GB < 3.25GB, no issues flagged) — the VRAM side of the quantization story behaves exactly as the literature predicts; it's specifically the *compute speed* side that inverts on T4/Turing.

**Relevance — why this matters beyond this project's own hardware:** T4 is not an obscure or arbitrary choice of GPU — it's a widely-deployed, real-world resource-constrained accelerator: AWS's G4 instance family, Google Cloud's T4 offering, and both Kaggle's and Colab's free tiers all run on T4. That means this finding is directly applicable to an actual population of resource-constrained deployments choosing between fp16 and quantized inference on affordable/free-tier hardware today, not just an artifact of whatever GPU happened to be free for this project. Anyone deploying on T4-class hardware and assuming "quantize for VRAM headroom, get a speed win too" should expect this project's data to plausibly generalize to their setup — anyone deploying on newer hardware (Ampere+, which does have efficient native int8/int4 tensor-core paths) should not assume the same trade-off holds; see ADR-005 and this document's Limitations section on scoping this finding to T4/Turing specifically.

**Context — this refines, rather than contradicts, actual ML-systems understanding:** the popular assumption this finding pushes back on is the oversimplified one — "quantization = always faster" — not the nuanced technical consensus among practitioners. `bitsandbytes`' own documentation already notes that its quantization speed benefits depend on the GPU having proper tensor-core support for low-precision ops; T4/Turing lacking that support is a known category of hardware limitation, not a novel discovery. What this project's four data points add is a *concrete, measured, four-technique-consistent quantitative confirmation* of that known caveat on a specific, widely-deployed piece of hardware — turning "quantization speed benefits are hardware-dependent" from a documented caveat into an actual measured number (1.5x-3x slower, not just "may vary") for anyone deciding whether to quantize on T4-class hardware specifically.

**Secondary finding: 8-bit vs 4-bit quality trade-off differs by task type, not uniformly worse.** 4-bit's more aggressive precision loss does not degrade every task equally:
- **CNN/summarization:** ROUGE1 held up slightly *better* under 4-bit (+8.15%, 0.2581 vs baseline 0.2387) than under 8-bit (+3.94%, 0.2481) — both are within noise of baseline, but 4-bit did not clearly do worse despite being the more aggressive quantization.
- **SQuAD/QA:** F1 degraded far more under 4-bit (-45.6%, F1 13.16, EM 2.0) than under 8-bit (-12.71%, F1 21.12, EM 5.5) — a large, non-noise-level quality hit.

Plausible explanation: ROUGE is a partial-overlap metric tolerant of small wording/precision-driven variation in a generated summary, while SQuAD's EM/F1 penalize short, exact answers much more harshly for the same degree of precision loss — a single wrong or missing token in a short extracted answer costs proportionally more than it does in a multi-sentence summary. **Report framing:** don't describe 4-bit as "uniformly worse than 8-bit" — it's task-dependent, and exact/short-answer tasks are where aggressive quantization's quality cost actually shows up on this project's data.

---

**`EXP-MIS-LORA-CNN`: ROUGE improvement is real but not "clean" summarization quality.** LoRA fine-tuning raised ROUGE-1/2/L by +21.1% over baseline (0.2890/0.1074/0.1965 vs 0.2387/0.0840/0.1607), but the raw predictions in `logs/debug_predictions/EXP-MIS-LORA-CNN.txt` show repetition loops (e.g. "I'm not going to say anything about the last few weeks" repeated verbatim 3x in example 0) and a rambling, run-on continuation style, rather than the terse bullet-point style CNN/DailyMail references use. This is consistent with a base (non-instruction-tuned) Mistral-7B model, LoRA r=8, only 1000 training examples, 2 epochs — a genuine quality limitation of this specific setup, not a pipeline defect. **Do not cite the +21.1% ROUGE gain in the report without this caveat** — ROUGE rewards n-gram overlap and is measurably fooled by verbose over-generation here.

**QLoRA's VRAM savings on Mistral-7B come at a real compute cost on this hardware — not a bug, a legitimate trade-off.** `EXP-MIS-QLORA-CNN`/`EXP-MIS-QLORA-SQUAD` used ~73% less peak VRAM than fp16 LoRA (1.84-1.88GB vs 6.92-6.93GB), exactly as expected for 4-bit quantized base weights. But training was ~14x slower *per step* (183.97s/step vs LoRA's 12.83s/step), and inference was slower too — CNN's QLoRA inference (16658.0ms) was even slower than the fp16 zero-shot baseline (9723.3ms, speedup_factor 0.58), and SQuAD's QLoRA inference (2465.5ms) beat baseline (speedup_factor 2.66) but was still ~4x slower than fp16 LoRA's post-EOS-fix 625.1ms. Plausible explanation: the T4 GPUs this project runs on (Turing architecture) lack efficient native int4/bf16 tensor-core paths that bitsandbytes' 4-bit compute kernels rely on, so the dequantize-on-the-fly compute path is doing real extra work per step that a newer architecture (Ampere+) wouldn't pay as heavily. **Report framing:** QLoRA is the right choice when VRAM is the binding constraint (e.g. it's what makes Llama-2-13B trainable at all on 16GB, unlike fp16 LoRA); it is not automatically the faster or cheaper choice in wall-clock/GPU-hour terms on older hardware, and this project's numbers are direct evidence of that trade-off, not an anomaly to explain away.

**`EXP-MIS-LORA-SQUAD`'s 10.49x inference speedup is not an inherent LoRA effect.** Latency dropped from baseline 6560.0ms to 625.1ms after the EOS-masking collator fix — but this is entirely because the fixed model now stops generating at EOS instead of running to `max_new_tokens` every single call (pre-fix latency was 9327.6ms, i.e. *slower* than baseline, because it always hit the generation cap). The `speedup_factor` field for this row measures "buggy-baseline-generation-length vs fixed-model-natural-stopping-length," not "LoRA-adapted inference vs base-model inference" holding generation behavior constant. **When writing up inference-latency comparisons, do not present this number as evidence that LoRA itself makes inference faster** — it doesn't, in general; the effect here is specific to fixing a training bug that had nothing to do with LoRA as a technique. Any real LoRA-vs-baseline latency comparison should control for generation length/stopping behavior on both sides.

---

## Limitations

Stated plainly so the report doesn't imply more rigor or generality than this project's design actually supports. These are scope boundaries, not defects — each is a reasonable trade-off given the GPU-hour budget (see `PROJECT_STATE.md` GPU Budget Tracking), but a reasonable trade-off is still a real limit on what the results can claim.

**1. Single hardware target (T4).** Every experiment — all 12 completed so far and everything still pending — has run exclusively on Kaggle's T4 GPU (see `logs/experiment_tracking.csv`'s `hardware` column, backfilled `T4` for every completed row). No experiment in this project has run on a different GPU architecture. This is a deliberate, ADR-documented choice (see `ARCHITECTURE.md` ADR-005), and T4 itself is a real, widely-deployed resource-constrained GPU (AWS G4, Google Cloud, Kaggle/Colab free tier) rather than an arbitrary pick — but that doesn't change the validity boundary: **every finding in this project, especially the T4/Turing quantization-slowdown finding above, is scoped to T4/Turing specifically and must not be generalized to "GPUs" or "quantization on any hardware" without saying so explicitly.** Ampere+ GPUs have proper tensor-core support for low-precision ops and would plausibly show a different (likely more favorable) speed trade-off for quantization — this project has no data on that, only a documented expectation of what direction it would move.

**2. Single run per experiment — no repeated trials, no seed variation.** Every one of the 22 experiments (12 completed, 10 pending/deferred) is a single run at a single seed. Every reported number — latency, VRAM, ROUGE, EM/F1, training time — is a **point estimate**, not a mean over repeated trials, and this project computes no confidence intervals, standard deviations, or statistical significance tests anywhere. This is a reasonable choice given the GPU-hour budget (30h/week cap makes even 2-3x reruns of the slower techniques, like QLoRA's 6-9 hour training runs, expensive), but it means: a reported "+16.3% quality" or "2.6x slower" number could, in principle, partly reflect run-to-run noise (sampling variance in the 200-example test set, minor nondeterminism in generation/kernel execution) rather than a stable, reproducible effect. Where multiple independent measurements happen to agree (e.g. the four-technique T4/Turing quantization-slowdown pattern, all pointing the same direction), that cross-technique consistency is doing the work statistical replication would normally do — but it is not a substitute for it, and single-technique, single-number results (most of the LoRA/QLoRA quality deltas) should be read as "this is what happened in this one run," not "this is the expected/average effect."

**3. Small scale relative to production fine-tuning.** Every training experiment (LoRA, QLoRA) uses 1000 training examples and 2 epochs; every quality/latency measurement uses 200 test samples. These sample sizes are intentionally small — see "Datasets" above — chosen to keep every experiment fast and rerunnable on a 30 GPU-hr/week budget, not chosen to represent realistic fine-tuning scale. Conclusions about technique behavior (e.g. "QLoRA trains ~14x slower per step than fp16 LoRA on T4") are specific to this scale and hyperparameter regime and may not transfer directly to production-scale fine-tuning — larger datasets (10k-1M+ examples), more epochs, larger effective batch sizes, and longer training runs could change the relative cost/benefit picture (e.g. QLoRA's per-step slowdown is a fixed multiplicative cost that matters more or less depending on how many steps you're willing to run, and larger datasets could shift where the VRAM-vs-wall-clock trade-off actually lands).

**4. The T4/Turing quantization-slowdown explanation is a hypothesis, not a profiled root cause.** The "Project-Level Finding" above is careful to separate two different claims that shouldn't be conflated: (a) **the observed pattern is confirmed** — four independent measurements (QLoRA training, QLoRA inference, 8-bit inference, 4-bit inference), all on Mistral-7B, all showing large VRAM reduction paired with slower compute, is a real, consistent, measured result; (b) **the architectural explanation for *why*** (T4/Turing lacking efficient native int8/int4 tensor-core paths that bitsandbytes' kernels rely on) **is a plausible hypothesis consistent with that pattern, not something this project verified directly.** No kernel-level profiling was performed — no `nsight-compute` run, no PyTorch profiler CUDA-kernel-level breakdown, nothing that would directly confirm tensor-core utilization or isolate dequantization overhead as the specific bottleneck. The hypothesis is well-grounded (it matches `bitsandbytes`' own documented caveat that speed benefits depend on tensor-core support — see the Relevance/Context discussion above) and is the most parsimonious explanation for the pattern, but an alternative or contributing cause (driver/kernel version quirks, a specific bitsandbytes version's implementation on this hardware, something dataset/prompt-length-specific) cannot be ruled out without direct profiling. **Report language should say "plausibly because" throughout, not "because."**

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

**Kaggle "Your notebook tried to allocate more memory than is available" during ONNX export specifically (not during normal inference), confirmed via nvidia-smi/Kaggle's crash banner, not a Python traceback:** Hit on `EXP-MIS-ONNX-CNN`/`EXP-MIS-ONNX-SQUAD` (2026-08-15), a genuine VRAM ceiling hit, not a script bug. Two compounding root causes, both missed by `EXPERIMENT_MATRIX.md`'s VRAM math because that math only ever projected *steady-state inference* memory, never the export step itself:
1. **Export-time trace peak:** the original `experiments/mistral/06_onnx.py` / `experiments/llama/05_onnx.py` called `ORTModelForCausalLM.from_pretrained(model_id, export=True, provider="CUDAExecutionProvider", ...)` — this places the model for tracing on whatever device the export machinery defaults to, and combines that with immediately standing up a CUDA `InferenceSession`. Tracing needs the full model resident **and** the ONNX graph being built simultaneously, which peaks above steady-state inference VRAM.
2. **fp32 export default (the bigger effect, verified by reading `optimum.exporters.onnx.main_export`'s source in this project's pinned version):** `main_export`'s `dtype` parameter defaults to `"fp32"` when not explicitly set, and the old code never set it. A fp32 ONNX export of a 7B model is ~28GB of weights — 2x the fp16 baseline's ~14GB that comfortably fits 16GB — so loading that graph onto a 16GB GPU via `CUDAExecutionProvider` OOMs on its own, independent of where tracing happened.

**Fix, applied in both `experiments/mistral/06_onnx.py` and `experiments/llama/05_onnx.py`:** call `optimum.exporters.onnx.main_export()` directly with `device="cpu", dtype="fp16"` to perform the export/trace step entirely off-GPU and at the same precision as the fp16 baseline, then load the resulting graph from `onnx_dir` via `ORTModelForCausalLM.from_pretrained(onnx_dir, provider="CUDAExecutionProvider")` for the actual benchmarked inference loop (this is what needs to run on GPU for realistic latency numbers). The "first run only, reuses `onnx_dir` after that" caching behavior — previously just a comment with no actual check — is now a real `os.path.isdir(onnx_dir) and any(...endswith(".onnx"))` guard, so reruns skip re-exporting. Verified offline only (`py_compile`, full module import, `inspect.signature` check that `main_export` accepts `device`/`dtype`/`task`/`token`) — **not yet run on a GPU**, since this fix hasn't had a Kaggle session yet. Confirm on real hardware before trusting the VRAM numbers.

**Open risk — deferred pending Ada cluster access, not scheduled on Kaggle:** Llama-2-13B's fp16 weights alone are ~26GB, already over Kaggle's 16GB VRAM before any export overhead — so `provider="CUDAExecutionProvider"` inference-loading step in the fix above will not fit as-is regardless of the CPU-export fix; CPU export may be the *only* viable path for Llama's ONNX experiment, not just an optimization, and even then it's unconfirmed whether Kaggle's CPU RAM can hold a ~26GB fp16 (or larger, if dtype ends up needing to stay fp32 for CPU-execution-provider inference) model for tracing. This has not been load-tested, and given Mistral-7B's ONNX export needed four attempts before being deferred as infeasible on this hardware tier (see below), Llama's larger/harder version of the same problem is treated the same way rather than attempted at all on Kaggle. See `PROJECT_STATE.md` Blockers & Risks for the corresponding open-risk entry.

**Kaggle "Your notebook tried to use more disk space than is available" / crash page shows "Output: ~20.94 GB" during ONNX export, on a genuinely fresh Kaggle session (not accumulated multi-session history):** Hit on `EXP-MIS-ONNX-CNN` (2026-08-15), the second consecutive ONNX failure — the first was the VRAM OOM above, this is a *disk* OOM after that fix let the export actually start running. Symptom looks like a raw-disk-space problem but isn't: `df -h` on the Kaggle instance showed ~1.1TB free at the underlying overlay filesystem level, which is irrelevant — Kaggle separately enforces a fixed **~20GB quota on the notebook's tracked *output*** (the default working directory the repo gets cloned into and `CHECKPOINTS_DIR` resolves under), independent of the raw disk. Root cause: the fp16 HF model cache (~14.5GB for Mistral-7B, under `~/.cache/huggingface` / wherever `HF_HOME` points) plus the ONNX export's own output (~14GB complete, ~6GB when it died) both land inside that same ~20GB-quota'd output path by default — together they exceed it regardless of whether the session has any prior history. `df -h` also showed `/kaggle/lib` (20GB, nearly empty) and `/opt/bin` (122GB, ~119GB free) as separate mounts, which are not the notebook's default output path and are presumed (not yet 100%-confirmed) to sit outside the same output-quota tracking.

**Fix attempt 1 (superseded — dead end):** redirect both the HF model cache and the ONNX export output off the default output path via `%env HF_HOME=/opt/bin/hf_cache` and `%env ONNX_CACHE_DIR=/opt/bin/onnx_cache`. **This did not work** — the very next Kaggle attempt failed with `OSError: [Errno 30] Read-only file system` on `/opt/bin`. That mount has free space but isn't writable on Kaggle; the README no longer recommends it. `ONNX_CACHE_DIR` as a mechanism (env var, falls back to the old `CHECKPOINTS_DIR`-based path when unset) was kept in the code anyway since it's harmless and still useful if a genuinely writable, larger mount is ever found — it just isn't pointed at `/opt/bin` by default anymore.

**Fix attempt 2 (the actual fix — third attempt at this experiment overall):** the real constraint is that `main_export()`/`ORTModelForCausalLM.from_pretrained(export=True)` both take a model *path/id* and load it internally, which needs the full fp16 HF cache (~14.5GB) **and** the growing ONNX output (~14GB) on disk *at the same time* — no alternate-mount redirect changes that math once nothing else turned out to be writable. Read `optimum.exporters.onnx.convert`'s source (same approach that found the fp32-default root cause previously) and found `onnx_export_from_model()` — a lower-level function `main_export()` itself calls into, that takes an **already-loaded PyTorch model object** instead of a path/id (its own docstring shows exactly this: `model = AutoModelForCausalLM.from_pretrained(...); onnx_export_from_model(model, output=...)`). Rewrote both ONNX scripts around this:
1. Load the model ourselves: `AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, token=HF_TOKEN)` (populates the HF cache on disk as a side effect, same as before).
2. Once loaded, free the on-disk cache via `huggingface_hub.scan_cache_dir().delete_revisions(*commit_hashes).execute()` — the official cache-management API, which removes the actual blob files, not just symlinks.
3. Export via `onnx_export_from_model(model=model, output=onnx_dir, task="text-generation-with-past", device="cpu")` against the in-memory object — disk now only ever holds the cache *or* the growing ONNX output, never both.

Safety of step 2 (deleting the cache while the model is still in use) was **verified by reading transformers' actual loading code, not assumed**: `transformers/modeling_utils.py`'s `load_state_dict()` loads each safetensors shard via `with safe_open(checkpoint_file, framework="pt") as f: ... f.get_tensor(k)`, which copies each tensor into a fresh torch-owned buffer and closes the file/mmap before the `with` block exits — no lingering file handle survives past `from_pretrained()` returning, so deleting the cache afterward is genuinely safe at the OS level, not just theoretically fine.

**Verified end-to-end this session (not just offline import-checking, unlike the previous two fix attempts)** — ran the complete pipeline (load fp16 → free cache → confirm the model still runs a forward pass post-deletion → export via `onnx_export_from_model` → reload the exported graph via `ORTModelForCausalLM.from_pretrained(onnx_dir, provider=...)` → `generate()`) against `hf-internal-testing/tiny-random-MistralForCausalLM`, the same architecture family as this project's actual model (RMSNorm-based, matching Mistral-7B). **Passed completely**, including the cache-deletion safety check. One caveat found along the way and worth recording: the same pipeline run against a tiny GPT2 model (LayerNorm-based, different architecture) failed at ONNX Runtime's internal graph-validation step with a `LayerNormalization` fp16/fp32 dtype mismatch — a known fp16-ONNX-export quirk. It did **not** reproduce on the Mistral-architecture test, consistent with Mistral using RMSNorm (no `LayerNormalization` op in its graph at all), so this specific failure mode is not expected to hit the real experiment — but this was confirmed on a tiny stand-in, not the actual 7B model, so scale-specific issues (a much larger graph, longer trace time) remain unverified until a real Kaggle run.

Third ONNX attempt on Mistral now in place (1st: VRAM OOM from fp32 default, fixed; 2nd: disk OOM from the cache+output combination, `/opt/bin` redirect attempted and failed; 3rd/current: in-memory export avoiding cache+output disk overlap entirely, verified end-to-end against a same-architecture stand-in model but not yet against the real 7B model on Kaggle).

**Llama-2-13B note:** this fix does *not* resolve Llama's disk problem the way it does Mistral's. Even with cache and ONNX output never held simultaneously, the *peak* usage during either single phase is still ~26GB (fp16 13B weights) — which alone exceeds the ~20GB output quota, independent of the already-known VRAM blocker for the final `CUDAExecutionProvider` inference-reload step. See the Open Risk entry above and `PROJECT_STATE.md` Blockers & Risks — not scheduled.

**FINAL OUTCOME (2026-08-16): Mistral-7B ONNX export marked infeasible on Kaggle's free tier — deferred pending Ada cluster access, not scheduled for further attempts there.** The in-memory-export fix (attempt 3 above), despite being verified end-to-end against a same-architecture stand-in model, **still hit a disk OOM on the real Kaggle run against the actual 7B model** — a fourth consecutive failure. This means the stand-in verification, while a genuinely stronger check than the first two fix attempts got, did not fully predict real-world behavior at 7B scale — plausible gaps include ONNX's external-data serialization for graphs over the 2GB single-file protobuf limit (not exercised by a tiny model, which stays under that limit entirely), or some other scale-dependent transient disk usage during export that a KB-scale stand-in can't surface. This was not re-diagnosed further — after four attempts (VRAM OOM → disk OOM → failed `/opt/bin` redirect → verified-but-still-failing in-memory export), the engineer made the call to stop spending Kaggle GPU hours on repeated attempts rather than pursue a fifth fix. `EXP-MIS-ONNX-CNN`/`EXP-MIS-ONNX-SQUAD` are deferred pending access to a different GPU (university Ada cluster), given the same treatment as Llama-2-13B's ONNX open risk above — not attempted again on Kaggle until then. See `PROJECT_STATE.md` Blockers & Risks and Component Status Summary for the corresponding update.
