# Daily Standup Log

One entry per working session (Kaggle or local). Append, don't overwrite. Copy the template below for each new entry.

---

## Template

### YYYY-MM-DD — [Kaggle / Local]

**Planned:** what you intended to run/do this session

**Completed:** what actually finished, with real numbers (GPU hours, exp IDs)

**Issues:** anything that failed or needed recovery, and what you did about it

**GPU hours used this session:** X.X

**Next session:** what's queued up next (should match `PROJECT_STATE.md` "What's Next")

---

### 2026-08-13 — Local

**Planned:** Verify local dev environment end-to-end before pushing to GitHub / starting Kaggle: venv + deps, HF auth, data pipeline sanity check, syntax/import-check on all experiment scripts.

**Completed:**
- `.venv` created (Python 3.11.14, chosen over system Python 3.14 for better ML-package wheel support)
- `pip install -r requirements.txt` succeeds (after fixing a real dependency conflict — see Issues)
- HF_TOKEN verified present in `.env` (non-placeholder, 37-char token)
- `python -m utils.data_loader` sanity check passed: CNN/DailyMail val=200/test=200, SQuAD val=200/test=200, no overlap by construction, both datasets
- `py_compile` passed on all 11 experiment scripts (`experiments/mistral/*.py`, `experiments/llama/*.py`) plus both `__init__.py` files
- `experiments.common`, `utils.config`, `utils.data_loader`, `utils.metrics`, `utils.validation` all import cleanly in the venv

**Issues:** `pip install -r requirements.txt` initially failed with a dependency-resolution conflict (`optimum[onnxruntime-gpu]` has no macOS wheel at all, forcing pip to backtrack through every `optimum` version and eventually collide with the `transformers>=4.40.0` pin). Fixed by switching the local requirement to `optimum[onnxruntime]` (CPU) and documenting a separate `pip install -U "optimum[onnxruntime-gpu]"` step for Kaggle sessions in `README.md`. Full detail logged in `knowledge/ai-usage-log/2026-08-13_env-verification.md`.

**GPU hours used this session:** 0.0 (no GPU on this machine — local-only checks per `CLAUDE.md` / `KAGGLE_SYNC.md`)

**Next session:** Push repo to GitHub (`sukhraj01/LLM_RCAS_Study`), start first Kaggle session per `README.md` Kaggle setup + `KAGGLE_SYNC.md` — run the extra `optimum[onnxruntime-gpu]` install step there, then begin Week 1 baseline experiments (4 runs: 2 models × 2 datasets).

---

### 2026-08-13 — Kaggle

**Planned:** Week 1 baseline experiments — 4 runs (Mistral-7B × CNN/SQuAD, Llama-2-13B × CNN/SQuAD), all in one Kaggle session.

**Completed:** 3 of 4 baselines finished — EXP-MIS-BASE-CNN, EXP-MIS-BASE-SQUAD, EXP-LLAMA-BASE-CNN. Real numbers merged into `logs/experiment_tracking.csv` and `results/mis_results.csv`/`results/llama_results.csv` from the engineer's downloaded CSVs.

**Issues:** EXP-LLAMA-BASE-SQUAD hung for 5+ hours and was killed by the 12h Kaggle session cap instead of completing or erroring. Root cause: `experiments/common.py`'s `run_inference_only_experiment()` never released the model between successive `load_model_and_tokenizer()` calls in the same process (once per dataset). Loading Llama-2-13B a second time without freeing the first instance left enough VRAM occupied that `device_map="auto"` fell back to CPU-offloading some layers, and `.generate()` essentially never finished — no error, no crash, just silence.

While merging the real numbers, two more issues surfaced (not requested, found by actually trying to load the data): (a) `quality_metrics` for both CNN rows was unparseable — `evaluate`'s rouge output is numpy `float64`, and numpy>=2.0's repr change (`"np.float64(x)"`) broke the `ast.literal_eval()` round-trip in `load_baseline_metrics()`, which every Week 2+ technique script depends on; fixed in `utils/metrics.py` (`compute_rouge()` now casts to native `float`). (b) The results file is actually `mis_results.csv`, not `mistral_results.csv` as `ARCHITECTURE.md`'s file tree claimed (`save_result()` uses `model_key.lower()`, and Mistral's key is `"MIS"`) — caught by testing `load_baseline_metrics()` against the file I'd written and getting `None` back; fixed the filename and the doc. (c) `EXP-MIS-BASE-SQUAD`'s `peak_vram_gb` (13.65GB) is ~2x `EXP-MIS-BASE-CNN`'s (6.91GB) — almost certainly the same un-released-model bug, just non-fatal for a 7B model (two un-freed Mistral-7B instances still fit under 16GB, unlike two 13B ones). Flagged in both results CSVs and `logs/experiment_tracking.csv`, not silently corrected — recommend rerunning alongside EXP-LLAMA-BASE-SQUAD.

**Fix applied** (same session, local): `experiments/common.py` now (1) explicitly releases the model (`del model; gc.collect(); torch.cuda.empty_cache()`) at the end of both `run_inference_only_experiment()` and `run_training_experiment()`; (2) checks parameter device placement right after model load (and after `apply_lora()` for training) via a new `_assert_fully_on_gpu()` guard, raising `RuntimeError` immediately instead of silently continuing into a slow `.generate()` loop; (3) `generate_predictions()` now prints progress every 20 samples so a stuck run is visible in the Kaggle log in real time. Documented as a new recovery procedure in `EXPERIMENT_MATRIX.md` and `CLAUDE.md`. Full detail in `knowledge/ai-usage-log/2026-08-13_kaggle-baseline-hang.md`.

**GPU hours used this session:** 12.0 (full 12h session cap — planned 2.0 for this phase; see `PROJECT_STATE.md` GPU Budget Tracking, now recorded honestly as over budget)

**Next session:** Rerun EXP-LLAMA-BASE-SQUAD and EXP-MIS-BASE-SQUAD on Kaggle with the fixed `common.py`/`metrics.py` (should now either complete cleanly or fail loudly and fast instead of hanging/silently inflating VRAM). Then proceed to Week 2 (LoRA, Mistral-7B) once the baseline phase is actually clean.

---

### 2026-08-13 — Local (follow-up: real root cause for the CPU-offload hang)

**Planned:** The device-placement guard from the previous entry worked (fast `RuntimeError` instead of a silent hang), but engineer determined it was treating a symptom — every technique script reloads the model once per dataset in the same process, and Llama-2-13B's second in-process load reliably fails even after explicit `del`/`gc.collect()`/`empty_cache()`, most likely CUDA allocator fragmentation rather than literal non-release (confirmed: a fresh process per dataset always works; Mistral-7B survives a second in-process load, Llama-2-13B doesn't). Restructure so each process loads its model once and loops datasets in-memory instead.

**Completed:** `experiments/common.py` split into already-loaded-model runners (`run_inference_only_experiment`, `run_training_experiment`) plus new orchestrators `run_inference_multi_dataset()` / `run_training_multi_dataset()` that load once, assert placement once, loop all of a technique's datasets, and release once. All 9 dependent scripts (`experiments/mistral/01–05`, `experiments/llama/01,02,03,04`) updated to call the orchestrators instead of looping the old per-dataset functions. ONNX scripts (`06_onnx.py`/`05_onnx.py`) already followed this pattern — left untouched, just verified they still import cleanly.

For LoRA/QLoRA specifically: naively reusing one loaded model across datasets would let dataset 2's training start from dataset 1's already-trained adapter weights, silently contaminating the comparison. `run_training_multi_dataset()` handles this with `PeftModel.unload()` — applies a **fresh** `apply_lora()` per dataset, trains, saves, then `unload()`s the adapter back to the clean base model (in place, no reload from disk) before the next dataset.

Verified offline (no GPU available locally): `py_compile` + import-checks pass on `common.py` and all 9 scripts; wrote a monkeypatched logic test exercising both orchestrators with fake model objects, confirming exactly 1 `load_model_and_tokenizer` call, 1 `_assert_fully_on_gpu` call, 1 `_release_model` call per orchestrator invocation, correct `exp_id` generation (including the `"baseline"`→`"BASE"` naming exception), and — the critical property — both datasets' `apply_lora()` calls wrap the *same* base model object (proving no reload happens between them).

**Issues:** None new — this is the real fix for the previous entry's CPU-offload hang, not a new failure. Updated `EXPERIMENT_MATRIX.md`/`CLAUDE.md`'s recovery procedure to describe the actual root cause (allocator fragmentation, not literal leak) and the actual fix (load-once architecture, not just cleanup calls) — the guard and cleanup calls stay in place as a safety net but are documented as insufficient on their own for 13B models. Updated `logs/experiment_tracking.csv`'s `EXP-LLAMA-BASE-SQUAD` note accordingly.

**GPU hours used this session:** 0.0 (local refactor + offline logic verification only; no GPU available here)

**Next session:** Rerun `EXP-LLAMA-BASE-SQUAD` and `EXP-MIS-BASE-SQUAD` on Kaggle with the restructured scripts — this is the first real test of the load-once architecture against actual hardware. Then proceed to Week 2 (LoRA, Mistral-7B) once the baseline phase is genuinely clean.

---

### 2026-08-14 — Kaggle

**Planned:** Rerun the baseline scripts on the restructured (load-once) `experiments/common.py` — first real-hardware test of the fix from the previous entry.

**Completed:** Both `experiments/mistral/01_baseline.py` and `experiments/llama/01_baseline.py` completed cleanly, all 4 Week 1 baselines now done, including `EXP-LLAMA-BASE-SQUAD` — the one that hung for 5+ hours two sessions ago. **The load-once fix is confirmed working on real hardware.** Real numbers merged into `results/mis_results.csv`, `results/llama_results.csv`, and `logs/experiment_tracking.csv`. Nice independent confirmation the fix targeted the right thing: `EXP-MIS-BASE-SQUAD`'s `peak_vram_gb` is now 6.91GB (matching `EXP-MIS-BASE-CNN`), down from the previous session's contaminated 13.65GB — exactly the ~2x-inflation theory from two sessions ago, now resolved.

**Issues:** Because the baseline scripts run both datasets per script (no way to rerun only the previously-failed dataset), rerunning `llama/01_baseline.py` and `mistral/01_baseline.py` re-appended fresh rows for `EXP-MIS-BASE-CNN` and `EXP-LLAMA-BASE-CNN` — which had already succeeded and already had rows in the committed `results/*.csv` from before — on top of the old ones (`save_result()` is append-only by design). Not itself a bug (the two Mistral-CNN measurements are consistent — 9723.3ms vs. 9616.9ms latency, ROUGE1 0.2386 both times — a nice reproducibility check), but it surfaced a real latent bug: `load_baseline_metrics()` returned the *first* matching row when scanning the CSV, not the *last* — meaning any technique script that reads a baseline after a rerun would have silently gotten stale, possibly pre-fix numbers instead of the fresh ones. Fixed: `load_baseline_metrics()` now scans the whole file and keeps the last match. Verified against the actual duplicated rows in both results files — all 4 baseline lookups (`MIS`/CNN, `MIS`/SQUAD, `LLAMA`/CNN, `LLAMA`/SQUAD) now correctly return the most recent measurement.

One result worth flagging for the eventual report, not a pipeline issue: Llama-2-13B's SQuAD quality (EM 3.0, F1 13.19) is notably worse than Mistral-7B's (EM 8.0, F1 24.19) despite being the bigger model — plausibly because `meta-llama/Llama-2-13b-hf` is a non-instruction-tuned base model and the zero-shot QA prompt format used here favors instruction-following. Not investigated further; noted rather than ignored per `CLAUDE.md`'s "document unexpected results" guidance.

**GPU hours used this session:** ~2.5h (estimated from per-sample timing — Mistral CNN+SQuAD ~54min, Llama CNN+SQuAD ~95min; not exact Kaggle accounting). Combined with the 12h from the 2026-08-13 first attempt, the Setup+baseline phase total is **≥14.5h against a 2h plan** — a real 7x miss on the phase-level estimate, see `PROJECT_STATE.md` GPU Budget Tracking for the full breakdown. Not a quota-cap risk, though: engineer confirmed the 30h figure is a per-week quota refreshing every Sunday, not a pool shared across all 5 phases — Weeks 2-4 each get their own fresh 30h regardless of this overrun. **Still open:** the intermediate guard-fix-test Kaggle session (the one that produced the fast `RuntimeError` instead of a hang, which is what led to diagnosing the real root cause) has real but never-recorded GPU-hours — the ≥14.5h figure is a confirmed floor, not the final number.

**Next session:** Week 2 — LoRA fine-tuning on Mistral-7B (`experiments/mistral/02_lora.py`, now using `run_training_multi_dataset()`), both datasets. Verify config against `EXPERIMENT_MATRIX.md` technique #2 before running. Worth deciding explicitly whether the Setup+baseline overrun changes how Weeks 2-4 get scheduled against the weekly quota.

---

### 2026-08-14 — Kaggle (LoRA env fix)

**Planned:** Week 2 — LoRA fine-tuning on Mistral-7B (`experiments/mistral/02_lora.py`), both datasets.

**Completed:** N/A — session blocked before training started; see Issues.

**Issues:**

1. `get_peft_model()` raised `ImportError` mentioning `torchao` version requirements when applying the LoRA adapter. Root cause: Kaggle's base image ships `torchao 0.10.0`, but `peft`'s LoRA dispatcher (`dispatch_torchao`) requires `torchao>=0.16.0` to import cleanly, even though this project's QLoRA path is bitsandbytes-based and never imports torchao directly. Fix: `!pip install -U torchao` before running any LoRA/QLoRA script. Added as a required setup line in `README.md` Kaggle setup section and as a new entry in `EXPERIMENT_MATRIX.md` Recovery Procedures. Full detail in `knowledge/ai-usage-log/2026-08-14_torchao-lora-importerror.md`.

2. After the torchao fix, `apply_lora()` → `get_peft_model()` succeeded (3.4M trainable / 7.2B total params, 0.047% — looked right), training started, `use_cache` got disabled for gradient checkpointing as expected, then the very first `trainer.train()` backward call failed: `UserWarning: None of the inputs have requires_grad=True. Gradients will be None` / `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`. Root cause: standard PEFT + gradient-checkpointing gotcha — with only LoRA adapter params trainable and gradient checkpointing on, the frozen base model's input embeddings output `requires_grad=False`, so checkpointing has no tensor to build a backward graph from and the graph never reaches the trainable adapter weights. Fix: `model.enable_input_require_grads()` added right after `get_peft_model()` returns inside `apply_lora()` in `experiments/common.py`, so it applies uniformly to every technique that calls `apply_lora()` (LoRA, QLoRA — Mistral and Llama both). Verified offline (`py_compile` + import check; no GPU available locally) — not yet tested on real hardware. Added as a new entry in `EXPERIMENT_MATRIX.md` Recovery Procedures. Full detail in `knowledge/ai-usage-log/2026-08-14_gradient-checkpointing-requires-grad.md`.

With both fixes in place, `experiments/mistral/02_lora.py` completed cleanly on both datasets — LoRA training on Mistral-7B, real numbers from `results/mis_results.csv`:

- `EXP-MIS-LORA-CNN`: ROUGE1/2/L 0.2719/0.0987/0.1834 (up from baseline 0.2386/0.0840/0.1607, +13.9%), training_time_hrs 0.86, peak_vram_gb 6.92 — clean result, adapter saved to `checkpoints/mistral_lora_cnn`
- `EXP-MIS-LORA-SQUAD`: EM 0.0/F1 7.76 (down from baseline EM 8.0/F1 24.19, -67.9%), training_time_hrs 0.41, peak_vram_gb 6.93 — **flagged, not accepted as-is.** EM collapsing to exactly 0/200 is unusual enough to investigate rather than log as a normal result. Code review (no GPU, code-only) ruled out a train/eval prompt-format mismatch (`format_example()` is shared between `tokenize_fn()` and `generate_predictions()`, so the prompt is identical). Leading unconfirmed hypothesis (at the time): `tokenizer.pad_token = tokenizer.eos_token` collides with `DataCollatorForLanguageModeling(mlm=False)`'s unconditional `pad_token_id` → `-100` label masking, wiping out gradient signal for every real per-example EOS token, not just padding — plausibly explains why SQuAD's exact-match (zero tolerance for trailing tokens) collapsed while CNN's ROUGE (partial overlap) improved under the same setup. Logged as an open question in `PROJECT_STATE.md` Current Blockers and `logs/experiment_tracking.csv`, not as a resolved/explained result.

Both rows merged into `logs/experiment_tracking.csv`.

3. Engineer confirmed the EOS-masking direction and asked for a fix. Root cause fixed: `experiments/common.py`'s `run_training_experiment()` now uses a custom `_CausalLMCollator` (masks labels by `attention_mask == 0`, i.e. actual padding position) instead of `DataCollatorForLanguageModeling(mlm=False)` (which masked by `pad_token_id`'s *value*, wrongly catching every real per-example EOS token since `pad_token_id == eos_token_id`). This is a labels-construction fix only — no new pad token added, no embedding resize, per the engineer's explicit scope constraint. Also added `save_debug_predictions()`, which dumps the first 5 predictions + references per experiment to `logs/debug_predictions/<exp_id>.txt`, called from both the training and inference-only paths, so a result like this can be spot-checked directly next time instead of guessing from aggregate metrics alone. Verified offline: `py_compile` + import check pass, and a fake-tensor smoke test confirms a token whose value equals `pad_token_id` but whose `attention_mask` is `1` is preserved in `labels` (not masked), while real `attention_mask == 0` padding positions correctly become `-100`. **Not yet confirmed on real hardware** — this affects every technique that trains (LoRA, QLoRA), both datasets, both models, since it's the one shared collator code path; `EXP-MIS-LORA-CNN` and `EXP-MIS-LORA-SQUAD` were both trained under the buggy version and are both now flagged `NEEDS RERUN` in `logs/experiment_tracking.csv` and `PROJECT_STATE.md` Current Blockers — CNN's improved ROUGE does not clear it of the same defect, since ROUGE is far more tolerant of over-generation than SQuAD's exact-match is. Full detail in `knowledge/ai-usage-log/2026-08-14_lora-squad-em-collapse.md`.

**GPU hours used this session:** ~1.3h (training_time_hrs 0.86 + 0.41 from the two completed runs; does not include the earlier blocked attempts covered above, which failed before any step ran; the collator fix itself required no GPU time, code-only + offline verification)

**Next session:** Rerun `EXP-MIS-LORA-CNN` and `EXP-MIS-LORA-SQUAD` on Kaggle with the fixed `_CausalLMCollator`, and check the new `logs/debug_predictions/` output to confirm the pre-fix predictions actually were verbose/run-on as hypothesized. Only proceed to QLoRA (Week 3) once both LoRA results are re-confirmed clean under the fix.

---

### 2026-08-14 — Kaggle (LoRA rerun confirmed)

**Planned:** Rerun `EXP-MIS-LORA-CNN` and `EXP-MIS-LORA-SQUAD` on Kaggle under the fixed `_CausalLMCollator`, check `logs/debug_predictions/` to confirm the EOS-masking hypothesis on real hardware, and close out the `NEEDS RERUN` status from the previous session.

**Completed:** Both experiments reran cleanly. Final numbers, `results/mis_results.csv`:
- `EXP-MIS-LORA-CNN`: ROUGE1/2/L 0.2890/0.1074/0.1965 (baseline 0.2387/0.0840/0.1607, +21.1%), training_time_hrs 0.89, peak_vram_gb 6.92
- `EXP-MIS-LORA-SQUAD`: EM 85.5/F1 91.96 (baseline EM 8.0/F1 24.19, +280%), training_time_hrs 0.42, peak_vram_gb 6.93, inference_latency_ms 625.1 (down from 9327.6ms pre-fix)

Spot-checked `logs/debug_predictions/` (5 examples/dataset) rather than trusting the aggregate metrics alone: SQuAD predictions now stop cleanly at EOS and match references almost exactly — confirms the EOS-masking root cause directly on real hardware, not just via code review. CNN predictions, despite the ROUGE gain, still show repetition loops and a rambling continuation style rather than the terse bullet-point style of the references — a genuine base-model (non-instruction-tuned) + small-LoRA (r=8, 1000 examples, 2 epochs) quality limit, not a pipeline defect. Documented as a caveat in `EXPERIMENT_MATRIX.md` "Qualitative Notes for Report" so the ROUGE number isn't cited without context later.

Also documented explicitly: SQuAD's 10.49x `speedup_factor` is the EOS fix letting the model stop generating early, not an inherent LoRA-vs-baseline inference speedup — flagged in `EXPERIMENT_MATRIX.md` and `logs/experiment_tracking.csv` so it isn't misread as "LoRA makes inference 10x faster" during report writing.

Both `EXP-MIS-LORA-CNN` and `EXP-MIS-LORA-SQUAD` rows in `logs/experiment_tracking.csv` updated from `NEEDS RERUN` to `CONFIRMED FINAL`. `PROJECT_STATE.md` blocker marked resolved (confirmed on real hardware, not just code review); Week 2 checklist closed out; Week 2 quality gate passed (no OOM, peak VRAM well under 16GB, training time reasonable) — proceeding to Week 3 (QLoRA).

**Issues:** None this session — this was the confirmation run for the previous session's fix, and it worked as diagnosed.

**GPU hours used this session:** ~1.31h (training_time_hrs 0.89 + 0.42 from the two reruns). Combined with the ~1.27h spent on the earlier discarded buggy-collator run, total Week 2 LoRA phase GPU-hours is ~2.6h against a 4h plan — see `PROJECT_STATE.md` GPU Budget Tracking.

**Next session:** Week 3 — QLoRA fine-tuning, both models, both datasets. Verify config against `EXPERIMENT_MATRIX.md` technique #3 before running.

---

### 2026-08-15 — Kaggle (QLoRA, Mistral-7B)

**Planned:** Week 3 — QLoRA fine-tuning on Mistral-7B, both datasets (Llama-2-13B QLoRA not attempted this session).

**Completed:** Both experiments finished. Final numbers, `results/mis_results.csv`:
- `EXP-MIS-QLORA-CNN`: ROUGE1/2/L 0.2775/0.0975/0.1868 (baseline 0.2387/0.0840/0.1607, +16.3%), training_time_hrs 6.44, peak_vram_gb 1.84, inference_latency_ms 16658.0 (slower than baseline's own 9723.3ms, speedup_factor 0.58)
- `EXP-MIS-QLORA-SQUAD`: EM 83.0/F1 90.03 (baseline EM 8.0/F1 24.19, +272%), training_time_hrs 3.42, peak_vram_gb 1.88, inference_latency_ms 2465.5 (speedup_factor 2.66 vs baseline, but slower than fp16 LoRA's 625.1ms)

Notable, report-worthy finding: QLoRA trained ~14x slower per step than fp16 LoRA on this hardware (183.97s/step vs 12.83s/step) despite ~73% less peak VRAM (1.84-1.88GB vs 6.92-6.93GB). Inference was also slower under QLoRA than fp16 LoRA across both datasets, and CNN's QLoRA inference was even slower than the zero-shot baseline. Plausible explanation, documented in `EXPERIMENT_MATRIX.md` "Qualitative Notes for Report": T4 GPUs (Turing architecture) lack efficient native int4/bf16 tensor-core paths that bitsandbytes' 4-bit compute relies on, so the VRAM savings come at a real compute cost on this specific hardware — a legitimate, reportable trade-off, not a bug.

Both rows merged into `logs/experiment_tracking.csv` as `CONFIRMED FINAL`.

**Issues:** None with the experiments themselves — both completed cleanly, no OOM, no NaN. But a real GPU-budget issue surfaced while merging results: Mistral-only QLoRA alone used 9.86 GPU-hours (6.44 + 3.42) against `EXPERIMENT_MATRIX.md`'s Mistral-only estimate of 3.0h (1.5h/dataset) — already exceeding the entire QLoRA phase's planned 9h budget (both models combined) before Llama-2-13B's two QLoRA experiments (budgeted ~6h) have even started. Per `CLAUDE.md`'s "stop and escalate" rule on resource constraints, this is flagged as an open blocker in `PROJECT_STATE.md` rather than silently proceeding to schedule Llama's QLoRA runs — actual Kaggle account quota needs to be confirmed against this before anything else is scheduled this week.

Also note: `results/mis_results.csv` did not yet contain these two rows when this session's results were reported — added them using the numbers reported, flagged explicitly for the engineer to verify against the actual Kaggle CSV download. **Update, same day:** engineer provided the actual `results/mis_results.csv` content afterward; the two QLoRA rows have been corrected to the real full-precision `quality_metrics` and real adapter-save-path notes, superseding the rounded placeholders added earlier.

**GPU hours used this session:** 9.86h (training_time_hrs 6.44 + 3.42) plus ~1h combined generation/eval, ~11h wall-clock total. This already exceeds the entire QLoRA phase's 9h plan on its own — see `PROJECT_STATE.md` GPU Budget Tracking.

**Budget update, same day:** this ~11h session crossed a Kaggle weekly-quota reset boundary partway through training — the session was **not** interrupted, both experiments completed cleanly straight through it. This resolves the budget-confirmation hold raised above: quota is fresh (30h) as of today, 2026-08-15. Honest accounting (not smoothed over): the prior Kaggle quota-week's real usage was at least ≥17.1h (12h CPU-offload hang + ~2.5h clean Week 1 rerun + ~2.6h Week 2 LoRA) plus an unknown, unrecoverable fraction of this session's ~11h that ran before the reset — plausibly at or over the 30h cap once combined with the still-uncaptured intermediate guard-fix-test session from 2026-08-13. For forward planning, conservatively treat the new week as ~11h already used / ~19h remaining (worst case, since the exact pre/post-reset split is unknown). Full detail in `PROJECT_STATE.md` GPU Budget Tracking "Kaggle Weekly Quota Reset." Not something to rely on recurring — added as a new Identified Risks row.

**Next session:** Run `EXP-MIS-8BIT-CNN`/`EXP-MIS-8BIT-SQUAD` and `EXP-MIS-4BIT-CNN`/`EXP-MIS-4BIT-SQUAD` on Kaggle (Mistral-7B, inference-only, expected fast) — good use of the freshly-reset quota. Llama-2-13B QLoRA remains deferred, to be rescheduled with a corrected GPU-hour estimate given Mistral's 3.3x miss on this technique.

---

### 2026-08-15 — Kaggle (8-bit inference, Mistral-7B)

**Planned:** Week 4 — 8-bit quantized inference on Mistral-7B, both datasets (base pretrained model, no LoRA adapter — zero-shot under `load_in_8bit=True`).

**Completed:** Both experiments finished. Real numbers, `results/mis_results.csv`:
- `EXP-MIS-8BIT-CNN`: ROUGE1/2/L 0.2481/0.0850/0.1656 (baseline 0.2387/0.0840/0.1607, +3.9% — essentially flat/noise-level), peak_vram_gb 3.25 (baseline 6.91, -53%), inference_latency_ms 25594.7 (baseline 9723.3, speedup_factor 0.38 — 2.6x SLOWER)
- `EXP-MIS-8BIT-SQUAD`: EM 5.5/F1 21.12 (baseline EM 8.0/F1 24.19, -12.7%), peak_vram_gb 3.25, inference_latency_ms 19562.4 (baseline 6560.0, speedup_factor 0.34 — ~3x SLOWER)

Both rows merged into `logs/experiment_tracking.csv` as `CONFIRMED FINAL`, referencing `results/mis_results.csv`. No `training_time_hrs` (inference-only technique, correctly left blank).

Report-worthy: this is the third consecutive bitsandbytes quantization technique (after QLoRA training and QLoRA inference) showing the same pattern — large VRAM reduction, substantially slower compute on this T4 hardware. Elevated from a per-experiment note to a single consolidated project-level finding in `EXPERIMENT_MATRIX.md` Qualitative Notes ("Project-Level Finding: T4/Turing Quantization Slowdown"), citing all three data points and flagging `EXP-MIS-4BIT-CNN`/`SQUAD` (not yet run) as the next confirming-or-disconfirming measurement.

**Issues:** None with the experiments themselves — both completed cleanly, inference-only so no training-side failure modes apply. SQuAD's quality drop (-12.7% F1) is the first quantization-technique result (of the three so far) where quality also degraded rather than staying flat — noted in `EXPERIMENT_MATRIX.md`, not investigated further this session since 8-bit is a well-understood lossy quantization and a modest EM/F1 drop on a 200-example zero-shot QA set is plausible without a pipeline defect.

**GPU hours used this session:** Not separately tracked (inference-only, fast — expected well under 0.5h per `EXPERIMENT_MATRIX.md`'s 0.2h/dataset estimate); no `training_time_hrs` to sum since this technique has none.

**Next session:** Run `EXP-MIS-4BIT-CNN`/`EXP-MIS-4BIT-SQUAD` on Kaggle (Mistral-7B, 4-bit inference-only) — the key data point for confirming or disconfirming the T4/Turing quantization-slowdown finding. Then Mistral ONNX (Week 4 remainder) before returning to the deferred Llama-2-13B QLoRA question.

---

### 2026-08-15 — Kaggle (4-bit inference, Mistral-7B)

**Planned:** Week 4 — 4-bit quantized inference on Mistral-7B, both datasets (base pretrained model, no LoRA adapter — zero-shot under `load_in_4bit=True`, NF4). This was the explicitly-flagged confirming/disconfirming data point for the T4/Turing quantization-slowdown finding.

**Completed:** Both experiments finished. Real numbers, `results/mis_results.csv`:
- `EXP-MIS-4BIT-CNN`: ROUGE1/2/L 0.2581/0.0881/0.1714 (baseline 0.2387/0.0840/0.1607, +8.15%), peak_vram_gb 1.81 (baseline 6.91, -74%), inference_latency_ms 14477.5 (baseline 9723.3, speedup_factor 0.66 — 1.5x SLOWER)
- `EXP-MIS-4BIT-SQUAD`: EM 2.0/F1 13.16 (baseline EM 8.0/F1 24.19, -45.6%), peak_vram_gb 1.81, inference_latency_ms 11292.7 (baseline 6560.0, speedup_factor 0.58 — 1.7x SLOWER)

Both rows merged into `logs/experiment_tracking.csv` as `CONFIRMED FINAL`, referencing `results/mis_results.csv`. No `training_time_hrs` (inference-only technique, correctly left blank).

**This confirms the T4/Turing quantization-slowdown finding.** 4-bit is now the fourth consistent data point (after QLoRA training, QLoRA inference, 8-bit inference) — lower VRAM, still meaningfully slower than baseline. `EXPERIMENT_MATRIX.md`'s Qualitative Notes section updated from "pending confirmation" to "CONFIRMED" with all four data points listed.

Also ran `utils/validation.py`'s `check_quant_vram_relationship(1.81, 3.25)` as a consistency check: returns no issues — 4-bit VRAM (1.81GB) is correctly lower than 8-bit VRAM (3.25GB), exactly as the quantization literature predicts. Good confirmation that while the *speed* side of quantization inverts on this hardware, the *VRAM* side behaves normally.

**New secondary finding, added to `EXPERIMENT_MATRIX.md`:** 4-bit's quality cost relative to 8-bit is task-dependent, not uniform. CNN/summarization ROUGE actually held up slightly better under 4-bit (+8.15%) than 8-bit (+3.94%) — both within noise of baseline. SQuAD/QA degraded far more under 4-bit (-45.6% F1) than 8-bit (-12.71% F1) — a real, non-noise-level hit. Plausible explanation: SQuAD's exact-match-driven metrics penalize short-answer precision loss much more harshly than ROUGE's partial-overlap scoring does for a generated summary. Report framing: don't describe 4-bit as "uniformly worse than 8-bit" — the quality cost shows up specifically on exact/short-answer tasks.

**Issues:** None — both experiments completed cleanly, inference-only so no training-side failure modes apply.

**GPU hours used this session:** Not separately tracked (inference-only, fast — expected well under 0.5h per `EXPERIMENT_MATRIX.md`'s 0.2h/dataset estimate); no `training_time_hrs` to sum since this technique has none.

**Next session:** Mistral-7B ONNX export/inference (`EXP-MIS-ONNX-CNN`/`EXP-MIS-ONNX-SQUAD`) — last Mistral-7B technique in the matrix, closes out Mistral's full 12-experiment set. Then decide on Llama-2-13B's deferred QLoRA and remaining quantization/ONNX experiments.

---

### 2026-08-15 — Kaggle (ONNX export, Mistral-7B) — OOM, code fix applied, rerun still needed

**Planned:** Mistral-7B ONNX export/inference (`EXP-MIS-ONNX-CNN`/`EXP-MIS-ONNX-SQUAD`) — last Mistral-7B technique in the matrix.

**Completed:** No experiment rows produced this session — export itself OOM'd before any inference ran. Root-caused and fixed in code (see Issues); **not yet reverified on Kaggle hardware.**

**Issues:** Export step hit Kaggle's "Your notebook tried to allocate more memory than is available" — confirmed via nvidia-smi/Kaggle's crash banner, not a Python traceback, so a genuine VRAM ceiling hit rather than a script bug. `EXPERIMENT_MATRIX.md`'s VRAM math never separately projected the export step, only steady-state inference — that's the gap that let this through planning.

Root-caused by reading `experiments/mistral/06_onnx.py` and the installed `optimum.exporters.onnx.main_export` source directly (pinned as `optimum[onnxruntime]>=1.19.0`, unpinned exact version — no `requirements.lock.txt` yet, so this is the best available proxy for what ran on Kaggle):
1. The old code called `ORTModelForCausalLM.from_pretrained(model_id, export=True, provider="CUDAExecutionProvider", token=HF_TOKEN)` — export/trace and CUDA `InferenceSession` creation happened together rather than the trace being kept off-GPU.
2. Bigger effect, confirmed by reading `main_export`'s signature: its `dtype` parameter defaults to `"fp32"` whenever not explicitly passed, and the old code never passed it. A fp32 export of a 7B model is ~28GB — double the fp16 baseline's ~14GB that fits fine — so loading that graph onto a 16GB GPU via `CUDAExecutionProvider` would OOM on its own, independent of where tracing happened.

**Fix applied** (same session, local, no GPU used): both `experiments/mistral/06_onnx.py` and `experiments/llama/05_onnx.py` now call `optimum.exporters.onnx.main_export()` directly with `device="cpu", dtype="fp16"` — export/trace entirely off-GPU, at the same precision as the fp16 baseline — then load the resulting graph from `onnx_dir` via `ORTModelForCausalLM.from_pretrained(onnx_dir, provider="CUDAExecutionProvider")` for the actual benchmarked GPU inference loop. Also turned the "first run only, reuses `onnx_dir`" docstring claim into a real `os.path.isdir(...)`/`.onnx`-file check — previously just a comment with no actual implementation, so every prior run would have silently re-exported from scratch. New Recovery Procedures entry added to `EXPERIMENT_MATRIX.md`; `PROJECT_STATE.md` Blockers & Risks updated with the same writeup plus an explicit open-risk flag for Llama-2-13B's ONNX export (13B fp16 weights alone are ~26GB, already over 16GB before export overhead — CPU export may be the only option there, unconfirmed whether Kaggle CPU RAM can even hold it for tracing; not scheduled until checked).

**Verified offline only:** `py_compile` on both scripts, full module-level import against the real installed `optimum`/`torch`/`transformers` stack (catches import/API-signature errors), and `inspect.signature(main_export)` confirms it accepts `device`/`dtype`/`task`/`token` as used. **No GPU available locally — cannot confirm the fix actually resolves the OOM until rerun on Kaggle.**

**GPU hours used this session:** 0.0 (OOM'd almost immediately on the export step before meaningful GPU time accrued; fix itself was code-only, local, offline-verified)

**Next session:** Rerun `experiments/mistral/06_onnx.py` on Kaggle — first real-hardware test of the CPU-export/fp16 fix. Confirm no OOM, confirm `peak_vram_gb` during the benchmarked inference loop is sane (should be in the same ballpark as the fp16 baseline, not 2x it), and confirm quality is ~unchanged vs. baseline as expected for ONNX export. Only mark Week 4's ONNX checklist item done once this is clean. Do not attempt Llama-2-13B's ONNX export until the open CPU-RAM risk noted above is separately checked.

---
