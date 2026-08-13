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

**GPU hours used this session:** ~2.5h (estimated from per-sample timing — Mistral CNN+SQuAD ~54min, Llama CNN+SQuAD ~95min; not exact Kaggle accounting). Combined with the 12h from the 2026-08-13 first attempt, the Setup+baseline phase total is **≥14.5h against a 2h plan** — see `PROJECT_STATE.md` GPU Budget Tracking for the full breakdown and the flag on Weeks 2-4 potentially pushing past the 30h/week cap. **Still open:** the intermediate guard-fix-test Kaggle session (the one that produced the fast `RuntimeError` instead of a hang, which is what led to diagnosing the real root cause) has real but never-recorded GPU-hours — the ≥14.5h figure is a confirmed floor, not the final number.

**Next session:** Week 2 — LoRA fine-tuning on Mistral-7B (`experiments/mistral/02_lora.py`, now using `run_training_multi_dataset()`), both datasets. Verify config against `EXPERIMENT_MATRIX.md` technique #2 before running. Worth deciding explicitly whether the Setup+baseline overrun changes how Weeks 2-4 get scheduled against the weekly quota.

---
