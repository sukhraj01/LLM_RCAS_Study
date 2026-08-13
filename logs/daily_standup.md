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
