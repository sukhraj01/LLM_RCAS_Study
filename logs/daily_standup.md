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

**Completed:** 3 of 4 baselines finished — EXP-MIS-BASE-CNN, EXP-MIS-BASE-SQUAD, EXP-LLAMA-BASE-CNN. Real numbers pending — engineer downloaded `mistral_results.csv`/`llama_results.csv` and will paste them in for `logs/experiment_tracking.csv`.

**Issues:** EXP-LLAMA-BASE-SQUAD hung for 5+ hours and was killed by the 12h Kaggle session cap instead of completing or erroring. Root cause: `experiments/common.py`'s `run_inference_only_experiment()` never released the model between successive `load_model_and_tokenizer()` calls in the same process (once per dataset). Loading Llama-2-13B a second time without freeing the first instance left enough VRAM occupied that `device_map="auto"` fell back to CPU-offloading some layers, and `.generate()` essentially never finished — no error, no crash, just silence.

**Fix applied** (same session, local): `experiments/common.py` now (1) explicitly releases the model (`del model; gc.collect(); torch.cuda.empty_cache()`) at the end of both `run_inference_only_experiment()` and `run_training_experiment()`; (2) checks parameter device placement right after model load (and after `apply_lora()` for training) via a new `_assert_fully_on_gpu()` guard, raising `RuntimeError` immediately instead of silently continuing into a slow `.generate()` loop; (3) `generate_predictions()` now prints progress every 20 samples so a stuck run is visible in the Kaggle log in real time. Documented as a new recovery procedure in `EXPERIMENT_MATRIX.md` and `CLAUDE.md`. Full detail in `knowledge/ai-usage-log/2026-08-13_kaggle-baseline-hang.md`.

**GPU hours used this session:** 12.0 (full 12h session cap — planned 2.0 for this phase; see `PROJECT_STATE.md` GPU Budget Tracking, now recorded honestly as over budget)

**Next session:** Rerun EXP-LLAMA-BASE-SQUAD on Kaggle with the fixed `common.py` (should now either complete normally or fail loudly and fast instead of hanging). Paste real numbers from `mistral_results.csv`/`llama_results.csv` into `logs/experiment_tracking.csv` for the 3 completed baselines. Then proceed to Week 2 (LoRA, Mistral-7B) once the baseline phase is actually clean.

---
