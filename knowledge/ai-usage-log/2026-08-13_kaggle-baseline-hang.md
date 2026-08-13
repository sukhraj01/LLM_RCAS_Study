# Session Log — 2026-08-13 — Kaggle Baseline Run: CPU-Offload Hang

## Prompts (verbatim, in order)

### Prompt 1

> The first Kaggle baseline run (see summary below) completed 3 of 4 experiments
> but hung for 5+ hours and got killed by the 12h session cap on the 4th. Root
> cause: experiments/common.py's run_inference_only_experiment() never releases
> the model between successive load_model_and_tokenizer() calls in the same
> process. For Llama-2-13B, loading it a second time (once per dataset, in the
> same script run) without freeing the first instance caused device_map="auto"
> to fall back to CPU-offloading some layers, which made .generate() essentially
> never finish.
>
> Fix experiments/common.py:
>
> 1. In both run_inference_only_experiment() and run_training_experiment(), add
>    explicit cleanup at the end (before returning): del model, then
>    gc.collect(), then torch.cuda.empty_cache(). Import gc at the top.
>
> 2. Add a guard right after load_model_and_tokenizer() in
>    run_inference_only_experiment() (and after apply_lora() in
>    run_training_experiment()) that checks whether any model parameter landed
>    on "meta" or "cpu" when it shouldn't have (iterate model.parameters(),
>    check .device.type). If any are found, raise a clear RuntimeError instead
>    of silently continuing into a slow generate() loop — something like
>    "Model partially offloaded to CPU (device_map couldn't fit it on GPU) —
>    aborting instead of running for hours. Free GPU memory or reduce batch/seq
>    length before retrying." This turns a silent multi-hour hang into an
>    immediate, loud failure.
>
> 3. In generate_predictions(), print progress every 20 samples (e.g. "50/200
>    done, Xs/sample avg") so a stuck run is visible in the Kaggle log in
>    real time instead of going silent for hours.
>
> 4. Update EXPERIMENT_MATRIX.md's "Recovery Procedures" section and CLAUDE.md's
>    "Failure Recovery Is Proceduralized" section: add a new entry for this
>    failure mode — "Model silently CPU-offloaded / hangs" — with the fix above
>    as the procedure, so it's documented before it happens again on a training
>    run (where it'd be worse).
>
> 5. Update PROJECT_STATE.md: GPU Budget Tracking actually used 12h against the
>    2h planned for Setup+baseline this session — record that honestly, don't
>    leave it at the estimate. Update logs/experiment_tracking.csv: EXP-MIS-BASE-CNN,
>    EXP-MIS-BASE-SQUAD, EXP-LLAMA-BASE-CNN → completed (I'll paste the actual
>    numbers from the downloaded CSVs once you ask for them). EXP-LLAMA-BASE-SQUAD
>    → status "failed", note "CPU-offload hang, killed by 12h session cap, fix
>    in common.py, will rerun."
>
> 6. Append a session entry to logs/daily_standup.md and a new dated file in
>    knowledge/ai-usage-log/ documenting this failure and fix, per CLAUDE.md.
>
> 7. Commit and push.
>
> I'll paste the contents of the downloaded mis_results.csv and llama_results.csv
> next so you can merge the real numbers into experiment_tracking.csv — ask me
> for them if you need them before finishing step 5.

### Prompt 2

> exp_id,model,technique,dataset,training_time_hrs,peak_vram_gb,inference_latency_ms,quality_metrics,status,notes,quality_degradation_percent,speedup_factor,validation_issues
> EXP-MIS-BASE-CNN,MIS,baseline,CNN,,6.91,9723.3,"{'rouge1': np.float64(0.23864457084089002), 'rouge2': np.float64(0.08410718202602918), 'rougeL': np.float64(0.15979991876555116)}",completed,,0.0,1.0,
> EXP-MIS-BASE-SQUAD,MIS,baseline,SQUAD,,13.65,6643.0,"{'exact_match': 8.0, 'f1': 24.19339741135814}",completed,,0.0,1.0,
>
> exp_id,model,technique,dataset,training_time_hrs,peak_vram_gb,inference_latency_ms,quality_metrics,status,notes,quality_degradation_percent,speedup_factor,validation_issues
> EXP-LLAMA-BASE-CNN,LLAMA,baseline,CNN,,12.39,15065.6,"{'rouge1': np.float64(0.25078525030817), 'rouge2': np.float64(0.08700461741796214), 'rougeL': np.float64(0.16905006023025135)}",completed,,0.0,1.0,

(Contents of the engineer's downloaded `mis_results.csv` and `llama_results.csv`, pasted so the real numbers could be merged into `logs/experiment_tracking.csv` per Prompt 1 step 5.)

## AI-Generated vs Human-Written

All changes this session are AI-generated, per the engineer's diagnosis and exact fix spec above (root cause and fix design were the engineer's, not derived independently this session):
- `experiments/common.py` — `import gc`; new `_assert_fully_on_gpu()` and `_release_model()` helpers; guard calls inserted after model load in `run_inference_only_experiment()` and after `apply_lora()` in `run_training_experiment()`; explicit cleanup calls before each function returns; progress printing added to `generate_predictions()`
- `EXPERIMENT_MATRIX.md` — new "Model silently CPU-offloaded / hangs" entry in Recovery Procedures
- `CLAUDE.md` — new "Model Silently CPU-Offloaded (Hang, No Error)" entry in Failure Recovery Is Proceduralized
- `PROJECT_STATE.md` — GPU Budget Tracking table updated with actual (12h) vs. planned (2h) hours for Setup + baseline
- `logs/experiment_tracking.csv` — 3 rows marked `completed`, later filled with real `peak_vram_gb`/latency/quality numbers once the engineer pasted them (Prompt 2); 1 row marked `failed` with root-cause note
- `logs/daily_standup.md` — new session entry
- `utils/metrics.py` — `compute_rouge()` fixed to cast to native `float` (see Issues Encountered below — found while merging Prompt 2's data, not requested directly, but blocking for Week 2)
- `results/mis_results.csv`, `results/llama_results.csv` — created from Prompt 2's pasted data (with the numpy-repr bug repaired in `quality_metrics` before writing — same measured values, corrected serialization; disclosed to engineer)
- `ARCHITECTURE.md` — one-line fix to the file-structure diagram (`mistral_results.csv` → `mis_results.csv`, matching what the code actually reads/writes)
- This file

## Key Decisions

- **Guard checks `p.device.type != "cuda"`, not an allowlist of "expected" devices.** This means the check also fires on `mps`/other non-cuda types, not just `cpu`/`meta` as literally named in the failure — correct, since any non-GPU placement on Kaggle is the same bug. Kept the error message focused on the two device types actually seen in practice (CPU offload, meta init) since that's what the recovery guidance addresses.
- **Cleanup (`_release_model`) placed after latency/VRAM measurement but before quality-metric computation** in both runners, rather than at the very end of the function. Quality metrics (ROUGE/F1) are CPU-only and don't need the model, so releasing GPU memory as early as possible reduces the window where a second model load (in a future multi-model-per-process script) could collide with this one still being resident.
- **Did not change `load_model_and_tokenizer()` itself** (e.g., to pass `max_memory` or disable `device_map="auto"` fallback) — the engineer's fix spec was explicit about failing loudly after the fact rather than changing the loading strategy, so kept the change minimal and exactly scoped to what was asked. If this recurs even with clean process-per-model discipline, revisit `device_map` strategy as a follow-up ADR.
- **`logs/experiment_tracking.csv` numeric columns (`gpu_hours_used`, `peak_vram_gb`, `start_time`, `end_time`) left blank for the 3 completed rows**, not fabricated — engineer has the real numbers in downloaded `mistral_results.csv`/`llama_results.csv` and offered to paste them in; per `CLAUDE.md` ("never write a result that hasn't actually been measured"), status was updated honestly but placeholder numbers were not invented to fill the row.
- **GPU Budget Tracking table flags the overrun explicitly** (🔴, "flagging per CLAUDE.md, not silently absorbing it") rather than just updating the number, per `CLAUDE.md`'s "if constraints are about to be violated, stop and escalate explicitly" and "push back if GPU budget is tightening." The Setup+baseline phase is 10h over its own 2h budget; total plan still has 17.4h of 29.4h left, but the 8h buffer week has effectively been partly pre-spent by this overrun.

## Experiments Executed (this Kaggle session, reported by engineer)

- EXP-MIS-BASE-CNN — completed: peak VRAM 6.91GB, latency 9723.3ms, ROUGE1/2/L 0.239/0.084/0.160
- EXP-MIS-BASE-SQUAD — completed but flagged: peak VRAM 13.65GB (suspect, see Issues below), latency 6643.0ms, EM 8.0 / F1 24.19
- EXP-LLAMA-BASE-CNN — completed: peak VRAM 12.39GB, latency 15065.6ms, ROUGE1/2/L 0.251/0.087/0.169
- EXP-LLAMA-BASE-SQUAD — failed (CPU-offload hang, killed by 12h session cap)

## Issues Encountered

**CPU-offload hang on EXP-LLAMA-BASE-SQUAD**, root-caused and fixed per the engineer's spec above. Full mechanism: `run_inference_only_experiment()` is called once per (model, dataset) pair within the same baseline script run. For Llama-2-13B, the second call (SQuAD, after CNN already ran and loaded a full 13B model onto GPU) called `load_model_and_tokenizer()` again without the first model instance having been freed. With the first ~13B-worth of GPU memory still held, `device_map="auto"` on the second load couldn't fit the whole model on GPU and silently placed some layers on CPU. `.generate()` continued to "work" — just at CPU-offloaded speed, effectively never finishing within any reasonable time, with no error and no log output to indicate anything was wrong until the 12h Kaggle session cap killed it.

**Fix:** explicit model release (`del model; gc.collect(); torch.cuda.empty_cache()`) after every experiment run, a fail-fast device-placement guard right after every model load, and periodic progress logging so a stuck run is visible immediately instead of silently consuming the entire session. Documented as a new recovery procedure in both `EXPERIMENT_MATRIX.md` and `CLAUDE.md` so it's checked before it recurs on a training run, where the same bug would be worse (burns GPU hours on a run that also can't produce a usable checkpoint).

**Not yet verified against real hardware** — the fix has been syntax/import-checked locally (`py_compile` + `import experiments.common` succeed, no GPU available locally to actually exercise `_assert_fully_on_gpu()` or the CPU-offload path) but has not yet been re-run on Kaggle. Next Kaggle session should rerun EXP-LLAMA-BASE-SQUAD first to confirm the guard actually catches the failure mode (or that it no longer occurs) before resuming the rest of the matrix.

---

**Two more issues found while merging Prompt 2's pasted CSV data — neither requested directly, both surfaced by actually trying to load/parse the real numbers rather than just copying them in:**

**1. `quality_metrics` was unparseable — blocking bug for every future technique script.** Both CNN rows' `quality_metrics` came through as `"{'rouge1': np.float64(0.238...), ...}"` (numpy-repr syntax embedded in the string), not valid Python literal syntax. Confirmed the cause: `evaluate`'s rouge metric returns numpy `float64` scalars, and NumPy >=2.0 (this venv has 2.4.6, installed in the prior session's `pip install`) changed `repr(np.float64(x))` from bare `x` to `"np.float64(x)"`. `experiments/common.py`'s `load_baseline_metrics()` reads this field back with `ast.literal_eval()`, which only parses literals — not function-call syntax like `np.float64(...)` — so it would raise `ValueError` on any of these baseline rows. Verified this concretely: `ast.literal_eval()` on the pasted string fails; every Week 2+ technique script (LoRA, QLoRA, 8-bit, 4-bit, ONNX) calls `load_baseline_metrics()` to compute `quality_degradation_percent`, so this would have crashed the very first CNN-dataset technique run. **Fixed** in `utils/metrics.py`: `compute_rouge()` now casts each value to native `float` before returning. Verified the fix with a live call + round-trip through `str()` → `ast.literal_eval()`. The SQuAD metric function (`compute_squad_metrics()`) was already unaffected — its `exact_match`/`f1` are computed via plain Python arithmetic (`100.0 * sum(...) / len(...)`), not numpy, which is exactly why the pasted SQuAD rows didn't show the same corruption and only the two CNN rows did.

**2. Results filename doesn't match `ARCHITECTURE.md`'s documented name.** `save_result()`/`load_baseline_metrics()` in `experiments/common.py` build the results path as `f"{model_key.lower()}_results.csv"`, and `model_key` for Mistral is `"MIS"` (per `utils/config.py`'s `MODELS` dict) — so the actual file is `mis_results.csv`, not `mistral_results.csv` as `ARCHITECTURE.md`'s file-structure diagram claims (this matches what the engineer's own message called it: "mis_results.csv"). Llama's happens to match by coincidence (`"LLAMA".lower() == "llama"`). Initially wrote the pasted data to `results/mistral_results.csv` by following the doc instead of the code, then caught it by testing `load_baseline_metrics()` against the file I'd just written and getting `None` back for both Mistral rows. Fixed by renaming to `results/mis_results.csv` (matching what the code actually reads/writes — safer to fix the file than to change `save_result()`'s path logic, which is what all 22 experiment scripts depend on) and correcting the one-line reference in `ARCHITECTURE.md`'s file tree. Re-verified: `load_baseline_metrics('MIS', 'CNN')` and `('MIS', 'SQUAD')` both now return the right data.

**3. `EXP-MIS-BASE-SQUAD`'s `peak_vram_gb` (13.65GB) is very likely contaminated by the same un-released-model bug that caused the Llama hang — not a clean baseline measurement.** `EXP-MIS-BASE-CNN` (the first Mistral load in that process) measured 6.91GB peak, and `EXP-MIS-BASE-SQUAD` (the second load, same process, pre-fix `common.py`) measured 13.65GB — almost exactly double (6.91 × 2 = 13.82, within ~1.2%). This is the same failure mode as the Llama hang, just non-fatal for a 7B model: two un-released Mistral-7B instances resident simultaneously still fit under 16GB (unlike two 13B instances, which didn't, hence the hang), so it didn't error or hang — it just silently produced an inflated VRAM number. Cross-checked against the multi-GPU hypothesis: Llama's clean first-load number (12.39GB) is close to half of Llama-2-13B's expected fp16 weight footprint (~26GB / 2 GPUs ≈ 13GB), consistent with `device_map="auto"` splitting across Kaggle's T4x2 pair — and 6.91GB for Mistral-7B (~14.4GB fp16 weights / 2 ≈ 7.2GB) fits the same pattern, which is what makes the ~2x jump on the second Mistral load stand out as a real anomaly rather than noise. **Not silently corrected** — the measured number (13.65GB) was kept as-is in `results/mis_results.csv` (it's what was actually measured, even if contaminated), but flagged in both `results/mis_results.csv`'s `notes` field and `logs/experiment_tracking.csv`, with a recommendation to rerun `EXP-MIS-BASE-SQUAD` alongside `EXP-LLAMA-BASE-SQUAD` once back on Kaggle with the fixed `common.py`, so the master benchmark CSV doesn't end up reporting a doubled VRAM figure as Mistral-7B's real SQuAD baseline footprint.
