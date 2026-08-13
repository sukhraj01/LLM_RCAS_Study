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

## AI-Generated vs Human-Written

All changes this session are AI-generated, per the engineer's diagnosis and exact fix spec above (root cause and fix design were the engineer's, not derived independently this session):
- `experiments/common.py` — `import gc`; new `_assert_fully_on_gpu()` and `_release_model()` helpers; guard calls inserted after model load in `run_inference_only_experiment()` and after `apply_lora()` in `run_training_experiment()`; explicit cleanup calls before each function returns; progress printing added to `generate_predictions()`
- `EXPERIMENT_MATRIX.md` — new "Model silently CPU-offloaded / hangs" entry in Recovery Procedures
- `CLAUDE.md` — new "Model Silently CPU-Offloaded (Hang, No Error)" entry in Failure Recovery Is Proceduralized
- `PROJECT_STATE.md` — GPU Budget Tracking table updated with actual (12h) vs. planned (2h) hours for Setup + baseline
- `logs/experiment_tracking.csv` — 3 rows marked `completed` (numeric fields left blank pending real numbers from the engineer), 1 row marked `failed` with root-cause note
- `logs/daily_standup.md` — new session entry
- This file

## Key Decisions

- **Guard checks `p.device.type != "cuda"`, not an allowlist of "expected" devices.** This means the check also fires on `mps`/other non-cuda types, not just `cpu`/`meta` as literally named in the failure — correct, since any non-GPU placement on Kaggle is the same bug. Kept the error message focused on the two device types actually seen in practice (CPU offload, meta init) since that's what the recovery guidance addresses.
- **Cleanup (`_release_model`) placed after latency/VRAM measurement but before quality-metric computation** in both runners, rather than at the very end of the function. Quality metrics (ROUGE/F1) are CPU-only and don't need the model, so releasing GPU memory as early as possible reduces the window where a second model load (in a future multi-model-per-process script) could collide with this one still being resident.
- **Did not change `load_model_and_tokenizer()` itself** (e.g., to pass `max_memory` or disable `device_map="auto"` fallback) — the engineer's fix spec was explicit about failing loudly after the fact rather than changing the loading strategy, so kept the change minimal and exactly scoped to what was asked. If this recurs even with clean process-per-model discipline, revisit `device_map` strategy as a follow-up ADR.
- **`logs/experiment_tracking.csv` numeric columns (`gpu_hours_used`, `peak_vram_gb`, `start_time`, `end_time`) left blank for the 3 completed rows**, not fabricated — engineer has the real numbers in downloaded `mistral_results.csv`/`llama_results.csv` and offered to paste them in; per `CLAUDE.md` ("never write a result that hasn't actually been measured"), status was updated honestly but placeholder numbers were not invented to fill the row.
- **GPU Budget Tracking table flags the overrun explicitly** (🔴, "flagging per CLAUDE.md, not silently absorbing it") rather than just updating the number, per `CLAUDE.md`'s "if constraints are about to be violated, stop and escalate explicitly" and "push back if GPU budget is tightening." The Setup+baseline phase is 10h over its own 2h budget; total plan still has 17.4h of 29.4h left, but the 8h buffer week has effectively been partly pre-spent by this overrun.

## Experiments Executed (this Kaggle session, reported by engineer)

- EXP-MIS-BASE-CNN — completed (numbers pending)
- EXP-MIS-BASE-SQUAD — completed (numbers pending)
- EXP-LLAMA-BASE-CNN — completed (numbers pending)
- EXP-LLAMA-BASE-SQUAD — failed (CPU-offload hang, killed by 12h session cap)

## Issues Encountered

**CPU-offload hang on EXP-LLAMA-BASE-SQUAD**, root-caused and fixed per the engineer's spec above. Full mechanism: `run_inference_only_experiment()` is called once per (model, dataset) pair within the same baseline script run. For Llama-2-13B, the second call (SQuAD, after CNN already ran and loaded a full 13B model onto GPU) called `load_model_and_tokenizer()` again without the first model instance having been freed. With the first ~13B-worth of GPU memory still held, `device_map="auto"` on the second load couldn't fit the whole model on GPU and silently placed some layers on CPU. `.generate()` continued to "work" — just at CPU-offloaded speed, effectively never finishing within any reasonable time, with no error and no log output to indicate anything was wrong until the 12h Kaggle session cap killed it.

**Fix:** explicit model release (`del model; gc.collect(); torch.cuda.empty_cache()`) after every experiment run, a fail-fast device-placement guard right after every model load, and periodic progress logging so a stuck run is visible immediately instead of silently consuming the entire session. Documented as a new recovery procedure in both `EXPERIMENT_MATRIX.md` and `CLAUDE.md` so it's checked before it recurs on a training run, where the same bug would be worse (burns GPU hours on a run that also can't produce a usable checkpoint).

**Not yet verified against real hardware** — the fix has been syntax/import-checked locally (`py_compile` + `import experiments.common` succeed, no GPU available locally to actually exercise `_assert_fully_on_gpu()` or the CPU-offload path) but has not yet been re-run on Kaggle. Next Kaggle session should rerun EXP-LLAMA-BASE-SQUAD first to confirm the guard actually catches the failure mode (or that it no longer occurs) before resuming the rest of the matrix.
