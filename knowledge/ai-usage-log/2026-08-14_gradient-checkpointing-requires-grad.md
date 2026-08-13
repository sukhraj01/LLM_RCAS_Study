# Session Log — 2026-08-14 — LoRA Backward Pass Failure: enable_input_require_grads()

## Prompts (verbatim, in order)

### Prompt 1

> LoRA training on Mistral-7B just failed on the very first backward pass with:
>
>   UserWarning: None of the inputs have requires_grad=True. Gradients will be None
>   RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
>
> Full traceback: apply_lora() -> get_peft_model() succeeds (3.4M trainable /
> 7.2B total params, 0.047% - looks right), training starts, use_cache gets
> disabled for gradient checkpointing as expected, then the very first
> trainer.train() backward call fails.
>
> Root cause: this is the standard PEFT + gradient-checkpointing gotcha. When
> only LoRA adapter params are trainable and gradient checkpointing is on, the
> frozen base model's input embeddings output requires_grad=False, so
> checkpointing has no tensor to build a backward graph from - the graph never
> reaches the trainable adapter weights. Fix is `model.enable_input_require_grads()`,
> called right after get_peft_model() in apply_lora() in experiments/common.py,
> so it applies uniformly to every technique that calls apply_lora() (lora,
> qlora - Mistral and Llama both).
>
> Please:
> 1. Find apply_lora() in experiments/common.py and add
>    model.enable_input_require_grads() right after get_peft_model() returns,
>    before the function returns the adapted model.
> 2. Check if gradient_checkpointing=True is set anywhere else in the training
>    path (TrainingArguments in run_training_experiment()) that would need the
>    same treatment - confirm the fix covers wherever gradient checkpointing is
>    actually enabled.
> 3. Verify offline (py_compile / import check - no GPU here) since this can't
>    be tested for real without Kaggle.
> 4. Add this to EXPERIMENT_MATRIX.md's Recovery Procedures (symptom: "None of
>    the inputs have requires_grad=True" / RuntimeError on backward during LoRA
>    or QLoRA training -> fix: enable_input_require_grads() after get_peft_model()).
> 5. Log in logs/daily_standup.md (today's Kaggle LoRA entry - this happened in
>    the same session as the torchao fix, after that was resolved) and in
>    knowledge/ai-usage-log/ with the verbatim traceback.
>
> Do not commit yet - report back what changed first.

## AI-Generated vs Human-Written

All changes this session are AI-generated, from a root cause and fix the engineer diagnosed and reported directly (not derived by Claude Code):
- `experiments/common.py` — `apply_lora()` (around line 113) now calls `model.enable_input_require_grads()` immediately after `get_peft_model()` returns, with a comment explaining why; function now assigns `get_peft_model()`'s return value to `model` before returning it instead of returning it inline
- `EXPERIMENT_MATRIX.md` — new Recovery Procedures entry: symptom ("None of the inputs have requires_grad=True" / `RuntimeError` on backward during LoRA/QLoRA training) → root cause (frozen base model's input embeddings have `requires_grad=False` under gradient checkpointing) → fix (`enable_input_require_grads()` after `get_peft_model()`, applied once inside `apply_lora()`)
- `logs/daily_standup.md` — today's `2026-08-14 — Kaggle (LoRA env fix)` entry extended with a second numbered Issues item for this failure (same session as the torchao fix, occurring after it was resolved); GPU-hours note adjusted to "blocked before any training completed a step"
- This file

## Key Decisions

- **Fix placed inside `apply_lora()`, not duplicated per training script.** `apply_lora()` is the single call site of `get_peft_model()` in the codebase (confirmed via `grep -rn "get_peft_model" .` — only `experiments/common.py:24` import and `:122`/now `:127` call), and every technique that trains an adapter (LoRA and QLoRA, both models) goes through `run_training_multi_dataset()` → `apply_lora()`. Fixing it once here, matching the engineer's request, means no per-script or per-technique duplication and no risk of a future technique script forgetting the call.
- **Confirmed step 2 (no other `gradient_checkpointing` site needing the same treatment) by grep, not assumption.** `grep -rn "gradient_checkpointing\|get_peft_model\|TrainingArguments(" --include="*.py" .` (excluding `.venv`) shows exactly one `TrainingArguments(...)` construction (`run_training_experiment()`, line ~358) with `gradient_checkpointing=lora_hparams["gradient_checkpointing"]`, and `gradient_checkpointing: True` set in two places in `utils/config.py` (LoRA and QLoRA hparam dicts, both consumed by that same single `TrainingArguments` call). No duplicate or parallel training-args construction elsewhere in the repo, so the single fix in `apply_lora()` covers every path that actually enables gradient checkpointing.
- **Verification was import/compile-only, as instructed** — `python -m py_compile experiments/common.py` and `python -c "import experiments.common"` both succeeded in the local venv. This confirms the code is syntactically valid and `enable_input_require_grads` is a real method being called correctly at parse/import time; it does **not** confirm the backward pass actually succeeds, since that requires a loaded model and a real GPU. Left as an open item for the next Kaggle session per the engineer's own framing ("can't be tested for real without Kaggle").
- **Did not commit**, per explicit instruction to report back first.

## Experiments Executed

None. Week 2 LoRA training (`experiments/mistral/02_lora.py`, both datasets) has not yet completed a training step — this is the second blocker in the same Kaggle session (after the torchao `ImportError`, see `2026-08-14_torchao-lora-importerror.md`). First real attempt with both fixes in place is queued as "Next session" in `logs/daily_standup.md`.

## Issues Encountered

Verbatim failure (as reported by engineer):

```
UserWarning: None of the inputs have requires_grad=True. Gradients will be None
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

Occurred on the first `trainer.train()` backward call, after `apply_lora()` → `get_peft_model()` succeeded (3.4M trainable / 7.2B total params, 0.047%) and `use_cache` was disabled for gradient checkpointing as expected. Standard PEFT + gradient-checkpointing interaction: the frozen base model's input embeddings output `requires_grad=False`, so activation checkpointing (which needs a `requires_grad=True` tensor somewhere in its recomputation path to attach a backward graph) never reaches the trainable LoRA adapter weights. Fixed by calling `model.enable_input_require_grads()` right after `get_peft_model()` — this forces the embedding output to require grad, giving checkpointing a valid graph entry point, without changing which parameters are actually trainable (the 0.047% figure is unaffected). No other issues this session beyond the torchao `ImportError` logged separately.
