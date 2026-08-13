# Session Log — 2026-08-13 — Environment Verification

## Prompts (verbatim, in order)

### Prompt 1

> Read PROJECT_STATE.md, CLAUDE.md, ARCHITECTURE.md, and EXPERIMENT_MATRIX.md first, per the
> "start of every session" rule in CLAUDE.md.
>
> Then do the following, in order, and stop to report back after each step if anything fails:
>
> 1. Verify environment: confirm a Python 3.10+ venv exists (create one at .venv if not),
>    activate it, and run `pip install -r requirements.txt`. Report any install failures.
>
> 2. Verify HF auth: check if HF_TOKEN is set (from a local .env file, copied from
>    .env.example). If missing, stop and tell me — I need to set it from
>    https://huggingface.co/settings/tokens, and I need to have requested access to
>    meta-llama/Llama-2-13b-hf (it's gated).
>
> 3. Sanity-check the data pipeline: run `python -m utils.data_loader` (the
>    sanity_check() at the bottom of utils/data_loader.py). This downloads small
>    samples of CNN/DailyMail and SQuAD and checks split sizes/no-overlap. This
>    doesn't need a GPU. Report the output.
>
> 4. Syntax/import-check all experiment scripts: for each file in experiments/mistral/
>    and experiments/llama/, run `python -m py_compile <file>` and separately try
>    importing experiments.common to catch missing dependencies or import errors
>    now, before we're burning Kaggle GPU quota debugging import errors.
>
> 5. Do NOT attempt to actually load Mistral-7B or Llama-2-13B or run any training/
>    inference locally — there's no GPU here, that all happens on Kaggle per
>    KAGGLE_SYNC.md.
>
> 6. Update PROJECT_STATE.md's Component Status table for whatever now actually
>    passed ("Environment setup", "Data pipeline" rows), and append an entry to
>    logs/daily_standup.md for this session. Do not mark anything as done that
>    didn't actually pass.
>
> 7. If everything above passes: tell me the exact next step is pushing this repo
>    to GitHub and starting the first Kaggle session per README.md, and give me
>    the git remote/push commands (I haven't pushed yet).
>
> gihub repo: https://github.com/sukhraj01/LLM_RCAS_Study.git

### Prompt 2

> bro hf token is avialble now you can chekc also jsut for you inof my pc supposrts pip3 without nay probms os you cn atry that insteadof pip or pything 3 for that metter

(Interpreted as: HF_TOKEN is now set in `.env` — re-check it. Also FYI the engineer's machine runs `pip3`/`python3` fine, offered as an alternative if `pip`/`python` had problems — informational, not a required change since the venv-based `pip`/`python` aliases were already working.)

### Prompt 3

> Apply option 1 from your last report:
>
> 1. In requirements.txt, change the line
>      optimum[onnxruntime-gpu]>=1.19.0
>    to
>      optimum[onnxruntime]>=1.19.0
>    and add a comment above it matching the existing style of the torch comment
>    at the top of the file — explain that local dev uses CPU onnxruntime since
>    onnxruntime-gpu has no macOS wheel, and that Kaggle sessions should instead
>    run `pip install -U "optimum[onnxruntime-gpu]"` as an extra install step
>    (on top of requirements.txt) to get GPU-accelerated ONNX inference there.
>
> 2. Update the Kaggle setup section in README.md to add that extra install line
>    after the existing `pip install -q -r requirements.txt --no-deps` step, so
>    Kaggle sessions actually get onnxruntime-gpu and don't silently fall back to
>    CPU-only ONNX Runtime (which would produce misleading/slow latency numbers
>    for the EXP-*-ONNX-* experiments).
>
> 3. Retry `pip install -r requirements.txt` in the venv. Report success or the
>    next failure — don't guess further fixes without telling me what broke.
>
> 4. If install succeeds, continue from where you stopped: run the HF_TOKEN
>    check (I've now set it in .env), then the data pipeline sanity check
>    (utils/data_loader.py), then the py_compile/import check on all experiment
>    scripts.
>
> 5. Log this dependency conflict and the fix in a new entry in
>    knowledge/ai-usage-log/ under today's date (create the file if one for
>    today doesn't exist yet) — what broke, why, and what you changed. This is
>    exactly the kind of "unexpected result" CLAUDE.md says to document, not
>    just silently fix.

## AI-Generated vs Human-Written

All changes this session are AI-generated:
- `requirements.txt` — one-line change (`optimum[onnxruntime-gpu]` → `optimum[onnxruntime]`) plus explanatory comment
- `README.md` — one added line in the Kaggle setup cell (`pip install -U "optimum[onnxruntime-gpu]"`)
- `PROJECT_STATE.md` — Component Status table rows for "Environment setup" and "Data pipeline"
- `logs/daily_standup.md` — new session entry
- This file

No application code (`experiments/`, `utils/`) was changed this session — only verified.

## Key Decisions

- **Local venv uses Python 3.11, not the system default 3.14.** 3.14 is ~3 months old at time of writing; ML packages (torch, bitsandbytes) commonly lag behind on wheel support for brand-new Python versions. 3.11 satisfies the 3.10+ requirement and has mature wheel coverage across the whole stack. Not an ADR-level decision (doesn't affect experiment validity/comparability — Kaggle's Python version is what actually matters for training), so not added to `ARCHITECTURE.md`.
- **Fixed `requirements.txt`'s `optimum[onnxruntime-gpu]` pin rather than working around it locally.** `onnxruntime-gpu` publishes no macOS wheel at all (confirmed via `pip download onnxruntime-gpu --platform macosx_11_0_arm64` → "no matching distribution found for any version"). This isn't a "no GPU, so slow" situation like the existing torch caveat in the file — it's "cannot be installed on this OS, period." While pip backtracked through every `optimum` release trying to satisfy the extra, it landed on `optimum==1.19.0`, which pins `transformers<4.40.0`, conflicting with this file's `transformers>=4.40.0`. Root cause was the unconditional `-gpu` extra, not a bad transformers/peft pin — so the fix was splitting local (CPU onnxruntime) from Kaggle (GPU onnxruntime, installed as an explicit extra step), matching the pattern the file already uses for torch itself.
- Followed CLAUDE.md's "escalate rather than silently substitute" guidance: reported the conflict and a recommended fix, waited for explicit approval before editing `requirements.txt`/`README.md`, rather than picking a resolution unilaterally.

## Experiments Executed

None — this session verified environment/tooling only, no training or inference.

## Issues Encountered

**`pip install -r requirements.txt` failed on first attempt** with:
```
ERROR: Cannot install -r requirements.txt (line 19), -r requirements.txt (line 9) and transformers>=4.40.0
because these package versions have conflicting dependencies.
The conflict is caused by:
    The user requested transformers>=4.40.0
    peft 0.10.0 depends on transformers
    optimum 1.19.0 depends on transformers<4.40.0 and >=4.26.0
```
Diagnosed root cause: `onnxruntime-gpu` (pulled in by `optimum[onnxruntime-gpu]>=1.19.0`) has no macOS wheel on PyPI for any version, confirmed via `pip index versions onnxruntime-gpu` (no versions listed for this platform) and a direct `pip download` attempt. pip's resolver backtracked through the entire `optimum` version history trying to satisfy the extra locally, and in doing so hit a version whose `transformers` upper bound conflicted with this file's `transformers>=4.40.0` floor.

**Fix (approved by engineer, see Prompt 3):**
1. `requirements.txt`: `optimum[onnxruntime-gpu]>=1.19.0` → `optimum[onnxruntime]>=1.19.0`, with a comment (matching the existing torch-comment style) explaining local dev gets CPU onnxruntime and Kaggle needs an extra `pip install -U "optimum[onnxruntime-gpu]"` step.
2. `README.md` Kaggle setup cell: added that extra install line after the existing `pip install -q -r requirements.txt --no-deps` step, so Kaggle sessions actually get GPU-accelerated ONNX Runtime instead of silently falling back to CPU (which would have produced misleading/slow latency numbers for the `EXP-*-ONNX-*` experiments — a correctness risk, not just a convenience issue).

**Retry succeeded** — full stack installed cleanly (torch 2.13.0, transformers 4.57.6, peft 0.20.0, bitsandbytes 0.50.0, accelerate 1.14.0, datasets 5.0.1, optimum 2.1.0 + optimum-onnx 0.1.0, onnx 1.22.0, onnxruntime 1.28.0 (CPU), plus fastapi/streamlit/dev tooling). Full `pip freeze` available via `.venv` if a `requirements.lock.txt` needs to be cut later, per the note at the bottom of `requirements.txt`.

No other issues — HF_TOKEN check, data pipeline sanity check, and py_compile/import checks all passed on first try after the install was fixed (see `logs/daily_standup.md` 2026-08-13 entry for full detail).
