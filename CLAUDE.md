# Claude Code Collaboration Model — LLM Optimization Project

This document defines the stable collaboration contract for the LLM Optimization project between the engineer and Claude Code.

It captures enduring engineering principles rather than project-specific information. Project state belongs in `PROJECT_STATE.md`. Exact experiment specs and hyperparameters belong in `EXPERIMENT_MATRIX.md`. The Kaggle/local handoff protocol belongs in `KAGGLE_SYNC.md`. This file should not duplicate any of those.

**Start of every session:** read `PROJECT_STATE.md` first. It is the single source of truth for what has actually happened. Only pull in `ARCHITECTURE.md` or `EXPERIMENT_MATRIX.md` sections relevant to the current task — don't load the whole archived guide.

---

## Planning Before Implementation

Claude Code should resist premature implementation.

Before running any experiment:

- ensure the experiment specification is fully understood (check `EXPERIMENT_MATRIX.md`),
- verify hyperparameters match the plan,
- identify hardware constraints (VRAM, GPU time, disk space) — **check the VRAM math actually works before scheduling, don't assume**,
- check that tracking/logging infrastructure is ready,
- confirm data and baselines are prepared,
- understand quality gates for this phase,
- identify recovery procedures if the experiment fails,
- and only then begin execution.

Planning is not overhead.

Good planning reduces failed runs, enables tracking, improves reproducibility, and produces defensible results.

---

## Resource Constraints Are Non-Negotiable

This project runs on Kaggle's free tier: **16GB VRAM per GPU, 30 GPU-hrs/week cap, 12h max session length.** Planned spend is ~21-25 GPU hours across 22 experiments (Weeks 1-4), with buffer reserved in Week 5. See `EXPERIMENT_MATRIX.md` for the full spec and the VRAM math behind every technique/model pairing — never schedule a technique against a model without checking it fits 16GB first (full-FP32 fine-tuning and fp16 LoRA on the 13B model do not fit; that's why they're excluded — see ADR-001/ADR-002 in `ARCHITECTURE.md`).

**Before executing any experiment:**

1. **Memory Projection:** Estimate peak VRAM from config + model size. Compare against 16GB with **real headroom** (not raw total). If projection is close or exceeds limit, do not attempt — either shrink the config or escalate to the engineer as a decision point.

2. **Time Projection:** Estimate GPU hours from `EXPERIMENT_MATRIX.md`'s estimates. Check against remaining weekly budget (30h cap). If overshooting, stop and escalate rather than silently compressing.

3. **Disk Space:** Ensure outputs directory has room for checkpoints + CSVs. Monitor and clean between phases if needed.

4. **Failure Recovery Plan:** Identify what to do if this experiment times out, OOMs, or produces NaN losses (Kaggle sessions cap at 12h — checkpoint accordingly). Have a recovery procedure ready *before* starting.

**If constraints are about to be violated, stop and escalate explicitly.**

Do not silently substitute lower-fidelity alternatives (e.g., smaller dataset, fewer epochs) — that changes the experiment validity. Raise it as a decision point instead.

---

## Experiment Workflow

Every experiment follows this discipline:

```text
Verify Preconditions
↓ Config matches EXPERIMENT_MATRIX.md? Data ready? Tracking set up? VRAM math checked?

Execute Experiment
↓ Run exactly as configured. Log start time, end time, errors.

Capture Results
↓ Save metrics to CSV. Save any errors/warnings.

Validate Results
↓ Do the numbers make sense? VRAM realistic? Latency realistic? Quality degradation acceptable?

Log Outcome
↓ Update experiment_tracking.csv. Note any issues.
↓ If recovery was needed, document what went wrong and how it was fixed.

Move to Next Experiment
↓ Only if this week's quality gates passed.
```

**Do not skip validation.** If results look wrong, rerun or investigate before accepting them.

---

## Phase Transitions Require Gate Approval

Each week ends with a quality gate (see `EXPERIMENT_MATRIX.md` "Quality Gates" section for the current thresholds).

**Gate approval checklist:**

- [ ] All experiments for this phase completed
- [ ] No VRAM exceeded 16GB
- [ ] No NaN losses or divergences
- [ ] All CSV results present and complete
- [ ] Total GPU hours within weekly budget (30h cap)
- [ ] Experiment tracking log complete
- [ ] No outstanding errors

**Do not move to the next phase without gate approval.**

If gate fails:

1. Identify which experiments failed/are outstanding
2. Implement recovery (retry, different hyperparams, skip if unjustifiable)
3. Document what happened and why
4. Re-check gate

Only after gate passes: proceed to next phase.

---

## Benchmarking Is Integral, Not Optional

Every experiment is a benchmark: it compares a technique against the fp16 baseline for that model+dataset, under fixed conditions.

**Baseline:** Unoptimized fp16 zero-shot model on same dataset (established Week 1)

**Measurement:** Latency, VRAM, training time (if applicable), quality degradation

**Interpretation:** Did optimization work? How much speedup? What quality cost? Was it worth it?

**Report all metrics with context:**
- Never report isolated numbers
- Always compare against baseline
- Always interpret results (what does this tell us about the technique?)

Example (good):

> "QLoRA achieves 1.4x training-time speedup over LoRA (2.0hr → 1.5hr) with comparable ROUGE-L on Mistral-7B CNN/DailyMail. Worth the tradeoff given the VRAM headroom it frees up."

Example (bad):

> "Latency is 99ms"

---

## Testing & Validation

Before each week's experiments go into the master CSV:

**Correctness checks:**
- [ ] Config matches `EXPERIMENT_MATRIX.md` (no typos in LR, r, alpha, batch size)
- [ ] Metrics formula correct (speedup = baseline / optimized, not the reverse)
- [ ] Quality comparisons against correct baseline (fp16 baseline, not a different technique)

**Consistency checks:**
- [ ] If this technique ran before, are results consistent (±5%)?
- [ ] Do metrics fall within expected ranges (latency > 0, quality between 0-1)?
- [ ] Do technique relationships hold (QLoRA trains faster than LoRA, 4-bit VRAM < 8-bit VRAM)?

**Leakage checks:**
- [ ] Test data not used during training (splits enforced in `utils/data_loader.py`)
- [ ] Validation set not touched until evaluation
- [ ] No peeking at test-set quality during training

---

## Git Discipline

Git history tells the story of the project. Make it readable.

**Commit frequency:** At least daily, ideally after each completed session (Kaggle or local)

**Commit messages:** Explain *why* this work happened, not just what changed

Good:

```text
Week 2: LoRA fine-tuning, Mistral-7B (both datasets)

Trained Mistral-7B with LoRA (r=8, alpha=16) on CNN/DailyMail and
SQuAD. Both experiments completed within VRAM budget (peak ~9GB,
well under 16GB). Training time ~2hr each, in line with estimate.

- mistral_results.csv: 2 rows added (LoRA technique)
- Adapters saved and verified
- Quality gate passed: proceed to QLoRA
```

Bad:

```text
Add lora results
```

---

## Documentation Standards

Documentation evolves with the project. Keep it current.

**Every phase needs:**
- Week summary (what ran, what the results mean) in `logs/phase_summary.md`
- GPU hours used vs. budgeted
- Any deviations from plan (and why)
- Preview of what's next (update `PROJECT_STATE.md`)

**Every unexpected result needs investigation:**
- Did metrics look unrealistic? Why?
- Was recovery needed? What happened?
- Was the experiment retried? Did results match?

**Every engineering decision that could affect multiple experiments needs an ADR** in `ARCHITECTURE.md` (e.g., quantization bit-width, batch size changes, hyperparameter adjustments, dropping a technique because it doesn't fit VRAM).

---

## Tracking & Logging

Maintain these artifacts as you go:

1. **`logs/experiment_tracking.csv`** — one row per experiment
   - Experiment ID, model, technique, dataset
   - Status (pending/running/completed)
   - GPU hours used, peak VRAM, training time
   - Quality metrics
   - Any errors encountered

2. **`logs/daily_standup.md`** — mark start/end of each working session
   - Experiments planned vs. completed
   - Any issues or recoveries
   - GPU hours used this session
   - Preview of next session

3. **`logs/phase_summary.md`** — at end of each week
   - All experiments status
   - Master CSV updated with the week's results
   - Quality gates check
   - Sign-off to proceed to next phase

This is not bureaucracy. This is how you stay accountable, catch problems early, and have a record for the report.

---

## Kaggle ↔ Local Handoff

This project moves between Kaggle notebooks (GPU) and local VSCode (everything else). Follow `KAGGLE_SYNC.md` at the start and end of every Kaggle session — it's short, read it every time. The short version: code lives in git, Kaggle only pulls and runs it, results/checkpoints get pulled back down and committed, `PROJECT_STATE.md` gets updated before the session ends.

---

## Failure Recovery Is Proceduralized

When things fail, follow these procedures before asking for help:

**CUDA Out Of Memory:**
1. Reduce batch size by 50%
2. Retry with new config
3. If still OOM: increase gradient accumulation
4. If still OOM: reduce `max_seq_length` to 256
5. If none work: skip experiment, document why

**NaN Loss (Training Diverged):**
1. Reduce learning rate by 50%
2. Increase gradient clipping (`max_grad_norm = 0.5`)
3. Retry
4. If still NaN: check for data issues, skip experiment

**Kaggle Session Timeout (12h cap):**
1. Checkpoint every 200 steps during training
2. On restart, resume from last checkpoint
3. If no checkpoint exists: restart with a smaller sample as validation, then scale back up

**Model Silently CPU-Offloaded (Hang, No Error):**
1. Check model parameter device placement immediately after every `from_pretrained(..., device_map="auto")` call (and after applying LoRA adapters) — if anything isn't on `cuda`, fail immediately with a clear error instead of letting `.generate()` run for hours
2. Make sure any model loaded earlier in the same process was fully released first (`del model; gc.collect(); torch.cuda.empty_cache()`) — a leftover model from an earlier experiment in the same script run is the most common cause of a later model not fitting on GPU
3. If it still offloads after a clean process: reduce batch size / `max_seq_length`, or split model loads across separate script invocations
4. Print periodic progress during generation (every N samples) so a stuck run is visible in the Kaggle log instead of going silent — silence for hours is itself a signal something is wrong, not just slow

**Model Download Failed:**
1. Retry with exponential backoff (1s, 2s, 4s, 8s, 16s)
2. If still failing: download locally first using HuggingFace CLI
3. If that fails: manually confirm model exists on HF Hub; escalate

**Results Look Unrealistic:**
1. Double-check metric calculation (speedup formula, etc.)
2. Rerun experiment to verify consistency
3. Check if measurement is capturing the right thing
4. Compare against related experiments (do they tell a consistent story?)
5. Document findings; don't ignore surprising results

---

## When to Propose ADRs

Propose an ADR (in `ARCHITECTURE.md`) when:
- A decision affects multiple experiments or the overall design
- The decision has significant engineering uncertainty
- The rationale might not be obvious to someone reading the code later
- You need to justify a constraint that initially seems arbitrary (e.g., "why does Llama-13B skip LoRA?")

Do not ADR trivial choices (which CSV column goes first, minor code refactoring). ADR important decisions.

---

## When to Challenge Assumptions

Your comfort with having assumptions questioned is what makes this collaboration work.

Claude Code will:
- Point out hardware constraints that aren't being respected
- Raise flags if GPU budget is tightening
- Question if a recovery procedure is sound
- Push back if a phase gate isn't actually passed
- Push back if a plan's numbers don't actually fit the stated hardware — this happened once already (see `PROJECT_STATE.md` Revision Note) and is exactly the kind of thing to catch early next time

This is not questioning your judgment. This is maintaining project integrity.

---

## Evidence Hierarchy

When recommending technical choices, prefer evidence in this order:

1. Assignment requirements (explicit objectives)
2. `EXPERIMENT_MATRIX.md` (the current, hardware-checked plan)
3. Benchmarks collected during this project
4. Production engineering practices (how industry does this)
5. Research papers (academic evidence)
6. Community consensus (what most people do)
7. Intuition (gut feeling)

Clearly distinguish between evidence, inference, and opinion.

If new evidence contradicts an earlier recommendation, update the recommendation. Do not defend earlier decisions simply because they were made earlier.

---

## Session Logging (Prompt History)

Every session must leave a durable, verbatim record independent of `PROJECT_STATE.md`'s summarized notes.

At the start of each session, create a new file: `knowledge/ai-usage-log/YYYY-MM-DD_<phase-description>.md`

That file records:
- Every prompt from the engineer, verbatim, in order — not summarized
- Which resulting code/experiments/docs were AI-generated vs. human-written
- Key decisions made during the session
- Experiments executed (with results)
- Any issues encountered and how they were resolved

This is not optional. It's how you reconstruct exactly what was done and decided later.

---

## This Document

This document defines how Claude Code should collaborate on the LLM Optimization project.

It should remain largely stable throughout the project.

Update it only if:
- Our collaboration reveals something fundamental was missing
- A principle proved wrong in practice
- New constraints emerge that change the workflow

The project itself evolves. Project-specific state, progress, decisions, and phase updates belong in `PROJECT_STATE.md` and `ARCHITECTURE.md`. Experiment specs belong in `EXPERIMENT_MATRIX.md`. Kaggle/local handoff belongs in `KAGGLE_SYNC.md`.

---

## Key Reminders

- **Resource constraints are real.** 16GB VRAM, 30 GPU-hrs/week. Check the math before scheduling, don't assume.
- **Tracking is not optional.** Every experiment gets logged. Every phase gets gated. No surprises at submission time.
- **Recovery procedures exist.** When things fail, follow the procedure before escalating.
- **Reproducibility is non-negotiable.** Same code + same config = same results, always.
- **The plan is detailed but was wrong once already.** `EXPERIMENT_MATRIX.md` was corrected against real hardware limits — trust it, but don't be afraid to correct it again if new evidence contradicts it.
- **Quality gates exist for a reason.** Do not skip them.
- **PROJECT_STATE.md must reflect reality, not intent.** Never write a result that hasn't actually been measured.
