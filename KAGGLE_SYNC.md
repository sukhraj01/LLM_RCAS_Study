# Kaggle ↔ Local Sync Protocol

This project splits across two environments: **Kaggle notebooks** (GPU, where experiments actually run) and **local VSCode** (where code is written, reviewed, and Claude Code sessions mostly happen). The two environments don't share a filesystem or a Claude Code context. This file is the bridge — follow it every time you switch, so no state gets lost and no Claude Code session has to guess what happened on the other side.

**Rule of thumb: git is the source of truth for code. `PROJECT_STATE.md` is the source of truth for status. Kaggle is disposable compute — nothing should exist only inside a Kaggle notebook.**

---

## Starting a Kaggle Session

1. Pull latest from git (Kaggle notebook's first cell: `git clone` or `git pull` your repo, or upload the current `experiments/`, `utils/`, `EXPERIMENT_MATRIX.md` as a Kaggle Dataset input if you're not running git directly in-notebook).
2. Read `PROJECT_STATE.md`'s "What's Next" section — confirm what this session is supposed to run before starting anything.
3. Confirm remaining weekly GPU quota (Kaggle shows this in the notebook settings panel) against the GPU-hour estimate for what you're about to run, from `EXPERIMENT_MATRIX.md`.
4. Run the experiment script(s) exactly as specified — no ad-hoc hyperparameter changes inside the notebook. If something needs to change, that's a decision for `ARCHITECTURE.md` (ADR), not a quiet notebook edit.

## During a Kaggle Session

- Checkpoint every 200 steps (Kaggle sessions cap at 12h and can disconnect without warning).
- Write results to CSV as you go, not just at the end — if the session dies, you want partial results, not nothing.
- If something fails, note it immediately in the notebook's own cell output (you'll transcribe it to `logs/daily_standup.md` at the end) — don't rely on remembering it later.

## Ending a Kaggle Session

1. Download from the Kaggle notebook: result CSVs, updated `logs/experiment_tracking.csv` rows, any checkpoints/adapters worth keeping (adapters are small, ~50-200MB — full checkpoints are not, don't casually download 10GB+ files unless you actually need them locally).
2. Commit those files to git from local (or push directly from the Kaggle notebook if you have git configured there — either works, just do it before closing the tab).
3. Update `PROJECT_STATE.md`:
   - Move completed experiments from "pending" to their real status in the Component Status table
   - Update GPU Budget Tracking with actual hours used (not the estimate — the real number)
   - Update "What's Next"
4. Append to `logs/daily_standup.md`: what ran, what the actual numbers were, GPU hours used, any issues.
5. If a new Claude Code chat will pick this up next (common if switching context to avoid pollution): the next session should only need to read `PROJECT_STATE.md` to know exactly where things stand. If it doesn't have enough info, that's a bug in this step — fix `PROJECT_STATE.md`, don't rely on chat memory.

---

## Starting a Local (VSCode) Session

1. `git pull` to get anything committed from the last Kaggle session.
2. Read `PROJECT_STATE.md` — same as above, it should already reflect the latest Kaggle results.
3. Local work is: writing/reviewing experiment scripts, updating `utils/`, building the API/dashboard (Weeks 6-7), writing the report (Weeks 8-9), fixing bugs found in Kaggle output. Local does not run GPU experiments — no GPU here.

## Ending a Local Session

1. Commit and push.
2. Update `PROJECT_STATE.md` if anything about status/plan changed.
3. If the next step is "run this on Kaggle," leave a clear note in `PROJECT_STATE.md`'s "What's Next" — exact script name, exact experiment IDs from `EXPERIMENT_MATRIX.md`.

---

## Starting a New Claude Code Chat (context reset)

You mentioned wanting to switch chats often to avoid context pollution — that's a good instinct given how much this project involves. Here's what makes that actually work:

1. New chat's first message should point Claude Code at the repo and say "read `PROJECT_STATE.md`."
2. Claude Code should NOT need to read `archive/HYBRID_APPROACH_DETAILED_IMPLEMENTATION_GUIDE.md` (the old 2000+ line guide) in a normal session. If it seems to need to, that's a sign something important got left out of `EXPERIMENT_MATRIX.md` or `ARCHITECTURE.md` and those files should be updated instead.
3. If picking up mid-experiment (e.g. "the QLoRA run OOM'd, help me fix it"), the relevant context is: the exact experiment ID, its spec in `EXPERIMENT_MATRIX.md`, and the error. That's it — no need to re-explain the whole project.
4. Before ending a chat that made a decision (dropped a technique, changed a hyperparameter, changed scope), make sure that decision is written into `ARCHITECTURE.md` (ADR) or `PROJECT_STATE.md` before the chat closes. If it's only in the chat transcript, it's lost the moment you switch.

---

## What Never Lives Only in One Place

| Thing | Lives in | Never only in |
|-------|----------|----------------|
| Code | git repo | a Kaggle notebook cell |
| Current status | `PROJECT_STATE.md` | a chat transcript |
| Hyperparameters | `EXPERIMENT_MATRIX.md` | a notebook variable typed from memory |
| Decisions/rationale | `ARCHITECTURE.md` (ADRs) | your own memory of "why we did that" |
| Raw prompts/session history | `knowledge/ai-usage-log/` | nowhere else — this is the only copy |
