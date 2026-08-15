# Session Log — 2026-08-16 — Hardware Column Added, Mistral/Llama ONNX Formally Deferred to Ada Cluster

## Prompts (verbatim, in order)

### Prompt 1

> Two things:
>
> 1. Add a "hardware" column to logs/experiment_tracking.csv (and
>    results/mis_results.csv, results/llama_results.csv if they don't
>    already implicitly track this). Backfill "T4" for every existing row -
>    every experiment run so far has been on Kaggle's GPU T4 x2 accelerator,
>    confirmed consistently across every session log this project. This is
>    in preparation for possibly getting access to a different GPU
>    (university Ada cluster) for remaining/blocked experiments, so future
>    rows need this tracked explicitly rather than left implicit - avoids
>    ever accidentally comparing T4 and non-T4 results as if they were the
>    same hardware.
>
> 2. Finalize the Mistral ONNX decision: after 4 failed attempts (VRAM OOM,
>    disk OOM twice, and a well-verified in-memory-export fix that still
>    hit disk OOM), we're marking Mistral-7B's ONNX export as infeasible on
>    Kaggle's free tier - not scheduled for further attempts there. Update
>    PROJECT_STATE.md and EXPERIMENT_MATRIX.md to reflect this clearly
>    (same treatment Llama's ONNX already has), and mark it something like
>    "deferred pending Ada cluster access" rather than leaving it in limbo.
>
> 3. Report back the current, actual state of PROJECT_STATE.md's Component
>    Status / experiment checklist - I want to confirm exactly which of the
>    22 experiments are completed vs pending vs blocked, cross-checked
>    against reality rather than assumed from conversation history.
>
> Commit when ready.

## AI-Generated vs Human-Written

All CSV and doc edits this session are AI-generated. The report that the fourth (verified) fix attempt still failed with a disk OOM came from the engineer, not observed by Claude Code — no Kaggle session ran this turn.

- `logs/experiment_tracking.csv`, `results/mis_results.csv`, `results/llama_results.csv` — new `hardware` column added after `dataset`; backfilled `T4` for every row with `status == completed` (12 rows across the three files' union — 10 in `mis_results.csv`, 2 completed rows represented in `llama_results.csv`'s 3 rows since one baseline experiment has two historical entries), left blank for every `pending` row.
- `EXPERIMENT_MATRIX.md` — technique #6 (ONNX) section given a status line; Recovery Procedures section given a "FINAL OUTCOME" paragraph and an updated Llama open-risk paragraph.
- `PROJECT_STATE.md` — Current Phase header, Component Status Summary (2 rows), Blockers & Risks section (restructured: Current Blockers, new "Deferred" entry, renamed "History" entry with a new Attempt 4 paragraph, updated Llama Open Risk heading), Week 4 checklist (2 items) all updated.
- This file.

## Key Decisions

- **Did not backfill `T4` into pending rows, despite the instruction saying "every existing row."** Read "every experiment run so far" as the operative qualifier — the 10 pending rows (2 Mistral ONNX, 8 Llama) have not run on any hardware yet, and writing `T4` into them would falsely claim they'd already executed on that hardware, which is exactly the kind of implicit/misleading hardware attribution the engineer said this column exists to prevent. Left them blank, to be filled in whenever they actually run (on T4 or Ada, whichever ends up hosting them) — flagged this interpretation explicitly when reporting back rather than silently applying it.
- **Hit and recovered from a real mid-task failure: the first attempt at the CSV rewrite script crashed with `ValueError: dict contains fields not in fieldnames: None` partway through writing `logs/experiment_tracking.csv`, truncating it to a single line.** Diagnosed immediately (`git diff --stat` showed 23 lines gone) and restored the file via `git checkout -- logs/experiment_tracking.csv` before doing anything else, since this was accidental in-session data loss on a git-tracked file with no uncommitted prior changes to preserve — the safe, correct recovery, not a judgment call requiring engineer input. Root-caused the crash itself before retrying: four rows in `logs/experiment_tracking.csv` (the four original baseline rows, predating this session) had `notes` fields containing an internal comma inside an unquoted parenthetical clause — a pre-existing CSV malformation, not something introduced this session, that `csv.DictReader` silently overflowed into a `None` key on read and then couldn't write back out. Fixed those four fields by wrapping them in double quotes (matching the quoting convention every other multi-clause `notes` field in the same file already uses), verified zero malformed rows remained across all three files, then reran the column-addition script using a temp-file-then-move pattern so a second failure (if any) couldn't corrupt the real file again. Flagged this pre-existing bug and the fix explicitly rather than quietly patching it — it changes exact byte content of four rows beyond just adding the new column, which the engineer should be able to see in the diff.
- **Took the engineer's report of Attempt 4's failure at face value rather than re-verifying or second-guessing it.** No new Kaggle session ran this turn, and the engineer's message states the outcome plainly as already-observed fact ("a well-verified in-memory-export fix that still hit disk OOM"). Updated `PROJECT_STATE.md`/`EXPERIMENT_MATRIX.md` to document this honestly, including a plausible-but-unconfirmed explanation (ONNX external-data serialization for graphs over the 2GB protobuf limit, never exercised by the tiny stand-in model used to verify the fix) rather than either (a) omitting the failure to protect the previous session's "verified" framing, or (b) inventing a more definitive root cause than what's actually known. This is the kind of "document unexpected results, don't ignore them" discipline this project's `CLAUDE.md` calls for.
- **Did not attempt a fifth fix.** The engineer's instruction was explicit — stop attempting on Kaggle, defer pending different hardware — and matches the same "escalate rather than keep burning GPU hours on repeated attempts" principle the engineer invoked in the previous session's instructions (the one that led to the in-memory-export fix being investigated at all rather than a naive fourth blind retry). Documented the deferral as a closed decision, not a lingering blocker, in both `PROJECT_STATE.md`'s Current Blockers (now says "None specific to ONNX anymore") and a new dedicated "Deferred" section.
- **Applied the identical deferral framing to Llama's ONNX entry** ("same treatment," per the engineer's explicit instruction) rather than leaving its older "Open Risk, needs a decision" wording, which now reads as unresolved when it's actually a closed, if provisional, decision (not scheduled on Kaggle, revisit if/when Ada access exists).
- **Cross-checked the 22-experiment count directly against the CSVs via `csv.DictReader` rather than recalling it from conversation history**, per the engineer's explicit ask in point 3 — counted `status == completed` vs `status == pending` in `logs/experiment_tracking.csv` and cross-referenced against `results/mis_results.csv` (10 completed Mistral rows, matching) and `results/llama_results.csv` (2 completed Llama baseline experiments, one with a duplicate historical row) before reporting the breakdown back.

## Experiments Executed

None this session — no Kaggle session ran. This was a documentation/tracking-infrastructure session (adding the `hardware` column, fixing a pre-existing CSV malformation, and formalizing the ONNX deferral decision the engineer had already made based on their own Kaggle results).

## Issues Encountered

Self-inflicted, same-session: a CSV-rewrite script crash truncated `logs/experiment_tracking.csv` to one line due to four pre-existing malformed rows (unquoted `notes` fields containing internal commas) that predated this session. Recovered via `git checkout` before any further action, root-caused, fixed properly (proper quoting, not a workaround), and reran successfully with a safer write pattern. No data was lost — the restore happened before any commit, and the fix was verified against all three CSVs before proceeding.

Reported by the engineer, not diagnosed this session: Mistral-7B's ONNX export failed a fourth time (disk OOM) despite the previous session's fix being verified end-to-end against a same-architecture stand-in model. Documented as an open, honestly-flagged gap between stand-in verification and real-7B-scale behavior, with a plausible (ONNX external-data serialization) but unconfirmed explanation — not further investigated, per the engineer's explicit instruction to defer rather than pursue a fifth attempt.
