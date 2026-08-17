# Session Log — 2026-08-17 — Limitations Section, Expanded T4/Turing Finding, ADR-005

## Prompts (verbatim, in order)

### Prompt 1

> Add proper documentation for two things discussed but not yet written
> down anywhere in the project:
>
> 1. A "Limitations" section (put it wherever fits the existing structure
>    best - EXPERIMENT_MATRIX.md or PROJECT_STATE.md, your call) covering:
>    - Single hardware target: all experiments so far ran on Kaggle's T4
>      GPU. Findings are scoped to T4 and should not be generalized to GPUs
>      broadly without saying so explicitly.
>    - Single run per experiment, no repeated trials/seeds: every result is
>      a point estimate, no confidence intervals or statistical testing.
>      Reasonable given the GPU-hour budget, but should be stated plainly,
>      not implied to be more statistically rigorous than it is.
>    - Small scale: 1000 training examples, 2 epochs, 200 test samples per
>      experiment. Conclusions apply at this scale; may not generalize
>      directly to production-scale fine-tuning (larger datasets, more
>      epochs).
>    - The T4/Turing quantization-slowdown explanation is a plausible
>      hypothesis consistent with the observed pattern across 4 data
>      points, not a profiled/verified root cause (no kernel-level
>      profiling was done). State it as "plausibly because," not "because."
>
> 2. Expand the existing T4/Turing quantization finding write-up in
>    EXPERIMENT_MATRIX.md with two additions:
>    - Relevance: T4 is a widely-deployed real-world resource-constrained
>      GPU (AWS G4 instances, Google Cloud, Kaggle/Colab free tier) - this
>      is directly applicable to a real population of resource-constrained
>      deployments, not an artifact of whatever GPU happened to be free.
>    - Context: this refines rather than contradicts actual ML systems
>      understanding - bitsandbytes' own documentation already notes that
>      quantization speed benefits depend on the GPU having proper tensor
>      core support for low-precision ops. The popular assumption being
>      contradicted is the oversimplified version ("quantization = always
>      faster"), not the nuanced technical consensus.
>
> Also check ARCHITECTURE.md's existing ADR pattern - if the "T4 as sole
> hardware target, with these validity implications" decision fits as a
> proper ADR alongside ADR-001 through ADR-004, add it there too, so it's
> captured the same way other scope decisions have been.
>
> Commit when ready.

## AI-Generated vs Human-Written

All content this session is AI-generated documentation (no code changes, no new experiment results). The specific bullet points, framing, and phrasing requirements (e.g. "plausibly because" not "because") came directly from the engineer's prompt; the surrounding prose, exact placement, and cross-referencing between files were written by Claude Code.

- `EXPERIMENT_MATRIX.md` — new `## Limitations` section (4 numbered points) added after the existing "Qualitative Notes for Report" content, before "Recovery Procedures"; new "Relevance" and "Context" paragraphs inserted into the existing "Project-Level Finding: T4/Turing Quantization Slowdown" section; the section's opening "Likely cause:" sentence reworded to "Plausible cause (a hypothesis..., not a profiled or kernel-level-verified root cause...)" for internal consistency with the new Limitations point #4, rather than leaving one part of the document hedged and another asserting the same claim as settled fact.
- `ARCHITECTURE.md` — new `## ADR-005: T4 is the sole hardware target for all 22 experiments`, matching the existing ADR-001 through ADR-004 `**Decision:**`/`**Why:**` format; existing "Known Limitations" bullet list's hardware bullet expanded with a pointer to `EXPERIMENT_MATRIX.md`'s new detailed section (to avoid the two limitations write-ups drifting out of sync over time).
- `PROJECT_STATE.md` — new ADR-005 row added to the "Recent Decisions (ADRs)" summary table, matching the existing ADR-001 through ADR-004 row format and length.
- This file.

## Key Decisions

- **Chose `EXPERIMENT_MATRIX.md` over `PROJECT_STATE.md` for the Limitations section**, per the engineer's explicit "your call." Reasoning: `PROJECT_STATE.md` is described in its own header as reflecting "reality, not the plan" — a live status/log document — whereas `EXPERIMENT_MATRIX.md` already contains "Quality Gates" and "Qualitative Notes for Report," both explicitly report-facing, stable, methodology-adjacent content. A Limitations section is exactly that category: stable methodological caveats meant to travel into the final technical report, not a snapshot of current status. Also noted that `ARCHITECTURE.md` already has a short "Known Limitations" bullet list (predating this session) that's about high-level project scope (2 models, 2 datasets, no full FT) rather than the methodological/statistical-rigor limitations being added now — updated that existing bullet to point at the new detailed section instead of writing a second, competing summary that could drift out of sync.
- **Reworded the existing "Likely cause:" sentence in the T4/Turing finding, not just added a new disclaimer elsewhere.** The engineer's ask (Limitations point 4) was specifically that the report say "plausibly because," not "because" — but the *existing* text in the Project-Level Finding section itself still read fairly assertively ("Likely cause: T4's Turing architecture lacks efficient native int8/int4 tensor-core paths... so its dequantize-on-the-fly kernels do real extra work"). Adding a Limitations-section caveat elsewhere while leaving the original claim's own wording unchanged would have left an internal inconsistency — a reader hitting the Project-Level Finding section first would still read it as a stated fact. Edited the sentence in place to explicitly frame it as "a hypothesis consistent with the observed pattern, not a profiled or kernel-level-verified root cause," and to separate two claims that were previously blurred together: the four-measurement *pattern* being confirmed (true) vs. the *architectural explanation* for the pattern being confirmed (not true, no profiling was done).
- **Grounded the "Relevance" paragraph's real-world claim (T4 on AWS G4/GCP/Kaggle/Colab) without inventing anything beyond what the engineer stated** — the specific platform names were given directly in the prompt; did not add unverified specifics (e.g., exact instance pricing, market-share claims) beyond what was provided.
- **Wrote the "Context" paragraph to make a precise distinction the engineer asked for**: the finding contradicts the *popular oversimplification* ("quantization = always faster"), not the *nuanced practitioner consensus* (which already caveats speed benefits on tensor-core support, per `bitsandbytes`' own docs). Framed this project's contribution as turning a known qualitative caveat into a concrete, four-technique-consistent quantitative measurement, rather than claiming to have discovered something novel — an accurate, not inflated, framing of what four single-run data points on one hardware class actually establish.
- **Did not add ADR-005 to `ARCHITECTURE.md` reflexively just because a template existed** — checked whether the "T4 as sole hardware target" decision genuinely fit the ADR pattern (a decision with project-wide consequences and non-obvious rationale, per `CLAUDE.md`'s "When to Propose ADRs" guidance) before adding it. It does: it's a real, consequential scope decision (not evaluated on a second hardware class) with a rationale that isn't obvious from the code alone (T4 being representative of real deployments, not just "whatever Kaggle gives you"), so it earned a full ADR entry rather than being folded into an existing one.
- **Cross-linked the three new/updated pieces (`ARCHITECTURE.md` ADR-005, `PROJECT_STATE.md`'s ADR table row, `EXPERIMENT_MATRIX.md`'s Limitations section) to point at each other** rather than each restating the full rationale independently, consistent with this project's existing pattern of `PROJECT_STATE.md` ADR-table rows being short summaries pointing to `ARCHITECTURE.md`'s full ADR text.
- **Committed at the end**, per the engineer's explicit "Commit when ready" instruction — no prior "report back first" round-trip requested this time.

## Experiments Executed

None — this was a documentation-only session, no code changes, no Kaggle session.

## Issues Encountered

None. Straightforward documentation addition; no blockers, no unexpected findings requiring investigation.
