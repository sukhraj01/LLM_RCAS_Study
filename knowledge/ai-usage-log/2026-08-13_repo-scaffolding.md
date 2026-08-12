# Session Log — 2026-08-13 — Repo Scaffolding

## Prompts (verbatim, in order)

### Prompt 1

> how to start workignoon vscodenow tell me doijaks thimto create lal thefinit fileslike toml etc from here guide me i hae kept all the aobe fiels in my main direcotyr

(Interpreted as: how to start working in VSCode now — create the init/config files like `pyproject.toml` etc. Engineer confirmed the 5 planning `.md` files from the previous session were saved in their main project directory, `/Users/test01/Desktop/independent_study`.)

## AI-Generated vs Human-Written

All files created this session are AI-generated first drafts:
- `pyproject.toml`, `requirements.txt`, `.gitignore`, `.env.example`, `README.md`
- `utils/config.py`, `utils/data_loader.py`, `utils/metrics.py`, `utils/validation.py`
- `experiments/common.py` and the 11 per-technique scripts (`experiments/mistral/*.py`, `experiments/llama/*.py`)
- `logs/experiment_tracking.csv`, `logs/daily_standup.md`, `logs/phase_summary.md`
- `knowledge/ai-usage-log/` (this file and its README)

**None of this code has run yet.** It's a first-draft skeleton matching `EXPERIMENT_MATRIX.md`'s hyperparameters — expect to debug library-version-specific API issues (transformers/peft/bitsandbytes/optimum move fast) on the first real Kaggle run. That debugging, and any resulting fixes, should be logged in the next session's entry as human-verified/human-fixed code.

## Key Decisions

- Connected the Cowork session to the engineer's local `independent_study` folder to write files directly rather than pasting code blocks for manual copy-paste.
- Structured `experiments/` as thin per-technique scripts calling into a shared `experiments/common.py`, so measurement logic (latency/VRAM tracking, quality metrics, baseline comparison) is written once and identical across all 22 experiments, per `CLAUDE.md`'s "always compare against the same baseline, measured the same way" rule.
- `run_inference_only_experiment`/`run_training_experiment` write to `results/<model>_results.csv` and read baseline rows back from that same CSV (via `load_baseline_metrics`), so technique scripts can run in separate Kaggle sessions from the baseline script without losing the comparison.

## Experiments Executed

None — scaffolding only, no GPU session run yet.

## Issues Encountered

None yet — first real issues expected on first Kaggle run (see "AI-Generated vs Human-Written" above for the expectation-setting on this).
