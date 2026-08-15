# LLM Optimization for Resource-Constrained Systems

Independent study project comparing LoRA, QLoRA, 8-bit/4-bit quantization, and ONNX Runtime inference optimization across Mistral-7B and Llama-2-13B on summarization (CNN/DailyMail) and QA (SQuAD).

## Start here

- **`PROJECT_STATE.md`** — current status, read this first, every session (Kaggle, local, or new Claude Code chat)
- **`CLAUDE.md`** — collaboration contract / how Claude Code should work on this repo
- **`ARCHITECTURE.md`** — system design, file structure, ADRs
- **`EXPERIMENT_MATRIX.md`** — the 22 experiments, exact hyperparameters, GPU-hour estimates
- **`KAGGLE_SYNC.md`** — protocol for handing off state between Kaggle and local VSCode

## Setup

### Local (VSCode) — for dev, API, dashboard, report. No GPU needed here.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in HF_TOKEN
```

### Kaggle — for actual GPU experiments

1. New Kaggle Notebook, enable GPU (Settings > Accelerator > GPU T4x2 or P100)
2. Add your `HF_TOKEN` via Notebook Settings > Secrets (don't paste it in a cell)
3. First cell:
   ```bash
   !git clone <your-repo-url> repo
   %cd repo
   !pip install -q -r requirements.txt --no-deps  # --no-deps avoids fighting Kaggle's preinstalled torch/CUDA
   !pip install -U "optimum[onnxruntime-gpu]"  # requirements.txt only pulls CPU onnxruntime (no macOS wheel for -gpu); Kaggle needs this extra step for GPU-accelerated ONNX experiments
   !pip install -U torchao  # Kaggle's base image ships torchao 0.10.0; peft's get_peft_model LoRA dispatch check requires torchao>=0.16.0 or it raises ImportError, even though this project's QLoRA is bitsandbytes-based and never imports torchao directly
   ```
4. Before running any ONNX export script (`experiments/mistral/06_onnx.py`, `experiments/llama/05_onnx.py`), redirect the HF model cache and the ONNX export output off Kaggle's default working directory:
   ```bash
   %env HF_HOME=/opt/bin/hf_cache
   %env ONNX_CACHE_DIR=/opt/bin/onnx_cache
   ```
   Kaggle enforces a fixed ~20GB quota on the notebook's tracked *output* (where `/kaggle/working` lives by default), separate from and much smaller than the raw disk. A model's fp16 HF cache plus its ONNX export easily exceed that on their own for a 7B+ model — `/opt/bin` has ~119GB free and isn't subject to the same output tracking. See `EXPERIMENT_MATRIX.md` Recovery Procedures "ONNX Export Disk OOM" for the full symptom/root-cause writeup.
5. Run the experiment script for whatever's next in `PROJECT_STATE.md` → "What's Next"
6. Before ending the session: follow `KAGGLE_SYNC.md` — download results, commit, update `PROJECT_STATE.md`

## Repo layout

See `ARCHITECTURE.md` "Appendix: File Structure" for the full annotated tree.

```
experiments/{mistral,llama}/   experiment scripts, one per technique
utils/                         data loading, metrics, config, validation — shared code
results/                       per-technique + master benchmark CSVs
logs/                          experiment_tracking.csv, daily_standup.md, phase_summary.md
knowledge/ai-usage-log/        verbatim session logs (see CLAUDE.md "Session Logging")
api/, dashboard/                Week 6-9 deliverables, empty for now
archive/                        superseded original plan — see archive/README.md before reading it
```

## Current status

Not started — see `PROJECT_STATE.md` for the live picture. Week 1 is: repo setup (this), env setup, data download, real baseline runs.
