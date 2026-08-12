# Archive — Read This First

`HYBRID_APPROACH_DETAILED_IMPLEMENTATION_GUIDE.md` in this folder is the **original** plan, kept for reference only.

**Do not use its numbers.** It assumed 24GB GPUs and full-FP32 fine-tuning of 7B/13B models — neither fits Kaggle's actual free-tier hardware (16GB VRAM, 30 GPU-hrs/week cap). It also budgeted 56 GPU-hours in a single week against that 30-hour cap. See `../PROJECT_STATE.md` ("Revision Note") for the full explanation of what was wrong and why.

**The corrected, hardware-checked plan lives in:**
- `../EXPERIMENT_MATRIX.md` — the 22-experiment matrix and hyperparameters that replace this guide's Section 3
- `../ARCHITECTURE.md` — system design and ADRs explaining what changed and why
- `../PROJECT_STATE.md` — current live status

**What's still worth reading in the old guide** (qualitative content unaffected by the hardware error):
- Section 2 (Model Selection & Justification) — the rationale for why Mistral/Llama/MPT/CodeLlama were chosen is still reasonable reading, even though the current plan only uses 2 of the 4 models
- Section 6 (Quality Gates philosophy) and Section 8 (Failure Recovery Procedures) — the structure and reasoning are sound, specific numbers were carried forward and corrected into `EXPERIMENT_MATRIX.md`

Skip Sections 3, 4, and 7 (Experiment Matrix, Phase Breakdown, Tracking Schema with the old GPU-hour numbers) — those are fully superseded.
