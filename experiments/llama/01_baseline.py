"""
EXP-LLAMA-BASE-CNN, EXP-LLAMA-BASE-SQUAD (see EXPERIMENT_MATRIX.md)
Run this FIRST, before any other Llama technique.

NOTE: Llama-2-13b-hf is gated on HuggingFace — you must have requested access
and set HF_TOKEN (see .env.example) before this will load.

Run: python experiments/llama/01_baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import run_inference_only_experiment

if __name__ == "__main__":
    for dataset_key in ("CNN", "SQUAD"):
        run_inference_only_experiment(
            exp_id=f"EXP-LLAMA-BASE-{dataset_key}",
            model_key="LLAMA",
            dataset_key=dataset_key,
            technique="baseline",
            quant_config=None,
            baseline_row=None,
        )
