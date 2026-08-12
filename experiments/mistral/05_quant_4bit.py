"""
EXP-MIS-4BIT-CNN, EXP-MIS-4BIT-SQUAD (see EXPERIMENT_MATRIX.md technique #5)
Inference only, on the base pretrained model. Requires 01_baseline.py first.

Run: python experiments/mistral/05_quant_4bit.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import load_baseline_metrics, run_inference_only_experiment
from utils.config import QUANT_4BIT_CONFIG

if __name__ == "__main__":
    for dataset_key in ("CNN", "SQUAD"):
        baseline = load_baseline_metrics("MIS", dataset_key)
        if baseline is None:
            raise RuntimeError(
                f"No baseline found for MIS/{dataset_key} — run experiments/mistral/01_baseline.py first."
            )
        run_inference_only_experiment(
            exp_id=f"EXP-MIS-4BIT-{dataset_key}",
            model_key="MIS",
            dataset_key=dataset_key,
            technique="4bit",
            quant_config=QUANT_4BIT_CONFIG,
            baseline_row=baseline,
        )
