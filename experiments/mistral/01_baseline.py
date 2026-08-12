"""
EXP-MIS-BASE-CNN, EXP-MIS-BASE-SQUAD (see EXPERIMENT_MATRIX.md)
Run this FIRST, before any other Mistral technique — every other script reads
its results back from results/mis_results.csv for comparison.

Run: python experiments/mistral/01_baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `utils`/`experiments` imports

from experiments.common import run_inference_only_experiment

if __name__ == "__main__":
    for dataset_key in ("CNN", "SQUAD"):
        run_inference_only_experiment(
            exp_id=f"EXP-MIS-BASE-{dataset_key}",
            model_key="MIS",
            dataset_key=dataset_key,
            technique="baseline",
            quant_config=None,
            baseline_row=None,  # this IS the baseline
        )
