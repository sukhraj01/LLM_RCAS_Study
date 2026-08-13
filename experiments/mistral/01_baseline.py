"""
EXP-MIS-BASE-CNN, EXP-MIS-BASE-SQUAD (see EXPERIMENT_MATRIX.md)
Run this FIRST, before any other Mistral technique — every other script reads
its results back from results/mis_results.csv for comparison.

Run: python experiments/mistral/01_baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `utils`/`experiments` imports

from experiments.common import run_inference_multi_dataset

if __name__ == "__main__":
    # Loads Mistral-7B ONCE and loops both datasets in-memory (see EXPERIMENT_MATRIX.md
    # "Model silently CPU-offloaded / hangs" — reloading per dataset in the same process is
    # what caused EXP-LLAMA-BASE-SQUAD's hang; this is the fix, not the workaround).
    run_inference_multi_dataset(
        model_key="MIS",
        technique="baseline",
        dataset_keys=["CNN", "SQUAD"],
        quant_config=None,
        baseline_lookup=lambda dataset_key: None,  # this IS the baseline
    )
