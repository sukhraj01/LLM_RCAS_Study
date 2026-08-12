"""
EXP-MIS-LORA-CNN, EXP-MIS-LORA-SQUAD (see EXPERIMENT_MATRIX.md technique #2)
Requires 01_baseline.py to have run already (reads its results for comparison).

Run: python experiments/mistral/02_lora.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import load_baseline_metrics, run_training_experiment
from utils.config import LORA_CONFIG, CHECKPOINTS_DIR
import os

if __name__ == "__main__":
    for dataset_key in ("CNN", "SQUAD"):
        baseline = load_baseline_metrics("MIS", dataset_key)
        if baseline is None:
            raise RuntimeError(
                f"No baseline found for MIS/{dataset_key} — run experiments/mistral/01_baseline.py first."
            )
        run_training_experiment(
            exp_id=f"EXP-MIS-LORA-{dataset_key}",
            model_key="MIS",
            dataset_key=dataset_key,
            technique="lora",
            lora_hparams=LORA_CONFIG,
            quant_config=None,  # fp16 base — this is plain LoRA, not QLoRA
            output_dir=os.path.join(CHECKPOINTS_DIR, f"mistral_lora_{dataset_key.lower()}"),
            baseline_row=baseline,
        )
