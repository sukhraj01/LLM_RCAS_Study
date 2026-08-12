"""
EXP-MIS-QLORA-CNN, EXP-MIS-QLORA-SQUAD (see EXPERIMENT_MATRIX.md technique #3)
Requires 01_baseline.py to have run already.

Run: python experiments/mistral/03_qlora.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import load_baseline_metrics, run_training_experiment
from utils.config import CHECKPOINTS_DIR, get_qlora_config

if __name__ == "__main__":
    qlora_hparams = get_qlora_config("MIS")
    for dataset_key in ("CNN", "SQUAD"):
        baseline = load_baseline_metrics("MIS", dataset_key)
        if baseline is None:
            raise RuntimeError(
                f"No baseline found for MIS/{dataset_key} — run experiments/mistral/01_baseline.py first."
            )
        run_training_experiment(
            exp_id=f"EXP-MIS-QLORA-{dataset_key}",
            model_key="MIS",
            dataset_key=dataset_key,
            technique="qlora",
            lora_hparams=qlora_hparams,
            quant_config=qlora_hparams,  # 4-bit base is baked into qlora_hparams
            output_dir=os.path.join(CHECKPOINTS_DIR, f"mistral_qlora_{dataset_key.lower()}"),
            baseline_row=baseline,
        )
