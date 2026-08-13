"""
EXP-MIS-QLORA-CNN, EXP-MIS-QLORA-SQUAD (see EXPERIMENT_MATRIX.md technique #3)
Requires 01_baseline.py to have run already.

Run: python experiments/mistral/03_qlora.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import require_baseline_metrics, run_training_multi_dataset
from utils.config import CHECKPOINTS_DIR, get_qlora_config

if __name__ == "__main__":
    qlora_hparams = get_qlora_config("MIS")
    # Loads Mistral-7B (4-bit NF4 base) ONCE, applies a fresh LoRA adapter per dataset.
    run_training_multi_dataset(
        model_key="MIS",
        technique="qlora",
        dataset_keys=["CNN", "SQUAD"],
        lora_hparams=qlora_hparams,
        quant_config=qlora_hparams,  # 4-bit base is baked into qlora_hparams
        output_dir_fn=lambda dataset_key: os.path.join(CHECKPOINTS_DIR, f"mistral_qlora_{dataset_key.lower()}"),
        baseline_lookup=lambda dataset_key: require_baseline_metrics("MIS", dataset_key),
    )
