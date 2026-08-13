"""
EXP-MIS-LORA-CNN, EXP-MIS-LORA-SQUAD (see EXPERIMENT_MATRIX.md technique #2)
Requires 01_baseline.py to have run already (reads its results for comparison).

Run: python experiments/mistral/02_lora.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os

from experiments.common import require_baseline_metrics, run_training_multi_dataset
from utils.config import LORA_CONFIG, CHECKPOINTS_DIR

if __name__ == "__main__":
    # Loads Mistral-7B ONCE, applies a fresh LoRA adapter per dataset (see
    # run_training_multi_dataset docstring for why datasets can't share one adapter).
    run_training_multi_dataset(
        model_key="MIS",
        technique="lora",
        dataset_keys=["CNN", "SQUAD"],
        lora_hparams=LORA_CONFIG,
        quant_config=None,  # fp16 base — this is plain LoRA, not QLoRA
        output_dir_fn=lambda dataset_key: os.path.join(CHECKPOINTS_DIR, f"mistral_lora_{dataset_key.lower()}"),
        baseline_lookup=lambda dataset_key: require_baseline_metrics("MIS", dataset_key),
    )
