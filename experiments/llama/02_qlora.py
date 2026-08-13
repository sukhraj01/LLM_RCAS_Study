"""
EXP-LLAMA-QLORA-CNN, EXP-LLAMA-QLORA-SQUAD (see EXPERIMENT_MATRIX.md technique #3)
Llama-2-13B does NOT run fp16 LoRA (doesn't fit 16GB — see ADR-002 in ARCHITECTURE.md).
QLoRA is the only fine-tuning technique run on this model. Requires 01_baseline.py first.

Run: python experiments/llama/02_qlora.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import require_baseline_metrics, run_training_multi_dataset
from utils.config import CHECKPOINTS_DIR, get_qlora_config

if __name__ == "__main__":
    qlora_hparams = get_qlora_config("LLAMA")  # smaller batch, more accumulation than Mistral
    # Loads Llama-2-13B (4-bit NF4 base) ONCE, applies a fresh LoRA adapter per dataset — this
    # is exactly the pattern that would otherwise hit the same CPU-offload failure the baseline
    # script hit (see run_training_multi_dataset docstring).
    run_training_multi_dataset(
        model_key="LLAMA",
        technique="qlora",
        dataset_keys=["CNN", "SQUAD"],
        lora_hparams=qlora_hparams,
        quant_config=qlora_hparams,
        output_dir_fn=lambda dataset_key: os.path.join(CHECKPOINTS_DIR, f"llama_qlora_{dataset_key.lower()}"),
        baseline_lookup=lambda dataset_key: require_baseline_metrics("LLAMA", dataset_key),
    )
