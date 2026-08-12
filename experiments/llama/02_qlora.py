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

from experiments.common import load_baseline_metrics, run_training_experiment
from utils.config import CHECKPOINTS_DIR, get_qlora_config

if __name__ == "__main__":
    qlora_hparams = get_qlora_config("LLAMA")  # smaller batch, more accumulation than Mistral
    for dataset_key in ("CNN", "SQUAD"):
        baseline = load_baseline_metrics("LLAMA", dataset_key)
        if baseline is None:
            raise RuntimeError(
                f"No baseline found for LLAMA/{dataset_key} — run experiments/llama/01_baseline.py first."
            )
        run_training_experiment(
            exp_id=f"EXP-LLAMA-QLORA-{dataset_key}",
            model_key="LLAMA",
            dataset_key=dataset_key,
            technique="qlora",
            lora_hparams=qlora_hparams,
            quant_config=qlora_hparams,
            output_dir=os.path.join(CHECKPOINTS_DIR, f"llama_qlora_{dataset_key.lower()}"),
            baseline_row=baseline,
        )
