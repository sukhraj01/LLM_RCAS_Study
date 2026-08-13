"""
EXP-MIS-8BIT-CNN, EXP-MIS-8BIT-SQUAD (see EXPERIMENT_MATRIX.md technique #4)
Inference only, on the base pretrained model (not fine-tuned) — isolates the
quality/latency/VRAM cost of quantization alone. Requires 01_baseline.py first.

Run: python experiments/mistral/04_quant_8bit.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import require_baseline_metrics, run_inference_multi_dataset
from utils.config import QUANT_8BIT_CONFIG

if __name__ == "__main__":
    run_inference_multi_dataset(
        model_key="MIS",
        technique="8bit",
        dataset_keys=["CNN", "SQUAD"],
        quant_config=QUANT_8BIT_CONFIG,
        baseline_lookup=lambda dataset_key: require_baseline_metrics("MIS", dataset_key),
    )
