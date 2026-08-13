"""
EXP-LLAMA-BASE-CNN, EXP-LLAMA-BASE-SQUAD (see EXPERIMENT_MATRIX.md)
Run this FIRST, before any other Llama technique.

NOTE: Llama-2-13b-hf is gated on HuggingFace — you must have requested access
and set HF_TOKEN (see .env.example) before this will load.

Run: python experiments/llama/01_baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import run_inference_multi_dataset

if __name__ == "__main__":
    # Loads Llama-2-13B ONCE and loops both datasets in-memory. This is the actual fix for the
    # CPU-offload hang on EXP-LLAMA-BASE-SQUAD — reloading a second 13B instance in the same
    # process was the root cause (allocator fragmentation), not just missing cleanup; see
    # EXPERIMENT_MATRIX.md "Model silently CPU-offloaded / hangs".
    run_inference_multi_dataset(
        model_key="LLAMA",
        technique="baseline",
        dataset_keys=["CNN", "SQUAD"],
        quant_config=None,
        baseline_lookup=lambda dataset_key: None,
    )
