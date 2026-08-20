"""
EXP-LLAMA-QLORA-CNN, EXP-LLAMA-QLORA-SQUAD (see EXPERIMENT_MATRIX.md technique #3)
Llama-2-13B does NOT run fp16 LoRA (doesn't fit 16GB — see ADR-002 in ARCHITECTURE.md).
QLoRA is the only fine-tuning technique run on this model. Requires 01_baseline.py first.

Run both datasets in one process (projected ~15-20h combined per EXPERIMENT_MATRIX.md's
"Llama-2-13B QLoRA time projection" — exceeds Kaggle's 12h session cap as a single
unattended run; a session-cap kill mid-CNN on 2026-08-19/20 is why --dataset exists below):
    python experiments/llama/02_qlora.py

Run a single dataset in isolation — recommended, e.g. SQuAD first as the calibration run
per EXPERIMENT_MATRIX.md's recommendation (its projected range fits the 12h cap with margin):
    python experiments/llama/02_qlora.py --dataset squad
    python experiments/llama/02_qlora.py --dataset cnn

Resume a single dataset from its last surviving checkpoint (e.g. after a session-cap kill —
requires --dataset, since a checkpoint directory belongs to exactly one dataset; fails loudly
via Trainer's own resume_from_checkpoint=True if no checkpoint-N exists under that dataset's
output dir; does NOT resume by default even if a checkpoint exists — must be requested):
    python experiments/llama/02_qlora.py --dataset cnn --resume
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import require_baseline_metrics, run_training_multi_dataset
from utils.config import CHECKPOINTS_DIR, get_qlora_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["cnn", "squad"], default=None,
        help="Run only this dataset. Omit to run both (CNN then SQuAD) in one process.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the latest checkpoint-N under --dataset's checkpoint directory "
             "instead of training fresh. Requires --dataset. Off by default.",
    )
    args = parser.parse_args()
    if args.resume and not args.dataset:
        parser.error("--resume requires --dataset (a checkpoint directory belongs to one dataset)")
    dataset_keys = [args.dataset.upper()] if args.dataset else ["CNN", "SQUAD"]

    qlora_hparams = get_qlora_config("LLAMA")  # smaller batch, more accumulation than Mistral
    # Loads Llama-2-13B (4-bit NF4 base) ONCE, applies a fresh LoRA adapter per dataset — this
    # is exactly the pattern that would otherwise hit the same CPU-offload failure the baseline
    # script hit (see run_training_multi_dataset docstring).
    run_training_multi_dataset(
        model_key="LLAMA",
        technique="qlora",
        dataset_keys=dataset_keys,
        lora_hparams=qlora_hparams,
        quant_config=qlora_hparams,
        output_dir_fn=lambda dataset_key: os.path.join(CHECKPOINTS_DIR, f"llama_qlora_{dataset_key.lower()}"),
        baseline_lookup=lambda dataset_key: require_baseline_metrics("LLAMA", dataset_key),
        resume=args.resume,
    )
