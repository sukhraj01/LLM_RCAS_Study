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

Resume from a checkpoint that only exists in a Kaggle "Create Notebook from Output" input
mount (read-only — e.g. /kaggle/input/<slug>/repo/checkpoints/llama_qlora_cnn, from an older
notebook version whose output you attached as this session's input). --checkpoint-source
copies that directory's contents into the real (writable) checkpoint dir before training, so
get_last_checkpoint() can find it AND future checkpoints (step 125, final adapter) can be
written normally — training can never resume in place against a read-only input mount:
    python experiments/llama/02_qlora.py --dataset cnn --resume \
        --checkpoint-source /kaggle/input/<slug>/repo/checkpoints/llama_qlora_cnn
"""

import argparse
import os
import shutil
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
    parser.add_argument(
        "--checkpoint-source", default=None,
        help="Path to a directory containing checkpoint-N subfolders to copy into the real "
             "(writable) checkpoint dir before training — e.g. a read-only Kaggle input mount "
             "from an older notebook's output. Requires --resume.",
    )
    args = parser.parse_args()
    if args.resume and not args.dataset:
        parser.error("--resume requires --dataset (a checkpoint directory belongs to one dataset)")
    if args.checkpoint_source and not args.resume:
        parser.error("--checkpoint-source requires --resume (copying it in is pointless without resuming)")
    dataset_keys = [args.dataset.upper()] if args.dataset else ["CNN", "SQUAD"]

    if args.checkpoint_source:
        dest = os.path.join(CHECKPOINTS_DIR, f"llama_qlora_{args.dataset}")
        print(f"Copying checkpoint(s) from read-only source into writable dir: "
              f"{args.checkpoint_source} -> {dest}")
        shutil.copytree(args.checkpoint_source, dest, dirs_exist_ok=True)

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
