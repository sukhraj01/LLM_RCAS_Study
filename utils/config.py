"""
Single source of truth for hyperparameters, mirroring EXPERIMENT_MATRIX.md exactly.

Rule: if you change a number here, change it in EXPERIMENT_MATRIX.md too (and vice versa).
Every experiment script should import from this file rather than hardcoding values —
that's what makes "config matches EXPERIMENT_MATRIX.md" (a CLAUDE.md checklist item)
checkable instead of a matter of trust.
"""

import os

SEED = 42

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS = {
    "MIS": "mistralai/Mistral-7B-v0.1",
    "LLAMA": "meta-llama/Llama-2-13b-hf",
}

# Techniques each model runs — Llama skips fp16 LoRA (doesn't fit 16GB VRAM,
# see EXPERIMENT_MATRIX.md "Hardware Reality"). Keep this in sync with the matrix.
TECHNIQUES_BY_MODEL = {
    "MIS": ["baseline", "lora", "qlora", "8bit", "4bit", "onnx"],
    "LLAMA": ["baseline", "qlora", "8bit", "4bit", "onnx"],
}

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

DATASETS = {
    "CNN": {
        "hf_name": "cnn_dailymail",
        "hf_config": "3.0.0",
        "task": "summarization",
        "text_field": "article",
        "target_field": "highlights",
        "n_train": 1000,
        "n_val": 200,
        "n_test": 200,
    },
    "SQUAD": {
        "hf_name": "squad",
        "hf_config": None,
        "task": "qa",
        "n_train": 1000,
        "n_val": 200,
        "n_test": 200,
    },
}

MAX_SEQ_LENGTH = 512

# ---------------------------------------------------------------------------
# Technique hyperparameters
# ---------------------------------------------------------------------------

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
    # training args
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "num_train_epochs": 2,
    "optim": "paged_adamw_8bit",
    "gradient_checkpointing": True,
}

QLORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
    # 4-bit base quantization
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    # training args — differ by model, see per-model overrides below
    "learning_rate": 2e-4,
    "num_train_epochs": 2,
    "optim": "paged_adamw_8bit",
    "gradient_checkpointing": True,
}

# Per-model overrides for QLoRA batch size (Llama-13B needs a smaller batch + more
# accumulation to stay under 16GB — see EXPERIMENT_MATRIX.md technique #3)
QLORA_BATCH_OVERRIDES = {
    "MIS": {"batch_size": 2, "gradient_accumulation_steps": 8},
    "LLAMA": {"batch_size": 1, "gradient_accumulation_steps": 16},
}


def get_qlora_config(model_key: str) -> dict:
    """QLORA_CONFIG merged with the per-model batch/accumulation override."""
    cfg = dict(QLORA_CONFIG)
    cfg.update(QLORA_BATCH_OVERRIDES[model_key])
    return cfg


QUANT_8BIT_CONFIG = {"load_in_8bit": True}

QUANT_4BIT_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16",
}

# ---------------------------------------------------------------------------
# GPU-hour estimates (for sanity-checking actual runs against plan — see
# EXPERIMENT_MATRIX.md "Full Matrix" table for the authoritative per-experiment list)
# ---------------------------------------------------------------------------

ESTIMATED_GPU_HOURS_TOTAL = 21.3  # core 22 experiments, before buffer

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

# Kaggle free tier = T4 only, the sole hardware target for this project (see ADR-005 in
# ARCHITECTURE.md). save_result() stamps every result row with this so it's never left
# blank for a new experiment to backfill by hand later — see EXPERIMENT_MATRIX.md
# Limitations #1. If a second GPU architecture is ever added (e.g. Ada cluster access),
# this becomes a per-run value instead of a flat constant.
HARDWARE = "T4"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# ---------------------------------------------------------------------------
# HF auth (Llama-2 is gated — see .env.example)
# ---------------------------------------------------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN")
