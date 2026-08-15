"""
EXP-MIS-ONNX-CNN, EXP-MIS-ONNX-SQUAD (see EXPERIMENT_MATRIX.md technique #6)
Exports the baseline fp16 model to ONNX and benchmarks inference speed via
ONNX Runtime's CUDAExecutionProvider. Quality should be ~identical to the
fp16 baseline (ONNX export shouldn't change model outputs materially) — this
technique is about latency/throughput, not quality.

Requires 01_baseline.py to have run already.

Run: python experiments/mistral/06_onnx.py

NOTE: optimum's ONNX export API has moved around across versions — if
`ORTModelForCausalLM.from_pretrained(..., export=True)` doesn't work as
written, check `optimum.onnxruntime` docs for your installed version.
Flag this as the first thing to verify on Kaggle, not something to assume works.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from optimum.exporters.onnx import main_export
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.utils.save_utils import maybe_save_preprocessors
from transformers import AutoTokenizer

from experiments.common import evaluate_quality, format_example, load_baseline_metrics, save_result
from utils.config import CHECKPOINTS_DIR, HF_TOKEN, MAX_SEQ_LENGTH, MODELS
from utils.data_loader import DataLoader
from utils.metrics import quality_degradation_percent, speedup_factor

_loader = DataLoader()

if __name__ == "__main__":
    model_id = MODELS["MIS"]
    # ONNX_CACHE_DIR lets a Kaggle session redirect this off the notebook's default output
    # path (Kaggle enforces a fixed ~20GB quota on tracked output, not the raw disk — see
    # EXPERIMENT_MATRIX.md Recovery Procedures "ONNX Export Disk OOM"). Falls back to the
    # previous CHECKPOINTS_DIR-based default when unset, so local/default behavior is
    # unchanged.
    onnx_base_dir = os.environ.get("ONNX_CACHE_DIR", CHECKPOINTS_DIR)
    onnx_dir = os.path.join(onnx_base_dir, "mistral_onnx")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if os.path.isdir(onnx_dir) and any(f.endswith(".onnx") for f in os.listdir(onnx_dir)):
        print(f"Reusing existing ONNX export at {onnx_dir} (first-run export skipped).")
    else:
        # Export on CPU with device="cpu": tracing needs the full model resident AND the
        # traced graph being built simultaneously, which peaks above steady-state inference
        # VRAM — that combined peak was never separately projected in EXPERIMENT_MATRIX.md
        # and OOM'd on Kaggle when the model was placed on GPU for export. dtype="fp16"
        # matters too: main_export defaults to fp32, which would produce a ~2x larger ONNX
        # graph than the fp16 baseline and blow VRAM again once loaded for inference below,
        # independent of the CPU/GPU export-device issue. See EXPERIMENT_MATRIX.md Recovery
        # Procedures "ONNX Export OOM".
        print(f"Exporting to ONNX on CPU, fp16 (first run only — writes to {onnx_dir})...")
        main_export(
            model_name_or_path=model_id,
            output=onnx_dir,
            task="text-generation-with-past",
            device="cpu",
            dtype="fp16",
            token=HF_TOKEN,
        )
        maybe_save_preprocessors(model_id, onnx_dir)

    print("Loading exported ONNX graph via onnxruntime-gpu CUDAExecutionProvider for benchmarking...")
    ort_model = ORTModelForCausalLM.from_pretrained(onnx_dir, provider="CUDAExecutionProvider")

    for dataset_key in ("CNN", "SQUAD"):
        baseline = load_baseline_metrics("MIS", dataset_key)
        if baseline is None:
            raise RuntimeError(f"No baseline found for MIS/{dataset_key} — run 01_baseline.py first.")

        test_examples = list(_loader.load(dataset_key, "test"))

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        predictions = []
        for ex in test_examples:
            prompt, _ = format_example(dataset_key, ex)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
            out = ort_model.generate(**inputs, max_new_tokens=128, do_sample=False)
            predictions.append(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - start
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        latency_ms = (elapsed_s / len(test_examples)) * 1000

        quality = evaluate_quality(dataset_key, predictions, test_examples)
        primary_metric_key = "rouge1" if dataset_key == "CNN" else "f1"

        row = {
            "training_time_hrs": None,
            "peak_vram_gb": round(peak_vram_gb, 2),
            "inference_latency_ms": round(latency_ms, 1),
            "quality_metrics": quality,
            "quality_degradation_percent": round(
                quality_degradation_percent(baseline["quality"][primary_metric_key], quality[primary_metric_key]), 2
            ),
            "speedup_factor": round(speedup_factor(baseline["latency_ms"], latency_ms), 2),
            "status": "completed",
            "notes": f"onnx model at {onnx_dir}",
        }
        save_result(f"EXP-MIS-ONNX-{dataset_key}", "MIS", "onnx", dataset_key, row)
