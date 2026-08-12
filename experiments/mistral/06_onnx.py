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

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

from experiments.common import evaluate_quality, format_example, load_baseline_metrics, save_result
from utils.config import CHECKPOINTS_DIR, HF_TOKEN, MAX_SEQ_LENGTH, MODELS
from utils.data_loader import DataLoader
from utils.metrics import quality_degradation_percent, speedup_factor

_loader = DataLoader()

if __name__ == "__main__":
    model_id = MODELS["MIS"]
    onnx_dir = f"{CHECKPOINTS_DIR}/mistral_onnx"

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Exporting to ONNX (first run only — reuses onnx_dir after that)...")
    ort_model = ORTModelForCausalLM.from_pretrained(
        model_id, export=True, provider="CUDAExecutionProvider", token=HF_TOKEN
    )
    ort_model.save_pretrained(onnx_dir)

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
