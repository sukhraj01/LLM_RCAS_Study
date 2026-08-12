"""
EXP-LLAMA-ONNX-CNN, EXP-LLAMA-ONNX-SQUAD (see EXPERIMENT_MATRIX.md technique #6)
Same approach as experiments/mistral/06_onnx.py — see that file's docstring for
the caveat about optimum's export API varying by version. Requires 01_baseline.py first.

Run: python experiments/llama/05_onnx.py
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
    model_id = MODELS["LLAMA"]
    onnx_dir = f"{CHECKPOINTS_DIR}/llama_onnx"

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Exporting to ONNX (first run only — reuses onnx_dir after that)...")
    ort_model = ORTModelForCausalLM.from_pretrained(
        model_id, export=True, provider="CUDAExecutionProvider", token=HF_TOKEN
    )
    ort_model.save_pretrained(onnx_dir)

    for dataset_key in ("CNN", "SQUAD"):
        baseline = load_baseline_metrics("LLAMA", dataset_key)
        if baseline is None:
            raise RuntimeError(f"No baseline found for LLAMA/{dataset_key} — run 01_baseline.py first.")

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
        save_result(f"EXP-LLAMA-ONNX-{dataset_key}", "LLAMA", "onnx", dataset_key, row)
