"""
EXP-LLAMA-ONNX-CNN, EXP-LLAMA-ONNX-SQUAD (see EXPERIMENT_MATRIX.md technique #6)
Same approach as experiments/mistral/06_onnx.py — see that file's docstring for
the caveat about optimum's export API varying by version. Requires 01_baseline.py first.

Run: python experiments/llama/05_onnx.py
"""

import gc
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from huggingface_hub import scan_cache_dir
from optimum.exporters.onnx import onnx_export_from_model
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.common import evaluate_quality, format_example, load_baseline_metrics, save_result
from utils.config import CHECKPOINTS_DIR, HF_TOKEN, MAX_SEQ_LENGTH, MODELS
from utils.data_loader import DataLoader
from utils.metrics import quality_degradation_percent, speedup_factor

_loader = DataLoader()


def _free_hf_cache(model_id: str):
    """See experiments/mistral/06_onnx.py's docstring for this function — identical here for
    consistency between the two ONNX scripts. Deletes the on-disk HF hub cache for model_id via
    huggingface_hub's own cache-management API; safe once the model is already loaded into
    memory (verified this session against a tiny Mistral-architecture test model, not yet
    against Llama-2-13B specifically — see the WARNING below)."""
    cache_info = scan_cache_dir()
    for repo in cache_info.repos:
        if repo.repo_id == model_id and repo.repo_type == "model":
            revisions = {rev.commit_hash for rev in repo.revisions}
            if revisions:
                cache_info.delete_revisions(*revisions).execute()
            return
    print(f"[WARNING] _free_hf_cache: no cached revisions found for {model_id} — nothing to free.")


if __name__ == "__main__":
    model_id = MODELS["LLAMA"]
    # ONNX_CACHE_DIR lets a Kaggle session redirect this if a writable, less-constrained mount
    # is ever found (Kaggle's default output path has a fixed ~20GB tracked-output quota — see
    # EXPERIMENT_MATRIX.md Recovery Procedures "ONNX Export Disk OOM"). /opt/bin turned out to
    # be read-only, so it's no longer recommended anywhere — the mechanism itself stays as a
    # harmless override, falling back to the previous CHECKPOINTS_DIR-based default when unset.
    onnx_base_dir = os.environ.get("ONNX_CACHE_DIR", CHECKPOINTS_DIR)
    onnx_dir = os.path.join(onnx_base_dir, "llama_onnx")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # WARNING: this is Llama-2-13B, and unlike Mistral, the load-then-free-cache fix below does
    # NOT resolve this model's disk problem, let alone its separate VRAM problem. fp16 weights
    # alone are ~26GB — so even with cache and ONNX output never held on disk simultaneously
    # (what the fix below achieves), the PEAK usage during either single phase is still ~26GB,
    # which alone exceeds Kaggle's ~20GB output quota with no redirect target currently known to
    # work. On top of that: fp16 weights alone are already over Kaggle's 16GB VRAM ceiling,
    # so the final ORTModelForCausalLM.from_pretrained(..., provider="CUDAExecutionProvider")
    # load below won't fit regardless. Do not run this script until both are separately
    # resolved. See PROJECT_STATE.md Blockers & Risks / EXPERIMENT_MATRIX.md Recovery
    # Procedures "ONNX Export OOM" / "ONNX Export Disk OOM" for the open risk this carries
    # beyond Mistral's fix.
    if os.path.isdir(onnx_dir) and any(f.endswith(".onnx") for f in os.listdir(onnx_dir)):
        print(f"Reusing existing ONNX export at {onnx_dir} (first-run export skipped).")
    else:
        print(f"Loading {model_id} (fp16, CPU) into memory...")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, token=HF_TOKEN)
        print(f"Model loaded — freeing its on-disk HF cache before export starts writing...")
        _free_hf_cache(model_id)

        print(f"Exporting to ONNX on CPU, fp16 (first run only — writes to {onnx_dir})...")
        onnx_export_from_model(
            model=model,
            output=onnx_dir,
            task="text-generation-with-past",
            device="cpu",
        )
        del model
        gc.collect()

    print("Loading exported ONNX graph via onnxruntime-gpu CUDAExecutionProvider for benchmarking...")
    ort_model = ORTModelForCausalLM.from_pretrained(onnx_dir, provider="CUDAExecutionProvider")

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
