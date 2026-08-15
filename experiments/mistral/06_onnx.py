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
    """Deletes the on-disk HF hub cache for model_id via huggingface_hub's own cache-management
    API (scan_cache_dir().delete_revisions().execute()) — this removes the actual cached blob
    files, not just directory symlinks. Safe to call once the model is already loaded into
    memory: transformers loads safetensors checkpoints via `with safe_open(...) as f: ...
    f.get_tensor(k)` (see transformers/modeling_utils.py load_state_dict()), which copies each
    tensor into a fresh torch-owned buffer and closes the file/mmap before from_pretrained()
    returns — verified by loading a model, freeing its cache, then running a forward pass on it
    successfully (see knowledge/ai-usage-log for this session's verification). No lingering file
    handle keeps the deleted blocks alive at the OS level, so the freed disk space is real."""
    cache_info = scan_cache_dir()
    for repo in cache_info.repos:
        if repo.repo_id == model_id and repo.repo_type == "model":
            revisions = {rev.commit_hash for rev in repo.revisions}
            if revisions:
                cache_info.delete_revisions(*revisions).execute()
            return
    print(f"[WARNING] _free_hf_cache: no cached revisions found for {model_id} — nothing to free.")


if __name__ == "__main__":
    model_id = MODELS["MIS"]
    # ONNX_CACHE_DIR lets a Kaggle session redirect this if a writable, less-constrained mount
    # is ever found (Kaggle's default output path has a fixed ~20GB tracked-output quota — see
    # EXPERIMENT_MATRIX.md Recovery Procedures "ONNX Export Disk OOM"). /opt/bin turned out to
    # be read-only, so it's no longer recommended anywhere — the mechanism itself stays as a
    # harmless override, falling back to the previous CHECKPOINTS_DIR-based default when unset.
    onnx_base_dir = os.environ.get("ONNX_CACHE_DIR", CHECKPOINTS_DIR)
    onnx_dir = os.path.join(onnx_base_dir, "mistral_onnx")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if os.path.isdir(onnx_dir) and any(f.endswith(".onnx") for f in os.listdir(onnx_dir)):
        print(f"Reusing existing ONNX export at {onnx_dir} (first-run export skipped).")
    else:
        # Third fix attempt (see EXPERIMENT_MATRIX.md Recovery Procedures "ONNX Export Disk
        # OOM" for the full history of attempts 1-2). main_export()/from_pretrained(export=True)
        # both take a model path/id and load it internally, which needs the full fp16 HF cache
        # (~14.5GB) AND the growing ONNX output (~14GB) on disk at the same time — that's what
        # exceeded Kaggle's ~20GB output quota even after a redirect. Loading the model
        # ourselves first, then freeing its on-disk cache before export starts writing, means
        # disk only ever holds the cache OR the growing ONNX output, never both — verified
        # end-to-end against a tiny Mistral-architecture test model this session (fp16 load,
        # cache freed, model still runs a forward pass, exports cleanly, reloads and generates).
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
