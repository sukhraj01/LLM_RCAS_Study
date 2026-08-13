"""
Shared logic for all experiment scripts. Each technique gets a thin script
(e.g. experiments/mistral/02_lora.py) that just calls into here — this is
where the actual model loading / training / inference logic lives, so it's
written once and every technique uses the same measurement code (required
for the "always compare against the same baseline, measured the same way"
rule in CLAUDE.md).

STATUS: first-draft skeleton, not yet run against real Kaggle hardware.
Expect to debug library-version quirks (transformers/peft/bitsandbytes APIs
move fast) on the first real run — that's normal, not a sign this was written
wrong. Log whatever you fix in knowledge/ai-usage-log/, per CLAUDE.md.
"""

import ast
import csv
import gc
import os
import time
from typing import Callable

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from utils.config import (
    DATASETS,
    HF_TOKEN,
    LORA_CONFIG,
    MAX_SEQ_LENGTH,
    MODELS,
    QUANT_4BIT_CONFIG,
    QUANT_8BIT_CONFIG,
    RESULTS_DIR,
    get_qlora_config,
)
from utils.data_loader import DataLoader
from utils.metrics import (
    LatencyVRAMTracker,
    compute_rouge,
    compute_squad_metrics,
    quality_degradation_percent,
    speedup_factor,
    vram_reduction_percent,
)
from utils.validation import validate_experiment_result

_loader = DataLoader()


# ---------------------------------------------------------------------------
# Prompting — task-specific format, dataset-agnostic interface
# ---------------------------------------------------------------------------


def format_example(dataset_key: str, example: dict) -> tuple[str, str]:
    """Returns (prompt, target). Keep prompts short and consistent across techniques —
    changing the prompt between techniques would confound the comparison."""
    if dataset_key == "CNN":
        prompt = f"Summarize the following article in 2-3 sentences.\n\nArticle: {example['article']}\n\nSummary:"
        target = example["highlights"]
        return prompt, target
    elif dataset_key == "SQUAD":
        prompt = (
            f"Answer the question based on the context. Be concise.\n\n"
            f"Context: {example['context']}\n\nQuestion: {example['question']}\n\nAnswer:"
        )
        # squad's 'answers' field is {'text': [...], 'answer_start': [...]}
        target = example["answers"]["text"][0] if example["answers"]["text"] else ""
        return prompt, target
    raise ValueError(f"Unknown dataset_key: {dataset_key}")


def get_references(dataset_key: str, example: dict) -> list[str]:
    """SQuAD allows multiple acceptable answers; CNN has one reference summary."""
    if dataset_key == "SQUAD":
        return example["answers"]["text"] or [""]
    return [example["highlights"]] if dataset_key == "CNN" else [""]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(model_key: str, quant_config: dict | None = None):
    model_id = MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"token": HF_TOKEN, "device_map": "auto"}

    if quant_config and quant_config.get("load_in_4bit"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=getattr(torch, quant_config["bnb_4bit_compute_dtype"]),
        )
        kwargs["quantization_config"] = bnb_config
    elif quant_config and quant_config.get("load_in_8bit"):
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model, tokenizer


def apply_lora(model, lora_hparams: dict):
    peft_config = LoraConfig(
        r=lora_hparams["r"],
        lora_alpha=lora_hparams["lora_alpha"],
        lora_dropout=lora_hparams["lora_dropout"],
        target_modules=lora_hparams["target_modules"],
        bias=lora_hparams["bias"],
        task_type=lora_hparams["task_type"],
    )
    model = get_peft_model(model, peft_config)
    # With only adapter params trainable and gradient checkpointing on, the frozen base
    # model's input embeddings output requires_grad=False, so checkpointing has no tensor
    # to build a backward graph from and the first backward pass fails with "None of the
    # inputs have requires_grad=True". See EXPERIMENT_MATRIX.md Recovery Procedures.
    model.enable_input_require_grads()
    return model


def _assert_fully_on_gpu(model, model_key: str):
    """device_map="auto" can silently spill layers to CPU (or 'meta') instead of erroring
    when a model doesn't fully fit in free VRAM — .generate() then keeps running, just
    slowly enough to burn an entire 12h Kaggle session without producing output. Fail loudly
    and immediately instead. See EXPERIMENT_MATRIX.md "Recovery Procedures" for what to do
    when this fires."""
    bad_devices = {p.device.type for p in model.parameters() if p.device.type != "cuda"}
    if bad_devices:
        raise RuntimeError(
            f"Model {model_key} partially offloaded to {sorted(bad_devices)} (device_map "
            "couldn't fit it on GPU) — aborting instead of running for hours. Free GPU memory "
            "(check no earlier model in this process was left unreleased) or reduce batch/seq "
            "length before retrying."
        )


def _release_model(model):
    """Explicit cleanup between successive model loads in the same process — without this,
    a leftover model can leave enough VRAM occupied that the next device_map="auto" load
    can't fit fully on GPU and silently falls back to CPU offload (see _assert_fully_on_gpu)."""
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Inference + quality
# ---------------------------------------------------------------------------


def generate_predictions(model, tokenizer, examples: list[dict], dataset_key: str, max_new_tokens=128):
    model.eval()
    predictions = []
    start = time.perf_counter()
    with torch.no_grad():
        for i, ex in enumerate(examples):
            prompt, _ = format_example(dataset_key, ex)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(model.device)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            predictions.append(text.strip())
            if (i + 1) % 20 == 0:
                avg_s = (time.perf_counter() - start) / (i + 1)
                print(f"{i + 1}/{len(examples)} done, {avg_s:.2f}s/sample avg")
    return predictions


def evaluate_quality(dataset_key: str, predictions: list[str], examples: list[dict]) -> dict:
    if dataset_key == "CNN":
        references = [ex["highlights"] for ex in examples]
        return compute_rouge(predictions, references)
    elif dataset_key == "SQUAD":
        references = [get_references(dataset_key, ex) for ex in examples]
        return compute_squad_metrics(predictions, references)
    raise ValueError(dataset_key)


# ---------------------------------------------------------------------------
# Result recording
# ---------------------------------------------------------------------------

RESULT_FIELDS = [
    "exp_id", "model", "technique", "dataset", "training_time_hrs", "peak_vram_gb",
    "inference_latency_ms", "quality_metrics", "quality_degradation_percent",
    "speedup_factor", "status", "notes",
]


def save_result(exp_id: str, model_key: str, technique: str, dataset_key: str, row: dict):
    """Appends one row to results/<model>_results.csv, creating it with a header if new."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{model_key.lower()}_results.csv")
    file_exists = os.path.isfile(path)

    full_row = {
        "exp_id": exp_id, "model": model_key, "technique": technique, "dataset": dataset_key,
        **row,
    }

    issues = validate_experiment_result(row)
    full_row["validation_issues"] = "; ".join(issues) if issues else ""

    fieldnames = list(full_row.keys())
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(full_row)

    if issues:
        print(f"[VALIDATION WARNING] {exp_id}: {full_row['validation_issues']}")
    print(f"Saved {exp_id} -> {path}")


def load_baseline_metrics(model_key: str, dataset_key: str) -> dict | None:
    """Reads the baseline row back out of results/<model>_results.csv, so technique
    scripts run in a *separate* Kaggle session from the baseline script can still
    compare against it. Returns None if baseline hasn't been run yet (run it first).

    save_result() only ever appends, so a rerun of a script that already succeeded once
    (e.g. rerunning a baseline script to pick up a previously-failed dataset in the same
    run) leaves multiple rows for the same exp_id/technique/dataset in the file — oldest
    first. Scans the whole file and keeps the LAST match, not the first, so a rerun's
    fresh numbers are what get compared against, not a stale earlier attempt."""
    path = os.path.join(RESULTS_DIR, f"{model_key.lower()}_results.csv")
    if not os.path.isfile(path):
        return None
    result = None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["technique"] == "baseline" and row["dataset"] == dataset_key:
                quality = ast.literal_eval(row["quality_metrics"])
                result = {"quality": quality, "latency_ms": float(row["inference_latency_ms"])}
    return result


def require_baseline_metrics(model_key: str, dataset_key: str) -> dict:
    """Same as load_baseline_metrics but fails fast if the baseline hasn't run yet, instead of
    letting a technique script silently proceed without a comparison. Use this (not
    load_baseline_metrics directly) as the baseline_lookup passed to the multi-dataset
    orchestrators below for every technique except baseline itself."""
    baseline = load_baseline_metrics(model_key, dataset_key)
    if baseline is None:
        raise RuntimeError(
            f"No baseline found for {model_key}/{dataset_key} — run the baseline script first."
        )
    return baseline


# Exp ID convention departs from the technique string in exactly one case ("baseline" -> "BASE");
# everything else is technique.upper(). See run_inference_multi_dataset / run_training_multi_dataset.
_EXP_ID_TECHNIQUE_TOKEN = {"baseline": "BASE"}


# ---------------------------------------------------------------------------
# High-level experiment runners — called by the multi-dataset orchestrators below,
# never directly by the per-technique scripts (see EXPERIMENT_MATRIX.md Recovery
# Procedures: "Model silently CPU-offloaded / hangs" for why).
# ---------------------------------------------------------------------------


def run_inference_only_experiment(model, tokenizer, exp_id: str, model_key: str, dataset_key: str,
                                   technique: str, baseline_row: dict | None = None):
    """For baseline, 8bit, 4bit — measure + evaluate on an ALREADY-LOADED model/tokenizer for
    ONE dataset. Caller (run_inference_multi_dataset) owns loading, device-placement checking,
    and releasing the model — this function never touches those, so it can be called once per
    dataset against the same model instance without reloading."""
    test_examples = list(_loader.load(dataset_key, "test"))

    with LatencyVRAMTracker() as tracker:
        predictions = generate_predictions(model, tokenizer, test_examples, dataset_key)

    latency_ms = tracker.latency_ms(len(test_examples))
    quality = evaluate_quality(dataset_key, predictions, test_examples)

    row = {
        "training_time_hrs": None,
        "peak_vram_gb": round(tracker.peak_vram_gb, 2),
        "inference_latency_ms": round(latency_ms, 1),
        "quality_metrics": quality,
        "status": "completed",
        "notes": "",
    }

    if baseline_row is not None:
        primary_metric_key = "rouge1" if dataset_key == "CNN" else "f1"
        row["quality_degradation_percent"] = round(
            quality_degradation_percent(baseline_row["quality"][primary_metric_key], quality[primary_metric_key]), 2
        )
        row["speedup_factor"] = round(
            speedup_factor(baseline_row["latency_ms"], latency_ms), 2
        )
    else:
        row["quality_degradation_percent"] = 0.0  # this IS the baseline
        row["speedup_factor"] = 1.0

    save_result(exp_id, model_key, technique, dataset_key, row)
    return {"quality": quality, "latency_ms": latency_ms, "vram_gb": tracker.peak_vram_gb}


def run_inference_multi_dataset(model_key: str, technique: str, dataset_keys: list[str],
                                 quant_config: dict | None,
                                 baseline_lookup: Callable[[str], dict | None]):
    """Loads the model ONCE for every dataset in dataset_keys, checks device placement ONCE,
    then releases ONCE at the end — this is the actual fix for the CPU-offload/allocator-
    fragmentation failure that hit EXP-LLAMA-BASE-SQUAD (see EXPERIMENT_MATRIX.md "Model
    silently CPU-offloaded / hangs"). Reloading a model a second time in the same process was
    the root cause; explicit del/gc.collect()/empty_cache() alone was necessary but not
    sufficient to make a second Llama-2-13B load reliably fit."""
    model, tokenizer = load_model_and_tokenizer(model_key, quant_config)
    _assert_fully_on_gpu(model, model_key)

    token = _EXP_ID_TECHNIQUE_TOKEN.get(technique, technique.upper())
    results = {}
    try:
        for dataset_key in dataset_keys:
            exp_id = f"EXP-{model_key}-{token}-{dataset_key}"
            results[dataset_key] = run_inference_only_experiment(
                model, tokenizer, exp_id, model_key, dataset_key, technique,
                baseline_row=baseline_lookup(dataset_key),
            )
    finally:
        _release_model(model)

    return results


def run_training_experiment(model, tokenizer, exp_id: str, model_key: str, dataset_key: str,
                             technique: str, lora_hparams: dict, output_dir: str, quantized: bool,
                             baseline_row: dict | None = None):
    """For LoRA and QLoRA — trains + evaluates on an ALREADY LoRA-wrapped model/tokenizer for
    ONE dataset. Caller (run_training_multi_dataset) owns loading the base model once, applying
    a FRESH adapter per dataset, and releasing the base model once at the end — this function
    only trains whatever adapter it's handed, so it stays correct whether that adapter is fresh
    or (if ever called directly) reused."""
    train_examples = list(_loader.load(dataset_key, "train"))
    test_examples = list(_loader.load(dataset_key, "test"))

    model.print_trainable_parameters()

    def tokenize_fn(example):
        prompt, target = format_example(dataset_key, example)
        full_text = f"{prompt} {target}{tokenizer.eos_token}"
        return tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LENGTH)

    train_ds = Dataset.from_list(train_examples).map(tokenize_fn, remove_columns=Dataset.from_list(train_examples).column_names)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=lora_hparams["batch_size"],
        gradient_accumulation_steps=lora_hparams["gradient_accumulation_steps"],
        learning_rate=lora_hparams["learning_rate"],
        num_train_epochs=lora_hparams["num_train_epochs"],
        optim=lora_hparams["optim"],
        gradient_checkpointing=lora_hparams["gradient_checkpointing"],
        save_steps=200,
        logging_steps=20,
        fp16=not quantized,  # bf16 compute already set inside 4-bit config when quantized
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    start = time.perf_counter()
    trainer.train()
    training_time_hrs = (time.perf_counter() - start) / 3600

    model.save_pretrained(output_dir)

    with LatencyVRAMTracker() as tracker:
        predictions = generate_predictions(model, tokenizer, test_examples, dataset_key)
    latency_ms = tracker.latency_ms(len(test_examples))

    quality = evaluate_quality(dataset_key, predictions, test_examples)

    row = {
        "training_time_hrs": round(training_time_hrs, 2),
        "peak_vram_gb": round(tracker.peak_vram_gb, 2),
        "inference_latency_ms": round(latency_ms, 1),
        "quality_metrics": quality,
        "status": "completed",
        "notes": f"adapter saved to {output_dir}",
    }

    if baseline_row is not None:
        primary_metric_key = "rouge1" if dataset_key == "CNN" else "f1"
        row["quality_degradation_percent"] = round(
            quality_degradation_percent(baseline_row["quality"][primary_metric_key], quality[primary_metric_key]), 2
        )
        row["speedup_factor"] = round(speedup_factor(baseline_row["latency_ms"], latency_ms), 2)
    else:
        row["quality_degradation_percent"] = None
        row["speedup_factor"] = None

    save_result(exp_id, model_key, technique, dataset_key, row)
    return {"quality": quality, "latency_ms": latency_ms, "vram_gb": tracker.peak_vram_gb}


def run_training_multi_dataset(model_key: str, technique: str, dataset_keys: list[str],
                                lora_hparams: dict, quant_config: dict | None,
                                output_dir_fn: Callable[[str], str],
                                baseline_lookup: Callable[[str], dict | None]):
    """Loads the base model ONCE, then for each dataset: applies a FRESH LoRA adapter (so
    datasets never train on top of each other's adapter weights — that would silently
    contaminate the comparison), trains + evaluates, saves the adapter, then strips it back to
    the clean base model via PeftModel.unload() (removes the adapter in place, no reload from
    disk) before the next dataset. The base model is released once, after the whole loop — this
    is the same "load once per process" fix as run_inference_multi_dataset, applied to training
    so it doesn't hit Llama-2-13B QLoRA in Week 3 the same way it hit the Llama baseline."""
    base_model, tokenizer = load_model_and_tokenizer(model_key, quant_config)
    _assert_fully_on_gpu(base_model, model_key)

    token = _EXP_ID_TECHNIQUE_TOKEN.get(technique, technique.upper())
    results = {}
    try:
        for dataset_key in dataset_keys:
            exp_id = f"EXP-{model_key}-{token}-{dataset_key}"
            adapted_model = apply_lora(base_model, lora_hparams)
            results[dataset_key] = run_training_experiment(
                adapted_model, tokenizer, exp_id, model_key, dataset_key, technique,
                lora_hparams, output_dir_fn(dataset_key), quantized=bool(quant_config),
                baseline_row=baseline_lookup(dataset_key),
            )
            base_model = adapted_model.unload()  # strip this dataset's adapter, back to clean base
    finally:
        _release_model(base_model)

    return results
