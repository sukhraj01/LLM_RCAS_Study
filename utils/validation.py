"""
Automated sanity checks on experiment results (ARCHITECTURE.md component #6).

Run this after every experiment before accepting the result — either import
`validate_experiment_result()` at the end of an experiment script, or run this
file directly against a results CSV row.
"""

VALID_RANGES = {
    "inference_latency_ms": (0, 60_000),   # generous upper bound, flag anything absurd
    "peak_vram_gb": (0, 16),               # hard ceiling — this is the whole point
    "rouge1": (0, 1),
    "rouge2": (0, 1),
    "rougeL": (0, 1),
    "exact_match": (0, 100),
    "f1": (0, 100),
}


def validate_experiment_result(result: dict) -> list[str]:
    """Returns a list of problems found (empty list = passed all checks)."""
    problems = []

    for field, (lo, hi) in VALID_RANGES.items():
        if field in result and result[field] is not None:
            val = result[field]
            if not (lo <= val <= hi):
                problems.append(f"{field}={val} outside expected range [{lo}, {hi}]")

    if result.get("peak_vram_gb", 0) > 15:
        problems.append(
            f"peak_vram_gb={result['peak_vram_gb']} is within 1GB of the 16GB hard limit — "
            "rerun with a safety margin before trusting this number, Kaggle's actual "
            "available VRAM is sometimes slightly under the nominal 16GB."
        )

    return problems


def check_technique_relationship(qlora_hours: float, lora_hours: float | None) -> list[str]:
    """QLoRA should train faster than LoRA (smaller effective batch overhead aside,
    the memory savings should translate to being able to run without CPU offload etc).
    lora_hours is None for Llama-13B, which doesn't run fp16 LoRA — skip the check there."""
    problems = []
    if lora_hours is not None and qlora_hours > lora_hours:
        problems.append(
            f"QLoRA took {qlora_hours}h, LoRA took {lora_hours}h — expected QLoRA <= LoRA. "
            "Not necessarily wrong (QLoRA has quantization overhead per step) but worth a second look."
        )
    return problems


def check_quant_vram_relationship(vram_4bit: float, vram_8bit: float) -> list[str]:
    problems = []
    if vram_4bit >= vram_8bit:
        problems.append(
            f"4-bit VRAM ({vram_4bit}GB) should be lower than 8-bit VRAM ({vram_8bit}GB) — investigate."
        )
    return problems


if __name__ == "__main__":
    # Example / smoke test
    example = {
        "inference_latency_ms": 180,
        "peak_vram_gb": 6.2,
        "rouge1": 0.42,
    }
    issues = validate_experiment_result(example)
    print("No issues found." if not issues else "\n".join(issues))
