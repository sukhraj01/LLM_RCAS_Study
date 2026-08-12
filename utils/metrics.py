"""
Quality + performance metrics, standardized across all experiments (ARCHITECTURE.md component #4).

Quality metrics:
- CNN/DailyMail (summarization): ROUGE-1 / ROUGE-2 / ROUGE-L
- SQuAD (QA): Exact Match + F1

Derived metrics (always computed relative to the fp16 baseline for the same model+dataset —
never compare across techniques directly, per CLAUDE.md "Benchmarking Is Integral"):
- quality_degradation_percent
- speedup_factor
- vram_reduction_percent
"""

import re
import string
import time
from collections import Counter

import torch
from evaluate import load as load_metric

_rouge = None


def _get_rouge():
    global _rouge
    if _rouge is None:
        _rouge = load_metric("rouge")
    return _rouge


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Returns {'rouge1':..., 'rouge2':..., 'rougeL':...}, each 0-1."""
    result = _get_rouge().compute(predictions=predictions, references=references)
    return {k: result[k] for k in ("rouge1", "rouge2", "rougeL")}


def _normalize_text(s: str) -> str:
    """SQuAD-standard normalization: lowercase, strip punctuation/articles/extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def compute_exact_match(prediction: str, references: list[str]) -> int:
    norm_pred = _normalize_text(prediction)
    return int(any(norm_pred == _normalize_text(ref) for ref in references))


def compute_f1(prediction: str, references: list[str]) -> float:
    def f1_single(pred, ref):
        pred_tokens = _normalize_text(pred).split()
        ref_tokens = _normalize_text(ref).split()
        if not pred_tokens or not ref_tokens:
            return float(pred_tokens == ref_tokens)
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    return max(f1_single(prediction, ref) for ref in references)


def compute_squad_metrics(predictions: list[str], references: list[list[str]]) -> dict:
    """references[i] is a list of acceptable answers for predictions[i] (SQuAD allows multiple)."""
    em_scores = [compute_exact_match(p, r) for p, r in zip(predictions, references)]
    f1_scores = [compute_f1(p, r) for p, r in zip(predictions, references)]
    return {
        "exact_match": 100.0 * sum(em_scores) / len(em_scores),
        "f1": 100.0 * sum(f1_scores) / len(f1_scores),
    }


class LatencyVRAMTracker:
    """Wrap a generate() call to measure latency and peak VRAM in one place, consistently."""

    def __enter__(self):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self.elapsed_s = time.perf_counter() - self.start
        self.peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

    def latency_ms(self, n_samples: int) -> float:
        return (self.elapsed_s / n_samples) * 1000


def quality_degradation_percent(baseline_score: float, technique_score: float) -> float:
    """Negative = worse than baseline, positive = better. Guard div-by-zero for safety."""
    if baseline_score == 0:
        return float("nan")
    return 100.0 * (technique_score - baseline_score) / baseline_score


def speedup_factor(baseline_latency_ms: float, technique_latency_ms: float) -> float:
    """>1 means faster than baseline. speedup = baseline / optimized — don't flip this (CLAUDE.md)."""
    if technique_latency_ms == 0:
        return float("nan")
    return baseline_latency_ms / technique_latency_ms


def vram_reduction_percent(baseline_vram_gb: float, technique_vram_gb: float) -> float:
    if baseline_vram_gb == 0:
        return float("nan")
    return 100.0 * (baseline_vram_gb - technique_vram_gb) / baseline_vram_gb
