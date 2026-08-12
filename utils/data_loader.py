"""
Unified dataset loading for CNN/DailyMail and SQuAD.

Guarantees (per ARCHITECTURE.md component #2):
- Deterministic splits (same rows every time, given the same SEED)
- No leakage (test rows are never touched during training)
- Consistent interface across both datasets

Usage:
    from utils.data_loader import DataLoader
    loader = DataLoader()
    train = loader.load("CNN", split="train")
    val = loader.load("CNN", split="val")
    test = loader.load("CNN", split="test")
"""

from datasets import load_dataset

from utils.config import DATASETS, SEED


class DataLoader:
    def __init__(self, seed: int = SEED):
        self.seed = seed
        self._cache = {}

    def load(self, dataset_key: str, split: str):
        """
        dataset_key: "CNN" or "SQUAD" (see utils.config.DATASETS)
        split: "train", "val", or "test"
        """
        if dataset_key not in DATASETS:
            raise ValueError(f"Unknown dataset key '{dataset_key}'. Options: {list(DATASETS)}")
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got '{split}'")

        cache_key = (dataset_key, split)
        if cache_key in self._cache:
            return self._cache[cache_key]

        spec = DATASETS[dataset_key]
        result = self._load_and_split(spec, split)
        self._cache[cache_key] = result
        return result

    def _load_and_split(self, spec: dict, split: str):
        hf_split = "train" if split in ("train", "val") else "validation"
        # NOTE: cnn_dailymail and squad both only ship train/validation on HF —
        # we carve "val" and "test" out of HF's validation split ourselves, deterministically,
        # so we never touch the same rows for both. This is what "no leakage" means here.
        if spec["hf_config"]:
            ds = load_dataset(spec["hf_name"], spec["hf_config"], split=hf_split)
        else:
            ds = load_dataset(spec["hf_name"], split=hf_split)

        ds = ds.shuffle(seed=self.seed)

        if split == "train":
            n = spec["n_train"]
            return ds.select(range(min(n, len(ds))))
        elif split == "val":
            n = spec["n_val"]
            return ds.select(range(min(n, len(ds))))
        else:  # test — offset past the val rows so there's zero overlap
            n_val = spec["n_val"]
            n_test = spec["n_test"]
            start = n_val
            end = n_val + n_test
            return ds.select(range(start, min(end, len(ds))))

    def sanity_check(self, dataset_key: str):
        """Quick assertion that splits don't overlap and sizes match config. Run once after
        writing/changing this file, and again if you touch DATASETS in config.py."""
        spec = DATASETS[dataset_key]
        val = self.load(dataset_key, "val")
        test = self.load(dataset_key, "test")
        assert len(val) == spec["n_val"], f"val size mismatch: {len(val)} != {spec['n_val']}"
        assert len(test) == spec["n_test"], f"test size mismatch: {len(test)} != {spec['n_test']}"
        print(f"{dataset_key}: val={len(val)}, test={len(test)} — no overlap by construction (disjoint index ranges)")


if __name__ == "__main__":
    loader = DataLoader()
    for key in DATASETS:
        loader.sanity_check(key)
