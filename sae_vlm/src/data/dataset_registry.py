"""Dataset registry — load any dataset by name using configs/datasets/registry.yaml."""

import json
import os
from pathlib import Path
from typing import List, Optional

import yaml
from datasets import Dataset, load_dataset

_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "datasets" / "registry.yaml"
_CLASSNAMES_ROOT = Path(__file__).resolve().parents[2] / "configs" / "classnames"


def load_registry(path: Optional[str] = None) -> dict:
    p = Path(path) if path else _REGISTRY_PATH
    with open(p) as f:
        return yaml.safe_load(f)


def get_dataset(name: str, registry: Optional[dict] = None,
                max_samples: Optional[int] = None,
                split: Optional[str] = None) -> Dataset:
    reg = registry or load_registry()
    cfg = reg[name]

    effective_split = split or cfg.get("split", "train")

    local = cfg.get("local_path")
    if local and os.path.isdir(local):
        ds = load_dataset("imagefolder", data_dir=local, split=effective_split,
                          trust_remote_code=cfg.get("trust_remote_code", False))
    else:
        kwargs = {
            "path": cfg["hf_path"],
            "split": effective_split,
            "trust_remote_code": cfg.get("trust_remote_code", False),
        }
        if "hf_name" in cfg:
            kwargs["name"] = cfg["hf_name"]
        ds = load_dataset(**kwargs)

    if isinstance(ds, dict):
        ds = ds[effective_split]

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds


def get_classnames(name: str, dataset: Optional[Dataset] = None,
                   registry: Optional[dict] = None) -> List[str]:
    reg = registry or load_registry()
    cfg = reg[name]

    # Prefer ClassLabel feature embedded in the dataset
    if dataset is not None:
        for key, feat in dataset.features.items():
            if hasattr(feat, "names"):
                return list(feat.names)

    # Fall back to classnames file
    cf = cfg.get("classnames_file")
    if cf:
        p = Path(__file__).resolve().parents[2] / cf
        if p.exists():
            if p.suffix == ".json":
                with open(p) as f:
                    data = json.load(f)
                return list(data.values()) if isinstance(data, dict) else data
            else:
                return [line.strip() for line in p.read_text().splitlines() if line.strip()]

    raise ValueError(f"No classnames found for dataset '{name}'. "
                     f"Add a classnames_file entry to configs/datasets/registry.yaml "
                     f"or ensure the dataset has a ClassLabel feature.")


def get_label_key(name: str, dataset: Dataset, registry: Optional[dict] = None) -> str:
    reg = registry or load_registry()
    cfg = reg.get(name, {})
    if "label_key" in cfg:
        return cfg["label_key"]
    # Autodetect
    for key in ("label", "fine_label", "labels", "class_label", "target", "y"):
        if key in dataset.features:
            return key
    for key, feat in dataset.features.items():
        if hasattr(feat, "names"):
            return key
    raise ValueError(f"Cannot find label column for dataset '{name}'. "
                     f"Columns: {list(dataset.features.keys())}")
