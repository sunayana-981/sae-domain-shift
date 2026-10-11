#!/usr/bin/env python3
"""Compute deterministic CLIP MMD² values from existing feature caches.

This is a feature-only preprocessing step: it does not load a model, images, an
SAE, or any outcome.  It matches ``../compute_distances.py``:

* OpenAI CLIP ViT-L/14 normalized image features;
* deterministic independent subsamples (default seed 42, n=2000);
* RBF kernel with the cross-distance median bandwidth heuristic;
* unbiased within-sample terms.

OfficeHome is sampled as one pooled mixture of its four domains.  Averaging four
per-domain MMD values would not equal the mixture MMD and is deliberately not
supported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "feature_cache"

DEFAULT_FEATURE_GROUPS: dict[str, list[str]] = {
    "caltech101": ["clip_caltech-101.npy"],
    "cub2002011": ["clip_cub200.npy"],
    "dtd": ["clip_dtd.npy"],
    "eurosat": ["clip_eurosat.npy"],
    "fgvc": ["clip_fgvc.npy"],
    "ucf101": ["clip_UCF101.npy"],
    "officehome": [
        "clip_oh_Art.npy",
        "clip_oh_Clipart.npy",
        "clip_oh_Product.npy",
        "clip_oh_Real.npy",
    ],
    "kitti": ["clip_kitti.npy"],
    "cityscapes": ["clip_cityscapes.npy"],
    "flowers102": ["clip_flowers102.npy"],
    "food101": ["clip_food101.npy"],
    "imagenet-a": ["clip_imagenet-a.npy"],
    "imagenet-o": ["clip_imagenet-o.npy"],
    "imagenet-r": ["clip_imagenet-r.npy"],
    "imagenetv2": ["clip_imagenetv2.npy"],
    "oxford_pets": ["clip_oxford_pets.npy"],
    "pathmnist": ["clip_pathmnist.npy"],
    "sketch": ["clip_sketch.npy"],
    "stanford_cars": ["clip_stanford_cars.npy"],
}


class MMDInputError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _require_new(path: Path) -> None:
    if path.exists():
        raise MMDInputError(
            f"Refusing to overwrite {path}; use a new output path for provenance."
        )


def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    return (
        np.sum(left**2, axis=1, keepdims=True)
        + np.sum(right**2, axis=1, keepdims=True).T
        - 2.0 * left @ right.T
    )


def compute_mmd2_rbf_unbiased(
    reference: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
    """Return (MMD², sigma) using the repository's exact estimator."""
    reference = np.asarray(reference, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    n, m = len(reference), len(target)
    if n < 2 or m < 2:
        raise MMDInputError("Unbiased MMD requires at least two samples per side")
    dxy = squared_distances(reference, target)
    positive = dxy[dxy > 0]
    sigma = float(np.sqrt(np.median(positive) / 2.0)) if len(positive) else 1.0
    if not math.isfinite(sigma) or sigma <= 0:
        raise MMDInputError(f"Invalid RBF bandwidth: {sigma}")
    kxx = np.exp(
        -squared_distances(reference, reference) / (2.0 * sigma**2)
    )
    kyy = np.exp(-squared_distances(target, target) / (2.0 * sigma**2))
    kxy = np.exp(-dxy / (2.0 * sigma**2))
    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)
    value = (
        kxx.sum() / (n * (n - 1))
        + kyy.sum() / (m * (m - 1))
        - 2.0 * kxy.mean()
    )
    return float(value), sigma


def _open_arrays(paths: Sequence[Path]) -> list[np.ndarray]:
    arrays = []
    feature_dim = None
    for path in paths:
        if not path.exists():
            raise MMDInputError(f"Feature cache not found: {path}")
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2 or len(array) < 2:
            raise MMDInputError(f"Expected a nontrivial 2-D feature cache: {path}")
        if feature_dim is None:
            feature_dim = array.shape[1]
        elif array.shape[1] != feature_dim:
            raise MMDInputError(
                f"Feature dimension mismatch in pooled group: {path} has "
                f"{array.shape[1]}, expected {feature_dim}"
            )
        arrays.append(array)
    return arrays


def deterministic_group_subsample(
    arrays: Sequence[np.ndarray], n: int, seed: int
) -> tuple[np.ndarray, int]:
    """Uniformly subsample the conceptual concatenation without materializing it."""
    lengths = np.asarray([len(array) for array in arrays], dtype=np.int64)
    total = int(lengths.sum())
    if total <= n:
        return np.concatenate([np.asarray(array) for array in arrays], axis=0), total
    rng = np.random.default_rng(seed)
    global_indices = rng.choice(total, n, replace=False)
    boundaries = np.cumsum(lengths)
    sampled = np.empty((n, arrays[0].shape[1]), dtype=np.float32)
    starts = np.concatenate([[0], boundaries[:-1]])
    for array, start, end in zip(arrays, starts, boundaries):
        mask = (global_indices >= start) & (global_indices < end)
        if np.any(mask):
            sampled[mask] = np.asarray(array[global_indices[mask] - start])
    return sampled, total


def parse_datasets(value: str) -> list[str]:
    datasets = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(datasets) - set(DEFAULT_FEATURE_GROUPS))
    if unknown:
        raise MMDInputError(
            f"Unknown datasets {unknown}; choices are "
            f"{sorted(DEFAULT_FEATURE_GROUPS)}"
        )
    return list(dict.fromkeys(datasets))


def compute_cached_mmd(
    *,
    cache_dir: Path,
    datasets: Sequence[str],
    output: Path,
    reference_name: str = "clip_imagenet_val.npy",
    sample_size: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    _require_new(output)
    if sample_size < 2:
        raise MMDInputError("sample_size must be at least 2")
    reference_path = cache_dir / reference_name
    reference_arrays = _open_arrays([reference_path])
    reference, reference_total = deterministic_group_subsample(
        reference_arrays, sample_size, seed
    )

    rows: list[dict[str, Any]] = []
    input_records: dict[str, Any] = {
        "reference": {
            "path": str(reference_path.resolve()),
            "sha256": sha256_file(reference_path),
            "n_total": reference_total,
        },
        "targets": {},
    }
    for dataset in datasets:
        cache_paths = [cache_dir / name for name in DEFAULT_FEATURE_GROUPS[dataset]]
        arrays = _open_arrays(cache_paths)
        if arrays[0].shape[1] != reference.shape[1]:
            raise MMDInputError(
                f"{dataset} feature width {arrays[0].shape[1]} does not match "
                f"reference width {reference.shape[1]}"
            )
        target, target_total = deterministic_group_subsample(
            arrays, sample_size, seed
        )
        mmd2, sigma = compute_mmd2_rbf_unbiased(reference, target)
        rows.append(
            {
                "dataset": dataset,
                "N": target_total,
                "MMD2_clip": mmd2,
                "rbf_sigma": sigma,
                "mmd_reference_n": len(reference),
                "mmd_target_n": len(target),
                "subsample_seed": seed,
                "feature_cache_files": ";".join(path.name for path in cache_paths),
            }
        )
        input_records["targets"][dataset] = [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "n": int(len(array)),
                "feature_dim": int(array.shape[1]),
            }
            for path, array in zip(cache_paths, arrays)
        ]
        print(
            f"{dataset:14s} N={target_total:6d} sample={len(target):4d} "
            f"MMD2_clip={mmd2:.8f} sigma={sigma:.6f}"
        )

    frame = pd.DataFrame(rows)
    _atomic_write(
        output, frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "estimator": (
            "unbiased RBF MMD^2; bandwidth sqrt(median positive cross squared "
            "distance / 2)"
        ),
        "feature_space": "OpenAI CLIP ViT-L/14 normalized image embedding",
        "sample_size_requested": sample_size,
        "subsample_seed": seed,
        "officehome_pooling": (
            "uniform sample over conceptual concatenation of Art, Clipart, "
            "Product, and Real World image features"
        ),
        "inputs": input_records,
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(f"Wrote {output}")
    print(f"Wrote {manifest_path}")
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument(
        "--datasets",
        required=True,
        help="comma-separated canonical dataset names",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-name", default="clip_imagenet_val.npy")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        compute_cached_mmd(
            cache_dir=Path(args.cache_dir).resolve(),
            datasets=parse_datasets(args.datasets),
            output=Path(args.output).resolve(),
            reference_name=args.reference_name,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    except MMDInputError as error:
        raise SystemExit(f"error: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
