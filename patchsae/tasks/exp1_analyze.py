#!/usr/bin/env python3
"""
exp1_analyze.py — turn per-dataset SAE unit activity into overlap-vs-distance.

Reads out/exp1_activity_overlap/activity.npz (written by exp1_activity_overlap.py)
and reports, for each target dataset, how much its important-unit set overlaps
ImageNet's, alongside the CLIP-MMD domain distances already in
../distance_results.csv.

Overlap at fraction k is |top-k(imagenet) ∩ top-k(dataset)| / |top-k|, i.e. 1.0
means the domain leans on exactly the units ImageNet leans on, and k means
chance level.
"""

import json
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
ACT = _root / "out/exp1_activity_overlap/activity.npz"
DIST = _root.parent / "distance_results.csv"

# DATASET_INFO key -> row(s) in distance_results.csv
DIST_KEY = {
    "imagenet": ["**imagenet_split_half**"],
    "caltech101": ["caltech-101"],
    "oxford_pets": ["oxford_pets"],
    "dtd": ["dtd"],
    "eurosat": ["eurosat"],
    "cub2002011": ["cub200"],
    "ucf101": ["UCF101"],
    "officehome": ["oh/Art", "oh/Clipart", "oh/Product", "oh/Real"],
    "medmnist": ["pathmnist"],
    # fgvc, kitti, cityscapes have no row in distance_results.csv
}

FRACS = [0.01, 0.05, 0.10, 0.20, 0.50]


def top_set(v, frac):
    k = max(1, round(frac * len(v)))
    return set(np.argsort(-v)[:k].tolist())


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb)))


def load_distances():
    if not DIST.exists():
        return {}
    rows = {}
    with open(DIST) as f:
        header = f.readline().strip().split(",")
        idx = header.index("MMD2_clip")
        for line in f:
            p = line.strip().split(",")
            if len(p) <= idx or not p[idx]:
                continue
            rows[p[0]] = float(p[idx])
    out = {}
    for ds, keys in DIST_KEY.items():
        vals = [rows[k] for k in keys if k in rows]
        if vals:
            out[ds] = float(np.mean(vals))
    return out


def main():
    d = np.load(ACT)
    names = sorted({k.rsplit("__", 1)[0] for k in d.files})
    if "imagenet" not in names:
        sys.exit("[exp1] imagenet reference missing from activity.npz")
    ref = d["imagenet__activity"]
    ref_fire = d["imagenet__fire_rate"]
    dists = load_distances()

    print(f"SAE units: {len(ref)}   datasets: {len(names)}\n")
    print("Overlap of top-k% most-active units with ImageNet's top-k%")
    hdr = f"{'dataset':<14}{'MMD2_clip':>10}" + "".join(f"{int(f*100):>7}%" for f in FRACS) + f"{'spearman':>10}{'dead':>8}"
    print(hdr)
    print("-" * len(hdr))

    table = []
    for name in names:
        if name == "imagenet":
            continue
        v = d[f"{name}__activity"]
        row = {"dataset": name, "mmd2_clip": dists.get(name)}
        cells = []
        for f in FRACS:
            ov = len(top_set(ref, f) & top_set(v, f)) / max(1, round(f * len(v)))
            row[f"overlap_{int(f*100)}"] = ov
            cells.append(f"{ov:>7.3f}")
        row["spearman"] = spearman(ref, v)
        row["dead_frac"] = float((v == 0).mean())
        dstr = "     n/a" if row["mmd2_clip"] is None else f"{row['mmd2_clip']:>10.4f}"
        print(f"{name:<14}{dstr}" + "".join(cells) +
              f"{row['spearman']:>10.3f}{row['dead_frac']:>8.3f}")
        table.append(row)

    # Chance level: a random top-k% set overlaps ImageNet's at rate k.
    print("\nchance level  " + " " * 10 + "".join(f"{f:>8.3f}" for f in FRACS))

    # Correlation between distance and overlap, over datasets that have both.
    have = [r for r in table if r["mmd2_clip"] is not None]
    if len(have) >= 3:
        print("\nCorrelation of top-k overlap with CLIP MMD^2 distance "
              f"(n={len(have)} datasets):")
        for f in FRACS:
            x = np.array([r["mmd2_clip"] for r in have])
            y = np.array([r[f"overlap_{int(f*100)}"] for r in have])
            print(f"  k={int(f*100):>3}%   pearson={np.corrcoef(x, y)[0,1]:+.3f}   "
                  f"spearman={spearman(x, y):+.3f}")

    out = _root / "out/exp1_activity_overlap/overlap_table.json"
    with open(out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
