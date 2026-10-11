#!/usr/bin/env python3
"""
Does masked fine-tuning make a feature's top-activating images stay loyal to
the target domain even when reference-domain (ImageNet) images are put back
in front of it?

Same class-selective-feature machinery as top_activating_grid.py (feature
selection still uses the target dataset's own per-class activation profile,
unchanged), but the top-9-image search runs over a POOL that mixes the
target dataset's eval images with a random sample of real ImageNet-val
images. If a G-SAE feature's "class-selective" signal was actually riding on
generic ImageNet-like texture, ImageNet images should crowd its top-9 out;
if masked fine-tuning genuinely specialised a feature to the target domain,
its top-9 should stay target-domain images even with ImageNet in the pool.

Usage:
    python tasks/top_activating_grid_mixed.py --dataset ucf101 \
        --sae_paths configs/controlled_sae_paths.json --n_select 20 \
        --mix_imagenet_n 500
"""
import argparse
import itertools
import json
import os
import random
import sys
import time
import heapq
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from top_activating import get_sae_acts
from eval_medmnist_sae import _hf_collate_fn
from tasks.controlled_common import (
    add_common_args, flush, load_common_registry_args, build_dataset_splits,
    get_vit, resolve_sae_path, get_sae,
)
from tasks.eval_steering import get_class_profile
from tasks.top_activating_grid import (
    select_class_selective_features, batched_patch_acts,
    save_feature_grid,
)

QUALITATIVE_SAE_CONDITIONS = ["gsae", "ftsae", "scratchsae", "masked", "masked_gated"]
DEFAULT_IMAGENET_VAL_DIR = "/path/to/sae-domain-shift/data/imagenet/val"
DEFAULT_IMAGENET_CLASSNAMES = str(
    Path(_project_root) / "configs/classnames/imagenet_classnames.txt")


# ═════════════════════════════════════════════════════════════════════════
# Mixed target-domain + ImageNet pool
# ═════════════════════════════════════════════════════════════════════════

def load_imagenet_classnames(path):
    """'<synset> <name>' per line -> {synset: name}."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            synset, name = line.split(" ", 1)
            out[synset] = name
    return out


class MixedSourcePool(Dataset):
    """Concatenates a target-domain eval split with a random ImageNet-val
    sample, both yielding (PIL image, int label) against one shared,
    source-tagged classnames list, so the downstream image title reveals
    which pool a top-activating image actually came from."""

    def __init__(self, target_ds, target_classnames, imagenet_items, imagenet_classnames):
        self.target_ds = target_ds
        self.n_target = len(target_ds)
        self.imagenet_items = imagenet_items  # list of (path, local_class_idx)
        self.classnames = (
            [f"[{{tag}}] {c}".format(tag="TARGET") for c in target_classnames]
            + [f"[IMAGENET] {c}" for c in imagenet_classnames]
        )
        self._offset = len(target_classnames)

    def __len__(self):
        return self.n_target + len(self.imagenet_items)

    def __getitem__(self, idx):
        if idx < self.n_target:
            return self.target_ds[idx]
        path, local_idx = self.imagenet_items[idx - self.n_target]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return img, self._offset + local_idx


def build_imagenet_sample(imagenet_dir, classnames_path, n, seed):
    """Randomly samples n (path, class_idx) pairs from an ImageFolder-style
    ImageNet val directory, plus the matching source classnames list (only
    the classes actually drawn, in the order ImageFolder assigns indices for
    those classes among themselves)."""
    imagenet = ImageFolder(imagenet_dir)
    synset_to_name = load_imagenet_classnames(classnames_path)
    rng = random.Random(seed)
    idx = rng.sample(range(len(imagenet.samples)), min(n, len(imagenet.samples)))

    # Remap to a compact 0..k-1 local index over only the classes drawn, in
    # first-seen order, so the classnames list isn't 1000 entries long.
    local_idx_of_class = {}
    items = []
    local_names = []
    for i in idx:
        path, global_cls = imagenet.samples[i]
        if global_cls not in local_idx_of_class:
            local_idx_of_class[global_cls] = len(local_names)
            synset = imagenet.classes[global_cls]
            local_names.append(synset_to_name.get(synset, synset))
        items.append((path, local_idx_of_class[global_cls]))
    return items, local_names


# ═════════════════════════════════════════════════════════════════════════
# Streaming top-N images per feature over the mixed pool
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def find_top_activating_images_mixed(vit, sae, cfg, dataset, feature_ids, classnames, args,
                                      top_n=9, max_batches=None):
    heaps = {fid: [] for fid in feature_ids}
    counter = itertools.count()
    fid_tensor = torch.tensor(feature_ids, device=args.device)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         collate_fn=_hf_collate_fn, num_workers=2)

    for bidx, (images, labels) in enumerate(tqdm(loader, desc="streaming mixed pool", leave=False)):
        if max_batches is not None and bidx >= max_batches:
            break
        acts = batched_patch_acts(vit, sae, cfg, images, args)
        sel = acts.index_select(-1, fid_tensor)
        max_per_feat, _ = sel.max(dim=1)
        max_per_feat = max_per_feat.cpu().numpy()

        for bi in range(len(images)):
            classname = classnames[labels[bi].item()]
            for fi, fid in enumerate(feature_ids):
                val = float(max_per_feat[bi, fi])
                item = (val, next(counter), images[bi].copy(), classname)
                if len(heaps[fid]) < top_n:
                    heapq.heappush(heaps[fid], item)
                elif val > heaps[fid][0][0]:
                    heapq.heapreplace(heaps[fid], item)

    return {fid: sorted(heaps[fid], key=lambda x: -x[0]) for fid in feature_ids}


def save_comparison_figure_mixed(selections_by_sae, top_items_by_sae, out_path, row_order):
    row_order = [s for s in row_order if s in selections_by_sae]
    ncols = max(len(selections_by_sae[s]) for s in row_order)
    fig, axes = plt.subplots(len(row_order), ncols,
                              figsize=(4.5 * ncols, 5.0 * len(row_order)))
    axes = np.atleast_2d(axes)

    for r, sae_condition in enumerate(row_order):
        selection = selections_by_sae[sae_condition]
        for c in range(ncols):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(selection):
                continue
            feat = selection[c]
            items = top_items_by_sae[sae_condition].get(feat["feature_id"], [])
            if not items:
                continue
            act, _, img, classname = items[0]
            ax.imshow(img)
            source = "IN" if classname.startswith("[TARGET]") else "OUT"
            color = "green" if source == "IN" else "red"
            ax.set_title(f"f{feat['feature_id']}  {classname}\nact={act:.2f}",
                          fontsize=12, color=color)
        row_top = 1 - r / len(row_order)
        row_bottom = 1 - (r + 1) / len(row_order)
        fig.text(0.01, (row_top + row_bottom) / 2, sae_condition.upper(),
                  fontsize=12, fontweight="bold", rotation=90, va="center", ha="left")

    fig.suptitle("Top-activating feature (rank 1) under a mixed target+ImageNet pool, "
                  "green=target domain won, red=ImageNet won\n"
                  + " vs ".join(s.upper() for s in row_order), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default="ucf101")
    p.add_argument("--vit_type", type=str, default="lora", choices=["lora"])
    p.add_argument("--sae_conditions", nargs="+", default=["gsae", "masked"],
                    choices=QUALITATIVE_SAE_CONDITIONS)
    p.add_argument("--n_candidates", type=int, default=500)
    p.add_argument("--n_select", type=int, default=20)
    p.add_argument("--top_n", type=int, default=9)
    p.add_argument("--mix_imagenet_n", type=int, default=500,
                    help="Random ImageNet-val images added to the top-9 search pool.")
    p.add_argument("--imagenet_dir", type=str, default=DEFAULT_IMAGENET_VAL_DIR)
    p.add_argument("--imagenet_classnames", type=str, default=DEFAULT_IMAGENET_CLASSNAMES)
    p = add_common_args(p, out_dir_default="out/controlled/qualitative_mixed")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(dataset_out_dir, exist_ok=True)
    if args.reuse_cache:
        os.makedirs(args.cache_dir, exist_ok=True)

    setattr(args, "datasets", [args.dataset])
    registry, lora_checkpoints, sae_paths = load_common_registry_args(args)

    print("=" * 78)
    print(f"TOP-ACTIVATING GRID -- MIXED WITH IMAGENET{'  [DRY RUN]' if args.dry_run else ''}")
    print(f"dataset={args.dataset}  mix_imagenet_n={args.mix_imagenet_n}  "
          f"sae_conditions={args.sae_conditions}")
    print("=" * 78)

    classnames, train_ds, eval_ds = build_dataset_splits(args.dataset, registry[args.dataset], args)
    num_classes = len(classnames)
    print(f"[OK] dataset '{args.dataset}': {len(train_ds)} train / {len(eval_ds)} eval images, "
          f"{num_classes} classes")

    print(f"[OK] sampling {args.mix_imagenet_n} random ImageNet-val images from {args.imagenet_dir}")
    imagenet_items, imagenet_local_names = build_imagenet_sample(
        args.imagenet_dir, args.imagenet_classnames, args.mix_imagenet_n, args.seed)
    print(f"[OK] drew {len(imagenet_items)} images across {len(imagenet_local_names)} ImageNet classes")

    mixed_pool = MixedSourcePool(eval_ds, classnames, imagenet_items, imagenet_local_names)
    mixed_classnames = mixed_pool.classnames
    print(f"[OK] mixed pool: {len(eval_ds)} {args.dataset} + {len(imagenet_items)} ImageNet "
          f"= {len(mixed_pool)} images")

    vit = get_vit(args.vit_type, args.dataset, lora_checkpoints, args)
    if vit is None:
        print(f"[FATAL] missing lora checkpoint for dataset '{args.dataset}'")
        sys.exit(1)

    max_batches = 1 if args.dry_run else None
    n_select = 2 if args.dry_run else args.n_select
    top_n = 3 if args.dry_run else args.top_n

    index_rows = []
    selections_by_sae, top_items_by_sae = {}, {}

    for sae_condition in args.sae_conditions:
        sae_path = resolve_sae_path(sae_condition, args.dataset, sae_paths, args)
        if not sae_path or not os.path.exists(sae_path):
            print(f"[SKIP] {sae_condition}: missing sae checkpoint")
            continue
        sae, cfg = get_sae(sae_path, args.device)

        print(f"\n[{sae_condition}] computing per-class activation profile on {args.dataset}...")
        profile = get_class_profile(vit, sae, cfg, train_ds, num_classes, args.dataset,
                                     args.vit_type, sae_condition, args, max_batches)

        selection = select_class_selective_features(profile, classnames, args.n_candidates, n_select)
        print(f"[{sae_condition}] selected {len(selection)} class-selective features: "
              + ", ".join(f"f{s['feature_id']}(H={s['label_entropy']:.2f},{s['dominant_class']})"
                           for s in selection))

        feature_ids = [s["feature_id"] for s in selection]
        top_items = find_top_activating_images_mixed(
            vit, sae, cfg, mixed_pool, feature_ids, mixed_classnames, args,
            top_n=top_n, max_batches=max_batches)

        sae_out_dir = os.path.join(dataset_out_dir, sae_condition)
        os.makedirs(sae_out_dir, exist_ok=True)
        for feat in selection:
            fid = feat["feature_id"]
            items = top_items.get(fid, [])
            fig_path = os.path.join(sae_out_dir, f"feature_{fid}.png")
            if items:
                save_feature_grid(items, fid, sae_condition, fig_path)
            n_target = sum(1 for it in items if it[3].startswith("[TARGET]"))
            n_imagenet = sum(1 for it in items if it[3].startswith("[IMAGENET]"))
            index_rows.append(dict(sae_condition=sae_condition, feature_id=fid,
                                    label_entropy=feat["label_entropy"],
                                    dominant_class=feat["dominant_class"],
                                    mean_act=feat["mean_act"],
                                    n_images_found=len(items),
                                    n_target_domain_in_top9=n_target,
                                    n_imagenet_in_top9=n_imagenet))

        selections_by_sae[sae_condition] = selection
        top_items_by_sae[sae_condition] = top_items
        flush()

    index_path = os.path.join(dataset_out_dir, "index.csv")
    pd.DataFrame(index_rows, columns=["sae_condition", "feature_id", "label_entropy",
                                       "dominant_class", "mean_act", "n_images_found",
                                       "n_target_domain_in_top9", "n_imagenet_in_top9"]
                 ).to_csv(index_path, index=False)
    print(f"\nWrote {index_path}")

    if len(selections_by_sae) >= 2:
        suffix = "_vs_".join(args.sae_conditions)
        comp_path = os.path.join(dataset_out_dir, f"comparison_{suffix}.png")
        save_comparison_figure_mixed(selections_by_sae, top_items_by_sae, comp_path,
                                      row_order=args.sae_conditions)
        print(f"Wrote {comp_path}")
    else:
        print("[SKIP] comparison figure: need >=2 sae_conditions with results")

    # Summary: fraction of each condition's top-9 slots won by the target domain.
    for sae_condition in selections_by_sae:
        rows = [r for r in index_rows if r["sae_condition"] == sae_condition]
        total_slots = sum(r["n_images_found"] for r in rows)
        total_target = sum(r["n_target_domain_in_top9"] for r in rows)
        frac = total_target / total_slots if total_slots else float("nan")
        print(f"[{sae_condition}] target-domain share of top-9 slots across "
              f"{len(rows)} features: {total_target}/{total_slots} ({frac:.1%})")

    summary_path = os.path.join(dataset_out_dir, f"{'dry_run_' if args.dry_run else ''}summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dry_run": args.dry_run,
            "args": {k: v for k, v in vars(args).items()},
            "sae_conditions_found": list(selections_by_sae.keys()),
            "n_features_per_sae": {k: len(v) for k, v in selections_by_sae.items()},
            "n_imagenet_classes_drawn": len(imagenet_local_names),
        }, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
