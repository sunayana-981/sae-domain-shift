#!/usr/bin/env python3
"""
exp1_activity_overlap.py — which SAE units does each domain use?

For a fixed backbone (base CLIP ViT-B/16, no adapter) and the pre-trained
ImageNet G-SAE, estimate per-unit activity on ImageNet and on each target
dataset, then measure how much the "important" unit sets overlap.

Motivation: the masked fine-tuning story assumes the top-activity units on
ImageNet are the ones carrying shared/generic content, so freezing them
preserves old concepts. That assumption is only interesting if the target
domain's own top units are a *different* set. This script measures that,
and lets the overlap be plotted against domain distance.

Two activity statistics are recorded per unit:
  activity  — mean |activation| over tokens (the criterion compute_protected_mask uses)
  fire_rate — fraction of tokens where the unit fires (act > 0)

Usage:
    python tasks/exp1_activity_overlap.py --datasets imagenet dtd eurosat
    python tasks/exp1_activity_overlap.py            # full roster
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
from datasets import load_dataset

from src.sae_training.config import ViTSAERunnerConfig
from src.sae_training.vit_activations_store import ViTActivationsStore
from tasks.utils import DATASET_INFO, get_classnames, load_hooked_vit, load_sae

ROSTER = [
    "imagenet", "caltech101", "oxford_pets", "dtd", "eurosat", "fgvc",
    "cub2002011", "ucf101", "officehome", "kitti", "cityscapes", "medmnist",
]


@torch.no_grad()
def estimate_activity(sae, store, n_batches, device):
    """Mean |activation| and fire rate per unit, over n_batches token batches."""
    sae.eval()
    act_sum = torch.zeros(sae.d_sae, device=device, dtype=torch.float64)
    fire_sum = torch.zeros(sae.d_sae, device=device, dtype=torch.float64)
    n_tokens = 0
    for _ in range(n_batches):
        x = store.get_batch_activations()
        _, acts, _ = sae.forward_standard(x)
        acts = acts.reshape(-1, sae.d_sae)
        act_sum += acts.abs().sum(dim=0).double()
        fire_sum += (acts > 0).sum(dim=0).double()
        n_tokens += acts.shape[0]
    return (act_sum / n_tokens).float().cpu().numpy(), (fire_sum / n_tokens).float().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae_checkpoint_path", default="data/sae_weight/base/out.pt")
    p.add_argument("--datasets", nargs="+", default=ROSTER)
    p.add_argument("--n_batches", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--block_layer", type=int, default=-2)
    p.add_argument("--model_name", default="openai/clip-vit-base-patch16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="out/exp1_activity_overlap/activity.npz")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[exp1] loading SAE {args.sae_checkpoint_path}")
    sae, sae_cfg = load_sae(args.sae_checkpoint_path, args.device)
    print(f"[exp1] d_in={sae.d_in} d_sae={sae.d_sae} layer={args.block_layer}")

    results = {}
    # Resume support: this loop is slow, so keep whatever is already done.
    if out_path.exists():
        prev = np.load(out_path)
        results = {k: prev[k] for k in prev.files}
        done = {k.rsplit("__", 1)[0] for k in results}
        print(f"[exp1] resuming, already have: {sorted(done)}")

    for name in args.datasets:
        if f"{name}__activity" in results:
            print(f"[exp1] skip {name} (cached)")
            continue
        if name not in DATASET_INFO:
            print(f"[exp1] WARN unknown dataset '{name}', skipping")
            continue

        t0 = time.time()
        print(f"\n[exp1] ===== {name} =====")
        cfg = ViTSAERunnerConfig(
            class_token=False,
            image_width=224, image_height=224,
            model_name=args.model_name,
            module_name="resid",
            block_layer=args.block_layer,
            dataset_path=DATASET_INFO[name]["path"],
            image_key="image", label_key="label",
            d_in=sae.d_in,
            expansion_factor=sae.d_sae // sae.d_in,
            batch_size=args.batch_size,
            device=args.device, seed=args.seed,
            dtype=torch.float32,
        )
        ds = load_dataset(**DATASET_INFO[name])
        classnames = get_classnames(name, ds)
        vit = load_hooked_vit(cfg, "base", args.model_name, args.device,
                              None, None, classnames, arch="clip")
        vit.eval()
        store = ViTActivationsStore(ds, args.batch_size, args.device, args.seed,
                                    vit, args.block_layer, "resid", False)

        activity, fire_rate = estimate_activity(sae, store, args.n_batches, args.device)
        results[f"{name}__activity"] = activity
        results[f"{name}__fire_rate"] = fire_rate
        np.savez(out_path, **results)
        print(f"[exp1] {name} done in {time.time()-t0:.0f}s  "
              f"mean={activity.mean():.3e} dead={(activity==0).mean():.3f}")

        del store, vit, ds
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n[exp1] saved {out_path}")


if __name__ == "__main__":
    main()
