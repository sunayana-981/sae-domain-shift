#!/usr/bin/env python3
"""
exp4_prune_sweep.py — how far can the SAE dictionary be compressed after training?

Keeps only the top-k% of SAE *units* (a global, permanent deletion from the
dictionary) and measures what that costs. This is a different axis from
eval_sae_topk.py, which keeps the top-K activations *per image* and leaves the
dictionary intact.

Pruning is implemented by zeroing the decoder rows of dropped units, so they
contribute nothing to the reconstruction while indices stay stable. b_dec is
left in place (it is not a unit).

Ranking criteria (all estimated on the reference domain, ImageNet):
    activity   mean |activation| per unit — what compute_protected_mask uses
    fire_rate  fraction of tokens on which the unit fires
    max_act    peak activation, which favours rare-but-sharp features
    random     control

Reported per (criterion, k):
    reconstruction MSE / cosine against the un-pruned activations
    zero-shot ImageNet top-1 with the pruned SAE spliced into the encoder
    surviving L0 and alive-unit count

Usage:
    python tasks/exp4_prune_sweep.py --keep_fracs 1.0 0.5 0.25 0.1 0.05 0.02
"""

import argparse
import json
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from eval_masked_sae import (
    ImageNetSubsetDataset,
    _hf_collate_fn,
    compute_accuracy_hf,
    get_processor_transform,
    get_text_features_hf,
    make_sae_hook_hf,
)
from src.sae_training.config import ViTSAERunnerConfig
from src.sae_training.vit_activations_store import ViTActivationsStore
from tasks.utils import DATASET_INFO, load_hooked_vit, load_sae


@torch.no_grad()
def collect_stats(sae, store, n_batches, device):
    """Per-unit activity, fire rate and peak activation on the reference domain."""
    sae.eval()
    act_sum = torch.zeros(sae.d_sae, device=device, dtype=torch.float64)
    fire_sum = torch.zeros(sae.d_sae, device=device, dtype=torch.float64)
    peak = torch.zeros(sae.d_sae, device=device)
    n = 0
    for _ in range(n_batches):
        x = store.get_batch_activations()
        _, acts, _ = sae.forward_standard(x)
        acts = acts.reshape(-1, sae.d_sae)
        act_sum += acts.abs().sum(0).double()
        fire_sum += (acts > 0).sum(0).double()
        peak = torch.maximum(peak, acts.max(0).values)
        n += acts.shape[0]
    return {
        "activity": (act_sum / n).float().cpu().numpy(),
        "fire_rate": (fire_sum / n).float().cpu().numpy(),
        "max_act": peak.cpu().numpy(),
    }


@torch.no_grad()
def reconstruction_and_l0(sae, store, keep_mask, n_batches, device):
    """MSE/cosine of the pruned SAE against the raw activations, plus L0."""
    sae.eval()
    mse, cos, l0, alive, n = 0.0, 0.0, 0.0, None, 0
    for _ in range(n_batches):
        x = store.get_batch_activations()
        flat = x.reshape(-1, sae.d_in)
        _, acts, _ = sae.forward_standard(x)
        acts = acts.reshape(-1, sae.d_sae) * keep_mask
        recon = acts @ sae.W_dec + sae.b_dec
        mse += (flat - recon).pow(2).mean().item()
        cos += torch.nn.functional.cosine_similarity(flat, recon, dim=-1).mean().item()
        l0 += (acts > 0).float().sum(-1).mean().item()
        fired = (acts > 0).any(0)
        alive = fired if alive is None else (alive | fired)
        n += 1
    return {
        "mse": mse / n,
        "cos_sim": cos / n,
        "l0": l0 / n,
        "alive_units": int(alive.sum().item()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae_checkpoint_path", default="data/sae_weight/base/out.pt")
    p.add_argument("--criteria", nargs="+",
                   default=["activity", "fire_rate", "max_act", "random"])
    p.add_argument("--keep_fracs", nargs="+", type=float,
                   default=[1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01])
    p.add_argument("--stat_batches", type=int, default=200)
    p.add_argument("--recon_batches", type=int, default=50)
    p.add_argument("--imagenet_n_samples", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_layer", type=int, default=-2)
    p.add_argument("--model_name", default="openai/clip-vit-base-patch16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip_accuracy", action="store_true")
    p.add_argument("--out", default="out/exp4_prune_sweep/results.json")
    args = p.parse_args()

    device = args.device
    torch.manual_seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sae, _ = load_sae(args.sae_checkpoint_path, device)
    print(f"[exp4] SAE d_in={sae.d_in} d_sae={sae.d_sae}")
    W_dec_orig = sae.W_dec.detach().clone()

    cfg = ViTSAERunnerConfig(
        class_token=False, image_width=224, image_height=224,
        model_name=args.model_name, module_name="resid",
        block_layer=args.block_layer,
        dataset_path=DATASET_INFO["imagenet"]["path"],
        image_key="image", label_key="label",
        d_in=sae.d_in, expansion_factor=sae.d_sae // sae.d_in,
        batch_size=args.batch_size, device=device, seed=args.seed,
        dtype=torch.float32,
    )
    imagenet_ds = load_dataset(**DATASET_INFO["imagenet"])
    vit = load_hooked_vit(cfg, "base", args.model_name, device)
    vit.eval()

    store = ViTActivationsStore(imagenet_ds, args.batch_size, device, args.seed,
                                vit, args.block_layer, "resid", False)

    print(f"[exp4] estimating unit stats over {args.stat_batches} batches...")
    stats = collect_stats(sae, store, args.stat_batches, device)
    rng = np.random.RandomState(args.seed)
    stats["random"] = rng.rand(sae.d_sae).astype(np.float32)

    # Zero-shot ImageNet setup
    text_feat, loader, classnames = None, None, None
    if not args.skip_accuracy:
        classnames = []
        with open("configs/classnames/imagenet_classnames.txt") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    classnames.append(parts[1])
        subset = ImageNetSubsetDataset(imagenet_ds, n_samples=args.imagenet_n_samples,
                                       transform=get_processor_transform(), seed=42)
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=_hf_collate_fn)
        text_feat = get_text_features_hf(vit, device, classnames, "base")
        acc_no_sae = compute_accuracy_hf(vit, text_feat, loader, "base", device)
        print(f"[exp4] ImageNet zero-shot, no SAE: {acc_no_sae:.2f}%")
    else:
        acc_no_sae = None

    results = {
        "sae_checkpoint": args.sae_checkpoint_path,
        "d_sae": int(sae.d_sae),
        "imagenet_no_sae_acc": acc_no_sae,
        "runs": [],
    }

    done_full = False   # keeping every unit is criterion-independent; measure it once
    for crit in args.criteria:
        score = torch.from_numpy(stats[crit]).to(device)
        order = torch.argsort(score, descending=True)
        for kf in args.keep_fracs:
            if kf >= 1.0:
                if done_full:
                    continue
                done_full = True
            k = max(1, round(kf * sae.d_sae))
            keep_idx = order[:k]
            keep_mask = torch.zeros(sae.d_sae, device=device)
            keep_mask[keep_idx] = 1.0

            rec = reconstruction_and_l0(sae, store, keep_mask, args.recon_batches, device)

            acc = None
            if not args.skip_accuracy:
                # Prune for real: dropped units contribute nothing to the decode.
                sae.W_dec.data.copy_(W_dec_orig * keep_mask.unsqueeze(1))
                hooks = make_sae_hook_hf(sae, cfg, "base")
                acc = compute_accuracy_hf(vit, text_feat, loader, "base", device, hooks)
                sae.W_dec.data.copy_(W_dec_orig)

            row = {"criterion": crit, "keep_frac": kf, "n_kept": k,
                   "imagenet_acc": acc, **rec}
            results["runs"].append(row)
            print(f"[exp4] {crit:>9s} keep={kf:<5} k={k:<6d} "
                  f"mse={rec['mse']:.5f} cos={rec['cos_sim']:.4f} "
                  f"L0={rec['l0']:.1f} alive={rec['alive_units']:<6d} "
                  f"acc={acc if acc is None else f'{acc:.2f}'}")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\n[exp4] saved {out_path}")


if __name__ == "__main__":
    main()
