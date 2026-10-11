#!/usr/bin/env python3
"""
exp5_residual_analysis.py — is the domain shift a small, low-dimensional residual?

The proposed mechanism for why fine-tuning only low-activity units works is:
the frozen high-activity units already reconstruct almost everything, so what
is left for the free units to encode is a small residual of low intrinsic
dimension — a handful of directions, which even a tiny free budget can span.

This measures that directly, on LoRA-adapted target-domain activations:

  1. Residual magnitude.  With only the protected units decoding, how much of
     the activation is left unexplained (relative residual norm, R^2)?
  2. Residual dimensionality.  Effective rank of the residual covariance
     (participation ratio of the eigenvalue spectrum) and the number of
     components needed for 90% / 95% of residual energy.
  3. Recruitment.  After masked fine-tuning, how many free units actually fire,
     and how much of the residual energy do their decoder directions span?

Usage:
    python tasks/exp5_residual_analysis.py --dataset pathmnist --condition masked_p80
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

from src.sae_training.config import ViTSAERunnerConfig
from src.sae_training.vit_activations_store import ViTActivationsStore
from tasks.utils import DATASET_INFO, get_classnames, load_hooked_vit, load_sae

LORA_ROOT = Path("/path/to/sae-domain-shift/lora_weights/vitb16")
LORA_KEY = {"pathmnist": "medmnist", "caltech101": "caltech101"}
TRAIN_KEY = {"pathmnist": "medmnist", "caltech101": "caltech101"}


def effective_rank(eigvals):
    """Participation ratio: (sum l)^2 / sum l^2. 1 for rank-1, n for flat spectrum."""
    l = np.clip(eigvals, 0, None)
    return float(l.sum() ** 2 / max((l ** 2).sum(), 1e-30))


def components_for(eigvals, frac):
    l = np.sort(np.clip(eigvals, 0, None))[::-1]
    c = np.cumsum(l) / max(l.sum(), 1e-30)
    return int(np.searchsorted(c, frac) + 1)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="pathmnist", choices=list(TRAIN_KEY))
    p.add_argument("--condition", default="masked_p80")
    p.add_argument("--base_sae", default="data/sae_weight/base/out.pt")
    p.add_argument("--registry", default="out/controlled/sae_registry.json")
    p.add_argument("--activity_batches", type=int, default=50)
    p.add_argument("--n_batches", type=int, default=60)
    p.add_argument("--chunk_size", type=int, default=2048,
                   help="Tokens encoded at once. The (chunk, d_sae) activation "
                        "matrix is the memory bottleneck, not the token count.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--block_layer", type=int, default=-2)
    p.add_argument("--model_name", default="openai/clip-vit-base-patch16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="out/exp5_residual/results.json")
    args = p.parse_args()

    device = args.device
    torch.manual_seed(args.seed)

    registry = json.load(open(args.registry))
    match = [r for r in registry
             if r["dataset"] == args.dataset and r["condition"] == args.condition]
    if not match:
        sys.exit(f"[exp5] no registry entry for {args.dataset}/{args.condition}")
    ft_path = match[-1]["checkpoint_path"]
    print(f"[exp5] base SAE      {args.base_sae}")
    print(f"[exp5] finetuned SAE {ft_path}")

    base_sae, _ = load_sae(args.base_sae, device)
    ft_sae, _ = load_sae(ft_path, device)
    d_sae, d_in = base_sae.d_sae, base_sae.d_in

    cfg = ViTSAERunnerConfig(
        class_token=False, image_width=224, image_height=224,
        model_name=args.model_name, module_name="resid",
        block_layer=args.block_layer,
        dataset_path=DATASET_INFO[TRAIN_KEY[args.dataset]]["path"],
        image_key="image", label_key="label",
        d_in=d_in, expansion_factor=d_sae // d_in,
        batch_size=args.batch_size, device=device, seed=args.seed,
        dtype=torch.float32,
    )

    # --- reference (ImageNet) activity on the BASE backbone defines protection ---
    print("[exp5] estimating ImageNet activity on base CLIP to rebuild the mask...")
    in_ds = load_dataset(**DATASET_INFO["imagenet"])
    base_vit = load_hooked_vit(cfg, "base", args.model_name, device)
    base_vit.eval()
    in_store = ViTActivationsStore(in_ds, args.batch_size, device, args.seed,
                                   base_vit, args.block_layer, "resid", False)
    acc = torch.zeros(d_sae, device=device)
    for _ in range(args.activity_batches):
        _, a, _ = base_sae.forward_standard(in_store.get_batch_activations())
        acc += a.abs().reshape(-1, d_sae).mean(0)
    activity = acc / args.activity_batches
    del in_store, in_ds, base_vit
    torch.cuda.empty_cache()

    results = {"dataset": args.dataset, "condition": args.condition,
               "finetuned_checkpoint": ft_path, "d_sae": int(d_sae), "protect": {}}

    # --- target-domain activations on the LoRA backbone ---
    from tasks.train_sae_lora_clip import load_lora_weights
    ds = load_dataset(**DATASET_INFO[TRAIN_KEY[args.dataset]])
    classnames = get_classnames(TRAIN_KEY[args.dataset], ds)
    vit = load_hooked_vit(cfg, "base", args.model_name, device, None, None, classnames, arch="clip")
    vit = load_lora_weights(vit, str(LORA_ROOT / LORA_KEY[args.dataset] / "16shots/seed1/lora_weights.pt"), device)
    vit.eval()
    store = ViTActivationsStore(ds, args.batch_size, device, args.seed,
                                vit, args.block_layer, "resid", False)

    print(f"[exp5] collecting {args.n_batches} batches of target activations...")
    X = torch.cat([store.get_batch_activations().reshape(-1, d_in).cpu()
                   for _ in range(args.n_batches)], dim=0).to(device)
    print(f"[exp5] X: {tuple(X.shape)}")

    chunk = args.chunk_size
    for protect_frac in (0.8, 0.9, 0.95, 0.98):
        k = round(protect_frac * d_sae)
        prot = torch.zeros(d_sae, device=device)
        prot[torch.topk(activity, k=k).indices] = 1.0
        prot_b = prot.bool()
        free_b = ~prot_b

        # Pass 1: stream the residual, accumulating only d_in x d_in moments.
        # Encoding all tokens at once would need an (N, d_sae) matrix -- tens of GB.
        sum_R = torch.zeros(d_in, device=device, dtype=torch.float64)
        sum_RR = torch.zeros(d_in, d_in, device=device, dtype=torch.float64)
        sum_X = torch.zeros(d_in, device=device, dtype=torch.float64)
        sum_XX = torch.zeros(d_in, d_in, device=device, dtype=torch.float64)
        rel_sum = 0.0
        free_fired = torch.zeros(d_sae, device=device, dtype=torch.bool)
        free_l0_sum = prot_l0_sum = 0.0
        n = 0
        R_cpu = []

        for i in range(0, X.shape[0], chunk):
            xb = X[i:i + chunk]
            _, a, _ = base_sae.forward_standard(xb)
            recon = (a * prot) @ base_sae.W_dec + base_sae.b_dec
            R = xb - recon
            del a, recon

            sum_R += R.sum(0).double()
            sum_RR += (R.T.double() @ R.double())
            sum_X += xb.sum(0).double()
            sum_XX += (xb.T.double() @ xb.double())
            rel_sum += (R.norm(dim=-1) / xb.norm(dim=-1).clamp_min(1e-8)).sum().item()
            n += xb.shape[0]
            R_cpu.append(R.cpu())

            _, af, _ = ft_sae.forward_standard(xb)
            fired = af > 0
            free_fired |= (fired & free_b).any(0)
            free_l0_sum += (fired & free_b).sum(-1).sum().item()
            prot_l0_sum += (fired & prot_b).sum(-1).sum().item()
            del af, fired, R
            torch.cuda.empty_cache()

        mean_R = sum_R / n
        cov_R = (sum_RR - n * torch.outer(mean_R, mean_R)) / (n - 1)
        mean_X = sum_X / n
        cov_X = (sum_XX - n * torch.outer(mean_X, mean_X)) / (n - 1)
        ev = torch.linalg.eigvalsh(cov_R).cpu().numpy()[::-1]

        rel = rel_sum / n
        r2 = 1.0 - float(torch.diagonal(cov_R).sum() / torch.diagonal(cov_X).sum().clamp_min(1e-12))

        n_free_alive = int(free_fired.sum().item())
        free_l0 = free_l0_sum / n
        prot_l0 = prot_l0_sum / n

        # Pass 2: how much residual energy lies in the span of the decoder
        # directions the fine-tune actually woke up?
        span_frac = None
        if n_free_alive > 0:
            D = ft_sae.W_dec[free_fired].double()
            Q, _ = torch.linalg.qr(D.T)
            num = den = 0.0
            mR = mean_R.float()
            for Rb in R_cpu:
                Rc = (Rb.to(device) - mR).double()
                num += (Rc @ Q).pow(2).sum().item()
                den += Rc.pow(2).sum().item()
            span_frac = num / max(den, 1e-30)
        del R_cpu
        torch.cuda.empty_cache()

        row = {
            "n_protected": k,
            "n_free": int(d_sae - k),
            "relative_residual_norm": rel,
            "r2_explained_by_protected": r2,
            "residual_effective_rank": effective_rank(ev),
            "residual_dims_90pct": components_for(ev, 0.90),
            "residual_dims_95pct": components_for(ev, 0.95),
            "n_free_units_alive_after_ft": n_free_alive,
            "free_l0_after_ft": free_l0,
            "protected_l0_after_ft": prot_l0,
            "residual_energy_spanned_by_recruited_free_units": span_frac,
        }
        results["protect"][f"{protect_frac}"] = row
        print(f"[exp5] protect={protect_frac}: rel_resid={rel:.4f} R2={r2:.4f} "
              f"eff_rank={row['residual_effective_rank']:.1f} "
              f"dims90={row['residual_dims_90pct']} "
              f"free_alive={n_free_alive} free_L0={free_l0:.2f} "
              f"span={span_frac if span_frac is None else f'{span_frac:.4f}'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.load(open(out)) if out.exists() else {}
    prev[f"{args.dataset}__{args.condition}"] = results
    json.dump(prev, open(out, "w"), indent=2)
    print(f"[exp5] saved {out}")


if __name__ == "__main__":
    main()
