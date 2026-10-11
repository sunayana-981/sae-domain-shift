#!/usr/bin/env python3
"""
eval_expanded_matrix_reconstruction.py — reconstruction-quality sweep for the
expanded backbone matrix (ALIGN, DINOv2, SigLIP2 x JumpReLU/TopK x 9 datasets).

Supersedes an attempt at zero-shot *accuracy* for this matrix: a destructive-hook
test (zero out the activation, check if the model's final output changes)
showed that this repo's forward-hook splicing silently no-ops for ALIGN and
SigLIP2 (fires, but the replacement never reaches downstream compute), and
DINOv2 has no text tower at all, so "zero-shot accuracy" isn't defined for it
regardless. See conversation notes; not re-litigated here.

This sidesteps that bug entirely: it never splices a reconstruction back into
a live forward pass. It only needs read-only activation extraction via
ViTActivationsStore -- the SAME code path used to train these checkpoints in
the first place (train_sae_masked_finetune.py), so it is proven correct for
every backbone here, unlike write-back hooking which was never exercised this
way before.

Metrics per (backbone, variant, dataset, split in {own, imagenet}):
    mse            reconstruction MSE
    cos_sim        cosine similarity, reconstruction vs original
    l0             mean number of active units per token
    dead_fraction  fraction of d_sae units that never fire in the sample

Usage:
    python tasks/eval_expanded_matrix_reconstruction.py --archs align dino siglip2
"""

import argparse
import gc
import json
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
from datasets import load_dataset

from src.sae_training.backbone_registry import get_backbone_spec
from src.sae_training.config import ViTSAERunnerConfig
from src.sae_training.vit_activations_store import ViTActivationsStore
from tasks.utils import DATASET_INFO, load_hooked_vit, load_sae

EXPANDED_ROOT = Path(_root) / "out/checkpoints/masked_finetune_expanded"
BASE_ROOT = Path(_root) / "out/checkpoints/base_expanded"
ARCHS = ["align", "dino", "siglip2"]
VARIANTS = ["jumprelu", "topk"]
DATASETS = ["caltech101", "cityscapes", "cub2002011", "dtd", "eurosat",
            "fgvc", "kitti", "officehome", "ucf101"]
BLOCK_LAYER = -2


@torch.no_grad()
def measure(sae, store, n_batches, device):
    """Reconstruction MSE / cosine sim / L0 / dead-fraction over n_batches."""
    sae.eval()
    d_sae = sae.d_sae
    mse_sum = cos_sum = l0_sum = 0.0
    ever_fired = torch.zeros(d_sae, dtype=torch.bool, device=device)
    n = 0
    for _ in range(n_batches):
        x = store.get_batch_activations()
        flat = x.reshape(-1, sae.d_in)
        sae_out, acts, _ = sae.forward(flat)
        mse_sum += (flat - sae_out).pow(2).mean().item()
        cos_sum += torch.nn.functional.cosine_similarity(flat, sae_out, dim=-1).mean().item()
        l0_sum += (acts > 0).float().sum(-1).mean().item()
        ever_fired |= (acts > 0).any(0)
        n += 1
    return {
        "mse": mse_sum / n, "cos_sim": cos_sum / n, "l0": l0_sum / n,
        "dead_fraction": float((~ever_fired).float().mean().item()),
    }


def find_ckpt(root, key):
    matches = list(root.glob(f"{key}/*/final_sparse_autoencoder_*/*.pt"))
    return str(matches[0]) if matches else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="+", default=ARCHS, choices=ARCHS)
    p.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--n_batches", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_path", default="out/eval_expanded_matrix/reconstruction.json")
    args = p.parse_args()

    device = args.device
    torch.manual_seed(args.seed)
    out_path = Path(_root) / args.save_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.load(open(out_path)) if out_path.exists() else {}

    imagenet_ds = None

    for arch in args.archs:
        print(f"\n{'='*70}\nARCH: {arch}\n{'='*70}")
        spec = get_backbone_spec(arch)
        model_id = spec.default_model_id
        d_in = spec.hidden_dim

        cfg = ViTSAERunnerConfig(
            class_token=False, image_width=224, image_height=224,
            model_name=model_id, module_name="resid", block_layer=BLOCK_LAYER,
            dataset_path=DATASET_INFO["caltech101"]["path"],
            image_key="image", label_key="label",
            d_in=d_in, expansion_factor=64, batch_size=args.batch_size,
            device=device, seed=args.seed, dtype=torch.float32,
        )
        vit = load_hooked_vit(cfg, "base", model_id, device, arch=arch)
        vit.eval()

        if imagenet_ds is None:
            imagenet_ds = load_dataset(**DATASET_INFO["imagenet"])
        in_store = ViTActivationsStore(imagenet_ds, args.batch_size, device, args.seed,
                                       vit, BLOCK_LAYER, "resid", False)

        results.setdefault(arch, {})
        for variant in args.variants:
            base_ckpt = find_ckpt(BASE_ROOT, f"{arch}_{variant}")
            if not base_ckpt:
                print(f"[WARN] no base_expanded checkpoint for {arch}_{variant}")
                continue
            base_sae, _ = load_sae(base_ckpt, device)
            print(f"[{arch}/{variant}] base SAE on ImageNet...")
            base_in_metrics = measure(base_sae, in_store, args.n_batches, device)
            print(f"  {base_in_metrics}")
            del base_sae
            gc.collect(); torch.cuda.empty_cache()

            for dataset in args.datasets:
                key = f"{dataset}/{variant}"
                if key in results[arch] and "own" in results[arch][key]:
                    print(f"[SKIP] {arch}/{key} already done")
                    continue
                ckpt = find_ckpt(EXPANDED_ROOT, f"{arch}_{variant}/{dataset}")
                if not ckpt:
                    continue
                print(f"\n--- {arch} / {dataset} / {variant} ---")
                try:
                    ds = load_dataset(**DATASET_INFO[dataset])
                except Exception as e:
                    print(f"[ERROR] loading {dataset}: {e}")
                    continue
                own_store = ViTActivationsStore(ds, args.batch_size, device, args.seed,
                                                vit, BLOCK_LAYER, "resid", False)
                sae, _ = load_sae(ckpt, device)
                own_metrics = measure(sae, own_store, args.n_batches, device)
                in_metrics = measure(sae, in_store, args.n_batches, device)
                print(f"  own:      {own_metrics}")
                print(f"  imagenet: {in_metrics}  (base-SAE ref: {base_in_metrics})")

                results[arch][key] = {
                    "checkpoint": ckpt, "own": own_metrics, "imagenet": in_metrics,
                    "imagenet_base_sae_ref": base_in_metrics,
                }
                json.dump(results, open(out_path, "w"), indent=2)
                del sae, own_store
                gc.collect(); torch.cuda.empty_cache()

        del vit, in_store
        gc.collect(); torch.cuda.empty_cache()

    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
