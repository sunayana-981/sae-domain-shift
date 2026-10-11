#!/usr/bin/env python3
"""
eval_expanded_backbone_sweep.py — zero-shot accuracy for the expanded backbone
matrix (ALIGN, SigLIP2; DINOv2 has no text tower and is out of scope here —
see the module docstring bottom for why).

Mirrors eval_masked_finetune_sweep.py's framing (no-SAE / base-SAE /
masked-SAE, own-dataset + ImageNet) for the masked_finetune_expanded /
base_expanded checkpoints trained on ALIGN and SigLIP2 (run_masked_finetune_
expanded.sh: base backbone, no adapter, protect_frac=0.2, block_layer=-2).

Each backbone's own HF model.get_image_features()/get_text_features() is used
for zero-shot classification; the SAE reconstruction hook is spliced into the
vision tower via the same Hook/HookedVisionTransformer machinery used
throughout this codebase, so get_image_features() naturally reflects the
SAE-reconstructed pathway.

Usage:
    python tasks/eval_expanded_backbone_sweep.py --archs align siglip2
    python tasks/eval_expanded_backbone_sweep.py --archs align --datasets eurosat dtd
"""

import argparse
import gc
import glob
import json
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.sae_training.config import ViTSAERunnerConfig
from src.sae_training.hooked_vit import Hook
from src.models.templates.openai_imagenet_templates import openai_imagenet_template
from tasks.utils import DATASET_INFO, get_classnames, load_hooked_vit, load_sae

EXPANDED_ROOT = Path(_root) / "out/checkpoints/masked_finetune_expanded"
BASE_ROOT = Path(_root) / "out/checkpoints/base_expanded"
VARIANTS = ["jumprelu", "topk"]
DATASETS = ["caltech101", "cityscapes", "cub2002011", "dtd", "eurosat",
            "fgvc", "kitti", "officehome", "ucf101"]
BLOCK_LAYER = -2


class HFImageDataset(Dataset):
    """Generic image+label dataset over an HF `datasets` split, preprocessed
    with a given processor's image transform at __getitem__ time."""
    def __init__(self, hf_dataset, processor, n_samples=None, seed=42):
        self.ds = hf_dataset
        self.processor = processor
        if n_samples is not None and n_samples < len(hf_dataset):
            rng = np.random.RandomState(seed)
            self.idx = sorted(rng.choice(len(hf_dataset), n_samples, replace=False).tolist())
        else:
            self.idx = list(range(len(hf_dataset)))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        item = self.ds[self.idx[i]]
        img = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        label = item["label"]
        px = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return px, label


def _collate(batch):
    px = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    return px, labels


@torch.no_grad()
def build_text_features(model, processor, classnames, device, n_templates=3):
    """Multi-template zero-shot text features, per-class averaged & renormalized.
    Capped at n_templates (default 3, not the full ~80) to keep the sweep's
    per-checkpoint text-encoding cost down across ~50 checkpoints x 2 datasets."""
    templates = openai_imagenet_template[:n_templates]
    feats = []
    for name in classnames:
        prompts = [t(name) for t in templates]
        toks = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        tf = model.get_text_features(**toks)
        tf = F.normalize(tf.float(), dim=-1).mean(0)
        feats.append(F.normalize(tf, dim=-1))
    return torch.stack(feats)  # [C, D]


@torch.no_grad()
def compute_accuracy(vit, text_features, loader, device, hooks=None):
    correct, total = 0, 0
    ctx = vit.hooks(hooks) if hooks else _nullctx()
    with ctx:
        for px, labels in loader:
            px = px.to(device)
            img_feat = vit.model.get_image_features(pixel_values=px)
            img_feat = F.normalize(img_feat.float(), dim=-1)
            logit_scale = getattr(vit.model, "logit_scale", None)
            scale = logit_scale.exp() if logit_scale is not None else 1.0
            sims = scale * (img_feat @ text_features.t())
            preds = sims.argmax(dim=-1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / max(total, 1)


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def make_sae_hook(sae, arch, block_layer):
    def fn(act):
        orig_dtype = act.dtype
        a = act.float()
        was_seq = a.ndim == 3
        if was_seq:
            b, s, d = a.shape
            flat = a.reshape(-1, d)
        else:
            flat = a
        sae_out = sae(flat)[0]
        out = sae_out.reshape(a.shape) if was_seq else sae_out
        return out.to(orig_dtype)
    return [Hook(block_layer, "resid", fn, backbone=arch, return_module_output=True)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="+", default=["align", "siglip2"], choices=["align", "siglip2"])
    p.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--own_n_samples", type=int, default=1000)
    p.add_argument("--imagenet_n_samples", type=int, default=1500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_templates", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_path", default="out/eval_expanded_matrix/results.json")
    args = p.parse_args()

    device = args.device
    out_path = Path(_root) / args.save_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.load(open(out_path)) if out_path.exists() else {}

    imagenet_ds = None  # loaded lazily once, reused across archs

    for arch in args.archs:
        print(f"\n{'='*70}\nARCH: {arch}\n{'='*70}")
        base_dir = BASE_ROOT
        base_ckpts = {}
        for variant in args.variants:
            matches = list(base_dir.glob(f"{arch}_{variant}/*/final_sparse_autoencoder_*/*.pt"))
            if matches:
                base_ckpts[variant] = str(matches[0])
        if not base_ckpts:
            print(f"[WARN] no base_expanded checkpoints for {arch}, skipping")
            continue

        from src.models.utils import get_base_backbone
        from src.sae_training.backbone_registry import get_backbone_spec
        default_model_id = get_backbone_spec(arch).default_model_id
        model, processor = get_base_backbone(arch)
        model = model.to(device).eval()
        cfg_stub = ViTSAERunnerConfig(
            class_token=False, image_width=224, image_height=224,
            model_name=default_model_id, module_name="resid", block_layer=BLOCK_LAYER,
            dataset_path=DATASET_INFO["caltech101"]["path"],
            image_key="image", label_key="label",
            d_in=768, expansion_factor=64, batch_size=args.batch_size,
            device=device, seed=42, dtype=torch.float32,
        )
        vit = load_hooked_vit(cfg_stub, "base", default_model_id, device, arch=arch)
        # load_hooked_vit builds its own model/processor via get_base_backbone;
        # reuse that one (avoids loading the HF model twice).
        model, processor = vit.model, vit.processor

        if imagenet_ds is None:
            imagenet_ds = load_dataset(**DATASET_INFO["imagenet"])
        with open(Path(_root) / "configs/classnames/imagenet_classnames.txt") as f:
            imagenet_classnames = [l.strip().split(" ", 1)[1] for l in f if len(l.strip().split(" ", 1)) == 2]

        print(f"[arch={arch}] building ImageNet text features ({args.n_templates} templates)...")
        in_text_feat = build_text_features(model, processor, imagenet_classnames, device, args.n_templates)
        in_subset = HFImageDataset(imagenet_ds, processor, n_samples=args.imagenet_n_samples)
        in_loader = DataLoader(in_subset, batch_size=args.batch_size, shuffle=False,
                               num_workers=4, collate_fn=_collate)

        no_sae_in_acc = compute_accuracy(vit, in_text_feat, in_loader, device)
        print(f"[arch={arch}] ImageNet no-SAE: {no_sae_in_acc:.2f}%")

        base_sae_acc = {}
        for variant, ckpt in base_ckpts.items():
            sae, _ = load_sae(ckpt, device)
            sae.eval()
            hooks = make_sae_hook(sae, arch, BLOCK_LAYER)
            acc = compute_accuracy(vit, in_text_feat, in_loader, device, hooks=hooks)
            base_sae_acc[variant] = acc
            print(f"[arch={arch}] ImageNet + base {variant} SAE: {acc:.2f}%")
            del sae; gc.collect(); torch.cuda.empty_cache()

        arch_key = arch
        results.setdefault(arch_key, {"imagenet_no_sae": no_sae_in_acc,
                                       "imagenet_base_sae": base_sae_acc, "datasets": {}})
        json.dump(results, open(out_path, "w"), indent=2)

        for dataset in args.datasets:
            for variant in args.variants:
                cond_dir = EXPANDED_ROOT / f"{arch}_{variant}" / dataset
                matches = list(cond_dir.glob("*/final_sparse_autoencoder_*/*.pt"))
                if not matches:
                    continue
                ckpt = str(matches[0])
                key = f"{dataset}/{variant}"
                if key in results[arch_key]["datasets"] and "own_no_sae" in results[arch_key]["datasets"][key]:
                    print(f"[SKIP] {arch}/{key} already done")
                    continue

                print(f"\n--- {arch} / {dataset} / {variant} ---")
                try:
                    ds = load_dataset(**DATASET_INFO[dataset])
                except Exception as e:
                    print(f"[ERROR] loading dataset {dataset}: {e}")
                    continue
                classnames = get_classnames(dataset, ds)
                own_text_feat = build_text_features(model, processor, classnames, device, args.n_templates)
                try:
                    own_subset = HFImageDataset(ds, processor, n_samples=args.own_n_samples)
                except KeyError as e:
                    print(f"[ERROR] dataset {dataset} missing key {e}, skipping")
                    continue
                own_loader = DataLoader(own_subset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=4, collate_fn=_collate)

                try:
                    own_no_sae = compute_accuracy(vit, own_text_feat, own_loader, device)
                except Exception as e:
                    print(f"[ERROR] own-domain no-SAE eval failed for {dataset}: {e}")
                    continue

                sae, sae_cfg = load_sae(ckpt, device)
                sae.eval()
                hooks = make_sae_hook(sae, arch, BLOCK_LAYER)
                own_masked = compute_accuracy(vit, own_text_feat, own_loader, device, hooks=hooks)
                in_masked = compute_accuracy(vit, in_text_feat, in_loader, device, hooks=hooks)

                row = {
                    "checkpoint": ckpt,
                    "own_no_sae": own_no_sae,
                    "own_base_sae": None,  # base SAE trained on ImageNet, not meaningful on own-domain here
                    "own_masked_sae": own_masked,
                    "imagenet_no_sae": no_sae_in_acc,
                    "imagenet_base_sae": base_sae_acc.get(variant),
                    "imagenet_masked_sae": in_masked,
                }
                results[arch_key]["datasets"][key] = row
                json.dump(results, open(out_path, "w"), indent=2)
                print(f"[{arch}/{key}] own: {own_no_sae:.2f} -> {own_masked:.2f}   "
                      f"ImageNet: {no_sae_in_acc:.2f} -> {in_masked:.2f} "
                      f"(base-SAE ref {base_sae_acc.get(variant):.2f})")

                del sae; gc.collect(); torch.cuda.empty_cache()

        del model, vit
        gc.collect(); torch.cuda.empty_cache()

    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
