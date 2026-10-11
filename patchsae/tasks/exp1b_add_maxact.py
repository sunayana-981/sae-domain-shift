#!/usr/bin/env python3
"""
exp1b_add_maxact.py — add an ImageNet peak-activation ranking to the exp1 cache.

exp4_prune_sweep.py found that ranking units by peak activation preserves
accuracy better than mean activation, but it computed that ranking in-process
and never persisted it. This computes it once and appends `imagenet__max_act` to out/exp1_activity_overlap/activity.npz.
"""

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
from tasks.utils import DATASET_INFO, load_hooked_vit, load_sae

NPZ = Path(_root) / "out/exp1_activity_overlap/activity.npz"


@torch.no_grad()
def main():
    device, n_batches, bs, layer = "cuda", 200, 16, -2
    model_name = "openai/clip-vit-base-patch16"

    sae, _ = load_sae(str(Path(_root) / "data/sae_weight/base/out.pt"), device)
    cfg = ViTSAERunnerConfig(
        class_token=False, image_width=224, image_height=224,
        model_name=model_name, module_name="resid", block_layer=layer,
        dataset_path=DATASET_INFO["imagenet"]["path"],
        image_key="image", label_key="label",
        d_in=sae.d_in, expansion_factor=sae.d_sae // sae.d_in,
        batch_size=bs, device=device, seed=42, dtype=torch.float32,
    )
    ds = load_dataset(**DATASET_INFO["imagenet"])
    vit = load_hooked_vit(cfg, "base", model_name, device)
    vit.eval()
    store = ViTActivationsStore(ds, bs, device, 42, vit, layer, "resid", False)

    peak = torch.zeros(sae.d_sae, device=device)
    for i in range(n_batches):
        _, acts, _ = sae.forward_standard(store.get_batch_activations())
        peak = torch.maximum(peak, acts.reshape(-1, sae.d_sae).max(0).values)
    peak = peak.cpu().numpy()

    data = {k: v for k, v in np.load(NPZ).items()}
    data["imagenet__max_act"] = peak
    np.savez(NPZ, **data)
    print(f"[exp1b] added imagenet__max_act  mean={peak.mean():.4f} "
          f"max={peak.max():.4f} zero={(peak==0).mean():.4f}")
    print(f"[exp1b] saved {NPZ}")


if __name__ == "__main__":
    main()
