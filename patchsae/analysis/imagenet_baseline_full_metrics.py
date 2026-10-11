"""Full SAE-quality metric suite (not just accuracy) for base CLIP + G-SAE
on ImageNet -- the "distance=0" anchor for every metric reported elsewhere
in the experiment notes, computed while the cluster ImageNet LoRA/FT-SAE jobs
run in parallel.

Reuses eval_matrix.py's own compute_zeroshot_acc / compute_sae_quality /
compute_label_entropy for the same metric definitions as every other row
in this file's tables -- this script only supplies the ImageNet data
loading (same HF mirror + WNID label-order verification as
imagenet_baseline.py / imagenet_anchor_eval.py) and wires the three
functions together for the base (non-adapted) model.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from tasks.eval_matrix import compute_label_entropy, compute_sae_quality, compute_zeroshot_acc  # noqa: E402
from tasks.eval_causal_steering import make_stratified_subset  # noqa: E402
from tasks.controlled_common import get_sae, get_vit  # noqa: E402

EVAL_PER_CLASS = 2   # matches imagenet_baseline.py's existing 2000-image eval scale
TRAIN_PER_CLASS = 2  # label_entropy needs a train-split class-activation profile
SEED = 42
BACKBONE = "openai/clip-vit-base-patch16"
IMAGENET_CLASSNAMES_PATH = _project_root / "configs" / "classnames" / "imagenet_classnames.txt"


class HFImageNetSplit(Dataset):
    def __init__(self, hf_dataset):
        self._ds = hf_dataset
        self.targets = torch.tensor(hf_dataset["label"]).numpy()

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        row = self._ds[idx]
        return row["image"].convert("RGB"), int(row["label"])


def load_imagenet_classnames() -> list[str]:
    names = []
    with IMAGENET_CLASSNAMES_PATH.open() as fh:
        for line in fh:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                names.append(parts[1])
    if len(names) != 1000:
        raise ValueError(f"expected 1000 ImageNet classnames, got {len(names)}")
    return names


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = SimpleNamespace(
        device=device, seed=SEED, batch_size=32, backbone=BACKBONE,
        l0_threshold=1e-5, label_entropy_topn=1000,
    )

    from datasets import load_dataset

    classnames = load_imagenet_classnames()
    print(f"[INFO] Loading ImageNet-1k val + train splits...")
    val_hf = load_dataset("evanarlian/imagenet_1k_resized_256", split="val")
    assert val_hf.features["label"].names[0].split(",")[0].strip().lower() == classnames[0].lower()
    eval_full = HFImageNetSplit(val_hf)
    eval_subset, _ = make_stratified_subset(eval_full, 1000, EVAL_PER_CLASS, SEED)
    print(f"[INFO] eval subset: {len(eval_subset)} images")

    train_hf = load_dataset("evanarlian/imagenet_1k_resized_256", split="train")
    train_full = HFImageNetSplit(train_hf)
    train_subset, _ = make_stratified_subset(train_full, 1000, TRAIN_PER_CLASS, SEED)
    print(f"[INFO] train subset (for label entropy): {len(train_subset)} images")

    vit = get_vit("base", "imagenet", {}, args)
    if vit is None:
        raise SystemExit("[FATAL] failed to load base CLIP model")

    rows = []

    # none: zeroshot only, no SAE
    acc, n = compute_zeroshot_acc(vit, None, None, classnames, eval_subset, args)
    print(f"[OK] base + none: acc={acc:.2f}% (n={n})")
    rows.append(dict(dataset="imagenet", vit_type="base", sae_condition="none",
                      zeroshot_acc=acc, l0=None, dead_frac=None, recon_cosine=None,
                      fve=None, label_entropy_mean=None, n_images=n))

    # gsae: zeroshot + full SAE-quality suite + label entropy
    sae, cfg = get_sae(str(_project_root / "data/sae_weight/base/out.pt"), device)
    acc, n = compute_zeroshot_acc(vit, sae, cfg, classnames, eval_subset, args)
    quality = compute_sae_quality(vit, sae, cfg, eval_subset, args)
    label_entropy = compute_label_entropy(vit, sae, cfg, train_subset, classnames, args)
    print(f"[OK] base + gsae: acc={acc:.2f}% l0={quality['l0']:.2f} "
          f"dead_frac={quality['dead_frac']:.4f} recon_cosine={quality['recon_cosine']:.4f} "
          f"fve={quality['fve']:.4f} label_entropy={label_entropy:.4f}")
    rows.append(dict(dataset="imagenet", vit_type="base", sae_condition="gsae",
                      zeroshot_acc=acc, l0=quality["l0"], dead_frac=quality["dead_frac"],
                      recon_cosine=quality["recon_cosine"], fve=quality["fve"],
                      label_entropy_mean=label_entropy, n_images=n))

    out_dir = _project_root / "out/controlled/imagenet_baseline_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "vit_type", "sae_condition", "zeroshot_acc", "l0",
                  "dead_frac", "recon_cosine", "fve", "label_entropy_mean", "n_images"]
    with (out_dir / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
