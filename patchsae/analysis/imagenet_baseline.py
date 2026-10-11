"""ImageNet baseline: what does G-SAE do on its own native domain?

The question this answers: "What is the performance of the reported r
metrics for ZS+G-SAE on ImageNet?" This is the distance=0 anchor point for
the whole MMD/degradation analysis (the experiment notes) -- every other row in
that table is a domain-shifted target; this one is the domain G-SAE was
actually trained on, using the BASE (non-adapted) CLIP model, so there is no
LoRA adaptation confound at all.

Reuses tasks/eval_matrix.py's own compute_zeroshot_acc (same accuracy
computation as every other row in controlled.md, for apples-to-apples
comparison) against a stratified subsample of the ImageNet-1k validation
set (evanarlian/imagenet_1k_resized_256 on HF -- a non-gated, resized
mirror; label indices match the standard 1000-class ordering used by
configs/classnames/imagenet_classnames.txt, verified against the HF
dataset's ClassLabel names before running).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from tasks.eval_matrix import compute_zeroshot_acc  # noqa: E402
from tasks.eval_causal_steering import make_stratified_subset  # noqa: E402
from tasks.controlled_common import get_sae, get_vit  # noqa: E402

PER_CLASS_CAP = 2  # ~2000 images across 1000 classes; matches the scale
                    # (n_images in the few-hundred-to-low-thousands range)
                    # used elsewhere in the experiment notes's table.
SEED = 42
BACKBONE = "openai/clip-vit-base-patch16"
IMAGENET_CLASSNAMES_PATH = _project_root / "configs" / "classnames" / "imagenet_classnames.txt"


class HFImageNetVal(Dataset):
    """Thin (PIL, int) adapter around the HF imagenet val split, with a
    .targets array for tasks.eval_causal_steering.extract_targets /
    make_stratified_subset."""

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
    args = SimpleNamespace(device=device, seed=SEED, batch_size=32, backbone=BACKBONE)

    print("[INFO] Loading ImageNet-1k val split (evanarlian/imagenet_1k_resized_256)...")
    from datasets import load_dataset

    hf_ds = load_dataset("evanarlian/imagenet_1k_resized_256", split="val")
    # Verify label ordering matches configs/classnames/imagenet_classnames.txt
    # before trusting any accuracy number against it.
    classnames = load_imagenet_classnames()
    hf_first_label_name = hf_ds.features["label"].names[0]
    assert hf_first_label_name.split(",")[0].strip().lower() == classnames[0].lower(), (
        f"label ordering mismatch: HF class 0 = {hf_first_label_name!r}, "
        f"imagenet_classnames.txt class 0 = {classnames[0]!r}"
    )
    print(f"[OK] label ordering verified against imagenet_classnames.txt (class 0 = {classnames[0]!r})")

    dataset = HFImageNetVal(hf_ds)
    subset, _indices = make_stratified_subset(dataset, num_classes=1000, per_class=PER_CLASS_CAP, seed=SEED)
    print(f"[INFO] Stratified eval subset: {len(subset)} images ({PER_CLASS_CAP}/class cap, 1000 classes)")

    vit = get_vit("base", "imagenet", {}, args)
    if vit is None:
        raise SystemExit("[FATAL] failed to load base CLIP model")

    rows = []
    for condition, sae_path in [("none", None), ("gsae", str(_project_root / "data/sae_weight/base/out.pt"))]:
        sae, cfg = (None, None)
        if sae_path is not None:
            sae, cfg = get_sae(sae_path, device)
        acc, n = compute_zeroshot_acc(vit, sae, cfg, classnames, subset, args)
        print(f"[OK] base ViT-B/16 + {condition}: acc={acc:.2f}% (n={n})")
        rows.append({"dataset": "imagenet", "vit_type": "base", "sae_condition": condition, "zeroshot_acc": acc, "n_images": n})

    degradation = rows[0]["zeroshot_acc"] - rows[1]["zeroshot_acc"]
    print(f"\n[RESULT] ImageNet (distance=0) degradation: {degradation:+.2f}pp (no-SAE {rows[0]['zeroshot_acc']:.2f}% -> G-SAE {rows[1]['zeroshot_acc']:.2f}%)")

    out_dir = _project_root / "out/controlled/imagenet_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dataset", "vit_type", "sae_condition", "zeroshot_acc", "n_images"])
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "results.json").open("w") as fh:
        json.dump({"rows": rows, "degradation_pp": degradation, "per_class_cap": PER_CLASS_CAP, "seed": SEED}, fh, indent=2)
    print(f"Wrote {out_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
