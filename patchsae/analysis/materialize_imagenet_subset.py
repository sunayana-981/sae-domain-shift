"""Materialize a small ImageFolder subset of ImageNet-1k for LoRA few-shot
training + SAE training, matching the 16-shot convention used for every
other dataset in this repo (lora_weights/vitb16/<dataset>/16shots/seed1/).

CLIP_LoRA/datasets/imagenet.py expects:
  root/imagenet/images/train/<class_folder>/*.jpg  (>= shots+val_shots per class,
                                                      randomly few-shot-sampled at train time)
  root/imagenet/images/val/<class_folder>/*.jpg    (used in full as the test set)

Folder names must be WordNet IDs sorted alphabetically to match
CLIP_LoRA/datasets/imagenet.py's hardcoded `imagenet_classes` list order
(verified: that list is in standard 0-999 class-index order, same order as
configs/classnames/imagenet_classnames.txt).
"""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

_project_root = Path(__file__).resolve().parent.parent
_repo_root = _project_root.parent
IMAGENET_CLASSNAMES_PATH = _project_root / "configs" / "classnames" / "imagenet_classnames.txt"
OUT_ROOT = _repo_root / "data" / "imagenet"

TRAIN_PER_CLASS = 10000  # exceeds any real per-class count (~732-1300) -> takes everything
VAL_PER_CLASS = 3         # kept small -- this becomes the LoRA test-set size


def load_wnids() -> list[str]:
    wnids = []
    with IMAGENET_CLASSNAMES_PATH.open() as fh:
        for line in fh:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                wnids.append(parts[0])
    if len(wnids) != 1000:
        raise ValueError(f"expected 1000 WNIDs, got {len(wnids)}")
    return wnids


def materialize(split: str, per_class: int, wnids: list[str]) -> None:
    out_dir = OUT_ROOT / ("train" if split == "train" else "val")
    if out_dir.is_dir() and any(out_dir.iterdir()):
        print(f"[SKIP] {out_dir} already has content")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading ImageNet-1k {split} split (metadata only)...")
    ds = load_dataset("evanarlian/imagenet_1k_resized_256", split=split)

    # Select indices per class from the label column only (no image decode)
    # before touching any images -- the dataset is grouped by class, so a
    # naive row-by-row scan would decode all ~1.28M images to reach class
    # 999 even though only ~28k images are needed.
    labels = ds["label"]
    counts = [0] * 1000
    chosen_indices: list[int] = []
    chosen_labels: list[int] = []
    for idx, label in enumerate(labels):
        if counts[label] < per_class:
            chosen_indices.append(idx)
            chosen_labels.append(label)
            counts[label] += 1
    missing = [wnids[i] for i, c in enumerate(counts) if c < per_class]
    print(f"[INFO] selected {len(chosen_indices)} indices ({per_class}/class); "
          f"classes below target: {len(missing)}")

    subset = ds.select(chosen_indices)
    per_class_counter = [0] * 1000
    for row, label in tqdm(zip(subset, chosen_labels), total=len(chosen_indices), desc=f"materializing {split}"):
        wnid = wnids[label]
        class_dir = out_dir / wnid
        class_dir.mkdir(exist_ok=True)
        img = row["image"].convert("RGB")
        img.save(class_dir / f"{per_class_counter[label]:03d}.jpg", quality=90)
        per_class_counter[label] += 1

    print(f"[OK] wrote {len(chosen_indices)} images to {out_dir}")
    if missing:
        print(f"     (first few classes below target: {missing[:5]})")


def main() -> int:
    wnids = load_wnids()
    materialize("train", TRAIN_PER_CLASS, wnids)
    materialize("val", VAL_PER_CLASS, wnids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
