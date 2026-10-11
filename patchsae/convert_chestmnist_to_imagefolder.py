"""
Convert ChestMNIST (multi-label, 14 disease tags) into a single-label
imagefolder for the accuracy pipelines here, which all assume one
integer class per image.

Scheme: an image keeps only its "no_finding" label if all 14 tags are
negative, or its single positive tag if exactly one is positive. Images with
two or more positive tags are dropped as label-ambiguous for a single-label
setup -- this is a real simplification of the underlying multi-label task
(around a third of ChestMNIST images have >1 finding), not a claim that the
condition is not comorbid; it exists to slot ChestMNIST into pipelines that
were not built for multi-label evaluation, and should be described as such
wherever these numbers are reported.

Usage:
    python convert_chestmnist_to_imagefolder.py \
        --medmnist_root /path/to/sae-domain-shift/data \
        --output_dir /path/to/sae-domain-shift/data/chestmnist_imagefolder \
        --split train
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm

try:
    import medmnist
    from medmnist import INFO
except ImportError:
    print("[FATAL] medmnist not installed. Run: pip install medmnist")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert ChestMNIST to single-label imagefolder")
    parser.add_argument("--medmnist_root", type=str,
                         default="/path/to/sae-domain-shift/data")
    parser.add_argument("--output_dir", type=str,
                         default="/path/to/sae-domain-shift/data/chestmnist_imagefolder")
    parser.add_argument("--split", type=str, default="train",
                         choices=["train", "val", "test", "all"])
    args = parser.parse_args()

    data_flag = "chestmnist"
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])
    label_names = {int(k): v for k, v in info["label"].items()}
    label_names[len(label_names)] = "no_finding"
    no_finding_idx = len(label_names) - 1

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    output_dir = Path(args.output_dir)
    total_saved, total_dropped_ambiguous = 0, 0

    for split in splits:
        print(f"\n[INFO] Processing split: {split}")
        dataset = DataClass(split=split, download=True, root=args.medmnist_root)

        for idx in tqdm(range(len(dataset)), desc=f"  Saving {split}"):
            image, label_vec = dataset[idx]
            positives = [i for i, v in enumerate(label_vec) if int(v) == 1]

            if len(positives) == 0:
                label_int = no_finding_idx
            elif len(positives) == 1:
                label_int = positives[0]
            else:
                total_dropped_ambiguous += 1
                continue

            classname = label_names[label_int].replace("/", "_").replace(" ", "_")
            class_dir = output_dir / classname
            class_dir.mkdir(parents=True, exist_ok=True)
            image.save(class_dir / f"{split}_{idx:06d}.png")
            total_saved += 1

    print(f"\n[SUCCESS] Saved {total_saved} single-label images to {output_dir}")
    print(f"[INFO] Dropped {total_dropped_ambiguous} images with >1 positive finding (label-ambiguous for single-label use)")
    for d in sorted(output_dir.iterdir()):
        if d.is_dir():
            print(f"  {d.name}/  ({len(list(d.glob('*.png')))} images)")

    import json
    classnames_dir = Path("configs/classnames")
    classnames_dir.mkdir(parents=True, exist_ok=True)
    ordered = [label_names[i] for i in range(len(label_names))]
    json.dump(ordered, open(classnames_dir / "chestmnist_classnames.json", "w"), indent=2)
    print(f"[INFO] Classnames saved to configs/classnames/chestmnist_classnames.json")


if __name__ == "__main__":
    main()
