"""Concept-overlap distance: does the target domain's *class vocabulary*
already overlap with ImageNet-1k, independent of raw CLIP-embedding MMD?

Motivation: the 8-domain MMD refit (2026-07-25) found oxford_pets has the
highest CLIP-embedding MMD^2 in the cohort but one of the lowest G-SAE
degradations -- backwards from the MMD hypothesis. Working hypothesis: raw
image-embedding MMD captures *distributional* shift (photography style,
composition), not *conceptual* novelty relative to what ImageNet -- and
therefore the G-SAE's dictionary -- already covers. ImageNet-1k has ~120
dog-breed classes and many cat/animal categories, so despite pets' large
embedding-space MMD, the relevant concepts may already be well-represented.

This script tests that account directly: embed each target dataset's class
names and ImageNet-1k's 1000 class names with the same CLIP text encoder
used for the existing MMD (ViT-L/14), and for each target class find its
nearest ImageNet class by cosine similarity. The per-dataset "concept gap"
is 1 minus the mean of those per-class max similarities: low gap means the
target's vocabulary is already well-covered by ImageNet; high gap means it
is not.

This is diagnostic only -- a candidate alternative/additional predictor to
correlate against the existing degradation_pp outcomes, not a replacement
fit and not something touching the held-out domains.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

_project_root = Path(__file__).resolve().parent.parent
import sys
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from eval_medmnist_sae import get_imagefolder_classnames  # noqa: E402
from tasks.controlled_common import _mapped_classnames  # noqa: E402

IMAGENET_CLASSNAMES_PATH = _project_root / "configs" / "classnames" / "imagenet_classnames.txt"
SKETCH_CLASSNAMES_PATH = _project_root / "configs" / "classnames" / "imagenet-sketch_classnames.txt"

# dataset -> imagefolder root used for classnames, taken directly from
# configs/controlled_datasets.json (verified against the file, not guessed).
DATASET_ROOTS = {
    "caltech101": "/path/to/sae-domain-shift/data/caltech-101/Caltech101/101_ObjectCategories",
    "dtd": "/path/to/sae-domain-shift/data/dtd/images",
    "ucf101": "/path/to/sae-domain-shift/data/UCF101/UCF-101-midframes",
    "cub2002011": "/path/to/sae-domain-shift/data/cub2002011/train",
    "fgvc": "/path/to/sae-domain-shift/data/fgvc_imagefolder/train",
    "eurosat": "/path/to/sae-domain-shift/data/eurosat/2750",
    "oxford_pets": "/path/to/sae-domain-shift/data/oxford_pets_imagefolder/train",
    "pathmnist": "/path/to/sae-domain-shift/data/pathmnist_imagefolder",
    "imagenet_r": "/path/to/sae-domain-shift/data/imagenet-r",
    "imagenet_a": "/path/to/sae-domain-shift/data/imagenet-a",
    "imagenet_sketch": "/path/to/sae-domain-shift/data/sketch",
    "imagenet_v2": "/path/to/sae-domain-shift/data/imagenetv2/imagenetv2-matched-frequency-format-val",
}

# Datasets whose folder names are WordNet synset IDs or numeric ImageNet
# indices (not human-readable words) -- same mapping used in
# tasks/controlled_common.py's build_dataset_splits() for eval_matrix.py, so
# classnames here are guaranteed consistent with what the model was actually
# evaluated against.
DATASET_CLASSNAME_MAPS = {
    "imagenet_r": (IMAGENET_CLASSNAMES_PATH, "wnid"),
    "imagenet_a": (IMAGENET_CLASSNAMES_PATH, "wnid"),
    "imagenet_sketch": (SKETCH_CLASSNAMES_PATH, "wnid"),
    "imagenet_v2": (IMAGENET_CLASSNAMES_PATH, "numeric_index"),
}


def classnames_for(dataset: str, root: str) -> list[str]:
    if dataset in DATASET_CLASSNAME_MAPS:
        map_path, mode = DATASET_CLASSNAME_MAPS[dataset]
        return _mapped_classnames(root, str(map_path), mode)
    return get_imagefolder_classnames(root)


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


@torch.no_grad()
def embed_texts(texts: list[str], model, tokenizer, device: str, batch_size: int = 256) -> np.ndarray:
    embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        prompts = [f"a photo of a {t}" for t in batch]
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
        out = model(**inputs).text_embeds
        out = out / out.norm(dim=-1, keepdim=True)
        embeds.append(out.cpu().numpy())
    return np.concatenate(embeds, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out",
        default=str(_project_root / "out/controlled/mmd_predictor_inputs/concept_overlap_distance.csv"),
    )
    args = parser.parse_args()

    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    print(f"[INFO] Loading {args.backbone} text encoder on {args.device}...")
    tokenizer = CLIPTokenizer.from_pretrained(args.backbone)
    model = CLIPTextModelWithProjection.from_pretrained(args.backbone).to(args.device).eval()

    imagenet_names = load_imagenet_classnames()
    print(f"[INFO] Embedding {len(imagenet_names)} ImageNet-1k class names...")
    imagenet_embeds = embed_texts(imagenet_names, model, tokenizer, args.device)

    rows = []
    for dataset, root in DATASET_ROOTS.items():
        if not Path(root).is_dir():
            print(f"[SKIP] {dataset}: root not found: {root}")
            continue
        classnames = classnames_for(dataset, root)
        target_embeds = embed_texts(classnames, model, tokenizer, args.device)
        sims = target_embeds @ imagenet_embeds.T  # (n_target_classes, 1000)
        max_sim_per_class = sims.max(axis=1)
        nearest_idx = sims.argmax(axis=1)
        concept_gap = float(1.0 - max_sim_per_class.mean())
        rows.append(
            {
                "dataset": dataset,
                "n_target_classes": len(classnames),
                "mean_max_cosine_sim_to_imagenet": float(max_sim_per_class.mean()),
                "concept_gap": concept_gap,
                "example_nearest_pairs": [
                    f"{classnames[i]} -> {imagenet_names[nearest_idx[i]]}"
                    for i in np.argsort(max_sim_per_class)[:3]
                ],
            }
        )
        print(f"[OK] {dataset}: n_classes={len(classnames)} concept_gap={concept_gap:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        import csv

        writer = csv.writer(fh)
        writer.writerow(["dataset", "n_target_classes", "mean_max_cosine_sim_to_imagenet", "concept_gap"])
        for row in rows:
            writer.writerow(
                [row["dataset"], row["n_target_classes"], row["mean_max_cosine_sim_to_imagenet"], row["concept_gap"]]
            )
    examples_path = out_path.with_suffix(".examples.json")
    with examples_path.open("w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"Wrote {out_path}")
    print(f"Wrote {examples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
