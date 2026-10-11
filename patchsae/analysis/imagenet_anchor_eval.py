"""Complete the ImageNet anchor row: no-SAE / G-SAE / FT-SAE accuracy on the
LoRA-adapted-on-ImageNet model, matching every other domain's table
structure in the experiment notes (which previously only had the BASE-model
G-SAE comparison, not the LoRA-FT one every other domain has).

Reuses eval_matrix.py's compute_zeroshot_acc, but with an explicit
classnames list instead of eval_matrix.py's automatic folder-name-derived
classnames: data/imagenet/{train,val} use WordNet-ID folder names (commas
and slashes in real ImageNet class names make them unsafe as folder names),
and WNIDs happen to sort alphabetically in the same order as
configs/classnames/imagenet_classnames.txt (verified), so the substitution
is a straight list swap, not a remapping.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from tasks.eval_matrix import compute_zeroshot_acc  # noqa: E402
from tasks.controlled_common import build_dataset_splits, get_sae, get_vit, load_registry  # noqa: E402

IMAGENET_CLASSNAMES_PATH = _project_root / "configs" / "classnames" / "imagenet_classnames.txt"


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
        backbone="openai/clip-vit-base-patch16", device=device, seed=42,
        max_images=100000, val_split=0.15, batch_size=8,
    )

    registry = load_registry(str(_project_root / "configs/controlled_datasets.json"))
    entry = registry["imagenet_anchor"]
    folder_classnames, _train_ds, eval_ds = build_dataset_splits("imagenet_anchor", entry, args)

    # Verify WNID-sorted folder order matches imagenet_classnames.txt order
    # before trusting the substitution.
    wnids = []
    with IMAGENET_CLASSNAMES_PATH.open() as fh:
        for line in fh:
            wnids.append(line.strip().split(" ", 1)[0])
    assert folder_classnames == wnids, "WNID folder order does not match imagenet_classnames.txt order"
    classnames = load_imagenet_classnames()
    print(f"[OK] verified WNID folder order matches imagenet_classnames.txt (class 0 = {classnames[0]!r})")
    print(f"[INFO] eval set: {len(eval_ds)} images")

    lora_checkpoints = {"imagenet_anchor": entry["lora_checkpoint"]}
    vit = get_vit("lora", "imagenet_anchor", lora_checkpoints, args)
    if vit is None:
        raise SystemExit("[FATAL] failed to load LoRA-adapted model")

    sae_paths = load_registry(str(_project_root / "configs/controlled_sae_paths.json"))
    conditions = {
        "none": None,
        "gsae": str(_project_root / "data/sae_weight/base/out.pt"),
        "ftsae": sae_paths["imagenet_anchor"]["ftsae"],
    }

    rows = []
    for condition, sae_path in conditions.items():
        sae, cfg = (None, None)
        if sae_path is not None:
            sae, cfg = get_sae(sae_path, device)
        acc, n = compute_zeroshot_acc(vit, sae, cfg, classnames, eval_ds, args)
        print(f"[OK] LoRA-ImageNet + {condition}: acc={acc:.2f}% (n={n})")
        rows.append({"dataset": "imagenet_anchor", "vit_type": "lora", "sae_condition": condition, "zeroshot_acc": acc, "n_images": n})

    out_dir = _project_root / "out/controlled/imagenet_anchor_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dataset", "vit_type", "sae_condition", "zeroshot_acc", "n_images"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
