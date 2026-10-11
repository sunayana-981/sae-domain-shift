#!/usr/bin/env python3
"""
TEASER.png-style qualitative figure (Arxiv-version/figures/TEASER.png):
for ONE specific input image, find its single top-activating SAE feature
under three pipeline configs, then show that feature's own top-N
activating images from across the dataset -- revealing what concept each
pipeline actually associated with the input image.

Configs (matches the teaser's 3 rows):
  1. G-SAE            -- base (non-LoRA) ViT + generic SAE   [red/bad]
  2. LoRA + G-SAE      -- LoRA-adapted ViT + generic SAE      [red/bad]
  3. LoRA + Masked-FT SAE -- LoRA-adapted ViT + masked SAE    [green/good]

Usage:
    python tasks/teaser_style_comparison.py --dataset ucf101 --query_class Rafting \
        --sae_paths configs/controlled_sae_paths.json
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torch.utils.data import ConcatDataset

from tasks.controlled_common import (
    add_common_args, flush, load_common_registry_args, build_dataset_splits,
    get_vit, resolve_sae_path, get_sae,
)
from tasks.top_activating_grid import (
    batched_patch_acts, find_top_activating_images, select_class_selective_features,
)
from tasks.eval_steering import get_class_profile

CONFIGS = [
    dict(key="gsae_base", label="G-SAE\n(base ViT)", vit_type="base", sae_condition="gsae", good=False),
    dict(key="gsae_lora", label="LoRA + G-SAE", vit_type="lora", sae_condition="gsae", good=False),
    dict(key="masked_lora", label="LoRA + Masked-FT SAE", vit_type="lora", sae_condition="masked", good=True),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default="ucf101")
    p.add_argument("--query_class", type=str, default="Rafting",
                    help="Class-folder name to draw the query image from (ImageFolder datasets).")
    p.add_argument("--query_index", type=int, nargs="+", default=[0],
                    help="Which image(s) within --query_class to use (sorted filename order). "
                         "Pass several to scan for a good example in one run.")
    p.add_argument("--query_image", type=str, default=None,
                    help="Explicit path to a query image, overrides --query_class/--query_index.")
    p.add_argument("--top_n", type=int, default=4, help="Top-activating images to show per row.")
    p.add_argument("--n_candidates", type=int, default=2000,
                    help="Pool of most-active features to search for class selectivity "
                         "(mirrors top_activating_grid.py).")
    p.add_argument("--n_candidate_pool", type=int, default=300,
                    help="Of --n_candidates, how many lowest-entropy (class-selective) "
                         "features form the pool the query image's top feature is picked "
                         "from. Restricting to this pool avoids picking a runaway/dead "
                         "SAE unit that wins raw-argmax for almost any input.")
    p = add_common_args(p, out_dir_default="out/controlled/teaser_style")
    return p.parse_args()


def resolve_query_images(args, registry):
    """Returns a list of (path, display_name, tag) for every requested query image."""
    if args.query_image:
        return [(args.query_image, os.path.basename(args.query_image), "custom")]
    entry = registry[args.dataset]
    if entry["type"] != "imagefolder":
        raise SystemExit(f"[FATAL] --query_class only supported for imagefolder datasets, "
                          f"'{args.dataset}' is type={entry['type']}")
    class_dir = os.path.join(entry["path"], args.query_class)
    if not os.path.isdir(class_dir):
        raise SystemExit(f"[FATAL] class dir not found: {class_dir}")
    files = sorted(os.listdir(class_dir))
    files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    out = []
    for qi in args.query_index:
        if qi >= len(files):
            print(f"[SKIP] --query_index {qi} out of range ({len(files)} images in {class_dir})")
            continue
        out.append((os.path.join(class_dir, files[qi]), files[qi], str(qi)))
    return out


@torch.no_grad()
def top_feature_for_image(vit, sae, cfg, image, args, candidate_ids):
    """Picks the query image's best-matching feature restricted to a pool of
    already class-selective candidates, rather than a raw argmax over the
    full dictionary -- the raw argmax is dominated by a runaway/dead unit
    (confirmed: the same single feature wins for unrelated query images
    across completely different classes) regardless of image content."""
    acts = batched_patch_acts(vit, sae, cfg, [image], args)  # [1, P, d_sae]
    max_per_feat = acts.max(dim=1).values[0]                  # [d_sae]
    cand_tensor = torch.tensor(candidate_ids, device=max_per_feat.device)
    cand_vals = max_per_feat.index_select(0, cand_tensor)
    local_argmax = int(cand_vals.argmax().item())
    fid = candidate_ids[local_argmax]
    val = float(cand_vals[local_argmax].item())
    return fid, val


def save_teaser_figure(query_img, query_label, results, top_n, out_path):
    nrows = len(results)
    label_col_w = 1.6
    fig = plt.figure(figsize=(3.0 * top_n + label_col_w + 1.5, 3.0 * nrows + 1.8))
    gs = fig.add_gridspec(nrows, top_n + 1, left=0.02, right=0.99, top=0.86, bottom=0.02,
                           wspace=0.06, hspace=0.15,
                           width_ratios=[label_col_w] + [1] * top_n)

    fig.text(0.5, 0.95, f"Input Image: {query_label}", fontsize=17,
              fontweight="bold", ha="center")
    ax_query = fig.add_axes([0.50 - 0.065, 0.865, 0.13, 0.09])
    ax_query.imshow(query_img)
    ax_query.set_xticks([]); ax_query.set_yticks([])
    for spine in ax_query.spines.values():
        spine.set_edgecolor("black")

    for r, res in enumerate(results):
        color = "#4caf50" if res["config"]["good"] else "#e57373"
        items = res["items"]

        ax_label = fig.add_subplot(gs[r, 0])
        ax_label.axis("off")
        ax_label.text(0.98, 0.5, f"{res['config']['label']}\nf{res['feature_id']}  "
                       f"(query act={res['query_act']:.2f})",
                       fontsize=11, fontweight="bold", va="center", ha="right",
                       color=color, transform=ax_label.transAxes, wrap=True)

        for c in range(top_n):
            ax = fig.add_subplot(gs[r, c + 1])
            ax.set_xticks([]); ax.set_yticks([])
            if c < len(items):
                _, _, img, classname = items[c]
                ax.imshow(img)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(dataset_out_dir, exist_ok=True)

    setattr(args, "datasets", [args.dataset])
    registry, lora_checkpoints, sae_paths = load_common_registry_args(args)

    print("=" * 78)
    print(f"TEASER-STYLE COMPARISON: dataset={args.dataset}  query_class={args.query_class}")
    print("=" * 78)

    classnames, train_ds, eval_ds = build_dataset_splits(args.dataset, registry[args.dataset], args)
    search_pool = ConcatDataset([train_ds, eval_ds])
    print(f"[OK] dataset '{args.dataset}': {len(search_pool)} images in search pool, "
          f"{len(classnames)} classes")

    query_images = resolve_query_images(args, registry)
    print(f"[OK] {len(query_images)} query image(s): "
          + ", ".join(name for _, name, _ in query_images))

    num_classes = len(classnames)

    # Load every (vit, sae) config once, reused across all query images, and
    # precompute each config's class-selective candidate pool (used to keep
    # the per-query top-feature search away from runaway/dead units).
    loaded_configs = []
    for cfg_spec in CONFIGS:
        vit = get_vit(cfg_spec["vit_type"], args.dataset, lora_checkpoints, args)
        if vit is None:
            print(f"[SKIP] {cfg_spec['key']}: missing lora checkpoint for {cfg_spec['vit_type']}")
            continue
        sae_path = resolve_sae_path(cfg_spec["sae_condition"], args.dataset, sae_paths, args)
        if not sae_path or not os.path.exists(sae_path):
            print(f"[SKIP] {cfg_spec['key']}: missing sae checkpoint")
            continue
        sae, cfg = get_sae(sae_path, args.device)

        print(f"[{cfg_spec['key']}] computing class-selective candidate pool...")
        profile = get_class_profile(vit, sae, cfg, train_ds, num_classes, args.dataset,
                                     cfg_spec["vit_type"], cfg_spec["sae_condition"], args)
        candidates = select_class_selective_features(
            profile, classnames, args.n_candidates, args.n_candidate_pool)
        candidate_ids = [c["feature_id"] for c in candidates]
        print(f"[{cfg_spec['key']}] candidate pool: {len(candidate_ids)} class-selective features "
              f"(of {args.n_candidates} most-active)")

        loaded_configs.append((cfg_spec, vit, sae, cfg, candidate_ids))

    for query_path, query_name, query_tag in query_images:
        print(f"\n--- query image: {query_path} ---")
        query_img = Image.open(query_path).convert("RGB")

        results = []
        for cfg_spec, vit, sae, cfg, candidate_ids in loaded_configs:
            fid, val = top_feature_for_image(vit, sae, cfg, query_img, args, candidate_ids)
            print(f"[{cfg_spec['key']}] top feature: f{fid} (act={val:.3f})")

            top_items = find_top_activating_images(vit, sae, cfg, search_pool, [fid], classnames,
                                                     args, top_n=args.top_n)
            results.append(dict(config=cfg_spec, feature_id=fid, query_act=val,
                                 items=top_items[fid]))
            flush()

        if len(results) < len(CONFIGS):
            print(f"[WARN] only {len(results)}/{len(CONFIGS)} configs produced results")

        out_path = os.path.join(dataset_out_dir, f"teaser_{args.query_class}_{query_tag}.png")
        save_teaser_figure(query_img, f"{args.query_class} ({args.dataset})", results, args.top_n, out_path)
        print(f"Wrote {out_path}")

        summary_path = os.path.join(
            dataset_out_dir, f"teaser_{args.query_class}_{query_tag}_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": args.dataset,
                "query_image": query_path,
                "results": [dict(key=r["config"]["key"], feature_id=r["feature_id"],
                                  query_act=r["query_act"],
                                  top_classes=[it[3] for it in r["items"]])
                            for r in results],
            }, f, indent=2)
        print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
