"""H5 (causal steerability), Part A: classifier-level necessity/sufficiency,
"cheap tier" -- reuses existing FT-SAE checkpoints, no new training.

The question, distilled: the paper shows G-SAE representations
degrade under shift (accuracy, reconstruction, monosemanticity) but never
tests whether that degradation matters for what SAEs are actually used for
-- explaining and steering. This is the classifier-level tier of that ask
(Part B, mirroring Kulkarni et al.'s LLaVA protocol, is a separate,
multi-day undertaking tracked elsewhere).

Reuses the exact error-preserving edit formula already validated in
tasks/eval_causal_steering.py (x' = x + D(z' - z), so an unmodified latent
is an exact no-op and reconstruction error is preserved) and its
run_prediction_pass/Hook machinery, generalized from single-latent to
multi-latent edits.

Two causal tests per (dataset, SAE condition, class):

  Necessity (ablation): zero the top-k class-conditional latents (ranked by
  class-mean activation on the *selection* split only), measure the
  resulting drop in that class's accuracy on a disjoint *evaluation* split,
  against a random-k-alive-latents control (same k, non-top latents).
  Selectivity = targeted_drop - control_drop.

  Sufficiency (injection): take evaluation images from *other* classes,
  multiply the target class's top-k latents by alpha in {2, 5, 10} (relative
  to each latent's own natural activation, following eval_causal_steering's
  amplify_mode="multiply"), measure the flip rate to the target class.
  Report flip-rate AUC over the alpha sweep (trapezoidal).

Scope (explicitly a "cheap tier v1", not the full grid): 4 datasets with
existing FT-SAE checkpoints spanning the MMD axis (caltech101=near,
dtd/eurosat=mid, pathmnist=far), first 6 classes per dataset (deterministic,
not cherry-picked), k in {5, 25} for ablation, k=10 fixed for injection.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import wilcoxon

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from mdmnist_sae_eval import calculate_text_features  # noqa: E402
from src.sae_training.hooked_vit import Hook  # noqa: E402
from tasks.eval_causal_steering import (  # noqa: E402
    extract_targets,
    make_stratified_subset,
    natural_sae_forward,
    run_prediction_pass,
)
from tasks.controlled_common import build_dataset_splits, get_sae, get_vit, load_registry  # noqa: E402

SEED = 42
SELECTION_PER_CLASS = 64
EVAL_PER_CLASS = 32
N_CLASSES_PER_DATASET = 6
ABLATE_KS = (5, 25)
INJECT_K = 10
INJECT_ALPHAS = (2.0, 5.0, 10.0)
CONTROL_POOL_OFFSET = 200  # skip this many top latents before sampling the "non-top" control pool

# (dataset key in configs/controlled_datasets.json, MMD2_clip from distance_results.csv,
#  ftsae checkpoint path override -- None means read from controlled_sae_paths.json)
DATASETS = ["caltech101", "dtd", "eurosat", "pathmnist"]
MMD2_BY_DATASET = {"caltech101": 0.028229, "dtd": 0.089749, "eurosat": 0.291092, "pathmnist": 0.422935}

# Prefer the single-seed causal_steering checkpoints where they exist (already
# the controlled protocol's designated ftsae for eurosat/pathmnist); fall back to the
# init-ablation seed42 checkpoints for caltech101/dtd, which used the same
# checkpoint-init/protect_frac=0 protocol.
FTSAE_OVERRIDE = {
    "caltech101": str(_project_root / "out/checkpoints/init_ablation/caltech101/ftsae/seed42/q9k3t6lq/final_sparse_autoencoder_openai/clip-vit-base-patch16_-2_resid_49152.pt"),
    "dtd": str(_project_root / "out/checkpoints/init_ablation/dtd/ftsae/seed42/25r0zz3x/final_sparse_autoencoder_openai/clip-vit-base-patch16_-2_resid_49152.pt"),
}


def error_preserving_multi_edit(sae, activations: torch.Tensor, feature_ids: np.ndarray, new_values: torch.Tensor) -> torch.Tensor:
    """Multi-latent generalization of eval_causal_steering.error_preserving_latent_edit:
    x' = x + D(z'_S - z_S) over a latent set S, exact no-op outside S."""
    reconstruction, latents = natural_sae_forward(sae, activations)
    feature_ids_t = torch.as_tensor(feature_ids, dtype=torch.long, device=latents.device)
    old_values = latents[..., feature_ids_t]
    delta_z = (new_values - old_values)  # (B, k)
    decoder_dirs = sae.W_dec[feature_ids_t].to(dtype=reconstruction.dtype, device=reconstruction.device)  # (k, d_in)
    edited_reconstruction = reconstruction + delta_z.to(reconstruction.dtype) @ decoder_dirs
    reconstruction_error = activations.to(reconstruction.dtype) - reconstruction
    return edited_reconstruction + reconstruction_error


def make_multi_feature_hook(sae, cfg, feature_ids: np.ndarray, mode: str, value: float = 0.0) -> Hook:
    """mode='ablate': zero the given latents. mode='multiply': scale them by `value`."""

    def hook_fn(activations: torch.Tensor):
        output = activations.clone()
        cls = output[:, 0, :]
        _, latents = natural_sae_forward(sae, cls)
        feature_ids_t = torch.as_tensor(feature_ids, dtype=torch.long, device=latents.device)
        if mode == "ablate":
            new_values = torch.zeros_like(latents[..., feature_ids_t])
        elif mode == "multiply":
            new_values = latents[..., feature_ids_t] * value
        else:
            raise ValueError(mode)
        edited = error_preserving_multi_edit(sae, cls, feature_ids, new_values)
        output[:, 0, :] = edited.to(output.dtype)
        return (output,)

    return Hook(cfg.block_layer, cfg.module_name, hook_fn, return_module_output=False, is_custom=False)


@torch.no_grad()
def compute_class_conditional_mass(vit, sae, cfg, selection_ds, num_classes: int, args) -> np.ndarray:
    """Per-class mean natural-latent activation on the selection split only.
    Returns (num_classes, d_sae)."""
    from torch.utils.data import DataLoader
    from tasks.eval_causal_steering import _hf_collate_fn

    d_sae = sae.d_sae
    sums = np.zeros((num_classes, d_sae), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)
    loader = DataLoader(selection_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_hf_collate_fn, num_workers=2)
    from tasks.eval_causal_steering import capture_cls_activations

    for images, labels in loader:
        cls = capture_cls_activations(vit, cfg, images, args.device)
        _, latents = natural_sae_forward(sae, cls)
        z = latents.cpu().numpy()
        for i, label in enumerate(labels.tolist()):
            sums[label] += z[i]
            counts[label] += 1
    counts = np.clip(counts, 1, None)
    return sums / counts[:, None]


def accuracy_on(predictions: dict, class_index: int | None = None) -> float:
    preds, labels = predictions["predictions"], predictions["labels"]
    if class_index is not None:
        mask = labels == class_index
        if mask.sum() == 0:
            return float("nan")
        return float((preds[mask] == labels[mask]).mean())
    return float((preds == labels).mean())


def flip_rate_to(predictions: dict, target_class: int) -> float:
    preds = predictions["predictions"]
    return float((preds == target_class).mean())


def trapezoid_auc(xs: list[float], ys: list[float]) -> float:
    return float(np.trapz(ys, xs) / (xs[-1] - xs[0]))


def run_condition(dataset_key: str, sae_condition: str, sae_path: str, device: str) -> list[dict]:
    print(f"\n{'=' * 70}\n{dataset_key} / {sae_condition}\n{'=' * 70}")
    args = SimpleNamespace(backbone="openai/clip-vit-base-patch16", device=device, seed=SEED,
                            max_images=100000, val_split=0.15, batch_size=16, num_workers=2,
                            debug_max_batches=None)
    registry = load_registry(str(_project_root / "configs/controlled_datasets.json"))
    entry = registry[dataset_key]
    classnames, train_ds, eval_ds = build_dataset_splits(dataset_key, entry, args)
    num_classes = len(classnames)
    n_target_classes = min(N_CLASSES_PER_DATASET, num_classes)

    lora_checkpoints = {dataset_key: entry.get("lora_checkpoint")}
    vit = get_vit("lora", dataset_key, lora_checkpoints, args)
    if vit is None:
        raise SystemExit(f"[FATAL] missing LoRA checkpoint for {dataset_key}")
    sae, cfg = get_sae(sae_path, device)

    selection_subset, _ = make_stratified_subset(train_ds, num_classes, SELECTION_PER_CLASS, SEED)
    eval_targets = extract_targets(eval_ds)
    eval_subset, eval_indices = make_stratified_subset(eval_ds, num_classes, EVAL_PER_CLASS, SEED)
    eval_labels_subset = eval_targets[eval_indices]

    class_mass = compute_class_conditional_mass(vit, sae, cfg, selection_subset, num_classes, args)
    text_features = calculate_text_features(vit, device, classnames)

    baseline_preds = run_prediction_pass(vit, eval_subset, text_features, args, hook=None)

    from torch.utils.data import Subset

    rows = []
    rng = np.random.default_rng(SEED)
    for target_class in range(n_target_classes):
        ranked = np.argsort(-class_mass[target_class])
        top_latents_max = ranked[: max(ABLATE_KS)]
        control_pool = ranked[CONTROL_POOL_OFFSET:]

        target_mask = eval_labels_subset == target_class
        other_mask = ~target_mask
        if target_mask.sum() == 0 or other_mask.sum() == 0:
            continue

        baseline_target_acc = accuracy_on(
            {"predictions": baseline_preds["predictions"][target_mask], "labels": baseline_preds["labels"][target_mask]}
        )
        baseline_other_acc = accuracy_on(
            {"predictions": baseline_preds["predictions"][other_mask], "labels": baseline_preds["labels"][other_mask]}
        )

        # --- Necessity: ablation ---
        for k in ABLATE_KS:
            top_k = top_latents_max[:k]
            control_k = rng.choice(control_pool, size=k, replace=False)

            hook_top = make_multi_feature_hook(sae, cfg, top_k, mode="ablate")
            preds_top = run_prediction_pass(vit, eval_subset, text_features, args, hook=hook_top)
            hook_ctrl = make_multi_feature_hook(sae, cfg, control_k, mode="ablate")
            preds_ctrl = run_prediction_pass(vit, eval_subset, text_features, args, hook=hook_ctrl)

            top_target_acc = accuracy_on({"predictions": preds_top["predictions"][target_mask], "labels": preds_top["labels"][target_mask]})
            top_other_acc = accuracy_on({"predictions": preds_top["predictions"][other_mask], "labels": preds_top["labels"][other_mask]})
            ctrl_target_acc = accuracy_on({"predictions": preds_ctrl["predictions"][target_mask], "labels": preds_ctrl["labels"][target_mask]})

            targeted_drop = baseline_target_acc - top_target_acc
            control_drop = baseline_target_acc - ctrl_target_acc
            collateral_drop = baseline_other_acc - top_other_acc
            rows.append(dict(
                dataset=dataset_key, sae_condition=sae_condition, test="necessity",
                target_class=target_class, classname=classnames[target_class], k=k,
                targeted_drop=targeted_drop, control_drop=control_drop,
                selectivity=targeted_drop - control_drop, collateral_drop=collateral_drop,
            ))

        # --- Sufficiency: injection ---
        top_k_inject = top_latents_max[:INJECT_K]
        other_subset = Subset(eval_subset, np.flatnonzero(other_mask).tolist())
        flip_rates = []
        for alpha in INJECT_ALPHAS:
            hook_inj = make_multi_feature_hook(sae, cfg, top_k_inject, mode="multiply", value=alpha)
            preds_inj = run_prediction_pass(vit, other_subset, text_features, args, hook=hook_inj)
            fr = flip_rate_to(preds_inj, target_class)
            flip_rates.append(fr)
        auc = trapezoid_auc(list(INJECT_ALPHAS), flip_rates)
        rows.append(dict(
            dataset=dataset_key, sae_condition=sae_condition, test="sufficiency",
            target_class=target_class, classname=classnames[target_class], k=INJECT_K,
            flip_rate_alpha2=flip_rates[0], flip_rate_alpha5=flip_rates[1], flip_rate_alpha10=flip_rates[2],
            flip_rate_auc=auc,
        ))
        print(f"  class {target_class} ({classnames[target_class]}): "
              f"necessity k=5 sel={rows[-3]['selectivity']:.3f} k=25 sel={rows[-2]['selectivity']:.3f} "
              f"sufficiency AUC={auc:.3f}")

    return rows


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae_paths_registry = load_registry(str(_project_root / "configs/controlled_sae_paths.json"))
    gsae_path = str(_project_root / "data/sae_weight/base/out.pt")

    all_rows: list[dict] = []
    for dataset_key in DATASETS:
        ftsae_path = FTSAE_OVERRIDE.get(dataset_key) or sae_paths_registry[dataset_key]["ftsae"]
        for condition, sae_path in [("gsae", gsae_path), ("ftsae", ftsae_path)]:
            all_rows.extend(run_condition(dataset_key, condition, sae_path, device))

    out_dir = _project_root / "out/controlled/h5_classifier_steering"
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in all_rows for k in row.keys()})
    with (out_dir / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    with (out_dir / "results.json").open("w") as fh:
        json.dump(all_rows, fh, indent=2, default=float)
    print(f"\nWrote {out_dir / 'results.csv'}")

    # --- Aggregate + Wilcoxon (paired by dataset x class x k/test) ---
    import pandas as pd

    df = pd.DataFrame(all_rows)
    summary_rows = []
    for dataset_key in DATASETS:
        sub = df[df.dataset == dataset_key]
        necessity = sub[sub.test == "necessity"]
        sufficiency = sub[sub.test == "sufficiency"]
        for k in ABLATE_KS:
            nk = necessity[necessity.k == k]
            g = nk[nk.sae_condition == "gsae"].sort_values("target_class")["selectivity"].values
            f = nk[nk.sae_condition == "ftsae"].sort_values("target_class")["selectivity"].values
            n = min(len(g), len(f))
            if n >= 2:
                try:
                    stat, p = wilcoxon(f[:n], g[:n])
                except ValueError:
                    stat, p = float("nan"), float("nan")
            else:
                stat, p = float("nan"), float("nan")
            summary_rows.append(dict(
                dataset=dataset_key, mmd2=MMD2_BY_DATASET[dataset_key], test="necessity", k=k,
                gsae_mean_selectivity=float(np.nanmean(g)) if len(g) else float("nan"),
                ftsae_mean_selectivity=float(np.nanmean(f)) if len(f) else float("nan"),
                gap=float(np.nanmean(f) - np.nanmean(g)) if len(f) and len(g) else float("nan"),
                wilcoxon_p=float(p),
            ))
        g = sufficiency[sufficiency.sae_condition == "gsae"].sort_values("target_class")["flip_rate_auc"].values
        f = sufficiency[sufficiency.sae_condition == "ftsae"].sort_values("target_class")["flip_rate_auc"].values
        n = min(len(g), len(f))
        if n >= 2:
            try:
                stat, p = wilcoxon(f[:n], g[:n])
            except ValueError:
                stat, p = float("nan"), float("nan")
        else:
            stat, p = float("nan"), float("nan")
        summary_rows.append(dict(
            dataset=dataset_key, mmd2=MMD2_BY_DATASET[dataset_key], test="sufficiency", k=INJECT_K,
            gsae_mean_selectivity=float(np.nanmean(g)) if len(g) else float("nan"),
            ftsae_mean_selectivity=float(np.nanmean(f)) if len(f) else float("nan"),
            gap=float(np.nanmean(f) - np.nanmean(g)) if len(f) and len(g) else float("nan"),
            wilcoxon_p=float(p),
        ))

    with (out_dir / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {out_dir / 'summary.csv'}")
    for row in summary_rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
