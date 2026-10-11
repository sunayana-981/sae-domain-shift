# When Do Sparse Autoencoders Transfer Across Adapted Vision Models?

Anonymous code release accompanying the submission. It covers every SAE
experiment in the paper: training generic (G-SAE) and fine-tuned (FT-SAE)
dictionaries on adapted encoders, the evaluation matrix, the latent
interventions, the retrain-versus-reuse predictor, the ablations, masked SAE
fine-tuning, and the language-model replication on Qwen3.5.

## Layout

| Directory | What it contains |
|---|---|
| `./` (top level) | Adapter training for CLIP, SigLIP2, DINOv2 and ALIGN (LoRA, DoRA, MaPLe, VPT-Deep), and the domain-distance computation |
| `CLIP_LoRA/` | CLIP-LoRA training code and dataset loaders used by the adapter scripts |
| `sae_vlm/` | Multi-backbone SAE pipeline: G-SAE vs. FT-SAE evaluation across the four encoders, neuron-level diagnostics, t-SNE and class-similarity figures |
| `patchsae/` | Patch-level SAE pipeline used for the controlled experiments: masked fine-tuning, causal steering, initialisation ablation, the domain-distance predictor, and the expanded backbone / SAE-architecture matrix |
| `qwenscope/` | Masked SAE fine-tuning of the released Qwen-Scope Top-k SAEs on LoRA-adapted Qwen3.5-2B/9B (see `qwenscope/README.md`) |

`sae_vlm/` and `patchsae/` both build on the public PatchSAE code base
(Lim et al., ICLR 2025, MIT license). Its license and third-party notices are
kept in `patchsae/LICENSE` and `patchsae/NOTICE`, with copies in `sae_vlm/`.

## Setup

```bash
conda create -n sae-shift python=3.10 && conda activate sae-shift
pip install torch torchvision open_clip_torch peft scikit-learn seaborn
pip install -r patchsae/requirements.txt
# Qwen experiments use a separate environment (transformers >= 5.17):
pip install -r qwenscope/requirements.txt
```

Scripts run from the directory they live in with `PYTHONPATH=./`, e.g.
`cd patchsae && PYTHONPATH=./ python tasks/train_sae_masked_finetune.py --help`.

**Paths.** Machine-specific absolute paths were replaced by the placeholder
`/path/to/sae-domain-shift` (the repository root). Datasets are expected under
`<repo>/data/`, adapter weights under `<repo>/lora_weights/` /
`<repo>/unified_weights/` / `<repo>/maple_weights/`, and SAE checkpoints under
`patchsae/out/` and `sae_vlm/out/`. Replace the placeholder or pass the
corresponding command-line flags. W&B logging uses `<wandb-entity>` as a
placeholder.

## Paper-to-code map

### Adapting the encoders

| Experiment | Entry point |
|---|---|
| LoRA / DoRA on CLIP, SigLIP2, DINOv2, ALIGN (16 datasets) | `unified_finetune.py`, `run_all_unified.sh`, `run_all_dora.sh` |
| MaPLe (CLIP) | `train_maple.py`, `run_all_maple.sh` |
| VPT-Deep (CLIP, SigLIP2, DINOv2) | `train_vpt.py`, `run_all_vpt.sh` |
| CLIP-LoRA (few-shot) | `lora_main.py`, `CLIP_LoRA/` |
| Domain distance to ImageNet (MMD², FID) | `compute_distances.py` |

### Generic vs. fine-tuned SAEs

| Experiment | Entry point |
|---|---|
| Generic SAE on ImageNet (CLIP / DINOv2 / ALIGN / SigLIP2) | `patchsae/tasks/train_sae_vit.py`, `patchsae/tasks/train_sae_dino_imagenet.py`, `patchsae/tasks/train_sae_align_imagenet.py`, `patchsae/run_train_base_sae_expanded.sh`, `sae_vlm/train_sae.py` |
| FT-SAE on adapted activations | `patchsae/tasks/train_sae_lora_clip.py`, `patchsae/run_all_sae_training.sh`, `patchsae/run_maple_sae_training.sh`, `sae_vlm/train_dora_sae.py` |
| Accuracy with the SAE reconstruction spliced in (ZS+G-SAE, FT+G-SAE, FT+FT-SAE) | `patchsae/eval_lora_sae_accuracy.py`, `patchsae/eval_maple_sae_accuracy.py`, `patchsae/eval_unified_metrics.py`, `sae_vlm/eval_lora_sae_accuracy.py`, `sae_vlm/eval_maple_sae_accuracy.py` |
| Reconstruction, sparsity, dead-neuron and cosine diagnostics | `patchsae/compute_sae_eval_metrics.py`, `patchsae/eval_all_saes_comprehensive.py`, `sae_vlm/generate_metrics_table.py` |
| t-SNE and class-similarity heat maps | `sae_vlm/visualize_neuron_tsne.py` |
| Top-activating images | `patchsae/run_sae_inference.py`, `patchsae/tasks/top_activating_grid.py` |
| Other SAE architectures (Top-k, JumpReLU, Matryoshka) | `patchsae/src/sae_training/sparse_autoencoder.py`, `patchsae/tasks/multiarch_lora.py` |

### Controlled experiments

| Experiment | Entry point |
|---|---|
| Arm definitions (G-SAE / FT-SAE / scratch / masked) and registry | `patchsae/tasks/controlled_common.py`, `patchsae/run_controlled_sae_training.sh`, `patchsae/tasks/check_matrix.py` |
| Single-latent ablation / amplification | `patchsae/tasks/eval_causal_steering.py` (protocol: `patchsae/tasks/CAUSAL_STEERING.md`) |
| Multi-latent necessity and sufficiency | `patchsae/analysis/h5_classifier_steering.py` |
| Initialisation: checkpoint vs. scratch | `patchsae/tasks/run_sae_initialization_ablation.py`, `patchsae/run_sae_initialization_ablation.sh`, `patchsae/tasks/summarize_sae_initialization_ablation.py` |
| Retrain-vs-reuse predictor (MMD² and concept distance) | `patchsae/analysis/mmd_degradation_predictor.py`, `patchsae/analysis/concept_overlap_distance.py`, `patchsae/analysis/degradation_regressor.py`, `patchsae/analysis/imagenet_baseline.py` (details: `patchsae/analysis/MMD_PREDICTOR_README.md`) |
| Dictionary-size ablation | `patchsae/tasks/train_sae_vit.py --expansion_factor {16,32,64,...}`, `sae_vlm/train_sae.py --expansion_factor` |
| Generic SAE trained on CC3M (DN-CBM dictionary) | `eval_base_sae_accuracy.py`, `eval_sae_concept_space.py`, `eval_sae_topk.py` |
| Training effort vs. domain distance | `plot_training_vs_mmd.py`, `sae_vlm/sae_convergence_plot.py` |

### Masked SAE fine-tuning

| Experiment | Entry point |
|---|---|
| Trainer (gradient-masked protected units, exact-protection checks) | `patchsae/src/sae_training/masked_sae_trainer.py`, `patchsae/tasks/train_sae_masked_finetune.py` |
| Masked vs. full FT-SAE, all datasets | `patchsae/run_masked_finetune_all_datasets.sh`, `patchsae/run_fullft_all_datasets.sh`, `patchsae/eval_masked_sae.py`, `patchsae/tasks/eval_matrix.py` |
| Protection-fraction sweep | `patchsae/run_masking_ablation.sh`, `patchsae/run_masked_finetune_protection_sweep.sh`, `patchsae/run_selection_and_fraction_sweep.sh`, `patchsae/eval_masked_finetune_sweep.py` |
| Activity overlap, selection-criterion pruning, residual rank | `patchsae/tasks/exp1_activity_overlap.py`, `patchsae/tasks/exp4_prune_sweep.py`, `patchsae/tasks/exp5_residual_analysis.py` |
| Expanded backbone × SAE-architecture × adapter matrix | `patchsae/run_masked_finetune_expanded.sh`, `patchsae/tasks/eval_expanded_matrix_reconstruction.py`, `patchsae/tasks/eval_expanded_backbone_sweep.py`, `run_expanded_matrix_all.sh` |
| Qwen3.5-2B (Pile corpora) and 9B (DKSplit) | `qwenscope/run_experiment.sh`, `qwenscope/scripts/`, `qwenscope/dksplit/` |
| Latent necessity on Qwen | `qwenscope/scripts/eval_steering.py` |

## Tests

```bash
cd patchsae && PYTHONPATH=./ python -m pytest tests -q
```

## Not included

Datasets, trained adapter and SAE checkpoints, and experiment outputs are not
part of this release because of their size. All of them can be regenerated
with the scripts above.
