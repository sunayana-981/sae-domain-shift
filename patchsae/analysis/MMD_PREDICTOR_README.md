# Held-out MMD → G-SAE degradation predictor

This workflow tests a fixed claim:

> A target domain's CLIP MMD² from ImageNet predicts how much a generic
> ImageNet SAE will degrade the accuracy of the LoRA-adapted model.

The primary outcome is fixed as

```text
degradation_pp = accuracy(LoRA, no SAE) - accuracy(LoRA + G-SAE)
```

Positive values mean the G-SAE intervention hurts accuracy. The predictor is
the repository's unbiased RBF MMD² in normalized OpenAI CLIP ViT-L/14 image
feature space. The model is ordinary least squares with an intercept and one
untransformed predictor. Do not choose another outcome, transformation, or
model after examining the three held-out outcomes.

## Audit result

The intended development cohort is the 15 non-PACS/non-OfficeHome datasets in
`../distance_results.csv`. Only five of those currently have paired no-SAE and
G-SAE outcomes in
`out/workshop_paper/matrix_final_standard/results.csv`: Caltech101, CUB-200,
DTD, EuroSAT, and UCF101. FGVC has a paired outcome but is not one of the
intended 15. Therefore:

- the planned 15-domain predictor cannot yet be fit;
- the available six-domain fit is exploratory and the code labels it as such;
- OfficeHome, KITTI, and Cityscapes remain the fixed held-out domains;
- an artifact created now cannot honestly be described as a prospective
  prediction if investigators have already inspected the existing holdout
  matrix.

The exact missing development outcomes, local checkpoints, and loader or
class-name blockers are recorded in
`configs/mmd_predictor_development_outcomes.json`. PathMNIST and Oxford Pets
are runnable with the current registry. ImageNet variants and Flowers need
semantic class-name mappings before zero-shot accuracy is valid; Cars and
Food101 need a parquet-capable evaluation loader; ImageNet-O also lacks a local
CLIP LoRA checkpoint.

For a defensible final analysis, rerun all 15 development domains under one
evaluation protocol rather than mixing the existing 225-image cells with new,
larger cells. Use the same LoRA checkpoint family, prompts, examples,
evaluation seed, and G-SAE checkpoint for every domain.

## Reproducible two-stage protocol

Run from `patchsae` with the project environment:

```bash
PY=/path/to/conda/envs/dncbm310/bin/python
```

First compute the missing MMD values from the already cached, outcome-free
features. OfficeHome is pooled at the image level; its four domain MMDs are not
averaged.

```bash
$PY analysis/compute_cached_clip_mmd.py \
  --datasets fgvc,officehome,kitti,cityscapes \
  --sample-size 2000 \
  --seed 42 \
  --output out/controlled/mmd_predictor_inputs/cached_mmd.csv
```

### Planned 15-domain fit

After all 15 paired outcomes have been evaluated, place only those development
rows in a train-only CSV:

```bash
$PY analysis/mmd_degradation_predictor.py extract-matrix \
  --input out/controlled/mmd_predictor_dev15/results.csv \
  --partition train \
  --output out/controlled/mmd_predictor_inputs/train15_outcomes.csv
```

Freeze predictions before producing or opening the holdout outcome file:

```bash
$PY analysis/mmd_degradation_predictor.py fit \
  --distance ../distance_results.csv \
  --distance out/controlled/mmd_predictor_inputs/cached_mmd.csv \
  --train-outcomes out/controlled/mmd_predictor_inputs/train15_outcomes.csv \
  --cohort planned \
  --out-dir out/controlled/mmd_predictor/blind_dev15
```

The command refuses a training outcome file containing OfficeHome, KITTI, or
Cityscapes, requires the exact 15-domain split, will not overwrite an existing
artifact, and hashes every blind prediction file.

Only after that command completes, derive and score the held-out outcomes:

```bash
$PY analysis/mmd_degradation_predictor.py extract-matrix \
  --input out/controlled/mmd_predictor_holdout/results.csv \
  --partition holdout \
  --output out/controlled/mmd_predictor_inputs/holdout_outcomes.csv

$PY analysis/mmd_degradation_predictor.py score \
  --artifact-dir out/controlled/mmd_predictor/blind_dev15 \
  --holdout-outcomes out/controlled/mmd_predictor_inputs/holdout_outcomes.csv \
  --out-dir out/controlled/mmd_predictor/scored_dev15
```

`score` verifies the frozen file hashes and never refits the model.

### Exploratory six-domain diagnostic

This is useful for checking the pipeline, but it is not the planned result:

```bash
$PY analysis/mmd_degradation_predictor.py extract-matrix \
  --input out/workshop_paper/matrix_final_standard/results.csv \
  --partition exploratory-train \
  --output out/controlled/mmd_predictor_inputs/train6_exploratory.csv

$PY analysis/mmd_degradation_predictor.py fit \
  --distance ../distance_results.csv \
  --distance out/controlled/mmd_predictor_inputs/cached_mmd.csv \
  --train-outcomes out/controlled/mmd_predictor_inputs/train6_exploratory.csv \
  --cohort exploratory-current \
  --out-dir out/controlled/mmd_predictor/blind_exploratory6
```

## Outputs

The blind artifact contains:

- `fit_metrics.json`: apparent and leave-one-domain-out R², Spearman ρ,
  Kendall τ-b, errors, coefficient intervals, and analysis status;
- `model.json`: exact OLS coefficients and interval parameters;
- `training_diagnostics.csv`: fitted and LODO predictions, residuals,
  leverage, and Cook's distance;
- `holdout_predictions.csv`: frozen predictions, mean confidence intervals,
  observation prediction intervals, and extrapolation flags;
- `fit_table.tex` and `blind_predictions_table.tex`: paper-ready fragments;
- `fit_manifest.json`: input and artifact hashes, split, environment, and
  leakage safeguards.

The revealed score directory adds `holdout_scored.csv`,
`score_metrics.json`, `holdout_table.tex`, and a separate score manifest.

## Interpretation limits

The dataset is the experimental unit. Apparent training R² is optimistic, so
report leave-one-domain-out R² and rank correlations beside it. Classical 95%
prediction intervals assume independent homoscedastic domain-level residuals
and are fragile with only 15 domains; extrapolation flags should be discussed
explicitly. With only three held-out domains, held-out R² and rank statistics
are highly discrete, so also report every prediction and absolute error.

MMD is computed with CLIP ViT-L/14 while the SAE intervention uses CLIP
ViT-B/16. This makes MMD an external domain-distance covariate, not a
mechanistic quantity from the evaluated model. A successful prediction is a
deployment diagnostic, not causal proof that MMD itself causes degradation.
