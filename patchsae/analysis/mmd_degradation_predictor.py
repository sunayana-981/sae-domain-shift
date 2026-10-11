#!/usr/bin/env python3
"""Freeze and score a held-out MMD -> G-SAE degradation prediction.

The primary endpoint is fixed before fitting:

    degradation_pp = accuracy(LoRA, no SAE) - accuracy(LoRA + G-SAE)

Positive values mean that inserting the generic ImageNet SAE harms the adapted
model.  The workflow deliberately has separate commands:

``extract-matrix``
    Derive the predeclared endpoint from the evaluation matrix.
``fit``
    Fit a one-predictor OLS model on training domains and freeze predictions for
    outcome-blind held-out domains.
``score``
    Reveal held-out outcomes, verify the frozen artifact hashes, and score the
    predictions without refitting.

The separation is an operational leakage safeguard, not proof that a
retrospective run was prospective.  See ``analysis/MMD_PREDICTOR_README.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy import stats


SCHEMA_VERSION = 1
SCRIPT_VERSION = "1.0.0"
DEFAULT_SPLIT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "mmd_predictor_split.json"
)
ENDPOINT_ID = "lora_no_sae_minus_lora_gsae_accuracy_pp"
ENDPOINT_LABEL = r"$\Delta_{\mathrm{G\text{-}SAE}}$ accuracy (pp)"


DATASET_ALIASES = {
    "caltech101": "caltech101",
    "cub200": "cub2002011",
    "cub2002011": "cub2002011",
    "dtd": "dtd",
    "eurosat": "eurosat",
    "fgvc": "fgvc",
    "fgvcaircraft": "fgvc",
    "ucf101": "ucf101",
    "officehome": "officehome",
    "kitti": "kitti",
    "cityscapes": "cityscapes",
    "pathmnist": "pathmnist",
    "medmnist": "medmnist",
    "oxfordpets": "oxford_pets",
    "stanfordcars": "stanford_cars",
    "food101": "food101",
    "flowers102": "flowers102",
    "oxfordflowers": "flowers102",
}


class PredictorError(ValueError):
    """Raised when an input violates a scientific or leakage safeguard."""


def canonical_dataset(value: Any) -> str:
    """Return a stable dataset key without conflating OfficeHome subdomains."""
    raw = str(value).strip()
    if not raw:
        raise PredictorError("Encountered an empty dataset name")
    lower = raw.lower().strip()
    if lower.startswith("oh/"):
        # Per-domain MMD is not the MMD of the pooled OfficeHome mixture.
        return lower
    compact = re.sub(r"[^a-z0-9]+", "", lower)
    return DATASET_ALIASES.get(compact, lower.replace("_", "-").replace(" ", "-"))


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [canonical_dataset(item) for item in value.split(",") if item.strip()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True)
    _atomic_write_bytes(path, (data + "\n").encode("utf-8"))


def write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    data = frame.to_csv(index=False, lineterminator="\n")
    _atomic_write_bytes(path, data.encode("utf-8"))


def _require_new_file(path: Path) -> None:
    if path.exists():
        raise PredictorError(
            f"Refusing to overwrite existing file: {path}. "
            "Use a new path so the frozen analysis trail remains intact."
        )


def _require_new_artifact_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise PredictorError(
            f"Refusing to overwrite non-empty artifact directory: {path}. "
            "Use a new directory for each frozen fit."
        )


def load_split_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PredictorError(f"Split config not found: {path}")
    with path.open() as handle:
        config = json.load(handle)
    required = {
        "heldout_datasets",
        "current_training_datasets",
        "planned_development_datasets",
        "planned_training_n",
        "endpoint_id",
    }
    missing = required - set(config)
    if missing:
        raise PredictorError(f"Split config lacks keys: {sorted(missing)}")
    if config["endpoint_id"] != ENDPOINT_ID:
        raise PredictorError(
            f"Split config endpoint {config['endpoint_id']!r} does not match "
            f"the implemented endpoint {ENDPOINT_ID!r}"
        )
    config["heldout_datasets"] = [
        canonical_dataset(x) for x in config["heldout_datasets"]
    ]
    config["current_training_datasets"] = [
        canonical_dataset(x) for x in config["current_training_datasets"]
    ]
    config["planned_development_datasets"] = [
        canonical_dataset(x) for x in config["planned_development_datasets"]
    ]
    planned_train = set(config["planned_development_datasets"])
    current_train = set(config["current_training_datasets"])
    holdout = set(config["heldout_datasets"])
    overlap = (planned_train | current_train) & holdout
    if overlap:
        raise PredictorError(f"Train/holdout overlap in split config: {sorted(overlap)}")
    return config


def _parse_skipped(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    true_values = {"true", "1", "yes", "y"}
    return series.fillna(False).map(lambda x: str(x).strip().lower() in true_values)


def extract_endpoint_from_matrix(
    matrix_path: Path,
    *,
    datasets: Sequence[str],
    vit_type: str = "lora",
    gsae_condition: str = "gsae",
) -> pd.DataFrame:
    """Extract one paired primary endpoint row per requested dataset."""
    frame = pd.read_csv(matrix_path)
    required = {
        "dataset",
        "vit_type",
        "sae_condition",
        "zeroshot_acc",
        "skipped",
    }
    missing = required - set(frame.columns)
    if missing:
        raise PredictorError(
            f"Evaluation matrix {matrix_path} lacks columns: {sorted(missing)}"
        )

    frame = frame.copy()
    frame["dataset"] = frame["dataset"].map(canonical_dataset)
    frame["skipped"] = _parse_skipped(frame["skipped"])
    requested = list(dict.fromkeys(canonical_dataset(x) for x in datasets))
    if not requested:
        raise PredictorError("At least one dataset must be requested")

    subset = frame[
        (frame["vit_type"].astype(str).str.lower() == vit_type.lower())
        & frame["sae_condition"].isin(["none", gsae_condition])
        & ~frame["skipped"]
        & frame["dataset"].isin(requested)
    ].copy()
    duplicated = subset.duplicated(["dataset", "sae_condition"], keep=False)
    if duplicated.any():
        cells = subset.loc[duplicated, ["dataset", "sae_condition"]]
        raise PredictorError(
            "Duplicate evaluation cells; aggregate seeds explicitly before using "
            f"this workflow:\n{cells.to_string(index=False)}"
        )

    pivot = subset.pivot(
        index="dataset", columns="sae_condition", values="zeroshot_acc"
    )
    missing_datasets = [
        dataset
        for dataset in requested
        if dataset not in pivot.index
        or "none" not in pivot.columns
        or gsae_condition not in pivot.columns
        or pd.isna(pivot.loc[dataset, "none"])
        or pd.isna(pivot.loc[dataset, gsae_condition])
    ]
    if missing_datasets:
        raise PredictorError(
            "Missing complete no-SAE/G-SAE pairs for: "
            f"{sorted(missing_datasets)}"
        )

    n_images: dict[str, int | None] = {}
    if "n_images" in subset.columns:
        for dataset, part in subset.groupby("dataset"):
            values = sorted(
                {
                    int(value)
                    for value in pd.to_numeric(part["n_images"], errors="coerce")
                    if np.isfinite(value)
                }
            )
            if len(values) > 1:
                raise PredictorError(
                    f"Paired cells use different n_images for {dataset}: {values}"
                )
            n_images[dataset] = values[0] if values else None

    rows = []
    for dataset in requested:
        ft_acc = float(pivot.loc[dataset, "none"])
        gsae_acc = float(pivot.loc[dataset, gsae_condition])
        rows.append(
            {
                "dataset": dataset,
                "endpoint_id": ENDPOINT_ID,
                "ft_no_sae_accuracy_pct": ft_acc,
                "ft_gsae_accuracy_pct": gsae_acc,
                "degradation_pp": ft_acc - gsae_acc,
                "n_images": n_images.get(dataset),
                "vit_type": vit_type,
                "gsae_condition": gsae_condition,
            }
        )
    return pd.DataFrame(rows)


def command_extract_matrix(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    _require_new_file(output)
    split_path = Path(args.split_config).resolve()
    split = load_split_config(split_path)
    if args.partition == "train":
        datasets = split["planned_development_datasets"]
    elif args.partition == "exploratory-train":
        datasets = split["current_training_datasets"]
    elif args.partition == "holdout":
        datasets = split["heldout_datasets"]
    else:
        datasets = parse_csv_list(args.datasets)
        if not datasets:
            raise PredictorError(
                "--datasets is required when --partition=custom"
            )

    frame = extract_endpoint_from_matrix(
        Path(args.input).resolve(),
        datasets=datasets,
        vit_type=args.vit_type,
        gsae_condition=args.gsae_condition,
    )
    write_csv(output, frame)
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    write_json(
        meta_path,
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "endpoint_id": ENDPOINT_ID,
            "endpoint_definition": (
                "LoRA no-SAE accuracy minus LoRA+G-SAE accuracy, percentage points"
            ),
            "partition": args.partition,
            "datasets": frame["dataset"].tolist(),
            "source_matrix": str(Path(args.input).resolve()),
            "source_matrix_sha256": sha256_file(Path(args.input).resolve()),
            "split_config": str(split_path),
            "split_config_sha256": sha256_file(split_path),
            "output_sha256": sha256_file(output),
        },
    )
    print(f"Wrote {output}")
    print(f"Wrote {meta_path}")


def load_distances(paths: Sequence[Path], x_column: str) -> pd.DataFrame:
    if not paths:
        raise PredictorError("At least one distance CSV is required")
    rows: list[dict[str, Any]] = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = {"dataset", x_column} - set(frame.columns)
        if missing:
            raise PredictorError(f"{path} lacks columns: {sorted(missing)}")
        for _, row in frame.iterrows():
            raw_dataset = str(row["dataset"])
            if raw_dataset.startswith("**"):
                continue
            value = pd.to_numeric(pd.Series([row[x_column]]), errors="coerce").iloc[0]
            if not np.isfinite(value):
                continue
            rows.append(
                {
                    "dataset": canonical_dataset(raw_dataset),
                    "mmd": float(value),
                    "distance_source": str(path.resolve()),
                }
            )
    if not rows:
        raise PredictorError(f"No finite values found for distance column {x_column!r}")
    frame = pd.DataFrame(rows)
    merged_rows = []
    for dataset, part in frame.groupby("dataset", sort=True):
        values = part["mmd"].to_numpy(dtype=float)
        tolerance = max(5e-7, 5e-6 * max(1.0, float(np.max(np.abs(values)))))
        if float(np.ptp(values)) > tolerance:
            detail = part[["mmd", "distance_source"]].to_dict("records")
            raise PredictorError(
                f"Conflicting distance values for {dataset}: {detail}"
            )
        merged_rows.append(
            {
                "dataset": dataset,
                "mmd": float(values[0]),
                "distance_source": ";".join(sorted(set(part["distance_source"]))),
            }
        )
    return pd.DataFrame(merged_rows)


def load_outcomes(path: Path, y_column: str = "degradation_pp") -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"dataset", y_column} - set(frame.columns)
    if missing:
        raise PredictorError(f"{path} lacks columns: {sorted(missing)}")
    frame = frame.copy()
    frame["dataset"] = frame["dataset"].map(canonical_dataset)
    frame["outcome"] = pd.to_numeric(frame[y_column], errors="coerce")
    bad = frame[~np.isfinite(frame["outcome"])]
    if len(bad):
        raise PredictorError(
            f"Non-finite outcomes for datasets: {bad['dataset'].tolist()}"
        )
    duplicates = frame.duplicated("dataset", keep=False)
    if duplicates.any():
        raise PredictorError(
            "Multiple rows per dataset are not silently pooled. Aggregate seeds "
            "first, or use a hierarchical analysis. Duplicate datasets: "
            f"{sorted(frame.loc[duplicates, 'dataset'].unique())}"
        )
    if "endpoint_id" in frame.columns:
        endpoint_ids = set(frame["endpoint_id"].dropna().astype(str))
        if endpoint_ids != {ENDPOINT_ID}:
            raise PredictorError(
                f"Outcome endpoint mismatch: found {sorted(endpoint_ids)}, "
                f"expected only {ENDPOINT_ID!r}"
            )
    return frame[["dataset", "outcome"]].sort_values("dataset").reset_index(drop=True)


@dataclass(frozen=True)
class LinearFit:
    intercept: float
    slope: float
    x_mean: float
    sxx: float
    n: int
    df_residual: int
    sse: float
    mse: float
    residual_sd: float
    r2: float
    adjusted_r2: float
    t_critical: float
    xtx_inverse: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * np.asarray(x, dtype=float)


def fit_linear(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> LinearFit:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise PredictorError("x and y must be one-dimensional and equally sized")
    if not 0 < alpha < 1:
        raise PredictorError("alpha must lie strictly between 0 and 1")
    if len(x) < 3:
        raise PredictorError("OLS with an intercept requires at least 3 datasets")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise PredictorError("x and y must be finite")
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    sxx = float(np.sum((x - x_mean) ** 2))
    if sxx <= np.finfo(float).eps:
        raise PredictorError("MMD has zero variance; the slope is unidentified")
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / sxx)
    intercept = float(y_mean - slope * x_mean)
    predictions = intercept + slope * x
    residuals = y - predictions
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y - y_mean) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    df = len(x) - 2
    mse = sse / df
    residual_sd = math.sqrt(max(0.0, mse))
    adjusted_r2 = (
        float(1.0 - (1.0 - r2) * (len(x) - 1) / df)
        if math.isfinite(r2)
        else float("nan")
    )
    design = np.column_stack([np.ones(len(x)), x])
    xtx_inverse = np.linalg.inv(design.T @ design)
    t_critical = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    return LinearFit(
        intercept=intercept,
        slope=slope,
        x_mean=x_mean,
        sxx=sxx,
        n=len(x),
        df_residual=df,
        sse=sse,
        mse=mse,
        residual_sd=residual_sd,
        r2=r2,
        adjusted_r2=adjusted_r2,
        t_critical=t_critical,
        xtx_inverse=xtx_inverse,
    )


def leave_one_out_predictions(
    x: np.ndarray, y: np.ndarray, alpha: float = 0.05
) -> np.ndarray:
    predictions = np.empty(len(x), dtype=float)
    for index in range(len(x)):
        keep = np.arange(len(x)) != index
        fold = fit_linear(x[keep], y[keep], alpha=alpha)
        predictions[index] = fold.predict(np.asarray([x[index]]))[0]
    return predictions


def r2_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    if denominator <= np.finfo(float).eps:
        return float("nan")
    return float(1.0 - np.sum((observed - predicted) ** 2) / denominator)


def rank_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    spearman = stats.spearmanr(observed, predicted)
    kendall = stats.kendalltau(observed, predicted, variant="b")
    return {
        "spearman_rho": float(spearman.statistic),
        "spearman_p_two_sided": float(spearman.pvalue),
        "kendall_tau_b": float(kendall.statistic),
        "kendall_p_two_sided": float(kendall.pvalue),
    }


def prediction_rows(
    model: LinearFit, holdout: pd.DataFrame
) -> pd.DataFrame:
    train_x_min = float(holdout.attrs["train_x_min"])
    train_x_max = float(holdout.attrs["train_x_max"])
    train_range = max(train_x_max - train_x_min, np.finfo(float).eps)
    rows = []
    for _, row in holdout.sort_values("dataset").iterrows():
        x0 = float(row["mmd"])
        design = np.asarray([1.0, x0])
        leverage_mean = float(design @ model.xtx_inverse @ design)
        se_mean = math.sqrt(max(0.0, model.mse * leverage_mean))
        se_prediction = math.sqrt(max(0.0, model.mse * (1.0 + leverage_mean)))
        predicted = float(model.predict(np.asarray([x0]))[0])
        outside_by = max(train_x_min - x0, x0 - train_x_max, 0.0)
        rows.append(
            {
                "dataset": row["dataset"],
                "MMD2_clip": x0,
                "predicted_degradation_pp": predicted,
                "mean_ci95_lower_pp": predicted - model.t_critical * se_mean,
                "mean_ci95_upper_pp": predicted + model.t_critical * se_mean,
                "prediction_interval95_lower_pp": (
                    predicted - model.t_critical * se_prediction
                ),
                "prediction_interval95_upper_pp": (
                    predicted + model.t_critical * se_prediction
                ),
                "mean_prediction_leverage": leverage_mean,
                "outside_training_mmd_range": bool(outside_by > 0),
                "extrapolation_distance_in_training_ranges": outside_by / train_range,
                "training_mmd_min": train_x_min,
                "training_mmd_max": train_x_max,
                "outcome_revealed": False,
            }
        )
    return pd.DataFrame(rows)


def training_diagnostics(
    frame: pd.DataFrame, model: LinearFit, loo_predictions: np.ndarray
) -> pd.DataFrame:
    x = frame["mmd"].to_numpy(dtype=float)
    y = frame["outcome"].to_numpy(dtype=float)
    fitted = model.predict(x)
    residual = y - fitted
    leverage = 1.0 / model.n + (x - model.x_mean) ** 2 / model.sxx
    if model.mse > 0:
        standardized = residual / np.sqrt(model.mse * np.maximum(1e-15, 1 - leverage))
        cooks = (
            (residual**2 / (2.0 * model.mse))
            * leverage
            / np.maximum(1e-15, (1.0 - leverage) ** 2)
        )
    else:
        standardized = np.zeros_like(residual)
        cooks = np.zeros_like(residual)
    result = frame[["dataset", "mmd", "outcome"]].copy()
    result = result.rename(
        columns={"mmd": "MMD2_clip", "outcome": "observed_degradation_pp"}
    )
    result["fitted_degradation_pp"] = fitted
    result["residual_pp"] = residual
    result["loo_predicted_degradation_pp"] = loo_predictions
    result["loo_error_pp"] = y - loo_predictions
    result["leverage"] = leverage
    result["standardized_residual"] = standardized
    result["cooks_distance"] = cooks
    return result


def _environment_record() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def fit_table_latex(metrics: Mapping[str, Any]) -> str:
    rows = [
        ("Training datasets", f"{metrics['n_training_datasets']}"),
        ("Apparent $R^2$", f"{metrics['apparent_r2']:.3f}"),
        ("Adjusted $R^2$", f"{metrics['adjusted_r2']:.3f}"),
        ("LODO $R^2$", f"{metrics['lodo_r2']:.3f}"),
        ("LODO Spearman $\\rho$", f"{metrics['lodo_spearman_rho']:.3f}"),
        ("LODO Kendall $\\tau_b$", f"{metrics['lodo_kendall_tau_b']:.3f}"),
        ("LODO RMSE (pp)", f"{metrics['lodo_rmse_pp']:.2f}"),
    ]
    lines = [
        "% Auto-generated; do not hand-edit.",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Statistic & Value \\",
        r"\midrule",
    ]
    lines.extend(f"{label} & {value} \\\\" for label, value in rows)
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def blind_predictions_table_latex(predictions: pd.DataFrame) -> str:
    lines = [
        "% Auto-generated before held-out outcomes are revealed.",
        r"\begin{tabular}{lrrlc}",
        r"\toprule",
        (
            r"Dataset & CLIP MMD$^2$ & Predicted degradation (pp) "
            r"& 95\% prediction interval & Extrapolation \\"
        ),
        r"\midrule",
    ]
    for _, row in predictions.iterrows():
        interval = (
            f"[{row['prediction_interval95_lower_pp']:.2f}, "
            f"{row['prediction_interval95_upper_pp']:.2f}]"
        )
        lines.append(
            f"{_latex_escape(row['dataset'])} & {row['MMD2_clip']:.4f} & "
            f"{row['predicted_degradation_pp']:.2f} & {interval} & "
            f"{'yes' if row['outside_training_mmd_range'] else 'no'} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def _artifact_hashes(directory: Path, names: Iterable[str]) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in names}


def fit_and_freeze(
    *,
    distance_paths: Sequence[Path],
    train_outcomes_path: Path,
    split_config_path: Path,
    out_dir: Path,
    cohort: str = "planned",
    x_column: str = "MMD2_clip",
    y_column: str = "degradation_pp",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fit on training domains and write a hash-locked blind artifact."""
    _require_new_artifact_dir(out_dir)
    split = load_split_config(split_config_path)
    holdout_names = split["heldout_datasets"]
    if cohort == "planned":
        planned_train_names = split["planned_development_datasets"]
    elif cohort == "exploratory-current":
        planned_train_names = split["current_training_datasets"]
    else:
        raise PredictorError(f"Unknown training cohort: {cohort}")
    distances = load_distances(distance_paths, x_column)
    outcomes = load_outcomes(train_outcomes_path, y_column)

    leaked = set(outcomes["dataset"]) & set(holdout_names)
    if leaked:
        raise PredictorError(
            "LEAKAGE GUARD: training outcome file contains held-out outcomes for "
            f"{sorted(leaked)}. Create a physically separate train-only CSV."
        )
    actual_train_names = set(outcomes["dataset"])
    expected_train_names = set(planned_train_names)
    if actual_train_names != expected_train_names:
        raise PredictorError(
            "Training outcomes do not exactly match the frozen current training "
            f"split. Missing={sorted(expected_train_names - actual_train_names)}, "
            f"unexpected={sorted(actual_train_names - expected_train_names)}"
        )

    train = outcomes.merge(distances, on="dataset", how="left", validate="one_to_one")
    missing_train_distance = train.loc[train["mmd"].isna(), "dataset"].tolist()
    if missing_train_distance:
        raise PredictorError(
            f"Missing MMD values for training datasets: {missing_train_distance}"
        )
    holdout = pd.DataFrame({"dataset": holdout_names}).merge(
        distances, on="dataset", how="left", validate="one_to_one"
    )
    missing_holdout_distance = holdout.loc[holdout["mmd"].isna(), "dataset"].tolist()
    if missing_holdout_distance:
        raise PredictorError(
            f"Missing MMD values for held-out datasets: {missing_holdout_distance}. "
            "Per-domain OfficeHome rows cannot substitute for pooled OfficeHome MMD."
        )

    train = train.sort_values("dataset").reset_index(drop=True)
    x = train["mmd"].to_numpy(dtype=float)
    y = train["outcome"].to_numpy(dtype=float)
    if len(train) < 4:
        raise PredictorError(
            "At least four training domains are required for leave-one-domain-out "
            "diagnostics with an intercept and slope"
        )
    model = fit_linear(x, y, alpha=alpha)
    loo_predictions = leave_one_out_predictions(x, y, alpha=alpha)
    loo_ranks = rank_metrics(y, loo_predictions)
    apparent_ranks = rank_metrics(y, model.predict(x))

    slope_se = math.sqrt(model.mse / model.sxx)
    intercept_se = math.sqrt(
        model.mse * (1.0 / model.n + model.x_mean**2 / model.sxx)
    )
    metrics = {
        "analysis_status": (
            "confirmatory_frozen_planned_development_cohort"
            if cohort == "planned"
            and model.n == int(split["planned_training_n"])
            else "exploratory_nonplanned_or_incomplete_training_cohort"
        ),
        "training_cohort": cohort,
        "n_training_datasets": model.n,
        "planned_training_n": int(split["planned_training_n"]),
        "n_heldout_datasets": len(holdout),
        "apparent_r2": model.r2,
        "adjusted_r2": model.adjusted_r2,
        "apparent_spearman_rho": apparent_ranks["spearman_rho"],
        "apparent_spearman_p_two_sided": apparent_ranks["spearman_p_two_sided"],
        "apparent_kendall_tau_b": apparent_ranks["kendall_tau_b"],
        "apparent_kendall_p_two_sided": apparent_ranks["kendall_p_two_sided"],
        "lodo_r2": r2_score(y, loo_predictions),
        "lodo_spearman_rho": loo_ranks["spearman_rho"],
        "lodo_spearman_p_two_sided": loo_ranks["spearman_p_two_sided"],
        "lodo_kendall_tau_b": loo_ranks["kendall_tau_b"],
        "lodo_kendall_p_two_sided": loo_ranks["kendall_p_two_sided"],
        "lodo_mae_pp": float(np.mean(np.abs(y - loo_predictions))),
        "lodo_rmse_pp": float(np.sqrt(np.mean((y - loo_predictions) ** 2))),
        "residual_sd_pp": model.residual_sd,
        "intercept_pp": model.intercept,
        "intercept_se_pp": intercept_se,
        "intercept_ci95_lower_pp": model.intercept
        - model.t_critical * intercept_se,
        "intercept_ci95_upper_pp": model.intercept
        + model.t_critical * intercept_se,
        "slope_pp_per_mmd2": model.slope,
        "slope_se_pp_per_mmd2": slope_se,
        "slope_ci95_lower_pp_per_mmd2": model.slope
        - model.t_critical * slope_se,
        "slope_ci95_upper_pp_per_mmd2": model.slope
        + model.t_critical * slope_se,
        "alpha": alpha,
        "interval_method": (
            "classical Student-t OLS mean and observation intervals; "
            "homoscedastic independent domain-level residuals assumed"
        ),
    }

    diagnostics = training_diagnostics(train, model, loo_predictions)
    holdout.attrs["train_x_min"] = float(x.min())
    holdout.attrs["train_x_max"] = float(x.max())
    predictions = prediction_rows(model, holdout)

    model_payload = {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "model_family": "ordinary_least_squares",
        "formula": "degradation_pp = intercept + slope * MMD2_clip",
        "feature_transform": "none",
        "outcome_transform": "none",
        "intercept": model.intercept,
        "slope": model.slope,
        "x_mean": model.x_mean,
        "sxx": model.sxx,
        "n": model.n,
        "df_residual": model.df_residual,
        "sse": model.sse,
        "mse": model.mse,
        "residual_sd": model.residual_sd,
        "t_critical": model.t_critical,
        "xtx_inverse": model.xtx_inverse.tolist(),
        "alpha": alpha,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "model.json": lambda path: write_json(path, model_payload),
        "fit_metrics.json": lambda path: write_json(path, metrics),
        "training_diagnostics.csv": lambda path: write_csv(path, diagnostics),
        "holdout_predictions.csv": lambda path: write_csv(path, predictions),
        "fit_table.tex": lambda path: write_text(path, fit_table_latex(metrics)),
        "blind_predictions_table.tex": lambda path: write_text(
            path, blind_predictions_table_latex(predictions)
        ),
    }
    for name, writer in files.items():
        writer(out_dir / name)

    canonical_rows = train[["dataset", "mmd", "outcome"]].to_csv(
        index=False, lineterminator="\n"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "status": "frozen_blind_predictions",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": metrics["analysis_status"],
        "training_cohort": cohort,
        "endpoint": {
            "id": ENDPOINT_ID,
            "definition": (
                "LoRA no-SAE accuracy minus LoRA+generic-SAE accuracy, "
                "percentage points; positive means G-SAE degradation"
            ),
        },
        "predictor": {
            "column": x_column,
            "definition": (
                "unbiased RBF-kernel MMD^2 from target images to ImageNet "
                "in normalized OpenAI CLIP ViT-L/14 feature space"
            ),
        },
        "predeclared_model": (
            "one-predictor OLS with intercept and no transform or holdout tuning"
        ),
        "training_datasets": train["dataset"].tolist(),
        "heldout_datasets": predictions["dataset"].tolist(),
        "planned_training_n": int(split["planned_training_n"]),
        "input_files": {
            "distance_csvs": [
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path.resolve()),
                }
                for path in distance_paths
            ],
            "train_outcomes": {
                "path": str(train_outcomes_path.resolve()),
                "sha256": sha256_file(train_outcomes_path.resolve()),
            },
            "split_config": {
                "path": str(split_config_path.resolve()),
                "sha256": sha256_file(split_config_path.resolve()),
            },
        },
        "canonical_training_rows_sha256": sha256_bytes(
            canonical_rows.encode("utf-8")
        ),
        "environment": _environment_record(),
        "artifact_files": _artifact_hashes(out_dir, files),
        "leakage_safeguards": [
            "held-out datasets are forbidden in the training outcome CSV",
            "training dataset identities must exactly match the frozen split",
            "model family, endpoint, and untransformed predictor are fixed",
            "blind prediction files are hash-locked before score is run",
            "score writes to a separate directory and never refits the model",
        ],
        "retrospective_caveat": (
            "Artifact chronology establishes when this script wrote predictions; "
            "it cannot establish that investigators had never seen outcomes."
        ),
    }
    write_json(out_dir / "fit_manifest.json", manifest)
    print(f"Frozen blind artifact: {out_dir}")
    print(
        f"Fit status: {metrics['analysis_status']} | n={model.n} | "
        f"apparent R2={model.r2:.3f} | LODO R2={metrics['lodo_r2']:.3f}"
    )
    return manifest


def command_fit(args: argparse.Namespace) -> None:
    fit_and_freeze(
        distance_paths=[Path(path).resolve() for path in args.distance],
        train_outcomes_path=Path(args.train_outcomes).resolve(),
        split_config_path=Path(args.split_config).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        cohort=args.cohort,
        x_column=args.x_column,
        y_column=args.y_column,
        alpha=args.alpha,
    )


def _validate_blind_artifact(artifact_dir: Path) -> dict[str, Any]:
    manifest_path = artifact_dir / "fit_manifest.json"
    if not manifest_path.exists():
        raise PredictorError(f"Blind fit manifest not found: {manifest_path}")
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "frozen_blind_predictions":
        raise PredictorError("Fit manifest is not a frozen blind artifact")
    for name, expected_hash in manifest.get("artifact_files", {}).items():
        path = artifact_dir / name
        if not path.exists():
            raise PredictorError(f"Frozen artifact file is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise PredictorError(
                f"LEAKAGE GUARD: frozen artifact hash mismatch for {name}; "
                f"expected {expected_hash}, got {actual_hash}"
            )
    return manifest


def scored_table_latex(scored: pd.DataFrame) -> str:
    lines = [
        "% Auto-generated after held-out outcomes were revealed.",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        (
            r"Dataset & CLIP MMD$^2$ & Predicted (pp) & Observed (pp) "
            r"& Error (pp) \\"
        ),
        r"\midrule",
    ]
    for _, row in scored.iterrows():
        lines.append(
            f"{_latex_escape(row['dataset'])} & {row['MMD2_clip']:.4f} & "
            f"{row['predicted_degradation_pp']:.2f} & "
            f"{row['observed_degradation_pp']:.2f} & "
            f"{row['prediction_error_pp']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def score_frozen_predictions(
    *,
    artifact_dir: Path,
    holdout_outcomes_path: Path,
    out_dir: Path,
    y_column: str = "degradation_pp",
) -> dict[str, Any]:
    """Score a frozen artifact without fitting or modifying it."""
    _require_new_artifact_dir(out_dir)
    manifest = _validate_blind_artifact(artifact_dir)
    predictions_path = artifact_dir / "holdout_predictions.csv"
    predictions = pd.read_csv(predictions_path)
    predictions["dataset"] = predictions["dataset"].map(canonical_dataset)
    outcomes = load_outcomes(holdout_outcomes_path, y_column)
    expected = set(manifest["heldout_datasets"])
    actual = set(outcomes["dataset"])
    if actual != expected:
        raise PredictorError(
            "Held-out outcome datasets must exactly match frozen predictions. "
            f"Missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )

    scored = predictions.merge(
        outcomes, on="dataset", how="inner", validate="one_to_one"
    )
    scored = scored.rename(columns={"outcome": "observed_degradation_pp"})
    scored["prediction_error_pp"] = (
        scored["predicted_degradation_pp"] - scored["observed_degradation_pp"]
    )
    scored["absolute_error_pp"] = scored["prediction_error_pp"].abs()
    scored["squared_error_pp2"] = scored["prediction_error_pp"] ** 2
    scored["covered_by_prediction_interval95"] = (
        scored["observed_degradation_pp"]
        >= scored["prediction_interval95_lower_pp"]
    ) & (
        scored["observed_degradation_pp"]
        <= scored["prediction_interval95_upper_pp"]
    )
    scored["outcome_revealed"] = True
    scored = scored.sort_values("dataset").reset_index(drop=True)

    observed = scored["observed_degradation_pp"].to_numpy(dtype=float)
    predicted = scored["predicted_degradation_pp"].to_numpy(dtype=float)
    ranks = rank_metrics(observed, predicted)
    metrics = {
        "n_heldout_datasets": len(scored),
        "heldout_r2": r2_score(observed, predicted),
        "heldout_spearman_rho": ranks["spearman_rho"],
        "heldout_spearman_p_two_sided": ranks["spearman_p_two_sided"],
        "heldout_kendall_tau_b": ranks["kendall_tau_b"],
        "heldout_kendall_p_two_sided": ranks["kendall_p_two_sided"],
        "heldout_mae_pp": float(np.mean(np.abs(predicted - observed))),
        "heldout_rmse_pp": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "heldout_mean_signed_error_pred_minus_observed_pp": float(
            np.mean(predicted - observed)
        ),
        "prediction_interval95_coverage": float(
            scored["covered_by_prediction_interval95"].mean()
        ),
        "n_extrapolation_datasets": int(
            scored["outside_training_mmd_range"].sum()
        ),
        "small_holdout_warning": (
            "R2 and rank statistics over three held-out domains are highly "
            "discrete and uncertain; report domain-level errors as well."
            if len(scored) <= 3
            else None
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    output_names = ["holdout_scored.csv", "score_metrics.json", "holdout_table.tex"]
    write_csv(out_dir / output_names[0], scored)
    write_json(out_dir / output_names[1], metrics)
    write_text(out_dir / output_names[2], scored_table_latex(scored))
    score_manifest = {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "status": "heldout_outcomes_scored_without_refit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "blind_artifact_dir": str(artifact_dir.resolve()),
        "blind_manifest_sha256": sha256_file(artifact_dir / "fit_manifest.json"),
        "blind_predictions_sha256": sha256_file(predictions_path),
        "holdout_outcomes": {
            "path": str(holdout_outcomes_path.resolve()),
            "sha256": sha256_file(holdout_outcomes_path.resolve()),
        },
        "heldout_datasets": scored["dataset"].tolist(),
        "artifact_files": _artifact_hashes(out_dir, output_names),
        "refit_performed": False,
    }
    write_json(out_dir / "score_manifest.json", score_manifest)
    print(f"Scored frozen predictions: {out_dir}")
    print(
        f"Held-out R2={metrics['heldout_r2']:.3f} | "
        f"Spearman rho={metrics['heldout_spearman_rho']:.3f} | "
        f"MAE={metrics['heldout_mae_pp']:.2f} pp"
    )
    return score_manifest


def command_score(args: argparse.Namespace) -> None:
    score_frozen_predictions(
        artifact_dir=Path(args.artifact_dir).resolve(),
        holdout_outcomes_path=Path(args.holdout_outcomes).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        y_column=args.y_column,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract-matrix", help="derive the fixed paired accuracy endpoint"
    )
    extract.add_argument("--input", required=True, help="eval_matrix results CSV")
    extract.add_argument("--output", required=True)
    extract.add_argument(
        "--partition",
        choices=["train", "exploratory-train", "holdout", "custom"],
        default="custom",
    )
    extract.add_argument(
        "--datasets",
        help="comma-separated datasets; required only for partition=custom",
    )
    extract.add_argument(
        "--split-config", default=str(DEFAULT_SPLIT_CONFIG)
    )
    extract.add_argument("--vit-type", default="lora")
    extract.add_argument("--gsae-condition", default="gsae")
    extract.set_defaults(func=command_extract_matrix)

    fit = subparsers.add_parser(
        "fit", help="fit on training domains and freeze blind predictions"
    )
    fit.add_argument(
        "--distance",
        action="append",
        required=True,
        help="distance CSV; repeat to add missing cached-feature distances",
    )
    fit.add_argument("--train-outcomes", required=True)
    fit.add_argument("--out-dir", required=True)
    fit.add_argument("--split-config", default=str(DEFAULT_SPLIT_CONFIG))
    fit.add_argument(
        "--cohort",
        choices=["planned", "exploratory-current"],
        default="planned",
        help=(
            "planned requires all 15 development domains; exploratory-current "
            "uses the six currently available matrix domains and is explicitly "
            "labelled exploratory"
        ),
    )
    fit.add_argument("--x-column", default="MMD2_clip")
    fit.add_argument("--y-column", default="degradation_pp")
    fit.add_argument("--alpha", type=float, default=0.05)
    fit.set_defaults(func=command_fit)

    score = subparsers.add_parser(
        "score", help="score frozen predictions after outcome reveal"
    )
    score.add_argument("--artifact-dir", required=True)
    score.add_argument("--holdout-outcomes", required=True)
    score.add_argument("--out-dir", required=True)
    score.add_argument("--y-column", default="degradation_pp")
    score.set_defaults(func=command_score)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "alpha") and not 0 < args.alpha < 1:
        parser.error("--alpha must lie strictly between 0 and 1")
    try:
        args.func(args)
    except PredictorError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
