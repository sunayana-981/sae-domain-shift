#!/usr/bin/env python3
"""Multivariate regressor for predicting G-SAE degradation from domain-distance features.

Gives a direct practical takeaway: instead of a single
scalar correlation (MMD^2, R^2=0.004 at n=8) or a single alternative predictor
(concept_gap alone, R^2=0.753), this fits every reasonable combination of the
distance features we have already measured for the 8-domain cohort --
MMD^2(CLIP), concept_gap, and n_target_classes (the class-cardinality confound
identified in the ImageNet baseline, section 1c) -- and selects among them by
leave-one-domain-out (LODO) R^2, not full-sample R^2, since full-sample R^2 is
guaranteed to look better with more predictors at n=8 regardless of whether
they generalize.

This is exploratory, not confirmatory: n=8, no interaction/nonlinear terms,
and the three domains reserved for held-out validation (OfficeHome, KITTI,
Cityscapes) are NOT touched here or anywhere else in this controlled. Output is
a candidate decision rule (predicted degradation vs. a floor derived from the
ImageNet self-degradation baseline), explicitly framed as a candidate, not a
validated tool.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MMD_DIAGNOSTICS = PROJECT_ROOT / "out/controlled/mmd_predictor/retrospective_exploratory8_fit/training_diagnostics.csv"
CONCEPT_GAP_CSV = PROJECT_ROOT / "out/controlled/mmd_predictor_inputs/concept_overlap_distance.csv"
OUT_DIR = PROJECT_ROOT / "out/controlled/mmd_predictor/degradation_regressor"

# Floor derived from section 1c: G-SAE costs 3.65pp even on its own training
# domain (ImageNet, no adaptation at all). Degradation predictions below this
# floor are not distinguishable from the SAE's intrinsic reconstruction cost.
DEGRADATION_FLOOR_PP = 3.65

ALL_PREDICTORS = ["mmd2", "concept_gap", "n_target_classes"]


def load_data() -> pd.DataFrame:
    mmd = pd.read_csv(MMD_DIAGNOSTICS)[["dataset", "MMD2_clip", "observed_degradation_pp"]]
    mmd = mmd.rename(columns={"MMD2_clip": "mmd2", "observed_degradation_pp": "degradation_pp"})
    cg = pd.read_csv(CONCEPT_GAP_CSV)[["dataset", "concept_gap", "n_target_classes"]]
    merged = mmd.merge(cg, on="dataset", how="inner", validate="one_to_one")
    if len(merged) != len(mmd):
        missing = set(mmd["dataset"]) - set(merged["dataset"])
        raise SystemExit(f"concept_gap/n_target_classes missing for: {missing}")
    return merged.sort_values("dataset").reset_index(drop=True)


def fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (coefficients incl. intercept, fitted values). X has no intercept column."""
    design = np.column_stack([np.ones(len(X)), X])
    coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefs
    return coefs, fitted


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst <= np.finfo(float).eps:
        return float("nan")
    return float(1.0 - np.sum((y - pred) ** 2) / sst)


def lodo_predictions(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(y)
    preds = np.empty(n, dtype=float)
    for i in range(n):
        keep = np.arange(n) != i
        coefs, _ = fit_ols(X[keep], y[keep])
        design_i = np.concatenate([[1.0], X[i]])
        preds[i] = float(design_i @ coefs)
    return preds


def evaluate_predictor_set(frame: pd.DataFrame, predictors: list[str]) -> dict:
    X = frame[predictors].to_numpy(dtype=float)
    y = frame["degradation_pp"].to_numpy(dtype=float)
    n, k = len(y), len(predictors)
    df_resid = n - k - 1
    if df_resid < 1:
        return {"predictors": predictors, "skipped": "insufficient df", "df_resid": df_resid}

    coefs, fitted = fit_ols(X, y)
    full_r2 = r2(y, fitted)
    loo_preds = lodo_predictions(X, y)
    lodo_r2 = r2(y, loo_preds)
    lodo_mae = float(np.mean(np.abs(y - loo_preds)))
    spearman = stats.spearmanr(y, loo_preds)

    return {
        "predictors": predictors,
        "n": n,
        "df_resid": df_resid,
        "intercept": float(coefs[0]),
        "coefficients": {p: float(c) for p, c in zip(predictors, coefs[1:])},
        "full_sample_r2": full_r2,
        "lodo_r2": lodo_r2,
        "lodo_mae_pp": lodo_mae,
        "lodo_spearman_rho": float(spearman.statistic),
        "lodo_spearman_p": float(spearman.pvalue),
        "loo_predictions": {d: float(p) for d, p in zip(frame["dataset"], loo_preds)},
    }


def main() -> int:
    frame = load_data()
    print(f"[INFO] {len(frame)} domains loaded: {list(frame['dataset'])}")

    candidate_sets = []
    for k in range(1, len(ALL_PREDICTORS) + 1):
        candidate_sets.extend(itertools.combinations(ALL_PREDICTORS, k))

    results = [evaluate_predictor_set(frame, list(preds)) for preds in candidate_sets]
    results = [r for r in results if "skipped" not in r]
    results.sort(key=lambda r: r["lodo_r2"], reverse=True)

    print("\n=== All candidate models, ranked by LODO R^2 (out-of-sample, not full-sample) ===")
    for r in results:
        print(
            f"  {'+'.join(r['predictors']):45s} "
            f"full_R2={r['full_sample_r2']:.3f}  LODO_R2={r['lodo_r2']:+.3f}  "
            f"LODO_MAE={r['lodo_mae_pp']:.2f}pp  LODO_rho={r['lodo_spearman_rho']:.2f}(p={r['lodo_spearman_p']:.3f})"
        )

    best = results[0]
    print(f"\n[SELECTED by LODO R^2] {'+'.join(best['predictors'])}")
    print(f"  intercept={best['intercept']:.3f}, coefficients={best['coefficients']}")

    # Decision table: LODO (out-of-sample) predicted degradation vs. actual,
    # scored against the floor from section 1c.
    decision_rows = []
    for _, row in frame.iterrows():
        pred = best["loo_predictions"][row["dataset"]]
        decision_rows.append(
            {
                "dataset": row["dataset"],
                "observed_degradation_pp": row["degradation_pp"],
                "lodo_predicted_degradation_pp": pred,
                "predicted_above_floor": pred > DEGRADATION_FLOOR_PP,
                "observed_above_floor": row["degradation_pp"] > DEGRADATION_FLOOR_PP,
                "recommendation_correct": (pred > DEGRADATION_FLOOR_PP)
                == (row["degradation_pp"] > DEGRADATION_FLOOR_PP),
            }
        )
    decision_df = pd.DataFrame(decision_rows).sort_values("observed_degradation_pp")
    n_correct = int(decision_df["recommendation_correct"].sum())
    print(f"\n=== Decision table (floor={DEGRADATION_FLOOR_PP}pp, LODO/out-of-sample predictions) ===")
    print(decision_df.to_string(index=False))
    print(f"\nLODO retrain/reuse recommendation matches observed floor-crossing in {n_correct}/{len(decision_df)} domains.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "model_comparison.json").open("w") as fh:
        json.dump(
            {"degradation_floor_pp": DEGRADATION_FLOOR_PP, "candidates": results, "selected": best},
            fh,
            indent=2,
        )
    decision_df.to_csv(OUT_DIR / "decision_table.csv", index=False)
    frame.to_csv(OUT_DIR / "input_features.csv", index=False)
    print(f"\nWrote {OUT_DIR}/model_comparison.json")
    print(f"Wrote {OUT_DIR}/decision_table.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
