#!/usr/bin/env python3
"""Beyond Pearson: run a battery of association/dependence measures between each
candidate distance predictor and degradation_pp, across the 8-domain cohort.

Pearson r only detects LINEAR relationships. Controlled section 1's failure of
MMD^2(CLIP) (and every other standard distance metric, see
standard_distance_metrics_comparison.csv) was diagnosed via Pearson/OLS and
Spearman. This asks: is there a nonlinear or non-monotonic relationship that
those two miss? Measures, by what kind of dependence they can detect:

  Pearson r          - linear only
  Spearman rho        - monotonic (any shape, rank-based)
  Kendall tau-b       - monotonic (rank-based, more robust to ties/small n)
  Biweight midcorr    - linear, but down-weights outliers (robust to leverage
                        points like Oxford Pets, unlike Pearson)
  Distance correlation - ANY dependence, linear or not, monotonic or not;
                        zero (asymptotically) iff the variables are independent
  Mutual information  - ANY dependence (kNN estimator); most flexible, but
                        also the least reliable at n=8 -- reported with that
                        caveat, not as a definitive result

All measures computed for: concept_gap (the winning predictor from 1b), MMD2_clip
(the failing original predictor), and the other 5 standard distance metrics
tested in standard_distance_metrics_comparison.csv, for a complete picture of
whether ANY association measure finds signal any of them missed.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import dcor
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MMD_DIAGNOSTICS = PROJECT_ROOT / "out/controlled/mmd_predictor/retrospective_exploratory8_fit/training_diagnostics.csv"
CONCEPT_GAP_CSV = PROJECT_ROOT / "out/controlled/mmd_predictor_inputs/concept_overlap_distance.csv"
DISTANCE_RESULTS = Path("/path/to/sae-domain-shift/distance_results.csv")
OUT_DIR = PROJECT_ROOT / "out/controlled/mmd_predictor/degradation_regressor"

NAME_MAP = {
    "eurosat": "eurosat", "dtd": "dtd", "caltech-101": "caltech101",
    "pathmnist": "pathmnist", "cub200": "cub2002011", "UCF101": "ucf101",
    "oxford_pets": "oxford_pets", "fgvc": "fgvc",
}
FGVC_ROW = {"dataset": "fgvc", "N": 6667, "FID_inception": 214.7679, "MMD2_inception": 0.315600,
            "Bhat_inception": 737.2605, "FID_clip": 0.6060, "MMD2_clip": 0.233219, "Bhat_clip": 143.7987}


def biweight_midcorrelation(x: np.ndarray, y: np.ndarray, c: float = 9.0) -> float:
    """Robust correlation that down-weights outliers via Tukey's biweight (no
    astropy dependency; standard formula, e.g. Wilcox 2012 robust statistics)."""

    def biweight_scale(v):
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        if mad == 0:
            return v - med, np.zeros_like(v, dtype=bool)
        u = (v - med) / (c * mad)
        mask = np.abs(u) < 1
        return u, mask

    ux, mx = biweight_scale(x)
    uy, my = biweight_scale(y)
    mask = mx & my
    if mask.sum() < 3:
        return float("nan")
    x_med, y_med = np.median(x), np.median(y)
    wx = (1 - ux[mask] ** 2) ** 2
    wy = (1 - uy[mask] ** 2) ** 2
    num = np.sum((x[mask] - x_med) * wx * (y[mask] - y_med) * wy)
    den = np.sqrt(np.sum(((x[mask] - x_med) * wx) ** 2) * np.sum(((y[mask] - y_med) * wy) ** 2))
    return float(num / den) if den > 0 else float("nan")


def load_predictors() -> pd.DataFrame:
    deg = pd.read_csv(MMD_DIAGNOSTICS)[["dataset", "observed_degradation_pp"]]
    cg = pd.read_csv(CONCEPT_GAP_CSV)[["dataset", "concept_gap"]]

    dist = pd.read_csv(DISTANCE_RESULTS)
    dist["dataset"] = dist["dataset"].map(lambda x: NAME_MAP.get(x, x))
    dist = dist[dist["dataset"].isin(NAME_MAP.values())]
    dist = pd.concat([dist, pd.DataFrame([FGVC_ROW])], ignore_index=True)

    merged = deg.merge(cg, on="dataset").merge(dist, on="dataset")
    assert len(merged) == 8, f"expected 8 domains, got {len(merged)}"
    return merged


def evaluate(x: np.ndarray, y: np.ndarray) -> dict:
    pear = stats.pearsonr(x, y)
    spear = stats.spearmanr(x, y)
    kendall = stats.kendalltau(x, y, variant="b")
    biweight = biweight_midcorrelation(x, y)
    dcorr = dcor.distance_correlation(x, y)
    # dcor's independence permutation test (exact-ish via permutations at n=8)
    dcorr_test = dcor.independence.distance_covariance_test(x, y, num_resamples=2000, random_state=0)
    mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=0, n_neighbors=3)[0]
    return {
        "pearson_r": pear.statistic, "pearson_p": pear.pvalue,
        "spearman_rho": spear.statistic, "spearman_p": spear.pvalue,
        "kendall_tau": kendall.statistic, "kendall_p": kendall.pvalue,
        "biweight_midcorr": biweight,
        "distance_corr": dcorr, "distance_corr_p": float(dcorr_test.pvalue),
        "mutual_info": mi,
    }


def main() -> int:
    frame = load_predictors()
    y = frame["observed_degradation_pp"].to_numpy(dtype=float)
    predictors = ["concept_gap", "MMD2_clip", "FID_clip", "Bhat_clip",
                  "MMD2_inception", "FID_inception", "Bhat_inception"]

    rows = []
    for p in predictors:
        x = frame[p].to_numpy(dtype=float)
        res = evaluate(x, y)
        res["predictor"] = p
        rows.append(res)

    df = pd.DataFrame(rows).set_index("predictor")
    df = df[["pearson_r", "pearson_p", "spearman_rho", "spearman_p", "kendall_tau", "kendall_p",
             "biweight_midcorr", "distance_corr", "distance_corr_p", "mutual_info"]]

    pd.set_option("display.width", 200)
    print(f"n={len(frame)} domains: {list(frame['dataset'])}\n")
    print(df.to_string(float_format=lambda v: f"{v:+.3f}"))

    print("\nNote on mutual_info: kNN estimator (n_neighbors=3) at n=8 is very unstable")
    print("(each fold has ~5 points per neighborhood) -- reported for completeness, not")
    print("treated as a reliable signal on its own; distance_corr's permutation p-value")
    print("is the more trustworthy nonlinear-dependence test at this sample size.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "association_measures_comparison.csv")
    print(f"\nWrote {OUT_DIR}/association_measures_comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
