#!/usr/bin/env python3
"""Search over correlation/fit measures using MMD^2(CLIP) alone -- no concept_gap,
no class-cardinality feature -- to check whether the collapse in section 1a
(full R^2=0.004, LODO R^2=-1.21 for plain OLS) is an artifact of the specific
fit method (ordinary least squares is not robust to Oxford Pets' outsized
leverage: Cook's distance 1.63, ~5x the next-highest domain) rather than a
genuine absence of signal in MMD alone.

Every variant below uses ONLY MMD^2(CLIP) as the predictor (possibly
transformed) -- concept_gap is intentionally excluded, per the request to see
how far MMD alone can be pushed before concluding a second feature is needed.

Candidates:
  - OLS on MMD^2, MMD (sqrt), log1p(MMD^2)          [transform search]
  - Huber regression on MMD^2                        [outlier-robust linear]
  - Theil-Sen on MMD^2                                [outlier-robust linear]
  - RANSAC on MMD^2                                   [outlier-robust linear]
  - Isotonic regression on MMD^2                      [monotonic, rank-only]

All evaluated by the same protocol as the rest of this controlled: full-sample
R^2, and leave-one-domain-out (LODO) R^2/MAE/Spearman with the model refit
from scratch on each fold -- LODO is the number that matters, not full-sample.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MMD_DIAGNOSTICS = PROJECT_ROOT / "out/controlled/mmd_predictor/retrospective_exploratory8_fit/training_diagnostics.csv"
OUT_DIR = PROJECT_ROOT / "out/controlled/mmd_predictor/mmd_alone_variants"


def load_data() -> pd.DataFrame:
    frame = pd.read_csv(MMD_DIAGNOSTICS)[["dataset", "MMD2_clip", "observed_degradation_pp"]]
    return frame.sort_values("dataset").reset_index(drop=True)


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst <= np.finfo(float).eps:
        return float("nan")
    return float(1.0 - np.sum((y - pred) ** 2) / sst)


def ols_fit_predict(x_train, y_train, x_test):
    coefs = np.polyfit(x_train, y_train, deg=1)
    return np.polyval(coefs, x_test), coefs


def robust_fit_predict(model_cls, x_train, y_train, x_test, **kwargs):
    model = model_cls(**kwargs)
    model.fit(x_train.reshape(-1, 1), y_train)
    return float(model.predict(x_test.reshape(-1, 1))[0])


def isotonic_fit_predict(x_train, y_train, x_test):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x_train, y_train)
    return float(iso.predict(x_test)[0])


def evaluate(name: str, x: np.ndarray, y: np.ndarray, fit_predict_fn) -> dict:
    n = len(y)
    # Full-sample fit
    full_pred = np.array([fit_predict_fn(x, y, np.array([x[i]]), full=True) for i in range(n)]) \
        if False else None
    # Full sample: fit once on all data, predict all (in-sample)
    full_preds = _fit_full(name, x, y)
    full_r2 = r2(y, full_preds)

    # LODO
    loo_preds = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        loo_preds[i] = _fit_one(name, x[keep], y[keep], x[i])
    lodo_r2 = r2(y, loo_preds)
    lodo_mae = float(np.mean(np.abs(y - loo_preds)))
    spearman = stats.spearmanr(y, loo_preds)
    return {
        "name": name,
        "full_sample_r2": full_r2,
        "lodo_r2": lodo_r2,
        "lodo_mae_pp": lodo_mae,
        "lodo_spearman_rho": float(spearman.statistic),
        "lodo_spearman_p": float(spearman.pvalue),
    }


def _transform(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "mmd2":
        return x
    if kind == "mmd":
        return np.sqrt(x)
    if kind == "log1p_mmd2":
        return np.log1p(x)
    if kind == "rank":
        return stats.rankdata(x)
    raise ValueError(kind)


def _fit_full(name: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _fit_one(name, x, y, x, batch=True)


def _fit_one(name: str, x_train: np.ndarray, y_train: np.ndarray, x_test, batch: bool = False):
    x_test_arr = np.atleast_1d(x_test)
    if name.startswith("ols_"):
        pred, _ = ols_fit_predict(x_train, y_train, x_test_arr)
    elif name == "huber_mmd2":
        pred = np.array([robust_fit_predict(HuberRegressor, x_train, y_train, np.array([v])) for v in x_test_arr])
    elif name == "theilsen_mmd2":
        pred = np.array([robust_fit_predict(TheilSenRegressor, x_train, y_train, np.array([v]), random_state=0) for v in x_test_arr])
    elif name == "ransac_mmd2":
        pred = np.array([robust_fit_predict(RANSACRegressor, x_train, y_train, np.array([v]), random_state=0, min_samples=0.7) for v in x_test_arr])
    elif name == "isotonic_mmd2":
        pred = np.array([isotonic_fit_predict(x_train, y_train, np.array([v])) for v in x_test_arr])
    else:
        raise ValueError(name)
    return pred if batch else float(pred[0])


def main() -> int:
    frame = load_data()
    mmd2 = frame["MMD2_clip"].to_numpy(dtype=float)
    y = frame["observed_degradation_pp"].to_numpy(dtype=float)
    print(f"[INFO] {len(frame)} domains: {list(frame['dataset'])}")

    candidates = [
        ("ols_mmd2", _transform(mmd2, "mmd2")),
        ("ols_mmd", _transform(mmd2, "mmd")),
        ("ols_log1p_mmd2", _transform(mmd2, "log1p_mmd2")),
        ("huber_mmd2", mmd2),
        ("theilsen_mmd2", mmd2),
        ("ransac_mmd2", mmd2),
        ("isotonic_mmd2", mmd2),
    ]

    results = []
    for name, x in candidates:
        res = evaluate(name, x, y, None)
        results.append(res)

    results.sort(key=lambda r: r["lodo_r2"], reverse=True)
    print("\n=== MMD-alone variants, ranked by LODO R^2 (out-of-sample) ===")
    print(f"{'variant':20s}  full_R2   LODO_R2   LODO_MAE  LODO_rho (p)")
    for r in results:
        print(
            f"{r['name']:20s}  {r['full_sample_r2']:+.3f}    {r['lodo_r2']:+.3f}    "
            f"{r['lodo_mae_pp']:.2f}pp    {r['lodo_spearman_rho']:+.2f} ({r['lodo_spearman_p']:.3f})"
        )

    print("\n=== Reference from section 1b (not MMD-alone) ===")
    print("concept_gap alone:            full_R2=0.753  LODO_R2=0.506  LODO_MAE=3.10pp")
    print("MMD^2 + concept_gap combined:  full_R2=0.857  LODO_R2=0.621  LODO_MAE=2.43pp")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUT_DIR / "mmd_alone_variant_comparison.csv", index=False)
    print(f"\nWrote {OUT_DIR}/mmd_alone_variant_comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
