"""Metrics used in the manuscript, plus paired bootstrap model comparison."""
from __future__ import annotations

import argparse
import json
from typing import Callable, Dict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
)


def _clean(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[keep], y_pred[keep]


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true, y_pred = _clean(y_true, y_pred)
    if not len(y_true):
        return {"rmse": float("nan"), "mae": float("nan"), "pcc": float("nan")}
    pcc = pearsonr(y_true, y_pred).statistic if len(y_true) > 1 else float("nan")
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pcc": float(pcc),
    }


def resistance_metrics(y_true, y_score, threshold: float = 1.36) -> Dict[str, float]:
    y_true, y_score = _clean(y_true, y_score)
    binary = (y_true > threshold).astype(int)
    predicted = (y_score > threshold).astype(int)
    if len(np.unique(binary)) < 2:
        auprc = float("nan")
    else:
        auprc = float(average_precision_score(binary, y_score))
    return {
        "auprc": auprc,
        "mcc": float(matthews_corrcoef(binary, predicted)),
        "f1": float(f1_score(binary, predicted, zero_division=0)),
        "positive_fraction": float(binary.mean()) if len(binary) else float("nan"),
    }


def all_metrics(y_true, y_pred, task: str, threshold: float = 1.36) -> Dict[str, float]:
    result = regression_metrics(y_true, y_pred)
    if task == "ddg":
        result.update(resistance_metrics(y_true, y_pred, threshold))
    return result


def paired_bootstrap(
    y_true,
    prediction_a,
    prediction_b,
    metric: str = "rmse",
    n_bootstrap: int = 10_000,
    seed: int = 2026,
) -> Dict[str, float]:
    """Paired resampling. Positive delta means model B is better for RMSE/MAE/PCC."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    prediction_a = np.asarray(prediction_a, dtype=float).reshape(-1)
    prediction_b = np.asarray(prediction_b, dtype=float).reshape(-1)
    if not (len(y_true) == len(prediction_a) == len(prediction_b)):
        raise ValueError("Predictions must be aligned")
    keep = np.isfinite(y_true) & np.isfinite(prediction_a) & np.isfinite(prediction_b)
    y_true, prediction_a, prediction_b = y_true[keep], prediction_a[keep], prediction_b[keep]
    if len(y_true) < 2:
        raise ValueError("At least two finite paired observations are required")

    def score(name: str, truth, pred) -> float:
        values = regression_metrics(truth, pred)
        return values[name]

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=float)
    higher_is_better = metric == "pcc"
    for i in range(n_bootstrap):
        index = rng.integers(0, len(y_true), len(y_true))
        sa = score(metric, y_true[index], prediction_a[index])
        sb = score(metric, y_true[index], prediction_b[index])
        deltas[i] = (sb - sa) if higher_is_better else (sa - sb)
    finite = deltas[np.isfinite(deltas)]
    return {
        "delta": float(np.mean(finite)),
        "ci_low": float(np.quantile(finite, 0.025)),
        "ci_high": float(np.quantile(finite, 0.975)),
        "p_two_sided": float(2 * min(np.mean(finite <= 0), np.mean(finite >= 0))),
        "n_bootstrap": int(len(finite)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--truth", default="label")
    parser.add_argument("--prediction", default="prediction")
    parser.add_argument("--task", choices=("affinity", "ddg"), default="affinity")
    parser.add_argument("--threshold", type=float, default=1.36)
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    print(json.dumps(all_metrics(frame[args.truth], frame[args.prediction], args.task,
                                 args.threshold), indent=2))


if __name__ == "__main__":
    main()
