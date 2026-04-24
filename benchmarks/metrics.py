"""Benchmark metric helpers."""

from __future__ import annotations

import time

import numpy as np


def _dist_scale_diagnostics(dist) -> dict[str, float]:
    """Extract scale percentiles from a fitted distribution for debugging."""

    scale = None
    if hasattr(dist, "scale"):
        scale = np.asarray(dist.scale).reshape(-1)
    elif hasattr(dist, "params") and isinstance(dist.params, dict):
        scale = np.asarray(dist.params.get("scale", [])).reshape(-1)
    if scale is None or len(scale) == 0:
        return {}
    return {
        "scale_p01": float(np.percentile(scale, 1)),
        "scale_p50": float(np.percentile(scale, 50)),
        "scale_p99": float(np.percentile(scale, 99)),
        "scale_min": float(np.min(scale)),
        "scale_max": float(np.max(scale)),
    }


def compute_metrics(model, x_test, y_test) -> dict[str, float]:
    """Compute speed and predictive quality metrics for a fitted model."""

    start = time.perf_counter()
    preds = model.predict(x_test)
    predict_time = time.perf_counter() - start
    rmse = float(np.sqrt(((preds - y_test) ** 2).mean()))

    try:
        start = time.perf_counter()
        dist = model.pred_dist(x_test)
        pred_dist_time = time.perf_counter() - start
        logpdf = np.asarray(dist.logpdf(y_test)).reshape(-1)
        nll = float(-np.mean(logpdf))
    except Exception:
        nll = np.nan
        pred_dist_time = np.nan
        dist = None

    metrics = {
        "RMSE": rmse,
        "NLL": nll,
        "PredictTime": float(predict_time),
        "PredDistTime": (
            float(pred_dist_time) if np.isfinite(pred_dist_time) else np.nan
        ),
        "NLL_bad": (not np.isfinite(nll)) or (nll > 1e3),
        "n_stages": int(len(getattr(model, "base_models", []))),
        "best_val_loss_itr": (
            float(v)
            if (v := getattr(model, "best_val_loss_itr", None)) is not None
            else np.nan
        ),
    }
    if dist is not None:
        metrics.update(_dist_scale_diagnostics(dist))
        if hasattr(dist, "scale") and hasattr(dist, "loc"):
            scale = np.asarray(dist.scale).reshape(-1)
            mu = np.asarray(dist.loc).reshape(-1)
            z = (y_test - mu) / np.maximum(scale, 1e-12)
            metrics["z2_mean"] = float(np.mean(z**2))
            metrics["log_scale_mean"] = float(np.mean(np.log(np.maximum(scale, 1e-12))))
    return metrics
