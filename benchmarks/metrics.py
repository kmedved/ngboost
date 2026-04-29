"""Benchmark metric helpers."""

from __future__ import annotations

import time

import numpy as np


def _dist_scale_diagnostics(dist) -> dict[str, float]:
    """Extract scale percentiles from a fitted distribution for debugging."""

    if hasattr(dist, "cov"):
        try:
            cov = np.asarray(dist.cov)
        except Exception:
            cov = None
        if cov is not None and cov.ndim == 3 and cov.shape[1] == cov.shape[2]:
            eigvals = np.linalg.eigvalsh(cov)
            eig_min = eigvals[:, 0]
            eig_max = eigvals[:, -1]
            safe_eig_min = np.maximum(eig_min, 1e-300)
            sign, logdet = np.linalg.slogdet(cov)
            condition = eig_max / safe_eig_min
            return {
                "cov_eig_min_p01": float(np.percentile(eig_min, 1)),
                "cov_eig_min_p50": float(np.percentile(eig_min, 50)),
                "cov_eig_min_p99": float(np.percentile(eig_min, 99)),
                "cov_eig_min_min": float(np.min(eig_min)),
                "cov_eig_max_p50": float(np.percentile(eig_max, 50)),
                "cov_eig_max_max": float(np.max(eig_max)),
                "cov_condition_p50": float(np.percentile(condition, 50)),
                "cov_condition_p99": float(np.percentile(condition, 99)),
                "cov_logdet_p50": float(np.percentile(logdet, 50)),
                "cov_logdet_p99": float(np.percentile(logdet, 99)),
                "cov_nonposdef_count": float(np.sum((sign <= 0) | (eig_min <= 0))),
            }

    scale = None
    if hasattr(dist, "scale"):
        scale = np.asarray(dist.scale).reshape(-1)
    elif hasattr(dist, "params") and isinstance(dist.params, dict):
        raw_scale = np.asarray(dist.params.get("scale", []))
        if raw_scale.ndim >= 2:
            return {}
        scale = raw_scale.reshape(-1)
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
        if hasattr(dist, "cov_inv") and hasattr(dist, "loc"):
            cov_inv = np.asarray(dist.cov_inv)
            loc = np.asarray(dist.loc)
            y_arr = np.asarray(y_test)
            if cov_inv.ndim == 3 and loc.ndim == 2 and y_arr.ndim == 2:
                diff = y_arr - loc
                maha = np.einsum("ni,nij,nj->n", diff, cov_inv, diff)
                metrics["mahalanobis_mean"] = float(np.mean(maha))
                metrics["mahalanobis_p50"] = float(np.percentile(maha, 50))
                metrics["mahalanobis_p95"] = float(np.percentile(maha, 95))
    return metrics
