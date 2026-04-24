"""Benchmark-only estimator variants and experiment registries."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_array, check_consistent_length

from ngboost import NGBRegressor
from ngboost.distns import MultivariateNormal, Normal
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)


def _optional_import(module_name: str):
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:  # pragma: no cover - exercised via registry behavior
        return None, exc


LGB_MODULE, LIGHTGBM_IMPORT_ERROR = _optional_import("lightgbm")
XGB_MODULE, XGBOOST_IMPORT_ERROR = _optional_import("xgboost")
NUMBA_MODULE, NUMBA_IMPORT_ERROR = _optional_import("numba")


if NUMBA_MODULE is not None:
    njit = NUMBA_MODULE.njit
else:

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


@dataclass(frozen=True)
class BackendStatus:
    """Availability state for an optional benchmark backend."""

    name: str
    available: bool
    detail: str = ""


@dataclass(frozen=True)
class ExperimentSpec:
    """One benchmark experiment entry."""

    name: str
    estimator_cls: Any | None
    extra_kwargs: dict[str, Any]
    backend: str | None = None
    skip_reason: str | None = None

    @property
    def runnable(self) -> bool:
        return self.estimator_cls is not None and self.skip_reason is None


def _backend_detail(error: Exception | None) -> str:
    if error is None:
        return ""
    return f"{type(error).__name__}: {error}"


DEFAULT_BACKEND_STATUSES = {
    "lightgbm": BackendStatus(
        "lightgbm",
        LGB_MODULE is not None,
        _backend_detail(LIGHTGBM_IMPORT_ERROR),
    ),
    "xgboost": BackendStatus(
        "xgboost",
        XGB_MODULE is not None,
        _backend_detail(XGBOOST_IMPORT_ERROR),
    ),
    "numba": BackendStatus(
        "numba",
        NUMBA_MODULE is not None,
        _backend_detail(NUMBA_IMPORT_ERROR),
    ),
}


def get_backend_statuses(
    overrides: dict[str, BackendStatus] | None = None,
) -> dict[str, BackendStatus]:
    """Return optional backend availability, allowing test overrides."""

    statuses = dict(DEFAULT_BACKEND_STATUSES)
    if overrides:
        statuses.update(overrides)
    return statuses


def _make_spec(
    name: str,
    estimator_cls,
    extra_kwargs: dict[str, Any] | None = None,
    *,
    backend: str | None = None,
    backend_statuses: dict[str, BackendStatus] | None = None,
) -> ExperimentSpec:
    extra_kwargs = extra_kwargs or {}
    skip_reason = None
    if backend is not None:
        statuses = get_backend_statuses(backend_statuses)
        status = statuses[backend]
        if not status.available:
            detail = f" ({status.detail})" if status.detail else ""
            skip_reason = f"missing optional dependency '{backend}'{detail}"
        elif estimator_cls is None:
            skip_reason = f"backend '{backend}' is unavailable in this interpreter"
    return ExperimentSpec(name, estimator_cls, extra_kwargs, backend, skip_reason)


def _check_xy(x, y):
    """Permissive X/Y validation that allows multi-output y."""

    x = check_array(x, accept_sparse=True, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    check_consistent_length(x, y)
    return x, y


def _check_x_y_compat(x, y, *, multi_output: bool = False):
    """sklearn-version-safe `check_X_y` wrapper."""

    from sklearn.utils import check_X_y

    try:
        return check_X_y(
            x,
            y,
            accept_sparse=True,
            ensure_all_finite="allow-nan",
            multi_output=multi_output,
            y_numeric=True,
        )
    except TypeError:  # pragma: no cover - compatibility shim
        return check_X_y(
            x,
            y,
            accept_sparse=True,
            force_all_finite="allow-nan",
            multi_output=multi_output,
            y_numeric=True,
        )


def _resolve_validation_aliases(x_val, y_val, kwargs):
    """Accept NGBoost's public X_val/Y_val names in benchmark subclasses."""

    kwargs = dict(kwargs)
    if "X_val" in kwargs:
        alias_value = kwargs.pop("X_val")
        if x_val is not None and alias_value is not x_val:
            raise TypeError("got both x_val and X_val")
        x_val = alias_value
    if "Y_val" in kwargs:
        alias_value = kwargs.pop("Y_val")
        if y_val is not None and alias_value is not y_val:
            raise TypeError("got both y_val and Y_val")
        y_val = alias_value
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected keyword argument(s): {unexpected}")
    return x_val, y_val


def get_mvn_class(k: int):
    """Return an NGBoost multivariate normal distribution for dimension `k`."""

    return MultivariateNormal(k)


_LOG_2PI = np.log(2.0 * np.pi)


class FastNormalLogScore(LogScore):
    """Normal log-score with inline logpdf and diagonal natural gradient."""

    def score(self, y):
        z = (y - self.loc) / self.scale
        return 0.5 * _LOG_2PI + np.log(self.scale) + 0.5 * z**2

    def d_score(self, y):
        gradients = np.zeros((len(y), 2))
        gradients[:, 0] = (self.loc - y) / self.var
        gradients[:, 1] = 1 - ((self.loc - y) ** 2) / self.var
        return gradients

    def metric(self):
        fisher = np.zeros((self.var.shape[0], 2, 2))
        fisher[:, 0, 0] = 1.0 / self.var
        fisher[:, 1, 1] = 2.0
        return fisher

    def _natural_gradient(self, grad, metric):
        nat_grad = np.empty_like(grad)
        nat_grad[:, 0] = grad[:, 0] * self.var
        nat_grad[:, 1] = grad[:, 1] / 2.0
        return nat_grad


class FastNormal(RegressionDistn):
    """Normal distribution that bypasses scipy frozen objects entirely."""

    n_params = 2
    scores = [FastNormalLogScore]
    _MIN_SCALE = 1e-6
    _MAX_SCALE = 1e6

    def __init__(self, params):
        params = np.asarray(params)
        if params.ndim == 1:
            params = params.reshape(-1, 1)
        self._params = params
        self.loc = params[0]
        log_scale = np.clip(
            params[1],
            np.log(self._MIN_SCALE),
            np.log(self._MAX_SCALE),
        )
        self.scale = np.exp(log_scale)
        self.var = np.maximum(self.scale**2, 1e-12)

    @staticmethod
    def fit(y):
        mean = np.mean(y)
        scale = np.std(y)
        return np.array([mean, np.log(max(scale, 1e-8))])

    def sample(self, m):
        n = np.size(self.loc)
        return np.random.normal(loc=self.loc, scale=self.scale, size=(m, n))

    def mean(self):
        return self.loc

    def logpdf(self, y):
        z = (y - self.loc) / self.scale
        return -0.5 * _LOG_2PI - np.log(self.scale) - 0.5 * z**2

    def cdf(self, y):
        from scipy.special import ndtr

        return ndtr((y - self.loc) / self.scale)

    def predict(self):
        return self.mean()

    @property
    def params(self):
        return {"loc": self.loc, "scale": self.scale}

    def __getitem__(self, key):
        params = self._params[:, key]
        if params.ndim == 1:
            params = params.reshape(-1, 1)
        return self.__class__(params)

    def __len__(self):
        return self._params.shape[1]


class FastMultiOutputNGBRegressor(NGBRegressor):
    """Multi-output NGBoost base learner with a shared update pipeline."""

    def _predict_stage(self, models, x_cols):
        if len(models) == 1:
            pred = models[0].predict(x_cols)
            return pred.reshape(-1, 1) if pred.ndim == 1 else pred
        return np.array([model.predict(x_cols) for model in models]).T

    def _predict_latest_stage(self, x_cols):
        return self._predict_stage(self.base_models[-1], x_cols)

    def _transform_stage_pred(self, stage_pred: np.ndarray) -> np.ndarray:
        return stage_pred

    def _post_scale_transform(self, delta: np.ndarray) -> np.ndarray:
        return delta

    def _compute_delta(self, direction: np.ndarray, scale: float) -> np.ndarray:
        delta = self.learning_rate * scale * direction
        return self._post_scale_transform(delta)

    def _apply_update(self, params: np.ndarray, delta: np.ndarray, idxs=None):
        if idxs is None:
            params -= delta
        else:
            params[idxs] -= delta
        return params

    def _get_fit_base_weights(self, manifold_batch, weight_batch):
        return weight_batch

    def _prepare_and_fit_base(
        self, manifold_batch, x_batch, y_batch, grads, sample_weight
    ):
        return self.fit_base(x_batch, grads, sample_weight)

    def fit_base(self, x, grads, sample_weight=None):
        model = clone(self.Base)
        if sample_weight is None:
            model.fit(x, grads)
        else:
            model.fit(x, grads, sample_weight=sample_weight)
        fitted = self._predict_stage([model], x)
        self.base_models.append([model])
        return fitted

    def _log_iteration_diagnostics(self, itr, params, loss, scale):
        if not getattr(self, "_collect_diagnostics", False):
            return
        if not hasattr(self, "diagnostics_"):
            self.diagnostics_ = []
        log_scale = params[:, 1] if params.shape[1] >= 2 else None
        row = {"itr": itr, "loss": float(loss), "ls_scale": float(scale)}
        if log_scale is not None:
            row.update(
                {
                    "log_scale_p01": float(np.percentile(log_scale, 1)),
                    "log_scale_p05": float(np.percentile(log_scale, 5)),
                    "log_scale_p50": float(np.percentile(log_scale, 50)),
                    "log_scale_p95": float(np.percentile(log_scale, 95)),
                    "log_scale_p99": float(np.percentile(log_scale, 99)),
                    "scale_min": float(np.exp(np.clip(log_scale.min(), -20, 20))),
                    "scale_max": float(np.exp(np.clip(log_scale.max(), -20, 20))),
                }
            )
        self.diagnostics_.append(row)

    def fit(
        self,
        x,
        y,
        x_val=None,
        y_val=None,
        sample_weight=None,
        val_sample_weight=None,
        train_loss_monitor=None,
        val_loss_monitor=None,
        early_stopping_rounds=None,
        **kwargs,
    ):
        x_val, y_val = _resolve_validation_aliases(x_val, y_val, kwargs)
        self.base_models = []
        self.scalings = []
        self.col_idxs = []
        if hasattr(self, "best_val_loss_itr"):
            delattr(self, "best_val_loss_itr")
        return self.partial_fit(
            x,
            y,
            x_val=x_val,
            y_val=y_val,
            sample_weight=sample_weight,
            val_sample_weight=val_sample_weight,
            train_loss_monitor=train_loss_monitor,
            val_loss_monitor=val_loss_monitor,
            early_stopping_rounds=early_stopping_rounds,
        )

    def pred_param(self, x, max_iter=None):
        m, _ = x.shape
        params = np.ones((m, self.Manifold.n_params)) * self.init_params
        for i, (models, scale, col_idx) in enumerate(
            zip(self.base_models, self.scalings, self.col_idxs)
        ):
            if max_iter is not None and i >= max_iter:
                break
            stage_pred = self._predict_stage(models, x[:, col_idx])
            direction = self._transform_stage_pred(stage_pred)
            delta = self._compute_delta(direction, scale)
            self._apply_update(params, delta)
        return params

    def staged_pred_dist(self, x, max_iter=None):
        predictions = []
        m, _ = x.shape
        params = np.ones((m, self.Dist.n_params)) * self.init_params
        for i, (models, scale, col_idx) in enumerate(
            zip(self.base_models, self.scalings, self.col_idxs),
            start=1,
        ):
            stage_pred = self._predict_stage(models, x[:, col_idx])
            direction = self._transform_stage_pred(stage_pred)
            delta = self._compute_delta(direction, scale)
            self._apply_update(params, delta)
            predictions.append(self.Dist(np.copy(params.T)))
            if max_iter is not None and i >= max_iter:
                break
        return predictions

    def partial_fit(
        self,
        x,
        y,
        x_val=None,
        y_val=None,
        sample_weight=None,
        val_sample_weight=None,
        train_loss_monitor=None,
        val_loss_monitor=None,
        early_stopping_rounds=None,
        **kwargs,
    ):
        x_val, y_val = _resolve_validation_aliases(x_val, y_val, kwargs)
        if len(self.base_models) != len(self.scalings) or len(self.base_models) != len(
            self.col_idxs
        ):
            raise RuntimeError(
                "Base models, scalings, and col_idxs are not the same length"
            )

        if early_stopping_rounds is None:
            early_stopping_rounds = self.early_stopping_rounds
        if early_stopping_rounds is not None:
            if x_val is None and y_val is None:
                if sample_weight is None:
                    x, x_val, y, y_val = train_test_split(
                        x,
                        y,
                        test_size=self.validation_fraction,
                        random_state=self.random_state,
                    )
                else:
                    (
                        x,
                        x_val,
                        y,
                        y_val,
                        sample_weight,
                        val_sample_weight,
                    ) = train_test_split(
                        x,
                        y,
                        sample_weight,
                        test_size=self.validation_fraction,
                        random_state=self.random_state,
                    )
            elif x_val is not None and y_val is not None:
                if sample_weight is not None and val_sample_weight is None:
                    raise ValueError(
                        "Training data has sample weights but validation data is missing them"
                    )
                if sample_weight is None and val_sample_weight is not None:
                    raise ValueError(
                        "sample weights mismatch between training and validation data"
                    )

        if y is None:
            raise ValueError("y cannot be None")

        x, y = _check_x_y_compat(x, y, multi_output=self.multi_output)
        self.n_features = x.shape[1]

        loss_list = []
        if len(self.base_models) == 0:
            self.fit_init_params_to_marginal(y)
        params = self.pred_param(x)

        if x_val is not None and y_val is not None:
            x_val, y_val = _check_x_y_compat(
                x_val,
                y_val,
                multi_output=self.multi_output,
            )
            val_params = self.pred_param(x_val)
            val_loss_list = []
            best_val_loss = np.inf

        if not train_loss_monitor:
            train_loss_monitor = (
                lambda manifold_batch, y_batch, weights: manifold_batch.total_score(
                    y_batch, sample_weight=weights
                )
            )
        if not val_loss_monitor:
            val_loss_monitor = (
                lambda manifold_batch, y_batch: manifold_batch.total_score(
                    y_batch, sample_weight=val_sample_weight
                )
            )

        start_itr = len(self.col_idxs)
        end_itr = self.n_estimators + len(self.col_idxs)
        for itr in range(start_itr, end_itr):
            idxs, col_idx, x_batch, y_batch, weight_batch, params_batch = self.sample(
                x, y, sample_weight, params
            )
            self.col_idxs.append(col_idx)

            manifold_batch = self.Manifold(params_batch.T)
            loss_list.append(train_loss_monitor(manifold_batch, y_batch, weight_batch))
            loss = loss_list[-1]
            grads = manifold_batch.grad(y_batch, natural=self.natural_gradient)

            base_weights = self._get_fit_base_weights(manifold_batch, weight_batch)
            proj_grad = self._prepare_and_fit_base(
                manifold_batch,
                x_batch,
                y_batch,
                grads,
                base_weights,
            )

            direction_batch = self._transform_stage_pred(proj_grad)
            scale = self.line_search(
                direction_batch, params_batch, y_batch, weight_batch
            )

            if idxs is None and x_batch.shape[0] == x.shape[0]:
                delta = self._compute_delta(direction_batch, scale)
                self._apply_update(params, delta)
            elif idxs is not None and len(idxs) == x.shape[0]:
                delta = self._compute_delta(direction_batch, scale)
                self._apply_update(params, delta, idxs)
            else:
                stage_pred_full = self._predict_latest_stage(x[:, col_idx])
                direction_full = self._transform_stage_pred(stage_pred_full)
                full_delta = self._compute_delta(direction_full, scale)
                self._apply_update(params, full_delta)

            self._log_iteration_diagnostics(itr, params, loss, scale)

            val_loss = 0.0
            if x_val is not None and y_val is not None:
                stage_pred_val = self._predict_latest_stage(x_val[:, col_idx])
                direction_val = self._transform_stage_pred(stage_pred_val)
                val_delta = self._compute_delta(direction_val, scale)
                self._apply_update(val_params, val_delta)

                val_loss = val_loss_monitor(self.Manifold(val_params.T), y_val)
                val_loss_list.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss, self.best_val_loss_itr = val_loss, itr

                if (
                    early_stopping_rounds is not None
                    and len(val_loss_list) > early_stopping_rounds
                    and best_val_loss
                    < np.min(np.array(val_loss_list[-early_stopping_rounds:]))
                ):
                    if self.verbose:
                        print("== Early stopping achieved.")
                        print(
                            "== Best iteration / VAL"
                            f"{self.best_val_loss_itr} (val_loss={best_val_loss:.4f})"
                        )
                    break

            if (
                self.verbose
                and int(self.verbose_eval) > 0
                and itr % int(self.verbose_eval) == 0
            ):
                grad_norm = np.linalg.norm(grads, axis=1).mean() * scale
                print(
                    f"[iter {itr}] loss={loss:.4f} val_loss={val_loss:.4f} "
                    f"scale={scale:.4f} norm={grad_norm:.4f}"
                )

            if np.linalg.norm(proj_grad, axis=1).mean() < self.tol:
                if self.verbose:
                    print(f"== Quitting at iteration / GRAD {itr}")
                break

        self.evals_result = {}
        metric = self.Score.__name__.upper()
        self.evals_result["train"] = {metric: loss_list}
        if x_val is not None and y_val is not None:
            self.evals_result["val"] = {metric: val_loss_list}
        return self

    @property
    def feature_importances_(self):
        from sklearn.tree import DecisionTreeRegressor

        if not self.base_models:
            return None
        model0 = self.base_models[0][0]
        if not isinstance(model0, DecisionTreeRegressor):
            return None

        importances = []
        for models, col_idx in zip(self.base_models, self.col_idxs):
            stage_importance = np.zeros(self.n_features)
            stage_importance[col_idx] = models[0].feature_importances_
            importances.append(stage_importance)

        average = np.average(importances, axis=0, weights=self.scalings)
        total = average.sum()
        if total > 0:
            average /= total
        return np.tile(average, (self.Manifold.n_params, 1))


def _clone_manifold_cls(manifold_cls, suffix: str):
    return type(f"{manifold_cls.__name__}_{suffix}", (manifold_cls,), {})


def make_diag_ng_manifold(manifold_cls, eps: float = 1e-8):
    new_cls = _clone_manifold_cls(manifold_cls, "DiagNG")

    def _natural_gradient(self_manifold, grad, metric):
        metric_diag = np.einsum("sii->si", metric) + eps
        return grad / metric_diag

    new_cls._natural_gradient = _natural_gradient
    return new_cls


def make_global_ng_manifold(manifold_cls, ridge: float = 1e-2):
    new_cls = _clone_manifold_cls(manifold_cls, "GlobalNG")

    def _natural_gradient(self_manifold, grad, metric):
        param_count = metric.shape[1]
        avg_metric = metric.mean(axis=0) + np.eye(param_count) * ridge
        try:
            return np.linalg.solve(avg_metric, grad.T).T
        except np.linalg.LinAlgError:
            return (np.linalg.pinv(avg_metric) @ grad.T).T

    new_cls._natural_gradient = _natural_gradient
    return new_cls


class DiagonalNGBRegressor(NGBRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Manifold = make_diag_ng_manifold(self.Manifold)


class GlobalNGBRegressor(NGBRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Manifold = make_global_ng_manifold(self.Manifold, ridge=1e-2)


class FastMO_DiagNGBRegressor(FastMultiOutputNGBRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Manifold = make_diag_ng_manifold(self.Manifold)


class FastMO_GlobalNGBRegressor(FastMultiOutputNGBRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Manifold = make_global_ng_manifold(self.Manifold, ridge=1e-2)


def _fixed_line_search(self, resids, start, y, sample_weight=None, scale_init=1.0):
    self.scalings.append(scale_init)
    return scale_init


class FixedStepNGBRegressor(NGBRegressor):
    line_search = _fixed_line_search


class FastMO_FixedStepNGBRegressor(FastMultiOutputNGBRegressor):
    line_search = _fixed_line_search


class FastMO_DiagNG_FixedStepNGBRegressor(FastMO_DiagNGBRegressor):
    line_search = _fixed_line_search


def _subsample_ls_arrays(
    resids, start, y, sample_weight, n_scalings, frac=0.05, max_n=1000
):
    n_rows = resids.shape[0]
    sub_n = min(max_n, max(200, int(n_rows * frac)))
    if sub_n >= n_rows:
        return resids, start, y, sample_weight
    rng = np.random.RandomState(GLOBAL_SEED + n_scalings)
    idx = rng.choice(n_rows, sub_n, replace=False)
    sub_weights = sample_weight[idx] if sample_weight is not None else None
    return resids[idx], start[idx], y[idx], sub_weights


def _verify_scale_on_full_data(self, resids, start, y, sample_weight, scale):
    loss0 = self.Manifold(start.T).total_score(y, sample_weight)
    loss1 = self.Manifold((start - resids * scale).T).total_score(y, sample_weight)

    if (not np.isfinite(loss1)) or (loss1 > loss0):
        for _ in range(10):
            scale *= 0.5
            loss1 = self.Manifold((start - resids * scale).T).total_score(
                y,
                sample_weight,
            )
            if np.isfinite(loss1) and loss1 <= loss0:
                break
        self.scalings[-1] = scale
    return self.scalings[-1]


def _line_search_core(
    self,
    resids,
    start,
    y,
    sample_weight,
    scale,
    *,
    max_up=256,
    max_down=256,
):
    loss_init = self.Manifold(start.T).total_score(y, sample_weight)

    for _ in range(max_up):
        scaled = resids * scale
        loss = self.Manifold((start - scaled).T).total_score(y, sample_weight)
        if (not np.isfinite(loss)) or (loss > loss_init) or (scale > 256):
            break
        scale *= 2

    for _ in range(max_down):
        scaled = resids * scale
        loss = self.Manifold((start - scaled).T).total_score(y, sample_weight)
        if np.mean(np.linalg.norm(scaled, axis=1)) < self.tol:
            break
        if np.isfinite(loss) and loss < loss_init:
            break
        scale *= 0.5

    scale = min(scale, 256.0)
    self.scalings.append(scale)
    return scale


def _subsampled_line_search(self, resids, start, y, sample_weight=None, scale_init=1):
    frac = getattr(self, "line_search_frac", 0.05)
    max_n = getattr(self, "line_search_max_n", 1000)
    res_sub, start_sub, y_sub, sw_sub = _subsample_ls_arrays(
        resids,
        start,
        y,
        sample_weight,
        len(self.scalings),
        frac,
        max_n,
    )
    _line_search_core(self, res_sub, start_sub, y_sub, sw_sub, scale_init)
    return _verify_scale_on_full_data(
        self,
        resids,
        start,
        y,
        sample_weight,
        self.scalings[-1],
    )


class SubsampledLineSearchNGBRegressor(NGBRegressor):
    line_search = _subsampled_line_search
    line_search_frac = 0.05
    line_search_max_n = 1000


class FastMO_SubsampledLSNGBRegressor(FastMultiOutputNGBRegressor):
    line_search = _subsampled_line_search
    line_search_frac = 0.05
    line_search_max_n = 1000


class FastMO_GradNormNGBRegressor(FastMultiOutputNGBRegressor):
    """Multi-output + gradient normalization to balance split selection."""

    def __init__(
        self,
        *args,
        grad_norm_mode="mad",
        grad_norm_alpha=0.5,
        grad_norm_max_ratio=10.0,
        grad_norm_freeze_after=None,
        grad_norm_anchor="median",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.grad_norm_mode = grad_norm_mode
        self.grad_norm_alpha = grad_norm_alpha
        self.grad_norm_max_ratio = grad_norm_max_ratio
        self.grad_norm_freeze_after = grad_norm_freeze_after
        self.grad_norm_anchor = grad_norm_anchor

    def _compute_grad_scales(self, grads: np.ndarray) -> np.ndarray:
        eps = 1e-8

        if self.grad_norm_mode == "mad":
            med = np.median(grads, axis=0)
            raw = np.median(np.abs(grads - med), axis=0) + eps
        elif self.grad_norm_mode == "std":
            raw = np.std(grads, axis=0) + eps
        else:
            raw = np.ones(grads.shape[1], dtype=np.float64)

        scales = raw ** float(self.grad_norm_alpha)
        scales = np.maximum(scales, eps)

        if self.grad_norm_max_ratio and len(scales) > 1:
            ratio = float(self.grad_norm_max_ratio)
            if self.grad_norm_anchor == "max":
                anchor = float(scales.max())
                lo = anchor / ratio
                hi = anchor
            else:
                anchor = float(np.median(scales))
                lo = anchor / ratio
                hi = anchor * ratio
            scales = np.clip(scales, max(lo, eps), hi)

        return scales

    def fit_base(self, x, grads, sample_weight=None):
        n_stages = len(self.base_models)
        if (
            self.grad_norm_freeze_after is not None
            and n_stages >= self.grad_norm_freeze_after
            and hasattr(self, "_grad_scales")
            and len(self._grad_scales) > 0
        ):
            scales = self._grad_scales[-1]
        else:
            scales = self._compute_grad_scales(grads)

        grads_normed = grads / scales

        model = clone(self.Base)
        if sample_weight is None:
            model.fit(x, grads_normed)
        else:
            model.fit(x, grads_normed, sample_weight=sample_weight)

        fitted_normed = self._predict_stage([model], x)
        fitted = fitted_normed * scales

        self.base_models.append([model])
        if not hasattr(self, "_grad_scales"):
            self._grad_scales = []
        self._grad_scales.append(scales)
        return fitted

    def _predict_stage_unscaled(self, models, x_cols, grad_scales):
        return self._predict_stage(models, x_cols) * grad_scales

    def pred_param(self, x, max_iter=None):
        m, _ = x.shape
        params = np.ones((m, self.Manifold.n_params)) * self.init_params
        scales_list = getattr(
            self,
            "_grad_scales",
            [np.ones(self.Manifold.n_params)] * len(self.base_models),
        )
        for i, (models, scale, col_idx, grad_scales) in enumerate(
            zip(self.base_models, self.scalings, self.col_idxs, scales_list)
        ):
            if max_iter is not None and i >= max_iter:
                break
            stage_pred = self._predict_stage_unscaled(
                models, x[:, col_idx], grad_scales
            )
            direction = self._transform_stage_pred(stage_pred)
            delta = self._compute_delta(direction, scale)
            self._apply_update(params, delta)
        return params

    def staged_pred_dist(self, x, max_iter=None):
        predictions = []
        m, _ = x.shape
        params = np.ones((m, self.Dist.n_params)) * self.init_params
        scales_list = getattr(
            self,
            "_grad_scales",
            [np.ones(self.Manifold.n_params)] * len(self.base_models),
        )
        for i, (models, scale, col_idx, grad_scales) in enumerate(
            zip(self.base_models, self.scalings, self.col_idxs, scales_list),
            start=1,
        ):
            stage_pred = self._predict_stage_unscaled(
                models, x[:, col_idx], grad_scales
            )
            direction = self._transform_stage_pred(stage_pred)
            delta = self._compute_delta(direction, scale)
            self._apply_update(params, delta)
            predictions.append(self.Dist(np.copy(params.T)))
            if max_iter is not None and i >= max_iter:
                break
        return predictions

    def _predict_latest_stage(self, x_cols):
        return self._predict_stage_unscaled(
            self.base_models[-1],
            x_cols,
            self._grad_scales[-1],
        )

    def partial_fit(self, x, y, x_val=None, y_val=None, **kwargs):
        if not hasattr(self, "_grad_scales"):
            self._grad_scales = []
        return super().partial_fit(x, y, x_val=x_val, y_val=y_val, **kwargs)

    def fit(
        self,
        x,
        y,
        x_val=None,
        y_val=None,
        sample_weight=None,
        val_sample_weight=None,
        **kwargs,
    ):
        self._grad_scales = []
        return super().fit(
            x,
            y,
            x_val=x_val,
            y_val=y_val,
            sample_weight=sample_weight,
            val_sample_weight=val_sample_weight,
            **kwargs,
        )


class FastMO_GradNorm_FixedStepNGBRegressor(FastMO_GradNormNGBRegressor):
    line_search = _fixed_line_search


class NoCopyNGBRegressor(NGBRegressor):
    def sample(self, x, y, sample_weight, params):
        if self.minibatch_frac == 1.0 and self.col_sample == 1.0:
            return (
                None,
                np.arange(x.shape[1]),
                x,
                y,
                sample_weight,
                params,
            )
        return super().sample(x, y, sample_weight, params)


class NoCopy_FixedStepNGBRegressor(NoCopyNGBRegressor):
    line_search = _fixed_line_search


def _capped_line_search(self, resids, start, y, sample_weight=None, scale_init=1):
    return _line_search_core(
        self,
        resids,
        start,
        y,
        sample_weight,
        scale_init,
        max_up=getattr(self, "_ls_max_up", 2),
        max_down=getattr(self, "_ls_max_down", 3),
    )


class CappedLSNGBRegressor(NGBRegressor):
    line_search = _capped_line_search
    _ls_max_up = 2
    _ls_max_down = 3


class FastMO_CappedLSNGBRegressor(FastMultiOutputNGBRegressor):
    line_search = _capped_line_search
    _ls_max_up = 2
    _ls_max_down = 3


class FastMO_GradNorm_CappedLSNGBRegressor(FastMO_GradNormNGBRegressor):
    line_search = _capped_line_search
    _ls_max_up = 2
    _ls_max_down = 3


class NoCopy_CappedLSNGBRegressor(NoCopyNGBRegressor):
    line_search = _capped_line_search
    _ls_max_up = 2
    _ls_max_down = 3


def _hybrid_line_search(self, resids, start, y, sample_weight=None, scale_init=1):
    frac = getattr(self, "_ls_frac", 0.05)
    max_n = getattr(self, "_ls_max_n", 1000)
    res_sub, start_sub, y_sub, sw_sub = _subsample_ls_arrays(
        resids,
        start,
        y,
        sample_weight,
        len(self.scalings),
        frac,
        max_n,
    )
    _line_search_core(
        self,
        res_sub,
        start_sub,
        y_sub,
        sw_sub,
        scale_init,
        max_up=getattr(self, "_ls_max_up", 2),
        max_down=getattr(self, "_ls_max_down", 3),
    )
    return _verify_scale_on_full_data(
        self,
        resids,
        start,
        y,
        sample_weight,
        self.scalings[-1],
    )


class HybridLSNGBRegressor(NGBRegressor):
    line_search = _hybrid_line_search
    _ls_max_up = 2
    _ls_max_down = 3
    _ls_frac = 0.05
    _ls_max_n = 1000


class FastMO_HybridLSNGBRegressor(FastMultiOutputNGBRegressor):
    line_search = _hybrid_line_search
    _ls_max_up = 2
    _ls_max_down = 3
    _ls_frac = 0.05
    _ls_max_n = 1000


class FastMO_GradNorm_HybridLSNGBRegressor(FastMO_GradNormNGBRegressor):
    line_search = _hybrid_line_search
    _ls_max_up = 2
    _ls_max_down = 3
    _ls_frac = 0.05
    _ls_max_n = 1000


class NewtonLeavesNGBRegressor(FastMultiOutputNGBRegressor):
    """Per-parameter Newton leaves with one tree per parameter."""

    def _prepare_and_fit_base(
        self, manifold_batch, x_batch, y_batch, grads, sample_weight
    ):
        vanilla_grads = manifold_batch.grad(y_batch, natural=False)
        hess_diag = np.einsum("sii->si", manifold_batch.metric())

        models = []
        preds = np.zeros_like(grads)
        for j in range(grads.shape[1]):
            hessian = hess_diag[:, j] + 1e-8
            target = vanilla_grads[:, j] / hessian
            weights = hessian if sample_weight is None else hessian * sample_weight
            model = clone(self.Base)
            model.fit(x_batch, target, sample_weight=weights)
            models.append(model)
            preds[:, j] = model.predict(x_batch)
        self.base_models.append(models)
        return preds


class NewtonLeaves_FixedStepNGBRegressor(NewtonLeavesNGBRegressor):
    line_search = _fixed_line_search


class FastMO_FisherHeuristicNGBRegressor(FastMultiOutputNGBRegressor):
    """Multi-output + averaged Fisher diagonal as heuristic sample weights."""

    def _get_fit_base_weights(self, manifold_batch, weight_batch):
        fisher_diag = np.einsum("sii->si", manifold_batch.metric())
        weights = fisher_diag.mean(axis=1) + 1e-6
        if weight_batch is not None:
            weights *= weight_batch
        return weights


class FastMO_FisherHeuristic_FixedStepNGBRegressor(FastMO_FisherHeuristicNGBRegressor):
    line_search = _fixed_line_search


def _warmstart_line_search(self, resids, start, y, sample_weight=None, scale_init=1):
    if hasattr(self, "scalings") and len(self.scalings) > 0:
        scale_init = min(256.0, float(self.scalings[-1]) * 2.0)
    return _line_search_core(self, resids, start, y, sample_weight, scale_init)


class WarmStartLSNGBRegressor(NGBRegressor):
    line_search = _warmstart_line_search


class FastMO_WarmStartLSNGBRegressor(FastMultiOutputNGBRegressor):
    line_search = _warmstart_line_search


def _armijo_line_search(self, resids, start, y, sample_weight=None, scale_init=1):
    armijo_tol = getattr(self, "_armijo_tol", 1e-4)
    max_down = getattr(self, "_armijo_max_down", 5)
    max_up = getattr(self, "_armijo_max_up", 4)

    loss_init = self.Manifold(start.T).total_score(y, sample_weight)

    scale = float(scale_init)
    best_scale = scale
    best_loss = loss_init
    loss = self.Manifold((start - resids * scale).T).total_score(y, sample_weight)

    if np.isfinite(loss) and (loss < loss_init - armijo_tol):
        best_loss = loss
        best_scale = scale
        for _ in range(max_up):
            scale_up = best_scale * 2.0
            if scale_up > 256.0:
                break
            loss_up = self.Manifold((start - resids * scale_up).T).total_score(
                y,
                sample_weight,
            )
            if (not np.isfinite(loss_up)) or (loss_up > best_loss):
                break
            best_loss = loss_up
            best_scale = scale_up

        self.scalings.append(best_scale)
        return best_scale

    scale = float(scale_init)
    for _ in range(max_down):
        scale *= 0.5
        scaled = resids * scale
        loss = self.Manifold((start - scaled).T).total_score(y, sample_weight)
        if np.mean(np.linalg.norm(scaled, axis=1)) < self.tol:
            break
        if np.isfinite(loss) and loss < loss_init:
            break

    self.scalings.append(scale)
    return scale


class ArmijoLSNGBRegressor(NGBRegressor):
    line_search = _armijo_line_search
    _armijo_tol = 1e-4
    _armijo_max_down = 5
    _armijo_max_up = 4


class FastMO_ArmijoLSNGBRegressor(FastMultiOutputNGBRegressor):
    line_search = _armijo_line_search
    _armijo_tol = 1e-4
    _armijo_max_down = 5
    _armijo_max_up = 4


class FastMO_DampedScaleNGBRegressor(FastMultiOutputNGBRegressor):
    """MO with damped non-mean parameter updates."""

    def __init__(self, *args, lr_scale_factor=0.5, log_scale_clip=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_scale_factor = float(lr_scale_factor)
        self.log_scale_clip = float(log_scale_clip)

    def _mean_dims(self) -> int:
        if getattr(self.Dist, "multi_output", False):
            n_params = int(self.Dist.n_params)
            k = int(round((-3.0 + math.sqrt(9.0 + 8.0 * n_params)) / 2.0))
            if k > 0 and (k * (k + 3)) // 2 == n_params:
                return k
        return 1

    def _transform_stage_pred(self, stage_pred: np.ndarray) -> np.ndarray:
        mean_dims = self._mean_dims()
        if stage_pred.ndim == 2 and stage_pred.shape[1] > mean_dims:
            out = stage_pred.copy()
            out[:, mean_dims:] *= self.lr_scale_factor
            return out
        return stage_pred

    def _post_scale_transform(self, delta: np.ndarray) -> np.ndarray:
        mean_dims = self._mean_dims()
        if delta.ndim == 2 and delta.shape[1] > mean_dims and self.log_scale_clip > 0:
            out = delta.copy()
            out[:, mean_dims:] = np.clip(
                out[:, mean_dims:],
                -self.log_scale_clip,
                self.log_scale_clip,
            )
            return out
        return delta


class FastMO_DampedScale_CappedLSNGBRegressor(FastMO_DampedScaleNGBRegressor):
    line_search = _capped_line_search
    _ls_max_up = 2
    _ls_max_down = 3


class FastMO_DampedScale_ArmijoLSNGBRegressor(FastMO_DampedScaleNGBRegressor):
    line_search = _armijo_line_search
    _armijo_tol = 1e-4
    _armijo_max_down = 5
    _armijo_max_up = 4


class FastMO_DampedScale_FixedStepNGBRegressor(FastMO_DampedScaleNGBRegressor):
    line_search = _fixed_line_search


class ParallelFitBaseNGBRegressor(NGBRegressor):
    """Parallel per-parameter tree fitting using joblib."""

    def __init__(self, *args, n_jobs_fit=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_jobs_fit = n_jobs_fit

    def fit_base(self, x, grads, sample_weight=None):
        def _fit_one(grad_column):
            model = clone(self.Base)
            if sample_weight is None:
                model.fit(x, grad_column)
            else:
                model.fit(x, grad_column, sample_weight=sample_weight)
            return model

        models = Parallel(n_jobs=self.n_jobs_fit, prefer="threads")(
            delayed(_fit_one)(grads[:, j]) for j in range(grads.shape[1])
        )
        fitted = np.array([model.predict(x) for model in models]).T
        self.base_models.append(models)
        return fitted


if XGB_MODULE is not None:

    class XGBoostMultiOutputEstimator(BaseEstimator, RegressorMixin):
        """XGBoost hist tree with `multi_output_tree` strategy."""

        def __init__(self, max_depth=3, learning_rate=1.0, n_jobs=4):
            self.max_depth = max_depth
            self.learning_rate = learning_rate
            self.n_jobs = n_jobs
            self.model = None

        def fit(self, x, y, sample_weight=None):
            dtrain = XGB_MODULE.DMatrix(x, label=y, weight=sample_weight)
            params = {
                "tree_method": "hist",
                "multi_strategy": "multi_output_tree",
                "objective": "reg:squarederror",
                "max_depth": self.max_depth,
                "eta": self.learning_rate,
                "n_jobs": self.n_jobs,
                "verbosity": 0,
                "seed": GLOBAL_SEED,
            }
            self.model = XGB_MODULE.train(params, dtrain, num_boost_round=1)
            return self

        def predict(self, x):
            return self.model.predict(XGB_MODULE.DMatrix(x))

else:  # pragma: no cover - registry uses skip behavior instead
    XGBoostMultiOutputEstimator = None


if LGB_MODULE is not None:

    class LightGBMPersistentNGBRegressor(NGBRegressor):
        """LightGBM hist trees that reuse a binned Dataset across iterations."""

        def fit(self, x, y, **kwargs):
            x, y = _check_xy(x, y)
            self.base_models = []
            self.scalings = []
            self.col_idxs = []
            self.fit_init_params_to_marginal(y)
            params = self.pred_param(x)

            self._train_ds = LGB_MODULE.Dataset(
                x,
                label=y if y.ndim == 1 else y[:, 0],
                free_raw_data=False,
                params={"verbose": -1},
            )

            lgb_params = {
                "objective": "regression",
                "verbosity": -1,
                "learning_rate": 1.0,
                "max_depth": 3,
                "num_leaves": 31,
                "min_data_in_leaf": 1,
                "bagging_fraction": 1.0,
                "feature_fraction": 1.0,
                "lambda_l1": 0.0,
                "lambda_l2": 0.0,
                "num_threads": 4,
                "seed": GLOBAL_SEED,
            }

            for _ in range(self.n_estimators):
                manifold_batch = self.Manifold(params.T)
                grads = manifold_batch.grad(y, natural=self.natural_gradient)

                layer_models = []
                layer_preds = np.zeros_like(grads)
                for k in range(grads.shape[1]):
                    self._train_ds.set_label(grads[:, k])
                    booster = LGB_MODULE.train(
                        lgb_params, self._train_ds, num_boost_round=1
                    )
                    layer_models.append(booster)
                    layer_preds[:, k] = booster.predict(x)

                self.base_models.append(layer_models)
                self.col_idxs.append(np.arange(x.shape[1]))
                scale = self.line_search(layer_preds, params, y)
                params -= self.learning_rate * scale * layer_preds
            return self

        def pred_param(self, x, max_iter=None):
            m = x.shape[0]
            params = np.ones((m, self.Manifold.n_params)) * self.init_params
            for i, (models, scale, _) in enumerate(
                zip(self.base_models, self.scalings, self.col_idxs)
            ):
                if max_iter is not None and i >= max_iter:
                    break
                resids = np.column_stack([model.predict(x) for model in models])
                params -= self.learning_rate * resids * scale
            return params

        def staged_pred_dist(self, x, max_iter=None):
            raise NotImplementedError(
                "LightGBMPersistentNGBRegressor does not support staged_pred_dist"
            )

    class LightGBM_FixedStepNGBRegressor(LightGBMPersistentNGBRegressor):
        line_search = _fixed_line_search

    class LightGBM_CappedLSNGBRegressor(LightGBMPersistentNGBRegressor):
        line_search = _capped_line_search
        _ls_max_up = 2
        _ls_max_down = 3

else:  # pragma: no cover - registry uses skip behavior instead
    LightGBMPersistentNGBRegressor = None
    LightGBM_FixedStepNGBRegressor = None
    LightGBM_CappedLSNGBRegressor = None


@njit(fastmath=True)
def _hist_split_kernel(
    x_binned_node,
    gradients_node,
    n_bins,
    n_features,
    n_outputs,
    reg_lambda,
):
    best_gain = -1.0
    best_feature = -1
    best_threshold = -1
    n_node = x_binned_node.shape[0]

    sum_g = np.zeros(n_outputs)
    for i in range(n_node):
        sum_g += gradients_node[i]

    for feature in range(n_features):
        hist_sum_g = np.zeros((n_bins, n_outputs))
        hist_cnt = np.zeros(n_bins)
        col_data = x_binned_node[:, feature]
        for i in range(n_node):
            bin_idx = col_data[i]
            hist_cnt[bin_idx] += 1
            hist_sum_g[bin_idx] += gradients_node[i]

        current_sum_g = np.zeros(n_outputs)
        current_cnt = 0.0
        for bin_idx in range(n_bins - 1):
            current_sum_g += hist_sum_g[bin_idx]
            current_cnt += hist_cnt[bin_idx]
            if current_cnt < 2 or (n_node - current_cnt) < 2:
                continue
            right_sum_g = sum_g - current_sum_g
            right_cnt = n_node - current_cnt
            gain = np.sum(current_sum_g**2) / (current_cnt + reg_lambda) + np.sum(
                right_sum_g**2
            ) / (right_cnt + reg_lambda)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = bin_idx

    return best_feature, best_threshold, best_gain


@njit(fastmath=True)
def _numba_batch_predict(
    x,
    features,
    thresholds,
    left_children,
    right_children,
    values,
    n_samples,
):
    n_outputs = values.shape[1]
    out = np.empty((n_samples, n_outputs))
    for i in range(n_samples):
        node = 0
        while features[node] >= 0:
            if x[i, features[node]] <= thresholds[node]:
                node = left_children[node]
            else:
                node = right_children[node]
        out[i] = values[node]
    return out


class NGBoostTree(BaseEstimator, RegressorMixin):
    """Purpose-built histogram tree for NGBoost."""

    def __init__(self, max_depth=3, n_bins=32, min_samples_leaf=5, reg_lambda=0.1):
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.n_outputs_ = 1
        self._features = None
        self._thresholds = None
        self._left_children = None
        self._right_children = None
        self._values = None
        self._train_preds = None

    def fit(self, x, y, sample_weight=None):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.n_outputs_ = y.shape[1]
        max_nodes = (1 << (self.max_depth + 1)) - 1

        self._features = np.full(max_nodes, -1, dtype=np.int32)
        self._thresholds = np.full(max_nodes, -1, dtype=np.int32)
        self._left_children = np.zeros(max_nodes, dtype=np.int32)
        self._right_children = np.zeros(max_nodes, dtype=np.int32)
        self._values = np.zeros((max_nodes, self.n_outputs_), dtype=np.float64)

        idx = np.arange(x.shape[0])
        leaf_assignments = np.empty(x.shape[0], dtype=np.int32)
        self._build(x, y, idx, 0, 0, leaf_assignments)

        self._train_preds = self._values[leaf_assignments].copy()
        return self

    def _build(self, x_binned, gradients, idx, depth, node_id, leaf_assignments):
        if depth >= self.max_depth or len(idx) < self.min_samples_leaf:
            self._features[node_id] = -1
            self._values[node_id] = np.mean(gradients[idx], axis=0)
            leaf_assignments[idx] = node_id
            return node_id + 1

        feature, threshold, gain = _hist_split_kernel(
            x_binned[idx],
            gradients[idx],
            self.n_bins,
            x_binned.shape[1],
            gradients.shape[1],
            self.reg_lambda,
        )
        if feature == -1:
            self._features[node_id] = -1
            self._values[node_id] = np.mean(gradients[idx], axis=0)
            leaf_assignments[idx] = node_id
            return node_id + 1

        left_mask = x_binned[idx, feature] <= threshold
        left_idx, right_idx = idx[left_mask], idx[~left_mask]
        if len(left_idx) == 0 or len(right_idx) == 0:
            self._features[node_id] = -1
            self._values[node_id] = np.mean(gradients[idx], axis=0)
            leaf_assignments[idx] = node_id
            return node_id + 1

        self._features[node_id] = feature
        self._thresholds[node_id] = threshold
        left_child_id = node_id + 1
        self._left_children[node_id] = left_child_id

        next_id = self._build(
            x_binned,
            gradients,
            left_idx,
            depth + 1,
            left_child_id,
            leaf_assignments,
        )
        right_child_id = next_id
        self._right_children[node_id] = right_child_id
        return self._build(
            x_binned,
            gradients,
            right_idx,
            depth + 1,
            right_child_id,
            leaf_assignments,
        )

    def predict(self, x):
        return _numba_batch_predict(
            x,
            self._features,
            self._thresholds,
            self._left_children,
            self._right_children,
            self._values,
            x.shape[0],
        )

    def predict_cached(self):
        return self._train_preds


class PreBinnedNumbaNGBRegressor(FastMO_FixedStepNGBRegressor):
    """Bins X once to uint8, then uses a purpose-built histogram tree."""

    def _to_dense(self, x):
        return x.toarray() if hasattr(x, "toarray") else x

    def _bin_x(self, x):
        x = self._to_dense(check_array(x, accept_sparse=True, dtype=np.float64))
        x_binned = np.zeros(x.shape, dtype=np.uint8)
        for feature in range(x.shape[1]):
            x_binned[:, feature] = np.searchsorted(
                self.bin_edges_[feature], x[:, feature]
            )
        return x_binned

    def fit(self, x, y, **kwargs):
        x, y = _check_xy(x, y)
        x = self._to_dense(x)

        self.n_bins = 32
        self.bin_edges_ = []
        self.x_binned = np.zeros(x.shape, dtype=np.uint8)
        for feature in range(x.shape[1]):
            edges = np.percentile(
                x[:, feature],
                np.linspace(0, 100, self.n_bins + 1)[1:-1],
            )
            self.bin_edges_.append(edges)
            self.x_binned[:, feature] = np.searchsorted(edges, x[:, feature])

        self.base_models = []
        self.scalings = []
        self.col_idxs = []
        self.fit_init_params_to_marginal(y)
        params = self._pred_binned(self.x_binned)

        for _ in range(self.n_estimators):
            manifold_batch = self.Manifold(params.T)
            grads = manifold_batch.grad(y, natural=self.natural_gradient)
            model = NGBoostTree(max_depth=3, n_bins=self.n_bins)
            model.fit(self.x_binned, grads)
            fitted = model.predict_cached()
            self.base_models.append([model])
            self.col_idxs.append(np.arange(x.shape[1]))
            direction = self._transform_stage_pred(fitted)
            scale = self.line_search(direction, params, y)
            delta = self._compute_delta(direction, scale)
            self._apply_update(params, delta)
        return self

    def _pred_binned(self, x_binned, max_iter=None):
        m = x_binned.shape[0]
        params = np.ones((m, self.Manifold.n_params)) * self.init_params
        for i, (models, scale) in enumerate(zip(self.base_models, self.scalings)):
            if max_iter is not None and i >= max_iter:
                break
            stage_pred = models[0].predict(x_binned)
            direction = self._transform_stage_pred(stage_pred)
            delta = self._compute_delta(direction, scale)
            self._apply_update(params, delta)
        return params

    def pred_param(self, x, max_iter=None):
        return self._pred_binned(self._bin_x(x), max_iter=max_iter)

    def staged_pred_dist(self, x, max_iter=None):
        x_binned = self._bin_x(x)
        predictions = []
        m = x_binned.shape[0]
        params = np.ones((m, self.Dist.n_params)) * self.init_params
        for i, (models, scale) in enumerate(
            zip(self.base_models, self.scalings), start=1
        ):
            if max_iter is not None and i >= max_iter:
                break
            stage_pred = models[0].predict(x_binned)
            direction = self._transform_stage_pred(stage_pred)
            delta = self._compute_delta(direction, scale)
            self._apply_update(params, delta)
            predictions.append(self.Dist(np.copy(params.T)))
        return predictions

    def predict(self, x, **kwargs):
        return self.pred_dist(x).predict()


DEFAULT_BASE = DecisionTreeRegressor(max_depth=3, random_state=GLOBAL_SEED)
HIST_BASE = HistGradientBoostingRegressor(
    max_iter=1,
    learning_rate=1.0,
    early_stopping=False,
    max_leaf_nodes=31,
    random_state=GLOBAL_SEED,
)
XGB_BASE = (
    XGBoostMultiOutputEstimator(max_depth=3, learning_rate=1.0, n_jobs=4)
    if XGBoostMultiOutputEstimator is not None
    else None
)

HYPER_CONFIGS = [
    {"tag": "T50_lr0.10", "n_estimators": 50, "learning_rate": 0.10},
    {"tag": "T200_lr0.05", "n_estimators": 200, "learning_rate": 0.05},
    {
        "tag": "T400_lr0.05_ES20",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "early_stopping_rounds": 20,
        "validation_fraction": 0.1,
    },
]

UNI_EXPERIMENTS = [
    ("Baseline", NGBRegressor, {}),
    ("Baseline (NoNatGrad)", NGBRegressor, {"natural_gradient": False}),
    ("Baseline+DiagNG", DiagonalNGBRegressor, {}),
    ("Baseline+GlobalNG", GlobalNGBRegressor, {}),
    ("Baseline+FixedStep", FixedStepNGBRegressor, {}),
    ("NoCopy", NoCopyNGBRegressor, {}),
    ("NoCopy+FixedStep", NoCopy_FixedStepNGBRegressor, {}),
    ("SubsampledLS(5%)", SubsampledLineSearchNGBRegressor, {}),
    ("CappedLS(2up+3dn)", CappedLSNGBRegressor, {}),
    ("NoCopy+CappedLS", NoCopy_CappedLSNGBRegressor, {}),
    ("HybridLS(sub+cap)", HybridLSNGBRegressor, {}),
    ("MO", FastMultiOutputNGBRegressor, {}),
    ("MO (NoNatGrad)", FastMultiOutputNGBRegressor, {"natural_gradient": False}),
    ("MO+DiagNG", FastMO_DiagNGBRegressor, {}),
    ("MO+FixedStep", FastMO_FixedStepNGBRegressor, {}),
    ("MO+DiagNG+FixedStep", FastMO_DiagNG_FixedStepNGBRegressor, {}),
    ("MO+SubsampledLS", FastMO_SubsampledLSNGBRegressor, {}),
    ("MO+CappedLS", FastMO_CappedLSNGBRegressor, {}),
    ("MO+HybridLS", FastMO_HybridLSNGBRegressor, {}),
    ("MO+GradNorm", FastMO_GradNormNGBRegressor, {}),
    ("MO+GradNorm(a=1.0)", FastMO_GradNormNGBRegressor, {"grad_norm_alpha": 1.0}),
    ("MO+GradNorm(a=0.3)", FastMO_GradNormNGBRegressor, {"grad_norm_alpha": 0.3}),
    ("MO+GradNorm+FixedStep", FastMO_GradNorm_FixedStepNGBRegressor, {}),
    ("MO+GradNorm+CappedLS", FastMO_GradNorm_CappedLSNGBRegressor, {}),
    ("MO+GradNorm+HybridLS", FastMO_GradNorm_HybridLSNGBRegressor, {}),
    ("MO+DampedScale", FastMO_DampedScaleNGBRegressor, {"log_scale_clip": 0.0}),
    ("MO+DampedScale+FixedStep", FastMO_DampedScale_FixedStepNGBRegressor, {}),
    (
        "MO+DampedScale+CappedLS",
        FastMO_DampedScale_CappedLSNGBRegressor,
        {"log_scale_clip": 0.0},
    ),
    (
        "MO+DampedScale+ArmijoLS",
        FastMO_DampedScale_ArmijoLSNGBRegressor,
        {"log_scale_clip": 0.0},
    ),
    ("ParallelFit(threads)", ParallelFitBaseNGBRegressor, {}),
    ("NewtonLeaves", NewtonLeavesNGBRegressor, {}),
    ("NewtonLeaves+FixedStep", NewtonLeaves_FixedStepNGBRegressor, {}),
    ("MO+FisherHeuristic", FastMO_FisherHeuristicNGBRegressor, {}),
    ("MO+FisherHeuristic+FixedStep", FastMO_FisherHeuristic_FixedStepNGBRegressor, {}),
    ("WarmStartLS", WarmStartLSNGBRegressor, {}),
    ("MO+WarmStartLS", FastMO_WarmStartLSNGBRegressor, {}),
    ("ArmijoLS", ArmijoLSNGBRegressor, {}),
    ("MO+ArmijoLS", FastMO_ArmijoLSNGBRegressor, {}),
    ("HistBase", NGBRegressor, {"Base": HIST_BASE}),
    ("HistBase+FixedStep", FixedStepNGBRegressor, {"Base": HIST_BASE}),
    ("XGB+MO+FixedStep", FastMO_FixedStepNGBRegressor, {"Base": XGB_BASE}, "xgboost"),
    ("LightGBM+FixedStep", LightGBM_FixedStepNGBRegressor, {}, "lightgbm"),
    ("LightGBM+CappedLS", LightGBM_CappedLSNGBRegressor, {}, "lightgbm"),
    ("PreBinnedNumba+FixedStep", PreBinnedNumbaNGBRegressor, {}, "numba"),
]

BIGK_EXPERIMENTS = [
    ("BigK:Baseline", NGBRegressor, {}),
    ("BigK:NoCopy", NoCopyNGBRegressor, {}),
    ("BigK:CappedLS", CappedLSNGBRegressor, {}),
    ("BigK:HybridLS", HybridLSNGBRegressor, {}),
    ("BigK:MO", FastMultiOutputNGBRegressor, {}),
    ("BigK:MO+DiagNG", FastMO_DiagNGBRegressor, {}),
    ("BigK:MO+GlobalNG", FastMO_GlobalNGBRegressor, {}),
    ("BigK:MO (NoNatGrad)", FastMultiOutputNGBRegressor, {"natural_gradient": False}),
    ("BigK:MO+FixedStep", FastMO_FixedStepNGBRegressor, {}),
    ("BigK:MO+CappedLS", FastMO_CappedLSNGBRegressor, {}),
    ("BigK:MO+HybridLS", FastMO_HybridLSNGBRegressor, {}),
    ("BigK:MO+SubsampledLS", FastMO_SubsampledLSNGBRegressor, {}),
    ("BigK:MO+GradNorm(a=1.0)", FastMO_GradNormNGBRegressor, {}),
    ("BigK:MO+GradNorm(a=0.5)", FastMO_GradNormNGBRegressor, {"grad_norm_alpha": 0.5}),
    ("BigK:MO+GradNorm+HybridLS", FastMO_GradNorm_HybridLSNGBRegressor, {}),
    ("BigK:NewtonLeaves", NewtonLeavesNGBRegressor, {}),
    ("BigK:MO+FisherHeuristic", FastMO_FisherHeuristicNGBRegressor, {}),
    ("BigK:MO+ArmijoLS", FastMO_ArmijoLSNGBRegressor, {}),
    ("BigK:ParallelFit", ParallelFitBaseNGBRegressor, {}),
    ("BigK:PreBinnedNumba+FixedStep", PreBinnedNumbaNGBRegressor, {}, "numba"),
]

SHORT_UNI_EXPERIMENTS = [
    ("Baseline", NGBRegressor, {}),
    ("FN:Baseline", NGBRegressor, {"Dist": FastNormal, "Score": FastNormalLogScore}),
    ("MO", FastMultiOutputNGBRegressor, {}),
    (
        "FN:MO",
        FastMultiOutputNGBRegressor,
        {"Dist": FastNormal, "Score": FastNormalLogScore},
    ),
    ("MO+CappedLS", FastMO_CappedLSNGBRegressor, {}),
    ("MO+ArmijoLS", FastMO_ArmijoLSNGBRegressor, {}),
    ("MO+GradNorm", FastMO_GradNormNGBRegressor, {}),
    (
        "FN:MO+GradNorm",
        FastMO_GradNormNGBRegressor,
        {"Dist": FastNormal, "Score": FastNormalLogScore},
    ),
    ("MO+DampedScale", FastMO_DampedScaleNGBRegressor, {"log_scale_clip": 0.0}),
    ("MO+DampedScale+FixedStep", FastMO_DampedScale_FixedStepNGBRegressor, {}),
    (
        "MO+DampedScale+CappedLS",
        FastMO_DampedScale_CappedLSNGBRegressor,
        {"log_scale_clip": 0.0},
    ),
    (
        "MO+DampedScale+ArmijoLS",
        FastMO_DampedScale_ArmijoLSNGBRegressor,
        {"log_scale_clip": 0.0},
    ),
    ("LightGBM+FixedStep", LightGBM_FixedStepNGBRegressor, {}, "lightgbm"),
    ("LightGBM+CappedLS", LightGBM_CappedLSNGBRegressor, {}, "lightgbm"),
    ("PreBinnedNumba+FixedStep", PreBinnedNumbaNGBRegressor, {}, "numba"),
    (
        "FN:PreBinnedNumba+FixedStep",
        PreBinnedNumbaNGBRegressor,
        {"Dist": FastNormal, "Score": FastNormalLogScore},
        "numba",
    ),
]

SHORT_BIGK_EXPERIMENTS = [
    ("BigK:Baseline", NGBRegressor, {}),
    ("BigK:MO", FastMultiOutputNGBRegressor, {}),
    ("BigK:MO+DiagNG", FastMO_DiagNGBRegressor, {}),
    ("BigK:MO+CappedLS", FastMO_CappedLSNGBRegressor, {}),
    ("BigK:MO+ArmijoLS", FastMO_ArmijoLSNGBRegressor, {}),
    ("BigK:MO+FisherHeuristic", FastMO_FisherHeuristicNGBRegressor, {}),
    ("BigK:ParallelFit", ParallelFitBaseNGBRegressor, {}),
    ("BigK:PreBinnedNumba+FixedStep", PreBinnedNumbaNGBRegressor, {}, "numba"),
]

PROFILE_EXPERIMENTS = [
    ("Baseline", NGBRegressor, {}),
    ("MO", FastMultiOutputNGBRegressor, {}),
    ("MO+GradNorm", FastMO_GradNormNGBRegressor, {}),
    ("NoCopy", NoCopyNGBRegressor, {}),
    ("NoCopy+CappedLS", NoCopy_CappedLSNGBRegressor, {}),
    ("MO+CappedLS", FastMO_CappedLSNGBRegressor, {}),
    ("MO+HybridLS", FastMO_HybridLSNGBRegressor, {}),
]

BIGK_SIZES = [5000, 10000]
BIGK_N_ESTIMATORS = [50, 100]

STRUCTURAL_SMOKE_METHODS = [
    ("Baseline", NGBRegressor, {}),
    ("FN:Baseline", NGBRegressor, {"Dist": FastNormal, "Score": FastNormalLogScore}),
    ("MO", FastMultiOutputNGBRegressor, {}),
    ("MO+DiagNG", FastMO_DiagNGBRegressor, {}),
    ("MO+GlobalNG", FastMO_GlobalNGBRegressor, {}),
    ("MO+GradNorm", FastMO_GradNormNGBRegressor, {}),
    ("MO+CappedLS", FastMO_CappedLSNGBRegressor, {}),
    ("MO+SubsampledLS", FastMO_SubsampledLSNGBRegressor, {}),
    ("MO+HybridLS", FastMO_HybridLSNGBRegressor, {}),
    ("MO+FisherHeuristic", FastMO_FisherHeuristicNGBRegressor, {}),
    ("MO+DampedScale", FastMO_DampedScaleNGBRegressor, {}),
    ("NewtonLeaves", NewtonLeavesNGBRegressor, {}),
    ("NoCopy", NoCopyNGBRegressor, {}),
    ("CappedLS", CappedLSNGBRegressor, {}),
    ("WarmStartLS", WarmStartLSNGBRegressor, {}),
    ("ArmijoLS", ArmijoLSNGBRegressor, {}),
    ("MO+WarmStartLS", FastMO_WarmStartLSNGBRegressor, {}),
    ("MO+ArmijoLS", FastMO_ArmijoLSNGBRegressor, {}),
    (
        "PreBinnedNumba+FixedStep",
        PreBinnedNumbaNGBRegressor,
        {},
        "numba",
    ),
]


def _build_registry(raw_specs, backend_statuses=None):
    registry = []
    for entry in raw_specs:
        if len(entry) == 3:
            name, cls, extra = entry
            backend = None
        else:
            name, cls, extra, backend = entry
        registry.append(
            _make_spec(
                name,
                cls,
                extra,
                backend=backend,
                backend_statuses=backend_statuses,
            )
        )
    return registry


def get_univariate_experiments(
    suite: str,
    *,
    backend_statuses: dict[str, BackendStatus] | None = None,
) -> list[ExperimentSpec]:
    raw_specs = SHORT_UNI_EXPERIMENTS if suite == "short" else UNI_EXPERIMENTS
    return _build_registry(raw_specs, backend_statuses)


def get_bigk_experiments(
    suite: str,
    *,
    backend_statuses: dict[str, BackendStatus] | None = None,
) -> list[ExperimentSpec]:
    raw_specs = SHORT_BIGK_EXPERIMENTS if suite == "short" else BIGK_EXPERIMENTS
    return _build_registry(raw_specs, backend_statuses)


def get_profile_experiments(
    *,
    backend_statuses: dict[str, BackendStatus] | None = None,
) -> list[ExperimentSpec]:
    return _build_registry(PROFILE_EXPERIMENTS, backend_statuses)


def get_structural_smoke_experiments(
    *,
    backend_statuses: dict[str, BackendStatus] | None = None,
) -> list[ExperimentSpec]:
    return _build_registry(STRUCTURAL_SMOKE_METHODS, backend_statuses)


def get_all_method_names(
    suite: str,
    *,
    backend_statuses: dict[str, BackendStatus] | None = None,
) -> set[str]:
    names = {
        spec.name
        for spec in get_univariate_experiments(suite, backend_statuses=backend_statuses)
    }
    names.update(
        spec.name
        for spec in get_bigk_experiments(suite, backend_statuses=backend_statuses)
    )
    names.update(
        spec.name for spec in get_profile_experiments(backend_statuses=backend_statuses)
    )
    return names


def warm_up_numba(verbose: bool = True) -> None:
    """Warm up numba kernels if numba-backed variants are available."""

    if NUMBA_MODULE is None:
        return
    _hist_split_kernel(
        np.zeros((16, 3), dtype=np.uint8),
        np.zeros((16, 4), dtype=np.float64),
        8,
        3,
        4,
        0.1,
    )
    _numba_batch_predict(
        np.zeros((4, 3), dtype=np.uint8),
        np.full(3, -1, dtype=np.int32),
        np.zeros(3, dtype=np.int32),
        np.zeros(3, dtype=np.int32),
        np.zeros(3, dtype=np.int32),
        np.zeros((3, 2), dtype=np.float64),
        4,
    )
    if verbose:
        print("Numba JIT warmed up (split kernel + batch predict).")


def warm_up_ngboost_paths(verbose: bool = True) -> None:
    """Warm up common fit paths before timed benchmarks."""

    warm_x = np.random.RandomState(0).randn(500, 8)
    warm_y = np.random.RandomState(1).randn(500)
    NGBRegressor(
        n_estimators=3,
        learning_rate=0.1,
        verbose=False,
        Dist=Normal,
    ).fit(warm_x, warm_y)
    NGBRegressor(
        n_estimators=3,
        learning_rate=0.1,
        verbose=False,
        Dist=FastNormal,
        Score=FastNormalLogScore,
    ).fit(warm_x, warm_y)
    FastMultiOutputNGBRegressor(
        n_estimators=3,
        learning_rate=0.1,
        verbose=False,
    ).fit(warm_x, warm_y)
    if verbose:
        print("NGBoost warmup complete.")


def smoke_test_all(
    *,
    backend_statuses: dict[str, BackendStatus] | None = None,
    verbose: bool = True,
) -> bool:
    """Run structural sanity checks for always-available core variants."""

    x = np.random.RandomState(42).randn(200, 8)
    y = np.random.RandomState(42).randn(200)
    n_estimators = 5

    passed = 0
    failed = 0
    skipped = 0
    for spec in get_structural_smoke_experiments(backend_statuses=backend_statuses):
        if not spec.runnable:
            skipped += 1
            if verbose:
                print(f"  SKIP: {spec.name}: {spec.skip_reason}")
            continue

        try:
            model_kwargs = {
                "Dist": Normal,
                "n_estimators": n_estimators,
                "learning_rate": 0.1,
                "verbose": False,
                "tol": 0.0,
            }
            model_kwargs.update(spec.extra_kwargs)
            model = spec.estimator_cls(**model_kwargs)
            model.fit(x, y)
            assert model.predict(x[:7]).shape == (
                7,
            ), f"{spec.name}: predict shape wrong"
            dist = model.pred_dist(x[:7])
            assert (
                np.asarray(dist.loc).shape[0] == 7
            ), f"{spec.name}: pred_dist shape wrong"
            staged = list(model.staged_pred_dist(x[:7]))
            assert (
                len(staged) == n_estimators
            ), f"{spec.name}: staged_pred_dist length {len(staged)} != {n_estimators}"
            assert (
                len(model.base_models) == n_estimators
            ), f"{spec.name}: base_models length {len(model.base_models)} != {n_estimators}"
            for stage_models in model.base_models:
                assert isinstance(
                    stage_models, list
                ), f"{spec.name}: base_models stage is not a list"
            passed += 1
            if verbose:
                print(f"  PASS: {spec.name}")
        except Exception as exc:
            failed += 1
            if verbose:
                print(f"  FAIL: {spec.name}: {exc}")

    if verbose:
        print(
            "\nSmoke test: "
            f"{passed} passed, {failed} failed, {skipped} skipped "
            f"out of {len(get_structural_smoke_experiments(backend_statuses=backend_statuses))}"
        )
    if failed > 0:
        raise RuntimeError(f"{failed} structural smoke tests failed")
    return True
