import builtins
import inspect
import pickle

import numpy as np
import pytest
import scipy.sparse
import scipy.stats
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBClassifier, NGBRegressor, NGBSurvival
from ngboost.distns import MultivariateNormal, Normal
from ngboost.distns.normal import NormalCRPScore, NormalLogScore
from ngboost.learners import LightGBMTreeLearner


def _score_instance(score_cls, dist):
    score = score_cls.__new__(score_cls)
    score.__dict__.update(dist.__dict__)
    return score


def _assert_constructor_accepts_get_params(model):
    params = model.get_params()
    signature = inspect.signature(type(model).__init__)
    constructor_keys = set(signature.parameters) - {"self"}

    assert set(params).issubset(constructor_keys)
    rebuilt = type(model)(**params)
    cloned = clone(model)

    for key in (
        "verbose_eval",
        "validation_fraction",
        "early_stopping_rounds",
        "fit_base_mode",
        "n_jobs_fit",
        "line_search_strategy",
        "line_search_max_up",
        "line_search_max_down",
    ):
        assert rebuilt.get_params()[key] == params[key]
        assert cloned.get_params()[key] == params[key]


@pytest.mark.parametrize("estimator_cls", [NGBRegressor, NGBClassifier, NGBSurvival])
@pytest.mark.parametrize("random_state", [None, 0])
def test_sklearn_clone_and_constructor_params_round_trip(estimator_cls, random_state):
    model = estimator_cls(
        n_estimators=2,
        verbose=False,
        verbose_eval=7,
        random_state=random_state,
        validation_fraction=0.2,
        early_stopping_rounds=3,
        fit_base_mode="parallel_separate",
        n_jobs_fit=2,
        line_search_strategy="loss_checked_capped",
        line_search_max_up=1,
        line_search_max_down=2,
    )

    _assert_constructor_accepts_get_params(model)


def test_capped_line_search_alias_canonicalizes_to_loss_checked_capped():
    model = NGBRegressor(line_search_strategy="capped", verbose=False)

    assert model.line_search_strategy == "loss_checked_capped"
    assert model.get_params()["line_search_strategy"] == "loss_checked_capped"


def test_set_params_canonicalizes_validates_and_rebuilds_manifold():
    model = NGBRegressor(random_state=0, verbose=False)

    model.set_params(
        line_search_strategy="capped",
        line_search_max_up=np.int64(0),
        line_search_max_down=np.int64(1),
    )

    assert model.line_search_strategy == "loss_checked_capped"
    assert model.line_search_max_up == 0
    assert model.line_search_max_down == 1
    assert model.get_params()["random_state"] == 0

    mvn2 = MultivariateNormal(2)
    model.set_params(Dist=mvn2)
    assert model.Dist is mvn2
    assert model.Manifold.n_params == mvn2.n_params
    assert model.multi_output is True

    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(not_a_real_param=1)
    with pytest.raises(ValueError, match="line_search_strategy"):
        model.set_params(line_search_strategy="fast")
    with pytest.raises(ValueError, match="nonnegative integer"):
        model.set_params(line_search_max_up=-1)


def test_survival_set_params_wraps_base_distribution():
    model = NGBSurvival(verbose=False)
    model.set_params(Dist=model.get_params()["Dist"], line_search_strategy="capped")

    assert model.get_params()["Dist"].__name__ == "LogNormal"
    assert model.line_search_strategy == "loss_checked_capped"


@pytest.mark.parametrize("bad_strategy", ["fast", "", None])
def test_invalid_line_search_strategy_fails(bad_strategy):
    with pytest.raises(ValueError, match="line_search_strategy"):
        NGBRegressor(line_search_strategy=bad_strategy)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"line_search_max_up": -1},
        {"line_search_max_down": -1},
        {"line_search_max_up": 1.5},
        {"line_search_max_down": True},
    ],
)
def test_invalid_line_search_caps_fail(kwargs):
    with pytest.raises(ValueError, match="nonnegative integer"):
        NGBRegressor(line_search_strategy="loss_checked_capped", **kwargs)


@pytest.mark.parametrize("n", [1, 3, 100])
def test_normal_logscore_matches_scipy_randomized(n):
    rng = np.random.RandomState(n)
    loc = rng.uniform(-10, 10, size=n)
    log_scale = rng.uniform(-8, 8, size=n)
    y = loc + rng.randn(n) * np.exp(log_scale)
    dist = Normal(np.vstack([loc, log_scale]))
    score = _score_instance(NormalLogScore, dist)
    grad = score.d_score(y)
    metric = score.metric()

    np.testing.assert_allclose(
        dist.logpdf(y),
        scipy.stats.norm.logpdf(y, loc=dist.loc, scale=dist.scale),
    )
    np.testing.assert_allclose(score.score(y), -dist.logpdf(y))
    np.testing.assert_allclose(
        score._natural_gradient(grad, metric),
        np.linalg.solve(metric, grad[..., None])[..., 0],
    )


def test_normal_crps_score_matches_reference_formula():
    params = np.array([[0.0, 1.0, -1.0], [0.0, np.log(2.0), np.log(0.5)]])
    y = np.array([0.25, -0.5, 0.0])
    dist = Normal(params)
    score = _score_instance(NormalCRPScore, dist)
    z = (y - dist.loc) / dist.scale
    expected = dist.scale * (
        z * (2 * scipy.stats.norm.cdf(z) - 1)
        + 2 * scipy.stats.norm.pdf(z)
        - 1 / np.sqrt(np.pi)
    )

    np.testing.assert_allclose(score.score(y), expected)


def _make_regression_data(target_dim, seed=0, n=120):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, 5)
    if target_dim == 1:
        y = rng.randn(n)
    else:
        y = rng.randn(n, target_dim)
    return x, y


def _assert_models_equivalent(left, right, x):
    np.testing.assert_allclose(left.pred_param(x), right.pred_param(x), atol=1e-12)
    np.testing.assert_allclose(left.predict(x), right.predict(x), atol=1e-12)
    np.testing.assert_allclose(
        left.pred_dist(x)._params,
        right.pred_dist(x)._params,
        atol=1e-12,
    )
    np.testing.assert_allclose(left.scalings, right.scalings, atol=1e-12)
    assert len(left.base_models) == len(right.base_models)
    assert left.evals_result.keys() == right.evals_result.keys()
    for dataset_name, left_metrics in left.evals_result.items():
        right_metrics = right.evals_result[dataset_name]
        assert left_metrics.keys() == right_metrics.keys()
        for metric_name, left_values in left_metrics.items():
            np.testing.assert_allclose(
                left_values,
                right_metrics[metric_name],
                atol=1e-12,
            )
    for left_cols, right_cols in zip(left.col_idxs, right.col_idxs):
        np.testing.assert_array_equal(left_cols, right_cols)

    reloaded = pickle.loads(pickle.dumps(right))
    np.testing.assert_allclose(right.pred_param(x), reloaded.pred_param(x), atol=1e-12)


@pytest.mark.parametrize(
    "dist,target_dim",
    [
        (Normal, 1),
        (MultivariateNormal(2), 2),
        (MultivariateNormal(5), 5),
    ],
)
def test_parallel_separate_matches_serial_fit_modes(dist, target_dim):
    x, y = _make_regression_data(target_dim, seed=target_dim)
    base = DecisionTreeRegressor(max_depth=2, random_state=0)
    common = dict(
        Dist=dist,
        Base=base,
        n_estimators=4,
        learning_rate=0.05,
        random_state=0,
        verbose=False,
    )
    serial = NGBRegressor(**common, fit_base_mode="separate")
    parallel = NGBRegressor(**common, fit_base_mode="parallel_separate", n_jobs_fit=2)

    serial.fit(x, y)
    parallel.fit(x, y)

    _assert_models_equivalent(serial, parallel, x[:20])


def test_parallel_separate_matches_serial_with_sampling_weights_and_validation():
    x, y = _make_regression_data(2, seed=7)
    x_train, x_val = x[:90], x[90:]
    y_train, y_val = y[:90], y[90:]
    sample_weight = np.linspace(0.5, 1.5, len(x_train))
    val_sample_weight = np.linspace(0.75, 1.25, len(x_val))
    base = DecisionTreeRegressor(max_depth=2, random_state=0)
    common = dict(
        Dist=MultivariateNormal(2),
        Base=base,
        n_estimators=5,
        learning_rate=0.05,
        random_state=0,
        verbose=False,
        minibatch_frac=0.8,
        col_sample=0.6,
        early_stopping_rounds=2,
    )
    serial = NGBRegressor(**common, fit_base_mode="separate")
    parallel = NGBRegressor(**common, fit_base_mode="parallel_separate", n_jobs_fit=2)

    serial.fit(
        x_train,
        y_train,
        X_val=x_val,
        Y_val=y_val,
        sample_weight=sample_weight,
        val_sample_weight=val_sample_weight,
    )
    parallel.fit(
        x_train,
        y_train,
        X_val=x_val,
        Y_val=y_val,
        sample_weight=sample_weight,
        val_sample_weight=val_sample_weight,
    )

    _assert_models_equivalent(serial, parallel, x[:20])


def test_parallel_separate_matches_serial_with_sparse_input():
    x, y = _make_regression_data(1, seed=17)
    x_sparse = scipy.sparse.csr_matrix(x)
    base = DecisionTreeRegressor(max_depth=2, random_state=0)
    common = dict(
        Dist=Normal,
        Base=base,
        n_estimators=4,
        learning_rate=0.05,
        random_state=0,
        verbose=False,
    )
    serial = NGBRegressor(**common, fit_base_mode="separate")
    parallel = NGBRegressor(**common, fit_base_mode="parallel_separate", n_jobs_fit=2)

    serial.fit(x_sparse, y)
    parallel.fit(x_sparse, y)

    _assert_models_equivalent(serial, parallel, x_sparse[:20])


def test_parallel_separate_matches_serial_classifier_fit():
    rng = np.random.RandomState(23)
    x = rng.randn(120, 5)
    y = (x[:, 0] + 0.25 * rng.randn(120) > 0).astype(int)
    base = DecisionTreeRegressor(max_depth=2, random_state=0)
    common = dict(
        Base=base,
        n_estimators=5,
        learning_rate=0.05,
        random_state=0,
        verbose=False,
    )
    serial = NGBClassifier(**common, fit_base_mode="separate")
    parallel = NGBClassifier(**common, fit_base_mode="parallel_separate", n_jobs_fit=2)

    serial.fit(x, y)
    parallel.fit(x, y)

    _assert_models_equivalent(serial, parallel, x[:20])
    np.testing.assert_allclose(
        serial.predict_proba(x[:20]),
        parallel.predict_proba(x[:20]),
        atol=1e-12,
    )


def test_parallel_separate_matches_serial_survival_fit_with_validation():
    rng = np.random.RandomState(29)
    x = rng.randn(140, 5)
    durations = np.exp(0.2 * x[:, 0] + rng.randn(140) * 0.1)
    events = rng.binomial(1, 0.8, size=140).astype(bool)
    base = DecisionTreeRegressor(max_depth=2, random_state=0)
    common = dict(
        Base=base,
        n_estimators=5,
        learning_rate=0.05,
        random_state=0,
        verbose=False,
        early_stopping_rounds=2,
    )
    serial = NGBSurvival(**common, fit_base_mode="separate")
    parallel = NGBSurvival(**common, fit_base_mode="parallel_separate", n_jobs_fit=2)

    serial.fit(
        x[:100],
        durations[:100],
        events[:100],
        X_val=x[100:],
        T_val=durations[100:],
        E_val=events[100:],
    )
    parallel.fit(
        x[:100],
        durations[:100],
        events[:100],
        X_val=x[100:],
        T_val=durations[100:],
        E_val=events[100:],
    )

    _assert_models_equivalent(serial, parallel, x[:20])


def test_loss_checked_line_search_never_returns_increasing_step():
    class DummyManifold:
        def __init__(self, params):
            self.params = params

        def total_score(self, y, sample_weight=None):
            return float(np.mean(self.params**2))

    model = NGBRegressor(
        n_estimators=1,
        verbose=False,
        line_search_strategy="loss_checked_capped",
        line_search_max_up=0,
        line_search_max_down=1,
    )
    model.Manifold = DummyManifold

    scale = model.line_search(
        np.ones((5, 1)),
        np.zeros((5, 1)),
        np.zeros(5),
        scale_init=1.0,
    )

    assert scale == 0.0
    assert model.scalings == [0.0]


@pytest.mark.parametrize(
    "dist,target_dim",
    [
        (Normal, 1),
        (MultivariateNormal(2), 2),
    ],
)
@pytest.mark.parametrize("weighted", [False, True])
def test_loss_checked_line_search_real_manifold_invariant(dist, target_dim, weighted):
    x, y = _make_regression_data(target_dim, seed=11 + target_dim, n=80)
    _ = x
    model = NGBRegressor(
        Dist=dist,
        n_estimators=1,
        verbose=False,
        line_search_strategy="loss_checked_capped",
        line_search_max_up=1,
        line_search_max_down=1,
    )
    init = model.Manifold.fit(y)
    start = np.ones((len(y), model.Manifold.n_params)) * init
    rng = np.random.RandomState(3)
    resids = rng.randn(*start.shape) * 0.05
    sample_weight = rng.uniform(0.5, 1.5, size=len(y)) if weighted else None
    loss0 = model.Manifold(start.T).total_score(y, sample_weight)

    scale = model.line_search(resids, start, y, sample_weight=sample_weight)

    loss1 = model.Manifold((start - resids * scale).T).total_score(y, sample_weight)
    assert scale == 0.0 or (np.isfinite(loss1) and loss1 <= loss0)


def test_lightgbm_tree_learner_is_optional_non_equivalent_backend():
    pytest.importorskip("lightgbm")

    rng = np.random.RandomState(1)
    x = rng.randn(80, 4)
    y = rng.randn(80)
    sample_weight = rng.uniform(0.5, 1.5, size=len(y))
    learner = LightGBMTreeLearner(max_depth=2, num_leaves=4, random_state=0, n_jobs=1)
    cloned_learner = clone(learner)
    model = NGBRegressor(
        Base=cloned_learner,
        n_estimators=3,
        learning_rate=0.05,
        line_search_strategy="loss_checked_capped",
        verbose=False,
    )
    cloned_model = clone(model)

    model.fit(x, y, sample_weight=sample_weight)

    assert model.predict(x[:5]).shape == (5,)
    assert len(model.base_models) == 3
    assert model.get_params()["line_search_strategy"] == "loss_checked_capped"
    assert cloned_model.get_params()["Base"].max_depth == learner.max_depth


def test_lightgbm_tree_learner_missing_dependency_message(monkeypatch):
    rng = np.random.RandomState(31)
    x = rng.randn(10, 2)
    y = rng.randn(10)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "lightgbm":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="requires lightgbm"):
        LightGBMTreeLearner().fit(x, y)


def test_benchmark_registry_names_public_speed_paths():
    from benchmarks.variants import (
        BackendStatus,
        BENCH_N_THREADS,
        get_bigk_experiments,
        get_univariate_experiments,
    )

    backend_statuses = {"lightgbm": BackendStatus("lightgbm", True)}
    uni_specs = {
        spec.name: spec
        for spec in get_univariate_experiments(
            "full", backend_statuses=backend_statuses
        )
    }
    bigk_specs = {
        spec.name: spec
        for spec in get_bigk_experiments("full", backend_statuses=backend_statuses)
    }

    assert uni_specs["PublicParallelSeparate"].extra_kwargs == {
        "fit_base_mode": "parallel_separate",
        "n_jobs_fit": BENCH_N_THREADS,
    }
    assert (
        uni_specs["PublicLightGBMTreeLearner+CappedLS"].extra_kwargs[
            "line_search_strategy"
        ]
        == "loss_checked_capped"
    )
    assert bigk_specs["BigK:PublicParallelSeparate"].extra_kwargs == {
        "fit_base_mode": "parallel_separate",
        "n_jobs_fit": BENCH_N_THREADS,
    }
