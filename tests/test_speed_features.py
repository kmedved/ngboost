import numpy as np
import pytest
import scipy.stats
from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBRegressor
from ngboost.distns import MultivariateNormal, Normal
from ngboost.distns.normal import NormalLogScore
from ngboost.learners import LightGBMTreeLearner


def test_normal_logscore_matches_scipy_and_diagonal_natural_gradient():
    params = np.array([[0.0, 1.0, -1.0], [0.0, np.log(2.0), np.log(0.5)]])
    y = np.array([0.25, -0.5, 0.0])
    dist = Normal(params)

    np.testing.assert_allclose(
        dist.logpdf(y),
        scipy.stats.norm.logpdf(y, loc=dist.loc, scale=dist.scale),
    )

    score = NormalLogScore.__new__(NormalLogScore)
    score.__dict__.update(dist.__dict__)
    grad = score.d_score(y)
    metric = score.metric()

    np.testing.assert_allclose(score.score(y), -dist.logpdf(y))
    np.testing.assert_allclose(
        score._natural_gradient(grad, metric),
        np.linalg.solve(metric, grad[..., None])[..., 0],
    )


def test_parallel_separate_matches_serial_multivariate_fit():
    rng = np.random.RandomState(0)
    x = rng.randn(80, 4)
    y = rng.randn(80, 2)
    dist = MultivariateNormal(2)
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
    parallel = NGBRegressor(
        **common,
        fit_base_mode="parallel_separate",
        n_jobs_fit=2,
    )

    serial.fit(x, y)
    parallel.fit(x, y)

    np.testing.assert_allclose(serial.pred_param(x[:10]), parallel.pred_param(x[:10]))
    np.testing.assert_allclose(serial.predict(x[:10]), parallel.predict(x[:10]))


def test_capped_line_search_never_returns_increasing_step():
    class DummyManifold:
        def __init__(self, params):
            self.params = params

        def total_score(self, y, sample_weight=None):
            return float(np.mean(self.params**2))

    model = NGBRegressor(
        n_estimators=1,
        verbose=False,
        line_search_strategy="capped",
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


def test_lightgbm_tree_learner_is_optional_backend():
    pytest.importorskip("lightgbm")

    rng = np.random.RandomState(1)
    x = rng.randn(80, 4)
    y = rng.randn(80)
    model = NGBRegressor(
        Base=LightGBMTreeLearner(max_depth=2, num_leaves=4, random_state=0),
        n_estimators=3,
        learning_rate=0.05,
        line_search_strategy="capped",
        verbose=False,
    )

    model.fit(x, y)

    assert model.predict(x[:5]).shape == (5,)
    assert len(model.base_models) == 3
    assert model.get_params()["line_search_strategy"] == "capped"
