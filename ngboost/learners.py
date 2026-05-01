import os

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

default_tree_learner = DecisionTreeRegressor(
    criterion="friedman_mse",
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_depth=3,
    splitter="best",
    random_state=None,
)

default_linear_learner = Ridge(alpha=0.0, random_state=None)


class LightGBMTreeLearner(BaseEstimator, RegressorMixin):
    """Optional one-tree LightGBM base learner for NGBoost.

    This wrapper keeps LightGBM as an optional dependency. It is intended to be
    passed explicitly as ``Base=LightGBMTreeLearner(...)``; NGBoost defaults
    remain unchanged.
    """

    def __init__(
        self,
        max_depth=3,
        num_leaves=31,
        min_child_samples=1,
        learning_rate=1.0,
        n_estimators=1,
        n_jobs=None,
        random_state=None,
        reg_alpha=0.0,
        reg_lambda=0.0,
        subsample=1.0,
        colsample_bytree=1.0,
        verbosity=-1,
    ):
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.verbosity = verbosity

    def _resolved_n_jobs(self):
        if self.n_jobs is not None:
            return self.n_jobs
        return int(os.environ.get("OMP_NUM_THREADS", "1"))

    def fit(self, X, y, sample_weight=None):
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "LightGBMTreeLearner requires lightgbm. Install lightgbm to use "
                "this optional NGBoost base learner."
            ) from exc

        self.model_ = lgb.LGBMRegressor(
            objective="regression",
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            n_jobs=self._resolved_n_jobs(),
            random_state=self.random_state,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            verbosity=self.verbosity,
        )
        self.model_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_
