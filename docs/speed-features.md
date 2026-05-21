# Safe Speed Features And Backend Guidance

This page documents the speed-related production API and the boundary between
safe optimizations, optional behavior-changing accelerators, and benchmark-only
experiments.

## Defaults

NGBoost defaults remain historical:

```python
from ngboost import NGBRegressor

model = NGBRegressor()
```

The default continues to fit separate per-parameter base learners serially with
the default sklearn decision-tree base learner and standard line search.

## Recommended Backend Choices

| Use case | Recommendation | Rationale |
| --- | --- | --- |
| Ordinary univariate Normal regression | Keep defaults | Only two distribution parameters; parallel overhead often eats the win |
| High-parameter distributions such as `MultivariateNormal(k)` | `fit_base_mode="parallel_separate"` | Preserves model semantics and parallelizes independent per-parameter fits |
| Large tabular data where behavior changes are acceptable | Optional `LightGBMTreeLearner` | Different tree backend; validate NLL/RMSE/calibration per dataset |
| Research on shared tree structures | Benchmark-only MO variants | Fast, but changes split selection and model behavior |
| Research on custom histogram trees | Benchmark-only PreBinnedNumba | Fast exploratory path, not production hardened |

## `fit_base_mode`

`fit_base_mode` controls how NGBoost fits the per-parameter base learners at each
boosting iteration.

```python
from ngboost import NGBRegressor
from ngboost.distns import MultivariateNormal

model = NGBRegressor(
    Dist=MultivariateNormal(5),
    fit_base_mode="parallel_separate",
    n_jobs_fit=4,
)
```

Available values:

| Value | Meaning |
| --- | --- |
| `"separate"` | Historical default. Fit one base learner per distribution parameter serially. |
| `"parallel_separate"` | Fit the same separate per-parameter learners concurrently with joblib. |

Safety scope:

- Expected to be semantics-preserving for deterministic, thread-safe base
  learners.
- Tested against serial fitting for Normal and `MultivariateNormal(2/5)`.
- Tested with sample weights, minibatching, column sampling, explicit validation,
  sparse inputs, classifier fitting, survival fitting, and pickle roundtrips.
- Not guaranteed to be bit-identical for stochastic base learners, base learners
  that consume global RNG state, or base learners with their own internal thread
  pools.

Threading guidance:

- Start with `n_jobs_fit=4` for high-parameter distributions.
- If the base learner has its own `n_jobs`, avoid nested parallelism. For example,
  do not combine `fit_base_mode="parallel_separate"` with a base learner also
  using all CPU cores unless you have measured it.
- Use environment variables such as `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and
  `OPENBLAS_NUM_THREADS` to keep comparisons fair.

## Normal LogScore Fast Path

The Normal distribution now avoids unnecessary SciPy frozen-distribution work in
the common LogScore path:

- `Normal.logpdf()` computes the log-density inline.
- `NormalLogScore.score()` mirrors `-logpdf`.
- `NormalLogScore._natural_gradient()` uses the exact diagonal Fisher solve for
  the Normal LogScore parameterization.

This is an internal exact optimization, not a separate user-facing backend. The
tests compare the inline logpdf against `scipy.stats.norm.logpdf`, verify
`score == -logpdf`, compare the diagonal natural gradient to a generic Fisher
solve, and include a CRPS regression check to ensure CRPS behavior was not
changed.

## Loss-Checked Capped Line Search

The historical line search remains the default:

```python
model = NGBRegressor(line_search_strategy="standard")
```

The opt-in bounded shortcut is:

```python
model = NGBRegressor(
    line_search_strategy="loss_checked_capped",
    line_search_max_up=2,
    line_search_max_down=3,
)
```

The old alias `"capped"` is accepted and canonicalized to
`"loss_checked_capped"`.

Important: this is a behavior-changing optimization. It is "loss checked" in the
sense that the selected step is required to avoid increasing the checked
training-batch loss, or else it falls back to a zero step. It is not equivalent
to the standard NGBoost line search and should not be described as a safe
accuracy-preserving speed feature.

## LightGBM Backend

`LightGBMTreeLearner` is an optional non-equivalent backend:

```python
from ngboost import NGBRegressor
from ngboost.learners import LightGBMTreeLearner

model = NGBRegressor(
    Base=LightGBMTreeLearner(
        max_depth=3,
        num_leaves=31,
        n_jobs=4,
    ),
    line_search_strategy="loss_checked_capped",
)
```

Use it when:

- You explicitly want a LightGBM tree builder.
- You can validate predictive quality and calibration on your own data.
- You are comfortable with changed tree-construction semantics.

Do not claim that public `LightGBMTreeLearner` has the same speedup as the
benchmark-only `LightGBMPersistentNGBRegressor` variant unless the public wrapper
has been benchmarked directly. The clean full run reported strong LightGBM
numbers for the persistent benchmark estimator, not for the public wrapper.

## Benchmark-Only Variants

The following should stay in `benchmarks/variants.py` unless a future design
proves them safe enough for production:

| Variant family | Why it stays benchmark-only |
| --- | --- |
| Multi-output shared trees (`MO`) | A single tree predicts multiple distribution-parameter updates, changing split selection |
| FixedStep | Removes adaptive line search and can degrade calibration |
| PreBinnedNumba | Custom tree is not production hardened for all sklearn semantics |
| Generic DiagNG/GlobalNG | Accuracy and calibration are not consistently safe |
| NewtonLeaves/Fisher heuristics | Research paths needing more validation |
| Persistent LightGBM benchmark estimator | Benchmark implementation, not public API |

## Quick Recipes

### Safe multivariate speedup

```python
from ngboost import NGBRegressor
from ngboost.distns import MultivariateNormal

model = NGBRegressor(
    Dist=MultivariateNormal(5),
    fit_base_mode="parallel_separate",
    n_jobs_fit=4,
)
```

### Historical univariate default

```python
from ngboost import NGBRegressor

model = NGBRegressor()
```

### Optional LightGBM experiment

```python
from ngboost import NGBRegressor
from ngboost.learners import LightGBMTreeLearner

model = NGBRegressor(
    Base=LightGBMTreeLearner(n_jobs=4),
    line_search_strategy="loss_checked_capped",
)
```

## Validation Checklist

Before using a speed option in production:

- Compare NLL and RMSE against baseline on a held-out set.
- Check calibration diagnostics, not just point prediction.
- Run with the same thread settings used in deployment.
- Confirm whether the option is exact (`parallel_separate` with deterministic
  base learners) or behavior-changing (`LightGBMTreeLearner`,
  `loss_checked_capped`).
- For multivariate outputs, inspect covariance eigenvalue and Mahalanobis
  diagnostics.

