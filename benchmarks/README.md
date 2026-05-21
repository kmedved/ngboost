# NGBoost Benchmark Suite

This package contains benchmark-only code for evaluating NGBoost speed changes
against predictive quality. It is intentionally separate from the production
estimators in `ngboost/`.

For the full methodology and result interpretation, see
[`docs/benchmarking.md`](../docs/benchmarking.md). For production API guidance,
see [`docs/speed-features.md`](../docs/speed-features.md).

## Package Layout

| File | Purpose |
| --- | --- |
| `run.py` | CLI entrypoint |
| `runner.py` | Benchmark execution engine |
| `variants.py` | Benchmark-only estimator variants and experiment registries |
| `datasets.py` | Synthetic, sklearn, and OpenML dataset builders |
| `metrics.py` | Runtime, RMSE, NLL, scale, covariance, and calibration metrics |
| `profiling.py` | Per-component profiling helpers |
| `plotting.py` | Timing, Pareto, BigK, and profiling plots |
| `manifest.py` | Environment and git manifest capture |

## Install Benchmark Dependencies

```sh
poetry install --with bench
```

Optional benchmark backends are detected at runtime:

- `lightgbm`
- `xgboost`
- `numba`

If one is missing, registry entries that require it are marked skipped instead
of failing the whole run.

## Run Benchmarks

Short suite:

```sh
python -m benchmarks.run --suite short --part all --no-show-plots
```

Full extended suite:

```sh
python -m benchmarks.run \
  --suite full \
  --part all \
  --include-extended-datasets \
  --trials 5 \
  --bigk-trials 5 \
  --resplit-trials \
  --output-dir results/benchmarks/full_extended_safe_speed_$(date +%Y%m%d_%H%M%S) \
  --no-show-plots
```

Focused run for public speed paths:

```sh
python -m benchmarks.run \
  --suite full \
  --part all \
  --include-extended-datasets \
  --trials 5 \
  --bigk-trials 5 \
  --resplit-trials \
  --methods Baseline,PublicParallelSeparate,BigK:Baseline,BigK:PublicParallelSeparate,PublicLightGBMTreeLearner+CappedLS \
  --no-show-plots
```

## Outputs

Each run writes to `results/benchmarks/<timestamp>/` unless `--output-dir` is
provided.

| Output | Description |
| --- | --- |
| `env_manifest.json` | Python/package versions, thread env vars, git commit, dirty status, CLI args |
| `ngboost_univariate_results.csv` | Univariate Normal benchmark results |
| `ngboost_bigk_results.csv` | Multivariate Normal BigK stress-test results |
| `ngboost_profiling_results.csv` | Per-component timing breakdown |
| `uni_*.png` | Per-cell univariate timing plots |
| `pareto_*.png` | Speed/accuracy Pareto plots |
| `bigk_T*.png` | BigK scaling plots |
| `profile_*.png` | Profiling breakdown plots |

The runner flushes partial CSVs as cells complete, so interrupted runs still
leave inspectable output.

## Experiment Registries

Registries live in `benchmarks/variants.py`:

| Registry | Used by |
| --- | --- |
| `UNI_EXPERIMENTS` | Full univariate suite |
| `SHORT_UNI_EXPERIMENTS` | Short univariate suite |
| `BIGK_EXPERIMENTS` | Full BigK suite |
| `SHORT_BIGK_EXPERIMENTS` | Short BigK suite |
| `PROFILE_EXPERIMENTS` | Profiling suite |
| `STRUCTURAL_SMOKE_METHODS` | Benchmark structural smoke tests |

Each registry entry is a tuple of:

```python
(method_name, estimator_class, extra_kwargs)
```

or, for optional dependencies:

```python
(method_name, estimator_class, extra_kwargs, backend_name)
```

Use explicit method names for public API paths. For example:

- `PublicParallelSeparate`
- `BigK:PublicParallelSeparate`
- `PublicLightGBMTreeLearner+CappedLS`

This keeps public API evidence separate from benchmark-only subclasses such as
`ParallelFitBaseNGBRegressor` or `LightGBMPersistentNGBRegressor`.

## Method Boundaries

Production candidates:

- Normal inline logpdf/log-score and exact Normal diagonal natural gradient.
- `fit_base_mode="parallel_separate"` for deterministic, thread-safe base
  learners, especially high-parameter distributions.

Optional behavior-changing accelerators:

- `line_search_strategy="loss_checked_capped"`.
- `LightGBMTreeLearner`.

Benchmark-only research variants:

- Shared multi-output trees.
- FixedStep.
- PreBinnedNumba.
- Generic DiagNG/GlobalNG.
- NewtonLeaves and Fisher heuristics.
- Persistent LightGBM benchmark estimator.

## Adding A Benchmark Method

1. Add the estimator or wrapper to `benchmarks/variants.py`.
2. Add a registry entry with a clear method name.
3. If it requires an optional dependency, add or reuse a backend status key.
4. Add a structural smoke entry when the estimator should support
   `predict`, `pred_dist`, and staged prediction.
5. Add tests in `tests/test_benchmarks.py` or `tests/test_speed_features.py`
   when the result will be cited for production API behavior.

## Reproducibility Rules

Before citing a result:

- Check `env_manifest.json`.
- Confirm `git.dirty` is `false`.
- Confirm `git.commit_short` is the commit being discussed.
- Confirm the benchmark method is the implementation being claimed.
- Report thread settings and optional backend versions.
- State whether the method is exact, optional non-equivalent, or benchmark-only.

