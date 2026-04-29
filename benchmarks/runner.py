"""Execution engine for NGBoost benchmark runs."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from benchmarks.datasets import build_datasets
from benchmarks.manifest import write_env_manifest
from benchmarks.metrics import compute_metrics
from benchmarks.plotting import plot_bigk, plot_pareto, plot_profiling, plot_univariate
from benchmarks.profiling import run_profiling
from benchmarks.variants import (
    BIGK_N_ESTIMATORS,
    BIGK_SIZES,
    DEFAULT_BASE,
    GLOBAL_SEED,
    HYPER_CONFIGS,
    Normal,
    get_all_method_names,
    get_backend_statuses,
    get_bigk_experiments,
    get_mvn_class,
    get_univariate_experiments,
    smoke_test_all,
    warm_up_ngboost_paths,
    warm_up_numba,
)

CSV_FILENAMES = {
    "univariate": "ngboost_univariate_results.csv",
    "bigk": "ngboost_bigk_results.csv",
    "profiling": "ngboost_profiling_results.csv",
}


@dataclass
class BenchmarkRunConfig:
    suite: str
    parts: tuple[str, ...]
    include_extended_datasets: bool
    output_dir: Path
    show_plots: bool
    trials: int
    bigk_trials: int
    selected_methods: set[str] | None = None
    selected_datasets: set[str] | None = None
    selected_configs: set[str] | None = None
    run_structural_smoke: bool = True
    verbose: bool = True
    randomize_method_order: bool = True
    order_seed: int = GLOBAL_SEED
    resplit_trials: bool = False


def _fit_eval_once(
    estimator_cls,
    fit_kwargs,
    x_train,
    y_train,
    x_test,
    y_test,
    metric_fn,
    trial,
):
    """Fit/evaluate one estimator trial."""

    kwargs = dict(fit_kwargs)
    random_state = kwargs.setdefault("random_state", GLOBAL_SEED + trial)

    fit_call_kwargs = {}
    fit_x, fit_y = x_train, y_train
    early_stopping_rounds = kwargs.get("early_stopping_rounds")
    validation_fraction = kwargs.get("validation_fraction")
    if early_stopping_rounds is not None and validation_fraction not in (None, 0.0):
        fit_x, x_val, fit_y, y_val = train_test_split(
            x_train,
            y_train,
            test_size=validation_fraction,
            random_state=random_state,
        )
        fit_call_kwargs = {"X_val": x_val, "Y_val": y_val}

    model = estimator_cls(**kwargs)
    start = time.perf_counter()
    model.fit(fit_x, fit_y, **fit_call_kwargs)
    elapsed = time.perf_counter() - start

    return elapsed, metric_fn(model, x_test, y_test)


def _aggregate_trial_results(all_times, all_metrics) -> dict[str, float]:
    """Aggregate per-trial timings and metric dictionaries."""

    result = {
        "Time_mean": float(np.mean(all_times)),
        "Time_std": float(np.std(all_times)),
    }
    all_keys = sorted(set().union(*[metrics.keys() for metrics in all_metrics]))
    for key in all_keys:
        values = [metrics.get(key, np.nan) for metrics in all_metrics]
        result[f"{key}_mean"] = float(np.nanmean(values))
        result[f"{key}_std"] = float(np.nanstd(values))

    n_stages = result.get("n_stages_mean", np.nan)
    result["Time_per_stage"] = (
        result["Time_mean"] / n_stages
        if np.isfinite(n_stages) and n_stages > 0
        else np.nan
    )
    return result


def run_one(
    estimator_cls, fit_kwargs, x_train, y_train, x_test, y_test, metric_fn, trials
):
    """Run one estimator across multiple trials and aggregate results."""

    all_times = []
    all_metrics = []
    for trial in range(trials):
        elapsed, metrics = _fit_eval_once(
            estimator_cls,
            fit_kwargs,
            x_train,
            y_train,
            x_test,
            y_test,
            metric_fn,
            trial,
        )
        all_times.append(elapsed)
        all_metrics.append(metrics)
    return _aggregate_trial_results(all_times, all_metrics)


def _base_row(status: str = "ok", detail: str = "") -> dict[str, object]:
    return {"Status": status, "Detail": detail}


def _flush_rows(
    rows: list[dict[str, object]],
    output_path: Path | None,
    transform=None,
) -> None:
    """Persist currently completed rows so interrupted runs leave usable CSVs."""

    if output_path is None:
        return
    dataframe = pd.DataFrame(rows)
    if transform is not None:
        dataframe = transform(dataframe)
    dataframe.to_csv(output_path, index=False)


def _select_methods(specs, selected_methods: set[str] | None):
    if selected_methods is None:
        return list(specs)
    return [spec for spec in specs if spec.name in selected_methods]


def _select_configs(selected_configs: set[str] | None):
    if selected_configs is None:
        return HYPER_CONFIGS
    filtered = [config for config in HYPER_CONFIGS if config["tag"] in selected_configs]
    missing = sorted(selected_configs - {config["tag"] for config in HYPER_CONFIGS})
    if missing:
        raise ValueError("Unknown config tags: " + ", ".join(missing))
    return filtered


def _ordered_specs(specs, config: BenchmarkRunConfig, *keys):
    """Return specs in deterministic randomized order for a benchmark cell."""

    ordered = list(specs)
    if not config.randomize_method_order or len(ordered) <= 1:
        return ordered
    seed_material = "|".join([str(config.order_seed), *[str(key) for key in keys]])
    seed = sum((idx + 1) * ord(char) for idx, char in enumerate(seed_material))
    rng = np.random.RandomState(seed % (2**32 - 1))
    rng.shuffle(ordered)
    return ordered


def _combine_splits(x_train, x_test, y_train, y_test):
    if sparse.issparse(x_train) or sparse.issparse(x_test):
        x_all = sparse.vstack([x_train, x_test])
    else:
        x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([np.asarray(y_train), np.asarray(y_test)], axis=0)
    return x_all, y_all


def _trial_split(x_train, x_test, y_train, y_test, *, trial: int, resplit: bool):
    if not resplit:
        return x_train, x_test, y_train, y_test
    x_all, y_all = _combine_splits(x_train, x_test, y_train, y_test)
    test_size = len(y_test) / len(y_all)
    return train_test_split(
        x_all,
        y_all,
        test_size=test_size,
        random_state=GLOBAL_SEED + trial,
    )


def _runs_part(config: BenchmarkRunConfig, part: str) -> bool:
    return "all" in config.parts or part in config.parts


def _needs_tabular_datasets(config: BenchmarkRunConfig) -> bool:
    return _runs_part(config, "univariate") or _runs_part(config, "profiling")


def _compute_univariate_deltas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["Speedup"] = np.nan
    df["Delta_NLL"] = np.nan
    df["Delta_RMSE"] = np.nan

    for (dataset_name, config_name), group in df.groupby(["Dataset", "Config"]):
        baseline = group[(group["Method"] == "Baseline") & (group["Status"] == "ok")]
        if baseline.empty:
            continue
        base_time = baseline["Time_mean"].iloc[0]
        base_nll = baseline["NLL_mean"].iloc[0]
        base_rmse = baseline["RMSE_mean"].iloc[0]
        mask = (df["Dataset"] == dataset_name) & (df["Config"] == config_name)
        df.loc[mask, "Speedup"] = base_time / df.loc[mask, "Time_mean"]
        df.loc[mask, "Delta_NLL"] = df.loc[mask, "NLL_mean"] - base_nll
        df.loc[mask, "Delta_RMSE"] = df.loc[mask, "RMSE_mean"] - base_rmse

    return df


def _compute_bigk_speedup(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["Speedup"] = np.nan
    for (size, n_estimators), group in df.groupby(["Size", "T"]):
        baseline = group[
            (group["Method"] == "BigK:Baseline") & (group["Status"] == "ok")
        ]
        if baseline.empty:
            continue
        base_time = baseline["Time_mean"].iloc[0]
        mask = (df["Size"] == size) & (df["T"] == n_estimators)
        df.loc[mask, "Speedup"] = base_time / df.loc[mask, "Time_mean"]
    return df


def _print_univariate_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No univariate benchmark rows produced.")
        return

    display_cols = [
        "Dataset",
        "Config",
        "Method",
        "Status",
        "Time_mean",
        "Speedup",
        "RMSE_mean",
        "Delta_RMSE",
        "NLL_mean",
        "Delta_NLL",
        "n_stages_mean",
        "Time_per_stage",
        "Detail",
    ]
    print("\n" + "=" * 80)
    print("UNIVARIATE SUMMARY")
    print("=" * 80)
    with pd.option_context("display.max_rows", 500, "display.width", 220):
        print(
            df.sort_values(["Dataset", "Config", "Time_mean"], na_position="last")[
                [column for column in display_cols if column in df.columns]
            ].to_string(index=False)
        )


def _print_bigk_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No BigK benchmark rows produced.")
        return

    print("\n" + "=" * 80)
    print("BIGK SUMMARY")
    print("=" * 80)
    with pd.option_context("display.max_rows", 500, "display.width", 220):
        print(df.to_string(index=False))


def _aggregated_rows_from_accumulators(accumulators):
    rows = []
    for accumulator in accumulators.values():
        row = dict(accumulator["row"])
        if accumulator["status"] != "ok":
            row.update(_base_row(accumulator["status"], accumulator["detail"]))
            rows.append(row)
            continue
        row["Trials_completed"] = len(accumulator["times"])
        row["TrialSeeds"] = ",".join(str(seed) for seed in accumulator["seeds"])
        if accumulator["times"]:
            row.update(
                _aggregate_trial_results(
                    accumulator["times"],
                    accumulator["metrics"],
                )
            )
        rows.append(row)
    return rows


def _make_accumulator(row):
    return {
        "row": row,
        "status": row["Status"],
        "detail": row["Detail"],
        "times": [],
        "metrics": [],
        "seeds": [],
    }


def _run_univariate(
    config: BenchmarkRunConfig,
    datasets,
    backend_statuses,
    *,
    output_path: Path | None = None,
) -> pd.DataFrame:
    experiments = _select_methods(
        get_univariate_experiments(config.suite, backend_statuses=backend_statuses),
        config.selected_methods,
    )
    hyper_configs = _select_configs(config.selected_configs)
    if not experiments:
        return pd.DataFrame()

    total_fits = len(datasets) * len(hyper_configs) * len(experiments) * config.trials
    mode_label = config.suite.upper()
    print("\n" + "=" * 80)
    print(f"PART 1: UNIVARIATE NORMAL [{mode_label}]  (~{total_fits} fits)")
    print("=" * 80)

    rows = []
    for dataset_name, (x_train, x_test, y_train, y_test) in datasets.items():
        for hyper_config in hyper_configs:
            tag = hyper_config["tag"]
            print(f"\n[ {dataset_name} | {tag} ]")

            accumulators = {}
            for spec in experiments:
                row = {
                    "Dataset": dataset_name,
                    "Config": tag,
                    "Method": spec.name,
                    **_base_row(),
                }
                accumulators[spec.name] = _make_accumulator(row)
                if not spec.runnable:
                    accumulators[spec.name]["status"] = "skipped"
                    accumulators[spec.name]["detail"] = spec.skip_reason or ""

            for trial in range(config.trials):
                trial_x_train, trial_x_test, trial_y_train, trial_y_test = _trial_split(
                    x_train,
                    x_test,
                    y_train,
                    y_test,
                    trial=trial,
                    resplit=config.resplit_trials,
                )
                ordered_experiments = _ordered_specs(
                    experiments,
                    config,
                    "univariate",
                    dataset_name,
                    tag,
                    trial,
                )

                for spec in ordered_experiments:
                    accumulator = accumulators[spec.name]
                    if accumulator["status"] != "ok":
                        continue

                    print(
                        f"  [trial {trial + 1}/{config.trials}] {spec.name:<35}",
                        end=" ... ",
                        flush=True,
                    )

                    fit_kwargs = {
                        "n_estimators": hyper_config["n_estimators"],
                        "learning_rate": hyper_config["learning_rate"],
                        "verbose": False,
                        "Dist": Normal,
                        "Base": DEFAULT_BASE,
                        **spec.extra_kwargs,
                    }
                    if "early_stopping_rounds" in hyper_config:
                        fit_kwargs["early_stopping_rounds"] = hyper_config[
                            "early_stopping_rounds"
                        ]
                    if "validation_fraction" in hyper_config:
                        fit_kwargs["validation_fraction"] = hyper_config[
                            "validation_fraction"
                        ]

                    try:
                        elapsed, metrics = _fit_eval_once(
                            spec.estimator_cls,
                            fit_kwargs,
                            trial_x_train,
                            trial_y_train,
                            trial_x_test,
                            trial_y_test,
                            compute_metrics,
                            trial,
                        )
                        accumulator["times"].append(elapsed)
                        accumulator["metrics"].append(metrics)
                        accumulator["seeds"].append(GLOBAL_SEED + trial)
                        rows = [
                            *[
                                row
                                for row in rows
                                if not (
                                    row.get("Dataset") == dataset_name
                                    and row.get("Config") == tag
                                )
                            ],
                            *_aggregated_rows_from_accumulators(accumulators),
                        ]
                        _flush_rows(rows, output_path, _compute_univariate_deltas)
                        print(
                            f"{elapsed:7.2f}s "
                            f"| RMSE {metrics['RMSE']:8.3f} "
                            f"| NLL {metrics['NLL']:8.3f}"
                        )
                    except Exception as exc:
                        accumulator["status"] = "failed"
                        accumulator["detail"] = str(exc)
                        rows = [
                            *[
                                row
                                for row in rows
                                if not (
                                    row.get("Dataset") == dataset_name
                                    and row.get("Config") == tag
                                )
                            ],
                            *_aggregated_rows_from_accumulators(accumulators),
                        ]
                        _flush_rows(rows, output_path, _compute_univariate_deltas)
                        print(f"FAILED: {exc}")

            rows = [
                *[
                    row
                    for row in rows
                    if not (
                        row.get("Dataset") == dataset_name and row.get("Config") == tag
                    )
                ],
                *_aggregated_rows_from_accumulators(accumulators),
            ]
            _flush_rows(rows, output_path, _compute_univariate_deltas)

    return _compute_univariate_deltas(pd.DataFrame(rows))


def _run_bigk(
    config: BenchmarkRunConfig,
    backend_statuses,
    *,
    output_path: Path | None = None,
) -> pd.DataFrame:
    experiments = _select_methods(
        get_bigk_experiments(config.suite, backend_statuses=backend_statuses),
        config.selected_methods,
    )
    if not experiments:
        return pd.DataFrame()

    bigk_dist = get_mvn_class(5)
    total_fits = (
        len(BIGK_SIZES) * len(BIGK_N_ESTIMATORS) * len(experiments) * config.bigk_trials
    )
    mode_label = config.suite.upper()
    print("\n" + "=" * 80)
    print(f"PART 2: HIGH-DIM MVN(5) STRESS TEST [{mode_label}]  (~{total_fits} fits)")
    print("=" * 80)

    rows = []
    for size in BIGK_SIZES:
        fixed_x_all, fixed_y_all = make_regression(
            n_samples=size,
            n_features=8,
            n_targets=5,
            noise=0.1,
            random_state=GLOBAL_SEED,
        )
        fixed_x_train, fixed_x_test, fixed_y_train, fixed_y_test = train_test_split(
            fixed_x_all,
            fixed_y_all,
            test_size=0.2,
            random_state=GLOBAL_SEED,
        )

        for n_estimators in BIGK_N_ESTIMATORS:
            print(f"\n[ BigK N={size}, k=5, T={n_estimators} ]")
            accumulators = {}
            for spec in experiments:
                row = {
                    "Size": size,
                    "T": n_estimators,
                    "Method": spec.name,
                    **_base_row(),
                }
                accumulators[spec.name] = _make_accumulator(row)
                if not spec.runnable:
                    accumulators[spec.name]["status"] = "skipped"
                    accumulators[spec.name]["detail"] = spec.skip_reason or ""

            for trial in range(config.bigk_trials):
                if config.resplit_trials:
                    x_all, y_all = make_regression(
                        n_samples=size,
                        n_features=8,
                        n_targets=5,
                        noise=0.1,
                        random_state=GLOBAL_SEED + trial,
                    )
                    x_train, x_test, y_train, y_test = train_test_split(
                        x_all,
                        y_all,
                        test_size=0.2,
                        random_state=GLOBAL_SEED + trial,
                    )
                else:
                    x_train, x_test = fixed_x_train, fixed_x_test
                    y_train, y_test = fixed_y_train, fixed_y_test

                ordered_experiments = _ordered_specs(
                    experiments,
                    config,
                    "bigk",
                    size,
                    n_estimators,
                    trial,
                )

                for spec in ordered_experiments:
                    accumulator = accumulators[spec.name]
                    if accumulator["status"] != "ok":
                        continue

                    print(
                        f"  [trial {trial + 1}/{config.bigk_trials}] {spec.name:<35}",
                        end=" ... ",
                        flush=True,
                    )

                    fit_kwargs = {
                        "Dist": bigk_dist,
                        "n_estimators": n_estimators,
                        "learning_rate": 0.05,
                        "verbose": False,
                        "Base": DEFAULT_BASE,
                        **spec.extra_kwargs,
                    }
                    try:
                        elapsed, metrics = _fit_eval_once(
                            spec.estimator_cls,
                            fit_kwargs,
                            x_train,
                            y_train,
                            x_test,
                            y_test,
                            compute_metrics,
                            trial,
                        )
                        accumulator["times"].append(elapsed)
                        accumulator["metrics"].append(metrics)
                        accumulator["seeds"].append(GLOBAL_SEED + trial)
                        rows = [
                            *[
                                row
                                for row in rows
                                if not (
                                    row.get("Size") == size
                                    and row.get("T") == n_estimators
                                )
                            ],
                            *_aggregated_rows_from_accumulators(accumulators),
                        ]
                        _flush_rows(rows, output_path, _compute_bigk_speedup)
                        print(
                            f"{elapsed:7.2f}s "
                            f"| RMSE {metrics['RMSE']:8.3f} "
                            f"| NLL {metrics['NLL']:8.3f}"
                        )
                    except Exception as exc:
                        accumulator["status"] = "failed"
                        accumulator["detail"] = str(exc)
                        rows = [
                            *[
                                row
                                for row in rows
                                if not (
                                    row.get("Size") == size
                                    and row.get("T") == n_estimators
                                )
                            ],
                            *_aggregated_rows_from_accumulators(accumulators),
                        ]
                        _flush_rows(rows, output_path, _compute_bigk_speedup)
                        print(f"FAILED: {exc}")

            rows = [
                *[
                    row
                    for row in rows
                    if not (row.get("Size") == size and row.get("T") == n_estimators)
                ],
                *_aggregated_rows_from_accumulators(accumulators),
            ]
            _flush_rows(rows, output_path, _compute_bigk_speedup)

    return _compute_bigk_speedup(pd.DataFrame(rows))


def run_benchmarks(config: BenchmarkRunConfig) -> dict[str, pd.DataFrame]:
    """Run the selected benchmark parts and persist outputs."""

    warnings.filterwarnings("ignore")
    backend_statuses = get_backend_statuses()
    all_method_names = get_all_method_names(
        config.suite, backend_statuses=backend_statuses
    )
    if config.selected_methods is not None:
        missing_methods = sorted(config.selected_methods - all_method_names)
        if missing_methods:
            raise ValueError("Unknown method names: " + ", ".join(missing_methods))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_env_manifest(config.output_dir / "env_manifest.json", run_config=config)

    if config.run_structural_smoke:
        print("=" * 80)
        print("SMOKE TESTS")
        print("=" * 80)
        smoke_test_all(backend_statuses=backend_statuses, verbose=config.verbose)
        print()

    warm_up_numba(verbose=config.verbose)
    warm_up_ngboost_paths(verbose=config.verbose)

    datasets = {}
    if _needs_tabular_datasets(config):
        datasets = build_datasets(
            include_extended=config.include_extended_datasets,
            selected_names=config.selected_datasets,
            verbose=config.verbose,
        )
        print(f"Datasets: {list(datasets.keys())}")

    results = {
        "univariate": pd.DataFrame(),
        "bigk": pd.DataFrame(),
        "profiling": pd.DataFrame(),
    }

    generated_outputs = []

    if _runs_part(config, "univariate"):
        uni_path = config.output_dir / CSV_FILENAMES["univariate"]
        results["univariate"] = _run_univariate(
            config,
            datasets,
            backend_statuses,
            output_path=uni_path,
        )
        results["univariate"].to_csv(uni_path, index=False)
        generated_outputs.append(uni_path)
        _print_univariate_summary(results["univariate"])

    if _runs_part(config, "bigk"):
        bigk_path = config.output_dir / CSV_FILENAMES["bigk"]
        results["bigk"] = _run_bigk(config, backend_statuses, output_path=bigk_path)
        results["bigk"].to_csv(bigk_path, index=False)
        generated_outputs.append(bigk_path)
        _print_bigk_summary(results["bigk"])

    if _runs_part(config, "profiling"):
        print("\n" + "=" * 80)
        print("PART 3: PROFILING (per-component time breakdown)")
        print("=" * 80)
        profile_path = config.output_dir / CSV_FILENAMES["profiling"]
        results["profiling"] = run_profiling(
            datasets,
            backend_statuses=backend_statuses,
            selected_methods=config.selected_methods,
            output_path=profile_path,
            verbose=config.verbose,
        )
        results["profiling"].to_csv(profile_path, index=False)
        generated_outputs.append(profile_path)

    if not results["univariate"].empty:
        plot_univariate(
            results["univariate"], config.output_dir, show_plots=config.show_plots
        )
        plot_pareto(
            results["univariate"], config.output_dir, show_plots=config.show_plots
        )
    if not results["bigk"].empty:
        plot_bigk(results["bigk"], config.output_dir, show_plots=config.show_plots)
    if not results["profiling"].empty:
        plot_profiling(
            results["profiling"], config.output_dir, show_plots=config.show_plots
        )

    print("\nResults saved to:")
    for path in generated_outputs:
        print(f"  {path}")

    return results
