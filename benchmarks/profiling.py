"""Profiling helpers for benchmark runs."""

from __future__ import annotations

import cProfile
import pstats
from pathlib import Path

import pandas as pd

from benchmarks.variants import (
    DEFAULT_BASE,
    GLOBAL_SEED,
    Normal,
    get_profile_experiments,
)


def profile_fit(estimator_cls, fit_kwargs, x_train, y_train):
    """Collect cProfile stats for one estimator fit."""

    model = estimator_cls(**fit_kwargs)
    profiler = cProfile.Profile()
    profiler.enable()
    model.fit(x_train, y_train)
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats("cumulative")
    breakdown = {}
    for (_, _, func_name), (
        _,
        _,
        total_time,
        cumulative_time,
        _,
    ) in stats.stats.items():
        if func_name in breakdown:
            prev_total, prev_cumulative = breakdown[func_name]
            breakdown[func_name] = (
                prev_total + total_time,
                prev_cumulative + cumulative_time,
            )
        else:
            breakdown[func_name] = (total_time, cumulative_time)
    return breakdown


def categorize_profile(breakdown: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Bucket raw profiler timings into higher-level categories."""

    exact_map = {
        "fit_base": "fit_base",
        "line_search": "line_search",
        "_capped_line_search": "line_search",
        "_fixed_line_search": "line_search",
        "_subsampled_line_search": "line_search",
        "_hybrid_line_search": "line_search",
        "_warmstart_line_search": "line_search",
        "_armijo_line_search": "line_search",
        "_line_search_core": "line_search",
        "total_score": "total_score",
        "sample": "sample",
        "predict": "predict",
        "pred_param": "predict",
    }
    linalg_keywords = ("dot", "solve", "inv", "cholesky", "svd", "linalg")
    fit_keywords = ("fit", "apply", "build", "splitter", "_fit")

    categories = {
        "fit_base": 0.0,
        "line_search": 0.0,
        "natural_grad": 0.0,
        "manifold_init": 0.0,
        "total_score": 0.0,
        "sample": 0.0,
        "predict": 0.0,
        "array_ops": 0.0,
        "other": 0.0,
    }

    for func_name, (total_time, _) in breakdown.items():
        lowered = func_name.lower()
        if lowered in exact_map:
            categories[exact_map[lowered]] += total_time
        elif ("natural" in lowered and "grad" in lowered) or lowered == "grad":
            categories["natural_grad"] += total_time
        elif lowered == "__init__" and total_time > 0.001:
            categories["manifold_init"] += total_time
        elif any(keyword in lowered for keyword in linalg_keywords):
            categories["array_ops"] += total_time
        elif any(keyword in lowered for keyword in fit_keywords):
            categories["fit_base"] += total_time
        else:
            categories["other"] += total_time

    return categories


def run_profiling(
    datasets: dict[str, tuple],
    *,
    backend_statuses=None,
    output_path: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Profile a subset of methods over a subset of datasets."""

    profile_datasets = {}
    for name in ["Synth-2k-P8", "Synth-10k-P32", "Heteroskedastic-10k", "Diabetes"]:
        if name in datasets:
            profile_datasets[name] = datasets[name]
    if not profile_datasets:
        for i, (name, value) in enumerate(datasets.items()):
            if i >= 2:
                break
            profile_datasets[name] = value

    rows = []
    hyper_config = {"n_estimators": 200, "learning_rate": 0.05}
    experiments = get_profile_experiments(backend_statuses=backend_statuses)

    for dataset_name, (x_train, _x_test, y_train, _y_test) in profile_datasets.items():
        if verbose:
            print(
                "\n  Profiling on "
                f"{dataset_name} (N={x_train.shape[0]}, P={x_train.shape[1]}, "
                f"T={hyper_config['n_estimators']})..."
            )

        for spec in experiments:
            base_row = {
                "Dataset": dataset_name,
                "Method": spec.name,
                "Status": "ok",
                "Detail": "",
            }
            if not spec.runnable:
                row = dict(base_row)
                row["Status"] = "skipped"
                row["Detail"] = spec.skip_reason or ""
                rows.append(row)
                if output_path is not None:
                    pd.DataFrame(rows).to_csv(output_path, index=False)
                if verbose:
                    print(f"    {spec.name:25s} SKIPPED: {row['Detail']}")
                continue

            fit_kwargs = {
                **hyper_config,
                "verbose": False,
                "Dist": Normal,
                "Base": DEFAULT_BASE,
                "random_state": GLOBAL_SEED,
                **spec.extra_kwargs,
            }
            try:
                breakdown = profile_fit(
                    spec.estimator_cls, fit_kwargs, x_train, y_train
                )
                categories = categorize_profile(breakdown)
                total = sum(categories.values())
                row = dict(base_row)
                row["Total_s"] = total
                for category, value in categories.items():
                    row[f"{category}_s"] = value
                    row[f"{category}_pct"] = 100.0 * value / total if total > 0 else 0.0
                rows.append(row)
                if output_path is not None:
                    pd.DataFrame(rows).to_csv(output_path, index=False)
                if verbose:
                    print(
                        f"    {spec.name:25s} total={total:.2f}s  "
                        f"fit_base={categories['fit_base']:.2f}s "
                        f"line_search={categories['line_search']:.2f}s "
                        f"natgrad={categories['natural_grad']:.2f}s "
                        f"total_score={categories['total_score']:.2f}s"
                    )
            except Exception as exc:
                row = dict(base_row)
                row["Status"] = "failed"
                row["Detail"] = str(exc)
                rows.append(row)
                if output_path is not None:
                    pd.DataFrame(rows).to_csv(output_path, index=False)
                if verbose:
                    print(f"    {spec.name:25s} FAILED: {exc}")

    return pd.DataFrame(rows)
