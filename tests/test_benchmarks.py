import json
import subprocess
import sys

import pytest

pytest.importorskip("pandas")

import pandas as pd  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

from benchmarks.manifest import write_env_manifest  # noqa: E402
from benchmarks.metrics import compute_metrics  # noqa: E402
from benchmarks.profiling import _select_profile_experiments  # noqa: E402
from benchmarks.runner import (  # noqa: E402
    BenchmarkRunConfig,
    _flush_rows,
    _needs_tabular_datasets,
    _ordered_specs,
    _trial_split,
)
from benchmarks.variants import (  # noqa: E402
    BackendStatus,
    CappedLSNGBRegressor,
    FastMultiOutputNGBRegressor,
    PreBinnedNumbaNGBRegressor,
    _line_search_core,
    get_mvn_class,
    get_profile_experiments,
    get_univariate_experiments,
)


def test_benchmark_registry_marks_missing_optional_backends_as_skipped():
    overrides = {
        "lightgbm": BackendStatus("lightgbm", False, "mock missing"),
        "numba": BackendStatus("numba", False, "mock missing"),
        "xgboost": BackendStatus("xgboost", False, "mock missing"),
    }

    experiments = get_univariate_experiments("short", backend_statuses=overrides)
    specs_by_name = {spec.name: spec for spec in experiments}

    assert specs_by_name["Baseline"].runnable
    assert specs_by_name["LightGBM+FixedStep"].skip_reason is not None
    assert specs_by_name["LightGBM+CappedLS"].skip_reason is not None
    assert specs_by_name["PreBinnedNumba+FixedStep"].skip_reason is not None
    assert specs_by_name["FN:PreBinnedNumba+FixedStep"].skip_reason is not None


def test_fast_multioutput_honors_public_validation_kwargs():
    import numpy as np

    x = np.random.RandomState(0).randn(80, 4)
    y = np.random.RandomState(1).randn(80)
    x_val = np.random.RandomState(2).randn(10, 4)
    y_val = np.random.RandomState(3).randn(11)

    model = FastMultiOutputNGBRegressor(
        n_estimators=3,
        early_stopping_rounds=1,
        validation_fraction=0.2,
        verbose=False,
    )
    with pytest.raises(ValueError):
        model.fit(x, y, X_val=x_val, Y_val=y_val)


def test_prebinned_numba_honors_public_validation_kwargs():
    import numpy as np

    x = np.random.RandomState(0).randn(80, 4)
    y = np.random.RandomState(1).randn(80)
    x_val = np.random.RandomState(2).randn(10, 4)
    y_val = np.random.RandomState(3).randn(11)

    model = PreBinnedNumbaNGBRegressor(
        n_estimators=3,
        early_stopping_rounds=1,
        validation_fraction=0.2,
        verbose=False,
    )
    with pytest.raises(ValueError):
        model.fit(x, y, X_val=x_val, Y_val=y_val)


def test_flush_rows_writes_incremental_csv(tmp_path):
    output_path = tmp_path / "partial.csv"
    rows = [{"Dataset": "Synth-2k-P8", "Method": "Baseline", "Status": "ok"}]

    _flush_rows(rows, output_path)

    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert df.to_dict("records") == rows


def test_bigk_only_runs_do_not_need_tabular_datasets(tmp_path):
    config = BenchmarkRunConfig(
        suite="short",
        parts=("bigk",),
        include_extended_datasets=False,
        output_dir=tmp_path,
        show_plots=False,
        trials=1,
        bigk_trials=1,
        selected_datasets={"NotAUnivariateDataset"},
    )

    assert not _needs_tabular_datasets(config)


def test_profile_method_selection_respects_selected_methods():
    experiments = get_profile_experiments()

    selected = _select_profile_experiments(experiments, {"Baseline"})

    assert [spec.name for spec in selected] == ["Baseline"]


def test_method_order_randomization_is_deterministic():
    experiments = get_univariate_experiments("short")
    config = BenchmarkRunConfig(
        suite="short",
        parts=("univariate",),
        include_extended_datasets=False,
        output_dir="unused",
        show_plots=False,
        trials=1,
        bigk_trials=1,
        order_seed=123,
    )

    first = _ordered_specs(experiments, config, "dataset", "config", 0)
    second = _ordered_specs(experiments, config, "dataset", "config", 0)
    different_trial = _ordered_specs(experiments, config, "dataset", "config", 1)

    assert [spec.name for spec in first] == [spec.name for spec in second]
    assert [spec.name for spec in first] != [spec.name for spec in experiments]
    assert [spec.name for spec in first] != [spec.name for spec in different_trial]


def test_resplit_trials_changes_train_test_split():
    import numpy as np

    x = np.arange(40).reshape(20, 2)
    y = np.arange(20)
    x_train, x_test = x[:16], x[16:]
    y_train, y_test = y[:16], y[16:]

    _, x_test_a, _, y_test_a = _trial_split(
        x_train,
        x_test,
        y_train,
        y_test,
        trial=0,
        resplit=True,
    )
    _, x_test_b, _, y_test_b = _trial_split(
        x_train,
        x_test,
        y_train,
        y_test,
        trial=1,
        resplit=True,
    )

    assert x_test_a.shape == x_test.shape
    assert y_test_a.shape == y_test.shape
    assert not np.array_equal(x_test_a, x_test_b)
    assert not np.array_equal(y_test_a, y_test_b)


def test_capped_line_search_falls_back_to_zero_when_loss_never_improves():
    import numpy as np

    class DummyManifold:
        def __init__(self, params):
            self.params = params

        def total_score(self, y, sample_weight=None):
            return float(np.mean(self.params**2))

    model = CappedLSNGBRegressor(n_estimators=1, verbose=False)
    model.Manifold = DummyManifold
    model.scalings = []
    model.tol = 0.0

    scale = _line_search_core(
        model,
        np.ones((5, 1)),
        np.zeros((5, 1)),
        np.zeros(5),
        None,
        1.0,
        max_up=0,
        max_down=1,
    )

    assert scale == 0.0
    assert model.scalings == [0.0]


def test_mvn_metrics_use_covariance_diagnostics():
    import numpy as np
    from ngboost import NGBRegressor

    rng = np.random.RandomState(0)
    x = rng.randn(80, 4)
    y = rng.randn(80, 2)
    dist = get_mvn_class(2)
    model = NGBRegressor(
        Dist=dist,
        n_estimators=2,
        learning_rate=0.05,
        verbose=False,
        Base=DecisionTreeRegressor(max_depth=1, random_state=0),
    )

    model.fit(x, y)
    metrics = compute_metrics(model, x[:10], y[:10])

    assert "cov_eig_min_min" in metrics
    assert "cov_condition_p99" in metrics
    assert "mahalanobis_mean" in metrics
    assert "scale_min" not in metrics


def test_env_manifest_captures_git_and_run_config(tmp_path):
    config = BenchmarkRunConfig(
        suite="short",
        parts=("univariate",),
        include_extended_datasets=False,
        output_dir=tmp_path,
        show_plots=False,
        trials=1,
        bigk_trials=1,
        selected_methods={"Baseline"},
    )
    manifest_path = tmp_path / "env_manifest.json"

    write_env_manifest(manifest_path, run_config=config)

    manifest = json.loads(manifest_path.read_text())
    assert "git" in manifest
    assert "argv" in manifest
    assert manifest["run_config"]["selected_methods"] == ["Baseline"]
    assert manifest["run_config"]["output_dir"] == str(tmp_path)


@pytest.mark.slow
def test_benchmark_cli_smoke(tmp_path):
    output_dir = tmp_path / "bench"
    command = [
        sys.executable,
        "-m",
        "benchmarks.run",
        "--suite",
        "short",
        "--part",
        "univariate",
        "--datasets",
        "Synth-2k-P8",
        "--configs",
        "T50_lr0.10",
        "--methods",
        "Baseline,MO",
        "--trials",
        "1",
        "--output-dir",
        str(output_dir),
        "--no-show-plots",
        "--skip-structural-smoke",
    ]
    subprocess.run(command, check=True)

    manifest_path = output_dir / "env_manifest.json"
    csv_path = output_dir / "ngboost_univariate_results.csv"

    assert manifest_path.exists()
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert not df.empty
    assert {
        "Dataset",
        "Config",
        "Method",
        "Status",
        "Time_mean",
        "RMSE_mean",
        "NLL_mean",
    } <= set(df.columns)
    assert set(df["Method"]) == {"Baseline", "MO"}
