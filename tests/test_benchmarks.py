import subprocess
import sys

import pytest

pytest.importorskip("pandas")

import pandas as pd  # noqa: E402

from benchmarks.runner import _flush_rows  # noqa: E402
from benchmarks.variants import (  # noqa: E402
    BackendStatus,
    FastMultiOutputNGBRegressor,
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


def test_flush_rows_writes_incremental_csv(tmp_path):
    output_path = tmp_path / "partial.csv"
    rows = [{"Dataset": "Synth-2k-P8", "Method": "Baseline", "Status": "ok"}]

    _flush_rows(rows, output_path)

    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert df.to_dict("records") == rows


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
