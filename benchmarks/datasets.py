"""Dataset loading for benchmark runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_diabetes,
    make_regression,
)
from sklearn.model_selection import train_test_split

GLOBAL_SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[1]
UCI_DATA_DIR = REPO_ROOT / "data" / "uci"

OPENML_DATASETS = [
    ("Concrete", 4353, None),
    ("EnergyEfficiency", 1471, None),
    ("Kin8nm", 189, None),
    ("PowerPlant", 41704, None),
    ("WineQualityRed", 287, None),
    ("Yacht", 44970, None),
    ("Abalone", 183, None),
    ("BikeSharing", 42712, "count"),
    ("Sulfur", 23515, None),
    ("CPUAct", 197, None),
    ("Elevators", 216, None),
]


def make_heteroskedastic(
    n_samples: int,
    n_features: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a regression problem with input-dependent noise."""

    rng = np.random.RandomState(random_state)
    x = rng.randn(n_samples, n_features)
    signal = x[:, 0] + 0.5 * x[:, 1] ** 2
    noise_scale = 0.5 + np.abs(x[:, 0])
    y = signal + rng.randn(n_samples) * noise_scale
    return x, y


def _split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(x, y, test_size=test_size, random_state=GLOBAL_SEED)


def _load_openml_numeric(
    data_id: int,
    target_col: str | None = None,
    *,
    max_samples: int = 20000,
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch an OpenML dataset by ID, keeping only numeric columns."""

    try:
        bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    except TypeError:
        bunch = fetch_openml(data_id=data_id, as_frame=True)

    frame = bunch.frame
    if frame is None:
        raise ValueError(f"OpenML {data_id}: frame is None")

    if target_col is None:
        target_col = bunch.target_names[0] if bunch.target_names else frame.columns[-1]
    if target_col not in frame.columns:
        raise ValueError(f"OpenML {data_id}: target '{target_col}' not in columns")

    y = pd.to_numeric(frame[target_col], errors="coerce")
    x_frame = frame.drop(columns=[target_col]).select_dtypes(include=[np.number])
    if x_frame.shape[1] == 0:
        raise ValueError(f"OpenML {data_id}: no numeric features after filtering")

    combined = pd.concat([x_frame, y.rename("__target__")], axis=1).dropna()
    if len(combined) < 100:
        raise ValueError(
            f"OpenML {data_id}: only {len(combined)} rows after dropping NaN"
        )

    x = combined.drop(columns=["__target__"]).values.astype(np.float64)
    y = combined["__target__"].values.astype(np.float64)
    if x.shape[0] > max_samples:
        rng = np.random.RandomState(GLOBAL_SEED)
        idx = rng.choice(x.shape[0], max_samples, replace=False)
        x, y = x[idx], y[idx]
    return x, y


def _load_local_uci_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load benchmark datasets already vendored in the repository."""

    local = {}

    kin8nm = pd.read_csv(UCI_DATA_DIR / "kin8nm.csv")
    local["Kin8nm"] = (
        kin8nm.iloc[:, :-1].values.astype(np.float64),
        kin8nm.iloc[:, -1].values.astype(np.float64),
    )

    naval = pd.read_csv(
        UCI_DATA_DIR / "naval-propulsion.txt",
        delim_whitespace=True,
        header=None,
    ).iloc[:, :-1]
    local["Naval"] = (
        naval.iloc[:, :-1].values.astype(np.float64),
        naval.iloc[:, -1].values.astype(np.float64),
    )

    power = pd.read_excel(UCI_DATA_DIR / "power-plant.xlsx")
    local["PowerPlant"] = (
        power.iloc[:, :-1].values.astype(np.float64),
        power.iloc[:, -1].values.astype(np.float64),
    )

    protein = pd.read_csv(UCI_DATA_DIR / "protein.csv")[
        ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "RMSD"]
    ]
    local["Protein"] = (
        protein.iloc[:, :-1].values.astype(np.float64),
        protein.iloc[:, -1].values.astype(np.float64),
    )

    return local


def build_datasets(
    *,
    include_extended: bool = False,
    selected_names: set[str] | None = None,
    verbose: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Build benchmark datasets, optionally including network-backed datasets."""

    datasets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def add_dataset(name: str, x: np.ndarray, y: np.ndarray) -> None:
        if selected_names is not None and name not in selected_names:
            return
        datasets[name] = _split_dataset(x, y)
        if verbose:
            print(f"  {name}: {x.shape}")

    if verbose:
        print("Loading datasets...")

    for n_samples, n_features in [(2000, 8), (10000, 8), (10000, 32), (25000, 8)]:
        name = f"Synth-{n_samples // 1000}k-P{n_features}"
        x, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=0.1,
            random_state=GLOBAL_SEED,
        )
        add_dataset(name, x, y)

    x, y = make_heteroskedastic(10000, 8, GLOBAL_SEED)
    add_dataset("Heteroskedastic-10k", x, y)

    diabetes_x, diabetes_y = load_diabetes(return_X_y=True)
    add_dataset("Diabetes", diabetes_x, diabetes_y)

    try:
        for name, (x, y) in _load_local_uci_datasets().items():
            add_dataset(name, x, y)
    except Exception as exc:
        if verbose:
            print(f"  [SKIP] Local UCI datasets: {exc}")

    if include_extended:
        try:
            california_x, california_y = fetch_california_housing(return_X_y=True)
            if california_x.shape[0] > 20000:
                rng = np.random.RandomState(GLOBAL_SEED)
                idx = rng.choice(california_x.shape[0], 20000, replace=False)
                california_x = california_x[idx]
                california_y = california_y[idx]
            add_dataset("CaliforniaHousing", california_x, california_y)
        except Exception as exc:
            if verbose:
                print(f"  [SKIP] CaliforniaHousing: {exc}")

        for name, data_id, target_col in OPENML_DATASETS:
            if selected_names is not None and name not in selected_names:
                continue
            if name in datasets:
                continue
            try:
                x, y = _load_openml_numeric(data_id, target_col)
                add_dataset(name, x, y)
            except Exception as exc:
                if verbose:
                    print(f"  [SKIP] {name} (OpenML {data_id}): {exc}")

    if selected_names is not None:
        missing = sorted(selected_names - set(datasets))
        if missing:
            raise ValueError(
                "Requested datasets were not available: " + ", ".join(missing)
            )

    return datasets
