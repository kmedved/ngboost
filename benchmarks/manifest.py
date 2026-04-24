"""Environment capture for benchmark reproducibility."""

from __future__ import annotations

import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)

MANIFEST_PACKAGES = (
    "ngboost",
    "scikit-learn",
    "numpy",
    "scipy",
    "pandas",
    "lightgbm",
    "xgboost",
    "numba",
    "threadpoolctl",
)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT INSTALLED"


def write_env_manifest(path: str | Path) -> None:
    """Persist Python, package, and thread settings for a benchmark run."""

    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in MANIFEST_PACKAGES},
        "env_threads": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
