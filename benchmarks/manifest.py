"""Environment capture for benchmark reproducibility."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
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
REPO_ROOT = Path(__file__).resolve().parents[1]


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT INSTALLED"


def _git_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _git_info() -> dict[str, object]:
    status = _git_output(["status", "--short"])
    return {
        "commit": _git_output(["rev-parse", "HEAD"]),
        "commit_short": _git_output(["rev-parse", "--short", "HEAD"]),
        "branch": _git_output(["branch", "--show-current"]),
        "dirty": bool(status),
        "status_short": status,
    }


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def write_env_manifest(path: str | Path, run_config=None) -> None:
    """Persist Python, package, and thread settings for a benchmark run."""

    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in MANIFEST_PACKAGES},
        "env_threads": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        "git": _git_info(),
        "argv": sys.argv,
    }
    if run_config is not None:
        manifest["run_config"] = _jsonable(run_config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
