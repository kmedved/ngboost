"""CLI entrypoint for the NGBoost benchmark suite."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

for env_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(env_var, "4")


from benchmarks.runner import BenchmarkRunConfig, run_benchmarks  # noqa: E402


def _parse_csv_set(raw_value: str | None) -> set[str] | None:
    if not raw_value:
        return None
    return {item.strip() for item in raw_value.split(",") if item.strip()}


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "benchmarks" / timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run NGBoost speed/accuracy benchmarks."
    )
    parser.add_argument(
        "--suite",
        choices=("short", "full"),
        default="short",
        help="Benchmark suite size.",
    )
    parser.add_argument(
        "--part",
        choices=("all", "univariate", "bigk", "profiling"),
        default="all",
        help="Benchmark subset to run.",
    )
    parser.add_argument(
        "--include-extended-datasets",
        action="store_true",
        help="Include network-backed CaliforniaHousing/OpenML datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where CSVs, plots, and manifests are written.",
    )
    parser.add_argument(
        "--show-plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show plots interactively in addition to saving them.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=2,
        help="Number of trials for univariate benchmarks.",
    )
    parser.add_argument(
        "--bigk-trials",
        type=int,
        default=1,
        help="Number of trials for BigK benchmarks.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated benchmark method names to include.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names to include.",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Comma-separated hyper-config tags to include.",
    )
    parser.add_argument(
        "--skip-structural-smoke",
        action="store_true",
        help="Skip the internal benchmark structural smoke test.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce benchmark progress logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = args.output_dir or _default_output_dir()
    config = BenchmarkRunConfig(
        suite=args.suite,
        parts=(args.part,),
        include_extended_datasets=args.include_extended_datasets,
        output_dir=output_dir,
        show_plots=args.show_plots,
        trials=args.trials,
        bigk_trials=args.bigk_trials,
        selected_methods=_parse_csv_set(args.methods),
        selected_datasets=_parse_csv_set(args.datasets),
        selected_configs=_parse_csv_set(args.configs),
        run_structural_smoke=not args.skip_structural_smoke,
        verbose=not args.quiet,
    )

    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=int(os.environ.get("OMP_NUM_THREADS", "4"))):
            return run_benchmarks(config)
    except ImportError:
        print("threadpoolctl not installed - relying on env vars for thread pinning")
        return run_benchmarks(config)


if __name__ == "__main__":
    main()
