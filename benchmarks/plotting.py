"""Plotting helpers for benchmark reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_univariate(
    df: pd.DataFrame, output_dir: str | Path, *, show_plots: bool
) -> None:
    """Plot fit time comparisons for univariate experiments."""

    output_dir = Path(output_dir)
    ok = df[df["Status"] == "ok"] if "Status" in df.columns else df
    for (dataset_name, config_name), group in ok.groupby(["Dataset", "Config"]):
        subset = group.dropna(subset=["Time_mean"]).sort_values("Time_mean")
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(14, max(6, len(subset) * 0.45)))
        colors = []
        for _, row in subset.iterrows():
            if row.get("Delta_NLL", 0) > 0.5:
                colors.append("#ff9999")
            elif row.get("Speedup", 1) > 2:
                colors.append("#66bb6a")
            else:
                colors.append("#90caf9")

        bars = ax.barh(subset["Method"], subset["Time_mean"], color=colors)
        ax.set_xlabel("Fit Time (s)")
        ax.set_title(f"NGBoost Variants - {dataset_name} [{config_name}]")
        ax.grid(axis="x", alpha=0.3)

        for bar, (_, row) in zip(bars, subset.iterrows()):
            speedup = row.get("Speedup", np.nan)
            delta_nll = row.get("Delta_NLL", np.nan)
            label = f" {row['Time_mean']:.2f}s"
            if not np.isnan(speedup):
                label += f" (x{speedup:.1f})"
            if not np.isnan(delta_nll):
                label += f"  dNLL={delta_nll:+.2f}"
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                fontsize=8,
            )

        plt.tight_layout()
        filename = output_dir / f"uni_{dataset_name}_{config_name}.png"
        plt.savefig(filename, dpi=150)
        if show_plots:
            plt.show()
        plt.close()


def plot_bigk(df: pd.DataFrame, output_dir: str | Path, *, show_plots: bool) -> None:
    """Plot BigK stress test timings."""

    output_dir = Path(output_dir)
    ok = df[df["Status"] == "ok"] if "Status" in df.columns else df
    if ok.empty:
        return

    for t_value in ok["T"].unique():
        subset = ok[ok["T"] == t_value]
        fig, ax = plt.subplots(figsize=(10, 6))
        for method in subset["Method"].unique():
            method_df = subset[subset["Method"] == method]
            ax.plot(method_df["Size"], method_df["Time_mean"], marker="o", label=method)
        ax.set_title(f"BigK MVN(5) Stress Test - T={t_value}")
        ax.set_xlabel("N (samples)")
        ax.set_ylabel("Fit Time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()
        filename = output_dir / f"bigk_T{t_value}.png"
        plt.savefig(filename, dpi=150)
        if show_plots:
            plt.show()
        plt.close()


def plot_pareto(df: pd.DataFrame, output_dir: str | Path, *, show_plots: bool) -> None:
    """Plot speedup vs NLL tradeoffs."""

    output_dir = Path(output_dir)
    ok = df[df["Status"] == "ok"] if "Status" in df.columns else df
    for (dataset_name, config_name), group in ok.groupby(["Dataset", "Config"]):
        subset = group.dropna(subset=["Speedup", "Delta_NLL"])
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(subset["Speedup"], subset["Delta_NLL"], s=60, zorder=5)
        for _, row in subset.iterrows():
            ax.annotate(
                row["Method"],
                (row["Speedup"], row["Delta_NLL"]),
                fontsize=7,
                ha="left",
                va="bottom",
                xytext=(4, 4),
                textcoords="offset points",
            )

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(1, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Speedup (x)")
        ax.set_ylabel("Delta NLL (lower = better)")
        ax.set_title(f"Pareto: Speed vs Accuracy - {dataset_name} [{config_name}]")
        ax.grid(True, alpha=0.3)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.fill_between([1, xlim[1]], ylim[0], 0, alpha=0.05, color="green")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        plt.tight_layout()
        filename = output_dir / f"pareto_{dataset_name}_{config_name}.png"
        plt.savefig(filename, dpi=150)
        if show_plots:
            plt.show()
        plt.close()


def plot_profiling(
    df: pd.DataFrame, output_dir: str | Path, *, show_plots: bool
) -> None:
    """Plot profiling time breakdowns."""

    output_dir = Path(output_dir)
    ok = df[df["Status"] == "ok"] if "Status" in df.columns else df
    if ok.empty:
        return

    categories = [
        column.replace("_s", "")
        for column in ok.columns
        if column.endswith("_s") and column != "Total_s"
    ]
    for dataset_name, group in ok.groupby("Dataset"):
        fig, ax = plt.subplots(figsize=(12, 6))
        methods = group["Method"].values
        x = np.arange(len(methods))
        bottom = np.zeros(len(methods))
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))

        for category, color in zip(categories, colors):
            column = f"{category}_s"
            if column not in group.columns:
                continue
            values = group[column].values
            ax.bar(x, values, bottom=bottom, label=category, color=color, width=0.6)
            bottom += values

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_ylabel("Time (seconds)")
        ax.set_title(f"Profile Breakdown - {dataset_name} (T=200, lr=0.05)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

        plt.tight_layout()
        filename = output_dir / f"profile_{dataset_name}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close()
