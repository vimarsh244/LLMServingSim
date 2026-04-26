#!/usr/bin/env python3
"""
Plot total simulation time scaling for provided ShareGPT workload sizes.

Usage:
  python benchmarks/plot_total_simulation_times.py
  python benchmarks/plot_total_simulation_times.py --out-dir figures/simulation_time_scaling
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT_DIR / "figures" / "simulation_time_scaling"


# Provided totals from simulation logs.
DATA_POINTS = [
    {"workload": "sharegpt100", "requests": 100, "seconds": 40.938},
    {"workload": "sharegpt300", "requests": 300, "seconds": 183.308},
    {"workload": "sharegpt750", "requests": 750, "seconds": 1144.539},
    {"workload": "sharegpt1000", "requests": 1000, "seconds": 1835.157},
    {"workload": "sharegpt1500", "requests": 1500, "seconds": 4116.779},
]


def _safe_style() -> None:
    for style in ["seaborn-v0_8-whitegrid", "ggplot", "default"]:
        try:
            plt.style.use(style)
            return
        except OSError:
            continue


def _format_hms(total_seconds: float) -> str:
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours}h {minutes}m {seconds:.3f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot total simulation time scaling for ShareGPT workloads")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where plot and source table are written",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _safe_style()

    df = pd.DataFrame(DATA_POINTS).sort_values("requests").reset_index(drop=True)
    df["minutes"] = df["seconds"] / 60.0
    df["hms"] = df["seconds"].apply(_format_hms)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        df["requests"],
        df["minutes"],
        color="#0f766e",
        marker="o",
        linewidth=2.5,
        markersize=7,
    )
    ax.fill_between(df["requests"], df["minutes"], color="#14b8a6", alpha=0.12)

    for row in df.itertuples(index=False):
        ax.annotate(
            f"{row.hms}",
            (row.requests, row.minutes),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    ax.set_xticks(df["requests"])
    ax.set_xlabel("Workload size (ShareGPT requests)")
    ax.set_ylabel("Total simulation time (minutes)")
    ax.set_title("Simulation Time Scaling vs Workload Size")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    png_path = args.out_dir / "sharegpt_total_simulation_time_scaling.png"
    pdf_path = args.out_dir / "sharegpt_total_simulation_time_scaling.pdf"
    csv_path = args.out_dir / "sharegpt_total_simulation_time_points.csv"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    df.to_csv(csv_path, index=False)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()