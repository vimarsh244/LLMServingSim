#!/usr/bin/env python3
"""
Create concise plots for HARP ablation outputs.

Inputs:
  output/tiered_kv/harp_ablation/sharegpt_1000_cpu_dram/metric_summary.csv
  output/tiered_kv/harp_ablation/sharegpt_1000_cpu_dram/delta_vs_baseline.csv

Outputs:
  output/tiered_kv/harp_ablation/sharegpt_1000_cpu_dram/plots/
    - harp_ablation_latency_line.png
    - harp_ablation_movement_line.png
    - harp_ablation_tradeoff_scatter.png
    - harp_ablation_delta_heatmap.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ABLATION_DIR = ROOT_DIR / "output" / "tiered_kv" / "harp_ablation" / "sharegpt_1000_cpu_dram"

RUN_ORDER = [
    "E0_baseline",
    "E1_no_stall",
    "E2_no_quality",
    "E3_no_fairness",
    "E4_fairness_high",
]


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    rank = {k: i for i, k in enumerate(RUN_ORDER)}
    return df.sort_values(by="experiment", key=lambda s: s.map(lambda v: rank.get(v, 999)))


def _plot_latency_line(df: pd.DataFrame, out_path: Path) -> None:
    x_labels = df["experiment"].tolist()
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax2 = ax.twinx()

    l1 = ax.plot(x, df["mean_ttft_ms"], marker="o", linewidth=2.2, color="#264653", label="Mean TTFT (ms)")
    l2 = ax2.plot(x, df["mean_tpot_ms"], marker="o", linewidth=2.2, color="#2A9D8F", label="Mean TPOT (ms)")
    l3 = ax2.plot(x, df["mean_itl_ms"], marker="o", linewidth=2.2, color="#E9C46A", label="Mean ITL (ms)")

    ax.set_title("HARP Ablation: Latency Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_ylabel("TTFT (ms)")
    ax2.set_ylabel("TPOT / ITL (ms)")
    ax.grid(alpha=0.25)

    lines = l1 + l2 + l3
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_movement_line(df: pd.DataFrame, out_path: Path) -> None:
    x_labels = df["experiment"].tolist()
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, df["tier_transition_mb_total"], marker="o", linewidth=2.2, label="Transition MB total")
    ax.plot(x, df["transition_mb_per_generated_token"], marker="o", linewidth=2.2, label="MB per generated token")

    ax.set_title("HARP Ablation: Data Movement")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_ylabel("MB / MB per token")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_tradeoff_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    x = df["tier_transition_mb_total"]
    y = df["mean_tpot_ms"]

    ax.scatter(x, y, s=80)
    for _, row in df.iterrows():
        ax.annotate(
            row["experiment"],
            (row["tier_transition_mb_total"], row["mean_tpot_ms"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
        )

    ax.set_xlabel("Tier transition total (MB)")
    ax.set_ylabel("Mean TPOT (ms)")
    ax.set_title("HARP Ablation Tradeoff: Movement vs TPOT")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_delta_heatmap(delta_df: pd.DataFrame, out_path: Path) -> None:
    metric_cols = [
        "delta_pct_mean_ttft_ms",
        "delta_pct_mean_tpot_ms",
        "delta_pct_mean_itl_ms",
        "delta_pct_tier_transition_mb_total",
        "delta_pct_transition_mb_per_generated_token",
        "delta_pct_shadow_hit_rate",
        "delta_pct_harp_avg_shadow_ratio",
    ]

    present = [c for c in metric_cols if c in delta_df.columns]
    if not present:
        return

    plot_df = delta_df[delta_df["experiment"] != "E0_baseline"].copy()
    plot_df = _ordered(plot_df)

    data = plot_df[present].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r")

    ax.set_title("HARP Ablation: Delta vs Baseline (%)")
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["experiment"].tolist())
    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels([c.replace("delta_pct_", "") for c in present], rotation=30, ha="right")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Delta %")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HARP ablation summaries")
    parser.add_argument("--ablation-dir", type=Path, default=DEFAULT_ABLATION_DIR)
    args = parser.parse_args()

    metric_path = args.ablation_dir / "metric_summary.csv"
    delta_path = args.ablation_dir / "delta_vs_baseline.csv"
    out_dir = args.ablation_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metric_path.exists():
        raise FileNotFoundError(f"Missing metric summary: {metric_path}")

    mdf = pd.read_csv(metric_path)
    mdf = _ordered(mdf)

    _plot_latency_line(mdf, out_dir / "harp_ablation_latency_line.png")
    _plot_movement_line(mdf, out_dir / "harp_ablation_movement_line.png")
    _plot_tradeoff_scatter(mdf, out_dir / "harp_ablation_tradeoff_scatter.png")

    if delta_path.exists():
        ddf = pd.read_csv(delta_path)
        ddf = _ordered(ddf)
        _plot_delta_heatmap(ddf, out_dir / "harp_ablation_delta_heatmap.png")

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
