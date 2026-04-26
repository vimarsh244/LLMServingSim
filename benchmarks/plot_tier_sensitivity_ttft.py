#!/usr/bin/env python3
"""Recreate tier sensitivity plot for mean TTFT across storage tiers.

Reads metric summary rows from tier policy matrix outputs and renders one panel per
workload in the requested order.

Example:
  python benchmarks/plot_tier_sensitivity_ttft.py \
    --root output/tiered_kv/tier_policy_matrix \
    --workloads sharegpt_1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = ROOT_DIR / "output" / "tiered_kv" / "tier_policy_matrix"

TIER_ORDER = ["cpu_dram", "cxl", "ssd", "pcie_nvme", "ethernet"]
TIER_LABELS = {
    "cpu_dram": "CPU DRAM",
    "cxl": "CXL",
    "ssd": "SSD",
    "pcie_nvme": "PCIe NVMe",
    "ethernet": "Ethernet",
}

POLICY_ORDER = ["fifo", "largest_kv", "evicpress", "smallest_kv", "random", "lru", "tail", "harp"]

POLICY_STYLE = {
    "fifo": {"color": "#9a9a9a", "marker": "h"},
    "largest_kv": {"color": "#8C564B", "marker": "X"},
    "evicpress": {"color": "#1B9E77", "marker": "s"},
    "smallest_kv": {"color": "#5AB4D5", "marker": "v"},
    "random": {"color": "#E69F00", "marker": "P"},
    "lru": {"color": "#D95F02", "marker": "^"},
    "tail": {"color": "#CC79A7", "marker": "D"},
    "harp": {"color": "#1F78B4", "marker": "o"},
}

POLICY_LABEL = {
    "fifo": "FIFO",
    "largest_kv": "Largest-KV",
    "evicpress": "EvicPress",
    "smallest_kv": "Smallest-KV",
    "random": "Random",
    "lru": "LRU",
    "tail": "Tail",
    "harp": "HARP",
}

WORKLOAD_LABEL = {
    "fixed_256": "Fixed-256",
    "sharegpt_100": "ShareGPT-100",
    "sharegpt_300": "ShareGPT-300",
    "sharegpt_750": "ShareGPT-750",
    "sharegpt_1000": "ShareGPT-1000",
    "sharegpt_1500": "ShareGPT-1500",
    "prefix_stress": "Prefix-Stress",
}


def _setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
        }
    )


def _load_metric_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing metric summary: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Metric summary is empty: {path}")

    for col in ["tier", "policy", "workload", "status"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    if "mean_ttft_ms" not in df.columns:
        raise ValueError("Expected column mean_ttft_ms in metric summary")

    df["mean_ttft_ms"] = pd.to_numeric(df["mean_ttft_ms"], errors="coerce")
    df = df.dropna(subset=["mean_ttft_ms"])

    # Keep successful runs only and normalize policy aliases.
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    df["policy"] = df["policy"].replace({"oldest": "fifo"})

    # Average any duplicate reruns for (tier, policy, workload).
    df = (
        df.groupby(["tier", "policy", "workload"], as_index=False)["mean_ttft_ms"]
        .mean()
        .reset_index(drop=True)
    )
    return df


def _plot_one_workload(ax: plt.Axes, df: pd.DataFrame, workload: str, show_ylabel: bool) -> str:
    wdf = df[df["workload"] == workload].copy()
    x = np.arange(len(TIER_ORDER), dtype=float)

    policies = [p for p in POLICY_ORDER if p in set(wdf["policy"].unique())]
    if not policies:
        ax.set_axis_off()
        return ""

    width = min(0.82 / max(1, len(policies)), 0.16)
    all_deltas = []
    baseline_tiers_used = []

    for i, policy in enumerate(policies):
        pdf = wdf[wdf["policy"] == policy]
        if pdf.empty:
            continue

        tier_values = {}
        for tier in TIER_ORDER:
            cell = pdf[pdf["tier"] == tier]
            if cell.empty:
                continue
            val = float(cell["mean_ttft_ms"].mean())
            if np.isfinite(val):
                tier_values[tier] = val

        if not tier_values:
            continue

        baseline_tier = "cpu_dram" if "cpu_dram" in tier_values else next((t for t in TIER_ORDER if t in tier_values), None)
        if baseline_tier is None:
            continue
        baseline_tiers_used.append(baseline_tier)

        baseline = float(tier_values[baseline_tier])
        if not np.isfinite(baseline) or baseline <= 0.0:
            continue

        deltas = []
        for tier in TIER_ORDER:
            if tier not in tier_values:
                deltas.append(np.nan)
                continue
            val = float(tier_values[tier])
            if not np.isfinite(val):
                deltas.append(np.nan)
                continue
            deltas.append((val - baseline) / baseline * 100.0)

        pos = x - 0.41 + width / 2 + i * width
        ax.bar(
            pos,
            deltas,
            width=width * 0.95,
            color=POLICY_STYLE[policy]["color"],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.5,
            label=POLICY_LABEL[policy],
        )

        all_deltas.extend([v for v in deltas if np.isfinite(v)])

    ax.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.8)
    ax.set_title(f"LLaMA-3 8B - {WORKLOAD_LABEL.get(workload, workload)}", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABELS[t] for t in TIER_ORDER], rotation=12, ha="right")
    ax.grid(True, axis="y", alpha=0.28, linestyle="--")
    ax.grid(False, axis="x")

    if show_ylabel:
        ax.set_ylabel("Mean TTFT change vs CPU DRAM (%)")

    if all_deltas:
        lo = min(all_deltas)
        hi = max(all_deltas)
        span = max(8.0, hi - lo)
        pad = 0.12 * span
        ax.set_ylim(lo - pad, hi + pad)

    if baseline_tiers_used:
        uniq = sorted(set(baseline_tiers_used), key=lambda t: TIER_ORDER.index(t))
        if len(uniq) == 1:
            return uniq[0]
        return ",".join(uniq)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate tier sensitivity TTFT change bar plot")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="tier_policy_matrix root directory",
    )
    parser.add_argument(
        "--metric-csv",
        type=Path,
        default=None,
        help="Optional explicit metric summary CSV path",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["sharegpt_1000"],
        help="Workloads to include as panels (example: sharegpt_1000)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <root>/plots/tier_sensitivity_mean_ttft_<workloads>.png)",
    )
    args = parser.parse_args()

    metric_csv = args.metric_csv or (args.root / "metric_summary.csv")
    workloads = [w.strip().lower() for w in args.workloads if w and w.strip()]

    _setup_style()
    df = _load_metric_summary(metric_csv)

    missing = [w for w in workloads if w not in set(df["workload"].unique())]
    if missing:
        raise ValueError(f"Workloads not found in metric summary: {missing}")

    n = len(workloads)
    fig, axes = plt.subplots(1, n, figsize=(8.8 * n, 4.9), squeeze=False)

    fig.suptitle("Tier Sensitivity Analysis - Mean TTFT Change Across Storage Tiers", fontsize=16, weight="bold", y=1.02)
    fig.text(
        0.5,
        0.98,
        "Each bar shows percent change in mean TTFT relative to each policy's baseline tier (0%).",
        ha="center",
        va="top",
        fontsize=10,
        style="italic",
        color="#555555",
    )

    baseline_notes = []
    for i, wl in enumerate(workloads):
        b = _plot_one_workload(axes[0, i], df, wl, show_ylabel=(i == 0))
        if b and b != "cpu_dram":
            baseline_notes.append(f"{WORKLOAD_LABEL.get(wl, wl)} uses {TIER_LABELS.get(b, b)} baseline")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Policy", loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=min(8, len(labels)), frameon=True)

    if baseline_notes:
        fig.text(0.5, 0.04, "; ".join(baseline_notes), ha="center", va="bottom", fontsize=9, color="#444444")

    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.92])

    out_path = args.out
    if out_path is None:
        safe_tag = "_".join(workloads)
        out_path = args.root / "plots" / f"tier_sensitivity_mean_ttft_change_bar_{safe_tag}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
