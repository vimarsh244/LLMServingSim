#!/usr/bin/env python3
"""
plot_tier_policy_matrix_visuals.py - Polished visuals for tier-policy matrix outputs.

Reads:
  output/tiered_kv/tier_policy_matrix/run_manifest.csv
  output/tiered_kv/tier_policy_matrix/metric_summary.csv

Writes:
  output/tiered_kv/tier_policy_matrix/plots/
    - run_status_counts.png
    - {tier}_absolute_metrics_heatmaps.png
        - {tier}_absolute_trends.png
        - {tier}_policy_scorecard_absolute.png
        - {tier}_combined_dashboard_absolute.png

Usage:
  python benchmarks/plot_tier_policy_matrix_visuals.py
  python benchmarks/plot_tier_policy_matrix_visuals.py \
      --root output/tiered_kv/tier_policy_matrix
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = ROOT_DIR / "output" / "tiered_kv" / "tier_policy_matrix"

DEFAULT_MANIFEST = DEFAULT_ROOT / "run_manifest.csv"
DEFAULT_METRIC = DEFAULT_ROOT / "metric_summary.csv"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "plots"

TIER_ORDER = ["cpu_dram", "cxl", "pcie_nvme", "ssd", "ethernet"]
WORKLOAD_ORDER = [
    "fixed_256",
    "sharegpt_100",
    "sharegpt_300",
    "sharegpt_750",
    "sharegpt_1000",
    "sharegpt_1500",
    "prefix_stress",
]
WORKLOAD_SHORT = {
    "fixed_256": "F256",
    "sharegpt_100": "SG100",
    "sharegpt_300": "SG300",
    "sharegpt_750": "SG750",
    "sharegpt_1000": "SG1000",
    "sharegpt_1500": "SG1500",
    "prefix_stress": "PFX",
}
POLICY_ORDER = ["tail", "harp", "largest_kv", "lru", "random", "smallest_kv", "oldest", "fifo", "evicpress"]

POLICY_COLORS: Dict[str, str] = {
    "tail": "#3A7CA5",
    "harp": "#2A9D8F",
    "largest_kv": "#E76F51",
    "lru": "#F4A261",
    "random": "#8D99AE",
    "smallest_kv": "#9C6644",
    "oldest": "#D62828",
    "fifo": "#6D597A",
    "evicpress": "#4D908E",
}


def _ordered(values: List[str], preferred: List[str]) -> List[str]:
    value_set = set(values)
    out = [v for v in preferred if v in value_set]
    out.extend(sorted(v for v in values if v not in out))
    return out


def _setup_style() -> None:
    for style_name in ["seaborn-v0_8-whitegrid", "ggplot", "default"]:
        try:
            plt.style.use(style_name)
            break
        except OSError:
            continue

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.family": "DejaVu Sans",
        }
    )


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{label} is empty: {path}")
    df.columns = df.columns.str.strip()
    return df


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_run_status(manifest_df: pd.DataFrame, out_dir: Path) -> Path | None:
    if "status" not in manifest_df.columns:
        return None

    status_order = ["ok", "skipped_existing", "dry_run", "failed", "timeout"]
    counts = manifest_df["status"].astype(str).value_counts()
    statuses = [s for s in status_order if s in counts.index] + [s for s in counts.index if s not in status_order]
    vals = [int(counts[s]) for s in statuses]

    color_map = {
        "ok": "#2A9D8F",
        "skipped_existing": "#577590",
        "dry_run": "#4D908E",
        "failed": "#E63946",
        "timeout": "#BC4749",
    }
    colors = [color_map.get(s, "#6C757D") for s in statuses]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bars = ax.bar(statuses, vals, color=colors, edgecolor="white", linewidth=0.9)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val}", ha="center", va="bottom", fontsize=11)

    total = sum(vals)
    ax.set_title(f"Run Status Overview (total runs: {total})")
    ax.set_xlabel("Status")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)

    out_path = out_dir / "run_status_counts.png"
    _save(fig, out_path)
    return out_path


def _draw_heatmap(
    ax: plt.Axes,
    pivot: pd.DataFrame,
    title: str,
    cmap: str,
    value_fmt: str,
    percent: bool = False,
    center_zero: bool = False,
):
    if pivot.empty:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(no data)")
        return None

    vals = pivot.values.astype(float)
    finite = np.isfinite(vals)
    if not finite.any():
        ax.set_axis_off()
        ax.set_title(f"{title}\n(no numeric data)")
        return None

    if center_zero:
        max_abs = float(np.nanmax(np.abs(vals)))
        if np.isclose(max_abs, 0.0):
            max_abs = 1.0
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
        im = ax.imshow(vals, aspect="auto", cmap=cmap, norm=norm)
    else:
        im = ax.imshow(vals, aspect="auto", cmap=cmap)

    ax.set_title(title)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticklabels(pivot.index)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isfinite(v):
                continue
            label = format(v, value_fmt)
            if percent:
                label += "%"
            color = "white" if center_zero and abs(v) > (0.55 * np.nanmax(np.abs(vals))) else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=8.5, color=color)

    return im


def _plot_absolute_metric_heatmaps(metric_df: pd.DataFrame, out_dir: Path) -> List[Path]:
    specs = [
        ("mean_ttft_ms", "Mean TTFT (ms)", "magma", ".0f"),
        ("mean_tpot_ms", "Mean TPOT (ms)", "viridis", ".1f"),
        ("mean_latency_ms", "Mean Latency (ms)", "plasma", ".0f"),
        ("p99_ttft_ms", "P99 TTFT (ms)", "magma", ".0f"),
        ("p99_tpot_ms", "P99 TPOT (ms)", "viridis", ".1f"),
        ("tier_transition_mb_total", "Tier Transition (MB)", "cividis", ".0f"),
    ]

    tiers = _ordered(metric_df["tier"].astype(str).unique().tolist(), TIER_ORDER)
    policies = _ordered(metric_df["policy"].astype(str).unique().tolist(), POLICY_ORDER)
    workloads = _ordered(metric_df["workload"].astype(str).unique().tolist(), WORKLOAD_ORDER)

    outputs: List[Path] = []
    for tier in tiers:
        tdf = metric_df[metric_df["tier"].astype(str) == tier].copy()
        if tdf.empty:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(22, 11))
        fig.suptitle(f"{tier}: Absolute Metrics by Policy and Workload", fontsize=17, y=1.01)

        for ax, (metric, title, cmap, fmt) in zip(axes.flat, specs):
            if metric not in tdf.columns:
                ax.set_axis_off()
                ax.set_title(f"{title}\n(missing)")
                continue

            pivot = (
                tdf.pivot_table(index="workload", columns="policy", values=metric, aggfunc="mean")
                .reindex(index=[w for w in workloads if w in tdf["workload"].astype(str).unique()])
                .reindex(columns=[p for p in policies if p in tdf["policy"].astype(str).unique()])
            )

            im = _draw_heatmap(ax, pivot, title, cmap=cmap, value_fmt=fmt)
            if im is not None:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cbar.ax.tick_params(labelsize=8)

        out_path = out_dir / f"{tier}_absolute_metrics_heatmaps.png"
        _save(fig, out_path)
        outputs.append(out_path)

    return outputs


def _plot_absolute_trends(metric_df: pd.DataFrame, out_dir: Path) -> List[Path]:
    specs = [
        ("mean_tpot_ms", "Mean TPOT (ms)"),
        ("p99_tpot_ms", "P99 TPOT (ms)"),
        ("tier_transition_mb_total", "Tier Transition (MB)"),
    ]

    tiers = _ordered(metric_df["tier"].astype(str).unique().tolist(), TIER_ORDER)
    policies = _ordered(metric_df["policy"].astype(str).unique().tolist(), POLICY_ORDER)
    workloads = _ordered(metric_df["workload"].astype(str).unique().tolist(), WORKLOAD_ORDER)
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

    outputs: List[Path] = []
    for tier in tiers:
        tdf = metric_df[metric_df["tier"].astype(str) == tier].copy()
        if tdf.empty:
            continue

        workloads_present = [w for w in workloads if w in set(tdf["workload"].astype(str))]
        x = np.arange(len(workloads_present), dtype=float)
        x_labels = [WORKLOAD_SHORT.get(w, w) for w in workloads_present]

        if not workloads_present:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(24, 6.8), sharex=True)
        fig.suptitle(f"{tier}: Absolute Trends Across Workloads", fontsize=17, y=1.02)

        for ax, (metric, ylabel) in zip(axes.flat, specs):
            if metric not in tdf.columns:
                ax.set_axis_off()
                ax.set_title(f"{ylabel}\n(missing)")
                continue

            for p_idx, policy in enumerate(policies):
                sub = tdf[tdf["policy"].astype(str) == policy]
                if sub.empty:
                    continue

                y_vals = []
                for wl in workloads_present:
                    cell = sub[sub["workload"].astype(str) == wl]
                    y_vals.append(float(cell[metric].mean()) if len(cell) > 0 else np.nan)

                if np.all(np.isnan(y_vals)):
                    continue

                ax.plot(
                    x,
                    y_vals,
                    marker=marker_cycle[p_idx % len(marker_cycle)],
                    linewidth=2.0,
                    markersize=6,
                    label=policy,
                    color=POLICY_COLORS.get(policy, "#666666"),
                    alpha=0.95,
                )

            ax.set_title(ylabel)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=0)
            ax.grid(alpha=0.25)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                title="Policy",
                ncol=1,
                loc="center left",
                bbox_to_anchor=(0.905, 0.5),
                frameon=True,
            )

        fig.subplots_adjust(left=0.06, right=0.88, bottom=0.12, wspace=0.24)

        out_path = out_dir / f"{tier}_absolute_trends.png"
        _save(fig, out_path)
        outputs.append(out_path)

    return outputs


def _plot_policy_scorecard_absolute(metric_df: pd.DataFrame, out_dir: Path) -> List[Path]:
    needed = ["mean_tpot_ms", "p99_tpot_ms", "tier_transition_mb_total"]
    if any(col not in metric_df.columns for col in needed):
        return []

    tiers = _ordered(metric_df["tier"].astype(str).unique().tolist(), TIER_ORDER)
    outputs: List[Path] = []

    for tier in tiers:
        tdf = metric_df[metric_df["tier"].astype(str) == tier].copy()
        if tdf.empty:
            continue

        grouped = tdf.groupby("policy", as_index=False)[needed].mean()
        for col in needed:
            vmin = float(grouped[col].min())
            vmax = float(grouped[col].max())
            if np.isclose(vmin, vmax):
                grouped[f"norm_{col}"] = 0.0
            else:
                grouped[f"norm_{col}"] = (grouped[col] - vmin) / (vmax - vmin)

        grouped["composite_score"] = grouped[[f"norm_{c}" for c in needed]].mean(axis=1)
        grouped = grouped.sort_values("composite_score", ascending=True).reset_index(drop=True)

        y = np.arange(len(grouped))
        vals = grouped["composite_score"].astype(float).values
        labels = grouped["policy"].astype(str).values
        colors = [POLICY_COLORS.get(lbl, "#666666") for lbl in labels]

        fig, ax = plt.subplots(figsize=(10.5, 6.0))
        bars = ax.barh(y, vals, color=colors, alpha=0.9, edgecolor="white", linewidth=0.9)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Composite score (0 best, 1 worst; normalized within tier)")
        ax.set_title(f"{tier}: Policy Scorecard (absolute metrics)")
        ax.grid(axis="x", alpha=0.25)

        for bar, v in zip(bars, vals):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", ha="left", fontsize=9)

        out_path = out_dir / f"{tier}_policy_scorecard_absolute.png"
        _save(fig, out_path)
        outputs.append(out_path)

    return outputs


def _plot_combined_dashboard_absolute(metric_df: pd.DataFrame, out_dir: Path) -> List[Path]:
    needed = ["mean_tpot_ms", "p99_tpot_ms", "tier_transition_mb_total"]
    if any(col not in metric_df.columns for col in needed):
        return []

    tiers = _ordered(metric_df["tier"].astype(str).unique().tolist(), TIER_ORDER)
    policies = _ordered(metric_df["policy"].astype(str).unique().tolist(), POLICY_ORDER)
    workloads = _ordered(metric_df["workload"].astype(str).unique().tolist(), WORKLOAD_ORDER)

    outputs: List[Path] = []

    for tier in tiers:
        tdf = metric_df[metric_df["tier"].astype(str) == tier].copy()
        if tdf.empty:
            continue

        workloads_present = [w for w in workloads if w in set(tdf["workload"].astype(str))]
        policies_present = [p for p in policies if p in set(tdf["policy"].astype(str))]

        if not workloads_present or not policies_present:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(
            f"{tier}: Combined Dashboard (absolute metrics)",
            fontsize=18,
            y=0.99,
        )

        # Panel 1: Mean TPOT heatmap
        pivot = (
            tdf.pivot_table(index="policy", columns="workload", values="mean_tpot_ms", aggfunc="mean")
            .reindex(index=policies_present)
            .reindex(columns=workloads_present)
        )
        im = _draw_heatmap(
            axes[0, 0],
            pivot,
            "Mean TPOT (ms)",
            cmap="viridis",
            value_fmt=".1f",
        )
        axes[0, 0].set_xticklabels([WORKLOAD_SHORT.get(w, w) for w in workloads_present], rotation=0)
        if im is not None:
            cbar = fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=8)

        # Panel 2: P99 TPOT heatmap
        pivot = (
            tdf.pivot_table(index="policy", columns="workload", values="p99_tpot_ms", aggfunc="mean")
            .reindex(index=policies_present)
            .reindex(columns=workloads_present)
        )
        im = _draw_heatmap(
            axes[0, 1],
            pivot,
            "P99 TPOT (ms)",
            cmap="magma",
            value_fmt=".1f",
        )
        axes[0, 1].set_xticklabels([WORKLOAD_SHORT.get(w, w) for w in workloads_present], rotation=0)
        if im is not None:
            cbar = fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=8)

        # Panel 3: Tier transition heatmap
        pivot = (
            tdf.pivot_table(index="policy", columns="workload", values="tier_transition_mb_total", aggfunc="mean")
            .reindex(index=policies_present)
            .reindex(columns=workloads_present)
        )
        im = _draw_heatmap(
            axes[1, 0],
            pivot,
            "Tier Transition (MB)",
            cmap="cividis",
            value_fmt=".0f",
        )
        axes[1, 0].set_xticklabels([WORKLOAD_SHORT.get(w, w) for w in workloads_present], rotation=0)
        if im is not None:
            cbar = fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=8)

        # Panel 4: Composite scorecard (absolute)
        agg = (
            tdf.groupby("policy", as_index=False)[needed]
            .mean()
            .set_index("policy")
            .reindex(policies_present)
        )

        for col in needed:
            vmin = float(agg[col].min())
            vmax = float(agg[col].max())
            if np.isclose(vmin, vmax):
                agg[f"norm_{col}"] = 0.0
            else:
                agg[f"norm_{col}"] = (agg[col] - vmin) / (vmax - vmin)

        agg["composite_score"] = agg[[f"norm_{c}" for c in needed]].mean(axis=1)
        comp = agg.reset_index().sort_values("composite_score", ascending=True)

        y = np.arange(len(comp), dtype=float)
        comp_colors = [POLICY_COLORS.get(p, "#666666") for p in comp["policy"].astype(str)]
        axes[1, 1].barh(
            y,
            comp["composite_score"].values,
            color=comp_colors,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
        )
        axes[1, 1].set_yticks(y)
        axes[1, 1].set_yticklabels(comp["policy"].tolist())
        axes[1, 1].set_title("Composite Policy Score (absolute)")
        axes[1, 1].set_xlabel("Score (0 best, 1 worst)")
        axes[1, 1].grid(axis="x", alpha=0.25)

        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.07, top=0.92, wspace=0.25, hspace=0.28)
        out_path = out_dir / f"{tier}_combined_dashboard_absolute.png"
        _save(fig, out_path)
        outputs.append(out_path)

    return outputs


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate polished visuals for tier_policy_matrix outputs")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Path to output/tiered_kv/tier_policy_matrix")
    parser.add_argument("--manifest-csv", type=Path, default=None, help="Optional explicit run_manifest.csv path")
    parser.add_argument("--metric-csv", type=Path, default=None, help="Optional explicit metric_summary.csv path")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for generated plots")
    args = parser.parse_args()

    root = args.root
    manifest_path = args.manifest_csv or (root / "run_manifest.csv")
    metric_path = args.metric_csv or (root / "metric_summary.csv")
    out_dir = args.out_dir or (root / "plots")

    _setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = _load_csv(manifest_path, "Run manifest")
    metric_df = _load_csv(metric_path, "Metric summary")

    _coerce_numeric(
        metric_df,
        [
            "mean_ttft_ms",
            "mean_tpot_ms",
            "mean_latency_ms",
            "p99_ttft_ms",
            "p99_tpot_ms",
            "tier_transition_mb_total",
        ],
    )

    generated: List[Path] = []

    status_plot = _plot_run_status(manifest_df, out_dir)
    if status_plot is not None:
        generated.append(status_plot)

    generated.extend(_plot_absolute_metric_heatmaps(metric_df, out_dir))
    generated.extend(_plot_absolute_trends(metric_df, out_dir))
    generated.extend(_plot_policy_scorecard_absolute(metric_df, out_dir))
    generated.extend(_plot_combined_dashboard_absolute(metric_df, out_dir))

    print(f"Saved {len(generated)} plot files to: {out_dir}")
    for path in generated:
        print(f" - {path}")


if __name__ == "__main__":
    main()
