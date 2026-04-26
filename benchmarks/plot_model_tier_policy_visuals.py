#!/usr/bin/env python3
"""
plot_model_tier_policy_visuals.py - Generate polished plots for model-tier-policy sweeps.

Reads:
  output/tiered_kv/model_tier_policy_matrix/metric_summary.csv

Writes (default):
  output/tiered_kv/model_tier_policy_matrix/plots/
    - {model}_mean_ttft_ms_grouped.png
    - {model}_mean_tpot_ms_grouped.png
    - {model}_p99_ttft_ms_grouped.png
    - {model}_tier_transition_mb_total_grouped.png
    - {model}_transition_heatmap.png
    - {model}_ttft_vs_tpot_tradeoff.png
    - leaderboard.csv

Usage:
  python benchmarks/plot_model_tier_policy_visuals.py
  python benchmarks/plot_model_tier_policy_visuals.py --workload sharegpt_100
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY = ROOT_DIR / "output" / "tiered_kv" / "model_tier_policy_matrix" / "metric_summary.csv"
DEFAULT_MANIFEST = ROOT_DIR / "output" / "tiered_kv" / "model_tier_policy_matrix" / "run_manifest.csv"
DEFAULT_OUT_DIR = ROOT_DIR / "output" / "tiered_kv" / "model_tier_policy_matrix" / "plots"

POLICY_COLORS = {
    "tail": "#4C78A8",
    "fifo": "#F58518",
    "oldest": "#F58518",
    "lru": "#FF9DA6",
    "largest_kv": "#54A24B",
    "smallest_kv": "#E45756",
    "random": "#B279A2",
    "evicpress": "#72B7B2",
    "harp": "#ECA400",
}

SHADOW_RATIO_MARKERS = [
    (0.90, "o", "shadow ratio >= 0.90"),
    (0.70, "s", "0.70-0.89"),
    (0.50, "^", "0.50-0.69"),
    (0.00, "D", "< 0.50"),
]


def _safe_style():
    # Fall back safely if style is not available.
    for style in ["seaborn-v0_8-whitegrid", "ggplot", "default"]:
        try:
            plt.style.use(style)
            return
        except OSError:
            continue


def _load_df(path: Path, workload: str | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metric summary not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Metric summary is empty. Run experiments first.")

    required = ["model", "tier", "policy", "workload"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Metric summary missing columns: {missing}")

    if workload:
        df = df[df["workload"] == workload].copy()
        if df.empty:
            raise ValueError(f"No rows found for workload='{workload}'")

    numeric_cols = [
        "mean_ttft_ms",
        "mean_tpot_ms",
        "p99_ttft_ms",
        "p99_tpot_ms",
        "mean_latency_ms",
        "tier_transition_mb_total",
        "transition_mb_per_generated_token",
        "stall_overlap_ratio",
        "mean_stall_ms_per_request",
        "shadow_hit_rate",
        "harp_avg_shadow_ratio",
        "evicpress_compression_events",
        "evicpress_compressed_bytes_saved",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _grouped_bar(
    df_model: pd.DataFrame,
    model: str,
    metric: str,
    ylabel: str,
    out_path: Path,
):
    pivot = (
        df_model.pivot_table(index="tier", columns="policy", values=metric, aggfunc="mean")
        .sort_index()
    )

    if pivot.empty:
        return

    tiers = list(pivot.index)
    policies = [p for p in pivot.columns if p in POLICY_COLORS] + [p for p in pivot.columns if p not in POLICY_COLORS]
    pivot = pivot[policies]

    x = np.arange(len(tiers))
    width = 0.85 / max(1, len(policies))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, policy in enumerate(policies):
        vals = pivot[policy].values
        positions = x - 0.425 + (i + 0.5) * width
        bars = ax.bar(
            positions,
            vals,
            width=width,
            label=policy,
            color=POLICY_COLORS.get(policy, "#999999"),
            alpha=0.92,
        )
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(tiers, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{model}: {metric} by Tier and Policy")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Policy", ncols=min(3, len(policies)), frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _heatmap(df_model: pd.DataFrame, model: str, out_path: Path):
    metric = "tier_transition_mb_total"
    pivot = (
        df_model.pivot_table(index="policy", columns="tier", values=metric, aggfunc="mean")
        .sort_index()
    )
    if pivot.empty:
        return

    vals = pivot.values
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(vals, aspect="auto", cmap="YlGnBu")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Tier Transition (MB)")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{model}: Policy/Tier Transition Heatmap")

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _tradeoff_scatter(df_model: pd.DataFrame, model: str, out_path: Path):
    needed = ["mean_ttft_ms", "mean_tpot_ms", "tier_transition_mb_total", "policy", "tier"]
    if any(c not in df_model.columns for c in needed):
        return

    plot_df = df_model.dropna(subset=["mean_ttft_ms", "mean_tpot_ms", "tier_transition_mb_total"]).copy()
    if plot_df.empty:
        return

    trans = plot_df["tier_transition_mb_total"].values
    max_trans = float(np.nanmax(trans)) if len(trans) else 1.0
    sizes = 60 + 540 * (trans / max(1e-9, max_trans))

    def _marker_for_ratio(ratio: float) -> str:
        for threshold, marker, _ in SHADOW_RATIO_MARKERS:
            if ratio >= threshold:
                return marker
        return "o"

    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        shadow_ratio = float(row.get("harp_avg_shadow_ratio", 1.0) or 1.0)
        ax.scatter(
            row["mean_tpot_ms"],
            row["mean_ttft_ms"],
            s=sizes[idx],
            alpha=0.75,
            color=POLICY_COLORS.get(row["policy"], "#999999"),
            marker=_marker_for_ratio(shadow_ratio),
            edgecolor="white",
            linewidth=0.8,
        )
        ax.text(
            row["mean_tpot_ms"],
            row["mean_ttft_ms"],
            f"{row['tier']}\n{row['policy']}\nr={shadow_ratio:.2f}",
            fontsize=7,
            ha="center",
            va="center",
        )

    ax.set_xlabel("Mean TPOT (ms)")
    ax.set_ylabel("Mean TTFT (ms)")
    ax.set_title(f"{model}: TTFT/TPOT Tradeoff (bubble size = transition MB)")
    ax.grid(alpha=0.3)

    # Manual policy legend.
    handles = []
    labels = []
    for p in sorted(plot_df["policy"].unique()):
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=POLICY_COLORS.get(p, "#999999"), markersize=8))
        labels.append(p)
    legend_policy = ax.legend(handles, labels, title="Policy", loc="best", frameon=True)
    ax.add_artist(legend_policy)

    marker_handles = []
    marker_labels = []
    for _, marker, label in SHADOW_RATIO_MARKERS:
        marker_handles.append(plt.Line2D([0], [0], marker=marker, color="black", linestyle="None", markersize=7))
        marker_labels.append(label)
    ax.legend(marker_handles, marker_labels, title="HARP shadow ratio", loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_harp_occupancy_timelines(manifest_path: Path, out_dir: Path, workload: str | None):
    if not manifest_path.exists():
        return

    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        return

    required = {"model", "tier", "policy", "workload", "status", "timeseries_csv"}
    if not required.issubset(set(manifest.columns)):
        return

    keep_status = {"ok", "skipped_existing"}
    rows = manifest[manifest["policy"] == "harp"]
    rows = rows[rows["status"].isin(keep_status)]
    if workload:
        rows = rows[rows["workload"] == workload]

    for rec in rows.itertuples(index=False):
        ts_path = Path(rec.timeseries_csv)
        if not ts_path.exists():
            continue
        ts_df = pd.read_csv(ts_path)
        needed_cols = {"sim_time_ns", "harp_hot_reqs", "harp_shadow_reqs", "harp_cold_reqs"}
        if not needed_cols.issubset(set(ts_df.columns)):
            continue

        fig, ax = plt.subplots(figsize=(11, 5.5))
        t = ts_df["sim_time_ns"].astype(float).values * 1e-9
        ax.plot(t, ts_df["harp_hot_reqs"].astype(float).values, color="#4C78A8", linewidth=2.0, label="Hot")
        ax.plot(t, ts_df["harp_shadow_reqs"].astype(float).values, color="#ECA400", linewidth=2.0, label="Shadow")
        ax.plot(t, ts_df["harp_cold_reqs"].astype(float).values, color="#E45756", linewidth=2.0, label="Cold")
        ax.fill_between(t, ts_df["harp_shadow_reqs"].astype(float).values, alpha=0.15, color="#ECA400")
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel("Active requests")
        ax.set_title(f"{rec.model} | {rec.tier} | {rec.workload}: HARP state occupancy")
        ax.grid(alpha=0.3)
        ax.legend(frameon=True)
        fig.tight_layout()

        out_path = out_dir / f"{rec.model}_{rec.tier}_{rec.workload}_harp_occupancy_timeline.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)


def _build_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    score_cols = ["mean_ttft_ms", "mean_tpot_ms", "tier_transition_mb_total"]
    agg_cols = [c for c in score_cols if c in df.columns]

    grouped = (
        df.groupby(["model", "tier", "policy"], as_index=False)[agg_cols]
        .mean()
    )

    # Normalize per model for fair comparisons.
    grouped["composite_score"] = 0.0
    for model, idxs in grouped.groupby("model").groups.items():
        sub = grouped.loc[idxs]
        score = np.zeros(len(sub), dtype=float)
        weights = {
            "mean_ttft_ms": 0.45,
            "mean_tpot_ms": 0.45,
            "tier_transition_mb_total": 0.10,
        }
        for col, w in weights.items():
            if col not in sub.columns:
                continue
            vals = sub[col].values.astype(float)
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            if np.isclose(vmax, vmin):
                norm = np.zeros_like(vals)
            else:
                norm = (vals - vmin) / (vmax - vmin)
            score += w * norm
        grouped.loc[idxs, "composite_score"] = score

    grouped = grouped.sort_values(["model", "composite_score", "mean_ttft_ms", "mean_tpot_ms"]).reset_index(drop=True)
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Plot rich visuals for model/tier/policy matrix outputs")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY, help="Path to metric_summary.csv")
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST, help="Path to run_manifest.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for plots")
    parser.add_argument(
        "--workload",
        type=str,
        default=None,
        help="Optional workload filter (e.g. sharegpt_100). If omitted, averages across all workloads.",
    )
    args = parser.parse_args()

    _safe_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_df(args.summary_csv, args.workload)

    # Average across workloads so visuals remain readable by default.
    agg_cols = [
        "mean_ttft_ms",
        "mean_tpot_ms",
        "p99_ttft_ms",
        "p99_tpot_ms",
        "mean_latency_ms",
        "tier_transition_mb_total",
        "transition_mb_per_generated_token",
        "stall_overlap_ratio",
        "mean_stall_ms_per_request",
        "shadow_hit_rate",
        "harp_avg_shadow_ratio",
        "evicpress_compression_events",
        "evicpress_compressed_bytes_saved",
    ]
    keep_cols = [c for c in agg_cols if c in df.columns]

    vis_df = (
        df.groupby(["model", "tier", "policy"], as_index=False)[keep_cols]
        .mean()
    )

    metric_specs = [
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("mean_tpot_ms", "Mean TPOT (ms)"),
        ("p99_ttft_ms", "P99 TTFT (ms)"),
        ("tier_transition_mb_total", "Tier Transition (MB)"),
        ("stall_overlap_ratio", "Stall Overlap Ratio"),
    ]

    models = sorted(vis_df["model"].unique())

    for model in models:
        mdf = vis_df[vis_df["model"] == model].copy()
        for metric, ylabel in metric_specs:
            if metric not in mdf.columns:
                continue
            out_path = args.out_dir / f"{model}_{metric}_grouped.png"
            _grouped_bar(mdf, model, metric, ylabel, out_path)

        _heatmap(mdf, model, args.out_dir / f"{model}_transition_heatmap.png")
        _tradeoff_scatter(mdf, model, args.out_dir / f"{model}_ttft_vs_tpot_tradeoff.png")

    _plot_harp_occupancy_timelines(args.manifest_csv, args.out_dir, args.workload)

    leaderboard = _build_leaderboard(vis_df)
    leaderboard_path = args.out_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    print(f"Saved plots to: {args.out_dir}")
    print(f"Saved leaderboard: {leaderboard_path}")


if __name__ == "__main__":
    main()
