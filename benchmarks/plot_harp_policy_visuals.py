#!/usr/bin/env python3
"""
plot_harp_policy_visuals.py - Focused visualizations for HARP eviction-policy outputs.

Reads per-request CSVs from a matrix output folder:
  output/tiered_kv/model_tier_policy_matrix/<accelerator>/<model>/<tier>/harp/<workload>/result.csv

Writes:
  <input-root>/harp_visuals/
    - harp_summary.csv
    - <model>_latency_overview.png
    - <model>_transition_breakdown.png
    - <model>_harp_behavior.png
    - harp_tradeoff_scatter.png

Usage:
  python benchmarks/plot_harp_policy_visuals.py
  python benchmarks/plot_harp_policy_visuals.py --input-root output/tiered_kv/model_tier_policy_matrix/A6000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from plot_utils import _setup_matplotlib

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_ROOT = ROOT_DIR / "output" / "tiered_kv" / "model_tier_policy_matrix" / "A6000"

BYTES_TO_MB = 1.0 / (1024.0 * 1024.0)
NS_TO_MS = 1e-6

TIER_ORDER = ["cpu_dram", "cxl", "ethernet", "pcie_nvme", "ssd"]
WORKLOAD_ORDER = [
    "fixed_256",
    "sharegpt_100",
    "sharegpt_300",
    "sharegpt_750",
    "sharegpt_1000",
    "sharegpt_1500",
]
MODEL_ORDER = ["llama8b", "phi_moe", "mixtral_8x7b", "llama70b"]

TIER_COLORS = {
    "cpu_dram": "#2E86AB",
    "cxl": "#4EA5D9",
    "ethernet": "#F6AA1C",
    "pcie_nvme": "#BC5090",
    "ssd": "#D1495B",
}

WORKLOAD_MARKERS = {
    "fixed_256": "o",
    "sharegpt_100": "^",
    "sharegpt_300": "s",
    "sharegpt_750": "D",
    "sharegpt_1000": "P",
    "sharegpt_1500": "X",
}


def _ordered(values: List[str], preferred: List[str]) -> List[str]:
    seen = set(values)
    out = [v for v in preferred if v in seen]
    out.extend(sorted(v for v in values if v not in out))
    return out


def _safe_mean_ms(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return np.nan
    return float(pd.to_numeric(df[col], errors="coerce").mean() * NS_TO_MS)


def _safe_p99_ms(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return np.nan
    series = pd.to_numeric(df[col], errors="coerce")
    return float(series.quantile(0.99) * NS_TO_MS)


def _sum_col(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def _collect_harp_summary(input_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    pattern = "*/*/harp/*/result.csv"

    for result_csv in sorted(input_root.glob(pattern)):
        rel = result_csv.relative_to(input_root)
        parts = rel.parts
        if len(parts) != 5:
            continue

        model, tier, _policy, workload, _file = parts

        df = pd.read_csv(result_csv)
        df.columns = df.columns.str.strip()

        nreq = float(len(df))
        if nreq <= 0:
            continue

        evict_cpu = _sum_col(df, "evict_npu_to_cpu_bytes")
        evict_cxl = _sum_col(df, "evict_npu_to_cxl_bytes")
        load_cpu = _sum_col(df, "load_cpu_to_npu_bytes")
        load_cxl = _sum_col(df, "load_cxl_to_npu_bytes")

        decode_tokens = _sum_col(df, "harp_decode_tokens")
        shadow_hit_tokens = _sum_col(df, "harp_shadow_hit_tokens")

        rows.append(
            {
                "model": model,
                "tier": tier,
                "workload": workload,
                "num_requests": nreq,
                "mean_ttft_ms": _safe_mean_ms(df, "TTFT"),
                "p99_ttft_ms": _safe_p99_ms(df, "TTFT"),
                "mean_tpot_ms": _safe_mean_ms(df, "TPOT"),
                "mean_latency_ms": _safe_mean_ms(df, "latency"),
                "p99_latency_ms": _safe_p99_ms(df, "latency"),
                "evict_npu_to_cpu_mb_per_req": (evict_cpu * BYTES_TO_MB) / nreq,
                "evict_npu_to_cxl_mb_per_req": (evict_cxl * BYTES_TO_MB) / nreq,
                "load_cpu_to_npu_mb_per_req": (load_cpu * BYTES_TO_MB) / nreq,
                "load_cxl_to_npu_mb_per_req": (load_cxl * BYTES_TO_MB) / nreq,
                "transition_mb_per_req": ((evict_cpu + evict_cxl + load_cpu + load_cxl) * BYTES_TO_MB) / nreq,
                "mean_stall_ms_per_req": _safe_mean_ms(df, "harp_stall_time_ns"),
                "mean_stall_events": float(pd.to_numeric(df.get("harp_stall_events", 0), errors="coerce").fillna(0).mean()),
                "mean_shadow_ratio": float(pd.to_numeric(df.get("harp_shadow_ratio", np.nan), errors="coerce").mean()),
                "shadow_hit_rate": (shadow_hit_tokens / decode_tokens) if decode_tokens > 0 else np.nan,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError(f"No HARP result.csv found under: {input_root}")

    summary["model"] = pd.Categorical(summary["model"], _ordered(summary["model"].unique().tolist(), MODEL_ORDER), ordered=True)
    summary["tier"] = pd.Categorical(summary["tier"], _ordered(summary["tier"].unique().tolist(), TIER_ORDER), ordered=True)
    summary["workload"] = pd.Categorical(summary["workload"], _ordered(summary["workload"].unique().tolist(), WORKLOAD_ORDER), ordered=True)

    return summary.sort_values(["model", "workload", "tier"]).reset_index(drop=True)


def _plot_latency_overview(summary: pd.DataFrame, out_dir: Path):
    plt, _sns = _setup_matplotlib()

    metrics = [
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("p99_ttft_ms", "P99 TTFT (ms)"),
        ("mean_tpot_ms", "Mean TPOT (ms)"),
        ("p99_latency_ms", "P99 Latency (ms)"),
    ]

    for model in summary["model"].cat.categories:
        mdf = summary[summary["model"] == model].copy()
        if mdf.empty:
            continue

        workloads = [w for w in summary["workload"].cat.categories if w in set(mdf["workload"].astype(str))]
        tiers = [t for t in summary["tier"].cat.categories if t in set(mdf["tier"].astype(str))]

        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        axes = axes.ravel()

        for ax, (metric, ylabel) in zip(axes, metrics):
            for wl in workloads:
                wdf = mdf[mdf["workload"].astype(str) == wl].set_index("tier")
                vals = [wdf.loc[t, metric] if t in wdf.index else np.nan for t in tiers]
                ax.plot(
                    tiers,
                    vals,
                    marker="o",
                    linewidth=2.4,
                    markersize=7,
                    label=wl,
                )

            ax.set_ylabel(ylabel)
            ax.set_xlabel("Tier")
            ax.grid(alpha=0.25)
            ax.tick_params(axis="x", rotation=20)

        axes[0].set_title("TTFT Profile")
        axes[1].set_title("Tail TTFT")
        axes[2].set_title("Token Latency")
        axes[3].set_title("End-to-End Tail Latency")

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, ncol=1, loc="best", frameon=True)

        fig.suptitle(f"{model} - HARP Latency Overview", fontsize=16, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_dir / f"{model}_latency_overview.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def _plot_transition_breakdown(summary: pd.DataFrame, out_dir: Path):
    plt, _sns = _setup_matplotlib()

    components = [
        ("evict_npu_to_cpu_mb_per_req", "Evict NPU->CPU", "#5DA5DA"),
        ("evict_npu_to_cxl_mb_per_req", "Evict NPU->CXL", "#F17CB0"),
        ("load_cpu_to_npu_mb_per_req", "Load CPU->NPU", "#60BD68"),
        ("load_cxl_to_npu_mb_per_req", "Load CXL->NPU", "#F5C04A"),
    ]

    for model in summary["model"].cat.categories:
        mdf = summary[summary["model"] == model].copy()
        if mdf.empty:
            continue

        workloads = [w for w in summary["workload"].cat.categories if w in set(mdf["workload"].astype(str))]
        fig, axes = plt.subplots(1, len(workloads), figsize=(7 * len(workloads), 5), squeeze=False)

        for i, wl in enumerate(workloads):
            ax = axes[0, i]
            wdf = mdf[mdf["workload"].astype(str) == wl].sort_values("tier")
            x = np.arange(len(wdf))
            bottoms = np.zeros(len(wdf), dtype=float)

            for col, label, color in components:
                vals = pd.to_numeric(wdf[col], errors="coerce").fillna(0).values
                ax.bar(x, vals, bottom=bottoms, color=color, edgecolor="white", linewidth=0.6, label=label)
                bottoms += vals

            ax.set_xticks(x)
            ax.set_xticklabels(wdf["tier"].astype(str).tolist(), rotation=20, ha="right")
            ax.set_ylabel("MB transferred per request")
            ax.set_title(f"{wl}")
            ax.grid(axis="y", alpha=0.25)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                ncol=2,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.01),
                frameon=True,
            )

        fig.suptitle(f"{model} - HARP Tier Transition Breakdown", fontsize=16, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_dir / f"{model}_transition_breakdown.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def _plot_harp_behavior(summary: pd.DataFrame, out_dir: Path):
    plt, _sns = _setup_matplotlib()

    for model in summary["model"].cat.categories:
        mdf = summary[summary["model"] == model].copy()
        if mdf.empty:
            continue

        workloads = [w for w in summary["workload"].cat.categories if w in set(mdf["workload"].astype(str))]
        fig, axes = plt.subplots(1, len(workloads), figsize=(7 * len(workloads), 5), squeeze=False)

        for i, wl in enumerate(workloads):
            ax = axes[0, i]
            wdf = mdf[mdf["workload"].astype(str) == wl].sort_values("tier")
            x = np.arange(len(wdf))

            shadow = pd.to_numeric(wdf["shadow_hit_rate"], errors="coerce").fillna(0).values * 100.0
            stall = pd.to_numeric(wdf["mean_stall_ms_per_req"], errors="coerce").fillna(0).values

            shadow_max = float(np.nanmax(shadow)) if len(shadow) else 0.0
            y_top = 100.0 if shadow_max >= 35.0 else max(5.0, np.ceil(shadow_max * 1.8))

            bars = ax.bar(
                x,
                shadow,
                color=[TIER_COLORS.get(t, "#999999") for t in wdf["tier"].astype(str).tolist()],
                edgecolor="white",
                linewidth=0.6,
                alpha=0.9,
            )
            ax.set_ylabel("Shadow hit rate (%)")
            ax.set_ylim(0, y_top)
            ax.set_xticks(x)
            ax.set_xticklabels(wdf["tier"].astype(str).tolist(), rotation=20, ha="right")
            ax.grid(axis="y", alpha=0.2)

            if float(np.nanmax(stall)) > 1e-9:
                ax2 = ax.twinx()
                ax2.plot(x, stall, color="#2E2E2E", marker="o", linewidth=2.0, label="stall ms/req")
                ax2.set_ylabel("Mean stall (ms/request)")
            else:
                ax.text(
                    0.98,
                    0.92,
                    "No HARP stalls observed",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    color="#444444",
                )

            for b, v in zip(bars, shadow):
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    b.get_height() + max(0.15, y_top * 0.015),
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            ax.set_title(f"{wl}")

        fig.suptitle(f"{model} - HARP Effectiveness and Stall Cost", fontsize=16, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_dir / f"{model}_harp_behavior.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def _plot_tradeoff_scatter(summary: pd.DataFrame, out_dir: Path):
    plt, _sns = _setup_matplotlib()

    df = summary.dropna(subset=["mean_ttft_ms", "transition_mb_per_req", "mean_tpot_ms"]).copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    model_colors = {
        model: color
        for model, color in zip(
            df["model"].cat.categories,
            ["#4C78A8", "#F58518", "#54A24B", "#B279A2"],
        )
    }

    highlight_rows = []
    for model in df["model"].cat.categories:
        mdf = df[df["model"] == model]
        if mdf.empty:
            continue

        for wl in sorted(mdf["workload"].astype(str).unique()):
            sub = mdf[mdf["workload"].astype(str) == wl]
            if sub.empty:
                continue
            marker = WORKLOAD_MARKERS.get(wl, "o")
            ax.scatter(
                sub["transition_mb_per_req"],
                sub["mean_ttft_ms"],
                s=70 + sub["mean_tpot_ms"].to_numpy() * 3.0,
                alpha=0.78,
                color=model_colors.get(model, "#777777"),
                marker=marker,
                edgecolor="white",
                linewidth=0.7,
            )

            highlight_rows.append(sub.loc[sub["mean_ttft_ms"].idxmin()])

    if highlight_rows:
        highlight_df = pd.DataFrame(highlight_rows).drop_duplicates(subset=["model", "workload", "tier"])
        for row in highlight_df.itertuples(index=False):
            ax.annotate(
                f"{row.model}: {row.workload} -> {row.tier}",
                xy=(float(row.transition_mb_per_req), float(row.mean_ttft_ms)),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.9,
            )

    ax.set_xlabel("Tier transition traffic (MB/request)")
    ax.set_ylabel("Mean TTFT (ms)")
    ax.set_title("HARP Tradeoff: TTFT vs Tier Traffic (bubble size = mean TPOT)")
    ax.grid(alpha=0.25)

    model_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=model_colors.get(model, "#777777"),
            markersize=8,
            label=str(model),
        )
        for model in df["model"].cat.categories
    ]
    workload_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=WORKLOAD_MARKERS.get(wl, "o"),
            color="#333333",
            linestyle="None",
            markersize=8,
            label=wl,
        )
        for wl in sorted(df["workload"].astype(str).unique())
    ]

    leg1 = ax.legend(handles=model_handles, title="Model", loc="lower left", frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=workload_handles, title="Workload", loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_dir / "harp_tradeoff_scatter.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate HARP-focused visuals from model-tier-policy matrix outputs")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Path to accelerator-level output root")
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional output directory (default: <input-root>/harp_visuals)")
    args = parser.parse_args()

    input_root = args.input_root
    out_dir = args.out_dir or (input_root / "harp_visuals")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _collect_harp_summary(input_root)
    summary_path = out_dir / "harp_summary.csv"
    summary.to_csv(summary_path, index=False)

    _plot_latency_overview(summary, out_dir)
    _plot_transition_breakdown(summary, out_dir)
    _plot_harp_behavior(summary, out_dir)
    _plot_tradeoff_scatter(summary, out_dir)

    print(f"Saved HARP summary: {summary_path}")
    print(f"Saved plots under: {out_dir}")


if __name__ == "__main__":
    main()
