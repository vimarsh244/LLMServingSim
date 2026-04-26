#!/usr/bin/env python3
"""
Build presentation-quality plots from raw A6000 matrix outputs
(model / tier / policy / workload folders with result.csv + result_tier_stats.json).

Writes to: output/tiered_kv/model_tier_policy_matrix/A6000/plots/

Usage:
  python benchmarks/plot_a6000_tier_policy_visuals.py
  python benchmarks/plot_a6000_tier_policy_visuals.py --root output/tiered_kv/model_tier_policy_matrix/A6000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_A6000_ROOT = ROOT_DIR / "output" / "tiered_kv" / "model_tier_policy_matrix" / "A6000"

POLICY_COLORS = {
    "tail": "#4C78A8",
    "fifo": "#F58518",
    "lru": "#FF9DA6",
    "largest_kv": "#54A24B",
    "smallest_kv": "#E45756",
    "random": "#B279A2",
    "evicpress": "#72B7B2",
    "harp": "#ECA400",
}

TIER_ORDER = ["cpu_dram", "cxl", "pcie_nvme", "ethernet", "ssd"]

POLICY_PREF = ["tail", "fifo", "lru", "largest_kv", "smallest_kv", "random", "evicpress", "harp"]


def _load_runner_helpers():
    path = ROOT_DIR / "benchmarks" / "run_model_tier_policy_matrix.py"
    mod_name = "_llmserving_run_model_tier_policy_matrix"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod._load_metrics, mod._load_tier_totals


def _eviction_event_total(tier_stats_json: Path) -> float:
    if not tier_stats_json.exists():
        return 0.0
    with open(tier_stats_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    count_suffix = "_count"
    total = 0.0
    for inst in raw.values():
        for k, v in inst.items():
            if not k.startswith("evict") or not k.endswith(count_suffix):
                continue
            try:
                total += float(v)
            except (TypeError, ValueError):
                continue
    return total


def _discover_runs(a6000_root: Path) -> List[Tuple[str, str, str, str, Path]]:
    """Return list of (model, tier, policy, workload, run_dir)."""
    out: List[Tuple[str, str, str, str, Path]] = []
    if not a6000_root.is_dir():
        return out
    for model_dir in sorted(a6000_root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        model = model_dir.name
        for tier_dir in sorted(model_dir.iterdir()):
            if not tier_dir.is_dir():
                continue
            tier = tier_dir.name
            for policy_dir in sorted(tier_dir.iterdir()):
                if not policy_dir.is_dir():
                    continue
                policy = policy_dir.name
                for wl_dir in sorted(policy_dir.iterdir()):
                    if not wl_dir.is_dir():
                        continue
                    workload = wl_dir.name
                    if (wl_dir / "result.csv").exists():
                        out.append((model, tier, policy, workload, wl_dir))
    return out


def _build_summary(a6000_root: Path) -> pd.DataFrame:
    load_metrics, load_tier = _load_runner_helpers()
    rows = []
    for model, tier, policy, workload, run_dir in _discover_runs(a6000_root):
        result_csv = run_dir / "result.csv"
        tier_json = run_dir / "result_tier_stats.json"
        row = {
            "model": model,
            "tier": tier,
            "policy": policy,
            "workload": workload,
            "accelerator": a6000_root.name if a6000_root.name else "A6000",
        }
        row.update(load_metrics(result_csv))
        row.update(load_tier(tier_json))
        if row.get("generated_tokens_total", 0.0) > 0:
            row["transition_mb_per_generated_token"] = row.get("tier_transition_mb_total", 0.0) / row[
                "generated_tokens_total"
            ]
        else:
            row["transition_mb_per_generated_token"] = 0.0
        row["eviction_events_total"] = _eviction_event_total(tier_json)
        rows.append(row)
    return pd.DataFrame(rows)


def _ordered_tiers(tiers: List[str]) -> List[str]:
    rest = sorted(t for t in tiers if t not in TIER_ORDER)
    return [t for t in TIER_ORDER if t in tiers] + rest


def _ordered_policies(policies: List[str]) -> List[str]:
    rest = sorted(p for p in policies if p not in POLICY_PREF)
    return [p for p in POLICY_PREF if p in policies] + rest


def _apply_large_style():
    plt.rcParams.update(
        {
            "figure.constrained_layout.use": True,
            "font.size": 15,
            "axes.titlesize": 17,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "legend.title_fontsize": 14,
        }
    )


def _grouped_metric_panel(
    ax,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    policies: List[str],
    tiers: List[str],
):
    if metric not in df.columns or df.empty:
        ax.set_visible(False)
        return
    pivot = df.pivot_table(index="tier", columns="policy", values=metric, aggfunc="mean")
    pivot = pivot.reindex(index=[t for t in tiers if t in pivot.index])
    pivot = pivot[[p for p in policies if p in pivot.columns]]
    if pivot.empty:
        ax.set_visible(False)
        return

    x = np.arange(len(pivot.index))
    n = max(1, len(pivot.columns))
    width = min(0.85 / n, 0.14)
    for i, policy in enumerate(pivot.columns):
        vals = pivot[policy].values.astype(float)
        pos = x - 0.425 + width / 2 + i * width
        color = POLICY_COLORS.get(str(policy), "#888888")
        ax.bar(pos, vals, width=width * 0.92, label=str(policy), color=color, alpha=0.9, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(list(pivot.index), rotation=22, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.35)


def _six_metric_figure(df_model_wl: pd.DataFrame, model_title: str, workload_label: str, out_path: Path):
    tiers = _ordered_tiers(sorted(df_model_wl["tier"].unique()))
    policies = _ordered_policies(sorted(df_model_wl["policy"].unique()))

    metrics = [
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("p99_ttft_ms", "P99 TTFT (ms)"),
        ("mean_tpot_ms", "Mean TPOT (ms)"),
        ("mean_latency_ms", "Mean end-to-end latency (ms)"),
        ("eviction_events_total", "Eviction events (tier stats)"),
        ("tier_transition_mb_total", "KV tier traffic (MB)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(f"{model_title} — {workload_label}\nPolicy comparison by storage tier (A6000)", fontsize=20, y=1.02)

    for ax, (col, ylab) in zip(axes.flat, metrics):
        _grouped_metric_panel(ax, df_model_wl, col, ylab, policies, tiers)
        ax.set_title(ylab.split("(")[0].strip(), fontsize=16)

    handles, labels = [], []
    for ax in axes.flat:
        h, lab = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, lab
            break
    if handles:
        fig.legend(
            handles,
            labels,
            title="Eviction policy",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(6, len(labels)),
            frameon=True,
        )

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _workload_compare_figure(
    df_model: pd.DataFrame,
    model_title: str,
    workload_a: str,
    workload_b: str,
    metric: str,
    ylabel: str,
    out_path: Path,
):
    sub = df_model[df_model["workload"].isin([workload_a, workload_b])].copy()
    if metric not in sub.columns or sub.empty:
        return

    tiers = _ordered_tiers(sorted(sub["tier"].unique()))
    policies = _ordered_policies(sorted(sub["policy"].unique()))
    n_t = len(tiers)
    n_cols = min(3, n_t)
    n_rows = int(np.ceil(n_t / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5.8 * n_rows), squeeze=False)
    fig.suptitle(
        f"{model_title}: {ylabel}\n{workload_a} vs {workload_b}",
        fontsize=19,
    )

    x = np.arange(len(policies))
    w = 0.36
    wl_colors = ("#2E5AAC", "#C45C3A")

    for idx, tier in enumerate(tiers):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        tdf = sub[sub["tier"] == tier]
        if tdf.empty:
            ax.set_visible(False)
            continue
        for i, wl in enumerate([workload_a, workload_b]):
            means = []
            for p in policies:
                cell = tdf[(tdf["policy"] == p) & (tdf["workload"] == wl)]
                means.append(float(cell[metric].mean()) if len(cell) else np.nan)
            offset = (i - 0.5) * w
            ax.bar(
                x + offset,
                means,
                width=w * 0.95,
                label=wl.replace("_", " "),
                alpha=0.88,
                color=wl_colors[i],
                edgecolor="white",
                linewidth=0.7,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=35, ha="right")
        ax.set_title(tier.replace("_", " ").upper(), fontsize=15)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.35)

    for j in range(len(tiers), n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r][c].set_visible(False)

    wl_labels = [workload_a.replace("_", " "), workload_b.replace("_", " ")]
    fig.legend(
        handles=[
            Rectangle((0, 0), 1, 1, fc="#2E5AAC", ec="white"),
            Rectangle((0, 0), 1, 1, fc="#C45C3A", ec="white"),
        ],
        labels=wl_labels,
        title="Workload",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=True,
    )
    for ax in axes.flat:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _four_aux_figure(df_model_wl: pd.DataFrame, model_title: str, workload_label: str, out_path: Path):
    tiers = _ordered_tiers(sorted(df_model_wl["tier"].unique()))
    policies = _ordered_policies(sorted(df_model_wl["policy"].unique()))
    metrics = [
        ("evicpress_compression_events", "EVICPRESS compression events"),
        ("transition_mb_per_generated_token", "Tier MB / generated token"),
        ("p99_tpot_ms", "P99 TPOT (ms)"),
        ("mean_stall_ms_per_request", "Mean HARP stall (ms / req)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        f"{model_title} — {workload_label}\nAuxiliary metrics (compression, tail latency, HARP stalls)",
        fontsize=19,
        y=1.01,
    )
    for ax, (col, ylab) in zip(axes.flat, metrics):
        if col not in df_model_wl.columns:
            ax.set_visible(False)
            continue
        _grouped_metric_panel(ax, df_model_wl, col, ylab, policies, tiers)
        ax.set_title(ylab, fontsize=15)

    handles, labels = [], []
    for ax in axes.flat:
        h, lab = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, lab
            break
    if handles:
        fig.legend(
            handles,
            labels,
            title="Eviction policy",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=min(5, len(labels)),
            frameon=True,
        )
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _model_display_name(model_key: str) -> str:
    if model_key == "llama8b":
        return "Llama 3.1 8B"
    if model_key == "phi_moe":
        return "Phi-mini-MoE"
    return model_key


def main():
    parser = argparse.ArgumentParser(description="Plot A6000 model/tier/policy matrix outputs")
    parser.add_argument("--root", type=Path, default=DEFAULT_A6000_ROOT, help="A6000 output root directory")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Plot output directory (default: <root>/plots)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = (args.out_dir or (root / "plots")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _apply_large_style()
    df = _build_summary(root)
    # oldest is identical to fifo in this simulator; show one label
    df["policy"] = df["policy"].replace({"oldest": "fifo"})
    id_cols = ["model", "tier", "policy", "workload", "accelerator"]
    val_cols = [c for c in df.columns if c not in id_cols]
    df = df.groupby(id_cols, as_index=False)[val_cols].mean()
    if df.empty:
        raise SystemExit(f"No runs found under {root}")

    summary_csv = out_dir / "a6000_metric_summary.csv"
    df.to_csv(summary_csv, index=False)

    workloads = sorted(df["workload"].unique())
    models = sorted(df["model"].unique())

    for model in models:
        mname = _model_display_name(model)
        mdf = df[df["model"] == model]
        for wl in workloads:
            sub = mdf[mdf["workload"] == wl]
            if sub.empty:
                continue
            wl_nice = wl.replace("_", " ")
            _six_metric_figure(
                sub,
                mname,
                wl_nice,
                out_dir / f"{model}_{wl}_policy_by_tier.png",
            )
            _four_aux_figure(
                sub,
                mname,
                wl_nice,
                out_dir / f"{model}_{wl}_auxiliary_metrics.png",
            )

        if "sharegpt_300" in workloads and "fixed_256" in workloads:
            for metric, ylab in [
                ("mean_ttft_ms", "Mean TTFT (ms)"),
                ("mean_tpot_ms", "Mean TPOT (ms)"),
                ("eviction_events_total", "Eviction events"),
                ("tier_transition_mb_total", "Tier transition (MB)"),
                ("p99_ttft_ms", "P99 TTFT (ms)"),
            ]:
                _workload_compare_figure(
                    mdf,
                    mname,
                    "sharegpt_300",
                    "fixed_256",
                    metric,
                    ylab,
                    out_dir / f"{model}_sharegpt300_vs_fixed256_{metric}.png",
                )

    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote plots under: {out_dir}")


if __name__ == "__main__":
    main()
