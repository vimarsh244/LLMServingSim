#!/usr/bin/env python3
"""
Plot CPU-DRAM ShareGPT grouped bars with DynMax in place of HARP.

This script scans output/tiered_kv/tier_policy_matrix/cpu_dram/*/*/result.csv
and the corresponding output.txt logs, filters out incomplete or unwanted
policies, and plots four metrics by workload:
    - TTFT
    - TPOT
    - ITL
    - throughput

Default behavior mirrors the screenshot style:
    - workloads: sharegpt_750 sharegpt_1000 sharegpt_1500
    - excluded policies: fifo evicpress adaptive_dynmax adaptive_dynamx
    - included policies: tail, largest_kv, lru, random, smallest_kv, harp, dynmax, dynmax_proactive

DynMax is treated as the HARP zero-lambda/no-compression configuration that was
copied into the tier_policy_matrix/cpu_dram/dynmax directory.
The proactive run stored under dynmax_proactive_32 is displayed as dynmax_proactive.
The older dynmax_proactive_16 run is intentionally ignored.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = ROOT_DIR / "output" / "tiered_kv" / "tier_policy_matrix" / "cpu_dram"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "plots"

WORKLOAD_ORDER = ["sharegpt_750", "sharegpt_1000", "sharegpt_1500"]
WORKLOAD_LABELS = {
    "sharegpt_750": "sharegpt_750",
    "sharegpt_1000": "sharegpt_1000",
    "sharegpt_1500": "sharegpt_1500",
}

POLICY_ORDER = ["tail", "largest_kv", "lru", "random", "smallest_kv", "harp", "dynmax", "dynmax_proactive"]
POLICY_ALIASES = {
    "dynmax_proactive_32": "dynmax_proactive",
}
POLICY_COLORS = {
    "tail": "#1F77B4",
    "largest_kv": "#FF7F0E",
    "lru": "#2CA02C",
    "random": "#D62728",
    "smallest_kv": "#9467BD",
    "harp": "#8C564B",
    "dynmax": "#17BECF",
    "dynmax_proactive": "#E377C2",
}

PROMPT_TP_RE = re.compile(r"Average prompt throughput \(tok/s\):\s*([0-9]+(?:\.[0-9]+)?)")
GEN_TP_RE = re.compile(r"Average generation throughput \(tok/s\):\s*([0-9]+(?:\.[0-9]+)?)")
TOTAL_TP_RE = re.compile(r"Total token throughput \(tok/s\):\s*([0-9]+(?:\.[0-9]+)?)")


def _load_runs(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for result_csv in root.glob("*/*/result.csv"):
        parts = result_csv.relative_to(root).parts
        if len(parts) != 3:
            continue
        policy, workload, _ = parts
        if policy in {"dynmax_proactive_16", "oldest"}:
            continue
        policy = POLICY_ALIASES.get(policy, policy)
        df = pd.read_csv(result_csv)
        if df.empty or "TTFT" not in df.columns or "TPOT" not in df.columns:
            continue
        df.columns = df.columns.str.strip()

        output_txt = result_csv.with_name("output.txt")
        prompt_tp = np.nan
        gen_tp = np.nan
        total_tp = np.nan
        if output_txt.exists():
            text = output_txt.read_text(encoding="utf-8", errors="ignore")
            prompt_matches = PROMPT_TP_RE.findall(text)
            gen_matches = GEN_TP_RE.findall(text)
            total_matches = TOTAL_TP_RE.findall(text)
            if prompt_matches:
                prompt_tp = float(prompt_matches[-1])
            if gen_matches:
                gen_tp = float(gen_matches[-1])
            if total_matches:
                total_tp = float(total_matches[-1])
            elif np.isfinite(prompt_tp) and np.isfinite(gen_tp):
                total_tp = float(prompt_tp + gen_tp)

        rows.append(
            {
                "policy": policy,
                "workload": workload,
                "mean_ttft_ms": float(df["TTFT"].astype(float).mean() * 1e-6),
                "mean_tpot_ms": float(df["TPOT"].astype(float).mean() * 1e-6),
                "mean_itl_ms": _mean_itl_ms(df["ITL"]) if "ITL" in df.columns else np.nan,
                "throughput_toks_s": total_tp,
                "prompt_toks_s": prompt_tp,
                "generation_toks_s": gen_tp,
                "result_csv": str(result_csv.relative_to(ROOT_DIR)),
                "output_txt": str(output_txt.relative_to(ROOT_DIR)) if output_txt.exists() else "",
            }
        )
    return pd.DataFrame(rows)


def _ordered(values: List[str], preferred: List[str]) -> List[str]:
    ordered = [v for v in preferred if v in values]
    ordered.extend(sorted(v for v in values if v not in ordered))
    return ordered


def _mean_itl_ms(itl_series: pd.Series) -> float:
    values: List[float] = []
    for item in itl_series.dropna():
        if isinstance(item, list):
            values.extend(float(v) for v in item)
            continue
        if isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, list):
                values.extend(float(v) for v in parsed)
    if not values:
        return np.nan
    return float(np.mean(values) * 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CPU-DRAM DynMax metrics with grouped bars")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--workloads", nargs="+", default=WORKLOAD_ORDER)
    parser.add_argument(
        "--exclude-policies",
        nargs="+",
        default=["fifo", "evicpress", "adaptive_dynmax", "adaptive_dynamx"],
    )
    args = parser.parse_args()

    df = _load_runs(args.root)
    if df.empty:
        raise RuntimeError(f"No result.csv files found under {args.root}")

    exclude = {p.strip() for p in args.exclude_policies if p.strip()}
    df = df[~df["policy"].isin(exclude)].copy()
    df = df[~df["policy"].str.startswith(("adaptive_dynmax", "adaptive_dynamx"), na=False)].copy()
    df = df[df["workload"].isin(args.workloads)].copy()
    if df.empty:
        raise RuntimeError("No rows left after filtering workloads/policies")

    policies = _ordered(df["policy"].unique().tolist(), POLICY_ORDER)
    workloads = _ordered(df["workload"].unique().tolist(), WORKLOAD_ORDER)

    metric_specs = [
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("mean_tpot_ms", "Mean TPOT (ms)"),
        ("mean_itl_ms", "Mean ITL (ms)"),
        ("throughput_toks_s", "Throughput (tok/s)"),
    ]
    pivots = {}
    for metric, _ in metric_specs:
        pivot = df.pivot_table(index="workload", columns="policy", values=metric, aggfunc="mean")
        pivot = pivot.reindex(index=workloads)
        pivot = pivot[[p for p in policies if p in pivot.columns]]
        pivots[metric] = pivot

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.out_dir / "cpu_dram_dynmax_metrics_summary.csv"
    df.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharex=False)
    axes = axes.flatten()
    for ax, (metric, title) in zip(axes, metric_specs):
        pivot = pivots[metric]
        x = np.arange(len(pivot.index), dtype=float)
        n = max(1, len(pivot.columns))
        width = min(0.82 / n, 0.14)

        for idx, policy in enumerate(pivot.columns):
            vals = pivot[policy].values.astype(float)
            pos = x - 0.41 + width / 2 + idx * width
            color = POLICY_COLORS.get(policy, "#777777")
            bars = ax.bar(pos, vals, width=width * 0.92, label=policy, color=color, alpha=0.92, edgecolor="white", linewidth=0.6)
            for bar, value in zip(bars, vals):
                if not np.isfinite(value):
                    continue
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.0f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([WORKLOAD_LABELS.get(w, w) for w in pivot.index], rotation=18, ha="right")
        ax.set_ylabel(title)
        ax.set_xlabel("Workload")
        ax.set_title(f"cpu_dram: {title}")
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Policy", ncol=4, loc="upper center", frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = args.out_dir / "cpu_dram_dynmax_metrics.png"
    pdf_path = args.out_dir / "cpu_dram_dynmax_metrics.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()