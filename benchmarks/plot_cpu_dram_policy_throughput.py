#!/usr/bin/env python3
"""
Plot throughput comparison across eviction policies for cpu_dram tier.

Reads summary lines from:
  output/tiered_kv/tier_policy_matrix/cpu_dram/<policy>/<workload>/output.txt

Extracted metrics:
  - Average prompt throughput (tok/s)
  - Average generation throughput (tok/s)
  - Total token throughput (tok/s)

Outputs:
  - throughput_summary.csv (all parsed rows)
    - throughput_<workload>_line.png (line graph)
    - throughput_all_workloads_line.png (multi-panel line graph)
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CPU_DRAM_ROOT = ROOT_DIR / "output" / "tiered_kv" / "tier_policy_matrix" / "cpu_dram"
DEFAULT_PLOTS_DIR = ROOT_DIR / "output" / "tiered_kv" / "tier_policy_matrix" / "plots"

PROMPT_RE = re.compile(r"Average prompt throughput \(tok/s\):\s*([0-9]+(?:\.[0-9]+)?)")
GEN_RE = re.compile(r"Average generation throughput \(tok/s\):\s*([0-9]+(?:\.[0-9]+)?)")
TOTAL_RE = re.compile(r"Total token throughput \(tok/s\):\s*([0-9]+(?:\.[0-9]+)?)")

POLICY_ORDER = [
    "tail",
    "fifo",
    "oldest",
    "lru",
    "largest_kv",
    "smallest_kv",
    "random",
    "evicpress",
    "harp",
]


@dataclass
class ThroughputRow:
    tier: str
    policy: str
    workload: str
    avg_prompt_toks: float
    avg_generation_toks: float
    total_toks: float
    source_log: str


def _extract_metric(pattern: re.Pattern[str], text: str) -> Optional[float]:
    matches = pattern.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def _scan_cpu_dram(root: Path) -> List[ThroughputRow]:
    rows: List[ThroughputRow] = []

    if not root.exists():
        return rows

    for policy_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        policy = policy_dir.name
        for workload_dir in sorted([w for w in policy_dir.iterdir() if w.is_dir()]):
            workload = workload_dir.name
            out_path = workload_dir / "output.txt"
            if not out_path.exists():
                continue

            text = out_path.read_text(encoding="utf-8", errors="ignore")
            prompt = _extract_metric(PROMPT_RE, text)
            generation = _extract_metric(GEN_RE, text)
            total = _extract_metric(TOTAL_RE, text)
            if prompt is None or generation is None or total is None:
                continue

            rows.append(
                ThroughputRow(
                    tier="cpu_dram",
                    policy=policy,
                    workload=workload,
                    avg_prompt_toks=prompt,
                    avg_generation_toks=generation,
                    total_toks=total,
                    source_log=str(out_path.relative_to(ROOT_DIR)),
                )
            )

    return rows


def _policy_sort_key(policy: str) -> tuple:
    if policy in POLICY_ORDER:
        return (0, POLICY_ORDER.index(policy), policy)
    return (1, 999, policy)


def _plot_workload_line(df: pd.DataFrame, workload: str, out_png: Path) -> None:
    wdf = df[df["workload"] == workload].copy()
    if wdf.empty:
        raise ValueError(f"No data found for workload '{workload}'")

    wdf = wdf.sort_values(by="policy", key=lambda s: s.map(lambda p: _policy_sort_key(str(p))))

    metrics = [
        ("avg_prompt_toks", "Avg prompt", "#2D6A4F"),
        ("avg_generation_toks", "Avg generation", "#40916C"),
        ("total_toks", "Total", "#1B4332"),
    ]

    policies = wdf["policy"].tolist()
    x = list(range(len(policies)))

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for col, label, color in metrics:
        vals = wdf[col].tolist()
        ax.plot(x, vals, marker="o", linewidth=2.0, markersize=5, label=label, color=color)
        for xi, yi in zip(x, vals):
            ax.text(
                xi,
                yi + max(5.0, 0.01 * max(vals)),
                f"{yi:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=25, ha="right")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title(f"CPU-DRAM throughput by eviction policy ({workload}) - line graph")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_all_workloads_line(df: pd.DataFrame, out_png: Path) -> None:
    workloads = sorted(df["workload"].unique().tolist())
    if not workloads:
        raise ValueError("No workloads available for plotting")

    n = len(workloads)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 4.5 * nrows), squeeze=False)

    metrics = [
        ("avg_prompt_toks", "Avg prompt", "#2D6A4F"),
        ("avg_generation_toks", "Avg generation", "#40916C"),
        ("total_toks", "Total", "#1B4332"),
    ]

    for idx, workload in enumerate(workloads):
        ax = axes[idx // ncols][idx % ncols]
        wdf = df[df["workload"] == workload].copy()
        wdf = wdf.sort_values(by="policy", key=lambda s: s.map(lambda p: _policy_sort_key(str(p))))

        policies = wdf["policy"].tolist()
        x = list(range(len(policies)))

        for col, label, color in metrics:
            vals = wdf[col].tolist()
            ax.plot(x, vals, marker="o", linewidth=1.8, markersize=4.5, label=label, color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=25, ha="right")
        ax.set_ylabel("tok/s")
        ax.set_title(workload)
        ax.grid(alpha=0.25)

    for idx in range(len(workloads), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)

    fig.suptitle("CPU-DRAM throughput by eviction policy - all workloads (line graph)", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CPU-DRAM throughput across eviction policies")
    parser.add_argument("--root", type=Path, default=DEFAULT_CPU_DRAM_ROOT, help="Path to cpu_dram policy outputs")
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR, help="Directory for output figures and CSV")
    parser.add_argument("--workload", type=str, default="sharegpt_1000", help="Workload to plot")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    args = parser.parse_args()

    rows = _scan_cpu_dram(args.root)
    if not rows:
        raise RuntimeError(f"No throughput summaries found under {args.root}")

    df = pd.DataFrame([r.__dict__ for r in rows])
    df = df.sort_values(by=["workload", "policy"])

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.plots_dir / "cpu_dram_policy_throughput_summary.csv"
    df.to_csv(csv_path, index=False)

    png_path = args.plots_dir / f"cpu_dram_policy_throughput_{args.workload}_line.png"
    _plot_workload_line(df, args.workload, png_path)

    all_png_path = args.plots_dir / "cpu_dram_policy_throughput_all_workloads_line.png"
    _plot_all_workloads_line(df, all_png_path)

    print(f"Parsed rows: {len(df)}")
    print(f"Summary CSV: {csv_path}")
    print(f"Plot:        {png_path}")
    print(f"All-workload plot: {all_png_path}")

    available = sorted(df["workload"].unique().tolist())
    print(f"Available workloads: {available}")

    print("\nSelected workload table:")
    view_cols = ["policy", "avg_prompt_toks", "avg_generation_toks", "total_toks", "source_log"]
    wdf = df[df["workload"] == args.workload].copy()
    if wdf.empty:
        print(f"No entries for workload={args.workload}")
    else:
        wdf = wdf.sort_values(by="policy", key=lambda s: s.map(lambda p: _policy_sort_key(str(p))))
        print(wdf[view_cols].to_string(index=False))

    if args.show:
        img = plt.imread(png_path)
        plt.figure(figsize=(12, 5.5))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
