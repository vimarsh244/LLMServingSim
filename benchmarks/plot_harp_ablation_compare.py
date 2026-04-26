#!/usr/bin/env python3
"""
Plot the HARP ablation comparison for baseline, HARP all-zero, HARP no-prefetch, and Tail.

Reads per-run artifacts from the ablation output tree and produces a four-panel figure:
  - TTFT
  - TPOT
  - ITL
  - Throughput

The script prefers the requested comparison set:
    - baseline from .../sharegpt1000_cpu_dram/E0_baseline
    - HARP all-zero from .../sharegpt1000_cpu_dram/E1_harp_all_zero
    - HARP no-prefetch from .../sharegpt1000_cpu_dram/E2_harp_no_prefetch
    - Tail from .../sharegpt1000_cpu_dram/E3_tail

Outputs are written under output/tiered_kv/harp_ablation/sharegpt1000_cpu_dram/plots.
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
CURRENT_ROOT = ROOT_DIR / "output" / "tiered_kv" / "harp_ablation" / "sharegpt1000_cpu_dram"
DEFAULT_PLOTS_DIR = CURRENT_ROOT / "plots"

THROUGHPUT_PROMPT_RE = re.compile(r"Average prompt throughput \(tok/s\):\s*([0-9]+(?:\.[0-9]+)?)")
THROUGHPUT_GEN_RE = re.compile(r"Average generation throughput \(tok/s\):\s*([0-9]+(?:\.[0-9]+)?)")
THROUGHPUT_TOTAL_RE = re.compile(r"Total token throughput \(tok/s\):\s*([0-9]+(?:\.[0-9]+)?)")


@dataclass(frozen=True)
class RunSpec:
    label: str
    directory: Path
    color: str


RUNS: List[RunSpec] = [
    RunSpec("Baseline", CURRENT_ROOT / "E0_baseline", "#264653"),
    RunSpec("HARP all-zero", CURRENT_ROOT / "E1_harp_all_zero", "#2a9d8f"),
    RunSpec("HARP no-prefetch", CURRENT_ROOT / "E2_harp_no_prefetch", "#e76f51"),
    RunSpec("Tail", CURRENT_ROOT / "E3_tail", "#457b9d"),
]


def _parse_itl_values(series: pd.Series) -> List[float]:
    values: List[float] = []
    for item in series.dropna():
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
    return values


def _load_result_metrics(result_csv: Path) -> Dict[str, float]:
    if not result_csv.exists():
        return {}

    df = pd.read_csv(result_csv)
    if df.empty:
        return {}

    df.columns = df.columns.str.strip()
    out: Dict[str, float] = {}

    def mean_ms(col: str, key: str) -> None:
        if col in df.columns:
            out[key] = float(df[col].astype(float).mean() * 1e-6)

    mean_ms("TTFT", "ttft_ms")
    mean_ms("TPOT", "tpot_ms")

    if "ITL" in df.columns:
        itl_values = _parse_itl_values(df["ITL"])
        if itl_values:
            itl_series = pd.Series(itl_values, dtype=float)
            out["itl_ms"] = float(itl_series.mean() * 1e-6)

    return out


def _load_tier_metrics(tier_stats_json: Path) -> Dict[str, float]:
    if not tier_stats_json.exists():
        return {}

    import json

    raw = json.loads(tier_stats_json.read_text(encoding="utf-8"))

    keys = [
        "evict_npu_to_cpu_bytes",
        "evict_npu_to_cpu_count",
    ]
    out = {k: 0.0 for k in keys}

    for inst in raw.values():
        for key in keys:
            out[key] += float(inst.get(key, 0.0))

    out["evict_npu_to_cpu_mb"] = out["evict_npu_to_cpu_bytes"] / (1024.0 * 1024.0)
    return out


def _extract_throughput(output_txt: Path) -> Dict[str, float]:
    if not output_txt.exists():
        return {}

    text = output_txt.read_text(encoding="utf-8", errors="ignore")

    def last_match(pattern: re.Pattern[str]) -> Optional[float]:
        matches = pattern.findall(text)
        if not matches:
            return None
        return float(matches[-1])

    prompt = last_match(THROUGHPUT_PROMPT_RE)
    generation = last_match(THROUGHPUT_GEN_RE)
    total = last_match(THROUGHPUT_TOTAL_RE)

    out: Dict[str, float] = {}
    if prompt is not None:
        out["prompt_toks_s"] = prompt
    if generation is not None:
        out["generation_toks_s"] = generation
    if total is not None:
        out["throughput_toks_s"] = total
    elif prompt is not None and generation is not None:
        out["throughput_toks_s"] = prompt + generation

    return out


def _load_run_row(run: RunSpec) -> Dict[str, object]:
    result_csv = run.directory / "result.csv"
    tier_stats_json = run.directory / "result_tier_stats.json"
    output_txt = run.directory / "output.txt"

    row: Dict[str, object] = {
        "experiment": run.label,
        "source_dir": str(run.directory.relative_to(ROOT_DIR)),
    }
    row.update(_load_result_metrics(result_csv))
    row.update(_load_tier_metrics(tier_stats_json))
    row.update(_extract_throughput(output_txt))
    return row


def _barplot(ax, df: pd.DataFrame, column: str, title: str, color_map: Dict[str, str]) -> None:
    order = [run.label for run in RUNS]
    plot_df = df.set_index("experiment").reindex(order).reset_index()
    values = plot_df[column].astype(float).tolist()

    colors = [color_map[label] for label in plot_df["experiment"].tolist()]
    bars = ax.bar(range(len(order)), values, color=colors, width=0.68)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=18, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.2f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _plot_metric_grid(df: pd.DataFrame, metric_specs: List[tuple], title: str, out_png: Path, color_map: Dict[str, str]) -> None:
    n = len(metric_specs)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 4.8 * nrows))

    if n == 1:
        axes = [axes]
    else:
        axes = list(axes.flatten())

    for ax, (column, metric_title) in zip(axes, metric_specs):
        _barplot(ax, df, column, metric_title, color_map)

    for ax in axes[len(metric_specs):]:
        ax.set_visible(False)

    fig.suptitle(title, y=0.99, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HARP ablation comparison metrics")
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    rows = [_load_run_row(run) for run in RUNS]
    df = pd.DataFrame(rows)

    required = ["ttft_ms", "tpot_ms", "itl_ms", "throughput_toks_s"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing metrics for plotting: {missing}")

    eviction_required = ["evict_npu_to_cpu_count", "evict_npu_to_cpu_bytes", "evict_npu_to_cpu_mb"]
    missing_eviction = [col for col in eviction_required if col not in df.columns]
    if missing_eviction:
        raise RuntimeError(f"Missing eviction metrics for plotting: {missing_eviction}")

    color_map = {run.label: run.color for run in RUNS}

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.plots_dir / "harp_ablation_compare_summary.csv"
    df.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    axes = axes.flatten()

    _barplot(axes[0], df, "ttft_ms", "Mean TTFT (ms)", color_map)
    _barplot(axes[1], df, "tpot_ms", "Mean TPOT (ms)", color_map)
    _barplot(axes[2], df, "itl_ms", "Mean ITL (ms)", color_map)
    _barplot(axes[3], df, "throughput_toks_s", "Throughput (tok/s)", color_map)

    fig.suptitle("ShareGPT-1000 CPU-DRAM HARP Ablation Comparison", y=0.99, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    png_path = args.plots_dir / "harp_ablation_compare_four_panel.png"
    pdf_path = args.plots_dir / "harp_ablation_compare_four_panel.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    eviction_png = args.plots_dir / "harp_ablation_eviction_compare.png"
    eviction_pdf = args.plots_dir / "harp_ablation_eviction_compare.pdf"
    _plot_metric_grid(
        df,
        [
            ("evict_npu_to_cpu_count", "Eviction Count (NPU → CPU)"),
            ("evict_npu_to_cpu_mb", "Eviction Data (MB, NPU → CPU)"),
        ],
        "ShareGPT-1000 CPU-DRAM HARP Ablation Eviction Comparison",
        eviction_png,
        color_map,
    )
    _plot_metric_grid(
        df,
        [
            ("evict_npu_to_cpu_count", "Eviction Count (NPU → CPU)"),
            ("evict_npu_to_cpu_mb", "Eviction Data (MB, NPU → CPU)"),
        ],
        "ShareGPT-1000 CPU-DRAM HARP Ablation Eviction Comparison",
        eviction_pdf,
        color_map,
    )

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {eviction_png}")
    print(f"Saved: {eviction_pdf}")
    print(f"Saved: {summary_csv}")
    print(df.to_string(index=False))

    if args.show:
        plt.figure(figsize=(13.5, 8.5))
        img = plt.imread(png_path)
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()