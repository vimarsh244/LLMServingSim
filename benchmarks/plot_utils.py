"""
plot_utils.py — Shared plotting utilities for tiered KV cache analysis.

Provides:
  - Consistent style, color palettes, and labels for tier configs
  - Data loading helpers for result CSVs and time-series CSVs
  - Unit conversion (ns → ms/s)
  - Figure saving wrapper (PDF + PNG)
"""

import os
import ast
import json
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np

# ─────────────────────── Lazy matplotlib imports ──────────────────────
# These are imported lazily so data-only scripts can use load helpers
# without requiring matplotlib.

def _setup_matplotlib():
    """Import and configure matplotlib + seaborn. Call once before plotting."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "font.family": "sans-serif",
    })
    return plt, sns


# ─────────────────────── Color palettes ───────────────────────────────

# Tier config color palette (consistent across all plots)
TIER_COLORS = {
    "npu_only":     "#1f77b4",  # blue
    "npu_cpu":      "#2ca02c",  # green
    "npu_cxl":      "#ff7f0e",  # orange
    "npu_cxl_cpu":  "#e377c2",  # pink  (3-tier: NPU→CXL→CPU)
    "npu_cxl_fast": "#d62728",  # red
    "npu_cxl_slow": "#9467bd",  # purple
    "cpu_dram":     "#17becf",  # cyan
    "cxl":          "#bcbd22",  # olive
    "pcie_nvme":    "#8c564b",  # brown
    "ssd":          "#7f7f7f",  # gray
    "ethernet":     "#1f77b4",  # blue
}

# Block-size sweep colors
BLOCK_COLORS = {
    "block_4":  "#1f77b4",  # blue
    "block_8":  "#ff7f0e",  # orange
    "block_16": "#2ca02c",  # green
    "block_32": "#d62728",  # red
    "block_64": "#9467bd",  # purple
}

# Prefix config colors
PREFIX_COLORS = {
    "no_prefix":    "#1f77b4",  # blue
    "prefix_npu":   "#2ca02c",  # green
    "prefix_cpu":   "#ff7f0e",  # orange
    "prefix_cxl":   "#d62728",  # red
}

# Memory tier colors (for stacked area charts)
MEM_TIER_COLORS = {
    "npu_kv":       "#1f77b4",  # blue
    "npu_prefix":   "#aec7e8",  # light blue
    "cpu_evicted":  "#2ca02c",  # green
    "cxl_evicted":  "#ff7f0e",  # orange
    "weight":       "#7f7f7f",  # gray
}

# ─────────────────────── Label mappings ───────────────────────────────

TIER_LABELS = {
    "npu_only":     "NPU Only",
    "npu_cpu":      "NPU + CPU",
    "npu_cxl":      "NPU + CXL",
    "npu_cxl_cpu":  "NPU → CXL → CPU",
    "npu_cxl_fast": "NPU + CXL (Fast)",
    "npu_cxl_slow": "NPU + CXL (Slow)",
    "cpu_dram":     "CPU DRAM Tier",
    "cxl":          "CXL Tier",
    "pcie_nvme":    "PCIe NVMe Tier",
    "ssd":          "SSD Tier",
    "ethernet":     "Ethernet Tier",
}

BLOCK_LABELS = {
    "block_4":  "Block 4",
    "block_8":  "Block 8",
    "block_16": "Block 16",
    "block_32": "Block 32",
    "block_64": "Block 64",
}

PREFIX_LABELS = {
    "no_prefix":    "No Prefix Cache",
    "prefix_npu":   "Prefix (NPU)",
    "prefix_cpu":   "Prefix (NPU+CPU)",
    "prefix_cxl":   "Prefix (NPU+CXL)",
}

WORKLOAD_LABELS = {
    "sharegpt_100":  "ShareGPT-100",
    "sharegpt_300":  "ShareGPT-300",
    "sharegpt_750":  "ShareGPT-750",
    "sharegpt_1000": "ShareGPT-1000",
    "sharegpt_1500": "ShareGPT-1500",
    "fixed_256":     "Fixed-256",
    "prefix_stress": "Prefix Stress",
    "pulse_prefix":  "Pulse Prefix",
}

METRIC_LABELS = {
    "ttft": "TTFT (ms)",
    "tpot": "TPOT (ms)",
    "itl":  "ITL (ms)",
    "latency": "Latency (ms)",
}


# ─────────────────────── Unit conversion ──────────────────────────────

NS_TO_MS = 1e-6  # nanoseconds to milliseconds
NS_TO_S  = 1e-9
BYTES_TO_MB = 1 / (1024 * 1024)
BYTES_TO_GB = 1 / (1024 * 1024 * 1024)


# ─────────────────────── Data loading ─────────────────────────────────

def load_result_csv(path: str) -> pd.DataFrame:
    """Load a per-request result CSV with proper types.
    
    Handles legacy and extended output formats.
    """
    df = pd.read_csv(path)
    # Strip whitespace from column names 
    df.columns = df.columns.str.strip()

    # Convert nanosecond timing columns to milliseconds
    ns_cols = ['arrival', 'end_time', 'latency', 'queuing_delay', 'TTFT', 'TPOT']
    for col in ns_cols:
        if col in df.columns:
            df[f"{col}_ms"] = df[col] * NS_TO_MS

    # Parse ITL list string
    if 'ITL' in df.columns:
        df['ITL_list'] = df['ITL'].apply(_parse_itl)
        df['ITL_ms_list'] = df['ITL_list'].apply(lambda x: [v * NS_TO_MS for v in x])
        df['mean_ITL_ms'] = df['ITL_ms_list'].apply(lambda x: np.mean(x) if x else 0)
        df['p99_ITL_ms'] = df['ITL_ms_list'].apply(lambda x: np.percentile(x, 99) if x else 0)

    # Ensure cache hit columns exist (backward compat)
    for col in ['npu_cache_hit', 'storage_cache_hit', 'prefix_cache_hit']:
        if col not in df.columns:
            df[col] = 0

    if 'tier_reload_source' not in df.columns:
        df['tier_reload_source'] = 'NPU'

    # Ensure per-request tier transition columns exist (backward compat)
    tier_cols = [
        'evict_npu_to_cpu_bytes',
        'evict_npu_to_cxl_bytes',
        'load_cpu_to_npu_bytes',
        'load_cxl_to_npu_bytes',
        'tier_transition_bytes_total',
    ]
    for col in tier_cols:
        if col not in df.columns:
            df[col] = 0
        df[f'{col}_mb'] = df[col] * BYTES_TO_MB

    if 'tier_transition_bytes_total' in df.columns:
        zero_total = (df['tier_transition_bytes_total'] == 0)
        if zero_total.any():
            recomputed = (
                df['evict_npu_to_cpu_bytes']
                + df['evict_npu_to_cxl_bytes']
                + df['load_cpu_to_npu_bytes']
                + df['load_cxl_to_npu_bytes']
            )
            df.loc[zero_total, 'tier_transition_bytes_total'] = recomputed[zero_total]

    df['tier_transition_mb_total'] = df['tier_transition_bytes_total'] * BYTES_TO_MB

    return df


def _parse_itl(itl_str) -> list:
    """Parse ITL column which is a string-encoded Python list."""
    if pd.isna(itl_str) or itl_str == '' or itl_str == '[]':
        return []
    try:
        return ast.literal_eval(str(itl_str))
    except (ValueError, SyntaxError):
        return []


def load_timeseries_csv(path: str) -> pd.DataFrame:
    """Load a time-series CSV with proper types."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Convert time to seconds for plotting
    if 'sim_time_ns' in df.columns:
        df['sim_time_s'] = df['sim_time_ns'] * NS_TO_S

    # Convert bytes to MB/GB 
    byte_cols = [c for c in df.columns if c.endswith('_bytes')]
    for col in byte_cols:
        df[f"{col.replace('_bytes', '_mb')}"] = df[col] * BYTES_TO_MB
        df[f"{col.replace('_bytes', '_gb')}"] = df[col] * BYTES_TO_GB

    return df


def load_tier_stats(path: str) -> dict:
    """Load tier_stats JSON."""
    with open(path) as f:
        return json.load(f)


def discover_experiments(results_dir: str, phase: str = None) -> List[Dict]:
    """Discover all completed experiments in a results directory.
    
    Returns list of dicts with keys: name, phase, config, workload, result_csv, 
    timeseries_csv, tier_stats_json, output_txt.
    """
    results_dir = Path(results_dir)
    experiments = []

    # Pattern: {phase}/{config}/{workload}/result.csv
    pattern = "*/*/*/result.csv" if phase is None else f"{phase}/*/*/result.csv"
    
    for result_csv in sorted(results_dir.glob(pattern)):
        rel = result_csv.relative_to(results_dir)
        parts = list(rel.parts)
        
        if len(parts) >= 3:
            exp_phase = parts[0]  # e.g. "phaseA"
            config = parts[1]     # e.g. "npu_only"
            workload = parts[2]   # e.g. "sharegpt_300"
        else:
            continue

        exp_dir = result_csv.parent
        experiments.append({
            "name": f"{exp_phase}/{config}/{workload}",
            "phase": exp_phase,
            "config": config,
            "workload": workload,
            "result_csv": str(result_csv),
            "timeseries_csv": str(exp_dir / "timeseries.csv") if (exp_dir / "timeseries.csv").exists() else None,
            "tier_stats_json": str(exp_dir / "result_tier_stats.json") if (exp_dir / "result_tier_stats.json").exists() else None,
            "output_txt": str(exp_dir / "output.txt") if (exp_dir / "output.txt").exists() else None,
        })

    return experiments


def load_all_results(results_dir: str, phase: str = None) -> pd.DataFrame:
    """Load and concatenate all result CSVs for a phase.
    
    Adds 'phase', 'config', 'workload' columns for grouping.
    """
    experiments = discover_experiments(results_dir, phase)
    dfs = []
    for exp in experiments:
        df = load_result_csv(exp["result_csv"])
        df["phase"] = exp["phase"]
        df["config"] = exp["config"]
        df["workload"] = exp["workload"]
        dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_all_timeseries(results_dir: str, phase: str = None) -> pd.DataFrame:
    """Load and concatenate all time-series CSVs for a phase."""
    experiments = discover_experiments(results_dir, phase)
    dfs = []
    for exp in experiments:
        if exp["timeseries_csv"] and os.path.exists(exp["timeseries_csv"]):
            df = load_timeseries_csv(exp["timeseries_csv"])
            df["phase"] = exp["phase"]
            df["config"] = exp["config"]
            df["workload"] = exp["workload"]
            dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_all_tier_stats(results_dir: str, phase: str = None) -> pd.DataFrame:
    """Load all tier_stats JSONs into a flat DataFrame."""
    experiments = discover_experiments(results_dir, phase)
    rows = []
    for exp in experiments:
        if exp["tier_stats_json"] and os.path.exists(exp["tier_stats_json"]):
            stats = load_tier_stats(exp["tier_stats_json"])
            for inst_key, inst_stats in stats.items():
                row = {"phase": exp["phase"], "config": exp["config"], 
                       "workload": exp["workload"], "instance": inst_key}
                row.update(inst_stats)
                rows.append(row)
    
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ─────────────────────── Figure saving ────────────────────────────────

def save_figure(fig, path: str, formats: list = None):
    """Save a figure in multiple formats.
    
    Args:
        fig: matplotlib Figure object
        path: base path without extension (e.g., 'figures/latency_bar')
        formats: list of formats (default: ['png'])
    """
    if formats is None:
        formats = ['png']
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        out = path.with_suffix(f'.{fmt}')
        fig.savefig(str(out), format=fmt, bbox_inches='tight')
    
    import matplotlib.pyplot as plt
    plt.close(fig)


# Auto-cycle colors for unknown configs
_AUTO_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
_auto_color_cache = {}

def get_color(config: str, palette: dict = None) -> str:
    """Get color for a config name, with auto-cycle fallback."""
    if palette is None:
        palette = {**TIER_COLORS, **PREFIX_COLORS, **BLOCK_COLORS}
    if config in palette:
        return palette[config]
    # Deterministic auto-assign for unknown configs
    if config not in _auto_color_cache:
        idx = len(_auto_color_cache) % len(_AUTO_COLORS)
        _auto_color_cache[config] = _AUTO_COLORS[idx]
    return _auto_color_cache[config]


def get_label(config: str, label_map: dict = None) -> str:
    """Get human-readable label for a config name."""
    if label_map is None:
        label_map = {**TIER_LABELS, **PREFIX_LABELS, **WORKLOAD_LABELS, **BLOCK_LABELS}
    return label_map.get(config, config)


# ─────────────────────── Summary table generation ─────────────────────

def generate_summary_table(results_df: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
    """Generate a summary table with key metrics per config × workload.
    
    Returns DataFrame and optionally saves as LaTeX and CSV.
    """
    if results_df.empty:
        return pd.DataFrame()

    summary = results_df.groupby(["config", "workload"]).agg(
        mean_TTFT_ms=("TTFT_ms", "mean"),
        p99_TTFT_ms=("TTFT_ms", lambda x: np.percentile(x, 99)),
        mean_TPOT_ms=("TPOT_ms", "mean"),
        p99_TPOT_ms=("TPOT_ms", lambda x: np.percentile(x, 99)),
        mean_ITL_ms=("mean_ITL_ms", "mean"),
        total_requests=("request id", "count"),
        mean_prefix_hit=("prefix_cache_hit", "mean"),
    ).reset_index()

    # Round for readability
    for col in summary.columns:
        if summary[col].dtype == float:
            summary[col] = summary[col].round(2)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(str(output_path.with_suffix('.csv')), index=False)
        
        # LaTeX table
        try:
            latex_str = summary.to_latex(index=False, float_format="%.2f", 
                                          caption="Tiered KV Cache Experiment Summary",
                                          label="tab:tiered_kv_summary")
            with open(str(output_path.with_suffix('.tex')), 'w') as f:
                f.write(latex_str)
        except Exception:
            pass  # LaTeX generation is optional

    return summary
