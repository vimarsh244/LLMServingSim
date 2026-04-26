#!/usr/bin/env python3
"""
analyze_results.py — Analysis and plotting pipeline for tiered KV cache experiments.

Reads experiment outputs from output/tiered_kv/ and generates publication-quality
plots covering:
  1. Latency comparison (bar charts)
  2. Latency distributions (CDF / box plots)
  3. Memory utilization time-series (stacked area)
  4. Cache efficiency (hit rates)
  5. Migration overhead (bytes moved)
  6. CXL sensitivity (heatmaps)
  7. Block size sensitivity
  8. Summary dashboard (Pareto frontier)

Usage:
    python benchmarks/analyze_results.py --results-dir output/tiered_kv --phase phaseA
    python benchmarks/analyze_results.py --results-dir output/tiered_kv --phase all
    python benchmarks/analyze_results.py --results-dir output/tiered_kv --phase all --output-dir figures/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_utils import (
    _setup_matplotlib, load_all_results, load_all_timeseries, load_all_tier_stats,
    discover_experiments, load_result_csv, load_timeseries_csv, load_tier_stats,
    save_figure, get_color, get_label, generate_summary_table,
    TIER_COLORS, PREFIX_COLORS, MEM_TIER_COLORS,
    TIER_LABELS, PREFIX_LABELS, WORKLOAD_LABELS, METRIC_LABELS,
    NS_TO_MS, NS_TO_S, BYTES_TO_MB, BYTES_TO_GB,
)

# ═══════════════════════ Plot Category 1: Latency Bars ════════════════

def plot_latency_bars(results_df, output_dir, phase):
    """Grouped bar charts for mean TTFT and TPOT across configs."""
    plt, sns = _setup_matplotlib()

    for metric, col in [("TTFT", "TTFT_ms"), ("TPOT", "TPOT_ms")]:
        if col not in results_df.columns:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        
        configs = results_df["config"].unique()
        workloads = results_df["workload"].unique()
        n_configs = len(configs)
        n_workloads = len(workloads)
        
        bar_width = 0.8 / n_configs
        x = np.arange(n_workloads)

        for i, cfg in enumerate(configs):
            means = []
            errs_low = []
            errs_high = []
            for wl in workloads:
                subset = results_df[(results_df["config"] == cfg) & (results_df["workload"] == wl)]
                if len(subset) > 0:
                    vals = subset[col].dropna()
                    mean = vals.mean()
                    p5 = vals.quantile(0.05) if len(vals) > 1 else mean
                    p95 = vals.quantile(0.95) if len(vals) > 1 else mean
                    means.append(mean)
                    errs_low.append(mean - p5)
                    errs_high.append(p95 - mean)
                else:
                    means.append(0)
                    errs_low.append(0)
                    errs_high.append(0)

            offset = (i - n_configs / 2 + 0.5) * bar_width
            ax.bar(x + offset, means, bar_width, 
                   yerr=[errs_low, errs_high], capsize=3,
                   label=get_label(cfg), color=get_color(cfg),
                   edgecolor='white', linewidth=0.5)

        ax.set_xlabel("Workload")
        ax.set_ylabel(f"Mean {metric} (ms)")
        ax.set_title(f"Mean {metric} by Tier Configuration — {phase}")
        ax.set_xticks(x)
        ax.set_xticklabels([get_label(w) for w in workloads], rotation=15, ha='right')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3)

        save_figure(fig, f"{output_dir}/{phase}_latency_{metric.lower()}_bar")

    # P99 chart
    for metric, col in [("TTFT", "TTFT_ms"), ("TPOT", "TPOT_ms")]:
        if col not in results_df.columns:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        configs = results_df["config"].unique()
        workloads = results_df["workload"].unique()
        n_configs = len(configs)
        bar_width = 0.8 / n_configs
        x = np.arange(len(workloads))

        for i, cfg in enumerate(configs):
            p99s = []
            for wl in workloads:
                subset = results_df[(results_df["config"] == cfg) & (results_df["workload"] == wl)]
                if len(subset) > 0:
                    p99s.append(subset[col].quantile(0.99))
                else:
                    p99s.append(0)

            offset = (i - n_configs / 2 + 0.5) * bar_width
            ax.bar(x + offset, p99s, bar_width,
                   label=get_label(cfg), color=get_color(cfg),
                   edgecolor='white', linewidth=0.5)

        ax.set_xlabel("Workload")
        ax.set_ylabel(f"P99 {metric} (ms)")
        ax.set_title(f"P99 {metric} by Tier Configuration — {phase}")
        ax.set_xticks(x)
        ax.set_xticklabels([get_label(w) for w in workloads], rotation=15, ha='right')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3)

        save_figure(fig, f"{output_dir}/{phase}_latency_{metric.lower()}_p99_bar")


# ═══════════════════════ Plot Category 2: Latency CDFs ════════════════

def plot_latency_cdfs(results_df, output_dir, phase):
    """CDF plots for TTFT and TPOT, one subplot per workload."""
    plt, sns = _setup_matplotlib()

    for metric, col in [("TTFT", "TTFT_ms"), ("TPOT", "TPOT_ms")]:
        if col not in results_df.columns:
            continue

        workloads = results_df["workload"].unique()
        n_wl = len(workloads)
        ncols = min(3, n_wl)
        nrows = (n_wl + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        configs = results_df["config"].unique()

        for idx, wl in enumerate(workloads):
            ax = axes[idx // ncols][idx % ncols]
            for cfg in configs:
                subset = results_df[(results_df["config"] == cfg) & (results_df["workload"] == wl)]
                if len(subset) == 0:
                    continue
                vals = np.sort(subset[col].dropna().values)
                cdf = np.arange(1, len(vals) + 1) / len(vals)
                ax.plot(vals, cdf, label=get_label(cfg), color=get_color(cfg), linewidth=1.5)

            ax.set_xlabel(f"{metric} (ms)")
            ax.set_ylabel("CDF")
            ax.set_title(get_label(wl))
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 1.05)

        # Hide unused subplots
        for idx in range(n_wl, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(f"{metric} CDF by Tier Configuration — {phase}", fontsize=14, y=1.02)
        fig.tight_layout()
        save_figure(fig, f"{output_dir}/{phase}_latency_{metric.lower()}_cdf")


def plot_itl_boxplot(results_df, output_dir, phase):
    """Box plot of inter-token latencies per config."""
    plt, sns = _setup_matplotlib()

    configs = results_df["config"].unique()
    workloads = results_df["workload"].unique()

    for wl in workloads:
        wl_df = results_df[results_df["workload"] == wl]
        if "ITL_ms_list" not in wl_df.columns:
            continue

        # Explode ITL lists into individual rows
        rows = []
        for _, req in wl_df.iterrows():
            for itl_val in req["ITL_ms_list"]:
                rows.append({"config": req["config"], "ITL_ms": itl_val})
        
        if not rows:
            continue

        itl_df = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(10, 6))
        order = [c for c in configs if c in itl_df["config"].unique()]
        palette = {c: get_color(c) for c in order}
        
        sns.boxplot(data=itl_df, x="config", y="ITL_ms", order=order, palette=palette,
                    ax=ax, showfliers=False, width=0.6)
        
        ax.set_xlabel("Tier Configuration")
        ax.set_ylabel("Inter-Token Latency (ms)")
        ax.set_title(f"ITL Distribution — {get_label(wl)} — {phase}")
        ax.set_xticklabels([get_label(c) for c in order], rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)

        save_figure(fig, f"{output_dir}/{phase}_itl_boxplot_{wl}")


# ═══════════════════════ Plot Category 3: Memory Time-Series ══════════

def plot_memory_timeseries(ts_df, output_dir, phase):
    """Stacked area chart of KV cache occupancy over time per experiment."""
    plt, sns = _setup_matplotlib()

    if ts_df.empty or 'sim_time_s' not in ts_df.columns:
        return

    for (cfg, wl), group in ts_df.groupby(["config", "workload"]):
        instances = group["instance_id"].unique()
        n_inst = len(instances)

        fig, axes = plt.subplots(1, n_inst, figsize=(8 * n_inst, 5), squeeze=False)

        for idx, inst_id in enumerate(instances):
            ax = axes[0][idx]
            inst_data = group[group["instance_id"] == inst_id].sort_values("sim_time_s")

            time_s = inst_data["sim_time_s"].values

            # Compute areas
            npu_total_gb = inst_data["npu_total_gb"].values if "npu_total_gb" in inst_data else np.zeros_like(time_s)
            npu_used_gb = inst_data["npu_used_gb"].values if "npu_used_gb" in inst_data else np.zeros_like(time_s)
            cpu_used_gb = inst_data["cpu_used_gb"].values if "cpu_used_gb" in inst_data else np.zeros_like(time_s)

            cxl_used_gb = inst_data["cxl_used_gb"].values if "cxl_used_gb" in inst_data else np.zeros_like(time_s)

            ax.fill_between(time_s, 0, npu_used_gb, 
                          alpha=0.7, color=MEM_TIER_COLORS["npu_kv"], label="NPU Used")
            ax.fill_between(time_s, npu_used_gb, npu_used_gb + cxl_used_gb,
                          alpha=0.7, color=MEM_TIER_COLORS["cxl_evicted"], label="CXL Evicted KV")
            ax.fill_between(time_s, npu_used_gb + cxl_used_gb, npu_used_gb + cxl_used_gb + cpu_used_gb,
                          alpha=0.7, color=MEM_TIER_COLORS["cpu_evicted"], label="CPU Evicted KV")
            
            if npu_total_gb.any():
                ax.axhline(y=npu_total_gb[0], color='red', linestyle='--', alpha=0.5, label="NPU Capacity")

            ax.set_xlabel("Simulation Time (s)")
            ax.set_ylabel("Memory Usage (GB)")
            ax.set_title(f"Instance {inst_id}")
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(alpha=0.3)

        fig.suptitle(f"Memory Utilization — {get_label(cfg)} / {get_label(wl)} — {phase}", fontsize=13)
        fig.tight_layout()
        save_figure(fig, f"{output_dir}/{phase}_memory_timeseries_{cfg}_{wl}")


def plot_memory_pressure(ts_df, output_dir, phase):
    """Line chart of NPU utilization % over time with highlight when near capacity."""
    plt, sns = _setup_matplotlib()

    if ts_df.empty:
        return

    for (cfg, wl), group in ts_df.groupby(["config", "workload"]):
        inst_data = group[group["instance_id"] == group["instance_id"].min()].sort_values("sim_time_s")

        if "npu_used_bytes" not in inst_data.columns or "npu_total_bytes" not in inst_data.columns:
            continue

        time_s = inst_data["sim_time_s"].values
        util_pct = (inst_data["npu_used_bytes"] / inst_data["npu_total_bytes"] * 100).values

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_s, util_pct, color=get_color(cfg), linewidth=1.5)
        ax.fill_between(time_s, 90, 100, alpha=0.15, color='red', label="Pressure Zone (>90%)")
        ax.axhline(y=90, color='red', linestyle=':', alpha=0.4)
        
        ax.set_xlabel("Simulation Time (s)")
        ax.set_ylabel("NPU Memory Utilization (%)")
        ax.set_title(f"Memory Pressure — {get_label(cfg)} / {get_label(wl)} — {phase}")
        ax.set_ylim(0, 105)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)

        save_figure(fig, f"{output_dir}/{phase}_mem_pressure_{cfg}_{wl}")


# ═══════════════════════ Plot Category 4: Cache Efficiency ════════════

def plot_cache_hit_rates(ts_df, output_dir, phase):
    """Hit rate over time (interval-delta) per config. Skips if all zero."""
    plt, sns = _setup_matplotlib()

    if ts_df.empty or "npu_hit_rate_interval" not in ts_df.columns:
        return

    # Check if ANY hit rate is > 0
    hit_vals = pd.to_numeric(ts_df["npu_hit_rate_interval"], errors='coerce').fillna(0)
    if hit_vals.max() <= 0:
        print("    (skipped cache hit rate — no prefix caching active)")
        return

    for wl in ts_df["workload"].unique():
        wl_data = ts_df[ts_df["workload"] == wl]
        configs = wl_data["config"].unique()

        fig, ax = plt.subplots(figsize=(10, 5))
        
        for cfg in configs:
            cfg_data = wl_data[(wl_data["config"] == cfg) & (wl_data["instance_id"] == 0)].sort_values("sim_time_s")
            if cfg_data.empty:
                continue
            # Convert string to float if needed
            hit_rate = pd.to_numeric(cfg_data["npu_hit_rate_interval"], errors='coerce').fillna(0)
            ax.plot(cfg_data["sim_time_s"].values, hit_rate.values,
                   label=get_label(cfg), color=get_color(cfg), linewidth=1.2, alpha=0.8)

        ax.set_xlabel("Simulation Time (s)")
        ax.set_ylabel("NPU Prefix Cache Hit Rate (%)")
        ax.set_title(f"Interval Cache Hit Rate — {get_label(wl)} — {phase}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(-5, 105)

        save_figure(fig, f"{output_dir}/{phase}_cache_hit_rate_{wl}")


def plot_cache_hit_bar(results_df, output_dir, phase):
    """Aggregate hit rate stacked bar: NPU hit / storage hit / miss. Skips if no hits."""
    plt, sns = _setup_matplotlib()

    if results_df.empty or "prefix_cache_hit" not in results_df.columns:
        return

    # Skip if there are zero cache hits across all experiments
    total_hits = results_df["npu_cache_hit"].sum() + results_df["storage_cache_hit"].sum()
    if total_hits <= 0:
        print("    (skipped cache hit breakdown — no cache hits)")
        return

    configs = results_df["config"].unique()
    workloads = results_df["workload"].unique()

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group labels
    group_labels = []
    npu_hits = []
    storage_hits = []
    misses = []

    for wl in workloads:
        for cfg in configs:
            subset = results_df[(results_df["config"] == cfg) & (results_df["workload"] == wl)]
            if len(subset) == 0:
                continue
            total_input = subset["input"].sum()
            total_npu_hit = subset["npu_cache_hit"].sum()
            total_storage_hit = subset["storage_cache_hit"].sum() - total_npu_hit  # incremental
            total_storage_hit = max(0, total_storage_hit)
            total_miss = total_input - total_npu_hit - total_storage_hit

            if total_input > 0:
                npu_hits.append(total_npu_hit / total_input * 100)
                storage_hits.append(total_storage_hit / total_input * 100)
                misses.append(total_miss / total_input * 100)
            else:
                npu_hits.append(0)
                storage_hits.append(0)
                misses.append(100)
            group_labels.append(f"{get_label(cfg)}\n{get_label(wl)}")

    if not group_labels:
        return

    x = np.arange(len(group_labels))
    ax.bar(x, npu_hits, label="NPU Hit", color=MEM_TIER_COLORS["npu_kv"], width=0.6)
    ax.bar(x, storage_hits, bottom=npu_hits, label="Storage Hit", color=MEM_TIER_COLORS["cpu_evicted"], width=0.6)
    ax.bar(x, misses, bottom=[n + s for n, s in zip(npu_hits, storage_hits)], 
           label="Miss", color="#d9d9d9", width=0.6)

    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Cache Hit Rate Breakdown — {phase}")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=7)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    save_figure(fig, f"{output_dir}/{phase}_cache_hit_breakdown")


def plot_cache_hit_scatter(results_df, output_dir, phase):
    """Scatter: input tokens vs prefix cache hit tokens, colored by config. Skips if no hits."""
    plt, sns = _setup_matplotlib()

    if results_df.empty or "prefix_cache_hit" not in results_df.columns:
        return

    # Skip if all prefix hits are zero
    if results_df["prefix_cache_hit"].sum() <= 0:
        print("    (skipped cache hit scatter — no prefix hits)")
        return

    configs = results_df["config"].unique()

    fig, ax = plt.subplots(figsize=(10, 6))
    for cfg in configs:
        subset = results_df[results_df["config"] == cfg]
        ax.scatter(subset["input"], subset["prefix_cache_hit"], 
                  label=get_label(cfg), color=get_color(cfg), 
                  alpha=0.5, s=20, edgecolors='none')

    ax.plot([0, results_df["input"].max()], [0, results_df["input"].max()], 
            'k--', alpha=0.3, label="100% hit")
    ax.set_xlabel("Input Token Count")
    ax.set_ylabel("Prefix Cache Hit (tokens)")
    ax.set_title(f"Cache Hit vs Input Length — {phase}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    save_figure(fig, f"{output_dir}/{phase}_cache_hit_scatter")


# ═══════════════════════ Plot Category 5: Migration Overhead ══════════

def plot_migration_bars(tier_stats_df, output_dir, phase):
    """Stacked bar of total bytes migrated per tier transition type."""
    plt, sns = _setup_matplotlib()

    if tier_stats_df.empty:
        return

    # Aggregate across instances
    agg_cols = {
        'evict_npu_to_cpu_bytes': 'sum',
        'load_cpu_to_npu_bytes': 'sum',
        'evict_npu_prefix_bytes': 'sum',
        'evict_storage_prefix_bytes': 'sum',
        'prefix_load_storage_to_npu_bytes': 'sum',
        'storage_cache_evicted_req_bytes': 'sum',
    }
    # Add CXL columns if present
    for col in ['evict_npu_to_cxl_bytes', 'load_cxl_to_npu_bytes']:
        if col in tier_stats_df.columns:
            agg_cols[col] = 'sum'
    agg = tier_stats_df.groupby(["config", "workload"]).agg(agg_cols).reset_index()

    if agg.empty:
        return

    migration_types = [
        ('evict_npu_to_cxl_bytes', 'NPU→CXL Evict', '#e377c2'),
        ('load_cxl_to_npu_bytes', 'CXL→NPU Reload', '#bcbd22'),
        ('evict_npu_to_cpu_bytes', 'NPU→CPU Evict', '#1f77b4'),
        ('load_cpu_to_npu_bytes', 'CPU→NPU Reload', '#2ca02c'),
        ('evict_npu_prefix_bytes', 'NPU Prefix Evict', '#ff7f0e'),
        ('prefix_load_storage_to_npu_bytes', 'Storage→NPU Prefix', '#d62728'),
        ('storage_cache_evicted_req_bytes', 'Evicted→Storage', '#9467bd'),
    ]

    labels = [f"{get_label(row['config'])}\n{get_label(row['workload'])}" for _, row in agg.iterrows()]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 6))
    bottom = np.zeros(len(labels))

    for col, name, color in migration_types:
        if col in agg.columns:
            vals = (agg[col].values * BYTES_TO_MB)
            ax.bar(x, vals, bottom=bottom, label=name, color=color, width=0.6)
            bottom += vals

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Data Migrated (MB)")
    ax.set_title(f"KV Cache Migration Volume — {phase}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    save_figure(fig, f"{output_dir}/{phase}_migration_volume")


def plot_migration_over_time(ts_df, output_dir, phase):
    """Cumulative migration bytes over simulation time."""
    plt, sns = _setup_matplotlib()

    if ts_df.empty:
        return

    evict_col = "evict_npu_to_cpu_bytes_total"
    load_col = "load_cpu_to_npu_bytes_total"
    evict_cxl_col = "evict_npu_to_cxl_bytes_total"
    load_cxl_col = "load_cxl_to_npu_bytes_total"
    
    has_any_col = any(c in ts_df.columns for c in [evict_col, evict_cxl_col])
    if not has_any_col:
        return

    for wl in ts_df["workload"].unique():
        wl_data = ts_df[ts_df["workload"] == wl]
        configs = wl_data["config"].unique()

        fig, ax = plt.subplots(figsize=(10, 5))
        for cfg in configs:
            cfg_data = wl_data[(wl_data["config"] == cfg) & (wl_data["instance_id"] == 0)].sort_values("sim_time_s")
            if cfg_data.empty:
                continue

            evict_mb = cfg_data[evict_col].values * BYTES_TO_MB if evict_col in cfg_data.columns else np.zeros(len(cfg_data))
            load_mb = cfg_data[load_col].values * BYTES_TO_MB if load_col in cfg_data.columns else np.zeros_like(evict_mb)
            evict_cxl_mb = cfg_data[evict_cxl_col].values * BYTES_TO_MB if evict_cxl_col in cfg_data.columns else np.zeros_like(evict_mb)
            load_cxl_mb = cfg_data[load_cxl_col].values * BYTES_TO_MB if load_cxl_col in cfg_data.columns else np.zeros_like(evict_mb)
            total_mb = evict_mb + load_mb + evict_cxl_mb + load_cxl_mb

            ax.plot(cfg_data["sim_time_s"].values, total_mb,
                   label=get_label(cfg), color=get_color(cfg), linewidth=1.5)

        ax.set_xlabel("Simulation Time (s)")
        ax.set_ylabel("Cumulative Data Moved (MB)")
        ax.set_title(f"Cumulative Migration Volume — {get_label(wl)} — {phase}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        save_figure(fig, f"{output_dir}/{phase}_migration_cumulative_{wl}")


# ═══════════════════════ Plot Category 5b: Workload Comparison ════════

def plot_workload_latency_comparison(results_df, output_dir, phase):
    """For phases with one config: horizontal bar chart comparing all workloads
    across multiple latency metrics in a single figure."""
    plt, sns = _setup_matplotlib()

    configs = results_df["config"].unique()
    if len(configs) != 1:
        return  # only useful for single-config phases

    workloads = results_df["workload"].unique()
    wl_labels = [get_label(w) for w in workloads]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = [
        ("Mean TTFT (ms)", "TTFT_ms", "mean"),
        ("Mean TPOT (ms)", "TPOT_ms", "mean"),
        ("Mean ITL (ms)", "mean_ITL_ms", "mean"),
    ]

    palette = sns.color_palette("viridis", n_colors=len(workloads))

    for ax, (label, col, agg) in zip(axes, metrics):
        vals = []
        for wl in workloads:
            subset = results_df[results_df["workload"] == wl]
            if len(subset) > 0 and col in subset.columns:
                vals.append(subset[col].mean())
            else:
                vals.append(0)

        y_pos = np.arange(len(workloads))
        bars = ax.barh(y_pos, vals, color=palette, edgecolor='white', linewidth=0.5,
                       height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(wl_labels)
        ax.set_xlabel(label)
        ax.grid(axis='x', alpha=0.3)

        # Annotate values
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                       xytext=(5, 0), textcoords='offset points', va='center', fontsize=9)

    fig.suptitle(f"Workload Comparison — {get_label(configs[0])} — {phase}", fontsize=14)
    fig.tight_layout()
    save_figure(fig, f"{output_dir}/{phase}_workload_comparison")


def plot_eviction_latency_correlation(results_df, tier_stats_df, output_dir, phase):
    """Scatter + annotation of total eviction volume vs mean TPOT per experiment."""
    plt, sns = _setup_matplotlib()

    if results_df.empty or tier_stats_df.empty:
        return

    # Compute per (config, workload) aggregates
    records = []
    for (cfg, wl), grp in results_df.groupby(["config", "workload"]):
        ts_sub = tier_stats_df[(tier_stats_df["config"] == cfg) & (tier_stats_df["workload"] == wl)]
        if ts_sub.empty:
            continue
        evict_cpu_mb = ts_sub["evict_npu_to_cpu_bytes"].sum() / (1024 * 1024) if "evict_npu_to_cpu_bytes" in ts_sub.columns else 0
        reload_cpu_mb = ts_sub["load_cpu_to_npu_bytes"].sum() / (1024 * 1024) if "load_cpu_to_npu_bytes" in ts_sub.columns else 0
        evict_cxl_mb = ts_sub["evict_npu_to_cxl_bytes"].sum() / (1024 * 1024) if "evict_npu_to_cxl_bytes" in ts_sub.columns else 0
        reload_cxl_mb = ts_sub["load_cxl_to_npu_bytes"].sum() / (1024 * 1024) if "load_cxl_to_npu_bytes" in ts_sub.columns else 0
        total_mig = evict_cpu_mb + reload_cpu_mb + evict_cxl_mb + reload_cxl_mb
        records.append({
            "config": cfg, "workload": wl,
            "total_migration_mb": total_mig,
            "evict_cpu_mb": evict_cpu_mb,
            "evict_cxl_mb": evict_cxl_mb,
            "reload_cpu_mb": reload_cpu_mb,
            "reload_cxl_mb": reload_cxl_mb,
            "mean_TPOT_ms": grp["TPOT_ms"].mean(),
            "mean_TTFT_ms": grp["TTFT_ms"].mean(),
            "num_evictions": (ts_sub.get("evict_npu_to_cpu_count", pd.Series([0])).sum() +
                              ts_sub.get("evict_npu_to_cxl_count", pd.Series([0])).sum()),
            "num_requests": len(grp),
        })

    if not records:
        return

    df = pd.DataFrame(records)

    # Skip if all migration is zero
    if df["total_migration_mb"].max() <= 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    import matplotlib.patches as mpatches

    # Left: migration vs TPOT
    ax = axes[0]
    for _, row in df.iterrows():
        ax.scatter(row["total_migration_mb"], row["mean_TPOT_ms"],
                   s=row["num_requests"] * 0.5 + 30,
                   color=get_color(row["config"]), alpha=0.7,
                   edgecolors='black', linewidth=0.3)
        ax.annotate(get_label(row["workload"]),
                    (row["total_migration_mb"], row["mean_TPOT_ms"]),
                    fontsize=7, xytext=(5, 3), textcoords='offset points')
    ax.set_xlabel("Total KV Data Migrated (MB)")
    ax.set_ylabel("Mean TPOT (ms)")
    ax.set_title("Migration Volume vs TPOT")
    ax.grid(alpha=0.3)

    # Right: eviction count vs TTFT
    ax = axes[1]
    for _, row in df.iterrows():
        ax.scatter(row["num_evictions"], row["mean_TTFT_ms"],
                   s=row["num_requests"] * 0.5 + 30,
                   color=get_color(row["config"]), alpha=0.7,
                   edgecolors='black', linewidth=0.3)
        ax.annotate(get_label(row["workload"]),
                    (row["num_evictions"], row["mean_TTFT_ms"]),
                    fontsize=7, xytext=(5, 3), textcoords='offset points')
    ax.set_xlabel("Total Eviction Count")
    ax.set_ylabel("Mean TTFT (ms)")
    ax.set_title("Eviction Count vs TTFT")
    ax.grid(alpha=0.3)

    # Config legend
    handles = []
    for cfg in df["config"].unique():
        handles.append(mpatches.Patch(color=get_color(cfg), label=get_label(cfg)))
    axes[0].legend(handles=handles, fontsize=8, loc='upper left')

    fig.suptitle(f"Eviction-Latency Correlation — {phase}", fontsize=14)
    fig.tight_layout()
    save_figure(fig, f"{output_dir}/{phase}_eviction_latency_corr")


def plot_migration_breakdown(tier_stats_df, output_dir, phase):
    """Side-by-side bars: eviction bytes vs reload bytes per workload."""
    plt, sns = _setup_matplotlib()

    if tier_stats_df.empty:
        return

    agg_cols = {
        'evict_npu_to_cpu_bytes': 'sum',
        'load_cpu_to_npu_bytes': 'sum',
        'evict_npu_to_cpu_count': 'sum',
        'load_cpu_to_npu_count': 'sum',
    }
    for col in ['evict_npu_to_cxl_bytes', 'load_cxl_to_npu_bytes', 'evict_npu_to_cxl_count', 'load_cxl_to_npu_count']:
        if col in tier_stats_df.columns:
            agg_cols[col] = 'sum'
    agg = tier_stats_df.groupby(["config", "workload"]).agg(agg_cols).reset_index()

    # Filter: at least some migration happened
    total_bytes = agg.get('evict_npu_to_cpu_bytes', 0) + agg.get('load_cpu_to_npu_bytes', 0)
    if 'evict_npu_to_cxl_bytes' in agg.columns:
        total_bytes = total_bytes + agg['evict_npu_to_cxl_bytes'] + agg.get('load_cxl_to_npu_bytes', 0)
    agg = agg[total_bytes > 0]
    if agg.empty:
        return

    labels = [f"{get_label(row['config'])}\n{get_label(row['workload'])}"
              for _, row in agg.iterrows()]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(labels) * 1.5), 6))

    # Volume subplot (stacked bars per direction)
    ax = axes[0]
    w = 0.35
    evict_cpu_mb = agg['evict_npu_to_cpu_bytes'].values / (1024 * 1024)
    reload_cpu_mb = agg['load_cpu_to_npu_bytes'].values / (1024 * 1024)
    evict_cxl_mb = agg['evict_npu_to_cxl_bytes'].values / (1024 * 1024) if 'evict_npu_to_cxl_bytes' in agg.columns else np.zeros(len(agg))
    reload_cxl_mb = agg['load_cxl_to_npu_bytes'].values / (1024 * 1024) if 'load_cxl_to_npu_bytes' in agg.columns else np.zeros(len(agg))

    ax.bar(x - w / 2, evict_cxl_mb, w, label="NPU\u2192CXL Evict", color="#e377c2", alpha=0.8)
    ax.bar(x - w / 2, evict_cpu_mb, w, bottom=evict_cxl_mb, label="NPU\u2192CPU Evict", color="#d62728", alpha=0.8)
    ax.bar(x + w / 2, reload_cxl_mb, w, label="CXL\u2192NPU Reload", color="#bcbd22", alpha=0.8)
    ax.bar(x + w / 2, reload_cpu_mb, w, bottom=reload_cxl_mb, label="CPU\u2192NPU Reload", color="#2ca02c", alpha=0.8)
    ax.set_ylabel("Data Volume (MB)")
    ax.set_title("Migration Volume: Evict vs Reload")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    # Count subplot
    ax = axes[1]
    evict_cpu_cnt = agg['evict_npu_to_cpu_count'].values
    reload_cpu_cnt = agg['load_cpu_to_npu_count'].values
    evict_cxl_cnt = agg['evict_npu_to_cxl_count'].values if 'evict_npu_to_cxl_count' in agg.columns else np.zeros(len(agg))
    reload_cxl_cnt = agg['load_cxl_to_npu_count'].values if 'load_cxl_to_npu_count' in agg.columns else np.zeros(len(agg))

    ax.bar(x - w / 2, evict_cxl_cnt, w, label="NPU\u2192CXL Evict", color="#e377c2", alpha=0.8)
    ax.bar(x - w / 2, evict_cpu_cnt, w, bottom=evict_cxl_cnt, label="NPU\u2192CPU Evict", color="#d62728", alpha=0.8)
    ax.bar(x + w / 2, reload_cxl_cnt, w, label="CXL\u2192NPU Reload", color="#bcbd22", alpha=0.8)
    ax.bar(x + w / 2, reload_cpu_cnt, w, bottom=reload_cxl_cnt, label="CPU\u2192NPU Reload", color="#2ca02c", alpha=0.8)
    ax.set_ylabel("Operation Count")
    ax.set_title("Migration Ops: Evict vs Reload Count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f"Migration Breakdown — {phase}", fontsize=14)
    fig.tight_layout()
    save_figure(fig, f"{output_dir}/{phase}_migration_breakdown")


def plot_memory_efficiency(ts_df, results_df, output_dir, phase):
    """Peak NPU utilization % vs throughput per (config, workload)."""
    plt, sns = _setup_matplotlib()

    if ts_df.empty or results_df.empty:
        return

    records = []
    for (cfg, wl), grp in ts_df.groupby(["config", "workload"]):
        inst_0 = grp[grp["instance_id"] == 0]
        if inst_0.empty or "npu_used_bytes" not in inst_0.columns:
            continue
        peak_pct = (inst_0["npu_used_bytes"] / inst_0["npu_total_bytes"]).max() * 100
        avg_pct = (inst_0["npu_used_bytes"] / inst_0["npu_total_bytes"]).mean() * 100

        res_sub = results_df[(results_df["config"] == cfg) & (results_df["workload"] == wl)]
        if res_sub.empty:
            continue
        total_time_s = (res_sub["end_time"].max() - res_sub["arrival"].min()) * NS_TO_S
        throughput = len(res_sub) / total_time_s if total_time_s > 0 else 0

        records.append({
            "config": cfg, "workload": wl,
            "peak_npu_pct": peak_pct, "avg_npu_pct": avg_pct,
            "throughput": throughput,
            "mean_TPOT_ms": res_sub["TPOT_ms"].mean(),
        })

    if not records:
        return

    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    labels = [f"{get_label(row['config'])}\n{get_label(row['workload'])}"
              for _, row in df.iterrows()]
    palette = [get_color(row["config"]) for _, row in df.iterrows()]
    x = np.arange(len(labels))

    # Left: peak NPU utilization bar
    ax = axes[0]
    ax.bar(x, df["peak_npu_pct"], color=palette, edgecolor='white', linewidth=0.5)
    ax.axhline(y=90, color='red', linestyle=':', alpha=0.5, label="90% threshold")
    ax.set_ylabel("Peak NPU Utilization (%)")
    ax.set_title("Peak NPU Memory Utilization")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Right: throughput vs avg memory utilization scatter
    ax = axes[1]
    for _, row in df.iterrows():
        ax.scatter(row["avg_npu_pct"], row["throughput"],
                   s=80, color=get_color(row["config"]), alpha=0.8,
                   edgecolors='black', linewidth=0.3)
        ax.annotate(get_label(row["workload"]),
                    (row["avg_npu_pct"], row["throughput"]),
                    fontsize=7, xytext=(5, 3), textcoords='offset points')
    ax.set_xlabel("Average NPU Utilization (%)")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Memory Efficiency: Utilization vs Throughput")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Memory Efficiency — {phase}", fontsize=14)
    fig.tight_layout()
    save_figure(fig, f"{output_dir}/{phase}_memory_efficiency")


# ═══════════════════════ Plot Category 6: CXL Sensitivity ═════════════

def plot_cxl_heatmap(results_df, output_dir, phase):
    """2D heatmap: CXL latency × bandwidth → mean TPOT."""
    plt, sns = _setup_matplotlib()

    if results_df.empty or "TPOT_ms" not in results_df.columns:
        return

    # Extract CXL params from config names like "cxl_lat250_bw60"
    import re
    pattern = re.compile(r"cxl_lat(\d+)_bw(\d+)")

    records = []
    for cfg in results_df["config"].unique():
        m = pattern.match(cfg)
        if m:
            lat, bw = int(m.group(1)), int(m.group(2))
            subset = results_df[results_df["config"] == cfg]
            records.append({
                "latency_ns": lat,
                "bandwidth_GBs": bw,
                "mean_TPOT_ms": subset["TPOT_ms"].mean(),
            })

    if not records:
        return

    heat_df = pd.DataFrame(records)
    pivot = heat_df.pivot_table(index="latency_ns", columns="bandwidth_GBs", values="mean_TPOT_ms")
    pivot = pivot.sort_index(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Mean TPOT (ms)"})
    ax.set_xlabel("CXL Bandwidth (GB/s)")
    ax.set_ylabel("CXL Latency (ns)")
    ax.set_title(f"CXL Parameter Sensitivity — Mean TPOT — {phase}")

    save_figure(fig, f"{output_dir}/{phase}_cxl_heatmap")


def plot_cxl_sensitivity_lines(results_df, output_dir, phase):
    """Line charts: TPOT vs CXL bandwidth, one line per latency."""
    plt, sns = _setup_matplotlib()

    import re
    pattern = re.compile(r"cxl_lat(\d+)_bw(\d+)")

    records = []
    for cfg in results_df["config"].unique():
        m = pattern.match(cfg)
        if m:
            lat, bw = int(m.group(1)), int(m.group(2))
            subset = results_df[results_df["config"] == cfg]
            records.append({
                "latency_ns": lat, "bandwidth_GBs": bw,
                "mean_TPOT_ms": subset["TPOT_ms"].mean(),
            })

    if not records:
        return

    df = pd.DataFrame(records)
    latencies = sorted(df["latency_ns"].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, lat in enumerate(latencies):
        lat_data = df[df["latency_ns"] == lat].sort_values("bandwidth_GBs")
        ax.plot(lat_data["bandwidth_GBs"], lat_data["mean_TPOT_ms"],
               marker='o', label=f"{lat}ns latency", color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel("CXL Bandwidth (GB/s)")
    ax.set_ylabel("Mean TPOT (ms)")
    ax.set_title(f"CXL Bandwidth Sensitivity — {phase}")
    ax.legend()
    ax.grid(alpha=0.3)

    save_figure(fig, f"{output_dir}/{phase}_cxl_bw_sensitivity")


# ═══════════════════════ Plot Category 7: Block Size ══════════════════

def plot_block_size(results_df, output_dir, phase):
    """Bar chart of TPOT/TTFT by block size."""
    plt, sns = _setup_matplotlib()

    import re
    pattern = re.compile(r"block_(\d+)")

    records = []
    for cfg in results_df["config"].unique():
        m = pattern.match(cfg)
        if m:
            bs = int(m.group(1))
            for wl in results_df["workload"].unique():
                subset = results_df[(results_df["config"] == cfg) & (results_df["workload"] == wl)]
                if len(subset) > 0:
                    records.append({
                        "block_size": bs, "workload": wl,
                        "mean_TPOT_ms": subset["TPOT_ms"].mean(),
                        "mean_TTFT_ms": subset["TTFT_ms"].mean(),
                    })

    if not records:
        return

    df = pd.DataFrame(records)
    workloads = df["workload"].unique()

    for metric in ["mean_TPOT_ms", "mean_TTFT_ms"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        n_wl = len(workloads)
        bar_width = 0.8 / n_wl
        block_sizes = sorted(df["block_size"].unique())
        x = np.arange(len(block_sizes))

        for i, wl in enumerate(workloads):
            wl_data = df[df["workload"] == wl].set_index("block_size")
            vals = [wl_data.loc[bs, metric] if bs in wl_data.index else 0 for bs in block_sizes]
            offset = (i - n_wl / 2 + 0.5) * bar_width
            ax.bar(x + offset, vals, bar_width, label=get_label(wl))

        metric_name = "TPOT" if "TPOT" in metric else "TTFT"
        ax.set_xlabel("Block Size (tokens)")
        ax.set_ylabel(f"Mean {metric_name} (ms)")
        ax.set_title(f"Block Size Impact on {metric_name} — {phase}")
        ax.set_xticks(x)
        ax.set_xticklabels(block_sizes)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        save_figure(fig, f"{output_dir}/{phase}_block_size_{metric_name.lower()}")


# ═══════════════════════ Plot Category 8: Summary ═════════════════════

def plot_pareto_frontier(results_df, tier_stats_df, output_dir, phase):
    """Pareto plot: Mean TPOT vs total migration bytes."""
    plt, sns = _setup_matplotlib()

    if results_df.empty or tier_stats_df.empty:
        return

    # Compute per-config aggregates
    tpot_agg = results_df.groupby("config")["TPOT_ms"].mean().reset_index()
    
    migration_cols = [c for c in tier_stats_df.columns if c.endswith('_bytes')]
    if not migration_cols:
        return
    
    mig_agg = tier_stats_df.groupby("config")[migration_cols].sum().reset_index()
    mig_agg["total_migration_mb"] = mig_agg[migration_cols].sum(axis=1) * BYTES_TO_MB

    merged = tpot_agg.merge(mig_agg[["config", "total_migration_mb"]], on="config", how="inner")

    if merged.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in merged.iterrows():
        ax.scatter(row["TPOT_ms"], row["total_migration_mb"],
                  s=100, color=get_color(row["config"]), 
                  label=get_label(row["config"]), zorder=5, edgecolors='black', linewidth=0.5)

    ax.set_xlabel("Mean TPOT (ms)")
    ax.set_ylabel("Total Data Migrated (MB)")
    ax.set_title(f"Latency vs Migration Overhead — {phase}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    save_figure(fig, f"{output_dir}/{phase}_pareto_frontier")


def plot_throughput_comparison(results_df, output_dir, phase):
    """Bar chart of request throughput (req/s) per config × workload."""
    plt, sns = _setup_matplotlib()

    if results_df.empty:
        return

    configs = results_df["config"].unique()
    workloads = results_df["workload"].unique()

    fig, ax = plt.subplots(figsize=(12, 6))
    n_configs = len(configs)
    bar_width = 0.8 / n_configs
    x = np.arange(len(workloads))

    for i, cfg in enumerate(configs):
        throughputs = []
        for wl in workloads:
            subset = results_df[(results_df["config"] == cfg) & (results_df["workload"] == wl)]
            if len(subset) > 0:
                total_time_s = (subset["end_time"].max() - subset["arrival"].min()) * NS_TO_S
                throughputs.append(len(subset) / total_time_s if total_time_s > 0 else 0)
            else:
                throughputs.append(0)

        offset = (i - n_configs / 2 + 0.5) * bar_width
        ax.bar(x + offset, throughputs, bar_width,
               label=get_label(cfg), color=get_color(cfg),
               edgecolor='white', linewidth=0.5)

    ax.set_xlabel("Workload")
    ax.set_ylabel("Request Throughput (req/s)")
    ax.set_title(f"Request Throughput — {phase}")
    ax.set_xticks(x)
    ax.set_xticklabels([get_label(w) for w in workloads], rotation=15, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    save_figure(fig, f"{output_dir}/{phase}_throughput_bar")


# ═══════════════════════ Main ═════════════════════════════════════════

def analyze_phase(results_dir, phase, output_dir):
    """Run all analysis for a single phase."""
    print(f"\n{'='*60}")
    print(f"  Analyzing {phase}")
    print(f"{'='*60}")

    results_df = load_all_results(results_dir, phase)
    ts_df = load_all_timeseries(results_dir, phase)
    tier_stats_df = load_all_tier_stats(results_dir, phase)

    if results_df.empty:
        print(f"  No results found for {phase} in {results_dir}")
        return

    print(f"  Loaded {len(results_df)} request records across {len(results_df['config'].unique())} configs")
    
    if not ts_df.empty:
        print(f"  Loaded {len(ts_df)} time-series rows")
    if not tier_stats_df.empty:
        print(f"  Loaded tier stats for {len(tier_stats_df)} instance-experiments")

    # Generate summary table
    summary = generate_summary_table(results_df, f"{output_dir}/{phase}_summary")
    if not summary.empty:
        print(f"\n  Summary table ({len(summary)} rows):")
        print(summary.to_string(index=False))

    # Category 1: Latency bars
    print("\n  Plotting latency bar charts...")
    plot_latency_bars(results_df, output_dir, phase)

    # Category 1b: Workload comparison (for single-config phases)
    plot_workload_latency_comparison(results_df, output_dir, phase)

    # Category 2: Latency CDFs + ITL box
    print("  Plotting latency CDFs...")
    plot_latency_cdfs(results_df, output_dir, phase)
    plot_itl_boxplot(results_df, output_dir, phase)

    # Category 3: Memory time-series
    if not ts_df.empty:
        print("  Plotting memory utilization time-series...")
        plot_memory_timeseries(ts_df, output_dir, phase)
        plot_memory_pressure(ts_df, output_dir, phase)

    # Category 3b: Memory efficiency
    if not ts_df.empty:
        print("  Plotting memory efficiency...")
        plot_memory_efficiency(ts_df, results_df, output_dir, phase)

    # Category 4: Cache efficiency (skip if no prefix caching data)
    print("  Plotting cache efficiency...")
    if not ts_df.empty:
        plot_cache_hit_rates(ts_df, output_dir, phase)
    plot_cache_hit_bar(results_df, output_dir, phase)
    plot_cache_hit_scatter(results_df, output_dir, phase)

    # Category 5: Migration overhead
    if not tier_stats_df.empty:
        print("  Plotting migration overhead...")
        plot_migration_bars(tier_stats_df, output_dir, phase)
        plot_migration_breakdown(tier_stats_df, output_dir, phase)
    if not ts_df.empty:
        plot_migration_over_time(ts_df, output_dir, phase)

    # Category 5b: Eviction-latency correlation
    if not tier_stats_df.empty:
        print("  Plotting eviction-latency correlation...")
        plot_eviction_latency_correlation(results_df, tier_stats_df, output_dir, phase)

    # Category 6: CXL sensitivity (Phase C specific)
    if "phaseC" in phase or phase == "all":
        print("  Plotting CXL sensitivity...")
        plot_cxl_heatmap(results_df, output_dir, phase)
        plot_cxl_sensitivity_lines(results_df, output_dir, phase)

    # Category 7: Block size (Phase D specific)
    if "phaseD" in phase or phase == "all":
        print("  Plotting block size sensitivity...")
        plot_block_size(results_df, output_dir, phase)

    # Category 8: Summary / Pareto
    print("  Plotting summary dashboard...")
    plot_pareto_frontier(results_df, tier_stats_df, output_dir, phase)
    plot_throughput_comparison(results_df, output_dir, phase)

    print(f"\n  Done! Figures saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Analyze tiered KV cache experiment results")
    parser.add_argument("--results-dir", type=str, default="output/tiered_kv",
                       help="Root directory of experiment results")
    parser.add_argument("--phase", type=str, default="all",
                       help="Phase to analyze (phaseA/phaseB/phaseC/phaseD/all)")
    parser.add_argument("--output-dir", type=str, default="figures/tiered_kv",
                       help="Directory for generated figures")
    parser.add_argument("--format", type=str, nargs='+', default=['pdf', 'png'],
                       help="Output figure formats (e.g., pdf png svg)")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir

    # Discover available phases
    experiments = discover_experiments(results_dir)
    available_phases = sorted(set(e["phase"] for e in experiments))

    if not available_phases:
        print(f"No experiment results found in {results_dir}")
        print("Run 'python benchmarks/run_baseline.py' first to generate results.")
        sys.exit(1)

    print(f"Available phases: {available_phases}")

    if args.phase == "all":
        phases = available_phases
    else:
        # Accept both "A" and "phaseA" formats
        p = args.phase if args.phase.startswith("phase") else f"phase{args.phase}"
        phases = [p]

    for phase in phases:
        if phase not in available_phases:
            print(f"Warning: Phase '{phase}' has no results, skipping.")
            continue
        analyze_phase(results_dir, phase, output_dir)


if __name__ == "__main__":
    main()
