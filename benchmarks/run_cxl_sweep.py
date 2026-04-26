#!/usr/bin/env python3
"""
run_cxl_sweep.py — CXL memory size sweep experiment.

Varies CXL capacity from 0 (no CXL, CPU-only fallback) through 4/8/16/32/64/128/256 GB,
running sharegpt_300 and fixed_256 workloads for each.

Usage:
    python benchmarks/run_cxl_sweep.py                  # run all
    python benchmarks/run_cxl_sweep.py --dry-run        # print commands only
    python benchmarks/run_cxl_sweep.py --plot-only       # skip runs, just plot
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "cluster_config"
OUTPUT_DIR = ROOT / "output" / "tiered_kv" / "cxl_sweep"
FIGURE_DIR = ROOT / "figures" / "cxl_sweep"
PYTHON = str(ROOT / "env" / "bin" / "python")

CXL_SIZES_GB = [0, 4, 8, 16, 32, 64, 128, 256]

WORKLOADS = {
    "sharegpt_300": {"dataset": "dataset/sharegpt_req300_rate10_llama.jsonl"},
    "fixed_256":    {"dataset": "dataset/fixed_in128_out512_req256_rate10.jsonl"},
}

# CXL device parameters (constant across sweep)
CXL_LATENCY = 150   # ns
CXL_BW = 120        # GB/s
CXL_NUM_DEVICES = 1


def make_config(cxl_size_gb: int) -> dict:
    """Generate a cluster config dict for the given CXL size."""
    cfg = {
        "num_nodes": 1,
        "link_bw": 112,
        "link_latency": 0,
        "nodes": [{
            "num_instances": 1,
            "cpu_mem": {"mem_size": 128, "mem_bw": 256, "mem_latency": 0},
            "instances": [{
                "model_name": "meta-llama/Llama-3.1-8B",
                "hardware": "A6000",
                "npu_mem": {"mem_size": 18, "mem_bw": 768, "mem_latency": 0},
                "npu_num": 1,
                "npu_group": 1,
                "pd_type": None,
                "placement": {
                    "default": {
                        "weights": "npu",
                        "kv_loc": "npu",
                        "kv_evict_loc": "cpu"
                    }
                }
            }]
        }]
    }
    if cxl_size_gb > 0:
        cfg["cxl_mem"] = {
            "mem_size": cxl_size_gb,
            "mem_latency": CXL_LATENCY,
            "mem_bw": CXL_BW,
            "num_devices": CXL_NUM_DEVICES,
        }
    return cfg


def write_configs():
    """Write all sweep configs, return {size_gb: path}."""
    paths = {}
    for sz in CXL_SIZES_GB:
        tag = f"cxl_{sz}GB" if sz > 0 else "no_cxl"
        p = CONFIG_DIR / f"sweep_{tag}.json"
        with open(p, "w") as f:
            json.dump(make_config(sz), f, indent=4)
        # Return relative path from ROOT so main.py's ../prefix works
        paths[sz] = str(p.relative_to(ROOT))
        print(f"  Config: {p.name}  (CXL = {sz} GB)")
    return paths


def run_experiments(configs: dict, dry_run=False):
    """Run all (cxl_size × workload) experiments."""
    total = len(CXL_SIZES_GB) * len(WORKLOADS)
    done = 0

    for sz in CXL_SIZES_GB:
        tag = f"cxl_{sz}GB" if sz > 0 else "no_cxl"
        cfg_path = configs[sz]

        for wl_name, wl_info in WORKLOADS.items():
            done += 1
            out_dir = OUTPUT_DIR / tag / wl_name
            result_csv = out_dir / "result.csv"
            ts_csv = out_dir / "timeseries.csv"

            # Skip if already done
            if result_csv.exists():
                print(f"  [{done}/{total}] SKIP {tag}/{wl_name} (already exists)")
                continue

            cmd = [
                PYTHON, "main.py",
                "--cluster-config", cfg_path,
                "--dataset", wl_info["dataset"],
                "--output", str(result_csv.relative_to(ROOT)),
                "--timeseries-output", str(ts_csv.relative_to(ROOT)),
                "--block-size", "16",
            ]

            print(f"  [{done}/{total}] RUN  {tag}/{wl_name} ...")
            if dry_run:
                print(f"    CMD: {' '.join(cmd)}")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                cmd, cwd=str(ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True
            )

            # Save stdout
            (out_dir / "output.txt").write_text(result.stdout)

            if result.returncode != 0:
                print(f"    FAILED (exit {result.returncode})")
                # Print last 10 lines for debugging
                for line in result.stdout.strip().split("\n")[-10:]:
                    print(f"      {line}")
            else:
                # Extract key metrics from output
                for line in result.stdout.strip().split("\n"):
                    if "NPU→CXL evictions:" in line or "NPU→CPU evictions:" in line:
                        print(f"    {line.strip()}")
                    if "Total data moved:" in line:
                        print(f"    {line.strip()}")
                print(f"    OK")


def plot_sweep():
    """Generate comparison plots across CXL sizes."""
    import numpy as np
    import pandas as pd

    # Lazy imports so running experiments doesn't require matplotlib
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from plot_utils import (
        _setup_matplotlib, save_figure, BYTES_TO_MB, BYTES_TO_GB, NS_TO_MS, NS_TO_S,
        load_result_csv, load_timeseries_csv, load_tier_stats,
    )
    plt, sns = _setup_matplotlib()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Collect data ──
    records = []
    ts_frames = []

    for sz in CXL_SIZES_GB:
        tag = f"cxl_{sz}GB" if sz > 0 else "no_cxl"
        for wl_name in WORKLOADS:
            base = OUTPUT_DIR / tag / wl_name

            result_csv = base / "result.csv"
            ts_csv = base / "timeseries.csv"
            tier_json = base / "result_tier_stats.json"

            if not result_csv.exists():
                continue

            df = load_result_csv(str(result_csv))
            rec = {
                "cxl_gb": sz,
                "label": f"{sz}GB" if sz > 0 else "No CXL",
                "workload": wl_name,
                "mean_TTFT_ms": df["TTFT_ms"].mean() if "TTFT_ms" in df else np.nan,
                "p99_TTFT_ms": df["TTFT_ms"].quantile(0.99) if "TTFT_ms" in df else np.nan,
                "mean_TPOT_ms": df["TPOT_ms"].mean() if "TPOT_ms" in df else np.nan,
                "p99_TPOT_ms": df["TPOT_ms"].quantile(0.99) if "TPOT_ms" in df else np.nan,
                "mean_ITL_ms": df["mean_ITL_ms"].mean() if "mean_ITL_ms" in df else np.nan,
                "num_requests": len(df),
            }

            # Tier stats
            if tier_json.exists():
                ts_data = load_tier_stats(str(tier_json))
                # Sum across instances
                for key in ["evict_npu_to_cpu_bytes", "evict_npu_to_cxl_bytes",
                            "load_cpu_to_npu_bytes", "load_cxl_to_npu_bytes",
                            "evict_npu_to_cpu_count", "evict_npu_to_cxl_count",
                            "load_cpu_to_npu_count", "load_cxl_to_npu_count"]:
                    rec[key] = sum(inst.get(key, 0) for inst in ts_data.values())

            records.append(rec)

            # Timeseries
            if ts_csv.exists():
                tsdf = load_timeseries_csv(str(ts_csv))
                tsdf["cxl_gb"] = sz
                tsdf["label"] = f"{sz}GB" if sz > 0 else "No CXL"
                tsdf["workload"] = wl_name
                ts_frames.append(tsdf)

    if not records:
        print("  No data found for CXL sweep. Run experiments first.")
        return

    summary = pd.DataFrame(records)
    ts_all = pd.concat(ts_frames, ignore_index=True) if ts_frames else pd.DataFrame()

    print(f"  Loaded {len(summary)} experiment results")
    print(summary[["cxl_gb", "workload", "mean_TTFT_ms", "mean_TPOT_ms"]].to_string(index=False))

    # Color palette: gradient from dark (no CXL) to bright (256GB)
    n = len(CXL_SIZES_GB)
    cmap = plt.cm.viridis
    size_colors = {sz: cmap(i / (n - 1)) for i, sz in enumerate(CXL_SIZES_GB)}
    size_labels = {sz: (f"{sz}GB" if sz > 0 else "No CXL") for sz in CXL_SIZES_GB}

    # ════════════════ Plot 1: Latency vs CXL Size (line) ════════════════
    for metric, col, ylabel in [
        ("TTFT", "mean_TTFT_ms", "Mean TTFT (ms)"),
        ("TPOT", "mean_TPOT_ms", "Mean TPOT (ms)"),
        ("P99 TTFT", "p99_TTFT_ms", "P99 TTFT (ms)"),
        ("P99 TPOT", "p99_TPOT_ms", "P99 TPOT (ms)"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for wl in WORKLOADS:
            wl_data = summary[summary["workload"] == wl].sort_values("cxl_gb")
            if wl_data.empty:
                continue
            ax.plot(wl_data["cxl_gb"], wl_data[col], "o-", label=wl, linewidth=2, markersize=8)

        ax.set_xlabel("CXL Memory Size (GB)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric} vs CXL Memory Size")
        ax.set_xticks(CXL_SIZES_GB)
        ax.set_xticklabels([size_labels[s] for s in CXL_SIZES_GB], rotation=30, ha="right")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        save_figure(fig, str(FIGURE_DIR / f"cxl_sweep_{metric.lower().replace(' ', '_')}"))

    # ════════════════ Plot 2: Migration Breakdown vs CXL Size ════════════
    for wl in WORKLOADS:
        wl_data = summary[summary["workload"] == wl].sort_values("cxl_gb")
        if wl_data.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(wl_data))
        labels = wl_data["label"].values
        w = 0.35

        evict_cxl = wl_data.get("evict_npu_to_cxl_bytes", pd.Series(dtype=float)).fillna(0).values / (1024**2)
        evict_cpu = wl_data.get("evict_npu_to_cpu_bytes", pd.Series(dtype=float)).fillna(0).values / (1024**2)
        reload_cxl = wl_data.get("load_cxl_to_npu_bytes", pd.Series(dtype=float)).fillna(0).values / (1024**2)
        reload_cpu = wl_data.get("load_cpu_to_npu_bytes", pd.Series(dtype=float)).fillna(0).values / (1024**2)

        ax = axes[0]
        ax.bar(x - w/2, evict_cxl, w, label="NPU→CXL", color="#e377c2")
        ax.bar(x - w/2, evict_cpu, w, bottom=evict_cxl, label="NPU→CPU", color="#d62728")
        ax.bar(x + w/2, reload_cxl, w, label="CXL→NPU", color="#bcbd22")
        ax.bar(x + w/2, reload_cpu, w, bottom=reload_cxl, label="CPU→NPU", color="#2ca02c")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Data Volume (MB)")
        ax.set_title(f"Migration Volume — {wl}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Event counts
        ax = axes[1]
        ec_cxl = wl_data.get("evict_npu_to_cxl_count", pd.Series(dtype=float)).fillna(0).values
        ec_cpu = wl_data.get("evict_npu_to_cpu_count", pd.Series(dtype=float)).fillna(0).values
        rc_cxl = wl_data.get("load_cxl_to_npu_count", pd.Series(dtype=float)).fillna(0).values
        rc_cpu = wl_data.get("load_cpu_to_npu_count", pd.Series(dtype=float)).fillna(0).values

        ax.bar(x - w/2, ec_cxl, w, label="NPU→CXL", color="#e377c2")
        ax.bar(x - w/2, ec_cpu, w, bottom=ec_cxl, label="NPU→CPU", color="#d62728")
        ax.bar(x + w/2, rc_cxl, w, label="CXL→NPU", color="#bcbd22")
        ax.bar(x + w/2, rc_cpu, w, bottom=rc_cxl, label="CPU→NPU", color="#2ca02c")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Event Count")
        ax.set_title(f"Migration Events — {wl}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"CXL Size Sweep — {wl}", fontsize=14)
        fig.tight_layout()
        save_figure(fig, str(FIGURE_DIR / f"cxl_sweep_migration_{wl}"))

    # ════════════════ Plot 3: Stacked area — total data moved ════════════
    fig, ax = plt.subplots(figsize=(10, 5))
    for wl in WORKLOADS:
        wl_data = summary[summary["workload"] == wl].sort_values("cxl_gb")
        if wl_data.empty:
            continue
        total_cxl = wl_data.get("evict_npu_to_cxl_bytes", pd.Series(dtype=float)).fillna(0).values + \
                    wl_data.get("load_cxl_to_npu_bytes", pd.Series(dtype=float)).fillna(0).values
        total_cpu = wl_data.get("evict_npu_to_cpu_bytes", pd.Series(dtype=float)).fillna(0).values + \
                    wl_data.get("load_cpu_to_npu_bytes", pd.Series(dtype=float)).fillna(0).values
        total_mb = (total_cxl + total_cpu) / (1024**2)
        cxl_mb = total_cxl / (1024**2)

        ax.plot(wl_data["cxl_gb"].values, total_mb, "o-", label=f"{wl} (total)", linewidth=2)
        ax.plot(wl_data["cxl_gb"].values, cxl_mb, "s--", label=f"{wl} (CXL portion)", linewidth=1.5, alpha=0.7)

    ax.set_xlabel("CXL Memory Size (GB)")
    ax.set_ylabel("Total Data Moved (MB)")
    ax.set_title("Total Migration Volume vs CXL Size")
    ax.set_xticks(CXL_SIZES_GB)
    ax.set_xticklabels([size_labels[s] for s in CXL_SIZES_GB], rotation=30, ha="right")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, str(FIGURE_DIR / "cxl_sweep_total_migration"))

    # ════════════════ Plot 4: Memory timeseries per CXL size ════════════
    if not ts_all.empty:
        for wl in WORKLOADS:
            wl_ts = ts_all[ts_all["workload"] == wl]
            if wl_ts.empty:
                continue

            sizes_with_data = sorted(wl_ts["cxl_gb"].unique())
            n_plots = len(sizes_with_data)
            if n_plots == 0:
                continue

            fig, axes = plt.subplots(2, (n_plots + 1) // 2, figsize=(6 * ((n_plots + 1) // 2), 10), squeeze=False)
            axes_flat = axes.flatten()

            for i, sz in enumerate(sizes_with_data):
                if i >= len(axes_flat):
                    break
                ax = axes_flat[i]
                data = wl_ts[(wl_ts["cxl_gb"] == sz) & (wl_ts["instance_id"] == 0)].sort_values("sim_time_s")
                if data.empty:
                    continue

                t = data["sim_time_s"].values
                npu_gb = data["npu_used_gb"].values if "npu_used_gb" in data else np.zeros_like(t)
                cxl_gb = data["cxl_used_gb"].values if "cxl_used_gb" in data else np.zeros_like(t)
                cpu_gb = data["cpu_used_gb"].values if "cpu_used_gb" in data else np.zeros_like(t)
                npu_cap = data["npu_total_gb"].values if "npu_total_gb" in data else np.zeros_like(t)

                ax.fill_between(t, 0, npu_gb, alpha=0.7, color="#1f77b4", label="NPU")
                ax.fill_between(t, npu_gb, npu_gb + cxl_gb, alpha=0.7, color="#ff7f0e", label="CXL")
                ax.fill_between(t, npu_gb + cxl_gb, npu_gb + cxl_gb + cpu_gb, alpha=0.7, color="#2ca02c", label="CPU")
                if npu_cap.any():
                    ax.axhline(y=npu_cap[0], color="red", linestyle="--", alpha=0.5, label="NPU Cap")

                lbl = f"{sz}GB" if sz > 0 else "No CXL"
                ax.set_title(lbl, fontsize=11)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Memory (GB)")
                if i == 0:
                    ax.legend(fontsize=7, loc="upper left")
                ax.grid(alpha=0.3)

            # Hide unused axes
            for j in range(i + 1, len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.suptitle(f"Memory Usage Over Time — {wl} — CXL Size Sweep", fontsize=14)
            fig.tight_layout()
            save_figure(fig, str(FIGURE_DIR / f"cxl_sweep_memory_{wl}"))

    # ════════════════ Plot 5: Combined latency bar chart ════════════════
    for wl in WORKLOADS:
        wl_data = summary[summary["workload"] == wl].sort_values("cxl_gb")
        if wl_data.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        x = np.arange(len(wl_data))
        labels = wl_data["label"].values
        colors = [size_colors[s] for s in wl_data["cxl_gb"]]

        for ax, (metric, col) in zip(axes, [
            ("Mean TTFT (ms)", "mean_TTFT_ms"),
            ("Mean TPOT (ms)", "mean_TPOT_ms"),
            ("Mean ITL (ms)",  "mean_ITL_ms"),
        ]):
            vals = wl_data[col].values
            bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.grid(axis="y", alpha=0.3)
            # Annotate
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.annotate(f"{v:.1f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)

        fig.suptitle(f"Latency Metrics — {wl} — CXL Size Sweep", fontsize=14)
        fig.tight_layout()
        save_figure(fig, str(FIGURE_DIR / f"cxl_sweep_latency_bars_{wl}"))

    # ════════════════ Summary CSV ════════════════
    csv_path = FIGURE_DIR / "cxl_sweep_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\n  Summary CSV: {csv_path}")
    print(f"  Figures saved to: {FIGURE_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="CXL memory size sweep experiment")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--plot-only", action="store_true", help="Skip experiments, just plot")
    args = parser.parse_args()

    print("=" * 60)
    print("  CXL Memory Size Sweep")
    print(f"  Sizes: {CXL_SIZES_GB} GB")
    print(f"  Workloads: {list(WORKLOADS.keys())}")
    print("=" * 60)

    if not args.plot_only:
        print("\n── Generating configs ──")
        configs = write_configs()

        print("\n── Running experiments ──")
        run_experiments(configs, dry_run=args.dry_run)

    if not args.dry_run:
        print("\n── Generating plots ──")
        plot_sweep()


if __name__ == "__main__":
    main()
