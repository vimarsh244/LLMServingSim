#!/usr/bin/env python3
"""
plot_tier_bandwidth_capacity.py - Tier capability visualization.

Generates a compact chart showing tier memory profiles:
- latency (ns)
- bandwidth (GB/s)
- capacity (GB)

Covered tiers: NPU HBM, CPU DRAM, CXL, PCIe NVMe, SSD, and Ethernet.

Usage:
    python benchmarks/plot_tier_bandwidth_capacity.py
"""

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLUSTER_DIR = ROOT / "cluster_config"
OUT_DIR = ROOT / "figures" / "tier_profiles"

TIER_COLORS = {
    "NPU_HBM": "#0f766e",
    "CPU_DRAM": "#2563eb",
    "CXL": "#f59e0b",
    "PCIE_NVME": "#6d28d9",
    "SSD": "#dc2626",
    "ETHERNET": "#4b5563",
}


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_profiles():
    cpu_cfg = _load_json(CLUSTER_DIR / "tiered_kv_tier_cpu_dram.json")
    cxl_cfg = _load_json(CLUSTER_DIR / "tiered_kv_tier_cxl.json")
    nvme_cfg = _load_json(CLUSTER_DIR / "tiered_kv_tier_pcie_nvme.json")
    ssd_cfg = _load_json(CLUSTER_DIR / "tiered_kv_tier_ssd.json")
    eth_cfg = _load_json(CLUSTER_DIR / "tiered_kv_tier_ethernet.json")

    npu = cpu_cfg["nodes"][0]["instances"][0]["npu_mem"]
    cpu = cpu_cfg["nodes"][0]["cpu_mem"]

    rows = [
        {
            "tier": "NPU_HBM",
            "capacity_gb": float(npu["mem_size"]),
            "bandwidth_gbs": float(npu["mem_bw"]),
            "latency_ns": float(npu["mem_latency"]),
        },
        {
            "tier": "CPU_DRAM",
            "capacity_gb": float(cpu["mem_size"]),
            "bandwidth_gbs": float(cpu["mem_bw"]),
            "latency_ns": float(cpu["mem_latency"]),
        },
    ]

    for cfg, name in [
        (cxl_cfg, "CXL"),
        (nvme_cfg, "PCIE_NVME"),
        (ssd_cfg, "SSD"),
        (eth_cfg, "ETHERNET"),
    ]:
        ext = cfg["external_kv_tier"]
        rows.append(
            {
                "tier": name,
                "capacity_gb": float(ext["mem_size"]),
                "bandwidth_gbs": float(ext["mem_bw"]),
                "latency_ns": float(ext["mem_latency"]),
            }
        )

    return pd.DataFrame(rows)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = _extract_profiles()
    df["latency_plot_ns"] = df["latency_ns"].clip(lower=1.0)
    df.to_csv(OUT_DIR / "tier_bandwidth_capacity_table.csv", index=False)

    tiers = df["tier"].tolist()
    colors = [TIER_COLORS.get(t, "#999999") for t in tiers]
    x = range(len(tiers))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Chart 1: Latency vs bandwidth (bubble size = capacity)
    ax = axes[0]
    bubble_sizes = ((df["capacity_gb"].values ** 0.55) * 90).clip(100, 1300)
    ax.scatter(
        df["latency_plot_ns"].values,
        df["bandwidth_gbs"].values,
        s=bubble_sizes,
        c=colors,
        alpha=0.86,
        linewidths=0.8,
        edgecolors="#111827",
    )
    for _, row in df.iterrows():
        ax.annotate(
            f"{row['tier']}\n{row['capacity_gb']:.0f}GB",
            (row["latency_plot_ns"], row["bandwidth_gbs"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=11,
            fontweight="bold",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Memory Latency (ns, log scale)", fontsize=12)
    ax.set_ylabel("Memory Bandwidth (GB/s, log scale)", fontsize=12)
    ax.set_title("Tier Latency vs Bandwidth (bubble size/label = capacity)", fontsize=13)
    ax.grid(which="both", alpha=0.25)
    if (df["latency_ns"] == 0).any():
        ax.text(
            0.02,
            0.02,
            "Note: 0ns values are displayed at 1ns for log-scale plotting.",
            transform=ax.transAxes,
            fontsize=10,
            color="#374151",
        )

    # Chart 2: Bandwidth bars + latency line
    ax = axes[1]
    bars = ax.bar(x, df["bandwidth_gbs"].values, color=colors, alpha=0.88)
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [f"{tier}\n{cap:.0f}GB" for tier, cap in zip(tiers, df["capacity_gb"].values)],
        rotation=16,
        ha="right",
        fontsize=12,
    )
    ax.set_title("Per-Tier Bandwidth and Latency (x-label includes capacity)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, df["bandwidth_gbs"].values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    ax2 = ax.twinx()
    ax2.plot(
        list(x),
        df["latency_plot_ns"].values,
        color="#111827",
        marker="o",
        linewidth=2.0,
        label="Latency",
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("Latency (ns, log scale)", fontsize=12)
    for xi, val, raw in zip(x, df["latency_plot_ns"].values, df["latency_ns"].values):
        ax2.annotate(
            f"{raw:.0f}ns",
            (xi, val),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=12,
            color="#111827",
        )

    ax.tick_params(axis="both", labelsize=11)
    ax2.tick_params(axis="y", labelsize=11)
    axes[0].tick_params(axis="both", labelsize=11)

    fig.suptitle("KV Memory Tier Profiles: Latency, Bandwidth, and Capacity", fontsize=16, fontweight="bold")
    fig.tight_layout()

    png_path = OUT_DIR / "tier_bandwidth_capacity.png"
    pdf_path = OUT_DIR / "tier_bandwidth_capacity.pdf"
    png_path_v2 = OUT_DIR / "tier_latency_bandwidth_overview.png"
    pdf_path_v2 = OUT_DIR / "tier_latency_bandwidth_overview.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path_v2, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path_v2, dpi=300, bbox_inches="tight")

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path_v2}")
    print(f"Saved: {pdf_path_v2}")
    print(f"Saved: {OUT_DIR / 'tier_bandwidth_capacity_table.csv'}")


if __name__ == "__main__":
    main()
