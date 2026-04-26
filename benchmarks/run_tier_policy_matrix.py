#!/usr/bin/env python3
"""
run_tier_policy_matrix.py - Sweep external KV tiers and eviction policies.

This runner executes the same workload matrix across:
  - KV spill tiers: CPU DRAM, CXL, PCIe NVMe, SSD, Ethernet-backed remote memory
    - KV eviction policies: tail, fifo, lru, largest_kv, smallest_kv, random, evicpress

Outputs are written under:
  output/tiered_kv/tier_policy_matrix/{tier}/{policy}/{workload}/

And aggregate CSVs are generated in:
  output/tiered_kv/tier_policy_matrix/
    - run_manifest.csv
    - metric_summary.csv
    - policy_delta_summary.csv

Usage examples:
  python benchmarks/run_tier_policy_matrix.py --dry-run
    python benchmarks/run_tier_policy_matrix.py --tiers cpu_dram cxl pcie_nvme --policies tail fifo lru largest_kv
  python benchmarks/run_tier_policy_matrix.py --workloads sharegpt_100 fixed_256 --rerun
    python benchmarks/run_tier_policy_matrix.py --workloads sharegpt_750 sharegpt_1000 sharegpt_1500 --policies all --rerun
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference_serving.eviction_policies import get_registered_policy_names

OUTPUT_ROOT = ROOT_DIR / "output" / "tiered_kv" / "tier_policy_matrix"

TIER_CONFIGS = {
    "cpu_dram": "cluster_config/tiered_kv_tier_cpu_dram.json",
    "cxl": "cluster_config/tiered_kv_tier_cxl.json",
    "pcie_nvme": "cluster_config/tiered_kv_tier_pcie_nvme.json",
    "ssd": "cluster_config/tiered_kv_tier_ssd.json",
    "ethernet": "cluster_config/tiered_kv_tier_ethernet.json",
}

WORKLOADS = {
    "sharegpt_100": {
        "dataset": "dataset/sharegpt_req100_rate10_llama.jsonl",
        "num_req": 100,
    },
    "sharegpt_300": {
        "dataset": "dataset/sharegpt_req300_rate10_llama.jsonl",
        "num_req": 300,
    },
    "sharegpt_750": {
        "dataset": "dataset/sharegpt_req750_rate10_llama.jsonl",
        "num_req": 750,
    },
    "sharegpt_1000": {
        "dataset": "dataset/sharegpt_req1000_rate10_llama.jsonl",
        "num_req": 1000,
    },
    "sharegpt_1500": {
        "dataset": "dataset/sharegpt_req1500_rate10_llama.jsonl",
        "num_req": 1500,
    },
    "fixed_256": {
        "dataset": "dataset/fixed_in128_out512_req256_rate10.jsonl",
        "num_req": 256,
    },
    "prefix_stress": {
        "dataset": "dataset/prefix_pool_stress.jsonl",
        "num_req": 100,
    },
}

POLICIES = get_registered_policy_names()


@dataclass
class RunRecord:
    tier: str
    policy: str
    workload: str
    returncode: int
    status: str
    output_dir: str
    result_csv: str
    tier_stats_json: str
    stdout_log: str


def _select_items(selected: List[str], table: Dict[str, object], label: str) -> List[str]:
    if not selected or "all" in selected:
        return list(table.keys())
    unknown = [item for item in selected if item not in table]
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}")
    return selected


def _select_policies(selected: List[str]) -> List[str]:
    if not selected or "all" in selected:
        return list(POLICIES)
    unknown = [p for p in selected if p not in POLICIES]
    if unknown:
        raise ValueError(f"Unknown policies: {unknown}")
    return selected


def _build_command(
    cluster_config: str,
    dataset: str,
    num_req: int,
    policy: str,
    network_backend: str,
    result_csv: Path,
    timeseries_csv: Path,
) -> List[str]:
    return [
        sys.executable,
        "main.py",
        "--cluster-config",
        cluster_config,
        "--dataset",
        dataset,
        "--num-req",
        str(num_req),
        "--output",
        str(result_csv.relative_to(ROOT_DIR)),
        "--timeseries-output",
        str(timeseries_csv.relative_to(ROOT_DIR)),
        "--network-backend",
        network_backend,
        "--kv-eviction-policy",
        policy,
    ]


def _load_metrics(result_csv: Path) -> Dict[str, float]:
    if not result_csv.exists():
        return {}

    df = pd.read_csv(result_csv)
    df.columns = df.columns.str.strip()

    metrics: Dict[str, float] = {}

    def _mean_ms(col: str, out_key: str):
        if col in df.columns and len(df) > 0:
            metrics[out_key] = float(df[col].mean() * 1e-6)

    def _p99_ms(col: str, out_key: str):
        if col in df.columns and len(df) > 0:
            metrics[out_key] = float(df[col].quantile(0.99) * 1e-6)

    _mean_ms("TTFT", "mean_ttft_ms")
    _mean_ms("TPOT", "mean_tpot_ms")
    _mean_ms("latency", "mean_latency_ms")
    _p99_ms("TTFT", "p99_ttft_ms")
    _p99_ms("TPOT", "p99_tpot_ms")
    metrics["num_requests"] = float(len(df))

    return metrics


def _load_tier_totals(tier_stats_json: Path) -> Dict[str, float]:
    if not tier_stats_json.exists():
        return {}

    with open(tier_stats_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    keys = [
        "evict_npu_to_cpu_bytes",
        "evict_npu_to_cxl_bytes",
        "load_cpu_to_npu_bytes",
        "load_cxl_to_npu_bytes",
    ]
    out = {k: 0.0 for k in keys}

    for inst_stats in raw.values():
        for k in keys:
            out[k] += float(inst_stats.get(k, 0.0))

    out["tier_transition_bytes_total"] = sum(out.values())
    out["tier_transition_mb_total"] = out["tier_transition_bytes_total"] / (1024.0 * 1024.0)
    return out


def _run_one(
    tier: str,
    policy: str,
    workload_name: str,
    cluster_config: str,
    dataset: str,
    num_req: int,
    network_backend: str,
    timeout: int,
    dry_run: bool,
    rerun: bool,
) -> RunRecord:
    out_dir = OUTPUT_ROOT / tier / policy / workload_name
    out_dir.mkdir(parents=True, exist_ok=True)

    result_csv = out_dir / "result.csv"
    timeseries_csv = out_dir / "timeseries.csv"
    tier_stats_json = out_dir / "result_tier_stats.json"
    stdout_log = out_dir / "output.txt"

    if result_csv.exists() and not rerun and not dry_run:
        return RunRecord(
            tier=tier,
            policy=policy,
            workload=workload_name,
            returncode=0,
            status="skipped_existing",
            output_dir=str(out_dir),
            result_csv=str(result_csv),
            tier_stats_json=str(tier_stats_json),
            stdout_log=str(stdout_log),
        )

    cmd = _build_command(
        cluster_config=cluster_config,
        dataset=dataset,
        num_req=num_req,
        policy=policy,
        network_backend=network_backend,
        result_csv=result_csv,
        timeseries_csv=timeseries_csv,
    )

    print(f"RUN [{tier}][{policy}] {workload_name}")
    print("  " + " ".join(cmd))

    if dry_run:
        return RunRecord(
            tier=tier,
            policy=policy,
            workload=workload_name,
            returncode=0,
            status="dry_run",
            output_dir=str(out_dir),
            result_csv=str(result_csv),
            tier_stats_json=str(tier_stats_json),
            stdout_log=str(stdout_log),
        )

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        stdout_log.write_text(completed.stdout, encoding="utf-8")
        status = "ok" if completed.returncode == 0 else "failed"
        return RunRecord(
            tier=tier,
            policy=policy,
            workload=workload_name,
            returncode=completed.returncode,
            status=status,
            output_dir=str(out_dir),
            result_csv=str(result_csv),
            tier_stats_json=str(tier_stats_json),
            stdout_log=str(stdout_log),
        )
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        stdout_log.write_text(partial, encoding="utf-8")
        return RunRecord(
            tier=tier,
            policy=policy,
            workload=workload_name,
            returncode=124,
            status="timeout",
            output_dir=str(out_dir),
            result_csv=str(result_csv),
            tier_stats_json=str(tier_stats_json),
            stdout_log=str(stdout_log),
        )


def _build_metric_summary(manifest_df: pd.DataFrame) -> pd.DataFrame:
    valid_status = {"ok", "skipped_existing"}
    rows = []

    for rec in manifest_df.itertuples(index=False):
        if rec.status not in valid_status:
            continue

        result_path = Path(rec.result_csv)
        if not result_path.exists():
            continue

        tier_path = Path(rec.tier_stats_json)
        metrics = _load_metrics(result_path)
        tier = _load_tier_totals(tier_path)

        row = {
            "tier": rec.tier,
            "policy": rec.policy,
            "workload": rec.workload,
            "status": rec.status,
        }
        row.update(metrics)
        row.update(tier)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _safe_pct_delta(base: float, other: float) -> float:
    if base == 0:
        return 0.0
    return (other - base) / base * 100.0


def _build_policy_delta_summary(summary_df: pd.DataFrame, baseline_policy: str = "tail") -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows = []
    grouped = summary_df.groupby(["tier", "workload"])

    for (tier, workload), group in grouped:
        by_policy = {row.policy: row for row in group.itertuples(index=False)}
        if baseline_policy not in by_policy:
            continue

        base_row = by_policy[baseline_policy]
        for policy in sorted(by_policy.keys()):
            if policy == baseline_policy:
                continue
            cmp_row = by_policy[policy]

            row = {
                "tier": tier,
                "workload": workload,
                "baseline_policy": baseline_policy,
                "compare_policy": policy,
            }

            for key in [
                "mean_ttft_ms",
                "mean_tpot_ms",
                "mean_latency_ms",
                "p99_ttft_ms",
                "p99_tpot_ms",
                "tier_transition_mb_total",
            ]:
                base_val = float(getattr(base_row, key, 0.0) or 0.0)
                cmp_val = float(getattr(cmp_row, key, 0.0) or 0.0)
                row[f"{baseline_policy}_{key}"] = base_val
                row[f"{policy}_{key}"] = cmp_val
                row[f"delta_{key}"] = cmp_val - base_val
                row[f"delta_pct_{key}"] = _safe_pct_delta(base_val, cmp_val)

            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Run tier and eviction-policy matrix for tiered KV cache")
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["all"],
        help="Tier names from TIER_CONFIGS or 'all'",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["tail", "fifo", "lru", "largest_kv"],
        help="Policy names or 'all'",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["sharegpt_100", "prefix_stress"],
        help="Workload names from WORKLOADS or 'all'",
    )
    parser.add_argument(
        "--network-backend",
        type=str,
        choices=["analytical", "ns3"],
        default="analytical",
        help="Backend for network simulation",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--rerun", action="store_true", help="Re-run even if outputs already exist")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-run timeout in seconds")
    parser.add_argument("--num-req-override", type=int, default=None, help="Override num_req for all workloads")
    args = parser.parse_args()

    selected_tiers = _select_items(args.tiers, TIER_CONFIGS, "tiers")
    selected_policies = _select_policies(args.policies)
    selected_workloads = _select_items(args.workloads, WORKLOADS, "workloads")

    print("=" * 96)
    print("Tier + Policy Matrix Runner")
    print(f"Tiers:     {selected_tiers}")
    print(f"Policies:  {selected_policies}")
    print(f"Workloads: {selected_workloads}")
    print(f"Backend:   {args.network_backend}")
    print(f"Output:    {OUTPUT_ROOT}")
    print("=" * 96)

    records: List[RunRecord] = []
    total = len(selected_tiers) * len(selected_policies) * len(selected_workloads)
    idx = 0

    for tier in selected_tiers:
        for policy in selected_policies:
            for wl_name in selected_workloads:
                idx += 1
                wl = WORKLOADS[wl_name]
                num_req = args.num_req_override if args.num_req_override is not None else wl["num_req"]
                print(f"[{idx}/{total}] {tier} | {policy} | {wl_name}")
                rec = _run_one(
                    tier=tier,
                    policy=policy,
                    workload_name=wl_name,
                    cluster_config=TIER_CONFIGS[tier],
                    dataset=wl["dataset"],
                    num_req=num_req,
                    network_backend=args.network_backend,
                    timeout=args.timeout,
                    dry_run=args.dry_run,
                    rerun=args.rerun,
                )
                records.append(rec)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.DataFrame([r.__dict__ for r in records])
    manifest_path = OUTPUT_ROOT / "run_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    metric_df = _build_metric_summary(manifest_df)
    metric_path = OUTPUT_ROOT / "metric_summary.csv"
    if not metric_df.empty:
        metric_df.to_csv(metric_path, index=False)

    delta_df = _build_policy_delta_summary(metric_df, baseline_policy="tail")
    delta_path = OUTPUT_ROOT / "policy_delta_summary.csv"
    if not delta_df.empty:
        delta_df.to_csv(delta_path, index=False)

    print("=" * 96)
    print("Run summary")
    if not manifest_df.empty:
        print(manifest_df["status"].value_counts().to_string())
    print(f"Manifest: {manifest_path}")
    if metric_df.empty:
        print("Metric summary not generated (no completed runs).")
    else:
        print(f"Metrics:  {metric_path}")
    if delta_df.empty:
        print("Policy delta summary not generated (no comparable baseline pairs).")
    else:
        print(f"Deltas:   {delta_path}")
        preview_cols = [
            "tier",
            "workload",
            "compare_policy",
            "delta_pct_mean_tpot_ms",
            "delta_pct_p99_tpot_ms",
            "delta_pct_tier_transition_mb_total",
        ]
        preview_cols = [c for c in preview_cols if c in delta_df.columns]
        print(delta_df[preview_cols].head(12).to_string(index=False))
    print("=" * 96)


if __name__ == "__main__":
    main()
