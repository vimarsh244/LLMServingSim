#!/usr/bin/env python3
"""
run_backend_diff.py - Compare simulator backends on identical tiered-KV scenarios.

This script runs the same config/workload matrix against multiple network backends
(typically analytical and ns3), then writes:
  1) run_manifest.csv: run status and output paths
  2) backend_diff_summary.csv: metric deltas across backend pairs

Usage examples:
  python benchmarks/run_backend_diff.py
  python benchmarks/run_backend_diff.py --dry-run
  python benchmarks/run_backend_diff.py --configs npu_cpu npu_cxl_cpu --workloads sharegpt_100
  python benchmarks/run_backend_diff.py --backends analytical ns3 --rerun
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

OUTPUT_ROOT = ROOT_DIR / "output" / "tiered_kv" / "backend_diff"

CONFIGS = {
    "npu_cpu": "cluster_config/tiered_kv_npu_cpu.json",
    "npu_cxl_cpu": "cluster_config/tiered_kv_npu_cxl_cpu.json",
    "npu_cxl_fast": "cluster_config/tiered_kv_npu_cxl_fast.json",
    "npu_cxl_slow": "cluster_config/tiered_kv_npu_cxl_slow.json",
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
    "fixed_256": {
        "dataset": "dataset/fixed_in128_out512_req256_rate10.jsonl",
        "num_req": 256,
    },
    "prefix_stress": {
        "dataset": "dataset/prefix_pool_stress.jsonl",
        "num_req": 100,
    },
}


@dataclass
class RunRecord:
    backend: str
    config: str
    workload: str
    returncode: int
    status: str
    output_dir: str
    result_csv: str
    tier_stats_json: str
    stdout_log: str


def _select_items(selected: List[str], table: Dict[str, str], label: str) -> List[str]:
    if not selected or "all" in selected:
        return list(table.keys())
    unknown = [item for item in selected if item not in table]
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}")
    return selected


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


def _build_command(
    backend: str,
    cluster_config: str,
    dataset: str,
    num_req: int,
    kv_eviction_policy: str,
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
        backend,
        "--kv-eviction-policy",
        kv_eviction_policy,
    ]


def _run_one(
    backend: str,
    config_name: str,
    workload_name: str,
    cluster_config: str,
    dataset: str,
    num_req: int,
    kv_eviction_policy: str,
    timeout: int,
    dry_run: bool,
    rerun: bool,
) -> RunRecord:
    out_dir = OUTPUT_ROOT / backend / config_name / workload_name
    out_dir.mkdir(parents=True, exist_ok=True)

    result_csv = out_dir / "result.csv"
    timeseries_csv = out_dir / "timeseries.csv"
    tier_stats_json = out_dir / "result_tier_stats.json"
    stdout_log = out_dir / "output.txt"

    if result_csv.exists() and not rerun and not dry_run:
        return RunRecord(
            backend=backend,
            config=config_name,
            workload=workload_name,
            returncode=0,
            status="skipped_existing",
            output_dir=str(out_dir),
            result_csv=str(result_csv),
            tier_stats_json=str(tier_stats_json),
            stdout_log=str(stdout_log),
        )

    cmd = _build_command(
        backend=backend,
        cluster_config=cluster_config,
        dataset=dataset,
        num_req=num_req,
        kv_eviction_policy=kv_eviction_policy,
        result_csv=result_csv,
        timeseries_csv=timeseries_csv,
    )

    print(f"RUN [{backend}] {config_name}/{workload_name}")
    print("  " + " ".join(cmd))

    if dry_run:
        return RunRecord(
            backend=backend,
            config=config_name,
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
            backend=backend,
            config=config_name,
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
        stdout_log.write_text(partial, encoding="utf-8")
        return RunRecord(
            backend=backend,
            config=config_name,
            workload=workload_name,
            returncode=124,
            status="timeout",
            output_dir=str(out_dir),
            result_csv=str(result_csv),
            tier_stats_json=str(tier_stats_json),
            stdout_log=str(stdout_log),
        )


def _safe_pct_delta(base: float, other: float) -> float:
    if base == 0:
        return 0.0
    return (other - base) / base * 100.0


def _build_diff_rows(manifest_df: pd.DataFrame, backends: List[str]) -> pd.DataFrame:
    if len(backends) < 2:
        return pd.DataFrame()

    lhs = backends[0]
    valid_status = {"ok", "skipped_existing"}
    rows = []

    grouped = manifest_df.groupby(["config", "workload"])
    for (config, workload), group in grouped:
        by_backend = {row.backend: row for row in group.itertuples(index=False)}
        if lhs not in by_backend:
            continue

        lhs_row = by_backend[lhs]
        lhs_result_path = Path(lhs_row.result_csv)
        lhs_tier_path = Path(lhs_row.tier_stats_json)
        if lhs_row.status not in valid_status or not lhs_result_path.exists():
            continue

        lhs_metrics = _load_metrics(lhs_result_path)
        lhs_tier = _load_tier_totals(lhs_tier_path)

        for rhs in backends[1:]:
            if rhs not in by_backend:
                continue

            rhs_row = by_backend[rhs]
            rhs_result_path = Path(rhs_row.result_csv)
            rhs_tier_path = Path(rhs_row.tier_stats_json)
            if rhs_row.status not in valid_status or not rhs_result_path.exists():
                continue

            rhs_metrics = _load_metrics(rhs_result_path)
            rhs_tier = _load_tier_totals(rhs_tier_path)

            row = {
                "config": config,
                "workload": workload,
                "baseline_backend": lhs,
                "compare_backend": rhs,
                "baseline_status": lhs_row.status,
                "compare_status": rhs_row.status,
            }

            for key in ["mean_ttft_ms", "mean_tpot_ms", "mean_latency_ms", "p99_ttft_ms", "p99_tpot_ms"]:
                base_val = float(lhs_metrics.get(key, 0.0))
                cmp_val = float(rhs_metrics.get(key, 0.0))
                row[f"{lhs}_{key}"] = base_val
                row[f"{rhs}_{key}"] = cmp_val
                row[f"delta_{key}"] = cmp_val - base_val
                row[f"delta_pct_{key}"] = _safe_pct_delta(base_val, cmp_val)

            for key in [
                "evict_npu_to_cpu_bytes",
                "evict_npu_to_cxl_bytes",
                "load_cpu_to_npu_bytes",
                "load_cxl_to_npu_bytes",
                "tier_transition_bytes_total",
                "tier_transition_mb_total",
            ]:
                base_val = float(lhs_tier.get(key, 0.0))
                cmp_val = float(rhs_tier.get(key, 0.0))
                row[f"{lhs}_{key}"] = base_val
                row[f"{rhs}_{key}"] = cmp_val
                row[f"delta_{key}"] = cmp_val - base_val
                row[f"delta_pct_{key}"] = _safe_pct_delta(base_val, cmp_val)

            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Run backend comparison for tiered KV cache scenarios")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["npu_cpu", "npu_cxl_cpu"],
        help="Config names from CONFIGS or 'all'",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["sharegpt_100", "prefix_stress"],
        help="Workload names from WORKLOADS or 'all'",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["analytical", "ns3"],
        help="Backends to compare (order matters; first is baseline)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--rerun", action="store_true", help="Re-run even if outputs already exist")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-run timeout in seconds")
    parser.add_argument("--num-req-override", type=int, default=None, help="Override num_req for all workloads")
    policy_choices = get_registered_policy_names()
    parser.add_argument(
        "--kv-eviction-policy",
        type=str,
        choices=policy_choices,
        default="tail",
        help="KV eviction policy to pass to main.py",
    )
    args = parser.parse_args()

    selected_configs = _select_items(args.configs, CONFIGS, "configs")
    selected_workloads = _select_items(args.workloads, WORKLOADS, "workloads")
    selected_backends = args.backends

    print("=" * 88)
    print("Backend Diff Runner")
    print(f"Configs:  {selected_configs}")
    print(f"Workloads:{selected_workloads}")
    print(f"Backends: {selected_backends}")
    print(f"Output:   {OUTPUT_ROOT}")
    print("=" * 88)

    records: List[RunRecord] = []

    total = len(selected_backends) * len(selected_configs) * len(selected_workloads)
    idx = 0
    for backend in selected_backends:
        for cfg_name in selected_configs:
            for wl_name in selected_workloads:
                idx += 1
                wl = WORKLOADS[wl_name]
                num_req = args.num_req_override if args.num_req_override is not None else wl["num_req"]
                print(f"[{idx}/{total}] {backend} | {cfg_name} | {wl_name}")
                rec = _run_one(
                    backend=backend,
                    config_name=cfg_name,
                    workload_name=wl_name,
                    cluster_config=CONFIGS[cfg_name],
                    dataset=wl["dataset"],
                    num_req=num_req,
                    kv_eviction_policy=args.kv_eviction_policy,
                    timeout=args.timeout,
                    dry_run=args.dry_run,
                    rerun=args.rerun,
                )
                records.append(rec)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.DataFrame([r.__dict__ for r in records])
    manifest_path = OUTPUT_ROOT / "run_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    diff_df = _build_diff_rows(manifest_df, selected_backends)
    diff_path = OUTPUT_ROOT / "backend_diff_summary.csv"
    if not diff_df.empty:
        diff_df.to_csv(diff_path, index=False)

    print("=" * 88)
    print("Run summary")
    if not manifest_df.empty:
        print(manifest_df["status"].value_counts().to_string())
    print(f"Manifest: {manifest_path}")
    if diff_df.empty:
        print("No backend pairs were available for summary diff.")
    else:
        print(f"Diff:     {diff_path}")
        preview_cols = [
            "config",
            "workload",
            "baseline_backend",
            "compare_backend",
            "delta_pct_mean_tpot_ms",
            "delta_pct_p99_tpot_ms",
            "delta_pct_tier_transition_mb_total",
        ]
        preview_cols = [c for c in preview_cols if c in diff_df.columns]
        print(diff_df[preview_cols].head(10).to_string(index=False))
    print("=" * 88)


if __name__ == "__main__":
    main()
