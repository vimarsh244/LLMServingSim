#!/usr/bin/env python3
"""
run_harp_ablation.py - Focused policy comparison runner.

Purpose:
- Compare HARP and Tail eviction on ShareGPT-1000 with CPU-DRAM tiering.

Outputs:
    output/tiered_kv/harp_ablation/sharegpt1000_cpu_dram/
        - run_manifest.csv
        - metric_summary.csv
        - delta_vs_baseline.csv
        - one subdir per configuration (result.csv/timeseries.csv/result_tier_stats.json/output.txt)
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT_DIR / "output" / "tiered_kv" / "harp_ablation" / "sharegpt1000_cpu_dram"


@dataclass
class AblationRun:
    key: str
    policy: str
    harp_lambda_stall: float = 0.0
    harp_lambda_quality: float = 0.0
    harp_lambda_fairness: float = 0.0
    harp_grace_candidates: str = "16,32,64"
    harp_ratios: str = "1.0"
    harp_compression_profile: str = "none"


RUNS: List[AblationRun] = [
    # HARP baseline with default lambda values; compression is disabled by the runner defaults.
    AblationRun(
        key="E0_baseline",
        policy="harp",
        harp_lambda_stall=1.0,
        harp_lambda_quality=0.5,
        harp_lambda_fairness=0.1,
        harp_grace_candidates="16,32,64",
    ),
    # HARP with all lambdas=0; shadow/grace/prefetch behavior left intact.
    AblationRun(
        key="E1_harp_all_zero",
        policy="harp",
        harp_lambda_stall=0.0,
        harp_lambda_quality=0.0,
        harp_lambda_fairness=0.0,
        harp_grace_candidates="16,32,64",
    ),
    # HARP no-prefetch approximation in script: force grace candidates to 0.
    AblationRun(
        key="E2_harp_no_prefetch",
        policy="harp",
        harp_lambda_stall=0.0,
        harp_lambda_quality=0.0,
        harp_lambda_fairness=0.0,
        harp_grace_candidates="0",
    ),
    # Tail eviction baseline: no compression, no prefetch.
    AblationRun(
        key="E3_tail",
        policy="tail",
    ),
]


def _build_command(run: AblationRun, args: argparse.Namespace, result_csv: Path, timeseries_csv: Path) -> List[str]:
    cmd = [
        sys.executable,
        "main.py",
        "--cluster-config",
        args.cluster_config,
        "--dataset",
        args.dataset,
        "--num-req",
        str(args.num_req),
        "--output",
        str(result_csv.relative_to(ROOT_DIR)),
        "--timeseries-output",
        str(timeseries_csv.relative_to(ROOT_DIR)),
        "--network-backend",
        args.network_backend,
    ]

    cmd.extend(
        [
            "--kv-eviction-policy",
            run.policy,
        ]
    )

    if run.policy == "harp":
        cmd.extend(
            [
                "--harp-grace-candidates",
                run.harp_grace_candidates,
                "--harp-ratios",
                run.harp_ratios,
                "--harp-lambda-stall",
                str(run.harp_lambda_stall),
                "--harp-lambda-quality",
                str(run.harp_lambda_quality),
                "--harp-lambda-fairness",
                str(run.harp_lambda_fairness),
                "--harp-fairness-epsilon",
                str(args.harp_fairness_epsilon),
                "--harp-compression-profile",
                run.harp_compression_profile,
            ]
        )

        if args.harp_compression_trace:
            cmd.extend(["--harp-compression-trace", args.harp_compression_trace])

    return cmd


def _flatten_itl_ns(series: pd.Series) -> List[float]:
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
    out: Dict[str, float] = {"num_requests": float(len(df))}

    def mean_ms(col: str, key: str) -> None:
        if col in df.columns:
            out[key] = float(df[col].astype(float).mean() * 1e-6)

    def p99_ms(col: str, key: str) -> None:
        if col in df.columns:
            out[key] = float(df[col].astype(float).quantile(0.99) * 1e-6)

    mean_ms("TTFT", "mean_ttft_ms")
    mean_ms("TPOT", "mean_tpot_ms")
    mean_ms("latency", "mean_latency_ms")
    p99_ms("TTFT", "p99_ttft_ms")
    p99_ms("TPOT", "p99_tpot_ms")

    if "ITL" in df.columns:
        itl_ns = _flatten_itl_ns(df["ITL"])
        if itl_ns:
            itl_series = pd.Series(itl_ns, dtype=float)
            out["mean_itl_ms"] = float(itl_series.mean() * 1e-6)
            out["p99_itl_ms"] = float(itl_series.quantile(0.99) * 1e-6)

    if "output" in df.columns and "input" in df.columns:
        generated = (df["output"].astype(float) - df["input"].astype(float)).clip(lower=0).sum()
        out["generated_tokens_total"] = float(generated)

    if "harp_stall_time_ns" in df.columns:
        out["mean_stall_ms_per_request"] = float(df["harp_stall_time_ns"].astype(float).mean() * 1e-6)

    if "harp_shadow_hit_tokens" in df.columns and "harp_decode_tokens" in df.columns:
        decode_total = float(df["harp_decode_tokens"].astype(float).sum())
        if decode_total > 0:
            out["shadow_hit_rate"] = float(df["harp_shadow_hit_tokens"].astype(float).sum() / decode_total)
        else:
            out["shadow_hit_rate"] = 0.0

    return out


def _load_tier_metrics(tier_stats_json: Path) -> Dict[str, float]:
    if not tier_stats_json.exists():
        return {}

    with open(tier_stats_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    keys = [
        "evict_npu_to_cpu_bytes",
        "evict_npu_to_cxl_bytes",
        "load_cpu_to_npu_bytes",
        "load_cxl_to_npu_bytes",
        "harp_prefetch_bytes_total",
        "harp_prefetch_overlap_bytes",
        "harp_shadow_hit_tokens",
        "harp_decode_tokens_total",
        "harp_shadow_ratio_sum",
        "harp_shadow_ratio_events",
    ]
    out = {k: 0.0 for k in keys}

    for inst in raw.values():
        for key in keys:
            out[key] += float(inst.get(key, 0.0))

    total_transition_bytes = (
        out["evict_npu_to_cpu_bytes"]
        + out["evict_npu_to_cxl_bytes"]
        + out["load_cpu_to_npu_bytes"]
        + out["load_cxl_to_npu_bytes"]
    )
    out["tier_transition_bytes_total"] = total_transition_bytes
    out["tier_transition_mb_total"] = total_transition_bytes / (1024.0 * 1024.0)

    if out["harp_prefetch_bytes_total"] > 0:
        out["stall_overlap_ratio"] = out["harp_prefetch_overlap_bytes"] / out["harp_prefetch_bytes_total"]
    else:
        out["stall_overlap_ratio"] = 0.0

    if out["harp_decode_tokens_total"] > 0:
        out["shadow_hit_rate"] = out["harp_shadow_hit_tokens"] / out["harp_decode_tokens_total"]

    if out["harp_shadow_ratio_events"] > 0:
        out["harp_avg_shadow_ratio"] = out["harp_shadow_ratio_sum"] / out["harp_shadow_ratio_events"]
    else:
        out["harp_avg_shadow_ratio"] = 1.0

    # CPU-only sanity: should be true in this study.
    out["cpu_only_sanity_ok"] = float(
        out["evict_npu_to_cxl_bytes"] == 0.0 and out["load_cxl_to_npu_bytes"] == 0.0
    )

    return out


def _safe_pct_delta(base: float, value: float) -> float:
    if base == 0.0:
        return 0.0
    return (value - base) / base * 100.0


def _build_delta_table(summary_df: pd.DataFrame, baseline_key: str = "E0_baseline") -> pd.DataFrame:
    if summary_df.empty or baseline_key not in set(summary_df["experiment"]):
        return pd.DataFrame()

    base = summary_df.loc[summary_df["experiment"] == baseline_key].iloc[0]
    metric_cols = [
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "p99_itl_ms",
        "mean_latency_ms",
        "tier_transition_mb_total",
        "transition_mb_per_generated_token",
        "mean_stall_ms_per_request",
        "stall_overlap_ratio",
        "shadow_hit_rate",
        "harp_avg_shadow_ratio",
    ]

    rows = []
    for row in summary_df.itertuples(index=False):
        d = {
            "experiment": row.experiment,
            "delta_against": baseline_key,
        }
        for col in metric_cols:
            if col not in summary_df.columns:
                continue
            base_val = float(base.get(col, 0.0) or 0.0)
            cur_val = float(getattr(row, col, 0.0) or 0.0)
            d[f"delta_{col}"] = cur_val - base_val
            d[f"delta_pct_{col}"] = _safe_pct_delta(base_val, cur_val)
        rows.append(d)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HARP/Tail policy comparison on CPU-DRAM + ShareGPT-1000")
    parser.add_argument("--cluster-config", type=str, default="cluster_config/tiered_kv_tier_cpu_dram.json")
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Optional output root relative to repo (or absolute path).",
    )
    parser.add_argument("--dataset", type=str, default="dataset/sharegpt_req1000_rate10_llama.jsonl")
    parser.add_argument("--num-req", type=int, default=1000)
    parser.add_argument("--network-backend", type=str, choices=["analytical", "ns3"], default="analytical")
    parser.add_argument("--timeout", type=int, default=7200, help="Per-run timeout in seconds")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--harp-fairness-epsilon", type=float, default=1e-6)
    parser.add_argument("--harp-compression-trace", type=str, default="")
    args = parser.parse_args()

    output_root = OUTPUT_ROOT
    if args.output_root:
        candidate = Path(args.output_root)
        output_root = candidate if candidate.is_absolute() else (ROOT_DIR / candidate)

    output_root.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []

    print("=" * 88)
    print("HARP/Tail Policy Runner")
    print(f"Cluster config: {args.cluster_config}")
    print(f"Dataset:        {args.dataset}")
    print(f"Num req:        {args.num_req}")
    print(f"Output root:    {output_root}")
    print("=" * 88)

    for run in RUNS:
        out_dir = output_root / run.key
        out_dir.mkdir(parents=True, exist_ok=True)
        result_csv = out_dir / "result.csv"
        timeseries_csv = out_dir / "timeseries.csv"
        tier_stats_json = out_dir / "result_tier_stats.json"
        stdout_log = out_dir / "output.txt"

        rec: Dict[str, object] = {
            "experiment": run.key,
            "policy": run.policy,
            "harp_lambda_stall": run.harp_lambda_stall,
            "harp_lambda_quality": run.harp_lambda_quality,
            "harp_lambda_fairness": run.harp_lambda_fairness,
            "harp_grace_candidates": run.harp_grace_candidates,
            "harp_ratios": run.harp_ratios,
            "harp_compression_profile": run.harp_compression_profile,
            "output_dir": str(out_dir),
            "result_csv": str(result_csv),
            "timeseries_csv": str(timeseries_csv),
            "tier_stats_json": str(tier_stats_json),
            "stdout_log": str(stdout_log),
        }

        if result_csv.exists() and not args.rerun and not args.dry_run:
            rec["status"] = "skipped_existing"
            rec["returncode"] = 0
            records.append(rec)
            print(f"SKIP [{run.key}] existing output")
            continue

        cmd = _build_command(run, args, result_csv, timeseries_csv)
        print(
            f"RUN  [{run.key}] policy={run.policy} "
            f"stall={run.harp_lambda_stall} quality={run.harp_lambda_quality} fairness={run.harp_lambda_fairness} "
            f"grace={run.harp_grace_candidates} ratios={run.harp_ratios} profile={run.harp_compression_profile}"
        )
        print("  " + " ".join(cmd))

        if args.dry_run:
            rec["status"] = "dry_run"
            rec["returncode"] = 0
            records.append(rec)
            continue

        try:
            completed = subprocess.run(
                cmd,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout,
            )
            stdout_log.write_text(completed.stdout, encoding="utf-8")
            rec["returncode"] = int(completed.returncode)
            rec["status"] = "ok" if completed.returncode == 0 else "failed"
        except subprocess.TimeoutExpired as exc:
            partial = exc.stdout or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            stdout_log.write_text(partial, encoding="utf-8")
            rec["returncode"] = 124
            rec["status"] = "timeout"

        records.append(rec)

    manifest_df = pd.DataFrame(records)
    manifest_path = output_root / "run_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    rows: List[Dict[str, object]] = []
    valid_status = {"ok", "skipped_existing"}
    for rec in records:
        if rec.get("status") not in valid_status:
            continue

        row: Dict[str, object] = {
            "experiment": rec["experiment"],
            "policy": rec["policy"],
            "harp_lambda_stall": rec["harp_lambda_stall"],
            "harp_lambda_quality": rec["harp_lambda_quality"],
            "harp_lambda_fairness": rec["harp_lambda_fairness"],
            "harp_grace_candidates": rec["harp_grace_candidates"],
            "harp_ratios": rec["harp_ratios"],
            "harp_compression_profile": rec["harp_compression_profile"],
            "status": rec["status"],
        }
        row.update(_load_result_metrics(Path(str(rec["result_csv"]))))
        row.update(_load_tier_metrics(Path(str(rec["tier_stats_json"]))))

        generated = float(row.get("generated_tokens_total", 0.0) or 0.0)
        moved_mb = float(row.get("tier_transition_mb_total", 0.0) or 0.0)
        if generated > 0:
            row["transition_mb_per_generated_token"] = moved_mb / generated
        else:
            row["transition_mb_per_generated_token"] = 0.0

        rows.append(row)

    metric_df = pd.DataFrame(rows)
    metric_path = output_root / "metric_summary.csv"
    if not metric_df.empty:
        metric_df.to_csv(metric_path, index=False)

    delta_df = _build_delta_table(metric_df, baseline_key="E0_baseline")
    delta_path = output_root / "delta_vs_baseline.csv"
    if not delta_df.empty:
        delta_df.to_csv(delta_path, index=False)

    print("=" * 88)
    print("Run summary")
    if not manifest_df.empty and "status" in manifest_df.columns:
        print(manifest_df["status"].value_counts().to_string())
    print(f"Manifest: {manifest_path}")
    if metric_df.empty:
        print("Metric summary not generated (no completed runs).")
    else:
        print(f"Metrics:  {metric_path}")
    if delta_df.empty:
        print("Delta table not generated (baseline or completed runs missing).")
    else:
        print(f"Deltas:   {delta_path}")
        preview_cols = [
            "experiment",
            "delta_pct_mean_ttft_ms",
            "delta_pct_mean_tpot_ms",
            "delta_pct_mean_itl_ms",
            "delta_pct_tier_transition_mb_total",
            "delta_pct_mean_stall_ms_per_request",
        ]
        preview_cols = [c for c in preview_cols if c in delta_df.columns]
        if preview_cols:
            print(delta_df[preview_cols].to_string(index=False))
    print("=" * 88)


if __name__ == "__main__":
    main()
