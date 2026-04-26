#!/usr/bin/env python3
"""
run_baseline.py — Automated experiment sweep for tiered KV cache research.

Runs the LLMServingSim with a matrix of:
  - Tier configurations (NPU-only, NPU+CPU, NPU+CXL variants)
  - Workloads (ShareGPT, fixed-length, prefix-stress)
  - Prefix caching modes (off, NPU, CPU, CXL)
  - Block sizes

Results are saved to output/tiered_kv/{phase}/{config}/{workload}/ with:
  - result.csv          (per-request latency)
  - timeseries.csv      (periodic memory/cache metrics)
  - output.txt          (stdout capture)
  - tier_stats.json     (aggregate tier transition stats)

Usage:
    python benchmarks/run_baseline.py --phase A
    python benchmarks/run_baseline.py --phase all
    python benchmarks/run_baseline.py --phase A --dry-run
"""

import os
import sys
import subprocess
import argparse
import itertools
import json
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference_serving.eviction_policies import get_registered_policy_names

OUTPUT_DIR = ROOT_DIR / "output" / "tiered_kv"

# ─────────────────────── Experiment definitions ───────────────────────

WORKLOADS = {
    "sharegpt_100":   {"dataset": "dataset/sharegpt_req100_rate10_llama.jsonl", "num_req": 100},
    "sharegpt_300":   {"dataset": "dataset/sharegpt_req300_rate10_llama.jsonl", "num_req": 300},
    "fixed_256":      {"dataset": "dataset/fixed_in128_out512_req256_rate10.jsonl", "num_req": 256},
    "prefix_stress":  {"dataset": "dataset/prefix_pool_stress.jsonl", "num_req": 100},
    "pulse_prefix":   {"dataset": "dataset/sharegpt_pulse_req50_n6_delay15_pc.jsonl", "num_req": 300},
}

TIER_CONFIGS = {
    "npu_only":     {"cluster": "cluster_config/tiered_kv_npu_only.json", "prefix_storage": "None"},
    "npu_cpu":      {"cluster": "cluster_config/tiered_kv_npu_cpu.json",  "prefix_storage": "None"},
    "npu_cxl":      {"cluster": "cluster_config/tiered_kv_npu_cxl.json", "prefix_storage": "None"},
    "npu_cxl_fast": {"cluster": "cluster_config/tiered_kv_npu_cxl_fast.json", "prefix_storage": "None"},
    "npu_cxl_slow": {"cluster": "cluster_config/tiered_kv_npu_cxl_slow.json", "prefix_storage": "None"},
}

PREFIX_CONFIGS = {
    "no_prefix":     {"enable_prefix_caching": False, "prefix_storage": "None"},
    "prefix_npu":    {"enable_prefix_caching": True,  "prefix_storage": "None"},
    "prefix_cpu":    {"enable_prefix_caching": True,  "prefix_storage": "CPU"},
    "prefix_cxl":    {"enable_prefix_caching": True,  "prefix_storage": "CXL"},
}

BLOCK_SIZES = [4, 8, 16, 32]

# CXL sensitivity sweep: (latency_ns, bandwidth_GBs)
CXL_SWEEP = [
    (150, 30), (150, 60), (150, 120),
    (250, 30), (250, 60), (250, 120),
    (500, 30), (500, 60), (500, 120),
]

# ─────────────────────── Phase definitions ────────────────────────────

def get_phase_a_experiments():
    """KV eviction pressure: npu_cpu and npu_cxl_cpu configs (18GB) × 5 workloads, prefix off."""
    experiments = []
    workloads = ["sharegpt_100", "sharegpt_300", "fixed_256", "prefix_stress", "pulse_prefix"]
    configs = [
        ("npu_cpu",     "cluster_config/tiered_kv_npu_cpu.json"),
        ("npu_cxl_cpu", "cluster_config/tiered_kv_npu_cxl_cpu.json"),
    ]
    for cfg_name, cfg_path in configs:
        for wl in workloads:
            experiments.append({
                "name": f"phaseA/{cfg_name}/{wl}",
                "cluster": cfg_path,
                "workload": WORKLOADS[wl],
                "prefix_caching": False,
                "prefix_storage": "None",
                "block_size": 16,
            })
    return experiments


def get_phase_b_experiments():
    """Prefix cache tier comparison: 4 prefix modes × 4 workloads."""
    experiments = []
    prefix_modes = ["no_prefix", "prefix_npu", "prefix_cpu", "prefix_cxl"]
    workloads = ["sharegpt_100", "sharegpt_300", "prefix_stress", "pulse_prefix"]
    for pm, wl in itertools.product(prefix_modes, workloads):
        pc = PREFIX_CONFIGS[pm]
        # Use 24GB configs for prefix caching (need headroom for prefix cache)
        if pc["prefix_storage"] == "CXL":
            cluster = "cluster_config/tiered_kv_prefix_cxl.json"
        else:
            cluster = "cluster_config/tiered_kv_prefix_cpu.json"
        experiments.append({
            "name": f"phaseB/{pm}/{wl}",
            "cluster": cluster,
            "workload": WORKLOADS[wl],
            "prefix_caching": pc["enable_prefix_caching"],
            "prefix_storage": pc["prefix_storage"],
            "block_size": 16,
        })
    return experiments


def get_phase_c_experiments():
    """CXL sensitivity sweep: 9 param combos × 2 workloads."""
    import json
    experiments = []
    workloads = ["sharegpt_300", "pulse_prefix"]
    for (lat, bw), wl in itertools.product(CXL_SWEEP, workloads):
        cfg_name = f"cxl_lat{lat}_bw{bw}"
        # Generate config on-the-fly
        cfg_path = ROOT_DIR / "cluster_config" / f"tiered_kv_{cfg_name}.json"
        if not cfg_path.exists():
            config = {
                "num_nodes": 1, "link_bw": 112, "link_latency": 0,
                "nodes": [{
                    "num_instances": 1,
                    "cpu_mem": {"mem_size": 128, "mem_bw": 256, "mem_latency": 0},
                    "instances": [{
                        "model_name": "meta-llama/Llama-3.1-8B",
                        "hardware": "A6000",
                        "npu_mem": {"mem_size": 24, "mem_bw": 768, "mem_latency": 0},
                        "npu_num": 1, "npu_group": 1, "pd_type": None,
                        "placement": {"default": {"weights": "npu", "kv_loc": "npu", "kv_evict_loc": "cpu"}}
                    }]
                }],
                "cxl_mem": {"mem_size": 256, "mem_latency": lat, "mem_bw": bw, "num_devices": 1}
            }
            with open(cfg_path, 'w') as f:
                json.dump(config, f, indent=4)

        experiments.append({
            "name": f"phaseC/{cfg_name}/{wl}",
            "cluster": f"cluster_config/tiered_kv_{cfg_name}.json",
            "workload": WORKLOADS[wl],
            "prefix_caching": True,
            "prefix_storage": "CXL",
            "block_size": 16,
        })
    return experiments


def get_phase_d_experiments():
    """Block size sensitivity: 4 sizes × 2 workloads."""
    experiments = []
    workloads = ["sharegpt_300", "fixed_256"]
    for bs, wl in itertools.product(BLOCK_SIZES, workloads):
        experiments.append({
            "name": f"phaseD/block_{bs}/{wl}",
            "cluster": "cluster_config/tiered_kv_npu_cpu.json",
            "workload": WORKLOADS[wl],
            "prefix_caching": False,
            "prefix_storage": "None",
            "block_size": bs,
        })
    return experiments


def get_phase_e_experiments():
    """Llama-70B baselines on 4 H100 GPUs with CPU KV offload (prefix caching disabled)."""
    experiments = []
    workloads = ["sharegpt_100", "sharegpt_300", "fixed_256", "prefix_stress", "pulse_prefix"]
    configs = [
        ("70b_npu_cpu", "cluster_config/tiered_kv_70b_npu_cpu.json"),
    ]
    for cfg_name, cfg_path in configs:
        for wl in workloads:
            experiments.append({
                "name": f"phaseE/{cfg_name}/{wl}",
                "cluster": cfg_path,
                "workload": WORKLOADS[wl],
                "prefix_caching": False,
                "prefix_storage": "None",
                "block_size": 16,
            })
    return experiments


PHASES = {
    "A": get_phase_a_experiments,
    "B": get_phase_b_experiments,
    "C": get_phase_c_experiments,
    "D": get_phase_d_experiments,
    "E": get_phase_e_experiments,
}


def warn_duplicate_cluster_configs(experiments):
    """Warn when different cluster config files have identical JSON content."""
    unique_paths = sorted(set(exp["cluster"] for exp in experiments))
    content_map = {}

    for rel_path in unique_paths:
        abs_path = ROOT_DIR / rel_path
        if not abs_path.exists():
            continue
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                normalized = json.dumps(json.load(f), sort_keys=True, separators=(",", ":"))
        except Exception:
            continue
        content_map.setdefault(normalized, []).append(rel_path)

    duplicates = [paths for paths in content_map.values() if len(paths) > 1]
    if not duplicates:
        return

    print("\n[WARNING] Detected semantically duplicate cluster config files:")
    for dup_group in duplicates:
        print(f"  - {dup_group}")
    print("  These files produce the same cluster behavior unless modified.")

# ─────────────────────── Runner ───────────────────────────────────────

def run_experiment(exp, dry_run=False, kv_eviction_policy="tail"):
    """Run a single experiment."""
    out_dir = OUTPUT_DIR / exp["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    result_csv = out_dir / "result.csv"
    timeseries_csv = out_dir / "timeseries.csv"
    stdout_log = out_dir / "output.txt"

    # Build command
    cmd = [
        sys.executable, str(ROOT_DIR / "main.py"),
        "--cluster-config", exp["cluster"],
        "--dataset", exp["workload"]["dataset"],
        "--num-req", str(exp["workload"]["num_req"]),
        "--output", str(result_csv.relative_to(ROOT_DIR)),
        "--timeseries-output", str(timeseries_csv.relative_to(ROOT_DIR)),
        "--block-size", str(exp["block_size"]),
        "--log-interval", "0.5",
        "--kv-eviction-policy", kv_eviction_policy,
    ]

    if exp["prefix_caching"]:
        cmd.append("--enable-prefix-caching")
    if exp["prefix_storage"] != "None":
        cmd.extend(["--prefix-storage", exp["prefix_storage"]])

    print(f"\n{'='*80}")
    print(f"  Experiment: {exp['name']}")
    print(f"  Command:    {' '.join(cmd)}")
    print(f"  Output:     {out_dir}")
    print(f"{'='*80}")

    if dry_run:
        print("  [DRY RUN] Skipping execution")
        return True

    try:
        with open(stdout_log, 'w') as log_f:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT_DIR),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=3600,  # 1 hour timeout
            )
        
        if proc.returncode != 0:
            print(f"  [FAILED] Exit code: {proc.returncode}")
            return False
        else:
            print(f"  [OK] Completed successfully")
            return True
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Experiment exceeded 1 hour")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tiered KV cache baseline experiments")
    parser.add_argument("--phase", type=str, choices=list(PHASES.keys()) + ["all"], default="A",
                       help="Which experimental phase to run (A/B/C/D/E/all)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print experiments without running them")
    parser.add_argument("--filter", type=str, default=None,
                       help="Only run experiments whose name contains this substring")
    policy_choices = get_registered_policy_names()
    parser.add_argument("--kv-eviction-policy", type=str,
                        choices=policy_choices,
                        default="tail",
                        help="KV eviction policy to pass to main.py")
    args = parser.parse_args()

    phases = list(PHASES.keys()) if args.phase == "all" else [args.phase]
    
    experiments = []
    for phase in phases:
        experiments.extend(PHASES[phase]())

    if args.filter:
        experiments = [e for e in experiments if args.filter in e["name"]]

    warn_duplicate_cluster_configs(experiments)

    print(f"\n{'#'*80}")
    print(f"  Tiered KV Cache Baseline Experiments")
    print(f"  Phases: {phases}")
    print(f"  Total experiments: {len(experiments)}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")

    results = {"success": 0, "failed": 0, "skipped": 0}
    for exp in experiments:
        out_dir = OUTPUT_DIR / exp["name"]
        result_csv = out_dir / "result.csv"
        
        # Skip if already completed
        if result_csv.exists() and not args.dry_run:
            print(f"\n  [SKIP] {exp['name']} — already has results")
            results["skipped"] += 1
            continue

        ok = run_experiment(exp, dry_run=args.dry_run, kv_eviction_policy=args.kv_eviction_policy)
        if args.dry_run:
            results["skipped"] += 1
        elif ok:
            results["success"] += 1
        else:
            results["failed"] += 1

    print(f"\n{'#'*80}")
    print(f"  Summary: {results['success']} succeeded, {results['failed']} failed, {results['skipped']} skipped")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
