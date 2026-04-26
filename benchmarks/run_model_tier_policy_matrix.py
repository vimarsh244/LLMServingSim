#!/usr/bin/env python3
"""
run_model_tier_policy_matrix.py - Sweep models x tiers x eviction policies.

This runner evaluates one dense model and one MoE model across tiered KV profiles
and eviction policies, logging all raw outputs and aggregate summaries.

Default models:
  - llama8b  -> meta-llama/Llama-3.1-8B
  - phi_moe  -> microsoft/Phi-mini-MoE-instruct

Additional models are auto-discovered from model_config/*.json and can be
selected using canonical keys (for example llama70b, mixtral_8x7b) or aliases
(for example mistral-8x7b, meta-llama/Llama-3.1-70B).

Outputs:
  output/tiered_kv/model_tier_policy_matrix/
    - run_manifest.csv
    - metric_summary.csv
    - policy_delta_summary.csv
    - _generated_cluster_configs/*.json
        - {accelerator}/{model}/{tier}/{policy}/{workload}/
        - result.csv
        - timeseries.csv
        - result_tier_stats.json
        - output.txt

Examples:
  python benchmarks/run_model_tier_policy_matrix.py --dry-run
  python benchmarks/run_model_tier_policy_matrix.py --num-req-override 80
  python benchmarks/run_model_tier_policy_matrix.py \
      --models llama8b phi_moe --tiers cpu_dram cxl pcie_nvme \
      --policies tail fifo lru largest_kv evicpress --workloads sharegpt_100 fixed_256
    python benchmarks/run_model_tier_policy_matrix.py \
            --models llama8b --tiers cpu_dram cxl pcie_nvme --policies all \
            --workloads sharegpt_750 sharegpt_1000 sharegpt_1500
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference_serving.eviction_policies import get_registered_policy_names

OUTPUT_ROOT = ROOT_DIR / "output" / "tiered_kv" / "model_tier_policy_matrix"
_out_root_override = os.environ.get("MODEL_TIER_OUT_ROOT", "").strip()
if _out_root_override:
    OUTPUT_ROOT = (ROOT_DIR / _out_root_override).resolve() if not Path(_out_root_override).is_absolute() else Path(_out_root_override)
GENERATED_CONFIG_ROOT = OUTPUT_ROOT / "_generated_cluster_configs"
MODEL_CONFIG_ROOT = ROOT_DIR / "model_config"

LLAMA_SHAREGPT_DATASETS = {
    "sharegpt_100": "dataset/sharegpt_req100_rate10_llama.jsonl",
    "sharegpt_300": "dataset/sharegpt_req300_rate10_llama.jsonl",
    "sharegpt_750": "dataset/sharegpt_req750_rate10_llama.jsonl",
    "sharegpt_1000": "dataset/sharegpt_req1000_rate10_llama.jsonl",
    "sharegpt_1500": "dataset/sharegpt_req1500_rate10_llama.jsonl",
}

PHI_SHAREGPT_DATASETS = {
    "sharegpt_100": "dataset/sharegpt_req100_rate10_phi.jsonl",
    "sharegpt_300": "dataset/sharegpt_req300_rate10_phi.jsonl",
    "sharegpt_750": "dataset/sharegpt_req750_rate10_phi.jsonl",
    "sharegpt_1000": "dataset/sharegpt_req1000_rate10_phi.jsonl",
    "sharegpt_1500": "dataset/sharegpt_req1500_rate10_phi.jsonl",
}

MIXTRAL_SHAREGPT_DATASETS = {
    "sharegpt_100": "dataset/sharegpt_req100_rate10_mixtral.jsonl",
    "sharegpt_300": "dataset/sharegpt_req300_rate10_mixtral.jsonl",
    "sharegpt_750": "dataset/sharegpt_req750_rate10_mixtral.jsonl",
    "sharegpt_1000": "dataset/sharegpt_req1000_rate10_mixtral.jsonl",
    "sharegpt_1500": "dataset/sharegpt_req1500_rate10_mixtral.jsonl",
}

CANONICAL_KEY_OVERRIDES = {
    "meta-llama/Llama-3.1-8B": "llama8b",
    "meta-llama/Llama-3.1-70B": "llama70b",
    "microsoft/Phi-mini-MoE-instruct": "phi_moe",
    "mistralai/Mixtral-8x7B-v0.1": "mixtral_8x7b",
}

MODEL_ALIAS_OVERRIDES = {
    "meta-llama/Llama-3.1-8B": ["llama-8b", "llama_8b"],
    "meta-llama/Llama-3.1-70B": ["llama-70b", "llama_70b"],
    "microsoft/Phi-mini-MoE-instruct": ["phi-moe", "phi-mini-moe", "phi-mini-moe-instruct"],
    "mistralai/Mixtral-8x7B-v0.1": [
        "mixtral-8x7b",
        "mixtral8x7b",
        "mistral-8x7b",
        "mistral_8x7b",
        "mistral8x7b",
    ],
}


def _slugify_model_key(raw: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", raw.lower())).strip("_")


def _datasets_for_model(model_name: str) -> Dict[str, str]:
    model_name_l = model_name.lower()
    if "phi-mini-moe" in model_name_l:
        return dict(PHI_SHAREGPT_DATASETS)
    if "mixtral" in model_name_l or "mistral" in model_name_l:
        return dict(MIXTRAL_SHAREGPT_DATASETS)
    return dict(LLAMA_SHAREGPT_DATASETS)


def _discover_model_registry() -> Tuple[Dict[str, Dict[str, object]], Dict[str, str]]:
    model_specs: Dict[str, Dict[str, object]] = {}
    alias_to_key: Dict[str, str] = {}

    for cfg_path in sorted(MODEL_CONFIG_ROOT.rglob("*.json")):
        model_name = cfg_path.relative_to(MODEL_CONFIG_ROOT).as_posix().removesuffix(".json")
        model_leaf = Path(model_name).name
        canonical = CANONICAL_KEY_OVERRIDES.get(model_name, _slugify_model_key(model_leaf))

        # Handle rare collisions in generated canonical keys by expanding with vendor path.
        if canonical in model_specs and model_specs[canonical]["model_name"] != model_name:
            canonical = _slugify_model_key(model_name)

        model_specs[canonical] = {
            "model_name": model_name,
            "datasets": _datasets_for_model(model_name),
        }

        aliases: Set[str] = {
            canonical,
            model_name,
            model_name.lower(),
            model_leaf,
            model_leaf.lower(),
            _slugify_model_key(model_leaf),
            _slugify_model_key(model_name),
        }
        aliases.update(MODEL_ALIAS_OVERRIDES.get(model_name, []))

        for alias in aliases:
            alias_to_key[alias.lower()] = canonical

    if not model_specs:
        raise FileNotFoundError(f"No model configs found under {MODEL_CONFIG_ROOT}")

    return model_specs, alias_to_key


MODEL_SPECS, MODEL_ALIASES = _discover_model_registry()

TIER_CONFIGS = {
    "cpu_dram": "cluster_config/tiered_kv_tier_cpu_dram.json",
    "cxl": "cluster_config/tiered_kv_tier_cxl.json",
    "pcie_nvme": "cluster_config/tiered_kv_tier_pcie_nvme.json",
    "ssd": "cluster_config/tiered_kv_tier_ssd.json",
    "ethernet": "cluster_config/tiered_kv_tier_ethernet.json",
}

WORKLOADS = {
    "sharegpt_100": {
        "default_dataset": "dataset/sharegpt_req100_rate10_llama.jsonl",
        "num_req": 100,
    },
    "sharegpt_300": {
        "default_dataset": "dataset/sharegpt_req300_rate10_llama.jsonl",
        "num_req": 300,
    },
    "sharegpt_750": {
        "default_dataset": "dataset/sharegpt_req750_rate10_llama.jsonl",
        "num_req": 750,
    },
    "sharegpt_1000": {
        "default_dataset": "dataset/sharegpt_req1000_rate10_llama.jsonl",
        "num_req": 1000,
    },
    "sharegpt_1500": {
        "default_dataset": "dataset/sharegpt_req1500_rate10_llama.jsonl",
        "num_req": 1500,
    },
    "fixed_256": {
        "default_dataset": "dataset/fixed_in128_out512_req256_rate10.jsonl",
        "num_req": 256,
    },
    "prefix_stress": {
        "default_dataset": "dataset/prefix_pool_stress.jsonl",
        "num_req": 100,
    },
}

POLICIES = get_registered_policy_names()


@dataclass
class RunRecord:
    accelerator: str
    model: str
    tier: str
    policy: str
    workload: str
    cluster_config: str
    dataset: str
    returncode: int
    status: str
    output_dir: str
    result_csv: str
    timeseries_csv: str
    tier_stats_json: str
    stdout_log: str
    evicpress_alpha: float = 1.0
    evicpress_ratios: str = "1.0,0.75,0.5,0.25"
    harp_grace_candidates: str = ""
    harp_ratios: str = ""
    harp_lambda_stall: float = 1.0
    harp_lambda_quality: float = 0.5
    harp_lambda_fairness: float = 0.1
    harp_fairness_epsilon: float = 1e-6
    harp_compression_profile: str = "balanced"
    harp_compression_trace: str = ""


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
    allowed = set(POLICIES) | {"dynmax"}
    unknown = [p for p in selected if p not in allowed]
    if unknown:
        raise ValueError(f"Unknown policies: {unknown}")
    return selected


def _select_models(selected: List[str]) -> List[str]:
    if not selected or any(item.lower() == "all" for item in selected):
        return list(MODEL_SPECS.keys())

    resolved: List[str] = []
    unknown: List[str] = []
    for item in selected:
        key = MODEL_ALIASES.get(item.lower())
        if key is None:
            unknown.append(item)
            continue
        if key not in resolved:
            resolved.append(key)

    if unknown:
        available = ", ".join(sorted(MODEL_SPECS.keys()))
        raise ValueError(f"Unknown models: {unknown}. Available model keys: {available}")

    return resolved


def _resolve_dataset(model_key: str, workload_key: str) -> str:
    model_spec = MODEL_SPECS[model_key]
    wl_spec = WORKLOADS[workload_key]
    return model_spec.get("datasets", {}).get(workload_key, wl_spec["default_dataset"])


def _sanitize_name(raw: str) -> str:
    """Build a filesystem-safe token from a user-provided value."""
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-. ")
    cleaned = "".join(ch if ch in keep else "_" for ch in raw).strip().replace(" ", "_")
    return cleaned or "default"


def _generate_model_tier_config(model_key: str, tier_key: str, accelerator: str) -> Path:
    """Create a model-specialized tier config by swapping model_name and hardware in template."""
    GENERATED_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    template_path = ROOT_DIR / TIER_CONFIGS[tier_key]
    with open(template_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_name = MODEL_SPECS[model_key]["model_name"]
    for node in cfg.get("nodes", []):
        for inst in node.get("instances", []):
            inst["model_name"] = model_name
            inst["hardware"] = accelerator

    accelerator_tag = _sanitize_name(accelerator)
    out_path = GENERATED_CONFIG_ROOT / f"{model_key}__{tier_key}__{accelerator_tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)

    return out_path


def _build_command(
    cluster_config_path: Path,
    dataset: str,
    num_req: int,
    policy: str,
    network_backend: str,
    result_csv: Path,
    timeseries_csv: Path,
    evicpress_alpha: float,
    evicpress_ratios: str,
    harp_grace_candidates: str,
    harp_ratios: str,
    harp_lambda_stall: float,
    harp_lambda_quality: float,
    harp_lambda_fairness: float,
    harp_fairness_epsilon: float,
    harp_compression_profile: str,
    harp_compression_trace: str,
) -> List[str]:
    cmd = [
        sys.executable,
        "main.py",
        "--cluster-config",
        str(cluster_config_path.relative_to(ROOT_DIR)),
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

    if policy == "evicpress":
        cmd.extend(
            [
                "--evicpress-alpha",
                str(evicpress_alpha),
                "--evicpress-ratios",
                evicpress_ratios,
            ]
        )
    elif policy == "harp":
        cmd.extend(
            [
                "--harp-grace-candidates",
                harp_grace_candidates,
                "--harp-ratios",
                harp_ratios,
                "--harp-lambda-stall",
                str(harp_lambda_stall),
                "--harp-lambda-quality",
                str(harp_lambda_quality),
                "--harp-lambda-fairness",
                str(harp_lambda_fairness),
                "--harp-fairness-epsilon",
                str(harp_fairness_epsilon),
                "--harp-compression-profile",
                harp_compression_profile,
            ]
        )
        if harp_compression_trace:
            cmd.extend(["--harp-compression-trace", harp_compression_trace])
    elif policy == "dynmax":
        cmd[cmd.index("--kv-eviction-policy") + 1] = "harp"
        cmd.extend(
            [
                "--harp-grace-candidates",
                "0",
                "--harp-ratios",
                "1.0",
                "--harp-lambda-stall",
                "0.0",
                "--harp-lambda-quality",
                "0.0",
                "--harp-lambda-fairness",
                "0.0",
                "--harp-fairness-epsilon",
                str(harp_fairness_epsilon),
                "--harp-compression-profile",
                "none",
            ]
        )

    return cmd


def _load_metrics(result_csv: Path) -> Dict[str, float]:
    if not result_csv.exists():
        return {}

    df = pd.read_csv(result_csv)
    df.columns = df.columns.str.strip()

    out: Dict[str, float] = {}

    def _mean_ms(col: str, key: str):
        if col in df.columns and len(df) > 0:
            out[key] = float(df[col].mean() * 1e-6)

    def _p99_ms(col: str, key: str):
        if col in df.columns and len(df) > 0:
            out[key] = float(df[col].quantile(0.99) * 1e-6)

    _mean_ms("TTFT", "mean_ttft_ms")
    _mean_ms("TPOT", "mean_tpot_ms")
    _mean_ms("latency", "mean_latency_ms")
    _p99_ms("TTFT", "p99_ttft_ms")
    _p99_ms("TPOT", "p99_tpot_ms")
    out["num_requests"] = float(len(df))

    if "output" in df.columns and "input" in df.columns and len(df) > 0:
        gen_tokens = (df["output"].astype(float) - df["input"].astype(float)).clip(lower=0).sum()
        out["generated_tokens_total"] = float(gen_tokens)

    if "harp_stall_time_ns" in df.columns and len(df) > 0:
        out["mean_stall_ms_per_request"] = float(df["harp_stall_time_ns"].astype(float).mean() * 1e-6)

    if "harp_shadow_hit_tokens" in df.columns and "harp_decode_tokens" in df.columns and len(df) > 0:
        decode_total = float(df["harp_decode_tokens"].astype(float).sum())
        if decode_total > 0:
            out["shadow_hit_rate"] = float(df["harp_shadow_hit_tokens"].astype(float).sum() / decode_total)
        else:
            out["shadow_hit_rate"] = 0.0

    return out


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
        "evicpress_compression_events",
        "evicpress_compressed_bytes_saved",
        "evicpress_ratio_sum",
        "harp_prefetch_bytes_total",
        "harp_prefetch_bytes_progress",
        "harp_prefetch_overlap_bytes",
        "harp_stall_time_ns",
        "harp_stall_events",
        "harp_shadow_hit_tokens",
        "harp_decode_tokens_total",
        "harp_shadow_ratio_sum",
        "harp_shadow_ratio_events",
    ]
    out = {k: 0.0 for k in keys}

    for inst_stats in raw.values():
        for k in keys:
            out[k] += float(inst_stats.get(k, 0.0))

    out["tier_transition_bytes_total"] = (
        out["evict_npu_to_cpu_bytes"]
        + out["evict_npu_to_cxl_bytes"]
        + out["load_cpu_to_npu_bytes"]
        + out["load_cxl_to_npu_bytes"]
    )
    out["tier_transition_mb_total"] = out["tier_transition_bytes_total"] / (1024.0 * 1024.0)

    if out["evicpress_compression_events"] > 0:
        out["evicpress_avg_ratio"] = out["evicpress_ratio_sum"] / out["evicpress_compression_events"]
    else:
        out["evicpress_avg_ratio"] = 1.0

    if out["harp_prefetch_bytes_total"] > 0:
        out["stall_overlap_ratio"] = out["harp_prefetch_overlap_bytes"] / out["harp_prefetch_bytes_total"]
    else:
        out["stall_overlap_ratio"] = 0.0

    if out["harp_decode_tokens_total"] > 0:
        out["shadow_hit_rate"] = out["harp_shadow_hit_tokens"] / out["harp_decode_tokens_total"]
    elif "shadow_hit_rate" not in out:
        out["shadow_hit_rate"] = 0.0

    if out["harp_shadow_ratio_events"] > 0:
        out["harp_avg_shadow_ratio"] = out["harp_shadow_ratio_sum"] / out["harp_shadow_ratio_events"]
    else:
        out["harp_avg_shadow_ratio"] = 1.0

    return out


def _run_one(
    accelerator: str,
    model: str,
    tier: str,
    policy: str,
    workload: str,
    cluster_config_path: Path,
    dataset: str,
    num_req: int,
    network_backend: str,
    timeout: int,
    dry_run: bool,
    rerun: bool,
    evicpress_alpha: float,
    evicpress_ratios: str,
    harp_grace_candidates: str,
    harp_ratios: str,
    harp_lambda_stall: float,
    harp_lambda_quality: float,
    harp_lambda_fairness: float,
    harp_fairness_epsilon: float,
    harp_compression_profile: str,
    harp_compression_trace: str,
) -> RunRecord:
    out_dir = OUTPUT_ROOT / _sanitize_name(accelerator) / model / tier / policy / workload
    out_dir.mkdir(parents=True, exist_ok=True)

    result_csv = out_dir / "result.csv"
    timeseries_csv = out_dir / "timeseries.csv"
    tier_stats_json = out_dir / "result_tier_stats.json"
    stdout_log = out_dir / "output.txt"

    if result_csv.exists() and not rerun and not dry_run:
        return RunRecord(
            accelerator=accelerator,
            model=model,
            tier=tier,
            policy=policy,
            workload=workload,
            cluster_config=str(cluster_config_path.relative_to(ROOT_DIR)),
            dataset=dataset,
            returncode=0,
            status="skipped_existing",
            output_dir=str(out_dir),
            result_csv=str(result_csv),
            timeseries_csv=str(timeseries_csv),
            tier_stats_json=str(tier_stats_json),
            stdout_log=str(stdout_log),
            evicpress_alpha=evicpress_alpha,
            evicpress_ratios=evicpress_ratios,
            harp_grace_candidates=harp_grace_candidates,
            harp_ratios=harp_ratios,
            harp_lambda_stall=harp_lambda_stall,
            harp_lambda_quality=harp_lambda_quality,
            harp_lambda_fairness=harp_lambda_fairness,
            harp_fairness_epsilon=harp_fairness_epsilon,
            harp_compression_profile=harp_compression_profile,
            harp_compression_trace=harp_compression_trace,
        )

    cmd = _build_command(
        cluster_config_path=cluster_config_path,
        dataset=dataset,
        num_req=num_req,
        policy=policy,
        network_backend=network_backend,
        result_csv=result_csv,
        timeseries_csv=timeseries_csv,
        evicpress_alpha=evicpress_alpha,
        evicpress_ratios=evicpress_ratios,
        harp_grace_candidates=harp_grace_candidates,
        harp_ratios=harp_ratios,
        harp_lambda_stall=harp_lambda_stall,
        harp_lambda_quality=harp_lambda_quality,
        harp_lambda_fairness=harp_lambda_fairness,
        harp_fairness_epsilon=harp_fairness_epsilon,
        harp_compression_profile=harp_compression_profile,
        harp_compression_trace=harp_compression_trace,
    )

    print(f"RUN [{model}][{tier}][{policy}] {workload}")
    print("  " + " ".join(cmd))

    if dry_run:
        return RunRecord(
            accelerator=accelerator,
            model=model,
            tier=tier,
            policy=policy,
            workload=workload,
            cluster_config=str(cluster_config_path.relative_to(ROOT_DIR)),
            dataset=dataset,
            returncode=0,
            status="dry_run",
            output_dir=str(out_dir),
            result_csv=str(result_csv),
            timeseries_csv=str(timeseries_csv),
            tier_stats_json=str(tier_stats_json),
            stdout_log=str(stdout_log),
            evicpress_alpha=evicpress_alpha,
            evicpress_ratios=evicpress_ratios,
            harp_grace_candidates=harp_grace_candidates,
            harp_ratios=harp_ratios,
            harp_lambda_stall=harp_lambda_stall,
            harp_lambda_quality=harp_lambda_quality,
            harp_lambda_fairness=harp_lambda_fairness,
            harp_fairness_epsilon=harp_fairness_epsilon,
            harp_compression_profile=harp_compression_profile,
            harp_compression_trace=harp_compression_trace,
        )

    try:
        with open(stdout_log, "w", encoding="utf-8") as log_f:
            completed = subprocess.run(
                cmd,
                cwd=str(ROOT_DIR),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        status = "ok" if completed.returncode == 0 else "failed"
        return RunRecord(
            accelerator=accelerator,
            model=model,
            tier=tier,
            policy=policy,
            workload=workload,
            cluster_config=str(cluster_config_path.relative_to(ROOT_DIR)),
            dataset=dataset,
            returncode=completed.returncode,
            status=status,
            output_dir=str(out_dir),
            result_csv=str(result_csv),
            timeseries_csv=str(timeseries_csv),
            tier_stats_json=str(tier_stats_json),
            stdout_log=str(stdout_log),
            evicpress_alpha=evicpress_alpha,
            evicpress_ratios=evicpress_ratios,
            harp_grace_candidates=harp_grace_candidates,
            harp_ratios=harp_ratios,
            harp_lambda_stall=harp_lambda_stall,
            harp_lambda_quality=harp_lambda_quality,
            harp_lambda_fairness=harp_lambda_fairness,
            harp_fairness_epsilon=harp_fairness_epsilon,
            harp_compression_profile=harp_compression_profile,
            harp_compression_trace=harp_compression_trace,
        )
    except subprocess.TimeoutExpired as exc:
        with open(stdout_log, "a", encoding="utf-8") as log_f:
            log_f.write(f"\n[runner-timeout] Command exceeded timeout={timeout}s\n")
        return RunRecord(
            accelerator=accelerator,
            model=model,
            tier=tier,
            policy=policy,
            workload=workload,
            cluster_config=str(cluster_config_path.relative_to(ROOT_DIR)),
            dataset=dataset,
            returncode=124,
            status="timeout",
            output_dir=str(out_dir),
            result_csv=str(result_csv),
            timeseries_csv=str(timeseries_csv),
            tier_stats_json=str(tier_stats_json),
            stdout_log=str(stdout_log),
            evicpress_alpha=evicpress_alpha,
            evicpress_ratios=evicpress_ratios,
            harp_grace_candidates=harp_grace_candidates,
            harp_ratios=harp_ratios,
            harp_lambda_stall=harp_lambda_stall,
            harp_lambda_quality=harp_lambda_quality,
            harp_lambda_fairness=harp_lambda_fairness,
            harp_fairness_epsilon=harp_fairness_epsilon,
            harp_compression_profile=harp_compression_profile,
            harp_compression_trace=harp_compression_trace,
        )


def _build_metric_summary(manifest_df: pd.DataFrame) -> pd.DataFrame:
    valid_status = {"ok", "skipped_existing"}
    rows = []

    for rec in manifest_df.itertuples(index=False):
        if rec.status not in valid_status:
            continue

        result_path = Path(rec.result_csv)
        tier_path = Path(rec.tier_stats_json)
        if not result_path.exists():
            continue

        row = {
            "accelerator": getattr(rec, "accelerator", "A6000"),
            "model": rec.model,
            "tier": rec.tier,
            "policy": rec.policy,
            "workload": rec.workload,
            "status": rec.status,
        }
        row.update(_load_metrics(result_path))
        row.update(_load_tier_totals(tier_path))
        if row.get("generated_tokens_total", 0.0) > 0:
            row["transition_mb_per_generated_token"] = row.get("tier_transition_mb_total", 0.0) / row["generated_tokens_total"]
        else:
            row["transition_mb_per_generated_token"] = 0.0
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
    grouped = summary_df.groupby(["accelerator", "model", "tier", "workload"])

    metric_keys = [
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_latency_ms",
        "p99_ttft_ms",
        "p99_tpot_ms",
        "tier_transition_mb_total",
        "transition_mb_per_generated_token",
        "stall_overlap_ratio",
        "mean_stall_ms_per_request",
        "shadow_hit_rate",
        "harp_avg_shadow_ratio",
        "evicpress_compression_events",
        "evicpress_compressed_bytes_saved",
    ]

    for (accelerator, model, tier, workload), group in grouped:
        by_policy = {row.policy: row for row in group.itertuples(index=False)}
        if baseline_policy not in by_policy:
            continue

        base_row = by_policy[baseline_policy]

        for policy, cmp_row in sorted(by_policy.items()):
            if policy == baseline_policy:
                continue

            row = {
                "accelerator": accelerator,
                "model": model,
                "tier": tier,
                "workload": workload,
                "baseline_policy": baseline_policy,
                "compare_policy": policy,
            }

            for key in metric_keys:
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
    available_model_keys = sorted(MODEL_SPECS.keys())
    parser = argparse.ArgumentParser(description="Run model x tier x policy matrix")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama8b", "phi_moe"],
        help=(
            "Model keys/aliases or 'all'. "
            f"Canonical keys: {', '.join(available_model_keys)}"
        ),
    )
    parser.add_argument("--tiers", nargs="+", default=["cpu_dram", "cxl", "pcie_nvme"], help="Tier keys or 'all'")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["tail", "fifo", "lru", "largest_kv", "evicpress", "harp"],
        help="Eviction policies or 'all'",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["sharegpt_100", "fixed_256"],
        help="Workload keys or 'all'",
    )
    parser.add_argument(
        "--network-backend",
        type=str,
        choices=["analytical", "ns3"],
        default="analytical",
        help="Backend for network simulation",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="A6000",
        help="Accelerator/NPU hardware type used for trace generation (e.g., A6000, H100)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--rerun", action="store_true", help="Re-run even if outputs exist")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-run timeout in seconds")
    parser.add_argument("--jobs", type=int, default=1, help="Number of concurrent runs")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bar (uses tqdm if installed, otherwise periodic text updates)",
    )
    parser.add_argument("--num-req-override", type=int, default=None, help="Override num_req for all workloads")
    parser.add_argument("--evicpress-alpha", type=float, default=1.0, help="EVICPRESS alpha")
    parser.add_argument(
        "--evicpress-ratios",
        type=str,
        default="1.0,0.75,0.5,0.25",
        help="EVICPRESS compression keep ratios",
    )
    parser.add_argument("--harp-grace-candidates", type=str, default="16,32,64", help="HARP grace-token candidates")
    parser.add_argument("--harp-ratios", type=str, default="1.0,0.75,0.5,0.25", help="HARP compression keep ratios")
    parser.add_argument("--harp-lambda-stall", type=float, default=1.0, help="HARP stall penalty weight")
    parser.add_argument("--harp-lambda-quality", type=float, default=0.5, help="HARP quality penalty weight")
    parser.add_argument("--harp-lambda-fairness", type=float, default=0.1, help="HARP fairness penalty weight")
    parser.add_argument("--harp-fairness-epsilon", type=float, default=1e-6, help="HARP fairness epsilon")
    parser.add_argument("--harp-compression-profile", type=str, default="balanced", help="HARP compression profile preset")
    parser.add_argument("--harp-compression-trace", type=str, default="", help="Optional HARP compression trace JSON/CSV")
    args = parser.parse_args()

    selected_models = _select_models(args.models)
    selected_tiers = _select_items(args.tiers, TIER_CONFIGS, "tiers")
    selected_policies = _select_policies(args.policies)
    selected_workloads = _select_items(args.workloads, WORKLOADS, "workloads")

    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")

    print("=" * 104)
    print("Model + Tier + Policy Matrix Runner")
    print(f"Models:    {selected_models}")
    print(f"Tiers:     {selected_tiers}")
    print(f"Policies:  {selected_policies}")
    print(f"Workloads: {selected_workloads}")
    print(f"Accel:     {args.accelerator}")
    print(f"Backend:   {args.network_backend}")
    print(f"Output:    {OUTPUT_ROOT}")
    print("=" * 104)

    config_paths = {
        (model, tier): _generate_model_tier_config(model, tier, args.accelerator)
        for model in selected_models
        for tier in selected_tiers
    }

    total = len(selected_models) * len(selected_tiers) * len(selected_policies) * len(selected_workloads)
    ordered_records: List[Optional[RunRecord]] = [None] * total

    run_specs = []
    for model in selected_models:
        for tier in selected_tiers:
            cfg_path = config_paths[(model, tier)]
            for policy in selected_policies:
                for wl_key in selected_workloads:
                    wl = WORKLOADS[wl_key]
                    num_req = args.num_req_override if args.num_req_override is not None else wl["num_req"]
                    dataset = _resolve_dataset(model, wl_key)
                    run_specs.append(
                        {
                            "model": model,
                            "tier": tier,
                            "policy": policy,
                            "workload": wl_key,
                            "cluster_config_path": cfg_path,
                            "dataset": dataset,
                            "num_req": num_req,
                        }
                    )

    if args.jobs == 1:
        iterator = enumerate(run_specs, start=1)
        use_tqdm = bool(args.progress and tqdm is not None)
        if use_tqdm:
            iterator = tqdm(iterator, total=total, desc="Runs", unit="run")

        if args.progress and tqdm is None:
            print("Progress bar requested, but tqdm is not installed; using text progress updates.")

        for idx, spec in iterator:
            if not use_tqdm:
                print(f"[{idx}/{total}] {spec['model']} | {spec['tier']} | {spec['policy']} | {spec['workload']}")

            rec = _run_one(
                accelerator=args.accelerator,
                model=spec["model"],
                tier=spec["tier"],
                policy=spec["policy"],
                workload=spec["workload"],
                cluster_config_path=spec["cluster_config_path"],
                dataset=spec["dataset"],
                num_req=spec["num_req"],
                network_backend=args.network_backend,
                timeout=args.timeout,
                dry_run=args.dry_run,
                rerun=args.rerun,
                evicpress_alpha=args.evicpress_alpha,
                evicpress_ratios=args.evicpress_ratios,
                harp_grace_candidates=args.harp_grace_candidates,
                harp_ratios=args.harp_ratios,
                harp_lambda_stall=args.harp_lambda_stall,
                harp_lambda_quality=args.harp_lambda_quality,
                harp_lambda_fairness=args.harp_lambda_fairness,
                harp_fairness_epsilon=args.harp_fairness_epsilon,
                harp_compression_profile=args.harp_compression_profile,
                harp_compression_trace=args.harp_compression_trace,
            )
            ordered_records[idx - 1] = rec
    else:
        print(f"Executing with parallel workers: {args.jobs}")
        future_to_idx = {}
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            for idx, spec in enumerate(run_specs, start=1):
                print(f"[queue {idx}/{total}] {spec['model']} | {spec['tier']} | {spec['policy']} | {spec['workload']}")
                future = executor.submit(
                    _run_one,
                    accelerator=args.accelerator,
                    model=spec["model"],
                    tier=spec["tier"],
                    policy=spec["policy"],
                    workload=spec["workload"],
                    cluster_config_path=spec["cluster_config_path"],
                    dataset=spec["dataset"],
                    num_req=spec["num_req"],
                    network_backend=args.network_backend,
                    timeout=args.timeout,
                    dry_run=args.dry_run,
                    rerun=args.rerun,
                    evicpress_alpha=args.evicpress_alpha,
                    evicpress_ratios=args.evicpress_ratios,
                    harp_grace_candidates=args.harp_grace_candidates,
                    harp_ratios=args.harp_ratios,
                    harp_lambda_stall=args.harp_lambda_stall,
                    harp_lambda_quality=args.harp_lambda_quality,
                    harp_lambda_fairness=args.harp_lambda_fairness,
                    harp_fairness_epsilon=args.harp_fairness_epsilon,
                    harp_compression_profile=args.harp_compression_profile,
                    harp_compression_trace=args.harp_compression_trace,
                )
                future_to_idx[future] = (idx, spec)

            completed_iter = as_completed(future_to_idx)
            use_tqdm = bool(args.progress and tqdm is not None)
            if use_tqdm:
                completed_iter = tqdm(completed_iter, total=total, desc="Completed", unit="run")
            elif args.progress and tqdm is None:
                print("Progress bar requested, but tqdm is not installed; using text completion updates.")

            done_count = 0
            for future in completed_iter:
                idx, spec = future_to_idx[future]
                rec = future.result()
                ordered_records[idx - 1] = rec
                done_count += 1
                if not use_tqdm:
                    print(
                        f"[done {done_count}/{total}] {spec['model']} | {spec['tier']} | "
                        f"{spec['policy']} | {spec['workload']} -> {rec.status}"
                    )

    records: List[RunRecord] = [r for r in ordered_records if r is not None]

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

    print("=" * 104)
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
    print("=" * 104)


if __name__ == "__main__":
    main()
