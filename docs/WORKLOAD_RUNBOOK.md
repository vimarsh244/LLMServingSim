# Workload Runbook

This document maps eviction-policy workload scripts to their purpose, commands, and outputs.

## Core Policy Set

Registered policies are discoverable in `inference_serving/eviction_policies/`:

- Basic: `tail`, `fifo`, `lru`, `oldest`, `largest_kv`, `smallest_kv`, `random`
- Compression-aware: `evicpress`
- Multi-objective staged eviction: `harp`, `dynmax`, `adaptive_dynmax`

## Main Single-Run Pattern

Use `main.py` when validating one policy/config pair before large sweeps:

```bash
python main.py \
  --cluster-config cluster_config/tiered_kv_tier_cpu_dram.json \
  --dataset dataset/sharegpt_req300_rate10_llama.jsonl \
  --num-req 300 \
  --kv-eviction-policy tail \
  --output output/sanity/result.csv \
  --timeseries-output output/sanity/timeseries.csv
```

## Sweep Runners

### Tier x Policy x Workload

```bash
python benchmarks/run_tier_policy_matrix.py \
  --tiers cpu_dram cxl pcie_nvme \
  --policies tail fifo lru largest_kv evicpress harp \
  --workloads sharegpt_100 fixed_256 \
  --rerun
```

Writes to `output/tiered_kv/tier_policy_matrix/`.

### HARP Ablation

```bash
python benchmarks/run_harp_ablation.py --rerun
```

Writes to `output/tiered_kv/harp_ablation/sharegpt1000_cpu_dram/`.

### Model x Tier x Policy

```bash
python benchmarks/run_model_tier_policy_matrix.py \
  --models llama8b phi_moe \
  --tiers cpu_dram cxl \
  --policies tail evicpress harp \
  --workloads sharegpt_100 fixed_256 \
  --jobs 2 --rerun
```

Writes to `output/tiered_kv/model_tier_policy_matrix/`.

### Backend Comparison

```bash
python benchmarks/run_backend_diff.py \
  --configs npu_cpu npu_cxl_cpu \
  --workloads sharegpt_100 prefix_stress \
  --backends analytical ns3 \
  --kv-eviction-policy tail \
  --rerun
```

Writes to `output/tiered_kv/backend_diff/`.

### Baseline Phases

```bash
python benchmarks/run_baseline.py --phase A --kv-eviction-policy tail
python benchmarks/run_baseline.py --phase all --dry-run
```

Writes to `output/tiered_kv/phase*/...`.

## Common Workload Keys

- `sharegpt_100`
- `sharegpt_300`
- `sharegpt_750`
- `sharegpt_1000`
- `sharegpt_1500`
- `fixed_256`
- `prefix_stress`

## Output Files Per Experiment

- `result.csv`: per-request latency metrics
- `timeseries.csv`: periodic utilization and migration metrics
- `result_tier_stats.json`: per-instance aggregate tier traffic stats
- `output.txt`: stdout/stderr capture

## Related Docs

- HARP policy behavior and tunables: `docs/HARP.md`
- Custom policy authoring: `inference_serving/eviction_policies/README.md`
- Policy research notes: `benchmarks/KV_EVICTION_POLICY_RESEARCH.md`