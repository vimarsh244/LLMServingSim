
---
# Changes made for DaSH/Marvell
## Tiered KV Cache: NPU → CXL → CPU Eviction

This section documents the **3-tier KV cache eviction** feature added on top of the v1.0.0
release. It implements a dynamic NPU → CXL → CPU eviction chain for KV cache blocks, along
with full instrumentation, experiment automation, and analysis tooling.

### Overview

When NPU memory is exhausted during inference serving, KV cache blocks must be evicted to
make room for new requests. The original simulator supported only NPU → CPU eviction. This
extension adds CXL (Compute Express Link) memory as an intermediate tier:

```
NPU (fast, small) → CXL (medium speed, medium capacity) → CPU (slow, large)
```

At eviction time, the scheduler attempts CXL first. If CXL has insufficient space, it falls
back to CPU. On reload, KV blocks are fetched from whichever tier they reside in.

### Modified Files

#### `inference_serving/memory_model.py`
- Added `self.cxl_used = 0` byte counter for KV cache tracking on CXL (independent of prefix
  cache RadixCache)
- Updated `allocate()`, `free()`, `is_avail()` for `Device.CXL` to use the simple byte
  counter when not in prefix caching mode, or delegate to `second_tier_prefix_cache` when
  prefix caching is enabled

#### `inference_serving/scheduler.py`
- **Eviction loop** (`schedule_base()`): Checks `self.memory.is_avail(size, Device.CXL)` first;
  if CXL has space, evicts there; otherwise falls back to CPU
- **Reload path**: Checks `req.evict_device` to free from the correct tier (CXL or CPU)
- **Tier stats**: Added 4 new counters: `evict_npu_to_cxl_bytes`, `evict_npu_to_cxl_count`,
  `load_cxl_to_npu_bytes`, `load_cxl_to_npu_count`
- **`print_tier_stats()`**: Prints NPU→CXL and CXL→NPU lines alongside existing CPU stats

#### `inference_serving/request.py`
- `Request.evict_device`: New field (`None` / `Device.CXL` / `Device.CPU`) tracking where
  KV blocks were evicted to, enabling correct reload routing
- `Batch.evict_cxl` / `Batch.load_cxl`: New fields carrying per-batch CXL transfer sizes
  for ASTRA-Sim trace generation

#### `inference_serving/trace_generator.py`
- Reads `batch.evict_cxl` and `batch.load_cxl` to emit separate `kv_load` / `kv_evict` trace
  entries targeting `CXL:0` device (listed before CPU entries, as the faster tier)
- CPU entries continue using `get_device(placement, ..., 'kv_evict_loc')` for the CPU device
  string

#### `main.py`
- Time-series CSV now includes `cxl_used_bytes`, `cxl_total_bytes`,
  `evict_npu_to_cxl_bytes_total`, `load_cxl_to_npu_bytes_total`
- Tier stats JSON (`*_tier_stats.json`) automatically includes CXL counters
- End-of-simulation output prints CXL eviction/reload stats

### New Files

#### `cluster_config/tiered_kv_npu_cxl_cpu.json`
3-tier cluster config: NPU 18 GB (768 GB/s), CXL 256 GB (150 ns, 120 GB/s), CPU 128 GB
(256 GB/s). Model: Llama-3.1-8B on A6000.

#### `benchmarks/run_baseline.py`
Automated multi-phase experiment sweep:
- **Phase A**: KV eviction pressure — `npu_cpu` and `npu_cxl_cpu` configs × workloads
- **Phase B**: Prefix cache tier comparison (4 modes × 4 workloads)
- **Phase C**: CXL sensitivity (9 latency × bandwidth combos)
- **Phase D**: Block size sensitivity (4 sizes × 2 workloads)

```bash
python benchmarks/run_baseline.py --phase A
python benchmarks/run_baseline.py --phase all --dry-run
```

#### `benchmarks/run_cxl_sweep.py`
CXL memory size sweep: varies CXL capacity across 0 / 4 / 8 / 16 / 32 / 64 / 128 / 256 GB,
running `sharegpt_300` and `fixed_256` workloads. Auto-generates cluster configs, runs all
experiments, and produces 5 categories of comparison plots.

```bash
python benchmarks/run_cxl_sweep.py              # run all + plot
python benchmarks/run_cxl_sweep.py --dry-run     # print commands only
python benchmarks/run_cxl_sweep.py --plot-only   # regenerate plots from existing data
```

#### `benchmarks/analyze_results.py`
Full analysis and plotting pipeline (1100+ lines). Generates publication-quality plots:
latency bars/CDFs, memory utilization time-series (3-tier stacked area), migration volume
breakdown (NPU→CXL + NPU→CPU), eviction-latency correlation, CXL sensitivity heatmaps,
block size sensitivity, and Pareto frontier dashboards.

```bash
python benchmarks/analyze_results.py --results-dir output/tiered_kv --phase phaseA
```

#### `benchmarks/plot_utils.py`
Shared plotting utilities: color palettes (including `npu_cxl_cpu`, CXL eviction tiers),
label maps, data loaders for result CSVs / time-series CSVs / tier stats JSON, experiment
discovery, and multi-format figure saving.

### Output Format

Each experiment produces:

| File | Contents |
|------|----------|
| `result.csv` | Per-request: instance ID, request ID, input/output tokens, latency, TTFT, TPOT, ITL |
| `timeseries.csv` | Per-interval: NPU/CPU/CXL memory usage, cache hits, throughput, cumulative tier stats |
| `result_tier_stats.json` | Aggregate per-instance: eviction/reload bytes and event counts for all tiers |

### Quick Start — 3-Tier Eviction Experiment

```bash
# Single run
python main.py \
    --cluster-config cluster_config/tiered_kv_npu_cxl_cpu.json \
    --dataset dataset/sharegpt_req300_rate10_llama.jsonl \
    --output output/test/result.csv \
    --timeseries-output output/test/timeseries.csv \
    --block-size 16

# CXL size sweep (0–256 GB)
python benchmarks/run_cxl_sweep.py

# Analyze and plot
python benchmarks/analyze_results.py \
    --results-dir output/tiered_kv --phase phaseA --output-dir figures/phaseA
```