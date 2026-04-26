# cluster_config

This directory contains cluster configuration files that define the hardware topology,
instance layout, memory hierarchy, and interconnect parameters for LLMServingSim.

Pass a config file to `main.py` via `--cluster-config cluster_config/{name}.json`.

## Configuration format

```json
{
  "num_nodes": 1,
  "link_bw": 112,
  "link_latency": 0,
  "nodes": [
    {
      "num_instances": 1,
      "cpu_mem": {
        "mem_size": 128,
        "mem_bw": 256,
        "mem_latency": 110
      },
      "instances": [
        {
          "model_name": "meta-llama/Llama-3.1-8B",
          "hardware": "A6000",
          "npu_mem": {
            "mem_size": 40,
            "mem_bw": 768,
            "mem_latency": 0
          },
          "npu_num": 1,
          "npu_group": 1,
          "pd_type": null
        }
      ]
    }
  ]
}
```

### Top-level fields

| Field | Type | Description |
| --- | --- | --- |
| `num_nodes` | Integer | Number of nodes in the cluster |
| `link_bw` | Float | Inter-node link bandwidth in GB/s |
| `link_latency` | Float | Inter-node link latency in ns |
| `kv_eviction_policy` | String (optional) | Default KV eviction policy for this cluster config (`tail`, `fifo`, `lru`, `largest_kv`, `smallest_kv`, `random`, `evicpress`, `harp`; `oldest` remains a compatibility alias for `fifo`) |
| `external_kv_tier` | Object (optional) | External KV spill tier profile (`name`, `mem_size`, `mem_bw`, `mem_latency`, `num_devices`) |

### Per-node fields

| Field | Type | Description |
| --- | --- | --- |
| `num_instances` | Integer | Number of instances on this node |
| `cpu_mem.mem_size` | Float | CPU memory capacity in GB |
| `cpu_mem.mem_bw` | Float | CPU memory bandwidth in GB/s |
| `cpu_mem.mem_latency` | Float | CPU memory latency in ns |

### Per-instance fields

| Field | Type | Description |
| --- | --- | --- |
| `model_name` | String | HuggingFace model identifier |
| `hardware` | String | Hardware target matching a profile in `llm_profile/perf_models/` |
| `npu_mem.mem_size` | Float | NPU memory capacity in GB |
| `npu_mem.mem_bw` | Float | NPU memory bandwidth in GB/s |
| `npu_mem.mem_latency` | Float | NPU memory latency in ns |
| `npu_num` | Integer | Number of NPUs in this instance |
| `npu_group` | Integer | NPU group size for tensor parallelism |
| `pd_type` | String or null | `"prefill"`, `"decode"`, or `null` for combined |

### Optional per-instance fields

| Field | Type | Description |
| --- | --- | --- |
| `placement` | Object | Per-layer placement rules for weights, KV cache, and experts |
| `pim_config` | String | Path to a PIM device INI file in `pim_config/` |
| `power` | Object | Power configuration for the power model |

### Optional external tier fields

| Field | Type | Description |
| --- | --- | --- |
| `cxl_mem` | Object | Legacy CXL memory expansion parameters (`mem_size`, `mem_bw`, `mem_latency`, `num_devices`) |
| `external_kv_tier` | Object | Generic external KV tier profile mapped to the simulator external-memory path |

### Modeling notes for CPU DRAM vs CXL

- Use non-zero `cpu_mem.mem_latency` (typically 80-150 ns) for realistic comparisons.
- Model CXL as a capacity tier first: set `cxl_mem.mem_size` or `external_kv_tier.mem_size` larger than CPU DRAM when possible.
- For most systems, CXL access latency should be higher than CPU DRAM latency.
- CXL bandwidth may be in the same order of magnitude as host-access paths but is often lower in practice due to protocol overhead and routing.

### Tiered-KV preset defaults (realistic starting points)

These presets are tuned for realistic ordering across tiers and are used in:

- `tiered_kv_tier_cpu_dram.json`
- `tiered_kv_tier_cxl.json`
- `tiered_kv_tier_ethernet.json`
- `tiered_kv_tier_pcie_nvme.json`
- `tiered_kv_tier_ssd.json`

| Tier | Latency (`mem_latency`, ns) | Bandwidth (`mem_bw`, GB/s) | Capacity example (`mem_size`, GB) |
| --- | --- | --- | --- |
| CPU DRAM | 110 | 220 | 512 |
| CXL memory expansion | 200 | 64 | 2048 |
| PCIe NVMe SSD | 80,000 | 6.8 | 4,096 |
| SATA-class SSD | 500,000 | 0.55 | 2,048 |
| Ethernet-backed external tier (conservative datacenter default) | 500,000 | 3 | 2,048 |

Interpretation notes:

- CXL is much faster than storage tiers, but still slower than directly attached DRAM.
- PCIe transport alone does not make NVMe DRAM-like; flash media access, SSD controller/FTL, queueing, and software stack dominate end-to-end latency.
- If you are modeling RDMA-backed remote memory (instead of generic Ethernet-backed external storage), use a lower-latency Ethernet profile (for example 20,000-50,000 ns) and higher effective bandwidth.

Source-backed ranges used for defaults:

| Tier | Evidence range (from sources) | Chosen default in presets |
| --- | --- | --- |
| CXL memory expansion | ~170-250 ns end-to-end for early CXL memory expansion modules; ~200 ns controller add is commonly cited in early deployments (`nextplatform.com`, "Just How Bad Is CXL Memory Latency?") | 200 ns |
| PCIe Gen4 NVMe SSD | Up to 6,800 MB/s sequential read for PM9A3 (`semiconductor.samsung.com` PM9A3 page / datasheet); enterprise SSD response times are commonly discussed around ~100 us averages (`techtarget.com` SSD benchmark primer) | 80,000 ns and 6.8 GB/s |
| SATA-class SSD | Typical SATA SSD random access commonly falls around ~0.5-0.6 ms class (`en.wikipedia.org/wiki/Solid-state_drive`, linked benchmark references) | 500,000 ns and 0.55 GB/s |
| Ethernet-backed external tier | Same-datacenter roundtrip is often modeled around ~500 us as an order-of-magnitude engineering baseline (`gist.github.com/jboner/2841832`) | 500,000 ns and 3 GB/s |

## Provided configurations

| File | Description |
| --- | --- |
| `single_node_single_instance.json` | Single node, single instance (default) |
| `single_node_single_instance_H100.json` | Single node, single instance on H100 |
| `single_node_multi_instance.json` | Single node, multiple instances |
| `single_node_pd_instance.json` | Single node with P/D disaggregation |
| `single_node_moe_single_instance.json` | Single node, single MoE instance |
| `single_node_moe_multi_instance.json` | Single node, multiple MoE instances |
| `single_node_moe_pd_instance.json` | Single node, MoE with P/D disaggregation |
| `single_node_cxl_instance.json` | Single node with CXL memory expansion |
| `single_node_pim_instance.json` | Single node with PIM-enabled memory |
| `single_node_power_instance.json` | Single node with power modeling enabled |
| `single_node_memory_instance.json` | Single node memory hierarchy configuration |
| `dual_node_multi_instance.json` | Dual node, multiple instances |
| `tiered_kv_tier_cpu_dram.json` | Tiered-KV baseline using CPU DRAM spill tier |
| `tiered_kv_tier_cxl.json` | Tiered-KV profile for CXL spill tier |
| `tiered_kv_tier_pcie_nvme.json` | Tiered-KV profile for PCIe NVMe spill tier |
| `tiered_kv_tier_ssd.json` | Tiered-KV profile for SSD spill tier |
| `tiered_kv_tier_ethernet.json` | Tiered-KV profile for Ethernet-backed external memory tier |
| `tiered_kv_tier_cxl_fifo.json` | Tiered-KV CXL profile with `kv_eviction_policy: fifo` |
| `tiered_kv_tier_cxl_lru.json` | Tiered-KV CXL profile with `kv_eviction_policy: lru` |
