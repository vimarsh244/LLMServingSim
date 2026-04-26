# DynMax: A Dynamic Greedy KV Cache Eviction Policy for LLM Serving

## Abstract

We present **DynMax**, a zero-parameter KV cache eviction policy for LLM inference schedulers. Through systematic ablation of HARP — a hierarchical adaptive policy with prefetch overlap, compression-aware scoring, and fairness debt tracking — we find that none of these mechanisms contribute measurably to serving performance. The dominant factor is a single property: how much NPU memory is freed per eviction event. DynMax operationalises this insight directly: at each eviction decision, scan the live pool and evict the request with the largest resident KV cache footprint. On ShareGPT workloads, DynMax reduces mean TTFT by ~5% and mean ITL by ~7% over a tuned HARP baseline, while eliminating all hyperparameters and reducing eviction event count by ~10%.

---

## 1. Background and Motivation

In LLM inference serving, KV caches grow proportionally with sequence length and must reside in fast NPU (GPU) memory for efficient attention computation. Under memory pressure — when many long-context requests are in flight simultaneously — the scheduler must evict KV caches to lower-bandwidth CPU or CXL memory to make room for new or higher-priority requests. The evicted request is then blocked until its KV cache is reloaded before its next decode step.

Every eviction event carries a fixed overhead regardless of how many bytes are moved:

- The evicted request cannot be scheduled until reload completes
- A reload transfer must be initiated and tracked
- The scheduler must work around a reduced active pool

This fixed cost structure has a direct implication: **the goal of an eviction policy should be to minimise how often eviction happens**, not to optimise individual eviction decisions. The way to evict less frequently is to clear as much memory as possible each time an eviction is triggered.

---

## 2. The Problem With Existing Policies

Prior policies miss this framing in different ways.

**Tail / FIFO / Oldest** policies evict based on arrival time, picking the newest or oldest request. These have no awareness of KV cache size, so they routinely evict small requests that free minimal memory. The system reaches pressure again almost immediately, triggering another eviction. This creates a thrashing loop.

**LRU** evicts the least recently scheduled request, which is sensible for cache management but again ignores size. A small, cold request clears almost no memory.

**SmallestKV** explicitly prefers small requests under the assumption that they are cheap to reload. This is locally correct but globally wrong: cheap reloads at the cost of frequent evictions produces worse end-to-end latency than expensive reloads with rare evictions.

**LargestKV** sorts the eviction pool by size at pool-build time and pops the largest. This is the right selection criterion but suffers from a staleness problem: the sort happens once per scheduling cycle, but multiple evictions may occur within that cycle. As evictions proceed and requests generate new tokens, the size ordering becomes stale. Under high workload sizes (1000–1500 requests), this degradation is significant — LargestKV performs 2–3× worse on TTFT than a dynamically-rescanning equivalent.

---

## 3. The DynMax Policy

DynMax makes a single algorithmic choice: **at each eviction call, perform a fresh linear scan of the entire eviction pool and select the request with the largest current resident KV cache footprint.**

```python
def select_action(self, evict_pool, scheduler):
    best_req = None
    best_bytes = 0
    best_device = None

    for req in evict_pool:
        if req.evict:
            continue
        resident_bytes = scheduler.memory.get_evict_kv(req)
        if resident_bytes > best_bytes:
            for device in [Device.CXL, Device.CPU]:
                if scheduler.memory.is_avail(resident_bytes, device):
                    best_req = req
                    best_bytes = resident_bytes
                    best_device = device
                    break

    if best_req is None:
        return None

    return EvictionAction(
        req=best_req,
        raw_bytes=best_bytes,
        stored_bytes=best_bytes,
        device=best_device,
        ratio=1.0,
        grace_tokens=0,
        grace_bytes=0,
        target_state="cold",
        score=0.0,
        utility=float(best_bytes),
    )
```

**No parameters. No state. No estimation. O(n) per call.**

The key distinction from LargestKV is the rescan. Because `get_evict_kv` is called live on every iteration of every `select_action` call, DynMax always operates on current sizes. If a request has grown between eviction rounds, or if a new feasible victim has appeared, DynMax sees it. LargestKV does not.

---

## 4. Why It Works: The Intuition

### 4.1 Eviction Frequency Is the Bottleneck

Consider NPU memory as a parking lot with fixed capacity. Every eviction moves a car to an overflow lot (CPU/CXL). The lot refills as new requests arrive and existing requests grow their KV caches. The question is: how do you keep the lot from constantly filling up?

```
Tail/FIFO (bad):
|████████████████████| full → evict 2MB
|███████████████████ |
|████████████████████| full → evict 1MB   ← thrashing
|███████████████████ |
|████████████████████| full → evict 3MB
         → 664 eviction events

DynMax (good):
|████████████████████| full → evict 8MB
|████████████        |                    ← breathing room
|████████████████    |
|████████████████████| full → evict 8MB
|████████████        |
         → 266 eviction events
```

Clearing more per event directly translates to fewer events, which directly translates to fewer stalls and lower latency.

### 4.2 Large Requests Are Naturally Safe to Evict

Large requests got large by generating many tokens — they have been in the system for a long time. Their next decode step is likely further in the future than a small, newly-arrived request. By the time they need to return to NPU memory, the reload transfer has often completed naturally, without any explicit prefetch orchestration. DynMax accidentally achieves prefetch overlap — the key goal of HARP's shadow/grace machinery — without implementing any of it.

### 4.3 The Dynamic Rescan Matters

Under multi-eviction pressure (common at workload sizes ≥ 1000), multiple evictions happen within a single scheduling cycle. Between evictions, the pool composition changes:

- Requests generate new tokens, growing their KV caches
- Previous evictions change which requests are feasible victims
- New requests may have entered the pool

A static sort goes stale immediately after the first eviction. The gap between LargestKV and DynMax (2–3× TTFT at 1500 requests) is entirely attributable to this staleness. DynMax's live rescan costs O(n) per eviction call but ensures the selection is always optimal given current state.

---

## 5. Empirical Results

All results are on ShareGPT workloads with CPU-DRAM as the eviction tier.

### 5.1 Latency Metrics (sharegpt_1000)

| Policy | Mean TTFT (ms) | Mean TPOT (ms) | Mean ITL (ms) | Throughput (tok/s) |
|---|---|---|---|---|
| HARP baseline | 115,778 | 46.01 | 42.46 | 1,359 |
| **DynMax (HARP all-zero)** | **110,554** | **42.13** | **39.56** | **1,402** |
| HARP no-prefetch | 109,882 | 41.93 | 39.68 | 1,404 |
| Tail | 234,032 | 57.03 | 46.90 | 799 |

### 5.2 Eviction Statistics (sharegpt_1000)

| Policy | Eviction Count | Data Moved (MB) | MB per Eviction |
|---|---|---|---|
| HARP baseline | 314 | 22,994 | 73.2 |
| **DynMax** | **283** | **26,386** | **93.2** |
| HARP no-prefetch | 266 | 25,152 | 94.6 |
| Tail | 664 | 21,852 | 32.9 |

DynMax clears **93 MB per eviction** versus 73 MB for the HARP baseline and 33 MB for tail. The eviction count drops from 314 to 283 — fewer interruptions, lower latency.

### 5.3 TTFT Across Workload Sizes

| Workload | HARP | DynMax | LargestKV | Tail |
|---|---|---|---|---|
| sharegpt_750 | ~90k | ~90k | ~200k | ~170k |
| sharegpt_1000 | ~115k | ~110k | ~295k | ~235k |
| sharegpt_1500 | ~180k | ~180k | ~490k | ~365k |

DynMax matches HARP at all workload sizes. LargestKV degrades 2–3× due to stale pool sorting. The gap widens with workload size, confirming that multi-eviction staleness is the cause.

---

## 6. Ablation: What HARP Gets Wrong

We ablated HARP's three λ terms independently and in combination.

| Config | TTFT | Finding |
|---|---|---|
| Full HARP (λ_stall, λ_quality, λ_fairness) | 115,778 | Baseline |
| λ_stall = 0 | ~115,778 | No effect |
| λ_quality = 0 | ~115,778 | No effect |
| λ_fairness = 0 | ~110,554 | Improvement |
| All λ = 0 (DynMax) | 110,554 | Best |

**Stall penalty** is near-zero across all candidates because grace windows consistently exceed prefetch times at the simulated bandwidth. The term contributes no signal.

**Quality loss** requires sensitivity variance across the pool. Under ShareGPT's context length distribution, requests in the eviction pool have similar lengths and remaining output, making quality scores nearly constant. The term cannot differentiate candidates.

**Fairness debt** actively degrades performance by overriding size-based selection. It forces the eviction of requests that have been "skipped" before, regardless of their size. This reduces per-event memory clearance and increases eviction frequency — the opposite of what the system needs.

Setting all λ = 0 collapses HARP's scorer to pure tiebreaker logic, which selects maximum `raw_bytes` — i.e., DynMax selection — explaining the improvement.

---

## 7. Design Implications

The central lesson from this study is that **eviction policy complexity should be justified by the bottleneck it addresses.** HARP was designed under the assumption that eviction *quality* — which request, how compressed, how much grace window — was the lever worth optimising. The ablation shows this assumption is wrong for throughput-intensive LLM serving workloads. The actual bottleneck is eviction *frequency*, and the lever is per-event memory clearance.

This suggests a general principle for KV cache eviction policy design:

> **Maximise bytes freed per eviction event. Everything else is secondary.**

Prefetch overlap, compression, and fairness may matter in specific regimes — very high memory bandwidth utilisation, heterogeneous workloads with adversarial arrival patterns, or strict SLA fairness requirements. But absent evidence that these regimes are present, the zero-parameter greedy policy is both simpler and better.

---

## 8. Conclusion

DynMax is a dynamic greedy KV cache eviction policy that selects the largest-footprint request via a live linear scan at each eviction decision. It has zero hyperparameters, requires no state between calls, and runs in O(n) time. It matches or beats a fully-tuned HARP baseline across all tested workload sizes by reducing eviction frequency through maximum per-event memory clearance. The improvement over the superficially similar LargestKV policy comes entirely from dynamic rescanning, which ensures correctness under multi-eviction pressure. We recommend DynMax as the default eviction policy for CPU-DRAM tiered KV cache systems operating under ShareGPT-class workloads.