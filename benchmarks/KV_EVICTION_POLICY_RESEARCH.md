# KV Cache Eviction Policy Landscape (LLM Serving)

This memo summarizes commonly used KV-cache eviction families and where they fit in LLM serving systems.

## 1) Classical cache policies (context/block level)

- LRU (Least Recently Used): evict entries with oldest recent access.
- LFU (Least Frequently Used): evict entries with lowest access frequency.
- FIFO / Random: simple baselines and stress tests.
- Size-aware variants: prioritize entries by utility-per-byte when entry sizes differ.

Where used:
- Many practical implementations start with LRU-style block/request eviction because it is cheap and robust under changing traffic.
- EVICPRESS explicitly compares against LRU-based eviction baselines and motivates why fixed LRU is suboptimal under heterogeneous compression sensitivity.

## 2) Request- or context-aware policies

- Recency + size: choose victims with lower expected near-term reuse and larger byte footprint.
- Remaining-length-aware: deprioritize contexts close to completion.
- Prefix-aware retention: retain shared/prefix-heavy KV regions to maximize future hit probability.

Why this matters:
- KV entries are not all equal: larger entries and high-reuse prefixes should be treated differently from one-off contexts.

## 3) Token-importance eviction / dropping

- Attention-score based dropping (e.g., heavy-hitter style): keep influential tokens, drop less important ones.
- Head-guided importance (e.g., IMPRESS-style): estimate token importance from selected attention heads.
- Structured token dropping (e.g., SnapKV-style methods): retain salient token subsets for quality/latency trade-offs.

Tradeoff:
- Better memory efficiency than blind LRU, but quality can degrade if importance estimation is off.

## 4) Compression-only policies

- Quantization-based KV compression.
- Token-dropping or merging-based compression.
- Fixed compression ratio for all contexts.

Limitation:
- Uniform compression is often suboptimal because contexts have different sensitivity to compression error.

## 5) Joint compression + eviction (EVICPRESS family)

Core idea (from EVICPRESS):
- Evaluate each context under multiple (method, ratio, tier) configurations.
- Score each configuration with a utility that combines quality and loading delay, weighted by access frequency.
- When a tier is full, update configurations using a greedy strategy that minimizes utility-score drop while satisfying capacity, recursively across lower tiers.

Utility form described in the paper:
- Util(method, ratio, device) = (alpha * quality - TTFT) * frequency

Why it outperforms fixed eviction:
- It adapts by context sensitivity and tier bandwidth/latency instead of applying one policy globally.

## 6) Practical policy checklist for simulators

When implementing policies in a simulator, compare at least:
- tail / fifo (oldest-arrival) / size-aware / random baselines
- an EVICPRESS-like utility policy with adaptive compression ratio selection

Track these metrics per run:
- TTFT mean/p99
- TPOT mean/p99
- tier transition bytes and events
- effective compression ratio and bytes saved
- quality proxy (if available)

## 7) What was implemented in this repo

Implemented policy set now includes:
- tail
- fifo (`oldest` alias)
- lru
- largest_kv
- smallest_kv
- random
- evicpress

The EVICPRESS policy in this simulator is a lightweight online approximation:
- Uses a utility-drop-per-byte greedy action at eviction time.
- Chooses (compression ratio, target tier) per candidate request.
- Uses tier bandwidth/latency and a request-level sensitivity proxy to balance quality and delay.

This matches the paper's design direction (utility + greedy updates), while remaining practical in a trace-driven simulator without full per-context offline quality profiling.
