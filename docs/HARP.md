# HARP KV Eviction Policy

HARP, implemented as `harp` in `inference_serving/eviction_policies/harp.py`, is a KV eviction policy that combines three decisions into one action:

1. Which request to evict.
2. How much of the request to keep as a grace window.
3. Which spill tier to use for the remaining KV bytes.

It is designed for decode requests, where the scheduler may need to free NPU KV memory while still preserving enough state to resume the request later with as little stall as possible.

## When HARP is active

HARP is selected when `--kv-eviction-policy harp` is set. In the scheduler, that policy is only enabled when prefix caching is disabled. If prefix caching is on, the scheduler rejects `harp` at initialization.

At runtime, HARP is consulted during the normal batch formation path in `schedule_base()`:

- Before scheduling, the scheduler advances any in-flight HARP prefetch progress.
- It decays each active request's fairness debt.
- It filters ready requests so that decode requests still waiting on HARP prefetch can stall until their data is ready or their grace window is active.
- If there is not enough NPU KV memory to form a batch, the scheduler asks the HARP policy which request to evict.

## What HARP does

HARP converts an evicted request into a staged state instead of treating eviction as a simple one-way spill.

The request is moved into one of two states:

- `shadow`: the request keeps a tail grace window in NPU-resident bytes for a bounded number of tokens.
- `cold`: the request is fully evicted with no grace tail.

The policy also chooses a spill tier:

- `CPU`
- `CXL` when the configured CXL memory tier exists

For the surviving KV payload, HARP stores only a compressed portion of the freed bytes in the chosen tier and records the rest as bytes that are effectively removed from the live NPU footprint.

## The three score terms

HARP chooses the action with the lowest combined score:

`score = lambda_stall * stall_penalty + lambda_quality * quality_loss + lambda_fairness * fairness_term`

Each term measures a different kind of cost:

- `stall_penalty` answers: will this eviction force the request to wait before it can decode again?
- `quality_loss` answers: how much information do we lose by compressing the spilled KV state?
- `fairness_term` answers: are we over-favoring this request compared with others in the pool?

The weights let you decide which cost matters most. Bigger `lambda_stall` pushes HARP to prefer actions that are ready sooner. Bigger `lambda_quality` makes it favor higher-retention actions, even if they are slower or take more space. Bigger `lambda_fairness` makes it spread evictions more evenly across requests.

In practice, HARP is trying to avoid a bad outcome where one request gets easy treatment repeatedly while others absorb all the eviction pressure.

## How the eviction decision is made

The policy evaluates candidate actions of the form:

`(request, spill tier, keep ratio, grace tokens)`

For each request in the eviction pool, HARP skips requests that are already marked for eviction. For the remaining request, it computes:

- `total_bytes`: the request's total evictable KV footprint
- `missing_raw_bytes`: bytes already accounted for as missing from prior HARP handling
- `resident_bytes = total_bytes - missing_raw_bytes`
- `fairness_debt`: a per-request debt score that increases as other requests are favored
- `sensitivity`: a request-length and remaining-output heuristic used to estimate quality impact

It then enumerates:

- grace token candidates, defaulting to `16, 32, 64`
- keep ratios, defaulting to `1.0, 0.75, 0.5, 0.25`
- spill tiers that have available capacity

For each candidate, HARP estimates a score:

`score = lambda_stall * stall_penalty + lambda_quality * quality_loss + lambda_fairness * fairness_term`

Where:

- `stall_penalty` is the estimated prefetch time minus the grace window, clamped at zero. If the grace window is long enough to cover prefetch, this term is zero.
- `quality_loss` comes from a compression-profile penalty multiplied by the request sensitivity. More compression usually means more loss, especially for sensitive requests.
- `fairness_term` is the request's fairness debt normalized by the amount of freed bytes. A request with more accumulated debt becomes more expensive to favor again.

The terms are not independent in effect:

- A larger grace window can reduce stall, but it also reduces how many bytes are actually freed.
- A lower keep ratio reduces storage usage, but usually increases quality loss.
- A request with high fairness debt becomes less attractive unless the other terms are clearly better.

The policy selects the action with the lowest score. If scores tie, it prefers the one that frees more raw bytes, and then the one that stores fewer bytes.

## What happens during eviction

When the scheduler applies a HARP eviction action, it:

1. Frees the full raw KV footprint from NPU memory.
2. Allocates the compressed stored bytes in the chosen spill tier.
3. Records the per-request and global tier-transition counters.
4. Sets up asynchronous prefetch bookkeeping for the stored bytes.
5. Saves the grace token count and tail byte count on the request.
6. Marks the request as `shadow` if a grace tail remains, otherwise `cold`.

This is why HARP is not just an eviction policy. It also models the cost of bringing the request back.

## How prefetch progresses

The scheduler advances HARP prefetch work every time it reaches the start of a scheduling step.

The progress model is simple:

- consume any remaining latency first
- then move data at the configured CPU or CXL bandwidth
- decrement the remaining source bytes accordingly
- free the transferred bytes from the spill tier
- accumulate reload counters and overlap statistics

If the request still has grace tokens left, the moved bytes are counted as overlapping useful work. Once the remaining bytes reach zero, the request is no longer stalled.

## When a request is allowed to run again

A decode request is schedulable if either:

- its HARP prefetch is already complete, or
- its grace-token window is still active

If neither condition is true, the scheduler marks the request as stalled and records the estimated stall time.

Prefill requests are not blocked by HARP in the same way. They are allowed through immediately.

## Request state tracked by HARP

Each request carries a small amount of HARP state in `inference_serving/request.py`:

- `harp_state`: `hot`, `shadow`, or `cold`
- `harp_grace_tokens_remaining`: remaining grace tokens
- `harp_grace_tail_bytes`: tail bytes preserved in NPU
- `harp_missing_raw_bytes`: raw bytes already removed from the live NPU footprint
- `harp_prefetch_remaining_bytes`: bytes left to reload from the spill tier
- `harp_storage_bytes_remaining`: bytes still resident in CPU or CXL storage
- `harp_prefetch_latency_ns_remaining`: latency still to pay before data transfer moves
- `harp_fairness_debt`: scheduling debt used to balance eviction choices

These fields let the scheduler reason about a request across multiple iterations instead of treating eviction as a one-time event.

## Metrics you can inspect

The scheduler tracks HARP-specific counters in `scheduler.py`, including:

- total prefetched bytes
- prefetch progress bytes
- overlap bytes during grace windows
- stall events and stall time
- shadow-hit tokens
- total decode tokens
- average shadow compression ratio
- shadow vs cold eviction counts

These values are reported in `print_tier_stats()` and written into the per-request CSV output.

## Tunables

The main knobs are:

- `harp_grace_candidates`: token counts to try for the grace window
- `harp_ratios`: stored-byte keep ratios to consider
- `harp_lambda_stall`: weight for stall avoidance
- `harp_lambda_quality`: weight for compression quality loss
- `harp_lambda_fairness`: weight for fairness debt
- `harp_fairness_epsilon`: denominator floor for fairness normalization
- `harp_compression_profile`: named compression profile used to estimate quality loss
- `harp_compression_trace`: optional trace file used by the compression profile loader

The defaults are chosen to make HARP conservative: it prefers actions that avoid visible stall, but it will still use compression and grace windows when they improve the score.

## Practical summary

In short, HARP tries to answer three questions every time the scheduler is out of KV memory:

- What can I evict with the least future pain?
- How much of that request should I preserve as a short grace window?
- Where should the remaining KV bytes live while the request is off NPU?

That makes HARP a staged eviction policy rather than a pure replacement policy.