import random
from typing import Any, List, Optional

from ..memory_model import Device
from .base import EvictionAction, EvictionPolicy
from .registry import register_policy


class _OrderedPolicy(EvictionPolicy):
    def _order(self, pool: List[Any], scheduler: Any) -> List[Any]:
        return pool

    def build_pool(self, gen_req: List[Any], scheduler: Any) -> List[Any]:
        return self._order(list(gen_req), scheduler)

    def select_action(self, evict_pool: List[Any], scheduler: Any) -> Optional[EvictionAction]:
        while evict_pool:
            req = evict_pool.pop()
            if req.evict:
                continue

            raw_bytes = scheduler.memory.get_evict_kv(req)
            if raw_bytes <= 0:
                continue

            stored_bytes = raw_bytes
            target_device = Device.CPU
            if scheduler.memory.cxl_mem > 0 and scheduler.memory.is_avail(stored_bytes, Device.CXL):
                target_device = Device.CXL
            elif not scheduler.memory.is_avail(stored_bytes, Device.CPU):
                # This request cannot be spilled to any lower tier right now.
                # Keep scanning in policy order for another feasible victim.
                continue

            return EvictionAction(
                req=req,
                raw_bytes=raw_bytes,
                stored_bytes=stored_bytes,
                device=target_device,
                ratio=1.0,
                utility=0.0,
            )

        return None


@register_policy("tail")
class TailPolicy(_OrderedPolicy):
    pass


@register_policy("fifo")
class FIFOPolicy(_OrderedPolicy):
    def _order(self, pool: List[Any], scheduler: Any) -> List[Any]:
        pool.sort(key=lambda req: (req.arrival, req.id), reverse=True)
        return pool


# Backward-compatible alias. Semantics are FIFO (arrival-order), not true LRU.
@register_policy("oldest")
class OldestPolicy(FIFOPolicy):
    pass


@register_policy("lru")
class LRUPolicy(_OrderedPolicy):
    def _order(self, pool: List[Any], scheduler: Any) -> List[Any]:
        del scheduler
        # Least recently used requests should be evicted first.
        # `recent_end` is updated when the request is last served.
        pool.sort(key=lambda req: (getattr(req, "recent_end", req.arrival), req.arrival, req.id), reverse=True)
        return pool


@register_policy("largest_kv")
class LargestKVPolicy(_OrderedPolicy):
    def _order(self, pool: List[Any], scheduler: Any) -> List[Any]:
        pool.sort(key=lambda req: (scheduler.memory.get_evict_kv(req), req.arrival, req.id))
        return pool


@register_policy("smallest_kv")
class SmallestKVPolicy(_OrderedPolicy):
    def _order(self, pool: List[Any], scheduler: Any) -> List[Any]:
        pool.sort(key=lambda req: (scheduler.memory.get_evict_kv(req), req.arrival, req.id), reverse=True)
        return pool


@register_policy("random")
class RandomPolicy(_OrderedPolicy):
    def _order(self, pool: List[Any], scheduler: Any) -> List[Any]:
        random.shuffle(pool)
        return pool
