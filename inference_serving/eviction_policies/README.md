# Custom KV Eviction Policies

Eviction policies are modularized under this folder.

## Built-in policies

- `tail`
- `fifo` (arrival-order eviction)
- `lru` (least-recently-used, based on request service recency)
- `oldest` (compatibility alias of `fifo`)
- `largest_kv`
- `smallest_kv`
- `random`
- `evicpress`

## Add a custom policy

1. Create a new Python module in this folder (for example, `my_policy.py`).
2. Implement a class that subclasses `EvictionPolicy`.
3. Register it with `@register_policy("my_policy")`.
4. Make sure your module is imported from `__init__.py` so registration runs.

Minimum shape:

```python
from .base import EvictionPolicy, EvictionAction
from .registry import register_policy


@register_policy("my_policy")
class MyPolicy(EvictionPolicy):
    def select_action(self, evict_pool, scheduler):
        # Return EvictionAction or None.
        return None
```

Then run:

```bash
python main.py --kv-eviction-policy my_policy ...
```
