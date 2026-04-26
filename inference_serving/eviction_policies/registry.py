from typing import Dict, List, Type

from .base import EvictionPolicy


_POLICY_REGISTRY: Dict[str, Type[EvictionPolicy]] = {}


def register_policy(name: str):
    key = (name or "").strip().lower()
    if not key:
        raise ValueError("Policy name must be a non-empty string")

    def decorator(policy_cls: Type[EvictionPolicy]) -> Type[EvictionPolicy]:
        if key in _POLICY_REGISTRY:
            raise ValueError(f"Eviction policy '{key}' is already registered")
        _POLICY_REGISTRY[key] = policy_cls
        policy_cls.name = key
        return policy_cls

    return decorator


def create_policy(name: str, **kwargs) -> EvictionPolicy:
    key = (name or "").strip().lower()
    if key not in _POLICY_REGISTRY:
        raise ValueError(
            f"Unsupported kv eviction policy '{name}'. "
            f"Choose one of {sorted(_POLICY_REGISTRY.keys())}"
        )
    return _POLICY_REGISTRY[key](**kwargs)


def get_registered_policy_names() -> List[str]:
    return sorted(_POLICY_REGISTRY.keys())
