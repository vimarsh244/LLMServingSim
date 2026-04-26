from .base import EvictionAction, EvictionPolicy
from .registry import create_policy, get_registered_policy_names, register_policy

# Ensure built-in policies are registered at import time.
from . import basic as _basic_policies  # noqa: F401
from . import evicpress as _evicpress_policy  # noqa: F401
from . import harp as _harp_policy  # noqa: F401

__all__ = [
    "EvictionAction",
    "EvictionPolicy",
    "create_policy",
    "get_registered_policy_names",
    "register_policy",
]
