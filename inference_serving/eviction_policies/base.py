from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

from ..memory_model import Device


@dataclass
class EvictionAction:
    req: Any
    raw_bytes: int
    stored_bytes: int
    device: Device
    ratio: float = 1.0
    utility: float = 0.0
    grace_tokens: int = 0
    grace_bytes: int = 0
    target_state: str = "cold"
    score: float = 0.0


class EvictionPolicy(ABC):
    name = "base"

    def __init__(self, **kwargs):
        del kwargs

    def build_pool(self, gen_req: List[Any], scheduler: Any) -> List[Any]:
        return list(gen_req)

    @abstractmethod
    def select_action(self, evict_pool: List[Any], scheduler: Any) -> Optional[EvictionAction]:
        raise NotImplementedError
