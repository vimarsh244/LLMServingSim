from typing import Any, Dict, List, Optional

from ..memory_model import Device
from .base import EvictionAction, EvictionPolicy
from .compression_profiles import load_compression_profile
from .registry import register_policy


@register_policy("evicpress")
class EvicPressPolicy(EvictionPolicy):
    """Online EVICPRESS-inspired greedy policy.

    This policy uses an online utility-drop heuristic at eviction time to select
    request/device/ratio actions, matching the paper's greedy design direction
    while staying compatible with this simulator's request scheduler.
    """

    def __init__(
        self,
        evicpress_alpha: float = 1.0,
        evicpress_ratios=None,
        evicpress_methods=None,
        evicpress_compression_trace: str = "",
        **kwargs,
    ):
        del kwargs
        self.evicpress_alpha = float(evicpress_alpha)
        if evicpress_ratios is None:
            evicpress_ratios = [1.0, 0.75, 0.5, 0.25]

        self.evicpress_ratios = sorted(
            {max(1e-6, min(float(r), 1.0)) for r in evicpress_ratios if float(r) > 0},
            reverse=True,
        )
        if not self.evicpress_ratios:
            self.evicpress_ratios = [1.0]

        if evicpress_methods is None:
            evicpress_methods = ["balanced"]
        self.evicpress_methods = []
        for method in evicpress_methods:
            key = str(method).strip().lower()
            if key:
                self.evicpress_methods.append(key)
        self.evicpress_methods = list(dict.fromkeys(self.evicpress_methods))
        if not self.evicpress_methods:
            self.evicpress_methods = ["balanced"]

        self.evicpress_compression_trace = str(evicpress_compression_trace or "")
        self.compression_profiles: Dict[str, Any] = {}
        for method in self.evicpress_methods:
            if method == "trace":
                if not self.evicpress_compression_trace:
                    raise ValueError(
                        "EVICPRESS method 'trace' requires a valid --evicpress-compression-trace path"
                    )
                profile = load_compression_profile("balanced", self.evicpress_compression_trace)
            else:
                profile = load_compression_profile(method, "")
            self.compression_profiles[method] = profile

    def build_pool(self, gen_req: List[Any], scheduler: Any) -> List[Any]:
        del scheduler
        return list(gen_req)

    def _frequency(self, req: Any) -> float:
        prompt_len = max(1, int(getattr(req, "original_input", req.input)))
        total_decode_budget = max(1, int(req.output - prompt_len))
        decoded_tokens = max(0, int(req.input - prompt_len))
        remaining_tokens = max(1, int(req.output - req.input))
        historical_accesses = max(0, int(getattr(req, "evicpress_access_count", 0)))

        # EVICPRESS uses profiled context access frequency. Approximate online by
        # combining observed decode accesses and expected future accesses.
        future_term = remaining_tokens / 32.0
        history_term = historical_accesses / max(16.0, float(total_decode_budget))
        progress_term = decoded_tokens / max(32.0, float(total_decode_budget))
        freq = 1.0 + min(24.0, future_term + history_term + progress_term)
        return float(freq)

    def _sensitivity(self, req: Any, scheduler: Any) -> float:
        max_pos = max(1, scheduler.config.get("max_position_embeddings", 1))
        prompt_len = max(1, int(getattr(req, "original_input", req.input)))
        total_decode_budget = max(1, int(req.output - prompt_len))
        context_len = max(prompt_len, int(req.input))
        remaining_tokens = max(1, int(req.output - req.input))

        length_term = min(1.0, context_len / max_pos)
        remaining_term = min(1.0, remaining_tokens / max(32.0, float(total_decode_budget)))
        sensitivity = 0.18 + 0.55 * length_term + 0.22 * remaining_term
        sensitivity = max(0.10, min(0.95, sensitivity))
        return float(sensitivity)

    def _quality(self, req: Any, ratio: float, method: str, scheduler: Any) -> float:
        sensitivity = self._sensitivity(req, scheduler)
        quality_drop = sensitivity * self.compression_profiles[method].penalty(ratio)
        quality = 1.0 - quality_drop
        return max(0.0, min(1.0, quality))

    def _ttft_seconds(self, moved_bytes: int, device: Device, scheduler: Any) -> float:
        if device == Device.CPU:
            bw = scheduler.cpu_tier_bw
            latency_ns = scheduler.cpu_tier_latency
        else:
            bw = scheduler.external_tier_bw
            latency_ns = scheduler.external_tier_latency

        bytes_per_sec = max(1e-9, bw * 1_000_000_000.0)
        return (latency_ns * 1e-9) + (moved_bytes / bytes_per_sec)

    def _utility(
        self,
        req: Any,
        method: str,
        ratio: float,
        moved_bytes: int,
        device: Device,
        scheduler: Any,
    ) -> float:
        freq = self._frequency(req)
        quality = self._quality(req, ratio, method, scheduler)
        ttft_seconds = self._ttft_seconds(moved_bytes, device, scheduler)
        return (self.evicpress_alpha * quality - ttft_seconds) * freq

    def select_action(self, evict_pool: List[Any], scheduler: Any) -> Optional[EvictionAction]:
        best_action = None
        best_utility_drop_per_saved_byte = None
        best_utility_drop = None
        best_saved_bytes = None
        eps = 1e-12

        device_candidates = []
        if scheduler.memory.cxl_mem > 0:
            device_candidates.append(Device.CXL)
        device_candidates.append(Device.CPU)

        for req in evict_pool:
            if req.evict:
                continue

            raw_bytes = scheduler.memory.get_evict_kv(req)
            if raw_bytes <= 0:
                continue

            # Baseline utility corresponds to keeping uncompressed KV on the
            # current tier (no reload TTFT penalty).
            baseline = self.evicpress_alpha * self._frequency(req)

            for method in self.evicpress_methods:
                for ratio in self.evicpress_ratios:
                    stored_bytes = max(1, int(raw_bytes * ratio))

                    for device in device_candidates:
                        if not scheduler.memory.is_avail(stored_bytes, device):
                            continue

                        option_utility = self._utility(req, method, ratio, stored_bytes, device, scheduler)
                        utility_drop = max(0.0, baseline - option_utility)
                        saved_bytes = max(0, raw_bytes - stored_bytes)
                        utility_drop_per_saved_byte = (
                            utility_drop / float(saved_bytes) if saved_bytes > 0 else float("inf")
                        )

                        action = EvictionAction(
                            req=req,
                            raw_bytes=raw_bytes,
                            stored_bytes=stored_bytes,
                            device=device,
                            ratio=ratio,
                            utility=option_utility,
                            score=utility_drop_per_saved_byte,
                        )

                        if best_action is None:
                            best_action = action
                            best_utility_drop_per_saved_byte = utility_drop_per_saved_byte
                            best_utility_drop = utility_drop
                            best_saved_bytes = saved_bytes
                            continue

                        if utility_drop_per_saved_byte + eps < best_utility_drop_per_saved_byte:
                            best_action = action
                            best_utility_drop_per_saved_byte = utility_drop_per_saved_byte
                            best_utility_drop = utility_drop
                            best_saved_bytes = saved_bytes
                        elif abs(utility_drop_per_saved_byte - best_utility_drop_per_saved_byte) <= eps:
                            if utility_drop + eps < best_utility_drop:
                                best_action = action
                                best_utility_drop_per_saved_byte = utility_drop_per_saved_byte
                                best_utility_drop = utility_drop
                                best_saved_bytes = saved_bytes
                            elif abs(utility_drop - best_utility_drop) <= eps:
                                if saved_bytes > best_saved_bytes:
                                    best_action = action
                                    best_utility_drop_per_saved_byte = utility_drop_per_saved_byte
                                    best_utility_drop = utility_drop
                                    best_saved_bytes = saved_bytes
                                elif (
                                    saved_bytes == best_saved_bytes
                                    and action.stored_bytes < best_action.stored_bytes
                                ):
                                    best_action = action
                                    best_utility_drop_per_saved_byte = utility_drop_per_saved_byte
                                    best_utility_drop = utility_drop
                                    best_saved_bytes = saved_bytes

        return best_action
