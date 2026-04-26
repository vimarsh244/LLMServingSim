from typing import Any, List, Optional

from ..memory_model import Device
from .base import EvictionAction, EvictionPolicy
from .compression_profiles import load_compression_profile
from .registry import register_policy


@register_policy("harp")
class HarpKVPolicy(EvictionPolicy):
    """HARP-KV: Hierarchical Adaptive Residual Prefetch policy.

    Chooses action tuple (tier, ratio, grace_tokens) with objective:
      J_i(a) = lambda_s * S_i + lambda_q * Q_i + lambda_f * D_i / max(F_i(a), eps)
    where S_i models stall risk, Q_i models compression quality loss, and D_i is
    fairness debt.
    """

    def __init__(
        self,
        harp_grace_candidates=None,
        harp_ratios=None,
        harp_lambda_stall: float = 0,
        harp_lambda_quality: float = 0,
        harp_lambda_fairness: float = 0,
        harp_fairness_epsilon: float = 0,
        harp_compression_profile: str = "balanced",
        harp_compression_trace: str = "",
        **kwargs,
    ):
        del kwargs
        if harp_grace_candidates is None:
            harp_grace_candidates = [0]
        if harp_ratios is None:
            harp_ratios = [1.0]

        self.harp_grace_candidates = sorted(
            {max(0, int(v)) for v in harp_grace_candidates}, reverse=True
        )
        if not self.harp_grace_candidates:
            self.harp_grace_candidates = [0]

        self.harp_ratios = sorted(
            {max(1e-6, min(float(r), 1.0)) for r in harp_ratios if float(r) > 0},
            reverse=True,
        )
        if not self.harp_ratios:
            self.harp_ratios = [1.0]

        self.harp_lambda_stall = max(0.0, float(harp_lambda_stall))
        self.harp_lambda_quality = max(0.0, float(harp_lambda_quality))
        self.harp_lambda_fairness = max(0.0, float(harp_lambda_fairness))
        self.harp_fairness_epsilon = max(1e-12, float(harp_fairness_epsilon))
        self.compression_profile = load_compression_profile(
            profile_name=harp_compression_profile,
            trace_path=harp_compression_trace,
        )

    def build_pool(self, gen_req: List[Any], scheduler: Any) -> List[Any]:
        # Start with the oldest requests at the end for tail-pop compatibility.
        del scheduler
        return sorted(gen_req, key=lambda req: (req.arrival, req.id), reverse=True)

    def _bytes_per_token(self, scheduler: Any) -> int:
        return max(1, int(scheduler.memory.get_kv(1)))

    def _token_time_ns(self, scheduler: Any) -> float:
        return max(1.0, float(getattr(scheduler, "harp_token_time_ns_ema", 2_000_000.0)))

    def _prefetch_seconds(self, payload_bytes: float, device: Device, scheduler: Any) -> float:
        if device == Device.CPU:
            bw = scheduler.cpu_tier_bw
            latency_ns = scheduler.cpu_tier_latency
        else:
            bw = scheduler.external_tier_bw
            latency_ns = scheduler.external_tier_latency

        bytes_per_sec = max(1e-9, bw * 1_000_000_000.0)
        return (latency_ns * 1e-9) + (max(0.0, payload_bytes) / bytes_per_sec)

    def _sensitivity(self, req: Any, scheduler: Any) -> float:
        max_pos = max(1, scheduler.config.get("max_position_embeddings", 1))
        remaining = max(0, req.output - req.input)
        total = max(req.output, req.input + 1)
        remaining_ratio = remaining / total
        length_ratio = min(1.0, req.input / max_pos)
        base = min(1.0, 0.2 + 0.5 * remaining_ratio + 0.3 * length_ratio)
        return max(0.0, base)

    def select_action(self, evict_pool: List[Any], scheduler: Any) -> Optional[EvictionAction]:
        best: Optional[EvictionAction] = None
        best_score: Optional[float] = None

        device_candidates = []
        if scheduler.memory.cxl_mem > 0:
            device_candidates.append(Device.CXL)
        device_candidates.append(Device.CPU)

        bytes_per_token = self._bytes_per_token(scheduler)
        token_time_ns = self._token_time_ns(scheduler)

        for req in evict_pool:
            if req.evict:
                continue

            total_bytes = max(0, int(scheduler.memory.get_evict_kv(req)))
            missing_raw_bytes = max(0, int(getattr(req, "harp_missing_raw_bytes", 0)))
            resident_bytes = max(0, total_bytes - missing_raw_bytes)
            if resident_bytes <= 0:
                continue

            fairness_debt = max(0.0, float(getattr(req, "harp_fairness_debt", 0.0)))
            sensitivity = self._sensitivity(req, scheduler)

            max_grace_tokens = max(0, int(resident_bytes // bytes_per_token))

            for grace_tokens_candidate in self.harp_grace_candidates:
                grace_tokens = min(max_grace_tokens, max(0, int(grace_tokens_candidate)))
                tail_bytes = min(resident_bytes, grace_tokens * bytes_per_token)
                freed_bytes = max(0, resident_bytes - tail_bytes)
                if freed_bytes <= 0:
                    continue

                for ratio in self.harp_ratios:
                    stored_bytes = max(1, int(round(freed_bytes * ratio)))
                    payload_bytes = ratio * max(0, total_bytes - tail_bytes)
                    compression_penalty = self.compression_profile.penalty(ratio)
                    quality_loss = sensitivity * compression_penalty

                    for device in device_candidates:
                        if not scheduler.memory.is_avail(stored_bytes, device):
                            continue

                        t_prefetch = self._prefetch_seconds(payload_bytes, device, scheduler)
                        t_grace = grace_tokens * token_time_ns * 1e-9
                        stall_penalty = max(0.0, t_prefetch - t_grace)

                        score = (
                            self.harp_lambda_stall * stall_penalty
                            + self.harp_lambda_quality * quality_loss
                            + self.harp_lambda_fairness
                            * (fairness_debt / max(float(freed_bytes), self.harp_fairness_epsilon))
                        )

                        action = EvictionAction(
                            req=req,
                            raw_bytes=int(freed_bytes),
                            stored_bytes=int(stored_bytes),
                            device=device,
                            ratio=float(ratio),
                            utility=-float(score),
                            grace_tokens=int(grace_tokens),
                            grace_bytes=int(tail_bytes),
                            target_state="shadow" if tail_bytes > 0 else "cold",
                            score=float(score),
                        )

                        if best is None:
                            best = action
                            best_score = score
                            continue

                        if score < best_score:
                            best = action
                            best_score = score
                        elif score == best_score:
                            if action.raw_bytes > best.raw_bytes:
                                best = action
                            elif action.raw_bytes == best.raw_bytes and action.stored_bytes < best.stored_bytes:
                                best = action

        return best


@register_policy("dynmax")
class DynMaxPolicy(HarpKVPolicy):
    """DynMax: HARP with zero lambdas, zero grace tokens, and no compression."""

    def __init__(self, **kwargs):
        kwargs["harp_grace_candidates"] = [0]
        kwargs["harp_ratios"] = [1.0]
        kwargs["harp_lambda_stall"] = 0.0
        kwargs["harp_lambda_quality"] = 0.0
        kwargs["harp_lambda_fairness"] = 0.0
        kwargs["harp_compression_profile"] = "none"
        kwargs["harp_compression_trace"] = ""
        super().__init__(**kwargs)


@register_policy("adaptive_dynmax")
class AdaptiveDynMaxPolicy(DynMaxPolicy):
    """Adaptive DynMax: aggressive early proactivity that decays over progress."""

    def __init__(
        self,
        adaptive_schedule: str = "linear",
        adaptive_progress_start: float = 0.10,
        adaptive_progress_end: float = 0.75,
        adaptive_final_trigger: float = 1.05,
        adaptive_final_target: float = 0.92,
        adaptive_final_steps_ahead: int = 12,
        adaptive_final_max_actions: int = 1,
        **kwargs,
    ):
        self.adaptive_schedule = str(adaptive_schedule or "linear").strip().lower()
        self.adaptive_progress_start = max(0.0, min(float(adaptive_progress_start), 1.0))
        self.adaptive_progress_end = max(self.adaptive_progress_start + 1e-6, min(float(adaptive_progress_end), 1.0))
        self.adaptive_final_trigger = max(0.0, min(float(adaptive_final_trigger), 1.5))
        self.adaptive_final_target = max(0.0, min(float(adaptive_final_target), self.adaptive_final_trigger))
        self.adaptive_final_steps_ahead = max(1, int(adaptive_final_steps_ahead))
        self.adaptive_final_max_actions = max(1, int(adaptive_final_max_actions))
        super().__init__(**kwargs)

    def _current_progress(self, scheduler: Any) -> float:
        total = max(1, int(getattr(scheduler, "req_num", 1)))
        done = max(0, int(len(getattr(scheduler, "done", []))))
        return max(0.0, min(1.0, done / total))

    def _blend(self, start_value: float, end_value: float, progress: float) -> float:
        if self.adaptive_schedule != "linear":
            raise NotImplementedError(f"Unsupported adaptive DynMax schedule '{self.adaptive_schedule}'")
        if progress <= self.adaptive_progress_start:
            return start_value
        if progress >= self.adaptive_progress_end:
            return end_value
        span = self.adaptive_progress_end - self.adaptive_progress_start
        alpha = (progress - self.adaptive_progress_start) / span
        return start_value + (end_value - start_value) * alpha

    def get_adaptive_proactive_settings(self, scheduler: Any):
        progress = self._current_progress(scheduler)
        trigger = self._blend(0.80, self.adaptive_final_trigger, progress)
        target = self._blend(0.60, self.adaptive_final_target, progress)
        steps_ahead = int(round(self._blend(48.0, float(self.adaptive_final_steps_ahead), progress)))
        max_actions = int(round(self._blend(4.0, float(self.adaptive_final_max_actions), progress)))
        steps_ahead = max(1, steps_ahead)
        max_actions = max(1, max_actions)

        if progress <= self.adaptive_progress_start:
            phase = "early"
        elif progress >= self.adaptive_progress_end:
            phase = "late"
        else:
            phase = "transition"

        return {
            "phase": phase,
            "progress": progress,
            "trigger": max(0.0, min(trigger, 1.5)),
            "target": max(0.0, min(target, trigger)),
            "steps_ahead": steps_ahead,
            "max_actions": max_actions,
        }
