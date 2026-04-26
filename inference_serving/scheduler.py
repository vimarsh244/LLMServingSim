import pandas as pd
from time import time
import csv
import math

from .request import *
from .utils import *
from .controller import *
from .memory_model import *
from .graph_generator import *
from .trace_generator import *
from .pim_model import *
from .eviction_policies import create_policy, get_registered_policy_names
import numpy as np

# class that shedules request of astra-sim
class Scheduler:
    def __init__(self, model, node_id, instance_id, max_batch, max_num_batched_tokens, 
                 npu_num, npu_group, npu_mem, cpu_mem, 
                 start_npu, pd_type, fp, block_size, req_num, 
                 prioritize_prefill, enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage,
                 cxl_mem=0, kv_eviction_policy="tail", external_tier_name="CXL",
                 cpu_tier_bw=256.0, cpu_tier_latency=100.0,
                 external_tier_bw=120.0, external_tier_latency=150.0,
                 evicpress_alpha=1.0, evicpress_ratios=None,
                 harp_grace_candidates=None, harp_ratios=None,
                 harp_lambda_stall=1.0, harp_lambda_quality=0.5, harp_lambda_fairness=0.1,
                 harp_fairness_epsilon=1e-6,
                 harp_compression_profile="balanced", harp_compression_trace="",
                 adaptive_dynmax_schedule="linear",
                 adaptive_dynmax_progress_start=0.10, adaptive_dynmax_progress_end=0.75,
                 adaptive_dynmax_final_trigger=1.05, adaptive_dynmax_final_target=0.92,
                 adaptive_dynmax_final_steps_ahead=12, adaptive_dynmax_final_max_actions=1,
                 enable_proactive_eviction=False, proactive_steps_ahead=32,
                 proactive_trigger=0.85, proactive_target=0.70,
                 proactive_max_actions=2):
        # all time realated variables are in using tick (system tick)
        # LLMServingSim uses Orca, vLLM technique at deafult
        self.model = model
        self.config = get_config(model)
        self.node_id = node_id
        self.instance_id = instance_id
        self.max_batch = max_batch
        self.max_num_batched_tokens = min(max_num_batched_tokens, self.config['max_position_embeddings'])
        self.npu_num = npu_num
        self.npu_group = npu_group
        self.req_num = req_num
        self.start_npu = start_npu
        self.pd_type = pd_type
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_prefix_sharing = enable_prefix_sharing
        self.prefix_storage = prefix_storage
        self.prioritize_prefill = prioritize_prefill
        self.kv_eviction_policy = (kv_eviction_policy or "tail").lower()
        available_evict_policies = get_registered_policy_names()
        if self.kv_eviction_policy not in available_evict_policies:
            raise ValueError(
                f"Unsupported kv eviction policy '{kv_eviction_policy}'. "
                f"Choose one of {available_evict_policies}"
            )
        if self.enable_prefix_caching and self.kv_eviction_policy in ("harp", "dynmax", "adaptive_dynmax"):
            raise ValueError("KV eviction policies 'harp', 'dynmax', and 'adaptive_dynmax' are currently supported only when prefix caching is disabled")
        self.external_tier_name = external_tier_name or "CXL"
        self.cpu_tier_bw = max(float(cpu_tier_bw), 1e-9)
        self.cpu_tier_latency = max(float(cpu_tier_latency), 0.0)
        self.external_tier_bw = max(float(external_tier_bw), 1e-9)
        self.external_tier_latency = max(float(external_tier_latency), 0.0)
        self.evicpress_alpha = float(evicpress_alpha)
        if evicpress_ratios is None:
            evicpress_ratios = [1.0, 0.75, 0.5, 0.25]
        self.evicpress_ratios = sorted(
            {max(1e-6, min(float(r), 1.0)) for r in evicpress_ratios if float(r) > 0},
            reverse=True,
        )
        if not self.evicpress_ratios:
            self.evicpress_ratios = [1.0]
        if harp_grace_candidates is None:
            harp_grace_candidates = [16, 32, 64]
        self.harp_grace_candidates = sorted({max(0, int(v)) for v in harp_grace_candidates}, reverse=True)
        if not self.harp_grace_candidates:
            self.harp_grace_candidates = [0]
        if harp_ratios is None:
            harp_ratios = [1.0, 0.75, 0.5, 0.25]
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
        self.harp_compression_profile = str(harp_compression_profile or "balanced")
        self.harp_compression_trace = str(harp_compression_trace or "")
        self.adaptive_dynmax_schedule = str(adaptive_dynmax_schedule or "linear").strip().lower()
        self.adaptive_dynmax_progress_start = max(0.0, min(float(adaptive_dynmax_progress_start), 1.0))
        self.adaptive_dynmax_progress_end = max(self.adaptive_dynmax_progress_start + 1e-6, min(float(adaptive_dynmax_progress_end), 1.0))
        self.adaptive_dynmax_final_trigger = max(0.0, min(float(adaptive_dynmax_final_trigger), 1.5))
        self.adaptive_dynmax_final_target = max(0.0, min(float(adaptive_dynmax_final_target), self.adaptive_dynmax_final_trigger))
        self.adaptive_dynmax_final_steps_ahead = max(1, int(adaptive_dynmax_final_steps_ahead))
        self.adaptive_dynmax_final_max_actions = max(1, int(adaptive_dynmax_final_max_actions))
        self.enable_proactive_eviction = bool(enable_proactive_eviction)
        self.proactive_steps_ahead = max(1, int(proactive_steps_ahead))
        self.proactive_trigger = max(0.0, min(float(proactive_trigger), 1.5))
        self.proactive_target = max(0.0, min(float(proactive_target), self.proactive_trigger))
        self.proactive_max_actions = max(1, int(proactive_max_actions))

        self.harp_token_time_ns_ema = 2_000_000.0
        self.harp_prefetch_last_time_ns = 0
        self.eviction_policy = create_policy(
            self.kv_eviction_policy,
            evicpress_alpha=self.evicpress_alpha,
            evicpress_ratios=self.evicpress_ratios,
            harp_grace_candidates=self.harp_grace_candidates,
            harp_ratios=self.harp_ratios,
            harp_lambda_stall=self.harp_lambda_stall,
            harp_lambda_quality=self.harp_lambda_quality,
            harp_lambda_fairness=self.harp_lambda_fairness,
            harp_fairness_epsilon=self.harp_fairness_epsilon,
            harp_compression_profile=self.harp_compression_profile,
            harp_compression_trace=self.harp_compression_trace,
            adaptive_schedule=self.adaptive_dynmax_schedule,
            adaptive_progress_start=self.adaptive_dynmax_progress_start,
            adaptive_progress_end=self.adaptive_dynmax_progress_end,
            adaptive_final_trigger=self.adaptive_dynmax_final_trigger,
            adaptive_final_target=self.adaptive_dynmax_final_target,
            adaptive_final_steps_ahead=self.adaptive_dynmax_final_steps_ahead,
            adaptive_final_max_actions=self.adaptive_dynmax_final_max_actions,
        )
        # lists are sorted in arrival time manner
        self.request = [] # list of requests
        self.inflight = [] # list of batches
        self.done = [] # list of requests
        # self.req_ids = -1
        self.batch_ids = -1
        self.first_arrival_time = 0

        # memory model
        self.memory = MemoryModel(model, instance_id, node_id, npu_num, npu_group, npu_mem, cpu_mem, block_size, fp, enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage, cxl_mem)

        # --- Tier transition counters for tiered KV cache analysis ---
        self.tier_stats = {
            'evict_npu_to_cpu_bytes': 0,
            'evict_npu_to_cpu_count': 0,
            'evict_npu_to_cxl_bytes': 0,
            'evict_npu_to_cxl_count': 0,
            'load_cpu_to_npu_bytes': 0,
            'load_cpu_to_npu_count': 0,
            'load_cxl_to_npu_bytes': 0,
            'load_cxl_to_npu_count': 0,
            'evict_npu_prefix_bytes': 0,
            'evict_npu_prefix_count': 0,
            'evict_storage_prefix_bytes': 0,
            'evict_storage_prefix_count': 0,
            'prefix_load_storage_to_npu_bytes': 0,
            'prefix_load_storage_to_npu_count': 0,
            'storage_cache_evicted_req_bytes': 0,
            'storage_cache_evicted_req_count': 0,
            'evicpress_compression_events': 0,
            'evicpress_compressed_bytes_saved': 0,
            'evicpress_ratio_sum': 0.0,
            'harp_prefetch_bytes_total': 0,
            'harp_prefetch_bytes_progress': 0,
            'harp_prefetch_overlap_bytes': 0,
            'harp_stall_time_ns': 0,
            'harp_stall_events': 0,
            'harp_shadow_hit_tokens': 0,
            'harp_decode_tokens_total': 0,
            'harp_shadow_ratio_sum': 0.0,
            'harp_shadow_ratio_events': 0,
            'harp_shadow_eviction_events': 0,
            'harp_cold_eviction_events': 0,
            'proactive_trigger_count': 0,
            'proactive_evict_events': 0,
            'proactive_evict_bytes': 0,
            'adaptive_early_checks': 0,
            'adaptive_transition_checks': 0,
            'adaptive_late_checks': 0,
        }

        # logger
        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)
    
 
    def schedule(self, current, sys, batch_id=-1):
        if self.enable_prefix_caching:
            return self.schedule_with_prefix(current, sys, batch_id)
        else:
            return self.schedule_base(current, sys, batch_id)

    def _is_harp_enabled(self):
        return self.kv_eviction_policy in ("harp", "dynmax", "adaptive_dynmax")

    def _is_proactive_enabled(self):
        return self.enable_proactive_eviction and self._is_harp_enabled() and not self.enable_prefix_caching

    def _get_proactive_settings(self):
        settings = {
            'phase': 'fixed',
            'progress': None,
            'trigger': self.proactive_trigger,
            'target': self.proactive_target,
            'steps_ahead': self.proactive_steps_ahead,
            'max_actions': self.proactive_max_actions,
        }

        if hasattr(self.eviction_policy, 'get_adaptive_proactive_settings'):
            adaptive_settings = self.eviction_policy.get_adaptive_proactive_settings(self)
            if isinstance(adaptive_settings, dict):
                settings.update(adaptive_settings)

        settings['trigger'] = max(0.0, min(float(settings.get('trigger', self.proactive_trigger)), 1.5))
        settings['target'] = max(0.0, min(float(settings.get('target', self.proactive_target)), settings['trigger']))
        settings['steps_ahead'] = max(1, int(settings.get('steps_ahead', self.proactive_steps_ahead)))
        settings['max_actions'] = max(1, int(settings.get('max_actions', self.proactive_max_actions)))
        settings['phase'] = str(settings.get('phase', 'fixed'))
        return settings

    def _collect_active_requests(self):
        active = []
        seen = set()
        for req in self.request:
            if req.id in seen:
                continue
            seen.add(req.id)
            active.append(req)
        for batch in self.inflight:
            for req in batch.requests:
                if req.id in seen:
                    continue
                seen.add(req.id)
                active.append(req)
        return active

    def _collect_active_decode_requests(self):
        return [req for req in self._collect_active_requests() if not req.is_init]

    def _projected_memory_usage(self, steps_ahead=None):
        if steps_ahead is None:
            steps_ahead = self.proactive_steps_ahead
        steps_ahead = max(1, int(steps_ahead))

        current_used = max(0, int(self.memory.npu_used))
        bytes_per_token = max(1, int(self.memory.get_kv(1)))
        active_decode = len(self._collect_active_decode_requests())
        projected_growth = active_decode * bytes_per_token * steps_ahead
        return current_used + projected_growth

    def _memory_pressure_ratio(self, steps_ahead=None):
        projected = float(self._projected_memory_usage(steps_ahead=steps_ahead))
        total = max(1.0, float(self.memory.npu_mem))
        return projected / total

    def _maybe_proactive_evict(self, current):
        del current
        if not self._is_proactive_enabled():
            return

        if not self._collect_active_decode_requests():
            return

        settings = self._get_proactive_settings()
        phase_key = f"adaptive_{settings['phase']}_checks"
        if phase_key in self.tier_stats:
            self.tier_stats[phase_key] += 1
        if phase_key not in self.tier_stats:
            phase_key = None

        pressure = self._memory_pressure_ratio(steps_ahead=settings['steps_ahead'])
        if pressure < settings['trigger']:
            return

        self.tier_stats['proactive_trigger_count'] += 1
        gen_req = self._collect_active_decode_requests()
        evict_pool = self._build_evict_pool(gen_req)

        actions = 0
        while actions < settings['max_actions']:
            if self._memory_pressure_ratio(steps_ahead=settings['steps_ahead']) <= settings['target']:
                break

            evict_action = self.eviction_policy.select_action(evict_pool, self)
            if evict_action is None:
                break

            evicted_req = evict_action.req
            cur_evict_size = int(evict_action.raw_bytes)
            if cur_evict_size <= 0:
                evict_pool = [req for req in evict_pool if req.id != evicted_req.id]
                continue

            (
                evicted_req,
                cur_evict_size,
                _stored_evict_size,
                _evict_target_device,
                _compression_ratio,
            ) = self._harp_apply_eviction_action(evict_action)
            self._harp_update_fairness_after_eviction(evict_pool, evicted_req)

            self.tier_stats['proactive_evict_events'] += 1
            self.tier_stats['proactive_evict_bytes'] += cur_evict_size
            evict_pool = [req for req in evict_pool if req.id != evicted_req.id]
            actions += 1

    def _harp_decay_fairness_debt(self):
        if not self._is_harp_enabled():
            return
        for req in self._collect_active_requests():
            req.harp_fairness_debt = max(0.0, float(getattr(req, 'harp_fairness_debt', 0.0)) * 0.995)

    def _harp_prefetch_eta_ns(self, req):
        device = getattr(req, 'harp_prefetch_device', None)
        if device == Device.CPU:
            bw_bytes_per_ns = self.cpu_tier_bw
        else:
            bw_bytes_per_ns = self.external_tier_bw
        bw_bytes_per_ns = max(1e-9, float(bw_bytes_per_ns))
        latency_ns = max(0.0, float(getattr(req, 'harp_prefetch_latency_ns_remaining', 0.0)))
        remaining_bytes = max(0.0, float(getattr(req, 'harp_prefetch_remaining_bytes', 0.0)))
        transfer_ns = remaining_bytes / bw_bytes_per_ns
        return int(math.ceil(latency_ns + transfer_ns))

    def _harp_progress_prefetch(self, current):
        if not self._is_harp_enabled():
            return

        if self.harp_prefetch_last_time_ns <= 0:
            self.harp_prefetch_last_time_ns = int(current)
            return

        delta_ns = max(0, int(current) - int(self.harp_prefetch_last_time_ns))
        self.harp_prefetch_last_time_ns = int(current)
        if delta_ns <= 0:
            return

        for req in self._collect_active_requests():
            remaining = max(0.0, float(getattr(req, 'harp_prefetch_remaining_bytes', 0.0)))
            if remaining <= 0:
                continue

            device = getattr(req, 'harp_prefetch_device', None)
            if device not in (Device.CPU, Device.CXL):
                continue

            overlap_active = int(getattr(req, 'harp_grace_tokens_remaining', 0)) > 0
            latency_rem = max(0.0, float(getattr(req, 'harp_prefetch_latency_ns_remaining', 0.0)))
            data_ns = float(delta_ns)
            if latency_rem > 0:
                consumed = min(latency_rem, data_ns)
                req.harp_prefetch_latency_ns_remaining = latency_rem - consumed
                data_ns -= consumed

            if data_ns <= 0:
                continue

            bw_bytes_per_ns = self.cpu_tier_bw if device == Device.CPU else self.external_tier_bw
            bw_bytes_per_ns = max(1e-9, float(bw_bytes_per_ns))
            moved = min(remaining, data_ns * bw_bytes_per_ns)
            if moved <= 0:
                continue
            storage_remaining = max(0, int(getattr(req, 'harp_storage_bytes_remaining', 0)))
            moved_int = max(0, min(storage_remaining, int(round(moved))))
            if moved_int == 0 and storage_remaining > 0:
                moved_int = 1
            req.harp_prefetch_remaining_bytes = max(0.0, remaining - moved_int)
            if moved_int > 0:
                self.memory.free(moved_int, device)
                req.harp_storage_bytes_remaining = storage_remaining - moved_int

                if device == Device.CPU:
                    req.load_cpu_to_npu_bytes += moved_int
                    self.tier_stats['load_cpu_to_npu_bytes'] += moved_int
                    self.tier_stats['load_cpu_to_npu_count'] += 1
                else:
                    req.load_cxl_to_npu_bytes += moved_int
                    self.tier_stats['load_cxl_to_npu_bytes'] += moved_int
                    self.tier_stats['load_cxl_to_npu_count'] += 1

                self.tier_stats['harp_prefetch_bytes_progress'] += moved_int
                if overlap_active:
                    self.tier_stats['harp_prefetch_overlap_bytes'] += moved_int

            if req.harp_prefetch_remaining_bytes <= 1e-9:
                req.harp_prefetch_remaining_bytes = 0.0
                req.harp_prefetch_latency_ns_remaining = 0.0
                req.harp_stall_active = False

    def _harp_filter_schedulable(self, ready_reqs):
        if not self._is_harp_enabled():
            return ready_reqs

        schedulable = []
        for req in ready_reqs:
            if req.is_init:
                schedulable.append(req)
                continue

            if req.harp_prefetch_remaining_bytes <= 0 or req.harp_grace_tokens_remaining > 0:
                req.harp_stall_active = False
                schedulable.append(req)
                continue

            if not req.harp_stall_active:
                req.harp_stall_active = True
                eta_ns = self._harp_prefetch_eta_ns(req)
                req.harp_stall_events += 1
                req.harp_stall_time_ns += eta_ns
                self.tier_stats['harp_stall_events'] += 1
                self.tier_stats['harp_stall_time_ns'] += eta_ns

        return schedulable

    def _harp_on_decode_token(self, req, itl_ns):
        if not self._is_harp_enabled() or req.is_init:
            return

        req.harp_decode_tokens += 1
        self.tier_stats['harp_decode_tokens_total'] += 1

        itl_ns = max(1.0, float(itl_ns))
        self.harp_token_time_ns_ema = 0.9 * self.harp_token_time_ns_ema + 0.1 * itl_ns

        if req.harp_state == "shadow":
            req.harp_shadow_hit_tokens += 1
            self.tier_stats['harp_shadow_hit_tokens'] += 1
            if req.harp_grace_tokens_remaining > 0:
                req.harp_grace_tokens_remaining -= 1
            if req.harp_grace_tokens_remaining <= 0:
                req.harp_grace_tokens_remaining = 0
                req.harp_state = "cold"
                req.harp_grace_tail_bytes = 0

    def _harp_update_fairness_after_eviction(self, evict_pool, evicted_req):
        if not self._is_harp_enabled():
            return

        for req in evict_pool:
            if req.id == evicted_req.id:
                req.harp_fairness_debt = max(0.0, float(req.harp_fairness_debt) * 0.5)
            else:
                req.harp_fairness_debt = min(1000.0, float(req.harp_fairness_debt) + 1.0)

    def _harp_apply_eviction_action(self, evict_action):
        evicted_req = evict_action.req
        cur_evict_size = int(evict_action.raw_bytes)
        stored_evict_size = int(evict_action.stored_bytes)
        evict_target_device = evict_action.device
        compression_ratio = float(evict_action.ratio)
        grace_tokens = int(getattr(evict_action, 'grace_tokens', 0))
        grace_bytes = int(getattr(evict_action, 'grace_bytes', 0))
        target_state = str(getattr(evict_action, 'target_state', 'cold'))

        evicted_req.current_evict_raw_bytes = cur_evict_size
        evicted_req.current_evict_stored_bytes = stored_evict_size
        evicted_req.current_evict_compression_ratio = compression_ratio
        evicted_req.evict = False
        evicted_req.evict_raw_bytes += cur_evict_size
        evicted_req.evict_stored_bytes += stored_evict_size
        evicted_req.evict_event_count += 1
        prev_ratio_sum = evicted_req.evict_compression_ratio * (evicted_req.evict_event_count - 1)
        evicted_req.evict_compression_ratio = (prev_ratio_sum + compression_ratio) / evicted_req.evict_event_count

        self.memory.free(cur_evict_size, Device.NPU)
        self.memory.allocate(stored_evict_size, evict_target_device)
        evicted_req.evict_device = evict_target_device

        if evict_target_device == Device.CXL:
            evicted_req.evict_npu_to_cxl_bytes += stored_evict_size
            self.tier_stats['evict_npu_to_cxl_bytes'] += stored_evict_size
            self.tier_stats['evict_npu_to_cxl_count'] += 1
        else:
            evicted_req.evict_npu_to_cpu_bytes += stored_evict_size
            self.tier_stats['evict_npu_to_cpu_bytes'] += stored_evict_size
            self.tier_stats['evict_npu_to_cpu_count'] += 1

        saved_bytes = max(0, cur_evict_size - stored_evict_size)
        if saved_bytes > 0:
            self.tier_stats['evicpress_compression_events'] += 1
            self.tier_stats['evicpress_compressed_bytes_saved'] += saved_bytes
            self.tier_stats['evicpress_ratio_sum'] += compression_ratio

        evicted_req.harp_prefetch_device = evict_target_device
        evicted_req.harp_prefetch_remaining_bytes += float(stored_evict_size)
        evicted_req.harp_storage_bytes_remaining += stored_evict_size
        if evict_target_device == Device.CPU:
            evicted_req.harp_prefetch_latency_ns_remaining += self.cpu_tier_latency
        else:
            evicted_req.harp_prefetch_latency_ns_remaining += self.external_tier_latency
        total_bytes = max(0, int(self.memory.get_evict_kv(evicted_req)))
        evicted_req.harp_missing_raw_bytes = min(total_bytes, int(evicted_req.harp_missing_raw_bytes + cur_evict_size))
        evicted_req.harp_grace_tokens_remaining = max(0, grace_tokens)
        evicted_req.harp_grace_tail_bytes = max(0, grace_bytes)
        evicted_req.harp_shadow_ratio = compression_ratio
        evicted_req.harp_state = "shadow" if target_state == "shadow" else "cold"

        self.tier_stats['harp_prefetch_bytes_total'] += stored_evict_size
        self.tier_stats['harp_shadow_ratio_sum'] += compression_ratio
        self.tier_stats['harp_shadow_ratio_events'] += 1
        if evicted_req.harp_state == "shadow":
            self.tier_stats['harp_shadow_eviction_events'] += 1
        else:
            self.tier_stats['harp_cold_eviction_events'] += 1

        return evicted_req, cur_evict_size, stored_evict_size, evict_target_device, compression_ratio

    def get_harp_state_counts(self):
        if not self._is_harp_enabled():
            return {
                'hot': 0,
                'shadow': 0,
                'cold': 0,
                'prefetch_remaining_bytes': 0.0,
            }

        hot = 0
        shadow = 0
        cold = 0
        prefetch_remaining = 0.0
        for req in self._collect_active_requests():
            state = getattr(req, 'harp_state', 'hot')
            if state == 'shadow':
                shadow += 1
            elif state == 'cold':
                cold += 1
            else:
                hot += 1
            prefetch_remaining += max(0.0, float(getattr(req, 'harp_prefetch_remaining_bytes', 0.0)))
        return {
            'hot': hot,
            'shadow': shadow,
            'cold': cold,
            'prefetch_remaining_bytes': prefetch_remaining,
        }

    # batch the request scheduling method
    def schedule_base(self, current, sys, batch_id=-1):
        # first NPU to process new batch
        if sys == self.start_npu:
            if self._is_harp_enabled():
                self._harp_progress_prefetch(current)
                self._harp_decay_fairness_debt()
                self._maybe_proactive_evict(current)

            # nothing to batch return None
            if len(self.request) != 0 and self.request[0].arrival > current:
                return None
            # constraint of inflight batches considering parallelism
            if len(self.inflight) >= self.npu_group:
                # wait it to be done
                return None

            # scheduling start
            batch_req = [req for req in self.request if req.arrival <= current]
            if self._is_harp_enabled():
                batch_req = self._harp_filter_schedulable(batch_req)
            batch_len = len(batch_req) if len(batch_req) <= self.max_batch else self.max_batch

            # nothing to batch
            if batch_len == 0:
                return None

            # can make batch and proceed
            batch_req = batch_req[:batch_len]

            kv_size = 0
            evict_size = 0
            evict_cxl_size = 0
            gen_req = [req for req in batch_req if not req.is_init]
            evict_pool = self._build_evict_pool(gen_req)
            
            if self.prioritize_prefill:
                prefill_req = [req for req in batch_req if req.is_init]

                if len(prefill_req) != 0:
                    batch_req = prefill_req
                    batch_len = len(batch_req) if len(batch_req) <= self.max_batch else self.max_batch
                    batch_req = batch_req[:batch_len]
            
            # check if there is request that need to enlarge the block
            temp_len = batch_len
            for i in range(batch_len, -1, -1):
                kv_size = self.memory.get_block_kv(batch_req, i) # includes evicted input, and initiation input
                if self.memory.is_avail(kv_size, Device.NPU):
                    temp_len = i
                    break
            
            # no memory to batch
            while temp_len == 0:
                # preempt request one by one untill there is enough space
                while evict_pool and evict_pool[-1].evict:
                    evict_pool.pop()

                if len(evict_pool) == 0:
                    return None

                evicted_req = None
                cur_evict_size = 0
                stored_evict_size = 0
                evict_target_device = Device.CPU
                compression_ratio = 1.0

                evict_action = self.eviction_policy.select_action(evict_pool, self)
                if evict_action is None:
                    return None

                evicted_req = evict_action.req
                cur_evict_size = int(evict_action.raw_bytes)
                stored_evict_size = int(evict_action.stored_bytes)
                evict_target_device = evict_action.device
                compression_ratio = float(evict_action.ratio)
                evicted_req.evicpress_utility_score = float(getattr(evict_action, 'utility', 0.0))
                self.logger.info("Eviction of the request #%d", evicted_req.id)

                if self._is_harp_enabled():
                    if cur_evict_size <= 0:
                        evict_pool = [req for req in evict_pool if req.id != evicted_req.id]
                        continue

                    (
                        evicted_req,
                        cur_evict_size,
                        stored_evict_size,
                        evict_target_device,
                        compression_ratio,
                    ) = self._harp_apply_eviction_action(evict_action)
                    self._harp_update_fairness_after_eviction(evict_pool, evicted_req)
                    if evict_target_device == Device.CXL:
                        evict_cxl_size += stored_evict_size
                    else:
                        evict_size += stored_evict_size
                else:
                    evict_pool = [req for req in evict_pool if req.id != evicted_req.id]

                    evicted_req.evict = True
                    evicted_req.current_evict_raw_bytes = cur_evict_size
                    evicted_req.current_evict_stored_bytes = stored_evict_size
                    evicted_req.current_evict_compression_ratio = compression_ratio
                    evicted_req.evict_raw_bytes += cur_evict_size
                    evicted_req.evict_stored_bytes += stored_evict_size
                    evicted_req.evict_event_count += 1
                    prev_ratio_sum = evicted_req.evict_compression_ratio * (evicted_req.evict_event_count - 1)
                    evicted_req.evict_compression_ratio = (prev_ratio_sum + compression_ratio) / evicted_req.evict_event_count
                    # spill to CXL first, fall back to CPU
                    self.memory.free(cur_evict_size, Device.NPU)
                    if evict_target_device == Device.CXL:
                        self.memory.allocate(stored_evict_size, Device.CXL)
                        evicted_req.evict_device = Device.CXL
                        evicted_req.evict_npu_to_cxl_bytes += stored_evict_size
                        evict_cxl_size += stored_evict_size
                        self.tier_stats['evict_npu_to_cxl_bytes'] += stored_evict_size
                        self.tier_stats['evict_npu_to_cxl_count'] += 1
                        self.logger.info(
                            "Evicted to %s (%d bytes, compression ratio %.2f)",
                            self.external_tier_name,
                            stored_evict_size,
                            compression_ratio,
                        )
                    else:
                        self.memory.allocate(stored_evict_size, Device.CPU)
                        evicted_req.evict_device = Device.CPU
                        evicted_req.evict_npu_to_cpu_bytes += stored_evict_size
                        evict_size += stored_evict_size
                        self.tier_stats['evict_npu_to_cpu_bytes'] += stored_evict_size
                        self.tier_stats['evict_npu_to_cpu_count'] += 1
                        self.logger.info(
                            "Evicted to CPU (%d bytes, compression ratio %.2f)",
                            stored_evict_size,
                            compression_ratio,
                        )

                    saved_bytes = max(0, cur_evict_size - stored_evict_size)
                    if saved_bytes > 0:
                        self.tier_stats['evicpress_compression_events'] += 1
                        self.tier_stats['evicpress_compressed_bytes_saved'] += saved_bytes
                        self.tier_stats['evicpress_ratio_sum'] += compression_ratio

                if len(evict_pool) < batch_len:
                    batch_len = len(evict_pool)

                # check if can batch
                for i in range(batch_len, -1, -1):
                    kv_size = self.memory.get_block_kv(batch_req, i)
                    if self.memory.is_avail(kv_size, Device.NPU):
                        temp_len = i
                        break

            batch_len = temp_len
            batch_req = batch_req[:batch_len]
            load_size = 0
            load_cxl_size = 0

            # check max_num_batched_tokens constraint
            total_len = 0
            for req in batch_req:
                if req.is_init:
                    total_len += req.input
                else:
                    total_len += 1

            while total_len > self.max_num_batched_tokens:
                if batch_req[-1].is_init:
                    total_len -= batch_req[-1].input
                else:
                    total_len -= 1
                
                batch_req = batch_req[:-1]
                batch_len -= 1

            # recompute kv_size
            kv_size = self.memory.get_block_kv(batch_req, batch_len) # includes evicted input, and initiation input

            # delete from request queue
            for req in batch_req:
                for i, req_ in enumerate(self.request):
                    if req_.id == req.id:
                        del self.request[i]
                        break

                if req.evict:
                    # load evicted kv cache from whichever tier it was evicted to
                    raw_kv_bytes = self.memory.get_evict_kv(req)
                    kv_bytes = getattr(req, 'current_evict_stored_bytes', raw_kv_bytes) or raw_kv_bytes
                    if req.evict_device == Device.CXL:
                        load_cxl_size += kv_bytes
                        req.load_cxl_to_npu_bytes += kv_bytes
                        req.last_kv_load_tier = self.external_tier_name
                    elif req.evict_device == Device.CPU:
                        load_size += kv_bytes
                        req.load_cpu_to_npu_bytes += kv_bytes
                        req.last_kv_load_tier = "CPU"
                    else:
                        # Defensive fallback: route unknown eviction source through CPU path.
                        load_size += kv_bytes
                        req.load_cpu_to_npu_bytes += kv_bytes
                        req.last_kv_load_tier = "CPU"
                        self.logger.warning(
                            "Request #%d had unknown evict_device; defaulted reload to CPU",
                            req.id,
                        )
                    req.evict = False
                    req.evict_device = None
                    req.current_evict_raw_bytes = 0
                    req.current_evict_stored_bytes = 0
                    req.current_evict_compression_ratio = 1.0
                    self.logger.info("Loading the request #%d", req.id)

            # Allocate Needed KV caches for current batch
            if kv_size > 0:
                self.memory.allocate(kv_size, Device.NPU)
            
            # load memory from CXL
            if load_cxl_size > 0:
                self.memory.free(load_cxl_size, Device.CXL)
                self.tier_stats['load_cxl_to_npu_bytes'] += load_cxl_size
                self.tier_stats['load_cxl_to_npu_count'] += 1

            # load memory from cpu (host)
            if load_size > 0:
                self.memory.free(load_size, Device.CPU)
                self.tier_stats['load_cpu_to_npu_bytes'] += load_size
                self.tier_stats['load_cpu_to_npu_count'] += 1
            
            total_len = 0
            kv_len = 0
            hit_len = 0
            num_prefill = 0
            num_decode = 0
            q_list = []
            k_list = []
            prefill_q_list = []
            prefill_k_list = []
            decode_k_list = []
            for req in batch_req:
                if req.is_init:
                    total_len += req.input
                    req.set_que_delay(current)
                    q_list.append(req.input)
                    prefill_q_list.append(req.input)
                    # For now, we don't assume chunked prefill
                    prefill_k_list.append(0)
                    num_prefill += 1
                else:
                    # Online access-frequency proxy used by EVICPRESS.
                    req.evicpress_access_count = int(getattr(req, 'evicpress_access_count', 0)) + 1
                    total_len += 1
                    q_list.append(1)
                    num_decode += 1
                    kv_len += req.input
                    decode_k_list.append(req.input)
                k_list.append(req.input)

            # make batch, output doesn't matter here!! always one iteration
            # batch is also 1
            batch = Batch(self.get_batch_id(), self.model, total_len, kv_len, hit_len, q_list, k_list, num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list, current, kv_size, evict_size, load_size, evict_cxl_size, load_cxl_size)
            # add alredy fired system
            batch.fired.append(sys)
            batch.requests.extend(batch_req)
            self.inflight.append(batch)
            self.logger.info(
                "Scheduling new batch #%d to NPU[%d]",
                batch.batch_id,
                sys,
            )
            return batch
        
        # Schedule already batched request
        else:
            if len(self.inflight) == 0:
                return None
            else:
                batch = None
                # find batch
                for b in self.inflight:
                    if b.batch_id == batch_id:
                        batch = b
                if batch == None:
                    return None
                # check if this has been runned in the system
                if sys in batch.fired:
                    return None
                else:
                    batch.fired.append(sys)
                    self.logger.info(
                        "Scheduling existing batch #%d to NPU[%d]",
                        batch.batch_id,
                        sys,
                    )
                    return batch
    
    def schedule_with_prefix(self, current, sys, batch_id=-1):
        if sys == self.start_npu:
            # nothing to batch return None
            if len(self.request) != 0 and self.request[0].arrival > current:
                return None
            # constraint of inflight batches considering parallelism
            if len(self.inflight) >= self.npu_group:
                # wait it to be done
                return None

            # scheduling start
            batch_req = [req for req in self.request if req.arrival <= current]
            batch_len = len(batch_req) if len(batch_req) <= self.max_batch else self.max_batch

            # nothing to batch
            if batch_len == 0:
                return None

            # can make batch and proceed
            batch_req = batch_req[:batch_len]

            if self.prioritize_prefill:
                prefill_req = [req for req in batch_req if req.is_init]

                if len(prefill_req) != 0:
                    batch_req = prefill_req
                    batch_len = len(batch_req) if len(batch_req) <= self.max_batch else self.max_batch
                    batch_req = batch_req[:batch_len]
        
            for req in batch_req:
                if req.is_init:
                    self.memory.prefix_match(req)
                    # self.memory.npu_lock_prefix(req)
                    self.memory.lock_prefix(req, Device.NPU)
            
            kv_size = 0
            evict_size = 0
            gen_req = [req for req in batch_req if not req.is_init]
            evict_pool = self._build_evict_pool(gen_req)
            # check if there is request that need to enlarge the block
            temp_len = batch_len
            total_useable_size = self.memory.avail_size(Device.NPU) + self.memory.evictable_size(Device.NPU)
            
            for i in range(batch_len, -1, -1):
                kv_size = self.memory.get_block_kv(batch_req, i) # includes evicted input, and initiation input
                if total_useable_size >= kv_size:
                    temp_len = i
                    break
            
            evicted_req = []
            # no memory to batch
            while temp_len == 0:
                while evict_pool and evict_pool[-1].evict:
                    evict_pool.pop()

                if len(evict_pool) == 0: # there is no request to evict but no memory
                    # rollback prefix cache lock ref
                    for req in batch_req:
                        if req.is_init:
                            # self.memory.npu_unlock_prefix(req)
                            self.memory.unlock_prefix(req, Device.NPU)
                            self.memory.erase_prefix_info(req)
                    return None
                
                # else
                # self.memory.npu_unlock_prefix(gen_req[-1])
                evict_candidate = evict_pool.pop()
                self.memory.unlock_prefix(evict_candidate, Device.NPU)
                self.memory.erase_prefix_info(evict_candidate)

                current_usable_size = self.memory.avail_size(Device.NPU) + self.memory.evictable_size(Device.NPU)

                evict_candidate.evict = True
                evicted_req.append(evict_candidate)
                self.logger.info("Eviction of the request #%d", evict_candidate.id)

                if len(evict_pool) < batch_len: # prefill is always at last
                    batch_len = len(evict_pool)
                
                # check if can batch
                for i in range(batch_len, -1, -1):
                    kv_size = self.memory.get_block_kv(batch_req, i)
                    if current_usable_size >= kv_size:
                        temp_len = i
                        break

            for req in batch_req[temp_len:]:
                if req.is_init:
                    # self.memory.npu_unlock_prefix(req)
                    self.memory.unlock_prefix(req, Device.NPU)
                    self.memory.erase_prefix_info(req)

            batch_len = temp_len
            batch_req = batch_req[:batch_len]

            # check max_num_batched_tokens constraint
            total_len = 0
            for req in batch_req:
                if req.is_init:
                    total_len += req.input
                else:
                    total_len += 1

            while total_len > self.max_num_batched_tokens:
                if batch_req[-1].is_init:
                    total_len -= batch_req[-1].input
                else:
                    total_len -= 1
                
                if batch_req[-1].is_init:
                    # self.memory.npu_unlock_prefix(batch_req[-1])
                    self.memory.unlock_prefix(batch_req[-1], Device.NPU)
                    self.memory.erase_prefix_info(batch_req[-1])

                batch_req = batch_req[:-1]
                batch_len -= 1

            # recompute kv_size
            kv_size = self.memory.get_block_kv(batch_req, batch_len) # includes evicted input, and initiation input
            evict_size = (kv_size - self.memory.avail_size(Device.NPU)) if kv_size > self.memory.avail_size(Device.NPU) else 0

            if evict_size > 0:
                # self.memory.npu_evict_prefix_cache(evict_size)
                self.memory.evict_prefix_cache(evict_size, Device.NPU)
                self.tier_stats['evict_npu_prefix_bytes'] += evict_size
                self.tier_stats['evict_npu_prefix_count'] += 1

            evict_load_size = 0
            prefix_load_size = 0
            for req in batch_req:
                for i, req_ in enumerate(self.request):
                    if req_.id == req.id:
                        del self.request[i]
                        break

                if req.is_init and req.storage_cache_hit > req.prefix_cache_hit:
                    # load prefix cache
                    _pls = (req.storage_cache_hit - req.prefix_cache_hit) * self.memory.get_kv(1)
                    prefix_load_size += _pls
                    self.tier_stats['prefix_load_storage_to_npu_bytes'] += _pls
                    self.tier_stats['prefix_load_storage_to_npu_count'] += 1

                if req.evict:
                    # load evicted kv cache
                    self.memory.prefix_match(req)
                    # self.memory.npu_lock_prefix(req)
                    self.memory.lock_prefix(req, Device.NPU)
                    # self.memory.cpu_unlock_prefix(req)
                    if self.prefix_storage is not None:
                        self.memory.unlock_prefix(req, Device.CPU)
                        if self.prefix_storage == Device.CXL:
                            req.last_kv_load_tier = self.external_tier_name
                        elif self.prefix_storage == Device.CPU:
                            req.last_kv_load_tier = "CPU"
                    evict_load_size += self.memory.get_evict_kv(req)
                    req.evict = False
                    self.logger.info("Loading the request #%d", req.id)

            total_len = 0
            kv_len = 0
            hit_len = 0
            num_prefill = 0
            num_decode = 0
            q_list = []
            k_list = []
            prefill_q_list = []
            prefill_k_list = []
            decode_k_list = []
            
            # evict cpu prefix cache if needed
            total_size = 0
            for req in batch_req:
                total_size += self.memory.get_total_kv(req) * self.npu_num
            for req in evicted_req:
                total_size += self.memory.get_total_kv(req) * self.npu_num
            
            if self.prefix_storage is not None:
                storage_evict_size = (total_size - self.memory.avail_size(self.prefix_storage)) if total_size > self.memory.avail_size(self.prefix_storage) else 0
                
                if storage_evict_size > 0:
                    # self.memory.cpu_evict_prefix_cache(cpu_evict_size)
                    self.memory.evict_prefix_cache(storage_evict_size, self.prefix_storage)
                    self.tier_stats['evict_storage_prefix_bytes'] += storage_evict_size
                    self.tier_stats['evict_storage_prefix_count'] += 1

            for req in batch_req:
                # Update the prefix cache for incoming batch
                self.memory.cache_unfinished_req(req, Device.NPU)
                if self.prefix_storage is not None:
                    self.memory.cache_unfinished_req(req, self.prefix_storage)
                if req.is_init:
                    total_len += req.input
                    req.set_que_delay(current)
                    if self.enable_prefix_caching and req.prefix_cache_hit > 0:
                        hit_len += req.prefix_cache_hit
                    q_list.append(max(req.input - req.prefix_cache_hit, 1))
                    num_prefill += 1
                    prefill_q_list.append(max(req.input - req.prefix_cache_hit, 1))
                    prefill_k_list.append(0)
                else:
                    total_len += 1    
                    q_list.append(1)
                    num_decode += 1
                    kv_len += req.input
                    decode_k_list.append(req.input)
                k_list.append(req.input)
            
            # cpu need to hold evicted cache
            if self.prefix_storage is not None:
                for req in evicted_req:
                    _evict_kv = self.memory.get_total_kv(req) * self.npu_num
                    self.memory.storage_cache_evicted_req(req)
                    self.tier_stats['storage_cache_evicted_req_bytes'] += _evict_kv
                    self.tier_stats['storage_cache_evicted_req_count'] += 1

            
            # For debugging
            # self.memory.npu_prefix_cache.pretty_print()
            # self.memory.npu_prefix_cache.print_prefix_info()
            batch = Batch(self.get_batch_id(), self.model, total_len, kv_len, hit_len, q_list, k_list, num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list, current, kv_size, evict_size, evict_load_size + prefix_load_size)
            # add alredy fired system
            batch.fired.append(sys)
            batch.requests.extend(batch_req)
            self.inflight.append(batch)
            self.logger.info(
                "Scheduling new batch #%d to NPU[%d]",
                batch.batch_id,
                sys,
            )
            return batch
        # Schedule already batched request
        else:
            if len(self.inflight) == 0:
                return None
            else:
                batch = None
                # find batch
                for b in self.inflight:
                    if b.batch_id == batch_id:
                        batch = b
                if batch is None or sys in batch.fired:
                    return None
                else:
                    batch.fired.append(sys)
                    self.logger.info(
                        "Scheduling existing batch #%d to NPU[%d]",
                        batch.batch_id,
                        sys,
                    )
                    return batch
        
    # pop inflight, add to done
    def add_done(self, id, sys, finish):
        prompt_t = 0
        gen_t = 0
        end_reqs = []
        if len(self.inflight) == 0:
            return prompt_t, gen_t, end_reqs
        batch = None
        # find batch
        id -= 1
        idx = 0
        for i, b in enumerate(self.inflight):
            if b.batch_id == id:
                batch = b
                idx = i
        # no batch return
        if batch == None:
            return prompt_t, gen_t, end_reqs
        # already done
        if sys in batch.end:
            return prompt_t, gen_t, end_reqs
        else:
            # add to done system
            batch.end.append(sys)
            # check all npus are done
            if self.pd_type != "prefill":
                if self.start_npu not in batch.end or (self.start_npu + self.npu_num - 1) not in batch.end:
                    return prompt_t, gen_t, end_reqs
            else:
                if self.start_npu not in batch.end or (self.start_npu + self.npu_num * 2 - 1) not in batch.end:
                    return prompt_t, gen_t, end_reqs
        self.logger.info(
            "Batch #%d is done",
            batch.batch_id,
        )
                
        pool = []
        for req in batch.requests:
            # change phase
            if req.is_init:
                req.is_init = False
                if self.pd_type != "prefill":
                    prompt_t += req.input
                    gen_t += 1
                    req.set_ttft(finish)
                else: # prefill instance
                    prompt_t += req.input
                    gen_t += 1
                    req.set_ttft(finish)
                    self.logger.info(
                    "Request #%d is prefill done",
                    req.id,
                    )
                        
                    # sending is done. clean this batch in prefill instance
                    self.logger.info("Request #%d is sent to decode instance", req.id)
                    req.input += 1
                    
                    # remove kv cache here
                    if self.enable_prefix_caching:
                        self.memory.unlock_prefix(req, Device.NPU)
                    else:
                        kv_size = self.memory.get_evict_kv(req)
                        self.memory.free(kv_size, Device.NPU)

                    end_reqs.append(req)
                    continue # pass generation phase and continue
            else:
                gen_t += 1
                req.add_itl(finish)
                if req.itl:
                    self._harp_on_decode_token(req, req.itl[-1])

            req.input += 1

            # check done
            if req.output <= req.input:
                self.logger.info("Request #%d is done", req.id)
                # remove kv cache here
                if self.enable_prefix_caching:
                    self.memory.cache_finished_req(req, Device.NPU) # insert happens here
                    if self.prefix_storage is not None:
                        self.memory.cache_finished_req(req, Device.CPU)
                else:
                    if self._is_harp_enabled():
                        kv_size = self.memory.get_evict_kv(req)
                        missing_raw = max(0, int(getattr(req, 'harp_missing_raw_bytes', 0)))
                        resident_kv = max(0, kv_size - missing_raw)
                        if resident_kv > 0:
                            self.memory.free(resident_kv, Device.NPU)

                        remaining_storage = max(0, int(getattr(req, 'harp_storage_bytes_remaining', 0)))
                        if remaining_storage > 0 and req.harp_prefetch_device in (Device.CPU, Device.CXL):
                            self.memory.free(remaining_storage, req.harp_prefetch_device)
                            req.harp_storage_bytes_remaining = 0
                            req.harp_prefetch_remaining_bytes = 0.0
                            req.harp_prefetch_latency_ns_remaining = 0.0
                    else:
                        kv_size = self.memory.get_evict_kv(req)
                        self.memory.free(kv_size, Device.NPU)
                req.add_latency(finish)
                self.done.append(req)
                end_reqs.append(req)

            # return to pool
            else:
                pool.append(req)
        # return to request pool, both are already sorted with arrival_time
        if self.prioritize_prefill:
            self.request = self._merge_by_arrival_id(pool, self.request)
        else:
            self.request = pool + self.request

        del self.inflight[idx]
        del batch
        return prompt_t, gen_t, end_reqs
    

    ##### Helper Functions ######
    # get new batch id
    def get_batch_id(self):
        self.batch_ids += 1
        return self.batch_ids

    def _build_evict_pool(self, gen_req):
        return self.eviction_policy.build_pool(gen_req, self)

    # add a request
    def add_request(self, req, is_init=True):
        new_req = Request(*(req), is_init=is_init)
        self.request.append(new_req)
        return
    
    # add decode request to decode instance from prefill instnace
    def add_decode(self, req):
        self.request.append(req)
        kv_size = self.memory.get_total_kv(req)
        self.memory.allocate(kv_size, Device.NPU)
    
    # get first request's arrival time
    def get_first_arrival_time(self):
        return self.first_arrival_time if self.first_arrival_time != 0 else 1 # need to add event handler at first
    
    # merge requests in the request pool, ensuring they are sorted by arrival time
    def _merge_by_arrival_id(self, left, right):
        if not left:  
            return right
        if not right: 
            return left

        # Fast path: if ranges don't overlap, just concatenate
        if (left[-1].arrival, left[-1].id) <= (right[0].arrival, right[0].id):
            return left + right
        if (right[-1].arrival, right[-1].id) <= (left[0].arrival, left[0].id):
            return right + left

        # General merge
        i = j = 0
        out = []
        while i < len(left) and j < len(right):
            li, rj = left[i], right[j]
            if (li.arrival, li.id) <= (rj.arrival, rj.id):
                out.append(li); i += 1
            else:
                out.append(rj); j += 1
        if i < len(left):  
            out.extend(left[i:])
        if j < len(right): 
            out.extend(right[j:])
        return out
    
    # print total system request metrics (TTFT, TPOT, ITL)
    def print_result(self):
        # Extract ttft, tpot, and itl values from the completed requests
        ttft_values = [req.ttft for req in self.done]
        tpot_values = [req.tpot for req in self.done]
        itl_values = [itl for req in self.done for itl in req.itl]

        print("------------------------------Time to First Token-------------------------------")
        if ttft_values:
            mean = np.mean(ttft_values) / 1000_000
            median = np.median(ttft_values) / 1000_000
            p99 = np.percentile(ttft_values, 99) / 1000_000
            print(f"Mean TTFT (ms):                                                     {mean:.2f}")
            print(f"Median TTFT (ms):                                                   {median:.2f}")
            print(f"P99 TTFT (ms):                                                      {p99:.2f}")
        else:
            print("No TTFT data available")

        print("--------------------Time per Output Token (excl. 1st token)---------------------")
        if tpot_values:
            mean = np.mean(tpot_values) / 1000_000
            median = np.median(tpot_values) / 1000_000
            p99 = np.percentile(tpot_values, 99) / 1000_000
            print(f"Mean TPOT (ms):                                                     {mean:.2f}")
            print(f"Median TPOT (ms):                                                   {median:.2f}")
            print(f"P99 TPOT (ms):                                                      {p99:.2f}")
        else:
            print("No TPOT data available")

        print("------------------------------Inter-token Latency-------------------------------")
        if itl_values:
            mean = np.mean(itl_values) / 1000_000
            median = np.median(itl_values) / 1000_000
            p99 = np.percentile(itl_values, 99) / 1000_000
            print(f"Mean ITL (ms):                                                      {mean:.2f}")
            print(f"Median ITL (ms):                                                    {median:.2f}")
            print(f"P99 ITL (ms):                                                       {p99:.2f}")
        else:
            print("No ITL data available")

    # print tier transition statistics
    def print_tier_stats(self):
        MB = 1024 * 1024
        print("----------------------------KV Cache Tier Statistics----------------------------")
        print(f"NPU→CPU evictions:   {self.tier_stats['evict_npu_to_cpu_count']:>6} events, {self.tier_stats['evict_npu_to_cpu_bytes']/MB:>10.2f} MB")
        print(f"NPU→{self.external_tier_name} evictions:   {self.tier_stats['evict_npu_to_cxl_count']:>6} events, {self.tier_stats['evict_npu_to_cxl_bytes']/MB:>10.2f} MB")
        print(f"CPU→NPU reloads:     {self.tier_stats['load_cpu_to_npu_count']:>6} events, {self.tier_stats['load_cpu_to_npu_bytes']/MB:>10.2f} MB")
        print(f"{self.external_tier_name}→NPU reloads:     {self.tier_stats['load_cxl_to_npu_count']:>6} events, {self.tier_stats['load_cxl_to_npu_bytes']/MB:>10.2f} MB")
        print(f"NPU prefix evictions:{self.tier_stats['evict_npu_prefix_count']:>6} events, {self.tier_stats['evict_npu_prefix_bytes']/MB:>10.2f} MB")
        print(f"Storage prefix evict:{self.tier_stats['evict_storage_prefix_count']:>6} events, {self.tier_stats['evict_storage_prefix_bytes']/MB:>10.2f} MB")
        print(f"Storage→NPU prefix:  {self.tier_stats['prefix_load_storage_to_npu_count']:>6} events, {self.tier_stats['prefix_load_storage_to_npu_bytes']/MB:>10.2f} MB")
        print(f"Evicted→Storage:     {self.tier_stats['storage_cache_evicted_req_count']:>6} events, {self.tier_stats['storage_cache_evicted_req_bytes']/MB:>10.2f} MB")
        if self.tier_stats['evicpress_compression_events'] > 0:
            avg_ratio = self.tier_stats['evicpress_ratio_sum'] / self.tier_stats['evicpress_compression_events']
            saved_mb = self.tier_stats['evicpress_compressed_bytes_saved'] / MB
            print(f"EVICPRESS compress:  {self.tier_stats['evicpress_compression_events']:>6} events, {saved_mb:>10.2f} MB saved, avg ratio {avg_ratio:>5.2f}")
        if self.tier_stats['harp_prefetch_bytes_total'] > 0 or self.tier_stats['harp_stall_events'] > 0:
            overlap_ratio = 0.0
            if self.tier_stats['harp_prefetch_bytes_total'] > 0:
                overlap_ratio = self.tier_stats['harp_prefetch_overlap_bytes'] / self.tier_stats['harp_prefetch_bytes_total']
            shadow_hit_rate = 0.0
            if self.tier_stats['harp_decode_tokens_total'] > 0:
                shadow_hit_rate = self.tier_stats['harp_shadow_hit_tokens'] / self.tier_stats['harp_decode_tokens_total']
            avg_shadow_ratio = 1.0
            if self.tier_stats['harp_shadow_ratio_events'] > 0:
                avg_shadow_ratio = self.tier_stats['harp_shadow_ratio_sum'] / self.tier_stats['harp_shadow_ratio_events']
            print(f"HARP prefetch:       {self.tier_stats['harp_prefetch_bytes_progress']/MB:>10.2f} MB progressed, overlap ratio {overlap_ratio:>6.3f}")
            print(f"HARP stalls:         {self.tier_stats['harp_stall_events']:>6} events, {self.tier_stats['harp_stall_time_ns']/1e6:>10.2f} ms total")
            print(f"HARP shadow hits:    {self.tier_stats['harp_shadow_hit_tokens']:>6} tokens, hit rate {shadow_hit_rate:>6.3f}, avg ratio {avg_shadow_ratio:>5.2f}")
        if self.tier_stats['adaptive_early_checks'] > 0 or self.tier_stats['adaptive_transition_checks'] > 0 or self.tier_stats['adaptive_late_checks'] > 0:
            print(
                f"Adaptive phases:     early {self.tier_stats['adaptive_early_checks']:>6} checks, "
                f"transition {self.tier_stats['adaptive_transition_checks']:>6} checks, "
                f"late {self.tier_stats['adaptive_late_checks']:>6} checks"
            )
        if self.tier_stats['proactive_trigger_count'] > 0 or self.tier_stats['proactive_evict_events'] > 0:
            print(f"Proactive triggers:  {self.tier_stats['proactive_trigger_count']:>6} checks, {self.tier_stats['proactive_evict_events']:>6} evictions, {self.tier_stats['proactive_evict_bytes']/MB:>10.2f} MB")

        moved_keys = [
            'evict_npu_to_cpu_bytes',
            'evict_npu_to_cxl_bytes',
            'load_cpu_to_npu_bytes',
            'load_cxl_to_npu_bytes',
            'evict_npu_prefix_bytes',
            'evict_storage_prefix_bytes',
            'prefix_load_storage_to_npu_bytes',
            'storage_cache_evicted_req_bytes',
        ]
        total_moved = sum(self.tier_stats.get(k, 0) for k in moved_keys)
        print(f"Total data moved:                                              {total_moved/MB:>10.2f} MB")

    def get_request_tier_totals(self):
        totals = {
            'evict_npu_to_cpu_bytes': 0,
            'evict_npu_to_cxl_bytes': 0,
            'load_cpu_to_npu_bytes': 0,
            'load_cxl_to_npu_bytes': 0,
        }

        for req in self.done:
            totals['evict_npu_to_cpu_bytes'] += getattr(req, 'evict_npu_to_cpu_bytes', 0)
            totals['evict_npu_to_cxl_bytes'] += getattr(req, 'evict_npu_to_cxl_bytes', 0)
            totals['load_cpu_to_npu_bytes'] += getattr(req, 'load_cpu_to_npu_bytes', 0)
            totals['load_cxl_to_npu_bytes'] += getattr(req, 'load_cxl_to_npu_bytes', 0)

        totals['tier_transition_bytes_total'] = (
            totals['evict_npu_to_cpu_bytes'] +
            totals['evict_npu_to_cxl_bytes'] +
            totals['load_cpu_to_npu_bytes'] +
            totals['load_cxl_to_npu_bytes']
        )
        return totals

    def validate_tier_accounting(self):
        req_totals = self.get_request_tier_totals()
        keys = [
            'evict_npu_to_cpu_bytes',
            'evict_npu_to_cxl_bytes',
            'load_cpu_to_npu_bytes',
            'load_cxl_to_npu_bytes',
        ]
        deltas = {}
        ok = True
        for key in keys:
            delta = req_totals[key] - self.tier_stats.get(key, 0)
            deltas[key] = delta
            if delta != 0:
                ok = False
        return ok, req_totals, deltas

    # print each request results
    def print_request_result(self):
        # sort in id order
        self.done.sort(key=lambda x : x.id)
        for i in self.done:
            print(i)
        return

    # check all the request is done
    def is_request_empty(self):
        if len(self.request) == 0 and len(self.inflight) == 0:
            return True
        else:
            return False
        
    # save requests information to an output file
    def save_output(self, output_file, is_append=False):
        output_file = f'../{output_file}'
        mode = 'a' if is_append else 'w'
        with open(output_file, mode=mode, newline='') as file:
            # Initialize the CSV writer
            writer = csv.writer(file)
            
            # Write the column headers
            if not is_append:
                writer.writerow(['instance id', 'request id', 'model', 'input', 'output', 
                                'arrival', 'end_time', 'latency', 
                                'queuing_delay', 'TTFT', 'TPOT', 'ITL',
                                'npu_cache_hit', 'storage_cache_hit', 'prefix_cache_hit',
                                'tier_reload_source',
                                'evict_npu_to_cpu_bytes', 'evict_npu_to_cxl_bytes',
                                'load_cpu_to_npu_bytes', 'load_cxl_to_npu_bytes',
                                'evict_raw_bytes', 'evict_stored_bytes', 'evict_compression_ratio',
                                'harp_state_final', 'harp_stall_time_ns', 'harp_stall_events',
                                'harp_shadow_hit_tokens', 'harp_decode_tokens', 'harp_shadow_ratio',
                                'tier_transition_bytes_total'])
            
            # Write each request's information
            for req in self.done:
                writer.writerow([
                    req.instance_id,
                    req.id,
                    req.model,
                    req.input,
                    req.output,
                    req.arrival,
                    req.end_time,
                    req.latency,
                    req.queuing_delay,
                    req.ttft,
                    req.tpot,
                    req.itl,
                    getattr(req, 'npu_cache_hit', 0),
                    getattr(req, 'storage_cache_hit', 0),
                    getattr(req, 'prefix_cache_hit', 0),
                    getattr(req, 'last_kv_load_tier', 'NPU'),
                    getattr(req, 'evict_npu_to_cpu_bytes', 0),
                    getattr(req, 'evict_npu_to_cxl_bytes', 0),
                    getattr(req, 'load_cpu_to_npu_bytes', 0),
                    getattr(req, 'load_cxl_to_npu_bytes', 0),
                    getattr(req, 'evict_raw_bytes', 0),
                    getattr(req, 'evict_stored_bytes', 0),
                    getattr(req, 'evict_compression_ratio', 1.0),
                    getattr(req, 'harp_state', 'hot'),
                    getattr(req, 'harp_stall_time_ns', 0),
                    getattr(req, 'harp_stall_events', 0),
                    getattr(req, 'harp_shadow_hit_tokens', 0),
                    getattr(req, 'harp_decode_tokens', 0),
                    getattr(req, 'harp_shadow_ratio', 1.0),
                    (
                        getattr(req, 'evict_npu_to_cpu_bytes', 0)
                        + getattr(req, 'evict_npu_to_cxl_bytes', 0)
                        + getattr(req, 'load_cpu_to_npu_bytes', 0)
                        + getattr(req, 'load_cxl_to_npu_bytes', 0)
                    ),
                ])
