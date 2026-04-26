import os
import subprocess
import argparse
import json
from time import time
from collections import defaultdict
import csv as csv_module

from inference_serving.scheduler import *
from inference_serving.request import *
from inference_serving.utils import *
from inference_serving.controller import *
from inference_serving.memory_model import *
from inference_serving.graph_generator import *
from inference_serving.trace_generator import *
from inference_serving.pim_model import *
from inference_serving.config_builder import *
from inference_serving.router import *
from inference_serving.power_model import *
from inference_serving.logger import *
from inference_serving.eviction_policies import get_registered_policy_names
import sys as flush

from pyinstrument import Profiler


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def main():
    # ----------------------------------------------------------------------------------------------
    # LLMServingSim runs in astra-sim directory for easy path configuration
    # your relative path should start from astra-sim directory
    cwd = os.getcwd()
    astra_sim = os.path.join(cwd, "astra-sim")
    os.chdir(astra_sim)

    # -------------------------------------- Argument parsing --------------------------------------
    parser = argparse.ArgumentParser(description='LLMServingSim') 
    
    parser.add_argument('--cluster-config', type=str, help='configuration of each node with instances', default='cluster_config/single_node_single_instance.json')
    parser.add_argument('--max-batch', type=int, help='maximum size of the batch', default=0)
    parser.add_argument('--max-num-batched-tokens', type=int, help='maximum number of tokens to be processed in a single iteration', default=2048)
    parser.add_argument('--fp', type=int, help='size of floating point in bit', default=16)
    parser.add_argument('--request-routing-policy', type=str, choices=['RR', 'RAND', 'CUSTOM'], help='request routing algorithm', default='RR')
    parser.add_argument('--expert-routing-policy', type=str, choices=['RR', 'RAND', 'FAST', 'CUSTOM'], help='expert routing algorithm', default='FAST')
    parser.add_argument('--enable-prefix-caching', action='store_true', help="enable prefix caching or not", default=False)
    parser.add_argument('--enable-prefix-sharing', action='store_true', help="enable prefix cache pooling in second-tier prefix storage side", default=False)
    parser.add_argument('--prefix-storage', type=str, choices=['None', 'CPU', 'CXL'], help='storage medium for second-tier prefix caching system', default='None')
    parser.add_argument('--enable-local-offloading', action='store_true', help="enable weight offloading to local (NPU) memory "
                        "(recommended to *disable* unless weight memory access is not counted in profiling)", default=False)
    parser.add_argument('--enable-attn-offloading', action='store_true', help="enable attention offloading to PIM", default=False)
    parser.add_argument('--enable-sub-batch-interleaving', action='store_true', help="enable sub-batch interleaving for better resource utilization", default=False)
    parser.add_argument('--enable-attn-prediction', action='store_true', help="enable realtime attention prediction", default=False)
    parser.add_argument('--prioritize-prefill', action='store_true', help="prioritize prefill", default=False)
    parser.add_argument('--block-size', type=int, help='kv cache block size unit of tokens', default=16)
    parser.add_argument('--dataset', type=str, help='dataset path', default=None)
    parser.add_argument('--output', type=str, help='output path', default=None)
    parser.add_argument('--timeseries-output', type=str, help='path for time-series CSV of per-interval memory/cache metrics', default=None)
    parser.add_argument('--stdout-log', type=str, help='optional path to mirror stdout/stderr logs', default=None)
    parser.add_argument('--gen', action='store_false', default=True, help='skip initiation phase')
    parser.add_argument('--num-req', type=int, help='number of requests to use', default=100)
    parser.add_argument('--log-interval', type=float, help='interval to log throughput (sec)', default=0.5)
    parser.add_argument('--log-level', type=str, choices=['WARNING', 'INFO', 'DEBUG'], help='log level to use', default='WARNING')
    parser.add_argument('--network-backend', type=str, choices=['analytical', 'ns3'], help='network backend to use', default='analytical')
    policy_choices = get_registered_policy_names()
    parser.add_argument(
        '--kv-eviction-policy',
        type=str,
        choices=policy_choices,
        help=(
            "policy for selecting decode requests to preempt when NPU KV memory is full "
            f"({', '.join(policy_choices)}). If omitted, falls back to cluster_config.kv_eviction_policy or 'tail'."
        ),
        default=None,
    )
    parser.add_argument(
        '--evicpress-alpha',
        type=float,
        default=1.0,
        help='quality-vs-delay tradeoff coefficient for EVICPRESS utility scoring',
    )
    parser.add_argument(
        '--evicpress-ratios',
        type=str,
        default='1.0,0.75,0.5,0.25',
        help='comma-separated compression keep ratios in (0,1], used by EVICPRESS',
    )
    parser.add_argument(
        '--evicpress-methods',
        type=str,
        default='balanced',
        help='comma-separated EVICPRESS compression methods/profiles (none, balanced, kivi, kvquant, gearkv, h2o, trace)',
    )
    parser.add_argument(
        '--evicpress-compression-trace',
        type=str,
        default='',
        help='optional path to JSON/CSV ratio-sensitivity trace; required when --evicpress-methods includes trace',
    )
    parser.add_argument(
        '--harp-grace-candidates',
        type=str,
        default='16,32,64',
        help='comma-separated grace-token candidates for HARP shadow state',
    )
    parser.add_argument(
        '--harp-ratios',
        type=str,
        default='1.0,0.75,0.5,0.25',
        help='comma-separated compression keep ratios in (0,1], used by HARP',
    )
    parser.add_argument('--harp-lambda-stall', type=float, default=1.0, help='HARP stall penalty weight')
    parser.add_argument('--harp-lambda-quality', type=float, default=0.5, help='HARP quality penalty weight')
    parser.add_argument('--harp-lambda-fairness', type=float, default=0.1, help='HARP fairness debt penalty weight')
    parser.add_argument('--harp-fairness-epsilon', type=float, default=1e-6, help='HARP fairness denominator epsilon')
    parser.add_argument(
        '--harp-compression-profile',
        type=str,
        default='balanced',
        help='HARP compression profile preset (none, balanced, kivi, kvquant, gearkv, h2o)',
    )
    parser.add_argument(
        '--harp-compression-trace',
        type=str,
        default='',
        help='optional path to JSON/CSV ratio-sensitivity trace for HARP compression modeling',
    )
    parser.add_argument(
        '--adaptive-dynmax-schedule',
        type=str,
        choices=['linear'],
        default='linear',
        help='adaptive DynMax schedule mode',
    )
    parser.add_argument('--adaptive-dynmax-progress-start', type=float, default=0.10,
                        help='progress ratio where adaptive DynMax starts decaying proactivity')
    parser.add_argument('--adaptive-dynmax-progress-end', type=float, default=0.75,
                        help='progress ratio where adaptive DynMax reaches its final setting')
    parser.add_argument('--adaptive-dynmax-final-trigger', type=float, default=1.05,
                        help='final proactive trigger ratio for adaptive DynMax')
    parser.add_argument('--adaptive-dynmax-final-target', type=float, default=0.92,
                        help='final proactive target ratio for adaptive DynMax')
    parser.add_argument('--adaptive-dynmax-final-steps-ahead', type=int, default=12,
                        help='final lookahead horizon for adaptive DynMax proactive eviction')
    parser.add_argument('--adaptive-dynmax-final-max-actions', type=int, default=1,
                        help='final proactive action budget for adaptive DynMax')
    parser.add_argument('--enable-proactive-eviction', action='store_true', default=False,
                        help='enable proactive KV eviction under projected NPU memory pressure (HARP/DynMax)')
    parser.add_argument('--proactive-steps-ahead', type=int, default=32,
                        help='decode steps used for projected NPU memory pressure forecast')
    parser.add_argument('--proactive-trigger', type=float, default=0.85,
                        help='projected pressure ratio threshold to trigger proactive eviction')
    parser.add_argument('--proactive-target', type=float, default=0.70,
                        help='projected pressure ratio target after proactive eviction')
    parser.add_argument('--proactive-max-actions', type=int, default=2,
                        help='maximum proactive eviction actions per scheduler tick')

    args = parser.parse_args()

    if args.stdout_log:
        if os.path.isabs(args.stdout_log):
            stdout_log_path = args.stdout_log
        else:
            stdout_log_path = os.path.join(cwd, args.stdout_log)
        os.makedirs(os.path.dirname(stdout_log_path), exist_ok=True)
        stdout_log_file = open(stdout_log_path, "w", encoding="utf-8")
        flush.stdout = _TeeStream(flush.stdout, stdout_log_file)
        flush.stderr = _TeeStream(flush.stderr, stdout_log_file)

    print_logo()
    print_input_config(args=args)
    print(bold(cyan("▶ Starting simulation...\n")))
    flush.stdout.flush()

    configure_logger(level=args.log_level)
    logger = get_logger("Main")
    
    max_batch=args.max_batch if args.max_batch != 0 else float('inf')
    max_num_batched_tokens=args.max_num_batched_tokens if args.max_num_batched_tokens != 0 else float('inf')
    block_size=args.block_size
    fp=args.fp
    request_routing_policy=args.request_routing_policy
    expert_routing_policy=args.expert_routing_policy
    enable_prefix_caching=args.enable_prefix_caching
    enable_prefix_sharing=args.enable_prefix_sharing
    prefix_storage=args.prefix_storage
    enable_local_offloading=args.enable_local_offloading
    enable_attn_offloading=args.enable_attn_offloading
    enable_sub_batch_interleaving=args.enable_sub_batch_interleaving
    if not enable_attn_offloading and enable_sub_batch_interleaving:
        raise RuntimeError("Sub-batch interleaving requires attention offloading to be enabled")
    enable_attn_prediction=args.enable_attn_prediction
    if enable_attn_prediction:
        logger.warning(
            "Realtime attention prediction is enabled. This may slow down the simulation."
        )
    prioritize_prefill=args.prioritize_prefill
    dataset=args.dataset
    output_file=args.output
    timeseries_output=args.timeseries_output
    is_init=args.gen
    num_req=args.num_req
    log_interval=args.log_interval
    network_backend = args.network_backend
    kv_eviction_policy_arg = args.kv_eviction_policy
    evicpress_alpha = args.evicpress_alpha
    try:
        evicpress_ratios = [float(v.strip()) for v in str(args.evicpress_ratios).split(',') if v.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --evicpress-ratios '{args.evicpress_ratios}': {exc}")
    if not evicpress_ratios:
        raise ValueError("--evicpress-ratios produced an empty list. Provide values in (0,1].")
    evicpress_methods = [v.strip().lower() for v in str(args.evicpress_methods).split(',') if v.strip()]
    if not evicpress_methods:
        raise ValueError("--evicpress-methods produced an empty list. Provide at least one method name.")
    evicpress_compression_trace = str(args.evicpress_compression_trace or '')
    try:
        harp_grace_candidates = [int(v.strip()) for v in str(args.harp_grace_candidates).split(',') if v.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --harp-grace-candidates '{args.harp_grace_candidates}': {exc}")
    if not harp_grace_candidates:
        raise ValueError("--harp-grace-candidates produced an empty list. Provide non-negative integers.")

    try:
        harp_ratios = [float(v.strip()) for v in str(args.harp_ratios).split(',') if v.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --harp-ratios '{args.harp_ratios}': {exc}")
    if not harp_ratios:
        raise ValueError("--harp-ratios produced an empty list. Provide values in (0,1].")

    harp_lambda_stall = float(args.harp_lambda_stall)
    harp_lambda_quality = float(args.harp_lambda_quality)
    harp_lambda_fairness = float(args.harp_lambda_fairness)
    harp_fairness_epsilon = float(args.harp_fairness_epsilon)
    harp_compression_profile = str(args.harp_compression_profile)
    harp_compression_trace = str(args.harp_compression_trace or '')
    adaptive_dynmax_schedule = str(args.adaptive_dynmax_schedule)
    adaptive_dynmax_progress_start = float(args.adaptive_dynmax_progress_start)
    adaptive_dynmax_progress_end = float(args.adaptive_dynmax_progress_end)
    adaptive_dynmax_final_trigger = float(args.adaptive_dynmax_final_trigger)
    adaptive_dynmax_final_target = float(args.adaptive_dynmax_final_target)
    adaptive_dynmax_final_steps_ahead = int(args.adaptive_dynmax_final_steps_ahead)
    adaptive_dynmax_final_max_actions = int(args.adaptive_dynmax_final_max_actions)
    enable_proactive_eviction = bool(args.enable_proactive_eviction)
    proactive_steps_ahead = int(args.proactive_steps_ahead)
    proactive_trigger = float(args.proactive_trigger)
    proactive_target = float(args.proactive_target)
    proactive_max_actions = int(args.proactive_max_actions)
    # ---------------------------------- Extract cluster config -----------------------------------
    cluster = build_cluster_config(astra_sim, args.cluster_config, args.enable_local_offloading, args.enable_attn_offloading)
    num_nodes = cluster["num_nodes"]
    num_instances = cluster["num_instances"]
    instances = cluster["instances"]
    inst2node_mapping = cluster["inst2node_mapping"]
    inst2npu_mapping = cluster["inst2npu_mapping"]
    npu2inst_mapping = cluster["npu2inst_mapping"]
    prefill_instance = cluster["prefill_instance"]
    decode_instance = cluster["decode_instance"]
    start_npu_ids = cluster["start_npu_ids"]
    end_npu_ids = cluster["end_npu_ids"]
    placement = cluster["placement"]
    block_mode_on = cluster["block_mode_on"]
    total_npu = cluster["total_npu"]
    cpu_mem_size = cluster["cpu_mem_size"]
    power_modeling = cluster["power_modeling"]
    power_configs = cluster["power_configs"]
    pim_models = cluster["pim_models"]
    external_tier_name = cluster.get("external_tier_name", "CXL")
    external_tier_bw = cluster.get("external_tier_bw", 0)
    external_tier_latency = cluster.get("external_tier_latency", 0)
    kv_eviction_policy = kv_eviction_policy_arg or cluster.get("kv_eviction_policy", "tail")
    # ----------------------------------------- Set config -----------------------------------------
    # Automatic network, memory configuration
    # If you want to set more specific information such as latency, look at config.py and each json file
    if network_backend == 'analytical':
        network=os.path.join(astra_sim, "inputs/network/network.yml")
        binary=os.path.join(astra_sim, "build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra")
    elif network_backend == 'ns3':
        network=os.path.join(astra_sim, "extern/network_backend/ns-3/scratch/config/config.txt")
        binary=os.path.join(astra_sim, "extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default")
        # make output files
        output_dir = os.path.join(astra_sim, "extern/network_backend/ns-3/scratch/output")
        os.makedirs(output_dir, exist_ok=True)
        open(os.path.join(output_dir, "flow.txt"), "w").close()
        open(os.path.join(output_dir, "trace.txt"), "w").close()
    else:
        raise NotImplementedError("Only analytical and ns3 network backend are supported")
    memory=os.path.join(astra_sim, 'inputs/memory/memory_expansion.json')
    system=os.path.join(astra_sim, "inputs/system/system.json")
    # ------------------------------------- Prepare simulation -------------------------------------
    # Need to extract each instance's memory accessability 
    node2inst_mapping = defaultdict(list)
    for inst_id, node_id in inst2node_mapping.items():
        node2inst_mapping[node_id].append(inst_id)
    node2inst_mapping = dict(node2inst_mapping)

    prefix_pool_inst_mapping = {}
    for i in range(num_instances):
        prefix_pool_inst_mapping[i] = None

    pool_device = None

    if prefix_storage == "CPU":
        pool_device = Device.CPU
    elif prefix_storage == "CXL":
        pool_device = Device.CXL

    if enable_prefix_caching and enable_prefix_sharing and prefix_storage != 'None':
        num_prefix_pool = num_nodes
        # make prefix pool objects based on num_prefix_pool
        prefix_pools = []
        if prefix_storage == 'CPU':
            for i in range(num_prefix_pool):
                if cpu_mem_size[i] > 0:
                    new_prefix_pool = RadixCache(
                                                node_id=0,
                                                device=prefix_storage, 
                                                page_size=256,
                                                capacity = cpu_mem_size[i] * GB_TO_BYTE,
                                                kv_size=131072,
                                                enable_kv_cache_events=True)
                    prefix_pools.append(new_prefix_pool)
                else:
                    raise RuntimeError(f"Memory size for prefix storage type {prefix_storage} is invalid")
            # This means one node shares one prefix pool
            prefix_pool_inst_mapping = inst2node_mapping

        elif prefix_storage == 'CXL':
            if cluster["cxl_mem_size"] > 0:
                new_prefix_pool = RadixCache(
                                            node_id=None,
                                            device=prefix_storage, 
                                            page_size=1,
                                            capacity = cluster["cxl_mem_size"] * GB_TO_BYTE, 
                                            kv_size=131072,
                                            enable_kv_cache_events=True)
                prefix_pools.append(new_prefix_pool)
                # This means every instance shares the same universal prefix pool (maybe fixed later)
                prefix_pool_inst_mapping = [0 for _ in range(num_instances)]
            else:
                raise RuntimeError(f"Memory size for prefix storage type {prefix_storage} is invalid")
        else:
            raise NotImplementedError(f"Prefix storage type {prefix_storage} is not supported or memory size is invalid")

    schedulers = []
    for instance_id, instance in enumerate(instances):
        prefix_pool_index = prefix_pool_inst_mapping[instance_id]
        prefix_pool = None
        if prefix_pool_index != None:
            prefix_pool = prefix_pools[prefix_pool_index]
        cxl_mem = 0
        if cluster["cxl_mem_size"] > 0:
            cxl_mem = cluster["cxl_mem_size"]        
        node_id = instance["node_id"]
        cpu_tier_bw = cluster["cpu_mem_bw"][node_id]
        cpu_tier_latency = cluster["cpu_mem_latency"][node_id]
        
        # Make scheduler for each instance
        schedulers.append(Scheduler(
            instance["model_name"], instance["node_id"], instance_id, max_batch, max_num_batched_tokens,
            instance["npu_num"], instance["npu_group"], instance["npu_mem"]["mem_size"], cpu_mem_size[instance["node_id"]],
            inst2npu_mapping[instance_id], instance["pd_type"], fp, block_size, num_req, 
            prioritize_prefill, enable_prefix_caching, enable_prefix_sharing, prefix_pool, pool_device,
            cxl_mem, kv_eviction_policy, external_tier_name,
            cpu_tier_bw, cpu_tier_latency, external_tier_bw, external_tier_latency,
            evicpress_alpha, evicpress_ratios,
            harp_grace_candidates, harp_ratios,
            harp_lambda_stall, harp_lambda_quality, harp_lambda_fairness,
            harp_fairness_epsilon, harp_compression_profile, harp_compression_trace,
            adaptive_dynmax_schedule, adaptive_dynmax_progress_start,
            adaptive_dynmax_progress_end, adaptive_dynmax_final_trigger,
            adaptive_dynmax_final_target, adaptive_dynmax_final_steps_ahead,
            adaptive_dynmax_final_max_actions,
            enable_proactive_eviction, proactive_steps_ahead,
            proactive_trigger, proactive_target, proactive_max_actions
        ))

    # Controller for astra-sim process communication
    controller = Controller(total_npu)
    # Global Request Router
    router = Router(num_instances, schedulers, num_req, request_routing_policy)
    # Power Modeling if enabled
    if power_modeling:
        power_model = PowerModel(power_configs)
    else:
        power_model = None

    # If there is no instance id, all requests are copied and added to each instance
    if dataset != None:
        router.generate(dataset, enable_prefix_caching=enable_prefix_caching, is_init=is_init)
    else:
        # Manually adding request
        for i in range(16):      # seq_len, end_len, arrival_time, instance_id
            for sched in schedulers:
                sched.add_request([i, sched.model, 64, 128, 0, i % num_instances])

    # Simulator start
    current = 0 # current tick of the system
    sys = 0 # current system id (NPU id)
    id = 0 # id of the request
    is_prefill_done = False # flag to check if prefill is done
    done_instance = [] # list of done instances
    done_inst_npus = [[] for _ in range(num_instances)]
    start_time = time()
    last_end_time = [0 for _ in range(num_instances)]
    last_calc_time = [0 for _ in range(num_instances)]
    waiting_request = [False for _ in range(num_instances)]

    # Calculating Simulator's Throughput
    throughput = []
    prompt_th = 0    # Avg Prompt Throguhput per Sec
    gen_th = 0       # Avg Generation Throughput per Sec
    last_log = 0    # last logged time
    FREQ = 1000_000_000 # 1 GHz (1e9 Hz)
    INTERVAL = log_interval*FREQ
    RATIO = FREQ//INTERVAL
    total_prompt = 0
    total_gen = 0
    total_latency = 0
    req_cnt = 0

    # Set Event Handler that loop with INTERVAL time until first request arrive (for all instances)
    first_arival_time = schedulers[0].get_first_arrival_time()
    if INTERVAL > first_arival_time:
        event_time = first_arival_time
    else:
        event_time = INTERVAL
    generate_event(int(event_time))
    # Make Chakra Grapth
    generate_graph(None, None, total_npu, event=True)
    # set first workload file
    workload = get_workload(None, None, event=True)
    # run subprocess
    args = [binary, "--workload-configuration="+workload, "--system-configuration="+system, "--network-configuration="+network, "--memory-configuration="+memory]
    if start_npu_ids != "":
        args.append("--start-npu-ids="+start_npu_ids)
    if end_npu_ids != "":
        args.append("--end-npu-ids="+end_npu_ids)
    if network_backend == 'ns3':
        args.append("--logical-topology-configuration="+astra_sim+"/inputs/logical_topology/logical_8nodes_1D.json")
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    # ----------------------------------- Start simulation loop ------------------------------------
    # Open time-series CSV if requested
    ts_csv_file = None
    ts_csv_writer = None
    if timeseries_output:
        ts_csv_path = os.path.join(cwd, timeseries_output)
        os.makedirs(os.path.dirname(ts_csv_path), exist_ok=True) if os.path.dirname(ts_csv_path) else None
        ts_csv_file = open(ts_csv_path, 'w', newline='')
        ts_csv_writer = csv_module.writer(ts_csv_file)
        ts_csv_writer.writerow([
            'sim_time_ns', 'instance_id',
            'npu_used_bytes', 'npu_total_bytes',
            'cpu_used_bytes', 'cpu_total_bytes',
            'cxl_used_bytes', 'cxl_total_bytes',
            'npu_prefix_tokens', 'storage_prefix_tokens',
            'evictable_tokens', 'protected_tokens',
            'npu_hit_total', 'storage_hit_total',
            'npu_hit_interval', 'storage_hit_interval', 'npu_hit_rate_interval',
            'prompt_throughput', 'gen_throughput',
            'evict_npu_to_cpu_bytes_total', 'load_cpu_to_npu_bytes_total',
            'evict_npu_to_cxl_bytes_total', 'load_cxl_to_npu_bytes_total',
            'evict_npu_prefix_bytes_total', 'evict_storage_prefix_bytes_total',
            'prefix_load_storage_to_npu_bytes_total',
            'harp_hot_reqs', 'harp_shadow_reqs', 'harp_cold_reqs',
            'harp_prefetch_remaining_bytes', 'harp_prefetch_overlap_ratio',
            'harp_stall_time_ns_total', 'harp_shadow_hit_rate',
        ])

    # Starting simulation, one while loop processes one iteration
    while True:
        out = controller.read_wait(p)
        out_dict = controller.parse_output(out[-2])

        if out_dict != None:
            sys = out_dict['sys']
            id = out_dict['id']
            current = out_dict['cycle']

        instance_id = npu2inst_mapping[sys]  # get instance id from NPU id
        node_id = inst2node_mapping[instance_id] # get node id from instance id

        # add stanby energy consumption for power modeling
        if power_modeling and sys == inst2npu_mapping[instance_id] and waiting_request[instance_id]:
            power_model.add_npu_standby_energy_consumption(instances[instance_id]["hardware"], node_id, current, 
                        last_end_time[instance_id], last_calc_time[instance_id], npu_nums=instances[instance_id]["npu_num"]) # total npus here
            last_calc_time[instance_id] = current

        # mark latest end time of the first NPU in the instance
        if sys == inst2npu_mapping[instance_id] and not waiting_request[instance_id]:
            last_end_time[instance_id] = current
            waiting_request[instance_id] = True

        # check request is done
        prompt_t, gen_t, reqs = schedulers[instance_id].add_done(id, sys, current)
        # add tokens in throughput
        prompt_th += prompt_t
        total_prompt += prompt_t
        gen_th += gen_t
        total_gen += gen_t
        # count only finished requests
        req_cnt += len(reqs) if instances[instance_id]["pd_type"] != "prefill" else 0

        # Add prefill ended requests to decode instance
        if instances[instance_id]["pd_type"] == "prefill" and len(reqs) > 0:
            router.transfer_prefill_request(reqs)

        # schedule requests
        new_req = schedulers[instance_id].schedule(current, sys, id)

        # runnable batch exists
        if new_req != None:
            if sys == inst2npu_mapping[instance_id]:  # if it is the first NPU of the instance, generate trace and graph
                # mark non-waiting state
                waiting_request[instance_id] = False
                instance = instances[instance_id]
                generate_trace(new_req, instance["hardware"], instance["npu_num"], instance["npu_group"], instance["pd_type"], 
                               node_id, instance_id, max_num_batched_tokens, placement[instance_id], block_mode_on[instance_id],
                               expert_routing_policy, enable_prefix_caching, enable_attn_offloading, power_model, pim_models[node_id], enable_attn_prediction, 
                               enable_sub_batch_interleaving, fp)
                generate_graph(new_req, instance["hardware"], instance["npu_num"], node_id,
                               instance_id, inst2npu_mapping[instance_id], enable_local_offloading)
            workload = get_workload(new_req, instance["hardware"], instance_id)
            controller.write_flush(p, workload)

        # check time to store throughput
        if current > last_log + INTERVAL:
            # store the prompt
            throughput.append((prompt_th*RATIO, gen_th*RATIO))
            last_log += INTERVAL
            log_time_str = f"[{last_log / FREQ:.1f}s]"
            log_time_len = len(log_time_str)
            log_indent = ' ' * log_time_len + '  '
            tree_indent = '├─'
            print(
                    log_time_str,
                    blue(f"Avg prompt throughput: {prompt_th * RATIO:.1f} tokens/s,"),
                    blue(f"Avg generation throughput: {gen_th * RATIO:.1f} tokens/s"),
                    end="\n"
                )
            prompt_th = 0
            gen_th = 0

            ######### Per Instance Metrics #########

            for inst_id in range(num_instances):
                running_reqs = sum([len(batch.requests) for batch in schedulers[inst_id].inflight] + [len([req for req in schedulers[inst_id].request if req.arrival <= current])])
                
                mem = schedulers[inst_id].memory
                npu_used_mb = mem.npu_used / MB_TO_BYTE
                npu_cap_mb = mem.npu_mem / MB_TO_BYTE if mem.npu_mem else 0.0
                npu_util = (mem.npu_used / mem.npu_mem * 100.0) if mem.npu_mem else 0.0
            
                print(f"{log_indent+tree_indent}Running Instance[{inst_id}]: {running_reqs} reqs,", end=' ')
                print(f"Total # {schedulers[inst_id].npu_num} NPUs, Each NPU Memory Usage {npu_used_mb:.2f} MB ({npu_util:.3f} % Used)", end='')
                if enable_prefix_caching:
                    schedulers[inst_id].memory.npu_prefix_cache.print_prefix_info()
                print()
            
            ######### Per Node Metrics #########
            if node2inst_mapping:
                num_nodes = len(node2inst_mapping)
                for i, (node_id, inst_ids) in enumerate(node2inst_mapping.items()):
                    node_cpu_usage = 0
                    if enable_prefix_sharing and prefix_storage == "CPU":
                        node_cpu_usage = (prefix_pools[node_id].total_size() * 131072)
                    else:
                        inst_usage = []
                        for inst_id in inst_ids:
                            inst_cpu_usage = schedulers[inst_id].memory.cpu_used
                            node_cpu_usage += inst_cpu_usage
                            inst_usage.append(inst_cpu_usage)

                    cpu_util = (node_cpu_usage / (cpu_mem_size[node_id]*GB_TO_BYTE)) * 100
                    if prefix_storage != "CXL" and not power_modeling and i == num_nodes - 1:
                        tree_indent = '└─'
                    print(f"{log_indent+tree_indent}Node[{node_id}]: Total CPU Memory Usage {node_cpu_usage/MB_TO_BYTE:.2f} MB, {cpu_util:.3f} % Used ", end='')
                    if enable_prefix_caching and enable_prefix_sharing and prefix_storage == "CPU":
                        prefix_pools[node_id].print_prefix_info()

                    if (enable_prefix_sharing and prefix_storage == "CPU") or (len(inst_ids) == 1):
                        print()
                    else:
                        for i, inst_cpu_usage in enumerate(inst_usage):
                            if i == 0:
                                print("(", end='')
                            inst_cpu_util = (inst_cpu_usage / node_cpu_usage)*100 if node_cpu_usage else 0
                            print(f"Instance[{inst_ids[i]}]: {inst_cpu_util:.2f} %", end='')
                            if i == len(inst_usage) - 1:
                                print(")", end='')
                            else:
                                print(", ", end='')
                        print()

            ######### Per CXL Metrics #########
            if prefix_storage == "CXL":
                if enable_prefix_sharing:
                    num_prefix_pool = len(prefix_pools)
                    for i, cxl_id, cxl_pool in enumerate(prefix_pools):
                        cxl_usage = (cxl_pool.total_size() * 131072)
                        cxl_util = cxl_usage / cxl_pool.capacity
                        if not power_modeling and i == num_prefix_pool - 1:
                            tree_indent = '└─'
                        print(f"{log_indent+tree_indent}CXL[{cxl_id}]: Total CXL Device Memory Usage {cxl_usage/MB_TO_BYTE:.2f}MB, {cxl_util:.3f} % Used")
                else:
                    # else only one instance could explictly use CXL
                    inst_id = 0
                    cxl_usage = (schedulers[inst_id].memory.second_tier_prefix_cache.total_size() * 131072)
                    cxl_util = cxl_usage / schedulers[inst_id].memory.second_tier_prefix_cache.capacity
                    if not power_modeling:
                        tree_indent = '└─'
                    print(f"{log_indent+tree_indent}CXL[0]: Total CXL Device Memory Usage {cxl_usage / MB_TO_BYTE:.2f} MB, {cxl_util:.3f} % Used")

            if power_modeling:
                tree_indent = '└─'
                print(f"{log_indent+tree_indent}Avg power consumption: {power_model.get_current_power(current)} W")

            # Write time-series CSV rows
            if ts_csv_writer:
                for inst_id in range(num_instances):
                    mem = schedulers[inst_id].memory
                    npu_prefix_tokens = 0
                    storage_prefix_tokens = 0
                    evictable_tokens = 0
                    protected_tokens = 0
                    npu_hit_total = 0
                    storage_hit_total = 0
                    npu_hit_interval = 0
                    storage_hit_interval = 0
                    npu_hit_rate_interval = 0.0

                    if enable_prefix_caching:
                        npu_prefix_tokens = mem.npu_prefix_cache.total_size()
                        evictable_tokens = mem.npu_prefix_cache.evictable_size()
                        protected_tokens = mem.npu_prefix_cache.protected_size()
                        npu_hit_total = mem.npu_prefix_cache.total_hit_tokens
                        _, npu_hit_interval, npu_hit_rate_interval = mem.npu_prefix_cache.get_interval_hit_rate()

                        if hasattr(mem, 'second_tier_prefix_cache') and mem.second_tier_prefix_cache is not None:
                            storage_prefix_tokens = mem.second_tier_prefix_cache.total_size()
                            storage_hit_total = mem.second_tier_prefix_cache.total_hit_tokens
                            _, storage_hit_interval, _ = mem.second_tier_prefix_cache.get_interval_hit_rate()

                    ts = schedulers[inst_id].tier_stats
                    harp_counts = schedulers[inst_id].get_harp_state_counts()
                    harp_overlap_ratio = 0.0
                    if ts['harp_prefetch_bytes_total'] > 0:
                        harp_overlap_ratio = ts['harp_prefetch_overlap_bytes'] / ts['harp_prefetch_bytes_total']
                    harp_shadow_hit_rate = 0.0
                    if ts['harp_decode_tokens_total'] > 0:
                        harp_shadow_hit_rate = ts['harp_shadow_hit_tokens'] / ts['harp_decode_tokens_total']
                    ts_csv_writer.writerow([
                        int(last_log), inst_id,
                        int(mem.npu_used), int(mem.npu_mem),
                        int(mem.cpu_used), int(mem.cpu_mem),
                        int(mem.cxl_used), int(mem.cxl_mem),
                        npu_prefix_tokens, storage_prefix_tokens,
                        evictable_tokens, protected_tokens,
                        npu_hit_total, storage_hit_total,
                        npu_hit_interval, storage_hit_interval, f"{npu_hit_rate_interval:.2f}",
                        int(prompt_th * RATIO), int(gen_th * RATIO),
                        ts['evict_npu_to_cpu_bytes'], ts['load_cpu_to_npu_bytes'],
                        ts['evict_npu_to_cxl_bytes'], ts['load_cxl_to_npu_bytes'],
                        ts['evict_npu_prefix_bytes'], ts['evict_storage_prefix_bytes'],
                        ts['prefix_load_storage_to_npu_bytes'],
                        harp_counts['hot'], harp_counts['shadow'], harp_counts['cold'],
                        int(harp_counts['prefetch_remaining_bytes']), f"{harp_overlap_ratio:.6f}",
                        int(ts['harp_stall_time_ns']), f"{harp_shadow_hit_rate:.6f}",
                    ])
                ts_csv_file.flush()

        # check if all requests are done for current instance
        if (instance_id not in decode_instance or is_prefill_done) and instance_id not in done_instance and schedulers[instance_id].is_request_empty():
            if sys not in done_inst_npus[instance_id]:
                done_inst_npus[instance_id].append(sys)

            if len(done_inst_npus[instance_id]) == (1 if instances[instance_id]["npu_num"] == 1 else 2): # start & end npu
                done_instance.append(instance_id)

            # check if all prefill instances are done
            if len(done_instance) == len(prefill_instance):
                is_prefill_done = True

            # check if all instances are done
            if len(done_instance) == num_instances:
                for inst_idx in range(num_instances):
                    schedulers[inst_idx].memory.free_prefix_cache()
                    schedulers[inst_idx].memory.free_weight()
                
                    if not schedulers[inst_idx].memory.is_free():
                        logger.error(f"Instance[{inst_idx}] has unfreed memory after all requests are done")

                print(SINGLE_BAR)
                print(bold(cyan("▶ Exiting simulation...\n")))
                controller.write_flush(p, "exit")
                break

            controller.write_flush(p, "done") # make done instances to sleep
        elif new_req == None:
            controller.write_flush(p, "pass")
        
        # flush
        flush.stdout.flush()

    # calculate simulation time
    end_time = time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # check all scheduled requests in astra-sim are well done
    controller.check_end(p)

    # calcuate prefix caching metrics
    total_requested_tokens = 0
    total_npu_hit_tokens = 0
    total_cpu_hit_tokens = 0
    if enable_prefix_caching:
        for i in range(num_instances):
            (temp_npu_a, temp_npu_b), (temp_cpu_a, temp_cpu_b) = schedulers[i].memory.return_prefix_info()
            if (not enable_prefix_sharing) and (prefix_storage != "None") and (temp_npu_a != temp_cpu_a):
                raise RuntimeError(f"Instance[{i}] prefix caching requested tokens mismatch between NPU ({temp_npu_a}) and CPU ({temp_cpu_a})")
            total_requested_tokens += temp_npu_a
            total_npu_hit_tokens += temp_npu_b
            if not enable_prefix_sharing:
                total_cpu_hit_tokens += temp_cpu_b
        
        if enable_prefix_sharing:
            for pool in prefix_pools:
                _, temp_cpu_b = pool.return_prefix_info()
                total_cpu_hit_tokens += temp_cpu_b
    
    # This is total system's throughput
    total_latency = current/FREQ
    print(SINGLE_BAR)
    print(bold(cyan("▶ Simulation results...\n")))
    print(f"Total simulation time: {int(hours)}h {int(minutes)}m {seconds:.3f}s")
    print(SINGLE_BAR)
    print(magenta(center('Throughput Results')))
    print(SINGLE_BAR)
    print(f"Total requests:                                                     {req_cnt}")
    print(f"Total clocks (ns):                                                  {current}")
    print(f"Total latency (s):                                                  {total_latency:.3f}")
    print(f"Total input tokens:                                                 {total_prompt}")
    print(f"Total generated tokens:                                             {total_gen}")
    print(f"Request throughput (req/s):                                         {req_cnt/total_latency:.2f}")
    print(f"Average prompt throughput (tok/s):                                  {total_prompt/total_latency:.2f}")
    print(f"Average generation throughput (tok/s):                              {total_gen/total_latency:.2f}")
    print(f"Total token throughput (tok/s):                                     {(total_prompt + total_gen)/total_latency:.2f}")
    print(f"Throughput per {1/RATIO} sec: {throughput}")
    print(SINGLE_BAR)
    if enable_prefix_caching:
        print(magenta(center("Prefix Caching Results")))
        print(SINGLE_BAR)
        print(f"Total requested prompt tokens:                                      {total_requested_tokens}")
        print(f"NPU prefix hit prompt tokens:                                       {total_npu_hit_tokens}")
        print(f"NPU prefix hit ratio (%):                                           {(total_npu_hit_tokens/total_requested_tokens)*100:.2f}")
        if prefix_storage != "None":
            print(f"{prefix_storage} prefix hit prompt tokens:                                       {total_cpu_hit_tokens}")
            print(f"{prefix_storage} prefix hit ratio (%):                                           {(total_cpu_hit_tokens/total_requested_tokens)*100:.2f}")
        print(f"Total prefix hit ratio (%):                                         {((total_npu_hit_tokens+total_cpu_hit_tokens)/total_requested_tokens)*100:.2f}")
        print(SINGLE_BAR)
    if power_modeling:
        print(magenta(center("Power Modeling Results")))
        print(SINGLE_BAR)
        total_energy = power_model.get_final_energy(current)
        print(f"Total energy consumption (kJ):                                      {total_energy/1000:.2f}")
        # Each node results
        power_model.print_power_summary()
        print(f"Power per {1/RATIO} sec (W): {power_model.power_time_series}")
        print(SINGLE_BAR)
    # Each instacne results
    for i in range(num_instances):
        print(magenta(center(f"Instance [{i}]")))
        print(SINGLE_BAR)
        schedulers[i].print_result()
        schedulers[i].print_tier_stats()
        acc_ok, req_totals, acc_deltas = schedulers[i].validate_tier_accounting()
        if acc_ok:
            print("Tier accounting check:                                             PASS")
        else:
            print("Tier accounting check:                                             WARNING")
            print("Request-tier totals and scheduler-tier totals do not match.")
            print(f"  evict_npu_to_cpu_bytes delta:                                   {acc_deltas['evict_npu_to_cpu_bytes']}")
            print(f"  evict_npu_to_cxl_bytes delta:                                   {acc_deltas['evict_npu_to_cxl_bytes']}")
            print(f"  load_cpu_to_npu_bytes delta:                                    {acc_deltas['load_cpu_to_npu_bytes']}")
            print(f"  load_cxl_to_npu_bytes delta:                                    {acc_deltas['load_cxl_to_npu_bytes']}")
        print(f"Request-level transition bytes total:                             {req_totals['tier_transition_bytes_total']}")
        print(SINGLE_BAR)
    
    # Important informations about metrics
    # The TTFT (Time to First Token) in our simulator differs from vllm. 
    # While vllm measures TTFT as the time when the client receives the first token,
    # Our simulator measures it as the time when the computation of the first token is completed.
    # Therefore, vllm gets much more higher TTFT.
    # (Ref: https://docs.vllm.ai/en/latest/design/metrics.html?utm_source=chatgpt.com#interval-calculations-vs-preemptions)

    if output_file != None:
        print(f"Saving each request's information to output file: {output_file}")
        for i in range(num_instances):
            schedulers[i].save_output(output_file, is_append=False if i == 0 else True)

    # Save tier stats JSON alongside the output file
    if output_file is not None:
        tier_stats_path = os.path.join(cwd, output_file.replace('.csv', '_tier_stats.json'))
        os.makedirs(os.path.dirname(tier_stats_path), exist_ok=True) if os.path.dirname(tier_stats_path) else None
        all_tier_stats = {}
        for i in range(num_instances):
            all_tier_stats[f"instance_{i}"] = schedulers[i].tier_stats
        with open(tier_stats_path, 'w') as f:
            json.dump(all_tier_stats, f, indent=2)
        print(f"Saving tier stats to: {tier_stats_path}")

    # Close time-series CSV
    if ts_csv_file:
        ts_csv_file.close()
        print(f"Saved time-series metrics to: {timeseries_output}")
    

if __name__ == "__main__":
    # For simulation time breakdown
    # profiler = Profiler()
    # profiler.start()
    main()
    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))