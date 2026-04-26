# class that manages request of astra-sim
class Request:
    def __init__(self, id, model, input, output, arrival, instance_id, input_hash_ids=None, output_hash_ids=None, is_init=True):
        self.id = id
        self.model = model
        self.input = input
        self.output = output
        self.arrival = arrival
        self.instance_id = instance_id
        self.is_init = is_init
        self.original_input = input
        self.evict = False
        self.evict_device = None   # Device enum: where KV was evicted to (CXL or CPU)
        # Per-request tier movement attribution for post-run analysis.
        self.last_kv_load_tier = "NPU"
        self.evict_npu_to_cpu_bytes = 0
        self.evict_npu_to_cxl_bytes = 0
        self.load_cpu_to_npu_bytes = 0
        self.load_cxl_to_npu_bytes = 0
        self.evict_raw_bytes = 0
        self.evict_stored_bytes = 0
        self.evict_compression_ratio = 1.0
        self.evict_event_count = 0
        self.current_evict_raw_bytes = 0
        self.current_evict_stored_bytes = 0
        self.current_evict_compression_ratio = 1.0
        self.evicpress_utility_score = 0.0
        self.harp_state = "hot"
        self.harp_grace_tokens_remaining = 0
        self.harp_grace_tail_bytes = 0
        self.harp_shadow_ratio = 1.0
        self.harp_missing_raw_bytes = 0
        self.harp_prefetch_remaining_bytes = 0.0
        self.harp_storage_bytes_remaining = 0
        self.harp_prefetch_latency_ns_remaining = 0.0
        self.harp_prefetch_device = None
        self.harp_stall_time_ns = 0
        self.harp_stall_events = 0
        self.harp_stall_active = False
        self.harp_shadow_hit_tokens = 0
        self.harp_decode_tokens = 0
        self.harp_fairness_debt = 0.0
        self.end_time = -1
        self.latency = -1
        self.queuing_delay = -1
        self.ttft = -1
        self.tpot = -1
        self.itl = []
        self.recent_end = 0

        # For prefix caching modeling
        self.input_hash_ids = input_hash_ids
        self.output_hash_ids = output_hash_ids
        self.prefix_cache_hit = 0
        self.npu_cache_hit = 0
        # self.cpu_cache_hit = 0
        self.storage_cache_hit = 0
        self.npu_last_node = None
        # self.cpu_last_node = None
        self.storage_last_node = None

    # to print the request information
    def __str__(self):
        return str(self.__dict__) 

    def add_latency(self, end_time):
        self.end_time = end_time
        self.latency = self.end_time - self.arrival
        # remove useless information
        self.input = self.original_input
        if self.output == self.input + 1:
            self.tpot = 0
        else:
            self.tpot = (self.latency - self.ttft) // (self.output - self.input - 1)
        
        del self.original_input
        del self.is_init
        del self.evict
        if hasattr(self, 'evict_device'):
            del self.evict_device
    
    def add_itl(self, current):
        self.itl.append(current - self.recent_end)
        self.recent_end = current

    def set_que_delay(self, current):
        self.queuing_delay = current - self.arrival
    
    def set_ttft(self, current):
        self.ttft = current - self.arrival
        self.recent_end = current


# class that manages batch of astra-sim
class Batch:
    def __init__(self, batch_id, model, total_len, kv_len, hit_len, q_list, k_list, num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list, batch_time, kv_size, evict=0, load=0, evict_cxl=0, load_cxl=0):
        self.batch_id = batch_id
        self.model = model
        self.total_len = total_len
        self.kv_len = kv_len
        self.hit_len = hit_len
        self.batch_time = batch_time
        self.fired = [] # systems that fired this batch
        self.requests = []
        self.end = []
        # vllm
        self.kv_size = kv_size
        self.evict = evict
        self.load = load
        self.evict_cxl = evict_cxl   # bytes evicted NPU → CXL
        self.load_cxl = load_cxl     # bytes loaded  CXL → NPU
        # for attn prediction
        self.q_list = q_list
        self.k_list = k_list
        self.num_prefill = num_prefill
        self.num_decode = num_decode
        self.prefill_q_list = prefill_q_list
        self.prefill_k_list = prefill_k_list
        self.decode_k_list = decode_k_list