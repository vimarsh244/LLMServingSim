# LLMServingSim Architecture Analysis

*Systems-Architecture Perspective*  
*Analysis Date: March 3, 2026*

---

## 1. Hardware Abstraction

### Where Hardware Characteristics Are Defined

#### **Declarative Configuration (JSON)**
Location: [`cluster_config/*.json`](cluster_config/)

Hardware characteristics defined:
- **NPU memory**: `npu_mem.{mem_size, mem_bw, mem_latency}` (GB, GB/s, ns)
- **CPU memory**: `cpu_mem.{mem_size, mem_bw, mem_latency}`
- **CXL memory**: `cxl_mem.{mem_size, mem_bw, mem_latency}` (optional)
- **Network**: `link_bw`, `link_latency`
- **Power specs**: `power.npu[hardware].*` (idle/standby/active power)

**Example** from [`single_node_single_instance.json`](cluster_config/single_node_single_instance.json#L10-L15):
```json
"npu_mem": {
    "mem_size": 40,
    "mem_bw": 768,
    "mem_latency": 0
}
```

#### **PIM Configuration Files**
Location: [`pim_config/*.ini`](pim_config/)

Defines DRAM timings and structure:
- Timing parameters: `tCK`, `CL`, `tRP`, `tRAS`, `tRFC`, etc.
- Structure: `bus_width`, `data_rate`, `rows`, `columns`, `bankgroups`
- Bandwidth/capacity computed analytically from these parameters

**Example** from [`HBM2_1GB_2000_pim.ini`](pim_config/HBM2_1GB_2000_pim.ini):
```ini
[dram_structure]
protocol = HBM
bankgroups = 4
banks_per_group = 4
rows = 32768
columns = 64
device_width = 128
```

#### **Operator Performance Database (Trace-Based)**
Location: `llm_profile/perf_models/{hardware}/{model}/tp{tp}/layers.csv`

Structure: `(layer_name, input, kv_cache, tp_size) → latency(ns)`

**Example** from A6000/meta-llama/Llama-3.1-8B/tp1/layers.csv:
```csv
layer_name,input,kv_cache,tp_size,latency(ns)
embedding,1,0,1,2575
q_proj,1,0,1,49731
attn,1,0,1,5310
```

This is **empirically profiled**, not analytically computed.

---

### Adding a Completely New Accelerator Type

#### Required Files to Modify:

**1. Create Performance Database** (REQUIRED)
- **Path**: `llm_profile/perf_models/{NEW_HW}/{model}/tp{N}/layers.csv`
- **Action**: Profile each operator on the new hardware
- **Operators to profile**:
  - Core: `embedding`, `q_proj`, `k_proj`, `v_proj`, `rope`, `attn`, `o_proj`
  - MLP: `gate_proj`, `up_proj`, `act_fn`, `down_proj`
  - MoE (if applicable): `expert.w1`, `expert.w2`, `expert.w3`, `gate`
  - Norm: `input_layernorm`, `post_layernorm`, `final_layernorm`
  - Head: `lm_head`

**2. Update Cluster Configuration** (REQUIRED)
- **File**: [`cluster_config/*.json`](cluster_config/)
- **Action**: Add hardware string to `instances[].hardware` field
- **Define**: `npu_mem` specifications (size, bandwidth, latency)

**3. Add Power Model** (OPTIONAL)
- **File**: Same cluster config JSON
- **Action**: Add entry to `power.npu[{NEW_HW}]`
- **Fields**: `idle_power`, `active_power`, `standby_power`, `standby_duration`

**4. Add PIM Configuration** (OPTIONAL - only for PIM-enabled accelerators)
- **File**: Create `pim_config/{NEW_PIM}.ini`
- **Code**: Add latency model to [`PIMModel.estimate_with_linear()`](inference_serving/pim_model.py#L123-L149)

**Key Files**:
- [`inference_serving/trace_generator.py`](inference_serving/trace_generator.py#L2000) - Loads perf CSV via hardcoded path
- [`inference_serving/config_builder.py`](inference_serving/config_builder.py) - Parses cluster config

---

## 2. Device Coupling

### Abstract Device Interface

**Status**: ❌ **No clean abstract base class exists**

#### Device Representation

**Memory tiers** defined as enum in [`memory_model.py`](inference_serving/memory_model.py#L11-L15):
```python
class Device(Enum):
    NPU = 1
    CPU = 2
    CXL = 3
```

**Hardware identity**: String-based (`"A6000"`, `"H100"`, `"TPU-v6e-1"`)

No `DeviceModel` base class or interface pattern.

---

### Tight Coupling Examples

#### **1. Hardware String Dependencies**
[`trace_generator.py`](inference_serving/trace_generator.py#L2000):
```python
file_path = f"../llm_profile/perf_models/{hardware}/{model}/tp{tp}/layers.csv"
```
- **Issue**: Hardcoded path resolution based on hardware name
- **Impact**: Adding new hardware requires matching directory structure

#### **2. Hardware-Specific Conditionals**
[`trace_generator.py`](inference_serving/trace_generator.py#L279-L285):
```python
if 'TPU' not in hardware: # TPU includes rope in attention latency
    rope_matching_row = _get_perf_row(perf_db, hardware, "rope", ...)
```
- **Issue**: Special-case logic scattered throughout codebase
- **Impact**: Fragile to hardware additions

#### **3. PIM Models Hardcoded**
[`pim_model.py`](inference_serving/pim_model.py#L123-L149):
```python
attn_model = {
    "LPDDR4X_2GB_4266_pim": {"slope": 432.4458, ...},
    "DDR4_8GB_3200_pim": {"slope": 333.2538, ...},
    "LPDDR5_2GB_6400_pim": {"slope": 282.4338, ...},
    "HBM2_1GB_2000_pim": {"slope": 242.0548, ...},
}
```
- **Issue**: No polymorphism, dictionary-based dispatch
- **Impact**: Must update hardcoded dict for new PIM types

---

### Coupling Assessment

| **Component** | **Coupling Level** | **Reason** |
|---------------|-------------------|------------|
| **Scheduler** → Hardware | ✅ Low | Never directly calls hardware logic |
| **Trace Generator** → Hardware | ❌ High | Hardcoded paths, string checks |
| **Memory Model** → Devices | 🟡 Medium | Enum-based but centralized |
| **Execution Logic** → Hardware | ✅ Low | Via CSV lookup (decoupled) |

**Overall**: Moderate-to-tight coupling. No interface pattern, but partial decoupling via CSV-based perf lookup.

---

## 3. Trace vs Parametric Modeling

### Hybrid Approach

#### **Trace-Based (Pre-Profiled)**

**Operator latencies** from empirical CSV data:

[`trace_generator.py`](inference_serving/trace_generator.py#L1988-L2036):
```python
def _load_perf_db_dict(hardware, model, tp):
    file_path = f"../llm_profile/perf_models/{hardware}/{model}/tp{tp}/layers.csv"
    df = pd.read_csv(file_path)
    perf_db[(layer_name, input_len, kv_cache, tp_size)] = latency(ns)
    return perf_db
```

**Not analytical** — requires real hardware profiling.

---

#### **Parametric (Analytical)**

**Memory bandwidth/latency** from declarative config:
- Parsed from [`cluster_config/*.json`](cluster_config/)
- Used to compute data transfer time

**Data transfer modeling** [`trace_generator.py`](inference_serving/trace_generator.py#L86-L111):
```python
load_size = batch.load
mem.append(["kv_load", '0', 'LOCAL', '0', cpu_dev, str(load_size), ...])
power_model.add_dram_energy_consumption(node_id, load_size)
```

**PIM latency** (linear analytical model) [`pim_model.py`](inference_serving/pim_model.py#L120-L149):
```python
def get_pim_latency(self, n_head, kv_head, head_dim, L, channel_split=1):
    return (slope * L + intercept) / channel_split  # ns
```

---

### Where Execution Time Is Computed

#### **Primary Location**
[`trace_generator.py::_synthesize_trace()`](inference_serving/trace_generator.py#L108-L500)

This function:
1. Loads performance database for (hardware, model, tp)
2. For each layer, queries: `perf_db[(layer_name, input_len, kv_cache, tp)]`
3. Writes trace file with operator latencies

#### **Operator Latency Lookup**
[`trace_generator.py`](inference_serving/trace_generator.py#L2140-L2165):
```python
def _get_perf_row(perf_db, hardware, layer_name, input_len, kv_cache_len, tp):
    key = (layer_name, input_len, kv_cache_len, tp)
    if key in perf_db:
        return perf_db[key]
    else:
        # Fallback: search for closest match or interpolate
```

#### **Attention Prediction (Optional ML Path)**
[`trace_generator.py`](inference_serving/trace_generator.py#L1877-L1930):
- XGBoost predictor for attention latency
- Enabled via `--enable-attn-prediction` flag
- Used when batch sizes vary dynamically

---

### Summary: Trace vs Parametric

| **Component** | **Modeling Type** | **Source** |
|---------------|-------------------|------------|
| Operator latency | **Trace-based** | CSV files from profiling |
| Memory transfer | **Parametric** | Bandwidth/latency from config |
| PIM operations | **Parametric** | Linear models (slope/intercept) |
| Attention (optional) | **ML-based** | XGBoost predictor |

**Hybrid nature** enables accuracy (traces) + flexibility (analytical).

---

## 4. Memory Hierarchy Modeling

### Fixed 3-Tier Hierarchy

#### **Device Enum**
[`memory_model.py`](inference_serving/memory_model.py#L11-L15):
```python
class Device(Enum):
    NPU = 1  # NPU on-device memory (HBM/GDDR)
    CPU = 2  # Host DRAM
    CXL = 3  # CXL.mem expansion
```

**Hardcoded** — not a pluggable tier system.

---

### Memory Management Implementation

[`memory_model.py`](inference_serving/memory_model.py#L218-L314):
```python
def allocate(self, size, device):
    if device == Device.NPU:
        if self.npu_used + size > self.npu_mem:
            raise RuntimeError(...)
        self.npu_used += size
    elif device == Device.CPU:
        self.cpu_used += size
    elif device == Device.CXL:
        self.cxl_used += size
    else:
        raise RuntimeError(f"Unsupported device {device}")
```

**If-else dispatch** — no polymorphism.

---

### Adding a New Memory Tier (e.g., On-Chip SRAM)

#### **Files to Modify**:

**1. Extend Device Enum** [`memory_model.py`](inference_serving/memory_model.py#L11):
```python
class Device(Enum):
    NPU = 1
    CPU = 2
    CXL = 3
    SRAM = 4  # <-- Add this
```

**2. Add Allocation Logic** [`memory_model.py`](inference_serving/memory_model.py#L218):
```python
def allocate(self, size, device):
    # ... existing cases ...
    elif device == Device.SRAM:
        if self.sram_used + size > self.sram_mem:
            raise RuntimeError(...)
        self.sram_used += size
```

**3. Add Free Logic** [`memory_model.py`](inference_serving/memory_model.py#L266):
```python
def free(self, size, device):
    # ... existing cases ...
    elif device == Device.SRAM:
        self.sram_used -= size
```

**4. Update Scheduler Eviction** [`scheduler.py`](inference_serving/scheduler.py#L120-L185):
- Add SRAM to eviction waterfall logic (which tier to evict to first)

**5. Update Trace Generator** [`trace_generator.py`](inference_serving/trace_generator.py#L86-L111):
- Add memory transfer nodes (`"kv_load"/"kv_evict"` with SRAM device string)

**6. Update Config Builder** [`config_builder.py`](inference_serving/config_builder.py):
- Parse SRAM specs from cluster config JSON

---

### Assessment: NOT Pluggable

**Impact**: ~4 files, ~50 lines of changes (manageable but requires multiple edits)

**Root Issue**: No `MemoryTier` interface/base class.

---

## 5. Operator Modeling

### Operator-Specific Modeling

#### **Operators Tracked**
[`trace_generator.py`](inference_serving/trace_generator.py#L108-L500):

| **Category** | **Operators** |
|--------------|---------------|
| Attention | `embedding`, `q_proj`, `k_proj`, `v_proj`, `rope`, `attn`, `o_proj` |
| MLP | `gate_proj`, `up_proj`, `act_fn`, `down_proj` |
| MoE | `expert.w1`, `expert.w2`, `expert.w3`, `gate` |
| Norm | `input_layernorm`, `post_layernorm`, `final_layernorm` |
| Head | `lm_head` |

**Not FLOPs-based** — each operator has independent empirical latency.

---

### Attention Special Handling

**3 Modeling Paths** [`trace_generator.py`](inference_serving/trace_generator.py#L195-L370):

1. **CSV Lookup (Default)**:
   - Pre-profiled `attn` latencies for various `(input, kv_cache)` pairs
   - Fast, accurate for profiled configurations

2. **ML Predictor** (`--enable-attn-prediction`):
   - XGBoost model trained on attention profiling data
   - Handles dynamic batch sizes gracefully
   - **Overhead**: Slower simulation due to model inference

3. **PIM Offload** (`--enable-attn-offloading`):
   - Linear model: `latency = (slope * L + intercept) / channel_split`
   - Defined in [`pim_model.py`](inference_serving/pim_model.py#L120)
   - Prefill on NPU, decode on PIM

---

### Data Size Computation

**Function**: [`memory_model.py::calculate_sizes()`](inference_serving/memory_model.py#L550-L784)

**Analytical per-operator formulas**:
```python
if layer_name == "embedding":
    input_size = seq * 8  # token IDs (int64)
    weight_size = vocab_size * n_embd * fp / tp
    output_size = seq * n_embd * fp

elif layer_name == "q_proj":
    input_size = seq * n_embd * fp
    weight_size = n_embd * n_embd * fp / tp
    output_size = seq * n_embd * fp

elif layer_name == "attn":
    input_size = seq * n_embd * fp  # Q
    weight_size = kv_len * kv_dim * fp * 2  # K, V cache
    output_size = seq * n_embd * fp
```

**Accounts for**:
- Tensor parallelism (`tp`)
- Floating-point precision (`fp`)
- KV cache size
- MoE expert routing

---

### Adding Operator-Specific Cost Model

**Scenario**: Add `flash_attn` kernel with specialized timing

#### **Steps**:

**1. Profile New Operator**:
- Generate CSV entries for `flash_attn` with various `(input, kv_cache)` pairs
- Add to `llm_profile/perf_models/{hardware}/{model}/tp{tp}/layers.csv`

**2. Update Trace Generator** [`trace_generator.py`](inference_serving/trace_generator.py#L280-L350):
```python
if use_flash_attn:
    flash_matching_row = _get_perf_row(perf_db, hardware, "flash_attn", 
                                        attn_len, kv_len, npus_per_group)
    block_res.append(formatter("flash_attn", 
                               str(flash_matching_row['latency(ns)']), ...))
```

**3. Add to Data Size Function** (if different from standard attention):
[`memory_model.py::calculate_sizes()`](inference_serving/memory_model.py#L550)

**Effort**: Low (~20 lines) — lookup-based system is extensible.

---

## 6. Extensibility Assessment

### Summary Matrix

| **Extension** | **Difficulty** | **Files to Modify** | **Coupling** |
|---------------|----------------|---------------------|--------------|
| **Add accelerator** | 🟡 Medium | `llm_profile/perf_models/{HW}/*.csv`<br>[`cluster_config/*.json`](cluster_config/) | Low (CSV-based) |
| **Add memory tier** | 🔴 High | [`memory_model.py`](inference_serving/memory_model.py)<br>[`scheduler.py`](inference_serving/scheduler.py)<br>[`trace_generator.py`](inference_serving/trace_generator.py)<br>[`config_builder.py`](inference_serving/config_builder.py) | **High (hardcoded enum)** |
| **Add operator cost model** | 🟢 Low | CSV + [`trace_generator.py`](inference_serving/trace_generator.py) | Low (lookup-based) |

---

### What Parts Are Modular ✅

1. **Operator Performance Database**
   - **Implementation**: [`trace_generator.py`](inference_serving/trace_generator.py#L26-L27)
   - **Why**: CSV-based lookup with caching
   - **Benefit**: Hardware-agnostic scheduler logic

2. **Model Configs**
   - **Location**: [`model_config/*.json`](model_config/)
   - **Why**: Declarative architecture specs (hidden_size, num_layers, etc.)
   - **Benefit**: No code changes to add new model architectures

3. **Power Modeling**
   - **Implementation**: [`power_model.py`](inference_serving/power_model.py)
   - **Why**: Decoupled energy accounting
   - **Benefit**: Optional, doesn't affect core simulation

4. **Request Routing**
   - **Implementation**: [`router.py`](inference_serving/router.py)
   - **Why**: Pluggable policies (`RR`, `RAND`, `CUSTOM`)
   - **Benefit**: Easy to add new routing algorithms

---

### What Parts Are Tightly Coupled ❌

1. **Memory Hierarchy**
   - **Location**: [`memory_model.py`](inference_serving/memory_model.py#L11-L314)
   - **Issue**: Enum-based dispatch, no interfaces
   - **Impact**: Adding tier requires changes to 4+ files

2. **Hardware-Specific Logic**
   - **Location**: [`trace_generator.py`](inference_serving/trace_generator.py#L279)
   - **Issue**: String-based checks (`if 'TPU' not in hardware`)
   - **Impact**: Fragile to hardware additions

3. **PIM Model**
   - **Location**: [`pim_model.py`](inference_serving/pim_model.py#L127)
   - **Issue**: Hardcoded for 4 DRAM types
   - **Impact**: Must update code for new PIM types

4. **Prefix Caching**
   - **Location**: [`memory_model.py`](inference_serving/memory_model.py#L47-L79)
   - **Issue**: Tangled with memory model initialization
   - **Impact**: Device-specific conditionals (CPU vs CXL)

5. **Graph Generation**
   - **Location**: [`graph_generator.py`](inference_serving/graph_generator.py)
   - **Issue**: Hardcoded dependency on Chakra converter
   - **Impact**: Locked to Astra-sim format

---

## Refactoring Recommendations

### 1. Introduce `DeviceModel` Interface

**Current Problem**: Enum-based dispatch limits extensibility.

**Proposed Solution** (in [`memory_model.py`](inference_serving/memory_model.py)):
```python
from abc import ABC, abstractmethod

class DeviceModel(ABC):
    @abstractmethod
    def allocate(self, size: int) -> None:
        """Allocate memory on this device."""
        pass
    
    @abstractmethod
    def free(self, size: int) -> None:
        """Free memory on this device."""
        pass
    
    @abstractmethod
    def get_bandwidth(self) -> float:
        """Return bandwidth in GB/s."""
        pass
    
    @abstractmethod
    def get_latency(self) -> float:
        """Return access latency in ns."""
        pass
    
    @abstractmethod
    def get_available(self) -> int:
        """Return available memory in bytes."""
        pass

class NPUMemory(DeviceModel):
    def __init__(self, capacity: int, bandwidth: float, latency: float):
        self.capacity = capacity
        self.used = 0
        self.bandwidth = bandwidth
        self.latency = latency
    
    def allocate(self, size: int) -> None:
        if self.used + size > self.capacity:
            raise RuntimeError(f"NPU OOM: {size} bytes requested")
        self.used += size
    
    def free(self, size: int) -> None:
        self.used -= size
    
    # ... implement other methods

class CPUMemory(DeviceModel):
    # Similar implementation
    pass

class CXLMemory(DeviceModel):
    # Similar implementation
    pass
```

**Benefits**:
- Adding new tier = create new class (no existing code changes)
- Polymorphic operations (no if-else chains)
- Testable in isolation

---

### 2. Extract `PerformanceLookup` Strategy

**Current Problem**: Hardcoded CSV path resolution.

**Proposed Solution** (in [`trace_generator.py`](inference_serving/trace_generator.py)):
```python
class PerformanceLookup(ABC):
    @abstractmethod
    def get_latency(self, op: str, params: dict) -> int:
        """Return operator latency in nanoseconds."""
        pass

class CSVLookup(PerformanceLookup):
    def __init__(self, hardware: str, model: str, tp: int):
        self.perf_db = self._load_csv(hardware, model, tp)
    
    def get_latency(self, op: str, params: dict) -> int:
        key = (op, params['input'], params['kv_cache'], params['tp'])
        return self.perf_db[key]['latency(ns)']

class AnalyticalLookup(PerformanceLookup):
    """For roofline models or FLOPs-based estimation."""
    def get_latency(self, op: str, params: dict) -> int:
        flops = self._compute_flops(op, params)
        compute_time = flops / self.tflops
        mem_time = params['data_size'] / self.bandwidth
        return max(compute_time, mem_time)

class MLLookup(PerformanceLookup):
    """XGBoost/neural network predictor."""
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
    
    def get_latency(self, op: str, params: dict) -> int:
        features = self._extract_features(op, params)
        return int(self.model.predict([features])[0])
```

**Benefits**:
- Supports multiple lookup strategies (CSV, analytical, ML)
- No hardcoded paths in core logic
- Easy to A/B test modeling approaches

---

### 3. Decouple PIM Models

**Current Problem**: Hardcoded dictionary in [`pim_model.py`](inference_serving/pim_model.py#L123).

**Proposed Solution**: Move to config file (`pim_config/*.json`):
```json
{
  "spec_name": "HBM2_1GB_2000_pim",
  "attn_model": {
    "slope": 242.0548,
    "intercept": 14513.5015
  },
  "dram_structure": {
    "protocol": "HBM",
    "bankgroups": 4,
    ...
  }
}
```

**Update** [`pim_model.py`](inference_serving/pim_model.py):
```python
def __init__(self, node_id, mem_size, pim_config_path):
    self.config = self._load_config(pim_config_path)
    self.attn_model = self.config.get("attn_model", {})
    
def get_pim_latency(self, n_head, kv_head, head_dim, L, channel_split=1):
    slope = self.attn_model["slope"]
    intercept = self.attn_model["intercept"]
    return (slope * L + intercept) / channel_split
```

**Benefits**:
- No code changes for new PIM types
- Config-driven extensibility

---

### 4. Unify Placement Logic

**Current Problem**: Scattered across:
- `placement` dict (layer → device mapping)
- `Device` enum
- `"LOCAL"/"REMOTE:{id}"` strings

**Proposed Solution**: Create `PlacementPolicy` class:
```python
class PlacementPolicy:
    def __init__(self, config: dict):
        self.tier_map = config  # layer → tier mapping
    
    def get_device(self, layer: str, node: int, instance: int) -> DeviceModel:
        """Return device object for given layer."""
        tier = self.tier_map.get(layer, "NPU")
        if tier == "NPU":
            return self.npu_memory[instance]
        elif tier == "CPU":
            return self.cpu_memory[node]
        # ...
    
    def get_location_string(self, layer: str, node: int) -> str:
        """Return Chakra-compatible location string."""
        if self.tier_map.get(layer) == "CPU":
            return f"REMOTE:{node}"
        return "LOCAL"
```

**Benefits**:
- Centralized placement decisions
- Type-safe device access
- Easy to implement new placement strategies

---

## Minimal Change Sets

### Adding a New Accelerator

**Effort**: 🟡 Medium (1-2 days profiling + config)

**Steps**:
1. **Profile operators** → Create `llm_profile/perf_models/{NEW_HW}/{model}/tp{N}/layers.csv`
   - Run profiling scripts on new hardware
   - Format: `layer_name,input,kv_cache,tp_size,latency(ns)`

2. **Add cluster config** → Edit/create [`cluster_config/*.json`](cluster_config/)
   ```json
   {
     "instances": [{
       "hardware": "NEW_ACCELERATOR",
       "npu_mem": {"mem_size": 80, "mem_bw": 2000, "mem_latency": 0}
     }]
   }
   ```

3. **(Optional) Power specs** → Same JSON
   ```json
   "power": {
     "npu": {
       "NEW_ACCELERATOR": {
         "idle_power": 50,
         "active_power": 300,
         "standby_power": 100,
         "standby_duration": 1000
       }
     }
   }
   ```

**No code changes required** — pure data/config extension.

---

### Adding a New Memory Tier

**Effort**: 🔴 High (1-2 days code + testing)

**Files to modify**:
1. [`memory_model.py`](inference_serving/memory_model.py#L11) — Extend `Device` enum
2. [`memory_model.py`](inference_serving/memory_model.py#L218) — Add `allocate()` case
3. [`memory_model.py`](inference_serving/memory_model.py#L266) — Add `free()` case
4. [`scheduler.py`](inference_serving/scheduler.py#L120-L185) — Update eviction logic
5. [`trace_generator.py`](inference_serving/trace_generator.py#L86-L111) — Add transfer trace entries
6. [`config_builder.py`](inference_serving/config_builder.py) — Parse from config

**Estimated changes**: ~100 lines across 5 files

---

### Adding Operator-Specific Cost Model

**Effort**: 🟢 Low (1-2 hours)

**Steps**:
1. **Add CSV entries** for new operator in perf database
2. **Insert lookup** in [`trace_generator.py::_synthesize_trace()`](inference_serving/trace_generator.py#L108-L500):
   ```python
   new_op_row = _get_perf_row(perf_db, hardware, "new_operator", 
                               input_len, kv_len, npus_per_group)
   block_res.append(formatter("new_operator", 
                              str(new_op_row['latency(ns)']), ...))
   ```
3. **(If needed)** Add to [`calculate_sizes()`](inference_serving/memory_model.py#L550) for data movement

**Estimated changes**: ~20 lines in 1-2 files

---

## Architectural Verdict

### **Strengths** ✅

1. **CSV-based operator modeling** is flexible and hardware-agnostic
2. **Scheduler logic** doesn't directly depend on hardware implementations
3. **Model configs** are declarative (JSON-based)
4. **Power modeling** is optional and decoupled

### **Weaknesses** ❌

1. **Memory tier system is rigid** (enum-based, requires multi-file changes)
2. **Hardware-specific conditionals** scattered in trace generator
3. **No abstract device interface** (prevents polymorphism)
4. **PIM models hardcoded** (not config-driven)

### **Path Forward** 🔧

**Recommended refactoring priorities**:

1. **Introduce `DeviceModel` interface** (memory_model.py)
   - Impact: High value, medium effort
   - Enables pluggable memory tiers

2. **Extract `PerformanceLookup` strategy** (trace_generator.py)
   - Impact: Medium value, low effort
   - Supports multiple modeling approaches

3. **Config-driven PIM models** (pim_model.py)
   - Impact: Low value, low effort
   - Quick win for extensibility

4. **Unify placement logic** (config_builder.py, memory_model.py)
   - Impact: Medium value, high effort
   - Reduces string-based device references

**After refactoring**: The simulator would achieve **true pluggability** with minimal changes for new hardware/memory tiers.

---

## File Reference Index

### Core Architecture Files

- [`main.py`](main.py) — Entry point, argument parsing
- [`inference_serving/scheduler.py`](inference_serving/scheduler.py) — Request batching, memory management
- [`inference_serving/controller.py`](inference_serving/controller.py) — Astra-sim communication
- [`inference_serving/memory_model.py`](inference_serving/memory_model.py) — Memory hierarchy, KV cache
- [`inference_serving/trace_generator.py`](inference_serving/trace_generator.py) — Operator latency synthesis
- [`inference_serving/graph_generator.py`](inference_serving/graph_generator.py) — Chakra graph conversion
- [`inference_serving/config_builder.py`](inference_serving/config_builder.py) — Cluster config parsing

### Configuration

- [`cluster_config/*.json`](cluster_config/) — Hardware specifications
- [`model_config/*.json`](model_config/) — Model architectures
- [`pim_config/*.ini`](pim_config/) — DRAM/PIM specifications

### Performance Database

- `llm_profile/perf_models/{hardware}/{model}/tp{tp}/layers.csv` — Operator latencies

### Supporting Modules

- [`inference_serving/pim_model.py`](inference_serving/pim_model.py) — PIM latency modeling
- [`inference_serving/power_model.py`](inference_serving/power_model.py) — Energy accounting
- [`inference_serving/router.py`](inference_serving/router.py) — Request routing
- [`inference_serving/request.py`](inference_serving/request.py) — Request/batch data structures
- [`inference_serving/utils.py`](inference_serving/utils.py) — Config loading, formatting

---

*End of Analysis*
