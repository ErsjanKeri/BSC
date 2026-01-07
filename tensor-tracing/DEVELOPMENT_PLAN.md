# Tensor Tracing - Comprehensive Development Plan
**Version:** 2.0
**Date:** 2026-01-07
**Status:** Planning Phase

---

## Executive Summary

This plan addresses **critical architectural issues** in the current tensor tracing implementation and establishes a roadmap for complete instrumentation of all operations, distinction between disk and buffer memory, and advanced visualization.

### Core Problems Identified:
1. **Data Synchronization**: We had logs for 5 tokens, but the graph for 1 token only, need to re-iterate the experiment for one token only and do our UI on that first
2. **Incomplete Memory Map**: Not using authoritative GGUF structure → missing tensors, incorrect offsets, because logs are only for mul_mat
3. **Incomplete Instrumentation**: Only `mul_mat` captured → missing ADD, ROPE, RMSNorm, embeddings, etc.
4. **No Memory Type Distinction**: Cannot differentiate disk-backed (GGUF) vs buffer (KV cache, activations)
5. **Heatmap Visualization Issues**: Overlapping tensors, incorrect positioning, hard to identify small tensors, coming from not proper partitioning in calculations from our part, need to carefully examine the gguf dumb and connect the logs with them 

### Solution Approach:
- **Phase 1**: Fix data collection infrastructure (complete instrumentation + buffer tracking)
- **Phase 2**: Fix memory map generation (use authoritative GGUF parser)
- **Phase 3**: Synchronize all data sources (single coherent experiment)
- **Phase 4**: Redesign heatmap visualization (dual-track: disk vs buffer)
- **Phase 5**: Advanced features (timeline analysis, correlation tools)

---

## PHASE 1: Complete Instrumentation & Buffer Tracking

**Goal:** Capture ALL tensor operations (not just mul_mat) and distinguish disk vs buffer memory.

**Duration:** ~1 week
**Priority:** CRITICAL

### 1.1: Generic Operation Instrumentation

**Current State:**
- File: `/Users/ersibesi/Desktop/LLAMA/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c`
- Lines: 1237-1268 (inside `ggml_compute_forward_mul_mat`)
- Only captures mul_mat operations

**Objective:** Instrument ALL operations at the operation dispatch level

**Implementation Strategy:**

**Step 1.1.1:** Locate Operation Dispatch Switch
```bash
# Find the main operation dispatch
cd /Users/ersibesi/Desktop/LLAMA/llama.cpp
grep -rn "switch.*->op" ggml/src/ggml-cpu/
grep -rn "ggml_compute_forward" ggml/src/ggml-cpu/
```

**Expected location:** Likely in `ggml_compute_forward_cpu()` or similar function that dispatches based on `op->type`.

**Step 1.1.2:** Add Generic Tracing Macro at Switch Entry
```c
// Inside ggml-cpu.c, at operation dispatch function
void ggml_compute_forward_cpu(
        const struct ggml_compute_params * params,
        struct ggml_tensor * tensor) {

    #ifdef GGML_TENSOR_TRACE
    if (params->ith == 0) {  // Only first thread logs
        // GENERIC TRACE CALL - captures all operations
        tensor_trace_log_operation(tensor, params);
    }
    #endif

    switch (tensor->op) {
        case GGML_OP_MUL_MAT:
            ggml_compute_forward_mul_mat(params, tensor);
            break;
        case GGML_OP_ADD:
            ggml_compute_forward_add(params, tensor);
            break;
        case GGML_OP_ROPE:
            ggml_compute_forward_rope(params, tensor);
            break;
        case GGML_OP_GET_ROWS:
            ggml_compute_forward_get_rows(params, tensor);
            break;
        // ... all other operations
    }
}
```

**Step 1.1.3:** Implement `tensor_trace_log_operation()` in `tensor_trace.c`
```c
// In tensor_trace.c
void tensor_trace_log_operation(
        const struct ggml_tensor * tensor,
        const struct ggml_compute_params * params) {

    struct TensorAccessLog entry = {0};
    entry.timestamp_ns = tensor_trace_get_timestamp_ns();
    entry.thread_id = tensor_trace_get_thread_id();

    // Determine operation type
    entry.operation_type = (uint8_t)tensor->op;  // Use ggml_op enum directly

    // Log ALL source tensors (inputs)
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] == NULL) continue;

        const struct ggml_tensor* src = tensor->src[i];

        // CRITICAL: Determine memory source (DISK vs BUFFER)
        entry.memory_source = tensor_trace_detect_memory_source(src->data);

        if (entry.memory_source == MEMORY_DISK) {
            entry.disk_offset = tensor_trace_get_gguf_offset(src->data);
            entry.buffer_id = 0;
        } else {
            entry.buffer_id = tensor_trace_get_buffer_id(src->data);
            entry.disk_offset = 0;
        }

        entry.tensor_ptr = (uint64_t)src->data;
        entry.size_bytes = (uint32_t)ggml_nbytes(src);
        strncpy(entry.tensor_name, src->name, sizeof(entry.tensor_name) - 1);
        entry.layer_id = tensor_trace_extract_layer_id(src->name);
        entry.tensor_idx = tensor_trace_lookup_idx(src->data);

        tensor_trace_log(&entry);
    }
}
```

**Benefits:**
✅ Captures ALL operations (ADD, ROPE, RMSNorm, embeddings, etc.)
✅ No need to modify each operation individually
✅ Future-proof: new operations automatically captured
✅ Single point of maintenance

**Testing:**
```bash
# Rebuild with new instrumentation
cmake --build build -j16

# Run test
rm -f /tmp/tensor_trace.bin
./build/bin/llama-cli -m model.gguf -p "test" -n 1

# Verify all operations captured
python3 tensor-tracing/tools/parse_trace.py /tmp/tensor_trace.bin --stats
# Expected: Should see MUL_MAT, ADD, ROPE, GET_ROWS, etc.
```

---

### 1.2: Memory Source Detection (Disk vs Buffer)

**Objective:** Distinguish between disk-backed GGUF parameters and runtime buffers.

**Step 1.2.1:** Add Memory Source Enum to trace format
here we remove some bytes from the 64 name buffer, and instead store there the memory source, also very carefully we adjust the parsing in the tools in our project as well (as currently it just reads the binary file and is used to the old format)

**File:** `/Users/ersibesi/Desktop/LLAMA/llama.cpp/ggml/include/tensor_trace.h`

```c
enum MemorySource {
    MEMORY_DISK = 0,   // GGUF file, mmap'd model parameters
    MEMORY_BUFFER = 1  // Runtime allocation (KV cache, intermediates)
};

struct TensorAccessLog {
    // ... existing fields ...
    uint8_t memory_source;   // NEW: MEMORY_DISK or MEMORY_BUFFER
    uint64_t buffer_id;      // NEW: Unique buffer ID (if BUFFER), else 0
    uint64_t disk_offset;    // UPDATED: If DISK: offset in GGUF, if BUFFER: 0
    // ... rest of fields ...
} __attribute__((packed));
```

**Step 1.2.2:** Implement Memory Source Detection, lets discuss this if this is the ideal best approach since it is kind of hardcoded 

**File:** `/Users/ersibesi/Desktop/LLAMA/llama.cpp/ggml/src/tensor_trace.c`

```c
// Global state: GGUF mmap range
static void* g_gguf_mmap_start = NULL;
static void* g_gguf_mmap_end = NULL;

// Initialize GGUF range during model load
void tensor_trace_set_gguf_range(void* start, void* end) {
    g_gguf_mmap_start = start;
    g_gguf_mmap_end = end;
}

// Detect if pointer is in GGUF mmap or runtime buffer
enum MemorySource tensor_trace_detect_memory_source(const void* ptr) {
    if (ptr >= g_gguf_mmap_start && ptr < g_gguf_mmap_end) {
        return MEMORY_DISK;  // Within GGUF mmap range
    }
    return MEMORY_BUFFER;  // Outside GGUF range → runtime allocation
}

// Get GGUF file offset for disk-backed tensors, careful: here might be problematic the fragmentation of the model file in the disk 
uint64_t tensor_trace_get_gguf_offset(const void* ptr) {
    if (g_gguf_mmap_start == NULL) return 0;
    return (uint64_t)((const char*)ptr - (const char*)g_gguf_mmap_start);
}
```

**Step 1.2.3:** Hook GGUF mmap initialization

**File:** Find where llama.cpp mmaps the GGUF file (likely `llama.cpp` or `llama-impl.h`)

```c
// After mmap() call for GGUF file:
void* gguf_data = mmap(...);
size_t gguf_size = ...;

#ifdef GGML_TENSOR_TRACE
tensor_trace_set_gguf_range(gguf_data, (char*)gguf_data + gguf_size);
#endif
```

**Testing:**
```bash
# Run trace
./build/bin/llama-cli -m model.gguf -p "test" -n 1

# Parse and verify memory sources
python3 tensor-tracing/tools/parse_trace.py /tmp/tensor_trace.bin --display

# Expected output:
# Entry #0: blk.0.attn_q.weight, DISK, offset=0x1A2B3C
# Entry #1: kv_cache_k, BUFFER, buffer_id=1
```

---

### 1.3: Buffer Tracking & Occupancy

**Objective:** Track runtime buffer allocations/deallocations and generate occupancy timeline.

**Design Decision Point:**

**Option A: Every allocation/deallocation (Precise)**
- ✅ Complete view of buffer lifecycle
- ✅ Precise occupancy at every moment
- ⚠️ Higher overhead (log on every malloc/free)
- ⚠️ Large log files

**Option B: Periodic sampling (Efficient)**
- ✅ Lower overhead
- ✅ Smaller log files
- ⚠️ Miss short-lived buffers
- ⚠️ Less precise

**RECOMMENDATION:** Option A (every alloc/dealloc) for now
- Reason: Research tool, need precision more than performance
- Can optimize later if overhead becomes issue

**Step 1.3.1:** Add Buffer Tracking Structure

**File:** `/Users/ersibesi/Desktop/LLAMA/llama.cpp/ggml/include/tensor_trace.h`

```c
struct BufferEvent {
    uint64_t timestamp_ns;
    uint8_t event_type;       // 0=ALLOC, 1=DEALLOC
    uint64_t buffer_id;       // Unique buffer identifier
    uint64_t buffer_ptr;      // Virtual address
    uint64_t size_bytes;      // Buffer size
    uint16_t layer_id;        // Associated layer (65535=N/A)
    char buffer_name[64];     // Name (e.g., "kv_cache_k_l5")
} __attribute__((packed));

void tensor_trace_log_buffer_alloc(uint64_t buffer_id, void* ptr, size_t size,
                                     const char* name, uint16_t layer_id);
void tensor_trace_log_buffer_dealloc(uint64_t buffer_id);
```

**Step 1.3.2:** Hook Buffer Allocations

Find buffer allocation points in llama.cpp:
```bash
cd /Users/ersibesi/Desktop/LLAMA/llama.cpp
grep -rn "ggml_backend_buffer_alloc" .
grep -rn "kv_cache" src/llama.cpp
```

**Expected locations:**
- KV cache allocation: `src/llama.cpp` (during context initialization)
- Scratch buffers: `ggml/src/ggml-backend-impl.h`

**Add instrumentation:**
```c
// In KV cache allocation code:
void* kv_cache_k = ggml_backend_buffer_alloc(...);

#ifdef GGML_TENSOR_TRACE
static uint64_t next_buffer_id = 1;
tensor_trace_log_buffer_alloc(
    next_buffer_id++,
    kv_cache_k,
    kv_cache_size,
    "kv_cache_k",
    layer_id
);
#endif
```

**Step 1.3.3:** Generate Buffer Stats Output

**File:** `tensor_trace.c`

```c
// Separate output file for buffer stats
static FILE* g_buffer_stats_file = NULL;

void tensor_trace_init_buffer_stats(void) {
    g_buffer_stats_file = fopen("/tmp/buffer_stats.json", "w");
    fprintf(g_buffer_stats_file, "{\"timeline\": [\n");
}

void tensor_trace_log_buffer_alloc(...) {
    // Log to buffer_stats.json
    fprintf(g_buffer_stats_file,
        "  {\"timestamp_ms\": %.3f, \"event\": \"alloc\", \"id\": %lu, "
        "\"name\": \"%s\", \"size\": %lu, \"layer\": %u},\n",
        timestamp_ns / 1e6, buffer_id, name, size, layer_id);
    fflush(g_buffer_stats_file);
}
```

**Output format:** `/tmp/buffer_stats.json`
```json
{
  "timeline": [
    {"timestamp_ms": 0.125, "event": "alloc", "id": 1, "name": "kv_cache_k", "size": 16777216, "layer": 0},
    {"timestamp_ms": 0.150, "event": "alloc", "id": 2, "name": "kv_cache_v", "size": 16777216, "layer": 0},
    {"timestamp_ms": 5.234, "event": "dealloc", "id": 1},
    {"timestamp_ms": 5.240, "event": "dealloc", "id": 2}
  ]
}
```

**Testing:**
```bash
# Run trace with buffer tracking
./build/bin/llama-cli -m model.gguf -p "test" -n 5

# Verify buffer stats generated
cat /tmp/buffer_stats.json

# Expected: Should see kv_cache allocations, scratch buffers, etc.
```

**Discussion Point: Buffer ID Assignment**

**Question:** How to assign unique buffer IDs?

**Options:**
1. **Sequential counter** (next_buffer_id++)
   - ✅ Simple, deterministic
   - ⚠️ Requires global state

2. **Pointer address as ID** (buffer_id = (uint64_t)ptr)
   - ✅ No global state needed
   - ⚠️ Reused addresses after free

3. **Hash of (ptr + timestamp)**
   - ✅ Unique even with reuse
   - ⚠️ Overkill complexity

**RECOMMENDATION:** Option 1 (sequential counter)
- Simple, works well for analysis
- Global state is fine for single-process tracing

---

## PHASE 2: Authoritative Memory Map Generation

**Goal:** Use official GGUF structure to generate accurate memory map.

**Duration:** ~2 days
**Priority:** HIGH

### 2.1: GGUF Parsing with gguf-dump

**Current State:**
- Command: `./build/bin/gguf-dump /path/to/model.gguf --csv > model_structure.csv`
- Already documented and used in experiments

**Step 2.1.1:** Verify gguf-dump Output Format
```bash
cd /Users/ersibesi/Desktop/LLAMA/llama.cpp
./build/bin/gguf-dump ../models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --csv | head -20
```

**Expected columns:**
- Tensor name
- Shape
- Type (dtype)
- Offset (bytes from file start)
- Size (bytes)

**Step 2.1.2:** Write Enhanced Parser Script

**File:** `/Users/ersibesi/Desktop/LLAMA/BSC/tensor-tracing/tools/generate_memory_map.py`

```python
#!/usr/bin/env python3
"""
Generate accurate memory_map.json from gguf-dump CSV output
"""
import csv
import json
import sys
import re

def infer_layer_from_name(name):
    """Extract layer ID from tensor name (e.g., 'blk.5.attn_q' → 5)"""
    match = re.search(r'blk\.(\d+)\.', name)
    return int(match.group(1)) if match else None

def infer_category(name):
    """Categorize tensor by name pattern"""
    if 'token_embd' in name or 'embd' in name:
        return 'input'
    elif 'attn' in name:
        return 'attention'
    elif 'ffn' in name:
        return 'ffn'
    elif 'norm' in name:
        return 'norm'
    elif 'output' in name:
        return 'output'
    return 'other'

def parse_gguf_csv(csv_path):
    """Parse gguf-dump CSV and generate memory map"""
    tensors = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tensor = {
                "name": row['name'],
                "offset_start": int(row['offset']),
                "offset_end": int(row['offset']) + int(row['size']),
                "size_bytes": int(row['size']),
                "shape": eval(row['shape']),  # Convert "[4096, 4096]" to list
                "dtype": row['type'],
                "layer_id": infer_layer_from_name(row['name']),
                "category": infer_category(row['name'])
            }
            tensors.append(tensor)

    # Sort by offset for sequential layout
    tensors.sort(key=lambda t: t['offset_start'])

    # Verify no overlaps or gaps
    for i in range(len(tensors) - 1):
        if tensors[i]['offset_end'] != tensors[i+1]['offset_start']:
            print(f"WARNING: Gap/overlap between {tensors[i]['name']} and {tensors[i+1]['name']}")

    total_size = tensors[-1]['offset_end'] if tensors else 0
    n_layers = max((t['layer_id'] for t in tensors if t['layer_id'] is not None), default=0) + 1

    return {
        "total_size_bytes": total_size,
        "num_tensors": len(tensors),
        "metadata": {
            "n_layers": n_layers,
            "source": "gguf-dump"
        },
        "tensors": tensors
    }

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: generate_memory_map.py <input.csv> <output.json>")
        sys.exit(1)

    memory_map = parse_gguf_csv(sys.argv[1])

    with open(sys.argv[2], 'w') as f:
        json.dump(memory_map, f, indent=2)

    print(f"Generated memory_map.json: {memory_map['num_tensors']} tensors, "
          f"{memory_map['total_size_bytes'] / 1e9:.2f} GB")
```

**Step 2.1.3:** Integrate into Workflow

**File:** `/Users/ersibesi/Desktop/LLAMA/BSC/tensor-tracing/scripts/run_experiment.sh`

```bash
#!/bin/bash
# Complete experiment workflow

MODEL_PATH="$1"
PROMPT="$2"
NUM_TOKENS="${3:-1}"

# Step 1: Generate GGUF memory map
echo "Generating memory map from GGUF..."
./build/bin/gguf-dump "$MODEL_PATH" --csv > /tmp/model_structure.csv
python3 tensor-tracing/tools/generate_memory_map.py /tmp/model_structure.csv /tmp/memory_map.json

# Step 2: Run traced inference
echo "Running traced inference..."
rm -f /tmp/tensor_trace.bin /tmp/buffer_stats.json
./build/bin/llama-cli -m "$MODEL_PATH" -p "$PROMPT" -n "$NUM_TOKENS"

# Step 3: Parse outputs
echo "Parsing trace..."
python3 tensor-tracing/tools/parse_trace.py /tmp/tensor_trace.bin --stats
```

---

## PHASE 3: Data Synchronization

**Goal:** Ensure all data sources come from same experiment.

**Duration:** ~1 day
**Priority:** HIGH

### 3.1: Single Coherent Experiment

**Step 3.1.1:** Add Experiment Metadata

**File:** `tensor_trace.c`

```c
void tensor_trace_init(void) {
    // ... existing init ...

    // Write experiment metadata
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", tm_info);

    FILE* metadata = fopen("/tmp/experiment_metadata.json", "w");
    fprintf(metadata, "{\n");
    fprintf(metadata, "  \"experiment_id\": \"%ld\",\n", now);
    fprintf(metadata, "  \"timestamp_start\": \"%s\",\n", timestamp);
    fprintf(metadata, "  \"model_file\": \"%s\",\n", g_model_path);
    fprintf(metadata, "  \"num_tokens_generated\": %d\n", g_num_tokens);
    fprintf(metadata, "}\n");
    fclose(metadata);
}
```

**Step 3.1.2:** Modified Run Script

```bash
#!/bin/bash
# run_synchronized_experiment.sh

EXPERIMENT_ID=$(date +%s)
OUTPUT_DIR="/tmp/experiments/${EXPERIMENT_ID}"
mkdir -p "$OUTPUT_DIR"

# Generate memory map
./build/bin/gguf-dump "$MODEL" --csv > "$OUTPUT_DIR/model_structure.csv"
python3 tools/generate_memory_map.py "$OUTPUT_DIR/model_structure.csv" "$OUTPUT_DIR/memory_map.json"

# Run traced inference (outputs go to $OUTPUT_DIR)
TENSOR_TRACE_DIR="$OUTPUT_DIR" ./build/bin/llama-cli -m "$MODEL" -p "$PROMPT" -n "$NUM_TOKENS"

# Copy experiment metadata
cp /tmp/experiment_metadata.json "$OUTPUT_DIR/"

# Verify all files present
ls "$OUTPUT_DIR"/tensor_trace.bin
ls "$OUTPUT_DIR"/buffer_stats.json
ls "$OUTPUT_DIR"/memory_map.json
ls "$OUTPUT_DIR"/experiment_metadata.json

echo "Experiment $EXPERIMENT_ID complete. Data in: $OUTPUT_DIR"
```

**Step 3.1.3:** Webui Validation

**File:** `webui/src/utils/dataLoader.ts`

```typescript
export async function loadExperiment(directory: string) {
  // Load metadata
  const metadata = await loadJSON(`${directory}/experiment_metadata.json`);

  // Load all data files
  const memoryMap = await loadJSON(`${directory}/memory_map.json`);
  const traceData = await loadBinary(`${directory}/tensor_trace.bin`);
  const bufferStats = await loadJSON(`${directory}/buffer_stats.json`);
  const graphData = await loadJSON(`${directory}/computation_graph.json`);

  // Validate all have same experiment_id
  const ids = [
    memoryMap.experiment_id,
    traceData.experiment_id,
    bufferStats.experiment_id,
    graphData.experiment_id
  ];

  if (new Set(ids).size > 1) {
    throw new Error("Data files from different experiments! IDs: " + ids.join(", "));
  }

  return { metadata, memoryMap, traceData, bufferStats, graphData };
}
```

---

## PHASE 4: Heatmap Visualization Redesign

**Goal:** Dual-track heatmap (disk vs buffer) with precise tensor identification.

**Duration:** ~3 days
**Priority:** MEDIUM

### 4.1: Dual-Track Heatmap Architecture

**Minimized View:**
```
┌─────────────────────────────────────────────────┐
│ Memory Heatmap (Disk + Buffer)                  │
├─────────────────────────────────────────────────┤
│ DISK (GGUF):  [──────────50px bar──────────]   │ ← Tensor layout by offset
│ BUFFER:       [──────────50px bar──────────]   │ ← Buffer allocations
└─────────────────────────────────────────────────┘
```

**Full-Screen View:**
```
┌─────────────────────────────────────────────────┐
│ Memory Heatmap                                  │
├─────────────────────────────────────────────────┤
│                                                 │
│ DISK (GGUF File Layout):                       │
│ L0:  [─────bar─────]  ← Layer 0 tensors       │
│ L1:  [─────bar─────]  ← Layer 1 tensors       │
│ ...                                             │
│                                                 │
│ BUFFER (Runtime Allocations):                  │
│ KV:  [─────bar─────]  ← KV cache              │
│ SCR: [─────bar─────]  ← Scratch buffers       │
│                                                 │
├─────────────────────────────────────────────────┤ ← 70% split
│ Trace Log (Filtered)                  [150 entries] │
│ #  Time      Op       Tensor         Layer Size    │
│ 0  0.125ms   MUL_MAT  attn_q.weight  5     2.5MB   │
│ 1  0.130ms   ADD      ...            5     1.0MB   │
└─────────────────────────────────────────────────┘
```

### 4.2: Tensor Identification Improvements

**For Small Tensors (< 5px width):**

**Technique 1: Minimum Width Rendering**
```typescript
const width = Math.max((endX - startX), 5);  // At least 5px
```

**Technique 2: Hover Magnification**
```typescript
if (hoveredTensor && width < 10) {
  // Draw magnified overlay
  const magnification = 5;
  const magWidth = width * magnification;
  drawMagnifiedRegion(hoveredTensor, magWidth);
}
```

**Technique 3: Arrow Pointer**
```html
<div class="tensor-pointer" style="left: ${exactX}px">
  ↓
  <div class="tooltip">
    attn_q.weight
    2.5 MB ████████
    Layer 5
  </div>
</div>
```

**Implementation:** Already partially done in current HeatmapView, needs refinement for small tensors.

---

## PHASE 5: Testing & Validation

**Goal:** Verify all components working together.

**Duration:** ~2 days
**Priority:** HIGH

### 5.1: Test Matrix

| Test Case | Input | Expected Output | Status |
|-----------|-------|-----------------|--------|
| Single token | -n 1 | All ops captured, 22/22 layers | ⏳ |
| Multi-token | -n 5 | Sequential pattern visible | ⏳ |
| Buffer tracking | -n 5 | KV cache alloc/dealloc logged | ⏳ |
| Memory map | GGUF | Exact tensor positions, no gaps | ⏳ |
| Data sync | Full experiment | All files same experiment_id | ⏳ |
| Heatmap | Webui | Disk+buffer tracks, no overlaps | ⏳ |

### 5.2: Validation Scripts

**File:** `tensor-tracing/tools/validate_experiment.py`

```python
def validate_experiment(experiment_dir):
    """Validate experiment data integrity"""

    # Check all files exist
    required = ['tensor_trace.bin', 'buffer_stats.json', 'memory_map.json', 'experiment_metadata.json']
    for file in required:
        assert os.path.exists(f"{experiment_dir}/{file}"), f"Missing {file}"

    # Check experiment_id consistency
    # ... (as shown in Phase 3.1.3)

    # Check memory map completeness
    memory_map = load_json(f"{experiment_dir}/memory_map.json")
    assert len(memory_map['tensors']) > 0, "Empty memory map"

    # Check trace has entries
    trace = parse_trace(f"{experiment_dir}/tensor_trace.bin")
    assert len(trace) > 0, "Empty trace"

    # Check buffer stats
    buffers = load_json(f"{experiment_dir}/buffer_stats.json")
    assert len(buffers['timeline']) > 0, "No buffer events"

    print("✅ All validations passed")
```

---

## Implementation Timeline

| Phase | Duration | Dependencies | Deliverables |
|-------|----------|--------------|-------------|
| **Phase 1** | 1 week | None | Complete instrumentation, buffer tracking |
| **Phase 2** | 2 days | None (parallel) | Accurate memory map generator |
| **Phase 3** | 1 day | Phase 1, 2 | Synchronized experiment workflow |
| **Phase 4** | 3 days | Phase 3 | Dual-track heatmap visualization |
| **Phase 5** | 2 days | Phase 1-4 | Validation suite, documentation |

**Total:** ~2 weeks

---

## Open Questions & Discussion Points

### Q1: Buffer ID Assignment Strategy
**Options:** Sequential counter vs pointer address vs hash
**Recommendation:** Sequential counter (simple, deterministic)
**Decision:** [ ] Approved [ ] Needs discussion

### Q2: Buffer Tracking Granularity
**Options:** Every alloc/dealloc vs periodic sampling
**Recommendation:** Every alloc/dealloc (precision > performance for research)
**Decision:** [ ] Approved [ ] Needs discussion

### Q3: Heatmap Layout (Full-Screen)
**Options:**
- A) By layer (vertically stacked, per-layer disk+buffer)
- B) By memory type (all disk, then all buffer)

**Recommendation:** Option B (clearer separation)
**Decision:** [ ] Approved [ ] Needs discussion

### Q4: Testing Strategy
**Options:**
- A) Test with 1 token only (clean, simple)
- B) Test with 5 tokens (more data, patterns visible)

**Recommendation:** Both (1 token for validation, 5 tokens for analysis)
**Decision:** [ ] Approved [ ] Needs discussion

### Q5: Computation Graph Generation
**Current:** Separate mechanism, unclear how it's generated
**Question:** How do we ensure graph is from same experiment?
**Action:** Need to investigate graph generation in llama.cpp
**Decision:** [ ] Investigate first [ ] Already understood

---

## Success Criteria

Phase 1 Complete when:
- ✅ All operation types captured (not just mul_mat)
- ✅ Memory source correctly identified (disk vs buffer)
- ✅ Buffer allocation/deallocation events logged
- ✅ Test run shows MUL_MAT, ADD, ROPE, GET_ROWS, etc.

Phase 2 Complete when:
- ✅ Memory map matches GGUF structure exactly
- ✅ No gaps or overlaps in tensor positions
- ✅ All tensors accounted for

Phase 3 Complete when:
- ✅ Single experiment generates all data files
- ✅ All files have same experiment_id
- ✅ Webui validates data consistency

Phase 4 Complete when:
- ✅ Dual-track heatmap renders correctly
- ✅ Small tensors identifiable (arrow pointer + magnification)
- ✅ Trace log panel filters by memory region

Phase 5 Complete when:
- ✅ All test cases pass
- ✅ Validation suite runs without errors
- ✅ Documentation updated

---

## Next Actions

1. **Review this plan** with user, get approval on decisions
2. **Phase 1.1.1:** Locate operation dispatch switch in ggml-cpu.c
3. **Phase 1.2.1:** Add memory source enum to trace format
4. **Phase 2.1.1:** Verify gguf-dump output format
5. **Start implementation** following phased approach

---

**Document Status:** Draft - Awaiting User Review
**Last Updated:** 2026-01-07
