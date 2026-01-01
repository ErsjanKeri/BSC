# TENSOR ACCESS TRACKER - Implementation Plan

**Project Goal**: Build a comprehensive tool to track, visualize, and analyze tensor access patterns during LLM inference in llama.cpp, enabling deep understanding of hot/cold parameters, memory access patterns, and optimization opportunities.

**Status**: Planning Phase
**Author**: Claude + Ersjan Këri
**Date**: January 1, 2026

---

## Table of Contents
1. [Existing Tools Research](#existing-tools-research)
2. [Why We Need This Tool](#why-we-need-this-tool)
3. [Core Requirements](#core-requirements)
4. [Async Logging Strategy Discussion](#async-logging-strategy-discussion)
5. [Data Recording Format](#data-recording-format)
6. [Instrumentation Strategy](#instrumentation-strategy)
7. [Implementation Phases](#implementation-phases)
8. [Visualization & Analysis Plan](#visualization--analysis-plan)
9. [Testing & Validation](#testing--validation)
10. [Timeline & Milestones](#timeline--milestones)

---

## 1. Existing Tools Research

### 1.1 What Currently Exists

**alphaXiv Tensor Trace** ([link](https://www.alphaxiv.org/labs/tensor-trace))
- Interactive 3D visualization of GGML tensor operations
- Shows tensor flow through layers with shapes
- **Limitation**: Static analysis only, not runtime tracking
- **Verdict**: Useful for understanding graph structure, but doesn't track actual access patterns

**llama-bench** ([GitHub](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md))
- Official performance testing tool for llama.cpp
- Outputs CSV/JSON/markdown with performance metrics
- **Limitation**: High-level benchmarking only (tokens/sec, latency), no tensor-level detail
- **Verdict**: Good for performance comparison, not for access pattern analysis

**GGML Computation Graph Logging** ([GitHub Issue #11039](https://github.com/ggml-org/llama.cpp/discussions/11039))
- Can log computation graph to CSV file
- **Limitation**: Terminates immediately after logging, not during inference
- **Verdict**: Shows graph structure but not runtime access patterns

**GGML_PERF** ([Removed in PR #8017](https://github.com/ggml-org/llama.cpp/discussions/6871))
- Was a built-in performance profiling feature
- Users now resort to Linux `perf` tool
- **Limitation**: `perf` is sampling-based, not granular, doesn't know tensor semantics
- **Verdict**: Too coarse-grained, missing semantic information

**NVIDIA Nsight/DLProf** ([NVIDIA Blog](https://developer.nvidia.com/blog/profiling-and-optimizing-deep-neural-networks-with-dlprof-and-pyprof/))
- GPU-focused profiling with TensorBoard integration
- Shows kernel execution, memory transfers
- **Limitation**: GPU-only, not applicable to CPU inference on hiwi-02
- **Verdict**: Not usable for our CPU-based setup

**Arm Streamline** ([ARM Learning Path](https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama_cpp_streamline/2_llama.cpp_intro/))
- Profiler for ARM-based systems
- Can profile llama.cpp inference workflow
- **Limitation**: General-purpose profiler, no tensor-level semantics
- **Verdict**: Could be complementary but doesn't solve our core need

### 1.2 Critical Gap in Existing Tools

**What's Missing:**
- ✗ Real-time tensor access tracking during inference
- ✗ Per-token, per-layer, per-operation granularity
- ✗ Attention head-level detail (which Q/K/V matrices accessed when)
- ✗ MoE expert activation tracking (which experts accessed per token)
- ✗ Memory access pattern visualization (temporal + spatial)
- ✗ File offset correlation (which bytes in GGUF file accessed)
- ✗ Page fault correlation (which accesses caused disk I/O)

**Why Academic Tools Don't Help:**
- [Graph Neural Network Memory Access Patterns](https://dl.acm.org/doi/abs/10.1145/3624062.3624168): Focused on GNN, not transformers
- [NTP Neural Network Topology Profiler](https://arxiv.org/abs/1905.09063): Architecture analysis, not runtime access tracking
- Most research uses simulators or custom frameworks, not production inference engines

### 1.3 Conclusion

**We need to build this tool ourselves.** No existing tool provides:
1. Tensor-level semantic tracking during real inference
2. Correlation with GGUF file structure
3. Detailed attention mechanism tracking
4. MoE-specific instrumentation
5. Integration with llama.cpp's actual execution flow

This validates the project's novelty and importance.

---

## 2. Why We Need This Tool

### 2.1 Primary Research Questions

**Q1: Which parameters are hot vs cold?**
- Under memory pressure, which tensors get re-read from SSD?
- Are embeddings accessed more frequently than layer weights?
- Do early layers get re-read more than late layers?

**Q2: What are the temporal access patterns?**
- Is access sequential (layer-by-layer) or random?
- Do MoE models show different patterns than standard transformers?
- Can we predict next access to enable prefetching?

**Q3: How do attention mechanisms access memory?**
- Which Q/K/V matrices are accessed in what order?
- Do different attention heads access memory differently?
- How does KV cache affect parameter access?

**Q4: What about MoE expert activation?**
- Which experts are "hot" (frequently activated)?
- Is there expert specialization (certain experts for certain tokens)?
- Can we cache hot experts to reduce I/O?

**Q5: Where are optimization opportunities?**
- What percentage of accesses could be eliminated by caching?
- Which layers are bottlenecks?
- Could we reorder computations to improve locality?

### 2.2 Three-Phase Workflow

**Phase 1: RECORDING** (Our Focus Now)
- Instrument llama.cpp to log every tensor access
- Capture: timestamp, token_id, layer_id, tensor_name, operation, size, offsets
- Output: High-fidelity binary log files

**Phase 2: VISUALIZATION** (Trivial Once Data is Clean)
- Parse logs into pandas DataFrames
- Generate heatmaps, timelines, statistics
- Interactive exploration (zoom, filter, animate)

**Phase 3: ANALYSIS** (Driven by Observations)
- Identify hot/cold parameters
- Quantify access patterns (sequential %, reuse distance)
- Propose optimization strategies based on findings

### 2.3 Success Criteria

**Minimum Viable Product (MVP):**
- ✅ Log every tensor access with full context
- ✅ Correlate accesses with GGUF file structure
- ✅ Generate basic visualizations (heatmap, timeline)
- ✅ Support standard transformers AND MoE models

**Extended Goals:**
- ✅ Attention head-level tracking
- ✅ Expert activation tracking for MoE
- ✅ Page fault correlation (with blktrace)
- ✅ Interactive visualization dashboard
- ✅ Animated access pattern playback

**The key insight**: Recording must be PERFECT. Visualization is straightforward once data is clean.

---

## 3. Core Requirements

### 3.1 Functional Requirements

**FR1: Comprehensive Logging**
- Track EVERY tensor access (no sampling)
- Capture full context: what, when, where, why
- Support multi-token inference (prompt + generation)

**FR2: Semantic Awareness**
- Know tensor names (e.g., "blk.5.attn_q.weight")
- Understand operation types (MUL_MAT, ADD, ROPE, etc.)
- Track layer progression and token boundaries

**FR3: Attention Mechanism Detail**
- Separate tracking for Q, K, V matrices
- Track attention head indices (for multi-head attention)
- Log KV cache accesses vs fresh computations

**FR4: MoE Support**
- Track expert selection (routing decisions)
- Log which experts activated per token
- Capture expert-specific weight accesses

**FR5: File Offset Correlation**
- Map tensor accesses → GGUF file byte offsets
- Enable correlation with blktrace block-level I/O
- Support analysis of sequential vs random access

**FR6: Low Recording Overhead**
- Async logging to minimize impact on inference
- Binary format for efficiency
- Batch writes to reduce syscalls

### 3.2 Non-Functional Requirements

**NFR1: Performance**
- Instrumentation overhead: acceptable (don't care about exact %, but should be measureable)
- Async logging: writes shouldn't block inference
- Scalable to 100+ tokens with large models

**NFR2: Maintainability**
- Clean code with `#ifdef TENSOR_TRACE` guards
- Minimal invasive changes to llama.cpp
- Easy to sync with upstream llama.cpp updates

**NFR3: Usability**
- Simple enable/disable via compile flag
- Output files are self-documenting
- Visualization scripts are standalone

**NFR4: Correctness**
- No missed accesses (complete trace)
- Accurate timestamps (monotonic clock)
- Validated against known patterns

---

## 4. Async Logging Strategy Discussion

### 4.1 The Challenge

**Problem**: Logging every tensor access synchronously would be SLOW.
- For 100 tokens × 32 layers × ~20 operations/layer = 64,000+ log entries
- Each `fprintf()` is a syscall (~300 cycles overhead)
- Disk I/O blocks the inference thread

**Goal**: Decouple logging from inference execution.

### 4.2 Option A: Memory-Mapped Ring Buffer

**How It Works:**
```c
// Pre-allocate large mmap'd file (e.g., 1 GB)
int log_fd = open("trace.bin", O_RDWR | O_CREAT, 0644);
ftruncate(log_fd, 1024*1024*1024); // 1 GB
void* log_buffer = mmap(NULL, 1024*1024*1024, PROT_WRITE, MAP_SHARED, log_fd, 0);

// Write log entries directly to mmap'd memory
struct LogEntry* entry = (struct LogEntry*)(log_buffer + offset);
*entry = {timestamp, token_id, ...};
offset += sizeof(LogEntry);

// OS flushes to disk asynchronously
```

**Pros:**
- Very fast writes (just memory copy)
- OS handles async disk writes
- Simple implementation

**Cons:**
- Fixed-size buffer (must estimate max trace size)
- If buffer fills, must stop logging or overwrite
- Memory overhead (1 GB mmap'd file)

**Verdict:** ⭐ **RECOMMENDED for MVP** - Simple, fast, good enough for our use case.

### 4.3 Option B: Lock-Free Queue + Background Thread

**How It Works:**
```c
// Main thread: push to lock-free queue
struct SPSCQueue {
    LogEntry entries[1024*1024];
    atomic_size_t head;
    atomic_size_t tail;
};

void log_access(...) {
    size_t tail = atomic_load(&queue.tail);
    queue.entries[tail % QUEUE_SIZE] = {...};
    atomic_store(&queue.tail, tail + 1);
}

// Background thread: consume queue and write to disk
void* logger_thread(void* arg) {
    while (running) {
        while (head < tail) {
            LogEntry* entry = &queue.entries[head % QUEUE_SIZE];
            fwrite(entry, sizeof(LogEntry), 1, log_file);
            head++;
        }
        fflush(log_file);
        usleep(1000); // 1ms sleep
    }
}
```

**Pros:**
- Unbounded logging (can write unlimited data)
- Inference thread never blocks on I/O
- Can compress or filter logs in background

**Cons:**
- More complex (thread synchronization)
- Potential queue overflow if logging too fast
- Requires pthreads

**Verdict:** 🔧 **For Future Enhancement** - Better for production, but overkill for MVP.

### 4.4 Option C: Batched Synchronous Writes

**How It Works:**
```c
LogEntry buffer[1024]; // In-memory buffer
size_t buffer_idx = 0;

void log_access(...) {
    buffer[buffer_idx++] = {...};

    if (buffer_idx >= 1024) {
        fwrite(buffer, sizeof(LogEntry), 1024, log_file);
        buffer_idx = 0;
    }
}
```

**Pros:**
- Simple to implement
- Reduces syscalls by 1000×
- No threading complexity

**Cons:**
- Still blocks on fwrite() every 1024 entries
- Potential data loss if crash before flush

**Verdict:** 👍 **Good Fallback** - If mmap has issues, use this.

### 4.5 Recommended Approach

**Phase 1 (MVP):** Use **Option A (mmap ring buffer)**
- Simplest to implement
- Fast enough for our needs
- Pre-allocate 2 GB mmap file (enough for ~100M log entries)

**Phase 2 (If Needed):** Upgrade to **Option B (lock-free queue)**
- If we need unlimited logging
- If we want to add filtering/compression
- If overhead becomes measurable

**Implementation Plan:**
```c
// Global state
static void* g_log_buffer = NULL;
static size_t g_log_offset = 0;
static size_t g_log_capacity = 0;

// Initialize at startup
void tensor_trace_init(const char* filename, size_t capacity_bytes) {
    int fd = open(filename, O_RDWR | O_CREAT, 0644);
    ftruncate(fd, capacity_bytes);
    g_log_buffer = mmap(NULL, capacity_bytes, PROT_WRITE, MAP_SHARED, fd, 0);
    g_log_capacity = capacity_bytes;
    g_log_offset = 0;
}

// Log entry (inline, fast)
static inline void tensor_trace_log(const LogEntry* entry) {
    if (g_log_offset + sizeof(LogEntry) > g_log_capacity) {
        fprintf(stderr, "TRACE BUFFER FULL!\n");
        return;
    }
    memcpy(g_log_buffer + g_log_offset, entry, sizeof(LogEntry));
    g_log_offset += sizeof(LogEntry);
}

// Cleanup
void tensor_trace_shutdown() {
    msync(g_log_buffer, g_log_offset, MS_SYNC); // Force flush
    munmap(g_log_buffer, g_log_capacity);
}
```

**Why This Works:**
- Write = single `memcpy()` (no syscall)
- OS flushes to disk in background
- Can write millions of entries/sec
- Simple error handling (just check capacity)

---

## 5. Data Recording Format

### 5.1 Binary Log Format (Efficiency First)

**LogEntry Structure:**
```c
struct TensorAccessLog {
    // Timestamp
    uint64_t timestamp_ns;        // Nanoseconds since trace start (8 bytes)

    // Context
    uint32_t token_id;            // Which token being processed (4 bytes)
    uint16_t layer_id;            // Which transformer layer (0-based, 2 bytes)
    uint8_t  operation_type;      // Enum: MUL_MAT, ADD, ROPE, etc. (1 byte)
    uint8_t  phase;               // Enum: PROMPT, GENERATE (1 byte)

    // Tensor identification
    uint32_t tensor_idx;          // Index into tensor name table (4 bytes)
    uint64_t tensor_ptr;          // Virtual address of tensor->data (8 bytes)
    uint64_t file_offset;         // Offset in GGUF file (8 bytes)
    uint32_t size_bytes;          // Size of access in bytes (4 bytes)

    // Attention-specific (if applicable)
    uint8_t  attention_head;      // Which attention head (0-127, or 255=N/A) (1 byte)
    uint8_t  qkv_type;            // Enum: Q, K, V, O, or N/A (1 byte)
    uint8_t  padding[2];          // Align to 8 bytes (2 bytes)

    // MoE-specific (if applicable)
    uint8_t  expert_id;           // Which expert (0-255, or 255=N/A) (1 byte)
    uint8_t  expert_rank;         // Routing rank (0=top expert, 1=second, etc.) (1 byte)
    uint16_t routing_score;       // Quantized routing score (0-65535) (2 bytes)
    uint32_t padding2;            // Align to 8 bytes (4 bytes)
};
// Total size: 64 bytes per entry
```

**Design Rationale:**
- **Fixed-size records**: Fast random access for analysis
- **64-byte alignment**: Cache-friendly, easy to index
- **Compact encoding**: 100k entries = 6.4 MB (reasonable)
- **Separate tensor name table**: Avoids repeating long strings

**Tensor Name Table (Separate File):**
```
# tensor_names.txt
0,token_embd.weight
1,output_norm.weight
2,blk.0.attn_norm.weight
3,blk.0.attn_q.weight
4,blk.0.attn_k.weight
...
```

### 5.2 Enums and Constants

```c
// Operation types
enum OperationType {
    OP_NONE = 0,
    OP_MUL_MAT = 1,
    OP_ADD = 2,
    OP_RMS_NORM = 3,
    OP_ROPE = 4,
    OP_MUL = 5,
    OP_CPY = 6,
    OP_RESHAPE = 7,
    OP_VIEW = 8,
    OP_PERMUTE = 9,
    OP_CONT = 10,
    OP_FLASH_ATTN = 11,
    OP_SWIGLU = 12,
    // ... add more as needed
};

// Inference phase
enum InferencePhase {
    PHASE_PROMPT = 0,    // Processing input prompt
    PHASE_GENERATE = 1,  // Generating tokens
};

// Q/K/V type
enum QKVType {
    QKV_NONE = 0,
    QKV_QUERY = 1,
    QKV_KEY = 2,
    QKV_VALUE = 3,
    QKV_OUTPUT = 4,
};
```

### 5.3 Metadata File (JSON)

**trace_metadata.json:**
```json
{
    "trace_version": "1.0",
    "start_time_utc": "2026-01-01T12:00:00Z",
    "model_file": "/path/to/model.gguf",
    "model_metadata": {
        "architecture": "llama",
        "num_layers": 32,
        "num_heads": 32,
        "hidden_dim": 4096,
        "vocab_size": 32000
    },
    "inference_config": {
        "prompt": "Tell me a story",
        "num_tokens_generated": 100,
        "temperature": 0.7,
        "top_k": 40
    },
    "system_info": {
        "hostname": "cli-hiwi-02.dis.cit.tum.de",
        "cpu": "AMD Ryzen 7 7700X",
        "ram_gb": 30,
        "ssd": "WUS4BB096D7P3E3"
    },
    "trace_stats": {
        "total_entries": 87234,
        "duration_seconds": 12.456,
        "log_file_size_mb": 5.58
    }
}
```

### 5.4 File Organization

**Output Directory Structure:**
```
traces/
├── run_20260101_120000/
│   ├── trace_metadata.json       # High-level info
│   ├── tensor_names.txt           # Tensor name lookup table
│   ├── tensor_access.bin          # Binary log (main data)
│   ├── gguf_structure.csv         # GGUF file map (offsets, sizes)
│   └── analysis/                  # Generated visualizations
│       ├── heatmap_token_layer.png
│       ├── timeline_offset.png
│       └── statistics.json
```

---

## 6. Instrumentation Strategy

### 6.1 Where to Hook Into llama.cpp

**Key Files to Modify:**

**6.1.1 GGML Operation Dispatch (Core Instrumentation)**
- **File**: `ggml/src/ggml.c` or backend-specific files (e.g., `ggml-cpu.c`)
- **Function**: `ggml_graph_compute()` and operation dispatch functions
- **What to log**: Every operation execution with input/output tensors

**6.1.2 Llama.cpp Inference Loop (Token Tracking)**
- **File**: `src/llama.cpp`
- **Function**: `llama_decode()` or the main generation loop
- **What to log**: Token boundaries, phase transitions (prompt→generate)

**6.1.3 Attention Mechanism (Head-Level Detail)**
- **File**: Backend-specific attention implementations
- **Function**: `ggml_compute_forward_flash_attn_ext()` or similar
- **What to log**: Which Q/K/V matrices, which heads

**6.1.4 MoE Router (Expert Selection)**
- **File**: MoE-specific computation in `ggml.c`
- **Function**: Expert routing and gating operations
- **What to log**: Expert selection, routing scores

### 6.2 Instrumentation Pattern

**Minimal Invasive Wrapper:**
```c
// In ggml.c (or backend-specific file)

#ifdef TENSOR_TRACE_ENABLED
    #include "tensor_trace.h"
    #define TRACE_TENSOR_ACCESS(op, src0, src1, dst) \
        tensor_trace_log_access(op, src0, src1, dst)
#else
    #define TRACE_TENSOR_ACCESS(op, src0, src1, dst) /* no-op */
#endif

// Example: Instrument matrix multiply
static void ggml_compute_forward_mul_mat(...) {
    struct ggml_tensor * src0 = dst->src[0];
    struct ggml_tensor * src1 = dst->src[1];

    TRACE_TENSOR_ACCESS(OP_MUL_MAT, src0, src1, dst);

    // ... existing computation code ...
}
```

**Implementation in tensor_trace.c:**
```c
void tensor_trace_log_access(
    enum OperationType op_type,
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1,
    const struct ggml_tensor* dst
) {
    // Build log entry
    struct TensorAccessLog entry = {
        .timestamp_ns = get_timestamp_ns(),
        .token_id = g_current_token,
        .layer_id = extract_layer_id(src0->name),
        .operation_type = op_type,
        .phase = g_current_phase,
        .tensor_idx = get_or_add_tensor_name(src0->name),
        .tensor_ptr = (uint64_t)src0->data,
        .file_offset = get_file_offset(src0),
        .size_bytes = ggml_nbytes(src0),
        // ... parse name for attention/MoE details ...
    };

    // Write to log (fast, async)
    tensor_trace_log(&entry);

    // Also log src1 and dst if needed
    // ...
}
```

### 6.3 Tensor Name Parsing

**Extract Semantic Information from Tensor Names:**

```c
// Parse layer ID from tensor name
// Examples:
//   "blk.5.attn_q.weight" → layer=5
//   "token_embd.weight" → layer=-1
int extract_layer_id(const char* name) {
    if (strncmp(name, "blk.", 4) == 0) {
        int layer;
        sscanf(name + 4, "%d", &layer);
        return layer;
    }
    return -1; // Not a layer tensor
}

// Parse attention head (if name contains head info)
int extract_attention_head(const char* name) {
    // Example: "blk.5.attn_q.weight.head_7" → 7
    const char* head_marker = strstr(name, ".head_");
    if (head_marker) {
        int head;
        sscanf(head_marker + 6, "%d", &head);
        return head;
    }
    return -1; // No head info
}

// Determine Q/K/V type
enum QKVType extract_qkv_type(const char* name) {
    if (strstr(name, "attn_q")) return QKV_QUERY;
    if (strstr(name, "attn_k")) return QKV_KEY;
    if (strstr(name, "attn_v")) return QKV_VALUE;
    if (strstr(name, "attn_output")) return QKV_OUTPUT;
    return QKV_NONE;
}

// Parse MoE expert ID
// Example: "blk.5.expert_3.ffn_up.weight" → expert=3
int extract_expert_id(const char* name) {
    const char* expert_marker = strstr(name, "expert_");
    if (expert_marker) {
        int expert;
        sscanf(expert_marker + 7, "%d", &expert);
        return expert;
    }
    return -1; // Not an expert tensor
}
```

### 6.4 GGUF File Offset Tracking

**Map Tensor Pointer → GGUF File Offset:**

```c
// During model loading, build lookup table
struct TensorOffsetMap {
    void* tensor_data_ptr;
    uint64_t file_offset;
    uint64_t size_bytes;
    char name[128];
};

static TensorOffsetMap g_tensor_map[MAX_TENSORS];
static size_t g_tensor_map_size = 0;

// Called during llama_load_model_from_file()
void tensor_trace_register_tensor(
    const char* name,
    void* data_ptr,
    uint64_t file_offset,
    uint64_t size
) {
    g_tensor_map[g_tensor_map_size++] = {
        .tensor_data_ptr = data_ptr,
        .file_offset = file_offset,
        .size_bytes = size,
    };
    strncpy(g_tensor_map[g_tensor_map_size-1].name, name, 127);
}

// During trace logging, lookup offset
uint64_t get_file_offset(const struct ggml_tensor* tensor) {
    for (size_t i = 0; i < g_tensor_map_size; i++) {
        if (g_tensor_map[i].tensor_data_ptr == tensor->data) {
            return g_tensor_map[i].file_offset;
        }
    }
    return UINT64_MAX; // Not found
}
```

### 6.5 Token Counter Tracking

**Track Current Token Being Processed:**

```c
// Global state
static uint32_t g_current_token = 0;
static enum InferencePhase g_current_phase = PHASE_PROMPT;

// In llama.cpp, increment token counter
// Example location: after each llama_decode() call

int llama_decode(struct llama_context* ctx, struct llama_batch batch) {
    #ifdef TENSOR_TRACE_ENABLED
        if (batch.n_tokens > 0) {
            // First batch = prompt phase
            if (g_current_token == 0) {
                g_current_phase = PHASE_PROMPT;
            } else {
                g_current_phase = PHASE_GENERATE;
            }
        }
    #endif

    // ... existing decode logic ...

    #ifdef TENSOR_TRACE_ENABLED
        g_current_token += batch.n_tokens;
    #endif

    return result;
}
```

---

## 7. Implementation Phases

### Phase 1: Foundation (Days 1-2)

**Goal**: Prove we can log tensor accesses with basic context.

**Tasks:**

**1.1 Create GGUF Structure Dumper (2-3 hours)**
- Modify or create tool to output: `tensor_name, file_offset, size_bytes, layer_id, component_type`
- Input: GGUF model file
- Output: `gguf_structure.csv`
- Test with llama-2-7b model

**Deliverable:** `gguf_structure.csv` for test models

**1.2 Design Logging Infrastructure (2 hours)**
- Define `TensorAccessLog` struct
- Implement mmap-based ring buffer
- Write initialization/shutdown functions
- Create test harness (can we log 1M entries?)

**Deliverable:** `tensor_trace.h` and `tensor_trace.c` with unit tests

**1.3 Minimal Instrumentation Test (3-4 hours)**
- Add `#ifdef TENSOR_TRACE_ENABLED` guards to build system
- Instrument JUST ONE operation (e.g., `ggml_compute_forward_mul_mat`)
- Add token counter in llama.cpp
- Run inference: llama-2-7b, prompt "test", 10 tokens
- Verify log file contains expected entries

**Deliverable:** Proof-of-concept binary log file

**1.4 Log Parser (2 hours)**
- Write Python script to parse binary log
- Convert to pandas DataFrame
- Print summary statistics (total accesses, unique tensors, etc.)

**Deliverable:** `parse_trace.py` script

**Phase 1 Success Criteria:**
- ✅ Can log tensor accesses during inference
- ✅ Binary log format works (can read it back)
- ✅ Token counter tracks correctly
- ✅ No crashes, no corruption

**Estimated Time:** 2 days

---

### Phase 2: Comprehensive Instrumentation (Days 3-5)

**Goal**: Log ALL operations with full context.

**Tasks:**

**2.1 Instrument All Operations (6-8 hours)**
- Add `TRACE_TENSOR_ACCESS()` to all operation types:
  - Matrix operations: MUL_MAT, MUL
  - Attention: FLASH_ATTN, ROPE
  - Normalization: RMS_NORM, LAYER_NORM
  - Activations: GELU, SWIGLU, SILU
  - Memory ops: CPY, VIEW, RESHAPE, PERMUTE
  - Arithmetic: ADD, SUB, MUL, DIV
- Ensure every major ggml operation is covered

**Deliverable:** Instrumented ggml.c (or backend files)

**2.2 Attention Head Tracking (3-4 hours)**
- Parse attention tensor names for head information
- If multi-head attention splits Q/K/V by head, track which head
- Log attention-specific context (Q/K/V type, head index)

**Deliverable:** Attention-aware logging

**2.3 MoE Expert Tracking (3-4 hours)**
- Identify MoE routing operations in GGML
- Log expert selection (which experts activated)
- If possible, capture routing scores
- Parse expert IDs from tensor names

**Deliverable:** MoE-aware logging

**2.4 File Offset Mapping (2-3 hours)**
- Hook into model loading to build tensor→offset map
- Store mapping in global lookup table
- During tracing, lookup offset for each tensor

**Deliverable:** Complete offset tracking

**2.5 Full Test Run (2-3 hours)**
- Run instrumented llama.cpp with:
  - llama-2-7b (standard transformer)
  - gpt-oss-20b (MoE, if available)
- Generate 100 tokens
- Verify log completeness:
  - Expected number of entries?
  - All layers represented?
  - Token progression correct?

**Deliverable:** Complete trace logs for multiple models

**Phase 2 Success Criteria:**
- ✅ Every operation type logged
- ✅ Attention heads tracked (if applicable)
- ✅ MoE experts tracked (if applicable)
- ✅ File offsets correct
- ✅ No performance regression (inference still fast enough)

**Estimated Time:** 3 days

---

### Phase 3: Visualization & Analysis (Days 6-8)

**Goal**: Generate insightful visualizations from trace data.

**Tasks:**

**3.1 Data Processing Pipeline (3-4 hours)**
- Parse binary logs → pandas DataFrame
- Merge with GGUF structure data
- Merge with tensor name table
- Add computed columns:
  - Layer name (e.g., "Layer 5")
  - Component type (e.g., "Attention", "FFN")
  - Access category (e.g., "Hot", "Cold")

**Deliverable:** `TraceAnalyzer` Python class

**3.2 Core Visualizations (4-6 hours)**

**Heatmap 1: Token × Tensor**
- Y-axis: Token ID (0-100)
- X-axis: Tensor name (grouped by layer)
- Color: Access count (or total bytes)
- Goal: See which tensors accessed per token

**Heatmap 2: Token × Layer**
- Y-axis: Token ID
- X-axis: Layer ID (0-31)
- Color: Number of accesses
- Goal: See layer-by-layer progression

**Timeline: File Offset × Time**
- Y-axis: File offset (bytes)
- X-axis: Time (seconds since start)
- Color: Token ID
- Goal: Visualize sequential vs random access

**Statistics Tables:**
- Top 10 most accessed tensors
- Access count per layer
- Access count per operation type
- Total bytes accessed per component type

**Deliverable:** Matplotlib/Seaborn visualizations

**3.3 Interactive Visualization (4-6 hours)**
- Use Plotly for interactive plots
- Features:
  - Zoom/pan
  - Hover for details (tensor name, timestamp)
  - Filter by layer, token, operation type
  - Export filtered data

**Deliverable:** Interactive HTML dashboards

**3.4 Animation (Optional, 2-3 hours)**
- Animate access pattern over time
- Show "wave" of computation flowing through layers
- Highlight active tensors at each timestamp

**Deliverable:** MP4 animation or animated GIF

**3.5 Analysis Notebook (2-3 hours)**
- Jupyter notebook with:
  - Data loading
  - All visualizations
  - Summary statistics
  - Findings and observations

**Deliverable:** `analysis.ipynb`

**Phase 3 Success Criteria:**
- ✅ Heatmaps clearly show access patterns
- ✅ Timeline reveals sequential vs random access
- ✅ Statistics identify hot/cold parameters
- ✅ Interactive exploration works smoothly
- ✅ Visualizations are publication-quality

**Estimated Time:** 3 days

---

### Phase 4: Advanced Features (Days 9-10, Optional)

**Goal**: Add nice-to-have features based on initial findings.

**Tasks:**

**4.1 Page Fault Correlation (3-4 hours)**
- Run blktrace alongside tensor tracing
- Parse blktrace output
- Correlate blktrace sector accesses with tensor file offsets
- Visualize: which tensor accesses caused page faults?

**Deliverable:** Blktrace correlation analysis

**4.2 Expert Heatmap (MoE-specific, 2-3 hours)**
- For MoE models, create:
  - Token × Expert matrix (which experts per token)
  - Expert "temperature" (activation frequency)
  - Expert specialization analysis

**Deliverable:** MoE-specific visualizations

**4.3 Attention Pattern Visualization (2-3 hours)**
- If attention heads tracked, visualize:
  - Which heads most active
  - Q/K/V access patterns
  - Head-specific access counts

**Deliverable:** Attention analysis

**4.4 Optimization Hints (2 hours)**
- Analyze trace to suggest:
  - Which tensors to pin in RAM (top 10% by access count)
  - Predicted prefetch sequence (next-access prediction)
  - Cache size recommendations

**Deliverable:** Optimization report

**Phase 4 Success Criteria:**
- ✅ Blktrace correlation working
- ✅ MoE insights actionable
- ✅ Optimization hints validated

**Estimated Time:** 2 days (optional)

---

## 8. Visualization & Analysis Plan

### 8.1 Critical Visualizations (Must-Have)

**1. Token-Layer Heatmap**
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load trace
df = pd.read_parquet('trace.parquet')

# Aggregate: access count per (token, layer)
heatmap_data = df.groupby(['token_id', 'layer_id']).size().unstack(fill_value=0)

# Plot
plt.figure(figsize=(16, 10))
sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': 'Access Count'})
plt.xlabel('Layer ID')
plt.ylabel('Token ID')
plt.title('Memory Access Heatmap: Token × Layer')
plt.savefig('heatmap_token_layer.png', dpi=300)
```

**Expected Pattern:**
- Horizontal bands if all layers accessed equally per token
- Vertical bands if certain layers accessed more often

**2. Temporal Access Timeline**
```python
# Scatter plot: file offset over time
plt.figure(figsize=(20, 8))
plt.scatter(df['timestamp_ns'] / 1e9, df['file_offset'],
            c=df['token_id'], cmap='viridis', alpha=0.5, s=1)
plt.xlabel('Time (seconds)')
plt.ylabel('File Offset (bytes)')
plt.title('File Access Pattern Over Time')
plt.colorbar(label='Token ID')
plt.savefig('timeline_offset.png', dpi=300)
```

**Expected Pattern:**
- Diagonal line = sequential access
- Horizontal bands = re-accessing same regions
- Scattered = random access

**3. Hot Tensor Identification**
```python
# Bar chart: top 20 most accessed tensors
top_tensors = df.groupby('tensor_name').size().sort_values(ascending=False).head(20)

plt.figure(figsize=(14, 8))
top_tensors.plot(kind='barh')
plt.xlabel('Access Count')
plt.title('Top 20 Most Accessed Tensors')
plt.tight_layout()
plt.savefig('hot_tensors.png', dpi=300)
```

**Expected Finding:**
- Embeddings likely hottest
- Certain layers may be hot spots

**4. Per-Layer Statistics**
```python
# Access count per layer
layer_stats = df.groupby('layer_id').agg({
    'tensor_name': 'count',
    'size_bytes': 'sum'
}).rename(columns={'tensor_name': 'access_count', 'size_bytes': 'total_bytes'})

print(layer_stats)
```

**5. Operation Type Breakdown**
```python
# Pie chart: operation type distribution
op_counts = df['operation_type'].value_counts()

plt.figure(figsize=(10, 10))
plt.pie(op_counts, labels=op_counts.index, autopct='%1.1f%%')
plt.title('Access Breakdown by Operation Type')
plt.savefig('operation_types.png', dpi=300)
```

### 8.2 Interactive Visualizations (Extended)

**Using Plotly for Interactivity:**

```python
import plotly.express as px
import plotly.graph_objects as go

# Interactive timeline with zoom
fig = px.scatter(df, x='timestamp_ns', y='file_offset', color='token_id',
                 hover_data=['tensor_name', 'layer_id', 'operation_type'],
                 title='Interactive Access Timeline')
fig.write_html('timeline_interactive.html')

# Interactive heatmap with filters
fig = px.density_heatmap(df, x='layer_id', y='token_id',
                          title='Token × Layer Access Heatmap')
fig.write_html('heatmap_interactive.html')
```

**Features:**
- Zoom into specific time ranges
- Click on point to see tensor details
- Filter by layer, operation type, phase

### 8.3 MoE-Specific Analysis

```python
# Filter for MoE expert tensors
moe_df = df[df['expert_id'] != 255]  # 255 = N/A

# Expert activation heatmap
expert_usage = moe_df.groupby(['token_id', 'expert_id']).size().unstack(fill_value=0)

plt.figure(figsize=(16, 10))
sns.heatmap(expert_usage, cmap='Blues', cbar_kws={'label': 'Activations'})
plt.xlabel('Expert ID')
plt.ylabel('Token ID')
plt.title('MoE Expert Activation Pattern')
plt.savefig('moe_expert_heatmap.png', dpi=300)

# Identify hot vs cold experts
expert_totals = expert_usage.sum(axis=0).sort_values(ascending=False)
print("Hot experts:", expert_totals.head(5).index.tolist())
print("Cold experts:", expert_totals.tail(5).index.tolist())
```

### 8.4 Attention Head Analysis

```python
# Filter for attention operations
attn_df = df[df['qkv_type'] != 0]  # 0 = QKV_NONE

# Breakdown by Q/K/V
qkv_counts = attn_df['qkv_type'].value_counts()
print("Q accesses:", qkv_counts.get(1, 0))
print("K accesses:", qkv_counts.get(2, 0))
print("V accesses:", qkv_counts.get(3, 0))

# Per-head breakdown (if head tracking enabled)
head_counts = attn_df.groupby('attention_head').size()
plt.figure(figsize=(12, 6))
head_counts.plot(kind='bar')
plt.xlabel('Attention Head')
plt.ylabel('Access Count')
plt.title('Access Count Per Attention Head')
plt.savefig('attention_heads.png', dpi=300)
```

---

## 9. Testing & Validation

### 9.1 Correctness Validation

**Test 1: Access Count Sanity Check**
```python
# For standard transformer with N layers, M operations per layer, T tokens:
# Expected total accesses ≈ N × M × T

expected_accesses = num_layers * ops_per_layer * num_tokens
actual_accesses = len(df)

assert 0.9 * expected_accesses < actual_accesses < 1.1 * expected_accesses, \
    f"Access count mismatch: expected ~{expected_accesses}, got {actual_accesses}"
```

**Test 2: Token Progression**
```python
# Tokens should be processed in order: 0, 1, 2, ..., N
token_sequence = df['token_id'].unique()
assert list(token_sequence) == list(range(max(token_sequence) + 1)), \
    "Token IDs are not sequential!"
```

**Test 3: Layer Completeness**
```python
# All layers should be represented
layers_accessed = df['layer_id'].unique()
expected_layers = set(range(num_layers))
assert set(layers_accessed) >= expected_layers, \
    f"Missing layers: {expected_layers - set(layers_accessed)}"
```

**Test 4: File Offset Validity**
```python
# All file offsets should be within GGUF file size
gguf_size = get_file_size(model_path)
invalid_offsets = df[df['file_offset'] > gguf_size]
assert len(invalid_offsets) == 0, \
    f"Found {len(invalid_offsets)} invalid file offsets!"
```

### 9.2 Performance Validation

**Benchmark: Overhead Measurement**
```bash
# Run without tracing
time ./llama-cli -m model.gguf -p "test" -n 100 > /dev/null
# Example: 12.5 seconds

# Run with tracing
TENSOR_TRACE_ENABLED=1 time ./llama-cli -m model.gguf -p "test" -n 100 > /dev/null
# Example: 13.8 seconds

# Overhead = (13.8 - 12.5) / 12.5 = 10.4%
```

**Acceptable overhead**: <20% for MVP (optimize later if needed)

### 9.3 Cross-Validation with Existing Tools

**Compare with --verbose output:**
```bash
# Run llama-cli with verbose logging
./llama-cli -m model.gguf -p "test" -n 10 --verbose 2>&1 | grep "load time"

# Compare load time with our trace timestamps
# First access timestamp should be ~= load time
```

**Compare with llama-bench:**
```bash
# Benchmark performance with llama-bench
./llama-bench -m model.gguf -n 100 -o csv

# Our trace should show similar token/sec if overhead is low
```

---

## 10. Timeline & Milestones

### 10.1 Development Timeline

| Phase | Tasks | Duration | Dependencies |
|-------|-------|----------|--------------|
| **Phase 1** | GGUF dumper + Logging infra + Minimal test | 2 days | None |
| **Phase 2** | Full instrumentation + MoE/Attention tracking | 3 days | Phase 1 |
| **Phase 3** | Visualizations + Analysis tools | 3 days | Phase 2 |
| **Phase 4** | Advanced features (optional) | 2 days | Phase 3 |
| **Total** | **MVP** | **8-10 days** | |

### 10.2 Milestones

**Milestone 1 (End of Phase 1):**
- ✅ Proof of concept working
- ✅ Can log tensor accesses
- ✅ Binary log format validated
- **Demo**: Show parsed log with 10 tokens

**Milestone 2 (End of Phase 2):**
- ✅ Comprehensive instrumentation complete
- ✅ Full trace logs for multiple models
- ✅ All operation types covered
- **Demo**: Show 100-token trace with all layers

**Milestone 3 (End of Phase 3):**
- ✅ Visualizations generated
- ✅ Analysis notebook ready
- ✅ Hot/cold parameters identified
- **Demo**: Present heatmaps and findings

**Milestone 4 (End of Phase 4, optional):**
- ✅ Advanced features implemented
- ✅ Optimization hints generated
- ✅ Tool ready for production use
- **Demo**: Full analysis with recommendations

### 10.3 Iterative Development Strategy

**Week 1: Foundation**
- Focus on getting basic logging working
- Don't worry about completeness yet
- Goal: Prove the concept

**Week 2: Completeness**
- Add all operation types
- Add MoE and attention tracking
- Goal: Comprehensive data collection

**Week 3: Insights**
- Build visualizations
- Analyze patterns
- Goal: Answer research questions

**Week 4 (if needed): Polish**
- Optimize performance
- Add advanced features
- Goal: Production-ready tool

### 10.4 Critical Path

**Must-Have for MVP:**
1. ✅ Binary logging infrastructure (mmap-based)
2. ✅ Operation instrumentation (at least major ops)
3. ✅ Token counter tracking
4. ✅ File offset mapping
5. ✅ Basic visualizations (heatmap, timeline)

**Can Be Added Later:**
- 🔧 Attention head tracking (nice to have)
- 🔧 MoE expert routing scores (if model supports)
- 🔧 Interactive visualizations (static first)
- 🔧 Animation (cool but not critical)
- 🔧 Blktrace correlation (Phase 4)

---

## 11. Open Questions & Decisions Needed

### 11.1 Implementation Decisions

**Q1: Logging granularity for large tensors?**
- Should we log access to a 2GB tensor as single entry or split by chunk?
- **Proposal**: Single entry with full size (simplify analysis)

**Q2: Multi-threading support?**
- llama.cpp can use multiple threads for compute
- Do we need thread-safe logging?
- **Proposal**: Thread-local buffers, merge on shutdown (Phase 2+)

**Q3: Attention head tracking - how granular?**
- Track individual heads or just operation type?
- **Proposal**: Start with operation type (Q/K/V), add head indices in Phase 2

**Q4: MoE routing scores - how to capture?**
- Need to hook into MoE router logic
- May require model-specific code
- **Proposal**: Start with expert ID only, add scores if easy

### 11.2 Analysis Questions

**Q5: What defines "hot" vs "cold"?**
- Top 10% by access count?
- Accessed >N times?
- **Proposal**: Use percentile (top 10% = hot, bottom 50% = cold)

**Q6: Sequential vs random threshold?**
- What gap size between accesses counts as "random"?
- **Proposal**: Use existing blktrace thresholds (128KB gap = random)

**Q7: Visualization color schemes?**
- Use consistent colors across all plots?
- **Proposal**: YlOrRd for heatmaps, viridis for timelines

### 11.3 Scope Clarifications

**Q8: Should we support GPU backends?**
- Current focus: CPU only (hiwi-02 has no GPU)
- **Decision**: CPU-only for MVP, design for extensibility

**Q9: Should tool work on unmodified llama.cpp?**
- Or is forking acceptable?
- **Decision**: Fork is acceptable, keep changes minimal with `#ifdef`

**Q10: Performance target?**
- What overhead is acceptable?
- **Decision**: <20% for MVP, optimize later if needed

---

## 12. Next Immediate Steps

### 12.1 Pre-Implementation Checklist

Before writing code, ensure:

- [ ] Review this plan with Ersjan - get approval on approach
- [ ] Decide on async logging strategy (mmap recommended)
- [ ] Confirm file organization (create `tensor-tracker/` folder?)
- [ ] Choose test model (llama-2-7b for MVP)
- [ ] Set up development branch in llama.cpp fork

### 12.2 Day 1 Action Items

**Morning (4 hours):**
1. Create `tensor-tracker/` directory structure
2. Write GGUF offset dumper
3. Test on llama-2-7b model
4. Verify output CSV is correct

**Afternoon (4 hours):**
1. Implement `tensor_trace.h` and `tensor_trace.c`
2. Write mmap-based logging
3. Create unit tests
4. Test: can we log 1M entries?

**End of Day 1 Deliverable:**
- Working GGUF dumper
- Working logging infrastructure
- Unit tests passing

### 12.3 Week 1 Goal

**By End of Week 1:**
- Phase 1 complete (foundation working)
- Can log tensor accesses during inference
- Have first visualizations (even if basic)
- Validated approach is sound

**Success Criteria:**
- Run llama-2-7b with 10 tokens
- Generate trace log
- Parse log to pandas
- Create basic heatmap

**If successful:** Proceed to Phase 2 (comprehensive instrumentation)

---

## 13. Sources & References

### Research & Tools
- [alphaXiv Tensor Trace](https://www.alphaxiv.org/labs/tensor-trace) - Interactive 3D GGML tensor visualization
- [llama-bench Documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) - Official benchmarking tool
- [GGML Performance Profiling Discussion](https://github.com/ggml-org/llama.cpp/discussions/6871) - Community discussion on profiling
- [GGML Computation Graph Logging](https://github.com/ggml-org/llama.cpp/discussions/11039) - Graph export feature
- [NVIDIA DLProf Guide](https://developer.nvidia.com/blog/profiling-and-optimizing-deep-neural-networks-with-dlprof-and-pyprof/) - GPU profiling reference
- [Graph Neural Network Memory Access Patterns](https://dl.acm.org/doi/abs/10.1145/3624062.3624168) - Academic research on memory patterns
- [NTP Neural Network Topology Profiler](https://arxiv.org/abs/1905.09063) - Neural network profiling methodology
- [GGUF File Format Documentation](https://huggingface.co/docs/hub/en/gguf) - Official GGUF spec
- [llama.cpp Architecture Guide](https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama_cpp_streamline/2_llama.cpp_intro/) - Detailed architecture explanation

### Internal References
- [30DEC.md](./Benchmark/journal/30DEC.md) - Original design document
- [SummarySoFar.md](./SummarySoFar.md) - Project context and background
- [HYPOTHESIS.md](./HYPOTHESIS.md) - Research hypotheses

---

## 14. Appendix: Technical Details

### A. GGUF File Structure Refresher

GGUF file layout:
```
[Header]
  - Magic number
  - Version
  - Tensor count
  - Metadata count

[Metadata]
  - Key-value pairs (architecture, hyperparameters, etc.)

[Tensor Info]
  - For each tensor:
    - Name (string)
    - Dimensions (n_dims, ne[0], ne[1], ...)
    - Type (F32, F16, Q4_K_M, etc.)
    - Offset (relative to data section)

[Alignment padding]

[Data Section] ← Tensors start here
  - Tensor 0 data
  - Tensor 1 data
  - ...
```

**Key insight**: `data_offset` (from header) + `tensor_offset` = absolute file offset

### B. GGML Computation Graph Basics

```c
struct ggml_cgraph {
    int n_nodes;                    // Number of operations
    struct ggml_tensor* nodes[...]; // Operations (topologically sorted)

    int n_leafs;                    // Number of constant tensors
    struct ggml_tensor* leafs[...]; // Model weights, inputs
};

struct ggml_tensor {
    enum ggml_type type;      // Data type
    int64_t ne[GGML_MAX_DIMS]; // Shape
    void* data;               // Actual data (may be mmap'd)
    char name[GGML_MAX_NAME]; // Tensor name

    enum ggml_op op;          // Operation that creates this tensor
    struct ggml_tensor* src[GGML_MAX_SRC]; // Input tensors
};
```

**Key insight**: Each node in graph is an operation; traverse graph = execute inference

### C. Example Tensor Names in GGUF

Standard transformer:
```
token_embd.weight
output_norm.weight
blk.0.attn_norm.weight
blk.0.attn_q.weight
blk.0.attn_k.weight
blk.0.attn_v.weight
blk.0.attn_output.weight
blk.0.ffn_norm.weight
blk.0.ffn_up.weight
blk.0.ffn_down.weight
blk.0.ffn_gate.weight
blk.1.attn_norm.weight
...
```

MoE model (hypothetical):
```
blk.5.expert_0.ffn_up.weight
blk.5.expert_0.ffn_down.weight
blk.5.expert_1.ffn_up.weight
blk.5.expert_1.ffn_down.weight
...
blk.5.expert_gate.weight  ← Router weights
```

### D. Timestamp Precision

```c
#include <time.h>

uint64_t get_timestamp_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
```

**Why CLOCK_MONOTONIC?**
- Guaranteed never goes backwards
- Not affected by NTP adjustments
- Suitable for performance measurement

---

## 15. Final Notes

### Key Success Factors

1. **Simplicity First**: Start with mmap logging, add complexity only if needed
2. **Iterate Quickly**: Get basic version working, then enhance
3. **Validate Early**: Test with small models and short runs first
4. **Document Findings**: Keep notes as we discover patterns
5. **Stay Focused**: Visualization is easy once data is right

### Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Logging overhead too high | Medium | High | Use mmap, optimize hot path |
| Missing tensor names | Low | Medium | Fall back to pointer addresses |
| GGUF format changes | Low | Low | Use stable llama.cpp version |
| Disk space exhaustion | Medium | Medium | Compress logs, limit trace size |
| Instrumentation bugs | High | High | Unit test each component |

### When to Pivot

**Abort Phase 1 if:**
- Cannot log tensor accesses without crashes
- Overhead >50% (unusable)
- Binary log format fundamentally flawed

**Simplify Phase 2 if:**
- Time constraints (skip attention heads initially)
- MoE models unavailable (focus on standard transformers)
- Instrumentation too complex (reduce operation coverage)

**Defer Phase 4 if:**
- Phases 1-3 take longer than expected
- Research questions answered with basic data
- Optimization work takes priority

---

## Approval & Sign-Off

**Plan Author**: Claude (AI Assistant)
**Plan Reviewed By**: Ersjan Këri
**Approval Date**: _____________
**Approved to Proceed**: [ ] Yes  [ ] No  [ ] With Modifications

**Modifications/Comments:**
_____________________________________________________________________________
_____________________________________________________________________________
_____________________________________________________________________________

---

**END OF TRACKER_PLAN.md**

*Version 1.0 - Comprehensive implementation plan for LLM Tensor Access Tracker*
*Last Updated: January 1, 2026*
