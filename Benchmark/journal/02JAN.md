# January 2, 2026 - Tensor Access Tracker: Critical Design Decisions

## Overview

Today we finalized the **critical architectural decisions** for the tensor access tracking tool. This is the foundation for all future optimization work (hot/cold parameter analysis, prefetching strategies, MoE optimization).

**Key Principle**: Get the recording infrastructure PERFECT first. Visualization becomes trivial once we have clean, complete data.

---

## Decision 1: Logging Strategy - Option A (mmap to tmpfs)

### The Choice

**Selected**: Option A - Memory-mapped ring buffer to tmpfs
**Alternative Considered**: Option B - Lock-free queue with background thread

### Why Option A?

**Performance Characteristics**:
```c
// Option A: Direct memory write (5-10 CPU cycles)
void log_access(...) {
    memcpy(log_buffer + offset, &entry, sizeof(entry));
    offset += sizeof(entry);
    // DONE! No syscalls, no locks, no thread synchronization
}

// Option B: Atomic queue operations (20-30 CPU cycles)
void log_access(...) {
    size_t tail = atomic_load(&queue.tail);
    queue.entries[tail % QUEUE_SIZE] = entry;
    atomic_store(&queue.tail, tail + 1);
    // Plus background thread overhead
}
```

**Decision Matrix**:

| Criterion | Option A (mmap) | Option B (thread) | Winner |
|-----------|----------------|-------------------|---------|
| **Speed** | 5-10 cycles | 20-30 cycles | **A** |
| **Simplicity** | 50 lines | 200+ lines | **A** |
| **Capacity** | Fixed (2GB) | Unbounded | B |
| **Overhead** | Minimal | Low (but more) | **A** |
| **Reliability** | No threads = no bugs | Thread sync risks | **A** |
| **Flexibility** | Limited | High (compress, filter) | B |

**Verdict**: Option A for Phase 1 (MVP), can upgrade to Option B later if needed.

### Implementation Details

**Location**: `/dev/shm/tensor_trace.bin` (tmpfs - writes to RAM, not disk)

**Why tmpfs?**
- **Zero I/O latency**: Writes to RAM, not SSD
- **No interference**: Won't affect inference I/O measurements
- **Safe**: Copy to permanent storage after inference completes
- **Fast**: Page cache is already in RAM

**Workflow**:
```bash
# 1. Before inference: Pre-allocate in tmpfs
truncate -s 2G /dev/shm/tensor_trace.bin

# 2. During inference: mmap and write
void* log_buffer = mmap(NULL, 2GB, PROT_WRITE, MAP_SHARED, fd, 0);
# ... log millions of entries (all to RAM) ...

# 3. After inference: Copy to permanent storage
cp /dev/shm/tensor_trace.bin ~/BSC/traces/experiment_001/trace.bin
rm /dev/shm/tensor_trace.bin
```

**Capacity Planning**:
- Entry size: 64 bytes
- Buffer size: 2 GB = 2,147,483,648 bytes
- Max entries: 2GB / 64 bytes = **33,554,432 entries**
- Example: 100 tokens × 32 layers × 20 ops/layer = 64,000 entries (0.2% of capacity)
- **Conclusion**: More than enough for any realistic experiment

### When to Upgrade to Option B

Only if we encounter:
1. **Capacity limits**: Need >30M entries per run (unlikely)
2. **Real-time streaming**: Want to visualize during inference (future feature)
3. **Compression**: Want to save disk space (nice-to-have)
4. **Measurable overhead**: If Option A shows >5% overhead (validate first)

**Current status**: No need for Option B yet. Keep it simple.

---

## Decision 2: Instrumentation Strategy - Hybrid Approach

### The Choice

**Selected**: Option 2 (Mid-Level) + Strategic Option 3 (Deep Dive)

**Option 2 (Base)**: Hook `ggml_compute_forward()` for all operations
**Option 3 (Deep)**: Add detailed hooks for critical operations only

### Why Hybrid?

**Three Levels Considered**:

**Level 1: High-Level (Graph Execution)**
```c
// In ggml_graph_compute_helper()
for (int i = 0; i < cgraph->n_nodes; i++) {
    struct ggml_tensor* node = cgraph->nodes[i];
    TRACE_NODE(node);  // One log entry per graph node
    ggml_compute_forward(ctx, node);
}
```

**Pros**: Simple, one hook
**Cons**: Too coarse - can't see individual tensor accesses
**Verdict**: ❌ Not detailed enough

---

**Level 2: Mid-Level (Operation Dispatch)** ⭐ **SELECTED AS BASE**
```c
// In ggml_compute_forward() - operation dispatch
static void ggml_compute_forward(struct ggml_context* ctx, struct ggml_tensor* tensor) {
    switch (tensor->op) {
        case GGML_OP_MUL_MAT:
            TRACE_OP_START(tensor, "MUL_MAT");
            TRACE_TENSOR_ACCESS(tensor->src[0], "weight_matrix");
            TRACE_TENSOR_ACCESS(tensor->src[1], "input_activations");
            ggml_compute_forward_mul_mat(ctx, tensor);
            TRACE_OP_END(tensor);
            break;

        case GGML_OP_ADD:
            TRACE_OP_START(tensor, "ADD");
            TRACE_TENSOR_ACCESS(tensor->src[0], "operand_a");
            TRACE_TENSOR_ACCESS(tensor->src[1], "operand_b");
            ggml_compute_forward_add(ctx, tensor);
            TRACE_OP_END(tensor);
            break;

        case GGML_OP_RMS_NORM:
            TRACE_OP_START(tensor, "RMS_NORM");
            TRACE_TENSOR_ACCESS(tensor->src[0], "input");
            ggml_compute_forward_rms_norm(ctx, tensor);
            TRACE_OP_END(tensor);
            break;

        // ... for all operation types
    }
}
```

**Pros**:
- ✅ Tensor-level detail (know which src tensor accessed)
- ✅ One location for all operations (maintainable)
- ✅ Semantic information (operation type, tensor roles)
- ✅ Moderate verbosity (not millions of entries)

**Cons**:
- ❌ Need to instrument each operation type (but there's only ~50)
- ❌ Don't see sub-operation details (e.g., cache line accesses)

**Verdict**: ⭐ **Perfect balance** of detail vs. maintainability

---

**Level 3: Low-Level (Inside Each Operation)** ⭐ **STRATEGIC USE ONLY**
```c
// Inside ggml-cpu.c - actual computation
static void ggml_compute_forward_mul_mat_f16_f32(...) {
    const struct ggml_tensor* src0 = dst->src[0];
    const struct ggml_tensor* src1 = dst->src[1];

    #ifdef TRACE_DETAILED_MATMUL
        TRACE_MATMUL_START(src0->name, src1->name, m, n, k);
    #endif

    // Matrix multiply inner loop
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                #ifdef TRACE_CACHE_LINE_ACCESS
                    TRACE_ACCESS(src0->data + offset);  // ← VERY detailed
                #endif
                sum += src0_row[p] * src1_col[p];
            }
            dst_row[j] = sum;
        }
    }

    #ifdef TRACE_DETAILED_MATMUL
        TRACE_MATMUL_END();
    #endif
}
```

**Pros**:
- ✅ Maximum detail (cache line level)
- ✅ Performance insights (memory access patterns)

**Cons**:
- ❌ MASSIVE overhead (logging inside inner loops)
- ❌ Gigabytes of log data (TB for long runs)
- ❌ Hard to maintain (backend-specific, duplicated code)

**Verdict**: ❌ **Too detailed for general use**, ✅ **But useful for specific deep dives**

---

### Hybrid Strategy (Best of Both Worlds)

**Base Instrumentation (Always On)**:
- Level 2 hooks in `ggml_compute_forward()` for ALL operations
- Logs: operation start/end, source tensors, sizes, timestamps

**Deep Instrumentation (Optional, Compile Flag)**:
- Level 3 hooks ONLY for:
  - `ggml_compute_forward_mul_mat` (most expensive operation)
  - `ggml_compute_forward_flash_attn` (attention mechanism details)
  - MoE routing operations (expert selection)
- Enabled via `#ifdef TRACE_DETAILED_MATMUL` etc.
- Used for targeted analysis, not general tracing

**Example**:
```c
// Always enabled (Level 2)
case GGML_OP_MUL_MAT:
    TRACE_OP_START(tensor, "MUL_MAT");
    TRACE_TENSOR_ACCESS(tensor->src[0], "weights");
    TRACE_TENSOR_ACCESS(tensor->src[1], "input");

    ggml_compute_forward_mul_mat(ctx, tensor);

    TRACE_OP_END(tensor);
    break;

// Optionally enabled (Level 3) - inside ggml_compute_forward_mul_mat()
#ifdef TRACE_DETAILED_MATMUL
    TRACE_MATMUL_DETAILS(m, n, k, src0_type, src1_type);
#endif
```

**Benefits**:
1. **Flexible**: Can enable deep tracing for specific operations
2. **Efficient**: Don't pay overhead unless needed
3. **Maintainable**: Base instrumentation is clean and simple
4. **Extensible**: Can add deep tracing for new operations later

---

## Decision 3: Binary Log Format (64 bytes, with thread_id)

### Final Structure

```c
struct TensorAccessLog {
    // === Timestamp (8 bytes) ===
    uint64_t timestamp_ns;        // Nanoseconds since trace start

    // === Execution Context (16 bytes) ===
    uint32_t token_id;            // Which token being processed
    uint16_t layer_id;            // Which transformer layer (0-based)
    uint16_t thread_id;           // CPU thread ID ← NEW!
    uint8_t  operation_type;      // Enum: MUL_MAT, ADD, ROPE, etc.
    uint8_t  phase;               // Enum: PROMPT, GENERATE
    uint16_t padding1;            // Alignment
    uint32_t padding1b;           // Alignment

    // === Tensor Identification (20 bytes) ===
    uint32_t tensor_idx;          // Index into tensor name table
    uint64_t tensor_ptr;          // Virtual address of tensor->data
    uint64_t file_offset;         // Offset in GGUF file
    uint32_t size_bytes;          // Size of access

    // === Attention-Specific (4 bytes) ===
    uint8_t  attention_head;      // Which attention head (0-127, or 255=N/A)
    uint8_t  qkv_type;            // Enum: Q, K, V, O, or N/A
    uint16_t padding2;            // Alignment

    // === MoE-Specific (8 bytes) ===
    uint8_t  expert_id;           // Which expert (0-255, or 255=N/A)
    uint8_t  expert_rank;         // Routing rank (0=top, 1=second, etc.)
    uint16_t routing_score;       // Quantized routing score (0-65535)
    uint32_t padding3;            // Alignment

    // Total: 64 bytes (cache-line aligned, power of 2)
};
```

### Why thread_id is Critical

**Use Cases**:

1. **Parallelism Analysis**:
   - Which threads are active?
   - Is llama.cpp using all available threads?
   - Thread utilization over time

2. **Contention Detection**:
   - Are multiple threads accessing same tensor?
   - Lock contention (if any)?
   - False sharing?

3. **Backend Analysis**:
   - Does CPU backend use threading effectively?
   - Per-thread workload distribution
   - Thread assignment patterns

4. **Debugging**:
   - Thread-specific bugs
   - Race conditions (if accesses interleaved)

**Example Analysis**:
```python
# Thread utilization heatmap
thread_usage = df.groupby(['timestamp_bucket', 'thread_id']).size()

plt.figure(figsize=(14, 6))
sns.heatmap(thread_usage.unstack(), cmap='YlOrRd')
plt.xlabel('Thread ID')
plt.ylabel('Time (seconds)')
plt.title('Thread Utilization Over Time')
```

**Implementation**:
```c
static inline uint16_t get_thread_id() {
    #ifdef _WIN32
        return (uint16_t)GetCurrentThreadId();
    #elif defined(__APPLE__)
        uint64_t tid;
        pthread_threadid_np(NULL, &tid);
        return (uint16_t)tid;
    #else
        return (uint16_t)pthread_self();
    #endif
}
```

---

## Decision 4: Instrumentation Locations in llama.cpp

### Primary Hook Point

**File**: `ggml/src/ggml.c`
**Function**: `ggml_compute_forward()`

**Why This Location?**
1. ✅ **Central dispatch**: All operations go through here
2. ✅ **Backend-agnostic**: Works for CPU, CUDA, Metal, etc.
3. ✅ **Complete coverage**: Every operation is logged
4. ✅ **Maintainable**: Single location, easy to update

**Code Structure**:
```c
// In ggml.c
static void ggml_compute_forward(struct ggml_context* ctx, struct ggml_tensor* tensor) {

    #ifdef TENSOR_TRACE_ENABLED
        // Log operation start
        tensor_trace_operation_start(tensor);

        // Log source tensor accesses
        for (int i = 0; i < GGML_MAX_SRC && tensor->src[i]; i++) {
            tensor_trace_tensor_access(tensor->src[i], i);
        }
    #endif

    // Original operation dispatch (unchanged)
    switch (tensor->op) {
        case GGML_OP_MUL_MAT:
            ggml_compute_forward_mul_mat(ctx, tensor);
            break;
        case GGML_OP_ADD:
            ggml_compute_forward_add(ctx, tensor);
            break;
        // ... all other operations
    }

    #ifdef TENSOR_TRACE_ENABLED
        // Log operation end
        tensor_trace_operation_end(tensor);
    #endif
}
```

### Secondary Hook Point (Token Tracking)

**File**: `src/llama.cpp`
**Function**: `llama_decode()` or main generation loop

**Purpose**: Track token boundaries

**Code**:
```c
// In llama.cpp, around token generation loop
int llama_decode(struct llama_context* ctx, struct llama_batch batch) {
    #ifdef TENSOR_TRACE_ENABLED
        // Mark token boundary
        if (batch.n_tokens > 0) {
            tensor_trace_new_token(batch.token[0]);
        }
    #endif

    // Original decode logic...

    return result;
}
```

### GGUF File Offset Mapping (During Load)

**File**: `src/llama.cpp`
**Function**: `llama_model_load_internal()`

**Purpose**: Build tensor name → file offset lookup table

**Code**:
```c
// During model loading, for each tensor
for (int i = 0; i < n_tensors; i++) {
    const char* name = gguf_get_tensor_name(ctx_gguf, i);
    size_t offset = gguf_get_tensor_offset(ctx_gguf, i);
    void* data = tensor->data;

    #ifdef TENSOR_TRACE_ENABLED
        // Register tensor for tracing
        tensor_trace_register_tensor(name, data, offset, ggml_nbytes(tensor));
    #endif
}
```

---

## File Organization

### New Directory Structure

```
/Users/ersibesi/Desktop/LLAMA/
├── llama.cpp/                    # Fork with instrumentation
│   ├── ggml/
│   │   ├── include/
│   │   │   └── tensor_trace.h    # ← NEW: Tracing API
│   │   └── src/
│   │       ├── ggml.c            # ← MODIFIED: Add hooks
│   │       └── tensor_trace.c    # ← NEW: Implementation
│   └── src/
│       └── llama.cpp             # ← MODIFIED: Token tracking
│
├── BSC/
│   ├── tensor-tracker/           # ← NEW: Analysis tools
│   │   ├── tools/
│   │   │   ├── gguf_offset_dump.cpp   # Extract GGUF structure
│   │   │   └── parse_trace.py         # Parse binary logs
│   │   ├── analysis/
│   │   │   ├── visualize.py           # Generate plots
│   │   │   └── statistics.py          # Compute stats
│   │   └── notebooks/
│   │       └── analysis.ipynb         # Interactive exploration
│   │
│   └── traces/                   # ← NEW: Trace data
│       └── experiment_001/
│           ├── trace_metadata.json
│           ├── tensor_names.txt
│           ├── tensor_access.bin  # Binary log
│           └── gguf_structure.csv # GGUF map
```

---

## Implementation Phases

### Phase 1: Foundation (Days 1-2)

**Task 1.1: Logging Infrastructure**
- Create `tensor_trace.h` and `tensor_trace.c`
- Implement mmap-based ring buffer
- Write to `/dev/shm/tensor_trace.bin`
- Add initialization/shutdown functions

**Deliverable**: Working logging system (unit tested)

**Task 1.2: GGUF Structure Dumper**
- Write `gguf_offset_dump.cpp`
- Extract tensor names, offsets, sizes
- Output `gguf_structure.csv`

**Deliverable**: GGUF structure map for test models

**Task 1.3: Minimal Instrumentation**
- Add hooks to `ggml_compute_forward()`
- Instrument just ONE operation (MUL_MAT)
- Add token counter in `llama.cpp`
- Test with llama-2-7b, 10 tokens

**Deliverable**: Proof-of-concept trace log

**Task 1.4: Binary Log Parser**
- Write `parse_trace.py`
- Convert binary → pandas DataFrame
- Merge with GGUF structure
- Print summary statistics

**Deliverable**: Working parse pipeline

**Success Criteria**:
- ✅ Can log tensor accesses
- ✅ Binary format reads correctly
- ✅ Token counter increments
- ✅ Overhead <10%

**Time Estimate**: 2 days

---

### Phase 2: Complete Instrumentation (Days 3-5)

**Task 2.1: All Operations**
- Add hooks for all ggml_op types:
  - Matrix ops: MUL_MAT, MUL
  - Attention: FLASH_ATTN, ROPE
  - Norms: RMS_NORM, LAYER_NORM
  - Activations: GELU, SWIGLU, SILU
  - Memory: CPY, VIEW, RESHAPE, PERMUTE
  - Arithmetic: ADD, SUB, MUL, DIV

**Task 2.2: Tensor Name Parsing**
- Extract layer_id from names (e.g., "blk.5" → layer=5)
- Extract attention head (if present)
- Extract expert_id for MoE (e.g., "expert_3" → expert=3)
- Determine Q/K/V type

**Task 2.3: File Offset Mapping**
- Hook model loading
- Build tensor→offset lookup table
- Store in global map for fast lookup

**Task 2.4: Thread ID Tracking**
- Implement `get_thread_id()` for Linux/macOS/Windows
- Add to log entries

**Task 2.5: Full Test**
- Run llama-2-7b, 100 tokens
- Run gpt-oss-20b, 100 tokens (if MoE)
- Verify log completeness

**Success Criteria**:
- ✅ All operations logged
- ✅ Attention details captured
- ✅ MoE experts tracked
- ✅ File offsets correct
- ✅ Thread IDs present

**Time Estimate**: 3 days

---

### Phase 3: Visualization & Analysis (Days 6-8)

**Task 3.1: Core Visualizations**
- Token × Layer heatmap
- Token × Tensor heatmap
- File offset timeline
- Hot tensor bar chart
- Per-layer statistics
- Operation type breakdown

**Task 3.2: Interactive Dashboard**
- Plotly-based interactive plots
- Zoom, pan, filter
- Hover for details
- Export HTML

**Task 3.3: Analysis Notebook**
- Jupyter notebook with all analysis
- Load trace → visualize → interpret
- Document findings

**Task 3.4: MoE-Specific Analysis**
- Expert activation heatmap
- Hot vs cold experts
- Routing score distributions

**Success Criteria**:
- ✅ All visualizations working
- ✅ Interactive exploration smooth
- ✅ Findings documented

**Time Estimate**: 3 days

---

### Phase 4: Advanced Features (Days 9-10, Optional)

**Task 4.1: alphaXiv-Style 3D View**
- 3D scatter: time × layer × file_offset
- Interactive rotation
- Animation over time

**Task 4.2: Detailed Operation Tracing**
- Add Level 3 hooks for MUL_MAT
- Trace cache line accesses
- Correlate with performance

**Task 4.3: Blktrace Correlation**
- Run blktrace alongside tensor trace
- Correlate sector accesses with tensor accesses
- Identify which tensor accesses caused page faults

**Success Criteria**:
- ✅ 3D visualization impressive
- ✅ Deep insights from detailed tracing
- ✅ Blktrace correlation working

**Time Estimate**: 2 days

---

## Critical Success Factors

### What Must Go Right

1. **Zero Data Loss**: Every tensor access must be logged
   - Verify: count accesses, compare to expected (N_layers × N_ops × N_tokens)
   - Validate: check for gaps in timestamps

2. **Accurate Timestamps**: Monotonic, high-resolution
   - Use `CLOCK_MONOTONIC`, not `CLOCK_REALTIME`
   - Nanosecond precision

3. **Correct File Offsets**: Must match GGUF structure
   - Cross-validate with `gguf-dump`
   - Verify: sum of all tensor sizes = GGUF data section size

4. **Low Overhead**: Instrumentation shouldn't slow inference >10%
   - Benchmark: baseline vs instrumented
   - Profile: where is overhead coming from?

5. **Thread Safety**: If llama.cpp uses multiple threads
   - Atomic operations for global counters
   - Thread-local buffers if needed

### What Could Go Wrong

**Risk 1: Log Buffer Overflow**
- **Symptom**: Trace stops mid-inference
- **Mitigation**: Pre-calculate max entries, check capacity
- **Detection**: Add overflow check in log function

**Risk 2: Performance Overhead**
- **Symptom**: Inference 2× slower with tracing
- **Mitigation**: Use mmap, avoid syscalls, profile hot path
- **Fallback**: Reduce logging frequency (sample every Nth access)

**Risk 3: Tensor Names Missing**
- **Symptom**: Many entries with tensor_idx = 0 (unknown)
- **Mitigation**: Fall back to pointer address if name missing
- **Workaround**: Create synthetic names based on operation

**Risk 4: File Offset Lookup Fails**
- **Symptom**: file_offset = UINT64_MAX for many entries
- **Mitigation**: Debug model loading, verify lookup table populated
- **Workaround**: Compute offset from tensor pointer and mmap base

**Risk 5: Concurrency Bugs**
- **Symptom**: Corrupted log entries, crashes
- **Mitigation**: Use atomic operations, thread-local buffers
- **Detection**: Validate log entries (checksums, magic numbers)

---

## Next Steps (Immediate Action Items)

### Tomorrow (January 3, 2026)

**Morning Session (4 hours)**:
1. Create `tensor_trace.h` and `tensor_trace.c` skeleton
2. Implement mmap-based logging to `/dev/shm/`
3. Write unit test (log 1M entries, verify)

**Afternoon Session (4 hours)**:
1. Write `gguf_offset_dump.cpp`
2. Test with llama-2-7b model
3. Verify output CSV is correct

**Evening**:
- Review progress
- Adjust plan if needed

### Day 2 (January 4, 2026)

**Morning**:
1. Add hooks to `ggml_compute_forward()` (one operation)
2. Add token counter in `llama.cpp`
3. Build instrumented llama.cpp

**Afternoon**:
1. Run first trace (10 tokens)
2. Write `parse_trace.py`
3. Verify log correctness

**Evening**:
- Celebrate first working trace! 🎉
- Plan Phase 2

---

## Visualization Roadmap (Future: Phase 2)

### alphaXiv-Style 3D Interactive Visualization

**Inspiration**: https://www.alphaxiv.org/labs/tensor-trace

**What We'll Build**:

```python
import plotly.graph_objects as go

# 3D scatter plot
fig = go.Scatter3d(
    x=df['timestamp_ns'] / 1e9,    # Time (X-axis)
    y=df['layer_id'],               # Layer (Y-axis)
    z=df['file_offset'],            # File offset (Z-axis)
    mode='markers',
    marker=dict(
        size=3,
        color=df['token_id'],       # Color by token
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Token ID')
    ),
    text=[
        f"Token: {row['token_id']}<br>"
        f"Layer: {row['layer_id']}<br>"
        f"Tensor: {row['tensor_name']}<br>"
        f"Size: {row['size_bytes']/1024:.1f} KB<br>"
        f"Thread: {row['thread_id']}"
        for _, row in df.iterrows()
    ],
    hoverinfo='text'
)

fig.update_layout(
    scene=dict(
        xaxis_title='Time (seconds)',
        yaxis_title='Layer ID',
        zaxis_title='File Offset (bytes)',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    title='Tensor Access Pattern - 3D Interactive View',
    hovermode='closest'
)

fig.write_html('tensor_access_3d.html')
```

**Features**:
- ✅ Rotate, zoom, pan (3D interaction)
- ✅ Hover shows tensor details
- ✅ Color by token, layer, thread, or operation
- ✅ Filter by layer, token range, operation type
- ✅ Animation: replay inference over time
- ✅ Export as HTML (shareable, no dependencies)

**Expected Patterns to Discover**:
1. **Sequential layers**: Vertical "bands" for each token
2. **Hot tensors**: Dense clusters at certain file offsets
3. **Thread parallelism**: Different colors (threads) interleaved
4. **MoE sparsity**: Gaps where experts not accessed

**This becomes Phase 2 after we have the data!**

---

## Conclusion

Today we made **critical architectural decisions** that will determine the success of the entire project:

1. ✅ **Logging**: mmap to tmpfs (Option A) - simplest, fastest
2. ✅ **Instrumentation**: Hybrid (Level 2 base + Level 3 strategic) - detailed yet maintainable
3. ✅ **Format**: 64-byte entries with thread_id - complete information
4. ✅ **Location**: `ggml_compute_forward()` - central, backend-agnostic

**Philosophy**: Perfect the foundation first. Visualization is easy once data is clean.

**Next**: Start implementing Phase 1 (logging infrastructure).

---

**Status**: Ready to begin implementation
**Confidence**: High (well-thought-out plan)
**Timeline**: 8-10 days to complete tracker
**Expected Impact**: Foundation for all future optimization work

---

---

## Decision 5: CPU-Only, ALL Operations (CRITICAL REFINEMENT)

**Date**: January 2, 2026 (Evening Update)
**Status**: FINAL ARCHITECTURE DECISION

### The Refinement

After critical review, we made a **fundamental shift** in instrumentation strategy:

**PREVIOUS PLAN**: Hook `ggml_compute_forward()` dispatcher (backend-agnostic, logical access)
**NEW PLAN**: Hook operations in `ggml-cpu.c` (CPU-specific, physical memory access) ⭐

### Why This Changes Everything

**User's Key Insight**: "I would love to have the REAL memory access, this way we could have the granularity of even single operations to log them right?"

The dispatcher-level approach (`ggml_compute_forward()`) logs **logical** tensor access:
- "Operation MUL_MAT accessed tensor X"
- But doesn't know if data came from RAM, page cache, or SSD
- Can't distinguish between fresh read vs cached read

The CPU backend approach (`ggml-cpu.c`) captures **physical** memory access:
- Each operation implementation actually touches `tensor->data`
- Can correlate with page faults, cache misses, SSD reads
- Enables blktrace correlation: "This tensor access caused this disk I/O"

### Implementation Approach

**Option A: Modify ggml-cpu.c directly** ⭐ **SELECTED**

```c
// In ggml/src/ggml-cpu/ggml-cpu.c

static void ggml_compute_forward_mul_mat_f32(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst) {

    const struct ggml_tensor* src0 = dst->src[0];
    const struct ggml_tensor* src1 = dst->src[1];

#ifdef GGML_TENSOR_TRACE
    // Log BEFORE accessing memory
    tensor_trace_log_access(dst, src0, src1, GGML_OP_MUL_MAT);
#endif

    // Actual computation - THIS is where memory is touched
    const float* wdata = (float*)src0->data;  // ← REAL ACCESS
    const float* xdata = (float*)src1->data;  // ← REAL ACCESS
    float* ddata = (float*)dst->data;

    // Matrix multiply inner loops...
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += wdata[i*K + k] * xdata[k*n + j];
            }
            ddata[i*n + j] = sum;
        }
    }
}
```

**Why not create `ggml-traced.c` (separate file)?**
- Would require duplicating all 7000+ lines of ggml-cpu.c
- Maintenance nightmare (need to keep in sync)
- Standard pattern in llama.cpp is compile-time flags (`#ifdef`)

**Compile-time flag approach** (like `GGML_PERF`, `GGML_DEBUG`):
```cmake
option(GGML_TENSOR_TRACE "Enable tensor access tracing" OFF)

if(GGML_TENSOR_TRACE)
    add_definitions(-DGGML_TENSOR_TRACE)
    message(STATUS "Tensor tracing ENABLED")
endif()
```

### What We Log (ALL Operations)

**User Requirement**: "I need truly ALL operations including intermediate tensors"

This means logging:

**1. Model Parameters (mmap'd from GGUF)**:
- ✅ Attention weights: `blk.*.attn_q.weight`, `attn_k.weight`, `attn_v.weight`
- ✅ FFN weights: `blk.*.ffn_up.weight`, `ffn_down.weight`, `ffn_gate.weight`
- ✅ Norms: `blk.*.attn_norm.weight`, `ffn_norm.weight`
- ✅ Embeddings: `token_embd.weight`
- ✅ Output projection: `output.weight`
- ✅ MoE experts: `blk.*.expert_*.ffn_*.weight` (if applicable)

**2. Intermediate Activations (allocated in RAM)**:
- ✅ Attention scores: `Qcur`, `Kcur`, `Vcur`
- ✅ Attention output: `attn_output`
- ✅ FFN activations: `ffn_up_output`, `ffn_gate_output`
- ✅ Residual connections: `inpL`, `cur`
- ✅ Normalized activations: `norm_output`

**3. KV Cache (mmap'd or RAM, depending on --mlock)**:
- ✅ Key cache: `kv_self.k` for each layer
- ✅ Value cache: `kv_self.v` for each layer
- ✅ Cache reads (retrieving context)
- ✅ Cache writes (storing new tokens)

**4. Input Embeddings**:
- ✅ Token embedding lookups: `token_embd.weight[token_id]`
- ✅ Position embeddings (if applicable)

**Why Log Everything?**

From user: "And yes all KV cache operations and even input embeddings accessing would be lovely to keep access of"

**Use Cases**:
1. **Memory Pressure Analysis**: Are KV cache accesses causing page faults?
2. **Cache Reuse**: How often are embeddings re-read?
3. **Activation Patterns**: Which intermediate tensors are largest?
4. **I/O Attribution**: Which parameter reads hit disk vs RAM?

**Example Analysis**:
```python
# Separate model parameters vs intermediate tensors
model_params = df[df['file_offset'] > 0]  # mmap'd from GGUF
intermediates = df[df['file_offset'] == 0]  # allocated in RAM

print(f"Model parameter accesses: {len(model_params):,}")
print(f"Intermediate tensor accesses: {len(intermediates):,}")
print(f"Ratio: {len(intermediates) / len(model_params):.2f}x")

# Expected: 3-5x more intermediate accesses than parameter accesses
```

### Log Volume Estimation

**Conservative Estimate** (100 tokens, llama-2-7b, 32 layers):

```
Per token:
- 32 layers × 15 operations/layer = 480 operations
- Operations include:
  - Embedding lookup: 1
  - Layer operations: 32 × (Q/K/V projection + attention + FFN + norms) = 32 × 15 = 480
  - Output projection: 1
  Total per token: ~500 operations

100 tokens × 500 ops = 50,000 log entries
50,000 × 64 bytes = 3.2 MB

100 tokens × 500 ops × 3 threads (parallelism) = 150,000 entries
150,000 × 64 bytes = 9.6 MB
```

**Buffer capacity**: 2 GB = 33.5M entries → plenty of room!

### Thread Safety (SIMPLIFIED)

**User Question**: "Are you sure? these tokens are processed sequentially? so it would not matter right?"

**Answer**: YES! Tokens are processed **sequentially** in autoregressive generation:

```c
// In llama.cpp - simplified generation loop
for (int n_gen = 0; n_gen < n_predict; n_gen++) {
    // ALL threads work on THIS token
    llama_decode(ctx, batch);  // ← Sequential, one token at a time

    // Sample next token
    int next_token = llama_sample(...);

    // Add to batch for NEXT iteration
    batch = llama_batch_init(...);
}
```

**Threading model**:
- Token N is processed by ALL threads **simultaneously**
- Threads compute different parts of the graph (e.g., different attention heads)
- Once token N completes, ALL threads move to token N+1
- **Never** multiple tokens in flight

**Implications for logging**:

```c
// Global token counter (NO atomics needed)
uint32_t g_current_token = 0;

// In llama_decode(), single-threaded context:
void llama_decode(...) {
    #ifdef GGML_TENSOR_TRACE
        // Increment BEFORE parallel execution
        g_current_token++;  // ← NO RACE: single thread here
    #endif

    // Now multiple threads execute graph
    ggml_graph_compute(...);  // ← All threads read g_current_token

    // All threads finish before we return
}
```

**Thread-local buffers still needed** because multiple threads write log entries simultaneously:

```c
// Thread-local log buffer (avoid contention)
__thread TensorAccessLog local_buffer[1024];
__thread size_t local_offset = 0;

void tensor_trace_log(const TensorAccessLog* entry) {
    local_buffer[local_offset++] = *entry;

    // Flush to global buffer when full
    if (local_offset >= 1024) {
        flush_to_global_buffer(local_buffer, local_offset);
        local_offset = 0;
    }
}
```

### File Offset Mapping (Clarified)

**User Clarification**: "I do not care for only loading! I could see that with blktrace, what interests me is granular parameter accessing to the file!!!"

**What user wants**: Not "when was the file loaded", but "during inference, which part of the GGUF file is hot/cold?"

**The mapping chain**:

```
1. Model Load (one-time):
   GGUF file → mmap() → tensor->data pointers
   Build map: tensor_ptr → {name, file_offset, size}

2. Inference (continuous):
   Operation accesses tensor->data
   Log entry records: {tensor_ptr, timestamp, token_id, ...}

3. Post-processing:
   Join log entries with mapping
   Result: "blk.5.attn_q.weight (file offset 0x1A2B3C) accessed at token 42"

4. Aggregate Analysis:
   Group by file_offset → identify hot regions
   Visualize: file offset vs access count
   Result: "First 500 MB (embeddings + early layers) = 80% of accesses"
```

**Why this is different from blktrace**:

```
blktrace shows:
  "Sector 12345678 was read at time T"
  ↓
  What is sector 12345678? Don't know!

Our tool shows:
  "blk.5.attn_q.weight at file offset 0x1A2B3C accessed at token 42"
  ↓
  Semantic meaning! Layer 5 attention query weights!
```

### Upstreaming to llama.cpp

**User Goal**: "would be a nice tool with llama.cpp and this is what we should strive for"

**How to make this upstreamable**:

1. **Zero impact when disabled**:
   ```c
   #ifndef GGML_TENSOR_TRACE
   // No code changes at all
   #endif
   ```

2. **Minimal invasiveness**:
   - Small `#ifdef` blocks in ggml-cpu.c
   - Self-contained `tensor_trace.c` library
   - Clean CMake option

3. **Useful beyond this thesis**:
   - **Performance profiling**: Which operations are slow?
   - **Memory optimization**: What to cache vs page?
   - **MoE research**: Expert activation patterns
   - **Mobile/edge**: Memory-constrained deployment

4. **Documentation**:
   - Clear README: "Build with `-DGGML_TENSOR_TRACE=ON`"
   - Example workflow: trace → analyze → optimize
   - Use cases: research, profiling, debugging

5. **Reference existing patterns**:
   - Similar to `GGML_PERF` (performance counters)
   - Similar to `GGML_DEBUG` (debug output)
   - Fits llama.cpp philosophy

**Potential PR pitch**:

> **Title**: Add tensor access tracing for memory profiling and optimization
>
> **Summary**: This PR adds optional tensor access tracing to ggml-cpu backend, enabling researchers and developers to analyze memory access patterns during inference. Use cases include:
> - Identifying hot/cold parameters for intelligent caching
> - Profiling memory access patterns for optimization
> - Analyzing MoE expert activation patterns
> - Debugging memory-constrained deployments
>
> **Implementation**: Compile-time flag (`GGML_TENSOR_TRACE`), zero overhead when disabled. Logs tensor accesses to binary file for post-processing.
>
> **Tools included**: Log parser, GGUF structure dumper, visualization scripts.

### Final Architecture Summary

**Instrumentation**:
- Location: `ggml/src/ggml-cpu/ggml-cpu.c`
- Method: `#ifdef GGML_TENSOR_TRACE` in each operation
- Scope: ALL operations (model params, intermediates, KV cache, embeddings)
- Backend: CPU only (sufficient for thesis, upstreamable for GPU later)

**Logging**:
- Method: mmap ring buffer to `/dev/shm/`
- Entry size: 64 bytes (fixed)
- Capacity: 2 GB (33.5M entries)
- Thread safety: Thread-local buffers, sequential token processing

**Mapping**:
- Build during model load: `tensor_ptr → {name, file_offset, size}`
- Use during inference: `log(tensor_ptr, ...)`
- Correlate during analysis: `tensor_ptr → file_offset → semantic name`

**Environment**:
- Linux server: `cli-hiwi-02.dis.cit.tum.de`
- No macOS concerns (tmpfs guaranteed)
- Standard development tools (gcc, cmake, python)

---

## Remaining Questions: RESOLVED ✅

### Q1: Backend Approach
**Resolved**: Modify ggml-cpu.c with `#ifdef GGML_TENSOR_TRACE`

### Q2: What to Log
**Resolved**: ALL operations (model params + intermediates + KV cache + embeddings)

### Q3: Thread Safety
**Resolved**: Sequential token processing (no atomics for token counter), thread-local buffers for logging

### Q4: File Offset Mapping
**Resolved**: Build during load, use during inference, correlate during analysis

### Q5: Upstreaming Strategy
**Resolved**: Minimal invasiveness, useful for community, document well, reference existing patterns

---

## Implementation Ready ✅

**Status**: All critical decisions made
**Confidence**: Very high (thoroughly discussed, user-validated)
**Next Action**: Begin Phase 1 implementation (create logging infrastructure)

**Go/No-Go Decision**: **GO** ✅

---

**End of Journal Entry - January 2, 2026**
