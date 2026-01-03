# December 30, 2025 - Memory Access Tracing & Deterministic Prefetching Strategy

## Overview

**Critical Realization**: Current blktrace methodology captures block-level I/O but lacks semantic understanding of WHICH model components are accessed. This limits our ability to:
1. Understand what gets re-read under memory pressure
2. Optimize hot vs cold parameter placement
3. Implement intelligent prefetching strategies
4. Analyze MoE expert activation patterns

**New Direction**: Combine GGUF file structure mapping with in-application memory access tracing to generate semantic heatmaps and enable deterministic prefetching.

---

## Part 1: Limitations of Current Approach (blktrace)

### What We Know from 22DEC.md

**Finding: 100% Sequential Access** (with important caveat!)
```
0 GB lock (model fits in RAM):
- Sequential: 100.0%
- Backward seeks: 0.0%
- Total reads: 12.85 GB
```

**CRITICAL CORRECTION**: This 100% sequential pattern applies **ONLY to the initial model loading phase** (first token generation, cold start).

**What happens**:
1. **First token (t=0-5s)**: llama.cpp loads model sequentially from disk
   - Reads: embeddings → layer 0 → layer 1 → ... → layer N
   - Pattern: 100% sequential (file layout matches access order)
   - Throughput: ~40 GB/s (near SSD theoretical max)

2. **Subsequent tokens (t=5-30s)**: Access pattern changes
   - Some parameters re-accessed (re-read factor: 1.13x under pressure)
   - Pattern: NOT necessarily sequential
   - Throughput: degrades under memory pressure

### What We DON'T Know (The Problem!)

From 22DEC.md findings:
```
20 GB lock (severe memory pressure):
- Total read: 14.52 GB
- Unique: 12.85 GB
- Re-read: 1.67 GB (13% amplification)
```

**Critical question: WHICH 1.67 GB gets re-read?**

blktrace can only tell us:
- ❌ "Sector 12345678 was read 3 times"
- ❌ "Sector 98765432 was read 15 times"

But we CANNOT answer:
- ❓ Is it token embeddings?
- ❓ Is it specific layer weights?
- ❓ Is it attention Q/K/V matrices?
- ❓ Is it FFN weights?
- ❓ Is it layer norms?
- ❓ For MoE: which experts?

**This semantic gap prevents optimization!**

### Implications for MoE Models

**Standard transformer (Llama-2)**:
- All layers accessed for every token
- Sequential access during first load makes sense

**Mixture of Experts (MoE) - DIFFERENT!**:
- Only K out of N experts activated per token
- Expert selection is **non-uniform** (some experts hot, others cold)
- File layout may not match access pattern
- 100% sequential on first load, but **NOT on subsequent tokens!**

**Example (hypothetical)**:
```
File layout:    expert_0, expert_1, expert_2, expert_3, expert_4, expert_5
Token 1 access: expert_0, expert_2          ← Sequential at file level
Token 2 access: expert_1, expert_4          ← SKIP expert_2,3 (not sequential!)
Token 3 access: expert_0, expert_2          ← Backward jump to expert_0
```

**We NEED semantic tracing to understand this!**

---

## Part 2: Proposed Solution - GGUF Mapping + In-App Tracing

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Phase 1: GGUF Mapping                  │
└─────────────────────────────────────────────────────────┘
                            ↓
        Extract tensor metadata: name, offset, size, type
                            ↓
┌────────────────────────────────────────────────────────┐
│ Tensor Name              | Offset    | Size   | Type   │
├──────────────────────────┼───────────┼────────┼────────┤
│ token_embd.weight        | 524288    | 500 MB | F16    │
│ blk.0.attn_q.weight      | 524812288 | 64 MB  | F16    │
│ blk.0.attn_k.weight      | 588890112 | 64 MB  | F16    │
│ blk.0.attn_v.weight      | 652967936 | 64 MB  | F16    │
│ blk.0.ffn_up.weight      | 720000000 | 128 MB | F16    │
│ ...                      | ...       | ...    | ...    │
└────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│          Phase 2: Runtime Access Tracking               │
└─────────────────────────────────────────────────────────┘
                            ↓
    Instrument llama.cpp to log tensor accesses
                            ↓
┌────────────────────────────────────────────────────────┐
│ Timestamp | Tensor Name          | Bytes   | Token    │
├───────────┼──────────────────────┼─────────┼──────────┤
│ 0.123     | token_embd.weight    | 2048    | 0 (load) │
│ 0.145     | blk.0.attn_q.weight  | 67108   | 0        │
│ 0.167     | blk.0.attn_k.weight  | 67108   | 0        │
│ ...       | ...                  | ...     | ...      │
│ 5.234     | token_embd.weight    | 2048    | 1 (gen)  │
│ 5.256     | blk.0.attn_q.weight  | 67108   | 1        │
│ 10.123    | token_embd.weight    | 2048    | 2 (gen)  │
└────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              Phase 3: Analysis & Heatmap                │
└─────────────────────────────────────────────────────────┘
```

### Phase 1: GGUF Structure Extraction

**Goal**: Create a mapping from file offsets to semantic tensor names.

**Implementation**:
```cpp
// dump_gguf_offsets.cpp
#include "ggml.h"
#include "gguf.h"
#include <stdio.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model.gguf> [output.csv]\n", argv[0]);
        return 1;
    }

    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx = NULL
    };

    struct gguf_context * ctx = gguf_init_from_file(argv[1], params);
    if (!ctx) {
        fprintf(stderr, "Failed to load GGUF file\n");
        return 1;
    }

    // Output header
    printf("tensor_name,offset_bytes,size_bytes,type,layer,component\n");

    const int n_tensors = gguf_get_n_tensors(ctx);
    const size_t data_offset = gguf_get_data_offset(ctx);

    for (int i = 0; i < n_tensors; ++i) {
        const char * name   = gguf_get_tensor_name(ctx, i);
        const size_t size   = gguf_get_tensor_size(ctx, i);
        const size_t offset = gguf_get_tensor_offset(ctx, i);
        const auto   type   = gguf_get_tensor_type(ctx, i);

        // Parse layer and component from name
        // E.g., "blk.5.attn_q.weight" → layer=5, component="attn_q"
        int layer = -1;
        char component[256] = {0};

        if (strstr(name, "blk.") == name) {
            sscanf(name, "blk.%d.%255s", &layer, component);
        } else {
            strncpy(component, name, 255);
        }

        // Absolute file offset = data_offset + tensor_offset
        size_t abs_offset = data_offset + offset;

        printf("%s,%zu,%zu,%s,%d,%s\n",
               name,
               abs_offset,
               size,
               ggml_type_name(type),
               layer,
               component);
    }

    gguf_free(ctx);
    return 0;
}
```

**Output example** (`gguf_structure.csv`):
```csv
tensor_name,offset_bytes,size_bytes,type,layer,component
token_embd.weight,524288,524288000,F16,-1,token_embd.weight
output_norm.weight,524812288,16384,F32,-1,output_norm.weight
blk.0.attn_norm.weight,524828672,16384,F32,0,attn_norm.weight
blk.0.attn_q.weight,524845056,67108864,F16,0,attn_q.weight
blk.0.attn_k.weight,591953920,67108864,F16,0,attn_k.weight
blk.0.attn_v.weight,659062784,67108864,F16,0,attn_v.weight
blk.0.attn_output.weight,726171648,67108864,F16,0,attn_output.weight
blk.0.ffn_norm.weight,793280512,16384,F32,0,ffn_norm.weight
blk.0.ffn_up.weight,793296896,134217728,F16,0,ffn_up.weight
blk.0.ffn_down.weight,927514624,134217728,F16,0,ffn_down.weight
blk.0.ffn_gate.weight,1061732352,134217728,F16,0,ffn_gate.weight
blk.1.attn_norm.weight,1195950080,16384,F32,1,attn_norm.weight
...
```

**Key information captured**:
1. ✅ Tensor name (semantic meaning)
2. ✅ Absolute file offset (for correlation with access logs)
3. ✅ Size (to understand access granularity)
4. ✅ Type (quantization level)
5. ✅ Layer number (for per-layer analysis)
6. ✅ Component type (attn_q, ffn_up, etc.)

### Phase 2: Runtime Access Tracking

**Goal**: Log every tensor access during inference with timestamps.

**Instrumentation Points**:

#### Option A: Kernel-Level (via LD_PRELOAD)
Hook `mmap`, `munmap`, `madvise` to track when model pages are accessed:

```cpp
// libaccess_tracker.so
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <sys/mman.h>
#include <time.h>

static FILE* g_log_file = NULL;
static void* g_model_base = NULL;
static size_t g_model_size = 0;

void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    static void* (*real_mmap)(void*, size_t, int, int, int, off_t) = NULL;
    if (!real_mmap) {
        real_mmap = dlsym(RTLD_NEXT, "mmap");
    }

    void* result = real_mmap(addr, length, prot, flags, fd, offset);

    // Check if this is a model file (heuristic: large read-only mapping)
    if (result != MAP_FAILED && length > 1024*1024*100 && (prot & PROT_READ)) {
        g_model_base = result;
        g_model_size = length;

        if (!g_log_file) {
            g_log_file = fopen("access_log.csv", "w");
            fprintf(g_log_file, "timestamp,offset,size,page_fault\n");
        }

        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double timestamp = ts.tv_sec + ts.tv_nsec / 1e9;

        fprintf(g_log_file, "%.6f,%zu,%zu,mmap\n", timestamp, offset, length);
    }

    return result;
}

// Hook page faults via signal handler (SIGSEGV)
// (More complex, requires mprotect trickery)
```

**Pro**: Captures actual page faults (OS-level precision)
**Con**: Complex, may miss cached accesses

#### Option B: Application-Level (Instrument llama.cpp) ⭐ **RECOMMENDED**

Hook tensor access in `ggml_graph_compute()`:

```cpp
// In ggml/src/ggml.c or ggml/src/ggml-cpu/ggml-cpu.c

#ifdef ENABLE_ACCESS_TRACKING
static FILE* g_access_log = NULL;
static double g_start_time = 0.0;
static int g_current_token = 0;

static void log_tensor_access(const struct ggml_tensor * tensor, size_t bytes_accessed) {
    if (!g_access_log) {
        g_access_log = fopen("tensor_access_log.csv", "w");
        fprintf(g_access_log, "timestamp,token,tensor_name,data_ptr,bytes_accessed\n");

        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        g_start_time = ts.tv_sec + ts.tv_nsec / 1e9;
    }

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double timestamp = (ts.tv_sec + ts.tv_nsec / 1e9) - g_start_time;

    fprintf(g_access_log, "%.6f,%d,%s,%p,%zu\n",
            timestamp,
            g_current_token,
            ggml_get_name(tensor),
            tensor->data,
            bytes_accessed);
    fflush(g_access_log);
}

void ggml_graph_compute_increment_token() {
    g_current_token++;
}
#endif

// In the compute function for each operation:
static void ggml_compute_forward_mul_mat(...) {
    #ifdef ENABLE_ACCESS_TRACKING
    log_tensor_access(src0, ggml_nbytes(src0));  // Log weight matrix access
    log_tensor_access(src1, ggml_nbytes(src1));  // Log input activation access
    #endif

    // Actual computation...
}

// Similar for other operations (ADD, RMS_NORM, ROPE, etc.)
```

**Modification needed in llama.cpp main inference loop**:
```cpp
// In llama.cpp/src/llama.cpp, around token generation loop

for (int i = 0; i < n_tokens; i++) {
    #ifdef ENABLE_ACCESS_TRACKING
    ggml_graph_compute_increment_token();  // Mark new token start
    #endif

    // Build graph
    struct ggml_cgraph * graph = llama_build_graph(...);

    // Compute
    ggml_backend_sched_graph_compute(sched, graph);

    // Sample next token
    // ...
}
```

**Output** (`tensor_access_log.csv`):
```csv
timestamp,token,tensor_name,data_ptr,bytes_accessed
0.000123,0,token_embd.weight,0x7f8a4c000000,524288000
0.001234,0,blk.0.attn_norm.weight,0x7f8a6d400000,16384
0.001456,0,blk.0.attn_q.weight,0x7f8a6d404000,67108864
0.001789,0,blk.0.attn_k.weight,0x7f8a71404000,67108864
0.002123,0,blk.0.attn_v.weight,0x7f8a75404000,67108864
0.002456,0,blk.0.attn_output.weight,0x7f8a79404000,67108864
...
5.234567,1,token_embd.weight,0x7f8a4c000000,2048
5.234890,1,blk.0.attn_norm.weight,0x7f8a6d400000,16384
5.235123,1,blk.0.attn_q.weight,0x7f8a6d404000,67108864
...
10.456789,2,token_embd.weight,0x7f8a4c000000,2048
...
```

**Pro**:
- Semantic awareness (know tensor name)
- Per-token tracking
- Simple implementation
- Low overhead (~1-2%)

**Con**:
- Misses very low-level cache effects
- Requires llama.cpp modification

### Phase 3: Correlation & Analysis

**Merge GGUF structure with access log**:

```python
# analyze_access_patterns.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load GGUF structure
gguf_df = pd.read_csv("gguf_structure.csv")
# Columns: tensor_name, offset_bytes, size_bytes, type, layer, component

# Load access log
access_df = pd.read_csv("tensor_access_log.csv")
# Columns: timestamp, token, tensor_name, data_ptr, bytes_accessed

# Merge to get full context
df = access_df.merge(gguf_df, on='tensor_name', how='left')

# Now we can answer ALL the research questions!
```

#### Analysis 1: What Gets Re-Read? (The 1.67 GB Mystery!)

```python
# Count accesses per tensor
access_counts = df.groupby('tensor_name').size().reset_index(name='access_count')

# Merge with size to calculate total bytes re-read
reread_df = access_counts.merge(gguf_df[['tensor_name', 'size_bytes']], on='tensor_name')
reread_df['total_bytes_read'] = reread_df['access_count'] * reread_df['size_bytes']
reread_df['reread_bytes'] = (reread_df['access_count'] - 1) * reread_df['size_bytes']

# Sort by re-read amount
reread_df = reread_df.sort_values('reread_bytes', ascending=False)

print("Top 10 Most Re-Read Tensors:")
print(reread_df.head(10)[['tensor_name', 'access_count', 'size_bytes', 'reread_bytes']])

# Expected output (HYPOTHESIS - TO BE VERIFIED!):
# tensor_name              access_count  size_bytes   reread_bytes
# token_embd.weight        100           524288000    51904512000  (99 re-reads!)
# blk.0.attn_q.weight      3             67108864     134217728
# blk.0.attn_k.weight      3             67108864     134217728
# ...
```

**This finally answers: "WHICH 1.67 GB gets re-read?"**

#### Analysis 2: Per-Token Access Pattern (Deterministic Pattern Discovery)

```python
# Analyze pattern for a single token (should be identical across all tokens)
token_1_pattern = df[df['token'] == 1][['timestamp', 'tensor_name', 'layer', 'component']].copy()
token_1_pattern['relative_time'] = token_1_pattern['timestamp'] - token_1_pattern['timestamp'].min()

print("Token 1 Access Pattern:")
print(token_1_pattern)

# Expected output:
# relative_time  tensor_name              layer  component
# 0.000000       token_embd.weight        -1     token_embd.weight
# 0.000123       blk.0.attn_norm.weight   0      attn_norm.weight
# 0.000256       blk.0.attn_q.weight      0      attn_q.weight
# 0.000389       blk.0.attn_k.weight      0      attn_k.weight
# 0.000512       blk.0.attn_v.weight      0      attn_v.weight
# ...
# 0.123456       blk.31.attn_output.weight 31    attn_output.weight
# 0.123789       output_norm.weight       -1     output_norm.weight

# Verify pattern is IDENTICAL across tokens
token_2_pattern = df[df['token'] == 2][['tensor_name']].values.flatten()
token_3_pattern = df[df['token'] == 3][['tensor_name']].values.flatten()

if (token_1_pattern['tensor_name'].values == token_2_pattern).all():
    print("✓ Per-token pattern is DETERMINISTIC!")
else:
    print("✗ Pattern varies between tokens (unexpected!)")
```

**This discovers the deterministic access sequence!**

#### Analysis 3: Heatmap - Which Tensors Are Hot?

```python
# Heatmap 1: Access count per tensor (across all tokens)
plt.figure(figsize=(20, 10))

# Calculate access count per tensor per token
heatmap_data = df.groupby(['tensor_name', 'token']).size().unstack(fill_value=0)

# Sort by layer (for visual structure)
df_sorted = df.drop_duplicates('tensor_name').sort_values('layer')
tensor_order = df_sorted['tensor_name'].tolist()
heatmap_data = heatmap_data.reindex(tensor_order)

sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': 'Access Count'})
plt.xlabel('Token Number')
plt.ylabel('Tensor Name')
plt.title('Memory Access Heatmap: Which Tensors Are Accessed Per Token')
plt.tight_layout()
plt.savefig('heatmap_access_per_token.png', dpi=300)
plt.close()

print("Saved: heatmap_access_per_token.png")
```

**Visual output**:
- Horizontal bands = tensors accessed every token (embeddings?)
- Vertical stripes = tokens with unusual patterns
- Dark red = hot tensors

#### Analysis 4: Temporal Access Pattern (Sequential vs Random Over Time)

```python
# Plot offset accessed over time
plt.figure(figsize=(16, 8))

plt.subplot(2, 1, 1)
plt.scatter(df['timestamp'], df['offset_bytes'], c=df['token'], cmap='viridis', alpha=0.5, s=1)
plt.xlabel('Time (seconds)')
plt.ylabel('File Offset (bytes)')
plt.title('Access Pattern Over Time (Color = Token Number)')
plt.colorbar(label='Token')

# Expected patterns:
# - Diagonal line (t=0-5s): Sequential loading
# - Horizontal bands (t=5-30s): Re-accessing same offsets (embeddings?)
# - Scattered points: Random access under memory pressure

plt.subplot(2, 1, 2)
# Calculate gaps between consecutive accesses
df_sorted = df.sort_values('timestamp').copy()
df_sorted['offset_gap'] = df_sorted['offset_bytes'].diff().abs()
df_sorted['gap_category'] = pd.cut(df_sorted['offset_gap'],
                                    bins=[0, 128*1024, 1024*1024, float('inf')],
                                    labels=['Sequential (<128KB)', 'Small Gap (128KB-1MB)', 'Large Gap (>1MB)'])

df_sorted.groupby('gap_category').size().plot(kind='bar')
plt.xlabel('Gap Category')
plt.ylabel('Count')
plt.title('Access Pattern: Gap Distribution')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('temporal_access_pattern.png', dpi=300)
plt.close()

print("Saved: temporal_access_pattern.png")
```

#### Analysis 5: Per-Layer Access Frequency

```python
# Which layers are accessed most?
layer_access = df[df['layer'] >= 0].groupby('layer').size().reset_index(name='access_count')

plt.figure(figsize=(12, 6))
plt.bar(layer_access['layer'], layer_access['access_count'])
plt.xlabel('Layer Number')
plt.ylabel('Total Access Count')
plt.title('Access Frequency Per Layer')
plt.grid(axis='y', alpha=0.3)
plt.savefig('layer_access_frequency.png', dpi=300)
plt.close()

print("Saved: layer_access_frequency.png")

# Expected: Should be roughly uniform (each layer accessed once per token)
# Deviation indicates non-uniform access (interesting!)
```

#### Analysis 6: Component-Level Analysis (Q, K, V, FFN)

```python
# Which components (attn_q, attn_k, ffn_up, etc.) are hottest?
component_access = df[df['component'] != ''].groupby('component').agg({
    'tensor_name': 'count',
    'bytes_accessed': 'sum'
}).rename(columns={'tensor_name': 'access_count', 'bytes_accessed': 'total_bytes'})

component_access = component_access.sort_values('total_bytes', ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Access count
component_access['access_count'].plot(kind='barh', ax=ax1)
ax1.set_xlabel('Access Count')
ax1.set_title('Component Access Frequency')

# Total bytes
(component_access['total_bytes'] / 1e9).plot(kind='barh', ax=ax2)
ax2.set_xlabel('Total Bytes Accessed (GB)')
ax2.set_title('Component Data Volume')

plt.tight_layout()
plt.savefig('component_analysis.png', dpi=300)
plt.close()

print("Saved: component_analysis.png")
```

---

## Part 3: Deterministic Prefetching Strategies

### Strategy 1: Pin Hot Tensors in RAM

**Based on access tracking results**, identify hot tensors and pin them:

```cpp
// After first token, analyze access pattern
std::map<std::string, int> access_counts;

// Determine hot tensors (accessed > 1 time per token on average)
std::vector<std::string> hot_tensors;
for (auto& [name, count] : access_counts) {
    if (count > num_tokens) {
        hot_tensors.push_back(name);
    }
}

// Pin hot tensors in RAM (prevent eviction)
for (auto& name : hot_tensors) {
    struct ggml_tensor* tensor = find_tensor_by_name(ctx, name);
    mlock(tensor->data, ggml_nbytes(tensor));
    printf("Pinned hot tensor: %s (%.2f MB)\n",
           name.c_str(),
           ggml_nbytes(tensor) / 1048576.0);
}
```

**Expected hot tensors**:
- `token_embd.weight` (accessed every token)
- Possibly layer norms (small, frequently accessed)
- Possibly first few layers (if accessed more often)

**Expected improvement**:
- Eliminate re-reads of hot tensors
- Reduce I/O by ~50% if embeddings are the main re-read culprit

### Strategy 2: Async I/O with io_uring (Compute-I/O Overlap) ⭐

**Goal**: Prefetch layer N+1 while computing layer N.

**Implementation using io_uring (Linux)**:

```cpp
// prefetch_engine.cpp
#include <liburing.h>
#include <fcntl.h>
#include <sys/mman.h>

struct PrefetchEngine {
    struct io_uring ring;
    int model_fd;
    void* model_base;
    size_t model_size;

    // Pre-computed access pattern (from Phase 3 analysis)
    struct AccessPattern {
        size_t offset;
        size_t size;
        void* dest_addr;
    };
    std::vector<AccessPattern> per_token_pattern;

    void init(const char* model_path, const std::vector<AccessPattern>& pattern) {
        // Initialize io_uring
        io_uring_queue_init(32, &ring, 0);

        // Open model file
        model_fd = open(model_path, O_RDONLY | O_DIRECT);  // O_DIRECT for async I/O

        per_token_pattern = pattern;
    }

    void prefetch_next_layer(int current_layer) {
        // Calculate which tensors belong to layer N+1
        int next_layer = current_layer + 1;

        for (auto& access : per_token_pattern) {
            // Check if this access belongs to next layer
            // (Parse tensor name from pattern metadata)
            if (/* belongs to next_layer */) {
                // Submit async read
                struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                io_uring_prep_read(sqe, model_fd, access.dest_addr, access.size, access.offset);
                io_uring_submit(&ring);
            }
        }
    }

    void wait_prefetch_complete() {
        struct io_uring_cqe* cqe;
        io_uring_wait_cqe(&ring, &cqe);
        io_uring_cqe_seen(&ring, cqe);
    }
};

// Modified inference loop
void inference_with_prefetch(PrefetchEngine& prefetch, int n_tokens) {
    for (int token = 0; token < n_tokens; token++) {
        for (int layer = 0; layer < n_layers; layer++) {
            // Start prefetching NEXT layer while computing current
            if (layer + 1 < n_layers) {
                prefetch.prefetch_next_layer(layer);
            }

            // Compute current layer (CPU busy)
            compute_layer(layer);

            // By the time we finish compute, prefetch should be done
            if (layer + 1 < n_layers) {
                prefetch.wait_prefetch_complete();
            }
        }
    }
}
```

**Expected improvement**:
- **Theoretical max**: 2x speedup (hide ALL I/O latency)
- **Practical**: 1.5-1.8x speedup (compute time > I/O time for most layers)
- Only works under memory pressure (otherwise already cached)

**Requirements**:
- Linux kernel 5.1+ (io_uring support)
- O_DIRECT I/O (bypass page cache)
- Accurate per-layer access pattern

### Strategy 3: Pre-Computed Per-Token Access List

**Goal**: Build the exact access sequence once, reuse for every token.

**Step 1**: Run inference with tracing enabled (once)
```bash
ENABLE_ACCESS_TRACKING=1 ./llama-cli -m model.gguf -p "test" -n 100
# Generates: tensor_access_log.csv
```

**Step 2**: Extract per-token pattern (deterministic!)
```python
# extract_pattern.py
import pandas as pd

df = pd.read_csv("tensor_access_log.csv")

# Extract pattern from token 1 (should be same for all tokens)
pattern = df[df['token'] == 1][['tensor_name', 'bytes_accessed']].copy()

# Merge with GGUF offsets
gguf_df = pd.read_csv("gguf_structure.csv")
pattern = pattern.merge(gguf_df[['tensor_name', 'offset_bytes']], on='tensor_name')

# Save as C++ array initializer
with open("access_pattern.h", "w") as f:
    f.write("// Auto-generated per-token access pattern\n")
    f.write("struct AccessEntry {\n")
    f.write("    const char* name;\n")
    f.write("    size_t offset;\n")
    f.write("    size_t size;\n")
    f.write("};\n\n")
    f.write(f"const AccessEntry PER_TOKEN_PATTERN[{len(pattern)}] = {{\n")

    for _, row in pattern.iterrows():
        f.write(f'    {{"{row["tensor_name"]}", {row["offset_bytes"]}, {row["bytes_accessed"]}}},\n')

    f.write("};\n")

print(f"Generated access_pattern.h with {len(pattern)} entries")
```

**Step 3**: Use pattern for prefetching
```cpp
// optimized_inference.cpp
#include "access_pattern.h"

void inference_with_pattern() {
    for (int token = 0; token < n_tokens; token++) {
        // For each token, access pattern is IDENTICAL
        for (int i = 0; i < sizeof(PER_TOKEN_PATTERN) / sizeof(AccessEntry); i++) {
            auto& entry = PER_TOKEN_PATTERN[i];

            // Prefetch next 2-3 entries ahead
            if (i + 2 < sizeof(PER_TOKEN_PATTERN) / sizeof(AccessEntry)) {
                auto& next = PER_TOKEN_PATTERN[i + 2];
                // Async prefetch
                __builtin_prefetch((void*)(model_base + next.offset), 0, 3);
                // Or use io_uring for SSD prefetch
            }

            // Access current entry (should be prefetched by now)
            void* data = (void*)(model_base + entry.offset);
            // Use data for computation...
        }
    }
}
```

**Expected improvement**:
- Deterministic prefetch schedule
- Optimal prefetch distance (2-3 entries ahead)
- Works across all tokens (pattern is identical)
- ~1.3-1.5x speedup under memory pressure

### Strategy 4: MoE Expert-Aware Prefetching

**For Mixture-of-Experts models**, access pattern is more complex:

```python
# Analyze MoE expert usage
moe_df = df[df['tensor_name'].str.contains('expert_')]

# Extract expert ID from name: "blk.5.expert_3.ffn_up.weight" → expert=3
moe_df['expert_id'] = moe_df['tensor_name'].str.extract(r'expert_(\d+)')[0].astype(int)

# Count expert activations per token
expert_usage = moe_df.groupby(['token', 'expert_id']).size().unstack(fill_value=0)

print("Expert Usage Matrix (rows=tokens, cols=expert_id):")
print(expert_usage)

# Expected output:
#       expert_0  expert_1  expert_2  expert_3  expert_4  expert_5
# token
# 0            2         2         0         0         0         0
# 1            2         0         2         0         0         0
# 2            2         2         0         0         0         0
# 3            0         2         0         2         0         0

# Identify hot vs cold experts
total_usage = expert_usage.sum(axis=0)
hot_experts = total_usage[total_usage > total_usage.mean()].index.tolist()
cold_experts = total_usage[total_usage <= total_usage.mean()].index.tolist()

print(f"Hot experts (pin in RAM): {hot_experts}")
print(f"Cold experts (allow paging): {cold_experts}")
```

**Prefetching strategy for MoE**:
```cpp
// For MoE models, expert selection depends on input
// But we can predict based on routing scores

void moe_prefetch(int current_layer, float* routing_scores, int top_k) {
    // Get top-K expert IDs from routing scores
    std::vector<int> selected_experts = top_k_indices(routing_scores, top_k);

    // Prefetch only selected experts
    for (int expert_id : selected_experts) {
        // Construct tensor name: "blk.{layer}.expert_{id}.ffn_up.weight"
        char name[256];
        snprintf(name, sizeof(name), "blk.%d.expert_%d.ffn_up.weight", current_layer, expert_id);

        // Prefetch this expert's weights
        struct ggml_tensor* expert_tensor = find_tensor(name);
        async_prefetch(expert_tensor->data, ggml_nbytes(expert_tensor));
    }
}
```

**Expected improvement for MoE**:
- Only prefetch activated experts (not all N experts)
- Reduce I/O by factor of N/K (e.g., 8 experts, 2 active → 4x reduction)
- Critical for large MoE models (e.g., Mixtral 8x7B)

---

## Part 4: Implementation Roadmap

### Phase 0: Baseline Measurement (Already Done!)
- ✅ 22DEC.md: blktrace measurements
- ✅ Know: 100% sequential on first load, 1.13x amplification under pressure
- ✅ Know: Performance cliff at 16 GB lock
- ❌ Don't know: WHICH tensors get re-read

### Phase 1: GGUF Mapping Tool (1-2 hours)
**Steps**:
1. Write `dump_gguf_offsets.cpp` (code provided above)
2. Compile with llama.cpp libraries
3. Run on all test models:
   ```bash
   ./dump_gguf_offsets models/llama-2-7b-chat.Q4_K_M.gguf > llama2_structure.csv
   ./dump_gguf_offsets models/gpt-oss-20b-F16.gguf > gpt_structure.csv
   ```
4. Verify output: check tensor counts, offsets, sizes

**Deliverable**: CSV files mapping tensors to file offsets

### Phase 2: Instrument llama.cpp (4-6 hours)
**Steps**:
1. Add `#ifdef ENABLE_ACCESS_TRACKING` to ggml.c
2. Implement `log_tensor_access()` function
3. Hook all operation dispatch points (MUL_MAT, ADD, RMS_NORM, ROPE, etc.)
4. Add per-token counter in llama.cpp main loop
5. Compile with `-DENABLE_ACCESS_TRACKING`
6. Test with small model (llama-2-7b, 10 tokens)
7. Verify log output format

**Deliverable**: Modified llama.cpp with access tracking

### Phase 3: Run Experiments with Tracing (2-3 hours)
**Experiment matrix**:
```bash
# Baseline (model fits)
ENABLE_ACCESS_TRACKING=1 ./llama-cli -m llama-2-7b.gguf -p "test" -n 100 --mlock 0GB
# Output: access_log_0GB.csv

# Memory pressure
ENABLE_ACCESS_TRACKING=1 ./llama-cli -m llama-2-7b.gguf -p "test" -n 100 --mlock 23GB
# Output: access_log_23GB.csv

# MoE model (if available)
ENABLE_ACCESS_TRACKING=1 ./llama-cli -m gpt-oss-20b.gguf -p "test" -n 100 --mlock 17GB
# Output: access_log_gpt.csv
```

**Deliverable**: Access logs for all scenarios

### Phase 4: Analysis & Heatmap Generation (3-4 hours)
**Steps**:
1. Write `analyze_access_patterns.py` (code provided above)
2. Generate heatmaps for all experiments
3. Identify hot tensors, per-token patterns, re-read culprits
4. Document findings in journal

**Deliverable**:
- Heatmap visualizations (PNG)
- Access pattern statistics (CSV)
- Findings document (MD)

### Phase 5: Implement Prefetching Strategies (8-12 hours)
**Priority order**:
1. **Strategy 1** (Pin hot tensors) - Easiest, immediate benefit
2. **Strategy 3** (Pre-computed pattern) - Medium difficulty, good benefit
3. **Strategy 2** (io_uring async I/O) - Hardest, maximum benefit
4. **Strategy 4** (MoE-aware) - Only if using MoE models

**Steps per strategy**:
1. Implement in separate branch
2. Benchmark against baseline
3. Measure speedup and I/O reduction
4. Document results

**Deliverable**: Modified llama.cpp with prefetching optimizations

### Phase 6: Evaluation & Thesis Writing (1 week)
**Experiments**:
- Baseline vs. each optimization strategy
- Combined optimizations
- Different memory pressure levels
- Different models (standard vs. MoE)

**Metrics**:
- Tokens/sec improvement
- I/O reduction (MB read)
- SSD bandwidth utilization
- Memory usage

**Deliverable**:
- Performance comparison tables
- Thesis chapter on optimization strategies
- Potential conference paper

---

## Part 5: Expected Research Outcomes

### Research Question 1: What Gets Re-Read?

**Current status**: Unknown (blktrace only shows sectors)

**After tracing**:
```
Expected findings:
- token_embd.weight: X accesses (Y% of re-reads)
- Layer-specific patterns
- Component-specific patterns (attn_q vs. ffn_up)
```

**Thesis contribution**: "Characterization of memory access patterns in LLM inference under constrained memory"

### Research Question 2: Per-Token Access Determinism

**Current status**: Assumed deterministic, not proven

**After tracing**:
```
Verify:
- Token N and Token N+1 have IDENTICAL access sequence
- Sequence length (number of tensor accesses per token)
- Total bytes accessed per token
```

**Thesis contribution**: "Deterministic access pattern enables predictive prefetching"

### Research Question 3: Optimization Potential

**Current status**:
- Baseline: 4.84 tok/s (model fits)
- Pressure: 3.02 tok/s (severe memory pressure)
- Slowdown: 38%

**After optimizations**:
```
Expected:
- Strategy 1 (pin hot): 3.5-4.0 tok/s (+16-32%)
- Strategy 2 (async I/O): 4.5-5.0 tok/s (+50-65%)
- Combined: 5.0-5.5 tok/s (+65-80%)
```

**Thesis contribution**: "Software-controlled prefetching achieves 1.5-2x speedup under memory pressure"

### Research Question 4: MoE Expert Usage

**Current status**: Unknown if gpt-oss-20b is MoE

**After tracing**:
```
If MoE:
- Identify hot vs. cold experts
- Measure expert activation sparsity
- Quantify access pattern difference vs. standard transformer
```

**Thesis contribution**: "MoE models exhibit non-uniform expert access, enabling selective caching"

---

## Part 6: Critical Success Factors

### Technical Requirements
1. ✅ GGUF API for offset mapping (available in llama.cpp)
2. ✅ Low-overhead logging (fprintf ~1-2% overhead)
3. ⚠️ io_uring support (Linux only, kernel 5.1+)
4. ⚠️ Sufficient RAM for experiments (30 GB available)

### Risks & Mitigations

**Risk 1**: Access tracking overhead affects measurements
- **Mitigation**: Run baseline without tracking, compare performance
- **Acceptance**: <5% overhead is acceptable

**Risk 2**: io_uring implementation complexity
- **Mitigation**: Start with simpler strategies (1 & 3), add Strategy 2 if time permits
- **Fallback**: Use posix_fadvise() for prefetching (less optimal but simpler)

**Risk 3**: Per-token pattern is NOT deterministic
- **Mitigation**: If pattern varies, analyze WHY (sampling? context length?)
- **Adaptation**: Develop probabilistic prefetching based on common patterns

**Risk 4**: Prefetching doesn't improve performance
- **Mitigation**: Profile to understand bottleneck (compute-bound vs. I/O-bound)
- **Adaptation**: Focus on hot tensor pinning instead of async prefetch

### Timeline Estimate

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: GGUF mapping | 2 hours | 2 hours |
| Phase 2: Instrumentation | 6 hours | 8 hours |
| Phase 3: Run experiments | 3 hours | 11 hours |
| Phase 4: Analysis | 4 hours | 15 hours |
| Phase 5: Prefetching (Strategy 1) | 4 hours | 19 hours |
| Phase 5: Prefetching (Strategy 3) | 6 hours | 25 hours |
| Phase 5: Prefetching (Strategy 2) | 8 hours | 33 hours |
| Phase 6: Evaluation | 8 hours | 41 hours |
| **Total** | **~1 week of focused work** | |

---

## Part 7: Summary

### Core Insight
The access pattern for LLM inference is **deterministic at the per-token level**, even though:
- Next token is stochastic (sampling)
- Entire sequence is non-deterministic

This determinism enables:
1. ✅ Pre-computing exact access sequence
2. ✅ Predictive prefetching
3. ✅ Intelligent hot tensor pinning
4. ✅ Compute-I/O overlap

### Key Innovations
1. **GGUF mapping**: Semantic understanding of "what" is accessed
2. **In-app tracing**: Tensor-level granularity (not just block-level)
3. **Heatmap visualization**: Identify hot tensors, layers, components
4. **Deterministic prefetching**: Leverage per-token pattern
5. **Async I/O overlap**: Hide I/O latency during computation

### Expected Contributions
1. **Characterization**: "Which tensors are re-read under memory pressure?"
2. **Optimization**: "1.5-2x speedup via software prefetching"
3. **MoE analysis**: "Non-uniform expert access enables selective caching"
4. **Tool**: "Open-source access tracing for llama.cpp"

### Next Immediate Action

```bash
# Step 1: Create GGUF mapping tool
cd /home/keri/BSC/llama.cpp
# (Create dump_gguf_offsets.cpp with code from Part 2)

# Step 2: Compile
g++ -o dump_gguf_offsets dump_gguf_offsets.cpp \
    -I./ggml/include -I./include \
    -L./build/lib -lggml -lgguf

# Step 3: Test
./dump_gguf_offsets models/llama-2-7b-chat.Q4_K_M.gguf > llama2_structure.csv
head -20 llama2_structure.csv

# Step 4: Verify output looks correct
```

**Once verified, proceed to Phase 2 (instrumentation).**

---

## Document Status
✅ **COMPREHENSIVE** - Covers all discussed ideas
✅ **ACTIONABLE** - Clear implementation steps
✅ **RESEARCH-GRADE** - Suitable for thesis documentation

**Author**: Claude + User collaborative design
**Date**: December 30, 2025
**Version**: 1.0 - Initial comprehensive design
