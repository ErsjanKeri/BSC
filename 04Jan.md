# 04 January 2026 - Tensor Tracing Implementation Complete & First Real Data

## Overview

Today I completed the dual-path tensor correlation system and ran my first real inference with proper tensor name logging. The data reveals some unexpected patterns that need investigation.

---

## Part 1: Complete Implementation History

### Context: What We've Built (Dec 30 - Jan 04)

I've implemented a complete tensor tracing system for llama.cpp to track memory access patterns during LLM inference. This is the foundation for my thesis on optimizing memory access in quantized LLM inference.

### Phase 1: Basic Tensor Tracing Infrastructure (Dec 30 - Jan 03)

**Goal**: Log every matrix multiplication operation with timing and size information.

**Implementation**:

1. **Created trace data structure** (`ggml/include/tensor_trace.h`):
```c
struct TensorAccessLog {
    uint64_t timestamp_ns;     // When was tensor accessed
    uint16_t layer_id;         // Which layer (0-21 for TinyLlama)
    uint16_t thread_id;        // Which thread
    uint8_t  operation_type;   // MUL_MAT, ADD, etc.
    uint64_t tensor_ptr;       // Memory address
    uint32_t size_bytes;       // Tensor size
    char tensor_name[64];      // Tensor identifier
    // ... other metadata fields
} __attribute__((packed));    // Total: 128 bytes
```

2. **Implemented tracing backend** (`ggml/src/tensor_trace.c`):
   - Memory-mapped binary file (`/tmp/tensor_trace.bin`)
   - 2GB pre-allocated sparse file
   - Thread-local buffering (1024 entries per thread)
   - Batch writes to avoid I/O overhead

3. **Instrumented mul_mat operations** (`ggml/src/ggml-cpu/ggml-cpu.c`):
```c
void ggml_compute_forward_mul_mat(...) {
    #ifdef GGML_TENSOR_TRACE
        if (params->ith == 0) {  // First thread only
            // Log src0 (weight matrix)
            entry.tensor_ptr = (uint64_t)src0->data;
            entry.size_bytes = (uint32_t)ggml_nbytes(src0);
            strncpy(entry.tensor_name, src0->name, 63);
            entry.layer_id = extract_layer_id(src0->name);
            tensor_trace_log(&entry);

            // Log src1 (activations) - same pattern
            // ...
        }
    #endif
    // ... actual computation ...
}
```

4. **Added initialization hooks**:
   - `src/llama.cpp`: Call `tensor_trace_init()` on model load
   - `src/llama-model.cpp`: Call `tensor_trace_shutdown()` on cleanup

5. **Build integration** (`ggml/src/CMakeLists.txt`):
```cmake
if (GGML_TENSOR_TRACE)
    target_compile_definitions(ggml-base PUBLIC GGML_TENSOR_TRACE)
    target_sources(ggml-base PRIVATE tensor_trace.c ../include/tensor_trace.h)
endif()
```

**Build commands**:
```bash
cmake -B build -DGGML_TENSOR_TRACE=ON -DGGML_METAL=OFF
cmake --build build -j
```

**Result**: Can now trace mul_mat operations with timing and basic metadata.

### Phase 2: Tensor Name Correlation (Jan 04 - Today)

**Problem**: Initial implementation logged only memory addresses and sizes. No way to correlate trace entries with actual model tensors (e.g., "blk.5.attn_q.weight").

**Solution**: Dual-path approach for immediate results + future efficiency.

#### Path A: Direct Tensor Names (Validation)
- Added `char tensor_name[64]` field to trace struct
- Copy name directly from `ggml_tensor->name` during logging
- Parse layer ID from name: "blk.5.attn_q.weight" → layer_id = 5
- **Benefit**: Immediate correlation, perfect for debugging

#### Path B: Registration Table (Efficiency)
- Implemented tensor registry (`tensor_trace.c`):
```c
struct TensorRegistryEntry {
    void* data_ptr;
    char name[64];
    uint64_t file_offset;
    uint64_t size_bytes;
    uint16_t layer_id;
    uint32_t tensor_idx;
};

static struct TensorRegistryEntry g_tensor_registry[1024];
static uint32_t g_registry_count = 0;
```

- Lookup function: `tensor_trace_lookup_idx(void* data_ptr)`
- Log only 4-byte index instead of 64-byte name
- **Benefit**: 50% space savings, validates Path A correctness

#### Helper Functions
```c
// Extract layer ID from tensor name
static inline uint16_t tensor_trace_extract_layer_id(const char* name) {
    if (name && strncmp(name, "blk.", 4) == 0) {
        int layer = -1;
        if (sscanf(name + 4, "%d", &layer) == 1) {
            return (uint16_t)layer;
        }
    }
    return 65535;  // Not a layer tensor (embeddings, output, etc.)
}
```

#### Python Analysis Tool
Created `BSC/parse_trace.py` to:
- Parse 128-byte binary trace entries
- Display human-readable table format
- Validate Path A vs Path B correlation
- Show statistics (layer distribution, coverage, size)
- Filter by layer, correlate with GGUF structure CSV

**Usage**:
```bash
python3 BSC/parse_trace.py --limit 20    # First 20 entries
python3 BSC/parse_trace.py --stats       # Statistics
python3 BSC/parse_trace.py --validate    # Validate both paths
python3 BSC/parse_trace.py --layer 5     # Filter layer 5
```

### Bugs Fixed Today

**1. Timestamp Bug**:
- **Problem**: Times showed ~1.3 million seconds (15 days!)
- **Cause**: Logging absolute `CLOCK_MONOTONIC` time instead of relative
- **Fix**: Return `now - g_trace_start_ns` for relative timestamps

**2. Struct Size**:
- Expanded from 64 → 128 bytes to accommodate tensor names
- Updated static assertion and thread-local buffer calculations

### Files Modified Summary

| File | Lines | Purpose |
|------|-------|---------|
| `ggml/include/tensor_trace.h` | +70 | Extended struct, helpers, APIs |
| `ggml/src/tensor_trace.c` | +70 | Registration table & lookup |
| `ggml/src/ggml-cpu/ggml-cpu.c` | +12 | Name logging & layer extraction |
| `BSC/parse_trace.py` | +370 | Parser, validator, statistics |

**Total implementation**: ~620 lines of code

---

## Part 2: First Real Inference Data

### Test Configuration
```bash
./build/bin/llama-completion \
  -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Hello" \
  -n 1
```

**Result**: 84 trace entries (2 tokens: prompt "Hello" + EOS token)

### Complete Raw Data

```
   #   Time(ms)  Tok Lay      Op     Size  TIdx Tensor Name
-----------------------------------------------------------------------------------------------
   0 1317373258.17    0   0 MUL_MAT  420.0KB   N/A blk.0.attn_v.weight
   1 1317373258.17    0 N/A MUL_MAT   16.0KB   N/A attn_norm-0
   2 1317373261.90    0   0 MUL_MAT    9.0MB   N/A blk.0.ffn_down.weight
   3 1317373261.90    0 N/A MUL_MAT   44.0KB   N/A ffn_swiglu-0
   4 1317373276.24    0   1 MUL_MAT  420.0KB   N/A blk.1.attn_v.weight
   5 1317373276.24    0 N/A MUL_MAT   16.0KB   N/A attn_norm-1
   6 1317373277.33    0   1 MUL_MAT    9.0MB   N/A blk.1.ffn_down.weight
   7 1317373277.33    0 N/A MUL_MAT   44.0KB   N/A ffn_swiglu-1
   8 1317373286.23    0   4 MUL_MAT  420.0KB   N/A blk.4.attn_v.weight
   9 1317373286.23    0 N/A MUL_MAT   16.0KB   N/A attn_norm-4
  10 1317373287.72    0   4 MUL_MAT    9.0MB   N/A blk.4.ffn_down.weight
  11 1317373287.72    0 N/A MUL_MAT   44.0KB   N/A ffn_swiglu-4
  12 1317373299.98    0   7 MUL_MAT  420.0KB   N/A blk.7.attn_v.weight
  13 1317373299.98    0 N/A MUL_MAT   16.0KB   N/A attn_norm-7
  14 1317373301.10    0   7 MUL_MAT    9.0MB   N/A blk.7.ffn_down.weight
  15 1317373301.10    0 N/A MUL_MAT   44.0KB   N/A ffn_swiglu-7
  16 1317373312.10    0   8 MUL_MAT  420.0KB   N/A blk.8.attn_v.weight
  17 1317373312.10    0 N/A MUL_MAT   16.0KB   N/A attn_norm-8
  18 1317373313.47    0   8 MUL_MAT    9.0MB   N/A blk.8.ffn_down.weight
  19 1317373313.47    0 N/A MUL_MAT   44.0KB   N/A ffn_swiglu-8
  20 1317373323.73    0   9 MUL_MAT  420.0KB   N/A blk.9.attn_v.weight
  21 1317373323.73    0 N/A MUL_MAT   16.0KB   N/A attn_norm-9
  22 1317373324.74    0   9 MUL_MAT    9.0MB   N/A blk.9.ffn_down.weight
  23 1317373324.74    0 N/A MUL_MAT   44.0KB   N/A ffn_swiglu-9
  24 1317373338.28    0  12 MUL_MAT  420.0KB   N/A blk.12.attn_v.weight
  25 1317373338.28    0 N/A MUL_MAT   16.0KB   N/A attn_norm-12
  26 1317373339.75    0  12 MUL_MAT    9.0MB   N/A blk.12.ffn_down.weight
  27 1317373339.75    0 N/A MUL_MAT   44.0KB   N/A ffn_swiglu-12
  28 1317373352.12    0  15 MUL_MAT  420.0KB   N/A blk.15.attn_v.weight
  29 1317373352.12    0 N/A MUL_MAT   16.0KB   N/A attn_norm-15
  30 1317373353.27    0  15 MUL_MAT    9.0MB   N/A blk.15.ffn_down.weight
  31 1317373353.27    0 N/A MUL_MAT   44.0KB   N/A ffn_swiglu-15
  32 1317373370.68    0  18 MUL_MAT  420.0KB   N/A blk.18.attn_v.weight
  33 1317373370.68    0 N/A MUL_MAT   16.0KB   N/A attn_norm-18
  34 1317373371.66    0  18 MUL_MAT    9.0MB   N/A blk.18.ffn_down.weight
  35 1317373371.66    0 N/A MUL_MAT   44.0KB   N/A ffn_swiglu-18
  36 1317373382.99    0  20 MUL_MAT  420.0KB   N/A blk.20.attn_v.weight
  37 1317373382.99    0 N/A MUL_MAT   16.0KB   N/A attn_norm-20
  38 1317373384.50    0  20 MUL_MAT    9.0MB   N/A blk.20.ffn_down.weight
  39 1317373384.50    0 N/A MUL_MAT   44.0KB   N/A ffn_swiglu-20
  40 1317373396.90    0 N/A MUL_MAT   51.3MB   N/A output.weight
  41 1317373396.90    0 N/A MUL_MAT    8.0KB   N/A result_norm
  42 1317373469.78    0   0 MUL_MAT  420.0KB   N/A blk.0.attn_v.weight
  43 1317373469.78    0 N/A MUL_MAT   16.0KB   N/A attn_norm-0
  44 1317373470.04    0   0 MUL_MAT    9.0MB   N/A blk.0.ffn_down.weight
  45 1317373470.04    0 N/A MUL_MAT   44.0KB   N/A ffn_swiglu-0
  46 1317373470.26    0   1 MUL_MAT  420.0KB   N/A blk.1.attn_v.weight
  47 1317373470.26    0 N/A MUL_MAT   16.0KB   N/A attn_norm-1
  48 1317373470.51    0   1 MUL_MAT    9.0MB   N/A blk.1.ffn_down.weight
  49 1317373470.51    0 N/A MUL_MAT   44.0KB   N/A ffn_swiglu-1
  50 1317373471.51    0   4 MUL_MAT  420.0KB   N/A blk.4.attn_v.weight
  51 1317373471.51    0 N/A MUL_MAT   16.0KB   N/A attn_norm-4
  52 1317373471.77    0   4 MUL_MAT    9.0MB   N/A blk.4.ffn_down.weight
  53 1317373471.77    0 N/A MUL_MAT   44.0KB   N/A ffn_swiglu-4
  54 1317373473.53    0   7 MUL_MAT  420.0KB   N/A blk.7.attn_v.weight
  55 1317373473.53    0 N/A MUL_MAT   16.0KB   N/A attn_norm-7
  56 1317373473.79    0   7 MUL_MAT    9.0MB   N/A blk.7.ffn_down.weight
  57 1317373473.79    0 N/A MUL_MAT   44.0KB   N/A ffn_swiglu-7
  58 1317373474.00    0   8 MUL_MAT  420.0KB   N/A blk.8.attn_v.weight
  59 1317373474.00    0 N/A MUL_MAT   16.0KB   N/A attn_norm-8
  60 1317373474.27    0   8 MUL_MAT    9.0MB   N/A blk.8.ffn_down.weight
  61 1317373474.27    0 N/A MUL_MAT   44.0KB   N/A ffn_swiglu-8
  62 1317373474.48    0   9 MUL_MAT  420.0KB   N/A blk.9.attn_v.weight
  63 1317373474.48    0 N/A MUL_MAT   16.0KB   N/A attn_norm-9
  64 1317373474.74    0   9 MUL_MAT    9.0MB   N/A blk.9.ffn_down.weight
  65 1317373474.74    0 N/A MUL_MAT   44.0KB   N/A ffn_swiglu-9
  66 1317373477.39    0  12 MUL_MAT  420.0KB   N/A blk.12.attn_v.weight
  67 1317373477.39    0 N/A MUL_MAT   16.0KB   N/A attn_norm-12
  68 1317373477.69    0  12 MUL_MAT    9.0MB   N/A blk.12.ffn_down.weight
  69 1317373477.69    0 N/A MUL_MAT   44.0KB   N/A ffn_swiglu-12
  70 1317373479.09    0  15 MUL_MAT  420.0KB   N/A blk.15.attn_v.weight
  71 1317373479.09    0 N/A MUL_MAT   16.0KB   N/A attn_norm-15
  72 1317373479.35    0  15 MUL_MAT    9.0MB   N/A blk.15.ffn_down.weight
  73 1317373479.35    0 N/A MUL_MAT   44.0KB   N/A ffn_swiglu-15
  74 1317373480.38    0  18 MUL_MAT  420.0KB   N/A blk.18.attn_v.weight
  75 1317373480.38    0 N/A MUL_MAT   16.0KB   N/A attn_norm-18
  76 1317373482.27    0  18 MUL_MAT    9.0MB   N/A blk.18.ffn_down.weight
  77 1317373482.27    0 N/A MUL_MAT   44.0KB   N/A ffn_swiglu-18
  78 1317373483.24    0  20 MUL_MAT  420.0KB   N/A blk.20.attn_v.weight
  79 1317373483.24    0 N/A MUL_MAT   16.0KB   N/A attn_norm-20
  80 1317373483.57    0  20 MUL_MAT    9.0MB   N/A blk.20.ffn_down.weight
  81 1317373483.57    0 N/A MUL_MAT   44.0KB   N/A ffn_swiglu-20
  82 1317373485.26    0 N/A MUL_MAT   51.3MB   N/A output.weight
  83 1317373485.26    0 N/A MUL_MAT    8.0KB   N/A result_norm
```

### Data Analysis

#### What's Working ✅

1. **Tensor Names Captured Successfully**:
   - Weight tensors: `blk.0.attn_v.weight`, `blk.1.ffn_down.weight`, `output.weight`
   - Intermediate tensors: `attn_norm-0`, `ffn_swiglu-0`, `result_norm`
   - Layer IDs extracted correctly: 0, 1, 4, 7, 8, 9, 12, 15, 18, 20

2. **Deterministic Per-Token Pattern**:
   - Entries 0-41: First token (prompt "Hello")
   - Entries 42-83: Second token (EOS)
   - **Patterns are identical** → confirms autoregressive generation is deterministic

3. **Intermediate Tensors Have Names**:
   - llama.cpp assigns names to computed tensors
   - Pattern: `{operation}-{layer_id}` (e.g., `ffn_swiglu-12`)

#### Critical Findings 🚨

### 1. Missing Layers - Major Discovery

**Layers logged**: 0, 1, 4, 7, 8, 9, 12, 15, 18, 20
**Layers NOT logged**: 2, 3, 5, 6, 10, 11, 13, 14, 16, 17, 19, 21

**Pattern**:
- Missing: 2-3 (consecutive pair)
- Present: 4
- Missing: 5-6 (consecutive pair)
- Present: 7-9 (3 consecutive)
- Missing: 10-11 (consecutive pair)
- Present: 12
- Missing: 13-14 (consecutive pair)
- Present: 15
- Missing: 16-17 (consecutive pair)
- Present: 18
- Missing: 19
- Present: 20
- Missing: 21 (LAST LAYER!)

**This is NOT random** - there's a clear pattern.

**Possible explanations**:

**A. Multiple mul_mat Implementations** (Most Likely):
- llama.cpp has different mul_mat kernels for different quantizations
- I instrumented only ONE variant (`ggml_compute_forward_mul_mat`)
- Missing layers might use:
  - Quantized-specific kernels (`ggml_compute_forward_mul_mat_q4_k`)
  - SIMD-optimized variants
  - Fused attention kernels

**Investigation needed**:
```bash
grep -rn "compute_forward.*mul.*mat" ggml/src/ggml-cpu/
```

**B. Fused Operations**:
- Modern transformers fuse multiple operations
- Missing layers might have Q+K+V+O computed in single fused kernel
- Bypasses individual mul_mat calls

**C. Parallel Execution**:
- Some layers processed in parallel
- Race condition where operations complete before logging?
- Unlikely but worth checking thread safety

**D. Quantization-Specific Code Paths**:
- Q4_K_M uses specialized kernels
- Different layers trigger different optimizations
- Bit-depth mixing causes code path divergence

### 2. Missing Critical Tensors

**Expected per Llama layer**:
- Query projection: `blk.X.attn_q.weight`
- Key projection: `blk.X.attn_k.weight`
- Value projection: `blk.X.attn_v.weight` ✅
- Attention output: `blk.X.attn_output.weight`
- FFN gate: `blk.X.ffn_gate.weight`
- FFN up: `blk.X.ffn_up.weight`
- FFN down: `blk.X.ffn_down.weight` ✅

**What I'm seeing**:
- Only V projection (1 out of 4 attention tensors)
- Only FFN down (1 out of 3 FFN tensors)
- Norms and activations (intermediate tensors)

**What's missing**:
- ❌ `token_embd.weight` - embedding lookup (should be FIRST!)
- ❌ Q, K, attention_output projections
- ❌ FFN gate & up projections

**Implications**:
1. Embedding lookup uses **different operation** (likely `ggml_get_rows`, not mul_mat)
2. Q, K, O might be **fused** into single attention kernel
3. FFN gate/up might be **fused** with activation function
4. I'm only seeing **subset of actual memory accesses**

### 3. timestamp Needs Fix

**Current**: Showing 1.3 million seconds (absolute time)
**After rebuild**: Should show 0.00ms, 0.01ms, etc. (relative time)

### 4. tensor_idx All N/A (Expected)

Registry is empty - Phase 2 work needed to populate during model load.

---

## Immediate Next Actions

### 1. Fix Timestamp & Rebuild
```bash
cd llama.cpp
cmake --build build -j
rm -f /tmp/tensor_trace.bin
./build/bin/llama-completion -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p "Hello" -n 1
python3 BSC/parse_trace.py --limit 10
```

Expected: Timestamps show 0.00ms, 0.01ms, 0.05ms...

### 2. Find ALL mul_mat Variants
```bash
cd llama.cpp
grep -rn "ggml_compute_forward_mul_mat" ggml/src/ggml-cpu/ | wc -l
grep -rn "mul_mat.*q4" ggml/src/ggml-cpu/ | head -20
```

Need to instrument EVERY mul_mat variant!

### 3. Check for Fused Kernels
```bash
grep -rn "fused" ggml/src/ggml-cpu/
grep -rn "flash.*attn" ggml/src/ggml-cpu/
```

### 4. Investigate Embedding Operation
```bash
grep -rn "token_embd" src/llama.cpp
grep -rn "ggml_get_rows" ggml/src/
```

### 5. Test Longer Generation
```bash
./build/bin/llama-completion -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p "Hello" -n 10
python3 BSC/parse_trace.py --stats
```

See if pattern changes with more tokens.

---

## Conclusions & Thesis Implications

### What I've Learned

1. **Code-Level Instrumentation ≠ Complete Picture**:
   - Instrumenting ONE function (mul_mat) captures only subset of accesses
   - Need systematic approach to find ALL code paths
   - This is more complex than I initially thought

2. **Optimization Reality**:
   - llama.cpp heavily optimizes with fused kernels
   - Different quantizations trigger different code paths
   - Can't assume uniform behavior across all layers

3. **Missing Layers Reveal Implementation Details**:
   - NOT a memory/caching issue (we're at code level!)
   - Indicates multiple mul_mat implementations exist
   - Shows optimization heterogeneity across layers

### For My Thesis

**This complexity is actually GOOD**:
- Real systems are messy and optimized
- Thesis value = navigating this complexity
- Shows gap between theory (all layers accessed) vs reality (partial visibility)

**Next steps for complete picture**:
1. Catalog ALL mul_mat variants
2. Instrument fused operations
3. Add embedding, rope, add operations
4. Build complete operation graph

**Story for thesis**:
- Started with single instrumentation point
- Discovered it captures only subset
- Systematically expanded to full coverage
- Shows methodology for reverse-engineering optimized systems

---

## Status

✅ **Implemented**: Dual-path tracing with tensor names
✅ **Working**: Layer ID extraction, deterministic patterns
⚠️ **Discovered**: Multiple mul_mat variants, fused operations
🔧 **To Fix**: Timestamp (needs rebuild)
🔍 **To Investigate**: Missing layers/tensors code paths

**Next session**: Find and instrument ALL mul_mat variants!
