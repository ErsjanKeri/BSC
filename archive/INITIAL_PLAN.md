# INITIAL IMPLEMENTATION PLAN - Tensor Access Tracker

**Date**: January 2, 2026
**Status**: Planning - READY TO IMPLEMENT ✅
**Author**: Ersjan Këri
**Last Updated**: January 2, 2026 (Evening - Final Architecture)

---

## CRITICAL ARCHITECTURE DECISION (Updated)

**FINAL APPROACH**: Instrument `ggml-cpu.c` (CPU backend) to capture REAL memory access

**Key Changes from Initial Plan**:
- ✅ **Location**: `ggml/src/ggml-cpu/ggml-cpu.c` (NOT `ggml.c` dispatcher)
- ✅ **Scope**: ALL operations including intermediate tensors, KV cache, embeddings
- ✅ **Backend**: CPU-only (sufficient for thesis, upstreamable)
- ✅ **Thread Safety**: Sequential token processing (simplified)
- ✅ **Environment**: Linux server only (`cli-hiwi-02.dis.cit.tum.de`)

**Why This Approach**:
- Captures **physical** memory access, not just logical operation dispatch
- Enables correlation with blktrace (page faults, disk I/O)
- More detailed than dispatcher-level hooks
- Standard pattern in llama.cpp (`#ifdef GGML_TENSOR_TRACE`)

See [02JAN.md](./journal/02JAN.md) Decision 5 for full rationale.

---

## PHASE 1: Foundation (Days 1-2)

### Step 1: Create Logging Infrastructure (4 hours)

**Location**: `/Users/ersibesi/Desktop/LLAMA/llama.cpp`

**Files to Create**:
```bash
ggml/include/tensor_trace.h
ggml/src/tensor_trace.c
```

**Tasks**:
- [ ] Define `TensorAccessLog` struct (64 bytes):
  ```c
  struct TensorAccessLog {
      uint64_t timestamp_ns;
      uint32_t token_id;
      uint16_t layer_id;
      uint16_t thread_id;
      uint8_t  operation_type;
      uint8_t  phase;
      uint16_t padding1;
      uint32_t padding1b;
      uint32_t tensor_idx;
      uint64_t tensor_ptr;
      uint64_t file_offset;
      uint32_t size_bytes;
      uint8_t  attention_head;
      uint8_t  qkv_type;
      uint16_t padding2;
      uint8_t  expert_id;
      uint8_t  expert_rank;
      uint16_t routing_score;
      uint32_t padding3;
  };
  ```

- [ ] Implement `tensor_trace_init(const char* log_path, size_t capacity_bytes)`:
  - Open/create file at `/dev/shm/tensor_trace.bin`
  - `ftruncate()` to 2GB
  - `mmap()` with `MAP_SHARED | PROT_WRITE`
  - Store buffer pointer and capacity globally

- [ ] Implement `tensor_trace_log(const TensorAccessLog* entry)`:
  - Check capacity (offset + 64 <= capacity)
  - `memcpy()` entry to buffer[offset]
  - Increment offset by 64
  - NO error handling (fast path)

- [ ] Implement `tensor_trace_shutdown()`:
  - `msync(MS_SYNC)` to force flush
  - `munmap()` buffer
  - Print statistics (total entries, MB written)

- [ ] Implement `tensor_trace_register_tensor()`:
  - Build global lookup table: `tensor_ptr → {name, file_offset, size}`
  - Use hash map or linear array (max ~1000 tensors)

- [ ] Add `#ifdef TENSOR_TRACE_ENABLED` guards everywhere

**Deliverable**: Standalone tensor_trace library (compiles, unit testable)

---

### Step 2: Build GGUF Structure Dumper (3 hours)

**Location**: `/Users/ersibesi/Desktop/LLAMA/BSC/tensor-tracker/tools`

**File to Create**:
```bash
gguf_offset_dump.cpp
```

**Tasks**:
- [ ] Include llama.cpp headers:
  ```cpp
  #include "ggml.h"
  #include "gguf.h"
  ```

- [ ] Load GGUF file:
  ```cpp
  struct gguf_init_params params = {.no_alloc = true, .ctx = NULL};
  struct gguf_context* ctx = gguf_init_from_file(argv[1], params);
  ```

- [ ] Extract metadata:
  - `gguf_get_n_tensors(ctx)`
  - `gguf_get_data_offset(ctx)` (start of tensor data section)
  - For each tensor: name, offset, size, type

- [ ] Parse layer_id from name:
  ```cpp
  int extract_layer_id(const char* name) {
      if (strncmp(name, "blk.", 4) == 0) {
          int layer;
          sscanf(name + 4, "%d", &layer);
          return layer;
      }
      return -1;  // Not a layer tensor
  }
  ```

- [ ] Determine component type:
  - "attn_q" → "Attention Q"
  - "attn_k" → "Attention K"
  - "attn_v" → "Attention V"
  - "attn_output" → "Attention Output"
  - "ffn_up" → "FFN Up"
  - "ffn_down" → "FFN Down"
  - "ffn_gate" → "FFN Gate"
  - "attn_norm" → "Attention Norm"
  - "ffn_norm" → "FFN Norm"
  - "token_embd" → "Token Embeddings"
  - "output" → "Output Projection"

- [ ] Output CSV:
  ```
  tensor_name,file_offset,size_bytes,layer_id,component_type
  token_embd.weight,524288,524288000,-1,Token Embeddings
  blk.0.attn_q.weight,524812288,67108864,0,Attention Q
  ...
  ```

**Compile**:
```bash
g++ -o gguf_offset_dump gguf_offset_dump.cpp \
    -I../../llama.cpp/ggml/include \
    -I../../llama.cpp/include \
    -L../../llama.cpp/build/lib \
    -lggml -lgguf
```

**Test**:
```bash
./gguf_offset_dump ~/models/llama-2-7b-chat.Q4_K_M.gguf > llama2_structure.csv
head -20 llama2_structure.csv
```

**Deliverable**: `gguf_structure.csv` for test models

---

### Step 3: Add Minimal Instrumentation (4 hours)

#### 3.1 Modify ggml-cpu.c (CPU Backend)

**File**: `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c`

**Tasks**:
- [ ] Add include at top:
  ```c
  #ifdef GGML_TENSOR_TRACE
  #include "tensor_trace.h"
  #endif
  ```

- [ ] Find `ggml_compute_forward_mul_mat_f32()` function (example operation)

- [ ] Add hook BEFORE memory access:
  ```c
  static void ggml_compute_forward_mul_mat_f32(
      const struct ggml_compute_params* params,
      struct ggml_tensor* dst) {

      const struct ggml_tensor* src0 = dst->src[0];
      const struct ggml_tensor* src1 = dst->src[1];

      #ifdef GGML_TENSOR_TRACE
          // Log BEFORE accessing memory (captures REAL access)
          tensor_trace_log_access(dst, src0, src1, GGML_OP_MUL_MAT);
      #endif

      // Actual computation - THIS is where memory is touched
      const float* wdata = (float*)src0->data;  // ← REAL ACCESS
      const float* xdata = (float*)src1->data;  // ← REAL ACCESS
      float* ddata = (float*)dst->data;

      // ... existing matrix multiply code ...
  }
  ```

**Note**: Start with ONE operation (MUL_MAT) for MVP, expand to all operations in Phase 2.

#### 3.2 Modify llama.cpp for Token Tracking

**File**: `llama.cpp/src/llama.cpp`

**Tasks**:
- [ ] Find `llama_decode()` function

- [ ] Add token counter (SIMPLIFIED - no atomics needed):
  ```c
  int llama_decode(struct llama_context* ctx, struct llama_batch batch) {
      #ifdef GGML_TENSOR_TRACE
          extern uint32_t g_current_token;
          extern uint8_t g_current_phase;

          // Increment BEFORE parallel execution (single-threaded here)
          // No atomics needed - tokens processed sequentially!
          if (batch.n_tokens > 0) {
              if (g_current_token == 0) {
                  g_current_phase = PHASE_PROMPT;
              } else {
                  g_current_phase = PHASE_GENERATE;
              }
              g_current_token++;
          }
      #endif

      // Now multiple threads execute graph (all read same g_current_token)
      // ... existing decode logic ...

      return result;
  }
  ```

- [ ] Find `llama_model_load_internal()` or similar

- [ ] Add tensor registration during model load:
  ```c
  // During model loading, for each tensor
  for (int i = 0; i < n_tensors; i++) {
      const char* name = gguf_get_tensor_name(ctx_gguf, i);
      size_t offset = gguf_get_tensor_offset(ctx_gguf, i);

      // ... existing loading code ...

      #ifdef GGML_TENSOR_TRACE
          // Build mapping: tensor_ptr → {name, file_offset, size}
          // This enables post-processing correlation
          tensor_trace_register_tensor(
              tensor->name,
              tensor->data,           // Pointer in mmap'd region
              offset,                 // Offset in GGUF file
              ggml_nbytes(tensor)
          );
      #endif
  }
  ```

#### 3.3 Modify CMakeLists.txt

**File**: `llama.cpp/ggml/CMakeLists.txt` (or main CMakeLists.txt)

**Tasks**:
- [ ] Add option:
  ```cmake
  option(GGML_TENSOR_TRACE "Enable tensor access tracing" OFF)
  ```

- [ ] Add compile definition:
  ```cmake
  if(GGML_TENSOR_TRACE)
      add_definitions(-DGGML_TENSOR_TRACE)
      message(STATUS "Tensor tracing ENABLED - all operations will be logged")
  endif()
  ```

- [ ] Add tensor_trace.c to sources:
  ```cmake
  set(GGML_SOURCES
      ggml/src/ggml.c
      ggml/src/ggml-alloc.c
      ggml/src/ggml-backend.c
      ggml/src/tensor_trace.c  # ← ADD THIS
      # ... other sources
  )
  ```

#### 3.4 Build Instrumented llama.cpp

```bash
cd /Users/ersibesi/Desktop/LLAMA/llama.cpp
mkdir -p build-trace
cd build-trace

cmake .. -DGGML_TENSOR_TRACE=ON -DCMAKE_BUILD_TYPE=Release
make -j8
```

**Expected Output**:
```
-- Tensor tracing ENABLED - all operations will be logged
...
[100%] Built target llama-cli
```

**Verify tracing is enabled**:
```bash
strings ./bin/llama-cli | grep tensor_trace
# Should see symbol names if tracing is compiled in
```

**Deliverable**: Instrumented llama-cli binary

---

### Step 4: First Test Run (2 hours)

#### 4.1 Run Inference

```bash
cd /Users/ersibesi/Desktop/LLAMA/llama.cpp/build-trace/bin

./llama-cli -m ~/models/llama-2-7b-chat.Q4_K_M.gguf \
            -p "Hello" \
            -n 10 \
            --no-cnv
```

**Expected**:
- Inference completes normally
- `/dev/shm/tensor_trace.bin` created
- File size: ~few KB (hundreds of entries for 10 tokens)

#### 4.2 Verify Log File

```bash
ls -lh /dev/shm/tensor_trace.bin
# Expected: -rw-r--r-- 1 user user 2.0G Jan  2 12:00 tensor_trace.bin
# (2GB allocated, only few KB used)

# Check actual used size
du -h /dev/shm/tensor_trace.bin
# Expected: 4.0K or similar (actual data written)
```

#### 4.3 Copy to Permanent Storage

```bash
mkdir -p ~/BSC/traces/test_001
cp /dev/shm/tensor_trace.bin ~/BSC/traces/test_001/trace.bin
rm /dev/shm/tensor_trace.bin
```

**Deliverable**: First binary trace log

---

### Step 5: Parse & Verify (3 hours)

#### 5.1 Create Parser

**File**: `BSC/tensor-tracker/tools/parse_trace.py`

```python
#!/usr/bin/env python3
import struct
import pandas as pd
from pathlib import Path

# 64-byte log entry format
LOG_ENTRY_FMT = (
    'Q'   # timestamp_ns (8 bytes)
    'I'   # token_id (4 bytes)
    'H'   # layer_id (2 bytes)
    'H'   # thread_id (2 bytes)
    'B'   # operation_type (1 byte)
    'B'   # phase (1 byte)
    'H'   # padding1 (2 bytes)
    'I'   # padding1b (4 bytes)
    'I'   # tensor_idx (4 bytes)
    'Q'   # tensor_ptr (8 bytes)
    'Q'   # file_offset (8 bytes)
    'I'   # size_bytes (4 bytes)
    'B'   # attention_head (1 byte)
    'B'   # qkv_type (1 byte)
    'H'   # padding2 (2 bytes)
    'B'   # expert_id (1 byte)
    'B'   # expert_rank (1 byte)
    'H'   # routing_score (2 bytes)
    'I'   # padding3 (4 bytes)
)

ENTRY_SIZE = struct.calcsize(LOG_ENTRY_FMT)
assert ENTRY_SIZE == 64, f"Entry size is {ENTRY_SIZE}, expected 64"

def parse_trace(trace_file):
    """Parse binary trace log to DataFrame"""

    data = Path(trace_file).read_bytes()
    num_entries = len(data) // ENTRY_SIZE

    print(f"File size: {len(data)} bytes")
    print(f"Entries: {num_entries}")

    entries = []
    for i in range(num_entries):
        offset = i * ENTRY_SIZE
        entry_data = struct.unpack(LOG_ENTRY_FMT, data[offset:offset+ENTRY_SIZE])

        entry = {
            'timestamp_ns': entry_data[0],
            'token_id': entry_data[1],
            'layer_id': entry_data[2],
            'thread_id': entry_data[3],
            'operation_type': entry_data[4],
            'phase': entry_data[5],
            'tensor_idx': entry_data[8],
            'tensor_ptr': entry_data[9],
            'file_offset': entry_data[10],
            'size_bytes': entry_data[11],
            'attention_head': entry_data[12],
            'qkv_type': entry_data[13],
            'expert_id': entry_data[15],
            'expert_rank': entry_data[16],
            'routing_score': entry_data[17],
        }
        entries.append(entry)

    df = pd.DataFrame(entries)

    # Convert timestamp to seconds
    df['timestamp_s'] = (df['timestamp_ns'] - df['timestamp_ns'].min()) / 1e9

    return df

def print_summary(df):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("TRACE SUMMARY")
    print("="*60)

    print(f"\nTotal entries: {len(df):,}")
    print(f"Tokens processed: {df['token_id'].max() + 1}")
    print(f"Layers observed: {sorted(df['layer_id'].unique())}")
    print(f"Threads used: {sorted(df['thread_id'].unique())}")

    print(f"\nTime range: {df['timestamp_s'].min():.3f}s - {df['timestamp_s'].max():.3f}s")
    print(f"Duration: {df['timestamp_s'].max() - df['timestamp_s'].min():.3f}s")

    print(f"\nOperation types:")
    print(df['operation_type'].value_counts())

    print(f"\nPhases:")
    print(df['phase'].value_counts())

    print(f"\nFile offsets:")
    print(f"  Min: {df['file_offset'].min():,} bytes")
    print(f"  Max: {df['file_offset'].max():,} bytes")
    print(f"  Range: {(df['file_offset'].max() - df['file_offset'].min()) / 1e9:.2f} GB")

    print(f"\nTotal data accessed: {df['size_bytes'].sum() / 1e9:.2f} GB")
    print(f"Unique file offsets: {df['file_offset'].nunique():,}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 parse_trace.py <trace.bin>")
        sys.exit(1)

    trace_file = sys.argv[1]
    df = parse_trace(trace_file)
    print_summary(df)

    # Save as parquet for further analysis
    output_file = Path(trace_file).with_suffix('.parquet')
    df.to_parquet(output_file, compression='zstd')
    print(f"\nSaved to: {output_file}")
```

#### 5.2 Run Parser

```bash
cd /Users/ersibesi/Desktop/LLAMA/BSC/tensor-tracker/tools
chmod +x parse_trace.py

python3 parse_trace.py ~/BSC/traces/test_001/trace.bin
```

**Expected Output**:
```
File size: 4096 bytes
Entries: 64

============================================================
TRACE SUMMARY
============================================================

Total entries: 64
Tokens processed: 10
Layers observed: [0, 1, 2, ..., 31]
Threads used: [1, 2, 3, 4]

Time range: 0.000s - 2.345s
Duration: 2.345s

Operation types:
21    320  # MUL_MAT (most common)
5     64   # ADD
...

Phases:
0    10   # PROMPT
1    54   # GENERATE

File offsets:
  Min: 524,288 bytes
  Max: 6,442,450,944 bytes
  Range: 6.00 GB

Total data accessed: 12.85 GB
Unique file offsets: 291

Saved to: ~/BSC/traces/test_001/trace.parquet
```

#### 5.3 Validate Correctness

**Checks**:
- [ ] Token count correct (10)
- [ ] Layers present (0-31 for llama-2)
- [ ] Timestamps monotonic increasing
- [ ] File offsets within GGUF size
- [ ] No entries with all zeros (corrupted)

**If validation fails**: Debug instrumentation, check for:
- Buffer overflow
- Incorrect struct packing
- Missing initialization
- Thread safety issues

**Deliverable**: Verified, parsed trace data

---

## PHASE 1 SUCCESS CRITERIA

### Must Pass:
- ✅ Binary log created in `/dev/shm/`
- ✅ Parser reads it without errors
- ✅ Token count matches expected (10)
- ✅ All layers represented
- ✅ Timestamps are monotonic
- ✅ File offsets are valid
- ✅ No corruption (no zero entries)
- ✅ Inference overhead <10%

### Performance Check:
```bash
# Baseline (no tracing)
time ./llama-cli-baseline -m model.gguf -p "Hello" -n 10

# With tracing
time ./llama-cli-trace -m model.gguf -p "Hello" -n 10

# Overhead = (trace_time - baseline_time) / baseline_time * 100%
# Target: <10%
```

### If Overhead >10%:
- Profile with `perf` to find hotspot
- Check if logging is called too frequently
- Ensure mmap writes are not blocking
- Consider reducing logging granularity

---

## NEXT STEPS (Phase 2)

**If Phase 1 succeeds**, proceed to:
- Step 6: Instrument all operations (not just MUL_MAT)
- Step 7: Add semantic parsing (layer_id, qkv_type, expert_id)
- Step 8: File offset mapping during model load
- Step 9: Thread ID tracking
- Step 10: Full 100-token test run

**If Phase 1 fails**, debug and iterate before proceeding.

---

## OPEN QUESTIONS - ALL RESOLVED ✅

### ~~1. Backend Dependency~~ → RESOLVED
**Decision**: CPU-only (ggml-cpu.c) is sufficient for thesis
- Captures real memory access patterns
- Upstreamable (GPU backends can be added later)
- See [02JAN.md Decision 5](./journal/02JAN.md)

### ~~2. Step Validation~~ → RESOLVED
**Struct layout**: Use `__attribute__((packed))` or static asserts
**tmpfs**: Linux server only (`/dev/shm/` guaranteed)
**Tensor registration**: Build during model load, use during inference
**Thread safety**: Sequential token processing (no atomics needed)

### ~~3. Upstream PR Potential~~ → RESOLVED
**Answer**: YES, with proper documentation and minimal invasiveness
- Similar to existing `GGML_PERF` and `GGML_DEBUG` flags
- Useful for performance profiling, memory optimization, MoE research
- See upstreaming strategy in [02JAN.md](./journal/02JAN.md)

### ~~4. Architecture Metadata~~ → RESOLVED
**Answer**: Extract from GGUF during model load
- Store in trace metadata JSON
- Use for post-processing correlation
- Not needed in binary log (keep entries small)

---

## RISK ASSESSMENT

### High Risk:
- ❌ **Backend specificity**: May need to modify multiple backend files
- ❌ **Performance overhead**: Logging millions of entries might slow inference >10%
- ❌ **Thread safety**: If llama.cpp uses multiple threads, need atomic operations
- ❌ **Correctness**: Difficult to validate without ground truth

### Medium Risk:
- ⚠️ **tmpfs availability**: `/dev/shm/` might not exist on all systems
- ⚠️ **Model loading hook**: May not capture all tensors if loading is complex
- ⚠️ **Struct packing**: Compiler-specific padding might break binary format
- ⚠️ **File offset accuracy**: Need to verify mapping is correct

### Low Risk:
- ✓ **mmap overflow**: Easy to detect (check offset < capacity)
- ✓ **Parsing errors**: Can validate with checksums
- ✓ **Disk space**: 2GB is manageable

---

## UNKNOWNS

**What We Don't Know Yet**:
1. How does llama.cpp load models internally? (need to read code)
2. Does llama.cpp use threading? (affects logging design)
3. What are all the ggml operation types? (need to enumerate)
4. How are MoE models represented in GGUF? (need examples)
5. Does this approach work for ALL model architectures? (Llama, GPT, Mistral, etc.)

**How to Resolve**:
- Read llama.cpp source code (especially model loading)
- Test with multiple model architectures
- Instrument and observe, iterate based on findings

---

## STATUS: READY TO IMPLEMENT ✅

**All critical decisions made** - see [02JAN.md](./journal/02JAN.md) for full discussion.

**Go/No-Go Decision**: **GO** ✅

**Next Action**: Begin Phase 1 - Step 1 (create logging infrastructure)

---

## IMPLEMENTATION CHECKLIST

**Phase 1 MVP** (Days 1-2):
- [ ] Step 1: Create tensor_trace.h and tensor_trace.c (logging infrastructure)
- [ ] Step 2: Build gguf_offset_dump.cpp (GGUF structure extractor)
- [ ] Step 3: Add minimal instrumentation (one operation: MUL_MAT)
- [ ] Step 4: First test run (10 tokens)
- [ ] Step 5: Parse & verify (parse_trace.py)

**Phase 2 Complete** (Days 3-5):
- [ ] Instrument ALL operations in ggml-cpu.c
- [ ] Add semantic parsing (layer_id, tensor names, etc.)
- [ ] Full test (100 tokens, multiple models)
- [ ] Verify correctness and overhead

**Phase 3 Analysis** (Days 6-8):
- [ ] Create visualization tools
- [ ] Generate hot/cold parameter analysis
- [ ] Document findings

**Target**: Working tensor access tracker in 8-10 days

---

**End of Initial Plan**
