# Implementation Plan: Tensor Name Correlation

## Goal
Enable 100% correlation between runtime trace entries and CSV structure by logging tensor names.

## Current Issues
1. All trace entries show `layer_id=0` (never populated)
2. Cannot correlate `tensor_ptr` (memory address) with CSV's `tensor_name`
3. Intermediate tensors have no `file_offset` to match

## Proposed Changes

### Change 1: Expand TensorAccessLog Structure

**File**: `llama.cpp/ggml/include/tensor_trace.h`

**Current**: 64-byte struct
**Modified**: 128-byte struct with tensor name field

```c
// 128-byte fixed-size log entry (2x cache-line aligned)
struct TensorAccessLog {
    // === Timestamp (8 bytes) ===
    uint64_t timestamp_ns;        // Nanoseconds since trace start

    // === Execution Context (16 bytes) ===
    uint32_t token_id;            // Which token being processed
    uint16_t layer_id;            // Which transformer layer (0-21, 65535=N/A)
    uint16_t thread_id;           // CPU thread ID
    uint8_t  operation_type;      // Enum: MUL_MAT, ADD, ROPE, etc.
    uint8_t  phase;               // Enum: PROMPT, GENERATE
    uint16_t padding1;            // Alignment
    uint32_t padding1b;           // Alignment

    // === Tensor Identification (20 bytes) ===
    uint32_t tensor_idx;          // Index into tensor name table (unused for now)
    uint64_t tensor_ptr;          // Virtual address of tensor->data
    uint64_t file_offset;         // Offset in GGUF file (0 = not in file/intermediate)
    uint32_t size_bytes;          // Size of tensor in bytes

    // === Attention-Specific (4 bytes) ===
    uint8_t  attention_head;      // Which attention head (0-127, or 255=N/A)
    uint8_t  qkv_type;            // Enum: Q, K, V, O, or N/A
    uint16_t padding2;            // Alignment

    // === MoE-Specific (12 bytes) ===
    uint8_t  expert_id;           // Which expert (0-255, or 255=N/A)
    uint8_t  expert_rank;         // Routing rank (0=top, 1=second, etc.)
    uint16_t routing_score;       // Quantized routing score (0-65535)
    uint32_t padding3;            // Alignment
    uint32_t padding4;            // Final padding

    // === NEW: Direct Tensor Name (64 bytes) ===
    char tensor_name[64];         // Tensor name from ggml_tensor->name (e.g., "blk.5.attn_q.weight")

    // Total: 128 bytes (verified at compile time)
} __attribute__((packed));

// Static assertion to ensure struct is exactly 128 bytes
_Static_assert(sizeof(struct TensorAccessLog) == 128,
               "TensorAccessLog must be exactly 128 bytes");
```

**Changes**:
- Added `char tensor_name[64]` field at end
- Updated static assertion: 64 → 128 bytes
- Updated comment to reflect 2x cache-line alignment

### Change 2: Extract Layer ID from Tensor Name

**File**: `llama.cpp/ggml/include/tensor_trace.h`

Add helper function before API functions:

```c
// === Helper Functions ===

// Extract layer ID from tensor name (e.g., "blk.5.attn_q.weight" → 5)
// Returns 65535 (UINT16_MAX) if not a layer tensor
static inline uint16_t tensor_trace_extract_layer_id(const char* name) {
    // Pattern: "blk.N.*"
    if (name && strncmp(name, "blk.", 4) == 0) {
        int layer = -1;
        if (sscanf(name + 4, "%d", &layer) == 1 && layer >= 0) {
            return (uint16_t)layer;
        }
    }
    return 65535;  // Not a layer tensor (embeddings, output, etc.)
}
```

### Change 3: Update Tracing Code to Log Names

**File**: `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c`

**Current code** (lines 1237-1256):
```c
#ifdef GGML_TENSOR_TRACE
    if (params->ith == 0) {
        struct TensorAccessLog entry = {0};
        entry.timestamp_ns = tensor_trace_get_timestamp_ns();
        entry.operation_type = 1;  // MUL_MAT
        entry.thread_id = tensor_trace_get_thread_id();

        // Log src0 access (weight matrix)
        entry.tensor_ptr = (uint64_t)src0->data;
        entry.size_bytes = (uint32_t)ggml_nbytes(src0);
        tensor_trace_log(&entry);

        // Log src1 access (input activations)
        entry.tensor_ptr = (uint64_t)src1->data;
        entry.size_bytes = (uint32_t)ggml_nbytes(src1);
        tensor_trace_log(&entry);
    }
#endif
```

**Modified code**:
```c
#ifdef GGML_TENSOR_TRACE
    if (params->ith == 0) {
        struct TensorAccessLog entry = {0};
        entry.timestamp_ns = tensor_trace_get_timestamp_ns();
        entry.operation_type = 1;  // MUL_MAT
        entry.thread_id = tensor_trace_get_thread_id();

        // Log src0 access (weight matrix)
        entry.tensor_ptr = (uint64_t)src0->data;
        entry.size_bytes = (uint32_t)ggml_nbytes(src0);
        // NEW: Copy tensor name and extract layer ID
        strncpy(entry.tensor_name, src0->name, sizeof(entry.tensor_name) - 1);
        entry.tensor_name[sizeof(entry.tensor_name) - 1] = '\0';  // Ensure null termination
        entry.layer_id = tensor_trace_extract_layer_id(src0->name);
        tensor_trace_log(&entry);

        // Log src1 access (input activations)
        entry.tensor_ptr = (uint64_t)src1->data;
        entry.size_bytes = (uint32_t)ggml_nbytes(src1);
        // NEW: Copy tensor name and extract layer ID
        strncpy(entry.tensor_name, src1->name, sizeof(entry.tensor_name) - 1);
        entry.tensor_name[sizeof(entry.tensor_name) - 1] = '\0';
        entry.layer_id = tensor_trace_extract_layer_id(src1->name);
        tensor_trace_log(&entry);
    }
#endif
```

**What this does**:
1. Copies tensor name from `src0->name` and `src1->name` into trace entry
2. Extracts layer ID using helper function (e.g., "blk.5" → 5)
3. For non-layer tensors (embeddings, output), layer_id = 65535

## Expected Results After Changes

### Trace Entry Example (src0 = weight matrix):
```
timestamp_ns:     125000000
token_id:         0
layer_id:         5              ← NOW POPULATED!
thread_id:        12345
operation_type:   1 (MUL_MAT)
tensor_ptr:       0x12A4B6000
size_bytes:       16777216
tensor_name:      "blk.5.attn_q.weight"  ← NOW POPULATED!
```

### CSV Match:
```csv
tensor_name,file_offset,size_bytes,layer_id,component_type
blk.5.attn_q.weight,143630336,16777216,5,Attention Q
```

### Correlation:
- Match by `tensor_name` field (100% accurate)
- Verify `layer_id` matches between trace and CSV
- `size_bytes` should match

## Impact Assessment

### File Size Impact:
- **Before**: 252 entries × 64 bytes = 16,128 bytes (~16 KB)
- **After**: 252 entries × 128 bytes = 32,256 bytes (~32 KB)
- **Growth**: 2x (negligible for test workload)

### Memory Impact (2GB mmap buffer):
- Before: 33,554,432 max entries
- After: 16,777,216 max entries
- Still plenty for long inference runs

### Performance Impact:
- Additional `strncpy()` per entry (minimal overhead)
- Additional `sscanf()` for layer ID parsing (one-time per entry)
- **Negligible** compared to actual matmul computation

## Build and Test Steps

1. **Modify tensor_trace.h** (add field, helper function)
2. **Modify ggml-cpu.c** (update tracing code)
3. **Rebuild**:
   ```bash
   cd llama.cpp
   cmake --build build -j
   ```
4. **Clear old trace**:
   ```bash
   rm -f /tmp/tensor_trace.bin
   ```
5. **Run inference**:
   ```bash
   ./build/bin/llama-completion -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p "Hello" -n 5
   ```
6. **Verify new data**:
   ```bash
   # Check file size (should be 32KB not 16KB)
   ls -lh /tmp/tensor_trace.bin

   # Hexdump should show ASCII tensor names
   hexdump -C /tmp/tensor_trace.bin | grep -A2 "blk\."
   ```

## Verification Script

```python
import struct

with open('/tmp/tensor_trace.bin', 'rb') as f:
    entry_num = 0
    while True:
        data = f.read(128)  # Now 128 bytes per entry!
        if len(data) < 128:
            break

        # Parse entry
        timestamp = struct.unpack('<Q', data[0:8])[0]
        if timestamp == 0:
            break

        token_id = struct.unpack('<I', data[8:12])[0]
        layer_id = struct.unpack('<H', data[12:14])[0]
        thread_id = struct.unpack('<H', data[14:16])[0]
        operation_type = struct.unpack('<B', data[16:17])[0]

        tensor_ptr = struct.unpack('<Q', data[24:32])[0]
        size_bytes = struct.unpack('<I', data[40:44])[0]

        # NEW: Extract tensor name (bytes 64-128)
        tensor_name_bytes = data[64:128]
        tensor_name = tensor_name_bytes.split(b'\x00')[0].decode('utf-8', errors='ignore')

        print(f"Entry {entry_num}: layer={layer_id}, size={size_bytes:,}, name={tensor_name}")
        entry_num += 1

        if entry_num >= 20:  # Show first 20
            break
```

Expected output:
```
Entry 0: layer=65535, size=262144000, name=token_embd.weight
Entry 1: layer=65535, size=524288, name=
Entry 2: layer=0, size=8192, name=blk.0.attn_norm.weight
Entry 3: layer=0, size=2097152, name=blk.0.attn_k.weight
Entry 4: layer=0, size=16777216, name=blk.0.attn_q.weight
...
```

## Next Steps After Implementation

1. **Verify layer_id extraction** works correctly (blk.0 → 0, blk.21 → 21)
2. **Match trace entries with CSV** by tensor_name field
3. **Analyze access patterns**:
   - Which tensors accessed most frequently?
   - Are they sequential through layers?
   - Do intermediate tensors have names?
4. **Plan optimizations** based on real data

## Questions to Consider

1. Do intermediate tensors (activations) have meaningful names, or are they empty/generic?
2. Should we handle anonymous tensors differently (layer_id=65535, name="")?
3. For thesis analysis, do you need to track which specific token is being processed (token_id field)?

---

**Status**: Ready to implement. All changes are well-defined and isolated to 3 files.
