# Implementation Plan: Expert-Level Granularity for MoE Heatmap

**Date:** 2026-01-14
**Status:** ✅ **COMPLETED** 2026-01-17
**Goal:** Modify tensor tracing pipeline to track and visualize individual MoE expert accesses

---

## ✅ COMPLETION SUMMARY (2026-01-17)

**All 5 phases completed successfully:**

✅ **Phase 1:** gguf-dump modified to split expert tensors into 32 entries
✅ **Phase 2:** Expert IDs added to trace format (1024-byte with expert_ids[16])
✅ **Phase 3:** parse_trace.py updated to extract expert IDs
✅ **Phase 4:** parse_csv.py handles expert entries with expert_id field
✅ **Phase 5:** Heatmap shows 32 individual expert bars with highlighting

**Critical Bug Fixed (2026-01-17):**
- MXFP4 quantization support added to gguf-dump.cpp
- Complete GGML type_traits table (40+ formats)
- Zero overlaps achieved (was 2,344)
- Expert tensor sizes correct: 4.2 MB (was 31.6 MB)

**See journal entry:** [2026-01-17.md](../journal/2026-01-17.md)

---

## ORIGINAL PLAN (2026-01-14)

---

## Overview

Currently, MoE expert tensors (e.g., `blk.0.ffn_gate_exps.weight`) are treated as single monolithic blocks (1 GB each). This plan adds expert-level granularity to detect which of the 32 experts are actually accessed during inference.

**Model:** GPT-OSS-20B F16
- 24 layers
- 32 experts per layer
- Top-4 expert selection per token

---

## Phase 1: Modify gguf-dump Tool

### Objective
Split 3D expert tensors `[2880, 2880, 32]` into 32 separate entries in the CSV output.

### Current Behavior
```csv
blk.0.ffn_down_exps.weight,0,1061683200,0,FFN Down,3,2880,2880,32,0
```

### Desired Behavior
```csv
blk.0.ffn_down_exps.weight[0],0,33177600,0,FFN Down Expert 0,2,2880,2880,0,0
blk.0.ffn_down_exps.weight[1],33177600,33177600,0,FFN Down Expert 1,2,2880,2880,0,0
...
blk.0.ffn_down_exps.weight[31],1028505600,33177600,0,FFN Down Expert 31,2,2880,2880,0,0
```

### Implementation Steps

#### Step 1.1: Detect MoE Expert Tensors
**File:** `llama.cpp/tools/gguf-dump/gguf-dump.cpp`

**Logic:**
```cpp
bool is_moe_expert_tensor(const char* name, uint32_t n_dims) {
    // Check if tensor name contains "_exps" and has 3 dimensions
    return (n_dims == 3 &&
            (strstr(name, "ffn_down_exps") != NULL ||
             strstr(name, "ffn_gate_exps") != NULL ||
             strstr(name, "ffn_up_exps") != NULL));
}
```

**Test:** For `blk.0.ffn_down_exps.weight` with dims `[2880, 2880, 32]`, should return `true`.

#### Step 1.2: Calculate Expert Slice Offsets
**Formula:**
```cpp
// For expert tensor with shape [dim0, dim1, n_experts]
uint64_t expert_size = (dim0 * dim1 * element_size);  // Size of one expert
uint64_t expert_offset = base_offset + (expert_id * expert_size);
```

**Example:**
- Base offset: 0
- Expert 0: offset = 0, size = 33,177,600
- Expert 1: offset = 33,177,600, size = 33,177,600
- ...
- Expert 31: offset = 1,028,505,600, size = 33,177,600

#### Step 1.3: Output Expert Entries
**Modification to main loop (line 289-301):**

```cpp
for (const auto& t : tensors) {
    // Check if this is an MoE expert tensor
    if (is_moe_expert_tensor(t.name.c_str(), t.n_dims) && t.n_dims == 3) {
        // Split into individual experts
        uint64_t n_experts = t.ne[2];
        size_t element_size = (tensor_type == 1) ? 2 : 4;
        uint64_t expert_size = t.ne[0] * t.ne[1] * element_size;

        for (uint64_t exp_id = 0; exp_id < n_experts; exp_id++) {
            uint64_t expert_offset = t.offset + (exp_id * expert_size);

            printf("%s[%llu],%llu,%llu,%d,%s Expert %llu,%u,%llu,%llu,0,0\n",
                   t.name.c_str(),
                   exp_id,
                   expert_offset,
                   expert_size,
                   t.layer_id,
                   t.component_type.c_str(),
                   exp_id,
                   2,  // n_dims = 2 (not 3 anymore)
                   t.ne[0],
                   t.ne[1]);
        }
    } else {
        // Normal tensor - output as before
        printf("%s,%llu,%llu,%d,%s,%u,%llu,%llu,%llu,%llu\n",
               t.name.c_str(), ...);
    }
}
```

### Verification Steps

**After modification:**

1. **Build the tool:**
   ```bash
   cd ~/llama.cpp
   cmake --build build --target llama-gguf-dump
   ```

2. **Run on test model:**
   ```bash
   ./build/bin/llama-gguf-dump models/gpt-oss-20b-F16.gguf 2>/dev/null | grep "ffn_down_exps.weight" | head -35
   ```

3. **Expected output:**
   - Should see 32 lines for `blk.0.ffn_down_exps.weight[0]` through `[31]`
   - Offsets should be incremental (0, 33177600, 66355200, ...)
   - Total of 459 + (32×3×24 - 3×24) = 459 + 2232 = **2,691 tensor entries**

4. **Verify no overlaps:**
   ```bash
   ./build/bin/llama-gguf-dump models/gpt-oss-20b-F16.gguf 2>/dev/null > /tmp/test_experts.csv
   python3 ~/BSC/tensor-tracing/tools/check_overlaps.py /tmp/test_experts.csv
   ```

### Issues to Handle

**Issue 1:** Need `tensor_type` available in the output loop (currently scoped)
- **Solution:** Store `tensor_type` in `tensor_info` struct

**Issue 2:** CSV format change might break existing parsers
- **Solution:** Keep backward compatibility - add `--split-experts` flag

**Issue 3:** File offset calculation needs data section offset
- **Solution:** Already handled in `parse_csv.py` - will work automatically

---

## Phase 2: Add Expert ID Tracking to Traces

### Objective
Capture which expert IDs were actually selected during MUL_MAT_ID operations.

### Current Trace Format
```c
struct TensorAccessLog {
    // ...existing fields...
    struct SourceTensorInfo sources[4];
    uint8_t padding2[232];
};
```

### Proposed Modification

**Option 1: Add expert_ids array (minimal change)**
```c
struct TensorAccessLog {
    // ...existing fields...
    struct SourceTensorInfo sources[4];

    // NEW: Expert IDs for MoE operations
    int32_t expert_ids[16];      // Up to 16 expert IDs (top-K, padded)
    uint8_t num_experts;         // How many expert IDs are valid
    uint8_t padding2[151];       // Adjusted padding (232 - 64 - 1 = 167, no wait...)
};
```

**Wait - this breaks the 1024-byte alignment!** Let me recalculate:
- Original padding: 232 bytes
- Expert IDs array: 16 × 4 = 64 bytes
- num_experts: 1 byte
- New padding: 232 - 64 - 1 = 167 bytes
- **Total still 1024 bytes ✓**

### Implementation Steps

#### Step 2.1: Modify tensor_trace.h
**File:** `llama.cpp/ggml/include/tensor_trace.h`

**Changes:**
1. Add expert_ids field to `TensorAccessLog` struct (line ~70)
2. Adjust padding2 size (line ~71)
3. Update static assertion (should still be 1024)

#### Step 2.2: Modify tensor_trace.c
**File:** `llama.cpp/ggml/src/tensor_trace.c`

**Add function to extract expert IDs:**
```c
// Extract expert IDs from MUL_MAT_ID operation
// Returns number of expert IDs extracted (0 if not MUL_MAT_ID)
static uint8_t extract_expert_ids(const struct ggml_tensor * dst, int32_t * out_ids, uint8_t max_ids) {
    if (dst->op != GGML_OP_MUL_MAT_ID) {
        return 0;
    }

    // src[2] is the ids tensor
    struct ggml_tensor * ids = dst->src[2];
    if (ids == NULL || ids->data == NULL) {
        return 0;
    }

    // Read expert IDs from ids tensor
    int32_t * id_data = (int32_t *)ids->data;

    // ids shape is [n_expert_used, n_tokens]
    // We want the expert IDs for all tokens
    uint64_t n_expert_used = ids->ne[0];
    uint64_t n_tokens = ids->ne[1];

    // For simplicity, collect top K experts (first row/column depending on layout)
    uint8_t count = 0;
    for (uint64_t i = 0; i < n_expert_used && count < max_ids; i++) {
        out_ids[count++] = id_data[i];  // Assuming row-major layout
    }

    return count;
}
```

**Modify `tensor_trace_log_operation()` function (line ~432):**
```c
void tensor_trace_log_operation(const struct ggml_tensor * dst, int ith) {
    // ...existing code...

    // NEW: Extract expert IDs if this is MUL_MAT_ID
    entry.num_experts = extract_expert_ids(dst, entry.expert_ids, 16);

    // ...rest of existing code...
}
```

#### Step 2.3: Verify Memory Layout
**Test:**
```c
// In ggml/tests/test_tensor_trace.c
printf("TensorAccessLog size: %zu bytes\n", sizeof(struct TensorAccessLog));
// Should print: 1024 bytes
```

### Verification Steps

**After modification:**

1. **Rebuild llama.cpp:**
   ```bash
   cd ~/llama.cpp
   cmake -B build -DGGML_TENSOR_TRACE=ON
   cmake --build build -j16
   ```

2. **Run quick test:**
   ```bash
   # On server
   cd ~/BSC/tensor-tracing
   ./run_experiment.py
   ```

3. **Check trace output:**
   ```python
   import struct

   with open('/tmp/tensor_trace.bin', 'rb') as f:
       entry_bytes = f.read(1024)
       # Unpack and verify expert_ids are populated
   ```

4. **Verify expert IDs are valid (0-31 range)**

---

## Phase 3: Update Parser to Handle Expert IDs

### Objective
Parse expert_ids from binary trace and include in JSON output.

### Implementation Steps

#### Step 3.1: Modify parse_trace.py
**File:** `BSC/tensor-tracing/tools/parse_trace.py`

**Update binary unpacking (around line ~50-80):**
```python
# Current format
ENTRY_FORMAT = '<QIHHBBBx5x128s640s232x'  # 1024 bytes

# New format (with expert_ids)
ENTRY_FORMAT = '<QIHHBBBx5x128s640s16iB151x'  # 1024 bytes
#                                     ^^^ ^^^^
#                                     |    num_experts + padding
#                                     16 expert IDs (int32)
```

**Parse expert IDs:**
```python
def parse_entry(data):
    unpacked = struct.unpack(ENTRY_FORMAT, data)
    # ...existing parsing...

    expert_ids_raw = unpacked[11:27]  # 16 int32 values
    num_experts = unpacked[27]

    expert_ids = list(expert_ids_raw[:num_experts]) if num_experts > 0 else []

    return {
        # ...existing fields...
        'expert_ids': expert_ids,
        'num_experts': num_experts
    }
```

**Add to JSON output:**
```python
entry_json = {
    # ...existing fields...
    'expert_ids': entry['expert_ids'],  # NEW
    'num_experts': entry['num_experts']  # NEW
}
```

### Verification Steps

1. **Run parser:**
   ```bash
   cd ~/BSC/tensor-tracing
   python3 tools/parse_trace.py /tmp/tensor_trace.bin --export-json webui/public/data/traces/
   ```

2. **Check JSON output:**
   ```bash
   grep -A 5 "MUL_MAT_ID" webui/public/data/traces/token-00000.json | grep "expert_ids"
   ```

3. **Expected:** For MUL_MAT_ID operations, should see `"expert_ids": [5, 12, 18, 27]` or similar

---

## Phase 4: Update parse_csv.py for Expert Entries

### Objective
Parse the expanded CSV with expert subdivisions and create granular memory-map.json.

### Implementation Steps

#### Step 4.1: Detect Expert Entries in CSV
**File:** `BSC/tensor-tracing/tools/parse_csv.py`

**Logic:**
```python
def is_expert_entry(tensor_name):
    """Check if tensor name has [N] expert index"""
    import re
    return re.search(r'\[\d+\]$', tensor_name) is not None

def extract_expert_id(tensor_name):
    """Extract expert ID from name like 'blk.0.ffn_down_exps.weight[5]'"""
    import re
    match = re.search(r'\[(\d+)\]$', tensor_name)
    return int(match.group(1)) if match else None
```

**Process expert entries:**
```python
for row in csv_reader:
    tensor_name = row['tensor_name']

    if is_expert_entry(tensor_name):
        expert_id = extract_expert_id(tensor_name)
        base_name = tensor_name.rsplit('[', 1)[0]  # Remove [N] suffix

        tensors.append({
            'name': tensor_name,  # Keep [N] in name
            'base_name': base_name,  # For grouping
            'expert_id': expert_id,  # NEW
            'offset_start': int(row['file_offset']) + data_offset,
            'size_bytes': int(row['size_bytes']),
            # ...rest of fields...
        })
    else:
        # Normal tensor - process as before
        tensors.append({...})
```

### Verification Steps

1. **Run parser:**
   ```bash
   cd ~/llama.cpp
   ./build/bin/llama-gguf-dump models/gpt-oss-20b-F16.gguf 2>/dev/null > /tmp/experts.csv

   cd ~/BSC/tensor-tracing
   python3 tools/parse_csv.py --csv /tmp/experts.csv --output /tmp/test_memory_map.json
   ```

2. **Check output:**
   ```bash
   cat /tmp/test_memory_map.json | python3 -m json.tool | grep -A 3 "ffn_down_exps.weight\[0\]"
   ```

3. **Expected:** Should see 32 entries for each expert tensor type (down, gate, up) per layer

4. **Verify total count:**
   ```python
   import json
   with open('/tmp/test_memory_map.json') as f:
       data = json.load(f)

   print(f"Total tensors: {len(data['tensors'])}")
   # Expected: 459 - 72 (original expert tensors) + 72×32 (expanded) = 2691

   expert_tensors = [t for t in data['tensors'] if 'expert_id' in t]
   print(f"Expert tensors: {len(expert_tensors)}")  # Expected: 72×32 = 2304
   ```

---

## Phase 5: Update Heatmap Visualization

### Objective
Display expert-level bars in heatmap and highlight accessed experts.

### Implementation Steps

#### Step 5.1: Modify Heatmap Data Processing
**File:** `BSC/tensor-tracing/webui/src/components/HeatmapView.tsx`

**Group experts for rendering:**
```typescript
// Separate expert tensors from regular tensors
const expertTensors = memoryMap.tensors.filter(t => t.expert_id !== undefined);
const regularTensors = memoryMap.tensors.filter(t => t.expert_id === undefined);

// For each expert tensor, track access counts
const expertAccessCounts = new Map<string, number>(); // "blk.0.ffn_down[5]" -> count

traceData.entries.forEach(entry => {
    if (entry.operation_type === 'MUL_MAT_ID' && entry.expert_ids) {
        // This operation used specific experts
        entry.expert_ids.forEach(expertId => {
            // Find which expert tensor was accessed
            entry.sources.forEach(src => {
                if (src.name.includes('_exps.weight')) {
                    const key = `${src.name}[${expertId}]`;
                    expertAccessCounts.set(key, (expertAccessCounts.get(key) || 0) + 1);
                }
            });
        });
    }
});
```

#### Step 5.2: Render Expert Bars
**Visual representation:**
```tsx
{expertTensors.map(tensor => {
    const accessCount = expertAccessCounts.get(tensor.name) || 0;
    const color = accessCount > 0 ? getHeatColor(accessCount, maxAccess) : '#374151';

    return (
        <rect
            x={tensor.offset_start / BYTES_PER_PIXEL}
            width={(tensor.size_bytes) / BYTES_PER_PIXEL}
            height={BAR_HEIGHT}
            fill={color}
            title={`${tensor.name} - Accessed ${accessCount} times`}
        />
    );
})}
```

### Verification Steps

1. **Start WebUI:**
   ```bash
   cd ~/BSC/tensor-tracing/webui
   npm run dev
   ```

2. **Visual checks:**
   - [ ] See 32 thin bars for each expert tensor (not one thick bar)
   - [ ] Accessed experts colored (red gradient)
   - [ ] Unaccessed experts gray
   - [ ] Hover tooltip shows expert ID

3. **Sanity check:**
   - [ ] Total memory span still equals file size (~13.8 GB)
   - [ ] No overlaps between expert bars
   - [ ] Layer filtering still works

---

## Testing & Validation

### Test Case 1: Expert Detection
**Goal:** Verify correct expert IDs are captured

**Method:**
```python
# Check if expert IDs make sense
with open('webui/public/data/traces/token-00000.json') as f:
    trace = json.load(f)

mul_mat_id_ops = [e for e in trace['entries'] if e['operation_type'] == 'MUL_MAT_ID']

for op in mul_mat_id_ops[:5]:
    print(f"{op['dst_name']}: experts {op['expert_ids']}")

# Expected: Expert IDs should be in range 0-31, top-4 selected
```

### Test Case 2: Heatmap Visual Verification
**Goal:** Ensure heatmap correctly shows expert granularity

**Method:**
- Open WebUI
- Navigate to Layer 0
- Check `blk.0.ffn_down_exps` region
- Should see 32 subdivisions
- 4 should be highlighted (accessed), 28 should be gray

### Test Case 3: Access Pattern Analysis
**Goal:** Confirm which experts are "hot" vs "cold"

**Method:**
```python
# Analyze expert usage across all layers
expert_usage = {}  # (layer, expert_id) -> count

for entry in trace['entries']:
    if entry.get('expert_ids'):
        layer = entry['layer_id']
        for exp_id in entry['expert_ids']:
            key = (layer, exp_id)
            expert_usage[key] = expert_usage.get(key, 0) + 1

# Show hottest experts
sorted_experts = sorted(expert_usage.items(), key=lambda x: x[1], reverse=True)
print("Top 10 hottest experts:")
for (layer, exp_id), count in sorted_experts[:10]:
    print(f"  Layer {layer}, Expert {exp_id}: {count} accesses")
```

---

## Rollback Plan

**If things break:**

### Rollback 1: gguf-dump tool
```bash
cd ~/llama.cpp
git checkout tools/gguf-dump/gguf-dump.cpp
cmake --build build --target llama-gguf-dump
```

### Rollback 2: tensor_trace.h/c
```bash
cd ~/llama.cpp
git checkout ggml/include/tensor_trace.h
git checkout ggml/src/tensor_trace.c
cmake --build build -j16
```

### Rollback 3: Parsers
```bash
cd ~/BSC/tensor-tracing
git checkout tools/parse_trace.py
git checkout tools/parse_csv.py
```

---

## Success Criteria

### Phase 1 Complete When:
- [  ] gguf-dump outputs 32 lines per expert tensor
- [  ] All offsets calculated correctly (no overlaps)
- [  ] CSV format compatible with existing pipeline

### Phase 2 Complete When:
- [  ] Trace captures expert IDs for MUL_MAT_ID ops
- [  ] expert_ids array contains valid IDs (0-31)
- [  ] num_experts matches model config (4 for GPT-OSS-20B)

### Phase 3 Complete When:
- [  ] parse_trace.py correctly extracts expert_ids
- [  ] JSON contains expert_ids for MUL_MAT_ID operations
- [  ] No parsing errors

### Phase 4 Complete When:
- [  ] parse_csv.py handles expert entries
- [  ] memory-map.json has 2,691 tensor entries
- [  ] Expert tensors have expert_id field

### Phase 5 Complete When:
- [  ] Heatmap shows 32 bars per expert tensor
- [  ] Accessed experts highlighted
- [  ] Hover shows expert ID and access count
- [  ] Visual verification confirms top-4 selection

---

## Open Questions to Resolve

**Before starting implementation:**

1. **Expert ID layout:** When reading `ids->data`, is it row-major or column-major?
   - Need to verify with small test in llama.cpp

2. **Top-K value:** Confirmed 4 experts, but need to verify this in actual trace data

3. **Backward compatibility:** Should old traces (without expert_ids) still work?
   - Proposed: Check `format_version` field, handle gracefully

4. **CSV flag:** Should expert splitting be optional (`--split-experts`) or always-on?
   - Proposed: Always-on for simplicity

---

## Estimated Timeline

- **Phase 1 (gguf-dump):** 2-3 hours (coding + testing)
- **Phase 2 (trace format):** 3-4 hours (careful struct modification + rebuild)
- **Phase 3 (parse_trace.py):** 1-2 hours (straightforward)
- **Phase 4 (parse_csv.py):** 1-2 hours (straightforward)
- **Phase 5 (heatmap):** 2-3 hours (visualization logic)

**Total:** ~10-14 hours of careful implementation + testing

---

## Next Step

**BEFORE I START CODING:**

Please confirm:
1. Is this plan correct and complete?
2. Any concerns or modifications needed?
3. Should I start with Phase 1 (gguf-dump modification)?

Once confirmed, I'll proceed **one phase at a time** with verification after each step.
