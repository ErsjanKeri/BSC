# 7 February 2026 - MAP_POPULATE Discovery and Sparse Access Analysis

## Summary

**Discovery**: Traced through llama.cpp code to understand model loading behavior. Found that `MAP_POPULATE` flag in `mmap()` causes eager page faulting, and identified sparse access patterns in embeddings and MoE expert selection.

**Key Finding**: Theoretical analysis suggests significant unused data:
- Embeddings: ~250MB container, potentially only KB accessed per token
- MoE Experts: Only 4 of 32 experts used per layer (87.5% unused)

**Status**: Code analysis complete. **Actual performance impact unknown** - needs measurement.

---

## Part 1: Code Discovery - Model Loading Path

### Entry Point: llama-model.cpp

**File**: `llama.cpp/src/llama-model.cpp`
**Line**: 6797

```cpp
ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);
                 ^^^^
                 prefetch=TRUE
```

This `true` parameter controls MAP_POPULATE behavior downstream.

### Memory Mapping: llama-mmap.cpp

**File**: `llama.cpp/src/llama-mmap.cpp`
**Lines**: 388-390

```cpp
#ifdef __linux__
    if (prefetch) { flags |= MAP_POPULATE; }
#endif

addr = mmap(NULL, file->size(), PROT_READ, flags, fd, 0);
```

**MAP_POPULATE behavior** (from `man 2 mmap`):
> Populate (prefault) page tables for a mapping. For a file mapping,
> this causes read-ahead on the file. This will help to reduce blocking
> on page faults later.

**What this means:**
- `MAP_POPULATE=ON`: Kernel faults in all pages immediately during mmap() call
- `MAP_POPULATE=OFF`: Virtual mapping created, pages faulted on first access

**Observation from htop:**
- With MAP_POPULATE: RAM grows from 2GB → 13GB over several seconds during "loading"
- This is the kernel reading from SSD and populating page cache

---

## Part 2: Sparse Access Pattern Analysis

### 2.1 Token Embeddings (GET_ROWS Operation)

**File**: `llama.cpp/src/llama-graph.cpp`
**Line**: 1219

```cpp
// GET_ROWS operation
cur = ggml_get_rows(ctx0, tok_embd, inp->tokens);
//                         ^^^^^^^^  ^^^^^^^^^^
//                         250MB     e.g., [42, 17, 99]
//                         tensor    only 3 rows needed!
```

**Implementation** (`ggml/src/ggml-cpu/ops.cpp`, line 4623):
```cpp
for (int64_t i = ir0; i < ir1; ++i) {
    const int64_t i01 = *(int32_t *) ((char *) src1->data + ...);

    // Access ONLY specific row
    dequantize_row_q(
        (const void *) ((char *) src0->data + i01*nb01 + ...),
        //              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        //              Pointer arithmetic to specific row
        (float *) ((char *) dst->data + ...), nc);
}
```

**Key insight**: Code accesses `src0->data + i01*nb01` (specific row offset), not entire tensor.

**Theoretical analysis:**
- Container: 250MB (32K vocab × 2880 dims × 2 bytes)
- Actual access: ~10 tokens × 5760 bytes = 57KB per inference
- **Potential waste: 99.98%** (if MAP_POPULATE loads entire 250MB)

### 2.2 MoE Experts (MUL_MAT_ID Operation)

**File**: `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c`
**Line**: 1628

```cpp
// Select which expert to use
const char * src0_cur = (const char *) src0->data + cur_a * nb02;
//                                                   ^^^^^^^^^^^^
//                                                   expert_id * bytes_per_expert
```

**Model structure (GPT-OSS-20B):**
- 32 experts per layer × 24 layers
- Top-4 expert selection (router chooses 4 out of 32)
- Each expert tensor: 2880 × 2880 × 32 = 134MB

**Theoretical analysis:**
- Container: 134MB (all 32 experts)
- Actual access: 4 experts × 4.2MB = 16.8MB per token
- **Potential waste: 87.5%** per layer
- **Total across 24 layers: ~2.8GB unused** (if MAP_POPULATE loads all experts)

---

## Part 3: What We DON'T Know Yet

### Critical Questions:

1. **Does MAP_POPULATE actually load unused data?**
   - Or does the kernel optimize this?
   - Does GET_ROWS trigger page faults for only accessed rows?

2. **What takes 5+ seconds during "loading"?**
   - Is it mmap() with MAP_POPULATE?
   - Is it subsequent access patterns?
   - Is it model metadata parsing?

3. **Page fault behavior:**
   - When do page faults actually happen?
   - Are they clustered at load time or spread during inference?

4. **Performance impact:**
   - Does disabling MAP_POPULATE improve load time?
   - Does it affect inference speed?
   - What's the actual memory usage?

### What We Need to Measure:

- [ ] Load time: MAP_POPULATE ON vs OFF
- [ ] Memory usage (RSS) over time
- [ ] Disk I/O patterns (iostat, iotop)
- [ ] Page fault counts (`perf` or `/usr/bin/time -l`)
- [ ] Inference throughput (tokens/sec)

---

## Part 4: Hypothesis (Untested)

**If sparse access is real:**
- MAP_POPULATE loads entire 13GB model
- But only ~4GB actually accessed (dense layers + 4-of-32 experts + few embedding rows)
- Disabling MAP_POPULATE could reduce loading overhead
- **But:** Pages might get faulted in anyway on first access

**Trade-offs to investigate:**
- Faster load time (avoid bulk page faulting)
- vs. Potential inference slowdown (page faults during generation)
- vs. Memory pressure (if model exceeds RAM)

**Expected outcomes:**
1. **Model fits in RAM**: Minor improvement (avoid upfront bulk fault-in)
2. **Model exceeds RAM**: Major improvement (avoid thrashing from trying to load 61GB into 30GB)

---

## Part 5: Next Steps

### Immediate Actions:

1. **Run timing experiments** (see time-tracking/run_experiments.py)
   - Compare MAP_POPULATE ON vs OFF
   - Measure: load time, inference speed, memory usage
   - Use iostat to monitor actual I/O

2. **Investigate load phase** (see time-tracking/investigate_load.sh)
   - What happens during the 5-second load?
   - How much data actually read from disk?
   - When do page faults occur?

3. **Validate sparse access theory**
   - Use tensor tracing to confirm which data accessed
   - Check if unused experts really not touched
   - Measure actual vs theoretical waste

### Documentation:

- Update thesis with code analysis findings
- Document MAP_POPULATE behavior (kernel vs theory)
- Present both theoretical analysis AND measured results
- Acknowledge unknowns and uncertainties

---

## Conclusion

**What we discovered:**
- ✅ MAP_POPULATE code path through llama.cpp
- ✅ Sparse access patterns in GET_ROWS and MUL_MAT_ID operations
- ✅ Theoretical waste calculation (embeddings, experts)

**What we DON'T know:**
- ❌ Actual performance impact
- ❌ Why load takes 5+ seconds without MAP_POPULATE
- ❌ Real page fault behavior
- ❌ Optimal configuration

**Philosophy:** Trust measurements over theory. Code analysis suggests possibilities, but only experiments reveal truth.

**Next session:** Run experiments, measure reality, update understanding.

---

**End of Entry**
