# December 21, 2025 - LLM Parameter Offloading Experiments

## PART 3: 1-Extent Experiment & Critical Findings

### 3.1 Motivation: Eliminating Fragmentation Artifacts

**Problem from Previous Experiments:**
- Model file had 19-26 extents (fragmented across disk)
- Backward seeks: **~75%** regardless of memory pressure
- Could not distinguish: fragmentation artifacts vs true transformer access pattern
- **Critical question:** Are backward seeks from file fragmentation or from LLM inference?

**Solution:**
- Move model to nvme0n1 (`/mnt/experiment_ssd`) - only 2% full vs nvme1n1's 68% full
- Fresh filesystem with abundant contiguous free space
- Result: **1 extent** (perfectly contiguous file)

---

### 3.2 Hardware Configuration

**Important Note: nvme0n1 is MUCH slower than nvme1n1**
- nvme0n1: Experiment SSD (slower hardware)
- nvme1n1: Root/system SSD (faster hardware)
- Performance differences expected and acceptable for this analysis

**Updated Configuration:**
```json
{
  "storage": {
    "block_device": "/dev/nvme0n1",
    "blktrace_staging": "/tmp/blktrace_staging"
  },
  "paths": {
    "models_dir": "/mnt/experiment_ssd"
  }
}
```

---

### 3.3 Results: 1-Extent vs Fragmented Comparison

| Configuration | Device | Extents | mlock | Time (s) | tok/s | Total Read (GB) | Unique (GB) | Coverage | Backward Seeks |
|---------------|--------|---------|-------|----------|-------|-----------------|-------------|----------|----------------|
| **Fragmented** | nvme1n1 | 19 | 20GB | 11.67 | 8.57 | 12.66 | 2.89 | 75.9% | **74.6%** |
| **Contiguous Run 1** | nvme0n1 | **1** | 20GB | 14.34 | 6.97 | 11.82 | 3.80 | **100.0%** | **67.0%** |
| **Contiguous Run 2** | nvme0n1 | **1** | 20GB | 14.48 | 6.91 | 11.84 | 3.80 | **100.0%** | **67.0%** |
| **Contiguous Run 3** | nvme0n1 | **1** | 21GB | 16.15 | 6.19 | 14.56 | 3.80 | **100.0%** | **67.6%** |

---

### 3.4 Critical Finding #1: Backward Seeks May Be Intrinsic (Not Conclusive)

**Key Observation:**
- Fragmented (19 extents): **74.6% backward seeks**
- Contiguous (1 extent): **67.0% backward seeks**
- **Only 8% improvement!**

**This suggests (but does not prove) that backward seeks might be intrinsic to transformer inference, not primarily from fragmentation.**

**However, we remain uncertain because:**
1. We don't fully understand the transformer architecture we're running
2. Could be a combination of both factors
3. Need more investigation to confirm

**Possible Explanations for Intrinsic Backward Seeks:**

1. **Multi-head attention accesses different layers non-sequentially**
   - Attention mechanisms jump between query, key, value matrices
   - Different heads access different parameter regions

2. **KV cache references scattered across model**
   - Key-value cache for attention requires non-contiguous memory access
   - Previous token states referenced from various model layers

3. **Embedding table lookups jump around vocabulary**
   - Token embeddings retrieved based on input, not sequentially
   - ~500MB embedding table, but only ~1MB accessed per inference

4. **Layer normalization parameters at different offsets**
   - Layer norms interspersed throughout model
   - Accessed in layer order, not file order

5. **Future consideration: Mixture of Experts (MoE)**
   - Not applicable to current llama-2-7b model
   - But relevant for future experiments with MoE architectures
   - Experts selected dynamically, causing scattered access

**Important:** We need to study llama.cpp architecture and transformer implementation details to validate these hypotheses.

---

### 3.5 Critical Finding #2: 100% Coverage - More Accurate Measurement

**Observation:**
- Fragmented (nvme1n1): 2.89 GB unique (75.9% coverage)
- Contiguous (nvme0n1): **3.80 GB unique (100.0% coverage)**

**Why This Makes Sense:**
- File size: 3.80 GB
- Unique sectors: 7,970,712 sectors × 512 bytes = 4,081,004,544 bytes = 3.80 GB
- **We're accessing the ENTIRE model file!**

**Interpretation:**
This is likely **more accurate** than the previous 75.9% measurement because:
1. Dynamic GGUF sector range detection (no hardcoded values)
2. Correct extent parsing from filefrag
3. Full model traversal during 100 token inference

**Why 100% Coverage for Only 100 Tokens?**

Possible reasons:
1. **Model initialization scan:** llama.cpp might scan entire file during startup
2. **Metadata/validation:** Full file read to validate .gguf structure
3. **All layers activated:** Even for 100 tokens, all transformer layers process data
4. **Embedding table:** Entire vocabulary table might be loaded (even if only subset used)

**Action Item:** Need to investigate llama.cpp source code to understand file access patterns during initialization vs inference.

---

### 3.6 Critical Finding #3: Gap Distribution Changes

**Sequential Access Improved:**

| Configuration | Perfect Sequential | Small Gaps | Medium Gaps | Large Gaps | Backward Seeks |
|---------------|-------------------|------------|-------------|------------|----------------|
| Fragmented (19 extents) | 21.7% | 1.9% | 1.3% | 0.5% | **74.6%** |
| Contiguous (1 extent) | **30.2%** | 1.3% | 0.9% | 0.6% | **67.0%** |

**Key Changes:**
- ✅ Perfect sequential: **21.7% → 30.2%** (+8.5% improvement)
- ✅ Backward seeks: **74.6% → 67.0%** (-7.6% improvement)

**Interpretation:**
- Some fragmentation artifacts eliminated (8% of backward seeks were fragmentation)
- **But majority (67%) of backward seeks remain** - possibly intrinsic to transformer access pattern
- Gap distribution more accurately reflects inference behavior

---

### 3.7 Critical Finding #4: Performance Degradation Expected (Slower Hardware)

**Performance Comparison:**

| Device | Hardware | Extents | Time | tok/s | Notes |
|--------|----------|---------|------|-------|-------|
| nvme1n1 | Faster SSD | 19 | 11.67s | 8.57 | Root/system drive |
| nvme0n1 | **Slower SSD** | 1 | 14.34s | 6.97 | Experiment drive |

**23% slower on nvme0n1 is expected and acceptable** because:
- nvme0n1 is inherently slower hardware
- Filesystem differences (XFS vs ext4)
- This experiment prioritizes **accurate access pattern measurement** over performance

---

### 3.8 CRITICAL CONCERN: 6GB RAM Required for 3.8GB Model

**Observed Memory Behavior:**

From experiments:
- Model file size: **3.80 GB**
- Minimum RAM needed to run without thrashing: **~6 GB**
- 20GB lock (10 GB free): Works well (11.67s, 8.57 tok/s)
- 21GB lock (9 GB free): Still works but slower (16.15s, 6.19 tok/s)
- 22GB lock (8 GB free): Severe thrashing (30s, 3.31 tok/s)
- 23GB lock (7 GB free): Fast again (suggests giving up on caching)

**The Problem:**
- File is only 3.8 GB
- But we need **~6 GB available RAM** to avoid thrashing
- **~2.2 GB overhead is suspicious!**

**Possible Explanations:**

1. **KV cache memory:**
   - Transformers maintain key-value cache for generated tokens
   - Grows with sequence length
   - Could explain 1-2 GB extra

2. **Activation memory:**
   - Intermediate activations during forward pass
   - Layer outputs temporarily stored
   - Could be 500MB-1GB

3. **llama.cpp internals:**
   - Model loading overhead
   - Graph computation buffers
   - Thread-local storage

4. **OS page cache overhead:**
   - Metadata for mmap'd regions
   - Page table entries
   - Buffer management structures

**Action Item:** **We need to study llama.cpp memory usage!**
- Read llama.cpp documentation
- Examine source code for memory allocation
- Use `pmap` or `/proc/[pid]/maps` to analyze actual memory layout
- Understand difference between file size and runtime memory requirements

This is **critical for thesis validity** - need to explain why 3.8GB model requires 6GB RAM.

---

### 3.9 Implications for Thesis

**What We Now Know:**

1. ✅ **Unique sectors metric is reliable** - consistent measurements with dynamic detection
2. ✅ **Fragmentation had minor impact** - only 8% of backward seeks from fragmentation
3. ❓ **Backward seeks likely intrinsic** - but need to confirm with architecture study
4. ❓ **100% coverage needs validation** - likely correct, but verify with llama.cpp analysis
5. ❌ **Memory overhead unexplained** - 6GB needed for 3.8GB model is concerning

**What We Still Need to Investigate:**

1. **llama.cpp memory architecture:**
   - Why 6GB for 3.8GB model?
   - Initialization vs inference memory usage
   - KV cache size and allocation

2. **Transformer access pattern validation:**
   - Are backward seeks truly intrinsic?
   - Can we trace specific attention/embedding accesses?
   - Profile with longer sequences (1000 tokens) to see if pattern changes

3. **Gap distribution interpretation:**
   - What do 67% backward seeks mean for SSD offloading?
   - How does this compare to CHEOPS assumptions?
   - Is random access pattern acceptable for thesis claims?

---

### 3.10 Next Steps (Prioritized)

**Priority 1: Understand llama.cpp Memory Usage**
```bash
# During inference, analyze memory layout
pmap -x [llama-cli-pid] > memory_map.txt

# Check which regions are mmap'd model file vs heap/stack
grep gguf /proc/[llama-cli-pid]/maps
```

**Priority 2: Validate 100% Coverage**
- Run experiment with fewer tokens (10 tokens) - should coverage drop?
- Run with more tokens (1000 tokens) - should coverage stay 100%?
- This will confirm whether we're reading entire file at init or during inference

**Priority 3: Study Transformer Architecture**
- Read llama.cpp source code (especially attention mechanisms)
- Understand attention.cpp, llama.cpp file structure
- Map backward seeks to specific transformer operations

**Priority 4: Compare with Other Models**
- Run same experiments with gpt-oss-20b (13 GB)
- See if gap distribution scales similarly
- Validate findings across different architectures

---

### 3.11 Updated Experimental Setup

**Final Configuration for Accurate Measurements:**

```json
{
  "storage": {
    "block_device": "/dev/nvme0n1",
    "blktrace_staging": "/tmp/blktrace_staging"
  },
  "paths": {
    "models_dir": "/mnt/experiment_ssd"
  }
}
```

**Model Location:**
- `/mnt/experiment_ssd/llama-2-7b-chat.Q4_K_M.gguf`
- **1 extent** (perfectly contiguous)
- Verified with: `filefrag /mnt/experiment_ssd/llama-2-7b-chat.Q4_K_M.gguf`

**Tracing:**
- Device: `/dev/nvme0n1` (where model lives)
- Staging: `/tmp/blktrace_staging` (on nvme1n1, not traced device)
- Zero RAM interference

---

### 3.12 Summary

**Major Achievements:**
1. ✅ Achieved 1-extent contiguous model file
2. ✅ Eliminated fragmentation as confounding variable
3. ✅ Dynamic GGUF sector range detection working
4. ✅ 100% coverage measurement (more accurate)

**Critical Findings:**
1. 🔍 Backward seeks **might be** intrinsic to transformers (67% with 1 extent)
2. 🔍 Sequential access improved only 8% (fragmentation had minor impact)
3. ❗ **6GB RAM needed for 3.8GB model** - needs explanation!
4. ✅ Gap distribution now reflects true inference pattern (not fragmentation)

**Remaining Concerns:**
1. ❓ Why does llama.cpp need 6GB for 3.8GB model?
2. ❓ Are backward seeks truly intrinsic or measurement artifact?
3. ❓ Does 100% coverage mean full file scan at init?
4. ❓ How do findings translate to CHEOPS comparison?

**For Thesis:**
- Document both fragmented and contiguous results
- Explain that gap analysis shows ~67% backward seeks (intrinsic to transformers)
- Address 6GB memory requirement mystery
- Validate assumptions with llama.cpp architecture study

---

## Appendix: Raw Data

### Experiment 1 (20GB lock, Run 1):
- Time: 14.34s
- Tokens/sec: 6.97
- Total reads: 11.82 GB
- Unique sectors: 7,970,712 (3.80 GB, 100%)
- Gap distribution: 30.2% seq, 67.0% backward
- Cache delta: 0.92 GB

### Experiment 2 (20GB lock, Run 2):
- Time: 14.48s
- Tokens/sec: 6.91
- Total reads: 11.84 GB
- Unique sectors: 7,970,712 (3.80 GB, 100%)
- Gap distribution: 29.8% seq, 67.0% backward
- Cache delta: 0.90 GB

### Experiment 3 (21GB lock):
- Time: 16.15s
- Tokens/sec: 6.19
- Total reads: 14.56 GB
- Unique sectors: 7,970,712 (3.80 GB, 100%)
- Gap distribution: 26.3% seq, 67.6% backward
- Cache delta: 0.99 GB

**Consistency:** Unique sectors identical across all runs (validates measurement reliability)
