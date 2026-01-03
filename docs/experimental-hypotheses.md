# Experimental Hypotheses - SSD-Backed Inference

## Primary Research Question

**How do access patterns (sequential vs random) in real LLM models affect SSD-backed inference performance?**

---

## Hypothesis 1: Model Architecture & Access Patterns

### H1.1: Standard Transformers are Sequential
**Hypothesis**: Standard transformer models (like Llama-2) will show **highly sequential access patterns** (>80%) during inference.

**Reasoning**:
- Transformers process layer-by-layer: Layer 1 → Layer 2 → ... → Layer N
- Parameters are accessed in order as computation flows through layers
- Weight matrices are typically stored contiguously in model files

**Expected Result**:
- Sequential read percentage: **80-95%**
- Pattern classification: **HIGHLY SEQUENTIAL**

**Implications**:
- SSD sequential read performance matters most
- Kernel readahead should be very effective
- Large prefetch buffers beneficial

---

### H1.2: MoE Models Show More Randomness
**Hypothesis**: Mixture-of-Experts (MoE) models will show **more random access patterns** (40-60% sequential) compared to standard transformers.

**Reasoning**:
- MoE uses sparse activation: only K out of N experts activated per token
- Expert selection varies by input → non-uniform parameter access
- Experts may be distributed across model file, not sequential

**Expected Result**:
- Sequential read percentage: **40-60%** (lower than standard transformers)
- Pattern classification: **MIXED** or **MOSTLY SEQUENTIAL**

**Implications**:
- Random read performance becomes more important
- Readahead less effective (reads unneeded data)
- Smaller page sizes might be better (less read amplification)

**Test**: Compare `llama-2-7b` (standard) vs `gpt-oss-20b` (if MoE)

---

## Hypothesis 2: Performance Degradation with SSD

### H2.1: Baseline (RAM) is Fastest
**Hypothesis**: Inference with model fully in RAM will be **fastest** (baseline performance).

**Expected Result**:
- No page faults, no disk I/O
- Pure compute-bound performance

---

### H2.2: 100% SSD Shows Significant Slowdown
**Hypothesis**: Inference with entire model on SSD will show **2-5x slowdown** compared to baseline.

**Reasoning**:
- Every parameter access requires disk read (first time)
- SSD latency: ~0.1ms per read
- NVMe sequential throughput: 3-7 GB/s (much slower than RAM's ~50 GB/s)

**Expected Result**:
- Tokens/sec reduced by 50-80%
- Slowdown factor: **2-5x**
- High major page fault count

---

### H2.3: 50% SSD is Sublinear Slowdown
**Hypothesis**: 50% on SSD will show **less than 2x** slowdown (not linear).

**Reasoning**:
- Kernel can prefetch from SSD while computing on cached parameters
- Overlapping I/O with computation
- Most frequently accessed parameters (embeddings, early layers) might stay in RAM

**Expected Result**:
- Slowdown factor: **1.3-1.8x** (not 2.5x)

---

## Hypothesis 3: Swappiness Impact on Access Patterns

### H3.1: High Swappiness (100) Causes More Random Access
**Hypothesis**: `vm.swappiness=100` will result in **more random access patterns** and **higher page thrashing**.

**Reasoning**:
- Kernel proactively swaps pages even when RAM not completely full
- Pages may be swapped out and back in multiple times during inference
- Aggressive swapping → less predictable page residence

**Expected Result**:
- More page faults (both major and minor)
- Lower sequential read percentage
- Higher total bytes read from SSD (due to thrashing)
- Potentially worse performance than low swappiness

---

### H3.2: Low Swappiness (0) Results in Sequential Access
**Hypothesis**: `vm.swappiness=0` will result in **more sequential access patterns** and **on-demand-only paging**.

**Reasoning**:
- Kernel avoids swapping until absolutely necessary
- Pages loaded once when first accessed, stay in RAM
- Inference processes layers sequentially → sequential page-in

**Expected Result**:
- Fewer total page faults (load once, keep in RAM)
- Higher sequential read percentage
- Lower total bytes read from SSD
- Better performance despite being "on SSD" (due to better caching)

---

### H3.3: Optimal Swappiness is Workload-Dependent
**Hypothesis**: There is an **optimal swappiness value** between 0-100 that balances:
- Keeping frequently-used parameters in RAM (lower swappiness)
- Allowing room for computation buffers (higher swappiness)

**Expected Result**:
- `swappiness=0`: Best for sequential workloads (standard transformers)
- `swappiness=60`: Balanced for mixed workloads
- `swappiness=100`: Stress test, shows worst-case performance

**Test**: Run experiments with swappiness ∈ {0, 60, 100}

---

## Hypothesis 4: SSD Throughput vs Theoretical Max

### H4.1: Measured Throughput Below Spec
**Hypothesis**: Measured SSD throughput during inference will be **40-70%** of theoretical sequential read spec.

**Reasoning**:
- Model file access is not purely sequential (some randomness)
- Page granularity (4KB) vs SSD optimal block size (128KB+)
- Kernel overhead (page fault handling, TLB misses)
- Swap file fragmentation

**Expected Result**:
- NVMe spec: 3-7 GB/s sequential
- Measured: **1.5-4 GB/s** during inference

---

### H4.2: Sequential Models Achieve Higher Throughput
**Hypothesis**: Models with higher sequential read % will achieve **closer to theoretical max** throughput.

**Expected Result**:
- Standard transformer (90% sequential) → 60-70% of spec
- MoE model (50% sequential) → 40-50% of spec

**Implication**: Access pattern directly correlates with achievable SSD performance

---

## Hypothesis 5: Page Size Impact (Future Work)

### H5.1: Larger Page Sizes Help Sequential Access
**Hypothesis**: Increasing page size (4KB → 2MB huge pages) will **improve performance** for sequential models.

**Reasoning**:
- Fewer page faults (fewer interrupts)
- Better alignment with SSD block sizes
- Reduced TLB pressure

**Expected Benefit**: +10-30% throughput for sequential workloads

---

### H5.2: Smaller Page Sizes Help Random Access
**Hypothesis**: Standard 4KB pages are **better for random access** (MoE models).

**Reasoning**:
- Large pages cause read amplification (read 2MB, use only 100KB)
- Wastes memory and bandwidth
- Small pages = fine-grained control

**Expected Result**: MoE models show minimal benefit (or regression) with huge pages

---

## Experimental Design to Test Hypotheses

### Phase 1: Basic Characterization (Current Plan)
Test matrix: 2 models × 3 scenarios = **6 experiments**

| Model | Scenario | Tests Hypotheses |
|-------|----------|------------------|
| llama-2-7b | baseline | H2.1 (RAM fastest) |
| llama-2-7b | 50% SSD | H2.3 (sublinear slowdown) |
| llama-2-7b | 100% SSD | H1.1 (sequential), H2.2 (slowdown) |
| gpt-oss-20b | baseline | H2.1 |
| gpt-oss-20b | 50% SSD | H2.3 |
| gpt-oss-20b | 100% SSD | H1.2 (MoE random?), H2.2 |

Compare: H1.1 vs H1.2 (sequential % difference between models)

---

### Phase 2: Swappiness Exploration (Extended)
Test matrix: 2 models × 1 scenario × 3 swappiness = **6 additional experiments**

Focus on 100% SSD scenario (most sensitive to swappiness):

| Model | Swappiness | Tests Hypotheses |
|-------|------------|------------------|
| llama-2-7b | 0 | H3.2 (sequential, on-demand) |
| llama-2-7b | 60 | H3.3 (balanced) |
| llama-2-7b | 100 | H3.1 (aggressive, random) |
| gpt-oss-20b | 0 | H3.2 |
| gpt-oss-20b | 60 | H3.3 |
| gpt-oss-20b | 100 | H3.1 |

**Total**: 12 experiments (6 baseline + 6 swappiness)

---

## Success Criteria

### Must Answer:
1. ✅ **Do real models show sequential or random access?**
   - Measure: Sequential read % from blktrace
   - Target: Quantify difference between standard transformer vs MoE

2. ✅ **How much does SSD slow down inference?**
   - Measure: Slowdown factor (baseline tokens/sec ÷ SSD tokens/sec)
   - Target: Quantify 50% vs 100% SSD impact

3. ✅ **What throughput do we actually achieve?**
   - Measure: MB/s from iostat during inference
   - Target: Compare to SSD spec sheet

### Nice to Have:
4. **Does swappiness affect access patterns?**
   - Measure: Sequential % vs swappiness value
   - Target: Find optimal swappiness for each model type

5. **Can we predict performance from access pattern?**
   - Measure: Correlation between sequential % and tokens/sec
   - Target: Build simple model: `performance = f(sequential%, SSD_spec)`

---

## Implications for Gabriel's Research

If hypotheses confirmed:

### H1.1 + H1.2 True (Different access patterns):
→ **Page size should be architecture-specific**
- Standard transformers: Large pages (2MB) + aggressive prefetch
- MoE models: Small pages (4KB) + conservative prefetch

### H3.1 + H3.2 True (Swappiness affects patterns):
→ **Kernel tuning matters for SSD-backed inference**
- Low swappiness better for production (fewer thrashing)
- Adaptive swappiness based on memory pressure

### H4.1 True (Gap between measured and theoretical):
→ **Significant optimization headroom exists**
- Current: 40-70% of spec
- Potential: Custom page cache, better alignment, direct I/O

---

## Open Questions

1. **Does prompt length affect access patterns?**
   - Longer context → more KV cache → different memory pressure
   - Test: Vary `--tokens` parameter (50, 100, 500)

2. **Does batch size matter?**
   - llama.cpp typically runs batch=1 for inference
   - Larger batches might amortize page faults

3. **What about model quantization?**
   - Q4_K_M (4-bit) vs F16 (16-bit)
   - Smaller models = less memory pressure = different patterns?

4. **First token latency vs sustained throughput?**
   - Cold start: All parameters on SSD
   - Warm: Some parameters cached
   - Which matters more for real applications?

---

## Timeline

- **Week 1**: Phase 1 (6 experiments) - Basic characterization
- **Week 2**: Phase 2 (6 experiments) - Swappiness exploration
- **Week 3**: Analysis, writing, presentation to Gabriel

**Priority**: Focus on Phase 1 first. Only do Phase 2 if time permits!
