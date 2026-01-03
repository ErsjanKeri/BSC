# Future Optimization Ideas

This document captures potential optimization strategies for improving SSD-backed LLM inference performance based on initial research and analysis.

## 1. Deterministic Prefetching for MoE Models

**Insight**: Deterministic prefetching might be particularly effective for Mixture-of-Experts (MoE) models.

**Inspiration**: PowerInfer's approach to sparsity (hot/cold neurons in ReLU activations)
- PowerInfer keeps hot neurons on GPU, cold neurons in RAM
- We could apply similar concept: hot tensors in RAM, cold tensors on SSD

**Key advantage**: If we can predict which experts will be activated, we can prefetch their parameters from SSD before they're needed.

**Status**: Hypothesis - needs validation with MoE model traces

---

## 2. Application-Level Buffer Manager

**Concept**: Implement a virtual memory manager inside the application layer, rather than relying on OS page cache.

**Benefits**:
- Fine-grained control over which tensors stay in RAM
- Predictive eviction based on inference patterns
- Bypass kernel overhead for known access patterns

**Challenges**:
- Requires deep ggml engine knowledge
- Complex to implement correctly
- May offer limited gains for dense models (still bound by SSD speed)

**Status**: Advanced idea - consider after validating access patterns

---

## 3. io_uring for Asynchronous I/O

**Current Problem**: llama.cpp likely uses synchronous I/O, potentially underutilizing the SSD's ~3.5 GB/s sequential read bandwidth.

**Solution**: Use Linux io_uring for true asynchronous I/O
- Submit multiple I/O requests without blocking
- Overlap computation with data loading
- Batch operations to saturate SSD bandwidth

**Challenges**:
- Requires deep kernel knowledge
- Significant refactoring of llama.cpp memory management
- Complexity may not justify gains for single-user inference

**Status**: Consider if tensor traces reveal predictable patterns suitable for prefetch

---

## 4. Dedicated Prefetch Thread

**Concept**: Spawn a background thread that continuously prefetches upcoming tensors based on:
- Known layer-by-layer execution order
- Tensor access patterns from traces
- MoE expert activation predictions

**Implementation**:
- Thread reads upcoming tensors into RAM before they're needed
- Main inference thread accesses already-loaded data
- Minimize blocking on SSD reads

**Prerequisites**:
- Complete tensor access pattern analysis
- Validate sequential vs random access behavior
- Ensure prefetch doesn't evict currently-needed data

---

## 5. Flash Attention Integration

**Note**: Flash Attention optimizes attention computation for memory hierarchy, primarily targeting GPU HBM/SRAM.

**Relevance to SSD-backed inference**: Limited
- Flash Attention focuses on reducing intermediate activation memory
- Our bottleneck is parameter loading from SSD, not activation memory
- May help reduce overall memory footprint (allowing larger models on SSD)

**Status**: Lower priority - focus on I/O optimization first

---

## 6. Swappiness as Experimental Variable

**Hypothesis**: Different swappiness values may reveal different access patterns.

### Scenario 1: Swappiness = 100 (Aggressive)
- **Behavior**: Kernel swaps pages proactively
- **Result**: Model pages swap even with available RAM
- **Use case**: Test worst-case SSD performance (maximum swapping)
- **Expected pattern**: More random access as pages swap in/out dynamically

### Scenario 2: Swappiness = 0 (Conservative)
- **Behavior**: Kernel avoids swapping until RAM is truly full
- **Result**: Model stays in RAM as long as possible
- **Use case**: Test on-demand paging (swap only when accessed)
- **Expected pattern**: More sequential access as pages load in order

### Scenario 3: Swappiness = 60 (Default)
- **Behavior**: Balanced approach
- **Result**: Middle ground between extremes
- **Use case**: Real-world baseline

### Experimental Design
```bash
# Test different swappiness values
./run_experiment.py --model deepseek-20b --scenario 100percent --swappiness 0
./run_experiment.py --model deepseek-20b --scenario 100percent --swappiness 60
./run_experiment.py --model deepseek-20b --scenario 100percent --swappiness 100
```

**Total experiments**: 2 models × 3 scenarios × 3 swappiness values = 18 experiments

**Potential finding**: Swappiness affects access patterns, which affects optimal prefetching strategy.

---

## Research Questions to Answer First

Before implementing optimizations:

1. **Are parameters accessed sequentially (layer-by-layer) or uniformly?**
   - Tensor traces will answer this definitively
   - Sequential → prefetching is viable
   - Uniform/random → cache optimization more important

2. **What is the actual bottleneck?**
   - SSD bandwidth saturation?
   - CPU processing overhead?
   - Memory copy overhead?
   - Synchronous I/O blocking?

3. **For MoE models: Can we predict expert activation?**
   - Are the same experts repeatedly activated?
   - Is there temporal locality?
   - Can we build a predictor?

4. **Does the CHEOPS paper's assumption of uniform access hold?**
   - CHEOPS claims parameters are accessed uniformly
   - Our hypothesis: Dense models are sequential (layer-by-layer)
   - Tensor traces will validate or refute

---

## Priority Order

Based on thesis timeline and expected impact:

1. **Complete tensor access pattern analysis** ← Current work
2. **Validate sequential vs uniform hypothesis**
3. **If sequential → Implement prefetch thread** (highest ROI)
4. **If uniform → Focus on cache optimization**
5. **Run swappiness experiments** (understand OS behavior)
6. **Consider io_uring** (if time permits, high complexity)
7. **MoE-specific optimizations** (depends on model availability)

---

## Implementation Constraints

- **No GPU**: CPU-only inference limits some optimization strategies
- **Timeline**: BSC thesis timeline favors simpler, proven approaches
- **Codebase**: llama.cpp is complex; minimize invasive changes
- **Validation**: All optimizations must be measurable with blktrace + tensor traces
