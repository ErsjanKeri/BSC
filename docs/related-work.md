# Related Work: Existing Tools for LLM Profiling

This document surveys existing tools for profiling and tracing LLM inference, explaining why we needed to build custom instrumentation for this thesis.

## Existing Tools Survey

### alphaXiv Tensor Trace
**Link**: https://www.alphaxiv.org/labs/tensor-trace

**What it does**:
- Interactive 3D visualization of GGML tensor operations
- Shows tensor flow through layers with shapes
- Displays computation graph structure

**Limitation**: Static analysis only - doesn't track runtime access patterns

**Verdict**: Useful for understanding graph structure, but doesn't capture actual memory accesses during inference.

---

### llama-bench
**Link**: https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md

**What it does**:
- Official performance testing tool for llama.cpp
- Outputs CSV/JSON/markdown with performance metrics
- Measures tokens/sec, latency, throughput

**Limitation**: High-level benchmarking only - no tensor-level detail

**Verdict**: Good for performance comparison across models, not for access pattern analysis.

---

### GGML Computation Graph Logging
**Link**: https://github.com/ggml-org/llama.cpp/discussions/11039

**What it does**:
- Can export computation graph to CSV file
- Shows tensor operations and dependencies

**Limitation**: Terminates immediately after logging - doesn't run during inference

**Verdict**: Shows graph structure but not runtime behavior.

---

### GGML_PERF (Removed)
**Link**: https://github.com/ggml-org/llama.cpp/discussions/6871

**What it was**:
- Built-in performance profiling feature in llama.cpp
- Removed in PR #8017
- Users now resort to Linux `perf` tool

**Limitation**:
- `perf` is sampling-based (not deterministic)
- Too coarse-grained (function-level, not tensor-level)
- Doesn't know tensor semantics (can't distinguish Q vs K vs V)

**Verdict**: Insufficient granularity for memory access analysis.

---

### NVIDIA Nsight / DLProf
**Link**: https://developer.nvidia.com/blog/profiling-and-optimizing-deep-neural-networks-with-dlprof-and-pyprof/

**What it does**:
- GPU-focused profiling with TensorBoard integration
- Shows kernel execution, memory transfers
- Detailed GPU performance metrics

**Limitation**: GPU-only - not applicable to CPU inference

**Verdict**: Not usable for our CPU-based server setup.

---

### Arm Streamline
**Link**: https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama_cpp_streamline/

**What it does**:
- System-wide profiler for ARM-based systems
- Can profile llama.cpp inference workflow
- Shows CPU utilization, cache misses, etc.

**Limitation**: General-purpose profiler without tensor-level semantics

**Verdict**: Could be complementary but doesn't solve our core need.

---

## Critical Gap in Existing Tools

### What's Missing

None of the existing tools provide:
- ✗ Real-time tensor access tracking during inference
- ✗ Per-token, per-layer, per-operation granularity
- ✗ Tensor name identification (which specific weight matrices accessed?)
- ✗ Attention head-level detail (Q/K/V matrix tracking)
- ✗ MoE expert activation tracking (which experts accessed per token)
- ✗ Memory access pattern visualization (temporal + spatial)
- ✗ File offset correlation (which bytes in GGUF file accessed)
- ✗ Page fault correlation (which accesses caused disk I/O)
- ✗ Correlation with disk I/O tracing (blktrace integration)

### Why This Matters for Our Thesis

Our research questions require:
1. **Tensor-level granularity**: Which specific tensors (blk.5.attn_q.weight) accessed when?
2. **Runtime behavior**: Not just graph structure, but actual access patterns during inference
3. **Disk I/O correlation**: Match tensor accesses to block-level I/O (blktrace)
4. **Memory hierarchy analysis**: Understand RAM vs SSD access patterns

**None of the existing tools provide this level of detail.**

---

## Our Solution: Custom Instrumentation

We implemented custom tensor tracing by:
1. Instrumenting llama.cpp at the CPU backend level (`ggml-cpu.c`)
2. Logging every tensor access with:
   - Tensor name (e.g., "blk.5.attn_q.weight")
   - Timestamp (nanosecond precision)
   - Layer ID, operation type, size
   - Memory address (for correlation with page faults)
3. Memory-mapped binary logging (minimal overhead)
4. Post-processing tools to analyze patterns

**Result**: Complete visibility into which tensors are accessed, when, and in what order - enabling correlation with disk I/O traces.

See [tensor-tracing/README.md](../tensor-tracing/README.md) for implementation details.
