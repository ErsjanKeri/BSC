# Related Work: Profiling and Tracing Ecosystem for LLM Inference

This document provides a comprehensive survey of the current state-of-the-art in performance profiling, trace analysis, and visualization for Large Language Model (LLM) inference runtimes. It specifically focuses on the llama.cpp and ggml ecosystem to contextualize the Dual-Path Tensor Correlation System developed in this repository.

The survey categorizes existing tools into high-level benchmarking, static analysis, system-level profiling, and academic research, highlighting the specific observability gaps—most notably the **"Semantic Gap"**—that necessitate our custom instrumentation approach.

---

## 1. High-Level Benchmarking Tools

These tools are designed to measure aggregate performance metrics such as throughput and latency. They treat the inference engine effectively as a "black box," providing output-oriented metrics without exposing internal memory access dynamics.

### llama-bench

**Source**: [ggml-org/llama.cpp/examples/llama-bench](https://github.com/ggml-org/llama.cpp/tree/master/examples/llama-bench)

**Methodology**: Application-level timing of the inference loop.

**Capabilities**:
- Measures standard metrics: Tokens per Second (t/s), Prompt Processing (PP) time, and Token Generation (TG) time.
- Outputs aggregate statistics in Markdown, CSV, or JSON formats.
- Effective for A/B testing hardware configurations (e.g., CPU vs. GPU offload) or model quantization formats (e.g., Q4_K vs. Q8_0).

**Limitations for Micro-Optimization**:
- **Coarse Granularity**: Reports performance per batch or token, not per individual tensor operation.
- **No Memory Insight**: Cannot detect micro-architectural bottlenecks such as cache thrashing, TLB misses, or specific inefficient tensor access patterns (e.g., stride misalignments in quantized kernels).

---

### GenAI-Perf / LLM-Perf

**Source**: [triton-inference-server/perf_analyzer](https://github.com/triton-inference-server/perf_analyzer)

**Methodology**: Client-side load generation and latency measurement.

**Capabilities**:
- Stress-tests inference servers (like llama-server) to measure concurrency, Time-To-First-Token (TTFT), and Inter-Token Latency (ITL).

**Limitations**:
- **Black-Box Analysis**: Completely agnostic to the internal state of the engine. It measures the symptom (high latency) but cannot diagnose the root cause (e.g., inefficient memory layout of the KV cache or specific operator overhead).

---

## 2. Static Analysis and Graph Visualization

These tools focus on the definition and topology of the computational graph. They are excellent for understanding the model architecture but do not capture execution behavior or dynamic memory access patterns during runtime.

### alphaXiv Tensor Trace

**Source**: [https://www.alphaxiv.org/labs/tensor-trace](https://www.alphaxiv.org/labs/tensor-trace)

**Methodology**: 3D Visualization of the static ggml computation graph structures.

**Capabilities**:
- Interactive 3D visualization of Transformer architectures (e.g., Llama 3).
- Displays theoretical tensor shapes, data flow dependencies, and layer structures.

**Limitations**:
- **Static Only**: Visualizes the execution plan, not the execution reality. It does not show dynamic behaviors such as CPU-GPU synchronization stalls, actual memory fragmentation, or the temporal order of memory accesses.
- **No Addressing**: Does not map logical tensors to physical virtual memory addresses, rendering it unsuitable for cache locality or NUMA analysis.

---

### GGML Graph Dump (ggml_graph_dump_dot)

**Source**: ggml-org/llama.cpp (ggml.c)

**Methodology**: Exports the ggml_cgraph structure to Graphviz (.dot) format.

**Capabilities**:
- Generates a Directed Acyclic Graph (DAG) of all operators in the compute graph.

**Limitations**:
- **Snapshot**: Captures the graph topology at a single instant (typically graph build time).
- **No Temporal Data**: Does not record when operations actually started/stopped, nor the sequence of memory reads/writes within fused kernels (e.g., FlashAttention).

---

## 3. Hardware and System Profilers

General-purpose profiling tools provide excellent visibility into hardware counters (PMU) and OS events but lack the **"Semantic Layer"** required to map these low-level events back to high-level application logic (Tensor IDs).

### Linux perf (perf_events)

**Documentation**: [Linux Kernel Profiling](https://perf.wiki.kernel.org/)

**Methodology**: Statistical sampling of hardware counters (CPU cycles, Cache Misses) and OS events (Page Faults).

**Capabilities**:
- `perf mem`: Samples load/store latency and physical memory addresses.
- `perf stat`: Counts L1/L2/L3 cache misses, TLB misses, and branch mispredictions.

**Critical Limitation (The "Semantic Gap")**:
- **Symbol Resolution**: While perf can map instruction pointers to function names (e.g., `ggml_compute_forward_mul_mat`), it cannot map data addresses to tensor names.
- It reports "Cache miss at address 0x7f...", but cannot identify if that address belongs to the Key Cache, a Weight Matrix, or an Activation Buffer. This makes it difficult to optimize specific data structures.

---

### NVIDIA Nsight Systems / Compute

**Source**: [https://developer.nvidia.com/nsight-systems](https://developer.nvidia.com/nsight-systems)

**Methodology**: Hardware instrumentation and kernel tracing.

**Capabilities**:
- Detailed timeline of GPU kernels, PCIe transfers (H2D/D2H), and CUDA API calls.

**Limitations**:
- **Opaque Kernels**: Without manual NVTX (NVIDIA Tools Extension) instrumentation injected into the llama.cpp source code, kernels appear as generic names.
- **CPU Blindness**: While it profiles CPU threads, it does not understand ggml's custom arena allocator (`ggml_gallocr`), effectively treating the CPU memory pool as a black box.

---

### Arm Streamline

**Source**: [https://developer.arm.com/Tools%20and%20Software/Streamline%20Performance%20Analyzer](https://developer.arm.com/Tools%20and%20Software/Streamline%20Performance%20Analyzer)

**Methodology**: System-wide profiling for Arm-based processors (e.g., Apple Silicon, AWS Graviton).

**Capabilities**:
- Visualizes CPU/GPU cluster activity, thermal throttling, and cache efficiency on Arm SoCs.

**Limitations**:
- Like Nsight, it requires "Annotation Channels" to provide semantic context. Without custom instrumentation, it cannot correlate a specific cache eviction with a specific layer of the LLM.

---

## 4. Academic Research & Experimental Tools

Emerging work attempts to bridge the gap between hardware counters and software semantics, validating the necessity of our approach.

### "Analysis of Memory Access Patterns for Large Language Model Inference"

**Author**: Max Henry Fisher (Virginia Tech, 2025)

**Source**: [https://vtechworks.lib.vt.edu/items/3d8bead1-eefc-4184-942a-ecc38371341f](https://vtechworks.lib.vt.edu/items/3d8bead1-eefc-4184-942a-ecc38371341f)

**Methodology**: Custom heatmap generation using hardware page sampling (PEBS) and kernel allocation interception.

**Key Contribution**: Demonstrated that LLM inference exhibits distinct "spatial locality" and "migrating hotspots" in memory, proving that memory accesses are not uniform.

**Contrast with Our Work**:
- **Hardware-Centric vs. Software-Centric**: Fisher's work focuses on Page Placement (optimizing NUMA binding) using statistical sampling. Our work focuses on Tensor Operations (optimizing computation kernels) using deterministic instrumentation.
- **Granularity**: Fisher's tool operates at the page level (4KB/2MB). Our tool operates at the tensor/operator level, allowing for finer-grained analysis of quantization formats and intra-tensor access patterns.

---

## 5. Critical Gap Analysis: Why Custom Instrumentation?

The survey reveals a fundamental observability gap in the current ecosystem: **The "Semantic Gap" between logical tensors and physical memory addresses**.

| Feature | llama-bench | perf mem | Nsight Systems | Fisher's Thesis | Our Approach |
|---------|-------------|----------|----------------|-----------------|--------------|
| Real-time Tracing | ❌ | ✅ | ✅ | ✅ | ✅ |
| Tensor Identity (e.g., blk.0.attn_q) | ❌ | ❌ | ⚠️ (Requires NVTX) | ❌ | ✅ |
| Exact Memory Address | ❌ | ✅ | ✅ | ✅ | ✅ |
| Quantization Awareness | ❌ | ❌ | ❌ | ❌ | ✅ |
| Determinism | N/A | ❌ (Sampling) | ✅ | ❌ (Sampling) | ✅ |

---

## Our Contribution

To address this, we implemented a **Dual-Path Tensor Correlation System** that:

1. **Instruments the Runtime**: Hooks directly into ggml's compute graph execution loop (`ggml-cpu` and `ggml-backend`).

2. **Bridges the Semantic Gap**: Maintains a dynamic registry mapping runtime pointers (`void*`) back to logical tensor metadata (Name, Shape, Data Type).

3. **Logs Granular Access**: Generates a deterministic, binary-structured log of every tensor operation, enabling precise reconstruction of the "Memory Lifecycle" for a single token generation step.

This capability is essential for correlating **"High Cache Miss Rates"** (hardware symptom) with **"Specific Quantized Tensor Layouts"** (software cause), enabling targeted optimization of memory access patterns in quantized inference.

---

## Implementation Details

See [tensor-tracing/README.md](../tensor-tracing/README.md) for complete technical documentation of our instrumentation approach.
