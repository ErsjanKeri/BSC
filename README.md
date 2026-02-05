# BSC Thesis: Storage-Backed Large Language Model Inference

## Tensor-Level Access Pattern Analysis and Optimization for SSD-Backed Sparse Expert Architectures

Research on understanding and optimizing large language model inference when model weights are stored on SSD rather than RAM, with focus on Mixture-of-Experts (MoE) architectures.

## Problem Statement

**Context**: Modern LLMs (20B+ parameters) don't fit in commodity RAM. SSD-backed inference enables running large models on consumer hardware, but performance characteristics are poorly understood.

**Research Questions**:
1. Are model parameters accessed **sequentially** (layer-by-layer) or **uniformly** (randomly)?
2. Does GGUF file layout (alphabetical tensor ordering) impact SSD performance?
3. How does expert sparsity (4-of-32 selection in MoE) affect access patterns?
4. What is the actual SSD bandwidth utilization during inference?
5. Can we implement prefetching optimizations based on observed patterns?

**Approach**: Application-level tensor tracing with offline analysis pipeline.

---

## Instrumentation Approach

### Primary Method: Tensor-Level Tracing
**Location**: [tensor-tracing/](tensor-tracing/) ([llama.cpp fork](https://github.com/ErsjanKeri/llama.cpp))

**Tool**: Custom instrumentation in llama.cpp GGML backend

**Captures**:
- Tensor operations (ALL 95 GGML ops)
- Memory source detection (DISK vs BUFFER)
- MoE expert selection (4-of-32 routing per layer)
- Multi-source tracking (up to 4 source tensors per operation)
- Timestamps (nanosecond precision)
- Computation graphs
- Buffer allocations

**Binary Format**: 1024-byte cache-aligned entries for <1% overhead

### Initial Exploration: Disk I/O Benchmarking (Historical)
**Location**: [disk-benchmarking/](disk-benchmarking/)

**Note**: This was an initial approach using Linux `blktrace` for OS-level block I/O tracing. While functional, it proved limiting compared to application-level tensor tracing. The tensor tracing approach provides superior granularity (tensor-level vs block-level) and captures semantic information (which experts, which layers) that block-level tracing cannot provide.

---

## Quick Start

### Tensor Tracing

**One command:**
```bash
cd tensor-tracing
python3 run_experiment.py
```

Automatically: cleans old traces → runs llama-completion → parses all data → moves to webui/public/data/

**View results:**
```bash
cd webui
npm install  # First time only
npm run dev  # http://localhost:5173
```

**Configuration**: Edit `tensor-tracing/settings.json`

See [tensor-tracing/README.md](tensor-tracing/README.md) for details.

### Disk I/O Benchmarking (Historical - Optional)

**Note**: This was an initial exploration approach. For most analysis, use tensor tracing above.

```bash
cd disk-benchmarking
sudo python3 run_experiment.py
```

Results in `results/experiment_TIMESTAMP/`

See [disk-benchmarking/README.md](disk-benchmarking/README.md) for details.

---

## Current Status (2026-02-05)

### Tensor Tracing ✅ INFRASTRUCTURE COMPLETE

**Working**:
- 1024-byte trace format (128-byte names, expert IDs, zero truncation)
- ALL operations logged (95 ggml ops)
- Expert-level MoE tracking (32 experts × 24 layers = 2,304 expert tensors)
- Buffer tracking, computation graphs
- Automated pipeline (`run_experiment.py`)
- WebUI (3-view dynamic layout: Graph + Trace + Heatmap)
- Name-based correlation (100% accuracy across all views)
- Temporal heatmap with zoom (0.1x - 500x)
- ✅ **GGUF parsing completely fixed** (supports all 40+ quantization formats)
- ✅ **Zero overlaps** (was 2,344 overlaps, now 0)
- ✅ **MXFP4 quantization** correctly handled (0.53125 bytes/element)

**All Critical Bugs Fixed**:
- ✅ Name truncation (expanded to 128 bytes - 2026-01-13)
- ✅ GGUF offset bug (data section offset added - 2026-01-13)
- ✅ Address correlation (switched to name-based - 2026-01-13)
- ✅ Quantization size inflation (gguf-dump fixed with type_traits table - 2026-01-17)

### Expert Activation Analysis ✅ DATA COLLECTED

**Completed** (2026-02-05):
- 5 domain experiments (code, math, creative, factual, mixed)
- 100 tokens per domain = 500 tokens total
- ~850 MB tensor trace data
- Expert activation patterns captured (72 MoE ops per token)
- Different expert subsets per domain confirmed

### Desktop UI (2026-01-25) ✅ PRODUCTION-READY

**Purpose**: Analyze large-scale traces (100+ tokens, 170,000+ entries) that browser WebUI cannot handle

**Features**:
- C++17 + ImGui + ImPlot (native performance)
- Dual visualization: Colored strip (viridis gradient) + Step function graph (quantitative)
- Temporal timeline scrubbing (0-649ms)
- Expert-level granularity (2,691 tensors)
- DISK-only access filtering
- Virtual scrolling table (handles millions of rows)
- Unified window with docking (professional layout)
- 110 FPS rendering performance

**Build**: [desktopui/README.md](desktopui/README.md)

---

## System Architecture

### Data Collection and Analysis Pipeline

```
┌─────────────────────────────────────────────────────┐
│ Layer 1: Data Collection (C/C++)                   │
│  - GGML backend hooks (tensor_trace.c)             │
│  - 1024-byte binary format (cache-aligned)         │
│  - <1% instrumentation overhead                    │
│  - Output: /tmp/tensor_trace.bin                   │
└──────────────────┬──────────────────────────────────┘
                   │ Binary trace file
┌──────────────────▼──────────────────────────────────┐
│ Layer 2: Processing Pipeline (Python)              │
│  - TraceParser: Binary → JSON (parse_trace.py)     │
│  - GGUFParser: Model metadata (parse_csv.py)       │
│  - GraphParser: Computation DAG (parse_dot.py)     │
│  - BufferParser: Memory events (parse_buffer.py)   │
│  - Automated workflow (run_experiment.py)          │
│  - Output: JSON files in webui/public/data/        │
└──────────────────┬──────────────────────────────────┘
                   │ Structured JSON data
┌──────────────────▼──────────────────────────────────┐
│ Layer 3: Visualization (TypeScript/C++)            │
│  - WebUI: React + Zustand (1-2 tokens)            │
│    - GraphView, TraceView, HeatmapView             │
│    - Name-based correlation, temporal filtering    │
│  - DesktopUI: C++ + ImGui + ImPlot (100+ tokens)  │
│    - Dual-plot visualization                       │
│    - Virtual scrolling, 110 FPS                    │
└─────────────────────────────────────────────────────┘
```

### Design Principles

**Modularity**: Each layer is independent
- Can swap parsers (Python → Rust) without touching instrumentation
- Can swap visualization (WebUI → DesktopUI) without re-parsing
- Binary format = stable API contract between layers

**Performance**: Optimized at every stage
- Binary format: 1000x faster than JSON for trace writes
- Cache-aligned structs: Minimize memory access overhead
- Virtual scrolling: Only render visible rows (10-20 of 170K)
- ImPlot GPU acceleration: 2,691 tensors at 110 FPS

**Testability**: Systematic validation
- Static assertions prevent struct size regressions
- Overlap detection validates GGUF parsing (156 → 0 overlaps)
- Immediate testing after every change

---

## Performance Metrics

### Instrumentation Overhead
- **Binary trace format**: <1% CPU overhead during inference
- **1024-byte structs**: Cache-aligned for minimal memory contention
- **Thread-local buffers**: Avoid lock contention across 16 threads

### Data Processing
- **Binary → JSON parsing**: ~1 second for 1,696 entries
- **GGUF parsing**: 2,691 tensors (12.85 GB) in <2 seconds
- **Expert expansion**: 459 base tensors → 2,691 indexed tensors

### Visualization Performance
**WebUI (Browser)**:
- Handles: 1-2 tokens (~1,700 entries)
- FPS: 60 (limited by React rendering)
- Memory: ~500 MB (browser limitation)

**DesktopUI (Native C++)**:
- Handles: 100+ tokens (170,000+ entries)
- FPS: 110 (ImPlot GPU-accelerated)
- Memory: ~200 MB (optimized data structures)
- Virtual scrolling: Renders only 10-20 visible rows

### Bug Discovery and Fixes

| Issue | Impact | Resolution | Date |
|-------|--------|------------|------|
| Name truncation (20 bytes) | Lost expert tensor info | Expanded to 128 bytes | 2026-01-13 |
| GGUF offset bug | 156 overlapping tensors | Added data section offset | 2026-01-13 |
| Q4_K size inflation | 7.11x oversized tensors | Fixed ggml type_traits | 2026-01-17 |
| MXFP4 quantization | Incorrect block size | Added MXFP4 support | 2026-01-17 |
| Address correlation | 0% match rate | Switched to name-based | 2026-01-13 |

---

## Related Work and Comparison

### Fisher (Virginia Tech, 2025) - NUMA Page Placement

**Similar**: Memory access pattern analysis for LLM inference (llama.cpp)

**Different**:
- **Memory tier**: NUMA (fast DRAM ↔ slow DRAM) vs **SSD-backed** (RAM ↔ Storage)
- **Latency**: 50ns ↔ 150ns vs **50ns ↔ 100μs (2000x slower!)**
- **Model**: Gemma 27B (dense) vs **GPT-OSS-20B (MoE with 32 experts)**
- **Granularity**: 4-16 MB granules (perf mem sampling) vs **Tensor-level (exact logging)**
- **Scope**: Model fits in RAM (348GB available) vs **Model exceeds RAM (limited memory)**

**Key Fisher finding**: "Once in page cache, layout doesn't matter" (warm start: 0.5% penalty)

**Why this doesn't apply to SSD-backed**:
- Assumes model stays in page cache
- Your scenario: Limited RAM → pages evicted → repeated SSD reads
- Physical file layout (alphabetical ordering) could matter for SSD seeks

### Alphabetical Tensor Ordering Discovery (2026-01-26)

**Finding**: GGUF files store tensors in alphabetical order (blk.0, blk.1, blk.10, blk.11, ..., blk.2, ...)

**Verified by**: Reading GGUF binary directly (not trusting tools)

**Impact on inference**:
- Inference accesses: L0 → L1 → L2 → ... → L23 (sequential)
- File layout: L0 @ 0.0GB, L1 @ 0.4GB, L10 @ 0.8GB, L2 @ 4.7GB (alphabetical)
- Seek pattern: L1→L2 requires +4.3 GB jump (skipping L10-L19 data)

**Open question**: Does alphabetical ordering hurt SSD performance?
- **Fisher's work**: Doesn't investigate (NUMA doesn't care about virtual address layout)
- **Opus research**: Says "no" for in-memory (page cache mitigates)
- **Your thesis**: Needs empirical measurement for SSD-backed scenario

**Proposed experiment**: Create layer-sequential GGUF version, compare with blktrace

---

## Key Hypothesis

**CHEOPS paper**: LLM parameters accessed uniformly (randomly)

**My hypothesis**: Dense models accessed **sequentially** (layer-by-layer)

**Validation**: Analyze tensor trace layer_id sequence to measure access patterns

---

## Potential Optimizations

If sequential access confirmed:

1. **Deterministic Prefetching**: Background thread prefetches next layer while current computes
2. **Async I/O (io_uring)**: Batch multiple reads, saturate SSD bandwidth
3. **Custom Buffer Manager**: Application-level page cache tailored to inference patterns

---

## Documentation

- [tensor-tracing/README.md](tensor-tracing/README.md) - Technical details (256-byte format, tools)
- [disk-benchmarking/README.md](disk-benchmarking/README.md) - blktrace workflow
- [journal/](journal/) - Research log (recommended reading)
- [docs/](docs/) - Reference material

**Key Journal Entries**:
- [2026-01-13.md](journal/2026-01-13.md) - Critical debugging: GGUF offset bug, Q4_K inflation, address correlation fix
- [2026-01-14.md](journal/2026-01-14.md) - Complete project summary, MoE expert tracking, MXFP4 bug discovered
- [2026-01-17.md](journal/2026-01-17.md) - gguf-dump.cpp completely fixed, quantization support, zero overlaps achieved
- [2026-01-25.md](journal/2026-01-25.md) - Desktop UI implementation: ImGui + ImPlot, dual visualization, viridis colormap
- [2026-01-26.md](journal/2026-01-26.md) - GGUF alphabetical ordering investigation, comparison with Fisher thesis, virtual vs physical memory

---

## Project Structure

```
BSC/
├── README.md                    ← You are here
├── journal/                     ← Research logs (2025-12 → 2026-01)
├── docs/                        ← Reference material
├── tensor-tracing/              ← Tensor-level instrumentation (primary approach)
│   ├── run_experiment.py        ← One-command pipeline
│   ├── settings.json            ← Configuration
│   ├── tools/                   ← Python parsers (binary/CSV/DOT/buffer)
│   └── webui/                   ← React visualization (1-2 tokens)
├── desktopui/                   ← C++ desktop analyzer (100+ tokens)
│   ├── CMakeLists.txt           ← Build configuration
│   ├── src/                     ← C++ source (main, parsers, views)
│   ├── external/                ← Dependencies (ImGui, ImPlot, json)
│   ├── data/                    ← JSON data files (from tensor-tracing)
│   └── build/                   ← Build output
└── disk-benchmarking/           ← OS-level disk I/O (historical exploration)
    ├── run_experiment.py        ← Automated blktrace
    ├── settings.json            ← Configuration
    └── analysis/                ← DuckDB analysis scripts
```

---

## Server

**Hardware**: TUM Server (cli-hiwi-02.dis.cit.tum.de)
- CPU: AMD Ryzen 7 7700X (8c/16t)
- RAM: 30 GiB
- Storage: 2× NVMe (Samsung 980 PRO 1TB, WD 960GB)

See [docs/server-setup.md](docs/server-setup.md) for specs.
