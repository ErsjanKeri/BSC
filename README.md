# BSC Thesis: Optimizing SSD-Backed LLM Inference

Research on optimizing large language model inference when model weights are stored on SSD rather than RAM.

## Problem Statement

**Goal**: Understand and optimize SSD-backed LLM inference performance.

**Research Questions**:
1. Are model parameters accessed **sequentially** (layer-by-layer) or **uniformly** (randomly)?
2. What is the actual SSD bandwidth utilization during inference?
3. Can we implement optimizations to improve throughput?

**Approach**: Two complementary instrumentation threads.

---

## Two-Thread Architecture

### Thread 1: Disk I/O Benchmarking
**Location**: [disk-benchmarking/](disk-benchmarking/)

**Tool**: Linux `blktrace` (OS-level block I/O tracing)

**Captures**: Raw disk reads/writes, request sizes, timestamps, actual SSD bandwidth

### Thread 2: Tensor-Level Tracing
**Location**: [tensor-tracing/](tensor-tracing/) ([llama.cpp fork](https://github.com/ErsjanKeri/llama.cpp))

**Tool**: Custom instrumentation in llama.cpp GGML backend

**Captures**: Tensor operations (ALL ops), memory source (DISK vs BUFFER), timestamps, computation graphs, buffer allocations

**Combined**: Correlate by timestamp to match tensor accesses with disk I/O events

---

## Quick Start

### Tensor Tracing (Thread 2)

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

### Disk I/O Benchmarking (Thread 1)

```bash
cd disk-benchmarking
sudo python3 run_experiment.py
```

Results in `results/experiment_TIMESTAMP/`

See [disk-benchmarking/README.md](disk-benchmarking/README.md) for configuration.

---

## Current Status (2026-01-17)

### Thread 2: Tensor Tracing ✅ INFRASTRUCTURE COMPLETE

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

### Thread 1: Disk Benchmarking ✅

**Working**:
- Automated blktrace experiments
- Memory pressure simulation
- DuckDB analysis pipeline

---

## Key Hypothesis

**CHEOPS paper**: LLM parameters accessed uniformly (randomly)

**My hypothesis**: Dense models accessed **sequentially** (layer-by-layer)

**Validation**: Analyze tensor trace layer_id sequence + correlate with blktrace sector patterns

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

---

## Project Structure

```
BSC/
├── README.md                    ← You are here
├── journal/                     ← Research logs (2025-12 → 2026-01)
├── docs/                        ← Reference docs
├── tensor-tracing/              ← Thread 2
│   ├── run_experiment.py        ← One-command pipeline
│   ├── settings.json            ← Configuration
│   ├── tools/                   ← Parsers
│   └── webui/                   ← Visualization
└── disk-benchmarking/           ← Thread 1
    ├── run_experiment.py        ← Automated blktrace
    └── settings.json            ← Configuration
```

---

## Server

**Hardware**: TUM Server (cli-hiwi-02.dis.cit.tum.de)
- CPU: AMD Ryzen 7 7700X (8c/16t)
- RAM: 30 GiB
- Storage: 2× NVMe (Samsung 980 PRO 1TB, WD 960GB)

See [docs/server-setup.md](docs/server-setup.md) for specs.
