# BSC Thesis: Optimizing SSD-Backed LLM Inference

This repository contains all research, implementation, and documentation for my Bachelor's thesis on optimizing large language model inference when model weights are stored on SSD rather than RAM.

## Problem Statement

**Research Goal**: Understand and optimize SSD-backed LLM inference performance. Current hypothesis: SSD bandwidth may not be fully utilized due to synchronous I/O, lack of predictive prefetching, or suboptimal access patterns.

**Research Questions**:
1. Are model parameters accessed **sequentially** (layer-by-layer) or **uniformly** (randomly)?
2. What is the actual SSD bandwidth utilization during inference?
3. Can we implement optimizations to improve throughput?

**Approach**: Two complementary instrumentation threads to understand the full picture.

---

## Research Architecture: Two Threads

### Thread 1: Disk I/O Benchmarking (OS-level)
**Location**: [disk-benchmarking/](disk-benchmarking/)

**Tool**: Linux `blktrace` to capture block layer I/O operations

**Visibility**:
- Raw disk reads/writes to NVMe device
- Request sizes, timestamps, sectors
- Actual SSD bandwidth utilization

**Purpose**: Measure real-world I/O patterns and bandwidth usage

---

### Thread 2: Tensor-Level Tracing (Application-level)
**Location**: [tensor-tracing/](tensor-tracing/) ([llama.cpp fork](https://github.com/ErsjanKeri/llama.cpp))

**Tool**: Custom instrumentation in `ggml-cpu.c`

**Visibility**:
- Which specific tensors accessed (e.g., "blk.5.attn_q.weight")
- When they're accessed (nanosecond timestamps)
- Layer-by-layer execution order
- Operation types (mul_mat, embeddings, etc.)

**Purpose**: Understand application-level tensor access patterns for correlation with disk I/O

---

### The Power of Correlation

**Combining both threads** enables answering questions neither can alone:

| Question | Thread 1 (blktrace) | Thread 2 (tensor trace) | Combined |
|----------|-------------------|----------------------|----------|
| Which tensor was accessed? | ❌ (just blocks) | ✅ | ✅ |
| Did it hit disk or RAM? | ✅ | ❌ | ✅ |
| Actual bandwidth used? | ✅ | ❌ | ✅ |
| Sequential or random? | ⚠️ (sectors only) | ⚠️ (logic only) | ✅ (match timestamps) |

**Workflow**: Run both traces simultaneously → correlate by timestamp → match tensor accesses to disk I/O events.

---

## Project Structure

```
BSC/
├── README.md                          ← You are here
│
├── journal/                           ← Chronological research logs (first-person)
│   ├── 2025-12-09-findings.md          Initial exploration
│   ├── 2025-12-20.md                   Early disk I/O learning
│   ├── 2025-12-21.md                   Continued exploration
│   ├── 2025-12-22.md                   System understanding
│   ├── 2025-12-30.md                   Pre-implementation planning
│   ├── 2026-01-02.md                   Model loading investigation
│   ├── 2026-01-02-critical-review.md   params_ith analysis
│   └── 2026-01-04.md                   Tensor tracing implementation + first data
│
├── docs/                              ← Reference documentation
│   ├── server-setup.md                 TUM server specs, hardware, memory concepts
│   ├── related-work.md                 Survey of existing tools
│   ├── future-work.md                  Optimization ideas (io_uring, MoE, swappiness)
│   └── experimental-hypotheses.md      Research predictions and test design
│
├── tensor-tracing/                    ← Thread 2: Application-level instrumentation
│   ├── README.md                       Complete technical documentation
│   ├── setup.md                        Quick start guide
│   └── tools/
│       └── parse_trace.py              Binary trace parser and analyzer
│
├── disk-benchmarking/                 ← Thread 1: OS-level block I/O experiments
│   ├── README.md                       blktrace workflow, tools documentation
│   ├── run_experiment.py               Automated experiment runner
│   ├── settings.json                   Experiment configuration
│   └── utils/                          Helper modules and tools
│       ├── setup_tools.py              System setup utilities
│       ├── analysis_tools.py           blktrace parsing
│       └── mlock_tool.cpp              Memory locking for forced SSD usage
│
└── archive/                           ← Historical content (gold extracted)
    ├── PLAN.md                         Original learning plan
    ├── INITIAL_PLAN.md                 Initial experimental design
    ├── IMPLEMENTATION_SUMMARY.md       Superseded by journal/2026-01-04.md
    ├── implementation_plan.md          Planning doc before implementation
    ├── TRACKER_PLAN.md                 Tool survey (→ docs/related-work.md)
    ├── SummarySoFar.md                 Historical learning summary
    ├── interesting.md                  Misc notes (→ docs/)
    └── server_specs.md                 Server info (→ docs/server-setup.md)
```

---

## Quick Start

### For Tensor Tracing (Thread 2)

**1. Clone the instrumented llama.cpp fork**:
```bash
git clone https://github.com/ErsjanKeri/llama.cpp
cd llama.cpp
```

**2. Build with tensor tracing enabled**:
```bash
cmake -B build \
  -DGGML_TENSOR_TRACE=ON \
  -DGGML_METAL=OFF
cmake --build build -j16
```

**3. Extract model structure** (one-time per model):
```bash
./build/bin/gguf-dump /path/to/model.gguf --csv > model_structure.csv
```

**4. Run traced inference**:
```bash
rm -f /tmp/tensor_trace.bin
./build/bin/llama-cli -m /path/to/model.gguf -p "Hello" -n 1
```

**5. Analyze trace**:
```bash
cd ../BSC
python3 tensor-tracing/tools/parse_trace.py /tmp/tensor_trace.bin --display
python3 tensor-tracing/tools/parse_trace.py /tmp/tensor_trace.bin --stats
```

See [tensor-tracing/README.md](tensor-tracing/README.md) for detailed workflow and troubleshooting.

---

### For Disk I/O Benchmarking (Thread 1)

**1. Configure experiment** (edit `settings.json`):
```json
{
  "experiment": {
    "model_file": "gpt-oss-20b-F16.gguf",
    "tokens_to_generate": 100,
    "prompt": "Once upon a time"
  },
  "memory": {
    "mlock_size_gb": 25
  }
}
```

**2. Run automated experiment**:
```bash
cd disk-benchmarking
sudo python3 run_experiment.py
```

The script automatically:
- Compiles mlock_tool.cpp if needed
- Drops page cache
- Starts blktrace on configured block device
- Launches mem_locker to force memory pressure
- Runs llama-cli inference
- Stops all processes and collects data
- Converts blktrace output to CSV
- Analyzes with DuckDB and generates `analysis.json`

Results saved to `results/experiment_TIMESTAMP/`

See [disk-benchmarking/README.md](disk-benchmarking/README.md) for detailed configuration.

---

## Key Research Hypothesis

**CHEOPS paper claims**: LLM parameters are accessed uniformly (randomly)

**My hypothesis**: Dense models (like GPT, LLaMA) are accessed **sequentially** (layer-by-layer during inference)

### Validation Method
Analyze tensor trace layer_id sequence and correlate with blktrace sector access patterns.

### Potential Optimization Strategies

If sequential access is confirmed, three optimization approaches could improve performance:

#### 1. Deterministic Prefetching
**Concept**: Predict which tensors will be accessed next based on layer-by-layer execution order

**Implementation**: Background prefetch thread loads upcoming layer parameters while current layer computes
- If layer N is computing, prefetch layer N+1 parameters from SSD
- Overlap I/O with computation to hide SSD latency
- Requires validation that access pattern is truly sequential

**Expected benefit**: Reduce blocking on SSD reads if I/O can be hidden behind computation

---

#### 2. Asynchronous I/O with io_uring
**Concept**: Replace synchronous I/O with Linux io_uring for true async operations

**Implementation**: Submit multiple I/O requests without blocking
- Batch multiple tensor reads in a single io_uring submission
- Kernel processes requests asynchronously
- Saturate SSD bandwidth with concurrent requests

**Expected benefit**: Increase SSD utilization from potential underutilization to closer to hardware limits

---

#### 3. Application-Level Buffer Manager
**Concept**: Implement custom page cache inside llama.cpp instead of relying on OS

**Implementation**: Virtual memory manager at application layer
- Fine-grained control over which tensors stay in RAM
- Predictive eviction based on known access patterns
- Bypass kernel overhead for known sequential access

**Expected benefit**: More efficient memory management tailored to inference patterns

---

**Status**: All three strategies require validation of access patterns via tensor tracing first. Current work focuses on data collection and pattern analysis.

---

## Server Environment

**Hardware**: TUM Server (cli-hiwi-02.dis.cit.tum.de)
- CPU: AMD Ryzen 7 7700X (8 cores / 16 threads)
- RAM: 30 GiB
- Storage: 2x NVMe
  - `/dev/nvme1n1`: Samsung 980 PRO 1TB (system)
  - `/dev/nvme0n1`: WD 960GB (experimental SSD for benchmarking)

See [docs/server-setup.md](docs/server-setup.md) for complete specifications and memory concepts.
