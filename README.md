# BSC Thesis: Optimizing SSD-Backed LLM Inference

This repository contains all research, implementation, and documentation for my Bachelor's thesis on optimizing large language model inference when model weights are stored on SSD rather than RAM.

## Problem Statement

**Current bottleneck**: llama.cpp achieves only ~10 GB/s throughput when loading model parameters from SSD, despite the server's NVMe drive supporting ~80 GB/s bandwidth (8x underutilization).

**Research questions**:
1. Are model parameters accessed **sequentially** (layer-by-layer) or **uniformly** (randomly)?
2. What causes the bandwidth underutilization?
3. Can we implement predictive prefetching or buffer management to saturate SSD bandwidth?

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

**Limitation**: No application-level semantics (doesn't know which tensor caused the I/O)

**Status**: Tools ready (mlock_tool.cpp), experiments planned after tensor tracing validation

---

### Thread 2: Tensor-Level Tracing (Application-level)
**Location**: [tensor-tracing/](tensor-tracing/) (llama.cpp instrumentation)

**Tool**: Custom instrumentation in `ggml-cpu.c`

**Visibility**:
- Which specific tensors accessed (e.g., "blk.5.attn_q.weight")
- When they're accessed (nanosecond timestamps)
- Layer-by-layer execution order
- Operation types (mul_mat, embeddings, etc.)

**Limitation**: Doesn't see disk I/O (operates at code level, not OS level)

**Status**: ✅ Implementation complete (Jan 4), first data collected, validation in progress

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
│   ├── utils/                          Helper modules for automation
│   │   ├── setup_tools.py
│   │   └── analysis_tools.py
│   └── tools/
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

**Build instrumented llama.cpp**:
```bash
cd ../llama.cpp
cmake -B build -DGGML_TENSOR_TRACE=ON -DGGML_METAL=OFF
cmake --build build -j16
```

**Run traced inference**:
```bash
./build/bin/llama-cli -m /path/to/model.gguf -p "Hello" -n 1
```

**Analyze trace**:
```bash
python BSC/tensor-tracing/tools/parse_trace.py /tmp/tensor_trace.bin --display
```

See [tensor-tracing/README.md](tensor-tracing/README.md) for complete workflow.

---

### For Disk I/O Benchmarking (Thread 1)

**Setup** (requires sudo):
```bash
# Clear page cache
sudo bash -c "echo 1 > /proc/sys/vm/drop_caches"

# Start blktrace
sudo blktrace -d /dev/nvme0n1 -o trace &
```

**Run experiment**:
```bash
# Force memory pressure (100% scenario)
./disk-benchmarking/tools/mlock_tool --size 28G --lock &

# Run inference (will hit SSD)
./llama-cli -m model.gguf -p "Test" -n 100

# Stop trace
sudo killall blktrace
```

**Analyze**:
```bash
blkparse -i trace -o trace.txt
grep -E " (Q|C) " trace.txt  # Queue/Complete events
```

See [disk-benchmarking/README.md](disk-benchmarking/README.md) for complete workflow.

---

## Current Status (as of 2026-01-04)

### Completed ✅
- [x] Custom tensor tracing infrastructure (128-byte binary logging)
- [x] Dual-path tensor correlation (direct names + indexed lookup)
- [x] Python parser for trace analysis
- [x] First real inference data collected (84 trace entries)
- [x] Layer ID extraction working (0-21 for layers)
- [x] Timestamp bug fixed (relative time)
- [x] Documentation restructured (journal/, docs/, clear separation of threads)

### In Progress 🔄
- [ ] Validate tensor tracing data quality
  - Only 10/22 layers logged in first test
  - Only V projection and FFN down logged (missing Q, K, O, gate, up)
  - Need to find and instrument all mul_mat variants
- [ ] Test rebuilt llama.cpp with timestamp fix
- [ ] Validate Path A (tensor_name) vs Path B (tensor_idx) correlation

### Planned ⏳
- [ ] Phase 2: Add tensor registration during model load
- [ ] Instrument other operations (ggml_get_rows, fused kernels)
- [ ] Run first blktrace experiment (baseline: no memory pressure)
- [ ] Run 100% memory pressure scenario
- [ ] Correlate tensor trace + blktrace by timestamp
- [ ] Answer research questions: Sequential vs uniform access?

---

## Key Research Hypothesis

**CHEOPS paper claims**: LLM parameters are accessed uniformly (randomly)

**My hypothesis**: Dense models (like GPT, LLaMA) are accessed **sequentially** (layer-by-layer during inference)

**If true** → Deterministic prefetching is viable
**If false** → Cache optimization more important

**Validation method**: Analyze tensor trace layer_id sequence and correlate with blktrace sector jumps.

---

## Server Environment

**Hardware**: TUM Server (cli-hiwi-02.dis.cit.tum.de)
- CPU: AMD Ryzen 7 7700X (8 cores / 16 threads)
- RAM: 30 GiB
- Storage: 2x NVMe
  - `/dev/nvme1n1`: Samsung 980 PRO 1TB (system)
  - `/dev/nvme0n1`: WD 960GB (experimental SSD, ~80 GB/s target)

See [docs/server-setup.md](docs/server-setup.md) for complete specs.

---

## Timeline

- **Dec 30 - Jan 3**: Tensor tracing infrastructure implementation
- **Jan 4**: First real data collection, documentation restructure
- **Jan 5-10**: Validation, fix missing layers/tensors, Phase 2 registration
- **Jan 11-15**: blktrace experiments, correlation analysis
- **Jan 16-20**: Optimization implementation (if patterns support it)
- **Jan 21-31**: Thesis writing, final experiments

---

## References

### Internal Documentation
- [Journal: 2026-01-04](journal/2026-01-04.md) - Complete implementation history
- [Tensor Tracing README](tensor-tracing/README.md) - Thread 2 workflow
- [Disk Benchmarking README](disk-benchmarking/README.md) - Thread 1 workflow
- [Related Work](docs/related-work.md) - Why existing tools aren't sufficient
- [Future Work](docs/future-work.md) - Optimization strategies

### External Resources
- llama.cpp: https://github.com/ggerganov/llama.cpp
- CHEOPS paper: (parameter access patterns for LLM inference)
- PowerInfer: https://github.com/SJTU-IPADS/PowerInfer (hot/cold neuron sparsity)

---

## Contact

**Author**: Ersi Besi
**Institution**: Technical University of Munich (TUM)
**Thesis Type**: Bachelor's Thesis
**Server**: cli-hiwi-02.dis.cit.tum.de
