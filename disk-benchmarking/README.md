# Disk I/O Benchmarking (Thread 1)

This directory contains tools and documentation for **OS-level block I/O tracing** experiments.

## Overview

This is **Thread 1** of the thesis research, focusing on understanding disk I/O patterns at the block device level when running LLM inference with SSD-backed model storage.

**Complementary to Thread 2 (Tensor Tracing)**:
- Thread 1 (this): OS-level block I/O via `blktrace` - sees raw read/write operations to NVMe device
- Thread 2 (tensor-tracing/): Application-level tensor access instrumentation - sees which tensors are accessed

**Combined view**: Correlate tensor accesses with actual disk I/O to understand memory hierarchy behavior.

---

## What is blktrace?

`blktrace` is a Linux kernel tool that traces block layer I/O operations.

**What it captures**:
- Every read/write request to a block device (e.g., `/dev/nvme0n1`)
- Request size (in 512-byte sectors)
- Timestamp (nanosecond precision)
- I/O queue operations (Q = queued, D = dispatched, C = completed)
- Physical disk offsets

**What it does NOT capture**:
- Which tensor caused the I/O (application-level semantic)
- Logical file structure (sees blocks, not files)
- Memory access patterns (sees disk I/O, not RAM accesses)

**Use case**: Measure actual SSD bandwidth utilization, I/O pattern (sequential vs random), request sizes.

---

## Distinction: blktrace vs Tensor Tracing

| Aspect | blktrace (Thread 1) | Tensor Tracing (Thread 2) |
|--------|-------------------|-------------------------|
| **Level** | OS kernel (block layer) | Application (ggml code) |
| **Visibility** | Disk I/O requests | Tensor accesses |
| **Granularity** | 512-byte sectors | Individual tensors |
| **Semantics** | None (just blocks) | Full (tensor names, layers) |
| **When triggered** | Page fault → disk read | Every mul_mat operation |
| **Tools** | `blktrace`, `blkparse` | Custom instrumentation |

**Example correlation**:
- Tensor trace: "Accessed blk.5.attn_q.weight at T=0.045s"
- blktrace: "Read 128 KB from sector 123456 at T=0.045s"
- **Inference**: Accessing that tensor caused a 128 KB disk read

---

## Tools in This Directory

### run_experiment.py

**Purpose**: Fully automated experiment runner for blktrace benchmarking

**What it does**:
- Loads configuration from `settings.json`
- Compiles `mlock_tool.cpp` if needed
- Drops page cache to ensure clean state
- Starts blktrace on configured block device
- Launches mem_locker to force memory pressure
- Runs llama-cli inference
- Captures memory snapshots before/after
- Stops all processes and collects data
- Converts blktrace binary output to CSV
- Analyzes with DuckDB (sequential vs random, bandwidth, etc.)
- Generates comprehensive `analysis.json`

**Usage**:
```bash
# 1. Edit settings.json to configure experiment
# 2. Run (no command-line arguments needed)
sudo python3 run_experiment.py
```

**Configuration**: All parameters in `settings.json` (see below)

**Output**: Results saved to `results/experiment_TIMESTAMP/`
- `config.json` - Experiment configuration
- `performance.json` - Timing and throughput metrics
- `blktrace/` - Raw blktrace data
- `blktrace.csv` - Parsed I/O events
- `analysis.json` - DuckDB analysis (sequential%, bandwidth, gaps)
- `memory_before.txt` / `memory_after.txt` - RAM snapshots
- `inference.log` - llama-cli output

### settings.json

**Purpose**: Experiment configuration file (JSON format)

**Structure**:
```json
{
  "experiment": {
    "model_file": "gpt-oss-20b-F16.gguf",    // Model filename (in models_dir)
    "tokens_to_generate": 100,                // How many tokens to generate
    "prompt": "Once upon a time"              // Prompt text
  },
  "memory": {
    "mlock_size_gb": 25                       // GB to lock (forces model to SSD)
  },
  "storage": {
    "block_device": "/dev/nvme0n1",           // Device to trace
    "blktrace_staging": "/tmp/blktrace_staging" // Temp dir for blktrace
  },
  "analysis": {
    "gap_small_sectors": 256,                 // Gap threshold for sequential
    "gap_medium_sectors": 2048                // Gap threshold for random
  },
  "paths": {
    "llama_cli": "llama.cpp/build/bin/llama-cli",
    "models_dir": "/mnt/experiment_ssd",      // Where models are stored
    "mlock_tool_cpp": "utils/mlock_tool.cpp",
    "mlock_bin": "mem_locker",
    "results_dir": "results"
  }
}
```

**Configuration tips**:
- `mlock_size_gb`: Set to force memory pressure (e.g., 25GB on 30GB system = 83% pressure)
- `model_file`: Must exist in `models_dir`
- `block_device`: The SSD where model is stored (use `lsblk` to verify)

### utils/

**Purpose**: Helper modules and tools for experiment automation

**Contents**:
- `setup_tools.py`: System setup and validation
- `analysis_tools.py`: blktrace output parsing and analysis
- `mlock_tool.cpp`: Memory locking tool (see below)

### utils/mlock_tool.cpp

**Purpose**: Control which parts of the model stay in RAM vs get swapped to disk.

**Why it's needed**:
- By default, OS might keep entire model in RAM (no disk I/O to measure)
- We need to force memory pressure to trigger SSD reads
- `mlock()` pins specific memory regions in RAM, preventing them from being swapped
- `munlock()` allows regions to be swapped out

**Typical usage**:
```bash
# Compile first (done automatically by run_experiment.py)
g++ utils/mlock_tool.cpp -o mem_locker

# Allocate 20 GB, lock it in RAM to force model to SSD
./mem_locker 20

# In another terminal: Run inference (will hit SSD)
./llama-cli -m model.gguf -p "Hello"

# Result: blktrace captures disk reads as model weights load from SSD
```

**Status**: Implemented, ready for experiments.

---

## Experimental Design

### Goal
Measure SSD I/O patterns when running inference on models that exceed available RAM.

### Variables

1. **Model size**:
   - Small model (~8 GB) - fits in RAM
   - Large model (~20 GB) - requires SSD offloading

2. **Memory pressure scenarios**:
   - `0percent`: All RAM available (baseline, no SSD)
   - `50percent`: Lock half of RAM
   - `100percent`: Lock all available RAM (maximum SSD usage)

3. **Swappiness** (optional, see [future-work.md](../docs/future-work.md)):
   - `0`: Conservative (swap only when necessary)
   - `60`: Default (balanced)
   - `100`: Aggressive (proactive swapping)

### Workflow

```bash
# 1. Clear page cache
sudo bash -c "echo 1 > /proc/sys/vm/drop_caches"

# 2. Set swappiness (optional)
sudo sysctl vm.swappiness=100

# 3. Start blktrace on experimental SSD
sudo blktrace -d /dev/nvme0n1 -o trace &

# 4. Lock memory to force SSD usage (if 100percent scenario)
./mlock_tool --size 28G --lock &

# 5. Run inference
./llama-cli -m /path/to/model.gguf -p "Test prompt" -n 100

# 6. Stop blktrace
sudo killall blktrace

# 7. Parse trace
blkparse -i trace -o trace.txt

# 8. Analyze
grep -E " (Q|C) " trace.txt  # Filter queue/complete events
```

---

## Expected Outputs

### blktrace Output Format
```
259,0    3        1     0.000000000  1234  Q  R 123456 + 256 [llama-cli]
259,0    3        2     0.000023456  1234  D  R 123456 + 256 [llama-cli]
259,0    3        3     0.000456789  1234  C  R 123456 + 256 [0]
```

**Fields**:
- `259,0`: Device (major, minor)
- `Q/D/C`: Queue/Dispatch/Complete
- `R`: Read operation
- `123456`: Starting sector
- `+ 256`: Request size (256 sectors = 128 KB)
- `[llama-cli]`: Process name

### Metrics to Extract

1. **Total bytes read from SSD**
2. **I/O bandwidth** (bytes/sec)
3. **Request size distribution** (histogram of block sizes)
4. **Sequential vs random ratio**:
   - Sequential: Consecutive sector numbers
   - Random: Large jumps in sector numbers
5. **I/O queue depth** (number of concurrent requests)

---

## Current Status

**Implementation**: ✅ Complete
- `mlock_tool.cpp` is ready
- blktrace workflow is documented

**Execution**: ⏳ Planned
- Waiting for tensor tracing validation (Thread 2)
- Will run experiments after correlating tensor access patterns

**Reason for delay**:
- Need to understand tensor access patterns first (Thread 2)
- Then correlate with block I/O traces
- This ensures we know WHICH tensors cause WHICH disk reads

---

## Integration with Tensor Tracing

### Correlation Workflow

1. **Run inference with BOTH traces enabled**:
   ```bash
   # Terminal 1: blktrace
   sudo blktrace -d /dev/nvme0n1 -o blk_trace

   # Terminal 2: Run instrumented llama.cpp
   GGML_TENSOR_TRACE=1 ./llama-cli -m model.gguf -p "Test"

   # Outputs:
   # - /tmp/tensor_trace.bin (tensor accesses)
   # - blk_trace.bin (disk I/O)
   ```

2. **Parse both traces**:
   ```bash
   python parse_trace.py /tmp/tensor_trace.bin > tensor.csv
   blkparse -i blk_trace -o blk.txt
   ```

3. **Correlate by timestamp**:
   - Match tensor access timestamp with blktrace timestamp
   - Identify which tensor access caused which disk read
   - Validate: Does accessing `blk.5.attn_q.weight` trigger 128 KB read?

4. **Analysis**:
   - Do all layers get loaded sequentially?
   - Are there repeated disk reads for the same tensor (cache misses)?
   - What is actual SSD bandwidth utilization?

---

## Research Questions

1. **What is the actual SSD bandwidth utilization during inference?**
   - SSD spec: ~3.5 GB/s sequential read (PCIe Gen 3.0 x4)
   - Question: Is llama.cpp achieving full bandwidth or underutilizing?
   - Measurement: blktrace will show actual MB/s throughput

2. **Are requests sequential or random?**
   - Hypothesis: Dense models access sequentially (layer-by-layer)
   - CHEOPS paper claims: Uniform access
   - Measurement: Analyze sector jump distances in blktrace

3. **What is the request size distribution?**
   - Hypothesis: llama.cpp issues small synchronous reads
   - Optimal: Large batched async reads (128 KB+ per request)
   - Measurement: Histogram of request sizes from blktrace

4. **Can we predict upcoming I/O?**
   - If access is sequential → prefetching is viable
   - If access is random → cache optimization more important
   - Measurement: Cross-correlate tensor trace with blktrace

---

## Next Steps

1. ✅ Complete tensor tracing implementation (Thread 2)
2. ⏳ Validate tensor trace data quality
3. ⏳ Run first blktrace experiment (baseline: no memory pressure)
4. ⏳ Run 100percent scenario (maximum SSD usage)
5. ⏳ Correlate traces by timestamp
6. ⏳ Analyze findings and answer research questions

---

## References

- [TUM Server Specs](../docs/server-setup.md) - Hardware details
- [Future Optimizations](../docs/future-work.md) - io_uring, prefetching ideas
- [Tensor Tracing](../tensor-tracing/README.md) - Thread 2 (application-level tracing)
