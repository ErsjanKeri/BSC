# Tensor-Level Tracing (Thread 2)

Application-level instrumentation for tracking tensor operations during LLM inference.

## What It Provides

- Tensor names & timestamps (ns precision)
- **ALL operation types** (95 ggml ops: MUL_MAT, ADD, ROPE, RMS_NORM, etc.)
- **Memory source detection** (DISK-backed GGUF vs BUFFER allocations)
- **Multi-source tracking** (up to 4 source tensors per operation)
- Layer-by-layer execution order
- Computation graphs (per token)
- Buffer allocation timeline

---

## Quick Start

### Build llama.cpp

```bash
cd llama.cpp
cmake -B build -DGGML_TENSOR_TRACE=ON -DGGML_METAL=OFF
cmake --build build -j16
```

### Run Experiment

```bash
cd tensor-tracing
python3 run_experiment.py
```

Automatically: cleans → runs llama-completion → parses → moves to webui/public/data/

**Configuration**: Edit `settings.json`

**View results**: `cd webui && npm run dev` → http://localhost:5173

---

## Trace Format (256-byte)

**Location**: `/tmp/tensor_trace.bin`

**Structure**:
```c
struct TensorTraceEntry {
    // Header (32 bytes)
    uint64_t timestamp_ns;
    uint32_t token_id;
    uint16_t layer_id;           // 0-21 or NULL
    uint16_t thread_id;
    uint8_t  phase;              // PROMPT/GENERATE
    uint8_t  operation_type;     // ggml_op enum
    uint8_t  num_sources;        // 1-4
    char     dst_name[20];       // 19 chars + null

    // Sources (56 bytes × 4 = 224 bytes)
    struct Source {
        char     name[20];
        uint64_t tensor_ptr;     // Correlation key
        uint32_t size_bytes;
        uint16_t layer_id;
        uint8_t  memory_source;  // DISK=0, BUFFER=1
        uint64_t disk_offset_or_buffer_id;
    } sources[4];
} __attribute__((packed));  // Total: 256 bytes
```

**Parsing**: `tools/parse_trace.py --export-json webui/public/data/traces/`

---

## Memory Source Detection

### DISK (memory_source = 0)
- Model weights from GGUF file
- Examples: `blk.0.attn_q.weight`, `token_embd.weight`
- Storage: `disk_offset` = byte offset in GGUF

### BUFFER (memory_source = 1)
- Runtime allocations (KV cache, intermediate tensors)
- Examples: `kv_cache_k`, `Qcur-0`
- Storage: `buffer_id` = unique buffer identifier

---

## Generated Data

After `run_experiment.py`:

```
webui/public/data/
├── memory-map.json           # GGUF structure (201 tensors)
├── buffer-timeline.json      # Buffer alloc/dealloc events
├── graphs/                   # Computation graphs (per token)
│   ├── token-00000.json
│   ├── token-00001.json
│   └── token-00002.json
└── traces/                   # Trace entries (per token)
    └── token-00000.json      # Currently all grouped here
```

**Known**: All traces in token-00000.json (C code doesn't increment token_id)

---

## Tools

### run_experiment.py
**Usage**: `python3 run_experiment.py`

**Pipeline**: verify → clean → run llama-completion → parse (GGUF, trace, graphs, buffers) → move to webui

**Config**: `settings.json`

### parse_trace.py
```bash
# Stats
python3 tools/parse_trace.py /tmp/tensor_trace.bin --stats

# Export to JSON
python3 tools/parse_trace.py /tmp/tensor_trace.bin \
  --export-json webui/public/data/traces/
```

### parse_dot.py
```bash
python3 tools/parse_dot.py \
  --dot /tmp/graphs/token_00001.dot \
  --output webui/public/data/graphs/token-00001.json
```

### parse_csv.py
```bash
# GGUF → memory map
llama.cpp/build/bin/llama-gguf-dump model.gguf > model.csv
python3 tools/parse_csv.py --csv model.csv --output webui/public/data/memory-map.json
```

### parse_buffer_stats.py
```bash
python3 tools/parse_buffer_stats.py /tmp/buffer_stats.jsonl \
  --output webui/public/data/buffer-timeline.json
```

---

## Status (2026-01-08)

### ✅ Working
- 256-byte format with multi-source tracking
- Memory source detection (DISK vs BUFFER)
- All 95 ggml operations captured
- Buffer allocation tracking
- Computation graph dumping
- Automated pipeline
- WebUI (basic 4-view layout)
- ✅ Token ID tracking: properly increments (0, 1, 2, ...) - fixed 2026-01-08
- ✅ Phase tracking: distinguishes PROMPT vs GENERATE - fixed 2026-01-08

### ⚠️ Known Issues
1. **Name truncation**: 19-char limit in struct

### 🔄 In Development
- WebUI: Full-screen views, heatmap improvements, timeline playback

---

## Research Questions

1. **Sequential vs uniform access?** → Analyze layer_id sequence
2. **Hot vs cold memory?** → Access counts per tensor
3. **Memory hierarchy behavior?** → DISK vs BUFFER patterns + correlate with blktrace
4. **Predictable patterns?** → Temporal sequence for prefetching
5. **Buffer allocation pattern?** → Memory pressure points

---

## Performance

**Overhead**: < 1% on inference time

**Why**: Logging only during op execution, memory-mapped I/O, minimal string ops, efficient binary format

**Footprint**: ~300 KB trace + ~10 KB buffers + ~1 KB thread-local = < 1 MB total

---

## Integration with Thread 1

Correlate by timestamp:
- Tensor trace: "Accessed blk.5.attn_q.weight at T=0.045s"
- blktrace: "Read 128 KB from sector 123456 at T=0.045s"
- **Result**: Which tensor accesses cause disk I/O

---

## File Locations

**llama.cpp fork**:
- Header: `llama.cpp/ggml/include/tensor_trace.h`
- Implementation: `llama.cpp/ggml/src/tensor_trace.c`
- Callback: `llama.cpp/ggml/src/ggml-backend.c`

**Parsers**: `tools/parse_*.py`

**Pipeline**: `run_experiment.py`, `settings.json`

**Output**: `webui/public/data/`

---

## Next Steps

**Immediate**: Fix token_id tracking in C code

**Short-term**: Validate sequential access hypothesis, correlate with blktrace

**Medium-term**: WebUI improvements, deterministic prefetcher prototype

---

## Documentation

- [Main README](../README.md) - Project overview
- [setup.md](setup.md) - Build & run guide
- [Journal 2026-01-07](../journal/2026-01-07.md) - 256-byte format
- [Journal 2026-01-08](../journal/2026-01-08.md) - Automated pipeline
- [webui/README.md](webui/README.md) - Visualization tool
