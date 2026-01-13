# Tensor Tracing Setup

Quick guide for building and running tensor tracing experiments.

---

## Build llama.cpp

```bash
cd llama.cpp

# Clean + configure
rm -rf build
cmake -B build -DGGML_TENSOR_TRACE=ON -DGGML_METAL=OFF

# Build (~5-10 min)
cmake --build build -j16

# Verify
./build/bin/llama-completion --help
```

**Why METAL=OFF?** CPU-only ensures tracing works (instrumentation in ggml-cpu backend).

---

## Run Experiment

### Recommended: Automated Pipeline

```bash
cd tensor-tracing
python3 run_experiment.py
```

Automatically: cleans → runs llama-completion → parses all data → moves to webui

**Configure**: Edit `settings.json` (model, prompt, tokens)

**View**: `cd webui && npm run dev` → http://localhost:5173

### Manual Workflow

For debugging:

```bash
# Clean
rm -f /tmp/tensor_trace.bin /tmp/buffer_stats.jsonl
rm -rf /tmp/graphs

# Run
cd llama.cpp
./build/bin/llama-completion -m model.gguf -p "Hello" -n 3 -no-cnv

# Parse
cd ../BSC
python3 tensor-tracing/tools/parse_trace.py /tmp/tensor_trace.bin \
  --export-json tensor-tracing/webui/public/data/traces/

# Parse graphs (for each token)
for dot in /tmp/graphs/token_*.dot; do
  id=$(basename "$dot" .dot | sed 's/token_//')
  python3 tensor-tracing/tools/parse_dot.py --dot "$dot" \
    --output "tensor-tracing/webui/public/data/graphs/token-${id}.json"
done

# Parse GGUF
llama.cpp/build/bin/llama-gguf-dump model.gguf > /tmp/model.csv
python3 tensor-tracing/tools/parse_csv.py --csv /tmp/model.csv \
  --output tensor-tracing/webui/public/data/memory-map.json

# Parse buffers
python3 tensor-tracing/tools/parse_buffer_stats.py /tmp/buffer_stats.jsonl \
  --output tensor-tracing/webui/public/data/buffer-timeline.json
```

---

## Verify Data

```bash
cd webui/public/data

# Check files exist
ls -lh memory-map.json buffer-timeline.json
ls -lh graphs/ traces/

# Check trace quality
python3 ../../tools/parse_trace.py /tmp/tensor_trace.bin --stats
```

**Expected**:
- memory-map.json (~62 KB, 201 tensors)
- buffer-timeline.json (~737 B)
- graphs/token-*.json (3 files, ~338 KB each)
- traces/token-*.json (3 files, split by token - fixed 2026-01-08)

---

## Known Issues

### 1. Name Truncation
**Problem**: 19-char limit in trace struct

**Example**: `"blk.0.attn_norm.weight"` → `"blk.0.attn_norm.wei"`

**Workaround**: Use address-based correlation (tensor_ptr field)

**Note**: Token ID tracking was fixed on 2026-01-08 - traces now properly split by token

---

## Troubleshooting

**Empty trace file?**
```bash
# Check build
cmake -B build -DGGML_TENSOR_TRACE=ON -DGGML_METAL=OFF
cmake --build build -j16
```

**Parser errors?**
```bash
# Verify format (should be multiple of 256 bytes)
ls -lh /tmp/tensor_trace.bin

# Check format
python3 tools/parse_trace.py /tmp/tensor_trace.bin --verify
```

**WebUI shows no data?**
```bash
# Re-run pipeline
cd tensor-tracing
python3 run_experiment.py
```

---

## Next Steps

1. Build llama.cpp with tracing
2. Run `python3 run_experiment.py`
3. View in WebUI (`npm run dev`)
4. Read [README.md](README.md) for technical details
5. See [journal/2026-01-08.md](../journal/2026-01-08.md) for analysis examples
