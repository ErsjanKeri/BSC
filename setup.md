# Tensor Tracing Setup - Quick Reference

## Prerequisites

- TinyLlama model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` (already in `llama.cpp/`)
- Modified llama.cpp with tensor tracing code:
  - `ggml/include/tensor_trace.h`
  - `ggml/src/tensor_trace.c`
  - `ggml/src/ggml-cpu/ggml-cpu.c` (mul_mat logging added)
  - `src/llama.cpp` (tensor_trace_init call added)
  - `src/llama-model.cpp` (tensor_trace_shutdown call added)

## Build Commands

```bash
cd llama.cpp

# Clean previous build
rm -rf build

# Configure with tensor tracing enabled and Metal disabled (CPU-only)
cmake -B build \
  -DGGML_TENSOR_TRACE=ON \
  -DGGML_METAL=OFF

# Build (takes ~5-10 minutes)
cmake --build build -j
```

## Extract Model Structure

```bash
cd llama.cpp

# Extract tensor metadata to CSV
./build/bin/llama-gguf-dump tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf > tinyllama_structure.csv
```

Output: `tinyllama_structure.csv` with 201 tensors (offsets, sizes, layer IDs, component types)

## Run Traced Inference

```bash
cd llama.cpp

# Clear old trace file
rm -f /tmp/tensor_trace.bin

# Run inference (CPU-only, no conversation mode)
./build/bin/llama-completion \
  -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Hello" \
  -n 5 \
  -no-cnv
```

Expected output:
```
[TENSOR_TRACE] Initialized: /tmp/tensor_trace.bin (2.00 GB capacity)
...model loads...
...generates text...
[TENSOR_TRACE] Shutdown: 252 entries logged (0.02 MB)
```

## Analyze Trace Data with Python Parser

### Step 1: Quick Verification
```bash
# Check trace file exists and size
ls -lh /tmp/tensor_trace.bin
# Expected: ~32 KB (252 entries × 128 bytes)

# View first entries with hexdump (should see tensor names in ASCII)
hexdump -C /tmp/tensor_trace.bin | head -40
# Look for readable tensor names like "blk.0.attn_q.weight"
```

### Step 2: Parse and Display First 20 Entries
```bash
cd /Users/ersibesi/Desktop/LLAMA

# Show first 20 trace entries in human-readable format
python3 BSC/parse_trace.py --limit 20
```

Expected output:
```
Parsed 252 trace entries

   # Time(ms)  Tok Lay      Op     Size  TIdx Tensor Name
---------------------------------------------------------------------------
   0       0.00    0 N/A MUL_MAT 250.0MB     0 token_embd.weight
   1       0.01    0 N/A MUL_MAT   0.5MB   N/A <anonymous>
   2       0.05    0   0 MUL_MAT   8.0KB     2 blk.0.attn_norm.weight
   3       0.06    0   0 MUL_MAT   2.0MB     7 blk.0.attn_k.weight
   4       0.10    0   0 MUL_MAT  16.0MB     9 blk.0.attn_q.weight
   ...
```

### Step 3: Validate Path A vs Path B Correlation
```bash
# Verify tensor_name (Path A) matches tensor_idx lookup (Path B)
python3 BSC/parse_trace.py --validate
```

Expected output:
```
=== Validating Path A vs Path B ===

Validated 252 entries
Unique tensor indices: 201

✅ All entries match! Path A and Path B are consistent.
```

### Step 4: Show Statistics
```bash
# Analyze layer distribution and coverage
python3 BSC/parse_trace.py --stats
```

Expected output:
```
=== Trace Statistics ===

Total entries: 252

Entries per layer:
  Layer N/A:  126 entries
  Layer   0:    6 entries
  Layer   1:    6 entries
  ...
  Layer  21:    6 entries

Entries per operation:
  MUL_MAT: 252 entries

Total data accessed: 6.2GB
Average access size: 25.2MB

Unique tensors accessed: 201
```

### Step 5: Correlate with CSV Structure
```bash
# Match trace entries with GGUF model structure
python3 BSC/parse_trace.py --limit 10 --csv llama.cpp/tinyllama_structure.csv
```

Expected output shows trace entry + CSV data side-by-side for correlation.

### Step 6: Filter by Layer
```bash
# Show only layer 5 accesses
python3 BSC/parse_trace.py --layer 5
```

Expected: 6 entries for layer 5 (attn_norm, attn_k, attn_q, attn_v, attn_output, ffn components)

## Trace File Format

Each entry is **128 bytes** with structure:
- `timestamp_ns` (8 bytes): Nanoseconds since trace start
- `token_id` (4 bytes): Which token being processed
- `layer_id` (2 bytes): Transformer layer (0-21 for TinyLlama, 65535=N/A)
- `thread_id` (2 bytes): CPU thread ID
- `operation_type` (1 byte): 1 = MUL_MAT
- `phase` (1 byte): 0 = PROMPT, 1 = GENERATE
- `tensor_idx` (4 bytes): Index into registry (Path B - efficient lookup)
- `tensor_ptr` (8 bytes): Memory address of tensor data
- `file_offset` (8 bytes): Offset in GGUF file (0 for intermediates)
- `size_bytes` (4 bytes): Tensor size in bytes
- `tensor_name` (64 bytes): Tensor name string (Path A - validation)
- Other metadata fields (attention, MoE, padding)

## Key Notes

- **Dual-path implementation**:
  - **Path A**: Direct tensor names logged for validation (64 bytes per entry)
  - **Path B**: Efficient tensor_idx lookup via registry (4 bytes per entry)
  - Both paths validate each other for 100% accuracy
- **CPU-only**: Metal (GPU) disabled to ensure tracing works on ggml-cpu backend
- **252 entries**: Each mul_mat logs 2 entries (src0=weights, src1=activations)
- **Entry size**: 128 bytes (was 64 bytes before adding tensor_name field)
- **~6 tokens**: 2 prompt tokens + 4-5 generated tokens
- **22 layers**: TinyLlama has 22 transformer layers (layer_id: 0-21)
- **Non-layer tensors**: Embeddings and output use layer_id = 65535 (N/A)
- **Trace location**: `/tmp/tensor_trace.bin` (~32 KB for 252 entries)
- **Model structure**: `tinyllama_structure.csv` (201 tensors from GGUF dump)
- **Parser tool**: `BSC/parse_trace.py` for human-readable analysis

## Validation Checklist

After running inference, verify:

- [ ] Trace file size is ~32 KB (252 entries × 128 bytes)
- [ ] Tensor names appear correctly in parsed output
- [ ] Layer IDs extracted properly:
  - `blk.0.*` through `blk.21.*` → layer_id 0-21
  - `token_embd.weight`, `output.weight` → layer_id 65535
- [ ] Path A vs Path B validation passes (✅ message)
- [ ] All 201 unique tensors accessed during inference
- [ ] Entries correlate with CSV by tensor_name
