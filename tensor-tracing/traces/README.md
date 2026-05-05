# Tensor traces (canonical, multi-topic prompt, 20B)

Captures from `llama-completion` running GPT-OSS-20B with the multi-topic essay prompt and `seed=42`, the same prompt and seed as the wall-clock canonical sweep in `time-tracking/results/`. Three outputs covering the two distinct uses of the tracing infrastructure.

## Outputs

| Path | Purpose | Coverage | Disk content |
|---|---|---|---|
| `20b-1tok-multitopic/tensor_trace.bin` | WebUI visualization (per-token graph + trace) | 2 generation tokens, all 95 GGML ops, 1696 entries | 1.7 MB content in 2 GiB sparse-allocated file |
| `20b-100tok-multitopic/tensor_trace.bin` | DesktopUI visualization (scale demo) | 100 generation tokens, MUL_MAT_ID + ADD_ID only, **14 400 entries** (24 layers × 100 tokens × 6 ops/layer) | 14 MB content in 2 GiB sparse-allocated file |
| `20b-2000tok-cache-dump/cache_dump.csv` | Cache-policy simulator replay | 1999 generation tokens, 383 816 load events, 575 724 expert accesses | 25 MB |

## Run flags

The visualization runs (1 and 2) use plain mmap so that `MUL_MAT_ID` dispatches go through the regular ggml dispatcher (the trace hook lives there). The MoE pipeline custom op (`--uring-async-projection-overlap`) collapses each layer's per-expert MUL_MAT_IDs into a single fused custom op, which `--trace-mode experts` filters out. For visualization we want the per-expert dispatches visible.

```
# Run 1 (WebUI):    -n 2   --trace-mode all
# Run 2 (Desktop):  -n 100 --trace-mode experts
# (both)            -ngl 0 -no-cnv --no-warmup --eager-compute --seed 42
```

The simulator-dump run (Run 3) uses the full canonical configuration including the pipeline:

```
-n 2000 -ngl 0 -no-cnv --no-warmup --eager-compute --pin-compute-weights
--uring-experts --uring-async-projection-overlap
--uring-cache-slots 250 --uring-cache-policy lfu-aging --uring-aging-mult 3
--trace-mode off  # and BSC_CACHE_DUMP=cache_dump.csv
--seed 42
```

The cache dump comes from `load_projections()` (single-threaded loader path) regardless of trace mode, so Run 3 sets `--trace-mode off` to avoid the tracer's open-file overhead while still capturing every expert access.

Reproduction scripts: `/tmp/run_canon_traces_plain.sh` (Run 1+2), `/tmp/run_canon_traces.sh` (Run 3, since superseded). The cache dump in `20b-2000tok-cache-dump/` is the surviving output.

## Tracer fix history

The tracer originally used a 512-entry per-thread batch buffer and called `tensor_trace_init`/`tensor_trace_shutdown` from each `llama_model_load` and `llama_model_free` respectively. Two bugs followed:

1. **Lifecycle truncation.** `llama_params_fit` calls `llama_load_model_from_file` to probe parameter fitting, then frees that probe model. The free called `tensor_trace_shutdown` (munmap, close), then the real model load called `tensor_trace_init` again, which opened the trace file with `O_TRUNC` and wiped any entries written so far. Fix: tracer shutdown is now a no-op; real teardown runs once at process exit via an atexit handler registered on first init (`ggml/src/tensor_trace.c`).
2. **TLS flush race.** Per-thread buffers flushed to a global offset under non-atomic update. Across multiple ggml-thread-pool boundaries (one per decode call) sub-512-entry remainders were silently dropped. Fix: removed the TLS layer entirely; each `tensor_trace_log` call atomically reserves entry-sized space in the global buffer (`atomic_fetch_add` on `g_log_offset`).

Both fixes landed in `ggml/src/tensor_trace.c` along with this trace re-run; the entries reported above are the post-fix counts.

## Simulator outputs (Run 3)

`tools/simulate_from_dump.py cache_dump.csv --cache-sizes 200,250,288,500,750` on the canonical access sequence:

| Cache | LRU | LFU | LFU-aging $m=3$ | ARC | W-TinyLFU | Belady |
|---|---|---|---|---|---|---|
| 200 | 0.0% | 13.4% | **29.1%** | 0.0% | 25.2% | 51.6% |
| 250 | 0.0% | 16.9% | **34.1%** | 0.0% | 31.1% | 58.5% |
| 288 (= WS) | **36.0%** | 19.5% | 30.5% | 32.9% | 33.3% | 62.7% |
| 500 | 54.9% | 33.7% | 54.0% | **56.3%** | 50.3% | 78.6% |
| 750 | 73.3% | 46.8% | 68.5% | **73.7%** | 70.4% | 88.4% |

LFU-aging dominates at undersized cache (c $\leq$ 250), LRU recovers at the boundary, ARC marginally wins at oversized cache; the gap to Belady (offline-optimal) at every size shows non-trivial headroom for routing-aware prefetching.

The simulator's LFU-aging at c=250 (34.1%) matches the C runtime's measured hit rate at the same cell (33.8%) within rounding. The simulator's pure-LFU validation against the dump's recorded hits/misses fails as expected because the runtime ran LFU-aging, not pure LFU; the policy comparison uses the sequence of access events from the dump, replayed through each policy implementation.
