# Meeting Notes — Key Findings Summary

## Infrastructure Built
- Custom phase timing in llama.cpp (8 phases: metadata, mmap, tensors, pinning, context, warmup, prompt_eval, generation)
- Discovered llama.cpp's timer reset hides all loading costs — our instrumentation bypasses it
- IO monitor: polls /proc/diskstats every 25ms, zero overhead (verified)
- blktrace integration attempted but causes feedback loop under memory pressure (trace writes compete for page cache)

## Key Discoveries

### 1. MAP_POPULATE and Warmup are harmful for MoE models exceeding RAM
- MAP_POPULATE reads entire model upfront — half gets evicted immediately when model > RAM
- Warmup activates ALL 128 experts (not just 4) — by design, to pre-fault pages
- Disabling both ("lazy" config): 100s → 39s wall clock for large model

### 2. llama.cpp's default KV cache wastes 3 GiB of RAM
- Context window default: 131k tokens → 3 GiB KV cache pre-allocated
- For 2000-token generation, only ~48 MiB actually used
- Under memory pressure, this wasted 3 GiB causes 2.7x performance degradation
- Setting `-c 2048` eliminates the waste (but is not the realistic scenario)

### 3. Swap acts as implicit KV cache offloading
- With swap ON: kernel pushes ~4 GiB of KV cache + buffers to swap file
- 24 GiB pressure + swap ON ≈ 20 GiB pressure + swap OFF (verified: 71.9 vs 71.0 ms/tok)
- ALL March 14-15 experiments had swap silently ON — data unreliable for absolute values
- Swap OFF is our standard going forward

### 4. Memory pinning (--pin-compute-weights) works
- Pins attention + output weights (2,329 MiB) via mlock(), skipping expert weights
- At 22 GiB pressure, swap OFF, ctx=131k, 2000 tokens: **37% faster** (1142s → 720s)
- Reduces total SSD reads from 3,912 GiB to 2,216 GiB (eliminates compute weight re-reads)
- Verified mlock'd pages do NOT survive process exit (no cache-drop bias)

### 5. SSD utilization measured via IO monitor
- Setup: 22 GiB pressure, swap OFF, ctx=131k, 2000 tokens, Samsung 980 PRO (7 GiB/s)
- Without pinning: 3,491 MiB/s average (50% utilization), model re-read 304x
- With pinning: 3,123 MiB/s average (45% utilization), model re-read 172x
- Throughput is rock-steady (~3.1 GiB/s) with periodic dips
- **Dips correspond exactly to per-token attention computation** (dip count = token count, interval = ms/token)
- ~50-55% SSD bandwidth unused — headroom for madvise prefetching

### 6. MoE expert computation details (verified in code)
- Router: small matrix multiply (hidden_state × gate_weight) → 32 scores → top-4 selection
- Expert MLP: 3 separate ggml_mul_mat_id calls (gate, up, down projections)
- Loop iterates ALL experts, skips unused ones — page faults happen at weight pointer access
- Each token: ~4 experts × 24 layers = 96 expert weight loads (this is where page faults concentrate)

## Experimental Standard
- 22 GiB memory pressure, swap OFF, ctx=131k, 2000 tokens, --seed 42
- Small model: gpt-oss-20b-F16 (12.85 GiB, 32 experts, 4 used/token)
- Machine: 30 GiB RAM, Samsung 980 PRO NVMe

## Next Steps
1. madvise(MADV_WILLNEED) prefetching after router selects experts
2. mlock2(MLOCK_ONFAULT) for zero-overhead lazy pinning
3. Large model experiments
4. Resolve March 21 vs March 23 performance discrepancy (2.3x difference, same config)
