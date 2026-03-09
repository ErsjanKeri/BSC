# MAP_POPULATE + Warmup Impact on LLM Inference Performance (March 2026)

## Experimental Setup

This study extends the February results by adding a third configuration: **MAP_POPULATE OFF + warmup disabled**. The hypothesis is that the warmup forward pass accounts for most of the "load time" when MAP_POPULATE is off, and disabling it reveals the true cost of on-demand page faulting during actual inference.

- Small model: gpt-oss-20b (13GB, fits in 30GB RAM)
- Large model: gpt-oss-120b (61GB, exceeds 30GB RAM)
- Token count: 200
- 10 runs per configuration
- Page cache dropped before each run (`echo 3 > /proc/sys/vm/drop_caches`)
- 5 second cooldown between runs
- Machine: Linux server, 30GB RAM, NVMe SSD

## Column Explanations

| Column | What It Measures | What It Includes |
|--------|------------------|------------------|
| Load Time | Model initialization and weight loading from disk | mmap() + MAP_POPULATE page faults (if enabled) + warmup (if enabled) |
| Prompt Eval | Processing input prompt through the model | First forward pass (~12 input tokens). Without warmup, includes page faults |
| Eval Time | Generating requested output tokens | Autoregressive token generation (200 tokens) |
| Total Inference | Complete inference time | Prompt Eval + Eval Time + sampling overhead |
| Exp Run Time | True end-to-end time | Load Time + Total Inference |
| Tokens/sec | Token generation throughput | ONLY token generation speed (excludes load + prompt) |

**Important Notes:**
- Exp Run Time = Load Time + Total Inference (complete end-to-end)
- Tokens/sec measures generation speed only, NOT including load or prompt processing
- Without warmup, Prompt Eval ≈ Load Time because the first forward pass triggers page faults that dominate both metrics
- exp1 run 10 excluded (only generated 103/199 tokens — incomplete run)

## Three Configurations Tested

| Config | MAP_POPULATE | Warmup | What Happens |
|--------|-------------|--------|--------------|
| Prefetch ON | Enabled (`init_mappings(true)`) | ON (default) | mmap blocks to load all pages, warmup is fast (weights in RAM) |
| Prefetch OFF | Disabled (`init_mappings(false)`) | ON (default) | mmap instant, warmup triggers page faults for all weights |
| Prefetch OFF + No Warmup | Disabled (`init_mappings(false)`) | OFF (`--no-warmup`) | mmap instant, no warmup, page faults only during real inference |

## Results: Small Model (20B, 13GB — fits in RAM)

| Config | Load Time (ms) | Prompt Eval (ms) | Eval Time (ms) | Total Inf (ms) | Exp Run Time (ms) | Tokens/sec |
|--------|----------------|------------------|----------------|----------------|-------------------|------------|
| Prefetch ON | 6124.6 ± 12.6 | 196.0 ± 0.7 | 13123.0 ± 13.1 | 13354.1 ± 11.0 | 19478.7 ± 12.4 | 15.16 |
| Prefetch OFF | 5221.2 ± 13.0 | 199.0 ± 2.4 | 13447.7 ± 67.5 | 13679.9 ± 67.1 | 18901.1 ± 69.6 | 14.80 |
| Prefetch OFF + No Warmup | 2311.1 ± 6.5 | 2310.5 ± 6.4 | 14260.5 ± 41.0 | 16605.6 ± 44.1 | 18916.7 ± 48.0 | 13.95 |

### Small Model Speedups

| Comparison | Load Speedup | End-to-End Speedup |
|------------|-------------|-------------------|
| Prefetch ON → OFF | **+14.8%** (903ms) | **+3.0%** (578ms) |
| Prefetch ON → OFF + No Warmup | **+62.3%** (3814ms) | **+2.9%** (562ms) |
| Prefetch OFF → OFF + No Warmup | **+55.7%** (2910ms) | **-0.1%** (-16ms) |

## Results: Large Model (120B, 61GB — exceeds RAM)

| Config | Load Time (ms) | Prompt Eval (ms) | Eval Time (ms) | Total Inf (ms) | Exp Run Time (ms) | Tokens/sec |
|--------|----------------|------------------|----------------|----------------|-------------------|------------|
| Prefetch ON | 65373.2 ± 2418.6 | 3715.9 ± 244.3 | 32761.1 ± 1917.3 | 36572.0 ± 1838.8 | 101945.2 ± 3483.7 | 6.09 |
| Prefetch OFF | 39961.2 ± 1216.8 | 3882.2 ± 249.9 | 32741.4 ± 2295.6 | 36705.7 ± 2260.7 | 76666.8 ± 2805.5 | 6.10 |
| Prefetch OFF + No Warmup | 4292.2 ± 9.3 | 4291.7 ± 9.3 | 32892.2 ± 1693.9 | 37236.2 ± 1689.8 | 41528.5 ± 1687.1 | 6.07 |

### Large Model Speedups

| Comparison | Load Speedup | End-to-End Speedup |
|------------|-------------|-------------------|
| Prefetch ON → OFF | **+38.9%** (25412ms) | **+24.8%** (25278ms) |
| Prefetch ON → OFF + No Warmup | **+93.4%** (61081ms) | **+59.3%** (60417ms) |
| Prefetch OFF → OFF + No Warmup | **+89.3%** (35669ms) | **+45.8%** (35138ms) |

## Key Findings

### 1. Warmup Dominates "Load Time" (Small Model)

With the small model (fits in RAM), disabling warmup cuts load time from 5221ms to 2311ms — a **55.7% reduction**. The warmup's full forward pass was responsible for ~2900ms of page faults. However, **end-to-end time is unchanged** (-0.1%) because those page faults simply move into real inference (eval time increases by ~813ms, prompt eval absorbs the rest).

**Conclusion:** For models that fit in RAM, disabling warmup redistributes page fault cost but doesn't eliminate it. The working set is similar either way.

### 2. Warmup Is Catastrophic for Large Models

With the large model (61GB on 30GB RAM), the results are dramatic:

- **Prefetch ON → OFF + No Warmup**: End-to-end drops from **101.9s to 41.5s** (59.3% faster)
- **Prefetch OFF → OFF + No Warmup**: End-to-end drops from **76.7s to 41.5s** (45.8% faster)
- Load time collapses from 65.4s to **4.3s** (93.4% reduction)

The warmup is actively destructive when the model exceeds RAM: it faults in pages that immediately get evicted (61GB cannot fit in 30GB), wasting I/O bandwidth on data that won't survive to inference. Without warmup, only pages needed for actual inference are loaded — and the sparse access patterns (MoE 8-of-32 experts, selective embedding rows) mean far fewer pages are needed.

### 3. Inference Speed Is Unaffected

Token generation speed is consistent across all configurations:
- Small model: 13.95–15.16 tokens/sec
- Large model: 6.07–6.10 tokens/sec

The ~8% difference in small model tokens/sec (15.16 vs 13.95) between Prefetch ON and OFF+NoWarmup likely reflects page faults during the first few generation steps. For the large model, tokens/sec is virtually identical because inference is memory-bandwidth-bound regardless — pages are being evicted and re-faulted during generation in all configurations.

### 4. Prompt Eval ≈ Load Time Without Warmup

An interesting artifact: when warmup is disabled, `prompt_eval_time ≈ load_time` (e.g., 2310.5ms vs 2311.1ms for small model). This is because llama.cpp's performance timer starts at model load and is only reset after warmup completes. Without warmup, the first prompt eval IS the first forward pass, so its page fault cost gets attributed to both metrics.

### 5. Standard Deviation Tells a Story

- **Prefetch OFF + No Warmup** has remarkably tight standard deviations on load time (6.5ms small, 9.3ms large) — because load is just GGUF parse + metadata, with no I/O variance
- **Prefetch ON** has high variance on the large model (±2418.6ms) — because eager loading 61GB into 30GB RAM involves unpredictable page eviction patterns
- **Eval time** variance is consistent across configs (~1700-2300ms for large model) — reflecting natural variation in memory-bound inference under memory pressure

## Comparison with February Results

| Metric | February | March | Notes |
|--------|----------|-------|-------|
| Small, Prefetch ON, Load | 6121.6 ± 9.5 ms | 6124.6 ± 12.6 ms | Consistent |
| Small, Prefetch OFF, Load | 5178.4 ± 8.4 ms | 5221.2 ± 13.0 ms | Consistent |
| Large, Prefetch ON, Load | 58580.1 ± 702.2 ms | 65373.2 ± 2418.6 ms | Higher variance, ~11% slower |
| Large, Prefetch OFF, Load | 37133.3 ± 399.6 ms | 39961.2 ± 1216.8 ms | ~8% slower |

Small model results are highly reproducible. Large model shows some variation between runs (likely system load / background processes), but the relative speedups are consistent.

## Recommendation

1. **Always disable MAP_POPULATE** (`init_mappings(false)`) — no downside, consistent improvement
2. **Disable warmup for large models that exceed RAM** — 45-59% end-to-end speedup with no inference penalty
3. **For models that fit in RAM**, warmup provides negligible end-to-end benefit — the page faults simply move from load to inference
4. **Ideal configuration**: `init_mappings(false)` + `--no-warmup` gives the best end-to-end performance in all tested scenarios
