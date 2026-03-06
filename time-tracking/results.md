# MAP_POPULATE Impact on LLM Inference Performance

## Experimental Setup

This study compares inference performance with MAP_POPULATE enabled (prefetch=true) versus disabled (prefetch=false) for:
- Small model: gpt-oss-20b (13GB, fits)
- Large model: gpt-oss-120b (61GB, exceeds)
- Token counts: 10, 100, 200
- 10 runs per configuration

## Column Explanations

| Column | What It Measures | What It Includes |
|--------|------------------|------------------|
| Load Time | Model initialization and weight loading from disk | mmap() + MAP_POPULATE page faults (if enabled) |
| Prompt Eval | Processing input prompt through the model | First forward pass (~12 input tokens) |
| Eval Time | Generating requested output tokens | Autoregressive token generation (10/100/200 tokens) |
| Total Inference | Complete inference time | Prompt Eval + Eval Time + sampling overhead |
| Exp Run Time | True end-to-end time | Load Time + Total Inference |
| Tokens/sec | Token generation throughput | ONLY token generation speed (excludes load + prompt) |

**Important Notes:**
- Exp Run Time = Load Time + Prompt Eval + Eval Time + overhead (complete end-to-end)
- Tokens/sec measures generation speed only, NOT including load or prompt processing
- Speedup percentages compare Exp Run Time (Prefetch ON vs OFF)
- Load speedup shows the improvement in loading phase specifically

## Results: 10 Tokens

### Small Model (20B, 13GB)

| Config | Load Time (ms) | Prompt Eval (ms) | Eval Time (ms) | Total Inf (ms) | Exp Run Time (ms) | Tokens/sec |
|--------|----------------|------------------|----------------|----------------|-------------------|------------|
| Prefetch ON | 6106.0 ± 14.6 | 195.2 ± 0.8 | 590.0 ± 0.7 | 787.5 ± 1.4 | 6893.5 ± 14.5 | 15.25 |
| Prefetch OFF | 5187.0 ± 14.7 | 199.4 ± 3.1 | 616.8 ± 6.9 | 818.6 ± 8.0 | 6005.6 ± 16.1 | 14.59 |
| **Speedup** | **+15.1%** (919ms) | - | - | - | **+12.9%** (888ms) | - |

### Large Model (120B, 61GB)

| Config | Load Time (ms) | Prompt Eval (ms) | Eval Time (ms) | Total Inf (ms) | Exp Run Time (ms) | Tokens/sec |
|--------|----------------|------------------|----------------|----------------|-------------------|------------|
| Prefetch ON | 59194.5 ± 671.4 | 3457.9 ± 235.8 | 1834.0 ± 185.8 | 5324.8 ± 230.3 | 64519.3 ± 821.9 | 4.91 |
| Prefetch OFF | 36843.9 ± 332.5 | 3298.4 ± 236.6 | 1762.0 ± 133.6 | 5092.1 ± 175.4 | 41936.0 ± 325.2 | 5.11 |
| **Speedup** | **+37.8%** (22351ms) | - | - | - | **+35.0%** (22583ms) | - |

## Results: 100 Tokens

### Small Model (20B, 13GB)

| Config | Load Time (ms) | Prompt Eval (ms) | Eval Time (ms) | Total Inf (ms) | Exp Run Time (ms) | Tokens/sec |
|--------|----------------|------------------|----------------|----------------|-------------------|------------|
| Prefetch ON | 6109.1 ± 10.3 | 195.0 ± 1.9 | 6498.4 ± 2.7 | 6711.2 ± 2.6 | 12820.3 ± 10.3 | 15.23 |
| Prefetch OFF | 5203.2 ± 17.8 | 197.5 ± 2.3 | 6688.6 ± 31.1 | 6902.9 ± 31.7 | 12106.2 ± 24.5 | 14.80 |
| **Speedup** | **+14.8%** (906ms) | - | - | - | **+5.6%** (714ms) | - |

### Large Model (120B, 61GB)

| Config | Load Time (ms) | Prompt Eval (ms) | Eval Time (ms) | Total Inf (ms) | Exp Run Time (ms) | Tokens/sec |
|--------|----------------|------------------|----------------|----------------|-------------------|------------|
| Prefetch ON | 60097.9 ± 1386.4 | 3443.3 ± 245.3 | 16685.9 ± 1363.0 | 20185.8 ± 1401.2 | 80283.6 ± 2540.6 | 5.93 |
| Prefetch OFF | 36916.9 ± 226.7 | 3510.5 ± 286.2 | 16026.4 ± 740.0 | 19592.8 ± 753.4 | 56509.7 ± 770.1 | 6.18 |
| **Speedup** | **+38.6%** (23181ms) | - | - | - | **+29.6%** (23774ms) | - |

## Results: 200 Tokens

### Small Model (20B, 13GB)

| Config | Load Time (ms) | Prompt Eval (ms) | Eval Time (ms) | Total Inf (ms) | Exp Run Time (ms) | Tokens/sec |
|--------|----------------|------------------|----------------|----------------|-------------------|------------|
| Prefetch ON | 6121.6 ± 9.5 | 195.3 ± 0.5 | 13128.1 ± 18.3 | 13357.9 ± 16.4 | 19479.5 ± 19.4 | 15.16 |
| Prefetch OFF | 5178.4 ± 8.4 | 198.5 ± 2.3 | 13406.8 ± 55.5 | 13638.6 ± 54.5 | 18817.0 ± 54.6 | 14.84 |
| **Speedup** | **+15.4%** (943ms) | - | - | - | **+3.4%** (663ms) | - |

### Large Model (120B, 61GB)

| Config | Load Time (ms) | Prompt Eval (ms) | Eval Time (ms) | Total Inf (ms) | Exp Run Time (ms) | Tokens/sec |
|--------|----------------|------------------|----------------|----------------|-------------------|------------|
| Prefetch ON | 58580.1 ± 702.2 | 3403.8 ± 225.7 | 30568.5 ± 1194.2 | 34051.7 ± 1199.4 | 92631.9 ± 1527.0 | 6.51 |
| Prefetch OFF | 37133.3 ± 399.6 | 3297.4 ± 229.6 | 30040.8 ± 1519.8 | 33416.9 ± 1531.0 | 70550.2 ± 1565.6 | 6.62 |
| **Speedup** | **+36.6%** (21447ms) | - | - | - | **+23.8%** (22082ms) | - |

## Key Findings

1. **Load Phase Impact:**
   - Small model: ~15% faster loading with MAP_POPULATE=OFF
   - Large model: ~35-40% faster loading (saves 20-23 seconds!)
   - MAP_POPULATE causes blocking when trying to load 61GB into 30GB RAM

2. **Inference Phase (Prompt + Token Generation):**
   - Nearly identical speed between MAP_POPULATE ON/OFF
   - Working set fits in RAM even for large model
   - Tokens/sec comparable: ~15 tok/s (small), ~5-6 tok/s (large)

3. **End-to-End Performance (Exp Run Time):**
   - Small model: 3-13% speedup with MAP_POPULATE=OFF
   - Large model: 24-35% speedup with MAP_POPULATE=OFF
   - Larger models benefit more because load time dominates

4. **Why Speedup Varies by Token Count:**
   - 10 tokens: Load time dominates total time (higher speedup %)
   - 200 tokens: Inference time dominates (lower speedup %)
   - Absolute load time savings constant (~1s small, ~23s large)
   - But as % of total, it matters less with more tokens

5. **Recommendation:**
   - Disable MAP_POPULATE (prefetch=false) when model size exceeds available RAM
   - Provides substantial speedup (20-35%) with no inference penalty
   - Even for models that fit in RAM, modest benefit (3-15%)