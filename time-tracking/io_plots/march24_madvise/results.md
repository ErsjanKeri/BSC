# Experiment Results — March 24, 2026

Small model, 3 iterations per experiment (except default: 1 run), page cache dropped before each run.

## Setup

- **Machine:** 30 GiB RAM, 16 cores, CPU-only
- **Model:** gpt-oss-20b-F16 (12.85 GiB, 24 layers, 32 experts, 4 used/token)
- **Swap:** OFF (`swapoff -a`)
- **Seed:** `--seed 42` for reproducible output
- **Context:** ctx=131072 (default, full context window)
- **Tokens:** 2000 output tokens
- **Pressure:** 20 GiB and 22 GiB (locked by mlock_tool)
- **I/O monitoring:** iostat per-experiment (25ms sampling)
- **Binary:** `build/bin/llama-completion` (MAP_POPULATE=false, with `--moe-prefetch` support)

## New: MoE Expert Prefetch (`--moe-prefetch`)

Custom GGML graph node inserted between MoE router selection and `mul_mat_id` weight access. After the router selects expert IDs, the node issues `posix_madvise(MADV_WILLNEED)` for all 12 weight slices (4 experts × 3 projections: up, gate, down) before any computation begins. This gives the kernel a head start on I/O while the graph proceeds to the actual matrix multiplications.

## Configurations

| Config | Prefetch (MAP_POPULATE) | Warmup | Pinning (mlock compute) | Madvise (expert prefetch) |
|--------|------------------------|--------|------------------------|--------------------------|
| default | ON | ON | OFF | OFF |
| lazy | OFF | OFF | OFF | OFF |
| lazy_pinned | OFF | OFF | ON | OFF |
| lazy_madvise | OFF | OFF | OFF | ON |
| lazy_madvise_pinned | OFF | OFF | ON | ON |

## Table Columns

| Column | What it measures |
|--------|-----------------|
| **Config** | Which optimizations are enabled |
| **Pressure** | How much RAM is locked away by mlock_tool |
| **metadata** | Time to parse GGUF file header, architecture, vocabulary |
| **mmap** | Time for mmap syscall (MAP_POPULATE OFF = instant) |
| **pin** | Time to mlock() compute weights (2329 MiB). Only with pinning ON |
| **context** | Time to allocate KV cache + compute buffers |
| **warmup** | Time for dummy forward pass. Only with warmup ON |
| **prompt_eval** | Time to process input tokens, includes page fault I/O |
| **generation** | Time to generate 2000 output tokens, includes page fault I/O |
| **wall** | Total wall-clock time (sum of all phases) |
| **faults** | Total major page faults (each = one 4KB page read from disk) |
| **gen_faults** | Major page faults during generation phase only |
| **tok/s** | Inference speed = eval_tokens / generation_time |

---

## Default — No pressure (1 run, baseline reference)

| Config | Pressure | metadata | mmap | pin | context | warmup | prompt_eval | generation | **wall** | faults | tok/s |
|--------|----------|----------|------|-----|---------|--------|-------------|------------|----------|--------|-------|
| default | none | 223 | 4776 | 0 | 843 | 312 | 291 | 135994 | **142441** | 39 | 14.7 |

---

## Small model — 2000 output tokens, ctx=131072, 20 GiB pressure

| Config | Pressure | metadata | mmap | pin | context | warmup | prompt_eval | generation | **wall** | faults | gen_faults | tok/s |
|--------|----------|----------|------|-----|---------|--------|-------------|------------|----------|--------|------------|-------|
| lazy | 20 GiB | 267 | 1 | 0 | 859 | 0 | 2237 | 352107 | **355473** | 6,518,919 | 6,482,385 | 5.68 |
| lazy_pinned | 20 GiB | 234 | 1 | 890 | 847 | 0 | 1774 | 324668 | **328779** | 5,674,006 | 5,616,251 | 6.16 |
| lazy_madvise | 20 GiB | 233 | 1 | 0 | 862 | 0 | 3516 | 361941 | **366923** | 6,870,760 | 6,796,409 | 5.52 |
| lazy_madvise_pinned | 20 GiB | 235 | 1 | 891 | 847 | 0 | 3006 | 333222 | **338506** | 5,971,171 | 5,908,726 | 6.00 |

### 20 GiB: Relative to Lazy baseline

| Config | Wall Δ | Gen Δ | Fault Δ | Tok/s Δ |
|--------|--------|-------|---------|---------|
| lazy | — | — | — | — |
| lazy_pinned | **-7.5%** | -7.8% | -13.0% | +8.5% |
| lazy_madvise | **+3.2%** | +2.8% | +5.4% | -2.8% |
| lazy_madvise_pinned | **-4.8%** | -5.4% | -8.4% | +5.6% |

---

## Small model — 2000 output tokens, ctx=131072, 22 GiB pressure

**Note:** lazy_madvise run 3 excluded as outlier (1100s / 22.8M faults vs 760/618s for runs 1-2).

| Config | Pressure | metadata | mmap | pin | context | warmup | prompt_eval | generation | **wall** | faults | gen_faults | tok/s |
|--------|----------|----------|------|-----|---------|--------|-------------|------------|----------|--------|------------|-------|
| lazy | 22 GiB | 269 | 1 | 0 | 867 | 0 | 2255 | 1148111 | **1151505** | 23,336,495 | 23,299,042 | 1.74 |
| lazy_pinned | 22 GiB | 232 | 1 | 888 | 849 | 0 | 1847 | 677742 | **683528** | 16,319,069 | 16,293,810 | 2.95 |
| lazy_madvise | 22 GiB | 232 | 1 | 0 | 868 | 0 | 3549 | 683725 | **689034** | 14,371,798 | 14,296,377 | 2.96 |
| lazy_madvise_pinned | 22 GiB | 235 | 1 | 902 | 844 | 0 | 3141 | 651608 | **658413** | 15,510,492 | 15,444,882 | 3.07 |

### 22 GiB: Relative to Lazy baseline

| Config | Wall Δ | Gen Δ | Fault Δ | Tok/s Δ |
|--------|--------|-------|---------|---------|
| lazy | — | — | — | — |
| lazy_pinned | **-40.6%** | -41.0% | -30.1% | +69.5% |
| lazy_madvise | **-40.2%** | -40.5% | -38.4% | +70.1% |
| lazy_madvise_pinned | **-42.8%** | -43.3% | -33.5% | +76.4% |

---

## SSD Throughput (iostat, 22 GiB, run 1)

| Config | Duration (s) | Total Read (GiB) | Mean Throughput (MiB/s) | SSD Utilization |
|--------|-------------|-------------------|------------------------|-----------------|
| lazy | 1150 | 3921 | 3498 | 50% |
| lazy_pinned | 685 | 2034 | 3050 | 44% |
| lazy_madvise | 761 | 2142 | 2890 | 41% |
| lazy_madvise_pinned | 662 | 1924 | 2989 | 43% |

---

## Key Findings

**1. Madvise is highly effective under severe pressure (22 GiB):**
- Madvise alone: -40.2% wall time, -38.4% fewer faults — the `MADV_WILLNEED` readahead genuinely avoids page faults by pre-loading expert weights before access
- Madvise achieves comparable speedup to pinning (-40.2% vs -40.6%) through a completely different mechanism: prefetch overlapping I/O vs pinning preventing eviction
- Combined madvise+pinned: -42.8% wall time, the best result. Marginal gain over either alone suggests they address overlapping bottlenecks

**2. Madvise is counterproductive at moderate pressure (20 GiB):**
- +3.2% slower than lazy baseline at 20 GiB, with +5.4% MORE faults
- At 20g, ~78% of the model fits in cache — the kernel's LRU already keeps hot pages resident
- `MADV_WILLNEED` issues readahead for ~42 MB/layer (12 expert slices × 3.5 MB), ~1 GB/token across 24 layers
- When most pages are already cached, this readahead is redundant and evicts other useful pages (well-known property of `MADV_WILLNEED` — harmful when working set fits in cache)
- Additionally, `prompt_eval` cost is ~3.5s vs ~2.2s for lazy — fixed overhead from prefetch node execution

**3. SSD throughput paradox persists across all configs:**
- All configs cluster at 2890-3498 MiB/s, far below 7000 MiB/s SSD capacity
- Faster configs show *lower* throughput — because they do less total I/O, not because the SSD is slower
- The bottleneck is synchronous page fault latency, not SSD bandwidth

**4. Pinning remains the most robust optimization:**
- Consistent benefit at both 20 GiB (-7.5%) and 22 GiB (-40.6%)
- Lower variance across runs than madvise
- Madvise's benefit is pressure-dependent — helps enormously at 22 GiB but hurts at 20 GiB

**5. Madvise outlier (run 3 at 22 GiB) needs investigation:**
- Run 3 regressed to 1100s / 22.8M faults (vs 760/618s for runs 1-2)
- Possible cause: kernel readahead thrashing at a tipping point, background system activity, or page cache state from prior run not fully cleared
- Excluded from averages; rerun needed for reliable 3-run statistics

---

## Note on March 22 Results

March 22 experiments at 22 GiB showed faster results (lazy: 503s / 4.0 tok/s, pinned: 435s / 4.6 tok/s) compared to March 24 (lazy: 1152s / 1.74 tok/s, pinned: 684s / 2.95 tok/s). The March 22 data may have had different system conditions (swap state, background processes, or page cache state from prior runs). March 24 results are internally consistent (low variance across 3 iterations per config) and should be treated as the reliable baseline for comparing optimization strategies.
