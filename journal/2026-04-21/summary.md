# GPT-OSS-120B on 30 GiB RAM — April 21

All numbers: GPT-OSS-120B-F16 (60.88 GiB), cgroup v2 `memory.max`, prompt "Write a detailed essay about the history of computer science", 2000 tokens, seed 42, 3 iters (lazy/pin from earlier same-week sweeps, overlap/v3 from tonight's bench).

## Architecture

| | Value |
|---|---|
| Layers | 36 |
| Hidden dim (n_embd) | 2880 |
| FFN dim (n_ff) | 2880 |
| KV heads | 8 · head_dim 45 |
| Experts / layer | 128 |
| Experts used / token | 4 |
| Per-token working set | **432 slots** = 4 · 3 · 36 |

## Memory budget

At `memory.max = N GiB`, every byte lives in exactly one of the buckets below. All values measured in tonight's runs via `[BSC_MEM]` dumps + cgroup `memory.current`.

| Bucket | Size | Notes |
|---|---|---|
| Pin (mlocked attention+output head, 470 tensors) | **2980 MiB** | `--pin-compute-weights` |
| KV cache (iSWA: 18 layers @ ctx 131072 + 18 @ ctx 768) | **4635 MiB** | 4608 non-SWA + 27 SWA |
| Compute buffer (eager-committed) | **413 MiB** | `--eager-compute` memset |
| Misc anon (threads, libs, io_uring rings) | **~100 MiB** | measured from RssAnon baseline |
| **Fixed non-reclaimable** | **~8100 MiB** | |
| Slot size (per expert-projection, MXFP4 + 512B align) | **4.203 MiB** | |

### Max safe cache at budget N

```
cache_slots_max ≈ (N · 1024 − 8100) / 4.203
```

| Budget | Slots available | Chosen c | Margin |
|---|---|---|---|
| 25 GiB | 4159 | **c=4000** | ~160 MiB |
| 20 GiB | 2937 | **c=2700** | ~1000 MiB |
| c/working-set ratio | 9.26× (25 G) / 6.25× (20 G) | | deep LRU regime on both |

## The 4 configurations

| Name | Flags |
|---|---|
| **lazy** | `--no-warmup --eager-compute --seed 42` (mmap + lazy page faults) |
| **mmap+pin** | + `--pin-compute-weights` (mlock attention + output) |
| **io_uring + overlap** | + `--uring-experts --uring-overlap --uring-cache-slots N --uring-cache-policy lru` (O_DIRECT reads into user-space cache; overlap down-read with up+gate compute) |
| **io_uring + pipeline v3** | + `--uring-pipeline-v3` instead of `--uring-overlap` (split-tag upgate/down + first-ready dynamic dispatch + canonical-order accumulation) |

## Pipeline v3 algorithm (decode, n_tokens=1, 1 fused custom op per layer)

**Tag scheme** (8 tags total, fits existing `LLAMA_IO_URING_MAX_TAGS=8`):
```
tag 2e   = upgate(e)   — 2 reads (gate + up)
tag 2e+1 = down(e)     — 1 read
```
for e ∈ {0, 1, 2, 3} selected experts per layer.

**Init (thread 0):**
```
for e in 0..3:
    load_upgate_async_tagged(layer, eid[e], tag=2e)       # bump epoch on e=0
    load_down_async_tagged  (layer, eid[e], tag=2e+1)
    capture sh->e_up/gate/down[e] = slot ptrs
quantise x
sh->dispatched_order[0] = wait_any_upgate_ready(tags=[0,2,4,6])   # first ready wins
```

**Compute loop (all 8 threads, row-split matmuls):**
```
for i in 0..3:
    e = sh->dispatched_order[i]
    up_matmul; gate_matmul; swiglu
    thread 0: quantise act; wait_expert_tagged(2e+1)
    down_matmul → sh->expert_weighted_out[e]   # per-expert, NOT into dst
    thread 0: if (i+1 < 4): sh->dispatched_order[i+1] = wait_any_upgate_ready(remaining tags)
```

**Final accumulation (all threads):**
```
for k in i0..i1:
    dst[k] = Σ_{e=0..3} expert_weighted_out[e·n_embd + k]   # canonical order = bit-exact vs v2
```

17 barriers per layer (same as v2). Bit-exact vs overlap verified at 30 / 200 / 2000 tokens.

## Results — tok/s

| | Lazy | Pin | Overlap | **V3** |
|---|---|---|---|---|
| **25 GiB** | 6.99 | 7.22 | 8.94 | **9.11** |
| **20 GiB** | 5.89 | 6.15 | 8.19 | **8.43** |

## Results — eval time (2000 tok) with σ

| | Lazy | Pin | Overlap | **V3** |
|---|---|---|---|---|
| **25 GiB** (s) | 286.02 ± 0.92 | 276.76 ± 0.42 | 223.62 ± 0.10 | **219.52 ± 0.06** |
| **20 GiB** (s) | 339.39 ± 0.83 | 324.85 ± 0.87 | 244.21 ± 0.06 | **237.13 ± 0.03** |

## Speedup vs lazy

| Step | 25 G | 20 G |
|---|---|---|
| lazy → pin | +3.3% | +4.5% |
| pin → overlap | +23.7% | +33.2% |
| overlap → **v3** | **+1.83%** | **+2.90%** |
| **Total lazy → v3** | **×1.30** (+30%) | **×1.43** (+43%) |

v3 vs overlap at 25 G: Δ = 4.10 s, pooled σ = 0.12 s → **t ≈ 34**.
v3 vs overlap at 20 G: Δ = 7.08 s, pooled σ = 0.07 s → **t ≈ 101**.

## Cache behavior (tonight's runs, from `bsc_phases_json`)

| | Hit rate | Misses | Evictions | Disk read |
|---|---|---|---|---|
| **25 G c=4000** (overlap / v3) | 94.57% | 46 863 | 42 863 | 192.3 GiB |
| **20 G c=2700** (overlap / v3) | 90.97% | 77 940 | 75 240 | 319.9 GiB |

Overlap and v3 show **identical cache metrics** within a budget → v3's advantage is pure timing on the same disk work, not different demand.

## NVMe bandwidth (io_monitor, 25 ms samples)

| | Decode avg | Active-reading avg | Peak |
|---|---|---|---|
| 25 G overlap | 913 MiB/s | 2120 MiB/s | 5134 MiB/s |
| **25 G v3** | **929** | **2250** (+6.1%) | **5422** (+5.6%) |
| 20 G overlap | 1365 | 2780 | 5518 |
| **20 G v3** | **1405** (+2.9%) | **2979** (+7.2%) | 5500 |

v3 extracts 6-7% more bandwidth during active reads → matches the wall-time speedup.

## Caveat on the 2000-token numbers

Model generates coherent text for ~500–1000 tokens, then degenerates (blank-line / repetition). All four methods produce bit-exact same output, so the **within-method speedup numbers are rigorous**. Absolute tok/s includes degenerate tokens (cheap, cache-resident) so is slightly inflated vs a pure coherent-generation benchmark.

## Plot index (same folder)

| File | Content |
|---|---|
| `plots/01_tok_per_sec.{png,pdf}` | tok/s headline bars |
| `plots/02_eval_time.{png,pdf}` | eval-time with σ |
| `plots/03_speedup_vs_lazy.{png,pdf}` | speedup factors |
| `plots/04_nvme_bandwidth_timeseries.{png,pdf}` | raw 25 ms NVMe read trace |
| `plots/05_nvme_bandwidth_avg.{png,pdf}` | sustained read MiB/s |
| `plots/06_bytes_read_from_nvme.{png,pdf}` | total disk I/O per run |
