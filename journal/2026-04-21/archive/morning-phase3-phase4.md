# 21 April 2026 — First GPT-OSS-120B cgroup benchmark; hugepages OOM; pipeline v2 collapses at loose budgets

## Summary

First end-to-end benchmark of GPT-OSS-120B-F16 on this hardware, under the cgroup v2 methodology established April 9. 5 configurations × 3 iterations × 2000 tokens at `memory.max=25 GiB`. Cache `c=3000 LFU-aging-3x` (reused Apr 14 20B canonical stack — **see "Cache miscalibration" below**). Best config: `uring_pipeline_25g @ 237.67 s → 8.42 tok/s` (+20.4% over lazy). Uncovered a system-level 15.2 GiB hugepages reservation that was silently reducing effective RAM from 30 GiB → ~14 GiB and causing global OOM of every uring run; fixed with `sysctl -w vm.nr_hugepages=0`. Pipeline v2's 20B advantage (2.08% at 7G, 0.97% at 9G) **collapsed to statistical noise at 120B/25G** — it's a memory-pressure optimization, not a universal one.

Result directory: `~/BSC/time-tracking/results/cgroup_20260420_200322/`
Settings: `time-tracking/settings_cgroup_april20_120b_fullsweep.json`
Pack file: `gpt-oss-120b-F16.gguf.bscexp` (56.74 GiB, 13824 slots, `--verify` passed, created 19:24).

## 1. Model structure confirmed via gguf-dump

```
block_count            = 36    (20B has 24)
embedding_length       = 2880  (same as 20B)
feed_forward_length    = 2880  (same)
attention.head_count   = 64
attention.head_count_kv = 8    (grouped query attention; interleaved SWA every other layer)
expert_count           = 128   (20B has 32)
expert_used_count      = 4     (same as 20B — pipelining.md §15 claimed 8, WRONG)
context_length         = 131072
```

Per-layer / per-token I/O is **identical to 20B** (4 experts × 3 projections × 4.4 MiB = 52.8 MiB per layer). Only the layer count scales (×1.5). Per-token working set = 432 slot accesses (vs 288 on 20B).

Model file: 61 GiB. The entire file is essentially expert weights (128 × 3 × 4.4 MiB × 36 = 60.75 GiB). Attention + output + embedding + norms < 1 GiB total.

## 2. The hugepages blocker (diagnosed, fixed, not our code)

**First dry-run (19:34)**: `lazy_25g_dry` and `pin_25g_dry` completed (peak 13.3 GiB), but both `overlap_25g_dry` and `pipeline_25g_dry` crashed after decode iter 2 with no error message. `dmesg` revealed:

```
oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=/,mems_allowed=0,
global_oom,task_memcg=/bscexp.slice/...
```

**System-wide OOM, not cgroup-level.** `/proc/meminfo` showed:

```
HugePages_Total: 7799     → 15.23 GiB reserved
HugePages_Free : 7799     → 0 in use
Hugetlb        : 15972352 kB
```

No persistent config (`grep -rn hugepage /etc/sysctl.d/ /etc/sysctl.conf` returned nothing) — ephemeral runtime reservation from some past workload. Effective RAM was 30 GiB − 15.23 GiB ≈ 14.8 GiB; the 25 GiB cgroup budget was larger than the physically available memory. lazy/pin peaked at 13.3 GiB RSS and squeezed under the ceiling. The uring configs allocated an additional 12.6 GiB of non-evictable anon (uring cache) and blew past it on decode iter 3 → global SIGKILL. **This would have silently invalidated ALL previous 120B data if we hadn't checked.**

Fix: `sysctl -w vm.nr_hugepages=0`. MemAvailable jumped from 13.6 GiB → 29.5 GiB immediately. Second dry-run and full sweep proceeded cleanly.

Writeup implication: worth verifying on any future server. The April 9 / 14 methodology documented cgroup v2 + MemorySwapMax=0 but did not check hugepages. On the TUM server this was benign for 20B (footprints always under 14 GiB effective) but would have broken any larger-model experiment.

## 3. Full sweep results

All configs: `--no-warmup --eager-compute --seed 42`, with pin where indicated, cache c=3000 LFU-aging-3x where uring.

| Config | Eval mean (s) | σ (s) | CV | Tok/s | Major faults | vs lazy | vs pin | vs overlap |
|---|---|---|---|---|---|---|---|---|
| lazy_25g | 286.02 | 1.12 | 0.39% | **6.99** | 2,390,918 | — | +3.3% | +20.2% |
| mmap_pin_25g | 276.76 | 0.52 | 0.19% | **7.23** | 2,152,865 | −3.2% | — | +16.3% |
| uring_plain_25g | 242.62 | **0.05** | 0.02% | **8.24** | 66,046 | −15.2% | −12.3% | +2.0% |
| uring_overlap_25g | 237.94 | **0.09** | 0.04% | **8.40** | 65,797 | −16.8% | −14.0% | — |
| uring_pipeline_25g | **237.67** | **1.57** | 0.66% | **8.42** | 68,559 | −16.9% | −14.1% | −0.11% |

**Statistical significance of deltas:**
- uring vs lazy/pin: >100σ (46,000 s vs 0.1 s pooled σ — undisputed)
- overlap vs uring_plain: Δ=4.68 s, pooled σ=0.10 s, **t=46** — highly significant, overlap works on 120B
- **pipeline vs overlap: Δ=0.27 s, pooled σ=1.57 s, t=0.17** — NOT significant

Pipeline iters: [236.84, 236.69, **239.48**]. The third iteration is a 3σ outlier relative to the first two. Can't discard without cause but it inflates pipeline's variance dramatically. With just iters 1-2, pipeline mean = 236.76 s (faster than overlap by ~1.2 s), but n=2 is meaningless statistically. 10-iteration re-run would disambiguate.

## 4. Cgroup memory behavior

| Config | memory.peak (MiB) | memory.current @ end (MiB) | events.max (reclaim events) |
|---|---|---|---|
| lazy_25g | 25600 (hit max) | 21,322 | **112,921** |
| mmap_pin_25g | 25600 | 21,313 | 116,929 |
| uring_plain_25g | 25600* | 8,144 | **3,338** — 34× less |
| uring_overlap_25g | 25600* | 8,147 | 3,338 |
| uring_pipeline_25g | 25600* | 8,147 | 3,334 |

* All uring configs hit 25600 peak, likely during prompt-eval when multi-token path pulls in mmap pages. Steady-state decode runs at only 8.1 GiB resident (= pin 2.98 + KV 4.6 + compute 0.4 + misc 0.1). The uring cache's 12.6 GiB allocation is anon-virtual, partially resident, and apparently reclaimed between peak and sample.

Pin for 120B: **2979.7 MiB** locking 470 tensors (vs 2329 MiB / 20B). KV cache at ctx=131072 with iSWA: 4635 MiB total (18 non-SWA × 4608 + 18 SWA × 27). Compute buffer: 413 MiB (same as 20B — not scaled, determined by n_ubatch not layer count).

## 5. Key finding: pipeline v2 is regime-dependent

| Model | Budget | Cache vs working set | Pipeline vs overlap |
|---|---|---|---|
| 20B | 7 GiB | 250 / 288 (0.87× undersized, LFU wins) | **−2.08%** (pipeline faster, 6σ) |
| 20B | 9 GiB | 740 / 288 (2.57× oversized, LRU wins) | **−0.97%** (pipeline faster) |
| 120B | 25 GiB | 3000 / 432 (**6.94× oversized**) | **−0.11%** (NOT significant, σ=1.57 s) |

Mechanistic interpretation: pipeline v2's submit-all-at-layer-entry-then-wait-per-expert trick only pays off when there are enough cache misses to keep the NVMe queue saturated at QD≈12. At 6.94× oversized cache, hit rate is very high, few misses, NVMe rarely actually loaded — overlap and pipeline both spend most of their "load" time on near-instant cache hits. The two converge.

**Thesis implication**: the pipelining.md 2% headline is **tight-budget**, not universal. pipelining.md §15 flagged this as an open question; this result provides the first data point at the "very loose budget" end of the curve. Cleanly defensible story: pipeline v2 is an optimization for the pressured regime; at loose budgets it degrades gracefully (no regression) but provides no speedup.

## 6. Cache miscalibration (self-critique)

**Used c=3000; should have used c=4000.** Arithmetic from the empirical memory.current:

- Measured fixed non-evictable (uring steady-state): **8.1 GiB**
- Budget: 25 GiB → available for cache: **16.9 GiB**
- Max slots at 4.203 MiB/slot: **c ≈ 4100**

I picked c=3000 because I did not have an empirical fixed-footprint measurement for 120B when writing the config (I'd estimated 9.3 GiB from 20B extrapolation). Now that we do, **25% of the budget was unused**. The actual peak memory usage per the cgroup CSV is only 8.15 GiB (steady-state) or 25.6 GiB (during prompt eval) — both modes fit comfortably with c=4000.

**Policy choice was also likely wrong.** Working set 432, cache 3000 = 6.94× oversized → deep LRU-wins regime per the April 9 20B finding (LRU beat LFU by 9-10% at oversized). Using LFU-aging-3x was a carry-over from the Apr 14 20B canonical (where it ties/beats LRU at the sizes tested), but that sweep did not include c > 740. At c=4000 / 6.94× oversized the LRU crossover almost certainly applies.

**Plausible headroom left on table**: 5-10% tok/s. So "best" 120B@25G config is probably closer to 9.0-9.2 tok/s, not 8.42. Need to re-run with c=4000 LRU to confirm.

## 7. Side observations

- **uring_plain σ = 0.05 s, CV 0.02%**. Cleanest number we've seen on this workload — io_uring's O_DIRECT determinism shows. Overlap σ = 0.09 s also sub-CV 0.1%. Pipeline σ is 17× larger and only hits CV 0.66%.
- **Pipeline has 4% more faults than overlap** (68559 vs 65797). Reproduced from the dry-run. Small absolute, systematic. Unexplained — possible mmap-adjacent access in fused op, or different prompt-eval → decode transition in the fused path. Not breaking anything.
- **Pin alone gives 3.3% on 120B@25G** (vs 36% on 20B@7G). Pin's value is proportional to pressure; at 25G there's enough page-cache room to hold attention weights without pin.
- **lazy_25g gen_faults = 2.4M per 2000 tokens → ~1200 faults/token**. Compared to 20B@7G mmap+pin = 14.3M/2000 = 7150 faults/token. 120B actually faults 6× LESS per token than 20B@7G, despite being 5× bigger, because the budget-to-model ratio is a bit looser (25/61 = 41% vs 7/12.85 = 54% — wait, 20B is LOOSER). Actually 20B at 7G should fault LESS. Hmm. Possible explanation: the 2000-token run on 120B is shorter-proportionally (not enough tokens to touch all experts) so fewer total unique experts accessed. Worth thinking about.

## 8. Next

Immediate follow-up sweep tomorrow (config drafted separately):

**25G — cache + policy study (5 configs)**: settle the c=4000 vs c=3000 and LRU vs LFU-aging questions at oversized cache. Directly completes the 2x2 started by today's c=3000 LFU.

**20G — full baseline sweep (6 configs)**: first data at this budget. Cache c=2800 (arithmetic max ≈ 2899 — leaves 420 MiB margin). Same policy A/B between overlap and pipeline.

Deferred (smaller budgets than 20G not prioritized by user):
- 15G / 12G / 10G where the LFU crossover + pipeline v2's advantage would re-emerge. Would complete the regime map but not needed for the immediate thesis-quality 120B story.

## Files / artifacts

- Results CSVs: `~/BSC/time-tracking/results/cgroup_20260420_200322/{lazy,mmap_pin,uring_plain,uring_overlap,uring_pipeline}_25g{,_cgroup,_meminfo}.csv`
- Dry-run results: `~/BSC/time-tracking/results/cgroup_20260420_195313/` (4 configs × 1 iter × 200 tok; post-hugepages-fix)
- Crashed dry-run results: `~/BSC/time-tracking/results/cgroup_20260420_193400/` (overlap/pipeline OOM'd pre-hugepages-fix) — kept as documentation of the diagnosis
- Pack file: `~/llama.cpp/models/gpt-oss-120b/gpt-oss-120b-F16.gguf.bscexp` (56.74 GiB, verified)
- Pack log: `/tmp/bsc-pack-120b.log`
- Sweep log: `/tmp/bsc-120b-phase3.log`
- Binary: `~/llama.cpp/build/bin/llama-completion` built 2026-04-17 20:23 from current working tree on top of commit `d7e480f4`
