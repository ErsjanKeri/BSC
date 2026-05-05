# April 22 — Pipeline sweep on coherent workload (both models)

## Motivation

April 21 benchmarks exposed a methodological problem: the single-topic prompt `"Write a detailed essay..."` caused GPT-OSS-120B to degenerate into newline-spam after ~token 500. Over a 2000-token run this drove the cache hit rate to an artificial 94.6% and deflated the pipeline/overlap delta (v2 +0.2%, v3 +1.9%) because ~75% of the measured workload had almost no I/O to overlap. Today's sweep uses a **multi-topic essay prompt** (5 topics × 5 sub-periods) that sustains coherent prose across the full 2000 tokens, then extends the comparison to GPT-OSS-20B as well.

## Experimental setup

- **Multi-topic prompt**: 5 topics (computer science, mathematics, physics, astronomy, biology) × 5 periods (ancient, medieval, 19C, 20C, modern), "write at least 5000 words".
- **Sampler**: `--seed 42`, no repeat-penalty, no chat template, `-no-cnv` — identical to prior benchmarks for comparability.
- **Budget/cache pairings**: 120B uses LRU at c=2700 (20 G) / c=4000 (25 G); 20B uses `lfu-aging + aging-mult=3` at c=250 (7 G) / c=498 (8 G) / c=740 (9 G) — April 14 canonical.
- **3 iterations** per config under cgroup v2 `memory.max`, drop_caches + 10 s cooldown between runs.
- **`io_monitor`** polling `/proc/diskstats` every 25 ms on `nvme1n1`, outside the cgroup.
- 15 configs × 3 iters = **45 runs** total, all in `cgroup_20260421_170500/`.

## Headline results

| Model · Budget | overlap | v2 | **v3** | **v3 vs overlap** |
|---|---|---|---|---|
| 120B · 20 G · c2700 LRU | 331.32 s (σ 0.78) · 6.03 tok/s | 325.29 (σ 0.51) · 6.14 | **309.39 (σ 0.25) · 6.46** | **−6.62% eval · +7.09% tok/s** |
| 120B · 25 G · c4000 LRU | 273.05 (σ 0.05) · 7.32 | 270.32 (σ 0.04) · 7.40 | **259.65 (σ 0.05) · 7.70** | **−4.91% · +5.16%** |
| 20B · 7 G · c250 lfua3 | 156.91 (σ 0.21) · 5.34 | 153.27 (σ 0.04) · 5.47 | **147.59 (σ 0.39) · 5.68** | **−5.94% · +6.31%** |
| 20B · 8 G · c498 lfua3 | 119.43 (σ 0.08) · 7.02 | 117.91 (σ 0.64) · 7.11 | **112.11 (σ 0.18) · 7.48** | **−6.13% · +6.53%** |
| 20B · 9 G · c740 lfua3 | 93.48 (σ 0.11) · 8.96 | 91.98 (σ 0.12) · 9.11 | **87.39 (σ 0.08) · 9.59** | **−6.51% · +6.97%** |

**V3 beats overlap by 5.2% – 7.1% on every (model, budget).** All CVs ≤ 0.54%, most under 0.2%.

V2 vs overlap is between +1.0% and +2.4%, confirming April 17's observation that "submit-all-tagged" alone helps modestly. The additional +4–5% from v3 on top of v2 is the split-tag (upgate/down) + first-ready dynamic dispatch pattern.

## Cache regime table

| Config | hit rate | misses/tok | avg I/O per tok | gen faults |
|---|---|---|---|---|
| 120B 20 G c2700 | 76.29% | 102.4 | 451 MiB | ~139 k |
| 120B 25 G c4000 | 86.59% | 57.9 | 255 MiB | ~137 k |
| 20B 7 G c250 (c < W.S. 288) | **37.16%** | 75.8 | 334 MiB | ~37 k |
| 20B 8 G c498 | 61.13% | 46.9 | 207 MiB | ~34 k |
| 20B 9 G c740 | 78.34% | 26.1 | 115 MiB | ~32 k |

The 20B @ 7 G row is the undersized-cache regime (250 slots vs 288 working-set); hit rate is the lowest across the sweep and LFU-aging is still ordering correctly. The 120B @ 20 G row (76.29% hit rate) matches the n=200 coherent-essay hit rate measured April 21 — confirming the prompt fix eliminated the degenerate-workload bias.

## NVMe bandwidth — the mechanism behind v3's win

Average active-read bandwidth during decode (mean of 3 iters, skipping first 2 s, filtering idle samples):

| Config | overlap | v2 | v3 | v3 advantage |
|---|---|---|---|---|
| 120B 20 G | 2644 MiB/s | 2694 | **2824** | +6.8% |
| 120B 25 G | 1968 | 1990 | **2070** | +5.2% |
| 20B 7 G | 3990 | 4082 | **4235** | +6.1% |
| 20B 8 G | 3245 | 3289 | **3451** | +6.3% |
| 20B 9 G | 2365 | 2407 | **2525** | +6.8% |

V3 sustains ~6% more NVMe bandwidth than overlap in every case. The mechanism (submit all 8 tags upfront, dispatch on first-ready upgate pair, wait for down only when its compute is about to start) keeps the io_uring queue deeper for longer — which is exactly what shows up in the bandwidth trace plots (`04_nvme_bandwidth_timeseries_granular.png`).

## Correctness

- Generation region bit-exact across overlap / v2 / v3 at spot-checked blocks (120B 25 G and 20B 9 G, run 1): md5 identical.
- Hit / miss counts match across modes within each (model, budget) block to fewer than 20 accesses — within normal warmup noise.
- Bit-exactness is strong evidence that no FP drift has been introduced by v3's dynamic dispatch. The canonical-order accumulation fix (`src/llama-moe-pipeline.cpp:625-680`) is working as designed.

## Generation quality

Verified by a background agent (no log text pulled into the analysis workspace):

- **120B — coherent throughout.** High-quality prose with precise dates (Antikythera 87 BCE, Boole 1815–1864, Babbage 1822). Cuts off mid-sentence at the 2000-token limit. Only 1 of 5 requested topics covered (Computer Science through 19C) — the model is thorough, not verbose. No repetition, no newline collapse, no degeneration.
- **20B — refuses the prompt.** Loops on meta-commentary ("I must produce a long essay", "That is a lot") 5+ times, then emits `[end of text]`. Zero actual topic coverage on all 5 topics. This is a 20B model-capability refusal behavior, not a pipeline bug. Identical refusal at both 7 G and 9 G, so it's not a memory constraint.

**Implication for the thesis**: the 20B pipeline-comparison numbers are methodologically valid (bit-exact access pattern across overlap/v2/v3, identical workload per mode) but they measure "20B refusing a complex prompt and looping" rather than "20B writing an essay". For thesis narrative, the **120B rows are the defensible headline** and the 20B rows need either a caveat or a re-run with a 20B-compatible prompt.

## Open items / next steps

1. **20B prompt replacement.** Either (a) a simpler single-topic prompt that 20B accepts, (b) a chat-templated prompt ("You are a tutor. Explain ..."), or (c) accept the refusal and disclose it. Option (b) is probably the cleanest.
2. **Cross-day reproducibility of the v3 win.** Today's numbers are all from a single sweep session. Re-running one config (overlap / v3 @ 120B 20 G, 3 iters) on a different day would close the methodological gap that exists in `pipelining.md`.
3. **Decide whether to cite v2 separately.** v2's 1–2% vs overlap is in the noise band at 25 G (CV 0.02%) but clean at 20 G and 20B budgets. Story is cleaner if v3 is presented directly vs overlap.
4. **Policy re-validation under coherent workload** — the April 14 LFU-aging+3 optimum was tuned on the degenerate workload. Worth a quick simulator pass on the new access traces before citing it as thesis-canonical for the 20B 7/8/9 G regime.

## Plots

- `plots/01_tok_per_sec.png` — Grouped bar, all 5 configs × 3 methods.
- `plots/02_eval_time.png` — Wall-clock eval time with σ error bars.
- `plots/03_v3_speedup_vs_overlap.png` — Headline claim, v2 & v3 deltas.
- `plots/04_nvme_bandwidth_timeseries_granular.png` — **5 × 3 panel grid**, raw 25 ms NVMe read samples for every (config, method). Per-panel dotted line is active-read mean.
- `plots/05_nvme_bandwidth_avg.png` — Bar chart of sustained read bandwidth.
- `plots/06_bytes_read.png` — Total GiB read from NVMe — identical across modes within a config (sanity check on bit-exactness of access pattern).
- `plots/07_cache_hit_rate.png` — Expert cache hit rate across configs.

### Zoomed NVMe read plots (match `io_plots/` style: filled area + SSD-ceiling reference)

`plots/io_zoom/` contains close-up views of the first 5 s and 10 s of each run, 3 stacked panels per file (overlap / v2 / v3) so you can see the per-layer burst structure directly. Files:

- `io_first10s_120b_20g.png`, `io_first10s_120b_25g.png`
- `io_first10s_20b_7g.png`, `io_first10s_20b_8g.png`, `io_first10s_20b_9g.png`
- `io_first10s_all_overlaid.png` — 5-row cross-config view, all 3 methods overlaid per panel
- `io_first5s_*` — same layout, 5-second zoom for maximum detail

In every 10 s view you can read three phases cleanly:
1. **`t ≈ 0.5 – 2.0 s`** — initial load / prompt-eval plateau (~2500–3000 MiB/s, dense)
2. **`t ≈ 2 – 3 s`** — brief quiescent gap before decode starts
3. **`t ≥ 3 s`** — decode proper: characteristic per-token bursty pattern where every layer triggers an io_uring read fan-out. This is the workload the thesis pipeline is actually optimizing; v3 shows visibly denser bursts at higher peaks (most clearly on 20B 7G and 20B 9G, where total I/O per token is higher).

All regeneratable from `scripts/plot_all.py` and `scripts/plot_io_granular.py`.

## Provenance

- Source data: `/home/keri/BSC/time-tracking/results/cgroup_20260421_170500/` (45 runs, 136 files).
- Driver settings: `time-tracking/settings_cgroup_april21_pipeline_sweep_multitopic.json`.
- Dry test that validated the multi-topic prompt on 120B: `cgroup_20260421_163155/`.
- Dry test that surfaced 20B refusal (after the fact): `cgroup_20260421_170234/`.
- Commit head during run: `d7e480f4` (atomic sense-reversing barrier) + uncommitted v3 changes in working tree.
