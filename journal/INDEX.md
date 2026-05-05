# Journal Index

Annotated table of contents for all research journal entries. Status tags reflect the state as of **2026-04-22** and are not revisions of the entries themselves — content is untouched; this file is read-only metadata.

## Status legend

| Tag | Meaning |
|---|---|
| **CANONICAL** | Findings, code, or methodology that is cited in CLAUDE.md and belongs in the thesis headline results. |
| **CURRENT** | Observations/discoveries that are still valid today. Absolute numbers may be stale if pre-April-9, but the qualitative finding holds. |
| **REFERENCE** | Useful historical / design / infrastructure document. Not a results entry to cite for thesis numbers, but worth keeping for context. |
| **DEPRECATED-NUMBERS** | Results entry whose numerical claims were invalidated by the April 9 methodology shift. Direction / mechanism may still be correct; exact numbers must be re-measured under cgroup before citation. See CLAUDE.md Data Reliability Guide. |

---

## Dec 2025 — blktrace era (pre-tensor-tracing)

| Entry | Status | One-line hook |
|---|---|---|
| `2025-12-09-findings.md` | REFERENCE | Early notes on `llama.cpp/src/llama-mmap.cpp` and page-cache behaviour — the starting mental model for the thesis. |
| `2025-12-20.md` | DEPRECATED-NUMBERS | First blktrace sweep on llama-2-7b-chat, varying mlock pressure. The unique-sectors metric was later found buggy (Dec 21). |
| `2025-12-21.md` | DEPRECATED-NUMBERS | Fixed the 88.9%-underestimate bug in the unique-sectors calc + attempted tmpfs isolation. Corrects Dec 20, but the entire workflow (blktrace + llama-2-7b + mlock_tool) was superseded in April. |
| `2025-12-22.md` | DEPRECATED-NUMBERS | Fixed blktrace action filtering; self-tagged as superseding the 21st. Still pre-cgroup. |
| `2025-12-30.md` | **CURRENT** | "blktrace has a semantic gap — can't tell which tensors are being read." This is the motivation document for the whole tensor-tracing approach and an important thesis-narrative pivot. |

---

## Jan 2026 — Tensor-tracing infrastructure

| Entry | Status | One-line hook |
|---|---|---|
| `2026-01-02-critical-review.md` | REFERENCE | Architecture review for thread-local trace buffer design. Design doc. |
| `2026-01-02.md` | REFERENCE | Tensor tracer design decisions (tmpfs, entry format v1). Design doc; format later evolved to 1024 bytes. |
| `2026-01-04.md` | REFERENCE | Moment-of-milestone: first real inference with tensor name logging. |
| `2026-01-07.md` | REFERENCE | 256-byte trace format + WebUI + buffer allocation tracking + trace↔graph correlation. Infrastructure milestone. |
| `2026-01-08.md` | REFERENCE | Automated experiment pipeline for tensor tracing. Infrastructure. |
| `2026-01-13.md` | **CURRENT** | Investigation of two `gguf-dump` bugs: (1) missing data section offset, (2) quantization-naive size calculation (7.11× inflation). Defines the current tool correctness. |
| `2026-01-14.md` | REFERENCE | Supervisor-meeting project summary as of day 24. Historical snapshot, partly superseded. |
| `2026-01-17.md` | **CURRENT** | Complete fix to `gguf-dump.cpp` for MXFP4 quantization. Establishes tool correctness for memory-map.json. |
| `2026-01-25.md` | REFERENCE | DesktopUI implementation (ImGui+ImPlot, 100+ token scale, 110 FPS). Infrastructure. |
| `2026-01-26.md` | **CURRENT** | **GGUF alphabetical tensor ordering discovery** (blk.0, blk.1, blk.10, ..., blk.2 ordering) — a thesis-quality finding about disk layout vs. logical layer order. |

---

## Feb 2026 — Characterization

| Entry | Status | One-line hook |
|---|---|---|
| `2026-02-05.md` | REFERENCE | DesktopUI refinements + 5-domain expert-activation experiment setup. Infrastructure. |
| `2026-02-07.md` | **CURRENT** | **MAP_POPULATE discovery** + sparse access characterization. Thesis finding; MAP_POPULATE is harmful for MoE models exceeding RAM. |

---

## Mar 2026 — Timing + madvise (methodology maturing)

| Entry | Status | One-line hook |
|---|---|---|
| `2026-03-08.md` | **CURRENT** | Load-time decomposition: 93% of "load time" is actually warmup, not file loading. Thesis-quality observation; numbers pre-cgroup. |
| `2026-03-09.md` | DEPRECATED-NUMBERS | Follow-up experiments with warmup disabled. Direction correct, numbers invalid. |
| `2026-03-13.md` | **CURRENT** | `llama.cpp` timer-reset bug documented, own phase-timing instrumentation built. **Methodology contribution.** 9-experiment matrix numbers are pre-cgroup so deprecated, but the instrumentation itself stands. |
| `2026-03-14.md` | **CURRENT (infra) / DEPRECATED-NUMBERS (results)** | Independent phase timing system + 90-experiment matrix. Infrastructure CURRENT; result numbers pre-cgroup. |
| `2026-03-15.md` | DEPRECATED-NUMBERS | Results from the 90-experiment sweep. Pre-cgroup; also had silent swap ON. Do not cite numbers. |
| `2026-03-16.md` | **CURRENT** | MoE expert computation code-path trace (`build_moe_ffn` → `ggml_compute_forward_mul_mat_id:1628`). Establishes the exact page-fault location cited in CLAUDE.md and the thesis. |
| `2026-03-22.md` | DEPRECATED-NUMBERS | Small-model-only experiment results. Pre-cgroup. |
| `2026-03-23.md` | **CURRENT** | Documented blktrace overhead under memory pressure — trace writes compete for page cache. Negative result that justifies the tensor-tracing approach. |
| `2026-03-24.md` | DEPRECATED-NUMBERS | MADV_WILLNEED prefetch implementation + results. The **code** is retained (`--moe-prefetch` flag). The 22 GiB / 20 GiB pressure comparison used mlock_tool, so numbers are invalid; the direction (WILLNEED helps at some pressures, hurts at others) needs cgroup re-validation before citation. |
| `2026-03-27.md` | **CURRENT (direction) / DEPRECATED-NUMBERS (magnitude)** | MADV_RANDOM is catastrophically harmful (35× more faults, 5.6× slower). Direction defensible, absolute numbers pre-cgroup. |

---

## Apr 2026 — Cgroup methodology + the optimization stack (THE CANONICAL ERA)

| Entry | Status | One-line hook |
|---|---|---|
| `2026-04-09.md` | **CANONICAL (methodology)** | The methodology reset. All prior numerical results invalidated. Adopted cgroup v2 `memory.max`, swap OFF, validated <1% CV. **The foundation under which every later number in this thesis was collected.** |
| `2026-04-09-results.md` | **CANONICAL** | First cgroup-standard benchmark: 14 configs × 2 iters, avg CV 0.04%. Best io_uring beats default lazy mmap by 2.67×. Cache policy crossover at working-set boundary (288) established here. |
| `2026-04-12.md` | **CANONICAL** | `--eager-compute` discovery: 377 MiB of compute buffer was lazy-faulted, hiding budget headroom. Every prior "7 G" run was actually effectively ~7.37 G. Budget arithmetic corrected; ceilings c254/c498/c742 defined. |
| `2026-04-13.md` | **CANONICAL** | Infrastructure day: `--uring-overlap` (phase1 sync + phase2 async, bit-exact), bit-exact cache simulator (2377/2377 matches), LFU-aging `aging_mult=3` tuning. All stack into the canonical April 14 numbers. |
| `2026-04-14.md` | **CANONICAL** | 54-run benchmark: best tok/s per budget on GPT-OSS-20B = 5.65 / 7.45 / 9.39 at 7 / 8 / 9 G. Cross-day-reproducible within 1%. Current 20B headline numbers. |

---

## Apr 21–22 — Pipeline v2 / v3 on both models (latest)

| Entry / folder | Status | One-line hook |
|---|---|---|
| `2026-04-21/` (folder: summary.md + plots + scripts) | **REFERENCE — CAVEATED** | April 21 v3_bench writeup. Single-topic prompt; later found to suffer from a newline-spam workload bias that artificially inflates cache hit rate to 94.6% and understates the v3 advantage. **Do not cite for thesis-final numbers**; the methodology finding (bias existed) is itself current and cited in `MEMORY.md`. |
| `2026-04-22/` (folder: summary.md + plots + scripts) | **CANONICAL — current headline** | 45-run multi-topic sweep on both models. V3 wins 5.2–7.1% over overlap across every (model, budget). Bit-exact output across overlap/v2/v3. Current ground truth for thesis pipeline claims. |

---

## Apr 30 – May 1 — Async-experts (v4) implementation, deadlock fix, and structural-ceiling finding

| Entry | Status | One-line hook |
|---|---|---|
| `2026-04-30.md` | **CURRENT (mechanism + fix)** | `compute_async_experts` (per-(expert, projection) tags, eager dispatch) implementation + the publish-overwrite race that caused 40% deadlock under cgroup 7G + the two-barrier handshake fix. 20/20 post-fix runs at CV 0.16%. Caveats: one budget, one model verified empirically; the two-barrier fix was REPLACED on May 1 by a 2-deep ring slot scheme (also race-free, also bit-exact). |
| `2026-05-01.md` | **CANONICAL (negative result with mechanism)** | Profiling-driven investigation of why `--uring-async-experts` is slightly slower than `--uring-async-projection-overlap`. Five rounds of fixes (spin_check stack-local, parity-slot, parallel quantize, picker decomposition) — all bit-exact, none move wall-clock. Conservation law: total `moe_barrier_wait_ms` ~ 160K regardless of barrier count. Root cause: thread 0 spends 16.35s blocked on `wait_any_upgate_ready`, and ae's smaller dispatch chunks expose I/O latency more than apo's larger phases hide it. Structural ~1% ceiling. Closing it requires background completion thread or cross-layer prefetching, neither in BSC scope. Also: this is the day v3/v4 were renamed to async-projection-overlap / async-experts in code. |

---

## Non-dated documents

| File | Status | One-line hook |
|---|---|---|
| `meeting_notes.md` | REFERENCE | Pre-April summary of findings for advisor discussion. Partly superseded by April canonical; keep for narrative. |

---

## How to use this index

- **For thesis writing**: cite entries tagged CANONICAL for headline claims and numbers. For qualitative findings (e.g., "MAP_POPULATE harmful", "MADV_RANDOM catastrophic"), CURRENT entries provide the direction; if you need exact numbers, re-measure under cgroup first.
- **For advisor discussion**: REFERENCE entries are fine narrative context; do not quote numbers from DEPRECATED-NUMBERS entries.
- **For code archaeology**: REFERENCE entries on tooling (Jan 2026 series) document design decisions that shaped the current codebase.

## Meta

- This index lives at `journal/INDEX.md` and is read-only metadata over untouched journal entries.
- Written on 2026-04-22 as part of the thesis prep cleanup. Entry contents are not modified.
- When new journal entries are added, append to the appropriate month section with a status tag.
