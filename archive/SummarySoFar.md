# Bachelor Thesis Progress Summary (Early Phase)
**Student:** Ersjan Këri
**Supervisor:** Gabriel Haas (Viktor Leis Department, TUM)
**Topic:** Tensor-Level Access Pattern Analysis for SSD-Backed LLM Inference

**Note:** This was an early-phase summary. Many claims here were speculative. See journal entries and tensor-tracing/ for actual verified results.

---

## Project Goal

Understand and characterize how LLM parameters are accessed during inference when model weights are stored on SSD rather than RAM, with focus on Mixture-of-Experts (MoE) architectures.

## Server

- **Host:** cli-hiwi-02.dis.cit.tum.de
- **CPU:** AMD Ryzen 7 7700X (8c/16t)
- **RAM:** 30 GiB
- **Storage:** 2× NVMe (Samsung 980 PRO 1TB, WD 960GB)
- **GPU:** None (CPU inference only)
- **Access:** SSH via ProxyJump through i13vm10.in.tum.de (requires TUM VPN)

## Models Tested

- **llama-2-7b-chat.Q4_K_M.gguf** (3.9 GB) — standard transformer, fits in RAM
- **gpt-oss-20b** (12.83 GB, F16) — MoE with 32 experts, top-4 activation
- **gpt-oss-120b** (61 GB) — MoE, exceeds RAM

## CHEOPS Paper Context

Paper characterized I/O patterns for SSD-offloaded LLM inference using NVMeVirt (RAM-emulated NVMe).

**Their findings:** 16.9 GiB/s hardware capacity, frameworks achieved only 2.6–4.9 GiB/s.

**Gabriel's concerns:**
1. Used dummy/synthetic models — real MoE models may behave differently
2. GPU offloading adds extra transfer step (GPU→RAM→SSD) — CPU-only would differ
3. "Uniform" access claim needs scrutiny — MoE expert selection creates sparse patterns
4. RAM-based NVMe emulation may miss real SSD characteristics

## What Was Actually Done (Verified)

1. **blktrace experiments (Dec 2025):** Measured block-level I/O under memory pressure. Found 100% sequential access at application level, file fragmentation caused misleading "backward seeks." See journal/2025-12-20.md through 2025-12-22.md.

2. **Tensor-level tracing (Jan–Feb 2026):** Custom instrumentation in llama.cpp ggml-cpu backend. Binary format, automated pipeline, WebUI + DesktopUI. See tensor-tracing/ directory.

3. **Expert activation experiments (Feb 2026):** 5 domains × 100 tokens. Captured MoE routing patterns.

4. **MAP_POPULATE experiments:** Compared prefetch ON vs OFF. See time-tracking/results.md.

## Early Speculative Claims (NOT VERIFIED)

The following claims from the original version of this document were never measured or verified:
- "Achieved 10 GB/s throughput" — no measurement supports this
- "Targeting 80 GB/s SSD bandwidth saturation" — spec sheet number, not a realistic target
- "Potential 8-12× speedup" — pure speculation
- Claims about llama.cpp MLA handling being incorrect — never investigated

---

*Original file was ~970 lines. Simplified 2026-03-04.*
