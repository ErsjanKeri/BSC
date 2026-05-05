# Experimental Hypotheses - SSD-Backed Inference

## Primary Research Question

**How do access patterns in real LLM models (especially MoE) differ from assumptions in prior work, and how does this affect SSD-backed inference performance?**

---

## H1: MoE Models Show Different Access Patterns Than Standard Transformers

Standard transformers process all layers identically per token, which is highly sequential. MoE models activate only K of N experts per token, creating sparse, input-dependent access patterns.

**What to measure:** Sequential read % for standard transformer vs MoE model under memory pressure.

**Status:** Partially explored via tensor tracing (expert activation data exists for 5 domains × 100 tokens). Not systematically compared with blktrace sequential % metrics.

---

## H2: SSD-Backed Inference Slowdown Is Sublinear

When only part of the model is on SSD, the slowdown should be less than proportional because:
- Kernel can prefetch from SSD while computing on cached parameters
- Frequently accessed tensors (embeddings, norms) stay in page cache

**What to measure:** Tokens/sec at various memory pressure levels.

**Status:** blktrace experiments (Dec 2025) showed performance degradation under mlock pressure. MAP_POPULATE experiments (Feb 2026) showed load time impact. No systematic sweep of partial-SSD scenarios.

---

## H3: Swappiness Affects Access Patterns

Higher swappiness → more aggressive page eviction → more re-reads → worse performance under memory pressure.

**What to measure:** Page fault counts and total bytes read at swappiness 0/60/100.

**Status:** Never tested.

---

## H4: Measured SSD Throughput Will Be Below Spec

Page-granularity access (4KB), kernel overhead, and non-sequential patterns from MoE routing will reduce effective throughput below hardware capability.

**Status:** No direct throughput measurement was done during inference. The "80 GB/s" target was a spec sheet number, not measured.

---

## What Was Not Tested

- Swappiness experiments (H3)
- Page size impact (huge pages vs 4KB)
- Batch size effects
- Prompt length variation
- Quantization effects on access patterns

These remain open questions.

---

*Original file was ~300 lines with formal hypothesis formatting and experiment matrices. Simplified 2026-03-04.*
