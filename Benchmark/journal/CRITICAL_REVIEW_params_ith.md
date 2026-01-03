# CRITICAL ARCHITECTURE REVIEW: Thread Logging Strategy

**Date**: January 2, 2026
**Status**: REQUIRES DECISION BEFORE PROCEEDING
**Reviewer**: Ersjan

---

## The Question

**Should we log tensor access once per operation (ith == 0) or once per thread (all threads)?**

Current implementation uses: `if (params->ith == 0)`
This logs only from **thread 0**, ignoring threads 1, 2, 3, ...

---

## Understanding the Architecture

### 1. File Structure (ggml-cpu/)

**ggml-cpu.c** (123KB):
- Main CPU backend implementation
- Contains `ggml_compute_forward_mul_mat()` - the REAL implementation
- Operation dispatcher (switch/case on operation type)
- This is where we added our hook

**ops.cpp** (353KB):
- Higher-level operations (Conv2D, etc.)
- Contains `ggml_call_mul_mat()` - a HELPER wrapper
- Creates temporary tensor structures and calls `ggml_compute_forward_mul_mat()`

**Relationship**:
```
ops.cpp: ggml_call_mul_mat()
    → Creates temp tensors
    → Calls ggml_compute_forward_mul_mat() (in ggml-cpu.c)
        → Our hook is HERE
        → Actual matrix multiply implementation
```

### 2. What is "compute_forward"?

**Terminology from Neural Networks**:
- **Forward pass**: Computing outputs from inputs (inference)
- **Backward pass**: Computing gradients (training)

`ggml_compute_forward_*` functions implement the **forward pass** of operations.

### 3. Understanding params->ith

```c
struct ggml_compute_params {
    // ith = thread index, nth = number of threads
    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void * wdata;

    struct ggml_threadpool * threadpool;
};
```

**Key facts**:
- `ith` = thread index (0, 1, 2, ..., nth-1)
- `nth` = total number of threads
- Each operation is executed by **ALL threads in parallel**
- Threads work on **different chunks** of the computation

---

## How Threads Actually Work in MUL_MAT

### Work Partitioning

From `ggml-cpu.c` lines 1395-1419:

```c
// distribute the work across threads
nchunk0 = nr0 > nr1 ? nth : 1;  // parallelize by src0 rows
nchunk1 = nr0 > nr1 ? 1 : nth;  // parallelize by src1 rows

// Each thread processes different chunks
int current_chunk = ith;  // Thread starts with its own chunk

while (current_chunk < nchunk0 * nchunk1) {
    // Each thread works on different part of the matrix
    const int64_t ith0 = current_chunk % nchunk0;
    const int64_t ith1 = current_chunk / nchunk0;

    const int64_t ir0_start = dr0 * ith0;
    const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

    // Process this chunk...
    ggml_compute_forward_mul_mat_one_chunk(..., ir0_start, ir0_end, ...);

    current_chunk = atomic_fetch_add(&params->threadpool->current_chunk, 1);
}
```

**Critical insight**:
- Threads process **DIFFERENT PARTS** of the same tensors
- Thread 0 works on rows 0-N
- Thread 1 works on rows N-2N
- Thread 2 works on rows 2N-3N
- etc.

---

## The Dilemma

### Option A: Log Once Per Operation (ith == 0) ← CURRENT

**Current code**:
```c
if (params->ith == 0) {  // Only thread 0 logs
    entry.tensor_ptr = (uint64_t)src0->data;  // Entire tensor pointer
    entry.size_bytes = (uint32_t)ggml_nbytes(src0);  // Entire tensor size
    tensor_trace_log(&entry);
}
```

**Pros**:
- ✅ No duplicate log entries
- ✅ Simple and clean
- ✅ One entry per operation
- ✅ Smaller log files

**Cons**:
- ❌ Loses per-thread timing information
- ❌ Can't see which thread triggered page faults
- ❌ Doesn't utilize thread_id field in log entry
- ❌ Wastes thread-local buffers (only one thread uses them)

### Option B: Log From All Threads

**Alternative code**:
```c
// NO if (ith == 0) check - all threads log
entry.tensor_ptr = (uint64_t)src0->data;
entry.size_bytes = (uint32_t)ggml_nbytes(src0);
entry.thread_id = tensor_trace_get_thread_id();  // Different for each thread
tensor_trace_log(&entry);
```

**Pros**:
- ✅ Captures per-thread access timing
- ✅ Can correlate which thread caused page faults
- ✅ Utilizes thread_id field
- ✅ Justifies thread-local buffers design
- ✅ More detailed for analysis

**Cons**:
- ❌ Multiple log entries for same tensor (one per thread)
- ❌ Larger log files (8 threads = 8× entries)
- ❌ Need to deduplicate during analysis

---

## What Do We Actually Want to Track?

### From Requirements (02JAN.md Decision 5):

> "I would love to have the REAL memory access, this way we could have the granularity of even single operations to log them"

> "And yes all KV cache operations and even input embeddings accessing would be lovely to keep access of"

**Goal**: Track memory access patterns to:
1. Identify hot/cold parameters (which parts of GGUF file are accessed)
2. Correlate with blktrace (page faults, disk I/O)
3. Analyze cache usage patterns
4. Understand thread behavior

### Key Architecture Decisions:

1. **We designed thread-local buffers** - suggests per-thread logging
2. **We included thread_id in log entry** - suggests per-thread tracking
3. **We're logging tensor_ptr (not byte offsets)** - suggests operation-level granularity

---

## Critical Questions

### 1. Do Different Threads Access Different Memory Pages?

**YES** - if the tensor is large enough and threads work on different chunks:
- Thread 0: accesses bytes 0 - N
- Thread 1: accesses bytes N - 2N
- These could be on **different memory pages** → **different page faults**

**For page fault correlation**: We need to know WHICH THREAD accessed WHICH PART.

But our current logging doesn't track byte ranges! We log:
```c
entry.tensor_ptr = (uint64_t)src0->data;  // Start of tensor
entry.size_bytes = (uint32_t)ggml_nbytes(src0);  // Entire tensor size
```

This says "the entire tensor was accessed" - not "thread 3 accessed bytes 1000-2000".

### 2. What Granularity Do We Need?

**Three levels**:

**A. Operation-level** (coarse):
- "MUL_MAT operation accessed tensor blk.5.attn_q.weight"
- Log once per operation (ith == 0)
- Sufficient for: "which tensors are hot/cold"

**B. Thread-level** (medium):
- "Thread 2 accessed tensor blk.5.attn_q.weight at time T"
- Log from all threads
- Sufficient for: "thread-level timing, which thread triggered page faults"

**C. Byte-range level** (fine):
- "Thread 2 accessed bytes 1000-2000 of tensor X"
- Need to log byte ranges (not currently implemented)
- Sufficient for: "precise page fault correlation, cache line analysis"

### 3. What Does blktrace Correlation Actually Need?

**blktrace output**:
```
  8,0    3        1     0.000000000  2000  A   R 12345678 + 8 <- (259,0) 12345678
  8,0    3        2     0.000000056  2000  Q   R 12345678 + 8 [python]
  8,0    3        3     0.000000126  2000  G   R 12345678 + 8 [python]
```

Shows: which **sector** was read, at what **time**, by which **process/thread**.

**To correlate**:
- We need: timestamp, thread_id, file_offset
- We have: ✅ timestamp_ns, ✅ thread_id, ✅ file_offset (via tensor_ptr lookup)
- **BUT**: We don't know which PART of the tensor each thread accessed!

---

## The Real Problem

### Our current approach logs:

```
[Thread 0, T=100ns]: Accessed src0 (entire tensor, 100MB)
[Thread 0, T=100ns]: Accessed src1 (entire tensor, 50MB)
```

### What we might need for page fault correlation:

```
[Thread 0, T=100ns]: Accessed src0[bytes 0-12MB]
[Thread 1, T=101ns]: Accessed src0[bytes 12-25MB]
[Thread 2, T=102ns]: Accessed src0[bytes 25-37MB]
...
```

**This requires knowing the byte range each thread works on!**

But we don't currently log that. We'd need to extract `ir0_start, ir0_end` from the chunking logic.

---

## Recommendation

### Option 1: Keep ith == 0, Accept Coarse Granularity (SIMPLE)

**For MVP Phase 1**:
- Log once per operation (current approach)
- Accept that we can't track per-thread page faults
- Sufficient for: "which tensors are accessed, how often, in what order"
- **Rationale**: Start simple, add detail later if needed

**Pros**: Simple, works NOW, achieves main goal (hot/cold parameter analysis)
**Cons**: Can't correlate per-thread page faults

### Option 2: Log From All Threads (MEDIUM COMPLEXITY)

**Remove `if (ith == 0)` check**:
- Log from every thread
- Deduplicate during analysis (group by tensor_ptr + timestamp bucket)
- Justifies thread_id field and thread-local buffers

**Pros**: Per-thread timing, better blktrace correlation
**Cons**: More log data, need smarter analysis

### Option 3: Log Byte Ranges (COMPLEX, FUTURE)

**Add byte range tracking**:
```c
entry.access_offset_start = chunk_start_bytes;
entry.access_offset_end = chunk_end_bytes;
```

**Pros**: Perfect granularity, precise page fault correlation
**Cons**: Requires refactoring, larger log entries, more complexity

---

## Decision Required

**Questions for user**:

1. **Primary goal**: Are we tracking "which tensors are accessed" (operation-level) or "which parts by which threads" (thread-level)?

2. **blktrace correlation**: Do we need to know which THREAD triggered which page fault, or just that the TENSOR was accessed?

3. **Log size trade-off**: Are we okay with 8× larger logs (if 8 threads) for per-thread detail?

4. **MVP scope**: Should Phase 1 be simple (ith == 0) with option to enhance later?

---

## My Recommendation for Phase 1 MVP

**Keep `if (ith == 0)` for now**, because:

1. **Achieves main goal**: Identify which tensors are hot/cold ✅
2. **Simple to implement**: No changes needed ✅
3. **Smaller logs**: Easier to test and debug ✅
4. **Upstreamable**: Less controversial ✅

**But document limitation**:
> "Phase 1 logs at operation granularity (one entry per operation).
> Future enhancement: per-thread logging for detailed page fault correlation."

**If needed later**: Remove `if (ith == 0)` check (5 seconds of work) and recompile.

---

## Alternative: Hybrid Approach

**Log differently based on operation importance**:

```c
#ifdef GGML_TENSOR_TRACE
    // For MUL_MAT: log from all threads (most important, worth detail)
    entry.thread_id = tensor_trace_get_thread_id();
    entry.tensor_ptr = (uint64_t)src0->data;
    tensor_trace_log(&entry);

    // For ADD, NORM, etc: log once (ith == 0) to reduce volume
#endif
```

---

**Status**: AWAITING USER DECISION
**Next Step**: User must choose Option 1, 2, 3, or Hybrid before we continue.

---

**Date**: January 2, 2026
**End of Critical Review**
