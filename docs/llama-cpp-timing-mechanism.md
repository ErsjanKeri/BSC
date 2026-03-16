# llama.cpp Timing Mechanism — Complete Reference

This document describes exactly how llama.cpp measures and reports performance timing: which variables are involved, where they are set/reset/overwritten, and what the final printed metrics actually mean.

All line numbers reference commit `91ed71d0` (build 7621).

---

## 1. The Inference Pipeline

Before understanding timing, it is critical to understand the main loop in `tools/completion/completion.cpp:587`. Everything else follows from this.

### Our Setup

- Prompt: 12 tokens (well under `n_batch = 2048`, so always processed in one `llama_decode()` call)
- Generation: 200 tokens (`n_predict = 200`)
- Context window: 131,072 (`n_ctx`), but we only use ~212 positions
- KV cache: pre-allocated for full context window (3090 MiB for small model, 4635 MiB for large). 12 layers use full attention (131k positions), 12 layers use sliding window attention (768 positions only — saves ~50% KV memory)
- Single instance, CPU-only, no parallel inference

### Key Concepts

**`embd`** — a list of token IDs (integer indices into the vocabulary). During prompt processing: `embd = [1042, 305, 7821, ...]` (12 indices). During generation: `embd = [15232]` (1 index).

**`n_batch`** (default 2048) — max tokens per single `llama_decode()` call. A "batch" just means feeding N tokens into one forward pass (one matrix multiplication of N×d instead of N separate 1×d multiplications). Only useful for prompt processing — during generation, we always decode 1 token at a time because each depends on the previous one. Irrelevant for our 12-token prompt.

**`n_remain`** — countdown of tokens left to generate. Starts at `n_predict` (200), decremented after each sampled token.

**`n_p_eval`** — total **prompt** tokens processed (the `p` stands for prompt). For us: always 12.

**`n_eval`** — total **generation** decode calls. For us: 199.

These two counters exist separately so llama.cpp can report prompt throughput and generation throughput independently. The distinction is purely for reporting — the same `llama_decode()` function runs both times.

### What `llama_decode()` Does

Every call to `llama_decode()` does the same thing:

1. **Embedding lookup**: `ggml_get_rows(tok_embd, token_ids)` — takes each token ID in `embd`, pulls its row from the embedding table (201,088 × 2880 matrix), producing a 2880-dimensional vector per token
2. **Forward pass through all layers**: attention (using KV cache) + expert MLPs + normalization
3. **Stores new K/V vectors** in the KV cache at the next available positions
4. **Outputs logits**: a float array of 201,088 values (one score per vocabulary word)

### The Main Loop

```
while (n_remain != 0) {
    if (!embd.empty()) {
        llama_decode(ctx, embd)    // forward pass: embeddings → layers → logits
        embd.clear()
        if (all prompt consumed) {
            index = sampler(logits)    // pick highest-scoring token ID
            embd.push_back(index)      // queue it for next decode
        }
    } else {
        push prompt token IDs into embd
    }
    display(embd)
}
```

**Iteration 1:** `embd` is empty. Prompt token IDs get pushed into `embd`. Display echoes the prompt text.

**Iteration 2:** `embd` has 12 token IDs. `llama_decode()` runs — looks up 12 embeddings, processes through all layers, stores 12 new K/V entries in KV cache, outputs logits. This is **prompt eval**. `embd` is cleared. Sampler picks index for **token_1** from logits → pushed into `embd`. Display prints token_1. **First generated token appears here.**

**Iteration 3:** `embd` has 1 token ID (token_1). `llama_decode()` runs — looks up 1 embedding, processes through layers (attention reads KV cache from all 13 previous positions), stores 1 new K/V entry, outputs logits. This is **generation**. Sampler picks token_2 → displayed.

**Repeat** 197 more times until `n_remain` hits 0.

### The Prompt Eval / Generation Distinction in Timing

The distinction is purely for reporting — `synchronize()` checks how many tokens were in the last decode call and adds the elapsed time to the appropriate counter (`src/llama-context.cpp:489-498`):

```cpp
if (n_queued_tokens == 1) {
    t_eval_us += ggml_time_us() - t_compute_start_us;   // generation counter
    n_eval++;
} else if (n_queued_tokens > 1) {
    t_p_eval_us += ggml_time_us() - t_compute_start_us; // prompt counter
    n_p_eval += n_queued_tokens;
}
```

`synchronize()` is called after every `llama_decode()`. Despite its name, it is not about thread synchronization — it updates timing counters and waits for backend compute to finish.

### What Goes Into Each Timing Counter

| Event | Tokens in call | Counter | Count |
|-------|---------------|---------|-------|
| Prompt decode | 12 | `t_p_eval_us` | `n_p_eval` += 12 |
| Each generated token decode | 1 | `t_eval_us` | `n_eval`++ |

`t_eval_us` includes the decode of the **first** generated token — there is no separate "first token" counter.

### MoE Expert Computation

For MoE models, `ggml_mul_mat_id` (`llama-graph.cpp:1094`) computes each (token, expert) pair **independently**. Each of the `n_expert_used × n_tokens` pairs loads and multiplies against the selected expert's weight matrix separately. (Verified in source: `build_lora_mm_id(up_exps, cur, selected_experts)` output shape is `[n_ff, n_expert_used, n_tokens]`.)

With 128 experts and 4 used per token: prompt eval (12 tokens) triggers 4×12 = 48 expert weight loads per layer; each generation step triggers 4×1 = 4 expert weight loads per layer.

---

## 2. Timing Variables

### In the model (`llama_model`, defined in `src/llama-model.h:489-490`)

| Variable | Type | Purpose |
|----------|------|---------|
| `t_start_us` | `int64_t` | Absolute timestamp (microseconds) when `llama_model_load()` began |
| `t_load_us` | `int64_t` | Elapsed time for model loading (set by RAII destructor) |

### In the context (`llama_context`, defined in `src/llama-context.h:300-317`)

| Variable | Type | Purpose |
|----------|------|---------|
| `t_start_us` | `int64_t` | Baseline timestamp for timing. Initialized from `model.t_start_us`, then **overwritten** by `perf_reset()` |
| `t_load_us` | `int64_t` | "Load time" — meaning depends on whether warmup is on/off (see Section 5) |
| `t_p_eval_us` | `int64_t` | Accumulated prompt evaluation time (microseconds) |
| `t_eval_us` | `int64_t` | Accumulated token generation time (microseconds) |
| `t_compute_start_us` | `int64_t` | Timestamp marking start of current encode/decode call |
| `n_queued_tokens` | `int64_t` | Number of tokens in current batch (used to distinguish prompt eval vs generation) |
| `n_p_eval` | `int32_t` | Total prompt tokens processed |
| `n_eval` | `int32_t` | Total generation steps (decode calls with 1 token) |
| `has_evaluated_once` | `bool` | Flag set after first synchronize(). **Never reset by perf_reset().** |

### RAII timer (`time_meas`, defined in `src/llama-impl.h:43-50`, implementation `src/llama-impl.cpp:20-26`)

```cpp
time_meas::time_meas(int64_t & t_acc, bool disable)
    : t_start_us(disable ? -1 : ggml_time_us()), t_acc(t_acc) {}

time_meas::~time_meas() {
    if (t_start_us >= 0) {
        t_acc += ggml_time_us() - t_start_us;
    }
}
```

Used in `llama_model_load()` to measure model loading time. Captures start time in constructor, adds elapsed time to accumulator in destructor.

---

## 3. The Timeline — Step by Step

### Phase A: Model Loading

**File:** `src/llama.cpp:783-790`

```cpp
static int llama_model_load(...) {
    model.t_load_us = 0;
    time_meas tm(model.t_load_us);           // A1: captures t_start_us = ggml_time_us()
    model.t_start_us = tm.t_start_us;        // A2: save absolute start time
    // ... metadata parsing, vocab, tensor loading ...
    // A3: ~time_meas destructor: model.t_load_us += elapsed
}
```

Inside model loading, the following sub-steps happen (in `src/llama-model.cpp`):

1. **Metadata + vocab parsing** — reads GGUF header, architecture, hyperparameters, vocabulary
2. **`init_mappings()`** (line 6797) — creates mmap. With MAP_POPULATE: kernel reads entire file. Without: instant (just sets up virtual address space)
3. **`load_all_data()`** (line 6858) — assigns tensor data pointers into the mmap'd region. No disk I/O
4. **`pin_compute_weights()`** (lines 6955-6968) — if `--pin-compute-weights`: iterates tensors, calls `mlock()` on attention/projection weights, skipping expert and embedding tensors

### Phase B: Context Construction

**File:** `src/llama-context.cpp:21-31`

```cpp
llama_context::llama_context(...)
    t_start_us = model.t_start_us;  // B1: inherit model's start time
    t_load_us  = model.t_load_us;   // B2: inherit model's load time
    // has_evaluated_once = false (default)
    // all other timing vars = 0
```

Then KV cache allocation (malloc), compute buffer allocation, graph reservation.

### Phase C: Warmup (if enabled)

**File:** `common/common.cpp:1271-1307`

```cpp
if (params.warmup) {
    llama_set_warmup(lctx, true);               // C1: sets cparams.warmup = true
    // decode BOS+EOS tokens                     // C2: triggers synchronize()
    llama_perf_context_reset(lctx);              // C3: RESET — zeros timing counters
    llama_set_warmup(lctx, false);               // C4: clears warmup flag
}
```

During warmup, `cparams.no_perf` is effectively true (via warmup flag), so `t_p_eval_us` and `t_eval_us` are NOT accumulated. However, the `has_evaluated_once` check in `synchronize()` (line 502-504) is NOT guarded by `no_perf`, so:
- `t_load_us` gets overwritten to `ggml_time_us() - t_start_us` (time from model load start to warmup completion)
- `has_evaluated_once` gets set to `true`

Then `perf_reset()` zeros `t_start_us`, `t_p_eval_us`, `t_eval_us`, `n_p_eval`, `n_eval`. But does NOT reset `t_load_us` or `has_evaluated_once`.

### Phase D: Pre-inference Reset

**File:** `tools/completion/completion.cpp:163-170`

```cpp
// print true model load time before the reset erases it
{
    auto perf_pre_reset = llama_perf_context(ctx);
    LOG_INF("[timing] true_model_load: %.2f ms (before perf reset)\n",
            perf_pre_reset.t_load_ms);
}

// start measuring performance timings from here
llama_perf_context_reset(ctx);
```

This is a second reset (the first was inside warmup at Phase C3, if warmup was enabled). After this reset:
- `t_start_us` = current time (new baseline)
- `t_p_eval_us` = 0, `t_eval_us` = 0, `n_p_eval` = 0, `n_eval` = 0
- `t_load_us` unchanged (NOT zeroed by reset)
- `has_evaluated_once` unchanged (NOT zeroed by reset)

### Phase E: Prompt Evaluation

**File:** `tools/completion/completion.cpp:698-711`

The prompt tokens are passed to `llama_decode()` as a batch:

```cpp
for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
    int n_eval = (int) embd.size() - i;
    // ...
    llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval));
    n_past += n_eval;
}
```

Inside `llama_decode()` (`src/llama-context.cpp:1095-1098`):
```cpp
if (t_compute_start_us == 0) {
    t_compute_start_us = ggml_time_us();       // E1: mark compute start
}
n_queued_tokens += n_tokens_all;                // E2: e.g., 12 prompt tokens
```

After computation, `synchronize()` is called (`src/llama-context.cpp:481-509`):
```cpp
// n_queued_tokens > 1, so this is prompt eval:
t_p_eval_us += ggml_time_us() - t_compute_start_us;   // E3: accumulate prompt eval time
n_p_eval += n_queued_tokens;                            // E4: count tokens

// first eval — load time overwrite:
if (n_queued_tokens > 0 && !has_evaluated_once) {       // E5
    t_load_us = ggml_time_us() - t_start_us;            // E6: OVERWRITE
    has_evaluated_once = true;                           // E7: set forever
}
```

**Critical:** At step E6, `t_start_us` is the value set by Phase D's `perf_reset()` (not the original model start). So `t_load_us` becomes "time from reset to end of first eval" — which includes prompt eval time but NOT model loading, NOT mlock, NOT warmup.

**Exception:** If warmup was ON, `has_evaluated_once` was already set to `true` in Phase C. So steps E5-E7 are skipped, and `t_load_us` retains its value from Phase C (time from model start to warmup completion).

### Phase F: Token Generation

**File:** `tools/completion/completion.cpp:737-743`

```cpp
const llama_token id = common_sampler_sample(smpl, ctx, -1);  // F1: sample
common_sampler_accept(smpl, id, true);                         // F2: accept
embd.push_back(id);                                            // F3: add to next batch
```

On the next loop iteration, this single token is decoded:
```cpp
llama_decode(ctx, llama_batch_get_one(&embd[i], 1));  // n_eval = 1
```

In `synchronize()`, since `n_queued_tokens == 1`:
```cpp
t_eval_us += ggml_time_us() - t_compute_start_us;     // F4: accumulate eval time
n_eval++;                                               // F5: count runs
```

This repeats for each generated token.

### Phase G: Performance Reporting

**File:** `common/sampling.cpp:347-390`

```cpp
llama_perf_context_data data = llama_perf_context(ctx);
const double t_end_ms = 1e-3 * ggml_time_us();
const double t_total_ms = t_end_ms - data.t_start_ms;
```

Printed output:
```
load time      = t_load_us (in ms)
prompt eval    = t_p_eval_us (in ms) / n_p_eval tokens
eval time      = t_eval_us (in ms) / n_eval runs
total time     = now - t_start_us (time since last reset)
unaccounted    = total - sampling - prompt_eval - eval
```

---

## 4. What `perf_reset()` Does and Doesn't Do

**File:** `src/llama-context.cpp:2129-2134`

```cpp
void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();    // new baseline
    t_eval_us   = n_eval = 0;        // zero generation timing
    t_p_eval_us = n_p_eval = 0;      // zero prompt timing
    n_reused    = 0;                  // zero graph reuse counter
}
```

| Variable | Reset? | New value |
|----------|--------|-----------|
| `t_start_us` | YES | `ggml_time_us()` (current time) |
| `t_p_eval_us` | YES | 0 |
| `t_eval_us` | YES | 0 |
| `n_p_eval` | YES | 0 |
| `n_eval` | YES | 0 |
| `n_reused` | YES | 0 |
| **`t_load_us`** | **NO** | unchanged |
| **`has_evaluated_once`** | **NO** | unchanged |
| **`t_compute_start_us`** | **NO** | unchanged |

---

## 5. What `load_time` Actually Reports

This is the most confusing metric. Its meaning depends on the execution path:

### Case 1: Warmup ON (default behavior)

1. Warmup's `synchronize()` sets `t_load_us = now - model_start` and `has_evaluated_once = true`
2. `perf_reset()` does NOT clear either
3. Real inference's `synchronize()` sees `has_evaluated_once = true` → no overwrite

**Result:** `load_time` = time from model load start to end of warmup decode. Includes: metadata parsing + mmap (+ MAP_POPULATE if on) + context construction + warmup forward pass.

### Case 2: Warmup OFF (`--no-warmup`)

1. No warmup, so `has_evaluated_once` remains `false`
2. `perf_reset()` sets `t_start_us = now`
3. First real inference's `synchronize()` sees `has_evaluated_once = false` → `t_load_us = now - t_start_us`

**Result:** `load_time` = time from reset to end of first prompt eval. Everything before the reset (model loading, mmap, mlock, context construction) is invisible.

### The inconsistency

`load_time` means completely different things depending on whether warmup is on or off:
- **Warmup ON:** captures model loading + warmup (seconds to minutes)
- **Warmup OFF:** captures just prompt eval, hiding all loading costs

This is why we added `[timing] true_model_load` instrumentation — to always have a reliable measurement of the pre-inference overhead.

---

## 6. What Our Instrumentation Adds

We added timing prints in two locations:

### In `src/llama-model.cpp` (inside `load_tensors()`)

After `init_mappings()` (line ~6797):
```
[timing] init_mappings_done: X.XX ms since model load start
```

After `load_all_data()` (line ~6858):
```
[timing] load_all_data_done: X.XX ms since model load start
```

After `pin_compute_weights()` (line ~6968):
```
load_tensors: pin_compute_weights: attempted N tensors, locked X.X MiB, failed N tensors, took X.XX ms
[timing] pin_compute_weights_done: X.XX ms since model load start
```

### In `tools/completion/completion.cpp` (line 164)

Right before the perf reset:
```
[timing] true_model_load: X.XX ms (before perf reset)
```

This captures the time from model load start to the reset point, which includes everything hidden by the reset: metadata, mmap/MAP_POPULATE, mlock, context construction, warmup (if enabled).

---

## 7. Summary: What Each Reported Metric Means

| Metric | What it measures | Includes loading costs? |
|--------|-----------------|------------------------|
| `load time` (warmup ON) | Model start → end of warmup | YES (but mixed with warmup) |
| `load time` (warmup OFF) | Reset → end of first prompt eval | NO — hidden by reset |
| `prompt eval time` | Time inside `llama_decode()` for multi-token batches | No loading costs, but includes page fault I/O within decode |
| `eval time` | Time inside `llama_decode()` for single-token batches | No loading costs, but includes page fault I/O within decode |
| `total time` | Reset → end of last token | NO — excludes pre-reset costs |
| `unaccounted time` | total - sampling - prompt_eval - eval | Should be tiny (<1%) |
| `[timing] true_model_load` | Model start → reset point | YES — our instrumentation |
| `[timing] init_mappings_done` | Model start → after mmap | YES — our instrumentation |
| `[timing] pin_compute_weights_done` | Model start → after mlock | YES — our instrumentation |

---

## 8. Plan: What We Need to Change

### Problem

The current timing mechanism conflates multiple phases and hides costs behind a `perf_reset()`. For our thesis, we need **granular, per-phase timing** that clearly separates:

1. **Model loading** — metadata parsing, GGUF header, vocab
2. **Memory mapping** — mmap setup (and MAP_POPULATE if enabled)
3. **Memory pinning** — `mlock()` of compute weights (if `--pin-compute-weights`)
4. **Context construction** — KV cache allocation, compute buffer allocation, graph reservation
5. **Warmup** — warmup forward pass (if enabled)
6. **Prompt eval** — forward pass for prompt tokens (includes page fault I/O on first access)
7. **Token generation** — per-token decode steps (includes page fault I/O)

### What needs to happen

- **Remove or bypass the perf_reset()** for our measurements, OR add separate counters that are not affected by the reset
- **Add per-phase timestamps** that are printed regardless of warmup/reset state
- **Separate page fault I/O from compute** within prompt eval and eval — this is the hardest part, as page faults happen inside `llama_decode()` and are invisible to the timing code. Options:
  - Read `/proc/self/stat` major fault counter before and after each `llama_decode()` call
  - Use `perf stat` externally
  - Use blktrace for disk-level tracing
- **Parse these new metrics in our Python runner** and record them in the CSV

### Open question

Should we modify llama.cpp's timing infrastructure directly (risk: diverging from upstream, harder to maintain), or wrap our measurements externally in the Python runner (risk: less precise, can't measure internal phases)? A hybrid approach — minimal instrumentation inside llama.cpp for per-phase timing, external measurement for page faults — may be the most practical path.
