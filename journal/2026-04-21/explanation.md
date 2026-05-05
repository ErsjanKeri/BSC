# MoE FFN graph + mul_mat_id + callbacks — visual walkthrough

Companion to `plots/07_graph_variants.png` and `plots/08_mul_mat_id_decode.png`.

---

## 1. What `ggml_mul_mat_id` is — with exact decode dimensions

See `plots/08_mul_mat_id_decode.png` for the visual.

In decode (`n_tokens = 1`), for a single layer, the inputs are:

| Tensor | Shape | Meaning |
|---|---|---|
| `x` | `[2880, 1]` | The single token's hidden state, post-attention |
| `selected_experts` | `[4]` | Router output: which 4 of the 128 experts were chosen (e.g. `{71, 4, 108, 33}`) |
| `up_exps` | `[2880, 2880, 128]` | **Bank of 128 weight matrices** stacked along dim-2. `up_exps[:, :, e]` is expert `e`'s up-projection matrix |

The call `ggml_mul_mat_id(up_exps, x, selected_experts)` produces an output of shape `[2880, 4, 1]`.

**Inside the op, for each `i` ∈ {0, 1, 2, 3}:**
```
e        = selected_experts[i]          (pick one of the 128 matrices)
W_e      = up_exps[:, :, e]              (2880×2880, the expert's up matrix)
out[:,i] = W_e @ x[:, 0]                 (matrix-vector, result is [2880])
```

All 4 iterations run inside a single op call, cooperatively across the 8 ggml worker threads (each thread owns a row-slice of the 2880 output dimension).

**The critical structural property**: these 4 iterations are **inside one atomic graph node**. There is no scheduler-visible boundary between iteration `i=1` and iteration `i=2`. You cannot insert "wait for expert `i=2`'s weights to arrive" between them. That's the constraint that motivates the fused op in pipeline v3.

## 2. The three `mul_mat_id` calls per layer

A single MoE FFN layer issues **three** of these calls back-to-back (right panel of figure 08):

1. **`up = mul_mat_id(up_exps, x, selected)`** → `[2880, 4, 1]`
2. **`gate = mul_mat_id(gate_exps, x, selected)`** → `[2880, 4, 1]`
3. **`down = mul_mat_id(down_exps, act, selected)`** → `[2880, 4, 1]` (where `act = swiglu(gate, up)`)

Then `moe_out = Σᵢ router_weights[i] · down[:, i, 0]` → `[2880, 1]`.

**Per layer per decode token**: 3 `mul_mat_id` calls × 4 experts read internally = **12 expert-projection slices** touched (= the 432 working-set slots we've been talking about, for 36 layers).

**Boundaries where external code can intervene**: only *between* these three ops (graph-node boundaries). *Inside* each op, we cannot intervene.

## 3. The three graph variants (figure 07)

See `plots/07_graph_variants.png`.

### LEFT — Stock mmap path (what llama.cpp does by default)

Just the three `mul_mat_id` nodes + `swiglu` + weighted sum. Expert weights are read via `tensor->data` which points into the mmap'd GGUF file. If that byte range isn't in the kernel page cache, the CPU thread stalls on a page fault while the kernel pulls a 4 KiB page from NVMe. Every miss → one synchronous stall. The op finishes when all the weight pages for all 4 selected experts are resident.

No external hooks for I/O. What you see is what ggml runs.

### MIDDLE — `io_uring + overlap` path

Same three `mul_mat_id` nodes remain (orange), but **our code inserts two new nodes of a special ggml type called `ggml_map_custom1/2` (teal)**:

- **Before the up/gate pair** — a `ggml_map_custom1` callback that fires before `mul_mat_id(up)` runs. Inside this callback, our C function submits 8 io_uring reads (up+gate for all 4 experts), does a sync `wait`, and patches each expert's `tensor->extra` pointer to point at the freshly loaded cache slot. Also queues 4 more async submits for the `down` projection.
- **Before the down `mul_mat_id`** — a `ggml_map_custom2` callback that waits for the down I/O (usually near-zero residual time because we submitted it ~3 ms ago, overlapped with the up+gate+swiglu compute).

The three `mul_mat_id` ops **are unchanged** — we just patched them at `ggml-cpu.c:1628` to read from `tensor->extra` instead of `tensor->data` when `extra` is non-NULL. When our callbacks set `extra` to a cache-slot pointer, the op reads from our user-space buffer instead of mmap. Backwards-compatible: if `extra` is NULL (every non-MoE tensor), the op behaves exactly as before.

Weights flow: NVMe → `io_uring_read` (O_DIRECT, bypasses kernel page cache) → our `posix_memalign` anonymous cache buffer → `tensor->extra` points at slot → `mul_mat_id` reads from there.

### RIGHT — Pipeline v3 path

Here we **throw out the three `mul_mat_id` nodes entirely** and replace them with **one single node of type `ggml_map_custom3`** (green). That one node IS the whole MoE FFN for this layer.

Inside the callback (`llama_moe_pipeline_compute_fused_v3`), all 8 ggml threads enter simultaneously and we implement:
- 8 io_uring submits tagged per-(expert, projection-group) — `tag 2e` = upgate, `tag 2e+1` = down
- a `wait_any_tag_ready` across upgate tags → which expert's up+gate finished first dictates compute order
- a compute loop where each iteration does up/gate/swiglu/down for one expert at a time, using 8-thread row-split matmul
- a per-expert `wait_expert_tagged` just before the `down` matmul
- a final canonical-order accumulation (so output is bit-exact vs overlap)

**We can insert io_uring waits at per-expert granularity** because we implement the compute loop ourselves — we're no longer constrained to the atomic "inside mul_mat_id" boundary.

## 4. How `ggml_map_custom*` callbacks actually work

This is the ggml mechanism that makes both overlap and v3 possible:

```
ggml_tensor * ggml_map_custom1/2/3(
    ggml_context * ctx,
    ggml_tensor * a,
    [ggml_tensor * b,]
    [ggml_tensor * c,]
    ggml_custom_op_t function_pointer,
    int n_threads,                // how many threads cooperate
    void * userdata);             // arbitrary pointer passed to callback
```

When the ggml scheduler walks the graph in topological order and reaches a `map_custom*` node, it:

1. Pauses normal op dispatch (no `mul_mat`, no `rope`, no `swiglu` — just our function)
2. Dispatches `n_threads` workers into our `function_pointer` with `(ith, nth, userdata, a_tensor, b_tensor, c_tensor, output_tensor)`
3. Our function does whatever it wants — read user_data, touch any memory, syscall, whatever
4. When all threads return, the scheduler moves to the next graph node

**This is how we smuggle io_uring into the graph.** From ggml's perspective our callback just takes some tensors in and produces a tensor out (even if the output tensor is dummy). From OUR perspective we've been given a scheduling slot where we can do I/O, barriers, and anything else we want.

**Overlap uses it as a "sandwich filler"**: normal `mul_mat_id` → callback (do I/O) → normal `mul_mat_id`. Sandwich stays 3 layers of bread; we just insert butter.

**V3 uses it as a "whole meal"**: instead of three `mul_mat_id` breads + swiglu butter, we write ONE map_custom3 node that does the whole thing — we get to decide when to do I/O between our own compute steps, not just between someone else's ops.

## 5. Summary — three levels of control

| Level | Where we can hook | What that enables |
|---|---|---|
| **Weight-pointer substitution** (patch to `mul_mat_id`) | Inside any `mul_mat_id` call, the weight pointer comes from `tensor->extra` when set | Make ggml read from our user-space cache instead of mmap. Backward-compatible (NULL → original behavior). |
| **Callback between ops** (`ggml_map_custom1/2` in overlap path) | Between the three `mul_mat_id` nodes of the standard MoE graph | Submit/wait io_uring reads at projection-group granularity (upgate vs down). Overlap down's I/O with up+gate's compute. |
| **Full fused op replacement** (`ggml_map_custom3` in v3) | Replace the entire 3-mul_mat_id sequence with one node whose implementation we write | Per-expert I/O granularity: wait for expert 0's upgate, compute expert 0, then wait for expert 1 — while 2 and 3's I/O finishes in the background. Cannot be done at the mul_mat_id boundary because mul_mat_id is atomic over all 4 experts. |

Each successive level lifts the granularity constraint of the previous one. V3 is the deepest hook — we control the entire per-layer compute + I/O interleaving ourselves.
