# llama.cpp Architecture: Inference Flow

**Last Updated:** 2026-03-07
**Model:** gpt-oss-20b-F16.gguf (Mixture of Experts)

---

## Overview

Inference happens in TWO distinct phases:

**Phase 1: Graph Building (Symbolic)**
- Creates computation DAG (directed acyclic graph)
- Defines WHAT operations to run and their connections
- NO actual computation, NO memory access to weights
- Result: `ggml_cgraph*` structure with nodes

**Phase 2: Graph Execution (Actual Computation)**
- Traverses graph and RUNS each operation
- Reads weights from memory, performs math
- Writes results to output tensors
- **OUR HOOKS trigger here**

---

## Token Generation Flow (Critical Foundation)

**Before understanding graphs, understand WHAT they're computing.**

### Example: "Hello there friend" → generate next token

**Token IDs:** `[15043, 612, 4333]`

---

### Phase A: Prompt Processing (Prefill)

**Input:** 3 tokens `[15043, 612, 4333]`

**Step 1: Embedding Lookup**
```
Token IDs: [15043, 612, 4333]
           ↓ lookup in embedding table [vocab_size=32000, n_embd=4096]
Embeddings: [3, 4096] ← 3 vectors, 4096 floats each
```

**Step 2: Pass Through 32 Layers**

Each layer processes ALL 3 tokens together:
```
Input:  [3, 4096]
        ↓
Attention (tokens see each other)  [3, 4096] → [3, 4096]
Residual add                       [3, 4096] + [3, 4096]
        ↓
MoE FFN (each token routed independently)  [3, 4096] → [3, 4096]
Residual add                                [3, 4096] + [3, 4096]
        ↓
Output: [3, 4096] (transformed representations)
```

**Critical: Shape stays [3, 4096] through ALL layers!**

**Step 3: Final Normalization**

**File:** `llama.cpp/src/models/openai-moe-iswa.cpp:110`
```cpp
cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
```

**What is RMS Normalization?**

NOT about vector length! Stabilizes values:

```
Before: [50000, -30000, 80000, ..., -40000] ← huge values
        ↓
1. RMS = sqrt(mean(x²)) = sqrt((50000² + 30000² + ...) / 4096)
2. Normalize: x / RMS
3. Scale: normalized * learned_weights
        ↓
After:  [0.8, -0.5, 1.2, ..., -0.7] ← stable range
```

**Applied PER TOKEN independently:**
- Token 0's 4096 values normalized separately
- Token 1's 4096 values normalized separately
- Token 2's 4096 values normalized separately

**Why?** Prevents numerical instability, ensures stable input to output layer.

**Step 4: Output Projection (NOT a layer!)**

**File:** `llama.cpp/src/models/openai-moe-iswa.cpp:118`
```cpp
cur = build_lora_mm(model.output, cur);
```

**What is `model.output`?**

**File:** `llama.cpp/src/llama-model.cpp` (search "output")
```cpp
model.output = create_tensor({n_embd, n_vocab});  // [4096, 32000]
```

**A weight matrix, NOT a transformer layer!**

**Computation:**
```
hidden: [3, 4096] ← representations
output_weights: [4096, 32000] ← vocabulary projection
logits = hidden @ output_weights = [3, 32000]
```

**What are logits?**

Raw scores for EVERY vocabulary word:
```
logits[0, :] = [32000 scores] for position after "Hello"
logits[1, :] = [32000 scores] for position after "Hello there"
logits[2, :] = [32000 scores] for position after "Hello there friend"
```

Example logits[2, :]:
```
[-2.3, 0.8, ..., 5.2, ..., -1.1]  ← 32000 values
                  ↑ score 5.2 at position 891 (word "how")
```

**Storage:**
- float32: 32,000 × 4 bytes = **128 KB per token**
- float16: 32,000 × 2 bytes = **64 KB per token**

**Step 5: Sample Next Token**

**File:** `llama.cpp/src/llama-context.cpp` (decode logic)
```cpp
// Use ONLY last token's logits
float* logits_out = &ctx->logits[n_vocab * (n_tokens - 1)];
```

**Why only last token?**
- We want: "What comes AFTER 'friend'?"
- Don't care about: "What comes after 'Hello'" (we already know: "there")

```
logits[2, :] = [-2.3, 0.8, ..., 5.2, ..., -1.1]
               ↓ find highest (or sample with temperature)
token_id = 891 (word "how")
```

**No "inverse lookup"!** Sampling directly gives token ID.

---

### Phase B: Generation (Autoregressive)

**Previous sequence:** `[15043, 612, 4333, 891]` = "Hello there friend how"

**Generate next token:**

**WITHOUT KV cache (slow):**
```
Input: ALL 4 tokens [4, 4096]
Process through 32 layers
Output: logits [4, 32000]
Use: logits[3, :] (last position)
```

**WITH KV cache (actual implementation):**
```
Already computed: tokens [15043, 612, 4333] ✓
Only compute: token [891]
Input: [1, 4096] ← ONLY new token
Process through 32 layers (using cached K,V from previous tokens)
Output: logits [1, 32000]
Use: logits[0, :] (only position)
```

**This is why input is [1, 4096] during generation!**

---

### Input Shape Summary

| Phase | Input Tokens | Shape | Why |
|-------|-------------|-------|-----|
| Prompt | "Hello there friend" (3) | [3, 4096] | Process all at once |
| Gen step 1 | "how" (1 new) | [1, 4096] | KV cache allows only new token |
| Gen step 2 | "are" (1 new) | [1, 4096] | Same reason |

**MoE routing happens PER TOKEN:**
- [3, 4096] input → each of 3 tokens routed independently (can pick different experts!)
- [1, 4096] input → single token routed to 8 experts

---

## 1. Graph Building (Symbolic Phase)

### Entry Point

**File:** `llama.cpp/src/llama-context.cpp:822-834`
```cpp
if (!graph_reuse_disable && res->can_reuse(gparams)) {
    n_reused++;  // Reuse if graph structure unchanged
} else {
    res->reset();  // ← Line 827: WIPE graph
    ggml_backend_sched_reset(sched.get());
    gf = model.build_graph(gparams);  // ← Line 834: REBUILD
}
```

**Graph lifecycle:**
- Graph stored in: `ggml_cgraph* gf` (inside `llm_graph_result* res`)
- Rebuilt per token (unless structure identical to previous)
- Line 827: `res->reset()` clears previous graph
- Line 834: `model.build_graph()` creates new graph

**Architecture Detection:**
- `llama.cpp/src/llama-arch.cpp:104`: `gpt-oss` → `LLM_ARCH_OPENAI_MOE`
- Dispatches to: `llama.cpp/src/models/openai-moe-iswa.cpp`

---

### What's in the Graph?

**Graph contains OPERATIONS (nodes), NOT data:**

Each node describes:
- Operation type (MUL_MAT, ADD, ROPE, etc.)
- Input tensor pointers (where to read from)
- Output tensor pointer (where to write)

**Example node:**
```cpp
node = ggml_mul_mat(ctx0, weight_tensor, input_tensor);
```
Creates node storing: "multiply `weight_tensor` with `input_tensor`, write to `node`"

**Important:** Tensors are just metadata at this point (shapes, pointers). Actual data not accessed yet.

---

### Matrix Multiply: LoRA-Aware

**File:** `llama.cpp/src/llama-graph.cpp:668-692`

**Why `build_lora_mm` instead of plain `ggml_mul_mat`?**

```cpp
ggml_tensor * build_lora_mm(ggml_tensor * w, ggml_tensor * cur) {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);  // Base matmul node

    // If LoRA adapters loaded, add correction:
    for (const auto & lora : *loras) {
        ab_cur = ggml_mul_mat(B, ggml_mul_mat(A, cur));  // LoRA low-rank
        res = ggml_add(res, scale * ab_cur);
    }
    return res;
}
```

**LoRA** = Low-Rank Adaptation (fine-tuning)
- If NO LoRA: just creates `ggml_mul_mat` node
- If LoRA loaded: creates `matmul + add(scaled_lora)` subgraph
- Still just nodes - no computation yet

---

### 1.1 Model-Specific Graph Builder

**File:** `llama.cpp/src/models/openai-moe-iswa.cpp:3-124`

For each layer (lines 16-107):

**1.1.1 Attention Block** (lines 19-71)
- Creates nodes for: Q/K/V projections, RoPE, attention, output projection
- No computation, just node creation

**1.1.2 MoE FFN Block** (lines 86-98)
- Calls `build_moe_ffn(cur, ...)` to create MoE subgraph

---

### 1.2 MoE Subgraph Building

**Context: What Goes Into MoE?**

**File:** `llama.cpp/src/models/openai-moe-iswa.cpp:77-98`
```cpp
ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);  // Residual after attention
cur = build_norm(cur, ...);                          // Normalize
cur = build_moe_ffn(cur, ...);                       // MoE processes this
```

**Input to MoE:** Residual stream (hidden states after attention)
- Shape: `[n_tokens, 4096]`
- n_tokens = 3 during prompt, 1 during generation
- Each token processed independently by MoE!

---

**File:** `llama.cpp/src/llama-graph.cpp:936-1200`

**What is "Gate"?**

The term "gate" appears in multiple contexts:

1. **LSTM gate** (information flow control): sigmoid output (0-1) that blocks/passes data
   - Forget gate, input gate, output gate
2. **SwiGLU gate** (`ffn_gate_exps`): activation that modulates features
   - `output = swish(gate) * up`
3. **"gate_inp"** (MISNOMER): Actually the **router** - selects which experts to use
   - Should be called `ffn_router`, but codebase uses `gate_inp`
   - Does NOT output 0-1 values, outputs expert selection indices

---

**MoE Components in Memory:**

**File:** `llama.cpp/src/llama-model.cpp:2747-2750`
```cpp
layer.ffn_gate_inp  = [..., {n_embd, n_expert}];        // [4096, 32] - router
layer.ffn_gate_exps = [..., {n_embd, n_ff, n_expert}];  // [4096, 16384, 32] - gate weights
layer.ffn_up_exps   = [..., {n_embd, n_ff, n_expert}];  // [4096, 16384, 32] - up weights
layer.ffn_down_exps = [..., {n_ff, n_embd, n_expert}];  // [16384, 4096, 32] - down weights
```

**Key:** All 32 experts stored in single 3D tensor (third dimension = expert index)

---

**MoE Nodes Created (Symbolic - No Execution):**

**1.2.1 Router Node** (line 964)
```cpp
logits = build_lora_mm(gate_inp, cur);
```
**Purpose:** Compute score for each expert (per token)

**Inputs:**
- `gate_inp`: `[4096, 32]` (router weight matrix)
- `cur`: `[n_tokens, 4096]` (hidden states)

**Examples:**

Prompt (3 tokens):
```
cur: [3, 4096]
logits: [3, 32] ← 32 scores for EACH of 3 tokens
  Row 0: [0.1, -0.5, 2.3, ...] ← token 0's expert scores
  Row 1: [0.8, 1.2, -0.3, ...] ← token 1's expert scores
  Row 2: [1.1, 0.2, 3.1, ...] ← token 2's expert scores
```

Generation (1 token):
```
cur: [1, 4096]
logits: [1, 32] ← 32 scores for this token
  Row 0: [0.1, -0.5, 2.3, 0.8, 1.1, ..., 0.9]
```

**Each token independently scored against all 32 experts!**

---

**1.2.2 Expert Selection Node** (line 1038)
```cpp
selected_experts = ggml_argsort_top_k(selection_probs, n_expert_used);
```
**Purpose:** Pick top K experts

**Input:** `selection_probs = [1, 32]` (derived from logits)

**Output:** `selected_experts = [8, 1]` (expert IDs as integers)
- Example: `[2, 30, 3, 0, 15, 21, 24, 28]` (top 8 expert IDs)
- These IDs determine which expert weights to access during execution

**Critical:** Expert selection happens during EXECUTION, not now. This just creates the node.

---

**1.2.3 Expert Weight Extraction** (line 1050)
```cpp
weights = ggml_get_rows(probs, selected_experts);
```
**Purpose:** Get mixing weights for selected experts

**Output:** `weights = [1, 8, 1]` (how much to weight each expert's output)

---

**1.2.4 Expert Computation Nodes** (lines 1089-1160)

**File:** `llama.cpp/ggml/src/ggml.c:3203-3210` (mul_mat_id documentation)

```cpp
up      = build_lora_mm_id(up_exps,   cur, selected_experts);
gate    = build_lora_mm_id(gate_exps, cur, selected_experts);
cur     = ggml_swiglu_oai(gate, up);
experts = build_lora_mm_id(down_exps, cur, selected_experts);
```

**What is `ggml_mul_mat_id`?**

"Indirect matrix multiply" - multiplies with selected slices of 3D tensor

**Signature:**
```c
as   -> [cols, rows, n_expert]         // ALL expert matrices stacked
b    -> [cols, n_expert_used, n_tokens] // input
ids  -> [n_expert_used, n_tokens]       // which expert to use (i32)
result -> [rows, n_expert_used, n_tokens]

// Computation: result[:, e, t] = as[:, :, ids[e,t]] @ b[:, e, t]
```

**Example with actual dimensions:**
```cpp
up = ggml_mul_mat_id(up_exps, cur, selected_experts);
```
- `up_exps`: `[4096, 16384, 32]` - ALL 32 experts in memory
  - `up_exps[:, :, 0]` = expert 0's weights `[4096, 16384]`
  - `up_exps[:, :, 2]` = expert 2's weights `[4096, 16384]`
  - ...
- `cur`: `[1, 4096]` (hidden state, broadcasted)
- `selected_experts`: `[8, 1]` = `[2, 30, 3, 0, 15, 21, 24, 28]`
- `up`: `[16384, 8, 1]` (output from 8 experts)

**During execution, for each slot e:**
```
expert_id = selected_experts[e]  // e.g., selected_experts[0] = 2
result[:, e, 0] = up_exps[:, :, expert_id] @ cur[0, :]
                  ^^^^^^^^^^^^^^^^^^^^^^^ selects expert 2's weights from memory
```

**Key:** Only selected 8 experts accessed! Remaining 24 experts untouched in memory.

---

**1.2.5 Weighted Sum Node** (line 1171)
```cpp
experts = ggml_mul(experts, weights);
```
**Purpose:** Multiply each expert's output by its weight

---

**1.2.6 Expert Aggregation Nodes** (lines 1180-1193)
```cpp
for (i = 1; i < n_expert_used; i++) {
    moe_out = ggml_add(moe_out, cur_experts[i]);
}
```
**Purpose:** Sum all weighted expert outputs

**Result:** `moe_out = [1, 4096]` (back to hidden dimension)

---

**Graph Building Complete:** All nodes created, but NO computation, NO memory access yet.

---

## 2. Graph Execution (Computation Phase)

### 2.1 Entry Point

**File:** `llama.cpp/src/llama-context.cpp:878`
```cpp
status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
```

**File:** `llama.cpp/src/llama-context.cpp:1566`
```cpp
status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
```
- NOW computation actually starts
- Traverses graph nodes in execution order

---

### 2.2 Operation Execution Loop

**File:** `llama.cpp/ggml/src/ggml-backend.c`

For each node in graph:
1. Determine which backend (CPU/GPU)
2. Call backend's compute function
3. Backend reads input tensors from memory
4. Backend performs computation
5. Backend writes output tensor

**THIS IS WHERE MEMORY ACCESS HAPPENS**

---

## Our Fork Modifications

### Tensor Tracing Infrastructure

**Files:**
- `llama.cpp/ggml/include/tensor_trace.h` - data structures
- `llama.cpp/ggml/src/tensor_trace.c` - implementation

**Key Structure:**

```c
struct TensorAccessLog {
    uint64_t timestamp_ns;
    uint32_t token_id;              // which token being processed
    uint16_t layer_id;              // which layer
    uint8_t  operation_type;        // GGML_OP_MUL_MAT, etc.
    uint8_t  phase;                 // PROMPT vs GENERATE

    char dst_name[128];             // output tensor name
    struct SourceTensorInfo sources[4];  // input tensors

    int32_t expert_ids[16];         // MoE: which experts used
    uint8_t num_experts;
};
```

**Hooks trigger during Phase 2 (execution)** - captures which tensors accessed when.

---

## Next: Investigate Execution Phase

**TODO:**
- Where exactly in backend execution do hooks trigger?
- How are expert IDs extracted during `ggml_mul_mat_id()` execution?
- How does memory access tracing work?