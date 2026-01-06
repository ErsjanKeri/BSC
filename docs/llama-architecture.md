# llama.cpp Architecture: Complete Internal Flow

**Last Updated**: January 6, 2026
**For**: Understanding tensor tracing instrumentation and inference flow

This document provides a comprehensive, verified walkthrough of llama.cpp's architecture, from entry point to final token output, with precise file locations and function call chains.

---

## 🗂️ Project Structure (Verified)

```
llama.cpp/
├── src/
│   ├── llama.cpp                    ← CORE: Model loading, context, inference orchestration
│   ├── llama-impl.h                 ← Internal utilities
│   ├── llama-sampling.cpp           ← Token sampling logic
│   └── models/
│       ├── deepseek.cpp             ← DeepSeek-specific graph building
│       ├── llama.cpp                ← LLaMA-specific graph building
│       └── ... (other architectures)
│
├── ggml/                            ← Tensor library (subproject)
│   ├── include/
│   │   ├── ggml.h                   ← Public API for tensor operations
│   │   ├── ggml-cpu.h               ← CPU backend API
│   │   └── tensor_trace.h           ← OUR ADDITION: Tracing structs
│   │
│   └── src/
│       ├── ggml.c                   ← Core tensor operations, graph execution
│       ├── tensor_trace.c           ← OUR ADDITION: Logging machinery
│       └── ggml-cpu/
│           └── ggml-cpu.c           ← OUR MODIFICATION: CPU backend with hooks
│
├── tools/
│   ├── cli/
│   │   └── cli.cpp                  ← Entry point for `llama-cli` binary
│   │
│   ├── completion/
│   │   └── completion.cpp           ← Entry point for `llama-completion` binary
│   │
│   └── gguf-dump/
│       └── gguf-dump.cpp            ← OUR MODIFICATION: Model structure export
│
├── common/
│   ├── common.cpp                   ← Common utilities (arg parsing, etc.)
│   ├── sampling.cpp                 ← Sampling strategies
│   └── json-schema-to-grammar.cpp   ← JSON mode support
│
└── build/bin/
    ├── llama-cli                    ← Compiled: Interactive chat
    ├── llama-completion             ← Compiled: One-shot completion
    └── llama-gguf-dump              ← Compiled: Model inspector
```

---

## 🎯 Entry Points (Actual Files)

### Option 1: `llama-cli` (Interactive Chat)

**Source**: `tools/cli/cli.cpp`

**Architecture**: Server-based (uses `server-context.h`)
- Runs an internal HTTP server
- Posts tasks to a queue
- Handles streaming responses

**Entry function**:
```cpp
int main(int argc, char ** argv) {
    // Parse args
    common_params params;

    // Initialize backend
    llama_backend_init();

    // Load model + create context (via server_context)
    cli_context cliCtx(params);
    cliCtx.ctx_server.initialize();

    // Main chat loop
    while (!should_stop()) {
        // Read user input
        // Post completion task
        // Stream response
    }
}
```

---

### Option 2: `llama-completion` (One-Shot, Simpler)

**Source**: `tools/completion/completion.cpp`

**Architecture**: Direct API calls (simpler, better for understanding)

**Entry function**:
```cpp
int main(int argc, char ** argv) {
    // 1. Parse arguments
    common_params params;
    params.prompt = "Hello";
    params.n_predict = 5;

    // 2. Initialize backend
    llama_backend_init();

    // 3. Load model
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_load_model_from_file(
        params.model.c_str(),
        model_params
    );

    // 4. Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    // 5. Tokenize prompt
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);

    // 6. MAIN GENERATION LOOP
    for (int i = 0; i < params.n_predict; i++) {
        // Decode (generate logits)
        llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));

        // Sample next token
        llama_token new_token = llama_sample_token(ctx);

        // Append
        tokens.push_back(new_token);

        // Print
        printf("%s", llama_token_to_piece(ctx, new_token).c_str());
    }

    // 7. Cleanup
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
```

**This is the clearest flow to understand!**

---

## 📚 Core API Functions (src/llama.cpp)

### 1. Model Loading

```cpp
llama_model * llama_load_model_from_file(
    const char * path,
    struct llama_model_params params
)
```

**Location**: `src/llama.cpp` (around line 17000+)

**What it does**:
```
1. Open GGUF file with mmap() or read()
2. Parse header: architecture, n_layers, vocab_size, etc.
3. Allocate tensor buffers:
   - token_embd.weight [vocab_size × n_embd]
   - For each layer:
     - wq, wk, wv, wo (attention)
     - ffn_gate, ffn_up, ffn_down (feed-forward)
4. Load tensor data from file into memory
5. Return llama_model struct
```

**Result**: Model weights in RAM (or mmapped from disk)

---

### 2. Context Creation

```cpp
llama_context * llama_new_context_with_model(
    struct llama_model * model,
    struct llama_context_params params
)
```

**Location**: `src/llama.cpp`

**What it does**:
```
1. Allocate KV cache:
   kv_cache.k [n_ctx × n_layers × n_embd_k]
   kv_cache.v [n_ctx × n_layers × n_embd_v]

2. Create ggml context (tensor arena):
   ctx_compute = ggml_init({ mem_size: 512 MB })

3. Allocate computation graph struct:
   gf = ggml_new_graph_custom(ctx_compute, LLAMA_MAX_NODES, false)

4. Allocate output buffer:
   logits [vocab_size floats]

5. Initialize scheduler:
   sched = ggml_backend_sched_new(...)

6. Return llama_context struct
```

**Key members of llama_context**:
```cpp
struct llama_context {
    llama_model * model;           // Pointer to loaded model

    llama_kv_cache kv_self;        // KV cache storage

    ggml_context * ctx_compute;    // Workspace for building graphs
    ggml_cgraph  * gf;             // Computation graph (reused)

    ggml_backend_sched * sched;    // Backend scheduler (CPU/GPU)

    float * logits;                // Output buffer [vocab_size]

    int n_ctx;                     // Max context length
    int n_batch;                   // Batch size
};
```

---

### 3. Decode (THE CORE FUNCTION)

```cpp
int llama_decode(
    struct llama_context * ctx,
    struct llama_batch batch
)
```

**Location**: `src/llama.cpp` (around line 15000+)

**This is where the magic happens!**

**Call chain**:
```
llama_decode()
  ├─ 1. llama_build_graph()
  │    └─ Detects architecture → calls llm_build_<ARCH>()
  │         └─ llm_build_deepseek() (for DeepSeek models)
  │              └─ src/models/deepseek.cpp
  │
  ├─ 2. ggml_backend_sched_graph_compute_async()
  │    └─ Executes the graph
  │         └─ For each node:
  │              └─ ggml_compute_forward_<OP>()
  │                   └─ 🎣 OUR HOOKS TRIGGER HERE
  │
  └─ 3. Copy logits to output buffer
```

---

## 🏗️ Graph Building (Symbolic Phase)

### Entry: `llama_build_graph()`

**Location**: `src/llama.cpp` (around line 13000+)

**Purpose**: Construct computation graph (SYMBOLIC, no actual math yet)

**Flow**:
```cpp
static struct ggml_cgraph * llama_build_graph(
    llama_context & lctx,
    llama_batch & batch,
    bool worst_case
) {
    const auto & model = lctx.model;
    const auto & hparams = model.hparams;

    // Detect architecture
    switch (model.arch) {
        case LLM_ARCH_DEEPSEEK:
            return llm_build_deepseek(lctx, batch);
        case LLM_ARCH_LLAMA:
            return llm_build_llama(lctx, batch);
        case LLM_ARCH_MISTRAL:
            return llm_build_mistral(lctx, batch);
        // ... other architectures
    }
}
```

---

### Model-Specific: `llm_build_deepseek()`

**Location**: `src/models/deepseek.cpp`

**Purpose**: Build DeepSeek-specific computation graph

**Simplified structure**:
```cpp
static struct ggml_cgraph * llm_build_deepseek(
    llama_context & lctx,
    llama_batch & batch
) {
    const auto & model = lctx.model;
    auto & kv_self = lctx.kv_self;

    struct ggml_cgraph * gf = lctx.gf;
    ggml_context * ctx = lctx.ctx_compute;

    // === INPUT EMBEDDING ===
    struct ggml_tensor * inpL = ggml_get_rows(
        ctx,
        model.tok_embd,          // [vocab_size × n_embd]
        batch.token              // [n_tokens]
    );
    // Result: inpL [n_tokens × n_embd]

    // === LAYER LOOP ===
    for (uint32_t il = 0; il < n_layer; ++il) {

        // --- Attention Block ---

        // 1. Input LayerNorm
        cur = llm_build_norm(ctx, inpL, ...);

        // 2. Q, K, V projections
        struct ggml_tensor * Qcur = ggml_mul_mat(
            ctx,
            model.layers[il].wq,    // Q weight [n_embd × n_embd]
            cur                     // Input [n_tokens × n_embd]
        );
        // 🎣 HOOK TRIGGERS DURING EXECUTION (not now!)

        struct ggml_tensor * Kcur = ggml_mul_mat(ctx, model.layers[il].wk, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx, model.layers[il].wv, cur);

        // 3. Reshape for multi-head attention
        Qcur = ggml_reshape_3d(ctx, Qcur, n_embd_head, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head_k, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head_v, n_head_kv, n_tokens);

        // 4. Apply RoPE (rotary position embeddings)
        Qcur = ggml_rope_ext(ctx, Qcur, ...);
        Kcur = ggml_rope_ext(ctx, Kcur, ...);

        // 5. Update KV cache (APPEND, don't recompute)
        ggml_build_forward_expand(gf,
            ggml_cpy(ctx, Kcur, ggml_view_1d(ctx, kv_self.k_l[il], ...))
        );
        ggml_build_forward_expand(gf,
            ggml_cpy(ctx, Vcur, ggml_view_1d(ctx, kv_self.v_l[il], ...))
        );

        // 6. Attention scores: Q @ K^T
        struct ggml_tensor * kq = ggml_mul_mat(
            ctx,
            ggml_view_3d(..., kv_self.k_l[il], ...),  // All cached K
            Qcur                                       // Current Q
        );

        // 7. Softmax
        kq = ggml_soft_max_ext(ctx, kq, mask, scale, ...);

        // 8. Attention output: softmax @ V
        struct ggml_tensor * kqv = ggml_mul_mat(
            ctx,
            ggml_view_3d(..., kv_self.v_l[il], ...),  // All cached V
            kq
        );

        // 9. Output projection
        cur = ggml_mul_mat(ctx, model.layers[il].wo, kqv);

        // 10. Residual connection
        cur = ggml_add(ctx, cur, inpL);
        inpL = cur;

        // --- Feed-Forward Block ---

        // 11. FFN LayerNorm
        cur = llm_build_norm(ctx, inpL, ...);

        // 12. Gate and Up projections
        struct ggml_tensor * tmp_gate = ggml_mul_mat(
            ctx,
            model.layers[il].ffn_gate,  // [n_embd × n_ff]
            cur
        );

        struct ggml_tensor * tmp_up = ggml_mul_mat(
            ctx,
            model.layers[il].ffn_up,    // [n_embd × n_ff]
            cur
        );

        // 13. SwiGLU activation
        cur = ggml_mul(ctx, ggml_silu(ctx, tmp_gate), tmp_up);

        // 14. Down projection
        cur = ggml_mul_mat(
            ctx,
            model.layers[il].ffn_down,  // [n_ff × n_embd]
            cur
        );

        // 15. Residual connection
        cur = ggml_add(ctx, cur, inpL);
        inpL = cur;

    } // End layer loop

    // === OUTPUT ===

    // 16. Final LayerNorm
    cur = llm_build_norm(ctx, inpL, ...);

    // 17. Output projection (get logits)
    cur = ggml_mul_mat(
        ctx,
        model.output,  // [n_embd × vocab_size]
        cur
    );

    // 18. Build the forward pass
    ggml_build_forward_expand(gf, cur);

    return gf;
}
```

**CRITICAL**: At this point, NO COMPUTATION has happened! We just built a recipe.

---

## ⚙️ Graph Execution (Actual Computation)

### Entry: `ggml_backend_sched_graph_compute_async()`

**Location**: `ggml/src/ggml-backend.c`

**Purpose**: Execute the graph (do the actual math)

**Flow**:
```cpp
void ggml_backend_sched_graph_compute_async(
    struct ggml_backend_sched * sched,
    struct ggml_cgraph * graph
) {
    // Split graph into backend-specific subgraphs
    // (some nodes on CPU, some on GPU, etc.)

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];

        // Determine which backend handles this op
        ggml_backend * backend = get_backend_for_node(node);

        // Execute the operation
        backend->graph_compute(node);
        // ↓
        // For CPU backend:
        ggml_compute_forward(node)
            ├─ switch (node->op):
            │    case GGML_OP_MUL_MAT:
            │        ggml_compute_forward_mul_mat(node)
            │         └─ 🎣 OUR HOOK HERE!
            │    case GGML_OP_ADD:
            │        ggml_compute_forward_add(node)
            │    case GGML_OP_SOFT_MAX:
            │        ggml_compute_forward_soft_max(node)
            │    ...
    }
}
```

---

### Our Hook Location: `ggml_compute_forward_mul_mat()`

**Location**: `ggml/src/ggml-cpu/ggml-cpu.c` (line ~1237)

**What happens**:
```cpp
static void ggml_compute_forward_mul_mat(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst
) {
    const struct ggml_tensor * src0 = dst->src[0];  // Weight matrix
    const struct ggml_tensor * src1 = dst->src[1];  // Input

    // ============================================
    // 🎣 OUR HOOK (if GGML_TENSOR_TRACE enabled)
    // ============================================
    #ifdef GGML_TENSOR_TRACE
        if (params->ith == 0) {  // Only first thread
            struct TensorAccessLog entry = {0};
            entry.timestamp_ns = tensor_trace_get_timestamp_ns();
            entry.tensor_ptr = (uint64_t)src0->data;
            strncpy(entry.tensor_name, src0->name, 63);
            entry.layer_id = tensor_trace_extract_layer_id(src0->name);
            tensor_trace_log(&entry);  // 📝 Write to /tmp/tensor_trace.bin
        }
    #endif
    // ============================================

    // ACTUAL MATRIX MULTIPLICATION
    // (BLAS call or manual loops)
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        1.0f,
        src0->data, K,
        src1->data, N,
        0.0f,
        dst->data, N
    );
}
```

**This is where our instrumentation captures tensor accesses!**

---

## 💾 KV Cache: How It Works

### The Problem

```
Generating "Hello world is"

Token 0: "Hello" [ID: 15043]
  → Compute K_hello, V_hello for all layers

Token 1: "world" [ID: 284]
  → Need K_world, V_world
  → BUT ALSO need K_hello, V_hello again (for attention)
  → Don't want to recompute K_hello, V_hello!
```

### The Solution: KV Cache

**Structure**:
```cpp
struct llama_kv_cache {
    struct ggml_tensor * k_l[n_layers];  // Cached K per layer
    struct ggml_tensor * v_l[n_layers];  // Cached V per layer

    // k_l[il] shape: [n_ctx × n_embd_k_gqa]
    // v_l[il] shape: [n_ctx × n_embd_v_gqa]
};
```

**During graph building** (`llm_build_deepseek`):
```cpp
// For layer il:

// 1. Compute ONLY current token's K, V
Kcur = ggml_mul_mat(ctx, wk, cur);  // [1 × n_embd_k]
Vcur = ggml_mul_mat(ctx, wv, cur);

// 2. APPEND to cache (at position kv_pos)
ggml_build_forward_expand(gf,
    ggml_cpy(ctx, Kcur,
        ggml_view_1d(ctx, kv_self.k_l[il], n_embd_k, kv_pos * n_embd_k)
    )
);

// 3. Now kv_self.k_l[il] contains:
// [K_token0, K_token1, K_token2, ..., K_current]
//  ↑ cached  ↑ cached  ↑ cached      ↑ just computed

// 4. Use ALL cached K for attention
struct ggml_tensor * kq = ggml_mul_mat(
    ctx,
    ggml_view_3d(ctx, kv_self.k_l[il], ...),  // ALL K (size: n_past + 1)
    Qcur                                       // Current Q
);
```

**Result**: O(n) instead of O(n²) computation!

---

## 🔄 The Complete Token Generation Loop

```
┌─────────────────────────────────────────────────────────────────┐
│ INITIALIZATION (Once)                                           │
├─────────────────────────────────────────────────────────────────┤
│ 1. llama_backend_init()                                         │
│ 2. model = llama_load_model_from_file("model.gguf")            │
│ 3. ctx = llama_new_context_with_model(model)                   │
│ 4. tokens = tokenize("Hello")  → [15043]                       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ LOOP: for i = 0 to n_predict                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ PHASE 1: BUILD GRAPH (Symbolic)                           │ │
│  ├───────────────────────────────────────────────────────────┤ │
│  │ llama_build_graph(ctx, batch)                             │ │
│  │   └─ llm_build_deepseek()                                 │ │
│  │        └─ For each layer:                                 │ │
│  │             ggml_mul_mat(wq, input)  ← Node in graph      │ │
│  │             ggml_mul_mat(wk, input)  ← Node in graph      │ │
│  │             ggml_mul_mat(wv, input)  ← Node in graph      │ │
│  │             ...                                            │ │
│  │   Result: gf (computation graph with ~500 nodes)          │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            ↓                                    │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ PHASE 2: EXECUTE GRAPH (Actual Math)                      │ │
│  ├───────────────────────────────────────────────────────────┤ │
│  │ ggml_backend_sched_graph_compute_async(sched, gf)         │ │
│  │   └─ For each node in gf:                                 │ │
│  │        ggml_compute_forward_mul_mat()                     │ │
│  │         ├─ 🎣 tensor_trace_log() ← OUR HOOK!              │ │
│  │         └─ cblas_sgemm() ← ACTUAL COMPUTATION             │ │
│  │   Result: logits[vocab_size] now contains actual scores   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            ↓                                    │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ PHASE 3: SAMPLE NEXT TOKEN                                │ │
│  ├───────────────────────────────────────────────────────────┤ │
│  │ llama_sample_token(ctx)                                   │ │
│  │   ├─ Apply temperature, top-k, top-p                      │ │
│  │   └─ Sample: new_token = 284 ("world")                    │ │
│  │   Result: new_token                                       │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            ↓                                    │
│  tokens.push_back(new_token)  → [15043, 284]                  │
│  kv_pos++  (advance KV cache position)                        │
│                                                                 │
│  LOOP BACK (if i < n_predict)                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: Generated text: "Hello world is great!"                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Computation Graph: Stored or Regenerated?

### Answer: **REGENERATED per token, but REUSES the same memory**

**Why regenerate?**
- Graph structure changes based on:
  - Number of tokens in batch (varies)
  - KV cache size (grows each token)
  - Attention mask shape (depends on position)

**But memory is reused**:
```cpp
// llama_context contains:
struct llama_context {
    ggml_context * ctx_compute;  // Arena allocator (512 MB)
    ggml_cgraph * gf;            // Graph struct (preallocated)
};

// Each token:
1. ctx_compute is RESET (arena allocator rewinds)
2. gf is REUSED (same struct, different nodes)
3. Graph is REBUILT with new node connections
4. Graph is EXECUTED
5. Repeat
```

**Analogy**: Like a whiteboard - you erase and redraw the same diagram each time, but the whiteboard itself is reused.

---

## 📊 Visualization: Generating a Graphviz Diagram

### You can export the computation graph!

**Add to your code**:
```cpp
// After llama_build_graph():
ggml_graph_dump_dot(gf, NULL, "computation_graph.dot");
```

**Then visualize**:
```bash
dot -Tpng computation_graph.dot -o graph.png
```

**This creates a visual diagram of all operations!**

---

## 🎯 Summary: Where Our Hooks Fit

```
Entry: llama-completion
  ↓
main()
  ↓
llama_load_model_from_file()  ← Loads weights (no hooks)
  ↓
llama_new_context_with_model()  ← Allocates KV cache (no hooks)
  ↓
FOR EACH TOKEN:
  ↓
  llama_decode()
    ↓
    llama_build_graph()  ← Symbolic (no hooks)
      ↓
      llm_build_deepseek()  ← Builds recipe (no hooks)
        ↓
        ggml_mul_mat()  ← Creates node (no hooks)
    ↓
    ggml_backend_sched_graph_compute_async()  ← EXECUTION!
      ↓
      FOR EACH NODE:
        ↓
        ggml_compute_forward_mul_mat()  ← 🎣 OUR HOOKS TRIGGER HERE!
          ↓
          tensor_trace_log()  ← 📝 Writes to /tmp/tensor_trace.bin
          ↓
          cblas_sgemm()  ← Actual math
  ↓
  llama_sample_token()  ← Pick next token (no hooks)
  ↓
LOOP
```

---

## 📁 File Location Reference

| Component | File | Line (approx) |
|-----------|------|---------------|
| Entry (completion) | `tools/completion/completion.cpp` | main() at ~50 |
| Entry (cli) | `tools/cli/cli.cpp` | main() at ~200 |
| Model loading | `src/llama.cpp` | ~17000+ |
| Context creation | `src/llama.cpp` | ~14000+ |
| Decode (core) | `src/llama.cpp` | ~15000+ |
| Graph building | `src/llama.cpp` | ~13000+ |
| DeepSeek graph | `src/models/deepseek.cpp` | llm_build_deepseek() |
| Graph execution | `ggml/src/ggml-backend.c` | ~2000+ |
| mul_mat (hook!) | `ggml/src/ggml-cpu/ggml-cpu.c` | ~1237 |
| Trace header | `ggml/include/tensor_trace.h` | Full file |
| Trace impl | `ggml/src/tensor_trace.c` | Full file |
| GGUF dump (our mod) | `tools/gguf-dump/gguf-dump.cpp` | --csv flag |

---

## 🔑 Key Takeaways

1. **Graph is SYMBOLIC** during building (no computation)
2. **Graph is EXECUTED** all at once (actual math)
3. **Graph is REGENERATED** each token (but memory reused)
4. **KV cache is PERSISTENT** across tokens (append-only)
5. **Our hooks trigger during EXECUTION** (ggml_compute_forward_*)
6. **Entry points**: `llama-cli` (complex) or `llama-completion` (simple)

---

**Next**: See [tensor-tracing/README.md](../tensor-tracing/README.md) for details on our instrumentation.



EXTRA: 
1) Complete Initialization Flow

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STEP 1: Backend Initialization                                             │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ llama_backend_init()                                                        │
  │   src/llama.cpp:751                                                         │
  │                                                                             │
  │   What it does:                                                             │
  │   1. ggml_time_init()           ← Initialize high-resolution timer         │
  │   2. ggml_init({ 0, NULL, false })  ← Dummy init to populate f16 tables    │
  │   3. ggml_free(ctx)             ← Immediately free (only needed tables)    │
  │                                                                             │
  │   Purpose: Set up global state (timing, fp16 lookup tables for CPU)        │
  └─────────────────────────────────────────────────────────────────────────────┘
                                        ↓
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STEP 2: Model Loading                                                      │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ llama_load_model_from_file(path, params)                                   │
  │   src/llama.cpp:989 → 837 (llama_model_load_from_file_impl)               │
  │                                                                             │
  │   Call chain:                                                               │
  │   llama_model_load_from_file()                                             │
  │     └─ llama_model_load_from_file_impl()  (line 837)                      │
  │          └─ llama_model_load()  (line 783)                                │
  │               ├─ llama_model_loader ml(fname, ...)  ← Open GGUF           │
  │               ├─ model.load_arch(ml)       ← Read architecture            │
  │               ├─ model.load_hparams(ml)    ← Read hyperparameters         │
  │               ├─ model.load_vocab(ml)      ← Read tokenizer               │
  │               └─ model.load_tensors(ml)    ← CRITICAL: mmap/load weights  │
  │                                                                             │
  │   What model.load_tensors() does:                                          │
  │   - For each tensor in GGUF metadata:                                      │
  │     1. Allocate buffer (or mmap region)                                    │
  │     2. Read tensor data from file offset                                   │
  │     3. Store in model.tensors map:                                         │
  │        "blk.0.attn_q.weight" → ggml_tensor*                               │
  │                                                                             │
  │   Result: All weights loaded into RAM (or mmapped)                         │
  └─────────────────────────────────────────────────────────────────────────────┘
                                        ↓
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STEP 2.5: Tensor Tracing Initialization (OUR ADDITION)                     │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ #ifdef GGML_TENSOR_TRACE                                                   │
  │   tensor_trace_init("/tmp/tensor_trace.bin", 2GB)                         │
  │   src/llama.cpp:975                                                         │
  │                                                                             │
  │   What tensor_trace_init() does (ggml/src/tensor_trace.c:70):             │
  │   1. Open log file: /tmp/tensor_trace.bin                                 │
  │   2. Allocate 2GB buffer for binary logs                                   │
  │   3. Initialize mutex for thread-safe logging                              │
  │   4. Start timestamp counter                                               │
  │   5. Initialize tensor registry (empty for now - Phase 2!)                 │
  │ #endif                                                                      │
  │                                                                             │
  │   NOTE: Registry is empty at this point! We need Phase 2 to populate it.   │
  └─────────────────────────────────────────────────────────────────────────────┘
                                        ↓
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STEP 3: Context Creation                                                   │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ llama_new_context_with_model(model, params)                                │
  │   src/llama-context.cpp:2466 → 2402 (llama_init_from_model)               │
  │                                                                             │
  │   What happens:                                                             │
  │   1. Validate params (n_ctx, n_batch, flash_attn settings)                 │
  │   2. Call: new llama_context(*model, params)  (line 2456)                 │
  │                                                                             │
  │   llama_context constructor does:                                          │
  │   ├─ Allocate KV cache buffers:                                            │
  │   │    kv_self.k_l[0..n_layer] each [n_ctx × n_embd_k] tensors           │
  │   │    kv_self.v_l[0..n_layer] each [n_ctx × n_embd_v] tensors           │
  │   │                                                                         │
  │   ├─ Create computation workspace:                                         │
  │   │    ctx_compute = ggml_init({ 512 MB arena })                          │
  │   │    (This is the "whiteboard" we rebuild the graph on each token!)     │
  │   │                                                                         │
  │   ├─ Allocate graph structure:                                             │
  │   │    gf = ggml_new_graph_custom(ctx_compute, LLAMA_MAX_NODES, false)    │
  │   │    (Pre-allocated slots for ~2048 nodes)                              │
  │   │                                                                         │
  │   ├─ Create backend scheduler:                                             │
  │   │    sched = ggml_backend_sched_new(backends, n_backends, ...)          │
  │   │    (Decides which ops run on CPU vs GPU)                              │
  │   │                                                                         │
  │   └─ Allocate output buffers:                                              │
  │        logits = new float[vocab_size]                                      │
  │        embeddings = new float[n_embd]  (if needed)                         │
  │                                                                             │
  │   Result: Ready to generate tokens!                                        │
  └─────────────────────────────────────────────────────────────────────────────┘

  Key Insights:

  1. Backend init is minimal: Just sets up timing and fp16 tables
  2. Model loading is heavy: Reads entire GGUF file, loads/mmaps all weights
  3. Tensor tracing inits AFTER model load: Binary log file and buffer ready
  4. Context creation allocates runtime memory: KV cache + workspace + scheduler
  5. Graph structure is pre-allocated but empty: Gets filled during each token decode

  ---
  Now let's move to #2: Graph Visualization Strategy. You're absolutely correct - the graph is complete in one place before
  execution!

  2) Graph Visualization Strategy - Discussion

  Your Observation (100% Correct!)

  "the full graph is in one place before we do decoding right?"

  YES! After llama_build_graph() completes, we have a complete ggml_cgraph* structure with all nodes for that token's forward
  pass.

  Current Flow:

  llama_decode(ctx, batch)
    ├─ 1. gf = llama_build_graph(ctx, batch)  ← GRAPH FULLY BUILT HERE
    │      Result: gf->nodes[0..n_nodes-1] all populated
    │              gf->n_nodes = ~500-800 (depends on architecture)
    │
    ├─ 2. ggml_backend_sched_graph_compute_async(sched, gf)  ← EXECUTE
    │      (This is where actual computation happens)
    │
    └─ 3. Copy logits to output

  Where to Insert Graph Dump:

  Option A: Immediately after graph building (RECOMMENDED)
  // src/llama.cpp, in llama_decode() function
  gf = llama_build_graph(lctx, ubatch, batch_all, worst_case);

  #ifdef GGML_TENSOR_TRACE  // Reuse same flag
      // Dump graph to Graphviz format
      char dot_filename[256];
      snprintf(dot_filename, sizeof(dot_filename),
               "/tmp/graphs/token_%05d.dot", lctx.n_outputs);
      ggml_graph_dump_dot(gf, NULL, dot_filename);
  #endif

  // Then execute...
  ggml_backend_sched_graph_compute_async(lctx.sched, gf);


