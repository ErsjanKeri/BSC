# Tensor Tracing Analysis and UI Planning

**Date**: 2026-01-08 (evening session)
**Status**: Investigation and planning phase

---

## Part 1: Understanding "Leaf" Nodes in llama.cpp

### What Are Leaf Nodes?

**Definition** (from `ggml.c:6750-6751`):
```c
if (node->op == GGML_OP_NONE && !(node->flags & GGML_TENSOR_FLAG_PARAM)) {
    // reached a leaf node, not part of the gradient graph (e.g. a constant)
```

A tensor is classified as a "leaf" if:
1. `node->op == GGML_OP_NONE` - No operation (not computed, it's an input)
2. `!(node->flags & GGML_TENSOR_FLAG_PARAM)` - Not a parameter (not a weight)

Otherwise, it's a "node" in the computation graph.

### Role in Computation

Leaf nodes are **global constant inputs** that are reused across the entire computation graph. They are **NOT** layer-specific infrastructure.

### Identified Leaf Nodes (TinyLlama)

Based on analysis of `token-00000.json`:

| Leaf | Shape | dtype | Purpose | Used By |
|------|-------|-------|---------|---------|
| `leaf_2` | [2, 1] | i32 | Token position indices | GET_ROWS (embedding lookup) |
| `leaf_4` | [2, 1] | i32 | ROPE position indices | ROPE (Q and K) in ALL 22 layers |
| `leaf_7` | [2, 1] | i64 | KV cache K positions | SET_ROWS (cache_k) in ALL 22 layers |
| `leaf_9` | [2, 1] | i64 | KV cache V positions | SET_ROWS (cache_v) in ALL 22 layers |
| `leaf_11` | [256, 2] | f32 | ROPE frequency table | COPY operation |
| `leaf_247` | [1, 1] | i32 | Final position index | GET_ROWS (output) |

### Memory Source

**Question**: Where do leaf nodes come from?

**Answer**:
- **Memory source**: BUFFER (runtime allocations)
- **Not from GGUF**: These are created at runtime by llama.cpp
- **Allocator**: Created by llama_decode() when building the computation graph
- **Purpose**: Position/index tracking for:
  - Token positions in sequence
  - Rotary position embeddings (ROPE)
  - KV cache management

### Why `layer_id = null`?

Leaf nodes have `layer_id = null` because they are **shared infrastructure**, not layer-specific:

- `leaf_4` is used by ROPE in **all 22 layers** - it's the same position tensor
- `leaf_7` and `leaf_9` update KV cache for **all 22 layers** - same position indices
- They are global inputs, not owned by any specific layer

**Analogy**: Like function parameters that are passed to all layers, not created by them.

---

## Part 2: Layer Filtering Problem

### Original Issue

When filtering graph by layer ID (e.g., "show layer 5"), we excluded nodes where `layer_id = null`.

**Problem**: This hid critical nodes:
1. **Leaf nodes** (position indices) - shared infrastructure
2. **Intermediate tensors** (Qcur-5, norm-12) - actually layer-specific but not detected
3. **Global operations** (embeddings, output)

### Root Cause #1: Incomplete Layer Extraction

**Problem**: `parse_dot.py` only extracted layer IDs from `blk.N.` pattern (weight tensors).

**Impact**: Intermediate computation tensors use `-N` suffix (e.g., "Qcur-5", "norm-12") where N = layer number. These were all marked as `layer_id = null`, hiding 550 nodes from layer filtering!

**Example**:
- ❌ Before: "Qcur-0" → layer_id = null (marked as infrastructure)
- ✅ After: "Qcur-0" → layer_id = 0 (correctly marked as layer-specific)

### Root Cause #2: Over-Strict Filtering

Filtering logic only showed nodes with exact layer match:
```python
filtered_nodes = [n for n in nodes if n['layer_id'] == selected_layer]
```

This excluded shared infrastructure (leaf nodes) needed by that layer.

### Solution Implemented (2026-01-08 evening)

**Fixed layer extraction in parse_dot.py**:

**Before** (only weights detected):
```python
def extract_layer_id(tensor_name: str) -> int:
    if tensor_name.startswith("blk."):
        match = re.search(r'blk\.(\d+)\.', tensor_name)
        if match:
            return int(match.group(1))
    return None
```

**After** (weights + intermediates):
```python
def extract_layer_id(tensor_name: str) -> int:
    # Pattern 1: "blk.N." for weights
    if tensor_name.startswith("blk."):
        match = re.search(r'blk\.(\d+)\.', tensor_name)
        if match:
            return int(match.group(1))

    # Pattern 2: "-N" suffix for intermediates
    # EXCLUDE: "leaf_N" and "node_N" (not layer-specific)
    if not tensor_name.startswith("leaf") and not tensor_name.startswith("node_"):
        match = re.search(r'-(\d+)$', tensor_name)
        if match:
            layer = int(match.group(1))
            if 0 <= layer < 100:  # Sanity check
                return layer

    return None
```

**Added node classification** (`node_type` field):
- `infrastructure`: Leaf nodes, KV cache, shared constants
- `layer`: Layer-specific tensors (now correctly identified!)
- `embedding`: Input embeddings
- `output`: Final output layer

**Results**:
- **Before**: 938 infrastructure nodes, 0 layer nodes ❌
- **After**: 550 layer nodes (25 per layer × 22 layers), 388 infrastructure ✅

### Divergence from Vanilla llama.cpp

**IMPORTANT**: This fix makes our graph parsing **different** from upstream llama.cpp.

**What changed**:
1. Added `-N` suffix pattern for layer extraction
2. Added `node_type` field for smart filtering
3. Enhanced `classify_node_type()` function

**Why this matters**:
- Our graph JSON files now have extra metadata
- Layer filtering will work correctly in our webui
- Should be considered for upstream contribution (better layer detection)

### Desired Behavior

When filtering for layer N, should show:
1. ✅ All nodes with `layer_id == N`
2. ✅ All leaf nodes (shared infrastructure)
3. ✅ Immediate inputs to layer N (even if from previous layer)
4. ✅ Immediate outputs from layer N (even if going to next layer)

**Goal**: Show a **self-contained subgraph** for that layer, including its dependencies.

---

## Part 3: Solution Design

### Proposed Layer Filtering Algorithm

```python
def get_layer_subgraph(graph, layer_id):
    """
    Extract subgraph for a specific layer, including:
    - Layer-specific nodes
    - Shared infrastructure (leaf nodes)
    - Direct inputs/outputs (cross-layer connections)
    """
    nodes = []
    edges = []

    # Step 1: Get all nodes for this layer
    layer_nodes = [n for n in graph['nodes'] if n['layer_id'] == layer_id]
    layer_node_ids = {n['id'] for n in layer_nodes}

    # Step 2: Get all leaf nodes (shared infrastructure)
    leaf_nodes = [n for n in graph['nodes']
                  if n['label'].startswith('leaf') or n['operation'] == 'CONST']

    # Step 3: Get direct inputs to layer nodes
    input_nodes = []
    for edge in graph['edges']:
        if edge['target'] in layer_node_ids:
            # Source node feeds into this layer
            source_node = next((n for n in graph['nodes'] if n['id'] == edge['source']), None)
            if source_node and source_node not in layer_nodes and source_node not in leaf_nodes:
                input_nodes.append(source_node)

    # Step 4: Get direct outputs from layer nodes
    output_nodes = []
    for edge in graph['edges']:
        if edge['source'] in layer_node_ids:
            # This layer feeds into target node
            target_node = next((n for n in graph['nodes'] if n['id'] == edge['target']), None)
            if target_node and target_node not in layer_nodes:
                output_nodes.append(target_node)

    # Step 5: Combine all nodes
    all_node_ids = set()
    for node_list in [layer_nodes, leaf_nodes, input_nodes, output_nodes]:
        for node in node_list:
            if node['id'] not in all_node_ids:
                nodes.append(node)
                all_node_ids.add(node['id'])

    # Step 6: Get all edges between included nodes
    for edge in graph['edges']:
        if edge['source'] in all_node_ids and edge['target'] in all_node_ids:
            edges.append(edge)

    return {'nodes': nodes, 'edges': edges}
```

### Node Categories for Visualization

Add visual distinction:

| Category | Color | Description |
|----------|-------|-------------|
| Layer-specific | Blue | Nodes with `layer_id == N` |
| Shared infrastructure | Green | Leaf nodes, global constants |
| Cross-layer inputs | Orange | Nodes from previous layers |
| Cross-layer outputs | Purple | Nodes going to next layers |

---

## Part 4: UI Improvement Plan

### Priority 1: Fix Graph Layer Filtering (Immediate)

**Goal**: Show complete, self-contained subgraph for each layer

**Tasks**:
1. Implement `get_layer_subgraph()` algorithm in webui
2. Add visual categories (colors) for different node types
3. Add legend explaining node colors
4. Test with layer 0, 5, 21 to ensure completeness

**Files to modify**:
- `webui/src/components/ComputationGraph.tsx` or similar

### Priority 2: Heatmap Improvements (High)

**Current Issues**:
- Hard to identify small tensors
- No magnification/zoom
- Can't click to filter graph

**Proposed Features**:
1. **Dual-track layout**: Separate DISK and BUFFER tracks
2. **Magnification**: Click region to zoom in
3. **Click-to-filter**: Click tensor → highlight in graph + show trace entries
4. **Hover tooltips**: Show tensor name, size, access count
5. **Color by access frequency**: Hot (red) vs cold (blue)

### Priority 3: Trace/Log Correlation (High)

**Goal**: Connect all views for per-token analysis

**Proposed Features**:
1. **Timeline playback**: Play/pause through trace entries
2. **Synchronized views**:
   - Click entry in trace → highlight in graph + heatmap
   - Click node in graph → show all trace entries for that tensor
   - Click heatmap region → filter trace entries
3. **Phase separation**: Toggle between PROMPT and GENERATE views
4. **Token navigation**: Next/previous token buttons

### Priority 4: Full-Screen Views (Medium)

**Goal**: Allow focusing on one visualization at a time

**Implementation**:
1. Add maximize/minimize buttons to each panel
2. When maximized, hide other panels
3. Keyboard shortcuts (F for full-screen, ESC to exit)

### Priority 5: Data Analytics Dashboard (Medium)

**Metrics to Show**:

**Per-Token**:
- Total operations (count)
- Duration (ms)
- Memory accessed (MB)
- DISK vs BUFFER ratio

**Per-Layer**:
- Operation count
- Average duration
- Hottest tensors (most accessed)
- Memory breakdown

**Overall**:
- Total inference time
- PROMPT vs GENERATE comparison
- Layer-by-layer breakdown (table + chart)
- Access pattern visualization (sequential vs random)

**Visualization Types**:
- Bar charts (operations per layer)
- Line charts (access pattern over time)
- Pie charts (DISK vs BUFFER distribution)
- Heatmap (layer × token access matrix)

---

## Part 5: Implementation Roadmap

### Phase 1: Fix Critical Issues (1-2 days)

**Tasks**:
1. ✅ Investigate leaf nodes (DONE)
2. ✅ Document findings (DONE)
3. ⏳ Implement improved layer filtering
4. ⏳ Add node color categories
5. ⏳ Test with all layers

**Deliverable**: Graph view correctly shows layer-specific subgraphs with infrastructure

### Phase 2: Enhance Heatmap (2-3 days)

**Tasks**:
1. Implement dual-track (DISK/BUFFER) layout
2. Add magnification/zoom
3. Add hover tooltips
4. Implement click-to-filter
5. Color by access frequency

**Deliverable**: Heatmap allows detailed exploration of memory access patterns

### Phase 3: View Synchronization (3-4 days)

**Tasks**:
1. Implement timeline playback controls
2. Connect trace table ↔ graph view
3. Connect trace table ↔ heatmap
4. Connect graph ↔ heatmap
5. Add phase toggle (PROMPT/GENERATE)

**Deliverable**: All views are synchronized and interactive

### Phase 4: Analytics Dashboard (2-3 days)

**Tasks**:
1. Calculate per-token metrics
2. Calculate per-layer metrics
3. Build charts (bar, line, pie)
4. Add comparison views (token-to-token, layer-to-layer)
5. Export to CSV/JSON

**Deliverable**: Data analytics panel shows quantitative insights

### Phase 5: Polish and Testing (1-2 days)

**Tasks**:
1. Full-screen mode
2. Keyboard shortcuts
3. Performance optimization (virtual scrolling, lazy loading)
4. User testing
5. Documentation

**Deliverable**: Production-ready visualization tool

---

## Part 6: Open Questions

### For Leaf Nodes
1. **Q**: Do leaf node addresses change between runs?
   **A**: Need to verify - likely yes (runtime allocation)

2. **Q**: Can we label leaf nodes more semantically?
   **A**: Yes, by analyzing their usage pattern:
   - Used by GET_ROWS → "token_positions"
   - Used by ROPE → "rope_positions"
   - Used by SET_ROWS → "cache_positions"

3. **Q**: Do different models have different leaf node counts?
   **A**: Likely yes - depends on model architecture

### For Layer Filtering
1. **Q**: Should we show cross-layer edges (e.g., layer 5 → layer 6)?
   **A**: Yes, but with different style (dashed line?)

2. **Q**: What about operations that span multiple layers (e.g., residual connections)?
   **A**: Include if they connect to the selected layer

3. **Q**: Should "layer_id = null" nodes have their own category?
   **A**: Yes - "Global" or "Infrastructure" category

---

## Part 7: Technical Notes

### Leaf Node Naming Enhancement

Could modify `tools/parse_dot.py` to detect and rename leaf nodes:

```python
def classify_leaf_node(node, edges):
    """Classify leaf node by analyzing its usage."""
    # Find all operations that use this leaf
    target_ops = [get_node(e['target']) for e in edges if e['source'] == node['id']]

    if any(op['operation'] == 'GET_ROWS' for op in target_ops):
        return 'token_positions'
    elif any(op['operation'] == 'ROPE' for op in target_ops):
        return 'rope_positions'
    elif any(op['operation'] == 'SET_ROWS' for op in target_ops):
        return 'cache_positions'
    else:
        return node['label']  # Keep original
```

### Performance Considerations

For large models (70B+ parameters):
- Graph may have 10,000+ nodes
- Need virtual scrolling for trace table (1M+ entries)
- Consider server-side filtering instead of client-side
- Implement progressive loading (load layer-by-layer)

---

## Part 8: Next Steps

**Immediate**:
1. Implement improved layer filtering in webui
2. Add node color categories
3. Test with all layers of TinyLlama

**Short-term**:
4. Enhance heatmap (dual-track, zoom)
5. Add hover tooltips
6. Implement timeline playback

**Medium-term**:
7. View synchronization
8. Analytics dashboard
9. Full-screen mode

**Long-term**:
10. Support for other models
11. Comparison mode (compare two runs)
12. Export reports

---

## Part 9: Success Criteria

### Graph View
- [ ] Layer filtering includes all relevant nodes (layer-specific + infrastructure)
- [ ] Visual distinction between node categories (colors)
- [ ] Legend explains node types
- [ ] Can navigate all 22 layers
- [ ] Cross-layer connections visible

### Heatmap
- [ ] DISK and BUFFER tracks separated
- [ ] Can zoom/magnify regions
- [ ] Click filters graph + trace
- [ ] Hover shows tensor details
- [ ] Color indicates access frequency

### Trace Table
- [ ] Virtual scrolling (handles 1M+ entries)
- [ ] Phase filter (PROMPT/GENERATE)
- [ ] Token navigation
- [ ] Click entry → highlight in graph + heatmap

### Analytics
- [ ] Per-token metrics displayed
- [ ] Per-layer breakdown table
- [ ] Charts visualize patterns
- [ ] Can export data

### Overall
- [ ] All views synchronized
- [ ] Responsive (handles large datasets)
- [ ] Intuitive UX
- [ ] Documented

---

## End of Analysis Document
