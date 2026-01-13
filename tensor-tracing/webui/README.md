# Tensor Trace Visualizer

Interactive visualization for analyzing tensor tracing and computation graphs.

**Status**: 🚧 Early development - Basic 4-view layout works, major improvements planned

---

## Current Features

**4-Panel Layout**:
1. Computation Graph (Cytoscape.js)
2. Timeline & Trace table
3. Memory Heatmap (basic)
4. 3D Transformer (placeholder)

**Working**: Load JSON, token selector, click nodes, view trace, basic zoom/pan

---

## Quick Start

```bash
# 1. Generate data
cd ..
python3 run_experiment.py

# 2. Install & run
cd webui
npm install
npm run dev  # → http://localhost:5173
```

---

## Data Structure

Expects in `public/data/`:
- `memory-map.json` - GGUF structure (201 tensors)
- `buffer-timeline.json` - Buffer alloc/dealloc events
- `graphs/token-*.json` - Computation graphs per token
- `traces/token-*.json` - Trace entries per token

---

## Planned Improvements

### High Priority
- Full-screen horizontal views (toggle minimize/maximize)
- Heatmap overhaul (dual-track DISK/BUFFER, magnification, click-to-filter)
- Timeline playback (play/pause/step)
- View synchronization

### Medium Priority
- Virtual scrolling for trace table
- 3D transformer model
- Layer-based filtering
- Performance optimization

---

## Known Issues

1. **Graph performance**: Slow with 700+ nodes
2. **Heatmap**: Hard to identify small tensors
3. **No correlation**: Views not synchronized yet

**Note**: Token ID tracking bug was fixed on 2026-01-08 - traces now properly split by token

---

## Development

```bash
npm install        # Dependencies
npm run dev        # Dev server
npm run build      # Production build
npm run preview    # Preview build
```

---

## Roadmap

**Phase 1** (Now): ✅ Basic 4-panel layout

**Phase 2** (2-4 weeks): Full-screen views, heatmap improvements, timeline playback

**Phase 3** (1-2 months): 3D model, advanced correlation, virtual scrolling

**Phase 4** (Future): Export/compare experiments, real-time tracing

---

## Documentation

- [Tensor Tracing README](../README.md) - Technical details
- [Setup Guide](../setup.md) - Build & run
- [Main README](../../README.md) - Project overview
- [Journal](../../journal/) - Development history
