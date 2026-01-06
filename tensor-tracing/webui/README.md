# Tensor Trace Visualizer - WebUI

Interactive 4-view visualization tool for analyzing llama.cpp tensor tracing and computation graphs.

## Features

**4 Synchronized Views:**
1. **Computation Graph** (Top-left) - Interactive graph visualization with Cytoscape.js
2. **Timeline & Trace** (Top-right) - Animated timeline with trace table
3. **Memory Heatmap** (Bottom-left) - GGUF file access visualization
4. **3D Transformer** (Bottom-right) - 3D transformer architecture

## Tech Stack

- **Framework:** React 18 + TypeScript 5.3 + Vite 5.0
- **State:** Zustand 4.5 (global state management)
- **Styling:** Tailwind CSS 3.4
- **Graph:** Cytoscape.js 3.28
- **3D:** React Three Fiber 8.15 + Three.js
- **Virtual Scrolling:** react-window 1.8

## Development

```bash
# Install dependencies
npm install

# Start dev server (running on http://localhost:5173/)
npm run dev

# Build for production
npm run build
```

## Current Status

✅ Phase 0: Preprocessing tools - COMPLETE
✅ Phase 1: React + TypeScript + Vite setup - COMPLETE
⏳ Phase 2: View 1 - Computation Graph
⏳ Phase 3: View 2 - Timeline & Trace
⏳ Phase 4: View 3 - Memory Heatmap
⏳ Phase 5: View 4 - 3D Transformer

See `../WEBUI.md` for complete implementation plan.
