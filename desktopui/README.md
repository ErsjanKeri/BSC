# Tensor Trace Analyzer - Desktop UI

C++ desktop application for analyzing large-scale tensor traces from LLM inference.

## Architecture

- **GUI**: Dear ImGui (immediate mode)
- **Graphics**: OpenGL 3.3+
- **Windowing**: GLFW
- **Language**: C++17

## Why Desktop UI?

The React WebUI works great for 1-2 tokens (~1,700 entries) but struggles with 100+ tokens (170,000+ entries). This C++ desktop app uses:
- Virtual scrolling (ImGuiListClipper) for millions of rows
- GPU-accelerated rendering
- Native performance

## Prerequisites (macOS)

1. **Xcode Command Line Tools**
   ```bash
   xcode-select --install
   ```

2. **Homebrew** (if not installed)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **CMake and GLFW**
   ```bash
   brew install cmake glfw
   ```

## Setup Dependencies

### 1. Download ImGui

```bash
cd external
git clone https://github.com/ocornut/imgui.git
cd imgui
git checkout v1.90.1  # Stable version
cd ../..
```

## Build

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
cmake --build .
```

## Usage

### Single Domain

```bash
./build/bin/tensor-trace-analyzer <domain-path>

# Example:
./build/bin/tensor-trace-analyzer ../tensor-tracing/expert-analysis-2026-01-26/domain-1-code
```

### All 5 Domains at Once

```bash
./launch_all_domains.sh
```

This will open 5 separate windows, one for each domain, allowing side-by-side comparison.

## Project Structure

```
desktopui/
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── data/                   # JSON data files
│   ├── memory-map.json     # 2,691 tensors (12.85 GB model)
│   ├── buffer-timeline.json
│   ├── traces/
│   │   └── token-00000.json  # 1,696 trace entries
│   └── graphs/
│       └── token-00000.json  # Computation graph
├── external/               # Third-party libraries
│   └── imgui/              # Dear ImGui (to be downloaded)
├── shaders/                # OpenGL shaders (future)
└── src/
    └── main.cpp            # Application entry point
```

## Current Status

- [x] Project structure created
- [x] CMakeLists.txt configured
- [x] Data files copied (memory-map, traces, graphs)
- [x] Dependencies installed
- [x] Multi-token visualization working
- [x] 100+ token support
- [x] Accumulated heatmap
- [x] Per-token heatmap rows
- [x] Command-line domain selection

## Features

### Multi-Token Heatmap Visualization

- **100 token rows**: Each row shows memory access pattern for one token
- **Accumulated row**: Bottom row shows total accesses across all 100 tokens
- **Scrollable**: Vertical scrolling to view all tokens
- **Interactive**: Hover shows token ID, tensor name, access count
- **File offset X-axis**: Shows access patterns across 12.85 GB model file

### Accumulated Access Graph

- Shows cumulative access pattern across all tokens
- Step function with quantitative Y-axis
- Helps identify hot memory regions

## Data Format

Each domain directory should contain:
```
domain-X-name/
├── memory-map.json          # GGUF model structure
├── traces/
│   ├── token-00000.json     # Trace for token 0
│   ├── token-00001.json     # Trace for token 1
│   └── ... (up to 100 tokens)
└── graphs/                   # Optional computation graphs
```

**Model**: GPT-OSS-20B (12.85 GB, 24 layers, 32 experts per layer)
**Tensors**: 2,691 (including expert-level granularity)
**MoE Operations**: ~72 per token (3 per layer × 24 layers)

## Next Steps

1. Install GLFW: `brew install glfw`
2. Download ImGui: `cd external && git clone https://github.com/ocornut/imgui.git`
3. Build minimal window
4. Add JSON loading
5. Implement visualization
