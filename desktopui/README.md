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

# Run
./bin/tensor-trace-analyzer
```

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
- [ ] Dependencies installed
- [ ] Minimal window working
- [ ] Data loading implemented
- [ ] Visualization implemented

## Data Files

The `data/` directory contains real trace data from GPT-OSS-20B model:

- **Model**: GPT-OSS-20B (12.85 GB, 24 layers, 32 experts per layer)
- **Tensors**: 2,691 (including expert-level granularity)
- **Trace entries**: 1,696 for single token
- **Duration**: 649.5ms inference time

## Next Steps

1. Install GLFW: `brew install glfw`
2. Download ImGui: `cd external && git clone https://github.com/ocornut/imgui.git`
3. Build minimal window
4. Add JSON loading
5. Implement visualization
