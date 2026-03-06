# Desktop UI Application Plan: Tensor Trace Analyzer

**Date:** 2026-01-17
**Project:** BSC Thesis - SSD-Backed LLM Inference Optimization
**Goal:** Build a C++ desktop application for analyzing large-scale tensor traces (100+ tokens, 170,000+ entries)

---

## Executive Summary

The current React WebUI is perfect for **validation** (1-2 tokens), but cannot handle **large-scale analysis** (100+ tokens) due to browser memory limits and performance constraints. This plan describes a C++ desktop application using ImGui (GUI) + DuckDB (analytics) that can:

- Handle **millions of trace entries** smoothly (60 FPS rendering)
- Run **analytical SQL queries** for pattern detection
- Render **GPU-accelerated heatmaps** (13 GB file layout)
- Export **publication-quality figures** for thesis
- Support **interactive exploration** with zoom/filter/correlation

**Development approach:** Incremental (build minimal prototype first, add features iteratively)
**Timeline:** 8-12 days total (3-4 days for working prototype)
**Platform:** macOS primary, Linux secondary (TUM server)

---

## Part 1: Requirements & Constraints

### Functional Requirements

**Must Have (Prototype):**
1. Load binary trace files (`tensor_trace.bin`, 1024-byte format)
2. Display trace entries in scrollable table (virtual scrolling for performance)
3. Show memory heatmap (file layout with access counts)
4. Basic filtering (by layer, token range)
5. Load memory map JSON (`memory-map.json`)

**Should Have (Full Version):**
6. SQL query panel (DuckDB integration)
7. Timeline scrubber (temporal visualization)
8. Click correlation (trace entry → heatmap highlight)
9. Expert hotness analysis (which of 32 experts are used)
10. Export figures (PNG heatmaps, CSV query results)

**Nice to Have (Future):**
11. Batch analysis (process multiple experiments)
12. Seek distance calculator (sequential vs random access)
13. Comparison mode (before/after optimization)

### Performance Requirements

- **Load time:** <5 seconds for 100 tokens (170,000 entries)
- **Frame rate:** 60 FPS with smooth scrolling
- **Memory usage:** <4 GB for 1,000 tokens (1.7M entries)
- **Query latency:** <100ms for common analytical queries

### Platform Requirements

- **Primary:** macOS (development laptop)
- **Secondary:** Linux (TUM server cli-hiwi-02)
- **C++ Standard:** C++17
- **Dependencies:** ImGui, DuckDB, GLFW, OpenGL 3.3+

---

## Part 2: Architecture Overview

### Technology Stack

**1. GUI Framework: Dear ImGui**
- License: MIT (permissive, thesis-appropriate)
- Size: ~30 KB source code
- Rendering: Immediate mode (game-loop style, 60 FPS)
- Virtual scrolling: ImGuiListClipper (handles millions of rows)

**2. Database: DuckDB (embedded)**
- License: MIT
- Size: ~10 MB binary (single-file embedding)
- Purpose: Analytical queries, aggregations, filtering
- No server: Runs in-process

**3. Windowing: GLFW**
- License: zlib (permissive)
- Size: ~200 KB
- Purpose: Window creation, input handling, OpenGL context

**4. OpenGL: Version 3.3+ (Core Profile)**
- Purpose: GPU-accelerated heatmap rendering
- Textures: 1D texture for 13 GB file layout
- Shaders: Fragment shader computes heat colors

**5. Export: stb_image_write (PNG), Cairo (SVG)**
- License: Public domain / MIT
- Purpose: Generate thesis figures

### Application Architecture (3 Layers)

```
┌─────────────────────────────────────────────────────────────┐
│                    UI LAYER (ImGui)                         │
│  - Main window layout                                       │
│  - Table view (virtual scrolling)                           │
│  - Heatmap view (OpenGL rendering)                          │
│  - Query panel (SQL editor)                                 │
│  - Controls (filters, timeline, export)                     │
├─────────────────────────────────────────────────────────────┤
│                   BUSINESS LOGIC LAYER                      │
│  - TraceDataManager (load/parse binary traces)             │
│  - MemoryMapManager (load/parse memory-map.json)           │
│  - CorrelationIndex (name-based tensor correlation)        │
│  - QueryBuilder (construct SQL from UI controls)           │
│  - ExportManager (generate PNG/SVG/CSV)                    │
├─────────────────────────────────────────────────────────────┤
│                    DATA LAYER (DuckDB)                      │
│  - traces table (all trace entries)                        │
│  - memory_map table (GGUF file layout)                     │
│  - Materialized views (expert_hotness, layer_timeline)     │
│  - Indices (token_id, layer_id, timestamp_ns)             │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 3: Data Model Design

### DuckDB Schema

#### Table 1: traces

```sql
CREATE TABLE traces (
    entry_id BIGINT PRIMARY KEY,
    timestamp_ns BIGINT NOT NULL,
    token_id INTEGER NOT NULL,
    layer_id SMALLINT,              -- NULL for non-layer tensors
    thread_id SMALLINT NOT NULL,
    operation_type VARCHAR NOT NULL, -- "MUL_MAT", "ADD", etc.
    phase VARCHAR NOT NULL,         -- "PROMPT" or "GENERATE"

    dst_name VARCHAR NOT NULL,      -- Destination tensor

    -- Source tensors (flattened, NULL if not present)
    src0_name VARCHAR,
    src0_offset BIGINT,
    src0_size BIGINT,
    src0_memory VARCHAR,            -- 'DISK' or 'BUFFER'

    src1_name VARCHAR,
    src1_offset BIGINT,
    src1_size BIGINT,
    src1_memory VARCHAR,

    src2_name VARCHAR,
    src2_offset BIGINT,
    src2_size BIGINT,
    src2_memory VARCHAR,

    src3_name VARCHAR,
    src3_offset BIGINT,
    src3_size BIGINT,
    src3_memory VARCHAR,

    -- MoE expert IDs (array, NULL for non-MoE ops)
    expert_ids INTEGER[],
    num_experts TINYINT
);

CREATE INDEX idx_token ON traces(token_id);
CREATE INDEX idx_layer ON traces(layer_id);
CREATE INDEX idx_timestamp ON traces(timestamp_ns);
CREATE INDEX idx_operation ON traces(operation_type);
```

#### Table 2: memory_map

```sql
CREATE TABLE memory_map (
    tensor_name VARCHAR PRIMARY KEY,
    offset_start BIGINT NOT NULL,
    offset_end BIGINT NOT NULL,
    size_bytes BIGINT NOT NULL,
    layer_id SMALLINT,
    component VARCHAR,              -- "query", "key", "value", "gate", etc.
    category VARCHAR NOT NULL,      -- "embedding", "attention", "ffn", "norm"
    expert_id SMALLINT              -- NULL for non-expert tensors
);

CREATE INDEX idx_offset ON memory_map(offset_start);
CREATE INDEX idx_layer_mem ON memory_map(layer_id);
CREATE INDEX idx_expert ON memory_map(expert_id) WHERE expert_id IS NOT NULL;
```

#### Materialized Views (Pre-computed)

```sql
-- Expert access hotness (layer × expert matrix)
CREATE VIEW expert_hotness AS
SELECT
    layer_id,
    UNNEST(expert_ids) as expert_id,
    COUNT(*) as access_count,
    COUNT(DISTINCT token_id) as token_count
FROM traces
WHERE operation_type = 'MUL_MAT_ID' AND expert_ids IS NOT NULL
GROUP BY layer_id, expert_id;

-- Layer execution timeline
CREATE VIEW layer_timeline AS
SELECT
    token_id,
    layer_id,
    MIN(timestamp_ns) as start_ns,
    MAX(timestamp_ns) as end_ns,
    (MAX(timestamp_ns) - MIN(timestamp_ns)) / 1e6 as duration_ms,
    COUNT(*) as operation_count
FROM traces
WHERE layer_id IS NOT NULL
GROUP BY token_id, layer_id;

-- Memory access hotness (which file regions are hot)
CREATE VIEW memory_hotness AS
SELECT
    m.tensor_name,
    m.offset_start,
    m.size_bytes,
    m.layer_id,
    m.expert_id,
    COUNT(DISTINCT t.entry_id) as access_count,
    COUNT(DISTINCT t.token_id) as token_count
FROM memory_map m
LEFT JOIN traces t ON (
    m.tensor_name = t.src0_name OR
    m.tensor_name = t.src1_name OR
    m.tensor_name = t.src2_name OR
    m.tensor_name = t.src3_name
)
GROUP BY m.tensor_name, m.offset_start, m.size_bytes, m.layer_id, m.expert_id;

-- Token statistics
CREATE VIEW token_stats AS
SELECT
    token_id,
    MIN(timestamp_ns) / 1e6 as start_ms,
    MAX(timestamp_ns) / 1e6 as end_ms,
    (MAX(timestamp_ns) - MIN(timestamp_ns)) / 1e6 as duration_ms,
    COUNT(*) as operation_count,
    COUNT(DISTINCT layer_id) as layers_accessed
FROM traces
GROUP BY token_id;
```

### C++ Data Structures

```cpp
// Compact in-memory representation (parsed from binary)
struct TraceEntry {
    uint64_t timestamp_ns;
    uint32_t token_id;
    uint16_t layer_id;      // 65535 = NULL
    uint16_t thread_id;
    uint8_t operation_type; // ggml_op enum
    uint8_t phase;          // 0=PROMPT, 1=GENERATE
    uint8_t num_sources;
    uint8_t num_experts;

    std::string dst_name;

    struct Source {
        std::string name;
        uint64_t offset;    // disk_offset or buffer_id
        uint32_t size_bytes;
        uint8_t memory_source; // 0=DISK, 1=BUFFER
    };
    Source sources[4];

    std::vector<int32_t> expert_ids;
};

// Memory map tensor
struct MemoryTensor {
    std::string name;
    uint64_t offset_start;
    uint64_t offset_end;
    uint64_t size_bytes;
    std::vector<uint64_t> shape;
    std::string category;
    int16_t layer_id;       // -1 = NULL
    std::string component;
    int16_t expert_id;      // -1 = NULL
};

// Heatmap data (GPU texture)
struct HeatmapData {
    uint64_t total_file_size;
    std::vector<float> heat_values;  // Per-megabyte access count
    GLuint texture_id;               // OpenGL texture handle
    float max_heat;                  // For normalization
};
```

---

## Part 4: Implementation Phases

### Phase 1: Foundation & Minimal Window (Days 1-2)

**Goal:** Get ImGui + GLFW + OpenGL running with empty window

**Tasks:**

**1.1: Project Setup**
- Create directory: `BSC/desktopui/`
- Create CMakeLists.txt
- Download dependencies (ImGui, GLFW, DuckDB)
- Verify build on macOS

**1.2: Dependencies Setup**

Directory structure:
```
BSC/desktopui/
├── CMakeLists.txt
├── external/
│   ├── imgui/              # Clone from GitHub
│   ├── glfw/               # Install via Homebrew or build
│   ├── duckdb/             # Download pre-built binary
│   └── stb/                # stb_image_write.h
├── src/
│   └── main.cpp           # Entry point
└── build/                 # CMake build directory
```

**1.3: Minimal ImGui Window**

`src/main.cpp` (minimal version):
```cpp
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <iostream>

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // OpenGL 3.3 Core Profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // macOS

    // Create window
    GLFWwindow* window = glfwCreateWindow(1920, 1080,
                                           "Tensor Trace Analyzer",
                                           nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // Setup ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Render UI
        ImGui::Begin("Tensor Trace Analyzer - Prototype");
        ImGui::Text("Application ready!");
        ImGui::Text("FPS: %.1f", io.Framerate);
        ImGui::End();

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
```

**Verification:**
```bash
cd BSC/desktopui
mkdir build && cd build
cmake ..
make
./tensor-trace-analyzer
```

**Expected:** Window opens, shows "Application ready!" and FPS counter

---

### Phase 2: Binary Trace Loader (Day 3)

**Goal:** Load and parse `tensor_trace.bin` into C++ structures

**2.1: Create TraceLoader class**

`src/trace_loader.h`:
```cpp
#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Match the binary format from tensor_trace.h
struct TraceEntry {
    uint64_t timestamp_ns;
    uint32_t token_id;
    uint16_t layer_id;
    uint16_t thread_id;
    uint8_t operation_type;
    uint8_t phase;
    uint8_t num_sources;
    uint8_t num_experts;

    std::string dst_name;

    struct Source {
        std::string name;
        uint64_t offset;
        uint32_t size_bytes;
        uint8_t memory_source; // 0=DISK, 1=BUFFER
    };
    std::vector<Source> sources;

    std::vector<int32_t> expert_ids;
};

class TraceLoader {
public:
    bool LoadFromFile(const std::string& filepath);
    const std::vector<TraceEntry>& GetEntries() const { return entries_; }
    size_t GetEntryCount() const { return entries_.size(); }

private:
    std::vector<TraceEntry> entries_;

    // Parse single 1024-byte binary entry
    bool ParseEntry(const uint8_t* data, TraceEntry& entry);
};
```

`src/trace_loader.cpp`:
```cpp
#include "trace_loader.h"
#include <fstream>
#include <cstring>
#include <iostream>

// Binary format offsets (must match tensor_trace.h exactly!)
constexpr size_t ENTRY_SIZE = 1024;
constexpr size_t DST_NAME_OFFSET = 24;
constexpr size_t DST_NAME_SIZE = 128;
constexpr size_t SOURCES_OFFSET = 152;
constexpr size_t SOURCE_SIZE = 160;
constexpr size_t EXPERT_IDS_OFFSET = 792;
constexpr size_t NUM_EXPERTS_OFFSET = 856;

bool TraceLoader::LoadFromFile(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open: " << filepath << std::endl;
        return false;
    }

    // Get file size
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size % ENTRY_SIZE != 0) {
        std::cerr << "Invalid trace file size: " << file_size
                  << " (not multiple of " << ENTRY_SIZE << ")" << std::endl;
        return false;
    }

    size_t entry_count = file_size / ENTRY_SIZE;
    std::cout << "Loading " << entry_count << " trace entries..." << std::endl;

    entries_.reserve(entry_count);

    // Read and parse entries
    std::vector<uint8_t> buffer(ENTRY_SIZE);
    size_t loaded = 0;

    while (file.read(reinterpret_cast<char*>(buffer.data()), ENTRY_SIZE)) {
        TraceEntry entry;
        if (ParseEntry(buffer.data(), entry)) {
            entries_.push_back(std::move(entry));
            loaded++;

            // Progress logging
            if (loaded % 10000 == 0) {
                std::cout << "  Loaded " << loaded << " / " << entry_count
                          << " entries..." << std::endl;
            }
        }
    }

    std::cout << "✓ Loaded " << entries_.size() << " entries" << std::endl;
    return true;
}

bool TraceLoader::ParseEntry(const uint8_t* data, TraceEntry& entry) {
    // Parse metadata (24 bytes)
    std::memcpy(&entry.timestamp_ns, data + 0, 8);
    std::memcpy(&entry.token_id, data + 8, 4);
    std::memcpy(&entry.layer_id, data + 12, 2);
    std::memcpy(&entry.thread_id, data + 14, 2);
    std::memcpy(&entry.operation_type, data + 16, 1);
    std::memcpy(&entry.phase, data + 17, 1);
    std::memcpy(&entry.num_sources, data + 18, 1);

    // Skip if timestamp is 0 (empty entry)
    if (entry.timestamp_ns == 0) {
        return false;
    }

    // Parse destination name (128 bytes at offset 24)
    char dst_name_buf[129] = {0};
    std::memcpy(dst_name_buf, data + DST_NAME_OFFSET, DST_NAME_SIZE);
    entry.dst_name = std::string(dst_name_buf);

    // Parse sources (4 × 160 bytes at offset 152)
    entry.sources.reserve(entry.num_sources);
    for (uint8_t i = 0; i < entry.num_sources && i < 4; i++) {
        size_t src_offset = SOURCES_OFFSET + (i * SOURCE_SIZE);

        TraceEntry::Source src;

        // Parse source name (128 bytes)
        char src_name_buf[129] = {0};
        std::memcpy(src_name_buf, data + src_offset, 128);
        src.name = std::string(src_name_buf);

        // Parse source metadata
        uint64_t tensor_ptr;
        std::memcpy(&tensor_ptr, data + src_offset + 128, 8);
        std::memcpy(&src.size_bytes, data + src_offset + 136, 4);
        std::memcpy(&src.memory_source, data + src_offset + 142, 1);
        std::memcpy(&src.offset, data + src_offset + 144, 8);

        entry.sources.push_back(src);
    }

    // Parse expert IDs (64 bytes at offset 792)
    std::memcpy(&entry.num_experts, data + NUM_EXPERTS_OFFSET, 1);

    if (entry.num_experts > 0) {
        entry.expert_ids.reserve(entry.num_experts);
        int32_t expert_id_buf[16];
        std::memcpy(expert_id_buf, data + EXPERT_IDS_OFFSET, 64);

        for (uint8_t i = 0; i < entry.num_experts && i < 16; i++) {
            entry.expert_ids.push_back(expert_id_buf[i]);
        }
    }

    return true;
}
```

**Verification:**
```cpp
// In main.cpp
TraceLoader loader;
if (loader.LoadFromFile("/tmp/tensor_trace.bin")) {
    ImGui::Text("Loaded %zu entries", loader.GetEntryCount());
}
```

**Expected:** Loads binary file, shows entry count in UI

---

### Phase 3: DuckDB Integration (Day 4)

**Goal:** Import trace data into DuckDB for SQL queries

**3.1: Create DatabaseManager class**

`src/database_manager.h`:
```cpp
#pragma once
#include "duckdb.hpp"
#include "trace_loader.h"
#include <memory>

class DatabaseManager {
public:
    DatabaseManager();
    ~DatabaseManager();

    // Initialize in-memory database
    bool Initialize();

    // Import trace data
    bool ImportTraces(const std::vector<TraceEntry>& entries);

    // Import memory map
    bool ImportMemoryMap(const std::string& json_path);

    // Execute query
    std::unique_ptr<duckdb::MaterializedQueryResult> Query(const std::string& sql);

    // Create materialized views
    bool CreateViews();

private:
    std::unique_ptr<duckdb::DuckDB> db_;
    std::unique_ptr<duckdb::Connection> conn_;

    bool CreateTracesTable();
    bool CreateMemoryMapTable();
};
```

`src/database_manager.cpp`:
```cpp
#include "database_manager.h"
#include <iostream>

DatabaseManager::DatabaseManager() = default;
DatabaseManager::~DatabaseManager() = default;

bool DatabaseManager::Initialize() {
    try {
        // Create in-memory database
        db_ = std::make_unique<duckdb::DuckDB>(nullptr);
        conn_ = std::make_unique<duckdb::Connection>(*db_);

        std::cout << "✓ DuckDB initialized (in-memory)" << std::endl;

        // Create schema
        if (!CreateTracesTable()) return false;
        if (!CreateMemoryMapTable()) return false;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "DuckDB initialization failed: " << e.what() << std::endl;
        return false;
    }
}

bool DatabaseManager::CreateTracesTable() {
    const char* sql = R"(
        CREATE TABLE traces (
            entry_id BIGINT PRIMARY KEY,
            timestamp_ns BIGINT,
            token_id INTEGER,
            layer_id SMALLINT,
            thread_id SMALLINT,
            operation_type VARCHAR,
            phase VARCHAR,
            dst_name VARCHAR,
            src0_name VARCHAR,
            src0_offset BIGINT,
            src0_size BIGINT,
            src0_memory VARCHAR,
            src1_name VARCHAR,
            src1_offset BIGINT,
            src1_size BIGINT,
            src1_memory VARCHAR,
            src2_name VARCHAR,
            src2_offset BIGINT,
            src2_size BIGINT,
            src2_memory VARCHAR,
            src3_name VARCHAR,
            src3_offset BIGINT,
            src3_size BIGINT,
            src3_memory VARCHAR,
            expert_ids INTEGER[],
            num_experts TINYINT
        )
    )";

    try {
        conn_->Query(sql);
        std::cout << "✓ Created traces table" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create traces table: " << e.what() << std::endl;
        return false;
    }
}

bool DatabaseManager::ImportTraces(const std::vector<TraceEntry>& entries) {
    std::cout << "Importing " << entries.size() << " entries to DuckDB..." << std::endl;

    try {
        // Use Appender for bulk insert (much faster than prepared statements)
        duckdb::Appender appender(*conn_, "traces");

        for (size_t i = 0; i < entries.size(); i++) {
            const auto& e = entries[i];

            appender.BeginRow();
            appender.Append<int64_t>(i);  // entry_id
            appender.Append<int64_t>(e.timestamp_ns);
            appender.Append<int32_t>(e.token_id);

            // Handle NULL layer_id
            if (e.layer_id == 65535) {
                appender.Append<int16_t>(nullptr);
            } else {
                appender.Append<int16_t>(e.layer_id);
            }

            appender.Append<int16_t>(e.thread_id);
            appender.Append<std::string>(OperationTypeToString(e.operation_type));
            appender.Append<std::string>(e.phase == 0 ? "PROMPT" : "GENERATE");
            appender.Append<std::string>(e.dst_name);

            // Append sources (NULL if not present)
            for (int s = 0; s < 4; s++) {
                if (s < e.sources.size()) {
                    appender.Append<std::string>(e.sources[s].name);
                    appender.Append<int64_t>(e.sources[s].offset);
                    appender.Append<int64_t>(e.sources[s].size_bytes);
                    appender.Append<std::string>(
                        e.sources[s].memory_source == 0 ? "DISK" : "BUFFER"
                    );
                } else {
                    appender.Append<std::string>(nullptr);
                    appender.Append<int64_t>(nullptr);
                    appender.Append<int64_t>(nullptr);
                    appender.Append<std::string>(nullptr);
                }
            }

            // Append expert IDs
            if (e.expert_ids.empty()) {
                appender.Append<std::vector<int32_t>>(nullptr);
                appender.Append<int8_t>(0);
            } else {
                appender.Append<std::vector<int32_t>>(e.expert_ids);
                appender.Append<int8_t>(e.num_experts);
            }

            appender.EndRow();

            // Progress
            if ((i + 1) % 10000 == 0) {
                std::cout << "  Inserted " << (i + 1) << " entries..." << std::endl;
            }
        }

        appender.Flush();

        std::cout << "✓ Imported " << entries.size() << " entries to DuckDB" << std::endl;

        // Create indices
        conn_->Query("CREATE INDEX idx_token ON traces(token_id)");
        conn_->Query("CREATE INDEX idx_layer ON traces(layer_id)");

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Import failed: " << e.what() << std::endl;
        return false;
    }
}
```

**Verification:**
```cpp
// In main.cpp
TraceLoader loader;
loader.LoadFromFile("/tmp/tensor_trace.bin");

DatabaseManager db;
db.Initialize();
db.ImportTraces(loader.GetEntries());

auto result = db.Query("SELECT COUNT(*) FROM traces");
// Display result in ImGui
```

**Expected:** DuckDB table populated, query returns correct count

---

### Phase 4: Virtual Scrolling Table View (Day 5)

**Goal:** Display trace entries in efficient scrolling table

**4.1: Create TraceTableView class**

`src/views/trace_table_view.h`:
```cpp
#pragma once
#include "../database_manager.h"
#include "imgui.h"
#include <vector>
#include <string>

struct DisplayEntry {
    int64_t entry_id;
    double time_ms;
    std::string operation;
    std::string destination;
    std::string sources;
    int16_t layer;
    std::string memory;
    double size_mb;
};

class TraceTableView {
public:
    TraceTableView(DatabaseManager* db);

    void Render();
    void SetFilter(const std::string& sql_where_clause);
    void Refresh();

private:
    DatabaseManager* db_;
    std::vector<DisplayEntry> entries_;
    std::string filter_;

    void LoadEntries();
    void RenderTable();
};
```

`src/views/trace_table_view.cpp`:
```cpp
#include "trace_table_view.h"
#include <sstream>

TraceTableView::TraceTableView(DatabaseManager* db) : db_(db) {
    LoadEntries();
}

void TraceTableView::LoadEntries() {
    std::string sql = R"(
        SELECT
            entry_id,
            timestamp_ns / 1e6 as time_ms,
            operation_type,
            dst_name,
            src0_name || CASE WHEN num_sources > 1 THEN ' +' || (num_sources-1) ELSE '' END as sources,
            layer_id,
            src0_memory as memory,
            (src0_size + COALESCE(src1_size, 0) + COALESCE(src2_size, 0) + COALESCE(src3_size, 0)) / (1024*1024.0) as size_mb
        FROM traces
    )";

    if (!filter_.empty()) {
        sql += " WHERE " + filter_;
    }

    sql += " ORDER BY entry_id LIMIT 100000";  // Safety limit

    auto result = db_->Query(sql);

    entries_.clear();
    if (!result || result->HasError()) {
        std::cerr << "Query failed: " << result->GetErrorObject() << std::endl;
        return;
    }

    // Convert to DisplayEntry
    for (size_t row = 0; row < result->RowCount(); row++) {
        DisplayEntry e;
        e.entry_id = result->GetValue(0, row).GetValue<int64_t>();
        e.time_ms = result->GetValue(1, row).GetValue<double>();
        e.operation = result->GetValue(2, row).GetValue<std::string>();
        e.destination = result->GetValue(3, row).GetValue<std::string>();
        e.sources = result->GetValue(4, row).GetValue<std::string>();

        if (result->GetValue(5, row).IsNull()) {
            e.layer = -1;
        } else {
            e.layer = result->GetValue(5, row).GetValue<int16_t>();
        }

        e.memory = result->GetValue(6, row).GetValue<std::string>();
        e.size_mb = result->GetValue(7, row).GetValue<double>();

        entries_.push_back(e);
    }

    std::cout << "✓ Loaded " << entries_.size() << " entries for table view" << std::endl;
}

void TraceTableView::Render() {
    ImGui::Begin("Trace Log");

    // Filter controls
    static char filter_buf[256] = "";
    if (ImGui::InputText("Filter (SQL WHERE)", filter_buf, sizeof(filter_buf),
                          ImGuiInputTextFlags_EnterReturnsTrue)) {
        SetFilter(filter_buf);
    }

    ImGui::SameLine();
    if (ImGui::Button("Refresh")) {
        Refresh();
    }

    ImGui::Text("Entries: %zu", entries_.size());
    ImGui::Separator();

    RenderTable();

    ImGui::End();
}

void TraceTableView::RenderTable() {
    // Virtual scrolling table
    ImGuiTableFlags flags = ImGuiTableFlags_ScrollY |
                            ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_BordersOuter |
                            ImGuiTableFlags_BordersV |
                            ImGuiTableFlags_Resizable;

    if (ImGui::BeginTable("traces_table", 8, flags)) {
        // Setup columns
        ImGui::TableSetupScrollFreeze(0, 1);  // Freeze header row
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 50.0f);
        ImGui::TableSetupColumn("Time (ms)", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Operation", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("Destination", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Sources", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Layer", ImGuiTableColumnFlags_WidthFixed, 50.0f);
        ImGui::TableSetupColumn("Mem", ImGuiTableColumnFlags_WidthFixed, 50.0f);
        ImGui::TableSetupColumn("Size (MB)", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableHeadersRow();

        // VIRTUAL SCROLLING - only render visible rows
        ImGuiListClipper clipper;
        clipper.Begin(entries_.size());

        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                const auto& e = entries_[row];

                ImGui::TableNextRow();

                ImGui::TableNextColumn();
                ImGui::Text("%lld", e.entry_id);

                ImGui::TableNextColumn();
                ImGui::Text("%.2f", e.time_ms);

                ImGui::TableNextColumn();
                ImGui::Text("%s", e.operation.c_str());

                ImGui::TableNextColumn();
                ImGui::Text("%s", e.destination.c_str());

                ImGui::TableNextColumn();
                ImGui::Text("%s", e.sources.c_str());

                ImGui::TableNextColumn();
                if (e.layer == -1) {
                    ImGui::Text("-");
                } else {
                    ImGui::Text("%d", e.layer);
                }

                ImGui::TableNextColumn();
                ImGui::Text("%s", e.memory.c_str());

                ImGui::TableNextColumn();
                ImGui::Text("%.1f", e.size_mb);
            }
        }

        ImGui::EndTable();
    }
}

void TraceTableView::SetFilter(const std::string& filter) {
    filter_ = filter;
    Refresh();
}

void TraceTableView::Refresh() {
    LoadEntries();
}
```

**Verification:**
```cpp
// In main.cpp
TraceTableView table_view(&db);
table_view.Render();
```

**Expected:** Table shows trace entries, scrolls smoothly with 100K+ rows

---

### Phase 5: Memory Heatmap (OpenGL Texture) (Day 6)

**Goal:** GPU-accelerated heatmap showing file layout and access patterns

**5.1: Create HeatmapRenderer class**

`src/views/heatmap_renderer.h`:
```cpp
#pragma once
#include <GL/gl3w.h>  // or glad/glew
#include <vector>
#include <cstdint>

class HeatmapRenderer {
public:
    HeatmapRenderer();
    ~HeatmapRenderer();

    // Initialize OpenGL resources
    bool Initialize(uint64_t file_size_bytes);

    // Update heat data from access counts
    void UpdateHeat(const std::vector<std::pair<uint64_t, uint32_t>>& offset_access_pairs);

    // Render heatmap
    void Render(float zoom, float pan_x, float pan_y, int width, int height);

    // Export to PNG
    bool ExportToPNG(const std::string& filepath, int width, int height);

private:
    GLuint texture_id_;
    GLuint shader_program_;
    GLuint vao_, vbo_;

    uint64_t file_size_bytes_;
    uint64_t granularity_bytes_;  // Bytes per texel (e.g., 1 MB)
    std::vector<float> heat_data_;
    float max_heat_;

    bool CompileShaders();
    void CreateQuadGeometry();
};
```

**5.2: Fragment Shader for Heat Coloring**

`shaders/heatmap.frag`:
```glsl
#version 330 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler1D heatTexture;  // Heat values (0.0 - 1.0)
uniform float maxHeat;

// Heat colormap (gray → dark red → bright red)
vec3 heatColor(float normalized) {
    if (normalized <= 0.0) {
        return vec3(0.22, 0.25, 0.32);  // Gray (no access)
    }

    // Dark red → bright red gradient
    float r = 0.55 + (0.45 * normalized);  // 0.55 → 1.0
    return vec3(r, 0.0, 0.0);
}

void main() {
    // Sample heat value from texture
    float heat = texture(heatTexture, TexCoord.x).r;

    // Normalize by max heat
    float normalized = heat / maxHeat;

    // Compute color
    vec3 color = heatColor(normalized);

    FragColor = vec4(color, 1.0);
}
```

**5.3: Render Implementation**

`src/views/heatmap_renderer.cpp` (key parts):
```cpp
bool HeatmapRenderer::Initialize(uint64_t file_size_bytes) {
    file_size_bytes_ = file_size_bytes;
    granularity_bytes_ = 1024 * 1024;  // 1 MB per texel

    // Texture size (1 texel = 1 MB)
    size_t texture_width = (file_size_bytes + granularity_bytes_ - 1) / granularity_bytes_;
    heat_data_.resize(texture_width, 0.0f);

    std::cout << "Heatmap texture: " << texture_width << " texels ("
              << file_size_bytes / (1024*1024*1024) << " GB file)" << std::endl;

    // Create 1D texture
    glGenTextures(1, &texture_id_);
    glBindTexture(GL_TEXTURE_1D, texture_id_);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, texture_width, 0,
                 GL_RED, GL_FLOAT, heat_data_.data());
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Compile shaders
    if (!CompileShaders()) {
        return false;
    }

    // Create quad geometry
    CreateQuadGeometry();

    return true;
}

void HeatmapRenderer::UpdateHeat(
    const std::vector<std::pair<uint64_t, uint32_t>>& offset_access_pairs) {

    // Reset heat
    std::fill(heat_data_.begin(), heat_data_.end(), 0.0f);
    max_heat_ = 0.0f;

    // Accumulate accesses
    for (const auto& [offset, count] : offset_access_pairs) {
        size_t texel = offset / granularity_bytes_;
        if (texel < heat_data_.size()) {
            heat_data_[texel] += count;
            max_heat_ = std::max(max_heat_, heat_data_[texel]);
        }
    }

    // Upload to GPU
    glBindTexture(GL_TEXTURE_1D, texture_id_);
    glTexSubImage1D(GL_TEXTURE_1D, 0, 0, heat_data_.size(),
                    GL_RED, GL_FLOAT, heat_data_.data());
}

void HeatmapRenderer::Render(float zoom, float pan_x, float pan_y,
                              int width, int height) {
    // Use shader program
    glUseProgram(shader_program_);

    // Bind texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_1D, texture_id_);
    glUniform1i(glGetUniformLocation(shader_program_, "heatTexture"), 0);
    glUniform1f(glGetUniformLocation(shader_program_, "maxHeat"), max_heat_);

    // Set transform uniforms (zoom/pan)
    glUniform1f(glGetUniformLocation(shader_program_, "zoom"), zoom);
    glUniform2f(glGetUniformLocation(shader_program_, "pan"), pan_x, pan_y);

    // Draw quad
    glBindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}
```

**Verification:**
```cpp
// In main.cpp - query heat data from DuckDB
auto heat_result = db.Query(R"(
    SELECT src0_offset, COUNT(*) as access_count
    FROM traces
    WHERE src0_memory = 'DISK'
    GROUP BY src0_offset
)");

std::vector<std::pair<uint64_t, uint32_t>> heat_data;
// ... convert result to heat_data ...

heatmap.UpdateHeat(heat_data);
heatmap.Render(1.0f, 0.0f, 0.0f, width, height);
```

**Expected:** Heatmap shows file layout with red gradient for hot regions

---

### Phase 6: SQL Query Panel (Day 7)

**Goal:** Interactive SQL query interface with template library

**6.1: Create QueryPanel class**

`src/views/query_panel.h`:
```cpp
#pragma once
#include "../database_manager.h"
#include "imgui.h"
#include <string>
#include <vector>

struct QueryTemplate {
    std::string name;
    std::string description;
    std::string sql;
};

class QueryPanel {
public:
    QueryPanel(DatabaseManager* db);

    void Render();

private:
    DatabaseManager* db_;
    char query_buffer_[8192];
    std::vector<QueryTemplate> templates_;

    // Query result display
    struct QueryResult {
        std::vector<std::string> column_names;
        std::vector<std::vector<std::string>> rows;
    };
    QueryResult current_result_;

    void InitializeTemplates();
    void ExecuteQuery();
    void RenderResults();
    void ExportResultsToCSV(const std::string& filepath);
};
```

**6.2: Query Templates**

```cpp
void QueryPanel::InitializeTemplates() {
    templates_ = {
        {
            "Expert Hotness (Global)",
            "Which experts are most frequently used across all layers?",
            R"(
SELECT
    UNNEST(expert_ids) as expert_id,
    COUNT(*) as total_accesses,
    COUNT(DISTINCT token_id) as tokens_using,
    COUNT(DISTINCT layer_id) as layers_using
FROM traces
WHERE operation_type = 'MUL_MAT_ID'
GROUP BY expert_id
ORDER BY total_accesses DESC
LIMIT 32
            )"
        },
        {
            "Expert Hotness (Per Layer)",
            "Expert usage breakdown for each layer",
            R"(
SELECT
    layer_id,
    UNNEST(expert_ids) as expert_id,
    COUNT(*) as access_count
FROM traces
WHERE operation_type = 'MUL_MAT_ID'
GROUP BY layer_id, expert_id
ORDER BY layer_id, access_count DESC
            )"
        },
        {
            "Layer Timeline",
            "Execution time for each layer per token",
            R"(
SELECT * FROM layer_timeline
ORDER BY token_id, layer_id
LIMIT 1000
            )"
        },
        {
            "Memory Hot Regions (100 MB chunks)",
            "Which file regions are accessed most frequently?",
            R"(
SELECT
    FLOOR(src0_offset / (100 * 1024 * 1024)) as region_100mb,
    COUNT(*) as access_count,
    SUM(src0_size) / (1024*1024) as total_mb_accessed
FROM traces
WHERE src0_memory = 'DISK'
GROUP BY region_100mb
ORDER BY region_100mb
            )"
        },
        {
            "Sequential vs Random Access",
            "Measure seek distances between consecutive layer accesses",
            R"(
WITH layer_accesses AS (
    SELECT
        token_id,
        layer_id,
        MIN(src0_offset) as layer_file_start
    FROM traces
    WHERE src0_memory = 'DISK'
    GROUP BY token_id, layer_id
)
SELECT
    token_id,
    layer_id,
    layer_file_start,
    layer_file_start - LAG(layer_file_start) OVER (
        PARTITION BY token_id ORDER BY layer_id
    ) as seek_distance_bytes,
    (layer_file_start - LAG(layer_file_start) OVER (
        PARTITION BY token_id ORDER BY layer_id
    )) / (1024*1024) as seek_distance_mb
FROM layer_accesses
ORDER BY token_id, layer_id
LIMIT 1000
            )"
        },
        {
            "Token Generation Performance",
            "Time per token breakdown",
            R"(
SELECT * FROM token_stats
ORDER BY token_id
            )"
        }
    };
}
```

**Verification:**
```cpp
// In main.cpp
QueryPanel query_panel(&db);
query_panel.Render();
```

**Expected:** SQL editor with templates, query execution, results table

---

## Part 5: CMake Build System

### CMakeLists.txt (Complete)

```cmake
cmake_minimum_required(VERSION 3.15)
project(TensorTraceAnalyzer VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenGL
find_package(OpenGL REQUIRED)

# GLFW (assume installed via Homebrew on macOS)
find_package(glfw3 REQUIRED)

# ImGui as static library
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/external/imgui)
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

# DuckDB
set(DUCKDB_DIR ${CMAKE_SOURCE_DIR}/external/duckdb)
add_library(duckdb SHARED IMPORTED)
set_target_properties(duckdb PROPERTIES
    IMPORTED_LOCATION "${DUCKDB_DIR}/libduckdb.dylib"  # macOS
    INTERFACE_INCLUDE_DIRECTORIES "${DUCKDB_DIR}/include"
)

# JSON parsing (nlohmann/json - header-only)
set(JSON_DIR ${CMAKE_SOURCE_DIR}/external/json)

# Main application
set(SOURCES
    src/main.cpp
    src/trace_loader.cpp
    src/database_manager.cpp
    src/views/trace_table_view.cpp
    src/views/heatmap_renderer.cpp
    src/views/query_panel.cpp
)

add_executable(tensor-trace-analyzer ${SOURCES})

target_include_directories(tensor-trace-analyzer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${JSON_DIR}/include
)

target_link_libraries(tensor-trace-analyzer
    imgui
    duckdb
    glfw
    OpenGL::GL
)

# Copy DuckDB library to build directory (for runtime)
add_custom_command(TARGET tensor-trace-analyzer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DUCKDB_DIR}/libduckdb.dylib"
        $<TARGET_FILE_DIR:tensor-trace-analyzer>
)

# Install targets
install(TARGETS tensor-trace-analyzer
    RUNTIME DESTINATION bin
)
```

---

## Part 6: Dependency Installation Guide

### macOS (Primary Platform)

**1. Install Homebrew packages:**
```bash
brew install cmake glfw
```

**2. Download ImGui:**
```bash
cd BSC/desktopui/external
git clone https://github.com/ocornut/imgui.git
cd imgui
git checkout v1.90.1  # Stable version
```

**3. Download DuckDB:**
```bash
cd BSC/desktopui/external
mkdir duckdb && cd duckdb

# Download pre-built binary for macOS
wget https://github.com/duckdb/duckdb/releases/download/v0.10.0/libduckdb-osx-universal.zip
unzip libduckdb-osx-universal.zip

# Should have:
# - libduckdb.dylib
# - duckdb.h
# - duckdb.hpp (C++ API)
```

**4. Download nlohmann/json (header-only):**
```bash
cd BSC/desktopui/external
git clone https://github.com/nlohmann/json.git
```

**5. Download STB (header-only):**
```bash
cd BSC/desktopui/external
mkdir stb && cd stb
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
```

### Linux (TUM Server)

**1. Install system packages:**
```bash
sudo apt-get update
sudo apt-get install cmake libglfw3-dev libgl1-mesa-dev
```

**2. Same steps for ImGui, DuckDB, nlohmann/json, STB**

---

## Part 7: Implementation Timeline

### Week 1: Foundation (Days 1-5)

**Day 1: Project Setup**
- Create BSC/desktopui/ directory
- Download all dependencies
- Create CMakeLists.txt
- Build minimal ImGui window
- **Deliverable:** Empty window opens at 60 FPS

**Day 2: Binary Trace Loader**
- Implement TraceLoader class
- Parse 1024-byte TensorAccessLog structs
- Handle expert IDs, multiple sources
- Test with /tmp/tensor_trace.bin (1-2 tokens)
- **Deliverable:** Trace entries loaded into C++ vectors

**Day 3: DuckDB Integration**
- Implement DatabaseManager class
- Create traces table schema
- Import trace data using Appender API
- Test simple queries (COUNT, WHERE filters)
- **Deliverable:** Data queryable via SQL

**Day 4: Virtual Scrolling Table**
- Implement TraceTableView class
- Use ImGuiListClipper for performance
- Query DuckDB for filtered data
- Display in table with columns
- **Deliverable:** Scrollable table with 100K+ rows

**Day 5: Basic Heatmap**
- Implement HeatmapRenderer class
- Create 1D OpenGL texture (file layout)
- Fragment shader for heat coloring
- Query access counts from DuckDB
- **Deliverable:** Static heatmap showing file layout

**Milestone:** **PROTOTYPE COMPLETE** (minimal working application)

---

### Week 2: Core Features (Days 6-10)

**Day 6: Memory Map Integration**
- Load memory-map.json (nlohmann/json)
- Import to DuckDB memory_map table
- Correlate tensor names with file offsets
- Show tensor boundaries in heatmap
- **Deliverable:** Heatmap shows labeled tensors

**Day 7: SQL Query Panel**
- Implement QueryPanel class
- Multi-line text editor for SQL
- Query execution with error handling
- Results table (reuse virtual scrolling)
- **Deliverable:** Interactive SQL queries working

**Day 8: Query Templates**
- Add 6+ predefined query templates
- Template picker (dropdown menu)
- Expert hotness query
- Layer timeline query
- Memory region analysis query
- **Deliverable:** One-click analytical queries

**Day 9: Timeline Controls**
- Token range slider (0-100)
- Layer filter (checkboxes for layers 0-23)
- Heatmap updates based on token range
- Table refreshes with filter
- **Deliverable:** Interactive filtering

**Day 10: Click Correlation**
- Click trace entry → highlight in heatmap
- Click heatmap region → filter table
- Bidirectional correlation (like WebUI)
- **Deliverable:** Full correlation working

**Milestone:** **FEATURE PARITY WITH WEBUI** (all essential features working)

---

### Week 3: Analysis & Export (Days 11-12)

**Day 11: Expert Analysis View**
- 2D heatmap: layer (Y-axis) × expert ID (X-axis)
- Color: access count
- Identify globally hot experts
- **Deliverable:** Expert usage visualization

**Day 12: Export for Thesis**
- Export heatmap to PNG (high-res, 8192px width)
- Export query results to CSV
- Batch export mode (all figures at once)
- **Deliverable:** Thesis-ready figures

**Milestone:** **PRODUCTION-READY TOOL**

---

## Part 8: File Structure

```
BSC/desktopui/
├── CMakeLists.txt
├── README.md
├── build/                      # CMake build output
├── external/                   # Third-party dependencies
│   ├── imgui/                  # Dear ImGui (git clone)
│   ├── duckdb/                 # DuckDB binary + headers
│   │   ├── libduckdb.dylib
│   │   ├── duckdb.h
│   │   └── duckdb.hpp
│   ├── json/                   # nlohmann/json
│   └── stb/                    # STB libraries
│       └── stb_image_write.h
├── shaders/                    # OpenGL shaders
│   ├── heatmap.vert
│   └── heatmap.frag
├── src/
│   ├── main.cpp                # Application entry point
│   ├── trace_loader.h/cpp      # Binary trace parser
│   ├── database_manager.h/cpp  # DuckDB wrapper
│   ├── memory_map_loader.h/cpp # JSON parser for memory map
│   ├── correlation_index.h/cpp # Name-based correlation
│   └── views/
│       ├── trace_table_view.h/cpp
│       ├── heatmap_renderer.h/cpp
│       ├── query_panel.h/cpp
│       └── export_panel.h/cpp
└── data/                       # Sample data for testing
    ├── sample_trace.bin
    └── sample_memory_map.json
```

---

## Part 9: Key Implementation Details

### Detail 1: Binary Parsing with memcpy

**Critical:** Must match tensor_trace.h layout exactly (1024 bytes)

**Offsets (from tensor_trace.h):**
- Metadata: 0-23 (24 bytes)
- dst_name: 24-151 (128 bytes)
- sources[4]: 152-791 (640 bytes = 4 × 160)
- expert_ids[16]: 792-855 (64 bytes)
- num_experts: 856 (1 byte)
- padding: 857-1023 (167 bytes)

**Parsing strategy:**
```cpp
std::memcpy(&entry.timestamp_ns, data + 0, 8);
std::memcpy(&entry.token_id, data + 8, 4);
// ... etc
```

**Safety checks:**
- Verify file size is multiple of 1024
- Check timestamp_ns != 0 (empty entry detection)
- Validate num_sources <= 4
- Validate num_experts <= 16

### Detail 2: Name-Based Correlation (Not Address-Based!)

**From WebUI investigation:** Address-based correlation FAILS because:
- Graph nodes have `ggml_tensor*` (struct address)
- Trace entries have `tensor->data*` (buffer address)
- These are different memory locations!

**Solution:** Normalize tensor names and match by string

```cpp
std::string NormalizeTensorName(const std::string& name) {
    std::string normalized = name;

    // Remove common suffixes
    const char* suffixes[] = {
        " (view)", " (reshaped)", " (permuted)", " (copy)"
    };

    for (const char* suffix : suffixes) {
        size_t pos = normalized.find(suffix);
        if (pos != std::string::npos) {
            normalized.erase(pos);
        }
    }

    // Trim whitespace
    // ... trim implementation ...

    return normalized;
}

// Build correlation index
std::unordered_map<std::string, std::vector<const TraceEntry*>> name_to_traces;
for (const auto& entry : all_entries) {
    std::string norm_name = NormalizeTensorName(entry.dst_name);
    name_to_traces[norm_name].push_back(&entry);

    for (const auto& src : entry.sources) {
        std::string norm_src = NormalizeTensorName(src.name);
        name_to_traces[norm_src].push_back(&entry);
    }
}
```

### Detail 3: DuckDB Appender API (Fast Bulk Insert)

**DO NOT use prepared statements for bulk data:**
```cpp
// SLOW (don't do this for 170,000 rows):
auto stmt = conn->Prepare("INSERT INTO traces VALUES (?, ?, ...)");
for (auto& entry : entries) {
    stmt->Execute(entry.field1, entry.field2, ...);
}

// FAST (use Appender instead):
duckdb::Appender appender(*conn, "traces");
for (auto& entry : entries) {
    appender.BeginRow();
    appender.Append<int64_t>(entry.field1);
    appender.Append<int32_t>(entry.field2);
    // ...
    appender.EndRow();
}
appender.Flush();
```

**Performance:** Appender is **100x faster** for bulk inserts

### Detail 4: ImGuiListClipper (Virtual Scrolling)

**Correct usage:**
```cpp
ImGuiListClipper clipper;
clipper.Begin(total_items);  // Tell clipper total item count

while (clipper.Step()) {
    // Only loop over VISIBLE items
    for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++) {
        ImGui::Text("Row %d: %s", i, items[i].name.c_str());
    }
}
// clipper.End() called automatically by destructor
```

**Requirements:**
- All items must have **same height**
- Need **random access** to data (can't iterate sequentially only)
- Works best with `std::vector` or similar

### Detail 5: OpenGL Texture for Heatmap

**Why 1D texture instead of vertices:**
- 13 GB file = 13,000 MB = 13,000 "pixels" at 1 MB/pixel
- At 100x zoom = 1,300,000 pixels (easily handled by GPU)
- Drawing 1M vertices = slow, drawing 1 textured quad = fast

**Implementation:**
```cpp
// Create 1D texture (one float per MB of file)
GLuint texture;
glGenTextures(1, &texture);
glBindTexture(GL_TEXTURE_1D, texture);

std::vector<float> heat_data(13000, 0.0f);  // 13,000 MB
glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, 13000, 0,
             GL_RED, GL_FLOAT, heat_data.data());

// Fragment shader samples texture
// Input: UV coordinate 0.0-1.0 (normalized file position)
// Output: Color based on heat value
```

---

## Part 10: Research Queries (SQL Templates)

### Query 1: Expert Selection Stability

**Research question:** Do the same experts get selected across tokens?

```sql
-- Per expert, how many tokens use it?
SELECT
    layer_id,
    UNNEST(expert_ids) as expert_id,
    COUNT(DISTINCT token_id) as token_count,
    100.0 * COUNT(DISTINCT token_id) / (SELECT MAX(token_id) + 1 FROM traces) as percentage
FROM traces
WHERE operation_type = 'MUL_MAT_ID'
GROUP BY layer_id, expert_id
HAVING token_count > 10
ORDER BY layer_id, percentage DESC;
```

**Expected output:**
```
layer_id | expert_id | token_count | percentage
---------|-----------|-------------|------------
0        | 9         | 98          | 98.0%      ← VERY STABLE
0        | 5         | 95          | 95.0%
0        | 24        | 87          | 87.0%
1        | 29        | 92          | 92.0%
```

**Interpretation:** If percentage > 80%, expert is "hot" → cache permanently

### Query 2: File Layout Analysis (Seek Distances)

**Research question:** How scattered is the file layout?

```sql
WITH layer_file_positions AS (
    SELECT
        token_id,
        layer_id,
        MIN(src0_offset) as layer_start_offset
    FROM traces
    WHERE src0_memory = 'DISK' AND layer_id IS NOT NULL
    GROUP BY token_id, layer_id
)
SELECT
    layer_id,
    AVG(ABS(layer_start_offset - LAG(layer_start_offset) OVER (
        PARTITION BY token_id ORDER BY layer_id
    ))) / (1024*1024*1024) as avg_seek_distance_gb
FROM layer_file_positions
GROUP BY layer_id
ORDER BY layer_id;
```

**Expected output:**
```
layer_id | avg_seek_distance_gb
---------|---------------------
0        | 0.00                 ← First layer (no seek)
1        | 2.85                 ← Large seek (scattered layout!)
2        | 5.12
...
```

**Interpretation:** If avg_seek > 1 GB, layout is scattered → re-ordering could help

### Query 3: Hot vs Cold Memory Regions

**Research question:** Which parts of the file are accessed most?

```sql
-- Divide file into 100 MB regions
SELECT
    FLOOR(offset_start / (100 * 1024 * 1024)) as region_100mb,
    COUNT(*) as tensor_count,
    SUM(access_count) as total_accesses,
    100.0 * SUM(access_count) / (SELECT SUM(access_count) FROM memory_hotness) as percentage
FROM memory_hotness
WHERE access_count > 0
GROUP BY region_100mb
ORDER BY total_accesses DESC
LIMIT 20;
```

**Expected output:**
```
region_100mb | tensor_count | total_accesses | percentage
-------------|--------------|----------------|------------
0-100 MB     | 45           | 12,450         | 42.3%      ← Embeddings (hot!)
500-600 MB   | 32           | 8,230          | 27.9%
...
```

**Interpretation:** If first 20% of file = 80% of accesses → layout matters!

---

## Part 11: Verification & Testing

### Verification Step 1: Data Integrity

**After loading trace file:**
```cpp
// Verify entry count matches binary file size
size_t expected_entries = file_size / 1024;
assert(loader.GetEntryCount() == expected_entries);

// Verify first entry has non-zero timestamp
assert(entries[0].timestamp_ns > 0);

// Verify token IDs are sequential (0, 1, 2, ...)
assert(entries.back().token_id == max_token_id);
```

### Verification Step 2: DuckDB Import

**After importing to database:**
```sql
-- Count matches
SELECT COUNT(*) FROM traces;
-- Expected: Same as loader.GetEntryCount()

-- Check for NULLs in required fields
SELECT COUNT(*) FROM traces WHERE timestamp_ns IS NULL;
-- Expected: 0

-- Verify expert IDs are valid (0-31)
SELECT MAX(UNNEST(expert_ids)) FROM traces WHERE expert_ids IS NOT NULL;
-- Expected: 31 (not > 31)
```

### Verification Step 3: Performance

**Frame rate test:**
```cpp
// In main loop
static int frame_count = 0;
static double last_time = glfwGetTime();

frame_count++;
double current_time = glfwGetTime();
if (current_time - last_time >= 1.0) {
    double fps = frame_count / (current_time - last_time);
    std::cout << "FPS: " << fps << std::endl;

    // Verify >= 55 FPS (target: 60, allow 10% slack)
    assert(fps >= 55.0);

    frame_count = 0;
    last_time = current_time;
}
```

### Verification Step 4: Memory Usage

**Test with 100 tokens:**
```bash
# Run application
./tensor-trace-analyzer

# In another terminal, monitor memory
ps aux | grep tensor-trace-analyzer

# Expected: <2 GB memory usage
```

### Verification Step 5: Query Correctness

**Cross-validate with Python:**
```python
# In Python, run same query
import duckdb
con = duckdb.connect(':memory:')
# ... load same data ...
result = con.execute("SELECT expert_id, COUNT(*) FROM traces GROUP BY expert_id").fetchall()

# Compare with C++ application result
# Should match exactly
```

---

## Part 12: Risk Mitigation

### Risk 1: DuckDB C++ API Instability

**Problem:** DuckDB's C++ API is marked as "not stable"

**Mitigation:**
- Use **C API** instead for production (if needed)
- Pin to specific DuckDB version (v0.10.0)
- Document API version in README
- Test on both macOS and Linux

**Fallback:** If C++ API breaks, switch to C API (more verbose but stable)

### Risk 2: Large File Memory Consumption

**Problem:** Loading 1,000 tokens (1.7M entries) might exhaust memory

**Mitigation:**
- Use **memory-mapped file** for binary trace (mmap)
- Stream parsing (don't load entire vector)
- DuckDB handles data on disk (not all in RAM)
- Lazy loading: only load visible token range

**Limit:** Cap at 1,000 tokens per session (1.6 GB trace file)

### Risk 3: OpenGL Compatibility

**Problem:** macOS deprecated OpenGL, Linux may have driver issues

**Mitigation:**
- Use OpenGL 3.3 (widely supported, not latest 4.6)
- Fallback to CPU rendering if GPU fails
- Consider Vulkan/Metal for future (via imgui backends)

**Testing:** Test on both macOS (development) and Linux (server)

### Risk 4: Build Complexity

**Problem:** Multiple dependencies, cross-platform builds

**Mitigation:**
- Use CMake FetchContent for dependencies (auto-download)
- Provide install scripts (install_deps.sh)
- Document build process thoroughly
- Test on clean systems

---

## Part 13: Future Enhancements (Post-Thesis)

### Enhancement 1: Real-Time Trace Monitoring

**Feature:** Stream trace data while llama.cpp is running

**Implementation:**
- Watch /tmp/tensor_trace.bin for changes (file system events)
- Parse new entries incrementally
- Live update heatmap and table
- Useful for debugging inference in real-time

### Enhancement 2: Comparison Mode

**Feature:** Load two trace files, compare before/after optimization

**UI:**
```
┌────────────────────────────────────────┐
│ Baseline       vs       Optimized      │
├───────────────────┬────────────────────┤
│ Heatmap (before)  │ Heatmap (after)    │
│ Seek: 2.5 GB avg  │ Seek: 0.3 GB avg   │
├───────────────────┴────────────────────┤
│ Improvement: 8.3x less seeking         │
└────────────────────────────────────────┘
```

### Enhancement 3: Blktrace Integration (Thread 1)

**Feature:** Correlate tensor accesses with disk I/O (blktrace sectors)

**Data flow:**
```
tensor_trace.bin   blktrace.bin
     ↓                  ↓
  DuckDB            DuckDB
     ↓                  ↓
     └──── JOIN ────────┘
           (by timestamp)
              ↓
   Which tensor caused which disk I/O?
```

**Query:**
```sql
SELECT
    t.dst_name,
    t.src0_offset as tensor_file_offset,
    b.sector * 512 as disk_sector_byte,
    ABS(t.src0_offset - b.sector * 512) as offset_diff
FROM traces t
JOIN blktrace b ON (
    ABS(t.timestamp_ns - b.timestamp_ns) < 1000000  -- Within 1ms
)
WHERE t.src0_memory = 'DISK'
ORDER BY t.timestamp_ns;
```

### Enhancement 4: Python Integration

**Feature:** Export analysis to Python for matplotlib/seaborn

**Implementation:**
- Export query results to CSV
- Generate Python plotting script
- User runs: `python generate_figures.py`
- Thesis-quality figures in `figures/`

---

## Part 14: Success Criteria

### Prototype Success (End of Week 1):

- [ ] Application window opens at 60 FPS
- [ ] Loads 100-token trace file (<5 seconds)
- [ ] Displays entries in scrollable table
- [ ] Shows basic heatmap (file layout)
- [ ] DuckDB queries working (COUNT, WHERE)
- [ ] Memory usage <2 GB

### Full Version Success (End of Week 2):

- [ ] SQL query panel with templates
- [ ] Timeline filtering (token range, layer)
- [ ] Click correlation (trace ↔ heatmap)
- [ ] Expert hotness visualization
- [ ] Export PNG/CSV
- [ ] Performance: 60 FPS with 170K entries

### Thesis Readiness (End of Week 3):

- [ ] Can answer: "Which experts are hot?"
- [ ] Can answer: "Is access sequential or random?"
- [ ] Can answer: "Which file regions are hot?"
- [ ] Generates all thesis figures (publication quality)
- [ ] Documented workflow for reproducibility

---

## Part 15: Comparison with WebUI

### When to Use WebUI (React):

✅ **Quick validation** (1-2 tokens)
✅ **Demo to supervisor** (visual appeal)
✅ **Web-based sharing** (send URL, no install)
✅ **Rapid prototyping** (iterate UI quickly)

### When to Use desktopui (C++):

✅ **Large-scale analysis** (100+ tokens)
✅ **SQL queries** (analytical research)
✅ **Thesis figure generation** (high-res export)
✅ **Performance** (millions of data points)
✅ **Offline work** (no npm run dev needed)

### Keep Both:

- WebUI: Validation tool (proves instrumentation works)
- desktopui: Analysis tool (answers research questions)

---

## Part 16: Next Steps After Plan Approval

### Immediate Actions:

1. Create BSC/desktopui/ directory
2. Download dependencies (ImGui, DuckDB, GLFW)
3. Create CMakeLists.txt
4. Build minimal window (Day 1 deliverable)
5. Verify 60 FPS rendering

### First Milestone (Day 5):

- Working prototype with virtual scrolling
- Can load 100-token trace
- Can query with DuckDB
- Can render basic heatmap

### Documentation:

- README.md (build instructions)
- ARCHITECTURE.md (design overview)
- QUERIES.md (SQL template library)

---

## Part 17: Open Questions

### Before Implementation:

1. **DuckDB C++ vs C API?**
   - C++ API: More convenient, less stable
   - C API: More verbose, guaranteed stable
   - **Recommendation:** Start with C++, switch to C if stability issues

2. **OpenGL vs Vulkan/Metal?**
   - OpenGL 3.3: Widely supported, deprecated on macOS
   - Metal: macOS native, best performance
   - Vulkan: Cross-platform, complex
   - **Recommendation:** OpenGL 3.3 for now (works everywhere)

3. **Memory-mapped files vs full load?**
   - mmap: Better for huge files (10 GB+)
   - Full load: Simpler code, faster random access
   - **Recommendation:** Full load for traces (<2 GB), mmap for future

4. **Persistent database vs in-memory?**
   - In-memory: Faster, no disk I/O
   - Persistent: Can save analysis state
   - **Recommendation:** In-memory for now, add persistence later

---

## Summary

**Architecture:** ImGui (UI) + DuckDB (queries) + OpenGL (heatmap) + C++17
**Timeline:** 12 days (3 days for prototype, 12 days for full version)
**Platform:** macOS primary, Linux secondary
**Dependencies:** All open-source with permissive licenses
**Performance:** 60 FPS, handles millions of entries
**Deliverable:** Production-ready analysis tool for thesis research

**Status:** Ready to implement pending approval
**Risk:** Low (proven technologies, incremental development)
**Confidence:** High (well-researched plan, clear milestones)

---

**End of Plan**
