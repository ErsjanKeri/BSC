cmake_minimum_required(VERSION 3.15)
project(TensorTraceAnalyzer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Print configuration
message(STATUS "Building Tensor Trace Analyzer")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find OpenGL (required for ImGui)
find_package(OpenGL REQUIRED)

# Find GLFW (will be installed via Homebrew)
find_package(glfw3 REQUIRED)

# JSON library (header-only)
set(JSON_DIR ${CMAKE_SOURCE_DIR}/external/json)
if(EXISTS ${JSON_DIR}/json.hpp)
    message(STATUS "Found nlohmann/json at: ${JSON_DIR}")
else()
    message(WARNING "nlohmann/json not found at ${JSON_DIR}")
endif()

# ImGui as static library (will be set up next)
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/external/imgui)

if(EXISTS ${IMGUI_DIR})
    message(STATUS "Found ImGui at: ${IMGUI_DIR}")
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
else()
    message(WARNING "ImGui not found at ${IMGUI_DIR} - will need to download")
endif()

# ImPlot as static library
set(IMPLOT_DIR ${CMAKE_SOURCE_DIR}/external/implot)

if(EXISTS ${IMPLOT_DIR})
    message(STATUS "Found ImPlot at: ${IMPLOT_DIR}")
    add_library(implot STATIC
        ${IMPLOT_DIR}/implot.cpp
        ${IMPLOT_DIR}/implot_items.cpp
    )
    target_include_directories(implot PUBLIC
        ${IMPLOT_DIR}
        ${IMGUI_DIR}
    )
    target_link_libraries(implot PUBLIC imgui)
else()
    message(WARNING "ImPlot not found at ${IMPLOT_DIR}")
endif()

# Main application
set(SOURCES
    src/main.cpp
    src/JSONLoader.cpp
    src/TraceTableView.cpp
    src/HeatmapView.cpp
)

add_executable(tensor-trace-analyzer ${SOURCES})

target_include_directories(tensor-trace-analyzer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${JSON_DIR}
)

if(EXISTS ${IMGUI_DIR})
    target_link_libraries(tensor-trace-analyzer
        imgui
        implot
        glfw
        OpenGL::GL
    )
endif()

# Set output directory
set_target_properties(tensor-trace-analyzer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

message(STATUS "Configuration complete")
