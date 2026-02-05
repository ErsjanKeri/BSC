#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include "JSONLoader.h"
#include "MemoryMap.h"
#include "TraceData.h"
#include "TraceTableView.h"
#include "HeatmapView.h"

// Helper: Render accumulated access graph
void renderAccumulatedGraph(const MemoryMap& memoryMap,
                           const std::map<std::string, uint32_t>& accumulatedCounts,
                           uint32_t maxCount) {
    ImGui::Separator();
    ImGui::Text("Accumulated Access Pattern (All 100 Tokens)");

    // Static hover state for tooltips
    static const MemoryTensor* hovered_tensor = nullptr;
    hovered_tensor = nullptr;  // Reset each frame

    if (ImPlot::BeginPlot("##accumulated_graph", ImVec2(-1, 450))) {
        ImPlot::SetupAxis(ImAxis_X1, "File Offset (GB)", ImPlotAxisFlags_None);
        ImPlot::SetupAxis(ImAxis_Y1, "Total Accesses", ImPlotAxisFlags_None);

        double max_gb = memoryMap.total_size_bytes / (1024.0 * 1024.0 * 1024.0);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, max_gb, ImGuiCond_Once);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, static_cast<double>(maxCount), ImGuiCond_Once);

        // Build step function
        std::vector<double> step_x, step_y;
        for (const auto& tensor : memoryMap.tensors) {
            uint32_t count = 0;
            auto it = accumulatedCounts.find(tensor.name);
            if (it != accumulatedCounts.end()) {
                count = it->second;
            }

            double start_gb = tensor.offset_start / (1024.0 * 1024.0 * 1024.0);
            double end_gb = tensor.offset_end / (1024.0 * 1024.0 * 1024.0);

            step_x.push_back(start_gb);
            step_y.push_back(static_cast<double>(count));
            step_x.push_back(end_gb);
            step_y.push_back(static_cast<double>(count));
        }

        if (!step_x.empty()) {
            // Blue color (same for fill and line)
            ImVec4 blue_color = ImVec4(0.2f, 0.5f, 0.8f, 1.0f);

            // Draw filled area
            ImPlot::PushStyleColor(ImPlotCol_Fill, blue_color);
            ImPlot::PlotShaded("##accumulated_fill", step_x.data(), step_y.data(), step_x.size(), 0.0);
            ImPlot::PopStyleColor();

            // Draw line (same color)
            ImPlot::PushStyleColor(ImPlotCol_Line, blue_color);
            ImPlot::PlotLine("##accumulated_line", step_x.data(), step_y.data(), step_x.size());
            ImPlot::PopStyleColor();
        }

        // Hover detection for tooltips
        if (ImPlot::IsPlotHovered()) {
            ImPlotPoint mouse_pos = ImPlot::GetPlotMousePos();
            double mouse_gb = mouse_pos.x;

            // Find which tensor mouse is over
            for (const auto& tensor : memoryMap.tensors) {
                double start_gb = tensor.offset_start / (1024.0 * 1024.0 * 1024.0);
                double end_gb = tensor.offset_end / (1024.0 * 1024.0 * 1024.0);
                if (mouse_gb >= start_gb && mouse_gb <= end_gb) {
                    hovered_tensor = &tensor;
                    break;
                }
            }
        }

        ImPlot::EndPlot();
    }

    // Show tooltip if hovering
    if (hovered_tensor) {
        ImGui::BeginTooltip();

        ImGui::Text("Tensor: %s", hovered_tensor->name.c_str());
        ImGui::Separator();

        if (hovered_tensor->layer_id >= 0) {
            ImGui::Text("Layer: %d", hovered_tensor->layer_id);
        } else {
            ImGui::Text("Layer: -");
        }

        if (hovered_tensor->expert_id >= 0) {
            ImGui::Text("Expert ID: %d", hovered_tensor->expert_id);
        }

        ImGui::Text("Category: %s", hovered_tensor->category.c_str());
        ImGui::Text("Component: %s", hovered_tensor->component_type.c_str());

        ImGui::Separator();

        // Size and position
        ImGui::Text("Size: %.2f MB", hovered_tensor->size_bytes / (1024.0 * 1024.0));
        ImGui::Text("Offset: %.2f - %.2f GB",
                    hovered_tensor->offset_start / (1024.0 * 1024.0 * 1024.0),
                    hovered_tensor->offset_end / (1024.0 * 1024.0 * 1024.0));

        // Accumulated access count
        auto it = accumulatedCounts.find(hovered_tensor->name);
        if (it != accumulatedCounts.end() && it->second > 0) {
            ImGui::Separator();
            uint32_t count = it->second;
            float intensity = static_cast<float>(count) / static_cast<float>(maxCount);
            ImGui::Text("Total Accesses: %u (%.1f%% of max)", count, intensity * 100.0f);
            ImGui::ProgressBar(intensity, ImVec2(-1, 0), "");
        } else {
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Not accessed across all tokens");
        }

        ImGui::EndTooltip();
    }
}

int main(int argc, char** argv) {
    // Check command-line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <domain-path>" << std::endl;
        std::cerr << "Example: " << argv[0] << " ../expert-analysis-2026-01-26/domain-1-code" << std::endl;
        return 1;
    }

    std::string domainPath = argv[1];
    std::string domainName = domainPath;

    // Extract domain name from path (e.g., "domain-1-code")
    size_t lastSlash = domainPath.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        domainName = domainPath.substr(lastSlash + 1);
    }

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // OpenGL 3.3 + Core Profile (macOS compatible)
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Required on macOS

    // Create window with domain name in title
    std::string windowTitle = "Tensor Trace Analyzer - " + domainName;
    GLFWwindow* window = glfwCreateWindow(1280, 720, windowTitle.c_str(), nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync (60 FPS)

    // Initialize Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable keyboard navigation
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;      // Enable docking

    // Setup ImGui style
    ImGui::StyleColorsDark();

    // Disable docking for fixed layout
    io.ConfigFlags &= ~ImGuiConfigFlags_DockingEnable;

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Initialize ImPlot
    ImPlot::CreateContext();

    std::cout << "Tensor Trace Analyzer - Desktop UI" << std::endl;
    std::cout << "Press ESC or close window to exit" << std::endl;
    std::cout << std::endl;

    // Load data files
    std::cout << "Loading domain data from: " << domainPath << std::endl;

    MemoryMap memoryMap;
    std::vector<TraceData> allTokens;
    bool memoryMapLoaded = false;

    // Load memory map (same for all tokens)
    std::string memoryMapPath = domainPath + "/memory-map.json";
    if (JSONLoader::loadMemoryMap(memoryMapPath, memoryMap)) {
        memoryMapLoaded = true;
        std::cout << "✓ Loaded memory map: " << memoryMap.tensors.size() << " tensors" << std::endl;
    } else {
        std::cerr << "Failed to load memory map: " << JSONLoader::getLastError() << std::endl;
    }

    // Load all 100 token traces
    std::cout << "Loading 100 token traces..." << std::endl;
    allTokens.reserve(100);

    for (int tokenId = 0; tokenId < 100; tokenId++) {
        char tokenPath[512];
        snprintf(tokenPath, sizeof(tokenPath), "%s/traces/token-%05d.json", domainPath.c_str(), tokenId);

        TraceData tokenData;
        if (JSONLoader::loadTraceData(tokenPath, tokenData)) {
            allTokens.push_back(std::move(tokenData));

            if ((tokenId + 1) % 10 == 0) {
                std::cout << "  Loaded " << (tokenId + 1) << "/100 tokens..." << std::endl;
            }
        } else {
            std::cerr << "Warning: Failed to load token " << tokenId << ": " << JSONLoader::getLastError() << std::endl;
        }
    }

    std::cout << "✓ Loaded " << allTokens.size() << " tokens" << std::endl;
    std::cout << std::endl;

    bool dataLoaded = memoryMapLoaded && !allTokens.empty();

    // Token selection state
    int currentTokenId = 0;
    int prevTokenId = -1;

    // Create views
    TraceTableView traceTableView;
    HeatmapView heatmapView;

    // Set memory map
    if (memoryMapLoaded) {
        heatmapView.setMemoryMap(&memoryMap);
    }

    // Calculate accumulated access counts across all tokens
    std::map<std::string, uint32_t> accumulatedCounts;
    uint32_t maxAccumulatedCount = 0;

    if (dataLoaded) {
        std::cout << "Calculating accumulated access counts..." << std::endl;
        for (const auto& tensor : memoryMap.tensors) {
            accumulatedCounts[tensor.name] = 0;
        }

        for (const auto& tokenData : allTokens) {
            for (const auto& entry : tokenData.entries) {
                for (const auto& source : entry.sources) {
                    if (source.memory_source != "DISK") continue;

                    bool is_expert = source.name.find("_exps.weight") != std::string::npos ||
                                    source.name.find("_exps.bias") != std::string::npos;

                    if (is_expert && !entry.expert_ids.empty()) {
                        size_t top_k = std::min(size_t(4), entry.expert_ids.size());
                        for (size_t i = 0; i < top_k; i++) {
                            std::string expertName = source.name + "[" + std::to_string(entry.expert_ids[i]) + "]";
                            accumulatedCounts[expertName]++;
                        }
                    } else {
                        accumulatedCounts[source.name]++;
                    }
                }
            }
        }

        for (const auto& pair : accumulatedCounts) {
            if (pair.second > maxAccumulatedCount) {
                maxAccumulatedCount = pair.second;
            }
        }
        std::cout << "✓ Accumulated counts calculated. Max: " << maxAccumulatedCount << std::endl;
    }

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Poll events
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Token selector bar (fullscreen at top)
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, 60));
        ImGui::Begin("Token Selector", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        ImGui::Text("Token Selector:");
        ImGui::SameLine();

        // Previous button
        if (ImGui::Button("<< Prev") && currentTokenId > 0) {
            currentTokenId--;
        }
        ImGui::SameLine();

        // Token slider
        ImGui::PushItemWidth(400);
        ImGui::SliderInt("##token", &currentTokenId, 0, (int)allTokens.size() - 1);
        ImGui::PopItemWidth();
        ImGui::SameLine();

        // Next button
        if (ImGui::Button("Next >>") && currentTokenId < (int)allTokens.size() - 1) {
            currentTokenId++;
        }
        ImGui::SameLine();

        ImGui::Text("Token %d / %zu", currentTokenId, allTokens.size());
        ImGui::SameLine(io.DisplaySize.x - 150);
        ImGui::Text("FPS: %.1f", io.Framerate);

        ImGui::End();

        // Update views when token changes
        if (dataLoaded && currentTokenId != prevTokenId && currentTokenId < (int)allTokens.size()) {
            heatmapView.setTraceData(&allTokens[currentTokenId]);
            traceTableView.setTraceData(&allTokens[currentTokenId]);
            prevTokenId = currentTokenId;
        }

        // 50/50 Split Layout (Trace Table left | Heatmap right)
        float split_y = 60.0f;  // Below token selector
        float split_width = io.DisplaySize.x * 0.5f;

        // Left: Trace Table (50%)
        ImGui::SetNextWindowPos(ImVec2(0, split_y));
        ImGui::SetNextWindowSize(ImVec2(split_width, io.DisplaySize.y - split_y));
        ImGui::Begin("Trace Table", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        if (dataLoaded) {
            traceTableView.render();
        }

        ImGui::End();

        // Right: Heatmap (50%)
        ImGui::SetNextWindowPos(ImVec2(split_width, split_y));
        ImGui::SetNextWindowSize(ImVec2(split_width, io.DisplaySize.y - split_y));
        ImGui::Begin("Heatmap", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        if (dataLoaded) {
            heatmapView.render();

            // Accumulated graph below heatmap
            renderAccumulatedGraph(memoryMap, accumulatedCounts, maxAccumulatedCount);
        }

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
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
