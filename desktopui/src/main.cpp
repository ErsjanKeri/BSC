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

int main(int argc, char** argv) {
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

    // Create window
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Tensor Trace Analyzer", nullptr, nullptr);
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

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Initialize ImPlot
    ImPlot::CreateContext();

    std::cout << "Tensor Trace Analyzer - Desktop UI" << std::endl;
    std::cout << "Press ESC or close window to exit" << std::endl;
    std::cout << std::endl;

    // Load data files
    std::cout << "Loading data files..." << std::endl;

    MemoryMap memoryMap;
    TraceData traceData;
    bool memoryMapLoaded = false;
    bool traceDataLoaded = false;

    // Load memory map
    if (JSONLoader::loadMemoryMap("../data/memory-map.json", memoryMap)) {
        memoryMapLoaded = true;
    } else {
        std::cerr << "Failed to load memory map: " << JSONLoader::getLastError() << std::endl;
    }

    // Load trace data
    if (JSONLoader::loadTraceData("../data/traces/token-00000.json", traceData)) {
        traceDataLoaded = true;
    } else {
        std::cerr << "Failed to load trace data: " << JSONLoader::getLastError() << std::endl;
    }

    std::cout << std::endl;

    // Create trace table view
    TraceTableView traceTableView;
    if (traceDataLoaded) {
        traceTableView.setTraceData(&traceData);
    }

    // Create heatmap view
    HeatmapView heatmapView;
    if (memoryMapLoaded) {
        heatmapView.setMemoryMap(&memoryMap);
    }
    if (traceDataLoaded) {
        heatmapView.setTraceData(&traceData);
    }

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Poll events
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Create fullscreen dockspace
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::SetNextWindowViewport(viewport->ID);

        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
        window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        window_flags |= ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
        window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

        ImGui::Begin("DockSpace", nullptr, window_flags);
        ImGui::PopStyleVar(3);

        // Create dockspace
        ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);

        // Menu bar
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Exit", "Ctrl+Q")) {
                    glfwSetWindowShouldClose(window, true);
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Heatmap", nullptr, true);
                ImGui::MenuItem("Trace Table", nullptr, true);
                ImGui::EndMenu();
            }

            // Status info in menu bar (right-aligned)
            float menu_bar_width = ImGui::GetWindowWidth();
            float fps_text_width = 150.0f;
            ImGui::SetCursorPosX(menu_bar_width - fps_text_width);
            ImGui::Text("FPS: %.1f", io.Framerate);

            ImGui::EndMenuBar();
        }

        ImGui::End();

        // Render heatmap view (will dock automatically)
        if (memoryMapLoaded) {
            heatmapView.render();
        }

        // Render trace table view (will dock automatically)
        if (traceDataLoaded) {
            traceTableView.render();
        }

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
