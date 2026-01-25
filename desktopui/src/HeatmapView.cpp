#include "HeatmapView.h"
#include "implot.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>

HeatmapView::HeatmapView()
    : memory_map_(nullptr)
    , trace_data_(nullptr)
    , zoom_level_(10.0f)  // Default: 10 pixels per MB
    , scroll_offset_(0.0f)
    , canvas_height_(30.0f)
    , current_time_ms_(0.0f)
    , max_time_ms_(0.0f)
    , max_access_count_(0)
    , hovered_tensor_(nullptr)
{
}

void HeatmapView::setMemoryMap(const MemoryMap* map) {
    memory_map_ = map;
    if (memory_map_ && trace_data_) {
        calculateAccessCounts();
    }
}

void HeatmapView::setTraceData(const TraceData* data) {
    trace_data_ = data;
    if (trace_data_) {
        // Set timeline range
        max_time_ms_ = trace_data_->metadata.duration_ms;
        current_time_ms_ = max_time_ms_;  // Start at end (show all accesses)

        // Calculate max from FULL timeline (stays fixed)
        if (memory_map_) {
            calculateMaxAccessCount();
        }
    }
    if (memory_map_ && trace_data_) {
        calculateAccessCounts();
    }
}

void HeatmapView::calculateMaxAccessCount() {
    max_access_count_ = 0;

    if (!trace_data_ || !memory_map_) {
        return;
    }

    // Temporary map to count accesses from FULL timeline
    std::map<std::string, uint32_t> full_counts;

    // Initialize
    for (const auto& tensor : memory_map_->tensors) {
        full_counts[tensor.name] = 0;
    }

    // Count ALL entries (full timeline, no time filtering)
    for (const auto& entry : trace_data_->entries) {
        for (const auto& source : entry.sources) {
            if (source.memory_source != "DISK") {
                continue;
            }

            bool is_expert_tensor = source.name.find("_exps.weight") != std::string::npos ||
                                   source.name.find("_exps.bias") != std::string::npos;

            if (is_expert_tensor && !entry.expert_ids.empty()) {
                size_t top_k = std::min(size_t(4), entry.expert_ids.size());
                for (size_t i = 0; i < top_k; i++) {
                    int expert_id = entry.expert_ids[i];
                    std::string expert_tensor_name = source.name + "[" + std::to_string(expert_id) + "]";
                    full_counts[expert_tensor_name]++;
                }
            } else {
                full_counts[source.name]++;
            }
        }
    }

    // Find the maximum
    for (const auto& pair : full_counts) {
        if (pair.second > max_access_count_) {
            max_access_count_ = pair.second;
        }
    }
}

void HeatmapView::calculateAccessCounts() {
    access_counts_.clear();
    // DO NOT reset max_access_count_ here! It's fixed from full timeline

    if (!trace_data_ || !memory_map_) {
        return;
    }

    // Initialize all memory map tensors with 0 count
    for (const auto& tensor : memory_map_->tensors) {
        access_counts_[tensor.name] = 0;
    }

    // Count how many times each tensor is accessed UP TO current timeline position
    for (const auto& entry : trace_data_->entries) {
        // Only count entries that happened before current_time_ms_ (temporal filtering)
        if (entry.timestamp_relative_ms > current_time_ms_) {
            break;  // Entries are sorted by time, so we can stop here
        }

        // Count DISK accesses only (not runtime buffers)
        for (const auto& source : entry.sources) {
            if (source.memory_source != "DISK") {
                continue;  // Skip buffer accesses
            }

            // Check if this is an expert tensor (contains "_exps.weight" or "_exps.bias")
            bool is_expert_tensor = source.name.find("_exps.weight") != std::string::npos ||
                                   source.name.find("_exps.bias") != std::string::npos;

            if (is_expert_tensor && !entry.expert_ids.empty()) {
                // For expert tensors: count access for each expert used
                // Use top-4 experts only (like WebUI)
                size_t top_k = std::min(size_t(4), entry.expert_ids.size());
                for (size_t i = 0; i < top_k; i++) {
                    int expert_id = entry.expert_ids[i];
                    // Build expert tensor name: "blk.0.ffn_down_exps.weight[0]"
                    std::string expert_tensor_name = source.name + "[" + std::to_string(expert_id) + "]";
                    access_counts_[expert_tensor_name]++;
                }
            } else {
                // Normal tensor access (non-expert)
                access_counts_[source.name]++;
            }
        }
    }

    // max_access_count_ stays fixed (from calculateMaxAccessCount)
}

void HeatmapView::render() {
    ImGui::Begin("Memory Access Heatmap", nullptr, ImGuiWindowFlags_None);

    if (!memory_map_) {
        ImGui::Text("No memory map loaded");
        ImGui::End();
        return;
    }

    // Render controls
    renderControls();

    ImGui::Separator();

    // Render statistics
    ImGui::Text("Model: %s", memory_map_->model_name.c_str());
    ImGui::Text("Total size: %.2f GB", memory_map_->getTotalSizeGB());
    ImGui::Text("Tensors: %zu", memory_map_->tensors.size());
    if (trace_data_) {
        ImGui::Text("Max accesses: %u", max_access_count_);
    }

    ImGui::Separator();

    // Render timeline widget
    renderTimelineWidget();

    ImGui::Separator();

    // Render heatmap canvas
    renderHeatmapCanvas();

    ImGui::End();
}

void HeatmapView::renderControls() {
    ImGui::Text("Zoom:");
    ImGui::SameLine();

    // Zoom buttons
    float zoom_levels[] = {1.0f, 5.0f, 10.0f, 20.0f, 50.0f, 100.0f};
    for (int i = 0; i < 6; i++) {
        if (i > 0) ImGui::SameLine();
        char label[16];
        snprintf(label, sizeof(label), "%.0fx", zoom_levels[i]);
        if (ImGui::Button(label)) {
            zoom_level_ = zoom_levels[i];
        }
    }

    ImGui::Text("%.0f pixels/MB", zoom_level_);
}

void HeatmapView::renderTimelineWidget() {
    if (!trace_data_) {
        return;
    }

    ImGui::Text("Timeline:");
    ImGui::SameLine();

    // Timeline slider
    ImGui::PushItemWidth(-100.0f);  // Leave space for the time display
    if (ImGui::SliderFloat("##timeline", &current_time_ms_, 0.0f, max_time_ms_, "%.1f ms")) {
        // Timeline changed - recalculate access counts
        calculateAccessCounts();
    }
    ImGui::PopItemWidth();

    // Show current time / total time
    ImGui::SameLine();
    ImGui::Text("%.1f / %.1f ms", current_time_ms_, max_time_ms_);
}

void HeatmapView::renderHeatmapCanvas() {
    if (!memory_map_) {
        return;
    }

    // Use ImPlot subplots for perfect X-axis alignment
    ImPlotSubplotFlags subplot_flags = ImPlotSubplotFlags_LinkCols | ImPlotSubplotFlags_NoTitle;
    float row_ratios[] = {0.25f, 0.75f};  // 1/4 for strip, 3/4 for graph

    if (ImPlot::BeginSubplots("##heatmap_subplots", 2, 1, ImVec2(-1, -1), subplot_flags, row_ratios, nullptr)) {

        // Top plot: Colored strip
        renderColoredStrip();

        // Bottom plot: Access count graph
        renderAccessGraph();

        ImPlot::EndSubplots();
    }

    // Show tooltip if hovering over a tensor
    if (hovered_tensor_) {
        renderTooltip(hovered_tensor_);
    }
}

void HeatmapView::renderColoredStrip() {
    // Colored strip plot - NO Y-axis, just visual (called from subplot)
    if (ImPlot::BeginPlot("##colored_strip")) {

        // Setup X-axis only (no labels/ticks) and Y-axis (no decorations)
        ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_NoTickLabels);
        ImPlot::SetupAxis(ImAxis_Y1, nullptr, ImPlotAxisFlags_NoDecorations);

        // Set ranges
        double max_gb = memory_map_->total_size_bytes / (1024.0 * 1024.0 * 1024.0);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, max_gb, ImGuiCond_Once);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImGuiCond_Always);

        // Push viridis colormap
        ImPlot::PushColormap(ImPlotColormap_Viridis);

        // Draw colored bars
        for (const auto& tensor : memory_map_->tensors) {
            uint32_t access_count = 0;
            auto it = access_counts_.find(tensor.name);
            if (it != access_counts_.end()) {
                access_count = it->second;
            }

            double start_gb = tensor.offset_start / (1024.0 * 1024.0 * 1024.0);
            double end_gb = tensor.offset_end / (1024.0 * 1024.0 * 1024.0);

            ImVec4 color;
            if (access_count == 0) {
                color = ImVec4(55.0f/255.0f, 65.0f/255.0f, 81.0f/255.0f, 1.0f);
            } else {
                float intensity = static_cast<float>(access_count) / static_cast<float>(max_access_count_);
                color = ImPlot::SampleColormap(intensity);
            }

            ImPlot::PushStyleColor(ImPlotCol_Fill, color);
            double xs[4] = {start_gb, end_gb, end_gb, start_gb};
            double ys[4] = {0.0, 0.0, 1.0, 1.0};
            ImPlot::PlotShaded("##tensor", xs, ys, 4);
            ImPlot::PopStyleColor();
        }

        ImPlot::PopColormap();

        // Hover detection
        if (ImPlot::IsPlotHovered()) {
            ImPlotPoint mouse_pos = ImPlot::GetPlotMousePos();
            double mouse_gb = mouse_pos.x;
            hovered_tensor_ = nullptr;
            for (const auto& tensor : memory_map_->tensors) {
                double start_gb = tensor.offset_start / (1024.0 * 1024.0 * 1024.0);
                double end_gb = tensor.offset_end / (1024.0 * 1024.0 * 1024.0);
                if (mouse_gb >= start_gb && mouse_gb <= end_gb) {
                    hovered_tensor_ = &tensor;
                    break;
                }
            }
        }

        ImPlot::EndPlot();
    }
}

void HeatmapView::renderAccessGraph() {
    // Step function graph with Y-axis (called from subplot)
    if (ImPlot::BeginPlot("##access_graph")) {

        // Setup axes
        ImPlot::SetupAxis(ImAxis_X1, "File Offset (GB)", ImPlotAxisFlags_None);
        ImPlot::SetupAxis(ImAxis_Y1, "Access Count", ImPlotAxisFlags_None);

        // Set ranges
        double max_gb = memory_map_->total_size_bytes / (1024.0 * 1024.0 * 1024.0);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, max_gb, ImGuiCond_Once);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, static_cast<double>(max_access_count_), ImGuiCond_Once);

        // Build step function data (X positions and Y heights)
        std::vector<double> step_x;
        std::vector<double> step_y;

        for (const auto& tensor : memory_map_->tensors) {
            uint32_t access_count = 0;
            auto it = access_counts_.find(tensor.name);
            if (it != access_counts_.end()) {
                access_count = it->second;
            }

            double start_gb = tensor.offset_start / (1024.0 * 1024.0 * 1024.0);
            double end_gb = tensor.offset_end / (1024.0 * 1024.0 * 1024.0);
            double count_d = static_cast<double>(access_count);

            // Add points for step function
            step_x.push_back(start_gb);
            step_y.push_back(count_d);
            step_x.push_back(end_gb);
            step_y.push_back(count_d);
        }

        // Draw step function
        if (!step_x.empty()) {
            ImPlot::PlotLine("##step", step_x.data(), step_y.data(), step_x.size());
        }

        // Hover detection
        if (ImPlot::IsPlotHovered()) {
            ImPlotPoint mouse_pos = ImPlot::GetPlotMousePos();
            double mouse_gb = mouse_pos.x;
            hovered_tensor_ = nullptr;
            for (const auto& tensor : memory_map_->tensors) {
                double start_gb = tensor.offset_start / (1024.0 * 1024.0 * 1024.0);
                double end_gb = tensor.offset_end / (1024.0 * 1024.0 * 1024.0);
                if (mouse_gb >= start_gb && mouse_gb <= end_gb) {
                    hovered_tensor_ = &tensor;
                    break;
                }
            }
        }

        ImPlot::EndPlot();
    }
}

void HeatmapView::renderTooltip(const MemoryTensor* tensor) {
    ImGui::BeginTooltip();

    ImGui::Text("Tensor: %s", tensor->name.c_str());
    ImGui::Separator();

    // Layer and expert info
    if (tensor->layer_id >= 0) {
        ImGui::Text("Layer: %d", tensor->layer_id);
    } else {
        ImGui::Text("Layer: -");
    }

    if (tensor->expert_id >= 0) {
        ImGui::Text("Expert ID: %d", tensor->expert_id);
    }

    ImGui::Text("Category: %s", tensor->category.c_str());
    ImGui::Text("Component: %s", tensor->component_type.c_str());

    ImGui::Separator();

    // Size and position
    ImGui::Text("Size: %s", formatSize(tensor->size_bytes).c_str());
    ImGui::Text("Offset: %s - %s",
                formatOffset(tensor->offset_start).c_str(),
                formatOffset(tensor->offset_end).c_str());

    // Access count with visual indicator
    auto it = access_counts_.find(tensor->name);
    if (it != access_counts_.end() && it->second > 0) {
        ImGui::Separator();
        uint32_t count = it->second;
        float intensity = static_cast<float>(count) / static_cast<float>(max_access_count_);

        // Show access count with color indicator
        ImGui::Text("Accesses: %u (%.1f%% of max)", count, intensity * 100.0f);

        // Visual bar showing relative heat
        ImGui::ProgressBar(intensity, ImVec2(-1, 0), "");
    } else {
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Not accessed in current timeline");
    }

    ImGui::EndTooltip();
}

ImU32 HeatmapView::getHeatColor(uint32_t access_count) const {
    if (access_count == 0) {
        // Gray for unaccessed tensors
        return IM_COL32(55, 65, 81, 255);  // gray-700
    }

    // Calculate intensity (0.0 - 1.0)
    float intensity = 0.0f;
    if (max_access_count_ > 0) {
        intensity = static_cast<float>(access_count) / static_cast<float>(max_access_count_);
    }

    // Dark red -> Bright red gradient
    int r = static_cast<int>(139 + (255 - 139) * intensity);
    return IM_COL32(r, 0, 0, 255);
}

std::string HeatmapView::formatSize(uint64_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_index < 3) {
        size /= 1024.0;
        unit_index++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
    return oss.str();
}

std::string HeatmapView::formatOffset(uint64_t offset) {
    // Format as MB for readability
    double offset_mb = offset / (1024.0 * 1024.0);

    std::ostringstream oss;
    if (offset_mb < 1024.0) {
        oss << std::fixed << std::setprecision(1) << offset_mb << " MB";
    } else {
        double offset_gb = offset_mb / 1024.0;
        oss << std::fixed << std::setprecision(2) << offset_gb << " GB";
    }
    return oss.str();
}
