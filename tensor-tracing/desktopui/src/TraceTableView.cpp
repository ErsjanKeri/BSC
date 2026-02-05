#include "TraceTableView.h"
#include <algorithm>
#include <sstream>
#include <iomanip>

TraceTableView::TraceTableView()
    : trace_data_(nullptr)
    , layer_filter_(-2)  // -2 = all layers
    , operation_filter_("")
    , memory_source_filter_("")
    , selected_entry_index_(-1)
{
}

void TraceTableView::setTraceData(const TraceData* data) {
    trace_data_ = data;
    applyFilters();
}

void TraceTableView::setLayerFilter(int layer_id) {
    layer_filter_ = layer_id;
    applyFilters();
}

void TraceTableView::setOperationFilter(const std::string& op_type) {
    operation_filter_ = op_type;
    applyFilters();
}

void TraceTableView::setMemorySourceFilter(const std::string& source) {
    memory_source_filter_ = source;
    applyFilters();
}

void TraceTableView::clearFilters() {
    layer_filter_ = -2;
    operation_filter_ = "";
    memory_source_filter_ = "";
    applyFilters();
}

void TraceTableView::applyFilters() {
    filtered_entries_.clear();

    if (!trace_data_) {
        return;
    }

    for (const auto& entry : trace_data_->entries) {
        // Apply layer filter
        if (layer_filter_ != -2) {
            if (layer_filter_ == -1 && entry.layer_id != -1) continue;
            if (layer_filter_ >= 0 && entry.layer_id != layer_filter_) continue;
        }

        // Apply operation filter
        if (!operation_filter_.empty() && entry.operation_type != operation_filter_) {
            continue;
        }

        // Apply memory source filter
        if (!memory_source_filter_.empty()) {
            bool has_matching_source = false;
            for (const auto& src : entry.sources) {
                if (src.memory_source == memory_source_filter_) {
                    has_matching_source = true;
                    break;
                }
            }
            if (!has_matching_source) continue;
        }

        filtered_entries_.push_back(&entry);
    }
}

void TraceTableView::render() {
    // Render directly into current window (caller provides window context)

    if (!trace_data_) {
        ImGui::Text("No trace data loaded");
        return;
    }

    // Render filter controls
    renderFilterControls();

    ImGui::Separator();
    ImGui::Text("Showing %zu / %zu entries", getVisibleEntryCount(), getTotalEntryCount());
    ImGui::Separator();

    // Render table
    renderTable();
}

void TraceTableView::renderFilterControls() {
    ImGui::Text("Filters:");

    // Layer filter
    ImGui::SameLine();
    if (ImGui::Button("All Layers")) {
        setLayerFilter(-2);
    }
    ImGui::SameLine();
    if (ImGui::Button("Non-Layer")) {
        setLayerFilter(-1);
    }

    // Show quick access to first few layers
    for (int i = 0; i < 5; i++) {
        ImGui::SameLine();
        char label[16];
        snprintf(label, sizeof(label), "L%d", i);
        if (ImGui::Button(label)) {
            setLayerFilter(i);
        }
    }

    // Memory source filter
    ImGui::SameLine();
    if (ImGui::Button("All Mem")) {
        setMemorySourceFilter("");
    }
    ImGui::SameLine();
    if (ImGui::Button("DISK")) {
        setMemorySourceFilter("DISK");
    }
    ImGui::SameLine();
    if (ImGui::Button("BUFFER")) {
        setMemorySourceFilter("BUFFER");
    }

    // Current filter status
    if (layer_filter_ != -2 || !operation_filter_.empty() || !memory_source_filter_.empty()) {
        ImGui::Text("Active filters:");
        ImGui::SameLine();
        if (layer_filter_ == -1) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "[Non-Layer]");
        } else if (layer_filter_ >= 0) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "[Layer %d]", layer_filter_);
        }
        if (!memory_source_filter_.empty()) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "[%s]", memory_source_filter_.c_str());
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Clear All")) {
            clearFilters();
        }
    }
}

void TraceTableView::renderTable() {
    // Table with virtual scrolling
    ImGuiTableFlags flags = ImGuiTableFlags_ScrollY |
                           ImGuiTableFlags_RowBg |
                           ImGuiTableFlags_BordersOuter |
                           ImGuiTableFlags_BordersV |
                           ImGuiTableFlags_Resizable |
                           ImGuiTableFlags_Reorderable |
                           ImGuiTableFlags_Hideable;

    if (ImGui::BeginTable("trace_table", 9, flags, ImVec2(0.0f, 0.0f))) {
        // Setup columns
        ImGui::TableSetupScrollFreeze(0, 1);  // Freeze header row
        ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 50.0f);
        ImGui::TableSetupColumn("Time (ms)", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Token", ImGuiTableColumnFlags_WidthFixed, 50.0f);
        ImGui::TableSetupColumn("Layer", ImGuiTableColumnFlags_WidthFixed, 50.0f);
        ImGui::TableSetupColumn("Phase", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Operation", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("Destination", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Sources", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableHeadersRow();

        // Virtual scrolling with ImGuiListClipper
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(filtered_entries_.size()));

        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                const TraceEntry* entry = filtered_entries_[row];

                ImGui::TableNextRow();
                ImGui::PushID(row);

                // Column 0: Entry ID
                ImGui::TableNextColumn();
                ImGui::Text("%u", entry->entry_id);

                // Column 1: Timestamp
                ImGui::TableNextColumn();
                ImGui::Text("%.2f", entry->timestamp_relative_ms);

                // Column 2: Token ID
                ImGui::TableNextColumn();
                ImGui::Text("%u", entry->token_id);

                // Column 3: Layer ID
                ImGui::TableNextColumn();
                if (entry->layer_id == -1) {
                    ImGui::Text("-");
                } else {
                    ImGui::Text("%d", entry->layer_id);
                }

                // Column 4: Phase
                ImGui::TableNextColumn();
                ImGui::Text("%s", entry->phase.c_str());

                // Column 5: Operation
                ImGui::TableNextColumn();
                ImGui::Text("%s", entry->operation_type.c_str());

                // Column 6: Destination
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(entry->dst_name.c_str());

                // Column 7: Number of sources
                ImGui::TableNextColumn();
                ImGui::Text("%u src", entry->num_sources);

                // Show memory source indicator
                bool has_disk = false;
                bool has_buffer = false;
                for (const auto& src : entry->sources) {
                    if (src.memory_source == "DISK") has_disk = true;
                    if (src.memory_source == "BUFFER") has_buffer = true;
                }
                ImGui::SameLine();
                if (has_disk && has_buffer) {
                    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "D+B");
                } else if (has_disk) {
                    ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "DSK");
                } else if (has_buffer) {
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "BUF");
                }

                // Column 8: Total input size
                ImGui::TableNextColumn();
                ImGui::Text("%s", formatSize(entry->getTotalInputSize()).c_str());

                // Tooltip on hover
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::Text("Entry ID: %u", entry->entry_id);
                    ImGui::Text("Destination: %s", entry->dst_name.c_str());
                    ImGui::Separator();
                    ImGui::Text("Sources (%u):", entry->num_sources);
                    for (size_t i = 0; i < entry->sources.size(); i++) {
                        const auto& src = entry->sources[i];
                        ImGui::BulletText("[%zu] %s", i, src.name.c_str());
                        ImGui::Indent();
                        ImGui::Text("%s • %s", src.memory_source.c_str(), formatSize(src.size_bytes).c_str());
                        if (src.memory_source == "DISK") {
                            ImGui::Text("Offset: 0x%llx", src.disk_offset);
                        }
                        ImGui::Unindent();
                    }
                    if (entry->num_experts > 0) {
                        ImGui::Separator();
                        ImGui::Text("Experts (%u): ", entry->num_experts);
                        ImGui::SameLine();
                        for (size_t i = 0; i < entry->expert_ids.size(); i++) {
                            ImGui::Text("%d", entry->expert_ids[i]);
                            if (i < entry->expert_ids.size() - 1) {
                                ImGui::SameLine();
                                ImGui::Text(",");
                                ImGui::SameLine();
                            }
                        }
                    }
                    ImGui::EndTooltip();
                }

                ImGui::PopID();
            }
        }

        ImGui::EndTable();
    }
}

void TraceTableView::renderEntryDetails(const TraceEntry* entry) {
    // This could be expanded for a detailed view in a separate panel
    ImGui::Text("Entry ID: %u", entry->entry_id);
    ImGui::Text("Operation: %s", entry->operation_type.c_str());
    ImGui::Text("Destination: %s", entry->dst_name.c_str());
}

std::string TraceTableView::formatSize(uint64_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_index < 3) {
        size /= 1024.0;
        unit_index++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << size << " " << units[unit_index];
    return oss.str();
}
