#pragma once

#include "TraceData.h"
#include "imgui.h"
#include <vector>
#include <string>

// View for displaying trace entries in a scrollable table
class TraceTableView {
public:
    TraceTableView();

    // Set the trace data to display
    void setTraceData(const TraceData* data);

    // Render the table view
    void render();

    // Filtering options
    void setLayerFilter(int layer_id);      // -2 = all layers, -1 = non-layer, 0-N = specific layer
    void setOperationFilter(const std::string& op_type);  // "" = all operations
    void setMemorySourceFilter(const std::string& source); // "" = all, "DISK", "BUFFER"

    // Clear all filters
    void clearFilters();

    // Get statistics
    size_t getVisibleEntryCount() const { return filtered_entries_.size(); }
    size_t getTotalEntryCount() const { return trace_data_ ? trace_data_->entries.size() : 0; }

private:
    const TraceData* trace_data_;

    // Filtering state
    int layer_filter_;                  // -2 = all
    std::string operation_filter_;      // "" = all
    std::string memory_source_filter_;  // "" = all

    // Filtered entries (pointers to entries in trace_data_)
    std::vector<const TraceEntry*> filtered_entries_;

    // UI state
    int selected_entry_index_;

    // Helper methods
    void applyFilters();
    void renderFilterControls();
    void renderTable();
    void renderEntryDetails(const TraceEntry* entry);

    // Helper to format sizes
    static std::string formatSize(uint64_t bytes);
};
