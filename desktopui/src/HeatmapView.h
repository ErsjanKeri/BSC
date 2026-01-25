#pragma once

#include "MemoryMap.h"
#include "TraceData.h"
#include "imgui.h"
#include <vector>
#include <map>
#include <string>

// Heatmap visualization for memory access patterns
class HeatmapView {
public:
    HeatmapView();

    // Set data sources
    void setMemoryMap(const MemoryMap* map);
    void setTraceData(const TraceData* data);

    // Render the heatmap
    void render();

    // Zoom control
    void setZoom(float zoom) { zoom_level_ = zoom; }
    float getZoom() const { return zoom_level_; }

    // Scroll position
    void setScrollOffset(float offset) { scroll_offset_ = offset; }
    float getScrollOffset() const { return scroll_offset_; }

private:
    const MemoryMap* memory_map_;
    const TraceData* trace_data_;

    // Rendering parameters
    float zoom_level_;          // Pixels per MB (default 1.0)
    float scroll_offset_;       // Horizontal scroll offset
    float canvas_height_;       // Height of heatmap bars

    // Timeline state
    float current_time_ms_;     // Current timeline position
    float max_time_ms_;         // Maximum time from trace data

    // Access count cache (temporal - only counts up to current_time_ms_)
    std::map<std::string, uint32_t> access_counts_;
    uint32_t max_access_count_;

    // UI state
    const MemoryTensor* hovered_tensor_;

    // Helper methods
    void calculateMaxAccessCount();  // Calculate max from FULL timeline (call once)
    void calculateAccessCounts();    // Calculate counts up to current_time_ms_ (call on timeline change)
    void renderHeatmapCanvas();
    void renderColoredStrip();       // Top: colored bars only (no Y-axis)
    void renderAccessGraph();        // Bottom: step function with Y-axis
    void renderControls();
    void renderTimelineWidget();
    void renderTooltip(const MemoryTensor* tensor);

    // Color calculation
    ImU32 getHeatColor(uint32_t access_count) const;

    // Formatting helpers
    static std::string formatSize(uint64_t bytes);
    static std::string formatOffset(uint64_t offset);
};
