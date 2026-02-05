#pragma once

#include <string>
#include <vector>
#include <cstdint>

// Represents a source tensor in a trace entry
struct TraceSource {
    std::string name;
    std::string tensor_ptr;      // Address as hex string
    uint64_t size_bytes;
    int layer_id;                // -1 for null
    std::string memory_source;   // "DISK" or "BUFFER"
    uint64_t disk_offset;        // Valid if memory_source == "DISK"
    uint64_t buffer_id;          // Valid if memory_source == "BUFFER"
};

// Represents a single trace entry (tensor operation)
struct TraceEntry {
    uint32_t entry_id;
    uint64_t timestamp_ns;
    double timestamp_relative_ms;
    uint32_t token_id;
    int layer_id;                // -1 for null
    uint16_t thread_id;
    std::string phase;           // "PROMPT" or "GENERATE"
    std::string operation_type;  // "MUL_MAT", "ADD", "GET_ROWS", etc.
    std::string dst_name;
    uint8_t num_sources;
    std::vector<TraceSource> sources;
    std::vector<int32_t> expert_ids;
    uint8_t num_experts;

    // Helper methods
    bool isDiskAccess() const {
        for (const auto& src : sources) {
            if (src.memory_source == "DISK") {
                return true;
            }
        }
        return false;
    }

    uint64_t getTotalInputSize() const {
        uint64_t total = 0;
        for (const auto& src : sources) {
            total += src.size_bytes;
        }
        return total;
    }
};

// Metadata about the trace
struct TraceMetadata {
    uint32_t total_entries;
    double duration_ms;
    uint64_t timestamp_start_ns;
    std::string format_version;
};

// Complete trace data
struct TraceData {
    TraceMetadata metadata;
    std::vector<TraceEntry> entries;

    // Helper methods
    size_t getEntryCount() const { return entries.size(); }

    // Get entries by layer
    std::vector<const TraceEntry*> getEntriesByLayer(int layer_id) const {
        std::vector<const TraceEntry*> result;
        for (const auto& entry : entries) {
            if (entry.layer_id == layer_id) {
                result.push_back(&entry);
            }
        }
        return result;
    }

    // Get disk access entries only
    std::vector<const TraceEntry*> getDiskAccessEntries() const {
        std::vector<const TraceEntry*> result;
        for (const auto& entry : entries) {
            if (entry.isDiskAccess()) {
                result.push_back(&entry);
            }
        }
        return result;
    }

    // Get entries with expert IDs
    std::vector<const TraceEntry*> getExpertEntries() const {
        std::vector<const TraceEntry*> result;
        for (const auto& entry : entries) {
            if (!entry.expert_ids.empty()) {
                result.push_back(&entry);
            }
        }
        return result;
    }
};
