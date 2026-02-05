#pragma once

#include "MemoryMap.h"
#include "TraceData.h"
#include <string>

// JSON loader utility class
class JSONLoader {
public:
    // Load memory map from JSON file
    // Returns true on success, false on failure
    static bool loadMemoryMap(const std::string& filepath, MemoryMap& out_map);

    // Load trace data from JSON file
    // Returns true on success, false on failure
    static bool loadTraceData(const std::string& filepath, TraceData& out_data);

    // Get last error message
    static const std::string& getLastError() { return last_error_; }

private:
    static std::string last_error_;
};
