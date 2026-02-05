#include "JSONLoader.h"
#include "json.hpp"
#include <fstream>
#include <iostream>

using json = nlohmann::json;

// Initialize static member
std::string JSONLoader::last_error_ = "";

bool JSONLoader::loadMemoryMap(const std::string& filepath, MemoryMap& out_map) {
    try {
        // Open file
        std::ifstream file(filepath);
        if (!file.is_open()) {
            last_error_ = "Failed to open file: " + filepath;
            return false;
        }

        // Parse JSON
        json j;
        file >> j;

        // Clear output structure
        out_map = MemoryMap();

        // Parse root fields
        out_map.model_name = j["model_name"].get<std::string>();
        out_map.total_size_bytes = j["total_size_bytes"].get<uint64_t>();

        // Parse metadata
        auto& meta_json = j["metadata"];
        out_map.metadata.n_layers = meta_json["n_layers"].get<int>();
        out_map.metadata.n_vocab = meta_json["n_vocab"].get<int>();
        out_map.metadata.n_embd = meta_json["n_embd"].get<int>();
        out_map.metadata.n_tensors = meta_json["n_tensors"].get<int>();

        // Parse tensors array
        auto& tensors_json = j["tensors"];
        out_map.tensors.reserve(tensors_json.size());

        for (const auto& tensor_json : tensors_json) {
            MemoryTensor tensor;
            tensor.name = tensor_json["name"].get<std::string>();
            tensor.offset_start = tensor_json["offset_start"].get<uint64_t>();
            tensor.offset_end = tensor_json["offset_end"].get<uint64_t>();
            tensor.size_bytes = tensor_json["size_bytes"].get<uint64_t>();

            // Parse shape array
            auto& shape_json = tensor_json["shape"];
            for (const auto& dim : shape_json) {
                tensor.shape.push_back(dim.get<uint64_t>());
            }

            tensor.category = tensor_json["category"].get<std::string>();

            // layer_id can be null
            if (tensor_json["layer_id"].is_null()) {
                tensor.layer_id = -1;
            } else {
                tensor.layer_id = tensor_json["layer_id"].get<int>();
            }

            tensor.component = tensor_json["component"].get<std::string>();
            tensor.component_type = tensor_json["component_type"].get<std::string>();

            // expert_id can be null
            if (tensor_json.contains("expert_id") && !tensor_json["expert_id"].is_null()) {
                tensor.expert_id = tensor_json["expert_id"].get<int>();
            } else {
                tensor.expert_id = -1;
            }

            out_map.tensors.push_back(tensor);
        }

        std::cout << "✓ Loaded memory map: " << out_map.model_name << std::endl;
        std::cout << "  Tensors: " << out_map.tensors.size() << std::endl;
        std::cout << "  Total size: " << out_map.getTotalSizeGB() << " GB" << std::endl;

        return true;

    } catch (const json::exception& e) {
        last_error_ = std::string("JSON parsing error: ") + e.what();
        std::cerr << "✗ " << last_error_ << std::endl;
        return false;
    } catch (const std::exception& e) {
        last_error_ = std::string("Error loading memory map: ") + e.what();
        std::cerr << "✗ " << last_error_ << std::endl;
        return false;
    }
}

bool JSONLoader::loadTraceData(const std::string& filepath, TraceData& out_data) {
    try {
        // Open file
        std::ifstream file(filepath);
        if (!file.is_open()) {
            last_error_ = "Failed to open file: " + filepath;
            return false;
        }

        // Parse JSON
        json j;
        file >> j;

        // Clear output structure
        out_data = TraceData();

        // Parse metadata
        auto& meta_json = j["metadata"];
        out_data.metadata.total_entries = meta_json["total_entries"].get<uint32_t>();
        out_data.metadata.duration_ms = meta_json["duration_ms"].get<double>();
        out_data.metadata.timestamp_start_ns = meta_json["timestamp_start_ns"].get<uint64_t>();
        out_data.metadata.format_version = meta_json["format_version"].get<std::string>();

        // Parse entries array
        auto& entries_json = j["entries"];
        out_data.entries.reserve(entries_json.size());

        for (const auto& entry_json : entries_json) {
            TraceEntry entry;
            entry.entry_id = entry_json["entry_id"].get<uint32_t>();
            entry.timestamp_ns = entry_json["timestamp_ns"].get<uint64_t>();
            entry.timestamp_relative_ms = entry_json["timestamp_relative_ms"].get<double>();
            entry.token_id = entry_json["token_id"].get<uint32_t>();

            // layer_id can be null
            if (entry_json["layer_id"].is_null()) {
                entry.layer_id = -1;
            } else {
                entry.layer_id = entry_json["layer_id"].get<int>();
            }

            entry.thread_id = entry_json["thread_id"].get<uint16_t>();
            entry.phase = entry_json["phase"].get<std::string>();
            entry.operation_type = entry_json["operation_type"].get<std::string>();
            entry.dst_name = entry_json["dst_name"].get<std::string>();
            entry.num_sources = entry_json["num_sources"].get<uint8_t>();

            // Parse sources array
            auto& sources_json = entry_json["sources"];
            for (const auto& source_json : sources_json) {
                TraceSource source;
                source.name = source_json["name"].get<std::string>();
                source.tensor_ptr = source_json["tensor_ptr"].get<std::string>();
                source.size_bytes = source_json["size_bytes"].get<uint64_t>();

                // layer_id can be null
                if (source_json["layer_id"].is_null()) {
                    source.layer_id = -1;
                } else {
                    source.layer_id = source_json["layer_id"].get<int>();
                }

                source.memory_source = source_json["memory_source"].get<std::string>();

                // disk_offset or buffer_id depending on memory_source
                if (source.memory_source == "DISK" && source_json.contains("disk_offset")) {
                    source.disk_offset = source_json["disk_offset"].get<uint64_t>();
                    source.buffer_id = 0;
                } else if (source.memory_source == "BUFFER" && source_json.contains("buffer_id")) {
                    source.buffer_id = source_json["buffer_id"].get<uint64_t>();
                    source.disk_offset = 0;
                } else {
                    source.disk_offset = 0;
                    source.buffer_id = 0;
                }

                entry.sources.push_back(source);
            }

            // Parse expert_ids array
            auto& expert_ids_json = entry_json["expert_ids"];
            for (const auto& expert_id : expert_ids_json) {
                entry.expert_ids.push_back(expert_id.get<int32_t>());
            }
            entry.num_experts = entry_json["num_experts"].get<uint8_t>();

            out_data.entries.push_back(entry);
        }

        std::cout << "✓ Loaded trace data: " << out_data.entries.size() << " entries" << std::endl;
        std::cout << "  Duration: " << out_data.metadata.duration_ms << " ms" << std::endl;
        std::cout << "  Format: " << out_data.metadata.format_version << std::endl;

        return true;

    } catch (const json::exception& e) {
        last_error_ = std::string("JSON parsing error: ") + e.what();
        std::cerr << "✗ " << last_error_ << std::endl;
        return false;
    } catch (const std::exception& e) {
        last_error_ = std::string("Error loading trace data: ") + e.what();
        std::cerr << "✗ " << last_error_ << std::endl;
        return false;
    }
}
