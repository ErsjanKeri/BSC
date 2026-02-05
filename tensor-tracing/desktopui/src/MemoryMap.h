#pragma once

#include <string>
#include <vector>
#include <cstdint>

// Represents a single tensor in the GGUF file
struct MemoryTensor {
    std::string name;
    uint64_t offset_start;
    uint64_t offset_end;
    uint64_t size_bytes;
    std::vector<uint64_t> shape;
    std::string category;       // "embedding", "attention", "ffn", "norm"
    int layer_id;               // -1 for non-layer tensors
    std::string component;      // "query", "key", "value", "gate", "down", "up", etc.
    std::string component_type; // Human-readable description
    int expert_id;              // -1 for non-expert tensors, 0-31 for experts
};

// Metadata about the model
struct MemoryMapMetadata {
    int n_layers;
    int n_vocab;
    int n_embd;
    int n_tensors;
};

// Complete memory map for the GGUF file
struct MemoryMap {
    std::string model_name;
    uint64_t total_size_bytes;
    MemoryMapMetadata metadata;
    std::vector<MemoryTensor> tensors;

    // Helper methods
    size_t getTensorCount() const { return tensors.size(); }
    double getTotalSizeGB() const { return total_size_bytes / (1024.0 * 1024.0 * 1024.0); }

    // Get tensors by category
    std::vector<const MemoryTensor*> getTensorsByCategory(const std::string& category) const {
        std::vector<const MemoryTensor*> result;
        for (const auto& tensor : tensors) {
            if (tensor.category == category) {
                result.push_back(&tensor);
            }
        }
        return result;
    }

    // Get tensors by layer
    std::vector<const MemoryTensor*> getTensorsByLayer(int layer_id) const {
        std::vector<const MemoryTensor*> result;
        for (const auto& tensor : tensors) {
            if (tensor.layer_id == layer_id) {
                result.push_back(&tensor);
            }
        }
        return result;
    }
};
