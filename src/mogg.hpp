// Copyright (c) 2026 Kenji8965
// SPDX-License-Identifier: MIT
// Programming and implementation assistance: ChatGPT 5.6 Sol (OpenAI)
// Human testing, validation, and release acceptance: Kenji8965
// Provided "AS IS"; use at your own risk. See LICENSE.

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <variant>
#include <vector>

namespace moge {

enum class MoggType : uint16_t {
    F32 = 1, F16 = 2, BF16 = 3, I32 = 4,
    Q4_K = 10, Q6_K = 11, Q8_0 = 12, IQ4_XS = 13,
};

enum TensorFlags : uint8_t {
    TF_SENSITIVE      = 1u << 0,
    TF_LINEARIZED_1X1 = 1u << 1,
    TF_PHASED_CONVT   = 1u << 2,
    TF_SPARSE         = 1u << 3,
    TF_SPARSE_PACKED  = 1u << 4,
    TF_PIXELSHUFFLE_PHASED = 1u << 5,
    TF_HADAMARD_ROTATED = 1u << 6,
};

using MetaValue = std::variant<int64_t, double, std::string,
                               std::vector<int64_t>, std::vector<std::string>, bool>;

struct TensorInfo {
    std::string name;
    MoggType type{};
    uint8_t ndim = 0;
    uint8_t flags = 0;
    int64_t ne[4] = {1,1,1,1};
    uint64_t offset = 0;
    uint64_t size = 0;
};

class MoggFile {
public:
    explicit MoggFile(const std::filesystem::path & path);
    // Parse a MOGG image supplied by the caller. When copy=true the bytes are
    // copied into storage owned by MoggFile; when false the caller must keep
    // the buffer alive and unchanged for the full Model lifetime.
    MoggFile(const void * data, size_t size, bool copy);
    ~MoggFile();
    MoggFile(const MoggFile &) = delete;
    MoggFile & operator=(const MoggFile &) = delete;

    const uint8_t * data() const { return data_; }
    uint8_t * mutable_data() const { return data_; }
    size_t size() const { return size_; }
    bool is_mapped() const { return mapped_; }

    const std::unordered_map<std::string, MetaValue> & metadata() const { return meta_; }
    const std::vector<TensorInfo> & tensors() const { return tensors_; }
    const TensorInfo * find_tensor(const std::string & name) const;
    const uint8_t * tensor_data(const TensorInfo & t) const;

    template<class T> T meta(const std::string & key) const {
        auto it = meta_.find(key);
        if (it == meta_.end()) throw std::runtime_error("missing metadata key: " + key);
        return std::get<T>(it->second);
    }
    template<class T> T meta_or(const std::string & key, T fallback) const {
        auto it = meta_.find(key);
        return it == meta_.end() ? fallback : std::get<T>(it->second);
    }

private:
    // Runtime loading uses a private writable mmap on POSIX so Metal can wrap
    // the model file with newBufferWithBytesNoCopy. The mapping is never
    // intentionally modified; MAP_PRIVATE simply satisfies backends that need
    // a non-const host pointer. Non-POSIX builds retain the vector fallback.
    uint8_t * data_ = nullptr;
    size_t size_ = 0;
    bool mapped_ = false;
    int fd_ = -1;
    std::vector<uint8_t> bytes_;
    std::unordered_map<std::string, MetaValue> meta_;
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, size_t> tensor_index_;

    void parse();
};

} // namespace moge
