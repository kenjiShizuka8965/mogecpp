// Copyright (c) 2026 Kenji8965
// SPDX-License-Identifier: MIT
// Programming and implementation assistance: ChatGPT 5.6 Sol (OpenAI)
// Human testing, validation, and release acceptance: Kenji8965
// Provided "AS IS"; use at your own risk. See LICENSE.

#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace moge {

inline constexpr int API_VERSION = 1;
inline constexpr int VERSION_MAJOR = 0;
inline constexpr int VERSION_MINOR = 4;
inline constexpr int VERSION_PATCH = 2;
inline constexpr const char * VERSION_STRING = "0.4.2";

enum class Backend { Auto, CPU, Vulkan, Metal };

struct LoadOptions {
    Backend backend = Backend::Auto;
    int threads = 0;                 // 0 = ggml default / hardware concurrency
    // ggml 0.25.3 requires CPU to be the final scheduler backend. When false
    // on a GPU primary backend, MoGe pins graph inputs to the GPU to avoid the
    // scheduler's default CPU-input placement; unsupported ops may still fall
    // back to the mandatory CPU backend.
    bool cpu_fallback = true;
    bool op_offload = true;
};

struct InferOptions {
    int num_tokens = 0;              // 0 => derive from resolution_level
    int resolution_level = 9;        // MoGe convention; normally 0..9
    int refine_steps = 3;            // MoGe-3 only
    bool force_projection = true;
    bool apply_mask = true;
    float fov_x_degrees = 0.0f;      // <=0 => recover from point map
    // Validation/debug only. When enabled, Result also carries the model's
    // resized/remapped forward outputs before camera focal/shift recovery.
    // Normal inference leaves these vectors empty to avoid an extra copy.
    bool capture_raw_forward = false;
};

struct CalibrationOptions {
    int num_tokens = 0;              // 0 => derive from resolution_level
    int resolution_level = 9;
    bool verify_first = false;       // one-shot host cross-check of Metal reduction
};

struct Result {
    int width = 0;
    int height = 0;
    std::vector<float> points;       // H*W*3, XYZ interleaved
    std::vector<float> depth;        // H*W
    std::vector<float> normal;       // H*W*3, optional
    std::vector<uint8_t> mask;       // H*W, optional
    float intrinsics[9] = {0};       // normalized K
    float metric_scale = 1.0f;

    // Optional validation capture matching upstream MoGeModel.forward():
    // affine_points is H*W*3 after output remap/resize but before focal/shift;
    // mask_probability is H*W after sigmoid; metric scale is exponentiated.
    std::vector<float> raw_affine_points;
    std::vector<float> raw_mask_probability;
    float raw_metric_scale = 1.0f;
};

class Model {
public:
    explicit Model(const std::string & mogg_path, const LoadOptions & options = {});

    // Load a MOGG image from memory. This constructor copies `mogg_size` bytes
    // into Model-owned storage, so the caller may release the input buffer as
    // soon as construction returns. On POSIX/macOS the owned copy is mmap-backed
    // so Metal can retain the mapped/shared-weight fast path.
    Model(const void * mogg_data, size_t mogg_size, const LoadOptions & options = {});

    // Zero-copy model-byte loading for callers that already own stable storage.
    // The buffer must remain alive and unchanged until the returned Model is
    // destroyed. Runtime backends may still copy/transform individual weights.
    static Model from_borrowed_memory(const void * mogg_data, size_t mogg_size,
                                      const LoadOptions & options = {});
    ~Model();
    Model(Model &&) noexcept;
    Model & operator=(Model &&) noexcept;
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;

    // RGB input is H*W*3 interleaved float in [0,1]. Batch size is intentionally
    // fixed to 1 in the optimized path; batching depth maps is rarely useful and
    // hurts the decoder's spatial locality.
    Result infer(const float * rgb, int width, int height, const InferOptions & options = {});

    // Native activation-importance calibration. This runs only the validated
    // dense forward path and accumulates E[x^2] for quantizable GEMM inputs;
    // MoGe postprocessing and the v3 sparse refiner are intentionally skipped.
    void calibration_reset();
    void calibration_add(const float * rgb, int width, int height,
                         const CalibrationOptions & options = {});
    void calibration_write_mogi(const std::string & path) const;
    size_t calibration_record_count() const;
    uint64_t calibration_image_count() const;
    bool calibration_verification_performed() const;

    int version() const;
    Backend backend() const;
    std::string backend_name() const;

private:
    struct BorrowedMemoryTag {};
    Model(BorrowedMemoryTag, const void * mogg_data, size_t mogg_size, const LoadOptions & options);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Library/API version. VERSION_STRING follows semantic versioning for the
// project; API_VERSION is incremented only for incompatible public API/ABI
// changes.
const char * version_string() noexcept;

} // namespace moge
