// Copyright (c) 2026 Kenji8965
// SPDX-License-Identifier: MIT
// Programming and implementation assistance: ChatGPT 5.6 Sol (OpenAI)
// Human testing, validation, and release acceptance: Kenji8965
// Provided "AS IS"; use at your own risk. See LICENSE.

#pragma once

#include <ggml.h>
#include <ggml-backend.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace moge {

// Opaque macOS Metal helper used only by the opt-in sparse hybrid path.
// Non-Apple builds provide a stub implementation that reports unavailable.
class SparseMetalExecutor {
public:
    static std::unique_ptr<SparseMetalExecutor> create(ggml_backend_t backend);
    ~SparseMetalExecutor();

    SparseMetalExecutor(const SparseMetalExecutor &) = delete;
    SparseMetalExecutor & operator=(const SparseMetalExecutor &) = delete;

    bool available() const;
    const char * backend_policy_for_channels(int channels) const;

    // norm_silu_f16 is [C,N] and already contains LN(x) -> SiLU in FP16.
    // conv1_f16 is reusable [C,N] scratch. state_f32 is both the residual
    // input and final output. The released MoGe-3 sparse blocks have C==Cout
    // and an identity residual; unsupported shapes/types throw so the caller
    // never silently changes semantics.
    double resblock_inplace(
        int level,
        ggml_tensor * norm_silu_f16,
        ggml_tensor * conv1_f16,
        ggml_tensor * state_f32,
        const std::vector<int32_t> & neighbors,
        ggml_tensor * conv1_weight,
        ggml_tensor * conv1_bias,
        ggml_tensor * conv2_weight,
        ggml_tensor * conv2_bias);

private:
    struct Impl;
    explicit SparseMetalExecutor(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace moge
