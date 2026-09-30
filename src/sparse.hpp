// Copyright (c) 2026 Kenji8965
// SPDX-License-Identifier: MIT
// Programming and implementation assistance: ChatGPT 5.6 Sol (OpenAI)
// Human testing, validation, and release acceptance: Kenji8965
// Provided "AS IS"; use at your own risk. See LICENSE.

#pragma once

#include "dense.hpp"

#include <ggml.h>
#include <ggml-backend.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace moge {

struct SparseConfig {
    bool present = false;
    int in_channels = 3;
    int out_channels = 1;
    int encoder_channels = 0;
    std::vector<int> model_channels;
    std::vector<int> encoder_blocks;
    std::vector<int> decoder_blocks;
    int bottleneck_blocks = 1;
    std::vector<int> downsample_factors;
    int encoder_downsample = 16;
    float depth_resolution = 256.0f;
};

SparseConfig read_sparse_config(const MoggFile & file);

// MoGe-3's changing z coordinate means sparse topology must be rebuilt after
// every log-depth refinement pass.  Topology construction intentionally stays
// on the host; feature gathering and all learned arithmetic remain ordinary
// ggml graph operations and can execute on CPU, Vulkan or Metal.
class SparseRefiner {
public:
    SparseRefiner(const WeightStore & weights, SparseConfig config);

    // factor_coord is H*W*3 interleaved (x/z, y/z, log(z)). encoder_feature is
    // the GPU-resident dense conditioning matrix [C,encoder_width*encoder_height].
    // Only log(z) is updated, exactly as in upstream MoGe-3.
    void refine_once(
        ggml_backend_sched_t sched,
        std::vector<float> & factor_coord,
        int width,
        int height,
        ggml_tensor * encoder_feature,
        int encoder_width,
        int encoder_height) const;

private:
    const WeightStore & w_;
    SparseConfig cfg_;
    // Opaque host-side cache used to reuse an identical quantized sparse topology
    // across consecutive refinement passes without exposing implementation types.
    mutable std::shared_ptr<void> topology_cache_;
    // Opaque backend-allocation cache. When the exact topology object is reused,
    // keep the packed sparse state/index arena alive across refinement passes.
    mutable std::shared_ptr<void> state_cache_;
};

} // namespace moge
