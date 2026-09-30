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

struct CalibrationTap {
    std::string weight_name;
    ggml_tensor * activation = nullptr;
};

struct CalibrationReduction {
    std::string weight_name;
    std::vector<float> sumsq;
    uint64_t samples = 0;
};

// Metal-only fast reducer used by native importance calibration. It reads
// already-computed ggml activations in-place from shared Metal buffers and
// returns only one sum-of-squares value per input channel.
class CalibrationMetalReducer {
public:
    static std::unique_ptr<CalibrationMetalReducer> create(ggml_backend_t backend);
    ~CalibrationMetalReducer();

    CalibrationMetalReducer(const CalibrationMetalReducer &) = delete;
    CalibrationMetalReducer & operator=(const CalibrationMetalReducer &) = delete;

    bool available() const;
    bool verified_first() const;
    std::vector<CalibrationReduction> reduce(const std::vector<CalibrationTap> & taps, bool verify_first = false);

private:
    struct Impl;
    explicit CalibrationMetalReducer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace moge
