// Copyright (c) 2026 Kenji8965
// SPDX-License-Identifier: MIT
// Programming and implementation assistance: ChatGPT 5.6 Sol (OpenAI)
// Human testing, validation, and release acceptance: Kenji8965
// Provided "AS IS"; use at your own risk. See LICENSE.

#include "calibration.hpp"

namespace moge {
struct CalibrationMetalReducer::Impl {};
CalibrationMetalReducer::CalibrationMetalReducer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CalibrationMetalReducer::~CalibrationMetalReducer() = default;
std::unique_ptr<CalibrationMetalReducer> CalibrationMetalReducer::create(ggml_backend_t) { return {}; }
bool CalibrationMetalReducer::available() const { return false; }
bool CalibrationMetalReducer::verified_first() const { return false; }
std::vector<CalibrationReduction> CalibrationMetalReducer::reduce(const std::vector<CalibrationTap> &, bool) { return {}; }
} // namespace moge
