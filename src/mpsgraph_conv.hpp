// Copyright (c) 2026 Kenji8965
// SPDX-License-Identifier: MIT
// Programming and implementation assistance: ChatGPT 5.6 Sol (OpenAI)
// Human testing, validation, and release acceptance: Kenji8965
// Provided "AS IS"; use at your own risk. See LICENSE.

#pragma once

#include <cstdint>
#include <vector>

struct ggml_tensor;

namespace moge {

struct RuntimeMpsConv;

bool runtime_mps_conv_supported();
// Bound zero-copy stripe convolution. Input tensor is F32 [W+2, rows+2, Cin, 1],
// output tensor is F32 [W, rows, Cout, 1]. Weight is OIHW F16 [3,3,Cin,Cout]
// in ggml/MOGG memory order (W,H,Cin,Cout), which is byte-compatible with OIHW.
RuntimeMpsConv * runtime_mps_conv_create_bound(int64_t width, int64_t rows,
                                                int64_t cin, int64_t cout,
                                                const uint16_t * weight_f16,
                                                ggml_tensor * input,
                                                ggml_tensor * output);
void runtime_mps_conv_free(RuntimeMpsConv * ctx);
double runtime_mps_conv_run_bound(RuntimeMpsConv * ctx);

} // namespace moge
