// Copyright (c) 2026 Kenji8965
// SPDX-License-Identifier: MIT
// Programming and implementation assistance: ChatGPT 5.6 Sol (OpenAI)
// Human testing, validation, and release acceptance: Kenji8965
// Provided "AS IS"; use at your own risk. See LICENSE.

#pragma once

#include <ggml.h>
#include <ggml-backend.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace moge::profiling {

inline bool env_enabled(const char * name) {
    const char * v = std::getenv(name);
    if (!v || !*v) return false;
    return std::strcmp(v, "0") != 0 && std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "no") != 0 && std::strcmp(v, "off") != 0;
}

inline bool op_profile_enabled() { return env_enabled("MOGE_OP_PROFILE"); }
inline bool memory_trace_enabled() { return env_enabled("MOGE_DEVICE_MEMORY_TRACE"); }

inline void device_memory_sample(ggml_backend_t backend, const char * tag) {
    if (!memory_trace_enabled() || !backend) return;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev) return;
    size_t free_bytes = 0, total_bytes = 0;
    ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
    const size_t used_bytes = total_bytes >= free_bytes ? total_bytes - free_bytes : 0;
    std::fprintf(stderr,
        "moge_device_memory: tag=%s device=\"%s\" free_bytes=%zu total_bytes=%zu used_bytes=%zu\n",
        tag ? tag : "unknown",
        ggml_backend_dev_description(dev) ? ggml_backend_dev_description(dev) : "unknown",
        free_bytes, total_bytes, used_bytes);
}

struct OpProfileState {
    const char * phase = nullptr;
    std::chrono::steady_clock::time_point started{};
    ggml_tensor * current = nullptr;
};

inline bool op_profile_callback(ggml_tensor * t, bool ask, void * user_data) {
    auto * state = static_cast<OpProfileState *>(user_data);
    if (!state || !t) return true;
    if (ask) {
        state->current = t;
        state->started = std::chrono::steady_clock::now();
        // Observe every node. The scheduler therefore submits and synchronizes
        // each node independently; these timings are diagnostic, not production
        // end-to-end latency measurements.
        return true;
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - state->started).count();
    const char * name = t->name[0] ? t->name : "-";
    std::fprintf(stderr,
        "moge_op_profile: phase=\"%s\" op=%s name=\"%s\" ms=%.6f bytes=%zu ne=[%lld,%lld,%lld,%lld]\n",
        state->phase ? state->phase : "unknown", ggml_op_name(t->op), name, ms,
        ggml_nbytes(t), static_cast<long long>(t->ne[0]), static_cast<long long>(t->ne[1]),
        static_cast<long long>(t->ne[2]), static_cast<long long>(t->ne[3]));
    state->current = nullptr;
    return true;
}

} // namespace moge::profiling
