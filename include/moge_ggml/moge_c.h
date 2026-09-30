// Copyright (c) 2026 Kenji8965
// SPDX-License-Identifier: MIT
// Programming and implementation assistance: ChatGPT 5.6 Sol (OpenAI)
// Human testing, validation, and release acceptance: Kenji8965
// Provided "AS IS"; use at your own risk. See LICENSE.

#pragma once

#include <stddef.h>
#include <stdint.h>

#define MOGE_API_VERSION 1
#define MOGE_VERSION_MAJOR 0
#define MOGE_VERSION_MINOR 4
#define MOGE_VERSION_PATCH 2
#define MOGE_VERSION_STRING "0.4.2"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct moge_model moge_model_t;
typedef struct moge_result moge_result_t;

typedef enum moge_backend {
    MOGE_BACKEND_AUTO = 0,
    MOGE_BACKEND_CPU = 1,
    MOGE_BACKEND_VULKAN = 2,
    MOGE_BACKEND_METAL = 3,
} moge_backend_t;

typedef struct moge_load_options {
    moge_backend_t backend;
    int threads;
    int cpu_fallback;
    int op_offload;
} moge_load_options_t;

typedef struct moge_infer_options {
    int num_tokens;
    int resolution_level;
    int refine_steps;
    int force_projection;
    int apply_mask;
    float fov_x_degrees;
} moge_infer_options_t;

// Defaults matching the C++ API.
moge_load_options_t moge_default_load_options(void);
moge_infer_options_t moge_default_infer_options(void);

// Returns NULL on failure. Use moge_last_error() for a thread-local message.
moge_model_t * moge_model_load_file(const char * path, const moge_load_options_t * options);

// Load a complete MOGG image from memory. If copy_bytes is non-zero, the model
// owns a copy and `data` may be freed immediately. If zero, the caller must keep
// the buffer alive and unchanged until moge_model_free().
moge_model_t * moge_model_load_memory(const void * data, size_t len, int copy_bytes,
                                      const moge_load_options_t * options);
void moge_model_free(moge_model_t * model);

moge_result_t * moge_model_infer(moge_model_t * model, const float * rgb_hwc,
                                 int width, int height,
                                 const moge_infer_options_t * options);
void moge_result_free(moge_result_t * result);

int moge_result_width(const moge_result_t * result);
int moge_result_height(const moge_result_t * result);
const float * moge_result_points(const moge_result_t * result);      // H*W*3
const float * moge_result_depth(const moge_result_t * result);       // H*W
const float * moge_result_normal(const moge_result_t * result);      // H*W*3 or NULL
const uint8_t * moge_result_mask(const moge_result_t * result);      // H*W or NULL
const float * moge_result_intrinsics(const moge_result_t * result);  // 9 floats
float moge_result_metric_scale(const moge_result_t * result);

int moge_model_version(const moge_model_t * model);
moge_backend_t moge_model_backend(const moge_model_t * model);
const char * moge_model_backend_name(const moge_model_t * model);
const char * moge_version_string(void);
const char * moge_last_error(void);

#ifdef __cplusplus
}
#endif
