// Copyright (c) 2026 Kenji8965
// SPDX-License-Identifier: MIT
// Programming and implementation assistance: ChatGPT 5.6 Sol (OpenAI)
// Human testing, validation, and release acceptance: Kenji8965
// Provided "AS IS"; use at your own risk. See LICENSE.

#include <moge_ggml/moge_c.h>
#include <moge_ggml/moge.hpp>

#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <string>

struct moge_model { std::unique_ptr<moge::Model> value; moge_backend_t backend = MOGE_BACKEND_AUTO; std::string backend_name; };
struct moge_result { moge::Result value; };

namespace {
thread_local std::string g_last_error;

moge::Backend convert_backend(moge_backend_t b) {
    switch (b) {
        case MOGE_BACKEND_CPU: return moge::Backend::CPU;
        case MOGE_BACKEND_VULKAN: return moge::Backend::Vulkan;
        case MOGE_BACKEND_METAL: return moge::Backend::Metal;
        case MOGE_BACKEND_AUTO:
        default: return moge::Backend::Auto;
    }
}


moge_backend_t convert_backend(moge::Backend b) {
    switch (b) {
        case moge::Backend::CPU: return MOGE_BACKEND_CPU;
        case moge::Backend::Vulkan: return MOGE_BACKEND_VULKAN;
        case moge::Backend::Metal: return MOGE_BACKEND_METAL;
        case moge::Backend::Auto:
        default: return MOGE_BACKEND_AUTO;
    }
}
moge::LoadOptions convert_load(const moge_load_options_t * in) {
    moge::LoadOptions out;
    if (!in) return out;
    out.backend = convert_backend(in->backend);
    out.threads = in->threads;
    out.cpu_fallback = in->cpu_fallback != 0;
    out.op_offload = in->op_offload != 0;
    return out;
}

moge::InferOptions convert_infer(const moge_infer_options_t * in) {
    moge::InferOptions out;
    if (!in) return out;
    out.num_tokens = in->num_tokens;
    out.resolution_level = in->resolution_level;
    out.refine_steps = in->refine_steps;
    out.force_projection = in->force_projection != 0;
    out.apply_mask = in->apply_mask != 0;
    out.fov_x_degrees = in->fov_x_degrees;
    return out;
}

template <class F, class R = decltype(std::declval<F>()())>
R guard(F && fn, R failure) noexcept {
    try {
        g_last_error.clear();
        return fn();
    } catch (const std::exception & e) {
        g_last_error = e.what();
    } catch (...) {
        g_last_error = "unknown moge-ggml error";
    }
    return failure;
}
}

extern "C" {

moge_load_options_t moge_default_load_options(void) {
    return {MOGE_BACKEND_AUTO, 0, 1, 1};
}

moge_infer_options_t moge_default_infer_options(void) {
    return {0, 9, 3, 1, 1, 0.0f};
}

moge_model_t * moge_model_load_file(const char * path, const moge_load_options_t * options) {
    return guard([&]() -> moge_model_t * {
        if (!path) throw std::runtime_error("moge_model_load_file: path is NULL");
        auto out = std::make_unique<moge_model>();
        out->value = std::make_unique<moge::Model>(std::string(path), convert_load(options));
        out->backend = convert_backend(out->value->backend());
        out->backend_name = out->value->backend_name();
        return out.release();
    }, static_cast<moge_model_t *>(nullptr));
}

moge_model_t * moge_model_load_memory(const void * data, size_t len, int copy_bytes,
                                      const moge_load_options_t * options) {
    return guard([&]() -> moge_model_t * {
        if (!data || !len) throw std::runtime_error("moge_model_load_memory: empty buffer");
        auto out = std::make_unique<moge_model>();
        const auto opt = convert_load(options);
        if (copy_bytes) out->value = std::make_unique<moge::Model>(data, len, opt);
        else out->value = std::make_unique<moge::Model>(moge::Model::from_borrowed_memory(data, len, opt));
        out->backend = convert_backend(out->value->backend());
        out->backend_name = out->value->backend_name();
        return out.release();
    }, static_cast<moge_model_t *>(nullptr));
}

void moge_model_free(moge_model_t * model) { delete model; }

moge_result_t * moge_model_infer(moge_model_t * model, const float * rgb_hwc,
                                 int width, int height,
                                 const moge_infer_options_t * options) {
    return guard([&]() -> moge_result_t * {
        if (!model || !model->value) throw std::runtime_error("moge_model_infer: model is NULL");
        if (!rgb_hwc || width <= 0 || height <= 0) throw std::runtime_error("moge_model_infer: invalid image");
        auto out = std::make_unique<moge_result>();
        out->value = model->value->infer(rgb_hwc, width, height, convert_infer(options));
        return out.release();
    }, static_cast<moge_result_t *>(nullptr));
}

void moge_result_free(moge_result_t * result) { delete result; }
int moge_result_width(const moge_result_t * r) { return r ? r->value.width : 0; }
int moge_result_height(const moge_result_t * r) { return r ? r->value.height : 0; }
const float * moge_result_points(const moge_result_t * r) { return r && !r->value.points.empty() ? r->value.points.data() : nullptr; }
const float * moge_result_depth(const moge_result_t * r) { return r && !r->value.depth.empty() ? r->value.depth.data() : nullptr; }
const float * moge_result_normal(const moge_result_t * r) { return r && !r->value.normal.empty() ? r->value.normal.data() : nullptr; }
const uint8_t * moge_result_mask(const moge_result_t * r) { return r && !r->value.mask.empty() ? r->value.mask.data() : nullptr; }
const float * moge_result_intrinsics(const moge_result_t * r) { return r ? r->value.intrinsics : nullptr; }
float moge_result_metric_scale(const moge_result_t * r) { return r ? r->value.metric_scale : 0.0f; }
int moge_model_version(const moge_model_t * m) { return guard([&](){ return m && m->value ? m->value->version() : 0; }, 0); }
moge_backend_t moge_model_backend(const moge_model_t * m) { return m ? m->backend : MOGE_BACKEND_AUTO; }
const char * moge_model_backend_name(const moge_model_t * m) { return m ? m->backend_name.c_str() : nullptr; }
const char * moge_version_string(void) { return MOGE_VERSION_STRING; }
const char * moge_last_error(void) { return g_last_error.c_str(); }

} // extern "C"
