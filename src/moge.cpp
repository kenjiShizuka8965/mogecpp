// Copyright (c) 2026 Kenji8965
// SPDX-License-Identifier: MIT
// Programming and implementation assistance: ChatGPT 5.6 Sol (OpenAI)
// Human testing, validation, and release acceptance: Kenji8965
// Provided "AS IS"; use at your own risk. See LICENSE.

#include "moge_ggml/moge.hpp"

#include "dense.hpp"
#include "calibration.hpp"
#include "mogg.hpp"
#include "mpsgraph_sdpa.hpp"
#include "mpsgraph_conv.hpp"
#include "postprocess.hpp"
#include "profiling.hpp"
#include "sparse.hpp"

#include <ggml.h>
#include <ggml-backend.h>
#ifdef __APPLE__
#include <ggml-metal.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <filesystem>
#include <fstream>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace moge {
namespace {

const char * backend_label(Backend backend) {
    switch (backend) {
        case Backend::Auto: return "auto";
        case Backend::CPU: return "cpu";
        case Backend::Vulkan: return "vulkan";
        case Backend::Metal: return "metal";
    }
    return "unknown";
}

std::string lower(std::string s) {
    for (char & c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool dev_matches(ggml_backend_dev_t d, Backend wanted) {
    const std::string n = lower(ggml_backend_dev_name(d) ? ggml_backend_dev_name(d) : "");
    const std::string desc = lower(ggml_backend_dev_description(d) ? ggml_backend_dev_description(d) : "");
    const auto reg = ggml_backend_dev_backend_reg(d);
    const std::string reg_name = lower(reg && ggml_backend_reg_name(reg) ? ggml_backend_reg_name(reg) : "");
    if (wanted == Backend::CPU) return ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU;
    // Current ggml names Metal devices MTL0/MTL1 and the registry MTL; the
    // human-readable description is the GPU name (e.g. "Apple M5"), so looking
    // only for the literal word "metal" incorrectly rejects a valid backend.
    if (wanted == Backend::Metal) {
        return reg_name == "mtl" || reg_name.find("metal") != std::string::npos ||
               n.rfind("mtl", 0) == 0 || n.find("metal") != std::string::npos ||
               desc.find("metal") != std::string::npos;
    }
    if (wanted == Backend::Vulkan) {
        return reg_name.find("vulkan") != std::string::npos || reg_name == "vk" ||
               n.find("vulkan") != std::string::npos || n.rfind("vk", 0) == 0 ||
               desc.find("vulkan") != std::string::npos;
    }
    return false;
}

ggml_backend_dev_t find_device(Backend wanted) {
    for (size_t i=0;i<ggml_backend_dev_count();++i) {
        auto d=ggml_backend_dev_get(i);
        if(dev_matches(d,wanted)) return d;
    }
    return nullptr;
}

ggml_backend_dev_t choose_auto() {
#if defined(__APPLE__)
    if(auto d=find_device(Backend::Metal)) return d;
#endif
    if(auto d=find_device(Backend::Vulkan)) return d;
    // Any registered GPU is a reasonable auto fallback (e.g. a downstream
    // backend) before CPU, but this project intentionally only tunes Vulkan/Metal.
    for(size_t i=0;i<ggml_backend_dev_count();++i) {
        auto d=ggml_backend_dev_get(i);
        if(ggml_backend_dev_type(d)==GGML_BACKEND_DEVICE_TYPE_GPU ||
           ggml_backend_dev_type(d)==GGML_BACKEND_DEVICE_TYPE_IGPU) return d;
    }
    return find_device(Backend::CPU);
}

bool env_flag(const char * name, bool default_value = false) {
    const char * v = std::getenv(name);
    if (!v || !*v) return default_value;
    const std::string x = lower(v);
    if (x == "0" || x == "false" || x == "no" || x == "off") return false;
    if (x == "1" || x == "true" || x == "yes" || x == "on") return true;
    return default_value;
}


constexpr int dense_final_tile_rows() { return 128; }
constexpr bool dense_tiled_final_level() { return true; }
constexpr bool dense_stream_level2() { return true; }
constexpr bool dense_stream_level3() { return true; }
constexpr bool dense_stream_final_level() { return true; }
constexpr bool dense_stream_stage_free() { return true; }
constexpr int dense_stream_tile_rows() { return 128; }
constexpr int dense_stream_level1_tile_rows() { return 64; }
constexpr bool dense_destructive_residuals() { return true; }

bool dense_trace() {
    // Sparse tracing is normally requested while debugging end-to-end MoGe-3.
    // Include dense phase boundaries too so a pre-refiner failure is localized
    // instead of looking like a silent Metal crash.
    return env_flag("MOGE_DENSE_TRACE") || env_flag("MOGE_SPARSE_TRACE");
}

ggml_type dense_state_type(Backend backend) {
    return backend == Backend::Vulkan ? GGML_TYPE_F16 : GGML_TYPE_F32;
}

int default_threads() {
    const unsigned n=std::thread::hardware_concurrency();
    return n ? static_cast<int>(n) : 4;
}

void set_backend_threads(ggml_backend_t backend, int threads) {
    if (!backend || threads <= 0) return;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) return;
    auto fn = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
    if (fn) fn(backend, threads);
}

std::vector<float> read_f32(ggml_tensor * t) {
    if(!t) return {};
    const size_t n=static_cast<size_t>(ggml_nelements(t));
    std::vector<float> out(n);
    if(t->type==GGML_TYPE_F32) {
        ggml_backend_tensor_get(t,out.data(),0,n*sizeof(float));
    } else if(t->type==GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> v(n); ggml_backend_tensor_get(t,v.data(),0,v.size()*sizeof(v[0]));
        ggml_fp16_to_fp32_row(v.data(),out.data(),static_cast<int64_t>(n));
    } else if(t->type==GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> v(n); ggml_backend_tensor_get(t,v.data(),0,v.size()*sizeof(v[0]));
        ggml_bf16_to_fp32_row(v.data(),out.data(),static_cast<int64_t>(n));
    } else throw std::runtime_error("unexpected non-float activation type");
    return out;
}


std::vector<float> read_mogg_f32(const MoggFile & file, const std::string & name) {
    const TensorInfo * ti = file.find_tensor(name);
    if (!ti) throw std::runtime_error("missing tensor: " + name);
    const size_t n = static_cast<size_t>(ti->ne[0]) * ti->ne[1] * ti->ne[2] * ti->ne[3];
    std::vector<float> out(n);
    const uint8_t * src = file.tensor_data(*ti);
    if (ti->type == MoggType::F32) {
        std::memcpy(out.data(), src, n * sizeof(float));
    } else if (ti->type == MoggType::F16) {
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(src), out.data(), static_cast<int64_t>(n));
    } else if (ti->type == MoggType::BF16) {
        ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(src), out.data(), static_cast<int64_t>(n));
    } else {
        throw std::runtime_error(name + " must remain floating point (do not quantize positional embeddings)");
    }
    return out;
}

struct CubicSample {
    int i[4]{};
    double w[4]{};
};

// Matches ATen's non-antialiased bicubic kernel (Keys cubic, a=-0.75) with
// align_corners=false. DINOv2 passes scale_factor rather than output size, so
// the source coordinate uses 1/scale_factor exactly instead of in/out.
double cubic1(double x) {
    constexpr double a = -0.75;
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
}

double cubic2(double x) {
    constexpr double a = -0.75;
    return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
}

std::vector<CubicSample> dino_axis_samples(int in_size, int out_size) {
    if (in_size <= 0 || out_size <= 0) throw std::runtime_error("invalid DINO positional interpolation shape");
    // Upstream dinov2: scale_factor = (out_size + 0.1) / in_size.
    const double scale_factor = (static_cast<double>(out_size) + 0.1) / static_cast<double>(in_size);
    const double scale = 1.0 / scale_factor;
    std::vector<CubicSample> table(static_cast<size_t>(out_size));
    for (int o = 0; o < out_size; ++o) {
        const double real = scale * (static_cast<double>(o) + 0.5) - 0.5;
        const int base = static_cast<int>(std::floor(real));
        const double t = real - static_cast<double>(base);
        auto & q = table[static_cast<size_t>(o)];
        q.i[0] = std::clamp(base - 1, 0, in_size - 1);
        q.i[1] = std::clamp(base + 0, 0, in_size - 1);
        q.i[2] = std::clamp(base + 1, 0, in_size - 1);
        q.i[3] = std::clamp(base + 2, 0, in_size - 1);
        q.w[0] = cubic2(t + 1.0);
        q.w[1] = cubic1(t);
        q.w[2] = cubic1(1.0 - t);
        q.w[3] = cubic2(2.0 - t);
    }
    return table;
}

std::vector<float> make_dino_pos_embed(const MoggFile & file, int embed_dim,
                                       int image_w, int image_h, int base_w, int base_h) {
    const TensorInfo * ti = file.find_tensor("encoder.backbone.pos_embed");
    if (!ti) throw std::runtime_error("missing encoder.backbone.pos_embed");
    if (ti->ne[0] != embed_dim || ti->ne[2] != 1 || ti->ne[3] != 1 || ti->ne[1] <= 1) {
        throw std::runtime_error("unexpected DINO positional embedding shape");
    }
    const int src_tokens = static_cast<int>(ti->ne[1]) - 1;
    const int m = static_cast<int>(std::llround(std::sqrt(static_cast<double>(src_tokens))));
    if (m * m != src_tokens) throw std::runtime_error("non-square DINO positional embedding");
    if (base_w <= 0 || base_h <= 0) throw std::runtime_error("invalid DINO target token grid");

    const std::vector<float> src = read_mogg_f32(file, "encoder.backbone.pos_embed");
    std::vector<float> out(static_cast<size_t>(embed_dim) * (1 + base_w * base_h));
    // Class token is never interpolated.
    std::copy_n(src.data(), embed_dim, out.data());

    // Exact upstream fast path: npatch == N and the *input image* is square.
    if (base_w * base_h == src_tokens && image_w == image_h) {
        std::copy_n(src.data() + embed_dim,
                    static_cast<size_t>(embed_dim) * src_tokens,
                    out.data() + embed_dim);
        return out;
    }

    const auto xs = dino_axis_samples(m, base_w);
    const auto ys = dino_axis_samples(m, base_h);
    // PyTorch/ggml logical tensor is [D,tokens], i.e. embedding dimension is
    // contiguous. A direct separable sampler avoids materializing a temporary
    // resized tensor and runs only when the cached plan shape changes.
    for (int oy = 0; oy < base_h; ++oy) {
        for (int ox = 0; ox < base_w; ++ox) {
            const size_t dst_token = 1u + static_cast<size_t>(oy) * base_w + ox;
            for (int d = 0; d < embed_dim; ++d) {
                double sum = 0.0;
                for (int ky = 0; ky < 4; ++ky) {
                    const int sy = ys[static_cast<size_t>(oy)].i[ky];
                    const double wy = ys[static_cast<size_t>(oy)].w[ky];
                    for (int kx = 0; kx < 4; ++kx) {
                        const int sx = xs[static_cast<size_t>(ox)].i[kx];
                        const double wx = xs[static_cast<size_t>(ox)].w[kx];
                        const size_t src_token = 1u + static_cast<size_t>(sy) * m + sx;
                        sum += static_cast<double>(src[src_token * embed_dim + d]) * wy * wx;
                    }
                }
                out[dst_token * embed_dim + d] = static_cast<float>(sum);
            }
        }
    }
    return out;
}


struct LinearAAAxisSample {
    int first = 0;
    std::vector<float> weights;
};

// Match PyTorch's float bilinear antialias=True kernel. This is the separable
// PIL-style triangle filter used by F.interpolate for floating-point tensors:
// support widens by the downsampling factor, coordinates are pixel-center
// based, and the clipped source weights are renormalized per destination pixel.
std::vector<LinearAAAxisSample> linear_aa_axis_samples(int input_size, int output_size) {
    if (input_size <= 0 || output_size <= 0) throw std::runtime_error("invalid antialias resize shape");
    const float scale = static_cast<float>(input_size) / static_cast<float>(output_size);
    const float support = scale >= 1.0f ? scale : 1.0f;
    const float invscale = scale >= 1.0f ? 1.0f / scale : 1.0f;
    const int max_interp_size = static_cast<int>(std::ceil(support)) * 2 + 1;
    std::vector<LinearAAAxisSample> table(static_cast<size_t>(output_size));
    for (int i = 0; i < output_size; ++i) {
        const float center = scale * (static_cast<float>(i) + 0.5f);
        // Deliberately use truncation (C++ float -> int64 semantics), matching
        // ATen's static_cast<int64_t> rather than floor().
        const int xmin = std::max(static_cast<int>(center - support + 0.5f), 0);
        int xsize = std::min(static_cast<int>(center + support + 0.5f), input_size) - xmin;
        xsize = std::clamp(xsize, 0, max_interp_size);
        auto & q = table[static_cast<size_t>(i)];
        q.first = xmin;
        q.weights.resize(static_cast<size_t>(xsize));
        float total = 0.0f;
        for (int j = 0; j < xsize; ++j) {
            const float x = std::abs((static_cast<float>(j + xmin) - center + 0.5f) * invscale);
            const float w = x < 1.0f ? 1.0f - x : 0.0f;
            q.weights[static_cast<size_t>(j)] = w;
            total += w;
        }
        if (total != 0.0f) {
            for (float & w : q.weights) w /= total;
        }
    }
    return table;
}

// Input is public-API RGB HWC float32. Output is ggml's [W,H,C,1] storage,
// i.e. planar C-major bytes. Horizontal then vertical pass intentionally
// mirrors ATen's separable 2-D AA interpolation order.
void resize_rgb_to_planar_aa(const float * rgb, int input_w, int input_h,
                             int output_w, int output_h,
                             const std::vector<LinearAAAxisSample> & xs,
                             const std::vector<LinearAAAxisSample> & ys,
                             std::vector<float> & tmp,
                             std::vector<float> & out) {
    if (!rgb || input_w <= 0 || input_h <= 0 || output_w <= 0 || output_h <= 0)
        throw std::runtime_error("invalid RGB resize input");
    if (xs.size() != static_cast<size_t>(output_w) || ys.size() != static_cast<size_t>(output_h))
        throw std::runtime_error("antialias resize table shape mismatch");

    const size_t out_npx = static_cast<size_t>(output_w) * output_h;
    out.resize(out_npx * 3);
    if (input_w == output_w && input_h == output_h) {
        for (size_t i = 0; i < out_npx; ++i) {
            out[i] = rgb[3*i + 0];
            out[out_npx + i] = rgb[3*i + 1];
            out[2*out_npx + i] = rgb[3*i + 2];
        }
        return;
    }

    // Keep channels interleaved in the horizontal temporary: each RGB triplet
    // is contiguous and only three values are accumulated per tap. The vector
    // capacity is retained in DensePlan across same-shape inference calls.
    tmp.resize(static_cast<size_t>(input_h) * output_w * 3);
    for (int y = 0; y < input_h; ++y) {
        const float * src_row = rgb + static_cast<size_t>(y) * input_w * 3;
        float * dst_row = tmp.data() + static_cast<size_t>(y) * output_w * 3;
        for (int ox = 0; ox < output_w; ++ox) {
            const auto & q = xs[static_cast<size_t>(ox)];
            float r = 0.0f, g = 0.0f, b = 0.0f;
            for (size_t k = 0; k < q.weights.size(); ++k) {
                const float w = q.weights[k];
                const float * px = src_row + static_cast<size_t>(q.first + static_cast<int>(k)) * 3;
                r += px[0] * w; g += px[1] * w; b += px[2] * w;
            }
            dst_row[3*ox + 0] = r;
            dst_row[3*ox + 1] = g;
            dst_row[3*ox + 2] = b;
        }
    }

    for (int oy = 0; oy < output_h; ++oy) {
        const auto & q = ys[static_cast<size_t>(oy)];
        for (int ox = 0; ox < output_w; ++ox) {
            float r = 0.0f, g = 0.0f, b = 0.0f;
            for (size_t k = 0; k < q.weights.size(); ++k) {
                const size_t off = (static_cast<size_t>(q.first + static_cast<int>(k)) * output_w + ox) * 3;
                const float w = q.weights[k];
                r += tmp[off + 0] * w; g += tmp[off + 1] * w; b += tmp[off + 2] * w;
            }
            const size_t d = static_cast<size_t>(oy) * output_w + ox;
            out[d] = r; out[out_npx + d] = g; out[2*out_npx + d] = b;
        }
    }
}

std::vector<float> make_uv_host(int image_w, int image_h, int width, int height) {
    if (image_w <= 0 || image_h <= 0 || width <= 0 || height <= 0)
        throw std::runtime_error("invalid UV shape");
    const float ar = static_cast<float>(image_w) / static_cast<float>(image_h);
    const float den = std::sqrt(1.0f + ar * ar);
    const float span_x = ar / den;
    const float span_y = 1.0f / den;
    const size_t plane = static_cast<size_t>(width) * height;
    std::vector<float> uv(plane * 2);
    for (int y = 0; y < height; ++y) {
        const float v = (2.0f * y - (height - 1.0f)) / height * span_y;
        for (int x = 0; x < width; ++x) {
            const size_t i = static_cast<size_t>(y) * width + x;
            uv[i] = (2.0f * x - (width - 1.0f)) / width * span_x;
            uv[plane + i] = v;
        }
    }
    return uv;
}

void normalize_planar_rgb(std::vector<float> & image, const std::array<float,3> & mean,
                          const std::array<float,3> & stdv) {
    if (image.size() % 3 != 0) throw std::runtime_error("invalid planar RGB buffer");
    const size_t npx = image.size() / 3;
    for (int c = 0; c < 3; ++c) {
        if (stdv[c] == 0.0f) throw std::runtime_error("zero image normalization stddev");
        float * p = image.data() + static_cast<size_t>(c) * npx;
        const float mu = mean[c], sigma = stdv[c];
        for (size_t i = 0; i < npx; ++i) p[i] = (p[i] - mu) / sigma;
    }
}

std::vector<float> whc_to_hwc(const std::vector<float> & raw, int w, int h, int c) {
    if(raw.size()!=static_cast<size_t>(w)*h*c) throw std::runtime_error("activation shape mismatch");
    std::vector<float> out(raw.size());
    for(int ch=0;ch<c;++ch) for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
        out[(static_cast<size_t>(y)*w+x)*c+ch]=raw[(static_cast<size_t>(ch)*h+y)*w+x];
    }
    return out;
}

} // namespace

struct Model::Impl {
    struct DensePhase {
        ggml_context * ctx = nullptr;
        ggml_cgraph * graph = nullptr;
        std::string label;
        // ggml_backend_sched_split_graph() may rewrite node->src[] to temporary
        // backend-copy tensors owned by the scheduler context. A cached phase
        // graph must never retain those pointers after the scheduler is reset or
        // freed. Keep the pristine source topology keyed by node identity so it
        // remains valid even if Metal graph optimization reorders graph->nodes.
        bool canonical_sources_captured = false;
        std::unordered_map<ggml_tensor *, std::array<ggml_tensor *, GGML_MAX_SRC>> canonical_sources;
        bool calibration_nodes_discovered = false;
        // Calibration must observe GEMM inputs while they are still live. ggml's
        // gallocr is free to recycle intermediate buffers later in the phase, so
        // storing src[1] pointers and reducing them only after graph completion is
        // incorrect. Keep only the pristine MUL_MAT node identity -> weight-name
        // mapping here; the scheduler eval callback reads the runtime src[1]
        // immediately after that node has executed (same pattern as llama.cpp's
        // imatrix collector).
        std::unordered_map<ggml_tensor *, std::string> calibration_nodes;
        bool run_mps_after = false;
        // final-decoder MPSGraph convolution hook. Prefix phases write a
        // padded stripe into a persistent shared Metal tensor, then compute_dense()
        // invokes the referenced MPSGraph conv before the following suffix phase.
        int mps_conv_spec = -1;
        int dino_stage_index = -1;
        ggml_tensor * dino_hidden_output = nullptr;
        ~DensePhase() { if (ctx) ggml_free(ctx); }
    };

    struct DenseMpsConvSlot {
        int rows = 0;
        int64_t width = 0;
        int64_t cin = 0;
        int64_t cout = 0;
        ggml_tensor * input = nullptr;   // F32 [W+2,R+2,Cin,1]
        ggml_tensor * output = nullptr;  // F32 [W,R,Cout,1]
        ggml_backend_buffer_t input_buffer = nullptr;
        ggml_backend_buffer_t output_buffer = nullptr;
    };

    struct DenseMpsConvSpec {
        size_t slot = 0;
        std::string weight_name;
        RuntimeMpsConv * runtime = nullptr;
    };

    struct DensePlan {
        // Monolithic compatibility path uses ctx/graph. The GPU-optimized path
        // uses state_ctx/state_buffer + several small cached phase graphs.
        bool segmented = false;
        ggml_context * ctx = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_context * state_ctx = nullptr;
        ggml_backend_buffer_t state_buffer = nullptr;
        std::vector<std::unique_ptr<DensePhase>> phases;
        ggml_tensor * image = nullptr;
        ggml_tensor * pos_embed = nullptr;
        std::array<ggml_tensor *, 5> uv{};
        // Segmented DINO uses ping-pong spill buffers. Never read and write the
        // same externally allocated tensor in one graph: ggml does not model
        // that alias safely across scheduler phases.
        std::array<ggml_tensor *, 2> hidden{};
        // Likewise accumulate the four DINO output projections in a two-buffer
        // ping-pong arena instead of retaining four full [D,tokens] tap tensors.
        std::array<ggml_tensor *, 2> encoder_accum{};
        ggml_tensor * final_cls = nullptr;
        // Decoder segmentation keeps the five exported neck features persistent.
        // Cross-level *transition* state is distinct: ConvStack resamples the
        // pre-output-block residual state, not the exported level_out tensor.
        // Neck transitions and the three output heads therefore share the same
        // two physical ping-pong arenas; their lifetimes are serial and never
        // overlap, so this preserves the monolithic ConvStack contract without
        // retaining another full pyramid.
        ggml_tensor * encoder_input = nullptr; // [encoder_out+2, baseW*baseH]
        std::array<ggml_tensor *, 5> neck_features{};
        std::array<ggml_tensor *, 4> neck_transition{};
        std::array<std::vector<ggml_tensor *>, 2> decoder_alias_tensors;
        std::array<ggml_backend_buffer_t, 2> decoder_alias_buffers{};
        // Shared full-resolution staging tensor for the final 2x bilinear
        // decoder transition. Neck/points/normal/mask use it serially, so one
        // physical allocation replaces four full-frame temporaries.
        ggml_tensor * decoder_highres_stage = nullptr;
        // shares the final-level high-resolution staging arena with the
        // level-3 pre-resampler tensors; these phases are strictly serial.
        std::vector<ggml_tensor *> decoder_shared_stage_tensors;
        ggml_backend_buffer_t decoder_highres_buffer = nullptr;
        // no longer needs the full-frame pre-residual arena: each stripe
        // computes the resampler 3x3 over its residual halo and immediately runs
        // the complete residual stack before copying only center rows to dst.
        bool dino_mpsgraph_f16 = false;
        bool dino_mpsgraph_direct = false;
        bool dino_mpsgraph_bridge = false;
        std::array<ggml_backend_buffer_t, 4> mps_buffers{};
        ggml_tensor * mps_q = nullptr;
        ggml_tensor * mps_k = nullptr;
        ggml_tensor * mps_v = nullptr;
        ggml_tensor * mps_attn16 = nullptr;
        RuntimeMpsSdpa * mps_sdpa = nullptr;
        bool dense_mpsgraph_final_conv = false;
        bool dense_mpsgraph_neck3_entry = false;
        bool dense_mpsgraph_neck3_residual0 = false;
        std::vector<DenseMpsConvSlot> dense_mps_conv_slots;
        std::vector<DenseMpsConvSpec> dense_mps_conv_specs;
        DenseOutputs out;
        int image_w = 0;
        int image_h = 0;
        int base_w = 0;
        int base_h = 0;
        int encoder_w = 0;
        int encoder_h = 0;
        std::vector<LinearAAAxisSample> resize_x;
        std::vector<LinearAAAxisSample> resize_y;
        std::vector<float> resize_tmp;
        std::vector<float> image_upload;

        ~DensePlan() {
            phases.clear();
            if (mps_sdpa) runtime_mps_sdpa_free(mps_sdpa);
            for (auto & c : dense_mps_conv_specs) if (c.runtime) runtime_mps_conv_free(c.runtime);
            for (auto & s : dense_mps_conv_slots) {
                if (s.input_buffer) ggml_backend_buffer_free(s.input_buffer);
                if (s.output_buffer) ggml_backend_buffer_free(s.output_buffer);
            }
            for (auto & b : mps_buffers) if (b) ggml_backend_buffer_free(b);
            for (auto & b : decoder_alias_buffers) if (b) ggml_backend_buffer_free(b);
            if (decoder_highres_buffer) ggml_backend_buffer_free(decoder_highres_buffer);
            if (state_buffer) ggml_backend_buffer_free(state_buffer);
            if (state_ctx) ggml_free(state_ctx);
            if (ctx) ggml_free(ctx);
        }
        bool matches(int iw, int ih, int bw, int bh) const {
            return iw == image_w && ih == image_h && bw == base_w && bh == base_h;
        }
    };

    MoggFile file;
    Architecture arch;
    SparseConfig sparse_cfg;
    LoadOptions options;
    ggml_backend_t primary=nullptr;
    ggml_backend_t cpu=nullptr;
    std::vector<ggml_backend_t> sched_backends;
    // Keep dense and sparse schedulers separate. The sparse graph must be reset
    // and rebuilt every refinement step as log-depth changes the voxel topology;
    // sharing that scheduler would invalidate an otherwise reusable dense graph.
    ggml_backend_sched_t dense_sched=nullptr;
    ggml_backend_sched_t sparse_sched=nullptr;
    std::unique_ptr<WeightStore> weights;
    std::unique_ptr<SparseRefiner> refiner;
    std::unique_ptr<DensePlan> dense_plan;
    Backend selected_backend = Backend::Auto;
    std::string backend_desc;
    std::array<float,3> image_mean{};
    std::array<float,3> image_std{};
    std::unique_ptr<CalibrationMetalReducer> calibration_reducer;
    std::unordered_map<std::string, std::vector<double>> calibration_sums;
    std::unordered_map<std::string, uint64_t> calibration_counts;
    uint64_t calibration_images = 0;
    bool calibration_active = false;
    bool calibration_verify_first = false;
    struct CalibrationEvalState {
        Impl * self = nullptr;
        DensePhase * phase = nullptr;
        std::exception_ptr error;
        size_t observed = 0;
    };

    void release_runtime() noexcept {
        // Scheduler reset first releases graph allocations; then discard the
        // metadata tensors that contain pointers into those allocations. This
        // helper also makes constructor failure cleanup deterministic because
        // the raw ggml handles are not member-RAII types.
        if(dense_sched && dense_plan) ggml_backend_sched_reset(dense_sched);
        dense_plan.reset();
        refiner.reset();
        weights.reset();
        if(sparse_sched) { ggml_backend_sched_free(sparse_sched); sparse_sched=nullptr; }
        if(dense_sched)  { ggml_backend_sched_free(dense_sched);  dense_sched=nullptr; }
        if(cpu && cpu!=primary) { ggml_backend_free(cpu); cpu=nullptr; }
        if(primary) { ggml_backend_free(primary); primary=nullptr; }
        cpu=nullptr;
        sched_backends.clear();
    }

    Impl(const std::string & path, const LoadOptions & opt):file(path),arch(read_architecture(file)),
        sparse_cfg(read_sparse_config(file)),options(opt) {
        initialize();
    }

    Impl(const void * data, size_t size, bool copy, const LoadOptions & opt):file(data, size, copy),
        arch(read_architecture(file)), sparse_cfg(read_sparse_config(file)), options(opt) {
        initialize();
    }

    void initialize() {
        try {
            const auto mean = read_mogg_f32(file, "encoder.image_mean");
            const auto stdv = read_mogg_f32(file, "encoder.image_std");
            if (mean.size() != 3 || stdv.size() != 3) throw std::runtime_error("invalid encoder image normalization constants");
            std::copy_n(mean.begin(), 3, image_mean.begin());
            std::copy_n(stdv.begin(), 3, image_std.begin());
            ggml_backend_load_all();
            ggml_backend_dev_t pd = options.backend==Backend::Auto ? choose_auto() : find_device(options.backend);
            if(!pd) {
                std::string message = std::string("requested ") + backend_label(options.backend) + " ggml backend is not available";
                if (options.backend == Backend::Vulkan) message += "; configure ggml with -DGGML_VULKAN=ON and install a Vulkan SDK/driver";
                else if (options.backend == Backend::Metal) message += "; configure ggml with -DGGML_METAL=ON";
                throw std::runtime_error(message);
            }
            primary=ggml_backend_dev_init(pd,nullptr);
            if(!primary) throw std::runtime_error("failed to initialize primary ggml backend");
            profiling::device_memory_sample(primary, "backend_initialized");
            if (dev_matches(pd, Backend::Metal)) selected_backend = Backend::Metal;
            else if (dev_matches(pd, Backend::Vulkan)) selected_backend = Backend::Vulkan;
            else if (dev_matches(pd, Backend::CPU)) selected_backend = Backend::CPU;
            else selected_backend = Backend::Auto; // downstream/unknown ggml accelerator
            backend_desc=ggml_backend_dev_description(pd) ? ggml_backend_dev_description(pd) : ggml_backend_name(primary);

            const int threads=options.threads>0?options.threads:default_threads();
            if(ggml_backend_dev_type(pd)==GGML_BACKEND_DEVICE_TYPE_CPU) {
                set_backend_threads(primary,threads); cpu=primary;
            } else {
                // ggml 0.25.3's multi-backend scheduler has a hard contract that
                // the final backend is CPU, even when the workload is intended to
                // stay GPU-resident. Keep CPU as the mandatory terminal fallback;
                // --no-cpu-fallback is implemented by explicitly pinning graph
                // inputs to the primary GPU below. Unsupported GPU ops may still
                // require CPU because the scheduler cannot be constructed without it.
                auto cd=find_device(Backend::CPU);
                if(!cd) throw std::runtime_error("ggml 0.25.3 scheduler requires a CPU backend after the GPU backend");
                cpu=ggml_backend_dev_init(cd,nullptr);
                if(!cpu) throw std::runtime_error("failed to initialize mandatory ggml CPU scheduler backend");
                set_backend_threads(cpu,threads);
            }
            sched_backends.push_back(primary);
            if(cpu && cpu!=primary) sched_backends.push_back(cpu);
            dense_sched = make_dense_scheduler();
            if(sparse_cfg.present) {
                sparse_sched = make_sparse_scheduler();
            }
            weights=std::make_unique<WeightStore>(file,primary);
            profiling::device_memory_sample(primary, "weights_loaded");
            if(sparse_cfg.present) refiner=std::make_unique<SparseRefiner>(*weights,sparse_cfg);
        } catch (...) {
            release_runtime();
            throw;
        }
    }

    ~Impl() { release_runtime(); }

    ggml_backend_sched_t make_dense_scheduler() {
        auto * sched = ggml_backend_sched_new(
            sched_backends.data(), nullptr, static_cast<int>(sched_backends.size()),
            32768, false, options.op_offload);
        if (!sched) throw std::runtime_error("failed to create dense ggml scheduler");
        return sched;
    }

    ggml_backend_sched_t make_sparse_scheduler() {
        auto * sched = ggml_backend_sched_new(
            sched_backends.data(), nullptr, static_cast<int>(sched_backends.size()),
            8192, false, options.op_offload);
        if (!sched) throw std::runtime_error("failed to create sparse ggml scheduler");
        return sched;
    }

    ggml_backend_sched_t make_phase_scheduler(size_t graph_size) {
        auto * sched = ggml_backend_sched_new(
            sched_backends.data(), nullptr, static_cast<int>(sched_backends.size()),
            graph_size, false, options.op_offload);
        if (!sched) throw std::runtime_error("failed to create phase-local ggml scheduler");
        return sched;
    }

    bool calibration_weight_eligible(const std::string & name) const {
        const auto * ti = file.find_tensor(name);
        if (!ti) return false;
        const bool floating = ti->type == MoggType::F32 || ti->type == MoggType::F16 || ti->type == MoggType::BF16;
        const int64_t nel = ti->ne[0] * ti->ne[1] * ti->ne[2] * ti->ne[3];
        // Match the Q4_K/Q6_K dense default quantizer eligibility. Q8_0 does not
        // consume importance, so calibration intentionally targets the K-quant
        // subset only. The complete refiner is marked TF_SPARSE by the converter.
        return floating && ti->ndim == 2 && nel >= 4096 &&
               !(ti->flags & TF_SENSITIVE) && !(ti->flags & TF_SPARSE) &&
               (ti->ne[0] % 256 == 0);
    }

    void discover_calibration_nodes(DensePhase & ph) {
        if (ph.calibration_nodes_discovered) return;
        ph.calibration_nodes_discovered = true;
        if (!ph.graph) return;
        for (int i = 0; i < ggml_graph_n_nodes(ph.graph); ++i) {
            auto * node = ggml_graph_node(ph.graph,i);
            if (!node || node->op != GGML_OP_MUL_MAT || !node->src[0] || !node->src[1]) continue;
            const char * raw = ggml_get_name(node->src[0]);
            if (!raw || !*raw) continue;
            const std::string name(raw);
            if (!calibration_weight_eligible(name)) continue;
            ph.calibration_nodes.emplace(node, name);
        }
    }

    void accumulate_calibration_reduction(CalibrationReduction && r) {
        auto & dst = calibration_sums[r.weight_name];
        if (dst.empty()) dst.assign(r.sumsq.size(), 0.0);
        if (dst.size() != r.sumsq.size()) throw std::runtime_error("calibration width changed for " + r.weight_name);
        for (size_t i = 0; i < dst.size(); ++i) dst[i] += static_cast<double>(r.sumsq[i]);
        calibration_counts[r.weight_name] += r.samples;
    }

    static bool calibration_eval_callback(ggml_tensor * node, bool ask, void * user_data) noexcept {
        auto * st = static_cast<CalibrationEvalState *>(user_data);
        if (!st || !st->self || !st->phase || st->error) return false;
        const auto it = st->phase->calibration_nodes.find(node);
        if (ask) return it != st->phase->calibration_nodes.end();
        if (it == st->phase->calibration_nodes.end()) return true;
        try {
            // The scheduler calls us immediately after computing `node` and
            // synchronizing its backend. Read the *runtime* src[1]: split_graph()
            // may have replaced a source with a backend-copy tensor, and this is
            // the exact activation consumed by the GEMM. Crucially, its gallocr
            // storage has not yet been recycled by subsequent graph nodes.
            if (!node->src[1]) throw std::runtime_error("calibration GEMM lost activation source for " + it->second);
            if (!st->self->calibration_reducer)
                st->self->calibration_reducer = CalibrationMetalReducer::create(st->self->primary);
            if (!st->self->calibration_reducer || !st->self->calibration_reducer->available())
                throw std::runtime_error("native importance calibration currently requires the Metal backend");
            CalibrationTap tap;
            tap.weight_name = it->second;
            tap.activation = node->src[1];
            auto reduced = st->self->calibration_reducer->reduce({tap}, st->self->calibration_verify_first);
            if (reduced.size() != 1) throw std::runtime_error("calibration reducer returned unexpected result count");
            st->self->accumulate_calibration_reduction(std::move(reduced.front()));
            ++st->observed;
            return true;
        } catch (...) {
            st->error = std::current_exception();
            return false;
        }
    }

    bool dense_phase_local_scheduler() const { return primary != cpu; }

    // ggml 0.25.3's gallocr initializes a view only when its *immediate*
    // view_src already has a backend buffer. With segmented execution we build
    // phase graphs before the persistent state context is backend-allocated, so
    // nested view chains rooted in persistent tensors can survive graph
    // allocation with an intermediate view still carrying buffer/data == null.
    // Metal's fusion/buffer lookup then dereferences that chain and reports
    // `tensor '... (view)' buffer is nil`. Refresh views parent-first *after*
    // sched_alloc_graph(), when both persistent roots and phase-local scratch
    // have real buffers. This is zero-copy and fixes the allocator ordering
    // issue without materializing cross-phase states.
    static size_t refresh_graph_views(ggml_cgraph * graph) {
        std::unordered_set<ggml_tensor *> seen;
        size_t initialized = 0;
        std::function<void(ggml_tensor *)> visit = [&](ggml_tensor * t) {
            if (!t || !seen.insert(t).second) return;
            if (t->view_src) visit(t->view_src);
            for (ggml_tensor * src : t->src) visit(src);
            if (t->view_src && t->buffer == nullptr &&
                t->view_src->buffer != nullptr && t->view_src->data != nullptr) {
                if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error(std::string("failed to initialize ggml view: ") +
                                             (t->name[0] ? t->name : "<unnamed>"));
                }
                ++initialized;
            }
        };
        const int n = ggml_graph_n_nodes(graph);
        for (int i = 0; i < n; ++i) visit(ggml_graph_node(graph, i));
        return initialized;
    }


    // A scheduler allocation writes backend buffer/data pointers directly into
    // graph tensor metadata. Those pointers remain after the scheduler (and its
    // gallocr buffers) is freed. Reusing a cached phase graph with a fresh
    // scheduler would therefore treat scratch as externally pre-allocated; on
    // Metal this becomes a use-after-free / nil MTLBuffer on the next inference.
    //
    // Do *not* blindly clear every phase-context tensor, though. ggml_cpy() and
    // many view ops create phase-local tensor metadata whose view_src is a
    // persistent destination/weight living in another context. Their backend
    // binding is durable and is also how ggml 0.25.3 seeds backend assignment
    // when a cached graph is split again. Clearing those external-rooted views
    // can leave a CPY source with backend id -1 on the next allocation.
    //
    // Therefore detach only scheduler-owned roots and views rooted in another
    // phase-local tensor. Views rooted directly in external persistent/model
    // tensors retain their live buffer/data/extra binding across scheduler
    // reset/free.
    static size_t restore_phase_graph_sources(DensePhase & ph) {
        if (!ph.graph) return 0;

        const int n = ggml_graph_n_nodes(ph.graph);
        if (!ph.canonical_sources_captured) {
            ph.canonical_sources.reserve(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) {
                ggml_tensor * node = ggml_graph_node(ph.graph, i);
                std::array<ggml_tensor *, GGML_MAX_SRC> srcs{};
                for (int j = 0; j < GGML_MAX_SRC; ++j) srcs[j] = node->src[j];
                ph.canonical_sources.emplace(node, srcs);
            }
            ph.canonical_sources_captured = true;
            return 0;
        }

        size_t restored = 0;
        for (int i = 0; i < n; ++i) {
            ggml_tensor * node = ggml_graph_node(ph.graph, i);
            const auto it = ph.canonical_sources.find(node);
            if (it == ph.canonical_sources.end()) {
                throw std::runtime_error(std::string("dense cached phase gained an unknown node: ") +
                                         (node->name[0] ? node->name : "<unnamed>"));
            }
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (node->src[j] != it->second[j]) {
                    node->src[j] = it->second[j];
                    ++restored;
                }
            }
        }
        return restored;
    }

    static size_t detach_phase_backend_bindings(ggml_context * ctx) {
        std::unordered_set<ggml_tensor *> owned;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            owned.insert(t);
        }

        auto view_chain_reaches_external = [&](ggml_tensor * t) {
            // checked only the immediate view_src.  That is insufficient for
            // chains such as outer_view -> inner_phase_view -> persistent_state:
            // clearing outer_view leaves ggml's fresh scheduler with an
            // unassigned immediate view_src during split_graph() on the next
            // cached inference.  Preserve the *entire* chain whenever any
            // ancestor leaves the phase context.
            std::unordered_set<ggml_tensor *> seen;
            for (ggml_tensor * v = t ? t->view_src : nullptr; v; v = v->view_src) {
                if (!seen.insert(v).second) break;
                if (owned.find(v) == owned.end()) return true;
            }
            return false;
        };

        size_t detached = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (view_chain_reaches_external(t)) continue;
            if (t->buffer || t->data || t->extra) {
                t->buffer = nullptr;
                t->data = nullptr;
                t->extra = nullptr;
                ++detached;
            }
        }
        return detached;
    }

    DensePlan & get_dense_plan(int iw, int ih, int bw, int bh) {
        if (dense_plan && dense_plan->matches(iw, ih, bw, bh)) return *dense_plan;

        if (dense_plan) {
            if (dense_sched) ggml_backend_sched_reset(dense_sched);
            dense_plan.reset();
        }

        auto plan = std::make_unique<DensePlan>();
        plan->image_w = iw; plan->image_h = ih; plan->base_w = bw; plan->base_h = bh;
        plan->encoder_w = bw * 14;
        plan->encoder_h = bh * 14;
        plan->resize_x = linear_aa_axis_samples(iw, plan->encoder_w);
        plan->resize_y = linear_aa_axis_samples(ih, plan->encoder_h);
        plan->resize_tmp.reserve(static_cast<size_t>(ih) * plan->encoder_w * 3);
        plan->image_upload.reserve(static_cast<size_t>(plan->encoder_w) * plan->encoder_h * 3);

        const std::vector<float> pos_data = make_dino_pos_embed(file,arch.embed_dim,iw,ih,bw,bh);
        plan->segmented = primary != cpu;

        if (!plan->segmented) {
            ggml_init_params p{}; p.mem_size=128u<<20; p.mem_buffer=nullptr; p.no_alloc=true;
            plan->ctx=ggml_init(p);
            if(!plan->ctx) throw std::runtime_error("ggml_init failed for dense graph");

            plan->image=ggml_new_tensor_4d(plan->ctx,GGML_TYPE_F32,plan->encoder_w,plan->encoder_h,3,1);
            ggml_set_name(plan->image,"image"); ggml_set_input(plan->image);
            plan->pos_embed=ggml_new_tensor_2d(plan->ctx,GGML_TYPE_F32,arch.embed_dim,1 + bw*bh);
            ggml_set_name(plan->pos_embed,"dino_pos_embed"); ggml_set_input(plan->pos_embed);
            for (int level = 0; level < 5; ++level) {
                const int uw = bw << level, uh = bh << level;
                plan->uv[static_cast<size_t>(level)] = ggml_new_tensor_4d(plan->ctx,GGML_TYPE_F32,uw,uh,2,1);
                ggml_set_name(plan->uv[static_cast<size_t>(level)], ("view_uv_" + std::to_string(level)).c_str());
                ggml_set_input(plan->uv[static_cast<size_t>(level)]);
            }
            DenseBuilder b(plan->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv);
            plan->out=b.build(plan->image);
            for(auto * t:{plan->out.raw_points,plan->out.raw_normal,plan->out.raw_mask,
                          plan->out.raw_scale,plan->out.encoder_feature}) if(t) ggml_set_output(t);
            plan->graph=ggml_new_graph_custom(plan->ctx,32768,false);
            for(auto * t:{plan->out.raw_points,plan->out.raw_normal,plan->out.raw_mask,
                          plan->out.raw_scale,plan->out.encoder_feature}) if(t) ggml_build_forward_expand(plan->graph,t);
            if (!options.cpu_fallback && primary != cpu) {
                ggml_backend_sched_set_tensor_backend(dense_sched, plan->image, primary);
                ggml_backend_sched_set_tensor_backend(dense_sched, plan->pos_embed, primary);
                for (auto * uv : plan->uv) if (uv) ggml_backend_sched_set_tensor_backend(dense_sched, uv, primary);
            }
            if(!ggml_backend_sched_alloc_graph(dense_sched,plan->graph)) {
                ggml_backend_sched_reset(dense_sched);
                throw std::runtime_error("failed to allocate dense graph");
            }
            ggml_backend_tensor_set(plan->pos_embed,pos_data.data(),0,pos_data.size()*sizeof(float));
            for (int level = 0; level < 5; ++level) {
                const int uw = bw << level, uh = bh << level;
                const std::vector<float> uv_data = make_uv_host(iw,ih,uw,uh);
                ggml_backend_tensor_set(plan->uv[static_cast<size_t>(level)],uv_data.data(),0,uv_data.size()*sizeof(float));
            }
            if (dense_trace()) {
                const double mib = static_cast<double>(ggml_backend_sched_get_buffer_size(dense_sched, primary))/(1024.0*1024.0);
                std::fprintf(stderr,"moge_dense_phase: monolithic scratch=%.1f MiB\n",mib);
            }
        } else {
            if (arch.intermediate_layers.empty()) throw std::runtime_error("segmented DINO requires intermediate layers");
            for (size_t i = 0; i < arch.intermediate_layers.size(); ++i) {
                const int64_t cur = arch.intermediate_layers[i];
                if (cur < 0 || cur >= arch.depth || (i && cur <= arch.intermediate_layers[i-1]))
                    throw std::runtime_error("invalid DINO intermediate layer ordering");
            }

            ggml_init_params sp{}; sp.mem_size=16u<<20; sp.mem_buffer=nullptr; sp.no_alloc=true;
            plan->state_ctx=ggml_init(sp);
            if(!plan->state_ctx) throw std::runtime_error("ggml_init failed for dense persistent state");
            const ggml_type stype = dense_state_type(selected_backend);
            const int64_t tokens = 1 + static_cast<int64_t>(bw) * bh;
            plan->image=ggml_new_tensor_4d(plan->state_ctx,GGML_TYPE_F32,plan->encoder_w,plan->encoder_h,3,1);
            ggml_set_name(plan->image,"dense.state.image"); ggml_set_input(plan->image);
            plan->pos_embed=ggml_new_tensor_2d(plan->state_ctx,GGML_TYPE_F32,arch.embed_dim,tokens);
            ggml_set_name(plan->pos_embed,"dense.state.pos_embed"); ggml_set_input(plan->pos_embed);
            for (int level = 0; level < 5; ++level) {
                const int uw = bw << level, uh = bh << level;
                plan->uv[static_cast<size_t>(level)] = ggml_new_tensor_4d(plan->state_ctx,GGML_TYPE_F32,uw,uh,2,1);
                ggml_set_name(plan->uv[static_cast<size_t>(level)], ("dense.state.uv." + std::to_string(level)).c_str());
                ggml_set_input(plan->uv[static_cast<size_t>(level)]);
            }
            for (int i = 0; i < 2; ++i) {
                plan->hidden[static_cast<size_t>(i)] = ggml_new_tensor_2d(plan->state_ctx,stype,arch.embed_dim,tokens);
                ggml_set_name(plan->hidden[static_cast<size_t>(i)], ("dense.state.hidden."+std::to_string(i)).c_str());
                ggml_set_input(plan->hidden[static_cast<size_t>(i)]);
                plan->encoder_accum[static_cast<size_t>(i)] = ggml_new_tensor_4d(
                    plan->state_ctx, stype, bw, bh, arch.encoder_out, 1);
                ggml_set_name(plan->encoder_accum[static_cast<size_t>(i)], ("dense.state.encoder_accum."+std::to_string(i)).c_str());
                ggml_set_input(plan->encoder_accum[static_cast<size_t>(i)]);
            }
            // Only the last normalized CLS token is required after the backbone
            // (for the metric scale head), so keep D floats rather than the
            // complete final DINO tap.
            plan->final_cls=ggml_new_tensor_2d(plan->state_ctx,GGML_TYPE_F32,arch.embed_dim,1);
            ggml_set_name(plan->final_cls,"dense.state.final_cls"); ggml_set_input(plan->final_cls);

            // Supported Metal path: use zero-copy MPSGraph F16 SDPA when the
            // runtime supports it. Other backends stay on the ggml path.
            const bool primary_is_metal = dev_matches(ggml_backend_get_device(primary), Backend::Metal);
            plan->dino_mpsgraph_f16 = primary_is_metal && stype == GGML_TYPE_F32 && runtime_mps_sdpa_supported();
            plan->dino_mpsgraph_direct = plan->dino_mpsgraph_f16;
            plan->dino_mpsgraph_bridge = false;
            if (plan->dino_mpsgraph_f16) {
                if (!dev_matches(ggml_backend_get_device(primary), Backend::Metal)) throw std::runtime_error("MPSGraph SDPA requires Metal");
                if (!runtime_mps_sdpa_supported()) throw std::runtime_error("MPSGraph F16 SDPA is unavailable on this system");
                if (stype != GGML_TYPE_F32) throw std::runtime_error("MPSGraph SDPA requires F32 dense state");
                const int64_t hd = arch.embed_dim / arch.heads;
                plan->mps_q = ggml_new_tensor_4d(plan->state_ctx,GGML_TYPE_F16,hd,tokens,arch.heads,1);
                plan->mps_k = ggml_new_tensor_4d(plan->state_ctx,GGML_TYPE_F16,hd,tokens,arch.heads,1);
                plan->mps_v = ggml_new_tensor_4d(plan->state_ctx,GGML_TYPE_F16,hd,tokens,arch.heads,1);
                plan->mps_attn16 = ggml_new_tensor_2d(plan->state_ctx,GGML_TYPE_F16,arch.embed_dim,tokens);
                for (auto * t : {plan->mps_q,plan->mps_k,plan->mps_v,plan->mps_attn16}) { ggml_set_input(t); ggml_set_output(t); }
                ggml_set_name(plan->mps_q,"dense.state.mps.q");
                ggml_set_name(plan->mps_k,"dense.state.mps.k");
                ggml_set_name(plan->mps_v,"dense.state.mps.v");
                ggml_set_name(plan->mps_attn16,"dense.state.mps.attn16");
            }

            auto make_phase = [&](const std::string & label, size_t mem_size) -> DensePhase * {
                auto ph=std::make_unique<DensePhase>();
                ph->label=label;
                ggml_init_params pp{}; pp.mem_size=mem_size; pp.mem_buffer=nullptr; pp.no_alloc=true;
                ph->ctx=ggml_init(pp);
                if(!ph->ctx) throw std::runtime_error("ggml_init failed for dense phase "+label);
                auto * raw=ph.get();
                plan->phases.push_back(std::move(ph));
                return raw;
            };

            const size_t nphases = arch.intermediate_layers.size();
            if (!plan->dino_mpsgraph_f16) {
                int begin_block=0;
                for (size_t si=0; si<nphases; ++si) {
                    const int end_block=static_cast<int>(arch.intermediate_layers[si]);
                    auto * ph=make_phase("dino."+std::to_string(begin_block)+"-"+std::to_string(end_block),32u<<20);
                    ph->dino_stage_index=static_cast<int>(si);
                    DenseBuilder b(ph->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,dense_destructive_residuals());
                    ggml_tensor * x=nullptr;
                    if (si==0) {
                        x=b.build_backbone_input(plan->image);
                    } else {
                        x=plan->hidden[(si-1)&1u];
                        if (x->type!=GGML_TYPE_F32) x=ggml_cast(ph->ctx,x,GGML_TYPE_F32);
                    }
                    x=b.run_backbone_blocks(x,begin_block,end_block);
                    auto * tap=b.normalized_backbone_tap(x);
                    ggml_tensor * prev_acc = nullptr;
                    if (si>0) {
                        prev_acc=plan->encoder_accum[(si-1)&1u];
                        if (prev_acc->type!=GGML_TYPE_F32) prev_acc=ggml_cast(ph->ctx,prev_acc,GGML_TYPE_F32);
                    }
                    auto * enc_sum=b.accumulate_backbone_projection(tap,si,prev_acc);
                    std::vector<ggml_tensor *> outs;
                    if (si+1<nphases) {
                        ph->dino_hidden_output=plan->hidden[si&1u];
                        outs.push_back(ggml_cpy(ph->ctx,x,ph->dino_hidden_output));
                    }
                    outs.push_back(ggml_cpy(ph->ctx,enc_sum,plan->encoder_accum[si&1u]));
                    if (si+1==nphases) {
                        auto * cls=ggml_view_2d(ph->ctx,tap,arch.embed_dim,1,tap->nb[1],0);
                        if (cls->type!=GGML_TYPE_F32) cls=ggml_cast(ph->ctx,cls,GGML_TYPE_F32);
                        outs.push_back(ggml_cpy(ph->ctx,cls,plan->final_cls));
                    }
                    ph->graph=ggml_new_graph_custom(ph->ctx,16384,false);
                    for(auto * o:outs) ggml_build_forward_expand(ph->graph,o);
                    begin_block=end_block+1;
                }
            } else {
                // Materialize the patch+position backbone input once so every
                // transformer block can use the same prefix -> MPS SDPA -> suffix
                // execution pattern with a durable residual source.
                {
                    auto * ph=make_phase("dino.mps.input",16u<<20);
                    DenseBuilder b(ph->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,dense_destructive_residuals());
                    auto * cp=ggml_cpy(ph->ctx,b.build_backbone_input(plan->image),plan->hidden[0]);
                    ph->graph=ggml_new_graph_custom(ph->ctx,8192,false);
                    ggml_build_forward_expand(ph->graph,cp);
                }
                int cur=0;
                size_t si=0;
                if (!plan->dino_mpsgraph_bridge) {
                    for (int block=0; block<arch.depth; ++block) {
                        auto * prefix=make_phase("dino.mps."+std::to_string(block)+".prefix",32u<<20);
                        {
                            DenseBuilder b(prefix->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,dense_destructive_residuals());
                            auto * x=plan->hidden[static_cast<size_t>(cur)];
                            auto * n=b.run_backbone_block_norm1(x,block);
                            auto * qkv=b.run_backbone_block_qkv(n,block);
                            auto qkv16=b.run_backbone_block_prepare_qkv_f16(qkv,block);
                            auto * cq=ggml_cpy(prefix->ctx,qkv16[0],plan->mps_q);
                            auto * ck=ggml_cpy(prefix->ctx,qkv16[1],plan->mps_k);
                            auto * cv=ggml_cpy(prefix->ctx,qkv16[2],plan->mps_v);
                            prefix->graph=ggml_new_graph_custom(prefix->ctx,16384,false);
                            ggml_build_forward_expand(prefix->graph,cq);
                            ggml_build_forward_expand(prefix->graph,ck);
                            ggml_build_forward_expand(prefix->graph,cv);
                            prefix->run_mps_after=true;
                        }
                        const int next=1-cur;
                        auto * suffix=make_phase("dino.mps."+std::to_string(block)+".suffix",32u<<20);
                        {
                            DenseBuilder b(suffix->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,dense_destructive_residuals());
                            auto * residual=plan->hidden[static_cast<size_t>(cur)];
                            auto * attn=ggml_cast(suffix->ctx,plan->mps_attn16,GGML_TYPE_F32);
                            auto * x=b.run_backbone_block_attn_project(residual,attn,block);
                            x=b.run_backbone_block_mlp(x,block);
                            std::vector<ggml_tensor *> outs;
                            suffix->dino_hidden_output=plan->hidden[static_cast<size_t>(next)];
                            outs.push_back(ggml_cpy(suffix->ctx,x,suffix->dino_hidden_output));
                            if (si<nphases && block==static_cast<int>(arch.intermediate_layers[si])) {
                                suffix->dino_stage_index=static_cast<int>(si);
                                auto * tap=b.normalized_backbone_tap(x);
                                ggml_tensor * prev_acc=nullptr;
                                if(si>0){ prev_acc=plan->encoder_accum[(si-1)&1u]; if(prev_acc->type!=GGML_TYPE_F32) prev_acc=ggml_cast(suffix->ctx,prev_acc,GGML_TYPE_F32); }
                                auto * enc_sum=b.accumulate_backbone_projection(tap,si,prev_acc);
                                outs.push_back(ggml_cpy(suffix->ctx,enc_sum,plan->encoder_accum[si&1u]));
                                if(si+1==nphases){ auto * cls=ggml_view_2d(suffix->ctx,tap,arch.embed_dim,1,tap->nb[1],0); if(cls->type!=GGML_TYPE_F32) cls=ggml_cast(suffix->ctx,cls,GGML_TYPE_F32); outs.push_back(ggml_cpy(suffix->ctx,cls,plan->final_cls)); }
                                ++si;
                            }
                            suffix->graph=ggml_new_graph_custom(suffix->ctx,16384,false);
                            for(auto * o:outs) ggml_build_forward_expand(suffix->graph,o);
                        }
                        cur=next;
                    }
                } else {
                    // Prime block 0 Q/K/V once. Thereafter each bridge phase finishes
                    // block N and immediately prepares block N+1 Q/K/V from the fresh
                    // F32 block output before spilling that output to the durable hidden
                    // ping-pong tensor used as block N+1's residual source.
                    {
                        auto * prefix=make_phase("dino.mps.0.prefix",32u<<20);
                        DenseBuilder b(prefix->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,dense_destructive_residuals());
                        auto * x=plan->hidden[0];
                        auto * n=b.run_backbone_block_norm1(x,0);
                        auto * qkv=b.run_backbone_block_qkv(n,0);
                        auto qkv16=b.run_backbone_block_prepare_qkv_f16(qkv,0);
                        auto * cq=ggml_cpy(prefix->ctx,qkv16[0],plan->mps_q);
                        auto * ck=ggml_cpy(prefix->ctx,qkv16[1],plan->mps_k);
                        auto * cv=ggml_cpy(prefix->ctx,qkv16[2],plan->mps_v);
                        prefix->graph=ggml_new_graph_custom(prefix->ctx,16384,false);
                        ggml_build_forward_expand(prefix->graph,cq);
                        ggml_build_forward_expand(prefix->graph,ck);
                        ggml_build_forward_expand(prefix->graph,cv);
                        prefix->run_mps_after=true;
                    }
                    for (int block=0; block<arch.depth; ++block) {
                        const int next=1-cur;
                        const bool has_next=(block+1)<arch.depth;
                        auto * ph=make_phase(has_next
                            ? "dino.mps."+std::to_string(block)+"-"+std::to_string(block+1)+".bridge"
                            : "dino.mps."+std::to_string(block)+".suffix",
                            48u<<20);
                        DenseBuilder b(ph->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,dense_destructive_residuals());
                        auto * residual=plan->hidden[static_cast<size_t>(cur)];
                        auto * attn=ggml_cast(ph->ctx,plan->mps_attn16,GGML_TYPE_F32);
                        auto * x=b.run_backbone_block_attn_project(residual,attn,block);
                        x=b.run_backbone_block_mlp(x,block);
                        std::vector<ggml_tensor *> outs;
                        ph->dino_hidden_output=plan->hidden[static_cast<size_t>(next)];
                        // Preserve the split-path materialization boundary even in the
                        // bridged graph. The next block prefix consumes the CPY result
                        // (a view of the durable hidden destination), not the transient
                        // block-N result directly.
                        auto * spilled_hidden = ggml_cpy(ph->ctx,x,ph->dino_hidden_output);
                        outs.push_back(spilled_hidden);
                        if (si<nphases && block==static_cast<int>(arch.intermediate_layers[si])) {
                            ph->dino_stage_index=static_cast<int>(si);
                            auto * tap=b.normalized_backbone_tap(x);
                            ggml_tensor * prev_acc=nullptr;
                            if(si>0){ prev_acc=plan->encoder_accum[(si-1)&1u]; if(prev_acc->type!=GGML_TYPE_F32) prev_acc=ggml_cast(ph->ctx,prev_acc,GGML_TYPE_F32); }
                            auto * enc_sum=b.accumulate_backbone_projection(tap,si,prev_acc);
                            outs.push_back(ggml_cpy(ph->ctx,enc_sum,plan->encoder_accum[si&1u]));
                            if(si+1==nphases){ auto * cls=ggml_view_2d(ph->ctx,tap,arch.embed_dim,1,tap->nb[1],0); if(cls->type!=GGML_TYPE_F32) cls=ggml_cast(ph->ctx,cls,GGML_TYPE_F32); outs.push_back(ggml_cpy(ph->ctx,cls,plan->final_cls)); }
                            ++si;
                        }
                        if (has_next) {
                            auto * n=b.run_backbone_block_norm1(spilled_hidden,block+1);
                            auto * qkv=b.run_backbone_block_qkv(n,block+1);
                            auto qkv16=b.run_backbone_block_prepare_qkv_f16(qkv,block+1);
                            outs.push_back(ggml_cpy(ph->ctx,qkv16[0],plan->mps_q));
                            outs.push_back(ggml_cpy(ph->ctx,qkv16[1],plan->mps_k));
                            outs.push_back(ggml_cpy(ph->ctx,qkv16[2],plan->mps_v));
                            ph->run_mps_after=true;
                        }
                        ph->graph=ggml_new_graph_custom(ph->ctx,24576,false);
                        for(auto * o:outs) ggml_build_forward_expand(ph->graph,o);
                        cur=next;
                    }
                }
                if (si!=nphases) throw std::runtime_error("MPSGraph DINO did not reach all intermediate taps");
            }

            // The spatial decoder is segmented much more aggressively than the
            // DINO trunk. A monolithic neck + 3 heads graph reached ~1.8 GiB of
            // Metal scratch at 640x480 and also exposed graph/alias instability.
            // Persist the compact neck pyramid once, then run each head one level
            // at a time through two shared ping-pong state arenas.
            if (arch.neck.dim_res.size() > plan->neck_features.size())
                throw std::runtime_error("segmented decoder supports at most five neck levels");
            for (size_t i=0; i+1<arch.neck.dim_res.size(); ++i) {
                if (arch.neck.dim_out[i] >= 0)
                    throw std::runtime_error("segmented decoder requires unprojected intermediate neck states");
            }
            auto validate_head_levels = [&](const ConvStackCfg & cfg, const char * name) {
                if (cfg.present && cfg.dim_res.size()!=arch.neck.dim_res.size())
                    throw std::runtime_error(std::string("segmented ")+name+" level count differs from neck");
            };
            validate_head_levels(arch.points,"points head");
            validate_head_levels(arch.normal,"normal head");
            validate_head_levels(arch.mask,"mask head");
            const ggml_type decoder_state_type = dense_state_type(selected_backend);
            auto persist_like = [&](ggml_tensor * src, const char * name, ggml_type type) -> ggml_tensor * {
                if (!src) return nullptr;
                auto * dst=ggml_new_tensor(plan->state_ctx,type,ggml_n_dims(src),src->ne);
                ggml_set_name(dst,name); ggml_set_input(dst); ggml_set_output(dst);
                return dst;
            };
            auto copy_to = [&](ggml_context * ctx, ggml_tensor * src, ggml_tensor * dst) -> ggml_tensor * {
                if (!src || !dst) return nullptr;
                if (src->type != dst->type) src=ggml_cast(ctx,src,dst->type);
                return ggml_cpy(ctx,src,dst);
            };

            const size_t decoder_last = arch.neck.dim_res.empty() ? 0 : arch.neck.dim_res.size()-1;
            auto can_tile_final = [&](const ConvStackCfg & cfg) -> bool {
                return cfg.present && cfg.dim_res.size()==arch.neck.dim_res.size() && decoder_last>0 &&
                       cfg.resamplers.size()>=decoder_last && cfg.resamplers[decoder_last-1]=="bilinear" &&
                       cfg.num_res_blocks.size()>decoder_last && cfg.num_res_blocks[decoder_last]==0 &&
                       cfg.dim_res[decoder_last-1]==arch.neck.dim_res[decoder_last-1];
            };
            bool tiled_final=dense_tiled_final_level() && can_tile_final(arch.neck);
            if (arch.points.present) tiled_final=tiled_final && can_tile_final(arch.points);
            if (arch.normal.present) tiled_final=tiled_final && can_tile_final(arch.normal);
            if (arch.mask.present) tiled_final=tiled_final && can_tile_final(arch.mask);
            const int final_tile_rows=dense_final_tile_rows();
            const bool stream_final_requested = dense_stream_final_level();
            const int64_t final_w=static_cast<int64_t>(bw) << decoder_last;
            const int64_t final_h=static_cast<int64_t>(bh) << decoder_last;
            const bool mps_conv_auto = dev_matches(ggml_backend_get_device(primary), Backend::Metal) &&
                                       decoder_state_type == GGML_TYPE_F32 && runtime_mps_conv_supported();
            plan->dense_mpsgraph_final_conv = mps_conv_auto && tiled_final && stream_final_requested;
            plan->dense_mpsgraph_neck3_entry = mps_conv_auto;
            plan->dense_mpsgraph_neck3_residual0 = false;
            if (plan->dense_mpsgraph_neck3_entry) {
                if (!dev_matches(ggml_backend_get_device(primary), Backend::Metal))
                    throw std::runtime_error("MPSGraph neck.3 convolution requires Metal");
                if (!runtime_mps_conv_supported())
                    throw std::runtime_error("MPSGraph convolution is unavailable on this system");
                if (decoder_state_type != GGML_TYPE_F32)
                    throw std::runtime_error("MPSGraph neck.3 convolution requires F32 dense state");
            }
            if (plan->dense_mpsgraph_final_conv) {
                if (!tiled_final || !stream_final_requested)
                    throw std::runtime_error("MPSGraph final convolution requires streamed final decoder");
                if (!dev_matches(ggml_backend_get_device(primary), Backend::Metal))
                    throw std::runtime_error("MPSGraph final convolution requires Metal");
                if (!runtime_mps_conv_supported())
                    throw std::runtime_error("MPSGraph convolution is unavailable on this system");
                if (decoder_state_type != GGML_TYPE_F32)
                    throw std::runtime_error("MPSGraph final convolution requires F32 dense state");
            }
            if (tiled_final && !stream_final_requested) {
                const int64_t stage_c=arch.neck.dim_res[decoder_last-1];
                plan->decoder_highres_stage=ggml_new_tensor_4d(plan->state_ctx,decoder_state_type,
                                                               final_w,final_h,stage_c,1);
                ggml_set_name(plan->decoder_highres_stage,"dense.alias.decoder_highres_stage");
                ggml_set_input(plan->decoder_highres_stage);
                ggml_set_output(plan->decoder_highres_stage);
                plan->decoder_shared_stage_tensors.push_back(plan->decoder_highres_stage);
            }

            // Level-1 streaming is part of the validated Metal profile and is
            // intentionally disabled on Vulkan, where this path is slower than the fixed release path.
            const bool stream_level1_requested = dev_matches(ggml_backend_get_device(primary), Backend::Metal);
            const bool stream_level2_requested = dense_stream_level2();
            const bool stream_level3_requested = dense_stream_level3();
            const int stream_tile_rows = dense_stream_tile_rows();
            const int stream_level1_tile_rows = dense_stream_level1_tile_rows();
            auto stream_level_enabled = [&](size_t level) -> bool {
                return (level == 1 && stream_level1_requested) ||
                       (level == 2 && stream_level2_requested) ||
                       (level == 3 && stream_level3_requested);
            };
            auto can_stream_cfg = [&](const ConvStackCfg & cfg, size_t level) -> bool {
                if (!stream_level_enabled(level) || !cfg.present || level == 0 || level >= decoder_last ||
                    level >= cfg.dim_res.size() || level - 1 >= cfg.resamplers.size() ||
                    level >= cfg.num_res_blocks.size()) return false;
                if (cfg.in_norm != "none" || cfg.hidden_norm != "none") return false;
                const std::string & rt = cfg.resamplers[level - 1];
                return rt == "conv_transpose" || rt == "nearest" || rt == "bilinear";
            };
            auto persist_level_shape = [&](const ConvStackCfg & cfg, size_t level, const std::string & name,
                                           ggml_type type) -> ggml_tensor * {
                const int64_t W=static_cast<int64_t>(bw) << level;
                const int64_t H=static_cast<int64_t>(bh) << level;
                const int64_t C=cfg.dim_res.at(level);
                auto * t=ggml_new_tensor_4d(plan->state_ctx,type,W,H,C,1);
                ggml_set_name(t,name.c_str()); ggml_set_input(t); ggml_set_output(t);
                return t;
            };
            auto make_stream_stage = [&](const ConvStackCfg & cfg, const std::string & pfx,
                                         size_t level, const char * kind,
                                         std::vector<ggml_tensor *> & arena) -> ggml_tensor * {
                auto * t=persist_level_shape(cfg,level,
                    "dense.alias.stream."+pfx+"."+std::to_string(level)+"."+kind,GGML_TYPE_F32);
                arena.push_back(t);
                return t;
            };

            auto dense_mps_slot_for = [&](int64_t width, int rows, int64_t cin, int64_t cout) -> size_t {
                for (size_t i=0; i<plan->dense_mps_conv_slots.size(); ++i) {
                    const auto & q=plan->dense_mps_conv_slots[i];
                    if (q.rows==rows && q.width==width && q.cin==cin && q.cout==cout) return i;
                }
                DenseMpsConvSlot q;
                q.rows=rows; q.width=width; q.cin=cin; q.cout=cout;
                q.input=ggml_new_tensor_4d(plan->state_ctx,GGML_TYPE_F32,width+2,rows+2,cin,1);
                q.output=ggml_new_tensor_4d(plan->state_ctx,GGML_TYPE_F32,width,rows,cout,1);
                const std::string stem="dense.state.mpsconv."+std::to_string(width)+"x"+std::to_string(rows)+"."+std::to_string(cin)+"x"+std::to_string(cout);
                ggml_set_name(q.input,(stem+".in").c_str());
                ggml_set_name(q.output,(stem+".out").c_str());
                ggml_set_input(q.input); ggml_set_output(q.input);
                ggml_set_input(q.output); ggml_set_output(q.output);
                plan->dense_mps_conv_slots.push_back(q);
                return plan->dense_mps_conv_slots.size()-1;
            };
            auto dense_mps_spec_for = [&](const std::string & weight_name, int64_t width, int rows, int64_t cin, int64_t cout) -> int {
                const size_t slot=dense_mps_slot_for(width,rows,cin,cout);
                for (size_t i=0; i<plan->dense_mps_conv_specs.size(); ++i) {
                    const auto & q=plan->dense_mps_conv_specs[i];
                    if (q.slot==slot && q.weight_name==weight_name) return static_cast<int>(i);
                }
                DenseMpsConvSpec q; q.slot=slot; q.weight_name=weight_name;
                plan->dense_mps_conv_specs.push_back(std::move(q));
                return static_cast<int>(plan->dense_mps_conv_specs.size()-1);
            };

            // Build one complete intermediate level with real phase boundaries.
            // There is no ggml_concat/SET chain spanning all stripes: each tile
            // is a separate graph whose scheduler is freed after compute.
            // proved this fused design for level 3; reuses the same exact
            // execution at level 2, whose untiled neck phase is now the peak.
            auto build_streamed_level = [&](const ConvStackCfg & cfg, const std::string & pfx,
                                            size_t level, ggml_tensor * prev, ggml_tensor * feature,
                                            ggml_tensor * dst) {
                if (!can_stream_cfg(cfg,level))
                    throw std::runtime_error("streamed config not supported: "+pfx+"."+std::to_string(level));
                if (!prev || !dst) throw std::runtime_error("streamed level missing state: "+pfx);
                const int64_t W=dst->ne[0], H=dst->ne[1], C=dst->ne[2];
                const std::string rp=pfx+".resamplers."+std::to_string(level-1);
                const bool stage_free = dense_stream_stage_free() &&
                                        cfg.resamplers[level-1]=="conv_transpose" &&
                                        weights->has(rp+".0.weight.mogg_phase");
                ggml_tensor * prepared=nullptr;
                if (!stage_free) {
                    prepared=make_stream_stage(cfg,pfx,level,"prepared",plan->decoder_shared_stage_tensors);
                    auto * ph=make_phase(pfx+"."+std::to_string(level)+".stream.prepare",32u<<20);
                    DenseBuilder b(ph->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,false);
                    ggml_tensor * src=prev;
                    if (src->type!=GGML_TYPE_F32) src=ggml_cast(ph->ctx,src,GGML_TYPE_F32);
                    auto * prep=b.conv_stack_stream_prepare_resample(cfg,pfx,level,src);
                    if (prep->ne[0]!=W || prep->ne[1]!=H || prep->ne[2]!=C)
                        throw std::runtime_error("streamed prepared shape mismatch: "+pfx+"."+std::to_string(level));
                    auto * cp=copy_to(ph->ctx,prep,prepared);
                    ph->graph=ggml_new_graph_custom(ph->ctx,8192,false);
                    ggml_build_forward_expand(ph->graph,cp);
                }
                const int level_tile_rows = level == 1 ? stream_level1_tile_rows : stream_tile_rows;
                const bool mps_neck3 = plan->dense_mpsgraph_neck3_entry && pfx=="neck" && level==3;
                const bool mps_residual0 = mps_neck3 && plan->dense_mpsgraph_neck3_residual0;
                if (mps_neck3 && !stage_free)
                    throw std::runtime_error("MPSGraph neck.3 entry requires stage-free ConvTranspose");
                const std::string entry_weight=rp+".1.weight";
                const std::string residual0_weight=pfx+".res_blocks."+std::to_string(level)+".0.layers.2.weight";
                ggml_tensor * entry_wt=mps_neck3 ? weights->get(entry_weight) : nullptr;
                ggml_tensor * residual0_wt=mps_residual0 ? weights->get(residual0_weight) : nullptr;
                if (mps_neck3 && (!entry_wt || entry_wt->type!=GGML_TYPE_F16 || entry_wt->ne[0]!=3 || entry_wt->ne[1]!=3))
                    throw std::runtime_error("MPSGraph neck.3 entry requires F16 3x3 weight: "+entry_weight);
                if (mps_residual0 && (!residual0_wt || residual0_wt->type!=GGML_TYPE_F16 || residual0_wt->ne[0]!=3 || residual0_wt->ne[1]!=3))
                    throw std::runtime_error("MPSGraph neck.3 residual0 requires F16 3x3 weight: "+residual0_weight);
                for (int64_t y0=0; y0<H; y0+=level_tile_rows) {
                    const int64_t y1=std::min<int64_t>(H,y0+level_tile_rows);
                    if (!mps_neck3) {
                        auto * ph=make_phase(pfx+"."+std::to_string(level)+(stage_free?".stream.direct.":".stream.fused.")+
                                            std::to_string(y0)+"-"+std::to_string(y1),64u<<20);
                        DenseBuilder b(ph->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,false);
                        auto * tile=stage_free ?
                            b.conv_stack_stream_phase_fused_tile(cfg,pfx,level,prev,feature,y0,y1) :
                            b.conv_stack_stream_fused_tile(cfg,pfx,level,prepared,feature,y0,y1);
                        auto * out=ggml_view_4d(ph->ctx,dst,W,y1-y0,C,1,
                                               dst->nb[1],dst->nb[2],dst->nb[3],
                                               static_cast<size_t>(y0)*dst->nb[1]);
                        auto * cp=copy_to(ph->ctx,tile,out);
                        ph->graph=ggml_new_graph_custom(ph->ctx,16384,false);
                        ggml_build_forward_expand(ph->graph,cp);
                        continue;
                    }
                    const int64_t radius=2LL*static_cast<int64_t>(cfg.num_res_blocks[level]);
                    const int64_t entry_y0=std::max<int64_t>(0,y0-radius);
                    const int64_t entry_y1=std::min<int64_t>(H,y1+radius);
                    const int rows=static_cast<int>(entry_y1-entry_y0);
                    const int spec_idx=dense_mps_spec_for(entry_weight,W,rows,entry_wt->ne[2],entry_wt->ne[3]);
                    auto & spec=plan->dense_mps_conv_specs.at(static_cast<size_t>(spec_idx));
                    auto & slot=plan->dense_mps_conv_slots.at(spec.slot);
                    {
                        auto * ph=make_phase("neck.3.mps.entry.prefix."+std::to_string(y0)+"-"+std::to_string(y1),64u<<20);
                        DenseBuilder b(ph->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,false);
                        auto * padded=b.conv_stack_stream_phase_prepare_entry_conv_tile(cfg,pfx,level,prev,y0,y1);
                        if (padded->ne[0]!=slot.input->ne[0] || padded->ne[1]!=slot.input->ne[1] || padded->ne[2]!=slot.input->ne[2])
                            throw std::runtime_error("MPSGraph neck.3 prepared stripe shape mismatch");
                        auto * cp=copy_to(ph->ctx,padded,slot.input);
                        ph->graph=ggml_new_graph_custom(ph->ctx,8192,false);
                        ggml_build_forward_expand(ph->graph,cp);
                        ph->mps_conv_spec=spec_idx;
                    }
                    if (!mps_residual0) {
                        auto * ph=make_phase("neck.3.mps.entry.suffix."+std::to_string(y0)+"-"+std::to_string(y1),64u<<20);
                        DenseBuilder b(ph->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,false);
                        auto * tile=b.conv_stack_stream_phase_finish_entry_conv_tile(cfg,pfx,level,slot.output,feature,y0,y1);
                        auto * out=ggml_view_4d(ph->ctx,dst,W,y1-y0,C,1,dst->nb[1],dst->nb[2],dst->nb[3],static_cast<size_t>(y0)*dst->nb[1]);
                        auto * cp=copy_to(ph->ctx,tile,out);
                        ph->graph=ggml_new_graph_custom(ph->ctx,16384,false);
                        ggml_build_forward_expand(ph->graph,cp);
                        continue;
                    }

                    // : preserve the exact pre-residual halo in scheduler-owned
                    // storage before reusing the shared MPS slot for residual0 layers.2.
                    auto * pre_residual=ggml_new_tensor_4d(plan->state_ctx,GGML_TYPE_F32,W,rows,C,1);
                    const std::string pre_name="dense.alias.neck3.pre_residual."+std::to_string(rows);
                    ggml_set_name(pre_residual,pre_name.c_str()); ggml_set_input(pre_residual); ggml_set_output(pre_residual);
                    plan->decoder_shared_stage_tensors.push_back(pre_residual);
                    {
                        auto * ph=make_phase("neck.3.mps.pre_residual."+std::to_string(y0)+"-"+std::to_string(y1),64u<<20);
                        DenseBuilder b(ph->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,false);
                        auto * pre=b.conv_stack_stream_phase_finish_entry_pre_residual_tile(cfg,pfx,level,slot.output,feature,y0,y1);
                        auto * cp=copy_to(ph->ctx,pre,pre_residual);
                        ph->graph=ggml_new_graph_custom(ph->ctx,16384,false);
                        ggml_build_forward_expand(ph->graph,cp);
                    }
                    const int residual_spec_idx=dense_mps_spec_for(residual0_weight,W,rows,residual0_wt->ne[2],residual0_wt->ne[3]);
                    {
                        auto * ph=make_phase("neck.3.mps.residual0.prefix."+std::to_string(y0)+"-"+std::to_string(y1),32u<<20);
                        DenseBuilder b(ph->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,false);
                        auto * padded=b.conv_stack_stream_residual0_prepare_conv_tile(cfg,pfx,level,pre_residual);
                        auto * cp=copy_to(ph->ctx,padded,slot.input);
                        ph->graph=ggml_new_graph_custom(ph->ctx,8192,false);
                        ggml_build_forward_expand(ph->graph,cp);
                        ph->mps_conv_spec=residual_spec_idx;
                    }
                    {
                        auto * ph=make_phase("neck.3.mps.residual0.suffix."+std::to_string(y0)+"-"+std::to_string(y1),64u<<20);
                        DenseBuilder b(ph->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,false);
                        auto * tile=b.conv_stack_stream_residual0_finish_conv_tile(cfg,pfx,level,pre_residual,slot.output,H,y0,y1);
                        auto * out=ggml_view_4d(ph->ctx,dst,W,y1-y0,C,1,dst->nb[1],dst->nb[2],dst->nb[3],static_cast<size_t>(y0)*dst->nb[1]);
                        auto * cp=copy_to(ph->ctx,tile,out);
                        ph->graph=ggml_new_graph_custom(ph->ctx,16384,false);
                        ggml_build_forward_expand(ph->graph,cp);
                    }
                }
            };

            // stream the final bilinear resize itself. Each tile
            // creates only the low-resolution source rows required for exact 2x
            // interpolation plus the trailing 3x3 halo, then writes directly to
            // the persistent final tensor. No full-frame high-resolution stage.
            auto build_streamed_final = [&](const ConvStackCfg & cfg, const std::string & pfx,
                                            ggml_tensor * prev, ggml_tensor * feature,
                                            ggml_tensor *& out_tensor, const char * out_name,
                                            ggml_type out_type) {
                if (!tiled_final || !stream_final_requested)
                    throw std::runtime_error("streamed final decoder unexpectedly disabled");
                if (!prev) throw std::runtime_error("streamed final decoder missing previous state");
                const int64_t cout=cfg.dim_out[decoder_last]>=0 ? cfg.dim_out[decoder_last] : cfg.dim_res[decoder_last];
                out_tensor=ggml_new_tensor_4d(plan->state_ctx,out_type,final_w,final_h,cout,1);
                ggml_set_name(out_tensor,out_name); ggml_set_input(out_tensor); ggml_set_output(out_tensor);

                const std::string rp=pfx+".resamplers."+std::to_string(decoder_last-1)+".1";
                ggml_tensor * mps_weight=nullptr;
                int64_t mps_cin=0,mps_cout=0;
                if (plan->dense_mpsgraph_final_conv) {
                    mps_weight=weights->get(rp+".weight");
                    if (!mps_weight || mps_weight->type!=GGML_TYPE_F16 || mps_weight->ne[0]!=3 || mps_weight->ne[1]!=3)
                        throw std::runtime_error("MPSGraph final decoder requires F16 3x3 trailing conv weight: "+rp+".weight");
                    mps_cin=mps_weight->ne[2]; mps_cout=mps_weight->ne[3];
                }

                for (int64_t y0=0; y0<final_h; y0+=final_tile_rows) {
                    const int64_t y1=std::min<int64_t>(final_h,y0+final_tile_rows);
                    if (!plan->dense_mpsgraph_final_conv) {
                        auto * ph=make_phase(pfx+"."+std::to_string(decoder_last)+".stream.final."+
                                            std::to_string(y0)+"-"+std::to_string(y1),32u<<20);
                        DenseBuilder b(ph->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,false);
                        auto * tile=b.conv_stack_final_level_fused_tile(cfg,pfx,decoder_last,prev,feature,y0,y1);
                        auto * dst=ggml_view_4d(ph->ctx,out_tensor,final_w,y1-y0,cout,1,
                                               out_tensor->nb[1],out_tensor->nb[2],out_tensor->nb[3],
                                               static_cast<size_t>(y0)*out_tensor->nb[1]);
                        auto * cp=copy_to(ph->ctx,tile,dst);
                        ph->graph=ggml_new_graph_custom(ph->ctx,8192,false);
                        ggml_build_forward_expand(ph->graph,cp);
                        continue;
                    }

                    const int rows=static_cast<int>(y1-y0);
                    const int spec_idx=dense_mps_spec_for(rp+".weight",final_w,rows,mps_cin,mps_cout);
                    auto & spec=plan->dense_mps_conv_specs.at(static_cast<size_t>(spec_idx));
                    auto & slot=plan->dense_mps_conv_slots.at(spec.slot);
                    {
                        auto * ph=make_phase(pfx+"."+std::to_string(decoder_last)+".mps.prefix."+
                                            std::to_string(y0)+"-"+std::to_string(y1),32u<<20);
                        DenseBuilder b(ph->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,false);
                        auto * padded=b.conv_stack_final_level_prepare_conv_tile(cfg,pfx,decoder_last,prev,y0,y1);
                        if (padded->ne[0]!=slot.input->ne[0] || padded->ne[1]!=slot.input->ne[1] || padded->ne[2]!=slot.input->ne[2])
                            throw std::runtime_error("MPSGraph final decoder prepared stripe shape mismatch: "+pfx);
                        auto * cp=copy_to(ph->ctx,padded,slot.input);
                        ph->graph=ggml_new_graph_custom(ph->ctx,8192,false);
                        ggml_build_forward_expand(ph->graph,cp);
                        ph->mps_conv_spec=spec_idx;
                    }
                    {
                        auto * ph=make_phase(pfx+"."+std::to_string(decoder_last)+".mps.suffix."+
                                            std::to_string(y0)+"-"+std::to_string(y1),32u<<20);
                        DenseBuilder b(ph->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,false);
                        auto * tile=b.conv_stack_final_level_finish_conv_tile(cfg,pfx,decoder_last,slot.output,feature,y0,y1);
                        auto * dst=ggml_view_4d(ph->ctx,out_tensor,final_w,y1-y0,cout,1,
                                               out_tensor->nb[1],out_tensor->nb[2],out_tensor->nb[3],
                                               static_cast<size_t>(y0)*out_tensor->nb[1]);
                        auto * cp=copy_to(ph->ctx,tile,dst);
                        ph->graph=ggml_new_graph_custom(ph->ctx,8192,false);
                        ggml_build_forward_expand(ph->graph,cp);
                    }
                }
            };

            // Build the last bilinear decoder level as one resize phase followed
            // by bounded row-tile convolution phases. The full 64-channel resize
            // is stored once in a shared persistent staging arena; all expensive
            // replicate-pad/im2col tensors are tile-local.
            auto build_tiled_final = [&](const ConvStackCfg & cfg, const std::string & pfx,
                                         ggml_tensor * prev, ggml_tensor * feature,
                                         ggml_tensor *& out_tensor, const char * out_name,
                                         ggml_type out_type) {
                if (!tiled_final) throw std::runtime_error("tiled final decoder unexpectedly disabled");
                if (!prev) throw std::runtime_error("tiled final decoder missing previous state");
                {
                    auto * ph=make_phase(pfx+"."+std::to_string(decoder_last)+".up",24u<<20);
                    DenseBuilder b(ph->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,false);
                    ggml_tensor * src=prev;
                    if (src->type!=GGML_TYPE_F32) src=ggml_cast(ph->ctx,src,GGML_TYPE_F32);
                    auto * up=b.upsample2_bilinear(src);
                    auto * cp=copy_to(ph->ctx,up,plan->decoder_highres_stage);
                    ph->graph=ggml_new_graph_custom(ph->ctx,8192,false);
                    ggml_build_forward_expand(ph->graph,cp);
                }
                const int64_t cout=cfg.dim_out[decoder_last]>=0 ? cfg.dim_out[decoder_last] : cfg.dim_res[decoder_last];
                out_tensor=ggml_new_tensor_4d(plan->state_ctx,out_type,final_w,final_h,cout,1);
                ggml_set_name(out_tensor,out_name); ggml_set_input(out_tensor); ggml_set_output(out_tensor);
                for (int64_t y0=0; y0<final_h; y0+=final_tile_rows) {
                    const int64_t y1=std::min<int64_t>(final_h,y0+final_tile_rows);
                    auto * ph=make_phase(pfx+"."+std::to_string(decoder_last)+".tile."+
                                        std::to_string(y0)+"-"+std::to_string(y1),24u<<20);
                    DenseBuilder b(ph->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,false);
                    auto * tile=b.conv_stack_final_level_tile(cfg,pfx,decoder_last,
                                                               plan->decoder_highres_stage,feature,y0,y1);
                    auto * dst=ggml_view_4d(ph->ctx,out_tensor,final_w,y1-y0,cout,1,
                                           out_tensor->nb[1],out_tensor->nb[2],out_tensor->nb[3],
                                           y0*out_tensor->nb[1]);
                    auto * cp=copy_to(ph->ctx,tile,dst);
                    ph->graph=ggml_new_graph_custom(ph->ctx,8192,false);
                    ggml_build_forward_expand(ph->graph,cp);
                }
            };

            ggml_tensor * enc_state=plan->encoder_accum[(nphases-1)&1u];

            // Materialize [encoder_out+2, baseW*baseH] once. This is both the
            // MoGe-3 sparse-conditioning tensor and the level-0 neck input, so
            // the expensive concat/permutation is not repeated in separate phases.
            {
                auto * ph=make_phase("encoder-input",24u<<20);
                ggml_tensor * enc=enc_state;
                if (enc->type!=GGML_TYPE_F32) enc=ggml_cast(ph->ctx,enc,GGML_TYPE_F32);
                DenseBuilder b(ph->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,false);
                auto * ef=b.build_encoder_feature(enc);
                plan->encoder_input=persist_like(ef,"dense.state.encoder_input",decoder_state_type);
                if (arch.version==3) plan->out.encoder_feature=plan->encoder_input;
                auto * cp=copy_to(ph->ctx,ef,plan->encoder_input);
                ph->graph=ggml_new_graph_custom(ph->ctx,8192,false);
                ggml_build_forward_expand(ph->graph,cp);
            }

            // Scale head is independent of all spatial decoder tensors.
            if (arch.scale_present) {
                auto * ph=make_phase("scale",8u<<20);
                DenseBuilder b(ph->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,false);
                auto * scale=b.build_scale_head(plan->final_cls);
                plan->out.raw_scale=persist_like(scale,"dense.out.scale",scale->type);
                auto * cp=copy_to(ph->ctx,scale,plan->out.raw_scale);
                ph->graph=ggml_new_graph_custom(ph->ctx,4096,false);
                ggml_build_forward_expand(ph->graph,cp);
            }

            // Neck: one graph per level except the final high-resolution
            // bilinear+3x3 stage, which is streamed by row tiles when supported.
            // This bounds im2col scratch while keeping the five neck features
            // persistent for the three output heads.
            for (size_t level=0; level<arch.neck.dim_res.size(); ++level) {
                ggml_tensor * prev = level ? plan->neck_transition[level-1] : nullptr;
                ggml_tensor * feature = level==0 ? nullptr : plan->uv[level];
                if (can_stream_cfg(arch.neck,level)) {
                    std::string name="dense.state.neck."+std::to_string(level);
                    plan->neck_features[level]=persist_level_shape(arch.neck,level,name,decoder_state_type);
                    build_streamed_level(arch.neck,"neck",level,prev,feature,plan->neck_features[level]);
                    // Segmented neck validation above guarantees intermediate
                    // dim_out=-1, so exported feature and transition are the
                    // same post-residual tensor exactly as in conv_stack_level.
                    plan->neck_transition[level]=plan->neck_features[level];
                    continue;
                }
                if (tiled_final && level==decoder_last) {
                    if (stream_final_requested) {
                        build_streamed_final(arch.neck,"neck",prev,feature,
                                             plan->neck_features[level],
                                             ("dense.state.neck."+std::to_string(level)).c_str(),
                                             decoder_state_type);
                    } else {
                        build_tiled_final(arch.neck,"neck",prev,feature,
                                          plan->neck_features[level],
                                          ("dense.state.neck."+std::to_string(level)).c_str(),
                                          decoder_state_type);
                    }
                    continue;
                }

                auto * ph=make_phase("neck."+std::to_string(level),48u<<20);
                DenseBuilder b(ph->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,false);
                if (prev && prev->type!=GGML_TYPE_F32) prev=ggml_cast(ph->ctx,prev,GGML_TYPE_F32);
                if (level==0) {
                    auto * cf=ggml_reshape_4d(ph->ctx,plan->encoder_input,
                                              arch.encoder_out+2,bw,bh,1);
                    feature=ggml_permute(ph->ctx,cf,2,0,1,3); // inverse of [W,H,C]->[C,W,H]
                    if (feature->type!=GGML_TYPE_F32) feature=ggml_cast(ph->ctx,feature,GGML_TYPE_F32);
                }
                const auto step=b.conv_stack_level(arch.neck,"neck",level,prev,feature);
                std::string name="dense.state.neck."+std::to_string(level);
                plan->neck_features[level]=persist_like(step.level_out,name.c_str(),decoder_state_type);
                ggml_tensor * root=copy_to(ph->ctx,step.level_out,plan->neck_features[level]);
                ph->graph=ggml_new_graph_custom(ph->ctx,16384,false);
                ggml_build_forward_expand(ph->graph,root);
                if (step.next) {
                    if (step.next == step.level_out) {
                        plan->neck_transition[level]=plan->neck_features[level];
                    } else {
                        std::string tn="dense.alias.neck_transition."+std::to_string(level);
                        plan->neck_transition[level]=persist_like(step.next,tn.c_str(),decoder_state_type);
                        const size_t parity=level&1u;
                        plan->decoder_alias_tensors[parity].push_back(plan->neck_transition[level]);
                        auto * cp_next=copy_to(ph->ctx,step.next,plan->neck_transition[level]);
                        ggml_build_forward_expand(ph->graph,cp_next);
                    }
                }
            }

            auto build_segmented_head = [&](const ConvStackCfg & cfg, const std::string & pfx,
                                            ggml_tensor *& final_out, const char * final_name) {
                if (!cfg.present) return;
                std::vector<ggml_tensor *> states(cfg.dim_res.size() > 0 ? cfg.dim_res.size()-1 : 0,nullptr);
                for (size_t level=0; level<cfg.dim_res.size(); ++level) {
                    ggml_tensor * prev = level ? states[level-1] : nullptr;
                    ggml_tensor * feature=plan->neck_features[level];
                    if (can_stream_cfg(cfg,level)) {
                        std::string n="dense.alias."+pfx+"."+std::to_string(level);
                        states[level]=persist_level_shape(cfg,level,n,decoder_state_type);
                        const size_t parity=level&1u;
                        plan->decoder_alias_tensors[parity].push_back(states[level]);
                        build_streamed_level(cfg,pfx,level,prev,feature,states[level]);
                        continue;
                    }
                    if (tiled_final && level==decoder_last) {
                        if (stream_final_requested)
                            build_streamed_final(cfg,pfx,prev,feature,final_out,final_name,GGML_TYPE_F32);
                        else
                            build_tiled_final(cfg,pfx,prev,feature,final_out,final_name,GGML_TYPE_F32);
                        continue;
                    }

                    auto * ph=make_phase(pfx+"."+std::to_string(level),48u<<20);
                    DenseBuilder b(ph->ctx,*weights,arch,iw,ih,bw,bh,plan->pos_embed,plan->uv,false);
                    if (prev && prev->type!=GGML_TYPE_F32) prev=ggml_cast(ph->ctx,prev,GGML_TYPE_F32);
                    if (feature && feature->type!=GGML_TYPE_F32) feature=ggml_cast(ph->ctx,feature,GGML_TYPE_F32);
                    const auto step=b.conv_stack_level(cfg,pfx,level,prev,feature);
                    ggml_tensor * root=nullptr;
                    if (level+1<cfg.dim_res.size()) {
                        std::string n="dense.alias."+pfx+"."+std::to_string(level);
                        states[level]=persist_like(step.next,n.c_str(),decoder_state_type);
                        const size_t parity=level&1u;
                        plan->decoder_alias_tensors[parity].push_back(states[level]);
                        root=copy_to(ph->ctx,step.next,states[level]);
                    } else {
                        final_out=persist_like(step.level_out,final_name,step.level_out->type);
                        root=copy_to(ph->ctx,step.level_out,final_out);
                    }
                    ph->graph=ggml_new_graph_custom(ph->ctx,16384,false);
                    ggml_build_forward_expand(ph->graph,root);
                }
            };

            build_segmented_head(arch.points,"points_head",plan->out.raw_points,"dense.out.points");
            build_segmented_head(arch.normal,"normal_head",plan->out.raw_normal,"dense.out.normal");
            build_segmented_head(arch.mask,"mask_head",plan->out.raw_mask,"dense.out.mask");

            // Explicit alias allocation for head cross-level state. Every tensor
            // in parity arena 0/1 is used by a different serial phase, so all may
            // safely share the same base address. This is the ggml equivalent of
            // compiler liveness reuse rather than allocating a pyramid per head.
            const auto buft=ggml_backend_get_default_buffer_type(primary);
            for (size_t parity=0; parity<2; ++parity) {
                size_t max_bytes=0;
                for (auto * t:plan->decoder_alias_tensors[parity])
                    max_bytes=std::max(max_bytes,ggml_backend_buft_get_alloc_size(buft,t));
                if (!max_bytes) continue;
                plan->decoder_alias_buffers[parity]=ggml_backend_buft_alloc_buffer(buft,max_bytes);
                if (!plan->decoder_alias_buffers[parity]) throw std::runtime_error("failed to allocate decoder alias arena");
                ggml_backend_buffer_set_usage(plan->decoder_alias_buffers[parity],GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                void * base=ggml_backend_buffer_get_base(plan->decoder_alias_buffers[parity]);
                for (auto * t:plan->decoder_alias_tensors[parity]) {
                    if (ggml_backend_tensor_alloc(plan->decoder_alias_buffers[parity],t,base)!=GGML_STATUS_SUCCESS)
                        throw std::runtime_error("failed to alias decoder state tensor");
                }
            }
            if (!plan->decoder_shared_stage_tensors.empty()) {
                size_t max_bytes=0;
                for (auto * t:plan->decoder_shared_stage_tensors)
                    max_bytes=std::max(max_bytes,ggml_backend_buft_get_alloc_size(buft,t));
                plan->decoder_highres_buffer=ggml_backend_buft_alloc_buffer(buft,max_bytes);
                if (!plan->decoder_highres_buffer) throw std::runtime_error("failed to allocate decoder shared staging arena");
                ggml_backend_buffer_set_usage(plan->decoder_highres_buffer,GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                void * base=ggml_backend_buffer_get_base(plan->decoder_highres_buffer);
                for (auto * t:plan->decoder_shared_stage_tensors) {
                    if (ggml_backend_tensor_alloc(plan->decoder_highres_buffer,t,base)!=GGML_STATUS_SUCCESS)
                        throw std::runtime_error("failed to bind decoder shared staging tensor");
                }
            }
            if (plan->dense_mpsgraph_final_conv || plan->dense_mpsgraph_neck3_entry || plan->dense_mpsgraph_neck3_residual0) {
                for (auto & slot : plan->dense_mps_conv_slots) {
                    const size_t ibytes=ggml_backend_buft_get_alloc_size(buft,slot.input);
                    const size_t obytes=ggml_backend_buft_get_alloc_size(buft,slot.output);
                    slot.input_buffer=ggml_backend_buft_alloc_buffer(buft,ibytes);
                    slot.output_buffer=ggml_backend_buft_alloc_buffer(buft,obytes);
                    if (!slot.input_buffer || !slot.output_buffer)
                        throw std::runtime_error("failed to allocate MPSGraph conv shared stripe buffers");
                    ggml_backend_buffer_set_usage(slot.input_buffer,GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                    ggml_backend_buffer_set_usage(slot.output_buffer,GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                    const char * ibn=ggml_backend_buffer_name(slot.input_buffer);
                    const char * obn=ggml_backend_buffer_name(slot.output_buffer);
                    void * ibase=ggml_backend_buffer_get_base(slot.input_buffer);
                    void * obase=ggml_backend_buffer_get_base(slot.output_buffer);
                    if (!ibn || !obn || std::string(ibn).find("_Private")!=std::string::npos ||
                        std::string(obn).find("_Private")!=std::string::npos || !ibase || !obase)
                        throw std::runtime_error("MPSGraph conv requires Metal shared buffers");
                    if (ggml_backend_tensor_alloc(slot.input_buffer,slot.input,ibase)!=GGML_STATUS_SUCCESS ||
                        ggml_backend_tensor_alloc(slot.output_buffer,slot.output,obase)!=GGML_STATUS_SUCCESS)
                        throw std::runtime_error("failed to bind MPSGraph conv stripe tensors");
                }
                for (auto & spec : plan->dense_mps_conv_specs) {
                    auto & slot=plan->dense_mps_conv_slots.at(spec.slot);
                    auto * wt=weights->get(spec.weight_name);
                    if (!wt || wt->type!=GGML_TYPE_F16 || wt->ne[0]!=3 || wt->ne[1]!=3 ||
                        wt->ne[2]!=slot.cin || wt->ne[3]!=slot.cout)
                        throw std::runtime_error("invalid MPSGraph conv weight: "+spec.weight_name);
                    std::vector<uint16_t> wh(static_cast<size_t>(ggml_nelements(wt)));
                    ggml_backend_tensor_get(wt,wh.data(),0,wh.size()*sizeof(uint16_t));
                    spec.runtime=runtime_mps_conv_create_bound(slot.width,slot.rows,slot.cin,slot.cout,
                                                               wh.data(),slot.input,slot.output);
                }
            }
            if (plan->dino_mpsgraph_direct) {
                const std::array<ggml_tensor *,4> mt = {plan->mps_q,plan->mps_k,plan->mps_v,plan->mps_attn16};
                for (size_t i=0;i<mt.size();++i) {
                    const size_t bytes=ggml_backend_buft_get_alloc_size(buft,mt[i]);
                    plan->mps_buffers[i]=ggml_backend_buft_alloc_buffer(buft,bytes);
                    if (!plan->mps_buffers[i]) throw std::runtime_error("failed to allocate MPSGraph direct shared buffer");
                    ggml_backend_buffer_set_usage(plan->mps_buffers[i],GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                    const char * bn=ggml_backend_buffer_name(plan->mps_buffers[i]);
                    void * base=ggml_backend_buffer_get_base(plan->mps_buffers[i]);
                    if (!bn || std::string(bn).find("_Private")!=std::string::npos || !base)
                        throw std::runtime_error("MPSGraph direct mode requires Metal shared buffers");
                    if (ggml_backend_tensor_alloc(plan->mps_buffers[i],mt[i],base)!=GGML_STATUS_SUCCESS)
                        throw std::runtime_error("failed to bind MPSGraph direct tensor");
                }
            }
            plan->state_buffer=ggml_backend_alloc_ctx_tensors(plan->state_ctx,primary);
            if(!plan->state_buffer) throw std::runtime_error("failed to allocate dense persistent GPU state");
            if (plan->dino_mpsgraph_f16) {
                const char * buf_name=ggml_backend_buffer_name(plan->state_buffer);
                if (!buf_name || std::string(buf_name).find("_Private")!=std::string::npos ||
                    !ggml_backend_buffer_get_base(plan->state_buffer) ||
                    !plan->mps_q->data || !plan->mps_k->data || !plan->mps_v->data || !plan->mps_attn16->data)
                    throw std::runtime_error("MPSGraph DINO requires CPU-addressable Metal shared state buffers");
                const int64_t hd=arch.embed_dim/arch.heads;
                const float scale=1.0f/std::sqrt(static_cast<float>(hd));
                if (plan->dino_mpsgraph_direct) {
#ifdef __APPLE__
                    std::array<void *,4> mb{};
                    const std::array<ggml_tensor *,4> mt = {plan->mps_q,plan->mps_k,plan->mps_v,plan->mps_attn16};
                    for (size_t i=0;i<mt.size();++i) {
                        size_t off=~size_t(0);
                        mb[i]=ggml_backend_metal_get_tensor_mtlbuffer(mt[i],&off);
                        if (!mb[i] || off!=0) throw std::runtime_error("MPSGraph direct tensor must begin at Metal buffer offset zero");
                    }
                    plan->mps_sdpa=runtime_mps_sdpa_create_bound(arch.heads,tokens,hd,scale,mb[0],mb[1],mb[2],mb[3]);
#else
                    throw std::runtime_error("MPSGraph direct mode is only available on Apple platforms");
#endif
                } else {
                    plan->mps_sdpa=runtime_mps_sdpa_create(arch.heads,tokens,hd,scale);
                }
            }
            ggml_backend_tensor_set(plan->pos_embed,pos_data.data(),0,pos_data.size()*sizeof(float));
            for (int level = 0; level < 5; ++level) {
                const int uw = bw << level, uh = bh << level;
                const std::vector<float> uv_data = make_uv_host(iw,ih,uw,uh);
                ggml_backend_tensor_set(plan->uv[static_cast<size_t>(level)],uv_data.data(),0,uv_data.size()*sizeof(float));
            }
            if (dense_trace()) {
                const double mib=static_cast<double>(ggml_backend_buffer_get_size(plan->state_buffer))/(1024.0*1024.0);
                const double alias0=plan->decoder_alias_buffers[0] ? static_cast<double>(ggml_backend_buffer_get_size(plan->decoder_alias_buffers[0]))/(1024.0*1024.0) : 0.0;
                const double alias1=plan->decoder_alias_buffers[1] ? static_cast<double>(ggml_backend_buffer_get_size(plan->decoder_alias_buffers[1]))/(1024.0*1024.0) : 0.0;
                const double highres=plan->decoder_highres_buffer ? static_cast<double>(ggml_backend_buffer_get_size(plan->decoder_highres_buffer))/(1024.0*1024.0) : 0.0;
                double mps_external=0.0;
                for (auto * b : plan->mps_buffers) if (b) mps_external += static_cast<double>(ggml_backend_buffer_get_size(b))/(1024.0*1024.0);
                double mps_conv_external=0.0;
                for (const auto & slot : plan->dense_mps_conv_slots) {
                    if (slot.input_buffer) mps_conv_external += static_cast<double>(ggml_backend_buffer_get_size(slot.input_buffer))/(1024.0*1024.0);
                    if (slot.output_buffer) mps_conv_external += static_cast<double>(ggml_backend_buffer_get_size(slot.output_buffer))/(1024.0*1024.0);
                }
                const double external=alias0+alias1+highres+mps_external+mps_conv_external;
                const double total=mib+external;
                const double model_mib=static_cast<double>(weights->resident_bytes())/(1024.0*1024.0);
                std::fprintf(stderr,"moge_dense_phase: segmented persistent=%.1f MiB phases=%zu state=%s tokens=%lld pingpong_hidden=2 encoder_accum=2 full_taps=0 decoder_alias=%.1f+%.1f MiB highres_stage=%.1f MiB stream_pre_residual=0.0 MiB mps_external=%.1f MiB mps_conv_external=%.1f MiB persistent_external=%.1f MiB persistent_total=%.1f MiB resident_model=%.1f MiB model_mapped=%d final_tile_rows=%d stream_final=%d stream_stage_free=%d stream_level1=%d stream_level2=%d stream_level3=%d stream_tile_rows=%d stream_level1_tile_rows=%d destructive_residuals=%d dino_mpsgraph_f16=%d dino_mpsgraph_direct=%d dino_mpsgraph_bridge=%d dense_mpsgraph_final_conv=%d dense_mpsgraph_neck3_entry=%d dense_mpsgraph_neck3_residual0=%d\n",
                             mib,plan->phases.size(),stype==GGML_TYPE_F16?"f16":"f32",static_cast<long long>(tokens),alias0,alias1,highres,mps_external,mps_conv_external,external,total,model_mib,weights->mapped_weights()?1:0,final_tile_rows,stream_final_requested?1:0,dense_stream_stage_free()?1:0,stream_level1_requested?1:0,stream_level2_requested?1:0,stream_level3_requested?1:0,stream_tile_rows,stream_level1_tile_rows,dense_destructive_residuals()?1:0,plan->dino_mpsgraph_f16?1:0,plan->dino_mpsgraph_direct?1:0,plan->dino_mpsgraph_bridge?1:0,plan->dense_mpsgraph_final_conv?1:0,plan->dense_mpsgraph_neck3_entry?1:0,plan->dense_mpsgraph_neck3_residual0?1:0);
            }
        }
        dense_plan=std::move(plan);
        return *dense_plan;
    }

    void compute_dense(DensePlan & plan) {
        if (!dense_sched) dense_sched = make_dense_scheduler();
        if (!plan.segmented) {
            const auto started = std::chrono::steady_clock::now();
            profiling::OpProfileState op_profile{"dense.monolithic"};
            if (profiling::op_profile_enabled()) ggml_backend_sched_set_eval_callback(dense_sched, profiling::op_profile_callback, &op_profile);
            profiling::device_memory_sample(primary, "dense.monolithic.before_compute");
            auto st=ggml_backend_sched_graph_compute(dense_sched,plan.graph);
            if (profiling::op_profile_enabled()) ggml_backend_sched_set_eval_callback(dense_sched, nullptr, nullptr);
            profiling::device_memory_sample(primary, "dense.monolithic.after_compute");
            if(st!=GGML_STATUS_SUCCESS) throw std::runtime_error("dense graph compute failed: "+std::string(ggml_status_to_string(st)));
            ggml_backend_sched_synchronize(dense_sched);
            if (dense_trace()) {
                const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
                const double mib=static_cast<double>(ggml_backend_sched_get_buffer_size(dense_sched,primary))/(1024.0*1024.0);
                std::fprintf(stderr,"moge_dense_phase: done monolithic %.3f ms scratch=%.1f MiB\n",ms,mib);
            }
            return;
        }

        const bool vulkan_primary = dev_matches(ggml_backend_get_device(primary), Backend::Vulkan);
        const bool async_dense = vulkan_primary && !calibration_active;
        const bool phase_local = async_dense ? false : dense_phase_local_scheduler();
        for (const auto & holder : plan.phases) {
            auto & ph=*holder;
            ggml_backend_sched_t owned=nullptr;
            ggml_backend_sched_t active=dense_sched;
            if (phase_local) {
                owned=make_phase_scheduler(ph.label=="heads" ? 32768 : 16384);
                active=owned;
            } else {
                ggml_backend_sched_reset(active);
            }
            const auto started=std::chrono::steady_clock::now();
            if (dense_trace()) std::fprintf(stderr,"moge_dense_phase: start %s phase_local=%d\n",ph.label.c_str(),phase_local?1:0);
            const size_t restored_before = restore_phase_graph_sources(ph);
            if (dense_trace() && restored_before) {
                std::fprintf(stderr,"moge_dense_phase: restored %zu stale scheduler source edges before %s\n",
                             restored_before,ph.label.c_str());
            }
            if (calibration_active) discover_calibration_nodes(ph);
            if(!ggml_backend_sched_alloc_graph(active,ph.graph)) {
                restore_phase_graph_sources(ph);
                detach_phase_backend_bindings(ph.ctx);
                if(owned) ggml_backend_sched_free(owned);
                else ggml_backend_sched_reset(active);
                throw std::runtime_error("failed to allocate dense phase "+ph.label);
            }
            const size_t refreshed_views = refresh_graph_views(ph.graph);
            if (dense_trace() && refreshed_views) {
                std::fprintf(stderr,"moge_dense_phase: refreshed %zu nested view buffers in %s\n",
                             refreshed_views,ph.label.c_str());
            }
            if (dense_trace()) {
                const double mib=static_cast<double>(ggml_backend_sched_get_buffer_size(active,primary))/(1024.0*1024.0);
                std::fprintf(stderr,"moge_dense_phase: allocated %s scratch=%.1f MiB\n",ph.label.c_str(),mib);
            }
            CalibrationEvalState calibration_eval;
            profiling::OpProfileState op_profile{ph.label.c_str()};
            const bool profile_ops = profiling::op_profile_enabled() && !(calibration_active && !ph.calibration_nodes.empty());
            if (calibration_active && !ph.calibration_nodes.empty()) {
                calibration_eval.self = this;
                calibration_eval.phase = &ph;
                ggml_backend_sched_set_eval_callback(active, calibration_eval_callback, &calibration_eval);
            } else if (profile_ops) {
                ggml_backend_sched_set_eval_callback(active, profiling::op_profile_callback, &op_profile);
            }
            const std::string memory_tag = "dense." + ph.label + ".allocated";
            profiling::device_memory_sample(primary, memory_tag.c_str());
            const auto st=ggml_backend_sched_graph_compute(active,ph.graph);
            if ((calibration_active && !ph.calibration_nodes.empty()) || profile_ops) {
                ggml_backend_sched_set_eval_callback(active, nullptr, nullptr);
            }
            if(st!=GGML_STATUS_SUCCESS || calibration_eval.error) {
                ggml_backend_sched_synchronize(active);
                restore_phase_graph_sources(ph);
                detach_phase_backend_bindings(ph.ctx);
                if(owned) ggml_backend_sched_free(owned);
                else ggml_backend_sched_reset(active);
                if (calibration_eval.error) std::rethrow_exception(calibration_eval.error);
                throw std::runtime_error("dense phase compute failed ("+ph.label+"): "+std::string(ggml_status_to_string(st)));
            }
            if (!async_dense) ggml_backend_sched_synchronize(active);
            if (dense_trace()) {
                const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
                std::fprintf(stderr,"moge_dense_phase: done %s %.3f ms\n",ph.label.c_str(),ms);
            }
            // split_graph() can rewrite node->src[] to scheduler-owned backend
            // copy tensors. Restore the original graph topology *before* the
            // scheduler context is reset/freed, otherwise the cached graph keeps
            // dangling source pointers into scheduler metadata. Metal graph
            // optimization may reorder nodes, so restoration is keyed by node
            // identity rather than graph index.
            const size_t restored_after = restore_phase_graph_sources(ph);
            if (dense_trace() && restored_after) {
                std::fprintf(stderr,"moge_dense_phase: restored %zu scheduler-rewritten source edges in %s\n",
                             restored_after,ph.label.c_str());
            }
            // sched_alloc_graph() also writes scheduler-owned buffer/data pointers
            // into graph tensor metadata. Detach those while the allocation is
            // still valid, before either phase-local free or shared-scheduler reset.
            const size_t detached=detach_phase_backend_bindings(ph.ctx);
            if (dense_trace()) std::fprintf(stderr,"moge_dense_phase: detached %zu scheduler bindings from %s\n",detached,ph.label.c_str());
            if(owned) ggml_backend_sched_free(owned);
            if (ph.mps_conv_spec >= 0) {
                const size_t idx=static_cast<size_t>(ph.mps_conv_spec);
                if (idx>=plan.dense_mps_conv_specs.size() || !plan.dense_mps_conv_specs[idx].runtime)
                    throw std::runtime_error("MPSGraph conv phase has no runtime context");
                const auto mps_started=std::chrono::steady_clock::now();
                const double mps_ms=runtime_mps_conv_run_bound(plan.dense_mps_conv_specs[idx].runtime);
                if (dense_trace()) {
                    const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-mps_started).count();
                    std::fprintf(stderr,"moge_dense_phase: mpsgraph conv %.3f ms wall=%.3f ms after %s\n",mps_ms,wall,ph.label.c_str());
                }
            }
            if (ph.run_mps_after) {
                if (!plan.mps_sdpa) throw std::runtime_error("MPSGraph DINO phase has no SDPA context");
                const auto mps_started=std::chrono::steady_clock::now();
                const double mps_ms=plan.dino_mpsgraph_direct
                    ? runtime_mps_sdpa_run_bound(plan.mps_sdpa)
                    : runtime_mps_sdpa_run_f16(
                        plan.mps_sdpa,
                        static_cast<const uint16_t *>(plan.mps_q->data),
                        static_cast<const uint16_t *>(plan.mps_k->data),
                        static_cast<const uint16_t *>(plan.mps_v->data),
                        static_cast<uint16_t *>(plan.mps_attn16->data));
                if (dense_trace()) {
                    const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-mps_started).count();
                    std::fprintf(stderr,"moge_dense_phase: mpsgraph sdpa %.3f ms wall=%.3f ms after %s\n",mps_ms,wall,ph.label.c_str());
                }
            }
        }
        // Vulkan queues preserve submission order, so the shared-scheduler async
        // path can reuse scratch across phase submissions without a CPU wait per
        // phase. Synchronize once before any CPU-visible dump/reset/free.
        if (async_dense) ggml_backend_sched_synchronize(dense_sched);
        // Every phase graph has had scheduler-owned scratch bindings detached
        // before this reset. Durable external-rooted views remain bound to their
        // persistent/model buffers so the next graph split can recover backend
        // placement without seeing stale gallocr pointers.
        ggml_backend_sched_reset(dense_sched);

        // MoGe-3 immediately enters a much larger sparse working set. Free the
        // dense scheduler arena after its persistent outputs have been written,
        // rather than keeping the largest DINO/head scratch allocation resident
        // throughout all sparse refinement steps. Graph metadata and weights stay
        // cached; a fresh scheduler is created lazily for the next inference.
        if (sparse_cfg.present && primary != cpu) {
            if (dense_trace()) std::fprintf(stderr,"moge_dense_phase: release scheduler scratch\n");
            ggml_backend_sched_free(dense_sched);
            dense_sched=nullptr;
        }
    }

    static void validate_image_request(int iw, int ih, int num_tokens, int resolution_level) {
        constexpr int kMaxInputSide = 32768;
        constexpr uint64_t kMaxInputPixels = 64ull * 1024ull * 1024ull;
        constexpr int kMaxTokens = 65536;
        if (iw <= 0 || ih <= 0) throw std::invalid_argument("invalid image dimensions");
        if (iw > kMaxInputSide || ih > kMaxInputSide) throw std::invalid_argument("image dimensions exceed library safety limit");
        const uint64_t pixels = static_cast<uint64_t>(iw) * static_cast<uint64_t>(ih);
        if (pixels > kMaxInputPixels) throw std::invalid_argument("image pixel count exceeds library safety limit");
        if (resolution_level < 0 || resolution_level > 9) throw std::invalid_argument("resolution_level must be in 0..9");
        if (num_tokens < 0 || num_tokens > kMaxTokens) throw std::invalid_argument("num_tokens exceeds library safety limit");
    }

    std::pair<int,int> token_grid(int iw, int ih, int num_tokens, int resolution_level) const {
        int nt=num_tokens;
        if(nt<=0) {
            if(arch.token_range.size()!=2) throw std::runtime_error("bad token range metadata");
            const float level=std::clamp(resolution_level,0,9)/9.0f;
            nt=static_cast<int>(arch.token_range[0]+level*(arch.token_range[1]-arch.token_range[0]));
        }
        const double ar=static_cast<double>(iw)/ih;
        const int bh=std::max(1,static_cast<int>(std::nearbyint(std::sqrt(nt/ar))));
        const int bw=std::max(1,static_cast<int>(std::nearbyint(std::sqrt(nt*ar))));
        return {bw,bh};
    }

    void upload_dense_input(DensePlan & plan, const float * rgb, int iw, int ih) {
        resize_rgb_to_planar_aa(rgb,iw,ih,plan.encoder_w,plan.encoder_h,
                                plan.resize_x,plan.resize_y,plan.resize_tmp,plan.image_upload);
        normalize_planar_rgb(plan.image_upload,image_mean,image_std);
        ggml_backend_tensor_set(plan.image,plan.image_upload.data(),0,plan.image_upload.size()*sizeof(float));
    }

    void calibration_reset() {
        calibration_sums.clear();
        calibration_counts.clear();
        calibration_images=0;
        calibration_reducer.reset();
        calibration_verify_first = false;
    }

    bool calibration_verification_performed() const {
        return calibration_reducer && calibration_reducer->verified_first();
    }

    void calibration_add(const float * rgb,int iw,int ih,const CalibrationOptions & co) {
        if(!rgb) throw std::invalid_argument("invalid calibration image");
        validate_image_request(iw, ih, co.num_tokens, co.resolution_level);
        const auto [bw,bh]=token_grid(iw,ih,co.num_tokens,co.resolution_level);
        DensePlan & plan=get_dense_plan(iw,ih,bw,bh);
        upload_dense_input(plan,rgb,iw,ih);
        calibration_verify_first = co.verify_first;
        calibration_active=true;
        try {
            compute_dense(plan);
            calibration_active=false;
            ++calibration_images;
        } catch (...) {
            calibration_active=false;
            throw;
        }
    }

    void calibration_write_mogi(const std::string & path) const {
        if (calibration_sums.empty() || calibration_images == 0)
            throw std::runtime_error("no native calibration samples collected");
        std::vector<std::string> names;
        names.reserve(calibration_sums.size());
        for (const auto & kv : calibration_sums) names.push_back(kv.first);
        std::sort(names.begin(),names.end());
        std::ofstream f(path,std::ios::binary);
        if (!f) throw std::runtime_error("cannot open calibration output: "+path);
        const char magic[8]={'M','O','G','I','\0','\0','\0','\1'};
        f.write(magic,8);
        const uint32_t version=1, nrec=static_cast<uint32_t>(names.size());
        f.write(reinterpret_cast<const char *>(&version),sizeof(version));
        f.write(reinterpret_cast<const char *>(&nrec),sizeof(nrec));
        for (const auto & name : names) {
            if (name.size()>UINT16_MAX) throw std::runtime_error("calibration tensor name too long");
            const auto & sum=calibration_sums.at(name);
            const auto itc=calibration_counts.find(name);
            if (itc==calibration_counts.end() || itc->second==0) throw std::runtime_error("zero calibration count for "+name);
            const uint16_t nl=static_cast<uint16_t>(name.size());
            const uint32_t n=static_cast<uint32_t>(sum.size());
            f.write(reinterpret_cast<const char *>(&nl),sizeof(nl));
            f.write(name.data(),static_cast<std::streamsize>(name.size()));
            f.write(reinterpret_cast<const char *>(&n),sizeof(n));
            const double den=static_cast<double>(itc->second);
            double mean=0.0;
            for (double s : sum) mean += s/den;
            mean /= std::max<size_t>(size_t{1}, sum.size());
            const float floor=static_cast<float>(std::max(mean*1e-8, 1e-20));
            for (double s : sum) {
                const float v=std::max(static_cast<float>(s/den), floor);
                f.write(reinterpret_cast<const char *>(&v),sizeof(v));
            }
        }
        if (!f) throw std::runtime_error("failed writing calibration output: "+path);
    }

    Result infer(const float * rgb,int iw,int ih,const InferOptions & io) {
        if(!rgb) throw std::invalid_argument("invalid image");
        validate_image_request(iw, ih, io.num_tokens, io.resolution_level);
        if (io.refine_steps < 0 || io.refine_steps > 64) throw std::invalid_argument("refine_steps exceeds library safety limit");
        if (!std::isfinite(io.fov_x_degrees) || (io.fov_x_degrees > 0.0f && !(io.fov_x_degrees >= 1.0f && io.fov_x_degrees < 179.0f)))
            throw std::invalid_argument("fov_x_degrees must be <=0 (auto) or in [1,179)");
        const auto [bw,bh]=token_grid(iw,ih,io.num_tokens,io.resolution_level);

        DensePlan & plan=get_dense_plan(iw,ih,bw,bh);
        DenseOutputs & o=plan.out;

        // Upstream MoGe resizes RGB to the exact DINO token raster before
        // mean/std normalization and patch embedding. Keep this off-graph so
        // all backends see identical PyTorch-compatible antialiased pixels.
        upload_dense_input(plan,rgb,iw,ih);
        compute_dense(plan);

        if(!o.raw_points) throw std::runtime_error("model has no points head");
        const int pw=static_cast<int>(o.raw_points->ne[0]), ph=static_cast<int>(o.raw_points->ne[1]);
        std::vector<float> points=whc_to_hwc(read_f32(o.raw_points),pw,ph,3);
        std::vector<float> normal,mask;
        if(o.raw_normal) normal=whc_to_hwc(read_f32(o.raw_normal),static_cast<int>(o.raw_normal->ne[0]),static_cast<int>(o.raw_normal->ne[1]),3);
        if(o.raw_mask) mask=whc_to_hwc(read_f32(o.raw_mask),static_cast<int>(o.raw_mask->ne[0]),static_cast<int>(o.raw_mask->ne[1]),1);
        float metric=1.0f;
        if(o.raw_scale) { auto s=read_f32(o.raw_scale); if(!s.empty()) metric=std::exp(s[0]); }

        if(arch.version==3) {
            if(!refiner || !sparse_sched) throw std::runtime_error("MoGe-3 metadata has no refiner scheduler");
            if(!o.encoder_feature) throw std::runtime_error("MoGe-3 conditioning tensor missing");
            const int ew=plan.base_w, eh=plan.base_h;
            for (int step = 0; step < std::max(0, io.refine_steps); ++step) {
                refiner->refine_once(sparse_sched, points, pw, ph, o.encoder_feature, ew, eh);
                // The refiner executes one U-Net level per graph and resets
                // scheduler scratch after every phase. Retain the scheduler here:
                // its gallocr buffer is now bounded by the largest *single level*
                // rather than the whole sparse U-Net, so recycling the scheduler
                // between refinement iterations would only add allocator/backend
                // churn without reducing peak residency further.
            }
            points=resize_hwc_bilinear(points,pw,ph,3,iw,ih);
        } else if(pw!=iw||ph!=ih) {
            points=resize_hwc_bilinear(points,pw,ph,3,iw,ih);
        }
        const size_t npx = static_cast<size_t>(iw) * ih;
        if(!normal.empty() && normal.size()!=npx*3) normal=resize_hwc_bilinear(normal,static_cast<int>(o.raw_normal->ne[0]),static_cast<int>(o.raw_normal->ne[1]),3,iw,ih);
        if(!mask.empty() && mask.size()!=npx) mask=resize_hwc_bilinear(mask,static_cast<int>(o.raw_mask->ne[0]),static_cast<int>(o.raw_mask->ne[1]),1,iw,ih);

        std::vector<float> captured_points;
        std::vector<float> captured_mask;
        if (io.capture_raw_forward) {
            // Match upstream MoGeModel.forward(): output resize has already
            // happened above; points are then remapped and mask logits sigmoid
            // transformed before infer() performs focal/shift recovery.
            captured_points=points;
            remap_points_inplace(captured_points,arch.remap);
            if (!mask.empty()) captured_mask=mask_probabilities(mask,true);
        }

        Result result=postprocess(std::move(points),std::move(normal),std::move(mask),iw,ih,arch.remap,metric,io,true);
        if (io.capture_raw_forward) {
            result.raw_affine_points=std::move(captured_points);
            result.raw_mask_probability=std::move(captured_mask);
            result.raw_metric_scale=metric;
        }
        return result;
    }
};

Model::Model(const std::string & path,const LoadOptions & o):impl_(std::make_unique<Impl>(path,o)) {}
Model::Model(const void * data,size_t size,const LoadOptions & o):impl_(std::make_unique<Impl>(data,size,true,o)) {}
Model::Model(BorrowedMemoryTag,const void * data,size_t size,const LoadOptions & o):impl_(std::make_unique<Impl>(data,size,false,o)) {}
Model Model::from_borrowed_memory(const void * data,size_t size,const LoadOptions & o) {
    return Model(BorrowedMemoryTag{}, data, size, o);
}
Model::~Model()=default;
Model::Model(Model&&) noexcept=default;
Model& Model::operator=(Model&&) noexcept=default;
Result Model::infer(const float * rgb,int w,int h,const InferOptions & o){return impl_->infer(rgb,w,h,o);}
void Model::calibration_reset(){impl_->calibration_reset();}
void Model::calibration_add(const float * rgb,int w,int h,const CalibrationOptions & o){impl_->calibration_add(rgb,w,h,o);}
void Model::calibration_write_mogi(const std::string & path) const{impl_->calibration_write_mogi(path);}
size_t Model::calibration_record_count() const{return impl_->calibration_sums.size();}
uint64_t Model::calibration_image_count() const{return impl_->calibration_images;}
bool Model::calibration_verification_performed() const{return impl_->calibration_verification_performed();}
int Model::version() const{return impl_->arch.version;}
Backend Model::backend() const{return impl_->selected_backend;}
std::string Model::backend_name() const{return impl_->backend_desc;}
const char * version_string() noexcept{return VERSION_STRING;}

} // namespace moge
