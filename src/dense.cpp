// Copyright (c) 2026 Kenji8965
// SPDX-License-Identifier: MIT
// Programming and implementation assistance: ChatGPT 5.6 Sol (OpenAI)
// Human testing, validation, and release acceptance: Kenji8965
// Provided "AS IS"; use at your own risk. See LICENSE.

#include "dense.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace moge {

static bool backend_is_metal(ggml_backend_t backend) {
    const char * raw = backend ? ggml_backend_name(backend) : nullptr;
    std::string x = raw ? raw : "";
    std::transform(x.begin(), x.end(), x.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return x.find("metal") != std::string::npos || x.find("mtl") != std::string::npos;
}

// DINO Flash-Attention policy for pinned ggml 0.25.3 Metal.
// Metal FLASH_ATTN_EXT requires F32 Q (hard backend contract), while K/V may
// remain F16. The Metal implementation does not consume the FA accumulation
// precision op parameter, so keep the native operation precision here.

namespace {

ggml_type to_ggml_type(MoggType t) {
    switch (t) {
        case MoggType::F32: return GGML_TYPE_F32;
        case MoggType::F16: return GGML_TYPE_F16;
        case MoggType::BF16: return GGML_TYPE_BF16;
        case MoggType::I32: return GGML_TYPE_I32;
        case MoggType::Q4_K: return GGML_TYPE_Q4_K;
        case MoggType::Q6_K: return GGML_TYPE_Q6_K;
        case MoggType::Q8_0: return GGML_TYPE_Q8_0;
        case MoggType::IQ4_XS: return GGML_TYPE_IQ4_XS;
    }
    throw std::runtime_error("unsupported MOGG tensor type");
}

std::string idx(const std::string & p, int i) {
    return p + "." + std::to_string(i);
}

ggml_tensor * cast_like(ggml_context * c, ggml_tensor * t, const ggml_tensor * like) {
    if (!t || t->type == like->type) return t;
    if ((t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16) &&
        (like->type == GGML_TYPE_F32 || like->type == GGML_TYPE_F16 || like->type == GGML_TYPE_BF16)) {
        return ggml_cast(c, t, like->type);
    }
    throw std::runtime_error("cannot cast affine tensor to activation type");
}


// ggml 0.25.3's CPU binary-op fallback requires src0 to have a unit element
// stride (nb[0] == sizeof(type)). Some views/permutations are legal ggml
// tensors but violate that requirement when an ADD/MUL is scheduled to CPU.
// Materialize only those first operands; contiguous tensors stay zero-copy.
ggml_tensor * binary_src0(ggml_context * c, ggml_tensor * x) {
    return x->nb[0] == ggml_type_size(x->type) ? x : ggml_cont(c, x);
}

ggml_tensor * add_safe(ggml_context * c, ggml_tensor * a, ggml_tensor * b) {
    return ggml_add(c, binary_src0(c, a), b);
}

ggml_tensor * mul_safe(ggml_context * c, ggml_tensor * a, ggml_tensor * b) {
    return ggml_mul(c, binary_src0(c, a), b);
}

ggml_tensor * add_inplace_safe(ggml_context * c, ggml_tensor * a, ggml_tensor * b) {
    if (a->nb[0] != ggml_type_size(a->type)) return add_safe(c, a, b);
    return ggml_add_inplace(c, a, b);
}

ggml_tensor * mul_inplace_safe(ggml_context * c, ggml_tensor * a, ggml_tensor * b) {
    if (a->nb[0] != ggml_type_size(a->type)) return mul_safe(c, a, b);
    return ggml_mul_inplace(c, a, b);
}

ggml_tensor * add_bias(ggml_context * c, ggml_tensor * x, ggml_tensor * b) {
    if (!b) return x;
    return add_inplace_safe(c, x, cast_like(c, b, x));
}

// Conv-stack tensors use [W,H,C,N].  PyTorch normalization's channel affine
// vector therefore needs to be viewed as [1,1,C,1] before broadcasting.
ggml_tensor * channel_affine(ggml_context * c, ggml_tensor * x,
                             ggml_tensor * w, ggml_tensor * b) {
    if (w) {
        auto * wr = ggml_reshape_4d(c, w, 1, 1, w->ne[0], 1);
        x = mul_inplace_safe(c, x, cast_like(c, wr, x));
    }
    if (b) {
        auto * br = ggml_reshape_4d(c, b, 1, 1, b->ne[0], 1);
        x = add_inplace_safe(c, x, cast_like(c, br, x));
    }
    return x;
}

ggml_tensor * replicate_pad2d(ggml_context * c, ggml_tensor * x, int p) {
    if (p <= 0) return x;
    const int64_t W=x->ne[0], H=x->ne[1], C=x->ne[2], N=x->ne[3];
    if (W <= 0 || H <= 0) throw std::runtime_error("replicate_pad2d empty input");

    // Do not construct replicate padding with a chain of SET_INPLACE nodes.
    // Each in-place SET is represented by ggml as a view of its destination;
    // eight border writes at pad=1 therefore form an eight-deep nested view
    // chain. Metal 0.25.3's fusion/buffer lookup can leave an intermediate view
    // unresolved and crash with `tensor '(view)...' buffer is nil`.
    //
    // Instead build the exact padded tensor with four CONCATs. Edge strips are
    // single views of the source (or repeated views when p>1); concat outputs are
    // ordinary phase-local tensors, so view ancestry stays depth 1 and allocator
    // lifetime is naturally bounded by the current segmented phase.
    auto * left  = ggml_view_4d(c,x,1,H,C,N,x->nb[1],x->nb[2],x->nb[3],0);
    auto * right = ggml_view_4d(c,x,1,H,C,N,x->nb[1],x->nb[2],x->nb[3],(W-1)*x->nb[0]);
    if (p > 1) {
        left  = ggml_repeat_4d(c,left, p,H,C,N);
        right = ggml_repeat_4d(c,right,p,H,C,N);
    }
    auto * horiz = ggml_concat(c,left,x,0);
    horiz = ggml_concat(c,horiz,right,0); // [W+2p,H,C,N]

    const int64_t WP=W+2*p;
    auto * top    = ggml_view_4d(c,horiz,WP,1,C,N,horiz->nb[1],horiz->nb[2],horiz->nb[3],0);
    auto * bottom = ggml_view_4d(c,horiz,WP,1,C,N,horiz->nb[1],horiz->nb[2],horiz->nb[3],(H-1)*horiz->nb[1]);
    if (p > 1) {
        top    = ggml_repeat_4d(c,top,WP,p,C,N);
        bottom = ggml_repeat_4d(c,bottom,WP,p,C,N);
    }
    auto * out = ggml_concat(c,top,horiz,1);
    return ggml_concat(c,out,bottom,1); // [W+2p,H+2p,C,N]
}

// Replicate-pad only the rows needed for one output tile. `y0:y1` is in
// unpadded output coordinates. Interior tiles borrow their 1-row halo directly
// from the persistent full-height source; only the true image top/bottom are
// replicated. This avoids materializing full-frame padding for high-resolution
// decoder convolutions.
ggml_tensor * replicate_pad2d_rows(ggml_context * c, ggml_tensor * x,
                                    int64_t y0, int64_t y1) {
    const int64_t W=x->ne[0], H=x->ne[1], C=x->ne[2], N=x->ne[3];
    if (y0 < 0 || y1 <= y0 || y1 > H) throw std::runtime_error("replicate_pad2d_rows range");
    const int64_t src_y0=std::max<int64_t>(0,y0-1);
    const int64_t src_y1=std::min<int64_t>(H,y1+1);
    const int64_t src_h=src_y1-src_y0;
    auto * crop=ggml_view_4d(c,x,W,src_h,C,N,x->nb[1],x->nb[2],x->nb[3],src_y0*x->nb[1]);

    // Horizontal p=1 replication. Keep edge views shallow and let concat own
    // the bounded tile-local storage.
    auto * left =ggml_view_4d(c,crop,1,src_h,C,N,crop->nb[1],crop->nb[2],crop->nb[3],0);
    auto * right=ggml_view_4d(c,crop,1,src_h,C,N,crop->nb[1],crop->nb[2],crop->nb[3],(W-1)*crop->nb[0]);
    auto * horiz=ggml_concat(c,left,crop,0);
    horiz=ggml_concat(c,horiz,right,0); // [W+2,src_h,C,N]

    // Interior tiles already contain both vertical halo rows. Only add a row
    // at the actual image boundary where the halo falls outside the tensor.
    const int64_t WP=W+2;
    ggml_tensor * out=horiz;
    if (y0==0) {
        auto * top=ggml_view_4d(c,out,WP,1,C,N,out->nb[1],out->nb[2],out->nb[3],0);
        out=ggml_concat(c,top,out,1);
    }
    if (y1==H) {
        auto * bottom=ggml_view_4d(c,out,WP,1,C,N,out->nb[1],out->nb[2],out->nb[3],
                                  (out->ne[1]-1)*out->nb[1]);
        out=ggml_concat(c,out,bottom,1);
    }
    if (out->ne[1] != (y1-y0)+2) throw std::runtime_error("replicate_pad2d_rows halo shape");
    return out;
}

ConvStackCfg read_stack(const MoggFile & f, const std::string & p) {
    ConvStackCfg c;
    c.present = f.meta_or<bool>(p + ".present", false);
    if (!c.present) return c;
    c.dim_in = f.meta<std::vector<int64_t>>(p + ".dim_in");
    c.dim_res = f.meta<std::vector<int64_t>>(p + ".dim_res_blocks");
    c.dim_out = f.meta<std::vector<int64_t>>(p + ".dim_out");
    c.resamplers = f.meta<std::vector<std::string>>(p + ".resamplers");
    c.hidden_mul = static_cast<int>(f.meta<int64_t>(p + ".dim_times_res_block_hidden"));
    c.num_res_blocks = f.meta<std::vector<int64_t>>(p + ".num_res_blocks");
    c.in_norm = f.meta<std::string>(p + ".res_block_in_norm");
    c.hidden_norm = f.meta<std::string>(p + ".res_block_hidden_norm");
    c.activation = f.meta<std::string>(p + ".activation");
    return c;
}

} // namespace

WeightStore::WeightStore(const MoggFile & file, ggml_backend_t backend) {
    rotation_group_ = static_cast<int>(file.meta_or<int64_t>("format.rotation.group", 0));
    // The context only holds tensor metadata; payloads either live in a normal
    // backend-owned weight buffer or directly in the mmap-backed MOGG file.
    ggml_init_params p{};
    p.mem_size = std::max<size_t>(16u << 20, (file.tensors().size() + 64) * 1024u);
    p.mem_buffer = nullptr;
    p.no_alloc = true;
    ctx_ = ggml_init(p);
    if (!ctx_) throw std::runtime_error("ggml_init failed for weight context");

    size_t max_tensor_size = 0;
    for (const auto & ti : file.tensors()) {
        int64_t ne[4] = {ti.ne[0], ti.ne[1], ti.ne[2], ti.ne[3]};
        auto * t = ggml_new_tensor(ctx_, to_ggml_type(ti.type), ti.ndim, ne);
        ggml_set_name(t, ti.name.c_str());
        tensors_.emplace(ti.name, t);
        flags_.emplace(ti.name, ti.flags);
        max_tensor_size = std::max(max_tensor_size, static_cast<size_t>(std::max<uint64_t>(ti.size, ggml_nbytes(t))));
    }

    auto * dev = ggml_backend_get_device(backend);
    ggml_backend_dev_props props{};
    if (dev) ggml_backend_dev_get_props(dev, &props);
    const bool can_map = file.is_mapped() && dev && props.caps.buffer_from_host_ptr && props.caps.mmap_support;
    const bool want_map = backend_is_metal(backend) && can_map;
    if (want_map) {
        buffer_ = ggml_backend_dev_buffer_from_host_ptr(dev, file.mutable_data(), file.size(), max_tensor_size);
        if (buffer_) {
            ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer_));
            if (base != file.mutable_data())
                throw std::runtime_error("mapped weight buffer base does not match MOGG mapping");
            for (const auto & ti : file.tensors()) {
                auto * t = tensors_.at(ti.name);
                const size_t expected = ggml_nbytes(t);
                if (expected != ti.size) {
                    std::ostringstream os;
                    os << "tensor byte-size mismatch for " << ti.name << ": file=" << ti.size
                       << " ggml=" << expected;
                    throw std::runtime_error(os.str());
                }
                if (ggml_backend_tensor_alloc(buffer_, t, base + ti.offset) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("failed to bind mapped weight tensor: " + ti.name);
            }
            mapped_weights_ = true;
            return;
        }
    }

    buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
    if (!buffer_) throw std::runtime_error("failed to allocate ggml weight buffer");
    ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (const auto & ti : file.tensors()) {
        auto * t = tensors_.at(ti.name);
        const bool shadow = t->type == GGML_TYPE_F16 && ti.type == MoggType::Q8_0;
        if (shadow) {
            const int64_t n = ggml_nelements(t);
            const auto * traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
            if (!traits || !traits->to_float) throw std::runtime_error("Q8_0 dequantizer unavailable");
            std::vector<float> f32(static_cast<size_t>(n));
            std::vector<ggml_fp16_t> f16(static_cast<size_t>(n));
            traits->to_float(file.tensor_data(ti), f32.data(), n);
            ggml_fp32_to_fp16_row(f32.data(), f16.data(), n);
            ggml_backend_tensor_set(t, f16.data(), 0, f16.size() * sizeof(ggml_fp16_t));
            continue;
        }
        const size_t expected = ggml_nbytes(t);
        if (expected != ti.size) {
            std::ostringstream os;
            os << "tensor byte-size mismatch for " << ti.name << ": file=" << ti.size
               << " ggml=" << expected;
            throw std::runtime_error(os.str());
        }
        ggml_backend_tensor_set(t, file.tensor_data(ti), 0, ti.size);
    }
}

WeightStore::~WeightStore() {
    if (buffer_) ggml_backend_buffer_free(buffer_);
    if (ctx_) ggml_free(ctx_);
}

ggml_tensor * WeightStore::get(const std::string & name) const {
    auto * t = maybe(name);
    if (!t) throw std::runtime_error("missing weight: " + name);
    return t;
}

ggml_tensor * WeightStore::maybe(const std::string & name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : it->second;
}

uint8_t WeightStore::flags(const std::string & name) const {
    auto it = flags_.find(name);
    return it == flags_.end() ? 0 : it->second;
}

Architecture read_architecture(const MoggFile & f) {
    Architecture a;
    a.version = static_cast<int>(f.meta<int64_t>("model.version"));
    a.remap = f.meta<std::string>("model.remap_output");
    a.token_range = f.meta<std::vector<int64_t>>("model.num_tokens_range");
    a.backbone = f.meta<std::string>("encoder.backbone");
    a.embed_dim = static_cast<int>(f.meta<int64_t>("encoder.embed_dim"));
    a.depth = static_cast<int>(f.meta<int64_t>("encoder.depth"));
    a.heads = static_cast<int>(f.meta<int64_t>("encoder.num_heads"));
    a.encoder_out = static_cast<int>(f.meta<int64_t>("encoder.dim_out"));
    a.ffn = f.meta<std::string>("encoder.ffn");
    a.intermediate_layers = f.meta<std::vector<int64_t>>("encoder.intermediate_layers");
    a.neck = read_stack(f, "neck");
    a.points = read_stack(f, "points_head");
    a.normal = read_stack(f, "normal_head");
    a.mask = read_stack(f, "mask_head");
    a.scale_present = f.meta_or<bool>("scale_head.present", false);
    if (a.scale_present) a.scale_dims = f.meta<std::vector<int64_t>>("scale_head.dims");
    a.refiner_present = f.meta_or<bool>("refiner.present", false);
    return a;
}

DenseBuilder::DenseBuilder(ggml_context * ctx, const WeightStore & w, const Architecture & arch,
                           int image_w, int image_h, int base_w, int base_h,
                           ggml_tensor * pos_embed_override,
                           const std::array<ggml_tensor *, 5> & uv_overrides,
                           bool destructive_residuals)
    : c_(ctx), w_(w), a_(arch), iw_(image_w), ih_(image_h), bw_(base_w), bh_(base_h),
      pos_embed_override_(pos_embed_override), uv_overrides_(uv_overrides),
      destructive_residuals_(destructive_residuals) {}

ggml_tensor * DenseBuilder::maybe_hadamard_rotate(ggml_tensor * x, const std::string & weight_name) {
    if (!(w_.flags(weight_name) & TF_HADAMARD_ROTATED)) return x;
    const int g = w_.rotation_group();
    if (g <= 0 || x->ne[0] % g != 0)
        throw std::runtime_error("invalid Hadamard rotation metadata for " + weight_name);
    auto * h = weight("__mogg.hadamard." + std::to_string(g));
    auto * xc = ggml_is_contiguous(x) ? x : ggml_cont(c_, x);
    const int64_t total = ggml_nelements(xc);
    auto * flat = ggml_reshape_2d(c_, xc, g, total / g);
    auto * rot = ggml_mul_mat(c_, h, flat);
    ggml_mul_mat_set_hint(rot, GGML_HINT_SRC0_IS_HADAMARD);
    switch (ggml_n_dims(x)) {
        case 1: return ggml_reshape_1d(c_, rot, x->ne[0]);
        case 2: return ggml_reshape_2d(c_, rot, x->ne[0], x->ne[1]);
        case 3: return ggml_reshape_3d(c_, rot, x->ne[0], x->ne[1], x->ne[2]);
        default: return ggml_reshape_4d(c_, rot, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
    }
}

ggml_tensor * DenseBuilder::matmul_weight(ggml_tensor * x, const std::string & weight_name) {
    x = maybe_hadamard_rotate(x, weight_name);
    auto * w = weight(weight_name);
    if (w->ne[0] != x->ne[0]) {
        throw std::runtime_error(
            "linear shape mismatch for " + weight_name +
            ": weight K=" + std::to_string(w->ne[0]) +
            " activation K=" + std::to_string(x->ne[0]) +
            " activation shape=[" + std::to_string(x->ne[0]) + "," +
            std::to_string(x->ne[1]) + "," + std::to_string(x->ne[2]) + "," +
            std::to_string(x->ne[3]) + "]");
    }
    return ggml_mul_mat(c_, w, x);
}

ggml_tensor * DenseBuilder::linear(ggml_tensor * x, const std::string & pfx) {
    auto * y = matmul_weight(x, pfx + ".weight");
    return add_bias(c_, y, maybe(pfx + ".bias"));
}

ggml_tensor * DenseBuilder::layer_norm(ggml_tensor * x, const std::string & pfx, float eps) {
    auto * y = ggml_norm(c_, x, eps);
    auto * w = maybe(pfx + ".weight");
    auto * b = maybe(pfx + ".bias");
    if (w) y = mul_inplace_safe(c_, y, cast_like(c_, w, y));
    if (b) y = add_inplace_safe(c_, y, cast_like(c_, b, y));
    return y;
}

ggml_tensor * DenseBuilder::affine_group_norm(ggml_tensor * x, const std::string & pfx,
                                               int groups, float eps) {
    auto * y = ggml_group_norm(c_, x, groups, eps);
    return channel_affine(c_, y, maybe(pfx + ".weight"), maybe(pfx + ".bias"));
}

ggml_tensor * DenseBuilder::conv1x1(ggml_tensor * x, const std::string & pfx) {
    const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], N = x->ne[3];
    auto * cf = ggml_cont(c_, ggml_permute(c_, x, 1, 2, 0, 3)); // [C,W,H,N]
    auto * m = ggml_reshape_2d(c_, cf, C, W * H * N);
    auto * y = linear(m, pfx);
    y = ggml_reshape_4d(c_, y, y->ne[0], W, H, N);
    return ggml_permute(c_, y, 2, 0, 1, 3); // [W,H,Cout,N]
}

ggml_tensor * DenseBuilder::conv2d_replicate(ggml_tensor * x, const std::string & pfx, int pad) {
    auto * k = weight(pfx + ".weight");
    auto * xp = replicate_pad2d(c_, x, pad);
    ggml_tensor * y = ggml_conv_2d(c_, k, xp, 1, 1, 0, 0, 1, 1);
    if (auto * b = maybe(pfx + ".bias")) {
        auto * br = ggml_reshape_4d(c_, b, 1, 1, b->ne[0], 1);
        y = add_inplace_safe(c_, y, cast_like(c_, br, y));
    }
    return y;
}

ggml_tensor * DenseBuilder::activate(ggml_tensor * x, const std::string & kind, bool inplace) {
    if (kind == "relu") return inplace ? ggml_relu_inplace(c_, x) : ggml_relu(c_, x);
    if (kind == "leaky_relu") return ggml_leaky_relu(c_, x, 0.2f, inplace);
    if (kind == "silu") return inplace ? ggml_silu_inplace(c_, x) : ggml_silu(c_, x);
    if (kind == "elu") return inplace ? ggml_elu_inplace(c_, x) : ggml_elu(c_, x);
    throw std::runtime_error("unsupported activation: " + kind);
}

ggml_tensor * DenseBuilder::make_uv(int width, int height) {
    // Exact normalized_view_plane_uv formula from MoGe.  Arange is used instead
    // of host constants so the graph remains self-contained across resolutions.
    const float ar = static_cast<float>(iw_) / static_cast<float>(ih_);
    const float den = std::sqrt(1.0f + ar * ar);
    const float span_x = ar / den;
    const float span_y = 1.0f / den;
    auto * u = ggml_arange(c_, 0.0f, static_cast<float>(width), 1.0f);
    auto * v = ggml_arange(c_, 0.0f, static_cast<float>(height), 1.0f);
    // u = (2*i-(W-1))/W * span_x, matching torch.linspace endpoints.
    u = ggml_scale_bias(c_, u, 2.0f * span_x / width, -span_x * (width - 1.0f) / width);
    v = ggml_scale_bias(c_, v, 2.0f * span_y / height, -span_y * (height - 1.0f) / height);
    auto * ur = ggml_reshape_4d(c_, u, width, 1, 1, 1);
    ur = ggml_repeat_4d(c_, ur, width, height, 1, 1);
    auto * vr = ggml_reshape_4d(c_, v, 1, height, 1, 1);
    vr = ggml_repeat_4d(c_, vr, width, height, 1, 1);
    return ggml_concat(c_, ur, vr, 2); // [W,H,2,1]
}

ggml_tensor * DenseBuilder::pixel_shuffle2(ggml_tensor * x) {
    // Converter rewrites c*4+phase into phase-major phase*C+c.  Interleave the
    // four phase slabs with CONCATs instead of repeatedly SET-ing into a shared
    // destination. ggml_set_inplace() returns a view of its destination; chaining
    // the eight scatter writes produced deep nested view chains that Metal 0.25.3
    // can fail to resolve during fusion ("tensor '(view)...' buffer is nil").
    //
    // Work in channel-first storage [4C,W,H]. For ky=0/1, concat kx=0/1 along a
    // singleton axis to obtain [C,2,W,H], then reshape that to [C,2W,1,H]. A
    // second concat along the singleton Y-phase axis yields [C,2W,2,H], whose
    // contiguous storage reshapes directly to [C,2W,2H]. No scatter buffer and
    // no nested destination views are required.
    const int64_t W = x->ne[0], H = x->ne[1], C4 = x->ne[2], N = x->ne[3];
    if (C4 % 4 != 0 || N != 1) throw std::runtime_error("pixel_shuffle2 shape");
    if (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16 && x->type != GGML_TYPE_BF16)
        throw std::runtime_error("pixel_shuffle2 expects F32/F16/BF16 activation");

    const int64_t C = C4 / 4;
    const size_t es = ggml_type_size(x->type);
    auto * cf = ggml_cont(c_, ggml_permute(c_, x, 1, 2, 0, 3)); // [4C,W,H,1]

    auto phase = [&](int p) -> ggml_tensor * {
        // Logical [C,1,W,H] view of one phase. dim-1 is singleton, so give it
        // the natural C stride while preserving the true W/H strides of cf.
        return ggml_view_4d(c_, cf, C, 1, W, H,
                            static_cast<size_t>(C) * es,
                            cf->nb[1], cf->nb[2],
                            static_cast<size_t>(p) * C * cf->nb[0]);
    };

    auto * row0 = ggml_concat(c_, phase(0), phase(1), 1); // [C,2,W,H], kx interleaved
    auto * row1 = ggml_concat(c_, phase(2), phase(3), 1); // [C,2,W,H]
    row0 = ggml_reshape_4d(c_, row0, C, W * 2, 1, H);
    row1 = ggml_reshape_4d(c_, row1, C, W * 2, 1, H);
    auto * out = ggml_concat(c_, row0, row1, 2);          // [C,2W,2,H], ky interleaved
    out = ggml_reshape_4d(c_, out, C, W * 2, H * 2, 1); // [C,2W,2H,1]
    return ggml_permute(c_, out, 2, 0, 1, 3);            // [2W,2H,C,1]
}

ggml_tensor * DenseBuilder::phase_convtranspose2(ggml_tensor * x, const std::string & pfx) {
    const int64_t W = x->ne[0], H = x->ne[1], Cin = x->ne[2];
    auto * cf = ggml_cont(c_, ggml_permute(c_, x, 1, 2, 0, 3));
    auto * m = ggml_reshape_2d(c_, cf, Cin, W * H);
    auto * y = matmul_weight(m, pfx + ".weight.mogg_phase"); // [4C,W*H]
    y = ggml_reshape_4d(c_, y, y->ne[0], W, H, 1);
    // y is phase-major already; transpose to [W,H,4C,1] for common helper.
    y = ggml_permute(c_, y, 2, 0, 1, 3);
    y = pixel_shuffle2(y);
    // ConvTranspose bias is per output channel, not per phase. Applying it
    // after interleave avoids constructing/repeating a temporary [4*C] bias.
    if (auto * b = maybe(pfx + ".bias")) {
        auto * br = ggml_reshape_4d(c_, b, 1, 1, b->ne[0], 1);
        y = add_inplace_safe(c_, y, cast_like(c_, br, y));
    }
    return y;
}

ggml_tensor * DenseBuilder::resample(ggml_tensor * x, const std::string & pfx, const std::string & type) {
    if (type == "pixel_shuffle") {
        x = conv2d_replicate(x, pfx + ".0", 1);
        x = pixel_shuffle2(x);
        return conv2d_replicate(x, pfx + ".2", 1);
    }
    if (type == "nearest" || type == "bilinear") {
        const uint32_t mode = type == "nearest" ? GGML_SCALE_MODE_NEAREST : GGML_SCALE_MODE_BILINEAR;
        x = ggml_interpolate(c_, x, x->ne[0] * 2, x->ne[1] * 2, x->ne[2], x->ne[3], mode);
        return conv2d_replicate(x, pfx + ".1", 1);
    }
    if (type == "conv_transpose") {
        if (maybe(pfx + ".0.weight.mogg_phase")) {
            x = phase_convtranspose2(x, pfx + ".0");
        } else {
            x = ggml_conv_transpose_2d_p0(c_, weight(pfx + ".0.weight"), x, 2);
            if (auto * b = maybe(pfx + ".0.bias")) {
                auto * br = ggml_reshape_4d(c_, b, 1, 1, b->ne[0], 1);
                x = add_inplace_safe(c_, x, cast_like(c_, br, x));
            }
        }
        return conv2d_replicate(x, pfx + ".1", 1);
    }
    if (type == "pixel_unshuffle") {
        // Space-to-depth for factor 2. This path is rarely used in released
        // MoGe-2/3 configs, but keep it exact via four strided views + concat.
        const int64_t W2 = x->ne[0] / 2, H2 = x->ne[1] / 2;
        std::vector<ggml_tensor *> phases;
        auto * cf = ggml_cont(c_, ggml_permute(c_, x, 1, 2, 0, 3)); // [C,W,H]
        for (int ky=0; ky<2; ++ky) for (int kx=0; kx<2; ++kx) {
            phases.push_back(ggml_view_3d(c_, cf, cf->ne[0], W2, H2,
                                         cf->nb[1]*2, cf->nb[2]*2,
                                         kx*cf->nb[1] + ky*cf->nb[2]));
        }
        auto * cat = phases[0];
        for (int i=1;i<4;++i) cat = ggml_concat(c_, cat, phases[i], 0);
        // Concatenating phase slabs gives p*C+c. PixelUnshuffle's channel
        // contract is c*4+p, so transpose the logical [C,phase] pair once.
        cat = ggml_reshape_4d(c_, cat, cf->ne[0], 4, W2, H2);
        cat = ggml_cont(c_, ggml_permute(c_, cat, 1, 0, 2, 3)); // [4,C,W,H]
        cat = ggml_reshape_4d(c_, cat, cf->ne[0] * 4, W2, H2, 1);
        cat = ggml_permute(c_, cat, 2, 0, 1, 3);
        return conv2d_replicate(cat, pfx + ".1", 1);
    }
    if (type == "avg_pool" || type == "max_pool") {
        x = conv2d_replicate(x, pfx + ".0", 1);
        return ggml_pool_2d(c_, x, type == "avg_pool" ? GGML_OP_POOL_AVG : GGML_OP_POOL_MAX,
                            2, 2, 2, 2, 0.0f, 0.0f);
    }
    throw std::runtime_error("unsupported ConvStack resampler: " + type);
}

ggml_tensor * DenseBuilder::conv_stack(const ConvStackCfg & cfg, const std::string & pfx,
                                        const std::vector<ggml_tensor *> & input,
                                        std::vector<ggml_tensor *> * all_outputs) {
    if (!cfg.present) return nullptr;
    ggml_tensor * x = nullptr;
    std::vector<ggml_tensor *> outs;
    for (size_t i=0; i<cfg.dim_res.size(); ++i) {
        ggml_tensor * feature = i < input.size() ? input[i] : nullptr;
        if (feature && cfg.dim_in[i] >= 0) feature = conv1x1(feature, idx(pfx + ".input_blocks", static_cast<int>(i)));
        if (i == 0) x = feature;
        else if (feature) x = add_inplace_safe(c_, x, feature);
        if (!x) throw std::runtime_error("ConvStack has no feature at level " + std::to_string(i));

        const int blocks = static_cast<int>(cfg.num_res_blocks[i]);
        for (int j=0; j<blocks; ++j) {
            const std::string q = idx(idx(pfx + ".res_blocks", static_cast<int>(i)), j);
            auto * skip = x;
            const bool projected_skip = maybe(q + ".skip_connection.weight") != nullptr;
            if (projected_skip) skip = conv1x1(skip, q + ".skip_connection");

            std::string np = q + ".layers.0";
            if (cfg.in_norm == "group_norm") x = affine_group_norm(x, np, static_cast<int>(x->ne[2] / 32));
            else if (cfg.in_norm == "layer_norm") x = affine_group_norm(x, np, 1);
            else if (cfg.in_norm == "instance_norm") x = ggml_group_norm(c_, x, static_cast<int>(x->ne[2]), 1e-5f);
            else if (cfg.in_norm != "none") throw std::runtime_error("unsupported conv norm " + cfg.in_norm);
            // Preserve the identity residual. Upstream computes skip from the
            // pre-activation input; an in-place first activation would mutate
            // that same tensor and turn x + F(x) into act(x) + F(x).
            x = activate(x, cfg.activation, projected_skip);
            x = conv2d_replicate(x, q + ".layers.2", 1);

            np = q + ".layers.3";
            if (cfg.hidden_norm == "group_norm") x = affine_group_norm(x, np, static_cast<int>(x->ne[2] / 32));
            else if (cfg.hidden_norm == "layer_norm") x = affine_group_norm(x, np, 1);
            else if (cfg.hidden_norm == "instance_norm") x = ggml_group_norm(c_, x, static_cast<int>(x->ne[2]), 1e-5f);
            else if (cfg.hidden_norm != "none") throw std::runtime_error("unsupported hidden norm " + cfg.hidden_norm);
            x = activate(x, cfg.activation);
            x = conv2d_replicate(x, q + ".layers.5", 1);
            x = add_inplace_safe(c_, x, skip);
        }

        ggml_tensor * out = x;
        if (cfg.dim_out[i] >= 0) out = conv1x1(out, idx(pfx + ".output_blocks", static_cast<int>(i)));
        outs.push_back(out);
        if (i + 1 < cfg.dim_res.size()) x = resample(x, idx(pfx + ".resamplers", static_cast<int>(i)), cfg.resamplers[i]);
    }
    if (all_outputs) *all_outputs = outs;
    return outs.empty() ? nullptr : outs.back();
}


DenseStackStep DenseBuilder::conv_stack_level(const ConvStackCfg & cfg, const std::string & pfx,
                                               size_t level, ggml_tensor * x, ggml_tensor * feature) {
    if (!cfg.present) throw std::runtime_error("segmented ConvStack is not present: " + pfx);
    if (level >= cfg.dim_res.size()) throw std::runtime_error("segmented ConvStack level out of range: " + pfx);
    if (level > 0) {
        if (!x) throw std::runtime_error("segmented ConvStack missing previous state: " + pfx);
        x = resample(x, idx(pfx + ".resamplers", static_cast<int>(level - 1)), cfg.resamplers[level - 1]);
    }

    if (feature && cfg.dim_in[level] >= 0) {
        feature = conv1x1(feature, idx(pfx + ".input_blocks", static_cast<int>(level)));
    }
    if (level == 0) {
        x = feature;
    } else if (feature) {
        x = add_inplace_safe(c_, x, feature);
    }
    if (!x) throw std::runtime_error("segmented ConvStack has no feature at level " + std::to_string(level));

    const int blocks = static_cast<int>(cfg.num_res_blocks[level]);
    for (int j = 0; j < blocks; ++j) {
        const std::string q = idx(idx(pfx + ".res_blocks", static_cast<int>(level)), j);
        auto * skip = x;
        const bool projected_skip = maybe(q + ".skip_connection.weight") != nullptr;
        if (projected_skip) skip = conv1x1(skip, q + ".skip_connection");

        std::string np = q + ".layers.0";
        if (cfg.in_norm == "group_norm") x = affine_group_norm(x, np, static_cast<int>(x->ne[2] / 32));
        else if (cfg.in_norm == "layer_norm") x = affine_group_norm(x, np, 1);
        else if (cfg.in_norm == "instance_norm") x = ggml_group_norm(c_, x, static_cast<int>(x->ne[2]), 1e-5f);
        else if (cfg.in_norm != "none") throw std::runtime_error("unsupported conv norm " + cfg.in_norm);
        // Preserve the identity residual exactly as PyTorch does.
        x = activate(x, cfg.activation, projected_skip);
        x = conv2d_replicate(x, q + ".layers.2", 1);

        np = q + ".layers.3";
        if (cfg.hidden_norm == "group_norm") x = affine_group_norm(x, np, static_cast<int>(x->ne[2] / 32));
        else if (cfg.hidden_norm == "layer_norm") x = affine_group_norm(x, np, 1);
        else if (cfg.hidden_norm == "instance_norm") x = ggml_group_norm(c_, x, static_cast<int>(x->ne[2]), 1e-5f);
        else if (cfg.hidden_norm != "none") throw std::runtime_error("unsupported hidden norm " + cfg.hidden_norm);
        x = activate(x, cfg.activation);
        x = conv2d_replicate(x, q + ".layers.5", 1);
        x = add_inplace_safe(c_, x, skip);
    }

    ggml_tensor * out = x;
    if (cfg.dim_out[level] >= 0) out = conv1x1(out, idx(pfx + ".output_blocks", static_cast<int>(level)));
    return { out, level + 1 < cfg.dim_res.size() ? x : nullptr };
}


ggml_tensor * DenseBuilder::upsample2_bilinear(ggml_tensor * x) {
    if (!x) throw std::runtime_error("upsample2_bilinear null input");
    return ggml_interpolate(c_,x,x->ne[0]*2,x->ne[1]*2,x->ne[2],x->ne[3],GGML_SCALE_MODE_BILINEAR);
}

ggml_tensor * DenseBuilder::conv_stack_final_level_tile(const ConvStackCfg & cfg, const std::string & pfx,
                                                         size_t level, ggml_tensor * upsampled,
                                                         ggml_tensor * feature,
                                                         int64_t y0, int64_t y1) {
    if (!cfg.present || !upsampled) throw std::runtime_error("tiled final ConvStack missing input");
    if (level+1 != cfg.dim_res.size() || level==0)
        throw std::runtime_error("tiled final ConvStack requires last nonzero level");
    if (cfg.resamplers[level-1] != "bilinear")
        throw std::runtime_error("tiled final ConvStack currently requires bilinear resampler");
    if (cfg.num_res_blocks[level] != 0)
        throw std::runtime_error("tiled final ConvStack currently requires zero final resblocks");
    const int64_t W=upsampled->ne[0], H=upsampled->ne[1];
    if (y0 < 0 || y1 <= y0 || y1 > H) throw std::runtime_error("tiled final ConvStack row range");

    // This is exactly the convolution half of resample(type="bilinear"):
    // interpolate has already been materialized into the shared persistent stage.
    auto * padded=replicate_pad2d_rows(c_,upsampled,y0,y1);
    if (padded->type!=GGML_TYPE_F32) padded=ggml_cast(c_,padded,GGML_TYPE_F32);
    auto * k=weight(idx(pfx+".resamplers",static_cast<int>(level-1))+".1.weight");
    auto * x = ggml_conv_2d(c_, k,padded,1,1,0,0,1,1);
    const std::string rp=idx(pfx+".resamplers",static_cast<int>(level-1))+".1";
    if (auto * bias=maybe(rp+".bias")) {
        auto * br=ggml_reshape_4d(c_,bias,1,1,bias->ne[0],1);
        x=add_inplace_safe(c_,x,cast_like(c_,br,x));
    }

    // Same-level feature is projected tile-locally. This avoids a second full
    // high-resolution 32-channel temporary for UV/neck skip projection.
    if (feature && cfg.dim_in[level]>=0) {
        auto * ft=ggml_view_4d(c_,feature,W,y1-y0,feature->ne[2],feature->ne[3],
                              feature->nb[1],feature->nb[2],feature->nb[3],y0*feature->nb[1]);
        if (ft->type!=GGML_TYPE_F32) ft=ggml_cast(c_,ft,GGML_TYPE_F32);
        ft=conv1x1(ft,idx(pfx+".input_blocks",static_cast<int>(level)));
        x=add_inplace_safe(c_,x,ft);
    }

    if (cfg.dim_out[level]>=0) x=conv1x1(x,idx(pfx+".output_blocks",static_cast<int>(level)));
    return x;
}





ggml_tensor * DenseBuilder::conv_stack_final_level_prepare_conv_tile(const ConvStackCfg & cfg, const std::string & pfx,
                                                                      size_t level, ggml_tensor * previous,
                                                                      int64_t y0, int64_t y1) {
    if (!cfg.present || !previous) throw std::runtime_error("streamed final ConvStack missing input");
    if (level + 1 != cfg.dim_res.size() || level == 0)
        throw std::runtime_error("streamed final ConvStack requires last nonzero level");
    if (cfg.resamplers[level - 1] != "bilinear")
        throw std::runtime_error("streamed final ConvStack currently requires bilinear resampler");
    if (cfg.num_res_blocks[level] != 0)
        throw std::runtime_error("streamed final ConvStack currently requires zero final resblocks");
    const int64_t src_w = previous->ne[0], src_h = previous->ne[1];
    const int64_t W = src_w * 2, H = src_h * 2;
    if (y0 < 0 || y1 <= y0 || y1 > H) throw std::runtime_error("streamed final ConvStack row range");
    const int64_t src_y0 = std::max<int64_t>(0, y0 / 2 - 1);
    const int64_t src_y1 = y1 == H ? src_h : std::min<int64_t>(src_h, (y1 + 1) / 2 + 1);
    if (src_y1 <= src_y0) throw std::runtime_error("streamed final ConvStack source row range");
    auto * src = ggml_view_4d(c_, previous, src_w, src_y1 - src_y0, previous->ne[2], previous->ne[3],
                              previous->nb[1], previous->nb[2], previous->nb[3],
                              static_cast<size_t>(src_y0) * previous->nb[1]);
    if (src->type != GGML_TYPE_F32) src = ggml_cast(c_, src, GGML_TYPE_F32);
    auto * local_up = ggml_interpolate(c_, src, src_w * 2, (src_y1 - src_y0) * 2,
                                       src->ne[2], src->ne[3], GGML_SCALE_MODE_BILINEAR);
    const int64_t local_y0 = y0 - 2 * src_y0;
    const int64_t local_y1 = y1 - 2 * src_y0;
    if (local_y0 < 0 || local_y1 > local_up->ne[1])
        throw std::runtime_error("streamed final ConvStack local crop range");
    auto * padded = replicate_pad2d_rows(c_, local_up, local_y0, local_y1);
    if (padded->type != GGML_TYPE_F32) padded = ggml_cast(c_, padded, GGML_TYPE_F32);
    (void)pfx;
    return padded;
}

ggml_tensor * DenseBuilder::conv_stack_final_level_finish_conv_tile(const ConvStackCfg & cfg, const std::string & pfx,
                                                                     size_t level, ggml_tensor * x,
                                                                     ggml_tensor * feature,
                                                                     int64_t y0, int64_t y1) {
    if (!x) throw std::runtime_error("streamed final ConvStack missing conv output");
    const int64_t W = x->ne[0];
    if (x->ne[1] != y1 - y0) throw std::runtime_error("streamed final ConvStack conv output row mismatch");

    // This hook is used by the external MPSGraph-convolution replay.  Its input
    // is a dedicated preallocated Metal tensor.  Do not build inplace ADD views
    // on that external buffer: ggml's scheduler must pin a preallocated op result
    // to the buffer backend, and Metal can reject the resulting broadcast/view
    // ADD before it has a chance to insert a materialization.  A graph-local DUP
    // preserves the exact F32 values and makes all following suffix ops ordinary
    // scheduler-owned tensors.
    if (x->buffer != nullptr || (x->view_src && x->view_src->buffer != nullptr)) {
        x = ggml_dup(c_, x);
    }
    const std::string rp = idx(pfx + ".resamplers", static_cast<int>(level - 1)) + ".1";
    if (auto * bias = maybe(rp + ".bias")) {
        auto * br = ggml_reshape_4d(c_, bias, 1, 1, bias->ne[0], 1);
        x = add_inplace_safe(c_, x, cast_like(c_, br, x));
    }
    if (feature && cfg.dim_in[level] >= 0) {
        auto * ft = ggml_view_4d(c_, feature, W, y1 - y0, feature->ne[2], feature->ne[3],
                                 feature->nb[1], feature->nb[2], feature->nb[3],
                                 static_cast<size_t>(y0) * feature->nb[1]);
        if (ft->type != GGML_TYPE_F32) ft = ggml_cast(c_, ft, GGML_TYPE_F32);
        ft = conv1x1(ft, idx(pfx + ".input_blocks", static_cast<int>(level)));
        // Metal binary ops require contiguous-row operands.  The 1x1 helper
        // returns a permuted logical [W,H,C] tensor, so explicitly materialize
        // it in this external-conv suffix before the ADD.
        if (!ggml_is_contiguous_rows(ft)) ft = ggml_cont(c_, ft);
        x = add_inplace_safe(c_, x, ft);
    }
    if (cfg.dim_out[level] >= 0) x = conv1x1(x, idx(pfx + ".output_blocks", static_cast<int>(level)));
    return x;
}

// Stream the final bilinear-resample level without a full-frame upsampled stage.
// For a 2x half-pixel bilinear resize, a local low-resolution row slice produces
// bit-identical global output rows when its local output coordinate is offset by
// exactly 2*src_y0.  We include enough source rows that replicate_pad2d_rows()
// sees the true one-row 3x3 halo; only the actual image top/bottom can clamp.
ggml_tensor * DenseBuilder::conv_stack_final_level_fused_tile(const ConvStackCfg & cfg, const std::string & pfx,
                                                               size_t level, ggml_tensor * previous,
                                                               ggml_tensor * feature,
                                                               int64_t y0, int64_t y1) {
    if (!cfg.present || !previous) throw std::runtime_error("streamed final ConvStack missing input");
    if (level + 1 != cfg.dim_res.size() || level == 0)
        throw std::runtime_error("streamed final ConvStack requires last nonzero level");
    if (cfg.resamplers[level - 1] != "bilinear")
        throw std::runtime_error("streamed final ConvStack currently requires bilinear resampler");
    if (cfg.num_res_blocks[level] != 0)
        throw std::runtime_error("streamed final ConvStack currently requires zero final resblocks");

    const int64_t src_w = previous->ne[0], src_h = previous->ne[1];
    const int64_t W = src_w * 2, H = src_h * 2;
    if (y0 < 0 || y1 <= y0 || y1 > H) throw std::runtime_error("streamed final ConvStack row range");

    // The trailing 3x3 needs global output rows y0-1 and y1 when interior.
    // Half-pixel 2x bilinear for an output row g references at most source row
    // floor((g+1)/2), and an even g may also reference the previous source row.
    // These bounds therefore avoid artificial slice-edge clamping exactly.
    const int64_t src_y0 = std::max<int64_t>(0, y0 / 2 - 1);
    const int64_t src_y1 = y1 == H ? src_h
        : std::min<int64_t>(src_h, (y1 + 1) / 2 + 1);
    if (src_y1 <= src_y0) throw std::runtime_error("streamed final ConvStack source row range");

    auto * src = ggml_view_4d(c_, previous, src_w, src_y1 - src_y0, previous->ne[2], previous->ne[3],
                              previous->nb[1], previous->nb[2], previous->nb[3],
                              static_cast<size_t>(src_y0) * previous->nb[1]);
    // The accepted final-level path casts decoder state to f32 before interpolate.
    // Cast only this bounded source slice so optional f16 state keeps identical
    // arithmetic without materializing a full low-resolution f32 temporary.
    if (src->type != GGML_TYPE_F32) src = ggml_cast(c_, src, GGML_TYPE_F32);
    auto * local_up = ggml_interpolate(c_, src, src_w * 2, (src_y1 - src_y0) * 2,
                                       src->ne[2], src->ne[3], GGML_SCALE_MODE_BILINEAR);
    const int64_t local_y0 = y0 - 2 * src_y0;
    const int64_t local_y1 = y1 - 2 * src_y0;
    if (local_y0 < 0 || local_y1 > local_up->ne[1])
        throw std::runtime_error("streamed final ConvStack local crop range");

    auto * padded = replicate_pad2d_rows(c_, local_up, local_y0, local_y1);
    if (padded->type != GGML_TYPE_F32) padded = ggml_cast(c_, padded, GGML_TYPE_F32);
    const std::string rp = idx(pfx + ".resamplers", static_cast<int>(level - 1)) + ".1";
    auto * k = weight(rp + ".weight");
    auto * x = ggml_conv_2d(c_, k, padded, 1, 1, 0, 0, 1, 1);
    if (auto * bias = maybe(rp + ".bias")) {
        auto * br = ggml_reshape_4d(c_, bias, 1, 1, bias->ne[0], 1);
        x = add_inplace_safe(c_, x, cast_like(c_, br, x));
    }

    if (feature && cfg.dim_in[level] >= 0) {
        auto * ft = ggml_view_4d(c_, feature, W, y1 - y0, feature->ne[2], feature->ne[3],
                                 feature->nb[1], feature->nb[2], feature->nb[3],
                                 static_cast<size_t>(y0) * feature->nb[1]);
        if (ft->type != GGML_TYPE_F32) ft = ggml_cast(c_, ft, GGML_TYPE_F32);
        ft = conv1x1(ft, idx(pfx + ".input_blocks", static_cast<int>(level)));
        // Keep the external-MPS suffix on the Metal-supported binary-op path.
        // conv1x1 may return a permuted logical tensor, while Metal ADD expects
        // contiguous-row operands here.
        if (!ggml_is_contiguous_rows(ft)) ft = ggml_cont(c_, ft);
        x = add_inplace_safe(c_, x, ft);
    }
    if (cfg.dim_out[level] >= 0) x = conv1x1(x, idx(pfx + ".output_blocks", static_cast<int>(level)));
    return x;
}

ggml_tensor * DenseBuilder::conv_stack_stream_prepare_resample(const ConvStackCfg & cfg,
                                                                 const std::string & pfx,
                                                                 size_t level,
                                                                 ggml_tensor * previous) {
    if (!cfg.present || !previous) throw std::runtime_error("streamed ConvStack missing previous state: " + pfx);
    if (level == 0 || level >= cfg.dim_res.size()) throw std::runtime_error("streamed ConvStack level out of range: " + pfx);
    const std::string rp = idx(pfx + ".resamplers", static_cast<int>(level - 1));
    const std::string & type = cfg.resamplers[level - 1];

    // Stop immediately before the resampler's trailing 3x3 convolution.  The
    // returned full-frame tensor is copied to persistent shared staging by the
    // runtime, so the following row-tile phases never coexist with this graph's
    // temporary storage.
    if (type == "nearest" || type == "bilinear") {
        const uint32_t mode = type == "nearest" ? GGML_SCALE_MODE_NEAREST : GGML_SCALE_MODE_BILINEAR;
        return ggml_interpolate(c_, previous, previous->ne[0] * 2, previous->ne[1] * 2,
                                previous->ne[2], previous->ne[3], mode);
    }
    if (type == "conv_transpose") {
        if (maybe(rp + ".0.weight.mogg_phase")) {
            return phase_convtranspose2(previous, rp + ".0");
        }
        auto * x = ggml_conv_transpose_2d_p0(c_, weight(rp + ".0.weight"), previous, 2);
        if (auto * b = maybe(rp + ".0.bias")) {
            auto * br = ggml_reshape_4d(c_, b, 1, 1, b->ne[0], 1);
            x = add_inplace_safe(c_, x, cast_like(c_, br, x));
        }
        return x;
    }
    throw std::runtime_error("streamed ConvStack unsupported resampler " + type + " in " + pfx);
}


ggml_tensor * DenseBuilder::conv_stack_stream_entry_tile(const ConvStackCfg & cfg,
                                                           const std::string & pfx,
                                                           size_t level,
                                                           ggml_tensor * prepared,
                                                           ggml_tensor * feature,
                                                           int64_t y0, int64_t y1) {
    if (!cfg.present || !prepared) throw std::runtime_error("streamed ConvStack missing prepared resample: " + pfx);
    if (level == 0 || level >= cfg.dim_res.size()) throw std::runtime_error("streamed ConvStack entry level out of range: " + pfx);
    const int64_t W = prepared->ne[0], H = prepared->ne[1];
    if (y0 < 0 || y1 <= y0 || y1 > H) throw std::runtime_error("streamed ConvStack entry row range");

    const std::string rp = idx(pfx + ".resamplers", static_cast<int>(level - 1));
    const std::string & type = cfg.resamplers[level - 1];
    std::string post;
    if (type == "nearest" || type == "bilinear" || type == "conv_transpose") post = rp + ".1";
    else throw std::runtime_error("streamed ConvStack unsupported entry resampler " + type + " in " + pfx);

    // The persistent prepared tensor contains the entire upsample/transpose
    // result, so an interior tile can borrow its true one-row halo.  Only the
    // actual image boundary is replicated.
    auto * padded = replicate_pad2d_rows(c_, prepared, y0, y1);
    if (padded->type != GGML_TYPE_F32) padded = ggml_cast(c_, padded, GGML_TYPE_F32);
    auto * k = weight(post + ".weight");
    auto * x = ggml_conv_2d(c_, k, padded, 1, 1, 0, 0, 1, 1);
    if (auto * bias = maybe(post + ".bias")) {
        auto * br = ggml_reshape_4d(c_, bias, 1, 1, bias->ne[0], 1);
        x = add_inplace_safe(c_, x, cast_like(c_, br, x));
    }

    // Same-level feature projection is spatially pointwise, so project only the
    // rows being written.  This avoids creating another full high-resolution
    // feature tensor in the streamed phase.
    if (feature && cfg.dim_in[level] >= 0) {
        auto * ft = ggml_view_4d(c_, feature, W, y1 - y0, feature->ne[2], feature->ne[3],
                                 feature->nb[1], feature->nb[2], feature->nb[3],
                                 static_cast<size_t>(y0) * feature->nb[1]);
        if (ft->type != GGML_TYPE_F32) ft = ggml_cast(c_, ft, GGML_TYPE_F32);
        ft = conv1x1(ft, idx(pfx + ".input_blocks", static_cast<int>(level)));
        x = add_inplace_safe(c_, x, ft);
    }
    return x;
}


ggml_tensor * DenseBuilder::conv_stack_stream_fused_tile(const ConvStackCfg & cfg,
                                                           const std::string & pfx,
                                                           size_t level,
                                                           ggml_tensor * prepared,
                                                           ggml_tensor * feature,
                                                           int64_t y0, int64_t y1) {
    if (!cfg.present || !prepared) throw std::runtime_error("streamed fused ConvStack missing prepared resample: " + pfx);
    if (level == 0 || level >= cfg.dim_res.size()) throw std::runtime_error("streamed fused ConvStack level out of range: " + pfx);
    if (cfg.in_norm != "none" || cfg.hidden_norm != "none")
        throw std::runtime_error("streamed fused ConvStack residuals require none/none normalization: " + pfx);

    const int blocks = static_cast<int>(cfg.num_res_blocks[level]);
    const int64_t W = prepared->ne[0], H = prepared->ne[1];
    if (y0 < 0 || y1 <= y0 || y1 > H) throw std::runtime_error("streamed fused ConvStack row range");

    // The requested output stripe depends on two source rows per residual block
    // on either side. Compute the resampler's trailing 3x3 for that expanded
    // interval directly from the full prepared tensor, so the local residual
    // graph begins with exact pre-residual values for the entire halo.
    const int64_t radius = 2LL * blocks;
    const int64_t src_y0 = std::max<int64_t>(0, y0 - radius);
    const int64_t src_y1 = std::min<int64_t>(H, y1 + radius);
    auto * x = conv_stack_stream_entry_tile(cfg, pfx, level, prepared, feature, src_y0, src_y1);

    // Run the complete residual sequence on the expanded stripe. Interior crop
    // boundaries are outside the dependency cone of the center rows; at y=0/H
    // the local replicated edge is the true image edge. Thus the returned core
    // is bit-equivalent to full-frame residual execution for none/none norms.
    for (int j = 0; j < blocks; ++j) {
        const std::string q = idx(idx(pfx + ".res_blocks", static_cast<int>(level)), j);
        auto * skip = x;
        const bool projected_skip = maybe(q + ".skip_connection.weight") != nullptr;
        if (projected_skip) skip = conv1x1(skip, q + ".skip_connection");
        x = activate(x, cfg.activation, projected_skip);
        x = conv2d_replicate(x, q + ".layers.2", 1);
        x = activate(x, cfg.activation);
        x = conv2d_replicate(x, q + ".layers.5", 1);
        x = add_inplace_safe(c_, x, skip);
    }

    const int64_t core_y = y0 - src_y0;
    if (core_y < 0 || core_y + (y1 - y0) > x->ne[1])
        throw std::runtime_error("streamed fused ConvStack center crop");
    return ggml_view_4d(c_, x, W, y1 - y0, x->ne[2], x->ne[3],
                        x->nb[1], x->nb[2], x->nb[3],
                        static_cast<size_t>(core_y) * x->nb[1]);
}


ggml_tensor * DenseBuilder::conv_stack_stream_phase_prepare_entry_conv_tile(const ConvStackCfg & cfg,
                                                                                const std::string & pfx,
                                                                                size_t level,
                                                                                ggml_tensor * previous,
                                                                                int64_t y0, int64_t y1) {
    if (!cfg.present || !previous) throw std::runtime_error("stage-free streamed ConvStack missing previous state: " + pfx);
    if (level == 0 || level >= cfg.dim_res.size()) throw std::runtime_error("stage-free streamed ConvStack level out of range: " + pfx);
    if (cfg.in_norm != "none" || cfg.hidden_norm != "none")
        throw std::runtime_error("stage-free streamed ConvStack residuals require none/none normalization: " + pfx);
    if (cfg.resamplers[level - 1] != "conv_transpose")
        throw std::runtime_error("stage-free streamed ConvStack requires conv_transpose: " + pfx);

    const std::string rp = idx(pfx + ".resamplers", static_cast<int>(level - 1));
    if (!maybe(rp + ".0.weight.mogg_phase"))
        throw std::runtime_error("stage-free streamed ConvStack requires phase-packed ConvTranspose: " + pfx);

    const int64_t src_w = previous->ne[0], src_h = previous->ne[1];
    const int64_t H = src_h * 2;
    if (y0 < 0 || y1 <= y0 || y1 > H) throw std::runtime_error("stage-free streamed ConvStack row range");

    const int blocks = static_cast<int>(cfg.num_res_blocks[level]);
    const int64_t radius = 2LL * blocks;
    const int64_t entry_y0 = std::max<int64_t>(0, y0 - radius);
    const int64_t entry_y1 = std::min<int64_t>(H, y1 + radius);
    const int64_t prep_y0 = std::max<int64_t>(0, entry_y0 - 1);
    const int64_t prep_y1 = std::min<int64_t>(H, entry_y1 + 1);
    const int64_t src_y0 = prep_y0 / 2;
    const int64_t src_y1 = (prep_y1 + 1) / 2;
    if (src_y1 <= src_y0 || src_y1 > src_h)
        throw std::runtime_error("stage-free streamed ConvStack source row range");

    auto * src = ggml_view_4d(c_, previous, src_w, src_y1 - src_y0,
                              previous->ne[2], previous->ne[3],
                              previous->nb[1], previous->nb[2], previous->nb[3],
                              static_cast<size_t>(src_y0) * previous->nb[1]);
    auto * prepared = phase_convtranspose2(src, rp + ".0");
    const int64_t global_prepared_y0 = 2 * src_y0;
    const int64_t local_entry_y0 = entry_y0 - global_prepared_y0;
    const int64_t local_entry_y1 = entry_y1 - global_prepared_y0;
    if (local_entry_y0 < 0 || local_entry_y1 > prepared->ne[1])
        throw std::runtime_error("stage-free streamed ConvStack local entry crop");

    auto * padded = replicate_pad2d_rows(c_, prepared, local_entry_y0, local_entry_y1);
    if (padded->type != GGML_TYPE_F32) padded = ggml_cast(c_, padded, GGML_TYPE_F32);
    return padded;
}


ggml_tensor * DenseBuilder::conv_stack_stream_phase_finish_entry_conv_tile(const ConvStackCfg & cfg,
                                                                               const std::string & pfx,
                                                                               size_t level,
                                                                               ggml_tensor * conv_output,
                                                                               ggml_tensor * feature,
                                                                               int64_t y0, int64_t y1) {
    if (!cfg.present || !conv_output) throw std::runtime_error("stage-free streamed ConvStack missing entry convolution output: " + pfx);
    if (level == 0 || level >= cfg.dim_res.size()) throw std::runtime_error("stage-free streamed ConvStack level out of range: " + pfx);
    if (cfg.in_norm != "none" || cfg.hidden_norm != "none")
        throw std::runtime_error("stage-free streamed ConvStack residuals require none/none normalization: " + pfx);

    const int blocks = static_cast<int>(cfg.num_res_blocks[level]);
    const int64_t H = feature ? feature->ne[1] : (y1 + 2LL * blocks);
    const int64_t W = conv_output->ne[0];
    const int64_t radius = 2LL * blocks;
    const int64_t entry_y0 = std::max<int64_t>(0, y0 - radius);
    const int64_t entry_y1 = std::min<int64_t>(H, y1 + radius);
    if (conv_output->ne[1] != entry_y1 - entry_y0)
        throw std::runtime_error("stage-free streamed ConvStack entry convolution row mismatch");

    const std::string rp = idx(pfx + ".resamplers", static_cast<int>(level - 1));
    const std::string post = rp + ".1";
    auto * x = conv_output;

    // The production MPSGraph entry-convolution path feeds this suffix from a
    // dedicated preallocated Metal tensor.  As with the validated final-level
    // MPSGraph suffix, do not let an in-place/broadcast ADD inherit that
    // external buffer: ggml_backend_sched must pin a preallocated op result to
    // the buffer backend and Metal may reject the resulting view/broadcast ADD
    // before it can materialize the value.  DUP makes the MPS result an
    // ordinary scheduler-owned graph-local tensor while preserving the exact
    // F32 values.
    if (x->buffer != nullptr || (x->view_src && x->view_src->buffer != nullptr)) {
        x = ggml_dup(c_, x);
    }

    if (auto * bias = maybe(post + ".bias")) {
        auto * br = ggml_reshape_4d(c_, bias, 1, 1, bias->ne[0], 1);
        x = add_inplace_safe(c_, x, cast_like(c_, br, x));
    }

    if (feature && cfg.dim_in[level] >= 0) {
        auto * ft = ggml_view_4d(c_, feature, W, entry_y1 - entry_y0,
                                 feature->ne[2], feature->ne[3],
                                 feature->nb[1], feature->nb[2], feature->nb[3],
                                 static_cast<size_t>(entry_y0) * feature->nb[1]);
        if (ft->type != GGML_TYPE_F32) ft = ggml_cast(c_, ft, GGML_TYPE_F32);
        ft = conv1x1(ft, idx(pfx + ".input_blocks", static_cast<int>(level)));
        // conv1x1 may return a permuted logical tensor; materialize it before
        // the Metal ADD in this external-MPS suffix, matching the accepted
        // final-level MPSGraph path.
        if (!ggml_is_contiguous_rows(ft)) ft = ggml_cont(c_, ft);
        x = add_inplace_safe(c_, x, ft);
    }

    for (int j = 0; j < blocks; ++j) {
        const std::string q = idx(idx(pfx + ".res_blocks", static_cast<int>(level)), j);
        auto * skip = x;
        const bool projected_skip = maybe(q + ".skip_connection.weight") != nullptr;
        if (projected_skip) skip = conv1x1(skip, q + ".skip_connection");
        x = activate(x, cfg.activation, projected_skip);
        x = conv2d_replicate(x, q + ".layers.2", 1);
        x = activate(x, cfg.activation);
        x = conv2d_replicate(x, q + ".layers.5", 1);
        x = add_inplace_safe(c_, x, skip);
    }

    const int64_t core_y = y0 - entry_y0;
    if (core_y < 0 || core_y + (y1 - y0) > x->ne[1])
        throw std::runtime_error("stage-free streamed ConvStack center crop");
    return ggml_view_4d(c_, x, W, y1 - y0, x->ne[2], x->ne[3],
                        x->nb[1], x->nb[2], x->nb[3],
                        static_cast<size_t>(core_y) * x->nb[1]);
}


ggml_tensor * DenseBuilder::conv_stack_stream_phase_finish_entry_pre_residual_tile(const ConvStackCfg & cfg,
                                                                                     const std::string & pfx,
                                                                                     size_t level,
                                                                                     ggml_tensor * conv_output,
                                                                                     ggml_tensor * feature,
                                                                                     int64_t y0, int64_t y1) {
    if (!cfg.present || !conv_output) throw std::runtime_error("stage-free streamed ConvStack missing entry convolution output: " + pfx);
    if (level == 0 || level >= cfg.dim_res.size()) throw std::runtime_error("stage-free streamed ConvStack level out of range: " + pfx);
    if (cfg.in_norm != "none" || cfg.hidden_norm != "none")
        throw std::runtime_error("stage-free streamed ConvStack residuals require none/none normalization: " + pfx);

    const int blocks = static_cast<int>(cfg.num_res_blocks[level]);
    const int64_t H = feature ? feature->ne[1] : (y1 + 2LL * blocks);
    const int64_t W = conv_output->ne[0];
    const int64_t radius = 2LL * blocks;
    const int64_t entry_y0 = std::max<int64_t>(0, y0 - radius);
    const int64_t entry_y1 = std::min<int64_t>(H, y1 + radius);
    if (conv_output->ne[1] != entry_y1 - entry_y0)
        throw std::runtime_error("stage-free streamed ConvStack pre-residual row mismatch");

    const std::string rp = idx(pfx + ".resamplers", static_cast<int>(level - 1));
    const std::string post = rp + ".1";
    auto * x = conv_output;
    if (x->buffer != nullptr || (x->view_src && x->view_src->buffer != nullptr)) x = ggml_dup(c_, x);
    if (auto * bias = maybe(post + ".bias")) {
        auto * br = ggml_reshape_4d(c_, bias, 1, 1, bias->ne[0], 1);
        x = add_inplace_safe(c_, x, cast_like(c_, br, x));
    }
    if (feature && cfg.dim_in[level] >= 0) {
        auto * ft = ggml_view_4d(c_, feature, W, entry_y1 - entry_y0,
                                 feature->ne[2], feature->ne[3],
                                 feature->nb[1], feature->nb[2], feature->nb[3],
                                 static_cast<size_t>(entry_y0) * feature->nb[1]);
        if (ft->type != GGML_TYPE_F32) ft = ggml_cast(c_, ft, GGML_TYPE_F32);
        ft = conv1x1(ft, idx(pfx + ".input_blocks", static_cast<int>(level)));
        if (!ggml_is_contiguous_rows(ft)) ft = ggml_cont(c_, ft);
        x = add_inplace_safe(c_, x, ft);
    }
    return x;
}

ggml_tensor * DenseBuilder::conv_stack_stream_residual0_prepare_conv_tile(const ConvStackCfg & cfg,
                                                                            const std::string & pfx,
                                                                            size_t level,
                                                                            ggml_tensor * pre_residual) {
    if (!cfg.present || !pre_residual) throw std::runtime_error("streamed ConvStack missing pre-residual state: " + pfx);
    if (level >= cfg.dim_res.size() || cfg.num_res_blocks[level] < 1)
        throw std::runtime_error("streamed ConvStack residual0 level/block out of range: " + pfx);
    if (cfg.in_norm != "none" || cfg.hidden_norm != "none")
        throw std::runtime_error("streamed ConvStack residual0 requires none/none normalization: " + pfx);
    const std::string q = idx(idx(pfx + ".res_blocks", static_cast<int>(level)), 0);
    const bool projected_skip = maybe(q + ".skip_connection.weight") != nullptr;
    auto * x = pre_residual;
    if (x->type != GGML_TYPE_F32) x = ggml_cast(c_, x, GGML_TYPE_F32);
    x = activate(x, cfg.activation, projected_skip);
    auto * padded = replicate_pad2d(c_, x, 1);
    if (padded->type != GGML_TYPE_F32) padded = ggml_cast(c_, padded, GGML_TYPE_F32);
    return padded;
}

ggml_tensor * DenseBuilder::conv_stack_stream_residual0_finish_conv_tile(const ConvStackCfg & cfg,
                                                                           const std::string & pfx,
                                                                           size_t level,
                                                                           ggml_tensor * pre_residual,
                                                                           ggml_tensor * conv_output,
                                                                           int64_t full_h,
                                                                           int64_t y0, int64_t y1) {
    if (!cfg.present || !pre_residual || !conv_output)
        throw std::runtime_error("streamed ConvStack residual0 missing state: " + pfx);
    if (level >= cfg.dim_res.size() || cfg.num_res_blocks[level] < 1)
        throw std::runtime_error("streamed ConvStack residual0 level/block out of range: " + pfx);
    if (cfg.in_norm != "none" || cfg.hidden_norm != "none")
        throw std::runtime_error("streamed ConvStack residual0 requires none/none normalization: " + pfx);

    const int blocks = static_cast<int>(cfg.num_res_blocks[level]);
    const int64_t radius = 2LL * blocks;
    const int64_t entry_y0 = std::max<int64_t>(0, y0 - radius);
    const int64_t entry_y1 = std::min<int64_t>(full_h, y1 + radius);
    if (pre_residual->ne[1] != entry_y1 - entry_y0 || conv_output->ne[1] != pre_residual->ne[1])
        throw std::runtime_error("streamed ConvStack residual0 row mismatch");

    const std::string q0 = idx(idx(pfx + ".res_blocks", static_cast<int>(level)), 0);
    auto * skip = pre_residual;
    const bool projected_skip = maybe(q0 + ".skip_connection.weight") != nullptr;
    if (projected_skip) skip = conv1x1(skip, q0 + ".skip_connection");

    auto * x = conv_output;
    if (x->buffer != nullptr || (x->view_src && x->view_src->buffer != nullptr)) x = ggml_dup(c_, x);
    if (auto * bias = maybe(q0 + ".layers.2.bias")) {
        auto * br = ggml_reshape_4d(c_, bias, 1, 1, bias->ne[0], 1);
        x = add_inplace_safe(c_, x, cast_like(c_, br, x));
    }
    x = activate(x, cfg.activation);
    x = conv2d_replicate(x, q0 + ".layers.5", 1);
    x = add_inplace_safe(c_, x, skip);

    for (int j = 1; j < blocks; ++j) {
        const std::string q = idx(idx(pfx + ".res_blocks", static_cast<int>(level)), j);
        auto * block_skip = x;
        const bool block_projected = maybe(q + ".skip_connection.weight") != nullptr;
        if (block_projected) block_skip = conv1x1(block_skip, q + ".skip_connection");
        x = activate(x, cfg.activation, block_projected);
        x = conv2d_replicate(x, q + ".layers.2", 1);
        x = activate(x, cfg.activation);
        x = conv2d_replicate(x, q + ".layers.5", 1);
        x = add_inplace_safe(c_, x, block_skip);
    }

    const int64_t core_y = y0 - entry_y0;
    if (core_y < 0 || core_y + (y1 - y0) > x->ne[1])
        throw std::runtime_error("streamed ConvStack residual0 center crop");
    return ggml_view_4d(c_, x, x->ne[0], y1 - y0, x->ne[2], x->ne[3],
                        x->nb[1], x->nb[2], x->nb[3],
                        static_cast<size_t>(core_y) * x->nb[1]);
}


ggml_tensor * DenseBuilder::conv_stack_stream_phase_fused_tile(const ConvStackCfg & cfg,
                                                                 const std::string & pfx,
                                                                 size_t level,
                                                                 ggml_tensor * previous,
                                                                 ggml_tensor * feature,
                                                                 int64_t y0, int64_t y1) {
    auto * padded = conv_stack_stream_phase_prepare_entry_conv_tile(cfg, pfx, level, previous, y0, y1);
    const std::string rp = idx(pfx + ".resamplers", static_cast<int>(level - 1));
    const std::string post = rp + ".1";
    auto * k = weight(post + ".weight");
    auto * x = ggml_conv_2d(c_, k, padded, 1, 1, 0, 0, 1, 1);
    return conv_stack_stream_phase_finish_entry_conv_tile(cfg, pfx, level, x, feature, y0, y1);
}


ggml_tensor * DenseBuilder::conv_stack_stream_residual_tile(const ConvStackCfg & cfg,
                                                              const std::string & pfx,
                                                              size_t level,
                                                              ggml_tensor * pre_residual,
                                                              int64_t y0, int64_t y1) {
    if (!cfg.present || !pre_residual) throw std::runtime_error("streamed ConvStack missing pre-residual state: " + pfx);
    if (level >= cfg.dim_res.size()) throw std::runtime_error("streamed ConvStack residual level out of range: " + pfx);
    if (cfg.in_norm != "none" || cfg.hidden_norm != "none")
        throw std::runtime_error("streamed ConvStack residuals require none/none normalization: " + pfx);

    const int blocks = static_cast<int>(cfg.num_res_blocks[level]);
    const int64_t W = pre_residual->ne[0], H = pre_residual->ne[1];
    if (y0 < 0 || y1 <= y0 || y1 > H) throw std::runtime_error("streamed ConvStack residual row range");
    if (blocks <= 0) {
        return ggml_view_4d(c_, pre_residual, W, y1 - y0, pre_residual->ne[2], pre_residual->ne[3],
                            pre_residual->nb[1], pre_residual->nb[2], pre_residual->nb[3],
                            static_cast<size_t>(y0) * pre_residual->nb[1]);
    }

    // Every residual block contains two stride-1 3x3 convolutions, so the final
    // center rows depend on at most two source rows per block on either side.
    // Run the complete block sequence on a crop with that halo.  Replication at
    // an interior crop edge can only affect rows inside the discarded halo;
    // when the crop touches y=0/H it exactly matches true image-edge replication.
    const int64_t radius = 2LL * blocks;
    const int64_t src_y0 = std::max<int64_t>(0, y0 - radius);
    const int64_t src_y1 = std::min<int64_t>(H, y1 + radius);
    auto * x = ggml_view_4d(c_, pre_residual, W, src_y1 - src_y0,
                            pre_residual->ne[2], pre_residual->ne[3],
                            pre_residual->nb[1], pre_residual->nb[2], pre_residual->nb[3],
                            static_cast<size_t>(src_y0) * pre_residual->nb[1]);
    if (x->type != GGML_TYPE_F32) x = ggml_cast(c_, x, GGML_TYPE_F32);

    for (int j = 0; j < blocks; ++j) {
        const std::string q = idx(idx(pfx + ".res_blocks", static_cast<int>(level)), j);
        auto * skip = x;
        const bool projected_skip = maybe(q + ".skip_connection.weight") != nullptr;
        if (projected_skip) skip = conv1x1(skip, q + ".skip_connection");
        x = activate(x, cfg.activation, projected_skip);
        x = conv2d_replicate(x, q + ".layers.2", 1);
        x = activate(x, cfg.activation);
        x = conv2d_replicate(x, q + ".layers.5", 1);
        x = add_inplace_safe(c_, x, skip);
    }

    const int64_t core_y = y0 - src_y0;
    if (core_y < 0 || core_y + (y1 - y0) > x->ne[1])
        throw std::runtime_error("streamed ConvStack residual center crop");
    return ggml_view_4d(c_, x, W, y1 - y0, x->ne[2], x->ne[3],
                        x->nb[1], x->nb[2], x->nb[3],
                        static_cast<size_t>(core_y) * x->nb[1]);
}

ggml_tensor * DenseBuilder::build_encoder_feature(ggml_tensor * enc) {
    if (!enc) throw std::runtime_error("encoder feature requires encoder projection sum");
    auto * uv = uv_overrides_[0];
    if (!uv) uv = make_uv(bw_, bh_);
    auto * f0 = ggml_concat(c_, enc, uv, 2);
    // f0 is [W,H,C,1]. ggml_permute arguments are the NEW POSITION of
    // each OLD axis, so [W,H,C] -> [C,W,H] is (1,2,0), not (2,0,1).
    // The segmented runtime later reconstructs [W,H,C] from planar
    // [C,W*H] with the inverse (2,0,1). Keeping these as true inverses
    // is essential: the old (2,0,1) here materialized [H,C,W] and then
    // reshaped those bytes as [C,W*H], spatially/channel scrambling the
    // level-0 neck input and MoGe-3 conditioning map.
    auto * ef = ggml_cont(c_, ggml_permute(c_, f0, 1, 2, 0, 3));
    return ggml_reshape_2d(c_, ef, f0->ne[2], static_cast<int64_t>(bw_) * bh_);
}

ggml_tensor * DenseBuilder::build_scale_head(ggml_tensor * final_cls) {
    if (!a_.scale_present) return nullptr;
    if (!final_cls) throw std::runtime_error("scale head requires final normalized CLS token");
    ggml_tensor * scale = final_cls;
    int li = 0;
    for (size_t d = 0; d + 1 < a_.scale_dims.size(); ++d) {
        scale = linear(scale, "scale_head." + std::to_string(li));
        if (d + 2 < a_.scale_dims.size()) {
            scale = ggml_relu_inplace(c_, scale);
            li += 2;
        }
    }
    return scale;
}

ggml_tensor * DenseBuilder::backbone_norm(ggml_tensor * x) {
    return layer_norm(x, "encoder.backbone.norm", 1e-6f);
}

ggml_tensor * DenseBuilder::run_backbone_block_norm1(ggml_tensor * x, int i) {
    if (i < 0 || i >= a_.depth) throw std::runtime_error("invalid DINO backbone block index");
    return layer_norm(x, "encoder.backbone.blocks." + std::to_string(i) + ".norm1", 1e-6f);
}

ggml_tensor * DenseBuilder::run_backbone_block_qkv(ggml_tensor * normalized, int i) {
    if (i < 0 || i >= a_.depth) throw std::runtime_error("invalid DINO backbone block index");
    return linear(normalized, "encoder.backbone.blocks." + std::to_string(i) + ".attn.qkv");
}

std::array<ggml_tensor *, 2> DenseBuilder::run_backbone_block_prepare_kv(ggml_tensor * qkv, int i) {
    if (i < 0 || i >= a_.depth) throw std::runtime_error("invalid DINO backbone block index");
    const int64_t D = a_.embed_dim, H = a_.heads, hd = D / H, N = qkv->ne[1];
    auto head_view = [&](int which) {
        return ggml_view_4d(c_, qkv, hd, H, N, 1,
                            ggml_row_size(qkv->type, hd),
                            qkv->nb[1], qkv->nb[1] * N,
                            ggml_row_size(qkv->type, which * D));
    };
    auto * k = ggml_permute(c_, head_view(1), 0, 2, 1, 3);
    auto * v = ggml_permute(c_, head_view(2), 0, 2, 1, 3);
    // Pinned ggml 0.25.3 Metal accepts compact F16 K/V while Q must remain F32.
    k = ggml_cast(c_, k, GGML_TYPE_F16);
    v = ggml_cast(c_, v, GGML_TYPE_F16);
    return {k, v};
}

std::array<ggml_tensor *, 3> DenseBuilder::run_backbone_block_prepare_qkv_f16(ggml_tensor * qkv, int i) {
    if (i < 0 || i >= a_.depth) throw std::runtime_error("invalid DINO backbone block index");
    const int64_t D = a_.embed_dim, H = a_.heads, hd = D / H, N = qkv->ne[1];
    auto kv = run_backbone_block_prepare_kv(qkv, i);
    auto * qh = ggml_view_4d(c_, qkv, hd, H, N, 1,
                             ggml_row_size(qkv->type, hd),
                             qkv->nb[1], qkv->nb[1] * N,
                             0);
    auto * q = ggml_cast(c_, ggml_permute(c_, qh, 0, 2, 1, 3), GGML_TYPE_F16);
    return {q, kv[0], kv[1]};
}

ggml_tensor * DenseBuilder::run_backbone_block_flash(ggml_tensor * qkv, ggml_tensor * k,
                                                       ggml_tensor * v, int i) {
    if (i < 0 || i >= a_.depth) throw std::runtime_error("invalid DINO backbone block index");
    const int64_t D = a_.embed_dim, H = a_.heads, hd = D / H, N = qkv->ne[1];
    auto * qh = ggml_view_4d(c_, qkv, hd, H, N, 1,
                             ggml_row_size(qkv->type, hd),
                             qkv->nb[1], qkv->nb[1] * N,
                             0);
    auto * q = ggml_permute(c_, qh, 0, 2, 1, 3);
    auto * attn = ggml_flash_attn_ext(c_, q, k, v, nullptr,
                                      1.0f/std::sqrt(static_cast<float>(hd)),
                                      0.0f, 0.0f);
    ggml_prec_set_acc(attn, GGML_PREC_F32);
    return ggml_reshape_2d(c_, attn, D, N);
}

ggml_tensor * DenseBuilder::run_backbone_block_attn_project(ggml_tensor * residual,
                                                             ggml_tensor * attn, int i) {
    if (i < 0 || i >= a_.depth) throw std::runtime_error("invalid DINO backbone block index");
    const std::string p = "encoder.backbone.blocks." + std::to_string(i);
    attn = linear(attn, p + ".attn.proj");
    if (auto * gamma = maybe(p + ".ls1.gamma")) attn = mul_inplace_safe(c_, attn, cast_like(c_, gamma, attn));
    return destructive_residuals_ ? add_inplace_safe(c_, residual, attn) : add_safe(c_, residual, attn);
}

ggml_tensor * DenseBuilder::run_backbone_block_attention(ggml_tensor * x, int i) {
    auto * n = run_backbone_block_norm1(x, i);
    auto * qkv = run_backbone_block_qkv(n, i);
    auto kv = run_backbone_block_prepare_kv(qkv, i);
    auto * attn = run_backbone_block_flash(qkv, kv[0], kv[1], i);
    return run_backbone_block_attn_project(x, attn, i);
}

ggml_tensor * DenseBuilder::run_backbone_block_mlp(ggml_tensor * x, int i) {
    if (i < 0 || i >= a_.depth) throw std::runtime_error("invalid DINO backbone block index");
    const std::string p = "encoder.backbone.blocks." + std::to_string(i);
    auto * n = layer_norm(x, p + ".norm2", 1e-6f);
    ggml_tensor * ff = nullptr;
    if (maybe(p + ".mlp.fc1.weight")) {
        ff = ggml_gelu_inplace(c_, linear(n, p + ".mlp.fc1"));
        ff = linear(ff, p + ".mlp.fc2");
    } else {
        // ViT-Giant DINOv2 SwiGLU.
        auto * x12 = linear(n, p + ".mlp.w12");
        const int64_t hidden = x12->ne[0] / 2;
        auto * x1 = ggml_view_2d(c_, x12, hidden, x12->ne[1], x12->nb[1], 0);
        auto * x2 = ggml_view_2d(c_, x12, hidden, x12->ne[1], x12->nb[1], hidden * x12->nb[0]);
        ff = mul_inplace_safe(c_, ggml_silu_inplace(c_, x1), x2);
        ff = linear(ff, p + ".mlp.w3");
    }
    if (auto * gamma = maybe(p + ".ls2.gamma")) ff = mul_inplace_safe(c_, ff, cast_like(c_, gamma, ff));
    return destructive_residuals_ ? add_inplace_safe(c_, x, ff) : add_safe(c_, x, ff);
}

ggml_tensor * DenseBuilder::transformer_block(ggml_tensor * x, int i) {
    x = run_backbone_block_attention(x, i);
    return run_backbone_block_mlp(x, i);
}

ggml_tensor * DenseBuilder::build_backbone_input(ggml_tensor * image) {
    // The public runtime pre-resizes and ImageNet-normalizes the RGB input to
    // [14*bw,14*bh]. Keeping resize/normalization off-graph gives identical
    // input pixels across CPU/Vulkan/Metal.
    auto * im = image;
    if (im->ne[0] != bw_ * 14 || im->ne[1] != bh_ * 14 || im->ne[2] != 3)
        throw std::runtime_error("DenseBuilder expects pre-resized DINO RGB input");

    auto * patch = ggml_conv_2d_sk_p0(c_, weight("encoder.backbone.patch_embed.proj.weight"), im);
    if (auto * b = maybe("encoder.backbone.patch_embed.proj.bias")) {
        auto * br = ggml_reshape_4d(c_, b, 1, 1, b->ne[0], 1);
        patch = add_inplace_safe(c_, patch, cast_like(c_, br, patch));
    }
    auto * pcf = ggml_reshape_3d(c_, patch, bw_ * bh_, a_.embed_dim, 1);
    pcf = ggml_cont(c_, ggml_transpose(c_, pcf));
    pcf = ggml_reshape_2d(c_, pcf, a_.embed_dim, bw_ * bh_);

    auto * cls0 = weight("encoder.backbone.cls_token");
    auto * cls = ggml_reshape_2d(c_, cls0, a_.embed_dim, 1);
    if (cls->type != pcf->type) cls = ggml_cast(c_, cls, pcf->type);
    auto * x = ggml_concat(c_, cls, pcf, 1); // [D,1+bw*bh]

    ggml_tensor * pos_dyn = pos_embed_override_;
    if (!pos_dyn) {
        auto * pos = weight("encoder.backbone.pos_embed");
        const int64_t pn = pos->ne[1] - 1;
        const int64_t pm = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(pn))));
        if (pm * pm != pn) throw std::runtime_error("non-square DINO positional embedding");
        auto * pos_cls = ggml_view_2d(c_, pos, a_.embed_dim, 1, pos->nb[1], 0);
        auto * pos_patch = ggml_view_2d(c_, pos, a_.embed_dim, pn, pos->nb[1], pos->nb[1]);
        pos_patch = ggml_reshape_4d(c_, pos_patch, a_.embed_dim, pm, pm, 1);
        pos_patch = ggml_permute(c_, pos_patch, 2, 0, 1, 3);
        pos_patch = ggml_interpolate(c_, pos_patch, bw_, bh_, a_.embed_dim, 1, GGML_SCALE_MODE_BICUBIC);
        pos_patch = ggml_cont(c_, ggml_permute(c_, pos_patch, 1, 2, 0, 3));
        pos_patch = ggml_reshape_2d(c_, pos_patch, a_.embed_dim, bw_ * bh_);
        pos_dyn = ggml_concat(c_, pos_cls, pos_patch, 1);
    }
    if (pos_dyn->type != x->type) pos_dyn = ggml_cast(c_, pos_dyn, x->type);
    return add_inplace_safe(c_, x, pos_dyn);
}

ggml_tensor * DenseBuilder::run_backbone_blocks(ggml_tensor * x, int begin_block,
                                                  int end_block_inclusive) {
    if (begin_block < 0 || end_block_inclusive < begin_block || end_block_inclusive >= a_.depth)
        throw std::runtime_error("invalid DINO backbone block range");
    for (int i = begin_block; i <= end_block_inclusive; ++i) x = transformer_block(x, i);
    return x;
}

ggml_tensor * DenseBuilder::normalized_backbone_tap(ggml_tensor * x) {
    return backbone_norm(x);
}

ggml_tensor * DenseBuilder::project_backbone_tap(ggml_tensor * normalized, size_t projection_index) {
    if (!normalized) throw std::runtime_error("null DINO tap");
    if (projection_index >= a_.intermediate_layers.size())
        throw std::runtime_error("DINO projection index out of range");
    auto * pt = ggml_view_2d(c_, normalized, a_.embed_dim, bw_ * bh_, normalized->nb[1], normalized->nb[1]);
    pt = ggml_reshape_4d(c_, pt, a_.embed_dim, bw_, bh_, 1);
    pt = ggml_cont(c_, ggml_permute(c_, pt, 2, 0, 1, 3));
    return conv1x1(pt, "encoder.output_projections." + std::to_string(projection_index));
}

ggml_tensor * DenseBuilder::accumulate_backbone_projection(ggml_tensor * normalized, size_t projection_index,
                                                             ggml_tensor * previous_sum) {
    auto * pr = project_backbone_tap(normalized, projection_index);
    if (!previous_sum) return pr;
    if (previous_sum->type != pr->type) previous_sum = ggml_cast(c_, previous_sum, pr->type);
    return add_inplace_safe(c_, pr, previous_sum);
}

DenseOutputs DenseBuilder::build_heads_from_encoder(ggml_tensor * enc, ggml_tensor * final_cls) {
    if (!enc) throw std::runtime_error("DINO head requires encoder projection sum");
    DenseOutputs out;

    std::vector<ggml_tensor *> features(5, nullptr);
    for (int level = 0; level < 5; ++level) {
        auto * uv = uv_overrides_[static_cast<size_t>(level)];
        if (!uv) uv = make_uv(bw_ << level, bh_ << level);
        features[level] = level == 0 ? ggml_concat(c_, enc, uv, 2) : uv;
    }
    if (a_.version == 3) out.encoder_feature = build_encoder_feature(enc);

    std::vector<ggml_tensor *> neck_features;
    conv_stack(a_.neck, "neck", features, &neck_features);
    out.raw_points = a_.points.present ? conv_stack(a_.points, "points_head", neck_features) : nullptr;
    out.raw_normal = a_.normal.present ? conv_stack(a_.normal, "normal_head", neck_features) : nullptr;
    out.raw_mask = a_.mask.present ? conv_stack(a_.mask, "mask_head", neck_features) : nullptr;

    out.raw_scale = build_scale_head(final_cls);

    if (a_.version == 2 && out.raw_points)
        out.raw_points = ggml_interpolate(c_, out.raw_points, iw_, ih_, out.raw_points->ne[2], 1, GGML_SCALE_MODE_BILINEAR);
    if (out.raw_normal)
        out.raw_normal = ggml_interpolate(c_, out.raw_normal, iw_, ih_, out.raw_normal->ne[2], 1, GGML_SCALE_MODE_BILINEAR);
    if (out.raw_mask)
        out.raw_mask = ggml_interpolate(c_, out.raw_mask, iw_, ih_, out.raw_mask->ne[2], 1, GGML_SCALE_MODE_BILINEAR);
    return out;
}

DenseOutputs DenseBuilder::build_heads(const std::vector<ggml_tensor *> & selected) {
    if (selected.size() != a_.intermediate_layers.size())
        throw std::runtime_error("DINO head requires all intermediate taps");
    ggml_tensor * enc = nullptr;
    for (size_t i = 0; i < selected.size(); ++i) {
        auto * pr = project_backbone_tap(selected[i], i);
        enc = enc ? add_inplace_safe(c_, enc, pr) : pr;
    }
    auto * clsn = selected.back();
    clsn = ggml_view_2d(c_, clsn, a_.embed_dim, 1, clsn->nb[1], 0);
    return build_heads_from_encoder(enc, clsn);
}

DenseOutputs DenseBuilder::build(ggml_tensor * image) {
    auto * x = build_backbone_input(image);
    std::vector<ggml_tensor *> selected;
    selected.reserve(a_.intermediate_layers.size());
    size_t next_sel = 0;
    for (int i = 0; i < a_.depth; ++i) {
        x = transformer_block(x, i);
        if (next_sel < a_.intermediate_layers.size() && i == a_.intermediate_layers[next_sel]) {
            selected.push_back(backbone_norm(x));
            ++next_sel;
        }
    }
    if (selected.size() != a_.intermediate_layers.size())
        throw std::runtime_error("failed to collect DINO intermediate layers");
    return build_heads(selected);
}

} // namespace moge
