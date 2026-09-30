#pragma once

#include <array>

#include "mogg.hpp"
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace moge {

struct ConvStackCfg {
    bool present = false;
    std::vector<int64_t> dim_in, dim_res, dim_out, num_res_blocks;
    std::vector<std::string> resamplers;
    int hidden_mul = 1;
    std::string in_norm = "layer_norm";
    std::string hidden_norm = "group_norm";
    std::string activation = "relu";
};

struct Architecture {
    int version = 2;
    std::string remap = "linear";
    std::vector<int64_t> token_range;
    std::string backbone;
    int embed_dim = 0, depth = 0, heads = 0, encoder_out = 0;
    std::string ffn;
    std::vector<int64_t> intermediate_layers;
    ConvStackCfg neck, points, normal, mask;
    bool scale_present = false;
    std::vector<int64_t> scale_dims;
    bool refiner_present = false;
};

class WeightStore {
public:
    WeightStore(const MoggFile & file, ggml_backend_t backend);
    ~WeightStore();
    WeightStore(const WeightStore &) = delete;

    ggml_tensor * get(const std::string & name) const;
    ggml_tensor * maybe(const std::string & name) const;
    uint8_t flags(const std::string & name) const;
    int rotation_group() const { return rotation_group_; }
    bool has(const std::string & name) const { return maybe(name) != nullptr; }
    size_t resident_bytes() const { return buffer_ ? ggml_backend_buffer_get_size(buffer_) : 0; }
    bool mapped_weights() const { return mapped_weights_; }
    ggml_context * context() const { return ctx_; }

private:
    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors_;
    std::unordered_map<std::string, uint8_t> flags_;
    int rotation_group_ = 0;
    bool mapped_weights_ = false;
};

Architecture read_architecture(const MoggFile & file);

struct DenseStackStep {
    ggml_tensor * level_out = nullptr;
    ggml_tensor * next = nullptr;
};

struct DenseOutputs {
    ggml_tensor * raw_points = nullptr;  // [W,H,3,1]
    ggml_tensor * raw_normal = nullptr;  // [W,H,3,1]
    ggml_tensor * raw_mask = nullptr;    // [W,H,1,1]
    ggml_tensor * raw_scale = nullptr;   // [1,1]
    ggml_tensor * encoder_feature = nullptr; // [C,baseW*baseH], refiner conditioning, GPU-resident
};

class DenseBuilder {
public:
    DenseBuilder(ggml_context * ctx, const WeightStore & w, const Architecture & arch,
                 int image_w, int image_h, int base_w, int base_h,
                 ggml_tensor * pos_embed_override = nullptr,
                 const std::array<ggml_tensor *, 5> & uv_overrides = {},
                 bool destructive_residuals = false);

    // Monolithic compatibility path. GPU runtimes may instead compose these
    // phase helpers into smaller graphs with persistent state between phases.
    DenseOutputs build(ggml_tensor * image);
    ggml_tensor * build_backbone_input(ggml_tensor * image);
    ggml_tensor * run_backbone_blocks(ggml_tensor * x, int begin_block, int end_block_inclusive);
    // cached profiler hooks. These expose the two mathematically exact halves
    // of one DINO transformer block so the replay harness can measure attention
    // versus MLP time without changing production execution. transformer_block()
    // remains their composition in the original order.
    ggml_tensor * run_backbone_block_attention(ggml_tensor * x, int block);
    // attention-stage profiler hooks. These expose the exact production
    // attention sub-operations so cached replay can distinguish normalization,
    // QKV projection, K/V conversion, Flash Attention proper, and output
    // projection/residual cost. Production attention remains their composition.
    ggml_tensor * run_backbone_block_norm1(ggml_tensor * x, int block);
    ggml_tensor * run_backbone_block_qkv(ggml_tensor * normalized, int block);
    std::array<ggml_tensor *, 2> run_backbone_block_prepare_kv(ggml_tensor * qkv, int block);
    std::array<ggml_tensor *, 3> run_backbone_block_prepare_qkv_f16(ggml_tensor * qkv, int block);
    ggml_tensor * run_backbone_block_flash(ggml_tensor * qkv, ggml_tensor * k, ggml_tensor * v, int block);
    ggml_tensor * run_backbone_block_attn_project(ggml_tensor * residual, ggml_tensor * attn, int block);
    ggml_tensor * run_backbone_block_mlp(ggml_tensor * x, int block);
    ggml_tensor * normalized_backbone_tap(ggml_tensor * x);
    // Project one normalized DINO tap immediately into the smaller MoGe encoder
    // space. Segmented execution can accumulate these projections and avoid
    // keeping four full [embed_dim, tokens] taps live across the backbone.
    ggml_tensor * project_backbone_tap(ggml_tensor * normalized, size_t projection_index);
    ggml_tensor * accumulate_backbone_projection(ggml_tensor * normalized, size_t projection_index,
                                                  ggml_tensor * previous_sum);
    // Build spatial heads from an already-summed encoder projection and the
    // final normalized CLS token. This is the liveness-optimized segmented
    // boundary used by the runtime.
    DenseOutputs build_heads_from_encoder(ggml_tensor * encoder_sum, ggml_tensor * final_cls);
    // Execute exactly one ConvStack level. This is the segmented decoder primitive:
    // `x` is the previous level state before this level's resampling (null for
    // level 0), while `feature` is the same-level skip/input feature. `next` is
    // the post-residual state that the following level will resample; it is null
    // on the final level.
    DenseStackStep conv_stack_level(const ConvStackCfg & cfg, const std::string & pfx,
                                    size_t level, ggml_tensor * x, ggml_tensor * feature);
    // Final high-resolution decoder level split primitive. The caller performs
    // the 2x bilinear resize once into persistent staging storage, then invokes
    // this for disjoint output-row tiles. This keeps replicate-pad + im2col
    // scratch bounded by tile height instead of the full ~0.9 Mpix raster.
    ggml_tensor * upsample2_bilinear(ggml_tensor * x);
    ggml_tensor * conv_stack_final_level_tile(const ConvStackCfg & cfg, const std::string & pfx,
                                              size_t level, ggml_tensor * upsampled,
                                              ggml_tensor * feature, int64_t y0, int64_t y1);
    // fully streamed final-level primitive. Upsample only the low-resolution
    // source rows needed by this output stripe (plus exact bilinear/3x3 halo),
    // then run the trailing 3x3 and pointwise skip/output projection locally.
    // This removes the full-frame 64-channel high-resolution staging tensor.
    ggml_tensor * conv_stack_final_level_fused_tile(const ConvStackCfg & cfg, const std::string & pfx,
                                                    size_t level, ggml_tensor * previous,
                                                    ggml_tensor * feature, int64_t y0, int64_t y1);
    // diagnostic hooks around the final-level trailing 3x3 convolution.
    // prepare() returns the exact padded F32 stripe consumed by that 3x3; finish()
    // applies the original bias, same-level feature projection, and output 1x1.
    // These hooks let cached replay substitute only the 3x3 implementation.
    ggml_tensor * conv_stack_final_level_prepare_conv_tile(const ConvStackCfg & cfg, const std::string & pfx,
                                                           size_t level, ggml_tensor * previous,
                                                           int64_t y0, int64_t y1);
    ggml_tensor * conv_stack_final_level_finish_conv_tile(const ConvStackCfg & cfg, const std::string & pfx,
                                                          size_t level, ggml_tensor * conv_output,
                                                          ggml_tensor * feature, int64_t y0, int64_t y1);
    // phase-level streamed intermediate decoder primitive.  The expensive
    // level is split at a real scheduler boundary: first materialize only the
    // cheap upsample/transpose result, then run the resampler's trailing 3x3
    // convolution by row stripe into a persistent pre-residual state.  Finally
    // execute the complete residual sequence for each stripe with a receptive-
    // field halo and write only the exact center rows to the persistent output.
    // This is exact only when both residual norms are spatially independent
    // (currently `none` in released MoGe-2/3 decoder configs); callers must gate
    // other norm modes rather than treating tile boundaries as image edges.
    ggml_tensor * conv_stack_stream_prepare_resample(const ConvStackCfg & cfg, const std::string & pfx,
                                                     size_t level, ggml_tensor * previous);
    ggml_tensor * conv_stack_stream_entry_tile(const ConvStackCfg & cfg, const std::string & pfx,
                                               size_t level, ggml_tensor * prepared,
                                               ggml_tensor * feature, int64_t y0, int64_t y1);
    // fused stripe primitive. Compute the resampler trailing 3x3 over the
    // residual receptive-field halo, run the complete residual stack locally,
    // then return only the requested center rows. This removes the full-
    // frame pre-residual staging tensor while preserving exact edge semantics.
    ggml_tensor * conv_stack_stream_fused_tile(const ConvStackCfg & cfg, const std::string & pfx,
                                               size_t level, ggml_tensor * prepared,
                                               ggml_tensor * feature, int64_t y0, int64_t y1);
    // stage-free intermediate primitive for converter phase-packed 2x
    // ConvTranspose resamplers. Slice only the low-resolution source rows that
    // can influence this output stripe, run the pointwise phase projection +
    // pixel shuffle locally, then immediately execute the trailing 3x3 and the
    // complete residual stack. This removes the full-frame `prepared` arena.
    ggml_tensor * conv_stack_stream_phase_fused_tile(const ConvStackCfg & cfg, const std::string & pfx,
                                                     size_t level, ggml_tensor * previous,
                                                     ggml_tensor * feature, int64_t y0, int64_t y1);
    // diagnostic split for stage-free intermediate levels. prepare() stops
    // immediately before the resampler entry 3x3 over the exact residual halo;
    // finish() applies bias/feature projection, the untouched residual stack, and
    // returns the requested center rows. This lets cached replay substitute only
    // the entry 3x3 with MPSGraph without changing the rest of level 3.
    ggml_tensor * conv_stack_stream_phase_prepare_entry_conv_tile(const ConvStackCfg & cfg, const std::string & pfx,
                                                                   size_t level, ggml_tensor * previous,
                                                                   int64_t y0, int64_t y1);
    ggml_tensor * conv_stack_stream_phase_finish_entry_conv_tile(const ConvStackCfg & cfg, const std::string & pfx,
                                                                  size_t level, ggml_tensor * conv_output,
                                                                  ggml_tensor * feature, int64_t y0, int64_t y1);
    // cached diagnostic split for the first level-3 residual 3x3. The first
    // helper stops after entry bias + same-level feature merge, yielding the
    // exact pre-residual halo. The second pair brackets only residual block 0
    // layers.2 (the first 3x3), so MPSGraph can replace that convolution while
    // the identity skip, layers.5, remaining residual blocks and center crop
    // remain unchanged ggml math.
    ggml_tensor * conv_stack_stream_phase_finish_entry_pre_residual_tile(const ConvStackCfg & cfg, const std::string & pfx,
                                                                          size_t level, ggml_tensor * conv_output,
                                                                          ggml_tensor * feature, int64_t y0, int64_t y1);
    ggml_tensor * conv_stack_stream_residual0_prepare_conv_tile(const ConvStackCfg & cfg, const std::string & pfx,
                                                                 size_t level, ggml_tensor * pre_residual);
    ggml_tensor * conv_stack_stream_residual0_finish_conv_tile(const ConvStackCfg & cfg, const std::string & pfx,
                                                                size_t level, ggml_tensor * pre_residual,
                                                                ggml_tensor * conv_output, int64_t full_h,
                                                                int64_t y0, int64_t y1);
    ggml_tensor * conv_stack_stream_residual_tile(const ConvStackCfg & cfg, const std::string & pfx,
                                                  size_t level, ggml_tensor * pre_residual,
                                                  int64_t y0, int64_t y1);
    // Build the sparse-conditioning tensor [C, baseW*baseH] without running
    // the neck or output heads.
    ggml_tensor * build_encoder_feature(ggml_tensor * encoder_sum);
    // Tiny metric-scale MLP isolated from the spatial decoder.
    ggml_tensor * build_scale_head(ggml_tensor * final_cls);
    DenseOutputs build_heads(const std::vector<ggml_tensor *> & selected);

private:
    ggml_context * c_;
    const WeightStore & w_;
    const Architecture & a_;
    int iw_, ih_, bw_, bh_;
    // Optional host-precomputed DINO positional embedding [D,1+bw*bh].
    // The runtime uses this for exact upstream +0.1 scale-factor interpolation
    // and to avoid an interpolation dispatch on every cached-graph execution.
    ggml_tensor * pos_embed_override_ = nullptr;
    std::array<ggml_tensor *, 5> uv_overrides_{};
    bool destructive_residuals_ = false;

    ggml_tensor * weight(const std::string & n) const { return w_.get(n); }
    ggml_tensor * maybe(const std::string & n) const { return w_.maybe(n); }
    ggml_tensor * maybe_hadamard_rotate(ggml_tensor * x, const std::string & weight_name);
    ggml_tensor * matmul_weight(ggml_tensor * x, const std::string & weight_name);
    ggml_tensor * linear(ggml_tensor * x, const std::string & pfx);
    ggml_tensor * layer_norm(ggml_tensor * x, const std::string & pfx, float eps=1e-6f);
    ggml_tensor * affine_group_norm(ggml_tensor * x, const std::string & pfx, int groups, float eps=1e-5f);
    ggml_tensor * conv1x1(ggml_tensor * x, const std::string & pfx);
    ggml_tensor * conv2d_replicate(ggml_tensor * x, const std::string & pfx, int pad=1);
    ggml_tensor * activate(ggml_tensor * x, const std::string & kind, bool inplace = true);
    ggml_tensor * make_uv(int width, int height);
    ggml_tensor * conv_stack(const ConvStackCfg & cfg, const std::string & pfx,
                             const std::vector<ggml_tensor *> & input,
                             std::vector<ggml_tensor *> * all_outputs=nullptr);
    ggml_tensor * resample(ggml_tensor * x, const std::string & pfx, const std::string & type);
    ggml_tensor * pixel_shuffle2(ggml_tensor * x);
    ggml_tensor * phase_convtranspose2(ggml_tensor * x, const std::string & pfx);
    ggml_tensor * transformer_block(ggml_tensor * x, int i);
    ggml_tensor * backbone_norm(ggml_tensor * x);
};

} // namespace moge
