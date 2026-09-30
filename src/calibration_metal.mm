// Copyright (c) 2026 Kenji8965
// SPDX-License-Identifier: MIT
// Programming and implementation assistance: ChatGPT 5.6 Sol (OpenAI)
// Human testing, validation, and release acceptance: Kenji8965
// Provided "AS IS"; use at your own risk. See LICENSE.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "calibration.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unistd.h>

namespace moge {
namespace {

static const char * kCalibrationMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Params {
    ulong ne1;
    ulong ne2;
    ulong ne3;
    ulong nb0;
    ulong nb1;
    ulong nb2;
    ulong nb3;
    uint channels;
    uint out_offset;
};

template<typename T>
kernel void channel_sumsq_impl(
    device const char * src_bytes [[buffer(0)]],
    device float * out [[buffer(1)]],
    constant Params & p [[buffer(2)]],
    threadgroup float * scratch [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 ntg [[threads_per_threadgroup]]) {
    const uint c = tg.x;
    const uint nth = ntg.x;
    if (c >= p.channels) return;
    const ulong rows = p.ne1 * p.ne2 * p.ne3;
    float acc = 0.0f;
    for (ulong r = tid; r < rows; r += nth) {
        ulong q = r;
        const ulong i1 = q % p.ne1; q /= p.ne1;
        const ulong i2 = q % p.ne2; q /= p.ne2;
        const ulong i3 = q;
        const ulong off = ulong(c) * p.nb0 + i1 * p.nb1 + i2 * p.nb2 + i3 * p.nb3;
        const device T * vptr = (device const T *)(src_bytes + off);
        const float v = float(*vptr);
        acc = fma(v, v, acc);
    }
    scratch[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[p.out_offset + c] = scratch[0];
}

typedef decltype(channel_sumsq_impl<float>) channel_sumsq_t;
template [[host_name("channel_sumsq_f32")]] kernel channel_sumsq_t channel_sumsq_impl<float>;
template [[host_name("channel_sumsq_f16")]] kernel channel_sumsq_t channel_sumsq_impl<half>;
)METAL";

struct Params {
    uint64_t ne1, ne2, ne3;
    uint64_t nb0, nb1, nb2, nb3;
    uint32_t channels, out_offset;
};

static bool is_metal_backend(ggml_backend_t backend) {
    if (!backend) return false;
    const char * n = ggml_backend_name(backend);
    return n && (std::strncmp(n, "MTL", 3) == 0 || std::strstr(n, "Metal") != nullptr);
}

static std::runtime_error metal_error(NSString * msg) {
    return std::runtime_error(msg ? msg.UTF8String : "Metal calibration error");
}

struct WrappedBuffer {
    id<MTLBuffer> buffer = nil;
    NSUInteger offset = 0;
};

} // namespace

struct CalibrationMetalReducer::Impl {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> lib = nil;
    id<MTLComputePipelineState> f32 = nil;
    id<MTLComputePipelineState> f16 = nil;
    bool verified_first = false;

    ~Impl() {
        if (f16) [f16 release];
        if (f32) [f32 release];
        if (lib) [lib release];
        if (queue) [queue release];
        if (dev) [dev release];
    }

    id<MTLComputePipelineState> make_pipeline(NSString * name) {
        NSError * err = nil;
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        if (!fn) throw metal_error([NSString stringWithFormat:@"missing calibration Metal function %@", name]);
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
        [fn release];
        if (!ps) throw metal_error(err.localizedDescription);
        return ps;
    }
};

CalibrationMetalReducer::CalibrationMetalReducer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CalibrationMetalReducer::~CalibrationMetalReducer() = default;

std::unique_ptr<CalibrationMetalReducer> CalibrationMetalReducer::create(ggml_backend_t backend) {
    if (!is_metal_backend(backend)) return {};
    auto impl = std::make_unique<Impl>();
    impl->dev = [MTLCreateSystemDefaultDevice() retain];
    if (!impl->dev) return {};
    impl->queue = [impl->dev newCommandQueue];
    if (!impl->queue) return {};
    NSError * err = nil;
    NSString * source = [NSString stringWithUTF8String:kCalibrationMetalSource];
    MTLCompileOptions * opts = [[MTLCompileOptions alloc] init];
    impl->lib = [impl->dev newLibraryWithSource:source options:opts error:&err];
    [opts release];
    if (!impl->lib) throw metal_error(err.localizedDescription);
    impl->f32 = impl->make_pipeline(@"channel_sumsq_f32");
    impl->f16 = impl->make_pipeline(@"channel_sumsq_f16");
    return std::unique_ptr<CalibrationMetalReducer>(new CalibrationMetalReducer(std::move(impl)));
}

bool CalibrationMetalReducer::available() const { return impl_ && impl_->dev && impl_->queue; }
bool CalibrationMetalReducer::verified_first() const { return impl_ && impl_->verified_first; }

std::vector<CalibrationReduction> CalibrationMetalReducer::reduce(const std::vector<CalibrationTap> & taps, bool verify_first) {
    if (!available()) throw std::runtime_error("native Metal calibration reducer unavailable");
    std::vector<CalibrationReduction> out;
    out.reserve(taps.size());
    uint64_t total_channels = 0;
    for (const auto & tap : taps) {
        auto * t = tap.activation;
        if (!t || !t->buffer || !t->data) throw std::runtime_error("calibration activation has no allocated backend buffer");
        if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16)
            throw std::runtime_error("Metal calibration currently expects F32/F16 activations for " + tap.weight_name);
        if (t->ne[0] <= 0 || t->ne[0] > INT32_MAX) throw std::runtime_error("invalid calibration channel count");
        total_channels += static_cast<uint64_t>(t->ne[0]);
        if (total_channels > UINT32_MAX) throw std::runtime_error("calibration phase output too large");
    }
    if (taps.empty()) return out;

    id<MTLBuffer> result = [impl_->dev newBufferWithLength:static_cast<NSUInteger>(total_channels * sizeof(float))
                                               options:MTLResourceStorageModeShared];
    if (!result) throw std::runtime_error("failed to allocate calibration result buffer");
    id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!cb || !enc) { [result release]; throw std::runtime_error("failed to create calibration command encoder"); }

    std::unordered_map<ggml_backend_buffer_t, id<MTLBuffer>> wrappers;
    uint32_t out_offset = 0;
    try {
        for (const auto & tap : taps) {
            ggml_tensor * t = tap.activation;
            const auto gb = t->buffer;
            id<MTLBuffer> mb = nil;
            auto it = wrappers.find(gb);
            if (it == wrappers.end()) {
                void * base = ggml_backend_buffer_get_base(gb);
                const size_t size = ggml_backend_buffer_get_size(gb);
                if (!base || size == 0) throw std::runtime_error("calibration requires shared Metal ggml buffers");
                const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
                const size_t rounded = ((size + page - 1) / page) * page;
                mb = [impl_->dev newBufferWithBytesNoCopy:base length:rounded options:MTLResourceStorageModeShared deallocator:nil];
                if (!mb) throw std::runtime_error("failed to wrap ggml Metal calibration buffer");
                wrappers.emplace(gb, mb);
            } else mb = it->second;
            void * base = ggml_backend_buffer_get_base(gb);
            const ptrdiff_t off = static_cast<uint8_t *>(t->data) - static_cast<uint8_t *>(base);
            if (off < 0) throw std::runtime_error("calibration tensor offset outside backend buffer");

            Params p{};
            p.ne1 = static_cast<uint64_t>(t->ne[1]); p.ne2 = static_cast<uint64_t>(t->ne[2]); p.ne3 = static_cast<uint64_t>(t->ne[3]);
            p.nb0 = static_cast<uint64_t>(t->nb[0]); p.nb1 = static_cast<uint64_t>(t->nb[1]); p.nb2 = static_cast<uint64_t>(t->nb[2]); p.nb3 = static_cast<uint64_t>(t->nb[3]);
            p.channels = static_cast<uint32_t>(t->ne[0]); p.out_offset = out_offset;
            auto ps = t->type == GGML_TYPE_F32 ? impl_->f32 : impl_->f16;
            [enc setComputePipelineState:ps];
            [enc setBuffer:mb offset:static_cast<NSUInteger>(off) atIndex:0];
            [enc setBuffer:result offset:0 atIndex:1];
            [enc setBytes:&p length:sizeof(p) atIndex:2];
            const NSUInteger nth = std::min<NSUInteger>(128, ps.maxTotalThreadsPerThreadgroup);
            [enc setThreadgroupMemoryLength:nth*sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(p.channels, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            out_offset += p.channels;
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) throw metal_error(cb.error.localizedDescription);

        const float * vals = static_cast<const float *>(result.contents);
        // Optional one-shot host cross-check requested by CalibrationOptions; it catches Metal
        // indexing/stride mistakes without returning full activations for every
        // layer or image.
        if (!impl_->verified_first && verify_first && !taps.empty()) {
            ggml_tensor * t = taps.front().activation;
            if (ggml_is_contiguous(t)) {
                const size_t nbytes = ggml_nbytes(t);
                std::vector<uint8_t> raw(nbytes);
                ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
                const size_t channels = static_cast<size_t>(t->ne[0]);
                const size_t rows = static_cast<size_t>(ggml_nrows(t));
                std::vector<double> ref(channels, 0.0);
                if (t->type == GGML_TYPE_F32) {
                    const float * x = reinterpret_cast<const float *>(raw.data());
                    for (size_t r = 0; r < rows; ++r)
                        for (size_t c = 0; c < channels; ++c) { const double v=x[r*channels+c]; ref[c]+=v*v; }
                } else {
                    const ggml_fp16_t * x = reinterpret_cast<const ggml_fp16_t *>(raw.data());
                    for (size_t r = 0; r < rows; ++r)
                        for (size_t c = 0; c < channels; ++c) { const double v=ggml_fp16_to_fp32(x[r*channels+c]); ref[c]+=v*v; }
                }
                double max_rel = 0.0, max_abs = 0.0;
                for (size_t c = 0; c < channels; ++c) {
                    const double got = vals[c], want = ref[c];
                    max_abs = std::max(max_abs, std::abs(got-want));
                    max_rel = std::max(max_rel, std::abs(got-want)/std::max(1e-12, std::abs(want)));
                }
                std::fprintf(stderr, "moge_calibration_verify: %s channels=%zu rows=%zu max_abs=%.6g max_rel=%.6g\n",
                             taps.front().weight_name.c_str(), channels, rows, max_abs, max_rel);
                if (!(max_rel < 5e-4 || max_abs < 5e-4))
                    throw std::runtime_error("Metal calibration reduction verification failed for " + taps.front().weight_name);
                impl_->verified_first = true;
            }
        }
        uint32_t cursor = 0;
        for (const auto & tap : taps) {
            CalibrationReduction r;
            r.weight_name = tap.weight_name;
            const size_t c = static_cast<size_t>(tap.activation->ne[0]);
            r.sumsq.assign(vals + cursor, vals + cursor + c);
            r.samples = static_cast<uint64_t>(tap.activation->ne[1]) * tap.activation->ne[2] * tap.activation->ne[3];
            cursor += static_cast<uint32_t>(c);
            out.push_back(std::move(r));
        }
    } catch (...) {
        for (auto & kv : wrappers) [kv.second release];
        [result release];
        throw;
    }
    for (auto & kv : wrappers) [kv.second release];
    [result release];
    return out;
}

} // namespace moge
