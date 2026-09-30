// Copyright (c) 2026 Kenji8965
// SPDX-License-Identifier: MIT
// Programming and implementation assistance: ChatGPT 5.6 Sol (OpenAI)
// Human testing, validation, and release acceptance: Kenji8965
// Provided "AS IS"; use at your own risk. See LICENSE.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "sparse_metal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unistd.h>

namespace moge {
namespace {

static const char * kSparseMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint C;
    uint Cout;
    uint N;
    uint silu_input;
};

struct GatherParams { uint C; uint N; uint tile; uint begin; uint tap; uint silu_input; };
struct AccumParams { uint total; uint first; };
struct FinalParams { uint Cout; uint N; uint tile; uint begin; uint has_bias; uint add_residual; };

inline half silu_half(half x, uint enabled) {
    if (!enabled) return x;
    const float f = float(x);
    return half(f / (1.0f + exp(-f)));
}

kernel void sparse27_p8_f16w_f16x_f16y(
    device const half * x [[buffer(0)]],
    device const half * w [[buffer(1)]],
    device const int  * rows [[buffer(2)]],
    device const half * bias [[buffer(3)]],
    device half * y [[buffer(4)]],
    constant Params & p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
    constexpr uint P = 8;
    threadgroup half sx[P * 512];
    const uint n0 = tg.y * P;
    const uint co = tg.x * 128u + tid;
    float acc[P];
    const float b = (co < p.Cout && bias != nullptr) ? float(bias[co]) : 0.0f;
    for (uint q = 0; q < P; ++q) acc[q] = b;
    for (uint tap = 0; tap < 27u; ++tap) {
        const uint work = P * p.C;
        for (uint li = tid; li < work; li += 128u) {
            const uint q = li / p.C;
            const uint ci = li - q * p.C;
            const uint n = n0 + q;
            half v = half(0.0h);
            if (n < p.N) {
                const int src = rows[n * 27u + tap];
                if (src >= 0 && uint(src) < p.N) v = silu_half(x[uint(src) * p.C + ci], p.silu_input);
            }
            sx[q * p.C + ci] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (co < p.Cout) {
            const uint wb = co * (27u * p.C) + tap * p.C;
            uint ci = 0;
            for (; ci + 3u < p.C; ci += 4u) {
                const float4 wf = float4(*((device const half4 *)(w + wb + ci)));
                for (uint q = 0; q < P; ++q) {
                    const float4 xv = float4(*((threadgroup half4 *)(sx + q * p.C + ci)));
                    acc[q] += dot(xv, wf);
                }
            }
            for (; ci < p.C; ++ci) {
                const float wf = float(w[wb + ci]);
                for (uint q = 0; q < P; ++q) acc[q] = fma(float(sx[q * p.C + ci]), wf, acc[q]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (co < p.Cout) {
        for (uint q = 0; q < P; ++q) {
            const uint n = n0 + q;
            if (n >= p.N) break;
            y[n * p.Cout + co] = half(acc[q]);
        }
    }
}

kernel void sparse27_p8_f16w_f16x_f32y_residual(
    device const half * x [[buffer(0)]],
    device const half * w [[buffer(1)]],
    device const int  * rows [[buffer(2)]],
    device const half * bias [[buffer(3)]],
    device float * residual_y [[buffer(4)]],
    constant Params & p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
    constexpr uint P = 8;
    threadgroup half sx[P * 512];
    const uint n0 = tg.y * P;
    const uint co = tg.x * 128u + tid;
    float acc[P];
    const float b = (co < p.Cout && bias != nullptr) ? float(bias[co]) : 0.0f;
    for (uint q = 0; q < P; ++q) acc[q] = b;
    for (uint tap = 0; tap < 27u; ++tap) {
        const uint work = P * p.C;
        for (uint li = tid; li < work; li += 128u) {
            const uint q = li / p.C;
            const uint ci = li - q * p.C;
            const uint n = n0 + q;
            half v = half(0.0h);
            if (n < p.N) {
                const int src = rows[n * 27u + tap];
                if (src >= 0 && uint(src) < p.N) v = silu_half(x[uint(src) * p.C + ci], p.silu_input);
            }
            sx[q * p.C + ci] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (co < p.Cout) {
            const uint wb = co * (27u * p.C) + tap * p.C;
            uint ci = 0;
            for (; ci + 3u < p.C; ci += 4u) {
                const float4 wf = float4(*((device const half4 *)(w + wb + ci)));
                for (uint q = 0; q < P; ++q) {
                    const float4 xv = float4(*((threadgroup half4 *)(sx + q * p.C + ci)));
                    acc[q] += dot(xv, wf);
                }
            }
            for (; ci < p.C; ++ci) {
                const float wf = float(w[wb + ci]);
                for (uint q = 0; q < P; ++q) acc[q] = fma(float(sx[q * p.C + ci]), wf, acc[q]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (co < p.Cout) {
        for (uint q = 0; q < P; ++q) {
            const uint n = n0 + q;
            if (n >= p.N) break;
            const uint off = n * p.Cout + co;
            residual_y[off] = residual_y[off] + acc[q];
        }
    }
}

kernel void sparsecsr_p8_f16w_f16x_f16y(
    device const half * x [[buffer(0)]],
    device const half * w [[buffer(1)]],
    device const uint * offsets [[buffer(2)]],
    device const uint * edges [[buffer(3)]],
    device const half * bias [[buffer(4)]],
    device half * y [[buffer(5)]],
    constant Params & p [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
    constexpr uint P = 8;
    constexpr uint SRC_MASK = 0x07ffffffu;
    threadgroup half sx[P * 512];
    threadgroup int src_for_tap[P * 27];
    const uint n0 = tg.y * P;
    const uint co = tg.x * 128u + tid;
    float acc[P];
    const float b = (co < p.Cout && bias != nullptr) ? float(bias[co]) : 0.0f;
    for (uint q = 0; q < P; ++q) acc[q] = b;
    for (uint li = tid; li < P * 27u; li += 128u) src_for_tap[li] = -1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < P) {
        const uint n = n0 + tid;
        if (n < p.N) {
            for (uint e = offsets[n]; e < offsets[n + 1u]; ++e) {
                const uint code = edges[e];
                const uint tap = code >> 27u;
                if (tap < 27u) src_for_tap[tid * 27u + tap] = int(code & SRC_MASK);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint tap = 0; tap < 27u; ++tap) {
        bool any = false;
        for (uint q = 0; q < P; ++q) any = any || (src_for_tap[q * 27u + tap] >= 0);
        if (!any) continue;
        const uint work = P * p.C;
        for (uint li = tid; li < work; li += 128u) {
            const uint q = li / p.C;
            const uint ci = li - q * p.C;
            const int src = src_for_tap[q * 27u + tap];
            sx[q * p.C + ci] = src >= 0 ? silu_half(x[uint(src) * p.C + ci], p.silu_input) : half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (co < p.Cout) {
            const uint wb = co * (27u * p.C) + tap * p.C;
            uint ci = 0;
            for (; ci + 3u < p.C; ci += 4u) {
                const float4 wf = float4(*((device const half4 *)(w + wb + ci)));
                for (uint q = 0; q < P; ++q) {
                    if (src_for_tap[q * 27u + tap] < 0) continue;
                    const float4 xv = float4(*((threadgroup half4 *)(sx + q * p.C + ci)));
                    acc[q] += dot(xv, wf);
                }
            }
            for (; ci < p.C; ++ci) {
                const float wf = float(w[wb + ci]);
                for (uint q = 0; q < P; ++q) if (src_for_tap[q * 27u + tap] >= 0) acc[q] = fma(float(sx[q * p.C + ci]), wf, acc[q]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (co < p.Cout) {
        for (uint q = 0; q < P; ++q) {
            const uint n = n0 + q;
            if (n >= p.N) break;
            y[n * p.Cout + co] = half(acc[q]);
        }
    }
}

kernel void sparsecsr_p8_f16w_f16x_f32y_residual(
    device const half * x [[buffer(0)]],
    device const half * w [[buffer(1)]],
    device const uint * offsets [[buffer(2)]],
    device const uint * edges [[buffer(3)]],
    device const half * bias [[buffer(4)]],
    device float * residual_y [[buffer(5)]],
    constant Params & p [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
    constexpr uint P = 8;
    constexpr uint SRC_MASK = 0x07ffffffu;
    threadgroup half sx[P * 512];
    threadgroup int src_for_tap[P * 27];
    const uint n0 = tg.y * P;
    const uint co = tg.x * 128u + tid;
    float acc[P];
    const float b = (co < p.Cout && bias != nullptr) ? float(bias[co]) : 0.0f;
    for (uint q = 0; q < P; ++q) acc[q] = b;
    for (uint li = tid; li < P * 27u; li += 128u) src_for_tap[li] = -1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < P) {
        const uint n = n0 + tid;
        if (n < p.N) {
            for (uint e = offsets[n]; e < offsets[n + 1u]; ++e) {
                const uint code = edges[e];
                const uint tap = code >> 27u;
                if (tap < 27u) src_for_tap[tid * 27u + tap] = int(code & SRC_MASK);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint tap = 0; tap < 27u; ++tap) {
        bool any = false;
        for (uint q = 0; q < P; ++q) any = any || (src_for_tap[q * 27u + tap] >= 0);
        if (!any) continue;
        const uint work = P * p.C;
        for (uint li = tid; li < work; li += 128u) {
            const uint q = li / p.C;
            const uint ci = li - q * p.C;
            const int src = src_for_tap[q * 27u + tap];
            sx[q * p.C + ci] = src >= 0 ? silu_half(x[uint(src) * p.C + ci], p.silu_input) : half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (co < p.Cout) {
            const uint wb = co * (27u * p.C) + tap * p.C;
            uint ci = 0;
            for (; ci + 3u < p.C; ci += 4u) {
                const float4 wf = float4(*((device const half4 *)(w + wb + ci)));
                for (uint q = 0; q < P; ++q) {
                    if (src_for_tap[q * 27u + tap] < 0) continue;
                    const float4 xv = float4(*((threadgroup half4 *)(sx + q * p.C + ci)));
                    acc[q] += dot(xv, wf);
                }
            }
            for (; ci < p.C; ++ci) {
                const float wf = float(w[wb + ci]);
                for (uint q = 0; q < P; ++q) if (src_for_tap[q * 27u + tap] >= 0) acc[q] = fma(float(sx[q * p.C + ci]), wf, acc[q]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (co < p.Cout) {
        for (uint q = 0; q < P; ++q) {
            const uint n = n0 + q;
            if (n >= p.N) break;
            const uint off = n * p.Cout + co;
            residual_y[off] = residual_y[off] + acc[q];
        }
    }
}

kernel void gather_tap_f16(
    device const half * x [[buffer(0)]],
    device const int * rows [[buffer(1)]],
    device half * b [[buffer(2)]],
    constant GatherParams & p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = p.C * p.tile;
    if (gid >= total) return;
    const uint ci = gid / p.tile;
    const uint q = gid - ci * p.tile;
    const uint n = p.begin + q;
    half v = half(0.0h);
    if (n < p.N) {
        const int src = rows[n * 27u + p.tap];
        if (src >= 0 && uint(src) < p.N) v = silu_half(x[uint(src) * p.C + ci], p.silu_input);
    }
    b[ci * p.tile + q] = v;
}

kernel void accum_f16_to_f32(
    device const half * src [[buffer(0)]],
    device float * dst [[buffer(1)]],
    constant AccumParams & p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.total) return;
    const float v = float(src[gid]);
    dst[gid] = p.first ? v : (dst[gid] + v);
}

kernel void final_scatter_f16(
    device const float * accum [[buffer(0)]],
    device const half * bias [[buffer(1)]],
    device half * dst [[buffer(2)]],
    constant FinalParams & p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = p.Cout * p.tile;
    if (gid >= total) return;
    const uint co = gid / p.tile;
    const uint q = gid - co * p.tile;
    const uint n = p.begin + q;
    if (n >= p.N) return;
    float v = accum[co * p.tile + q];
    if (p.has_bias) v += float(bias[co]);
    dst[n * p.Cout + co] = half(v);
}

kernel void final_scatter_f32_residual(
    device const float * accum [[buffer(0)]],
    device const half * bias [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant FinalParams & p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = p.Cout * p.tile;
    if (gid >= total) return;
    const uint co = gid / p.tile;
    const uint q = gid - co * p.tile;
    const uint n = p.begin + q;
    if (n >= p.N) return;
    const uint off = n * p.Cout + co;
    float v = accum[co * p.tile + q];
    if (p.has_bias) v += float(bias[co]);
    if (p.add_residual) v += dst[off];
    dst[off] = v;
}
)METAL";

struct Params { uint32_t C, Cout, N, silu_input; };
struct GatherParams { uint32_t C, N, tile, begin, tap, silu_input; };
struct AccumParams { uint32_t total, first; };
struct FinalParams { uint32_t Cout, N, tile, begin, has_bias, add_residual; };

enum class Policy { P8, CSR, MPS };

struct WrappedTensor {
    id<MTLBuffer> buffer = nil;
    NSUInteger offset = 0;
};

static std::runtime_error metal_error(NSString * msg) {
    return std::runtime_error(msg ? msg.UTF8String : "Metal error");
}

static bool is_metal_backend(ggml_backend_t backend) {
    if (!backend) return false;
    const char * n = ggml_backend_name(backend);
    if (!n) return false;
    return std::strncmp(n, "MTL", 3) == 0 || std::strstr(n, "Metal") != nullptr;
}

static void require_tensor(ggml_tensor * t, ggml_type type, int64_t c, int64_t n, const char * what) {
    if (!t || t->type != type || t->ne[0] != c || t->ne[1] != n || t->nb[0] != ggml_type_size(type)) {
        throw std::runtime_error(std::string("Metal sparse hybrid tensor mismatch: ") + what);
    }
}

} // namespace

struct SparseMetalExecutor::Impl {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> lib = nil;
    id<MTLComputePipelineState> p8_half = nil;
    id<MTLComputePipelineState> p8_f32_res = nil;
    id<MTLComputePipelineState> csr_half = nil;
    id<MTLComputePipelineState> csr_f32_res = nil;
    id<MTLComputePipelineState> gather = nil;
    id<MTLComputePipelineState> accum = nil;
    id<MTLComputePipelineState> final_half = nil;
    id<MTLComputePipelineState> final_f32_res = nil;

    std::unordered_map<ggml_backend_buffer_t, id<MTLBuffer>> wrapped;

    const int32_t * index_key = nullptr;
    int64_t index_n = 0;
    Policy index_policy = Policy::P8;
    id<MTLBuffer> rows = nil;
    id<MTLBuffer> csr_offsets = nil;
    id<MTLBuffer> csr_edges = nil;

    id<MTLBuffer> mps_gather = nil;
    id<MTLBuffer> mps_temp = nil;
    id<MTLBuffer> mps_accum = nil;
    size_t mps_gather_bytes = 0;
    size_t mps_temp_bytes = 0;
    size_t mps_accum_bytes = 0;
    int mps_tile = 2048;

    Impl() = default;
    ~Impl() {
        for (auto & kv : wrapped) [kv.second release];
        if (rows) [rows release];
        if (csr_offsets) [csr_offsets release];
        if (csr_edges) [csr_edges release];
        if (mps_gather) [mps_gather release];
        if (mps_temp) [mps_temp release];
        if (mps_accum) [mps_accum release];
        if (final_f32_res) [final_f32_res release];
        if (final_half) [final_half release];
        if (accum) [accum release];
        if (gather) [gather release];
        if (csr_f32_res) [csr_f32_res release];
        if (csr_half) [csr_half release];
        if (p8_f32_res) [p8_f32_res release];
        if (p8_half) [p8_half release];
        if (lib) [lib release];
        if (queue) [queue release];
        if (dev) [dev release];
    }

    id<MTLComputePipelineState> pipeline(NSString * name) {
        NSError * err = nil;
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        if (!fn) throw metal_error([NSString stringWithFormat:@"missing Metal function %@", name]);
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
        [fn release];
        if (!ps) throw metal_error(err.localizedDescription);
        return ps;
    }

    WrappedTensor wrap(ggml_tensor * t) {
        if (!t || !t->buffer || !t->data) throw std::runtime_error("Metal sparse hybrid requires allocated tensor buffers");
        const auto gb = t->buffer;
        id<MTLBuffer> mb = nil;
        auto it = wrapped.find(gb);
        if (it == wrapped.end()) {
            void * base = ggml_backend_buffer_get_base(gb);
            const size_t size = ggml_backend_buffer_get_size(gb);
            if (!base || size == 0) throw std::runtime_error("Metal sparse hybrid requires shared ggml buffers");
            const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
            const size_t rounded = ((size + page - 1) / page) * page;
            mb = [dev newBufferWithBytesNoCopy:base length:rounded options:MTLResourceStorageModeShared deallocator:nil];
            if (!mb) throw std::runtime_error("failed to wrap ggml Metal buffer for sparse hybrid path");
            wrapped.emplace(gb, mb);
        } else {
            mb = it->second;
        }
        void * base = ggml_backend_buffer_get_base(gb);
        const ptrdiff_t off = static_cast<uint8_t *>(t->data) - static_cast<uint8_t *>(base);
        if (off < 0 || static_cast<size_t>(off) + ggml_nbytes(t) > ggml_backend_buffer_get_size(gb)) {
            throw std::runtime_error("Metal sparse hybrid tensor offset outside backend buffer");
        }
        return {mb, static_cast<NSUInteger>(off)};
    }

    Policy policy_for(int C) const {
        if (C <= 32) return Policy::P8;
        if (C <= 128) return Policy::CSR;
        return Policy::MPS;
    }

    void clear_indices() {
        if (rows) { [rows release]; rows = nil; }
        if (csr_offsets) { [csr_offsets release]; csr_offsets = nil; }
        if (csr_edges) { [csr_edges release]; csr_edges = nil; }
        index_key = nullptr;
        index_n = 0;
    }

    void prepare_indices(const std::vector<int32_t> & neighbors, int64_t N, Policy p) {
        if (neighbors.size() != static_cast<size_t>(27 * N)) throw std::runtime_error("Metal sparse hybrid neighbour shape mismatch");
        if (index_key == neighbors.data() && index_n == N && index_policy == p) return;
        clear_indices();
        index_key = neighbors.data(); index_n = N; index_policy = p;
        if (p == Policy::CSR) {
            std::vector<uint32_t> offsets(static_cast<size_t>(N) + 1);
            std::vector<uint32_t> edges;
            edges.reserve(static_cast<size_t>(N) * 12u);
            for (int64_t n = 0; n < N; ++n) {
                offsets[static_cast<size_t>(n)] = static_cast<uint32_t>(edges.size());
                for (uint32_t tap = 0; tap < 27; ++tap) {
                    const int32_t src = neighbors[static_cast<size_t>(n) * 27u + tap];
                    if (src < 0 || src >= N) continue;
                    if (static_cast<uint32_t>(src) > 0x07ffffffu) throw std::runtime_error("Metal sparse CSR source index overflow");
                    edges.push_back((tap << 27u) | static_cast<uint32_t>(src));
                }
            }
            offsets[static_cast<size_t>(N)] = static_cast<uint32_t>(edges.size());
            csr_offsets = [dev newBufferWithBytes:offsets.data() length:offsets.size()*sizeof(uint32_t) options:MTLResourceStorageModeShared];
            csr_edges = [dev newBufferWithBytes:edges.data() length:edges.size()*sizeof(uint32_t) options:MTLResourceStorageModeShared];
            if (!csr_offsets || !csr_edges) throw std::runtime_error("failed to allocate Metal sparse CSR indices");
        } else {
            rows = [dev newBufferWithBytes:neighbors.data() length:neighbors.size()*sizeof(int32_t) options:MTLResourceStorageModeShared];
            if (!rows) throw std::runtime_error("failed to allocate Metal sparse neighbour indices");
        }
    }

    void ensure_mps_scratch(int C, int Cout) {
        const size_t gb = static_cast<size_t>(C) * mps_tile * sizeof(uint16_t);
        const size_t tb = static_cast<size_t>(Cout) * mps_tile * sizeof(uint16_t);
        const size_t ab = static_cast<size_t>(Cout) * mps_tile * sizeof(float);
        if (gb > mps_gather_bytes) {
            if (mps_gather) [mps_gather release];
            mps_gather = [dev newBufferWithLength:gb options:MTLResourceStorageModePrivate];
            mps_gather_bytes = gb;
        }
        if (tb > mps_temp_bytes) {
            if (mps_temp) [mps_temp release];
            mps_temp = [dev newBufferWithLength:tb options:MTLResourceStorageModePrivate];
            mps_temp_bytes = tb;
        }
        if (ab > mps_accum_bytes) {
            if (mps_accum) [mps_accum release];
            mps_accum = [dev newBufferWithLength:ab options:MTLResourceStorageModePrivate];
            mps_accum_bytes = ab;
        }
        if (!mps_gather || !mps_temp || !mps_accum) throw std::runtime_error("failed to allocate MPS sparse scratch");
    }

    void encode_direct(id<MTLCommandBuffer> cb, Policy p, bool output_half,
                       ggml_tensor * x, ggml_tensor * w, ggml_tensor * bias, ggml_tensor * y,
                       int64_t C, int64_t N, bool silu_input) {
        WrappedTensor tx=wrap(x), tw=wrap(w), ty=wrap(y); WrappedTensor tb{};
        if (bias) tb=wrap(bias);
        id<MTLComputePipelineState> ps = nil;
        if (p == Policy::P8) ps = output_half ? p8_half : p8_f32_res;
        else ps = output_half ? csr_half : csr_f32_res;
        id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:ps];
        [enc setBuffer:tx.buffer offset:tx.offset atIndex:0];
        [enc setBuffer:tw.buffer offset:tw.offset atIndex:1];
        if (p == Policy::P8) {
            [enc setBuffer:rows offset:0 atIndex:2];
            [enc setBuffer:bias ? tb.buffer : nil offset:bias ? tb.offset : 0 atIndex:3];
            [enc setBuffer:ty.buffer offset:ty.offset atIndex:4];
            Params prm{static_cast<uint32_t>(C), static_cast<uint32_t>(C), static_cast<uint32_t>(N), static_cast<uint32_t>(silu_input)};
            [enc setBytes:&prm length:sizeof(prm) atIndex:5];
        } else {
            [enc setBuffer:csr_offsets offset:0 atIndex:2];
            [enc setBuffer:csr_edges offset:0 atIndex:3];
            [enc setBuffer:bias ? tb.buffer : nil offset:bias ? tb.offset : 0 atIndex:4];
            [enc setBuffer:ty.buffer offset:ty.offset atIndex:5];
            Params prm{static_cast<uint32_t>(C), static_cast<uint32_t>(C), static_cast<uint32_t>(N), static_cast<uint32_t>(silu_input)};
            [enc setBytes:&prm length:sizeof(prm) atIndex:6];
        }
        [enc dispatchThreadgroups:MTLSizeMake((static_cast<NSUInteger>(C)+127)/128,(static_cast<NSUInteger>(N)+7)/8,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [enc endEncoding];
    }

    void encode_mps_conv(id<MTLCommandBuffer> cb, bool output_half,
                         ggml_tensor * x, ggml_tensor * w, ggml_tensor * bias, ggml_tensor * y,
                         int C, int N, bool silu_input) {
        ensure_mps_scratch(C,C);
        WrappedTensor tx=wrap(x), tw=wrap(w), ty=wrap(y); WrappedTensor tb{};
        if (bias) tb=wrap(bias);
        const NSUInteger row_bytes_w = static_cast<NSUInteger>(27*C*sizeof(uint16_t));
        const NSUInteger row_bytes_b = static_cast<NSUInteger>(mps_tile*sizeof(uint16_t));
        MPSMatrixDescriptor * dw=[MPSMatrixDescriptor matrixDescriptorWithRows:C columns:C rowBytes:row_bytes_w dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor * db=[MPSMatrixDescriptor matrixDescriptorWithRows:C columns:mps_tile rowBytes:row_bytes_b dataType:MPSDataTypeFloat16];
        MPSMatrix * B=[[MPSMatrix alloc] initWithBuffer:mps_gather descriptor:db];
        MPSMatrix * T=[[MPSMatrix alloc] initWithBuffer:mps_temp descriptor:db];
        MPSMatrixMultiplication * mm=[[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO resultRows:C resultColumns:mps_tile interiorColumns:C alpha:1.0 beta:0.0];
        std::vector<MPSMatrix *> W; W.reserve(27);
        for (int tap=0;tap<27;++tap) {
            const NSUInteger off=tw.offset + static_cast<NSUInteger>(tap*C*sizeof(uint16_t));
            W.push_back([[MPSMatrix alloc] initWithBuffer:tw.buffer offset:off descriptor:dw]);
        }
        for (int begin=0; begin<N; begin+=mps_tile) {
            for (int tap=0; tap<27; ++tap) {
                id<MTLComputeCommandEncoder> ge=[cb computeCommandEncoder];
                [ge setComputePipelineState:gather]; [ge setBuffer:tx.buffer offset:tx.offset atIndex:0]; [ge setBuffer:rows offset:0 atIndex:1]; [ge setBuffer:mps_gather offset:0 atIndex:2];
                GatherParams gp{static_cast<uint32_t>(C),static_cast<uint32_t>(N),static_cast<uint32_t>(mps_tile),static_cast<uint32_t>(begin),static_cast<uint32_t>(tap),static_cast<uint32_t>(silu_input)};
                [ge setBytes:&gp length:sizeof(gp) atIndex:3];
                const NSUInteger total=static_cast<NSUInteger>(C)*mps_tile; const NSUInteger th=std::min<NSUInteger>(256,gather.maxTotalThreadsPerThreadgroup);
                [ge dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(th,1,1)]; [ge endEncoding];
                [mm encodeToCommandBuffer:cb leftMatrix:W[static_cast<size_t>(tap)] rightMatrix:B resultMatrix:T];
                id<MTLComputeCommandEncoder> ae=[cb computeCommandEncoder]; [ae setComputePipelineState:accum]; [ae setBuffer:mps_temp offset:0 atIndex:0]; [ae setBuffer:mps_accum offset:0 atIndex:1];
                AccumParams ap{static_cast<uint32_t>(static_cast<size_t>(C)*mps_tile),static_cast<uint32_t>(tap==0)}; [ae setBytes:&ap length:sizeof(ap) atIndex:2];
                const NSUInteger atotal=static_cast<NSUInteger>(C)*mps_tile; const NSUInteger ath=std::min<NSUInteger>(256,accum.maxTotalThreadsPerThreadgroup);
                [ae dispatchThreads:MTLSizeMake(atotal,1,1) threadsPerThreadgroup:MTLSizeMake(ath,1,1)]; [ae endEncoding];
            }
            id<MTLComputeCommandEncoder> fe=[cb computeCommandEncoder]; [fe setComputePipelineState:output_half ? final_half : final_f32_res];
            [fe setBuffer:mps_accum offset:0 atIndex:0]; [fe setBuffer:bias ? tb.buffer : nil offset:bias ? tb.offset : 0 atIndex:1]; [fe setBuffer:ty.buffer offset:ty.offset atIndex:2];
            FinalParams fp{static_cast<uint32_t>(C),static_cast<uint32_t>(N),static_cast<uint32_t>(mps_tile),static_cast<uint32_t>(begin),static_cast<uint32_t>(bias?1:0),static_cast<uint32_t>(output_half?0:1)};
            [fe setBytes:&fp length:sizeof(fp) atIndex:3];
            const NSUInteger ftotal=static_cast<NSUInteger>(C)*mps_tile;
            id<MTLComputePipelineState> fps = output_half ? final_half : final_f32_res;
            const NSUInteger fth=std::min<NSUInteger>(256,fps.maxTotalThreadsPerThreadgroup);
            [fe dispatchThreads:MTLSizeMake(ftotal,1,1) threadsPerThreadgroup:MTLSizeMake(fth,1,1)]; [fe endEncoding];
        }
        for (auto * m:W) [m release]; [mm release]; [B release]; [T release];
    }
};

SparseMetalExecutor::SparseMetalExecutor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SparseMetalExecutor::~SparseMetalExecutor() = default;

std::unique_ptr<SparseMetalExecutor> SparseMetalExecutor::create(ggml_backend_t backend) {
    @autoreleasepool {
        if (!is_metal_backend(backend)) return nullptr;
        auto impl=std::make_unique<Impl>();
        impl->dev=MTLCreateSystemDefaultDevice();
        if (!impl->dev) return nullptr;
        impl->queue=[impl->dev newCommandQueue];
        if (!impl->queue) throw std::runtime_error("failed to create sparse hybrid Metal command queue");
        NSError * err=nil;
        impl->lib=[impl->dev newLibraryWithSource:[NSString stringWithUTF8String:kSparseMetalSource] options:nil error:&err];
        if (!impl->lib) throw metal_error(err.localizedDescription);
        impl->p8_half=impl->pipeline(@"sparse27_p8_f16w_f16x_f16y");
        impl->p8_f32_res=impl->pipeline(@"sparse27_p8_f16w_f16x_f32y_residual");
        impl->csr_half=impl->pipeline(@"sparsecsr_p8_f16w_f16x_f16y");
        impl->csr_f32_res=impl->pipeline(@"sparsecsr_p8_f16w_f16x_f32y_residual");
        impl->gather=impl->pipeline(@"gather_tap_f16");
        impl->accum=impl->pipeline(@"accum_f16_to_f32");
        impl->final_half=impl->pipeline(@"final_scatter_f16");
        impl->final_f32_res=impl->pipeline(@"final_scatter_f32_residual");
        return std::unique_ptr<SparseMetalExecutor>(new SparseMetalExecutor(std::move(impl)));
    }
}

bool SparseMetalExecutor::available() const { return impl_ && impl_->dev && impl_->queue; }

const char * SparseMetalExecutor::backend_policy_for_channels(int channels) const {
    if (!impl_) return "unavailable";
    switch (impl_->policy_for(channels)) {
        case Policy::P8: return "p8-f16";
        case Policy::CSR: return "csr-p8-f16";
        case Policy::MPS: return "mps-tap-stream";
    }
    return "unknown";
}

double SparseMetalExecutor::resblock_inplace(
        int level,
        ggml_tensor * norm_silu_f16,
        ggml_tensor * conv1_f16,
        ggml_tensor * state_f32,
        const std::vector<int32_t> & neighbors,
        ggml_tensor * conv1_weight,
        ggml_tensor * conv1_bias,
        ggml_tensor * conv2_weight,
        ggml_tensor * conv2_bias) {
    @autoreleasepool {
        if (!available()) throw std::runtime_error("Metal sparse hybrid executor unavailable");
        const int64_t C=state_f32->ne[0], N=state_f32->ne[1];
        if (C <= 0 || C > 512 || N <= 0 || conv1_weight->ne[0] != 27*C || conv1_weight->ne[1] != C || conv2_weight->ne[0] != 27*C || conv2_weight->ne[1] != C) {
            throw std::runtime_error("Metal sparse hybrid supports released identity-channel sparse residual blocks only");
        }
        require_tensor(norm_silu_f16,GGML_TYPE_F16,C,N,"norm scratch");
        require_tensor(conv1_f16,GGML_TYPE_F16,C,N,"conv1 scratch");
        require_tensor(state_f32,GGML_TYPE_F32,C,N,"state");
        if (conv1_weight->type != GGML_TYPE_F16 || conv2_weight->type != GGML_TYPE_F16 ||
            (conv1_bias && conv1_bias->type != GGML_TYPE_F16) || (conv2_bias && conv2_bias->type != GGML_TYPE_F16)) {
            throw std::runtime_error("Metal sparse hybrid currently requires F16 sparse convolution weights/biases");
        }
        const Policy p=impl_->policy_for(static_cast<int>(C));
        impl_->prepare_indices(neighbors,N,p);
        const auto started=std::chrono::steady_clock::now();
        id<MTLCommandBuffer> cb=[impl_->queue commandBuffer];
        if (!cb) throw std::runtime_error("failed to create sparse hybrid Metal command buffer");
        if (p == Policy::MPS) {
            impl_->encode_mps_conv(cb,true,norm_silu_f16,conv1_weight,conv1_bias,conv1_f16,static_cast<int>(C),static_cast<int>(N),false);
            impl_->encode_mps_conv(cb,false,conv1_f16,conv2_weight,conv2_bias,state_f32,static_cast<int>(C),static_cast<int>(N),true);
        } else {
            impl_->encode_direct(cb,p,true,norm_silu_f16,conv1_weight,conv1_bias,conv1_f16,C,N,false);
            impl_->encode_direct(cb,p,false,conv1_f16,conv2_weight,conv2_bias,state_f32,C,N,true);
        }
        [cb commit]; [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) throw metal_error(cb.error.localizedDescription ?: @"sparse hybrid Metal command failed");
        (void)level;
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    }
}

} // namespace moge
