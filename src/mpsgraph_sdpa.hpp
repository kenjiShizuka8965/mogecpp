#pragma once
#include <cstdint>

namespace moge {

struct RuntimeMpsSdpa;

bool runtime_mps_sdpa_supported();
RuntimeMpsSdpa * runtime_mps_sdpa_create(int64_t heads, int64_t tokens, int64_t head_dim, float scale);
// Zero-copy production path. Buffers are borrowed id<MTLBuffer> handles with
// zero byte offset; Q/K/V are H,N,D F16 and output is written N,H,D F16.
RuntimeMpsSdpa * runtime_mps_sdpa_create_bound(int64_t heads, int64_t tokens, int64_t head_dim, float scale,
                                               void * q_mtl, void * k_mtl, void * v_mtl, void * out_mtl);
void runtime_mps_sdpa_free(RuntimeMpsSdpa * ctx);
// Inputs are contiguous H,N,D F16. Output is contiguous N,H,D F16.
// Returns wall-clock milliseconds including staging, execution, sync and layout conversion.
double runtime_mps_sdpa_run_f16(RuntimeMpsSdpa * ctx,
                                 const uint16_t * q_hnf,
                                 const uint16_t * k_hnf,
                                 const uint16_t * v_hnf,
                                 uint16_t * out_nhf);
// Executes the graph against buffers supplied to runtime_mps_sdpa_create_bound().
double runtime_mps_sdpa_run_bound(RuntimeMpsSdpa * ctx);

} // namespace moge
