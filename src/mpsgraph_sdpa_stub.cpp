#include "mpsgraph_sdpa.hpp"
#include <stdexcept>
namespace moge {
struct RuntimeMpsSdpa {};
bool runtime_mps_sdpa_supported() { return false; }
RuntimeMpsSdpa * runtime_mps_sdpa_create(int64_t, int64_t, int64_t, float) { throw std::runtime_error("MPSGraph SDPA is only available on Apple platforms"); }
RuntimeMpsSdpa * runtime_mps_sdpa_create_bound(int64_t, int64_t, int64_t, float, void *, void *, void *, void *) { throw std::runtime_error("MPSGraph SDPA is only available on Apple platforms"); }
void runtime_mps_sdpa_free(RuntimeMpsSdpa *) {}
double runtime_mps_sdpa_run_f16(RuntimeMpsSdpa *, const uint16_t *, const uint16_t *, const uint16_t *, uint16_t *) { throw std::runtime_error("MPSGraph SDPA is only available on Apple platforms"); }
double runtime_mps_sdpa_run_bound(RuntimeMpsSdpa *) { throw std::runtime_error("MPSGraph SDPA is only available on Apple platforms"); }
} // namespace moge
