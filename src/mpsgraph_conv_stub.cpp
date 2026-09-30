#include "mpsgraph_conv.hpp"
#include <stdexcept>
namespace moge {
bool runtime_mps_conv_supported() { return false; }
RuntimeMpsConv * runtime_mps_conv_create_bound(int64_t,int64_t,int64_t,int64_t,const uint16_t*,ggml_tensor*,ggml_tensor*) {
    throw std::runtime_error("MPSGraph convolution is only available on macOS");
}
void runtime_mps_conv_free(RuntimeMpsConv *) {}
double runtime_mps_conv_run_bound(RuntimeMpsConv *) { throw std::runtime_error("MPSGraph convolution is only available on macOS"); }
}
