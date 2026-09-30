#include "sparse_metal.hpp"

#include <stdexcept>

namespace moge {

struct SparseMetalExecutor::Impl {};

SparseMetalExecutor::SparseMetalExecutor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SparseMetalExecutor::~SparseMetalExecutor() = default;

std::unique_ptr<SparseMetalExecutor> SparseMetalExecutor::create(ggml_backend_t) {
    return nullptr;
}

bool SparseMetalExecutor::available() const { return false; }
const char * SparseMetalExecutor::backend_policy_for_channels(int) const { return "unavailable"; }

double SparseMetalExecutor::resblock_inplace(
    int, ggml_tensor *, ggml_tensor *, ggml_tensor *, const std::vector<int32_t> &,
    ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *) {
    throw std::runtime_error("Sparse Metal execution is only available on macOS Metal builds");
}

} // namespace moge
