#include "calibration.hpp"

namespace moge {
struct CalibrationMetalReducer::Impl {};
CalibrationMetalReducer::CalibrationMetalReducer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CalibrationMetalReducer::~CalibrationMetalReducer() = default;
std::unique_ptr<CalibrationMetalReducer> CalibrationMetalReducer::create(ggml_backend_t) { return {}; }
bool CalibrationMetalReducer::available() const { return false; }
bool CalibrationMetalReducer::verified_first() const { return false; }
std::vector<CalibrationReduction> CalibrationMetalReducer::reduce(const std::vector<CalibrationTap> &, bool) { return {}; }
} // namespace moge
