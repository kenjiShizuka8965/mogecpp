#pragma once

#include "moge_ggml/moge.hpp"

#include <string>
#include <vector>

namespace moge {

// These helpers are exposed internally so the validation dumper can capture
// exactly the same pre-camera outputs that postprocess() consumes.  They mirror
// MoGeModel.forward()'s output remap and mask sigmoid respectively.
void remap_points_inplace(std::vector<float> & points, const std::string & kind);
std::vector<float> mask_probabilities(const std::vector<float> & mask_raw,
                                      bool mask_is_logit = true);

// Convert a raw MoGe affine/factorized point map to public camera-space output.
// raw_points is HWC interleaved and raw_mask is a probability/logit selected by
// mask_is_logit. metric_scale is already exponentiated.
Result postprocess(std::vector<float> raw_points,
                   std::vector<float> raw_normal,
                   std::vector<float> raw_mask,
                   int width, int height,
                   const std::string & remap,
                   float metric_scale,
                   const InferOptions & options,
                   bool mask_is_logit = true);

std::vector<float> resize_hwc_bilinear(const std::vector<float> & src,
                                       int sw, int sh, int channels,
                                       int dw, int dh);

} // namespace moge
