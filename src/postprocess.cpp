// Copyright (c) 2026 Kenji8965
// SPDX-License-Identifier: MIT
// Programming and implementation assistance: ChatGPT 5.6 Sol (OpenAI)
// Human testing, validation, and release acceptance: Kenji8965
// Provided "AS IS"; use at your own risk. See LICENSE.

#include "postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <thread>
#include <cstdlib>
#include <cstring>

namespace moge {
namespace {

constexpr float kPi = 3.14159265358979323846f;

int post_threads(size_t work_items) {
    const char * env = std::getenv("MOGE_POST_THREADS");
    int n = 0;
    if (env && *env) {
        char * end = nullptr;
        const long v = std::strtol(env, &end, 10);
        if (!end || *end || v < 1 || v > 64) throw std::runtime_error("MOGE_POST_THREADS must be an integer in [1,64]");
        n = static_cast<int>(v);
    } else {
        const unsigned hw = std::thread::hardware_concurrency();
        n = static_cast<int>(hw ? std::min<unsigned>(8, hw) : 4);
    }
    if (work_items < 32768) return 1;
    return std::max(1, std::min<int>(n, static_cast<int>(work_items / 16384)));
}

template <class Fn>
void post_parallel_for(size_t begin, size_t end, Fn && fn) {
    const size_t n = end - begin;
    const int nth = post_threads(n);
    if (nth <= 1) { fn(begin, end); return; }
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(nth));
    for (int t = 0; t < nth; ++t) {
        const size_t b = begin + n * static_cast<size_t>(t) / static_cast<size_t>(nth);
        const size_t e = begin + n * static_cast<size_t>(t + 1) / static_cast<size_t>(nth);
        workers.emplace_back([&, b, e] { fn(b, e); });
    }
    for (auto & w : workers) w.join();
}

inline float sigmoid(float x) {
    if (x >= 0.0f) {
        const float e = std::exp(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = std::exp(x);
    return e / (1.0f + e);
}

} // namespace

void remap_points_inplace(std::vector<float> & p, const std::string & kind) {
    if (kind == "linear") return;
    if (kind != "sinh" && kind != "exp" && kind != "sinh_exp")
        throw std::runtime_error("unsupported MoGe output remap: " + kind);
    const size_t n = p.size() / 3;
    post_parallel_for(0, n, [&](size_t b, size_t e) {
        for (size_t px = b; px < e; ++px) {
            const size_t i = px * 3;
            float & x = p[i + 0];
            float & y = p[i + 1];
            float & z = p[i + 2];
            if (kind == "sinh") {
                x = std::sinh(x); y = std::sinh(y); z = std::sinh(z);
            } else if (kind == "exp") {
                z = std::exp(z); x *= z; y *= z;
            } else {
                x = std::sinh(x); y = std::sinh(y); z = std::exp(z);
            }
        }
    });
}

std::vector<float> mask_probabilities(const std::vector<float> & mask_raw,
                                      bool mask_is_logit) {
    std::vector<float> out(mask_raw.size());
    post_parallel_for(0, mask_raw.size(), [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) out[i] = mask_is_logit ? sigmoid(mask_raw[i]) : mask_raw[i];
    });
    return out;
}

namespace {

void normalize_normals(std::vector<float> & n) {
    const size_t count = n.size() / 3;
    post_parallel_for(0, count, [&](size_t b, size_t e) {
        for (size_t px = b; px < e; ++px) {
            const size_t i = px * 3;
            const double x = n[i], y = n[i+1], z = n[i+2];
            const double l = std::sqrt(x*x + y*y + z*z);
            if (l > 1e-12) {
                n[i] = static_cast<float>(x/l);
                n[i+1] = static_cast<float>(y/l);
                n[i+2] = static_cast<float>(z/l);
            } else {
                n[i] = n[i+1] = n[i+2] = 0.0f;
            }
        }
    });
}

struct Sample { double u, v, x, y, z; };

std::vector<Sample> lowres_samples(const std::vector<float> & points,
                                   const std::vector<uint8_t> * mask,
                                   int w, int h) {
    constexpr int OW = 64, OH = 64;
    const double ar = static_cast<double>(w) / h;
    const double den = std::sqrt(1.0 + ar*ar);
    const double span_x = ar / den;
    const double span_y = 1.0 / den;
    std::vector<Sample> out;
    out.reserve(OW * OH);
    // torch.interpolate(..., mode='nearest') maps output coordinate o to
    // floor(o * input/output) for the ordinary integer-ratio-free case.
    for (int oy = 0; oy < OH; ++oy) {
        const int sy = std::min(h - 1, (oy * h) / OH);
        for (int ox = 0; ox < OW; ++ox) {
            const int sx = std::min(w - 1, (ox * w) / OW);
            const size_t pi = static_cast<size_t>(sy) * w + sx;
            if (mask && !(*mask)[pi]) continue;
            const double u = ((2.0 * sx - (w - 1.0)) / w) * span_x;
            const double v = ((2.0 * sy - (h - 1.0)) / h) * span_y;
            out.push_back({u, v, points[3*pi], points[3*pi+1], points[3*pi+2]});
        }
    }
    return out;
}

// Small dependency-free LM solver. Upstream minimizes only the z-shift and
// analytically eliminates focal at every residual evaluation. Keep that exact
// objective here; the only deliberate difference is using an analytic Jacobian
// instead of SciPy's numerical one.
std::pair<double,double> solve_focal_shift(const std::vector<Sample> & s) {
    if (s.size() < 2) return {1.0, 0.0};

    struct Eval {
        double focal = 1.0;
        double cost = std::numeric_limits<double>::infinity();
        double grad = 0.0; // J^T r
        double hess = 0.0; // J^T J (Gauss-Newton)
        bool valid = false;
    };

    auto eval = [&](double sh, bool jacobian) -> Eval {
        // p = xy/(z+s), f(s) = <p,uv>/<p,p>.
        double num = 0.0, den = 0.0;
        double num_p = 0.0, den_p = 0.0;
        for (const auto & q : s) {
            const double d = q.z + sh;
            if (!std::isfinite(d) || std::abs(d) < 1e-10) return {};
            const double inv = 1.0 / d;
            const double px = q.x * inv, py = q.y * inv;
            num += px * q.u + py * q.v;
            den += px * px + py * py;
            if (jacobian) {
                const double inv2 = inv * inv;
                const double ppx = -q.x * inv2, ppy = -q.y * inv2;
                num_p += ppx * q.u + ppy * q.v;
                den_p += 2.0 * (px * ppx + py * ppy);
            }
        }
        if (!std::isfinite(den) || den <= 1e-20) return {};

        Eval e;
        e.focal = num / den;
        const double f_p = jacobian ? (num_p * den - num * den_p) / (den * den) : 0.0;
        e.cost = 0.0;
        for (const auto & q : s) {
            const double d = q.z + sh;
            if (!std::isfinite(d) || std::abs(d) < 1e-10) return {};
            const double inv = 1.0 / d;
            const double px = q.x * inv, py = q.y * inv;
            const double rx = e.focal * px - q.u;
            const double ry = e.focal * py - q.v;
            e.cost += rx * rx + ry * ry;
            if (jacobian) {
                const double inv2 = inv * inv;
                const double ppx = -q.x * inv2, ppy = -q.y * inv2;
                const double jx = f_p * px + e.focal * ppx;
                const double jy = f_p * py + e.focal * ppy;
                e.grad += jx * rx + jy * ry;
                e.hess += jx * jx + jy * jy;
            }
        }
        e.valid = std::isfinite(e.focal) && std::isfinite(e.cost) &&
                  (!jacobian || (std::isfinite(e.grad) && std::isfinite(e.hess)));
        return e;
    };

    double shift = 0.0;
    double lambda = 1e-3;
    Eval cur = eval(shift, true);
    if (!cur.valid) return {1.0, 0.0};

    for (int iter = 0; iter < 60; ++iter) {
        const double denom = cur.hess + lambda;
        if (!std::isfinite(denom) || denom <= 1e-24) break;
        double ds = -cur.grad / denom;

        // Do not let an LM proposal jump across z + shift == 0. SciPy will
        // naturally reject such singular residuals; clipping keeps this tiny
        // standalone implementation finite while preserving the same basin.
        double min_abs_d = std::numeric_limits<double>::infinity();
        for (const auto & q : s) min_abs_d = std::min(min_abs_d, std::abs(q.z + shift));
        if (std::isfinite(min_abs_d)) ds = std::clamp(ds, -0.45 * min_abs_d, 0.45 * min_abs_d);
        if (std::abs(ds) <= 1e-12 * std::max(1.0, std::abs(shift))) break;

        const double next_shift = shift + ds;
        Eval next = eval(next_shift, false);
        if (next.valid && next.cost < cur.cost) {
            const double old_cost = cur.cost;
            shift = next_shift;
            cur = eval(shift, true);
            if (!cur.valid) break;
            lambda = std::max(1e-12, lambda * 0.3);
            // Match upstream's deliberately loose ftol=1e-3 termination.
            if (std::abs(old_cost - cur.cost) <= 1e-3 * std::max(1.0, old_cost)) break;
        } else {
            lambda = std::min(1e12, lambda * 10.0);
        }
    }

    Eval out = eval(shift, false);
    if (!out.valid || !std::isfinite(shift)) return {1.0, 0.0};
    return {out.focal, shift};
}

double solve_shift_fixed_focal(const std::vector<Sample> & s, double f) {
    if (s.size() < 2) return 0.0;
    double sh=0, lambda=1e-3;
    for (int it=0; it<40; ++it) {
        double a=lambda,b=0,cost=0;
        for (const auto & q:s) {
            double d=q.z+sh;
            if (std::abs(d)<1e-7) d=std::copysign(1e-7,d==0?1.0:d);
            double inv=1/d, inv2=inv*inv;
            double rx=f*q.x*inv-q.u, ry=f*q.y*inv-q.v;
            double jx=-f*q.x*inv2, jy=-f*q.y*inv2;
            a+=jx*jx+jy*jy; b-=jx*rx+jy*ry; cost+=rx*rx+ry*ry;
        }
        double ds=b/a;
        double min_abs_d=std::numeric_limits<double>::infinity();
        for (const auto&q:s) min_abs_d=std::min(min_abs_d,std::abs(q.z+sh));
        ds=std::clamp(ds,-0.45*min_abs_d,0.45*min_abs_d);
        const double ns=sh+ds;
        double nc=0;
        for (const auto&q:s) {
            double d=q.z+ns;
            if (std::abs(d)<1e-7) d=std::copysign(1e-7,d==0?1.0:d);
            double rx=f*q.x/d-q.u, ry=f*q.y/d-q.v; nc+=rx*rx+ry*ry;
        }
        if (nc<cost) { sh=ns; lambda=std::max(1e-9,lambda*.3); if(std::abs(ds)<1e-7) break; }
        else lambda=std::min(1e9,lambda*10.0);
    }
    return std::isfinite(sh)?sh:0.0;
}

} // namespace

std::vector<float> resize_hwc_bilinear(const std::vector<float> & src,
                                       int sw, int sh, int c,
                                       int dw, int dh) {
    if (sw == dw && sh == dh) return src;
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0 || c <= 0 ||
        src.size() != static_cast<size_t>(sw)*sh*c) throw std::runtime_error("bad bilinear resize shape");
    std::vector<float> dst(static_cast<size_t>(dw)*dh*c);
    // PyTorch align_corners=False mapping.
    post_parallel_for(0, static_cast<size_t>(dh), [&](size_t yb, size_t ye) {
        for (size_t yu = yb; yu < ye; ++yu) {
            const int y = static_cast<int>(yu);
            const float fy=(y+0.5f)*sh/dh-0.5f;
            int y0=static_cast<int>(std::floor(fy)); float wy=fy-y0;
            if (y0<0) { y0=0; wy=0; } int y1=std::min(sh-1,y0+1);
            if (y0>=sh-1) { y0=sh-1; y1=y0; wy=0; }
            for (int x=0;x<dw;++x) {
                const float fx=(x+0.5f)*sw/dw-0.5f;
                int x0=static_cast<int>(std::floor(fx)); float wx=fx-x0;
                if (x0<0) { x0=0; wx=0; } int x1=std::min(sw-1,x0+1);
                if (x0>=sw-1) { x0=sw-1; x1=x0; wx=0; }
                for (int ch=0; ch<c; ++ch) {
                    auto at=[&](int yy,int xx){return src[(static_cast<size_t>(yy)*sw+xx)*c+ch];};
                    const float a=at(y0,x0)*(1-wx)+at(y0,x1)*wx;
                    const float b=at(y1,x0)*(1-wx)+at(y1,x1)*wx;
                    dst[(static_cast<size_t>(y)*dw+x)*c+ch]=a*(1-wy)+b*wy;
                }
            }
        }
    });
    return dst;
}

Result postprocess(std::vector<float> points,
                   std::vector<float> normal,
                   std::vector<float> mask_raw,
                   int width, int height,
                   const std::string & remap,
                   float metric_scale,
                   const InferOptions & options,
                   bool mask_is_logit) {
    const size_t n=static_cast<size_t>(width)*height;
    if (points.size()!=n*3) throw std::runtime_error("postprocess point shape mismatch");
    remap_points_inplace(points,remap);

    std::vector<uint8_t> mask(n,1);
    if (!mask_raw.empty()) {
        if(mask_raw.size()!=n) throw std::runtime_error("postprocess mask shape mismatch");
        const auto probability = mask_probabilities(mask_raw, mask_is_logit);
        for(size_t i=0;i<n;++i) mask[i]=probability[i]>0.5f;
    }
    if(!normal.empty()) {
        if(normal.size()!=n*3) throw std::runtime_error("postprocess normal shape mismatch");
        normalize_normals(normal);
    }

    const double ar=static_cast<double>(width)/height;
    const double diag=std::sqrt(1.0+ar*ar);
    auto samples=lowres_samples(points,mask_raw.empty()?nullptr:&mask,width,height);
    double focal=1.0, shift=0.0;
    if(options.fov_x_degrees>0) {
        const double rad=options.fov_x_degrees*kPi/180.0;
        focal=ar/diag/std::tan(rad/2.0);
        shift=solve_shift_fixed_focal(samples,focal);
    } else {
        auto fs=solve_focal_shift(samples); focal=fs.first; shift=fs.second;
        // Current upstream consumers also guard pathological negative/nonfinite
        // focal estimates. 60 degrees is a conservative failure fallback.
        if(!std::isfinite(focal)||focal<=0) {
            focal=ar/diag/std::tan((60.0*kPi/180.0)/2.0);
            shift=solve_shift_fixed_focal(samples,focal);
        }
    }
    const double fdiag=focal/2.0*diag;
    const float fx=static_cast<float>(fdiag/ar), fy=static_cast<float>(fdiag);

    Result r; r.width=width; r.height=height; r.mask=mask; r.metric_scale=metric_scale;
    r.intrinsics[0]=fx; r.intrinsics[1]=0; r.intrinsics[2]=0.5f;
    r.intrinsics[3]=0; r.intrinsics[4]=fy; r.intrinsics[5]=0.5f;
    r.intrinsics[6]=0; r.intrinsics[7]=0; r.intrinsics[8]=1;
    r.depth.resize(n); r.points.resize(n*3);
    if(!normal.empty()) r.normal=std::move(normal);

    post_parallel_for(0, n, [&](size_t b, size_t e) {
        for(size_t i=b;i<e;++i) {
            const int y=static_cast<int>(i/static_cast<size_t>(width));
            const int x=static_cast<int>(i-static_cast<size_t>(y)*width);
            float d=points[3*i+2]+static_cast<float>(shift);
            if(d<=0) r.mask[i]=0;
            d*=metric_scale; r.depth[i]=d;
            if(options.force_projection) {
                const float u=(x+0.5f)/width, v=(y+0.5f)/height;
                r.points[3*i+0]=(u-0.5f)/fx*d;
                r.points[3*i+1]=(v-0.5f)/fy*d;
                r.points[3*i+2]=d;
            } else {
                r.points[3*i+0]=points[3*i+0]*metric_scale;
                r.points[3*i+1]=points[3*i+1]*metric_scale;
                r.points[3*i+2]=d;
            }
            if(options.apply_mask&&!r.mask[i]) {
                const float inf=std::numeric_limits<float>::infinity();
                r.points[3*i]=r.points[3*i+1]=r.points[3*i+2]=inf; r.depth[i]=inf;
                if(!r.normal.empty()) r.normal[3*i]=r.normal[3*i+1]=r.normal[3*i+2]=0;
            }
        }
    });
    return r;
}

} // namespace moge
