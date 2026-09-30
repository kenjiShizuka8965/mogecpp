#include "sparse.hpp"
#include "profiling.hpp"
#include "sparse_metal.hpp"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

namespace moge {
namespace {


int sparse_topology_threads(size_t work_items) {
    const char * env = std::getenv("MOGE_SPARSE_TOPOLOGY_THREADS");
    long requested = 0;
    if (env && *env) {
        char * end = nullptr;
        requested = std::strtol(env, &end, 10);
        if (!end || *end != '\0' || requested < 1 || requested > 64) {
            throw std::runtime_error("MOGE_SPARSE_TOPOLOGY_THREADS must be in [1,64]");
        }
    } else {
        const unsigned hw = std::thread::hardware_concurrency();
        requested = static_cast<long>(std::min<unsigned>(hw ? hw : 4u, 8u));
    }
    if (work_items < 16384) return 1;
    return static_cast<int>(std::min<size_t>(static_cast<size_t>(requested), std::max<size_t>(1, work_items / 4096)));
}

template <class Fn>
void sparse_parallel_for(size_t begin, size_t end, Fn && fn) {
    const size_t n = end > begin ? end - begin : 0;
    const int nth = sparse_topology_threads(n);
    if (nth <= 1 || n == 0) {
        fn(begin, end);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(nth));
    for (int t = 0; t < nth; ++t) {
        const size_t b = begin + n * static_cast<size_t>(t) / static_cast<size_t>(nth);
        const size_t e = begin + n * static_cast<size_t>(t + 1) / static_cast<size_t>(nth);
        workers.emplace_back([&, b, e] { fn(b, e); });
    }
    for (auto & w : workers) w.join();
}

struct Coord3 {
    int32_t x = 0; // sparse dimension 0: image row in upstream naming
    int32_t y = 0; // sparse dimension 1: image column
    int32_t z = 0;
    bool operator==(const Coord3 & o) const { return x == o.x && y == o.y && z == o.z; }
};

struct CoordHash {
    size_t operator()(const Coord3 & c) const noexcept {
        // SplitMix-style integer mixing.  Sparse image coordinates are small,
        // but log-z can be several thousand bins wide.
        uint64_t h = static_cast<uint32_t>(c.x);
        h = (h * 0x9e3779b185ebca87ULL) ^ static_cast<uint32_t>(c.y);
        h = (h * 0xc2b2ae3d27d4eb4fULL) ^ static_cast<uint32_t>(c.z);
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        return static_cast<size_t>(h);
    }
};


class FlatCoordMap {
public:
    explicit FlatCoordMap(size_t expected) {
        size_t cap = 16;
        const size_t want = std::max<size_t>(16, expected * 2);
        while (cap < want) cap <<= 1;
        keys_.resize(cap);
        values_.assign(cap, -1);
        mask_ = cap - 1;
    }

    std::pair<int32_t, bool> emplace(const Coord3 & key, int32_t value) {
        size_t slot = CoordHash{}(key) & mask_;
        for (;;) {
            int32_t & current = values_[slot];
            if (current < 0) {
                keys_[slot] = key;
                current = value;
                return {value, true};
            }
            if (keys_[slot] == key) return {current, false};
            slot = (slot + 1) & mask_;
        }
    }

    int32_t find(const Coord3 & key) const noexcept {
        size_t slot = CoordHash{}(key) & mask_;
        for (;;) {
            const int32_t current = values_[slot];
            if (current < 0) return -1;
            if (keys_[slot] == key) return current;
            slot = (slot + 1) & mask_;
        }
    }

private:
    std::vector<Coord3> keys_;
    std::vector<int32_t> values_;
    size_t mask_ = 0;
};

struct SparseLevel {
    std::vector<Coord3> coords;
    // Canonical target-major [target][27 taps]. Output-point tiling can slice
    // this directly without runtime repacking.
    std::vector<int32_t> neighbors;
};

struct Transition {
    int factor = 2;
    int slots = 8;
    // Parent-major, slot-fastest.  Missing children use child_count.
    std::vector<int32_t> pool_rows;
    std::vector<float> inv_count;
    // One parent row per fine target for nearest upsampling.
    std::vector<int32_t> up_rows;
};

struct SparseTopology {
    std::vector<SparseLevel> levels;
    std::vector<Transition> transitions;
};

std::vector<int> to_ints(const std::vector<int64_t> & v) {
    std::vector<int> out;
    out.reserve(v.size());
    for (auto x : v) out.push_back(static_cast<int>(x));
    return out;
}

int64_t product3(int f) { return static_cast<int64_t>(f) * f * f; }

bool sparse_trace() {
    const char * env = std::getenv("MOGE_SPARSE_TRACE");
    return env && *env && std::strcmp(env, "0") != 0;
}


thread_local bool g_vulkan_balanced_default = false;

struct ScopedVulkanBalancedDefault {
    bool previous = false;
    explicit ScopedVulkanBalancedDefault(bool enabled)
        : previous(g_vulkan_balanced_default) { g_vulkan_balanced_default = enabled; }
    ~ScopedVulkanBalancedDefault() { g_vulkan_balanced_default = previous; }
};

bool backend_is_vulkan(ggml_backend_t backend) {
    if (!backend) return false;
    auto dev = ggml_backend_get_device(backend);
    auto reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    const char * raw = reg ? ggml_backend_reg_name(reg) : nullptr;
    if (!raw) return false;
    std::string name(raw);
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return name == "vk" || name.find("vulkan") != std::string::npos;
}

// Release policy: Vulkan uses the validated balanced state/budget policy.
// CPU and Metal retain their full-precision reference state.
bool sparse_low_precision_state() { return g_vulkan_balanced_default; }

ggml_tensor * sparse_append_zero_row(ggml_context * c, ggml_tensor * x) {
    auto * zero = ggml_fill_inplace(c, ggml_new_tensor_2d(c, x->type, x->ne[0], 1), 0.0f);
    return ggml_concat(c, x, zero, 1);
}

ggml_tensor * sparse_get_rows_source(ggml_context * c, ggml_tensor * x) {
    // The validated Vulkan small GET_ROWS kernel treats sentinel index N as an
    // implicit zero row. Other backends retain the explicit appended zero row.
    return g_vulkan_balanced_default ? x : sparse_append_zero_row(c, x);
}

ggml_backend_sched_t make_derived_scheduler(ggml_backend_sched_t parent) {
    const int n = ggml_backend_sched_get_n_backends(parent);
    if (n <= 0) throw std::runtime_error("sparse parent scheduler has no backends");
    std::vector<ggml_backend_t> backends(static_cast<size_t>(n));
    std::vector<ggml_backend_buffer_type_t> bufts(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        backends[static_cast<size_t>(i)] = ggml_backend_sched_get_backend(parent, i);
        bufts[static_cast<size_t>(i)] = ggml_backend_sched_get_buffer_type(parent, backends[static_cast<size_t>(i)]);
    }
    return ggml_backend_sched_new(backends.data(), bufts.data(), n, 4096, false, true);
}

size_t sparse_gather_budget_bytes(size_t level = static_cast<size_t>(-1)) {
    const long mib = g_vulkan_balanced_default ? ((level == 0) ? 96 : 64) : 128;
    return static_cast<size_t>(mib) * 1024u * 1024u;
}

ggml_type sparse_gather_type(size_t = static_cast<size_t>(-1)) {
    // Sparse gather materialization stays in F32 to preserve the validated
    // cross-backend numerical contract.
    return GGML_TYPE_F32;
}

int64_t sparse_tile_points(int64_t channels, int64_t npoints, size_t element_size = sizeof(float),
                           size_t level = static_cast<size_t>(-1)) {
    const size_t bytes_per_target = static_cast<size_t>(27) * static_cast<size_t>(channels) * element_size;
    int64_t tile = static_cast<int64_t>(sparse_gather_budget_bytes(level) / std::max<size_t>(1, bytes_per_target));
    tile = (tile / 256) * 256;
    tile = std::max<int64_t>(1024, tile);
    return std::max<int64_t>(1, std::min<int64_t>(npoints, tile));
}

SparseLevel make_level(std::vector<Coord3> coords) {
    SparseLevel l;
    l.coords = std::move(coords);
    const int32_t n = static_cast<int32_t>(l.coords.size());
    // Canonical target-major layout: [target][tap]. This is directly sliceable
    // for output-point tiled sparse convolution and independent of runtime tile size.
    l.neighbors.resize(static_cast<size_t>(n) * 27);

    FlatCoordMap flat(l.coords.size());
    for (int32_t i = 0; i < n; ++i) flat.emplace(l.coords[static_cast<size_t>(i)], i);

    // FlexGEMM's canonical v order is x/y/z with z fastest.  In MoGe coords
    // are (row, col, zbin), so dx below means row offset and dy column offset.
    // The map is immutable after construction, so neighbour lookup is safely
    // parallel across targets. This is a sizeable CPU-side cost at every
    // refinement step and does not alter ordering or numerics.
    sparse_parallel_for(0, static_cast<size_t>(n), [&](size_t ib, size_t ie) {
        for (size_t ii = ib; ii < ie; ++ii) {
            int v = 0;
            const Coord3 c = l.coords[ii];
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz, ++v) {
                        Coord3 q{c.x + dx, c.y + dy, c.z + dz};
                        int32_t found = n;
                        const int32_t idx = flat.find(q);
                        if (idx >= 0) found = idx;
                        l.neighbors[ii * 27 + static_cast<size_t>(v)] = found;
                    }
                }
            }
        }
    });
    return l;
}

SparseLevel make_dense_image_level(std::vector<Coord3> coords, int width, int height) {
    if (coords.size() != static_cast<size_t>(width) * height)
        throw std::runtime_error("dense sparse level shape mismatch");
    SparseLevel l;
    l.coords=std::move(coords);
    const int32_t n=static_cast<int32_t>(l.coords.size());
    l.neighbors.assign(static_cast<size_t>(n)*27,n);

    // Level 0 contains exactly one point per image pixel. Avoid constructing a
    // ~million-entry hash map and 27 hash probes per pixel: the x/y neighbour
    // index is direct, and only the z-bin delta decides which of the three z
    // taps exists for that pixel neighbour. Rows are independent, so parallelize
    // the dominant level-0 topology pass.
    sparse_parallel_for(0, static_cast<size_t>(height), [&](size_t yb, size_t ye) {
        for (size_t yu=yb; yu<ye; ++yu) {
            const int y=static_cast<int>(yu);
            for (int x=0;x<width;++x) {
                const int32_t i=static_cast<int32_t>(static_cast<size_t>(y)*width+x);
                const int32_t z=l.coords[static_cast<size_t>(i)].z;
                for (int dy=-1;dy<=1;++dy) {
                    const int yy=y+dy;
                    if (yy<0 || yy>=height) continue;
                    for (int dx=-1;dx<=1;++dx) {
                        const int xx=x+dx;
                        if (xx<0 || xx>=width) continue;
                        const int32_t j=static_cast<int32_t>(static_cast<size_t>(yy)*width+xx);
                        const int dz=l.coords[static_cast<size_t>(j)].z-z;
                        if (dz < -1 || dz > 1) continue;
                        const int tap=(dy+1)*9+(dx+1)*3+(dz+1);
                        l.neighbors[static_cast<size_t>(i)*27+static_cast<size_t>(tap)]=j;
                    }
                }
            }
        }
    });
    return l;
}

std::pair<std::vector<Coord3>, Transition> downsample(const SparseLevel & fine, int factor) {
    if (factor <= 1) throw std::runtime_error("sparse downsample factor must be > 1");
    const int32_t nfine = static_cast<int32_t>(fine.coords.size());
    const int slots = static_cast<int>(product3(factor));

    std::vector<Coord3> parent;
    std::vector<int32_t> fine_to_parent(static_cast<size_t>(nfine));
    std::vector<int32_t> counts;
    parent.reserve(fine.coords.size() / 2 + 1);

    {
        FlatCoordMap pmap(fine.coords.size());
        for (int32_t i = 0; i < nfine; ++i) {
            const auto & c = fine.coords[static_cast<size_t>(i)];
            Coord3 p{c.x / factor, c.y / factor, c.z / factor};
            const int32_t next = static_cast<int32_t>(parent.size());
            auto [pi, inserted] = pmap.emplace(p, next);
            if (inserted) {
                parent.push_back(p);
                counts.push_back(0);
            }
            fine_to_parent[static_cast<size_t>(i)] = pi;
            ++counts[static_cast<size_t>(pi)];
        }
    }

    Transition tr;
    tr.factor = factor;
    tr.slots = slots;
    const size_t np=parent.size();
    tr.pool_rows.assign(np * static_cast<size_t>(slots), nfine);
    tr.inv_count.resize(np);
    tr.up_rows = fine_to_parent;
    std::vector<int32_t> fill(np,0);
    for (size_t p=0;p<np;++p) {
        const int32_t c=counts[p];
        if (c<=0 || c>slots) throw std::runtime_error("invalid sparse pool occupancy");
        tr.inv_count[p]=1.0f/static_cast<float>(c);
    }
    for (int32_t i=0;i<nfine;++i) {
        const int32_t pi=fine_to_parent[static_cast<size_t>(i)];
        const int32_t slot=fill[static_cast<size_t>(pi)]++;
        tr.pool_rows[static_cast<size_t>(slot)*np+static_cast<size_t>(pi)]=i;
    }
    return {std::move(parent), std::move(tr)};
}

struct QuantizedTopologyKey {
    std::vector<int32_t> z;
};

QuantizedTopologyKey quantize_topology_z(const std::vector<float> & coord, int width, int height,
                                         const SparseConfig & cfg) {
    if (coord.size() != static_cast<size_t>(width) * height * 3) {
        throw std::runtime_error("factor coordinate size mismatch");
    }
    const size_t n = static_cast<size_t>(width) * height;
    std::vector<int64_t> qz(n);
    const int nth = sparse_topology_threads(n);
    std::vector<int64_t> local_min(static_cast<size_t>(nth), std::numeric_limits<int64_t>::max());
    if (nth <= 1) {
        for (size_t i=0;i<n;++i) {
            const float z=coord[i*3+2];
            const int64_t q=static_cast<int64_t>(std::nearbyint(static_cast<double>(z*cfg.depth_resolution)));
            qz[i]=q; local_min[0]=std::min(local_min[0],q);
        }
    } else {
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(nth));
        for (int t=0;t<nth;++t) {
            const size_t b=n*static_cast<size_t>(t)/static_cast<size_t>(nth);
            const size_t e=n*static_cast<size_t>(t+1)/static_cast<size_t>(nth);
            workers.emplace_back([&,t,b,e] {
                int64_t mn=std::numeric_limits<int64_t>::max();
                for (size_t i=b;i<e;++i) {
                    const float z=coord[i*3+2];
                    const int64_t q=static_cast<int64_t>(std::nearbyint(static_cast<double>(z*cfg.depth_resolution)));
                    qz[i]=q; mn=std::min(mn,q);
                }
                local_min[static_cast<size_t>(t)]=mn;
            });
        }
        for (auto & w:workers) w.join();
    }
    const int64_t zmin=*std::min_element(local_min.begin(),local_min.end());
    const int64_t zmax=*std::max_element(qz.begin(),qz.end());
    if (zmax-zmin > std::numeric_limits<int32_t>::max()) throw std::runtime_error("z-bin extent overflow");

    QuantizedTopologyKey key;
    key.z.resize(n);
    sparse_parallel_for(0,n,[&](size_t b,size_t e) {
        for (size_t i=b;i<e;++i) key.z[i]=static_cast<int32_t>(qz[i]-zmin);
    });
    return key;
}

SparseTopology build_topology_from_key(const QuantizedTopologyKey & key, int width, int height,
                                       const SparseConfig & cfg) {
    const size_t n = static_cast<size_t>(width) * height;
    if (key.z.size() != n) throw std::runtime_error("quantized topology size mismatch");
    std::vector<Coord3> l0(n);
    sparse_parallel_for(0,n,[&](size_t b,size_t e) {
        for (size_t i=b;i<e;++i) {
            const int y=static_cast<int>(i/static_cast<size_t>(width));
            const int x=static_cast<int>(i-static_cast<size_t>(y)*width);
            l0[i]=Coord3{y,x,key.z[i]};
        }
    });

    SparseTopology top;
    top.levels.reserve(cfg.model_channels.size());
    top.transitions.reserve(cfg.downsample_factors.size());
    top.levels.push_back(make_dense_image_level(std::move(l0), width, height));
    for (int f : cfg.downsample_factors) {
        auto [parent, tr] = downsample(top.levels.back(), f);
        top.transitions.push_back(std::move(tr));
        top.levels.push_back(make_level(std::move(parent)));
    }
    return top;
}

struct SparseTopologyCache {
    int width = 0;
    int height = 0;
    QuantizedTopologyKey key;
    std::shared_ptr<SparseTopology> topology;
    uint64_t hits = 0;
    uint64_t misses = 0;
};


std::shared_ptr<SparseTopology> get_topology_cached(std::shared_ptr<void> & opaque_cache,
                                                    const std::vector<float> & coord,
                                                    int width, int height,
                                                    const SparseConfig & cfg) {
    auto key = quantize_topology_z(coord, width, height, cfg);
    std::shared_ptr<SparseTopologyCache> cache;
    if (opaque_cache) cache = std::static_pointer_cast<SparseTopologyCache>(opaque_cache);
    else {
        cache = std::make_shared<SparseTopologyCache>();
        opaque_cache = cache;
    }
    if (cache->topology && cache->width == width && cache->height == height && cache->key.z == key.z) {
        ++cache->hits;
        if (sparse_trace()) std::fprintf(stderr, "moge_sparse_topology_cache: hit=%llu miss=%llu points=%zu\n",
                                         static_cast<unsigned long long>(cache->hits),
                                         static_cast<unsigned long long>(cache->misses), key.z.size());
        return cache->topology;
    }
    auto topology = std::make_shared<SparseTopology>(build_topology_from_key(key, width, height, cfg));
    ++cache->misses;
    cache->width = width;
    cache->height = height;
    cache->key = std::move(key);
    cache->topology = topology;
    if (sparse_trace()) std::fprintf(stderr, "moge_sparse_topology_cache: hit=%llu miss=%llu points=%zu\n",
                                     static_cast<unsigned long long>(cache->hits),
                                     static_cast<unsigned long long>(cache->misses),
                                     static_cast<size_t>(width) * height);
    return topology;
}

// All sparse learned features use [C,N] so LayerNorm is over ggml ne0 and a
// PyTorch nn.Linear converted as [Cin,Cout] is directly consumable by mul_mat.
ggml_tensor * cast_like(ggml_context * c, ggml_tensor * t, const ggml_tensor * like) {
    if (!t || t->type == like->type) return t;
    if ((t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16) &&
        (like->type == GGML_TYPE_F32 || like->type == GGML_TYPE_F16 || like->type == GGML_TYPE_BF16)) {
        return ggml_cast(c, t, like->type);
    }
    throw std::runtime_error("cannot cast sparse affine tensor to activation type");
}


// Keep binary src0 compatible with ggml 0.25.3's CPU fallback. Sparse graph
// views can otherwise reach binary-ops.cpp with nb[0] != sizeof(type).
ggml_tensor * binary_src0(ggml_context * c, ggml_tensor * x) {
    return x->nb[0] == ggml_type_size(x->type) ? x : ggml_cont(c, x);
}

ggml_tensor * add_safe(ggml_context * c, ggml_tensor * a, ggml_tensor * b) {
    return ggml_add(c, binary_src0(c, a), b);
}

ggml_tensor * mul_safe(ggml_context * c, ggml_tensor * a, ggml_tensor * b) {
    return ggml_mul(c, binary_src0(c, a), b);
}

ggml_tensor * add_inplace_safe(ggml_context * c, ggml_tensor * a, ggml_tensor * b) {
    if (a->nb[0] != ggml_type_size(a->type)) return add_safe(c, a, b);
    return ggml_add_inplace(c, a, b);
}

ggml_tensor * mul_inplace_safe(ggml_context * c, ggml_tensor * a, ggml_tensor * b) {
    if (a->nb[0] != ggml_type_size(a->type)) return mul_safe(c, a, b);
    return ggml_mul_inplace(c, a, b);
}

ggml_tensor * add_bias(ggml_context * c, ggml_tensor * x, ggml_tensor * b) {
    if (!b) return x;
    return add_inplace_safe(c, x, cast_like(c, b, x));
}

ggml_tensor * linear(ggml_context * c, const WeightStore & w, ggml_tensor * x, const std::string & pfx) {
    auto * y = ggml_mul_mat(c, w.get(pfx + ".weight"), x);
    return add_bias(c, y, w.maybe(pfx + ".bias"));
}

ggml_tensor * layer_norm(ggml_context * c, const WeightStore & w, ggml_tensor * x,
                         const std::string & pfx) {
    auto * y = ggml_norm(c, x, 1e-6f);
    if (auto * a = w.maybe(pfx + ".weight")) y = mul_inplace_safe(c, y, cast_like(c, a, y));
    if (auto * b = w.maybe(pfx + ".bias")) y = add_inplace_safe(c, y, cast_like(c, b, y));
    return y;
}

struct RuntimeInputs {
    // Input and final residual have disjoint lifetimes. Back them with one raw
    // F32 arena: feature consumes 3*N0 at the start of the down path, while
    // residual reuses the first 1*N0 words only after the final up phase.
    ggml_tensor * io_arena = nullptr;
    ggml_tensor * feature = nullptr;
    // Reusable GPU index arena. Only one U-Net level is active at a time, so
    // persisting every level's neighbour/pool/up tables wastes hundreds of MiB.
    ggml_tensor * neighbor_arena = nullptr; // I32, max 27*Nlevel
    ggml_tensor * rows_arena = nullptr;     // I32, max pool/up/encoder rows
    ggml_tensor * inv_arena = nullptr;      // F32, max parent count
};

struct PersistentSparseState {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    RuntimeInputs in;
    std::vector<ggml_tensor *> level;
    ggml_tensor * residual = nullptr;
    ggml_tensor * hybrid_norm_arena = nullptr;
    ggml_tensor * hybrid_conv1_arena = nullptr;
    std::vector<ggml_tensor *> hybrid_norm;
    std::vector<ggml_tensor *> hybrid_conv1;
    enum ggml_type state_type = GGML_TYPE_F32;
    bool metal_hybrid = false;

    ~PersistentSparseState() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
};

enum ggml_type sparse_state_type(bool metal_hybrid) {
    if (metal_hybrid) return GGML_TYPE_F32;
    return sparse_low_precision_state() ? GGML_TYPE_F16 : GGML_TYPE_F32;
}

ggml_tensor * state_read(ggml_context * c, ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(c, t, GGML_TYPE_F32);
}

ggml_tensor * make_input(ggml_context * c, enum ggml_type type, int64_t ne0, int64_t ne1,
                         const char * name) {
    auto * t = ggml_new_tensor_2d(c, type, ne0, ne1);
    ggml_set_name(t, name);
    ggml_set_input(t);
    return t;
}



ggml_tensor * sparse_conv(ggml_context * c, const WeightStore & w, ggml_tensor * x,
                          ggml_tensor * neighbor, const std::string & pfx, size_t level) {
    const int64_t C = x->ne[0];
    const int64_t N = x->ne[1];
    if (N <= 0 || neighbor->ne[0] != 27 * N) {
        throw std::runtime_error("sparse neighbor index shape mismatch");
    }

    auto * packed_w = w.get(pfx + ".weight.mogg_sparse27");
    if (packed_w->ne[0] != 27 * C) {
        throw std::runtime_error("sparse packed weight input size mismatch");
    }
    const int64_t Cout = packed_w->ne[1];

    // Optionally quantize only the gather-side activation to fp16. MUL_MAT still
    // accumulates/returns f32 in ggml 0.25.3, but this halves the dominant sparse
    // neighbourhood buffer. It is opt-in because it changes numerical results.
    ggml_tensor * gather_x = x;
    const ggml_type gtype = sparse_gather_type(level);
    if (gather_x->type != gtype) gather_x = ggml_cast(c, gather_x, gtype);

    // One zero sentinel is enough for every tile. Missing sparse neighbours use
    // index N, so padding x to [C,N+1] keeps GET_ROWS branch-free.
    // Vulkan GET_ROWS can treat the sentinel index N as an implicit zero row,
    // avoiding a FILL+CONCAT materialization for every sparse convolution.
    // Other backends retain the historical explicit [C,N+1] padding tensor.
    auto * padded = sparse_get_rows_source(c, gather_x); // [C,N] Vulkan, [C,N+1] fallback

    // Tile over *output points*, not taps. Each tile still performs one full
    // 27-tap GEMM, preserving the efficient packed checkpoint layout, but the
    // huge [27*C,N] gather becomes [27*C,tile]. Results are written into one
    // destination through chained in-place SET ops, so no concat ladder or
    // second full output is created.
    const int64_t tile = sparse_tile_points(C, N, ggml_type_size(gather_x->type), level);
    // Every output column is overwritten exactly once by the SET chain, so the
    // destination needs no zero-fill. Bias is applied once after tile assembly.
    ggml_tensor * y = ggml_new_tensor_2d(c, GGML_TYPE_F32, Cout, N);
    for (int64_t begin = 0; begin < N; begin += tile) {
        const int64_t count = std::min<int64_t>(tile, N - begin);
        const size_t row_off = static_cast<size_t>(begin) * 27u * neighbor->nb[0];
        auto * rows = ggml_view_1d(c, neighbor, 27 * count, row_off);
        auto * gathered = ggml_get_rows(c, padded, rows); // [C,27*count]
        gathered = ggml_reshape_2d(c, gathered, 27 * C, count);
        auto * tile_y = ggml_mul_mat(c, packed_w, gathered); // [Cout,count]
        y = ggml_set_2d_inplace(c, y, tile_y, y->nb[1], static_cast<size_t>(begin) * y->nb[1]);
    }
    y = add_bias(c, y, w.maybe(pfx + ".bias"));
    return y;
}

ggml_tensor * res_block(ggml_context * c, const WeightStore & w, ggml_tensor * x,
                        ggml_tensor * neighbor, const std::string & pfx, size_t level) {
    // SparseResBlock3d in MoGe has a single affine LayerNorm before conv1,
    // SiLU before each convolution, and an identity skip for released configs.
    auto * h = layer_norm(c, w, x, pfx + ".norm1");
    h = ggml_silu_inplace(c, h);
    h = sparse_conv(c, w, h, neighbor, pfx + ".conv1", level);
    h = ggml_silu_inplace(c, h);
    h = sparse_conv(c, w, h, neighbor, pfx + ".conv2", level);
    if (w.has(pfx + ".skip_connection.weight")) {
        x = linear(c, w, x, pfx + ".skip_connection");
    }
    return add_inplace_safe(c, h, x);
}



ggml_tensor * pool_down(ggml_context * c, const WeightStore & w, ggml_tensor * x,
                        ggml_tensor * rows, ggml_tensor * inv, int slots,
                        const std::string & pfx) {
    const int64_t C = x->ne[0];
    const int64_t Nc = x->ne[1];
    const int64_t Np = inv->ne[1];
    if (rows->ne[0] != slots * Np) throw std::runtime_error("sparse pool index size mismatch");

    // Missing children use sentinel Nc. Stream one child slot at a time so the
    // temporary is [C,Np], not [C,slots,Np]. This portable ggml path is both
    // preserves the validated pooling behavior across the supported backends.
    auto * padded = sparse_get_rows_source(c, x);
    ggml_tensor * sum = nullptr;
    for (int slot = 0; slot < slots; ++slot) {
        auto * slot_rows = ggml_view_1d(c, rows, Np,
                                        static_cast<size_t>(slot) * static_cast<size_t>(Np) * rows->nb[0]);
        auto * g = ggml_get_rows(c, padded, slot_rows); // [C,Np]
        sum = sum ? add_inplace_safe(c, sum, g) : g;
    }
    sum = mul_inplace_safe(c, sum, cast_like(c, inv, sum));
    (void) C; (void) Nc;
    return linear(c, w, sum, pfx + ".linear");
}

ggml_tensor * nearest_up(ggml_context * c, const WeightStore & w, ggml_tensor * x,
                         ggml_tensor * rows, const std::string & pfx) {
    x = linear(c, w, x, pfx + ".linear");
    return ggml_get_rows(c, x, rows); // [C,Nfine]
}

RuntimeInputs make_inputs(ggml_context * c, const SparseConfig & cfg,
                          const SparseTopology & top, bool need_neighbor_arena) {
    RuntimeInputs r;
    const int64_t n0 = static_cast<int64_t>(top.levels[0].coords.size());
    const int64_t io_channels = std::max(cfg.in_channels, cfg.out_channels);
    r.io_arena = ggml_new_tensor_2d(c, GGML_TYPE_F32, io_channels, n0);
    ggml_set_name(r.io_arena, "refiner.state.io_arena");
    r.feature = ggml_view_2d(c, r.io_arena, cfg.in_channels, n0,
                             static_cast<size_t>(cfg.in_channels) * sizeof(float), 0);
    ggml_set_name(r.feature, "refiner.input");
    ggml_set_input(r.feature);

    size_t max_neighbors = 0;
    size_t max_rows = top.levels.back().coords.size();
    size_t max_inv = 1;
    for (const auto & level : top.levels) max_neighbors = std::max(max_neighbors, level.neighbors.size());
    for (const auto & tr : top.transitions) {
        max_rows = std::max(max_rows, tr.pool_rows.size());
        max_rows = std::max(max_rows, tr.up_rows.size());
        max_inv = std::max(max_inv, tr.inv_count.size());
    }
    if (need_neighbor_arena) {
        r.neighbor_arena = make_input(c, GGML_TYPE_I32, static_cast<int64_t>(max_neighbors), 1,
                                      "refiner.index.neighbor_arena");
    }
    r.rows_arena = make_input(c, GGML_TYPE_I32, static_cast<int64_t>(max_rows), 1,
                              "refiner.index.rows_arena");
    r.inv_arena = make_input(c, GGML_TYPE_F32, 1, static_cast<int64_t>(max_inv),
                             "refiner.index.inv_arena");
    return r;
}

ggml_tensor * neighbor_view(ggml_context * c, const RuntimeInputs & r, int64_t npoints) {
    return ggml_view_1d(c, r.neighbor_arena, 27 * npoints, 0);
}

ggml_tensor * rows_view(ggml_context * c, const RuntimeInputs & r, int64_t count) {
    return ggml_view_1d(c, r.rows_arena, count, 0);
}

ggml_tensor * inv_view(ggml_context * c, const RuntimeInputs & r, int64_t count) {
    return ggml_view_2d(c, r.inv_arena, 1, count, r.inv_arena->nb[1], 0);
}

std::unique_ptr<PersistentSparseState> make_persistent_state(
        ggml_backend_t backend,
        const SparseConfig & cfg,
        const SparseTopology & top,
        bool metal_hybrid) {
    if (!backend) throw std::runtime_error("missing primary backend for sparse state");

    auto s = std::make_unique<PersistentSparseState>();
    ggml_init_params p{};
    // Metadata only. Tensor storage is packed by ggml into one backend buffer
    // below, which avoids hundreds of scheduler-owned sparse input allocations.
    p.mem_size = 16u << 20;
    p.mem_buffer = nullptr;
    p.no_alloc = true;
    s->ctx = ggml_init(p);
    if (!s->ctx) throw std::runtime_error("ggml_init failed for persistent sparse state");

    s->metal_hybrid = metal_hybrid;
    s->in = make_inputs(s->ctx, cfg, top, !metal_hybrid);
    s->state_type = sparse_state_type(metal_hybrid);
    s->level.reserve(cfg.model_channels.size());
    for (size_t i = 0; i < cfg.model_channels.size(); ++i) {
        const int64_t n = static_cast<int64_t>(top.levels[i].coords.size());
        auto * t = ggml_new_tensor_2d(s->ctx, s->state_type, cfg.model_channels[i], n);
        ggml_set_name(t, ("refiner.state.level." + std::to_string(i)).c_str());
        s->level.push_back(t);
    }
    if (metal_hybrid) {
        int64_t max_elems = 0;
        for (size_t i = 0; i < cfg.model_channels.size(); ++i) {
            max_elems = std::max<int64_t>(max_elems, static_cast<int64_t>(cfg.model_channels[i]) * static_cast<int64_t>(top.levels[i].coords.size()));
        }
        s->hybrid_norm_arena = ggml_new_tensor_1d(s->ctx, GGML_TYPE_F16, max_elems);
        ggml_set_name(s->hybrid_norm_arena, "refiner.hybrid.norm_arena");
        s->hybrid_conv1_arena = ggml_new_tensor_1d(s->ctx, GGML_TYPE_F16, max_elems);
        ggml_set_name(s->hybrid_conv1_arena, "refiner.hybrid.conv1_arena");
        s->hybrid_norm.reserve(cfg.model_channels.size());
        s->hybrid_conv1.reserve(cfg.model_channels.size());
        for (size_t i = 0; i < cfg.model_channels.size(); ++i) {
            const int64_t C = cfg.model_channels[i];
            const int64_t N = static_cast<int64_t>(top.levels[i].coords.size());
            auto * a = ggml_view_2d(s->ctx, s->hybrid_norm_arena, C, N, static_cast<size_t>(C) * sizeof(ggml_fp16_t), 0);
            ggml_set_name(a, ("refiner.hybrid.norm." + std::to_string(i)).c_str());
            auto * b = ggml_view_2d(s->ctx, s->hybrid_conv1_arena, C, N, static_cast<size_t>(C) * sizeof(ggml_fp16_t), 0);
            ggml_set_name(b, ("refiner.hybrid.conv1." + std::to_string(i)).c_str());
            s->hybrid_norm.push_back(a);
            s->hybrid_conv1.push_back(b);
        }
    }

    const int64_t n0 = static_cast<int64_t>(top.levels[0].coords.size());
    // Reuse the now-dead input arena for the final residual. This view is
    // contiguous even though the arena was originally described as [Cin,N].
    s->residual = ggml_view_2d(s->ctx, s->in.io_arena, cfg.out_channels, n0,
                               static_cast<size_t>(cfg.out_channels) * sizeof(float), 0);
    ggml_set_name(s->residual, "refiner.state.residual");

    s->buffer = ggml_backend_alloc_ctx_tensors(s->ctx, backend);
    if (!s->buffer) throw std::runtime_error("failed to allocate persistent sparse backend buffer");
    return s;
}


struct SparseStateCache {
    const SparseTopology * topology = nullptr;
    ggml_backend_t backend = nullptr;
    bool metal_hybrid = false;
    ggml_type state_type = GGML_TYPE_F32;
    std::shared_ptr<PersistentSparseState> state;
    uint64_t hits = 0;
    uint64_t misses = 0;
};


std::shared_ptr<PersistentSparseState> get_persistent_state_cached(
        std::shared_ptr<void> & opaque_cache,
        ggml_backend_t backend,
        const SparseConfig & cfg,
        const std::shared_ptr<SparseTopology> & topology,
        bool metal_hybrid) {
    std::shared_ptr<SparseStateCache> cache;
    if (opaque_cache) cache = std::static_pointer_cast<SparseStateCache>(opaque_cache);
    if (!cache) {
        cache = std::make_shared<SparseStateCache>();
        opaque_cache = cache;
    }

    const ggml_type wanted_type = sparse_state_type(metal_hybrid);
    if (cache->state && cache->topology == topology.get() && cache->backend == backend &&
        cache->metal_hybrid == metal_hybrid && cache->state_type == wanted_type) {
        ++cache->hits;
        if (sparse_trace()) {
            std::fprintf(stderr, "moge_sparse_state_cache: hit=%llu miss=%llu bytes=%zu\n",
                         static_cast<unsigned long long>(cache->hits),
                         static_cast<unsigned long long>(cache->misses),
                         ggml_backend_buffer_get_size(cache->state->buffer));
        }
        return cache->state;
    }

    auto fresh = make_persistent_state(backend, cfg, *topology, metal_hybrid);
    auto state = std::shared_ptr<PersistentSparseState>(fresh.release());
    cache->topology = topology.get();
    cache->backend = backend;
    cache->metal_hybrid = metal_hybrid;
    cache->state_type = state->state_type;
    cache->state = state;
    ++cache->misses;
    if (sparse_trace()) {
        std::fprintf(stderr, "moge_sparse_state_cache: hit=%llu miss=%llu bytes=%zu\n",
                     static_cast<unsigned long long>(cache->hits),
                     static_cast<unsigned long long>(cache->misses),
                     ggml_backend_buffer_get_size(state->buffer));
    }
    return state;
}

ggml_context * make_phase_context() {
    ggml_init_params p{};
    // Each segmented graph contains at most one encoder/decoder U-Net level,
    // not the whole refinement network. Keep only tensor metadata here; the
    // scheduler owns transient activation storage.
    p.mem_size = 32u << 20;
    p.mem_buffer = nullptr;
    p.no_alloc = true;
    auto * c = ggml_init(p);
    if (!c) throw std::runtime_error("ggml_init failed for sparse phase graph");
    return c;
}

void run_phase(ggml_backend_sched_t parent_sched, ggml_context * c, const char * label,
               const std::vector<ggml_tensor *> & outputs) {
    if (outputs.empty()) throw std::runtime_error("sparse phase has no outputs");
    ggml_backend_sched_t owned_sched = make_derived_scheduler(parent_sched);
    if (!owned_sched) throw std::runtime_error("failed to create phase-local sparse scheduler");
    ggml_backend_sched_t sched = owned_sched;
    auto * graph = ggml_new_graph_custom(c, 4096, false);
    for (auto * t : outputs) {
        if (!t) throw std::runtime_error("null sparse phase output");
        ggml_build_forward_expand(graph, t);
    }

    // Persistent state tensors already own a primary-backend buffer. Resetting
    // only the scheduler here frees/reuses phase scratch while leaving those
    // level/skip/topology buffers intact for the next submission.
    ggml_backend_sched_reset(sched);
    const auto started = std::chrono::steady_clock::now();
    if (sparse_trace()) std::fprintf(stderr, "moge_sparse_phase: start %s\n", label);
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        if (owned_sched) ggml_backend_sched_free(owned_sched);
        throw std::runtime_error("failed to allocate segmented sparse phase graph");
    }
    auto * primary = ggml_backend_sched_get_backend(sched, 0);
    if (sparse_trace()) {
        const double mib = static_cast<double>(ggml_backend_sched_get_buffer_size(sched, primary)) / (1024.0 * 1024.0);
        std::fprintf(stderr, "moge_sparse_phase: allocated %s scratch=%.1f MiB\n", label, mib);
    }
    profiling::OpProfileState op_profile{label};
    if (profiling::op_profile_enabled()) ggml_backend_sched_set_eval_callback(sched, profiling::op_profile_callback, &op_profile);
    std::string memory_tag = std::string("sparse.") + label + ".allocated";
    profiling::device_memory_sample(primary, memory_tag.c_str());
    const auto st = ggml_backend_sched_graph_compute(sched, graph);
    if (profiling::op_profile_enabled()) ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
    if (st != GGML_STATUS_SUCCESS) {
        if (owned_sched) ggml_backend_sched_free(owned_sched);
        throw std::runtime_error("segmented sparse phase compute failed");
    }
    ggml_backend_sched_synchronize(sched);
    if (sparse_trace()) {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        std::fprintf(stderr, "moge_sparse_phase: done %s %.3f ms\n", label, ms);
    }
    ggml_backend_sched_reset(sched);
    if (owned_sched) ggml_backend_sched_free(owned_sched);
}

void run_hybrid_resblock(ggml_backend_sched_t sched, const WeightStore & w,
                         PersistentSparseState & state, SparseMetalExecutor & metal,
                         const SparseLevel & level_data, size_t level,
                         const std::string & pfx, const std::string & label) {
    if (w.has(pfx + ".skip_connection.weight")) {
        throw std::runtime_error("Metal sparse hybrid does not yet support projected sparse residual skips");
    }
    ggml_context * c = make_phase_context();
    try {
        auto * h = layer_norm(c, w, state.level[level], pfx + ".norm1");
        h = ggml_silu(c, h);
        auto * out = ggml_cpy(c, h, state.hybrid_norm[level]);
        const std::string norm_label = label + ".norm";
        run_phase(sched, c, norm_label.c_str(), {out});
    } catch (...) {
        ggml_free(c);
        throw;
    }
    ggml_free(c);

    const double ms = metal.resblock_inplace(
        static_cast<int>(level), state.hybrid_norm[level], state.hybrid_conv1[level], state.level[level],
        level_data.neighbors,
        w.get(pfx + ".conv1.weight.mogg_sparse27"), w.maybe(pfx + ".conv1.bias"),
        w.get(pfx + ".conv2.weight.mogg_sparse27"), w.maybe(pfx + ".conv2.bias"));
    if (sparse_trace()) {
        std::fprintf(stderr, "moge_sparse_metal: done %s policy=%s %.3f ms\n",
                     label.c_str(), metal.backend_policy_for_channels(static_cast<int>(state.level[level]->ne[0])), ms);
    }
}

void upload_feature(const RuntimeInputs & r, const std::vector<float> & factor_coord,
                    size_t n0) {
    std::vector<float> input(n0 * 3);
    for (size_t i = 0; i < n0; ++i) {
        input[i] = factor_coord[3*i + 0];
        input[n0 + i] = factor_coord[3*i + 1];
        input[2*n0 + i] = factor_coord[3*i + 2];
    }
    ggml_backend_tensor_set(r.feature, input.data(), 0, input.size() * sizeof(float));
}

void upload_neighbor(const RuntimeInputs & r, const SparseLevel & level) {
    if (!r.neighbor_arena) throw std::runtime_error("sparse neighbour arena unavailable in Metal hybrid mode");
    if (level.neighbors.size() > static_cast<size_t>(r.neighbor_arena->ne[0])) {
        throw std::runtime_error("sparse neighbour arena overflow");
    }
    ggml_backend_tensor_set(r.neighbor_arena, level.neighbors.data(), 0,
                            level.neighbors.size() * sizeof(int32_t));
}

void upload_pool(const RuntimeInputs & r, const Transition & tr) {
    if (tr.pool_rows.size() > static_cast<size_t>(r.rows_arena->ne[0]) ||
        tr.inv_count.size() > static_cast<size_t>(r.inv_arena->ne[1])) {
        throw std::runtime_error("sparse pool arena overflow");
    }
    ggml_backend_tensor_set(r.rows_arena, tr.pool_rows.data(), 0,
                            tr.pool_rows.size() * sizeof(int32_t));
    ggml_backend_tensor_set(r.inv_arena, tr.inv_count.data(), 0,
                            tr.inv_count.size() * sizeof(float));
}

void upload_up(const RuntimeInputs & r, const Transition & tr) {
    if (tr.up_rows.size() > static_cast<size_t>(r.rows_arena->ne[0])) {
        throw std::runtime_error("sparse upsample arena overflow");
    }
    ggml_backend_tensor_set(r.rows_arena, tr.up_rows.data(), 0,
                            tr.up_rows.size() * sizeof(int32_t));
}

void upload_encoder_rows(const RuntimeInputs & r, const SparseLevel & deep, int ew, int eh) {
    if (deep.coords.size() > static_cast<size_t>(r.rows_arena->ne[0])) {
        throw std::runtime_error("encoder-row arena overflow");
    }
    std::vector<int32_t> rows(deep.coords.size());
    for (size_t i = 0; i < deep.coords.size(); ++i) {
        const auto & q = deep.coords[i];
        if (q.y < 0 || q.x < 0 || q.y >= ew || q.x >= eh) {
            throw std::runtime_error("deep sparse coord outside encoder feature map");
        }
        rows[i] = static_cast<int32_t>(q.x * ew + q.y);
    }
    ggml_backend_tensor_set(r.rows_arena, rows.data(), 0, rows.size() * sizeof(int32_t));
}


} // namespace

SparseConfig read_sparse_config(const MoggFile & f) {
    SparseConfig c;
    c.present = f.meta_or<bool>("refiner.present", false);
    if (!c.present) return c;
    c.in_channels = static_cast<int>(f.meta<int64_t>("refiner.in_channels"));
    c.out_channels = static_cast<int>(f.meta<int64_t>("refiner.out_channels"));
    c.encoder_channels = static_cast<int>(f.meta<int64_t>("refiner.encoder_channels"));
    c.model_channels = to_ints(f.meta<std::vector<int64_t>>("refiner.model_channels"));
    c.encoder_blocks = to_ints(f.meta<std::vector<int64_t>>("refiner.encoder_blocks_per_level"));
    c.decoder_blocks = to_ints(f.meta<std::vector<int64_t>>("refiner.decoder_blocks_per_level"));
    c.bottleneck_blocks = static_cast<int>(f.meta<int64_t>("refiner.bottleneck_blocks"));
    c.downsample_factors = to_ints(f.meta<std::vector<int64_t>>("refiner.downsample_factors"));
    c.encoder_downsample = static_cast<int>(f.meta<int64_t>("refiner.encoder_downsample"));
    c.depth_resolution = static_cast<float>(f.meta<double>("refiner.depth_resolution"));
    if (c.model_channels.size() < 2 || c.encoder_blocks.size() != c.model_channels.size() ||
        c.decoder_blocks.size() + 1 != c.model_channels.size() ||
        c.downsample_factors.size() + 1 != c.model_channels.size()) {
        throw std::runtime_error("invalid sparse refiner metadata");
    }
    int prod = 1;
    for (int x : c.downsample_factors) prod *= x;
    if (prod != c.encoder_downsample) throw std::runtime_error("refiner encoder_downsample mismatch");
    return c;
}

SparseRefiner::SparseRefiner(const WeightStore & weights, SparseConfig config)
    : w_(weights), cfg_(std::move(config)) {
    if (!cfg_.present) throw std::runtime_error("SparseRefiner created for a model without a refiner");
}

void SparseRefiner::refine_once(ggml_backend_sched_t sched,
                                std::vector<float> & factor_coord,
                                int width, int height,
                                ggml_tensor * encoder_feature,
                                int encoder_width, int encoder_height) const {
    if (width != encoder_width * cfg_.encoder_downsample ||
        height != encoder_height * cfg_.encoder_downsample) {
        std::ostringstream os;
        os << "refiner grid " << width << 'x' << height << " does not match encoder "
           << encoder_width << 'x' << encoder_height << " * " << cfg_.encoder_downsample;
        throw std::runtime_error(os.str());
    }

    const auto top_ptr = get_topology_cached(topology_cache_, factor_coord, width, height, cfg_);
    const auto & top = *top_ptr;
    ggml_backend_t primary = ggml_backend_sched_get_backend(sched, 0);
    if (!primary) throw std::runtime_error("sparse scheduler has no primary backend");
    // Keep sparse gather memory bounded on Vulkan while allowing a larger CPU/Metal budget.
    // materially lower peak device memory with effectively unchanged latency
    // and golden-reference accuracy. The thread-local scope avoids changing
    // CPU/Metal behavior or leaking policy between concurrent model instances.
    ScopedVulkanBalancedDefault vulkan_default(backend_is_vulkan(primary));

    std::unique_ptr<SparseMetalExecutor> metal;
    if (auto executor = SparseMetalExecutor::create(primary); executor && executor->available()) {
        metal = std::move(executor);
    }

    // The old implementation built the entire sparse U-Net as one graph. At
    // 640x480 this could produce a multi-GiB Metal command-buffer residency set
    // even when the actual live activation set was much smaller. Keep only the
    // tensors that must cross U-Net stage boundaries in one packed backend
    // allocation, and execute each level as its own scheduler graph. The supported
    // Metal hybrid replaces only residual-block sparse convolutions; all other
    // graph operations continue through ggml.
    auto state = get_persistent_state_cached(state_cache_, primary, cfg_, top_ptr, static_cast<bool>(metal));
    if (sparse_trace()) {
        const double mib = static_cast<double>(ggml_backend_buffer_get_size(state->buffer)) / (1024.0 * 1024.0);
        std::fprintf(stderr,
                     "moge_sparse_phase: persistent=%.1f MiB levels=%zu state=%s gather=%s tile0=%lld gather_budget=%zuMiB phase_local_sched=%d points0=%zu metal_hybrid=%d\n",
                     mib, cfg_.model_channels.size(), state->state_type == GGML_TYPE_F16 ? "f16" : "f32",
                     sparse_gather_type(0) == GGML_TYPE_F16 ? "f16" : "f32",
                     static_cast<long long>(sparse_tile_points(cfg_.model_channels[0], static_cast<int64_t>(top.levels[0].coords.size()), ggml_type_size(sparse_gather_type(0)), 0)),
                     sparse_gather_budget_bytes(0) / (1024u * 1024u), 1, top.levels[0].coords.size(), metal ? 1 : 0);
        for (size_t i = 0; i < top.levels.size(); ++i) {
            const size_t npoints = top.levels[i].coords.size();
            const size_t state_bytes = static_cast<size_t>(cfg_.model_channels[i]) * npoints * ggml_type_size(state->state_type);
            const size_t neigh_bytes = top.levels[i].neighbors.size() * sizeof(int32_t);
            size_t valid_edges = 0;
            int max_degree = 0;
            for (size_t n = 0; n < npoints; ++n) {
                int degree = 0;
                const int32_t sentinel = static_cast<int32_t>(npoints);
                for (int tap = 0; tap < 27; ++tap) {
                    degree += top.levels[i].neighbors[n * 27u + static_cast<size_t>(tap)] != sentinel;
                }
                valid_edges += static_cast<size_t>(degree);
                max_degree = std::max(max_degree, degree);
            }
            // Proposed direct-kernel CSR encoding: one U32 offset per point +
            // one U32 packed (tap:5, source:27) word per valid neighbour.
            const size_t packed_edge_bytes = (npoints + 1 + valid_edges) * sizeof(uint32_t);
            const double avg_degree = npoints ? static_cast<double>(valid_edges) / static_cast<double>(npoints) : 0.0;
            std::fprintf(stderr,
                         "moge_sparse_phase: level=%zu points=%zu channels=%d spill=%.1f MiB neighbor_host=%.1f MiB degree_avg=%.2f degree_max=%d packed_edges=%.1f MiB\n",
                         i, npoints, cfg_.model_channels[i],
                         static_cast<double>(state_bytes)/(1024.0*1024.0),
                         static_cast<double>(neigh_bytes)/(1024.0*1024.0),
                         avg_degree, max_degree,
                         static_cast<double>(packed_edge_bytes)/(1024.0*1024.0));
        }
    }
    if (!encoder_feature || encoder_feature->ne[0] != cfg_.encoder_channels ||
        encoder_feature->ne[1] != static_cast<int64_t>(encoder_width) * encoder_height) {
        throw std::runtime_error("encoder feature matrix shape mismatch");
    }
    upload_feature(state->in, factor_coord, top.levels[0].coords.size());

    const size_t deep = cfg_.model_channels.size() - 1;

    if (!metal) {

        // Down path. Each phase saves its post-resblock feature in level[level]
        // (the future skip) and, except at the deepest level, writes the pooled
        // input for the next phase directly into level[level+1]. The latter is
        // overwritten in-place with that level's skip on the following phase.
        for (size_t level = 0; level < cfg_.model_channels.size(); ++level) {
            // The index arena is overwritten only after the previous phase has
            // synchronized, so peak topology residency is one level rather than
            // the sum of the full sparse pyramid.
            upload_neighbor(state->in, top.levels[level]);
            if (level < deep) upload_pool(state->in, top.transitions[level]);
            else upload_encoder_rows(state->in, top.levels[deep], encoder_width, encoder_height);

            ggml_context * c = make_phase_context();
            try {
                auto * neigh = neighbor_view(c, state->in, static_cast<int64_t>(top.levels[level].coords.size()));
                ggml_tensor * x = nullptr;
                if (level == 0) {
                    x = linear(c, w_, state->in.feature, "refiner.input_proj");
                } else {
                    x = state_read(c, state->level[level]);
                }

                for (int b = 0; b < cfg_.encoder_blocks[level]; ++b) {
                    x = res_block(c, w_, x, neigh,
                                  "refiner.down_stages." + std::to_string(level) + "." + std::to_string(b), level);
                }

                std::vector<ggml_tensor *> outputs;
                if (level < deep) {
                    const auto & tr = top.transitions[level];
                    auto * pool_rows = rows_view(c, state->in, static_cast<int64_t>(tr.pool_rows.size()));
                    auto * pool_inv = inv_view(c, state->in, static_cast<int64_t>(tr.inv_count.size()));
                    auto * pooled = pool_down(c, w_, x, pool_rows, pool_inv, tr.slots,
                                              "refiner.downsample_blocks." + std::to_string(level));
                    outputs.push_back(ggml_cpy(c, x, state->level[level]));
                    outputs.push_back(ggml_cpy(c, pooled, state->level[level + 1]));
                } else {
                    auto * encoder_rows = rows_view(c, state->in, static_cast<int64_t>(top.levels[deep].coords.size()));
                    auto * enc_sampled = ggml_get_rows(c, encoder_feature, encoder_rows);
                    auto * enc = linear(c, w_, enc_sampled, "refiner.encoder_fuse");
                    x = ggml_concat(c, x, enc, 0);
                    x = linear(c, w_, x, "refiner.fuse_proj.0");
                    x = ggml_silu(c, x);
                    x = linear(c, w_, x, "refiner.fuse_proj.2");
                    for (int b = 0; b < cfg_.bottleneck_blocks; ++b) {
                        x = res_block(c, w_, x, neigh,
                                      "refiner.bottleneck_stage." + std::to_string(b), deep);
                    }
                    outputs.push_back(ggml_cpy(c, x, state->level[deep]));
                }

                const std::string label = "down." + std::to_string(level);
                run_phase(sched, c, label.c_str(), outputs);
            } catch (...) {
                ggml_free(c);
                throw;
            }
            ggml_free(c);
        }

        // Up path. level[target] still contains the encoder skip. Once consumed,
        // reuse that same persistent allocation for the decoded feature. This cuts
        // the persistent feature working set roughly in half versus keeping a
        // second decoder pyramid alive.
        for (size_t u = 0; u < cfg_.downsample_factors.size(); ++u) {
            const size_t target = cfg_.model_channels.size() - 2 - u;
            upload_neighbor(state->in, top.levels[target]);
            upload_up(state->in, top.transitions[target]);

            ggml_context * c = make_phase_context();
            try {
                auto * neigh = neighbor_view(c, state->in, static_cast<int64_t>(top.levels[target].coords.size()));
                auto * up_rows = rows_view(c, state->in, static_cast<int64_t>(top.transitions[target].up_rows.size()));
                auto * x = nearest_up(c, w_, state_read(c, state->level[target + 1]), up_rows,
                                      "refiner.upsample_blocks." + std::to_string(u));
                x = add_safe(c, x, state_read(c, state->level[target]));
                for (int b = 0; b < cfg_.decoder_blocks[u]; ++b) {
                    x = res_block(c, w_, x, neigh,
                                  "refiner.up_stages." + std::to_string(u) + "." + std::to_string(b), target);
                }

                ggml_tensor * out = nullptr;
                if (target == 0) {
                    x = linear(c, w_, x, "refiner.out_proj");
                    out = ggml_cpy(c, x, state->residual);
                } else {
                    out = ggml_cpy(c, x, state->level[target]);
                }
                const std::string label = "up." + std::to_string(u) + ".to." + std::to_string(target);
                run_phase(sched, c, label.c_str(), {out});
            } catch (...) {
                ggml_free(c);
                throw;
            }
            ggml_free(c);
        }

    } else {
        // Hybrid Metal path: keep all non-convolution arithmetic in ggml, but
        // execute the two sparse 3x3 convolutions in each residual block using
        // the real-shape policy measured on M5. Each residual block first writes
        // LN(x)->SiLU to a reusable FP16 arena, then the Metal helper performs
        // conv1 -> FP16 scratch and conv2 -> F32 state + identity residual.
        for (size_t level = 0; level < cfg_.model_channels.size(); ++level) {
            if (level < deep) upload_pool(state->in, top.transitions[level]);
            else upload_encoder_rows(state->in, top.levels[deep], encoder_width, encoder_height);

            if (level == 0) {
                ggml_context * c = make_phase_context();
                try {
                    auto * x = linear(c, w_, state->in.feature, "refiner.input_proj");
                    auto * out = ggml_cpy(c, x, state->level[0]);
                    run_phase(sched, c, "down.0.input", {out});
                } catch (...) { ggml_free(c); throw; }
                ggml_free(c);
            }

            for (int b = 0; b < cfg_.encoder_blocks[level]; ++b) {
                const std::string pfx = "refiner.down_stages." + std::to_string(level) + "." + std::to_string(b);
                const std::string label = "down." + std::to_string(level) + ".block." + std::to_string(b);
                run_hybrid_resblock(sched, w_, *state, *metal, top.levels[level], level, pfx, label);
            }

            if (level < deep) {
                const auto & tr = top.transitions[level];
                ggml_context * c = make_phase_context();
                try {
                    auto * pool_rows = rows_view(c, state->in, static_cast<int64_t>(tr.pool_rows.size()));
                    auto * pool_inv = inv_view(c, state->in, static_cast<int64_t>(tr.inv_count.size()));
                    auto * pooled = pool_down(c, w_, state->level[level], pool_rows, pool_inv, tr.slots,
                                              "refiner.downsample_blocks." + std::to_string(level));
                    auto * out = ggml_cpy(c, pooled, state->level[level + 1]);
                    const std::string label = "down." + std::to_string(level) + ".pool";
                    run_phase(sched, c, label.c_str(), {out});
                } catch (...) { ggml_free(c); throw; }
                ggml_free(c);
            } else {
                ggml_context * c = make_phase_context();
                try {
                    auto * encoder_rows = rows_view(c, state->in, static_cast<int64_t>(top.levels[deep].coords.size()));
                    auto * enc_sampled = ggml_get_rows(c, encoder_feature, encoder_rows);
                    auto * enc = linear(c, w_, enc_sampled, "refiner.encoder_fuse");
                    auto * x = ggml_concat(c, state->level[deep], enc, 0);
                    x = linear(c, w_, x, "refiner.fuse_proj.0");
                    x = ggml_silu(c, x);
                    x = linear(c, w_, x, "refiner.fuse_proj.2");
                    auto * out = ggml_cpy(c, x, state->level[deep]);
                    run_phase(sched, c, "down.deep.fuse", {out});
                } catch (...) { ggml_free(c); throw; }
                ggml_free(c);
                for (int b = 0; b < cfg_.bottleneck_blocks; ++b) {
                    const std::string pfx = "refiner.bottleneck_stage." + std::to_string(b);
                    const std::string label = "bottleneck." + std::to_string(b);
                    run_hybrid_resblock(sched, w_, *state, *metal, top.levels[deep], deep, pfx, label);
                }
            }
        }

        // Decoder: first materialize nearest-up + skip into the target level,
        // then update that same persistent tensor in-place through custom sparse
        // residual blocks. This preserves the original one-feature-per-level
        // liveness plan.
        for (size_t u = 0; u < cfg_.downsample_factors.size(); ++u) {
            const size_t target = cfg_.model_channels.size() - 2 - u;
            upload_up(state->in, top.transitions[target]);
            ggml_context * c = make_phase_context();
            try {
                auto * up_rows = rows_view(c, state->in, static_cast<int64_t>(top.transitions[target].up_rows.size()));
                auto * x = nearest_up(c, w_, state->level[target + 1], up_rows,
                                      "refiner.upsample_blocks." + std::to_string(u));
                x = add_safe(c, x, state->level[target]);
                auto * out = ggml_cpy(c, x, state->level[target]);
                const std::string label = "up." + std::to_string(u) + ".to." + std::to_string(target) + ".merge";
                run_phase(sched, c, label.c_str(), {out});
            } catch (...) { ggml_free(c); throw; }
            ggml_free(c);

            for (int b = 0; b < cfg_.decoder_blocks[u]; ++b) {
                const std::string pfx = "refiner.up_stages." + std::to_string(u) + "." + std::to_string(b);
                const std::string label = "up." + std::to_string(u) + ".to." + std::to_string(target) + ".block." + std::to_string(b);
                run_hybrid_resblock(sched, w_, *state, *metal, top.levels[target], target, pfx, label);
            }

            if (target == 0) {
                c = make_phase_context();
                try {
                    auto * x = linear(c, w_, state->level[0], "refiner.out_proj");
                    auto * out = ggml_cpy(c, x, state->residual);
                    run_phase(sched, c, "up.final.out_proj", {out});
                } catch (...) { ggml_free(c); throw; }
                ggml_free(c);
            }
        }
    }

    const size_t n = static_cast<size_t>(width) * height;
    std::vector<float> residual(n * static_cast<size_t>(cfg_.out_channels));
    ggml_backend_tensor_get(state->residual, residual.data(), 0, residual.size() * sizeof(float));
    if (cfg_.out_channels != 1) throw std::runtime_error("MoGe-3 refiner out_channels != 1 unsupported");
    for (size_t i = 0; i < n; ++i) factor_coord[3*i + 2] += residual[i];
}


} // namespace moge
