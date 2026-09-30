#include "moge_ggml/moge.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__APPLE__) || defined(__unix__)
#include <sys/resource.h>
#endif

namespace {
moge::Backend backend_from(const std::string & s) {
    if (s == "auto") return moge::Backend::Auto;
    if (s == "cpu") return moge::Backend::CPU;
    if (s == "vulkan") return moge::Backend::Vulkan;
    if (s == "metal") return moge::Backend::Metal;
    throw std::runtime_error("--backend must be auto|cpu|vulkan|metal");
}

double percentile_nearest_rank(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t rank = std::max<size_t>(1, static_cast<size_t>(std::ceil(q * v.size())));
    return v[std::min(rank - 1, v.size() - 1)];
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

double peak_rss_mib() {
#if defined(__APPLE__)
    struct rusage u{};
    return getrusage(RUSAGE_SELF, &u) == 0 ? static_cast<double>(u.ru_maxrss) / (1024.0 * 1024.0) : 0.0;
#elif defined(__unix__)
    struct rusage u{};
    return getrusage(RUSAGE_SELF, &u) == 0 ? static_cast<double>(u.ru_maxrss) / 1024.0 : 0.0;
#else
    return 0.0;
#endif
}

std::string samples_csv(const std::vector<double> & v) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ',';
        os << v[i];
    }
    return os.str();
}

}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
            std::cout << "moge-bench " << moge::version_string() << "\nBenchmark deterministic MoGe inference.\n\n"
                      << "USAGE\n  moge-bench MODEL.moge [options]\n\n"
                      << "EXAMPLE\n  moge-bench v3_q8.moge --backend metal --width 1920 --height 1080 --runs 3\n\n"
                      << "OPTIONS\n"
                      << "  --backend <name>            auto, cpu, vulkan, or metal. Default: auto.\n"
                      << "  --width <px>                Synthetic input width. Default: 640.\n"
                      << "  --height <px>               Synthetic input height. Default: 480.\n"
                      << "  --tokens <n>                Internal token count; 0 uses model default. Default: 0.\n"
                      << "  --threads <n>               CPU thread count; 0 uses backend default. Default: 0.\n"
                      << "  --refine <n>                MoGe-3 refinement steps. Default: 3.\n"
                      << "  --warmup <n>                Warmup inferences. Default: 1.\n"
                      << "  --runs <n>                  Timed inferences. Default: 3.\n"
                      << "  --gpu-resident-inputs       Pin graph inputs to GPU where supported. Default: disabled.\n"
                      << "  --version                   Show version and exit.\n"
                      << "  -h, --help                  Show this help and exit.\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--version") { std::cout << "moge-bench " << moge::version_string() << "\n"; return 0; }
        if (argc < 2) {
            std::cerr << "moge-bench: missing MODEL.moge (run --help for usage)\n";
            return 2;
        }
        std::string model = argv[1], backend = "auto";
        int w = 640, h = 480, tokens = 0, threads = 0, refine = 3, warmup = 1, runs = 3;
        bool cpu_fallback = true;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            auto val = [&]() -> std::string { if (++i >= argc) throw std::runtime_error("missing value for " + a); return argv[i]; };
            if (a == "--backend") backend = val();
            else if (a == "--width") w = std::stoi(val());
            else if (a == "--height") h = std::stoi(val());
            else if (a == "--tokens") tokens = std::stoi(val());
            else if (a == "--threads") threads = std::stoi(val());
            else if (a == "--refine") refine = std::stoi(val());
            else if (a == "--warmup") warmup = std::stoi(val());
            else if (a == "--runs") runs = std::stoi(val());
            else if (a == "--gpu-resident-inputs" || a == "--no-cpu-fallback") cpu_fallback = false;
            else throw std::runtime_error("unknown option " + a);
        }
        if (w <= 0 || h <= 0 || runs <= 0 || warmup < 0) throw std::runtime_error("invalid benchmark dimensions/counts");

        std::vector<float> rgb(static_cast<size_t>(w) * h * 3);
        // Deterministic, non-constant signal; enough to exercise every path.
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            const size_t p = (static_cast<size_t>(y) * w + x) * 3;
            rgb[p + 0] = static_cast<float>(x) / std::max(1, w - 1);
            rgb[p + 1] = static_cast<float>(y) / std::max(1, h - 1);
            rgb[p + 2] = 0.5f + 0.25f * std::sin(0.013f * static_cast<float>(x + 3*y));
        }

        moge::LoadOptions lo;
        lo.backend = backend_from(backend);
        lo.threads = threads;
        lo.cpu_fallback = cpu_fallback;
        moge::Model m(model, lo);
        moge::InferOptions io;
        io.num_tokens = tokens;
        io.refine_steps = refine;
        io.apply_mask = false; // keep checksum finite for benchmark diagnostics

        for (int i = 0; i < warmup; ++i) (void) m.infer(rgb.data(), w, h, io);
        std::vector<double> ms;
        moge::Result last;
        for (int i = 0; i < runs; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            last = m.infer(rgb.data(), w, h, io);
            const auto t1 = std::chrono::steady_clock::now();
            ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        const double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size();
        const double best = *std::min_element(ms.begin(), ms.end());
        const double med = median(ms);
        const double p90 = percentile_nearest_rank(ms, 0.90);
        double variance = 0.0;
        for (double x : ms) variance += (x - mean) * (x - mean);
        const double stddev = std::sqrt(variance / ms.size());
        double checksum = 0.0;
        for (float z : last.depth) if (std::isfinite(z)) checksum += z;
        std::cout << "model_version=" << m.version() << "\nbackend=" << m.backend_name()
                  << "\nimage=" << w << "x" << h << "\nruns=" << runs
                  << "\nmean_ms=" << std::fixed << std::setprecision(3) << mean
                  << "\nmedian_ms=" << med
                  << "\np90_ms=" << p90
                  << "\nbest_ms=" << best
                  << "\nstddev_ms=" << stddev
                  << "\nmean_mpix_s=" << (static_cast<double>(w) * h / 1e6) / (mean / 1000.0)
                  << "\nsamples_ms=" << samples_csv(ms)
                  << "\npeak_rss_mib=" << std::setprecision(3) << peak_rss_mib()
                  << "\ndepth_checksum=" << std::setprecision(9) << checksum << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
