#include <moge_ggml/moge.hpp>
#include <ggml.h>

// Keep the CLI decoder surface intentionally small: PNG/JPEG are enough for
// the normal user workflow and avoiding unused legacy image formats reduces
// parser attack surface for untrusted inputs.
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_MAX_DIMENSIONS 8192
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
#include <sys/resource.h>

namespace fs = std::filesystem;

namespace {

constexpr int kHardMinImageSide = 32;
constexpr int kHardMaxImageSide = 8192;
constexpr uint64_t kHardMaxImagePixels = 16ull * 1024ull * 1024ull;
constexpr uint64_t kHardMaxInputBytes = 128ull * 1024ull * 1024ull;
constexpr int kHardMaxTokens = 65536;
constexpr int kHardMaxRefineSteps = 16;

struct Args {
    std::string model;
    std::string image;
    fs::path output = "moge-output";
    moge::Backend backend = moge::Backend::Auto;
    moge::InferOptions infer;
    int min_image_side = kHardMinImageSide;
    int max_image_side = kHardMaxImageSide;
    uint64_t max_image_pixels = kHardMaxImagePixels;
    uint64_t max_input_bytes = kHardMaxInputBytes;
    bool verbose = false;
};

[[noreturn]] void usage_error(const std::string & msg) {
    throw std::runtime_error(msg);
}

void usage(const char * argv0) {
    std::cout
        << "moge-cli " << moge::version_string() << "\n"
        << "Run MoGe geometry inference on a PNG or JPEG image.\n\n"
        << "USAGE\n  " << argv0 << " --model MODEL.moge [options] IMAGE\n\n"
        << "EXAMPLE\n  " << argv0 << " --model v3_q8.moge --backend metal --output output photo.jpg\n\n"
        << "OPTIONS\n"
        << "  -m, --model <file>         Model file. Required.\n"
        << "  -o, --output <dir>         Output directory. Default: moge-output.\n"
        << "  -b, --backend <name>       auto, cpu, metal, or vulkan. Default: auto.\n"
        << "  --resolution-level <n>     MoGe resolution level 0..9. Default: 9.\n"
        << "  --tokens <n>               Internal token count; 0 uses model default. Default: 0.\n"
        << "  --refine <n>               MoGe-3 refinement steps 0..16. Default: 3.\n"
        << "  --fov-x <degrees>           Horizontal FOV; 0 uses automatic estimate. Default: 0.\n"
        << "  --no-mask                   Do not apply predicted validity mask. Default: mask enabled.\n"
        << "  --min-image-side <px>       Reject smaller images. Default/hard minimum: 32.\n"
        << "  --max-image-side <px>       Reject larger images. Default/hard maximum: 8192.\n"
        << "  --max-image-pixels <n>      Reject excessive decoded pixels. Default/hard maximum: 16777216.\n"
        << "  --max-input-mib <MiB>       Reject excessive encoded input size. Default/hard maximum: 128.\n"
        << "  -v, --verbose               Show ggml/backend diagnostic logs. Default: disabled.\n"
        << "  --version                   Show version and exit.\n"
        << "  -h, --help                  Show this help and exit.\n";
}


void quiet_ggml_log(enum ggml_log_level level, const char * text, void *) {
    // Normal CLI output should describe the inference, not Metal kernel setup.
    // Keep warnings/errors visible; --verbose restores ggml's default logger.
    if (level >= GGML_LOG_LEVEL_WARN && text) std::cerr << text;
}

double seconds_since(const std::chrono::steady_clock::time_point & t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

double peak_rss_mib() {
    struct rusage u {};
    if (getrusage(RUSAGE_SELF, &u) != 0) return 0.0;
#if defined(__APPLE__)
    return static_cast<double>(u.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(u.ru_maxrss) / 1024.0;
#endif
}

moge::Backend parse_backend(const std::string & s) {
    if (s == "auto") return moge::Backend::Auto;
    if (s == "cpu") return moge::Backend::CPU;
    if (s == "metal") return moge::Backend::Metal;
    if (s == "vulkan") return moge::Backend::Vulkan;
    throw std::runtime_error("unknown backend: " + s);
}

int parse_int_strict(const std::string & s, const char * flag) {
    int v = 0;
    const char * b = s.data();
    const char * e = b + s.size();
    const auto r = std::from_chars(b, e, v);
    if (r.ec != std::errc{} || r.ptr != e) usage_error(std::string(flag) + " requires an integer");
    return v;
}

uint64_t parse_u64_strict(const std::string & s, const char * flag) {
    uint64_t v = 0;
    const char * b = s.data();
    const char * e = b + s.size();
    const auto r = std::from_chars(b, e, v);
    if (r.ec != std::errc{} || r.ptr != e) usage_error(std::string(flag) + " requires a non-negative integer");
    return v;
}

float parse_float_strict(const std::string & s, const char * flag) {
    char * end = nullptr;
    errno = 0;
    const float v = std::strtof(s.c_str(), &end);
    if (errno != 0 || end != s.c_str() + s.size() || !std::isfinite(v))
        usage_error(std::string(flag) + " requires a finite number");
    return v;
}

size_t checked_mul(size_t a, size_t b, const char * what) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
        throw std::runtime_error(std::string(what) + " size overflow");
    return a * b;
}

Args parse_args(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto value = [&](const char * flag) -> std::string {
            if (++i >= argc) throw std::runtime_error(std::string(flag) + " requires a value");
            return argv[i];
        };
        if (s == "-h" || s == "--help") { usage(argv[0]); std::exit(0); }
        else if (s == "--version") { std::cout << "moge-ggml " << moge::version_string() << " (API " << moge::API_VERSION << ")\n"; std::exit(0); }
        else if (s == "--model" || s == "-m") a.model = value("--model");
        else if (s == "--output" || s == "-o") a.output = value("--output");
        else if (s == "--backend" || s == "-b") a.backend = parse_backend(value("--backend"));
        else if (s == "--resolution-level") a.infer.resolution_level = parse_int_strict(value("--resolution-level"), "--resolution-level");
        else if (s == "--tokens") a.infer.num_tokens = parse_int_strict(value("--tokens"), "--tokens");
        else if (s == "--refine") a.infer.refine_steps = parse_int_strict(value("--refine"), "--refine");
        else if (s == "--fov-x") a.infer.fov_x_degrees = parse_float_strict(value("--fov-x"), "--fov-x");
        else if (s == "--no-mask") a.infer.apply_mask = false;
        else if (s == "--verbose" || s == "-v") a.verbose = true;
        else if (s == "--min-image-side") a.min_image_side = parse_int_strict(value("--min-image-side"), "--min-image-side");
        else if (s == "--max-image-side") a.max_image_side = parse_int_strict(value("--max-image-side"), "--max-image-side");
        else if (s == "--max-image-pixels") a.max_image_pixels = parse_u64_strict(value("--max-image-pixels"), "--max-image-pixels");
        else if (s == "--max-input-mib") {
            const uint64_t mib = parse_u64_strict(value("--max-input-mib"), "--max-input-mib");
            if (mib > kHardMaxInputBytes / (1024ull * 1024ull)) usage_error("--max-input-mib exceeds hard limit of 128 MiB");
            a.max_input_bytes = mib * 1024ull * 1024ull;
        }
        else if (!s.empty() && s[0] == '-') throw std::runtime_error("unknown option: " + s);
        else if (a.image.empty()) a.image = s;
        else throw std::runtime_error("only one input image is supported per invocation");
    }
    if (a.model.empty()) throw std::runtime_error("--model is required");
    if (a.image.empty()) throw std::runtime_error("input image is required");
    if (a.infer.resolution_level < 0 || a.infer.resolution_level > 9) usage_error("--resolution-level must be in 0..9");
    if (a.infer.num_tokens < 0 || a.infer.num_tokens > kHardMaxTokens) usage_error("--tokens must be 0..65536 (0 selects model default)");
    if (a.infer.refine_steps < 0 || a.infer.refine_steps > kHardMaxRefineSteps) usage_error("--refine must be in 0..16");
    if (a.infer.fov_x_degrees != 0.0f && !(a.infer.fov_x_degrees >= 1.0f && a.infer.fov_x_degrees < 179.0f)) usage_error("--fov-x must be 0 (auto) or in [1,179)");
    if (a.min_image_side < kHardMinImageSide) usage_error("--min-image-side cannot be below hard minimum 32");
    if (a.max_image_side <= 0 || a.max_image_side > kHardMaxImageSide) usage_error("--max-image-side must be 1..8192");
    if (a.min_image_side > a.max_image_side) usage_error("minimum image side exceeds maximum image side");
    if (a.max_image_pixels == 0 || a.max_image_pixels > kHardMaxImagePixels) usage_error("--max-image-pixels must be 1..16777216");
    if (a.max_input_bytes == 0 || a.max_input_bytes > kHardMaxInputBytes) usage_error("encoded input limit must be 1..128 MiB");
    return a;
}

std::vector<uint8_t> read_bounded_file(const fs::path & path, uint64_t max_bytes) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec) || ec) throw std::runtime_error("input image is not a regular file");
    const uintmax_t n = fs::file_size(path, ec);
    if (ec) throw std::runtime_error("cannot stat input image");
    if (n == 0) throw std::runtime_error("input image is empty");
    if (n > max_bytes) throw std::runtime_error("encoded input exceeds configured size limit");
    if (n > static_cast<uintmax_t>(std::numeric_limits<int>::max())) throw std::runtime_error("encoded input is too large for image decoder");
    std::vector<uint8_t> bytes(static_cast<size_t>(n));
    std::ifstream f(path, std::ios::binary);
    if (!f || !f.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error("failed reading input image");
    return bytes;
}

void validate_image_dimensions(int w, int h, const Args & args) {
    if (w <= 0 || h <= 0) throw std::runtime_error("image decoder returned invalid dimensions");
    if (w < args.min_image_side || h < args.min_image_side)
        throw std::runtime_error("image is below minimum side length of " + std::to_string(args.min_image_side));
    if (w > args.max_image_side || h > args.max_image_side)
        throw std::runtime_error("image exceeds maximum side length of " + std::to_string(args.max_image_side));
    const uint64_t pixels = static_cast<uint64_t>(w) * static_cast<uint64_t>(h);
    if (pixels > args.max_image_pixels)
        throw std::runtime_error("image exceeds maximum pixel count of " + std::to_string(args.max_image_pixels));
}

void validate_result(const moge::Result & r, int expected_w, int expected_h) {
    if (r.width != expected_w || r.height != expected_h || r.width <= 0 || r.height <= 0)
        throw std::runtime_error("model returned unexpected output dimensions");
    const size_t n = checked_mul(static_cast<size_t>(r.width), static_cast<size_t>(r.height), "output pixel");
    if (r.depth.size() != n) throw std::runtime_error("model returned malformed depth output");
    if (r.points.size() != checked_mul(n, size_t{3}, "point output")) throw std::runtime_error("model returned malformed point output");
    if (!r.normal.empty() && r.normal.size() != checked_mul(n, size_t{3}, "normal output")) throw std::runtime_error("model returned malformed normal output");
    if (!r.mask.empty() && r.mask.size() != n) throw std::runtime_error("model returned malformed mask output");
}

void write_pfm(const fs::path & path, const std::vector<float> & depth, int w, int h) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot create " + path.string());
    f << "Pf\n" << w << ' ' << h << "\n-1.0\n";
    for (int y = h - 1; y >= 0; --y) {
        f.write(reinterpret_cast<const char *>(depth.data() + static_cast<size_t>(y) * static_cast<size_t>(w)),
                static_cast<std::streamsize>(static_cast<size_t>(w) * sizeof(float)));
    }
    if (!f) throw std::runtime_error("failed writing " + path.string());
}

void write_depth_preview(const fs::path & path, const moge::Result & r) {
    float lo = std::numeric_limits<float>::infinity();
    float hi = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < r.depth.size(); ++i) {
        if (!r.mask.empty() && !r.mask[i]) continue;
        const float d = r.depth[i];
        if (!std::isfinite(d)) continue;
        lo = std::min(lo, d); hi = std::max(hi, d);
    }
    if (!(hi > lo)) { lo = 0.0f; hi = 1.0f; }
    std::vector<uint8_t> img(r.depth.size());
    for (size_t i = 0; i < r.depth.size(); ++i) {
        if ((!r.mask.empty() && !r.mask[i]) || !std::isfinite(r.depth[i])) { img[i] = 0; continue; }
        const float t = std::clamp((r.depth[i] - lo) / (hi - lo), 0.0f, 1.0f);
        img[i] = static_cast<uint8_t>(std::lround(t * 255.0f));
    }
    if (!stbi_write_png(path.string().c_str(), r.width, r.height, 1, img.data(), r.width))
        throw std::runtime_error("failed writing " + path.string());
}

void write_mask(const fs::path & path, const moge::Result & r) {
    if (r.mask.empty()) return;
    std::vector<uint8_t> img(r.mask.size());
    for (size_t i = 0; i < img.size(); ++i) img[i] = r.mask[i] ? 255 : 0;
    if (!stbi_write_png(path.string().c_str(), r.width, r.height, 1, img.data(), r.width))
        throw std::runtime_error("failed writing " + path.string());
}

void write_normal(const fs::path & path, const moge::Result & r) {
    if (r.normal.empty()) return;
    std::vector<uint8_t> img(r.normal.size());
    for (size_t i = 0; i < img.size(); ++i) {
        const float v = std::isfinite(r.normal[i]) ? r.normal[i] : 0.0f;
        img[i] = static_cast<uint8_t>(std::lround(std::clamp(v * 0.5f + 0.5f, 0.0f, 1.0f) * 255.0f));
    }
    if (!stbi_write_png(path.string().c_str(), r.width, r.height, 3, img.data(), r.width * 3))
        throw std::runtime_error("failed writing " + path.string());
}

void write_ply(const fs::path & path, const moge::Result & r) {
    const size_t n = static_cast<size_t>(r.width) * static_cast<size_t>(r.height);
    size_t valid = 0;
    for (size_t i = 0; i < n; ++i) {
        const float * p = r.points.data() + i * 3;
        if ((!r.mask.empty() && !r.mask[i]) || !std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) continue;
        ++valid;
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot create " + path.string());
    f << "ply\nformat binary_little_endian 1.0\n"
      << "element vertex " << valid << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "end_header\n";
    for (size_t i = 0; i < n; ++i) {
        const float * p = r.points.data() + i * 3;
        if ((!r.mask.empty() && !r.mask[i]) || !std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) continue;
        f.write(reinterpret_cast<const char *>(p), 3 * sizeof(float));
    }
    if (!f) throw std::runtime_error("failed writing " + path.string());
}

void write_camera(const fs::path & path, const moge::Result & r, const std::string & backend) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot create " + path.string());
    f << "{\n  \"width\": " << r.width << ",\n  \"height\": " << r.height
      << ",\n  \"backend\": \"" << backend << "\",\n  \"metric_scale\": " << r.metric_scale
      << ",\n  \"intrinsics\": [";
    for (int i = 0; i < 9; ++i) { if (i) f << ", "; f << r.intrinsics[i]; }
    f << "]\n}\n";
    if (!f) throw std::runtime_error("failed writing " + path.string());
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const auto total_t0 = std::chrono::steady_clock::now();
        const Args args = parse_args(argc, argv);
        if (!args.verbose) ggml_log_set(quiet_ggml_log, nullptr);

        std::cerr << "[1/4] Reading image..." << std::flush;
        const std::vector<uint8_t> encoded = read_bounded_file(args.image, args.max_input_bytes);

        int w = 0, h = 0, comp = 0;
        if (!stbi_info_from_memory(encoded.data(), static_cast<int>(encoded.size()), &w, &h, &comp))
            throw std::runtime_error(std::string("cannot inspect image: ") + stbi_failure_reason());
        validate_image_dimensions(w, h, args);

        stbi_uc * pixels = stbi_load_from_memory(encoded.data(), static_cast<int>(encoded.size()), &w, &h, &comp, 3);
        if (!pixels) throw std::runtime_error(std::string("cannot load image: ") + stbi_failure_reason());
        validate_image_dimensions(w, h, args);
        const size_t rgb_count = checked_mul(checked_mul(static_cast<size_t>(w), static_cast<size_t>(h), "image"), size_t{3}, "RGB image");
        std::vector<float> rgb(rgb_count);
        for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = pixels[i] / 255.0f;
        stbi_image_free(pixels);
        std::cerr << " done (" << w << "x" << h << ")\n";

        std::cerr << "[2/4] Loading model..." << std::flush;
        const auto load_t0 = std::chrono::steady_clock::now();
        moge::LoadOptions load;
        load.backend = args.backend;
        moge::Model model(args.model, load);
        std::cerr << " done (" << model.backend_name() << ", " << std::fixed << std::setprecision(2) << seconds_since(load_t0) << " s)\n";
        std::cerr << "[3/4] Inferring..." << std::flush;
        const auto infer_t0 = std::chrono::steady_clock::now();
        const moge::Result result = model.infer(rgb.data(), w, h, args.infer);
        const double infer_s = seconds_since(infer_t0);
        std::cerr << " done (" << std::fixed << std::setprecision(2) << infer_s << " s)\n";
        validate_result(result, w, h);

        std::cerr << "[4/4] Writing outputs..." << std::flush;
        std::error_code ec;
        if (fs::exists(args.output, ec) && !ec && !fs::is_directory(args.output, ec))
            throw std::runtime_error("output path exists and is not a directory");
        fs::create_directories(args.output);
        write_pfm(args.output / "depth.pfm", result.depth, result.width, result.height);
        write_depth_preview(args.output / "depth.png", result);
        write_ply(args.output / "points.ply", result);
        write_mask(args.output / "mask.png", result);
        write_normal(args.output / "normal.png", result);
        write_camera(args.output / "camera.json", result, model.backend_name());
        std::cerr << " done\n";
        const double total_s = seconds_since(total_t0);
        const double rss = peak_rss_mib();
        std::cout << "Done\n"
                  << "  backend:   " << model.backend_name() << "\n"
                  << "  image:     " << w << "x" << h << "\n"
                  << "  inference: " << std::fixed << std::setprecision(2) << infer_s << " s\n"
                  << "  total:     " << std::fixed << std::setprecision(2) << total_s << " s\n";
        if (rss > 0.0) std::cout << "  peak RSS:  " << std::fixed << std::setprecision(1) << rss << " MiB\n";
        std::cout << "  output:    " << args.output.string() << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "moge-cli: " << e.what() << "\n";
        return 1;
    }
}
