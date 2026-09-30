#include "moge_ggml/moge.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

moge::Backend backend_from(const std::string & s) {
    if (s == "auto") return moge::Backend::Auto;
    if (s == "cpu") return moge::Backend::CPU;
    if (s == "vulkan") return moge::Backend::Vulkan;
    if (s == "metal") return moge::Backend::Metal;
    throw std::runtime_error("--backend must be auto|cpu|vulkan|metal");
}

bool parse_bool(const std::string & s) {
    if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
    if (s == "0" || s == "false" || s == "no" || s == "off") return false;
    throw std::runtime_error("expected boolean 0|1|false|true, got: " + s);
}

template <typename T>
void write_binary(const fs::path & path, const std::vector<T> & data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open output " + path.string());
    if (!data.empty()) {
        f.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
    }
    if (!f) throw std::runtime_error("failed writing " + path.string());
}

std::vector<float> read_rgb(const fs::path & path, int width, int height) {
    const size_t n = static_cast<size_t>(width) * height * 3;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open input " + path.string());
    const auto bytes = f.tellg();
    if (bytes != static_cast<std::streamoff>(n * sizeof(float))) {
        std::ostringstream os;
        os << "RGB file size mismatch: expected " << n * sizeof(float) << " bytes, got " << bytes;
        throw std::runtime_error(os.str());
    }
    f.seekg(0);
    std::vector<float> rgb(n);
    f.read(reinterpret_cast<char *>(rgb.data()), static_cast<std::streamsize>(n * sizeof(float)));
    if (!f) throw std::runtime_error("failed reading RGB input");
    return rgb;
}

std::vector<float> deterministic_rgb(int width, int height) {
    std::vector<float> rgb(static_cast<size_t>(width) * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t p = (static_cast<size_t>(y) * width + x) * 3;
            rgb[p + 0] = static_cast<float>(x) / std::max(1, width - 1);
            rgb[p + 1] = static_cast<float>(y) / std::max(1, height - 1);
            rgb[p + 2] = 0.5f + 0.25f * std::sin(0.013f * static_cast<float>(x + 3*y));
        }
    }
    return rgb;
}

std::string json_escape(const std::string & s) {
    std::ostringstream os;
    for (char c : s) {
        switch (c) {
            case '\\': os << "\\\\"; break;
            case '"': os << "\\\""; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default: os << c;
        }
    }
    return os.str();
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: moge-dump MODEL.mogg --out DIR [--input RGB.f32] [--backend auto|cpu|vulkan|metal] "
                         "[--width 640] [--height 480] [--tokens N] [--resolution-level 9] [--threads N] "
                         "[--refine N] [--apply-mask 0|1] [--force-projection 0|1] [--fov-x DEG] "
                         "[--capture-raw-forward] [--gpu-resident-inputs]\n";
            return 2;
        }

        std::string model = argv[1];
        std::string backend = "auto";
        std::string input_path;
        std::string out_dir;
        int width = 640, height = 480, tokens = 0, resolution_level = 9, threads = 0, refine = 3;
        bool apply_mask = false, force_projection = true, cpu_fallback = true, capture_raw_forward = false;
        float fov_x = 0.0f;

        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto val = [&]() -> std::string {
                if (++i >= argc) throw std::runtime_error("missing value for " + a);
                return argv[i];
            };
            if (a == "--backend") backend = val();
            else if (a == "--input") input_path = val();
            else if (a == "--out") out_dir = val();
            else if (a == "--width") width = std::stoi(val());
            else if (a == "--height") height = std::stoi(val());
            else if (a == "--tokens") tokens = std::stoi(val());
            else if (a == "--resolution-level") resolution_level = std::stoi(val());
            else if (a == "--threads") threads = std::stoi(val());
            else if (a == "--refine") refine = std::stoi(val());
            else if (a == "--apply-mask") apply_mask = parse_bool(val());
            else if (a == "--force-projection") force_projection = parse_bool(val());
            else if (a == "--fov-x") fov_x = std::stof(val());
            else if (a == "--capture-raw-forward") capture_raw_forward = true;
            else if (a == "--gpu-resident-inputs" || a == "--no-cpu-fallback") cpu_fallback = false;
            else throw std::runtime_error("unknown option " + a);
        }

        if (out_dir.empty()) throw std::runtime_error("--out is required");
        if (width <= 0 || height <= 0 || refine < 0 || resolution_level < 0 || resolution_level > 9) {
            throw std::runtime_error("invalid dimensions/refinement/resolution arguments");
        }

        fs::create_directories(out_dir);
        std::vector<float> rgb = input_path.empty() ? deterministic_rgb(width, height) : read_rgb(input_path, width, height);

        moge::LoadOptions lo;
        lo.backend = backend_from(backend);
        lo.threads = threads;
        lo.cpu_fallback = cpu_fallback;
        moge::Model m(model, lo);

        moge::InferOptions io;
        io.num_tokens = tokens;
        io.resolution_level = resolution_level;
        io.refine_steps = refine;
        io.apply_mask = apply_mask;
        io.force_projection = force_projection;
        io.fov_x_degrees = fov_x;
        io.capture_raw_forward = capture_raw_forward;

        moge::Result r = m.infer(rgb.data(), width, height, io);
        const fs::path dir(out_dir);
        write_binary(dir / "input.rgb.f32", rgb);
        write_binary(dir / "points.f32", r.points);
        write_binary(dir / "depth.f32", r.depth);
        write_binary(dir / "normal.f32", r.normal);
        write_binary(dir / "mask.u8", r.mask);
        if (capture_raw_forward) {
            write_binary(dir / "raw-affine-points.f32", r.raw_affine_points);
            write_binary(dir / "raw-mask-probability.f32", r.raw_mask_probability);
        }

        std::vector<float> intrinsics(r.intrinsics, r.intrinsics + 9);
        write_binary(dir / "intrinsics.f32", intrinsics);

        std::ofstream meta(dir / "meta.json");
        if (!meta) throw std::runtime_error("cannot write meta.json");
        meta << std::setprecision(9)
             << "{\n"
             << "  \"format\": \"moge-dump-v1\",\n"
             << "  \"model\": \"" << json_escape(model) << "\",\n"
             << "  \"model_version\": " << m.version() << ",\n"
             << "  \"backend\": \"" << json_escape(m.backend_name()) << "\",\n"
             << "  \"width\": " << r.width << ",\n"
             << "  \"height\": " << r.height << ",\n"
             << "  \"requested_width\": " << width << ",\n"
             << "  \"requested_height\": " << height << ",\n"
             << "  \"num_tokens\": " << tokens << ",\n"
             << "  \"resolution_level\": " << resolution_level << ",\n"
             << "  \"refine_steps\": " << refine << ",\n"
             << "  \"apply_mask\": " << (apply_mask ? "true" : "false") << ",\n"
             << "  \"force_projection\": " << (force_projection ? "true" : "false") << ",\n"
             << "  \"fov_x_degrees\": " << fov_x << ",\n"
             << "  \"metric_scale\": " << r.metric_scale << ",\n"
             << "  \"capture_raw_forward\": " << (capture_raw_forward ? "true" : "false") << ",\n"
             << "  \"raw_metric_scale\": " << r.raw_metric_scale << ",\n"
             << "  \"has_raw_affine_points\": " << (!r.raw_affine_points.empty() ? "true" : "false") << ",\n"
             << "  \"has_raw_mask_probability\": " << (!r.raw_mask_probability.empty() ? "true" : "false") << ",\n"
             << "  \"has_normal\": " << (!r.normal.empty() ? "true" : "false") << ",\n"
             << "  \"has_mask\": " << (!r.mask.empty() ? "true" : "false") << ",\n"
             << "  \"float_byte_order\": \"native-little-endian-required\"\n"
             << "}\n";

        std::cout << "backend=" << m.backend_name() << "\n"
                  << "model_version=" << m.version() << "\n"
                  << "output=" << fs::absolute(dir).string() << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
