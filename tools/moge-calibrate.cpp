#include "moge_ggml/moge.hpp"

#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_MAX_DIMENSIONS 8192
#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace fs = std::filesystem;

namespace {

moge::Backend backend_from(const std::string & s) {
    if (s == "metal") return moge::Backend::Metal;
    if (s == "cpu") return moge::Backend::CPU;
    if (s == "auto") return moge::Backend::Auto;
    throw std::runtime_error("--backend must be metal|cpu|auto");
}

std::string lower(std::string s) {
    for (char & c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool supported_image(const fs::path & p) {
    static const std::set<std::string> exts = {".jpg", ".jpeg", ".png"};
    return exts.count(lower(p.extension().string())) != 0;
}

std::vector<fs::path> list_images(const fs::path & root) {
    std::vector<fs::path> out;
    if (!fs::exists(root)) throw std::runtime_error("image directory does not exist: " + root.string());
    for (const auto & e : fs::recursive_directory_iterator(root, fs::directory_options::skip_permission_denied)) {
        std::error_code ec;
        if (!e.is_regular_file(ec) || ec) continue;
        if (supported_image(e.path())) out.push_back(e.path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<fs::path> spread_sample(const std::vector<fs::path> & all, size_t limit) {
    if (limit == 0 || all.size() <= limit) return all;
    std::vector<fs::path> out;
    out.reserve(limit);
    if (limit == 1) { out.push_back(all[all.size()/2]); return out; }
    for (size_t i = 0; i < limit; ++i) {
        const size_t idx = static_cast<size_t>(std::llround(static_cast<double>(i) * (all.size()-1) / (limit-1)));
        out.push_back(all[std::min(idx, all.size()-1)]);
    }
    return out;
}

std::pair<int,int> bucket_for(int sw, int sh, int long_side) {
    const double ar = static_cast<double>(sw) / sh;
    if (ar >= 1.20) return {long_side, (long_side * 3) / 4};
    if (ar <= 1.0/1.20) return {(long_side * 3) / 4, long_side};
    const int sq = (long_side * 6) / 7; // 768 when long_side=896
    return {sq, sq};
}

std::vector<float> resize_cover_rgb(const uint8_t * src, int sw, int sh, int tw, int th) {
    if (!src || sw <= 0 || sh <= 0 || tw <= 0 || th <= 0) throw std::runtime_error("invalid image resize");
    std::vector<float> out(static_cast<size_t>(tw) * th * 3);
    const double scale = std::max(static_cast<double>(tw)/sw, static_cast<double>(th)/sh);
    const double scaled_w = sw * scale, scaled_h = sh * scale;
    const double crop_x = 0.5 * (scaled_w - tw), crop_y = 0.5 * (scaled_h - th);
    for (int y = 0; y < th; ++y) {
        const double sy = (y + crop_y + 0.5) / scale - 0.5;
        const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, sh-1);
        const int y1 = std::min(y0 + 1, sh-1);
        const float fy = static_cast<float>(std::clamp(sy - std::floor(sy), 0.0, 1.0));
        for (int x = 0; x < tw; ++x) {
            const double sx = (x + crop_x + 0.5) / scale - 0.5;
            const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, sw-1);
            const int x1 = std::min(x0 + 1, sw-1);
            const float fx = static_cast<float>(std::clamp(sx - std::floor(sx), 0.0, 1.0));
            for (int c = 0; c < 3; ++c) {
                const float a = src[(static_cast<size_t>(y0)*sw+x0)*3+c] * (1.0f-fx) + src[(static_cast<size_t>(y0)*sw+x1)*3+c] * fx;
                const float b = src[(static_cast<size_t>(y1)*sw+x0)*3+c] * (1.0f-fx) + src[(static_cast<size_t>(y1)*sw+x1)*3+c] * fx;
                out[(static_cast<size_t>(y)*tw+x)*3+c] = (a*(1.0f-fy)+b*fy) * (1.0f/255.0f);
            }
        }
    }
    return out;
}

std::vector<float> synthetic_image(int index, int w, int h) {
    std::vector<float> out(static_cast<size_t>(w)*h*3);
    const float phase=0.31f*index;
    for (int y=0;y<h;++y) for(int x=0;x<w;++x) {
        const size_t p=(static_cast<size_t>(y)*w+x)*3;
        out[p+0]=static_cast<float>((x+17*index)%std::max(w,2))/std::max(1,w-1);
        out[p+1]=static_cast<float>((y+11*index)%std::max(h,2))/std::max(1,h-1);
        out[p+2]=std::clamp(0.5f+0.35f*std::sin(0.017f*x+0.013f*y+phase),0.0f,1.0f);
    }
    return out;
}

double peak_rss_gib() {
    rusage u{};
    if (getrusage(RUSAGE_SELF,&u)!=0) return 0.0;
#if defined(__APPLE__)
    return static_cast<double>(u.ru_maxrss)/(1024.0*1024.0*1024.0);
#else
    return static_cast<double>(u.ru_maxrss)*1024.0/(1024.0*1024.0*1024.0);
#endif
}

std::string tail_name(const std::string & s, size_t n=38) {
    if (s.size() <= n) return s;
    return "..." + s.substr(s.size()-(n-3));
}

void progress(size_t done, size_t total, int w, int h, double sec, const std::string & name) {
    constexpr int barw=28;
    const double frac=total?static_cast<double>(done)/total:0.0;
    const int fill=static_cast<int>(std::round(frac*barw));
    std::ostringstream os;
    os << '\r' << '[';
    for(int i=0;i<barw;++i) os << (i<fill?'#':'-');
    os << "] " << std::setw(3) << done << '/' << total << ' '
       << std::setw(5) << std::fixed << std::setprecision(1) << (100.0*frac) << "% "
       << w << 'x' << h << ' ' << std::setw(7) << std::setprecision(1) << sec << "s "
       << "rss_peak=" << std::setprecision(2) << peak_rss_gib() << "GiB " << tail_name(name);
    std::cerr << os.str() << std::flush;
    if (done==total) std::cerr << '\n';
}

std::string json_escape(const std::string & s) {
    std::ostringstream os;
    for (char c : s) {
        if (c=='\\' || c=='\"') os << '\\' << c;
        else if (c=='\n') os << "\\n";
        else os << c;
    }
    return os.str();
}

} // namespace

int main(int argc,char ** argv) {
    try {
        if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
            std::cout << "moge-calibrate " << moge::version_string() << "\nCollect activation-importance calibration data for model conversion and analysis.\n\n"
                      << "USAGE\n  moge-calibrate MODEL.moge --output FILE.mogi [options]\n\n"
                      << "EXAMPLE\n  moge-calibrate v3_f16.moge --output calibration.mogi --images ./images\n\n"
                      << "OPTIONS\n"
                      << "  --output <file>             Calibration output. Required.\n"
                      << "  --images <dir>              PNG/JPEG calibration directory. Default: none.\n"
                      << "  --limit <n>                 Maximum real images. Default: 128.\n"
                      << "  --synthetic-count <n>       Synthetic calibration samples. Default: 8.\n"
                      << "  --backend <name>            Compute backend. Default: metal.\n"
                      << "  --resolution-level <n>      MoGe resolution level. Default: 9.\n"
                      << "  --bucket-long-side <px>     Calibration resize long side. Default: 896.\n"
                      << "  --max-source-pixels <n>     Maximum decoded source pixels. Default: 33554432.\n"
                      << "  --summary <file>            Optional JSON summary. Default: none.\n"
                      << "  --verify-first              Verify first sample. Default: enabled.\n"
                      << "  --no-verify-first           Disable first-sample verification.\n"
                      << "  --version                   Show version and exit.\n"
                      << "  -h, --help                  Show this help and exit.\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--version") { std::cout << "moge-calibrate " << moge::version_string() << "\n"; return 0; }
        if (argc < 2) {
            std::cerr << "moge-calibrate: missing MODEL.moge (run --help for usage)\n";
            return 2;
        }
        std::string model=argv[1], output, images, backend="metal", summary;
        int limit=128, synthetic_count=8, resolution_level=9, bucket_long=896;
        uint64_t max_source_pixels=33554432ull;
        bool verify_first=true;
        for (int i=2;i<argc;++i) {
            const std::string a=argv[i];
            auto val=[&](){if(++i>=argc) throw std::runtime_error("missing value for "+a); return std::string(argv[i]);};
            if(a=="--output") output=val();
            else if(a=="--images") images=val();
            else if(a=="--limit") limit=std::stoi(val());
            else if(a=="--synthetic-count") synthetic_count=std::stoi(val());
            else if(a=="--backend") backend=val();
            else if(a=="--resolution-level") resolution_level=std::stoi(val());
            else if(a=="--bucket-long-side") bucket_long=std::stoi(val());
            else if(a=="--max-source-pixels") max_source_pixels=static_cast<uint64_t>(std::stoull(val()));
            else if(a=="--summary") summary=val();
            else if(a=="--verify-first") verify_first=true;
            else if(a=="--no-verify-first") verify_first=false;
            else throw std::runtime_error("unknown option "+a);
        }
        if(output.empty()) throw std::runtime_error("--output is required");
        if(limit<1 || synthetic_count<1 || bucket_long<224) throw std::runtime_error("invalid calibration count/shape option");
        std::cerr << "native calibration verify_first=" << (verify_first ? "on" : "off") << '\n';

        moge::LoadOptions lo;
        lo.backend=backend_from(backend);
        lo.cpu_fallback=false;
        moge::Model m(model,lo);
        m.calibration_reset();
        moge::CalibrationOptions co;
        co.resolution_level=resolution_level;
        co.verify_first=verify_first;

        std::vector<fs::path> selected;
        if(!images.empty()) selected=spread_sample(list_images(images),static_cast<size_t>(limit));
        const size_t total=images.empty()?static_cast<size_t>(synthetic_count):selected.size();
        if(total==0) throw std::runtime_error("no supported calibration images found");
        const auto started=std::chrono::steady_clock::now();
        progress(0,total,0,0,0.0,"starting");

        size_t completed=0, skipped=0;
        if(images.empty()) {
            for(int i=0;i<synthetic_count;++i) {
                const int short_side=(bucket_long*3)/4, square=(bucket_long*6)/7;
                const auto [w,h]=(i%3==0?std::pair<int,int>{bucket_long,short_side}:i%3==1?std::pair<int,int>{short_side,bucket_long}:std::pair<int,int>{square,square});
                auto rgb=synthetic_image(i,w,h);
                m.calibration_add(rgb.data(),w,h,co);
                ++completed;
                const double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
                progress(completed,total,w,h,sec,"synthetic-"+std::to_string(i));
            }
        } else {
            for(size_t i=0;i<selected.size();++i) {
                const auto & path=selected[i];
                int sw=0,sh=0,comp=0;
                std::error_code size_ec;
                const uintmax_t encoded_bytes = fs::file_size(path, size_ec);
                if (size_ec || encoded_bytes == 0 || encoded_bytes > 128ull*1024ull*1024ull) {
                    ++skipped; std::cerr << "\nwarning: encoded image exceeds 128 MiB cap or cannot be sized, skipping: " << path << '\n';
                    const double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
                    progress(i+1,total,0,0,sec,path.string()+" [skipped]"); continue;
                }
                if(!stbi_info(path.string().c_str(),&sw,&sh,&comp) || sw<=0 || sh<=0) {
                    ++skipped; std::cerr << "\nwarning: cannot inspect image, skipping: " << path << '\n';
                    const double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
                    progress(i+1,total,0,0,sec,path.string()+" [skipped]"); continue;
                }
                const uint64_t source_pixels=static_cast<uint64_t>(sw)*sh;
                if(max_source_pixels>0 && source_pixels>max_source_pixels) {
                    ++skipped; std::cerr << "\nwarning: source image exceeds hard decode cap (" << sw << 'x' << sh << "), skipping: " << path << '\n';
                    const double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
                    progress(i+1,total,0,0,sec,path.string()+" [skipped]"); continue;
                }
                uint8_t * raw=stbi_load(path.string().c_str(),&sw,&sh,&comp,3);
                if(!raw) { ++skipped; std::cerr << "\nwarning: decode failed, skipping: " << path << " (" << stbi_failure_reason() << ")\n";
                    const double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
                    progress(i+1,total,0,0,sec,path.string()+" [skipped]"); continue; }
                const auto [tw,th]=bucket_for(sw,sh,bucket_long);
                auto rgb=resize_cover_rgb(raw,sw,sh,tw,th);
                stbi_image_free(raw);
                m.calibration_add(rgb.data(),tw,th,co);
                ++completed;
                const double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
                progress(i+1,total,tw,th,sec,path.string());
            }
        }
        if(completed==0) throw std::runtime_error("all calibration images were skipped");
        if (verify_first && !m.calibration_verification_performed())
            throw std::runtime_error("requested Metal calibration verification never ran");
        m.calibration_write_mogi(output);
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        std::cerr << "native calibration complete: images=" << completed << " skipped=" << skipped
                  << " records=" << m.calibration_record_count() << " seconds=" << std::fixed << std::setprecision(2) << seconds
                  << " peak_rss_gib=" << std::setprecision(3) << peak_rss_gib() << '\n';
        if(!summary.empty()) {
            std::ofstream f(summary);
            if(!f) throw std::runtime_error("cannot open summary output: "+summary);
            f << "{\n"
              << "  \"schema\": 1,\n"
              << "  \"method\": \"native_metal_activation_input_mean_square\",\n"
              << "  \"model\": \"" << json_escape(model) << "\",\n"
              << "  \"backend\": \"" << json_escape(m.backend_name()) << "\",\n"
              << "  \"images\": " << completed << ",\n"
              << "  \"skipped\": " << skipped << ",\n"
              << "  \"records\": " << m.calibration_record_count() << ",\n"
              << "  \"resolution_level\": " << resolution_level << ",\n"
              << "  \"bucket_long_side\": " << bucket_long << ",\n"
              << "  \"max_source_pixels\": " << max_source_pixels << ",\n"
              << "  \"seconds\": " << std::fixed << std::setprecision(3) << seconds << ",\n"
              << "  \"peak_rss_gib\": " << std::setprecision(4) << peak_rss_gib() << ",\n"
              << "  \"verify_first\": " << (verify_first ? "true" : "false") << ",\n"
              << "  \"verification_performed\": " << (m.calibration_verification_performed() ? "true" : "false") << "\n"
              << "}\n";
        }
        return 0;
    } catch(const std::exception & e) {
        std::cerr << "\nerror: " << e.what() << '\n';
        return 1;
    }
}
