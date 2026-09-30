#include <ggml.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::string lower(std::string s) {
    for (char & c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool matches(ggml_backend_dev_t dev, const std::string & wanted) {
    if (!dev) return false;
    const auto reg = ggml_backend_dev_backend_reg(dev);
    const std::string rn = lower(reg && ggml_backend_reg_name(reg) ? ggml_backend_reg_name(reg) : "");
    const std::string name = lower(ggml_backend_dev_name(dev) ? ggml_backend_dev_name(dev) : "");
    const std::string desc = lower(ggml_backend_dev_description(dev) ? ggml_backend_dev_description(dev) : "");
    if (wanted == "cpu") return ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    if (wanted == "vulkan") return rn == "vk" || rn.find("vulkan") != std::string::npos ||
        name.rfind("vk", 0) == 0 || name.find("vulkan") != std::string::npos || desc.find("vulkan") != std::string::npos;
    if (wanted == "metal") return rn == "mtl" || rn.find("metal") != std::string::npos ||
        name.rfind("mtl", 0) == 0 || name.find("metal") != std::string::npos || desc.find("metal") != std::string::npos;
    return false;
}

ggml_backend_dev_t find_device(const std::string & wanted) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        if (matches(dev, wanted)) return dev;
    }
    return nullptr;
}
}

int main(int argc, char ** argv) {
    try {
        const std::string wanted = argc > 1 ? lower(argv[1]) : "vulkan";
        if (wanted != "cpu" && wanted != "vulkan" && wanted != "metal") {
            throw std::runtime_error("usage: moge-backend-smoke [cpu|vulkan|metal]");
        }

        ggml_backend_load_all();
        auto dev = find_device(wanted);
        if (!dev) throw std::runtime_error("requested backend device is not available");
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) throw std::runtime_error("backend initialization failed");

        ggml_init_params params{};
        params.mem_size = 1u << 20;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        if (!ctx) throw std::runtime_error("ggml_init failed");

        ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        ggml_tensor * c = ggml_add(ctx, a, b);
        ggml_set_input(a);
        ggml_set_input(b);
        ggml_set_output(c);

        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, c);

        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) throw std::runtime_error("backend tensor allocation failed");

        const float av[4] = {1.f, 2.f, 3.f, 4.f};
        const float bv[4] = {4.f, 3.f, 2.f, 1.f};
        ggml_backend_tensor_set(a, av, 0, sizeof(av));
        ggml_backend_tensor_set(b, bv, 0, sizeof(bv));
        const auto status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("backend graph compute failed");
        ggml_backend_synchronize(backend);

        float out[4] = {};
        ggml_backend_tensor_get(c, out, 0, sizeof(out));
        for (float x : out) {
            if (std::fabs(x - 5.f) > 1e-5f) throw std::runtime_error("backend compute returned an incorrect result");
        }

        std::printf("backend=%s\ndevice=%s\nstatus=ok\n",
            wanted.c_str(), ggml_backend_dev_description(dev) ? ggml_backend_dev_description(dev) : ggml_backend_name(backend));

        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "moge-backend-smoke: %s\n", e.what());
        return 1;
    }
}
