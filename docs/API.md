The stable public contract is API v1 in moge-ggml 0.4.2. See STABILITY.md.

# Public API

`moge-ggml` exposes a small C++17 API and a C ABI intended for bindings. Backend-specific tuning and validation environment variables are implementation details and are not part of the stable API.

## C++

```cpp
#include <moge_ggml/moge.hpp>

moge::LoadOptions load;
load.backend = moge::Backend::Auto;

moge::Model model("v3_q8.moge", load);
moge::Result result = model.infer(rgb_f32_hwc, width, height);
```

Input is one interleaved HWC RGB image of `float` values in `[0, 1]`. Reuse the same `Model` for repeated inference; the runtime caches shape-dependent plans and backend resources.

`Result` owns its output arrays:

- `depth`: `H*W` floats;
- `points`: `H*W*3` XYZ floats;
- `normal`: optional `H*W*3` floats;
- `mask`: optional `H*W` bytes;
- `intrinsics`: normalized 3x3 camera matrix;
- `metric_scale`: recovered metric scale.

### Model bytes from memory

Owning form:

```cpp
moge::Model model(model_bytes, model_size, load);
```

The constructor copies the complete MOGG image. The caller may release the original storage as soon as construction returns. On POSIX/macOS the owned copy is anonymous-mmap-backed so the runtime can retain the mapped/shared-weight path where the backend permits it.

Borrowed form:

```cpp
auto model = moge::Model::from_borrowed_memory(model_bytes, model_size, load);
```

The caller must keep the bytes alive and unchanged until the `Model` is destroyed. Borrowing avoids the model-image copy, but a backend may still copy or transform individual tensors when required by alignment or backend representation.

### Backend selection

```cpp
moge::Backend::Auto
moge::Backend::CPU
moge::Backend::Vulkan
moge::Backend::Metal
```

`Auto` selects an available accelerator. Metal-specific MPSGraph/Tensor-API choices are capability-gated internally and are intentionally not represented in the public API.

`Vulkan` uses ggml's Vulkan backend and the same dense/sparse graph implementation as the other generic ggml backends. A Vulkan build must configure ggml with `-DGGML_VULKAN=ON`; selecting `Vulkan` at runtime fails with an explicit diagnostic when no Vulkan device is registered. ggml 0.25.3 still requires a CPU backend as the terminal scheduler fallback, even when graph inputs are pinned to the GPU.

## C ABI

```c
#include <moge_ggml/moge_c.h>

moge_load_options_t load = moge_default_load_options();
load.backend = MOGE_BACKEND_AUTO;

moge_model_t *model = moge_model_load_file("v3_q8.moge", &load);
if (!model) {
    fprintf(stderr, "%s\n", moge_last_error());
    return 1;
}

moge_result_t *result = moge_model_infer(model, rgb, width, height, NULL);
if (!result) {
    fprintf(stderr, "%s\n", moge_last_error());
    moge_model_free(model);
    return 1;
}

const float *depth = moge_result_depth(result);
const float *points = moge_result_points(result);

moge_result_free(result);
moge_model_free(model);
```

The C result/model types are opaque. Returned array pointers remain valid until `moge_result_free()`.

Memory model loading is first-class:

```c
// Own a copy; caller can release data immediately after this returns.
moge_model_t *a = moge_model_load_memory(data, len, 1, &load);

// Borrow bytes; data must outlive the model.
moge_model_t *b = moge_model_load_memory(data, len, 0, &load);
```

All failing constructors/inference functions return `NULL`. `moge_last_error()` provides a thread-local diagnostic string.

## ABI/versioning

The current public API version is `1` (`moge::API_VERSION` / `MOGE_API_VERSION`). The C++ API requires C++17. The C ABI is the recommended base for language bindings that need ABI stability across C++ toolchains.

## Version and selected backend

C++:

```cpp
std::cout << moge::version_string() << "\n";       // 0.4.2
std::cout << moge::API_VERSION << "\n";            // 1
auto kind = model.backend();                         // Auto/CPU/Vulkan/Metal
std::cout << model.backend_name() << "\n";          // concrete ggml device description
```

C:

```c
printf("%s\n", moge_version_string());
printf("%d\n", MOGE_API_VERSION);
moge_backend_t kind = moge_model_backend(model);
printf("%s\n", moge_model_backend_name(model));
```


## Safety limits

The public inference entry point validates dimensions and high-risk numeric options before allocating backend plans. API v1 currently rejects dimensions above 32768 on either side, more than 64 Mi pixels, explicit token counts above 65536, `resolution_level` outside 0..9, refinement counts above 64, and non-finite/invalid positive FOV values. These are hard sanity limits, not recommended operating sizes; applications handling untrusted uploads should normally apply tighter limits before calling the library. `moge-cli` does so by default.

MOGG file and memory-buffer loads validate directory bounds, metadata/list counts, duplicate names, tensor types, shapes, alignment, extents, and shape/type-derived payload sizes before model construction proceeds.
