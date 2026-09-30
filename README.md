# moge-ggml

Copyright © 2026 **Kenji8965**.

**Programming and implementation assistance:** ChatGPT 5.6 Sol (OpenAI)

**Human testing, validation, and release acceptance:** Kenji8965

This software is provided under the MIT License **“AS IS”**, without warranty of any kind. Use of the software and its outputs is at your own risk. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the complete terms and third-party notices.

**Release: 0.4.2 · Public API: v1 · Bundled ggml: 0.25.3**

Native MoGe inference on ggml 0.25.3 with CPU, Metal, and Vulkan backends. The release exposes a small C++17 API, a stable C ABI, `moge-cli`, and checkpoint conversion tools. Backend execution policy is fixed internally; applications do not need tuning environment variables.

This is an independent native inference implementation for the MoGe model family; see [Credits](CREDITS.md) for research attribution and upstream acknowledgements.

Release documentation: [0.4.2 release notes](docs/RELEASE_0.4.2.md) · [changelog](CHANGELOG.md) · [API](docs/API.md) · [stability](docs/STABILITY.md) · [third-party notices](THIRD_PARTY_NOTICES.md).

Project credits: [CREDITS.md](CREDITS.md).

## Requirements

- CMake 3.20 or newer and a C++17 compiler.
- CPU builds require only the normal ggml host dependencies.
- Vulkan builds additionally require Vulkan development headers/loader and `glslc`.
- Metal builds require macOS with the Metal, MetalPerformanceShaders, and MPSGraph frameworks.

Model checkpoints are not included in the source release. Convert a supported checkpoint with `moge-convert` or use an existing `.moge`/MOGG model.

## Build

The release includes the validated ggml source tree at `third_party/ggml-0.25.3`.

### CPU

```bash
cmake -S . -B build \
  -DGGML_SOURCE_DIR=third_party/ggml-0.25.3 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Vulkan

A Vulkan build requires the Vulkan loader/headers and `glslc`.

```bash
cmake -S . -B build-vulkan \
  -DGGML_SOURCE_DIR=third_party/ggml-0.25.3 \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j
```

### Metal

```bash
cmake -S . -B build-metal \
  -DGGML_SOURCE_DIR=third_party/ggml-0.25.3 \
  -DGGML_METAL=ON \
  -DGGML_ACCELERATE=ON \
  -DGGML_BLAS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-metal -j
```

The Metal release path pins its MPSGraph helpers to the Metal GPU and disables
MPSGraph cross-hardware placement, avoiding spurious Apple Neural Engine (ANE)
compatibility diagnostics during inference.

Project options:

- `MOGE_BUILD_CLI=ON|OFF` — build `moge-cli` (default ON)
- `MOGE_BUILD_CONVERT=ON|OFF` — build `moge-convert` (default ON)
- `MOGE_BUILD_DEV_TOOLS=ON|OFF` — build benchmark/calibration/quantization tools (default OFF)
- `MOGE_BUILD_TESTS=ON|OFF` — build native validation utilities (default OFF)
- `MOGE_HOMEBREW_OPENMP=ON|OFF` — detect Homebrew libomp for AppleClang (default ON)

A system-installed ggml can be used by omitting `GGML_SOURCE_DIR` and providing a CMake package exporting `ggml::ggml` or `ggml`.

## CLI

```bash
build/moge-cli \
  --model v3_q8.moge \
  --backend auto \
  --output output \
  photo.jpg
```

Backends are `auto`, `cpu`, `metal`, and `vulkan`. Normal inference controls include `--resolution-level`, `--tokens`, `--refine`, and `--fov-x`.

## C++ API

```cpp
#include <moge_ggml/moge.hpp>

moge::LoadOptions load;
load.backend = moge::Backend::Auto;

moge::Model model("v3_q8.moge", load);
moge::Result result = model.infer(rgb_f32_hwc, width, height);
```

Models can also be loaded from memory. The owning constructor copies the MOGG image; `Model::from_borrowed_memory()` avoids that copy and requires the caller to keep the bytes alive for the lifetime of the model.

A `Model` may be reused sequentially and caches shape-dependent resources. Concurrent `infer()` calls on the same instance are unsupported; serialize access or use one model instance per concurrent inference stream.

## C API

```c
#include <moge_ggml/moge_c.h>

moge_load_options_t load = moge_default_load_options();
load.backend = MOGE_BACKEND_AUTO;

moge_model_t *model = moge_model_load_file("v3_q8.moge", &load);
moge_result_t *result = moge_model_infer(model, rgb, width, height, NULL);
if (!result) fprintf(stderr, "%s\n", moge_last_error());

moge_result_free(result);
moge_model_free(model);
```

`moge_model_load_memory()` supports copied and borrowed bytes. The public API version is `1` (`moge::API_VERSION` / `MOGE_API_VERSION`). See `docs/API.md` and `docs/STABILITY.md`.

## Model conversion

Build `moge-convert`, then convert a supported PyTorch checkpoint:

```bash
build/moge-convert model.pt --quant q8 --output v3_q8.moge
```

The Python converter is retained for development/reference workflows. See `docs/CONVERSION.md`.

## Diagnostics

Production execution policy is fixed in code. The supported runtime diagnostics are:

- `MOGE_DENSE_TRACE=1` — dense execution timing and scratch diagnostics
- `MOGE_SPARSE_TRACE=1` — sparse execution timing and scratch diagnostics
- `MOGE_OP_PROFILE=1` — per-operator profiling
- `MOGE_DEVICE_MEMORY_TRACE=1` — backend memory samples
- `MOGE_POST_THREADS=<n>` — override post-processing thread count
- `MOGE_SPARSE_TOPOLOGY_THREADS=<n>` — override sparse topology host thread count

These controls observe or schedule supported work; they do not select alternate inference algorithms.

## Install / CMake consumer

```bash
cmake --install build --prefix /your/prefix
```

When configured against an installed ggml CMake package, moge-ggml installs an exported CMake package:

```cmake
find_package(moge-ggml CONFIG REQUIRED)
target_link_libraries(your_app PRIVATE moge::moge-ggml)
```

## Release status

0.4.2 has been release-validated on Apple M5 / Metal and AMD Radeon RX 7900 XTX / RADV Vulkan, in addition to the CPU build/test surface. Exact validation measurements and scope are recorded in `docs/RELEASE_0.4.2.md`; they are hardware-specific regression data rather than general performance guarantees.

## Credits

- **Kenji8965** — project direction, target definition, hardware validation, performance steering, release acceptance, and testing.
- **ChatGPT 5.6 Sol (OpenAI)** — primary programming and implementation assistance for the 0.4.2 development cycle.

See [`CREDITS.md`](CREDITS.md) for the complete attribution statement.

## License and third-party attribution

moge-ggml is released under the MIT License: **Copyright (c) 2026 Kenji8965**. See [`LICENSE`](LICENSE).

Bundled dependencies retain their own copyrights and licenses. In particular, the vendored ggml tree remains under its upstream MIT license and the stb headers retain their embedded MIT/public-domain terms. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for details. Model checkpoints are not included and may be governed by separate terms.
