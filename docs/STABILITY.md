# API and release stability

moge-ggml 0.4.2 uses public API/ABI version 1.

## Stable API v1

The following are public and intended to remain source-compatible throughout API version 1:

- `include/moge_ggml/moge.hpp`
- `include/moge_ggml/moge_c.h`
- `moge::Backend::{Auto,CPU,Vulkan,Metal}` and matching C enum values
- file, owning-memory, and borrowed-memory model loading
- batch-1 HWC RGB float32 inference
- `moge::Result` geometry outputs and C result accessors
- selected-backend name/class accessors
- CMake target `moge::moge-ggml`
- documented `moge-cli` behavior

`moge::API_VERSION` and `MOGE_API_VERSION` are `1`. An incompatible public API/ABI change requires incrementing that value. Project semantic versioning is separate and is reported by `moge::version_string()`, `moge_version_string()`, and `moge-cli --version`.

## Backend policy

`Backend::Auto` chooses the best supported backend in the current build. Metal and Vulkan may use backend-specific kernels and segmented execution internally, but those implementation details do not create additional public backends.

Production algorithm selection is fixed by the release. Diagnostic/profiling environment variables are not API and may evolve without an API-version bump, but they must not change the supported mathematical inference policy.

## Thread safety

A model may be reused sequentially. Concurrent inference on one `Model` instance is unsupported because the implementation owns mutable scheduler state, plan caches, and reusable buffers. Use external serialization or separate model instances for concurrent streams.

## Developer surface

Everything under `tests/`, benchmark/calibration utilities enabled by `MOGE_BUILD_DEV_TOOLS`, profiling output formats, and private backend implementation details are developer interfaces and may change without an API-version bump.

## Model format

The `.moge`/MOGG container carries its own model-format version, reported by `Model::version()` / `moge_model_version()`. Model-format versioning is separate from API version 1 and project version 0.4.2.
