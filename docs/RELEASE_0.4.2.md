# moge-ggml 0.4.2 release notes

moge-ggml 0.4.2 is the first consolidated release after the CPU, Metal, and Vulkan implementation was consolidated from the development/tuning phase into a fixed production policy.

## Highlights

- Small C++17 API plus a stable C ABI (API version 1).
- Native inference on CPU, Apple Metal, and ggml Vulkan backends.
- Native checkpoint conversion to the MOGG container.
- File, owning-memory, and borrowed-memory model loading.
- Shape-dependent plan/resource caching for repeated inference.
- Release-oriented build, smoke, benchmark, golden-reference, and packaging checks.

## Metal

The release Metal path uses the validated GPU policy and explicitly prevents MPSGraph from running the cross-hardware placement pass that probes Apple Neural Engine compatibility. This removes the repeated `Incompatible element type for ANE` diagnostics seen during earlier development builds while retaining the validated Metal execution behavior.

Final Apple M5 validation with a model-version-2 Q8 checkpoint and a 1440x1799 input completed in 2.66 s inference time, 3.09 s total, with 381.7 MiB peak RSS and no ANE diagnostics.

## Vulkan

The release no longer exposes the development sweep matrix used to tune the backend. The accepted Vulkan policy is fixed in code.

Final AMD Radeon RX 7900 XTX / RADV NAVI31 validation at 640x480:

| Workload | Golden status | Median | Best | Peak RSS |
| --- | --- | ---: | ---: | ---: |
| v3 Q8 | PASS | 1155.349 ms | 1136.141 ms | 962.059 MiB |
| model-version-2 Q8 | PASS | 218.523 ms | 211.057 ms | 283.746 MiB |

The v3 final validation had mask agreement/IoU of 1.0/1.0. The model-version-2 run also had mask agreement/IoU of 1.0/1.0, depth RMSE 0.00390, depth mean relative error 0.163%, and normal mean angular error 0.299 degrees against its reference.

These measurements are release-regression data for the tested machine/driver, not general performance promises.

## Compatibility

- Project version: 0.4.2
- Public API/ABI version: 1
- C++ requirement: C++17
- Bundled ggml source: 0.25.3
- Model container versioning is independent of the project/API version.

## Build notes

The source release bundles the validated ggml 0.25.3 tree. Vulkan builds require the Vulkan development files and `glslc`. Metal builds require the Apple Metal/MPS/MPSGraph frameworks provided by the macOS SDK.

See the top-level README for build commands and `docs/API.md` / `docs/STABILITY.md` for the public API contract.
