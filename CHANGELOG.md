# Changelog

All notable changes to moge-ggml are documented here.

The project uses semantic versioning for the library release. The public C/C++ API has a separate API version; moge-ggml 0.4.2 uses API version 1.

## 0.4.2 - 2026-09-30

First release-quality CPU, Metal, and Vulkan package after backend consolidation and hardware validation.

### Added

- Stable C++17 API and C ABI (API version 1).
- File, copied-memory, and borrowed-memory MOGG model loading.
- CPU, Metal, and Vulkan backend selection behind one inference API.
- `moge-cli` and native `moge-convert` tools.
- Installable CMake target `moge::moge-ggml` when building against an installed ggml package.
- Release validation scripts for CPU/Metal/Vulkan, including golden-output comparison and Vulkan profiling/GPU-state capture.
- Defensive MOGG validation for bounds, counts, duplicate names, tensor shapes/types, alignment, and payload sizes.

### Performance and backend work

- Consolidated the validated Metal execution path, including MPSGraph helpers pinned to Metal GPU execution without cross-hardware ANE placement probes.
- Consolidated the validated Vulkan execution policy for AMD Navi31 / RX 7900 XTX on RADV.
- Added cached dense/sparse execution plans, segmented sparse execution, bounded sparse convolution, decoder streaming where beneficial, and reusable backend memory arenas.
- Removed release-time algorithm/tuning selectors used during development; production algorithm selection is fixed internally.

### Release cleanup

- Removed historical experiment/sweep scripts and handoff artifacts from the release tree.
- Reduced the retained test surface to release-contract, API, security, conversion, build, smoke, benchmark, and backend regression coverage.
- Fixed release packaging so build products, test results, Python caches, and temporary artifacts are excluded.

### Validated release targets

- Apple M5 / Metal: model-version-2 inference validated without ANE compatibility diagnostics; 1440x1799 v2 Q8 inference measured at approximately 2.66 s in the final validation run.
- AMD Radeon RX 7900 XTX / Vulkan (RADV NAVI31): 640x480 v3 Q8 golden-reference parity validated; final 11-run median measured at 1155.349 ms.
- AMD Radeon RX 7900 XTX / Vulkan (RADV NAVI31): model-version-2 Q8 golden-reference parity validated; 640x480 median measured at 218.523 ms.

Performance numbers describe the tested systems and workloads only; they are not portability guarantees.
