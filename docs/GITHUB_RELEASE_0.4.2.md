# GitHub release copy — moge-ggml 0.4.2

**Tag:** `v0.4.2`  
**Title:** `moge-ggml 0.4.2`

moge-ggml 0.4.2 is the first consolidated release after the CPU, Metal, and Vulkan development/tuning phase was reduced to a fixed production policy.

## Highlights

- Native MoGe inference through ggml on CPU, Apple Metal, and Vulkan.
- Small C++17 API and stable C ABI (API version 1).
- Native `.moge`/MOGG checkpoint loading plus conversion tooling.
- File, owning-memory, and borrowed-memory model loading.
- Shape-dependent execution-plan/resource caching.
- Release-focused smoke, benchmark, golden-reference, and packaging validation.
- Metal MPSGraph execution pinned to the GPU path, removing earlier ANE compatibility diagnostics.
- Vulkan experimentation controls removed from the release-facing runtime; validated production policy is fixed internally.

## Validated release targets

- **Apple M5 / Metal:** model-version-2 Q8, 1440x1799 input — 2.66 s inference, 381.7 MiB peak RSS, no ANE diagnostics.
- **AMD Radeon RX 7900 XTX / RADV NAVI31 / Vulkan:** v3 Q8 at 640x480 — golden PASS, 1155.349 ms median, 1136.141 ms best, 962.059 MiB peak RSS.
- **AMD Radeon RX 7900 XTX / RADV NAVI31 / Vulkan:** model-version-2 Q8 at 640x480 — golden PASS, 218.523 ms median, 211.057 ms best, 283.746 MiB peak RSS.

These are release-regression measurements for the stated systems, not general performance guarantees.

## Compatibility

- Project version: **0.4.2**
- Public API/ABI version: **1**
- C++ requirement: **C++17**
- Bundled ggml: **0.25.3**

Model checkpoints are not included in the source release and may have separate licensing terms.

## Credits

- **Kenji8965** — project direction, target definition, hardware validation, performance steering, release acceptance, and testing.
- **ChatGPT 5.6 Sol (OpenAI)** — primary programming and implementation assistance for the 0.4.2 development cycle.
- **ggml authors/contributors** and **stb authors/contributors** — upstream software retained under their respective licenses; see `THIRD_PARTY_NOTICES.md`.

## Assets

Attach:

- `moge-ggml-0.4.2.tar.gz`
- `moge-ggml-0.4.2.tar.gz.sha256`

Verify the checksum before publication.
