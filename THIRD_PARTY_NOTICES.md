# Third-party notices

This project redistributes or interfaces with third-party software. Copyright and license ownership remains with the respective upstream authors. Nothing in the moge-ggml license supersedes an upstream license.

The source archive keeps authoritative license text with each redistributed component wherever upstream provides it. This file is a convenience summary, not a replacement for those license texts.

## ggml 0.25.3

- Location: `third_party/ggml-0.25.3/`
- Upstream project: ggml
- License: MIT License
- Copyright: **Copyright (c) 2023-2026 The ggml authors**
- Authoritative bundled license: `third_party/ggml-0.25.3/LICENSE`

moge-ggml includes a validated ggml 0.25.3 source tree with project-specific backend changes. The upstream MIT copyright and permission notice are retained in full.

## stb_image 2.28

- Location: `third_party/stb_image.h`
- Upstream project: stb
- Primary author: Sean Barrett, with contributors listed in the header
- License: MIT License **or** public domain, at the user's option
- MIT copyright notice: **Copyright (c) 2017 Sean Barrett**
- Authoritative license text: embedded in `third_party/stb_image.h`

## stb_image_write 1.16

- Location: `third_party/stb_image_write.h`
- Upstream project: stb
- Primary author: Sean Barrett, with contributors listed in the header
- License: MIT License **or** public domain, at the user's option
- MIT copyright notice: **Copyright (c) 2017 Sean Barrett**
- Authoritative license text: embedded in `third_party/stb_image_write.h`

## Platform and system dependencies

Depending on the selected backend and host platform, builds may link against software supplied by the operating system, SDK, package manager, or graphics driver rather than redistributed in this repository. Examples include:

- Vulkan loader, headers, shader compiler and GPU driver components
- Apple Metal, MetalPerformanceShaders, MPSGraph, Accelerate, and related macOS SDK frameworks
- BLAS implementations
- OpenMP runtimes
- a separately installed ggml package when the bundled source tree is not used

These components remain subject to their own licenses and terms. They are not relicensed by moge-ggml.

## Model weights and research assets

Model checkpoints are **not** redistributed by this source release. MoGe checkpoints, research code, datasets, sample media, and other externally obtained assets may have terms different from this repository's MIT license. Users must review and comply with the licenses or usage terms attached to those assets.
