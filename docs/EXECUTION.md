# Execution architecture

moge-ggml uses one public inference interface across CPU, Metal, and Vulkan. Backend-specific scheduling and memory policy are internal implementation details; the release does not expose algorithm-selection environment variables.

## Dense path

The dense model is partitioned at natural DINO/neck/decoder boundaries so GPU scheduler scratch can be released between phases while persistent state remains externally owned. Shape-dependent graph metadata is cached and reused across sequential calls.

The decoder uses streamed high-resolution phases to bound convolution workspace instead of materializing every full-resolution intermediate. The production tile geometry is fixed in code. On Vulkan, cross-phase dense state uses the validated reduced-precision representation; other backends retain their supported native state path.

Cached phase graphs require careful scheduler lifetime handling. Scheduler-owned backend bindings are detached before a phase-local arena is destroyed or reset; views retain topology metadata and are rebound on the next allocation. Persistent model/state tensors are allocated separately and are not detached.

## Sparse refiner

The sparse U-Net keeps topology and cross-phase feature state persistent while executing natural down/up phases with short-lived schedulers. Level-0 neighbors use direct image/factor-coordinate indexing; higher levels use the compact coordinate map.

Sparse convolution bounds gather memory by output-point tiling. Vulkan uses fixed release budgets by level and F32 gathered operands. Outputs are assembled with in-place SET operations and bias is applied after tile assembly. The small-channel Vulkan GET_ROWS path uses the validated 64-thread scalar-channel kernel.

Pooling gathers one child slot at a time and accumulates into the output rather than constructing an eight-child temporary. Topology and reusable sparse state are cached across compatible sequential inferences.

On supported Apple systems, the sparse Metal executor is selected automatically when available. It is an implementation detail, not a separate public backend.

## Diagnostics

`MOGE_DENSE_TRACE=1` and `MOGE_SPARSE_TRACE=1` report phase timing and scratch information. `MOGE_OP_PROFILE=1` and `MOGE_DEVICE_MEMORY_TRACE=1` are used by the release validation harness for operator and memory reporting.

These diagnostics do not alter the mathematical execution path.

## Concurrency

A `Model` instance owns mutable scheduler state, caches, and reusable buffers. Sequential reuse is supported and expected. Concurrent `infer()` calls on the same instance are unsupported; callers that need parallel inference should serialize access or create one model instance per concurrent stream.
