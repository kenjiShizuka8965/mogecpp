# Release validation

The release test directory contains only supported build, smoke, benchmark, parity, profiling, packaging, and regression helpers. Historical tuning harnesses are intentionally not shipped.

## Python/static regression suite

```bash
tests/run_python_tests.sh
```

The suite covers stable numerical helpers, conversion/container behavior, API/security contracts, and release-policy invariants.

## Build helpers

```bash
tests/build.sh cpu
tests/build.sh metal
tests/build.sh vulkan
```

By default these use the bundled `third_party/ggml-0.25.3` tree. An alternate ggml source tree may be passed as the second argument.

Developer tools can be enabled for a build with:

```bash
MOGE_BUILD_DEV_TOOLS=ON tests/build.sh cpu
```

Build logs are written under `test-results/`.

## Smoke tests

```bash
tests/smoke.sh cpu MODEL.moge
tests/smoke.sh metal MODEL.moge
tests/smoke.sh vulkan MODEL.moge
```

GPU smoke tests use resident inputs. A warmup plus measured execution exercises cached plan reuse.

## Benchmarks

The benchmark helpers require `moge-bench`, so build with `MOGE_BUILD_DEV_TOOLS=ON` first.

```bash
tests/bench_cpu.sh MODEL.moge
tests/bench_metal.sh MODEL.moge
tests/bench_vulkan.sh MODEL.moge
```

Useful harness controls are `MOGE_WIDTH`, `MOGE_HEIGHT`, `MOGE_RUNS`, `MOGE_WARMUP`, and `MOGE_REFINE`. CPU benchmarking also accepts `MOGE_CPU_THREADS`.

## Reference parity

For a pre-generated Vulkan golden output:

```bash
tests/vulkan_reference_test.sh MODEL.moge INPUT.jpg REFERENCE_DIR_OR_ZIP
```

The default reference comparison enforces conservative geometry thresholds.

## Vulkan release validation

On the validated RX 7900 XTX target, run:

```bash
tests/vulkan_release_test.sh \
  --reference ../v3_q8.moge ../emiru.jpg ../moge-output.zip
```

The command performs environment capture, a Vulkan release build, backend/model smoke checks, host-contention detection, repeated benchmarking, golden-reference enforcement, operator/memory profiling, and read-only GPU-state sampling. It does not change clocks or select alternate algorithms.

Results are sanitized and packaged as `../moge-vulkan-test-results.tar.gz` by default.

## Packaging

Create a source release with:

```bash
tests/package_release.sh 0.4.2
```

This emits `../moge-ggml-0.4.2.tar.gz` by default and excludes builds, test output, caches, local models, and calibration data.
