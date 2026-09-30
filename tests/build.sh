#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

backend="${1:-cpu}"
ggml_arg="${2:-}"
ggml_dir="$(ensure_ggml "$ggml_arg")"
build_dir="${MOGE_BUILD_DIR:-$(build_dir_for "$backend")}"

case "$backend" in
  cpu)
    opts=(-DGGML_METAL=OFF -DGGML_VULKAN=OFF)
    ;;
  metal)
    [[ "$(uname -s)" == "Darwin" ]] || { echo "Metal build requires macOS" >&2; exit 2; }
    opts=(-DGGML_METAL=ON -DGGML_VULKAN=OFF -DGGML_ACCELERATE=ON -DGGML_BLAS=ON)
    ;;
  vulkan)
    opts=(-DGGML_METAL=OFF -DGGML_VULKAN=ON)
    ;;
  *)
    echo "usage: tests/build.sh cpu|metal|vulkan [GGML_SOURCE_DIR]" >&2
    exit 2
    ;;
esac

# Apple Clang does not search Homebrew's keg-only libomp automatically.
# When available, teach CMake's FindOpenMP exactly where brew installed it.
# Set MOGE_OPENMP=0 to deliberately disable OpenMP discovery.
openmp_opts=()
if [[ "$(uname -s)" == "Darwin" && "${MOGE_OPENMP:-1}" != "0" ]] && command -v brew >/dev/null 2>&1; then
    omp_prefix="$(brew --prefix libomp 2>/dev/null || true)"
    if [[ -n "$omp_prefix" && -f "$omp_prefix/include/omp.h" && -f "$omp_prefix/lib/libomp.dylib" ]]; then
        omp_flags="-Xpreprocessor -fopenmp -isystem$omp_prefix/include"
        openmp_opts+=(
          "-DOpenMP_C_FLAGS=$omp_flags"
          "-DOpenMP_CXX_FLAGS=$omp_flags"
          "-DOpenMP_C_LIB_NAMES=omp"
          "-DOpenMP_CXX_LIB_NAMES=omp"
          "-DOpenMP_omp_LIBRARY=$omp_prefix/lib/libomp.dylib"
        )
        echo "Homebrew OpenMP: $omp_prefix"
    else
        echo "Homebrew libomp not found; continuing without explicit OpenMP hints"
    fi
fi

jobs="${MOGE_BUILD_JOBS:-}"
if [[ -z "$jobs" ]]; then
    if command -v sysctl >/dev/null 2>&1; then jobs="$(sysctl -n hw.ncpu 2>/dev/null || true)"; fi
    if [[ -z "$jobs" ]] && command -v nproc >/dev/null 2>&1; then jobs="$(nproc)"; fi
    jobs="${jobs:-4}"
fi

mkdir -p "$RESULTS_DIR"
log="$RESULTS_DIR/build-$backend.log"

# Prefer faster local build tools on fresh build directories. Existing CMake
# directories keep their configured generator to avoid generator mismatch.
generator_opts=()
if [[ ! -f "$build_dir/CMakeCache.txt" ]] && command -v ninja >/dev/null 2>&1; then
    generator_opts=(-G Ninja)
fi
launcher_opts=()
if command -v ccache >/dev/null 2>&1; then
    launcher_opts=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
fi
{
  cmake -S "$ROOT" -B "$build_dir" "${generator_opts[@]}" \
    -DGGML_SOURCE_DIR="$ggml_dir" \
    -DGGML_NATIVE=ON \
    -DMOGE_BUILD_TESTS="${MOGE_BUILD_TESTS:-ON}" \
    -DMOGE_BUILD_DEV_TOOLS="${MOGE_BUILD_DEV_TOOLS:-OFF}" \
    -DCMAKE_BUILD_TYPE=Release \
    "${launcher_opts[@]}" \
    "${opts[@]}" \
    "${openmp_opts[@]}"
  if [[ -n "${MOGE_BUILD_TARGET:-}" ]]; then
    # CMake accepts multiple names after --target. Split the environment value
    # explicitly so macOS/Bash does not pass a whitespace-separated list as one
    # literal make target. Single-target callers continue to behave identically.
    IFS=' ' read -r -a build_targets <<< "$MOGE_BUILD_TARGET"
    cmake --build "$build_dir" --target "${build_targets[@]}" --parallel "$jobs"
  else
    cmake --build "$build_dir" --parallel "$jobs"
  fi
  echo "build_dir=$build_dir"
  echo "ggml_source=$ggml_dir"
} 2>&1 | tee "$log"
