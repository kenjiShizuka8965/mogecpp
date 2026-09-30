#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

[[ $# -eq 4 && "$1" == "--reference" ]] || {
    echo "usage: tests/vulkan_release_test.sh --reference MODEL.mogg INPUT.png|jpg REFERENCE_DIR_OR_ZIP" >&2
    exit 2
}
model="$2"; image="$3"; reference="$4"
require_file "$model"; require_file "$image"
if [[ ! -f "$reference" && ! -d "$reference" ]]; then
    echo "reference not found: $reference" >&2
    exit 2
fi

rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"
"$ROOT/tests/env_report.sh" "$RESULTS_DIR/environment.txt" >/dev/null

# Build the release backend and the small set of native validation/dev tools used below.
MOGE_BUILD_DEV_TOOLS=ON MOGE_BUILD_TESTS=ON "$ROOT/tests/build.sh" vulkan
build_dir="$(build_dir_for vulkan)"

"$build_dir/moge-backend-smoke" vulkan 2>&1 | tee "$RESULTS_DIR/backend-smoke.txt"
"$ROOT/tests/smoke.sh" vulkan "$model"

# Reject benchmark numbers collected while the host is obviously busy.
"$ROOT/tests/check_host_contention.sh"
MOGE_RUNS="${MOGE_RUNS:-11}" MOGE_WARMUP="${MOGE_WARMUP:-3}" \
    "$ROOT/tests/bench_vulkan.sh" "$model"

"$ROOT/tests/vulkan_reference_test.sh" "$model" "$image" "$reference"
"$ROOT/tests/profile_vulkan.sh" "$model"
"$ROOT/tests/vulkan_gpu_state.sh" "$model"

archive="${MOGE_RESULTS_ARCHIVE:-$ROOT/../moge-vulkan-test-results.tar.gz}"
python3 "$ROOT/tests/collect_results.py" --root "$ROOT" --output "$archive"
echo "release_validation=PASS"
echo "results_archive=$archive"
