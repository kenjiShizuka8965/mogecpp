#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
[[ $# -eq 1 ]] || { echo "usage: tests/profile_vulkan.sh MODEL.mogg" >&2; exit 2; }
model="$1"; require_file "$model"
build_dir="$(build_dir_for vulkan)"; bench="$build_dir/moge-bench"; require_exe "$bench"
mkdir -p "$RESULTS_DIR"
log="$RESULTS_DIR/vulkan-profile.log"
width="${MOGE_PROFILE_WIDTH:-640}"; height="${MOGE_PROFILE_HEIGHT:-480}"; refine="${MOGE_PROFILE_REFINE:-3}"
MOGE_OP_PROFILE=1 MOGE_DEVICE_MEMORY_TRACE=1 \
  "$bench" "$model" --backend vulkan --width "$width" --height "$height" --runs 1 --warmup 0 --refine "$refine" --gpu-resident-inputs \
  2>&1 | tee "$log"
python3 "$ROOT/tests/parse_vulkan_profile.py" "$log" --output "$RESULTS_DIR/vulkan-profile.json" > "$RESULTS_DIR/vulkan-profile-summary.txt"
cat "$RESULTS_DIR/vulkan-profile-summary.txt"
