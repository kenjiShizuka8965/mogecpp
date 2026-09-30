#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

backend="${1:-cpu}"
model="${2:-}"
[[ -n "$model" ]] || { echo "usage: tests/smoke.sh cpu|metal|vulkan MODEL.mogg" >&2; exit 2; }
require_file "$model"
build_dir="$(build_dir_for "$backend")"
bench="$build_dir/moge-bench"
dump="$build_dir/moge-dump"
require_exe "$bench"
require_exe "$dump"
extra=()
if [[ "$backend" == "metal" || "$backend" == "vulkan" ]]; then extra+=(--gpu-resident-inputs); fi

label="$(basename "$model" .mogg)"
label="${label//[^A-Za-z0-9._-]/_}"
out="$RESULTS_DIR/smoke-$backend-$label"
rm -rf "$out"; mkdir -p "$out"
# Include one warmup so the same cached dense plan is executed twice in one
# process. This catches scheduler-buffer lifetime regressions that a single-pass
# smoke cannot see (notably phase-local Metal scratch reuse).
"$bench" "$model" --backend "$backend" --width 224 --height 224 --tokens 256 --warmup 1 --runs 1 "${extra[@]}" 2>&1 | tee "$out/bench.txt"
"$dump" "$model" --backend "$backend" --width 224 --height 224 --tokens 256 --refine "${MOGE_SMOKE_REFINE:-0}" --out "$out/dump" "${extra[@]}" 2>&1 | tee "$out/dump.txt"
