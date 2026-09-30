#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

[[ $# -ge 1 ]] || { echo "usage: tests/bench_cpu.sh MODEL.mogg [MODEL2.mogg ...]" >&2; exit 2; }
build_dir="$(build_dir_for cpu)"
bench="$build_dir/moge-bench"
require_exe "$bench"

width="${MOGE_WIDTH:-640}"; height="${MOGE_HEIGHT:-480}"
runs="${MOGE_RUNS:-5}"; warmup="${MOGE_WARMUP:-2}"; refine="${MOGE_REFINE:-3}"
threads="${MOGE_CPU_THREADS:-1,2,4,8}"
args=(--bench "$bench" --backend cpu --width "$width" --height "$height" --runs "$runs" --warmup "$warmup" --refine "$refine")
IFS=',' read -r -a ts <<< "$threads"
for t in "${ts[@]}"; do args+=(--threads "$t"); done
for model in "$@"; do require_file "$model"; args+=(--model "$model"); done
python3 "$ROOT/tools/bench_matrix.py" "${args[@]}" --output "$RESULTS_DIR/cpu-bench.json"
