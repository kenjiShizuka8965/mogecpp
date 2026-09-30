#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${MOGE_RESULTS_DIR:-$ROOT/test-results}"
GGML_VERSION="0.25.3"
GGML_DEFAULT="$ROOT/third_party/ggml-$GGML_VERSION"

mkdir -p "$RESULTS_DIR"

ensure_ggml() {
    local requested="${1:-}"
    local source="${requested:-$GGML_DEFAULT}"
    [[ -f "$source/CMakeLists.txt" ]] || {
        echo "ggml source not found: $source" >&2
        echo "The release tarball should contain third_party/ggml-$GGML_VERSION; alternatively pass a ggml source directory explicitly." >&2
        return 2
    }
    (cd "$source" && pwd)
}

build_dir_for() {
    printf '%s/build-%s\n' "$ROOT" "$1"
}

require_file() {
    [[ -f "$1" ]] || { echo "required file not found: $1" >&2; exit 2; }
}

require_exe() {
    [[ -x "$1" ]] || { echo "required executable not found: $1" >&2; exit 2; }
}
