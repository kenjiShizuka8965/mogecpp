#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

version="${1:-0.4.2}"
out="${2:-$ROOT/../moge-ggml-${version}.tar.gz}"
name="moge-ggml-${version}"
[[ -f "$GGML_DEFAULT/CMakeLists.txt" ]] || { echo "missing bundled ggml source: $GGML_DEFAULT" >&2; exit 2; }

stage="$(mktemp -d "${TMPDIR:-/tmp}/moge-release.XXXXXX")"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/$name"

tar -C "$ROOT" -cf - \
  --exclude='./.git' \
  --exclude='./.deps' \
  --exclude='./build-*' \
  --exclude='./install-*' \
  --exclude='./test-results' \
  --exclude='./models' \
  --exclude='./calibration' \
  --exclude='.pytest_cache' \
  --exclude='*/.pytest_cache' \
  --exclude='__pycache__' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='./*.tar.gz' \
  . | tar -C "$stage/$name" -xf -

tar -C "$stage" -czf "$out" "$name"
echo "$out"
