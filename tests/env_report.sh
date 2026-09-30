#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

out="${1:-$RESULTS_DIR/environment.txt}"
mkdir -p "$(dirname "$out")"
{
  echo "date_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "kernel=$(uname -srm)"
  echo
  echo "== tools =="
  for cmd in cmake python3 clang clang++ cc c++ ninja make; do
    if command -v "$cmd" >/dev/null 2>&1; then
      echo "-- $cmd --"
      "$cmd" --version 2>&1 | head -n 4 || true
    fi
  done
  echo
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "== macOS =="
    sw_vers 2>&1 || true
    xcodebuild -version 2>&1 || true
    xcrun -find metal 2>&1 || true
    for key in hw.model hw.machine hw.ncpu hw.physicalcpu hw.logicalcpu hw.memsize \
               hw.perflevel0.physicalcpu hw.perflevel0.logicalcpu \
               hw.perflevel1.physicalcpu hw.perflevel1.logicalcpu; do
      printf '%s=' "$key"; sysctl -n "$key" 2>/dev/null || echo unavailable
    done
    # Hardware/device details are useful, but never include serial/UUID/provisioning identifiers.
    system_profiler SPHardwareDataType SPDisplaysDataType 2>&1 | \
      sed -E '/Serial Number \(system\):/d; /Hardware UUID:/d; /Provisioning UDID:/d; /Activation Lock Status:/d' || true
  else
    echo "== CPU =="
    command -v lscpu >/dev/null 2>&1 && lscpu || true
    echo
    echo "== Vulkan =="
    command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo --summary || true
  fi
} > "$out"
echo "$out"
