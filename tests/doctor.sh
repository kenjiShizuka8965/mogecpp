#!/usr/bin/env bash
set -u
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

status=0
echo "== MoGe-ggml preflight =="
echo "project: $ROOT"
echo "python:  $(command -v python3 2>/dev/null || echo MISSING)"
python3 --version 2>&1 || status=1

echo
echo "== Python modules =="
python3 - <<'PY' || status=1
import sys
print("executable:", sys.executable)
for name in ("numpy", "torch"):
    try:
        m = __import__(name)
        print(f"{name}: {getattr(m, '__version__', 'present')}")
    except Exception as e:
        print(f"{name}: MISSING ({e})")
if 'torch' in sys.modules:
    import torch
    print("torch.cuda.available:", torch.cuda.is_available())
    print("torch.mps.available:", bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()))
PY

echo
if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "== Homebrew OpenMP =="
    if command -v brew >/dev/null 2>&1; then
        omp_prefix="$(brew --prefix libomp 2>/dev/null || true)"
        if [[ -n "$omp_prefix" && -f "$omp_prefix/include/omp.h" && -f "$omp_prefix/lib/libomp.dylib" ]]; then
            echo "libomp: $omp_prefix"
            echo "status: ready; tests/build.sh will pass explicit Apple-Clang OpenMP hints to CMake"
        else
            echo "libomp: not found via brew --prefix libomp"
        fi
    else
        echo "brew: not installed/on PATH"
    fi
fi

echo
echo "== Registered upstream checkpoints =="
found=0
for f in "$ROOT"/checkpoints/moge2.pt "$ROOT"/checkpoints/moge2.pth "$ROOT"/checkpoints/moge2.safetensors \
         "$ROOT"/checkpoints/moge3.pt "$ROOT"/checkpoints/moge3.pth "$ROOT"/checkpoints/moge3.safetensors; do
    if [[ -e "$f" ]]; then
        found=1
        ls -lh "$f"
    fi
done
if [[ "$found" == 0 ]]; then
    echo "none"
    echo "none (pass checkpoints/models explicitly to conversion and validation commands)"
fi

echo
echo "== Converted models =="
shopt -s nullglob
mogg=("$ROOT"/models/*.mogg)
if (( ${#mogg[@]} )); then
    ls -lh "${mogg[@]}"
else
    echo "none"
fi

echo
echo "== Native builds =="
for backend in cpu metal vulkan; do
    d="$(build_dir_for "$backend")"
    if [[ -x "$d/moge-bench" ]]; then
        echo "$backend: ready ($d)"
    else
        echo "$backend: not built"
    fi
done

echo
if [[ "$status" != 0 ]]; then
    echo "Preflight found a missing Python dependency. Activate the environment that contains PyTorch, then rerun tests/doctor.sh." >&2
fi
exit "$status"
