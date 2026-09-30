#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "$ROOT"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_*.py
python3 -m py_compile tools/convert_moge.py tools/inspect_mogg.py tools/bench_matrix.py tests/collect_results.py
