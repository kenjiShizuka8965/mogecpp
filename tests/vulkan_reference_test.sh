#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
[[ $# -eq 3 ]] || { echo "usage: tests/vulkan_reference_test.sh MODEL.mogg INPUT.png|jpg REFERENCE_DIR_OR_ZIP" >&2; exit 2; }
model="$1"; image="$2"; ref_arg="$3"
require_file "$model"; require_file "$image"
build_dir="$(build_dir_for vulkan)"; cli="$build_dir/moge-cli"; require_exe "$cli"
work="$build_dir/vulkan-reference-work"; rm -rf "$work"; mkdir -p "$work/candidate"
reference="$ref_arg"
case "$ref_arg" in
  *.zip)
    require_file "$ref_arg"; mkdir -p "$work/reference"
    python3 - "$ref_arg" "$work/reference" <<'PY'
import sys,zipfile
from pathlib import Path
src=Path(sys.argv[1]); dst=Path(sys.argv[2]).resolve()
with zipfile.ZipFile(src) as z:
    for info in z.infolist():
        p=(dst/info.filename).resolve()
        if dst not in p.parents and p!=dst: raise SystemExit('unsafe path in reference zip')
    z.extractall(dst)
PY
    if [[ -f "$work/reference/depth.pfm" ]]; then reference="$work/reference"
    elif [[ -f "$work/reference/moge-output/depth.pfm" ]]; then reference="$work/reference/moge-output"
    else echo "reference ZIP does not contain depth.pfm" >&2; exit 2; fi
    ;;
  *) [[ -d "$ref_arg" ]] || { echo "missing reference directory/ZIP: $ref_arg" >&2; exit 2; } ;;
esac
for f in depth.pfm points.ply camera.json; do require_file "$reference/$f"; done

args=(--model "$model" --backend vulkan --output "$work/candidate")
[[ -n "${MOGE_REFERENCE_RESOLUTION_LEVEL:-}" ]] && args+=(--resolution-level "$MOGE_REFERENCE_RESOLUTION_LEVEL")
[[ -n "${MOGE_REFERENCE_TOKENS:-}" ]] && args+=(--tokens "$MOGE_REFERENCE_TOKENS")
[[ -n "${MOGE_REFERENCE_REFINE:-}" ]] && args+=(--refine "$MOGE_REFERENCE_REFINE")
[[ -n "${MOGE_REFERENCE_FOV_X:-}" ]] && args+=(--fov-x "$MOGE_REFERENCE_FOV_X")
mkdir -p "$RESULTS_DIR"
"$cli" "${args[@]}" "$image" 2>&1 | tee "$RESULTS_DIR/vulkan-reference-inference.txt"
cmp=("$reference" "$work/candidate" --json "$RESULTS_DIR/accuracy.json")
# The default thresholds are a conservative release gate for the validated
# cross-backend golden output. Set MOGE_REFERENCE_REPORT_ONLY=1 only when
# establishing a new trusted reference.
if [[ "${MOGE_REFERENCE_REPORT_ONLY:-0}" != "1" ]]; then
  cmp+=(--enforce
    --depth-rmse-max "${MOGE_DEPTH_RMSE_MAX:-0.03}"
    --depth-p95-rel-max "${MOGE_DEPTH_P95_REL_MAX:-0.03}"
    --points-rmse-max "${MOGE_POINTS_RMSE_MAX:-0.03}"
    --normal-p95-deg-max "${MOGE_NORMAL_P95_DEG_MAX:-2.0}"
    --mask-iou-min "${MOGE_MASK_IOU_MIN:-0.999}"
    --intrinsics-max-abs "${MOGE_INTRINSICS_MAX_ABS:-0.03}"
    --metric-scale-rel-max "${MOGE_METRIC_SCALE_REL_MAX:-0.02}")
fi
python3 "$ROOT/tests/compare_moge_output.py" "${cmp[@]}" | tee "$RESULTS_DIR/vulkan-reference-summary.txt"
cp "$RESULTS_DIR/accuracy.json" "$RESULTS_DIR/vulkan-reference-metrics.json"
