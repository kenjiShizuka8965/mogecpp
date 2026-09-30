#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

model="${1:-}"
[[ -n "$model" && -f "$model" ]] || { echo "usage: $0 MODEL" >&2; exit 2; }
build_dir="$(build_dir_for vulkan)"
bench="$build_dir/moge-bench"
[[ -x "$bench" ]] || { echo "required executable not found: $bench" >&2; exit 2; }

out="$RESULTS_DIR/vulkan-gpu-state"
mkdir -p "$out"

gpu_dev=""
for d in /sys/class/drm/card*/device; do
  [[ -r "$d/vendor" ]] || continue
  [[ "$(cat "$d/vendor" 2>/dev/null || true)" == "0x1002" ]] || continue
  gpu_dev="$d"
  [[ "$(cat "$d/boot_vga" 2>/dev/null || true)" == "1" ]] && break
done

if [[ -z "$gpu_dev" ]]; then
  echo "No AMD DRM device sysfs path found" > "$out/README.txt"
  exit 0
fi

printf 'gpu_device=%s\n' "$gpu_dev" > "$out/device.txt"
for f in power_dpm_force_performance_level pp_power_profile_mode pp_dpm_sclk pp_dpm_mclk gpu_busy_percent mem_busy_percent; do
  if [[ -r "$gpu_dev/$f" ]]; then
    { echo "### $f"; cat "$gpu_dev/$f"; echo; } >> "$out/device.txt" 2>/dev/null || true
  fi
done

hwmon=""
for h in "$gpu_dev"/hwmon/hwmon*; do [[ -d "$h" ]] && hwmon="$h" && break; done
[[ -n "$hwmon" ]] && printf 'hwmon=%s\n' "$hwmon" >> "$out/device.txt"

{
  echo "### /dev/dri users before benchmark"
  command -v fuser >/dev/null 2>&1 && fuser -v /dev/dri/renderD* /dev/dri/card* 2>&1 || true
  echo
  echo "### likely GPU processes"
  ps -eo pid,ppid,ni,psr,pcpu,pmem,etimes,comm,args --sort=-pcpu | head -80 || true
} > "$out/processes-before.txt"

read1() { [[ -r "$1" ]] && tr '\n' ' ' < "$1" 2>/dev/null || true; }
star_clock() {
  local f="$1"
  [[ -r "$f" ]] || return 0
  awk '/\*/ { for (i=1;i<=NF;i++) if ($i ~ /[0-9]+Mhz/) {gsub(/Mhz/,"",$i); print $i; exit} }' "$f" 2>/dev/null || true
}
hwval() { local n="$1"; [[ -n "$hwmon" && -r "$hwmon/$n" ]] && cat "$hwmon/$n" 2>/dev/null || true; }

sample_loop() {
  local file="$1"
  printf 'epoch_ms\tgpu_busy_pct\tmem_busy_pct\tsclk_mhz\tmclk_mhz\tfreq1_hz\tfreq2_hz\tpower_uw\ttemp_mC\tperf_level\n' > "$file"
  while :; do
    now="$(date +%s%3N 2>/dev/null || date +%s000)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$now" \
      "$(read1 "$gpu_dev/gpu_busy_percent")" \
      "$(read1 "$gpu_dev/mem_busy_percent")" \
      "$(star_clock "$gpu_dev/pp_dpm_sclk")" \
      "$(star_clock "$gpu_dev/pp_dpm_mclk")" \
      "$(hwval freq1_input)" \
      "$(hwval freq2_input)" \
      "$(hwval power1_average)" \
      "$(hwval temp1_input)" \
      "$(read1 "$gpu_dev/power_dpm_force_performance_level")" >> "$file"
    sleep 0.2
  done
}

run_case() {
  local log="$out/auto-bench.txt"
  local samples="$out/auto-samples.tsv"
  sample_loop "$samples" &
  sampler=$!
  set +e
  "$bench" "$model" --backend vulkan --width 640 --height 480 --refine 3 --warmup 2 --runs 7 --gpu-resident-inputs > "$log" 2>&1
  rc=$?
  set -e
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  echo "$rc" > "$out/auto-rc.txt"
  return "$rc"
}

run_case || true

{
  echo "### /dev/dri users after benchmark"
  command -v fuser >/dev/null 2>&1 && fuser -v /dev/dri/renderD* /dev/dri/card* 2>&1 || true
} > "$out/processes-after.txt"

python3 - "$out" <<'PY'
import csv, json, pathlib, re, statistics, sys
root = pathlib.Path(sys.argv[1])
report = {"schema": 1, "cases": {}}
for p in sorted(root.glob("*-samples.tsv")):
    name = p.name[:-12]
    rows = list(csv.DictReader(p.open(), delimiter='\t'))
    case = {"samples": len(rows)}
    for key, scale in [("gpu_busy_pct",1),("mem_busy_pct",1),("sclk_mhz",1),("mclk_mhz",1),("freq1_hz",1e-6),("freq2_hz",1e-6),("power_uw",1e-6),("temp_mC",1e-3)]:
        vals=[]
        for r in rows:
            try:
                s=(r.get(key) or '').strip()
                if s: vals.append(float(s)*scale)
            except ValueError: pass
        if vals:
            vals.sort()
            case[key] = {"min": vals[0], "median": statistics.median(vals), "max": vals[-1], "mean": statistics.fmean(vals)}
    log = root / f"{name}-bench.txt"
    if log.exists():
        text=log.read_text(errors='ignore')
        m=re.search(r"median(?:_ms)?\s*[:=]\s*([0-9.]+)", text, re.I)
        if m: case["median_ms"] = float(m.group(1))
        case["bench_tail"] = text[-4000:]
    rc = root / f"{name}-rc.txt"
    if rc.exists():
        try: case["rc"] = int(rc.read_text().strip())
        except: pass
    report["cases"][name]=case
(root/"summary.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY
