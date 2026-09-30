#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
out="$RESULTS_DIR/host-contention.txt"
threshold="${MOGE_HOST_CONTENTION_CPU_PCT:-200}"
mkdir -p "$RESULTS_DIR"
{
  echo "threshold_cpu_pct=$threshold"
  echo "timestamp=$(date -Is 2>/dev/null || date)"
  ps -eo pid,ppid,ni,psr,pcpu,pmem,comm,args --sort=-pcpu | head -25
} > "$out"
python3 - "$out" "$threshold" $$ "$PPID" <<'PY'
import sys,re
p=sys.argv[1]; threshold=float(sys.argv[2]); selfpid=int(sys.argv[3]); parent=int(sys.argv[4])
lines=open(p,errors='ignore').read().splitlines()
bad=[]
for line in lines:
    m=re.match(r'\s*(\d+)\s+(\d+)\s+[-\d]+\s+\d+\s+([\d.]+)\s+', line)
    if not m: continue
    pid,ppid,cpu=int(m.group(1)),int(m.group(2)),float(m.group(3))
    if pid in (selfpid,parent) or ppid in (selfpid,parent): continue
    if cpu >= threshold: bad.append((cpu,line.strip()))
if bad:
    print('HOST_CONTENTION_DETECTED')
    for cpu,line in bad: print(line)
    raise SystemExit(3)
print('HOST_CONTENTION_OK')
PY
