#!/usr/bin/env python3
"""Run moge-bench across model/backend combinations and persist comparable results.

The native benchmark intentionally emits simple key=value lines. This wrapper keeps
all individual samples plus backend diagnostics. It also emits case-level progress
and periodic heartbeats so slow single-thread CPU measurements never look hung.
"""

from __future__ import annotations

import argparse
import json
import os
import selectors
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


NUMERIC_KEYS = {
    "model_version": int,
    "runs": int,
    "mean_ms": float,
    "median_ms": float,
    "p90_ms": float,
    "best_ms": float,
    "stddev_ms": float,
    "mean_mpix_s": float,
    "depth_checksum": float,
}


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: str


class BenchmarkFailure(RuntimeError):
    def __init__(self, message: str, *, stdout: str = "", stderr: str = "",
                 elapsed: float = 0.0, returncode: int | None = None) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr
        self.elapsed = elapsed
        self.returncode = returncode


def parse_model(value: str) -> ModelSpec:
    if "=" in value:
        label, path = value.split("=", 1)
        if not label or not path:
            raise argparse.ArgumentTypeError("--model must be LABEL=PATH or PATH")
        return ModelSpec(label, path)
    path = Path(value)
    return ModelSpec(path.stem, value)


def parse_kv(text: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for raw in text.splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key, value = key.strip(), value.strip()
        if key == "samples_ms":
            out[key] = [float(x) for x in value.split(",") if x]
        elif key in NUMERIC_KEYS:
            out[key] = NUMERIC_KEYS[key](value)
        else:
            out[key] = value
    return out


def memory_trace(log: str) -> dict[str, object]:
    dense_scratch = [float(x) for x in re.findall(r"moge_dense_phase: allocated [^\n]* scratch=([0-9.]+) MiB", log)]
    dense_scratch += [float(x) for x in re.findall(r"moge_dense_phase: (?:monolithic )?scratch=([0-9.]+) MiB", log)]
    sparse_scratch = [float(x) for x in re.findall(r"moge_sparse_phase: allocated [^\n]* scratch=([0-9.]+) MiB", log)]
    dense_persistent = [float(x) for x in re.findall(r"moge_dense_phase: segmented persistent=([0-9.]+) MiB", log)]
    dense_persistent_total = [float(x) for x in re.findall(r"persistent_total=([0-9.]+) MiB", log)]
    resident_model = [float(x) for x in re.findall(r"resident_model=([0-9.]+) MiB", log)]
    sparse_persistent = [float(x) for x in re.findall(r"moge_sparse_phase: persistent=([0-9.]+) MiB", log)]
    out: dict[str, object] = {}
    if dense_scratch: out["dense_peak_scratch_mib"] = max(dense_scratch)
    if sparse_scratch: out["sparse_peak_scratch_mib"] = max(sparse_scratch)
    if dense_persistent: out["dense_persistent_mib"] = max(dense_persistent)
    if dense_persistent_total: out["dense_persistent_total_mib"] = max(dense_persistent_total)
    if resident_model: out["resident_model_mib"] = max(resident_model)
    if dense_scratch and dense_persistent_total and resident_model:
        out["estimated_dense_working_set_mib"] = max(dense_scratch) + max(dense_persistent_total) + max(resident_model)
    if sparse_persistent: out["sparse_peak_persistent_mib"] = max(sparse_persistent)
    return out


def metal_tensor_status(log: str) -> str:
    lower = log.lower()
    if "ggml-tensor.metallib not found" in lower or "disabling the tensor api" in lower:
        return "disabled"
    if "has tensor" in lower:
        for line in lower.splitlines():
            if "has tensor" in line:
                if "true" in line:
                    return "enabled"
                if "false" in line:
                    return "disabled"
    if "ggml-tensor.metallib" in lower and "loaded" in lower:
        return "enabled"
    return "unknown"


def command_for(args: argparse.Namespace, model: ModelSpec, backend: str, threads: int) -> list[str]:
    cmd = [args.bench, model.path, "--backend", backend,
           "--width", str(args.width), "--height", str(args.height),
           "--refine", str(args.refine), "--warmup", str(args.warmup),
           "--runs", str(args.runs)]
    if args.tokens:
        cmd += ["--tokens", str(args.tokens)]
    if threads:
        cmd += ["--threads", str(threads)]
    if backend in {"metal", "vulkan"} and not args.allow_cpu_fallback:
        # ggml 0.25.3 requires a CPU backend at the end of the scheduler.
        # This flag therefore pins user graph inputs to the requested GPU; it
        # does not remove the mandatory CPU sentinel from the scheduler.
        cmd += ["--gpu-resident-inputs"]
    return cmd


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run_one(
    cmd: list[str],
    *,
    progress_label: str = "benchmark",
    progress_seconds: float = 10.0,
    stream_child: bool = False,
) -> tuple[dict[str, object], str, str, float]:
    """Run one benchmark while retaining output and reporting liveness.

    stdout/stderr are drained concurrently so a verbose ggml backend cannot block the
    child on a full pipe. By default only a heartbeat is printed; --stream-child also
    mirrors child output live with stdout/stderr prefixes.
    """
    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    assert proc.stdout is not None and proc.stderr is not None

    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ, ("stdout", proc.stdout))
    sel.register(proc.stderr, selectors.EVENT_READ, ("stderr", proc.stderr))
    captured: dict[str, list[str]] = {"stdout": [], "stderr": []}
    next_heartbeat = start + max(progress_seconds, 0.1)

    try:
        while sel.get_map():
            now = time.monotonic()
            timeout = max(0.0, next_heartbeat - now) if progress_seconds > 0 else None
            events = sel.select(timeout)
            if not events:
                now = time.monotonic()
                if progress_seconds > 0 and now >= next_heartbeat:
                    _progress(f"    ... {progress_label}: still running ({now - start:.1f}s elapsed)")
                    next_heartbeat = now + progress_seconds
                continue

            for key, _ in events:
                kind, pipe = key.data
                line = pipe.readline()
                if line:
                    captured[kind].append(line)
                    if stream_child:
                        prefix = "out" if kind == "stdout" else "err"
                        print(f"    [{prefix}] {line}", end="", file=sys.stderr, flush=True)
                else:
                    sel.unregister(pipe)

            now = time.monotonic()
            if progress_seconds > 0 and now >= next_heartbeat:
                _progress(f"    ... {progress_label}: still running ({now - start:.1f}s elapsed)")
                next_heartbeat = now + progress_seconds
    except KeyboardInterrupt:
        elapsed = time.monotonic() - start
        _progress(f"    !! {progress_label}: interrupted after {elapsed:.1f}s; terminating child")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise
    finally:
        sel.close()

    rc = proc.wait()
    elapsed = time.monotonic() - start
    stdout = "".join(captured["stdout"])
    stderr = "".join(captured["stderr"])
    if rc != 0:
        raise BenchmarkFailure(
            f"benchmark failed ({rc}): {' '.join(cmd)}",
            stdout=stdout, stderr=stderr, elapsed=elapsed, returncode=rc,
        )
    metrics = parse_kv(stdout)
    required = {"mean_ms", "median_ms", "p90_ms", "best_ms", "depth_checksum"}
    missing = sorted(required - metrics.keys())
    if missing:
        raise BenchmarkFailure(
            f"benchmark output missing {missing}",
            stdout=stdout, stderr=stderr, elapsed=elapsed, returncode=rc,
        )
    return metrics, stdout, stderr, elapsed


def print_rows(rows: Iterable[dict[str, object]]) -> None:
    print(f"{'model':18} {'backend':12} {'thr':>4} {'median ms':>10} {'p90 ms':>10} {'mean ms':>10} {'wall s':>8} {'tensor':>9}")
    print("-" * 91)
    for row in rows:
        m = row.get("metrics")
        if not isinstance(m, dict):
            print(f"{str(row['model']):18.18} {str(row['requested_backend']):12.12} "
                  f"{int(row['requested_threads']):4d} {'FAILED':>10} {'-':>10} {'-':>10} "
                  f"{float(row.get('wall_seconds', 0.0)):8.1f} {str(row.get('metal_tensor_api', 'unknown')):>9}")
            continue
        print(f"{str(row['model']):18.18} {str(row['requested_backend']):12.12} "
              f"{int(row['requested_threads']):4d} {float(m['median_ms']):10.3f} "
              f"{float(m['p90_ms']):10.3f} {float(m['mean_ms']):10.3f} "
              f"{float(row.get('wall_seconds', 0.0)):8.1f} {str(row['metal_tensor_api']):>9}")


def write_payload(args: argparse.Namespace, threads_list: list[int], rows: list[dict[str, object]], *, complete: bool) -> None:
    payload = {
        "schema": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "settings": {
            "width": args.width, "height": args.height, "tokens": args.tokens,
            "threads": threads_list, "refine": args.refine,
            "warmup": args.warmup, "runs": args.runs,
            "progress_seconds": args.progress_seconds,
        },
        "results": rows,
        "failed_cases": sum(1 for row in rows if row.get("status") == "failed"),
        "successful": complete and all(row.get("status") != "failed" for row in rows),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", default="build/moge-bench", help="path to moge-bench executable")
    p.add_argument("--model", action="append", type=parse_model, required=True,
                   help="model as LABEL=PATH (repeatable); LABEL defaults to filename stem")
    p.add_argument("--backend", action="append", choices=["auto", "cpu", "vulkan", "metal"],
                   help="backend to benchmark (repeatable; default metal)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--tokens", type=int, default=0)
    p.add_argument("--threads", type=int, action="append",
                   help="thread count to test (repeatable; 0/omitted uses runtime default)")
    p.add_argument("--refine", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=7)
    p.add_argument("--progress-seconds", type=float,
                   default=float(os.environ.get("MOGE_PROGRESS_SECONDS", "5")),
                   help="heartbeat interval for slow cases; 0 disables (default: 5s)")
    p.add_argument("--stream-child", action="store_true",
                   help="also mirror child stdout/stderr live (very verbose for Metal)")
    p.add_argument("--allow-cpu-fallback", action="store_true",
                   help="allow scheduler CPU fallback for GPU backends (GPU-only is the default)")
    p.add_argument("--keep-going", action="store_true",
                   help="record failed cases and continue with the rest of the matrix")
    p.add_argument("--output", default="bench-results.json")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.width <= 0 or args.height <= 0 or args.runs <= 0 or args.warmup < 0:
        raise SystemExit("invalid benchmark dimensions/counts")
    if args.progress_seconds < 0:
        raise SystemExit("--progress-seconds must be >= 0")
    backends = args.backend or ["metal"]
    threads_list = args.threads or [0]
    if any(t < 0 for t in threads_list):
        raise SystemExit("--threads must be >= 0")

    jobs: list[tuple[ModelSpec, str, int]] = [
        (model, backend, threads)
        for model in args.model
        for backend in backends
        for threads in threads_list
    ]
    rows: list[dict[str, object]] = []
    write_payload(args, threads_list, rows, complete=False)
    _progress(
        f"benchmark matrix: {len(jobs)} case(s), {args.width}x{args.height}, "
        f"warmup={args.warmup}, runs={args.runs}; progress every {args.progress_seconds:g}s"
    )

    try:
        for index, (model, backend, threads) in enumerate(jobs, 1):
            thread_label = str(threads) if threads else "default"
            label = f"{index}/{len(jobs)} model={model.label} backend={backend} threads={thread_label}"
            _progress(f"==> START {label}")
            cmd = command_for(args, model, backend, threads)
            try:
                metrics, stdout, stderr, elapsed = run_one(
                    cmd,
                    progress_label=label,
                    progress_seconds=args.progress_seconds,
                    stream_child=args.stream_child,
                )
            except BenchmarkFailure as exc:
                combined = exc.stdout + "\n" + exc.stderr
                row = {
                    "status": "failed",
                    "model": model.label,
                    "path": model.path,
                    "requested_backend": backend,
                    "requested_threads": threads,
                    "metal_tensor_api": metal_tensor_status(combined) if backend == "metal" else "n/a",
                    "command": cmd,
                    "stdout": exc.stdout,
                    "stderr": exc.stderr,
                    "error": str(exc),
                    "returncode": exc.returncode,
                    "wall_seconds": exc.elapsed,
                    "memory": memory_trace(combined),
                }
                rows.append(row)
                write_payload(args, threads_list, rows, complete=False)
                _progress(f"<!! FAIL  {label} after {exc.elapsed:.1f}s: {exc}")
                if not args.keep_going:
                    raise
                continue

            row = {
                "status": "ok",
                "model": model.label,
                "path": model.path,
                "requested_backend": backend,
                "requested_threads": threads,
                "metal_tensor_api": metal_tensor_status(stdout + "\n" + stderr) if backend == "metal" else "n/a",
                "metrics": metrics,
                "command": cmd,
                "stderr": stderr,
                "wall_seconds": elapsed,
                "memory": memory_trace(stdout + "\n" + stderr),
            }
            rows.append(row)
            write_payload(args, threads_list, rows, complete=False)
            _progress(
                f"<== DONE  {label} in {elapsed:.1f}s "
                f"(median={float(metrics['median_ms']):.3f} ms, p90={float(metrics['p90_ms']):.3f} ms)"
            )
    except KeyboardInterrupt:
        write_payload(args, threads_list, rows, complete=False)
        _progress(f"benchmark interrupted; preserved {len(rows)}/{len(jobs)} completed case(s) in {args.output}")
        raise

    write_payload(args, threads_list, rows, complete=True)
    print_rows(rows)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
