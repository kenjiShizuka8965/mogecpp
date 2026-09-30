import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bench_matrix", ROOT / "tools/bench_matrix.py")
bench = importlib.util.module_from_spec(spec)
sys.modules["bench_matrix"] = bench
spec.loader.exec_module(bench)


def test_parse_benchmark_kv():
    parsed = bench.parse_kv("""\
model_version=3
backend=Metal
runs=7
mean_ms=12.300
median_ms=12.100
p90_ms=13.000
best_ms=11.800
stddev_ms=0.400
mean_mpix_s=25.0
samples_ms=12.0,13.0,11.8
depth_checksum=1.25
""")
    assert parsed["model_version"] == 3
    assert parsed["backend"] == "Metal"
    assert parsed["median_ms"] == 12.1
    assert parsed["samples_ms"] == [12.0, 13.0, 11.8]


def test_metal_tensor_status_from_ggml_logs():
    assert bench.metal_tensor_status("ggml_metal_device_init: has tensor = true") == "enabled"
    assert bench.metal_tensor_status("ggml_metal_device_init: has tensor = false") == "disabled"
    assert bench.metal_tensor_status(
        "ggml_metal_library_init: ggml-tensor.metallib not found - disabling the tensor API"
    ) == "disabled"
    assert bench.metal_tensor_status("ordinary Metal startup") == "unknown"


def test_run_one_reports_heartbeat_for_slow_child(capsys):
    import sys

    code = """
import time
time.sleep(0.12)
print('model_version=2')
print('runs=1')
print('mean_ms=1.0')
print('median_ms=1.0')
print('p90_ms=1.0')
print('best_ms=1.0')
print('stddev_ms=0.0')
print('mean_mpix_s=1.0')
print('depth_checksum=2.0')
"""
    metrics, stdout, stderr, elapsed = bench.run_one(
        [sys.executable, "-c", code],
        progress_label="unit-slow-case",
        progress_seconds=0.02,
    )
    assert metrics["median_ms"] == 1.0
    assert "depth_checksum=2.0" in stdout
    assert stderr == ""
    assert elapsed >= 0.1
    captured = capsys.readouterr()
    assert "unit-slow-case: still running" in captured.err


def test_run_one_failure_preserves_diagnostics():
    import sys

    code = """
import sys
print('partial_metric=1')
print('simulated GPU OOM', file=sys.stderr)
raise SystemExit(7)
"""
    try:
        bench.run_one([sys.executable, "-c", code], progress_seconds=0)
    except bench.BenchmarkFailure as exc:
        assert exc.returncode == 7
        assert "partial_metric=1" in exc.stdout
        assert "simulated GPU OOM" in exc.stderr
        assert exc.elapsed >= 0
    else:
        raise AssertionError("expected BenchmarkFailure")


def test_gpu_command_requests_gpu_resident_inputs_by_default():
    import argparse
    a = argparse.Namespace(bench="moge-bench", width=640, height=480, refine=3,
                           warmup=1, runs=2, tokens=0, allow_cpu_fallback=False)
    cmd = bench.command_for(a, bench.ModelSpec("m", "m.mogg"), "metal", 0)
    assert "--gpu-resident-inputs" in cmd
    cmd_cpu = bench.command_for(a, bench.ModelSpec("m", "m.mogg"), "cpu", 4)
    assert "--gpu-resident-inputs" not in cmd_cpu
