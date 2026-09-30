# Contributing

Contributions are welcome when they preserve the small public API, deterministic release behavior, and backend portability goals of moge-ggml.

## Before opening a pull request

1. Build the affected backend in Release mode.
2. Run `tests/run_python_tests.sh`.
3. Run the relevant smoke/benchmark test when changing execution code.
4. Keep model weights, benchmark result archives, build directories, and generated outputs out of commits.
5. Preserve third-party copyright and license notices when modifying vendored code.

Backend performance changes should include measurements before and after the change on the same machine, driver/runtime, model, input, and benchmark procedure. A faster result must not weaken golden-output validation.

The public C/C++ API follows `docs/STABILITY.md`. API/ABI changes should be explicit rather than incidental.
