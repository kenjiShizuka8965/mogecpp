from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_release_has_small_supported_shell_surface():
    scripts = sorted(p.name for p in (ROOT / "tests").glob("*.sh"))
    assert len(scripts) <= 20
    assert not any(re.search(r"v\d+|sweep|probe|experiment|compare_.*_m5", n, re.I) for n in scripts)
    required = {
        "build.sh", "smoke.sh", "bench_cpu.sh", "bench_metal.sh", "bench_vulkan.sh",
        "vulkan_reference_test.sh", "vulkan_release_test.sh", "package_release.sh",
    }
    assert required <= set(scripts)


def test_release_has_no_experimental_build_switches():
    cmake = (ROOT / "CMakeLists.txt").read_text()
    build = (ROOT / "tests/build.sh").read_text()
    forbidden = [
        "MOGE_BUILD_EXPERIMENTAL_METAL", "MOGE_VULKAN_MINIMAL",
        "MOGE_VULKAN_OPT_SWEEP", "MOGE_VULKAN_DECODER_TILE_SWEEP",
    ]
    for name in forbidden:
        assert name not in cmake
        assert name not in build


def test_release_uses_bundled_ggml_source_directly():
    common = (ROOT / "tests/common.sh").read_text()
    assert 'GGML_DEFAULT="$ROOT/third_party/ggml-$GGML_VERSION"' in common
    assert not (ROOT / "third_party/ggml-0.25.3.zip").exists()
    assert (ROOT / "third_party/ggml-0.25.3/CMakeLists.txt").is_file()
    assert ".deps" not in common
    assert "unzip" not in common


def test_release_validation_has_no_experiment_matrix():
    s = (ROOT / "tests/vulkan_release_test.sh").read_text()
    assert "--reference" in s
    assert "vulkan_reference_test.sh" in s
    assert "bench_vulkan.sh" in s
    assert "profile_vulkan.sh" in s
    assert "vulkan_gpu_state.sh" in s
    assert "SWEEP" not in s
    assert "MOGE_VK_" not in s
    assert "MOGE_DENSE_" not in s
    assert "MOGE_SPARSE_" not in s


def test_release_tree_has_no_handoff_journal():
    assert not (ROOT / "CONTINUATION.md").exists()
    assert not (ROOT / "log.md").exists()
    assert not (ROOT / "docs/M5.md").exists()


def test_vulkan_submit_threshold_remains_mutable():
    src = (ROOT / "third_party/ggml-0.25.3/src/ggml-vulkan/ggml-vulkan.cpp").read_text()
    assert "uint64_t flops_per_submit = std::min(flops_cap, ctx->last_total_flops / 40u);" in src
    assert "const uint64_t flops_per_submit" not in src
    assert "flops_per_submit *= 2;" in src


def test_release_package_excludes_python_caches():
    s = (ROOT / "tests/package_release.sh").read_text()
    assert "--exclude='*/__pycache__'" in s
    assert "--exclude='*/.pytest_cache'" in s
    assert "--exclude='*.pyc'" in s
    assert "--exclude='./install-*'" in s


def test_mpsgraph_release_paths_disable_cross_hardware_placement():
    conv = (ROOT / "src/mpsgraph_conv.mm").read_text()
    sdpa = (ROOT / "src/mpsgraph_sdpa.mm").read_text()
    for src in (conv, sdpa):
        assert "MPSGraphOptimizationLevel0" in src
        assert ".optimizationLevel=MPSGraphOptimizationLevel0;" in src
    # Fused SDPA must be precompiled for the explicit Metal device too; using
    # graph.run directly would fall back to the framework's default compiler.
    assert "compileWithDevice:gd" in sdpa
    assert "[c->graph runWithMTLCommandQueue" not in sdpa


def test_release_publication_docs_and_install_surface():
    for rel in ("CHANGELOG.md", "THIRD_PARTY_NOTICES.md", "CREDITS.md", "SECURITY.md", "CONTRIBUTING.md", "docs/RELEASE_0.4.2.md", "docs/GITHUB_RELEASE_0.4.2.md", "docs/GITHUB_RELEASE_CHECKLIST.md"):
        assert (ROOT / rel).is_file()
    api = (ROOT / "docs/API.md").read_text()
    assert "// 0.4.2" in api
    assert "// 0.3.2" not in api
    cmake = (ROOT / "CMakeLists.txt").read_text()
    assert "THIRD_PARTY_NOTICES.md" in cmake
    assert "CHANGELOG.md" in cmake
    assert "CREDITS.md" in cmake
    assert "SECURITY.md" in cmake
    assert "docs/RELEASE_0.4.2.md" in cmake
    license_text = (ROOT / "LICENSE").read_text()
    assert "Copyright (c) 2026 Kenji8965" in license_text
    credits = (ROOT / "CREDITS.md").read_text()
    assert "ChatGPT 5.6 Sol (OpenAI)" in credits
    assert "project direction" in credits.lower()
