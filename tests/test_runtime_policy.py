from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_vulkan_policy_is_fixed_not_environment_selected():
    vk = (ROOT / "third_party/ggml-0.25.3/src/ggml-vulkan/ggml-vulkan.cpp").read_text()
    gen = (ROOT / "third_party/ggml-0.25.3/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp").read_text()
    assert "_small_vec2" not in vk and "_small_vec4" not in vk
    assert "_small_vec2" not in gen and "_small_vec4" not in gen
    assert "MOGE_VK_" not in vk
    assert "last_total_flops / 40u" in vk
    assert "src0->ne[0] <= 64" in vk


def test_sparse_release_policy_is_fixed():
    s = (ROOT / "src/sparse.cpp").read_text()
    forbidden = [
        "MOGE_MEMORY_PROFILE", "MOGE_SPARSE_GATHER_TYPE", "MOGE_SPARSE_OUTPUT_ASSEMBLY",
        "MOGE_SPARSE_STATE_TYPE", "MOGE_SPARSE_STATE_CACHE", "MOGE_SPARSE_TOPOLOGY_CACHE",
        "MOGE_SPARSE_PHASE_LOCAL_SCHED", "MOGE_SPARSE_METAL_HYBRID",
    ]
    for name in forbidden:
        assert name not in s
    assert "GGML_TYPE_F32" in s
    assert "ggml_set_2d_inplace" in s


def test_dense_release_policy_is_fixed():
    s = (ROOT / "src/moge.cpp").read_text()
    d = (ROOT / "src/dense.cpp").read_text()
    forbidden = [
        "MOGE_DENSE_FINAL_TILE_ROWS", "MOGE_DENSE_STREAM_TILE_ROWS", "MOGE_DENSE_STREAM_LEVEL1",
        "MOGE_DENSE_STATE_TYPE", "MOGE_DENSE_DIRECT_CONV", "MOGE_DENSE_IM2COL_TILE_ROWS",
        "MOGE_DINO_WEIGHT_SHADOW_F16", "MOGE_DINO_Q8_ACT_F16", "MOGE_MAPPED_WEIGHTS",
    ]
    for name in forbidden:
        assert name not in s
        assert name not in d
    assert "return 128;" in s


def test_only_supported_runtime_environment_controls_remain_in_core():
    names = set()
    for path in (ROOT / "src").glob("*.*"):
        if path.suffix not in {".cpp", ".hpp", ".mm"}:
            continue
        text = path.read_text(errors="ignore")
        import re
        names.update(re.findall(r'getenv\("(MOGE_[A-Z0-9_]+)"\)', text))
    assert names <= {
        "MOGE_SPARSE_TOPOLOGY_THREADS", "MOGE_SPARSE_TRACE", "MOGE_DENSE_TRACE",
        "MOGE_POST_THREADS",
    }
