from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_memory_loading_is_public_and_explicit():
    h = (ROOT / "include/moge_ggml/moge.hpp").read_text()
    c = (ROOT / "include/moge_ggml/moge_c.h").read_text()
    assert "Model(const void * mogg_data, size_t mogg_size" in h
    assert "from_borrowed_memory" in h
    assert "moge_model_load_memory" in c
    assert "copy_bytes" in c
    assert "MOGE_API_VERSION 1" in c


def test_cli_and_install_surface_are_documented():
    readme = (ROOT / "README.md").read_text()
    cmake = (ROOT / "CMakeLists.txt").read_text()
    assert "moge-cli" in readme
    assert "moge::moge-ggml" in readme
    assert "moge_model_load_memory" in readme
    assert "add_executable(moge-cli" in cmake
    assert "install(DIRECTORY include/moge_ggml" in cmake


def test_cmake_is_the_single_documented_build_path():
    readme = (ROOT / "README.md").read_text().lower()
    assert not (ROOT / "build.zig").exists()
    assert "zig build" not in readme
    assert "ggml_source_dir" in readme
    assert "third_party/ggml-0.25.3" in readme
    assert (ROOT / "third_party/ggml-0.25.3/src/ggml.c").exists()


def test_public_api_docs_cover_model_memory_ownership():
    api = (ROOT / "docs/API.md").read_text()
    assert "Model::from_borrowed_memory" in api
    assert "moge_model_load_memory" in api
    assert "caller must keep the bytes alive" in api


def test_cli_uses_only_public_api():
    cli = (ROOT / "examples/moge-cli.cpp").read_text()
    assert "#include <moge_ggml/moge.hpp>" in cli
    assert '#include "../src/' not in cli
    assert "moge::Model model" in cli
    assert "model.infer" in cli
