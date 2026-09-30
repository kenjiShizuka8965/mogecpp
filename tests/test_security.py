from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_bounds_untrusted_image_inputs_before_decode():
    s = (ROOT / "examples/moge-cli.cpp").read_text()
    for token in ["STBI_ONLY_PNG", "STBI_ONLY_JPEG", "STBI_MAX_DIMENSIONS 8192", "stbi_info_from_memory",
                  "kHardMaxImagePixels = 16ull * 1024ull * 1024ull", "kHardMaxInputBytes = 128ull * 1024ull * 1024ull",
                  "checked_mul", "validate_result"]:
        assert token in s


def test_mogg_parser_bounds_counts_and_payloads():
    s = (ROOT / "src/mogg.cpp").read_text()
    for token in ["kMaxMetadataEntries", "kMaxTensorEntries", "expected_tensor_bytes",
                  "duplicate MOGG tensor name", "unreasonable MOGG metadata count", "unreasonable MOGG tensor count"]:
        assert token in s


def test_library_has_embedding_safety_caps():
    s = (ROOT / "src/moge.cpp").read_text()
    assert "validate_image_request" in s
    assert "kMaxInputPixels = 64ull * 1024ull * 1024ull" in s
    assert "num_tokens exceeds library safety limit" in s
    assert "refine_steps exceeds library safety limit" in s
