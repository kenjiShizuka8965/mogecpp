import importlib.util
import struct
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("conv", ROOT / "tools/convert_moge.py")
conv = importlib.util.module_from_spec(spec)
sys.modules["conv"] = conv
spec.loader.exec_module(conv)


def config(v3=True):
    dims = [1024, 256, 128, 64, 32]
    base_head = dict(
        dim_in=dims, dim_out=[None, None, None, None, 3],
        dim_res_blocks=dims, num_res_blocks=[0, 1, 1, 1, 0],
        res_block_in_norm="none", res_block_hidden_norm="none",
        resamplers=["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
    )
    c = {
        "encoder": {"backbone": "dinov2_vitl14", "intermediate_layers": [5, 11, 17, 23], "dim_out": 1024},
        "neck": dict(dim_in=[1026, 2, 2, 2, 2], dim_out=None, dim_res_blocks=dims,
                     num_res_blocks=[0, 2, 2, 2, 0], res_block_in_norm="none",
                     res_block_hidden_norm="none", resamplers=["conv_transpose"]*3+["bilinear"]),
        "points_head": base_head,
        "normal_head": base_head,
        "mask_head": {**base_head, "dim_out": [None, None, None, None, 1]},
        "scale_head": {"dims": [1024, 1024, 1024, 1]},
        "remap_output": "exp",
        "num_tokens_range": [1200, 3600],
    }
    if v3:
        c["refiner"] = dict(in_channels=3, out_channels=1, encoder_channels=1026,
                            model_channels=[32,64,128,256,512], encoder_blocks_per_level=1,
                            decoder_blocks_per_level=1, bottleneck_blocks=1,
                            downsample_factors=[2,2,2,2], encoder_downsample=16)
        c["refiner_depth_resolution"] = 256
    return c


def minimal_sd():
    sd = {
        "encoder.backbone.cls_token": torch.zeros(1, 1, 1024),
        "encoder.backbone.pos_embed": torch.zeros(1, 257, 1024),
        "encoder.backbone.patch_embed.proj.weight": torch.zeros(1024, 3, 14, 14),
        "encoder.output_projections.0.weight": torch.zeros(1024, 1024, 1, 1),
        "neck.input_blocks.0.weight": torch.zeros(1024, 1026, 1, 1),
        "neck.resamplers.0.0.weight": torch.arange(1024*256*4, dtype=torch.float32).reshape(1024,256,2,2),
        "neck.resamplers.0.0.bias": torch.zeros(256),
        "points_head.output_blocks.4.weight": torch.zeros(3,32,1,1),
        "points_head.output_blocks.4.bias": torch.zeros(3),
        "refiner.down_stages.0.0.conv1.weight": torch.zeros(32,3,3,3,32),
    }
    for i in range(24):
        sd[f"encoder.backbone.blocks.{i}.norm1.weight"] = torch.ones(1024)
    return sd


def test_prepack_shapes_and_types():
    sd = minimal_sd()
    embed, depth, heads, ffn = conv.dino_info(sd)
    assert (embed, depth, heads, ffn) == (1024, 24, 16, "gelu")

    one = sd["neck.input_blocks.0.weight"][:, :, 0, 0]
    typ, payload = conv.tensor_payload(one, "f16")
    assert typ == conv.T_F16 and len(payload) == one.numel() * 2

    ct = conv.phase_convtranspose(sd["neck.resamplers.0.0.weight"])
    assert ct.shape == (4 * 256, 1024)
    # phase 0 / output 0 contains each input's [0,0] coefficient
    assert torch.equal(ct[0], sd["neck.resamplers.0.0.weight"][:, 0, 0, 0])

    sp = sd["refiner.down_stages.0.0.conv1.weight"].reshape(32, 27 * 32)
    assert sp.shape == (32, 864)


def test_mogg_writer_roundtrip_header(tmp_path):
    w = conv.MoggWriter(256)
    w.add("model.version", 3)
    w.add("encoder.intermediate_layers", [5,11,17,23])
    w.add("refiner.present", True)
    typ, data = conv.tensor_payload(torch.arange(6, dtype=torch.float32).reshape(2,3), "f16")
    w.add_tensor(conv.TensorRec("x", (2,3), typ, conv.TF_LINEARIZED_1X1, data))
    p = tmp_path / "x.mogg"
    w.write(p)
    b = p.read_bytes()
    assert b[:8] == conv.MAGIC
    ver,nm,nt,align,mo,ds = struct.unpack_from("<IIIIQQ", b, 8)
    assert (ver,nm,nt,align,mo) == (1,3,1,256,40)
    assert ds % 256 == 0 and ds < len(b)


def test_phase_convtranspose_is_exact():
    torch.manual_seed(7)
    x = torch.randn(1, 3, 5, 4)
    w = torch.randn(3, 2, 2, 2)  # PyTorch ConvTranspose2d: [Cin,Cout,kH,kW]
    b = torch.randn(2)
    expected = torch.nn.functional.conv_transpose2d(x, w, b, stride=2)

    packed = conv.phase_convtranspose(w)  # [4*Cout,Cin], phase-major
    actual = torch.empty_like(expected)
    cout = w.shape[1]
    for ky in range(2):
        for kx in range(2):
            phase = ky * 2 + kx
            rows = packed[phase * cout:(phase + 1) * cout]
            slab = torch.einsum("oc,bchw->bohw", rows, x)
            actual[:, :, ky::2, kx::2] = slab + b[None, :, None, None]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_pixelshuffle_channel_preorder_is_exact():
    torch.manual_seed(11)
    cin, cout, h, w = 5, 3, 3, 4
    x = torch.randn(1, cin, h, w)
    weight = torch.randn(cout * 4, cin, 1, 1)
    bias = torch.randn(cout * 4)
    original = torch.nn.functional.pixel_shuffle(
        torch.nn.functional.conv2d(x, weight, bias), 2
    )

    # Converter changes standard PixelShuffle order [c*4 + phase] to
    # runtime phase-major [phase*C + c].
    wp, bp = conv.phase_pixelshuffle(weight), conv.phase_pixelshuffle(bias)
    phase_ch = torch.nn.functional.conv2d(x, wp, bp)
    actual = torch.empty_like(original)
    for ky in range(2):
        for kx in range(2):
            phase = ky * 2 + kx
            actual[:, :, ky::2, kx::2] = phase_ch[:, phase * cout:(phase + 1) * cout]
    torch.testing.assert_close(actual, original, rtol=1e-6, atol=1e-6)


def test_mogg_index_reverses_row_major_shape_for_ggml(tmp_path):
    """PyTorch row-major [rows,cols] bytes must be ggml ne=[cols,rows]."""
    w = conv.MoggWriter(256)
    w.add("x", 1)
    typ, data = conv.tensor_payload(torch.arange(6, dtype=torch.float32).reshape(2, 3), "f32")
    w.add_tensor(conv.TensorRec("matrix", (2, 3), typ, 0, data))
    p = tmp_path / "shape.mogg"
    w.write(p)

    b = p.read_bytes()
    _, nm, nt, _, mo, _ = struct.unpack_from("<IIIIQQ", b, 8)
    assert (nm, nt) == (1, 1)
    pos = mo
    # Skip the one integer metadata record.
    klen = struct.unpack_from("<H", b, pos)[0]
    pos += 2 + klen
    tag, n = struct.unpack_from("<BI", b, pos)
    assert tag == conv.META_I64
    pos += 5 + n
    nlen = struct.unpack_from("<H", b, pos)[0]
    pos += 2
    assert b[pos:pos+nlen].decode() == "matrix"
    pos += nlen
    _, nd, _ = struct.unpack_from("<HBB", b, pos)
    pos += 4
    ne = struct.unpack_from("<4q", b, pos)
    assert nd == 2
    assert ne == (3, 2, 1, 1)


def test_sparse27_pack_order_matches_runtime_neighbor_order():
    # Encode each [d,h,w,cin] slot with a unique value. A contiguous flatten of
    # [Cout,D,H,W,Cin] must make Cin fastest, then w/z-bin, h/column, d/row.
    t = torch.empty(1, 3, 3, 3, 2, dtype=torch.float32)
    for d in range(3):
        for h in range(3):
            for z in range(3):
                for ci in range(2):
                    t[0, d, h, z, ci] = 1000*d + 100*h + 10*z + ci
    packed = conv.pack_sparse27(t)
    expected = []
    for d in range(3):
        for h in range(3):
            for z in range(3):
                for ci in range(2):
                    expected.append(1000*d + 100*h + 10*z + ci)
    assert packed.shape == (1, 54)
    assert packed[0].tolist() == expected


def test_sparse27_tapwise_sum_matches_full_gather_gemm():
    # The low-memory runtime evaluates the packed sparse convolution as a sum
    # of 27 tap GEMMs instead of materializing [27*C,N]. This must be exactly
    # the same contraction as the original packed gather-GEMM formulation.
    torch.manual_seed(23)
    cin, cout, n = 3, 5, 11
    x = torch.randn(cin, n + 1)
    x[:, n] = 0.0  # sentinel row for missing neighbours
    rows = torch.randint(0, n + 1, (n, 27), dtype=torch.int64)
    w = torch.randn(cout, 27 * cin)

    gathered = torch.cat([x[:, rows[:, tap]] for tap in range(27)], dim=0)
    full = w @ gathered

    tapwise = torch.zeros_like(full)
    for tap in range(27):
        tapwise += w[:, tap * cin:(tap + 1) * cin] @ x[:, rows[:, tap]]

    torch.testing.assert_close(tapwise, full, rtol=2e-6, atol=2e-6)


def test_component_storage_dtype_overrides():
    import argparse
    a = argparse.Namespace(dtype="f16", encoder_dtype="bf16", decoder_dtype="f32", refiner_dtype=None)
    assert conv.storage_dtype_for_tensor("encoder.backbone.blocks.0.attn.qkv.weight", a) == "bf16"
    assert conv.storage_dtype_for_tensor("neck.input_blocks.0.weight", a) == "f32"
    assert conv.storage_dtype_for_tensor("points_head.output_blocks.4.weight", a) == "f32"
    assert conv.storage_dtype_for_tensor("refiner.input_proj.weight", a) == "f16"


def test_pos_embed_is_always_quantization_sensitive():
    c = config(v3=False)
    assert conv.sensitive("encoder.backbone.pos_embed", c)
    # This guard is semantic: it remains true even when a user chooses a lower
    # floating storage dtype with --no-keep-sensitive-f32.
    assert not conv.sensitive("encoder.backbone.cls_token", c)
