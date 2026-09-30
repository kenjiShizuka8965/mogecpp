import math

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional


def _axis_samples(n_in: int, n_out: int):
    # Python mirror of src/moge.cpp::linear_aa_axis_samples. Use float32 at
    # every scalar step because ATen's float input path computes AA weights in
    # scalar_t=float.
    f32 = lambda x: torch.tensor(x, dtype=torch.float32).item()
    scale = f32(f32(n_in) / f32(n_out))
    support = scale if scale >= 1.0 else 1.0
    invscale = f32(1.0 / scale) if scale >= 1.0 else 1.0
    max_interp = math.ceil(support) * 2 + 1
    out = []
    for i in range(n_out):
        center = f32(scale * f32(i + 0.5))
        # C++ truncates float -> int toward zero.
        xmin = max(int(f32(center - support + 0.5)), 0)
        xsize = min(int(f32(center + support + 0.5)), n_in) - xmin
        xsize = min(max(xsize, 0), max_interp)
        w = []
        total = 0.0
        for j in range(xsize):
            x = abs(f32(f32(f32(j + xmin) - center + 0.5) * invscale))
            ww = f32(1.0 - x) if x < 1.0 else 0.0
            w.append(ww)
            total = f32(total + ww)
        if total != 0.0:
            w = [f32(v / total) for v in w]
        out.append((xmin, w))
    return out


def _resize_aa(x: torch.Tensor, oh: int, ow: int) -> torch.Tensor:
    # x NCHW, float32; horizontal then vertical like ATen's separable kernel.
    assert x.dtype == torch.float32 and x.ndim == 4
    n, c, ih, iw = x.shape
    xs = _axis_samples(iw, ow)
    ys = _axis_samples(ih, oh)
    tmp = torch.empty((n, c, ih, ow), dtype=torch.float32)
    for ox, (first, w) in enumerate(xs):
        acc = torch.zeros((n, c, ih), dtype=torch.float32)
        for k, wk in enumerate(w):
            acc += x[:, :, :, first + k] * wk
        tmp[:, :, :, ox] = acc
    out = torch.empty((n, c, oh, ow), dtype=torch.float32)
    for oy, (first, w) in enumerate(ys):
        acc = torch.zeros((n, c, ow), dtype=torch.float32)
        for k, wk in enumerate(w):
            acc += tmp[:, :, first + k, :] * wk
        out[:, :, oy, :] = acc
    return out


@pytest.mark.parametrize("shape,target", [
    ((1, 3, 7, 11), (17, 23)),   # upsample
    ((1, 3, 17, 23), (7, 11)),   # downsample
    ((1, 3, 13, 29), (19, 11)),  # mixed up/down
    ((1, 3, 48, 64), (56, 84)),  # MoGe-like 4:3 upsample
    ((1, 3, 83, 127), (42, 56)), # stronger downsample
])
def test_host_antialias_bilinear_matches_torch(shape, target):
    torch.manual_seed(123)
    x = torch.rand(shape, dtype=torch.float32)
    got = _resize_aa(x, *target)
    ref = F.interpolate(x, size=target, mode="bilinear", align_corners=False, antialias=True)
    torch.testing.assert_close(got, ref, rtol=2e-6, atol=2e-6)
