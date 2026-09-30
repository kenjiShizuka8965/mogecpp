"""Numerical check for the host-side DINOv2 positional interpolation formula.

This mirrors src/moge.cpp. Keeping the reference in Python lets CI compare the
coordinate/Keys-cubic semantics directly with torch even when ggml headers are
not installed in the test environment.
"""

import math

import torch
import torch.nn.functional as F


def cubic1(x: float) -> float:
    a = -0.75
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0


def cubic2(x: float) -> float:
    a = -0.75
    return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a


def axis_samples(nin: int, nout: int):
    sf = (nout + 0.1) / nin
    scale = 1.0 / sf
    out = []
    for o in range(nout):
        real = scale * (o + 0.5) - 0.5
        base = math.floor(real)
        t = real - base
        ix = [min(max(base + q, 0), nin - 1) for q in (-1, 0, 1, 2)]
        wt = [cubic2(t + 1.0), cubic1(t), cubic1(1.0 - t), cubic2(2.0 - t)]
        out.append((ix, wt))
    return out


def host_formula(src: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    # src [1,C,M,M], matching the patch portion after DINO reshape/permute.
    _, c, m, m2 = src.shape
    assert m == m2
    ys, xs = axis_samples(m, out_h), axis_samples(m, out_w)
    dst = torch.empty((1, c, out_h, out_w), dtype=torch.float32)
    for oy in range(out_h):
        yi, yw = ys[oy]
        for ox in range(out_w):
            xi, xw = xs[ox]
            for ch in range(c):
                acc = 0.0
                for ky in range(4):
                    for kx in range(4):
                        acc += float(src[0, ch, yi[ky], xi[kx]]) * yw[ky] * xw[kx]
                dst[0, ch, oy, ox] = acc
    return dst


def test_dino_scale_factor_bicubic_matches_torch():
    torch.manual_seed(123)
    for m, oh, ow in ((4, 3, 5), (5, 8, 2), (16, 24, 32), (16, 16, 16)):
        src = torch.randn(1, 3, m, m, dtype=torch.float32)
        sy, sx = (oh + 0.1) / m, (ow + 0.1) / m
        ref = F.interpolate(src, scale_factor=(sy, sx), mode="bicubic", antialias=False)
        got = host_formula(src, oh, ow)
        assert ref.shape == got.shape
        torch.testing.assert_close(got, ref, rtol=3e-4, atol=5e-6)
