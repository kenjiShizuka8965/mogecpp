#!/usr/bin/env python3
"""Convert upstream MoGe-2 / MoGe-3 checkpoints to MOGG for moge-ggml.

MOGG is a small inference container deliberately tailored to this port.  It keeps
PyTorch parameter names but permits offline graph-preserving prepacking that is
awkward to express in a generic checkpoint format:

* 1x1 Conv2d -> 2D matrix for MUL_MAT and K-quantization.
* stride-2/kernel-2 ConvTranspose2d -> phase-major [4*Cout,Cin] matrix.  Runtime
  executes one GEMM + exact pixel shuffle instead of a transposed-convolution op.
* MoGe-3 sparse 3x3x3 kernels -> [Cout,27*Cin] matrices used with gathered
  neighbor features.
* PixelShuffle resampler channels -> phase-major order so runtime shuffling is a
  cheap strided scatter.

The converter consumes the embedded ``model_config`` in official .pt checkpoints
whenever present.  Safetensors without config require ``--config``.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    from safetensors.torch import load_file as safetensors_load_file
except Exception:
    safetensors_load_file = None

MAGIC = b"MOGG\0\0\0\x01"
VERSION = 1
DEFAULT_ALIGNMENT = 256

# Must match src/mogg.hpp.
T_F32, T_F16, T_BF16, T_I32 = 1, 2, 3, 4
TF_SENSITIVE = 1 << 0
TF_LINEARIZED_1X1 = 1 << 1
TF_PHASED_CONVT = 1 << 2
TF_SPARSE = 1 << 3
TF_SPARSE_PACKED = 1 << 4
TF_PIXELSHUFFLE_PHASED = 1 << 5

META_I64, META_F64, META_STRING = 1, 2, 3
META_I64_LIST, META_STRING_LIST, META_BOOL = 4, 5, 6


def align_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def p16(s: str) -> bytes:
    b = s.encode("utf-8")
    if len(b) > 0xFFFF:
        raise ValueError(f"string too long for MOGG: {s[:80]!r}")
    return struct.pack("<H", len(b)) + b


def bf16_bytes(x: np.ndarray) -> bytes:
    x = np.asarray(x, dtype=np.float32, order="C")
    u = x.view(np.uint32)
    # IEEE round-to-nearest-even before dropping low 16 bits.
    u = u + np.uint32(0x7FFF) + ((u >> 16) & 1)
    return (u >> 16).astype("<u2", copy=False).tobytes(order="C")


def tensor_payload(t: torch.Tensor, dtype: str) -> tuple[int, bytes]:
    t = t.detach().cpu().contiguous()
    if not torch.is_floating_point(t):
        if t.dtype in (torch.int32, torch.int64):
            a = t.to(torch.int32).numpy().astype("<i4", copy=False)
            return T_I32, a.tobytes(order="C")
        raise TypeError(f"unsupported tensor dtype {t.dtype}")
    a = t.float().numpy()
    if dtype == "f32":
        return T_F32, np.asarray(a, dtype="<f4", order="C").tobytes(order="C")
    if dtype == "f16":
        return T_F16, np.asarray(a, dtype="<f2", order="C").tobytes(order="C")
    if dtype == "bf16":
        return T_BF16, bf16_bytes(a)
    raise ValueError(dtype)


@dataclass
class TensorRec:
    name: str
    shape: tuple[int, ...]  # logical row-major / PyTorch shape
    typ: int
    flags: int
    data: bytes
    offset: int = 0


class MoggWriter:
    def __init__(self, alignment: int = DEFAULT_ALIGNMENT):
        if alignment < 64 or alignment & (alignment - 1):
            raise ValueError("alignment must be a power of two >= 64")
        self.alignment = alignment
        self.meta: dict[str, Any] = {}
        self.tensors: list[TensorRec] = []

    def add(self, key: str, value: Any) -> None:
        if key in self.meta:
            raise ValueError(f"duplicate metadata key {key}")
        self.meta[key] = value

    def add_tensor(self, r: TensorRec) -> None:
        if not 1 <= len(r.shape) <= 4:
            raise ValueError(f"{r.name}: MOGG supports 1..4D after prepack, got {r.shape}")
        self.tensors.append(r)

    @staticmethod
    def _meta_payload(v: Any) -> tuple[int, bytes]:
        if isinstance(v, bool):
            return META_BOOL, struct.pack("<B", int(v))
        if isinstance(v, int):
            return META_I64, struct.pack("<q", v)
        if isinstance(v, float):
            return META_F64, struct.pack("<d", v)
        if isinstance(v, str):
            b = v.encode("utf-8")
            return META_STRING, struct.pack("<I", len(b)) + b
        if isinstance(v, (list, tuple)):
            if all(isinstance(x, (bool, int)) for x in v):
                return META_I64_LIST, struct.pack("<I", len(v)) + b"".join(struct.pack("<q", int(x)) for x in v)
            if all(isinstance(x, str) for x in v):
                return META_STRING_LIST, struct.pack("<I", len(v)) + b"".join(p16(x) for x in v)
        raise TypeError(f"unsupported MOGG metadata value {type(v).__name__}: {v!r}")

    def _index_bytes(self, absolute_offsets: list[int]) -> bytes:
        out = bytearray()
        for r, off in zip(self.tensors, absolute_offsets):
            ne = list(reversed(r.shape)) + [1] * (4 - len(r.shape))
            out += p16(r.name)
            out += struct.pack("<HBB", r.typ, len(r.shape), r.flags)
            out += struct.pack("<4q", *ne)
            out += struct.pack("<QQ", off, len(r.data))
        return bytes(out)

    def write(self, path: os.PathLike[str] | str) -> None:
        meta_blob = bytearray()
        for k, v in self.meta.items():
            tag, payload = self._meta_payload(v)
            meta_blob += p16(k) + struct.pack("<BI", tag, len(payload)) + payload

        header_size = 8 + 4 * 4 + 8 * 2
        meta_offset = header_size
        # Tensor index is fixed-size except names, so first make a dummy index.
        dummy_offsets = [0] * len(self.tensors)
        index0 = self._index_bytes(dummy_offsets)
        data_start = align_up(meta_offset + len(meta_blob) + len(index0), self.alignment)
        offsets: list[int] = []
        cur = data_start
        for r in self.tensors:
            cur = align_up(cur, self.alignment)
            offsets.append(cur)
            cur += len(r.data)
        index = self._index_bytes(offsets)
        assert len(index) == len(index0)

        with open(path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<IIIIQQ", VERSION, len(self.meta), len(self.tensors), self.alignment,
                                meta_offset, data_start))
            f.write(meta_blob)
            f.write(index)
            if f.tell() > data_start:
                raise AssertionError("MOGG index overran data_start")
            f.write(b"\0" * (data_start - f.tell()))
            for r, off in zip(self.tensors, offsets):
                f.write(b"\0" * (off - f.tell()))
                f.write(r.data)


def load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
    if path.suffix.lower() == ".safetensors":
        if safetensors_load_file is None:
            raise RuntimeError("install safetensors to load .safetensors files")
        return dict(safetensors_load_file(str(path), device="cpu")), None
    # Official MoGe checkpoints are safe weights + dictionaries.
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = None
    if isinstance(obj, Mapping) and "model" in obj and isinstance(obj["model"], Mapping):
        cfg = obj.get("model_config")
        obj = obj["model"]
    elif isinstance(obj, Mapping) and "state_dict" in obj and isinstance(obj["state_dict"], Mapping):
        cfg = obj.get("model_config")
        obj = obj["state_dict"]
    if not isinstance(obj, Mapping):
        raise TypeError("checkpoint is not a state_dict-like mapping")
    sd = {str(k): v for k, v in obj.items() if torch.is_tensor(v)}
    for prefix in ("module.", "model."):
        if sd and all(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):]: v for k, v in sd.items()}
    return sd, cfg


def load_config(path: str | None, embedded: Any) -> dict[str, Any]:
    if path:
        cfg = json.loads(Path(path).read_text())
        if "model" in cfg and isinstance(cfg["model"], Mapping):
            cfg = cfg["model"]
        return dict(cfg)
    if isinstance(embedded, Mapping):
        # checkpoint['model_config'] is normally already the constructor dict,
        # but also accept an entire training config for convenience.
        if "model" in embedded and isinstance(embedded["model"], Mapping) and "encoder" not in embedded:
            return dict(embedded["model"])
        return dict(embedded)
    raise ValueError("checkpoint has no embedded model_config; pass --config configs/train/v2.json or v3.json")


def model_version(sd: Mapping[str, torch.Tensor], override: str) -> int:
    if override != "auto":
        return int(override)
    return 3 if any(k.startswith("refiner.") for k in sd) else 2


def dino_info(sd: Mapping[str, torch.Tensor]) -> tuple[int, int, int, str]:
    p = "encoder.backbone."
    cls = sd[p + "cls_token"]
    embed = int(cls.shape[-1])
    blocks = sorted({int(k[len(p + "blocks."):].split(".", 1)[0]) for k in sd if k.startswith(p + "blocks.")})
    if not blocks:
        raise ValueError("no encoder.backbone.blocks.* parameters found")
    depth = max(blocks) + 1
    # DINOv2 released variants use 64-wide heads.
    heads = embed // 64
    ffn = "swiglu" if any(k.startswith(p + "blocks.0.mlp.w12") for k in sd) else "gelu"
    return embed, depth, heads, ffn


def as_list(v: Any, n: int, *, none: int = -1) -> list[int]:
    if v is None:
        return [none] * n
    if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
        if len(v) != n:
            raise ValueError(f"expected {n} entries, got {len(v)}: {v}")
        return [none if x is None else int(x) for x in v]
    return [int(v)] * n


def str_list(v: Any, n: int) -> list[str]:
    if isinstance(v, str):
        return [v] * n
    if len(v) != n:
        raise ValueError(f"expected {n} entries, got {len(v)}")
    return [str(x) for x in v]


def add_stack_meta(w: MoggWriter, name: str, cfg: Mapping[str, Any] | None) -> None:
    p = name
    w.add(p + ".present", cfg is not None)
    if cfg is None:
        return
    dims = [int(x) for x in cfg["dim_res_blocks"]]
    n = len(dims)
    w.add(p + ".dim_in", as_list(cfg["dim_in"], n))
    w.add(p + ".dim_res_blocks", dims)
    w.add(p + ".dim_out", as_list(cfg.get("dim_out"), n))
    w.add(p + ".resamplers", str_list(cfg["resamplers"], n - 1))
    w.add(p + ".dim_times_res_block_hidden", int(cfg.get("dim_times_res_block_hidden", 1)))
    w.add(p + ".num_res_blocks", as_list(cfg.get("num_res_blocks", 1), n))
    w.add(p + ".res_block_in_norm", str(cfg.get("res_block_in_norm", "layer_norm")))
    w.add(p + ".res_block_hidden_norm", str(cfg.get("res_block_hidden_norm", "group_norm")))
    w.add(p + ".activation", str(cfg.get("activation", "relu")))


def is_1x1(t: torch.Tensor) -> bool:
    return t.ndim == 4 and tuple(t.shape[-2:]) == (1, 1)


def sensitive(name: str, cfg: Mapping[str, Any]) -> bool:
    # DINO interpolation explicitly promotes pos_embed to float in upstream,
    # and this matrix is tiny relative to the ViT trunk. Keeping it out of the
    # low-bit quantizer also lets the runtime reproduce the historical +0.1
    # scale-factor interpolation exactly from the MOGG payload.
    if name == "encoder.backbone.pos_embed":
        return True

    # Final point/normal output projections are explicitly fp32 in upstream.
    for head in ("points_head", "normal_head"):
        hc = cfg.get(head)
        if not hc or not name.startswith(head + ".output_blocks."):
            continue
        dims = list(hc["dim_res_blocks"])
        last = len(dims) - 1
        if name.startswith(f"{head}.output_blocks.{last}."):
            return True
    return False


def resampler_kind_for_tensor(name: str, cfg: Mapping[str, Any]) -> tuple[str | None, int | None]:
    # Returns stack name + resampler index for names like neck.resamplers.0.0.weight.
    for stack in ("neck", "points_head", "normal_head", "mask_head"):
        prefix = stack + ".resamplers."
        if name.startswith(prefix):
            tail = name[len(prefix):]
            try:
                idx = int(tail.split(".", 1)[0])
            except ValueError:
                return None, None
            sc = cfg.get(stack)
            if not sc:
                return None, None
            rs = str_list(sc["resamplers"], len(sc["dim_res_blocks"]) - 1)
            return rs[idx], idx
    return None, None


def phase_convtranspose(t: torch.Tensor) -> torch.Tensor:
    # PyTorch ConvTranspose2d [Cin,Cout,2,2] -> phase-major [4*Cout,Cin].
    if t.ndim != 4 or tuple(t.shape[-2:]) != (2, 2):
        raise ValueError(f"expected kernel-2 ConvTranspose2d, got {tuple(t.shape)}")
    cin, cout = int(t.shape[0]), int(t.shape[1])
    return t.permute(2, 3, 1, 0).contiguous().reshape(4 * cout, cin)


def phase_pixelshuffle(t: torch.Tensor) -> torch.Tensor:
    # Standard conv output order is c*4+p. Runtime wants p*C+c.
    out = int(t.shape[0])
    if out % 4:
        raise ValueError("pixel-shuffle projection output is not divisible by 4")
    c = out // 4
    if t.ndim == 1:
        return t.reshape(c, 4).t().contiguous().reshape(out)
    return t.reshape(c, 4, *t.shape[1:]).permute(1, 0, *range(2, t.ndim + 1)).contiguous().reshape_as(t)


def pack_sparse27(t: torch.Tensor) -> torch.Tensor:
    """Pack a FlexGEMM 3x3x3 kernel into the runtime gather-GEMM matrix.

    MoGe-3/FlexGEMM stores sparse kernels as [Cout,Kd,Kh,Kw,Cin], with Cin
    fastest. Runtime neighbor maps enumerate the matching (d,h,w) offset with
    w/z-bin fastest, so a contiguous reshape preserves the exact tap order.
    """
    if t.ndim != 5 or tuple(t.shape[1:4]) != (3, 3, 3):
        raise ValueError(f"expected [Cout,3,3,3,Cin], got {tuple(t.shape)}")
    return t.contiguous().reshape(t.shape[0], 27 * t.shape[-1])


def storage_dtype_for_tensor(name: str, args: argparse.Namespace) -> str:
    """Resolve persistent floating storage for one logical tensor."""
    if name.startswith("encoder."):
        return args.encoder_dtype or args.dtype
    if name.startswith("refiner."):
        return args.refiner_dtype or args.dtype
    return args.decoder_dtype or args.dtype


def convert(args: argparse.Namespace) -> None:
    src = Path(args.input)
    sd, embedded = load_checkpoint(src)
    cfg = load_config(args.config, embedded)
    ver = model_version(sd, args.version)
    embed, depth, heads, ffn = dino_info(sd)

    ecfg = dict(cfg["encoder"])
    inter = ecfg.get("intermediate_layers")
    if isinstance(inter, int):
        inter = list(range(depth - inter, depth))
    inter = [int(x) for x in inter]

    writer = MoggWriter(args.alignment)
    writer.add("format.name", "MOGG")
    writer.add("format.version", VERSION)
    component_dtypes = {
        "encoder": args.encoder_dtype or args.dtype,
        "decoder": args.decoder_dtype or args.dtype,
        "refiner": args.refiner_dtype or args.dtype,
    }
    unique_storage = set(component_dtypes.values())
    writer.add("format.storage", next(iter(unique_storage)) if len(unique_storage) == 1 else "mixed")
    writer.add("format.storage.profile", "default")
    writer.add("format.storage.encoder", component_dtypes["encoder"])
    writer.add("format.storage.decoder", component_dtypes["decoder"])
    writer.add("format.storage.refiner", component_dtypes["refiner"])
    writer.add("format.prepack.1x1", not args.no_prepack_1x1)
    writer.add("format.prepack.convtranspose_phase", not args.no_prepack_convt)
    writer.add("format.prepack.sparse27", ver == 3)
    writer.add("format.prepack.pixelshuffle_phase", True)
    writer.add("format.rotation", "none")
    writer.add("format.rotation.group", 0)
    writer.add("source.name", src.name)
    writer.add("model.version", ver)
    writer.add("model.remap_output", "exp" if ver == 3 else str(cfg.get("remap_output", "linear")))
    writer.add("model.num_tokens_range", [int(x) for x in cfg.get("num_tokens_range", [1200, 3600])])
    writer.add("encoder.backbone", str(ecfg["backbone"]))
    writer.add("encoder.embed_dim", embed)
    writer.add("encoder.depth", depth)
    writer.add("encoder.num_heads", heads)
    writer.add("encoder.dim_out", int(ecfg["dim_out"]))
    writer.add("encoder.ffn", ffn)
    writer.add("encoder.intermediate_layers", inter)

    for name in ("neck", "points_head", "normal_head", "mask_head"):
        add_stack_meta(writer, name, cfg.get(name))

    sh = cfg.get("scale_head")
    writer.add("scale_head.present", sh is not None)
    if sh:
        writer.add("scale_head.dims", [int(x) for x in sh["dims"]])

    rcfg = cfg.get("refiner") if ver == 3 else None
    writer.add("refiner.present", rcfg is not None)
    if rcfg:
        channels = [int(x) for x in rcfg["model_channels"]]
        writer.add("refiner.in_channels", int(rcfg.get("in_channels", 3)))
        writer.add("refiner.out_channels", int(rcfg.get("out_channels", 1)))
        writer.add("refiner.encoder_channels", int(rcfg["encoder_channels"]))
        writer.add("refiner.model_channels", channels)
        writer.add("refiner.encoder_blocks_per_level", as_list(rcfg.get("encoder_blocks_per_level", 1), len(channels)))
        writer.add("refiner.decoder_blocks_per_level", as_list(rcfg.get("decoder_blocks_per_level", 1), len(channels) - 1))
        writer.add("refiner.bottleneck_blocks", int(rcfg.get("bottleneck_blocks", 1)))
        writer.add("refiner.downsample_factors", as_list(rcfg.get("downsample_factors", 2), len(channels) - 1))
        writer.add("refiner.encoder_downsample", int(rcfg.get("encoder_downsample", 16)))
        writer.add("refiner.depth_resolution", float(cfg.get("refiner_depth_resolution", 256)))

    stats = {"linearized_1x1": 0, "phase_convt": 0, "sparse27": 0, "pixelshuffle": 0,
             "sensitive_f32": 0}
    seen: set[str] = set()

    for orig_name in sorted(sd):
        t = sd[orig_name].detach().cpu().contiguous()
        if not (torch.is_floating_point(t) or t.dtype in (torch.int32, torch.int64)):
            continue
        name = orig_name
        flags = 0
        kind, _ = resampler_kind_for_tensor(name, cfg)

        # Exact ConvTranspose2d(k=2,s=2) -> GEMM+shuffle prepack.
        if (not args.no_prepack_convt and kind == "conv_transpose" and name.endswith(".0.weight")):
            t = phase_convtranspose(t)
            name = name + ".mogg_phase"
            flags |= TF_PHASED_CONVT
            stats["phase_convt"] += 1
        # Sparse FlexGEMM kernel: [Cout,3,3,3,Cin] -> matrix [Cout,27*Cin].
        elif ver == 3 and name.startswith("refiner.") and t.ndim == 5 and tuple(t.shape[1:4]) == (3, 3, 3):
            t = pack_sparse27(t)
            name = name + ".mogg_sparse27"
            flags |= TF_SPARSE | TF_SPARSE_PACKED
            stats["sparse27"] += 1
        else:
            # PixelShuffle's first conv is identifiable by .0.{weight,bias}.
            if kind == "pixel_shuffle" and (name.endswith(".0.weight") or name.endswith(".0.bias")):
                t = phase_pixelshuffle(t)
                flags |= TF_PIXELSHUFFLE_PHASED
                stats["pixelshuffle"] += 1
            if not args.no_prepack_1x1 and is_1x1(t):
                t = t[:, :, 0, 0].contiguous()
                flags |= TF_LINEARIZED_1X1
                stats["linearized_1x1"] += 1

        # Every MoGe-3 refiner tensor belongs to the sparse subsystem, not only
        # the packed 3x3x3 kernels.  Mark the whole refiner so dense-model
        # quantization keeps it in floating point unless a dedicated converter explicitly handles sparse/refiner weights.
        if ver == 3 and orig_name.startswith("refiner."):
            flags |= TF_SPARSE

        is_sensitive = sensitive(orig_name, cfg)
        if is_sensitive:
            # Sensitivity is a semantic quantization guard. By default these
            # tensors remain F32 to preserve numerically sensitive parameters.
            flags |= TF_SENSITIVE
        keep_f32 = args.keep_sensitive_f32 and is_sensitive
        if keep_f32:
            stats["sensitive_f32"] += 1
        tdtype = "f32" if keep_f32 else storage_dtype_for_tensor(orig_name, args)
        typ, data = tensor_payload(t, tdtype)
        if name in seen:
            raise ValueError(f"duplicate converted tensor name {name}")
        seen.add(name)
        writer.add_tensor(TensorRec(name, tuple(int(x) for x in t.shape), typ, flags, data))

    # Older exports may omit buffers; synthesize exact DINO normalization values.
    for name, values in (("encoder.image_mean", [0.485, 0.456, 0.406]),
                         ("encoder.image_std", [0.229, 0.224, 0.225])):
        if name not in seen:
            t = torch.tensor(values, dtype=torch.float32).reshape(1, 3, 1, 1)
            # Keep normalization constants exact enough that changing a component
            # storage profile cannot perturb the input affine transform.
            typ, data = tensor_payload(t, "f32")
            writer.add_tensor(TensorRec(name, tuple(t.shape), typ, 0, data))

    writer.write(args.output)
    payload = sum(len(t.data) for t in writer.tensors)
    print(json.dumps({
        "output": str(args.output), "model_version": ver, "backbone": ecfg["backbone"],
        "embed_dim": embed, "depth": depth, "heads": heads, "ffn": ffn,
        "tensors": len(writer.tensors), "payload_mib": round(payload / 2**20, 2), **stats,
    }, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="upstream model.pt/.pth/.safetensors")
    p.add_argument("output", help="output .mogg")
    p.add_argument("--config", help="model or training JSON config; required if checkpoint has no model_config")
    p.add_argument("--version", choices=["auto", "2", "3"], default="auto")
    p.add_argument("--dtype", choices=["f32", "f16", "bf16"], default="f16",
                   help="default storage dtype for floating-point weights")
    p.add_argument("--encoder-dtype", choices=["f32", "f16", "bf16"],
                   help="override storage dtype for encoder.* tensors")
    p.add_argument("--decoder-dtype", choices=["f32", "f16", "bf16"],
                   help="override storage dtype for neck/heads/scale tensors")
    p.add_argument("--refiner-dtype", choices=["f32", "f16", "bf16"],
                   help="override storage dtype for refiner.* tensors (MoGe-3)")
    p.add_argument("--alignment", type=int, default=DEFAULT_ALIGNMENT)
    p.add_argument("--no-prepack-1x1", action="store_true")
    p.add_argument("--no-prepack-convt", action="store_true")
    p.add_argument("--keep-sensitive-f32", action=argparse.BooleanOptionalAction, default=True)
    return p


if __name__ == "__main__":
    convert(parser().parse_args())
