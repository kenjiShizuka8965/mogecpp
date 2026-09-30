# Model conversion

`moge-convert` is the normal model-preparation tool. Official MoGe checkpoints are distributed as `model.pt`; the checkpoint contains both `model_config` and `model`, so no separate config file is required.

## Recommended

```sh
build/moge-convert model.pt
```

Default: `--quant q8`. The output filename is automatic: `v{model_version}_{quant}.moge`, for example `v3_q8.moge`.

## Precision choices

```sh
build/moge-convert model.pt --quant q8   # default; recommended deployment format
build/moge-convert model.pt --quant f16  # high-fidelity reference/storage format
build/moge-convert model.pt --quant f32  # debugging/reference only
```

Use `--output FILE.moge` only when you want to override the automatic name.

Q8 uses the validated Q8_0 policy: eligible dense 2D matrices of at least 4096 elements are quantized, while sensitive tensors and sparse-refiner tensors remain floating point.

## Safety

The `.pt` reader does not invoke Python or unpickle arbitrary classes. It reads the PyTorch ZIP container and accepts a restricted pickle subset needed for `model_config`, tensor descriptors, and tensor storages. Unsupported globals/opcodes are rejected.

## Calibration

When activation-importance data is needed for an external model-preparation workflow, build with `MOGE_BUILD_DEV_TOOLS=ON` and use `moge-calibrate`. Standard Q8/F16/F32 conversion does not require calibration.
