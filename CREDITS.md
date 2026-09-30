# Credits

## Project

- **Kenji8965** — project direction, target definition, human testing, hardware validation, performance steering, release acceptance, and repository maintenance.
- **ChatGPT 5.6 Sol (OpenAI)** — programming and implementation assistance for the 0.4.2 development cycle, including code changes, refactoring, testing infrastructure, backend optimization, release cleanup, documentation, and packaging support.

Copyright in this project remains with **Kenji8965**. AI-assisted programming does not transfer project copyright to OpenAI or ChatGPT.

The software is provided under the MIT License **“AS IS”**, without warranty of any kind. Use of the software and its outputs is at your own risk. See [LICENSE](LICENSE) for the complete license terms.

## MoGe research

This project is an independent native inference implementation for the **MoGe** family of monocular geometry models. The original MoGe research and official model ecosystem are by the Microsoft Research / MoGe authors. Please cite the corresponding MoGe paper(s) when using the models in research.

- MoGe: Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, Jiaolong Yang — CVPR 2025.
- MoGe-2: Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, Jiaolong Yang — 2025.
- Official project: https://github.com/microsoft/MoGe

The official MoGe repository states that its code is MIT-licensed, with DINOv2 portions under Apache-2.0. moge-ggml does not redistribute the official Python source tree or model checkpoints in this release.

## Upstream and third-party projects

moge-ggml depends on substantial upstream work, especially **ggml** and **stb**. Their authors and contributors retain their respective copyrights and licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the license texts retained in the vendored source tree.

The MoGe model architecture and original research/checkpoints are separate from this native inference implementation. Users are responsible for complying with the terms that apply to any model weights they obtain or convert; model checkpoints are not included in this source release.
