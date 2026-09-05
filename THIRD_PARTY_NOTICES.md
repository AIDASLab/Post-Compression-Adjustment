# Third-Party Notices

This repository contains selected code derived from existing MoE compression
implementations. Copyright and license notices in the method directories are
retained and govern those files.

Except where otherwise noted, the authors' original repository code is
licensed under the top-level Apache License 2.0. That top-level license does not
replace the licenses or copyright notices attached to third-party components.

| Included directory | Origin represented by the archived code | Retained license |
| --- | --- | --- |
| `qwen3/compression_qwen3/REAP` | [REAP](https://github.com/CerebrasResearch/reap) expert-pruning implementation | Apache License 2.0 |
| `gemma4/compression_gemma4/AIMER` | [AIMER](https://github.com/ZongfangLiu/AIMER) adaptation; upstream AIMER builds on [REAP](https://github.com/CerebrasResearch/reap) | Apache License 2.0 |
| `qwen3/compression_qwen3/HC-SMoE` | [HC-SMoE](https://github.com/wazenmai/HC-SMoE) implementation; upstream HC-SMoE is based on the [MC-SMoE](https://github.com/UNITES-Lab/MC-SMoE) codebase | MIT License |
| `gemma4/compression_gemma4/MC-SMoE` | M-SMoE expert-merging implementation adapted from [MC-SMoE](https://github.com/UNITES-Lab/MC-SMoE) | MIT License |

External resources are not redistributed here:

- [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [allenai/c4](https://huggingface.co/datasets/allenai/c4)
- [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
- Qwen3-30B-A3B-Instruct-2507 model files
- gemma-4-26B-A4B-it model files
