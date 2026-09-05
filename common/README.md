# Shared Data and Evaluation Harness

Large public datasets and the evaluator checkout are deliberately not copied
into this repository. This directory records the exact resources used by the
experiments and how the original launchers expect to find them.

## C4 calibration shard

All experiments described as using C4 use the first English training shard:

- source: [allenai/c4, `en/c4-train.00000-of-01024.json.gz`](https://huggingface.co/datasets/allenai/c4/blob/main/en/c4-train.00000-of-01024.json.gz)
- local uncompressed filename: `c4-train.00000-of-01024.json`
- format: one JSON object per line; the adjustment loaders read the `text` field

Download and decompress it outside the repository, then export the path before
running a C4 launcher:

```bash
export C4_DATASET_PATH=/absolute/path/to/c4-train.00000-of-01024.json
```

The main adjustment setting consumes the first 3,000 non-empty texts. The
sample-count robustness setting uses the same code and file with
`MAX_CALIB_SAMPLES=1024`.

Qwen3 REAP refers to the same public shard through the `allenai/c4` dataset
identifier. Its dataset loader resolves the exact shard URL above.

## OpenR1-Math-220k

The math-domain robustness experiment loads
[`open-r1/OpenR1-Math-220k`](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k),
configuration `all`, split `train`, through the Hugging Face `datasets`
library. It takes the first 1,024 examples. This setting is used only for
Qwen3 HC-SMoE and Gemma 4 AIMER, matching the paper.

## lm-evaluation-harness

Benchmarks were run with
[`EleutherAI/lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
version `0.4.13.dev0`.

Install a separate checkout outside this repository. Conda environment names
are local choices and are not part of the experiment configuration. The
following public-facing names are examples and may be changed. Create a Qwen3
environment as follows:

```bash
ENV_NAME=beyond-moe-qwen3
conda create -n "$ENV_NAME" python=3.12 -y
conda activate "$ENV_NAME"

# Install the CUDA-enabled PyTorch build appropriate for the host first.
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
python -m pip install "transformers==5.8.1" "vllm==0.19.1"
python -m pip install -e ".[vllm,ifeval,math,sentencepiece]"
export HARNESS_DIR="$PWD"
```

Create the separate Gemma4 environment with the same harness version:

```bash
ENV_NAME=beyond-moe-gemma4
conda create -n "$ENV_NAME" python=3.12 -y
conda activate "$ENV_NAME"

# Install the CUDA-enabled PyTorch build appropriate for the host first.
cd /path/to/lm-evaluation-harness
python -m pip install "transformers==5.8.1" "vllm==0.19.1"
python -m pip install -e ".[vllm,ifeval,math,sentencepiece]"
export HARNESS_DIR="$PWD"
```

Separate environments allow backbone-specific dependencies while using the
same evaluator version and required optional dependencies. The `ifeval` and
`math` extras provide the task-specific dependencies used by the paper suite,
`sentencepiece` supports the model tokenizers, and `vllm` provides the
evaluation backend.

The archived evaluation environment was:

| Component | Version |
| --- | --- |
| Python | 3.12.12 |
| PyTorch | 2.10.0+cu128 |
| Transformers | 5.8.1 |
| vLLM | 0.19.1 |
| lm-evaluation-harness | 0.4.13.dev0 |

CUDA-enabled PyTorch wheels are platform-specific, so install the appropriate
build for the host driver rather than treating the recorded CUDA build as a
portable package constraint. Compression methods may require older,
method-specific Transformers environments; follow the README and requirements
file in the corresponding compression directory.

The evaluation launchers use vLLM with BF16, tensor parallel size 1, maximum
model length 4,096, automatic batch sizing, chat templates, multi-turn few-shot
formatting, sample logging, and seed 42. GPU-memory utilization is 0.8 for MCQA
and 0.9 for other task groups. Gemma 4 disables thinking for all categories;
Qwen3 disables it for AIME. AIME uses at most 2,780 generated tokens with greedy
decoding.

Set caches outside the Git checkout when running the experiments, for example:

```bash
export HF_HOME=/scratch/$USER/huggingface
export XDG_CACHE_HOME=/scratch/$USER/cache
```
