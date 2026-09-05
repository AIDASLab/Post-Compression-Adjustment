# Additional Experiments: Behavioral Proxies

This directory contains the additional behavioral-proxy evaluation pipelines
reported in the paper for Qwen3 and Gemma4. It includes only the six reported
proxy tasks. Generated results, logs, caches, checkpoints, and model weights
are deliberately omitted.

## Scope

The camera-ready paper reports six behavioral proxies:

- `ifeval`
- `truthfulqa_mc1`
- `toxigen`
- `wmdp`
- `crows_pairs_english`
- `winogender` (evaluated with the `winogender_all` harness task)

These metrics have different meanings and directions and are reported
separately. They must not be combined into a scalar safety score. The paper's
aggregate for each proxy is the arithmetic mean over the 12 combinations of
backbone, compression method, and retention ratio.

The retained compression methods are:

| Backbone | Methods |
| --- | --- |
| Qwen3 | `HC-SMoE`, `REAP` |
| Gemma4 | `AIMER`, `MC-SMoE` |

Each method covers 50%, 62.5%, and 75% expert retention. Evaluation runners
cover the original backbone, compressed-only checkpoints, and the 13
paper-reported adjustment strategies.

## Layout

```text
qwen3/
  compression_qwen3/original_compression_qwen3/
  PCA_c4/All_strategy_qwen3_c4/lm-harness_results_qwen3/
gemma4/
  compression_gemma4/original_compression_gemma4/
  PCA_c4/All_strategy_gemma4_c4/lm-harness_results_gemma4/
```

The `compression_*` branches evaluate original and compressed-only models. The
`PCA_c4/All_strategy_*` branches evaluate adjusted models.

## Output policy

No benchmark result, run metadata, log, telemetry, cache, generated table, or
model weight is distributed in this repository. The evaluation runners create
`result.json` and `run_info.json` under `results_*` directories at runtime; the
retained collectors can consume those locally generated files. All such output
paths are ignored by Git.

## Re-running evaluations

Obtain the lm-evaluation-harness checkout described in
[`../common/README.md`](../common/README.md). Point each runner to external
checkpoints because model weights are not included.

Run the following commands from the repository root. First enter this
directory so that the `qwen3/` and `gemma4/` paths below resolve to the
additional-evaluation runners rather than the main benchmark runners:

```bash
cd additional_experiments
```

For original and compressed-only Qwen3 models:

```bash
HARNESS_DIR=/path/to/lm-evaluation-harness \
COMPRESSED_MODELS_ROOT=/path/to/qwen3/compression_qwen3 \
bash qwen3/compression_qwen3/original_compression_qwen3/run_eval_method.sh \
  original HC-SMoE REAP --dry-run
```

For adjusted Qwen3 models, `MODEL_ROOT` contains the `HC-SMoE/` and `REAP/`
directories with the original `ratio_*/<strategy>/` layout:

```bash
HARNESS_DIR=/path/to/lm-evaluation-harness \
MODEL_ROOT=/path/to/All_strategy_qwen3_c4 \
bash qwen3/PCA_c4/All_strategy_qwen3_c4/lm-harness_results_qwen3/run_eval_method.sh \
  HC-SMoE --dry-run
```

For original and compressed-only Gemma4 models:

```bash
HARNESS_DIR=/path/to/lm-evaluation-harness \
COMPRESSED_MODELS_ROOT=/path/to/gemma4/compression_gemma4 \
bash gemma4/compression_gemma4/original_compression_gemma4/run_eval_method.sh \
  original AIMER MC-SMoE --dry-run
```

For adjusted Gemma4 models:

```bash
HARNESS_DIR=/path/to/lm-evaluation-harness \
MODEL_ROOT=/path/to/All_strategy_gemma4_c4 \
bash gemma4/PCA_c4/All_strategy_gemma4_c4/lm-harness_results_gemma4/run_eval_method.sh \
  AIMER MC-SMoE --dry-run
```

The examples use `--dry-run` to print and validate the planned jobs without
launching them. Remove `--dry-run` to run the evaluations. Passing
`--categories safety_alignment` is only a convenience alias that launches all
six proxy tasks; it does not compute or report a combined safety score.
