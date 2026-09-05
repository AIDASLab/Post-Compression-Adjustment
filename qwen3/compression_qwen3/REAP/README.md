# Qwen3 REAP compression

This directory preserves the REAP pruning code and launch structure used for
`Qwen/Qwen3-30B-A3B-Instruct-2507`. The implementation is derived from the
[Cerebras REAP repository](https://github.com/CerebrasResearch/reap). See
`LICENSE` for the retained Apache-2.0 license.

## Environment

This release contains the adapted Qwen3 entry point and the upstream REAP
modules in its transitive runtime dependency closure. Optional upstream
evaluation suites, third-party subprojects, lock files, and packaging metadata
are not vendored. The retained upstream environment metadata uses Python 3.12,
PyTorch 2.7.1, and Transformers 4.55.0. A pruning-only environment can be
prepared without installing the omitted evaluation subprojects:

```bash
ENV_NAME=beyond-moe-reap
conda create -n "$ENV_NAME" python=3.12 -y
conda activate "$ENV_NAME"
# Install the CUDA-enabled torch==2.7.1 build appropriate for the host first.
python -m pip install "transformers==4.55.0" \
  "accelerate>=1.7.0" "datasets>=3.6.0,<4.0.0" \
  numpy scipy scikit-learn tqdm pyyaml python-dotenv
```

These packages cover the imports reached by `src/reap/prune.py`; the full
[upstream REAP setup](https://github.com/CerebrasResearch/reap) additionally
installs evaluation and development tools that are not needed for these
compression launchers. The launchers add this directory's `src/` tree to
`PYTHONPATH`, and the compression entry point is `src/reap/prune.py`.

By default, the launchers use the public first English C4 shard through the
exact URL embedded in `src/reap/data.py`. Set `REAP_LOCAL_FILES_ONLY=true` only
when all required Hugging Face model files are already cached.

## Compression runs

Run one ratio or all ratios from this directory:

```bash
bash run_qwen3_reap_c4_keep50.sh
bash run_qwen3_reap_c4_all.sh
```

| Retention | Pruning ratio | Calibration batches | Output suffix |
|---:|---:|---:|---|
| 50% | 0.5 | 1024 | `reap-renorm_true-seed_42-0.5` |
| 62.5% | 0.375 | 1024 | `reap-renorm_true-seed_42-0.375` |
| 75% | 0.25 | 1024 | `reap-renorm_true-seed_42-0.25` |

All runs use batch size 1, cosine distance, seed 42, router-weight
renormalization, pruning-metric-only observation, and the author-confirmed
1,024 C4 calibration batches.

Artifacts keep the original hierarchy under
`artifacts/Qwen3-30B-A3B-Instruct-2507/c4/`, and each launcher invokes its
corresponding `test_pruned_qwen_*.py` validator. Model weights and observations
are generated at runtime and are not included in this repository.

`CUDA_VISIBLE_DEVICES` defaults to `7` and can be overridden:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_qwen3_reap_c4_keep75.sh
```

## Citation

```bibtex
@inproceedings{lasby2026reap,
  title={{REAP} the Experts: Why Pruning Prevails for One-Shot MoE Compression},
  author={Mike Lasby and Ivan Lazarevich and Nish Sinnadurai and Sean Lie and Yani Ioannou and Vithursan Thangarasa},
  booktitle={International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=ukGxWd2aDG}
}
```
