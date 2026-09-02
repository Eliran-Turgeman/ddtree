<h1 align="center">DDTree</h1>

<p align="center">
  Official implementation of <strong>DDTree (Diffusion Draft Tree)</strong> from
  <em>Accelerating Speculative Decoding with Block Diffusion Draft Trees</em>.
</p>

<p align="center">
  Liran Ringel, Yaniv Romano
</p>

<p align="center">
  <a href="https://liranringel.github.io/ddtree/">🌐 Project Page</a>
  &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2604.12989">📄 Paper</a>
</p>

## Setup

This codebase is intended for a CUDA-enabled PyTorch environment.

```bash
pip install -r requirements.txt
```

## Run Experiments

```bash
bash run_benchmark.sh
```

This produces benchmark outputs in `runs/` and logs in `logs/`.

To run a limited GSM8K benchmark on one Lambda Cloud GPU:

```bash
bash run_benchmark.sh \
  --gpus 0 \
  --task gsm8k:32 \
  --model-draft-pair 'Qwen/Qwen3-4B|z-lab/Qwen3-4B-DFlash-b16' \
  --temperature 0.0 \
  --mode sdpa
```

This runs the baseline, DFlash, and DDTree methods for 32 examples using one
worker. Use `bash run_benchmark.sh --help` to see all sweep parameters. The
model weights, dataset, and FlashAttention dependency are downloaded on the
first run, so the Lambda instance needs Hugging Face access and enough local
storage for both models.

## Reproduce Paper Artifacts

Generate the plots:

```bash
python3 plot_results.py
```

Generate the LaTeX table:

```bash
python3 make_latex_table.py
```

## Citation

```bibtex
@article{ringel2026ddtree,
  title={Accelerating Speculative Decoding with Block Diffusion Draft Trees},
  author={Ringel, Liran and Romano, Yaniv},
  journal={arXiv preprint arXiv:2604.12989},
  year={2026}
}
```
