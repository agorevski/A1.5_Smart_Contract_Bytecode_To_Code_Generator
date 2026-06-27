# Model Details

## Base Model

**Qwen2.5-Coder-7B-Instruct** (`Qwen/Qwen2.5-Coder-7B-Instruct`)

| Property | Value |
|----------|-------|
| Parameters | 7.6B |
| Context window | 128K tokens |
| Primary role | Code-specialized instruction model for TAC-to-Solidity generation |

## LoRA Fine-Tuning

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha (α) | 32 |
| Dropout | 0.1 |
| Target modules | q/k/v/o and MLP gate/up/down projections |
| Trainable params | ~40M with the default Qwen 7B target modules |

LoRA decomposes weight updates as: $W' = W + \frac{BA}{\alpha}$ where $B \in \mathbb{R}^{r \times d}$, $A \in \mathbb{R}^{d \times r}$.

## Quantization

Training defaults to full-precision LoRA (`use_quantization=False`) to avoid quantization-related quality changes on the 4-GPU default path. Use `--quantization` to enable 4-bit NF4 loading via `bitsandbytes` when VRAM is constrained, or `--no-quantization` to make the choice explicit in manifests.

Precision is controlled with `--precision {auto,bf16,fp16,fp32}`. `auto` selects BF16 on Ampere+ CUDA devices, FP16 on older CUDA devices, and FP32 on CPU.

8-bit loading is not currently exposed as a separate training mode; update `ModelConfig` before documenting or relying on 8-bit quantization.

## Training Defaults

| Parameter | Value |
|-----------|-------|
| Batch size | 4 per device |
| Gradient accumulation | Auto from target global batch |
| Effective batch size | 16 |
| Learning rate | 2e-4 (cosine schedule) |
| Optimizer | AdamW |
| Epochs | 3 |
| Max sequence length | 2,048 tokens |
| GPUs | 4 by default via automatic `torchrun` relaunch |
| Tokenization cache | Enabled by default for repeat runs |
| Trainer validation | `--train-eval-strategy auto` (`steps` for large splits, `epoch` for small splits) |

## Inference

| Parameter | Default |
|-----------|---------|
| Temperature | 0.1 |
| Sampling | Greedy by default |
| Max new tokens | 1,024 |
| Repetition penalty | 1.15 |

## Target Metrics

| Metric | Target |
|--------|--------|
| Semantic similarity (avg) | > 0.82 |
| Functions > 0.8 similarity | > 78% |
| Normalized edit distance < 0.4 | > 82% |
| Structured replication precision/recall/F1 | Track by category; higher is better |

## Hardware Requirements

| Operation | Min VRAM |
|-----------|----------|
| Inference (4-bit) | 4 GB |
| Training (full LoRA) | 24 GB+ recommended |
| Training (4-bit + LoRA) | 16 GB |