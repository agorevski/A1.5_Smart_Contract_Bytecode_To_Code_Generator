# Model Details

## Base Model

**Llama 3.2 3B** (`meta-llama/Llama-3.2-3B`)

| Property | Value |
|----------|-------|
| Parameters | 3.21B |
| Layers | 32 transformer blocks |
| Hidden size | 3,072 |
| Attention heads | 24 |
| Context window | 20,000 tokens |
| Vocabulary | 128,256 tokens |

## LoRA Fine-Tuning

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha (α) | 32 |
| Dropout | 0.1 |
| Target modules | All linear layers |
| Trainable params | ~30M (0.9% of base) |

LoRA decomposes weight updates as: $W' = W + \frac{BA}{\alpha}$ where $B \in \mathbb{R}^{r \times d}$, $A \in \mathbb{R}^{d \times r}$.

## Quantization

8-bit loading via `bitsandbytes` reduces GPU memory from ~12 GB (FP32) to ~3.5 GB.

## Training Defaults

| Parameter | Value |
|-----------|-------|
| Batch size | 4 per device |
| Gradient accumulation | 8 steps |
| Effective batch size | 32 |
| Learning rate | 2e-4 (cosine schedule) |
| Optimizer | AdamW |
| Epochs | 3 |
| Max sequence length | 4,096 tokens |

## Inference

| Parameter | Default |
|-----------|---------|
| Temperature | 0.3 |
| Top-p | 0.9 |
| Max new tokens | 512 |
| Repetition penalty | 1.1 |

## Target Metrics

| Metric | Target |
|--------|--------|
| Semantic similarity (avg) | > 0.82 |
| Functions > 0.8 similarity | > 78% |
| Normalized edit distance < 0.4 | > 82% |

## Hardware Requirements

| Operation | Min VRAM |
|-----------|----------|
| Inference (8-bit) | 4 GB |
| Training (8-bit + LoRA) | 16 GB |