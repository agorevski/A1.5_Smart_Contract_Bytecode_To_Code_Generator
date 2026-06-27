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

The code defaults to 4-bit NF4 loading via `bitsandbytes` (`use_quantization=True`, `load_in_4bit=True`). This keeps the base model small enough for LoRA fine-tuning on a single CUDA GPU while preserving the normal Hugging Face/PEFT training flow.

8-bit loading is not currently exposed as a separate training mode; update `ModelConfig` before documenting or relying on 8-bit quantization.

## Training Defaults

| Parameter | Value |
|-----------|-------|
| Batch size | 4 per device |
| Gradient accumulation | 4 steps |
| Effective batch size | 16 |
| Learning rate | 2e-4 (cosine schedule) |
| Optimizer | AdamW |
| Epochs | 3 |
| Max sequence length | 2,048 tokens |

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
| Training (4-bit + LoRA) | 16 GB |