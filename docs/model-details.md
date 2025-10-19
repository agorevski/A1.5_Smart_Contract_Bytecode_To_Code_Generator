# Model Configuration and Details

This document provides in-depth information about the model architecture, configuration, and training parameters.

## Base Model: Llama 3.2 3B

### Architecture Overview

| Component | Specification |
|-----------|--------------|
| **Parameters** | 3.21 billion |
| **Architecture** | Transformer decoder |
| **Layers** | 32 transformer blocks |
| **Hidden Size** | 3,072 |
| **Attention Heads** | 24 |
| **Intermediate Size** | 8,192 |
| **Context Window** | 20,000 tokens |
| **Vocabulary Size** | 128,256 tokens |

### Model Capabilities

- **Code Understanding**: Pretrained on large code corpus
- **Natural Language**: Strong reasoning and explanation
- **Context Length**: 20,000 tokens handles complex functions
- **Multilingual**: Supports multiple programming languages

## LoRA Configuration

Low-Rank Adaptation (LoRA) enables efficient fine-tuning with minimal parameters.

### Standard Configuration (Paper Specification)

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,                          # Rank
    lora_alpha=32,                 # Alpha (2 * rank)
    target_modules=[
        "q_proj",                  # Query projection
        "k_proj",                  # Key projection
        "v_proj",                  # Value projection
        "o_proj"                   # Output projection
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    modules_to_save=None
)
```

### Parameter Breakdown

| Parameter | Value | Description |
|-----------|-------|-------------|
| **r** (rank) | 16 | Decomposition rank |
| **lora_alpha** | 32 | Scaling factor (typically 2*r) |
| **target_modules** | q/k/v/o_proj | Attention layers to adapt |
| **lora_dropout** | 0.05 | Dropout for regularization |
| **trainable_params** | ~30M | Only 0.9% of base model |

### LoRA Mathematics

LoRA decomposes weight updates as low-rank matrices:

```text
W' = W + BA/α
```

Where:

- W: Original weights (frozen)
- B: Low-rank matrix (r × d)
- A: Low-rank matrix (d × r)
- α: Scaling factor
- r: Rank (much smaller than d)

**Memory Savings**:

- Original: 3.21B parameters
- LoRA: ~30M trainable parameters (~99% reduction)

## Quantization

4-bit quantization reduces memory footprint for training and inference.

### BitsAndBytes Configuration

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True  # Nested quantization
)
```

### Memory Impact

| Configuration | GPU Memory | Relative |
|---------------|------------|----------|
| **FP32 (full precision)** | ~12GB | 100% |
| **FP16 (half precision)** | ~6GB | 50% |
| **INT8 (8-bit)** | ~3.5GB | 29% |
| **NF4 (4-bit)** | ~2GB | 17% |

## Training Hyperparameters

### Learning Rate Schedule

```python
from transformers import get_linear_schedule_with_warmup

# Configuration
learning_rate = 2e-4
warmup_steps = 500
total_steps = 75000  # ~3 epochs on 238k samples

# Schedule
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

**Schedule Phases**:

1. **Warmup** (0-500 steps): Linear increase from 0 to 2e-4
2. **Linear Decay** (500-75000): Linear decrease to 0

### Optimizer Configuration

```python
from torch.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=2e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)
```

### Batch Configuration

```python
training_args = TrainingArguments(
    per_device_train_batch_size=4,        # Per GPU
    gradient_accumulation_steps=4,         # Accumulate 4 batches
    # Effective batch size = 4 * 4 = 16
    
    per_device_eval_batch_size=4,
    eval_accumulation_steps=4
)
```

### Complete Training Arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # Output
    output_dir="models/checkpoints",
    logging_dir="output/logs",
    
    # Batch configuration
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    
    # Learning rate
    learning_rate=2e-4,
    warmup_steps=500,
    lr_scheduler_type="linear",
    
    # Optimization
    optim="adamw_torch",
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    # Training duration
    num_train_epochs=3,
    max_steps=-1,
    
    # Evaluation
    evaluation_strategy="steps",
    eval_steps=500,
    
    # Checkpointing
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # Logging
    logging_steps=100,
    report_to="tensorboard",
    
    # Memory optimization
    gradient_checkpointing=True,
    fp16=False,
    bf16=True,
    
    # System
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    
    # Reproducibility
    seed=42
)
```

## Generation Configuration

### Inference Parameters

```python
from transformers import GenerationConfig

generation_config = GenerationConfig(
    max_new_tokens=2048,           # Maximum output length
    min_new_tokens=10,             # Minimum output length
    
    # Sampling strategy
    do_sample=True,
    temperature=0.3,               # Low for deterministic
    top_p=0.9,                     # Nucleus sampling
    top_k=50,                      # Top-k sampling
    
    # Repetition control
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
    
    # Special tokens
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    
    # Beam search (optional)
    num_beams=1,                   # Greedy by default
    early_stopping=False,
    
    # Return options
    num_return_sequences=1,
    output_scores=False,
    return_dict_in_generate=True
)
```

### Temperature Effects

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| **0.1-0.3** | Very deterministic | Production decompilation |
| **0.4-0.6** | Balanced | General use |
| **0.7-0.9** | Creative | Exploratory analysis |
| **1.0+** | Highly random | Not recommended |

### Sampling Strategies

#### Greedy Decoding

```python
generation_config = GenerationConfig(
    do_sample=False,
    num_beams=1
)
```

#### Top-p (Nucleus) Sampling

```python
generation_config = GenerationConfig(
    do_sample=True,
    top_p=0.9,
    top_k=0  # Disable top-k
)
```

#### Top-k Sampling

```python
generation_config = GenerationConfig(
    do_sample=True,
    top_k=50,
    top_p=1.0  # Disable top-p
)
```

#### Beam Search

```python
generation_config = GenerationConfig(
    do_sample=False,
    num_beams=5,
    early_stopping=True
)
```

## Model Loading

### Standard Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "models/final/smart_contract_decompiler",
    is_trainable=False
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B"
)
```

### Merged Model Loading

```python
# Merge LoRA weights with base model for faster inference
model = model.merge_and_unload()

# Save merged model
model.save_pretrained("models/merged/smart_contract_decompiler")
tokenizer.save_pretrained("models/merged/smart_contract_decompiler")
```

## Performance Characteristics

### Inference Speed

| Hardware | Tokens/Second | Function Time |
|----------|---------------|---------------|
| **A100 (40GB)** | ~50-80 | 1-2s |
| **RTX 4090** | ~30-50 | 2-3s |
| **RTX 3090** | ~20-30 | 3-4s |
| **CPU (32 cores)** | ~5-10 | 10-20s |

### Memory Requirements

| Operation | 4-bit | 8-bit | FP16 |
|-----------|-------|-------|------|
| **Training** | 8-12GB | 14-18GB | 24-32GB |
| **Inference** | 4-6GB | 6-8GB | 10-12GB |
| **Batch=8** | 12-16GB | 20-24GB | 40GB+ |

## Model Variants

### Alternative LoRA Configurations

#### High Capacity

```python
# More parameters for complex patterns
lora_config = LoraConfig(
    r=32,                # Higher rank
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"]  # More modules
)
```

#### Memory Efficient

```python
# Fewer parameters for limited memory
lora_config = LoraConfig(
    r=8,                 # Lower rank
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"]  # Fewer modules
)
```

### Alternative Base Models

| Model | Size | Context | Notes |
|-------|------|---------|-------|
| **Llama 3.2 1B** | 1.2B | 20K | Faster, less accurate |
| **Llama 3.2 3B** | 3.2B | 20K | **Recommended** |
| **Llama 3.1 8B** | 8B | 128K | Higher quality, slower |
| **CodeLlama 7B** | 7B | 16K | Code-specialized |

## Model Evaluation Metrics

### Training Metrics

- **Training Loss**: Target <0.5 by end
- **Validation Loss**: Should track training loss
- **Perplexity**: Lower is better
- **Learning Rate**: Follows schedule

### Quality Metrics

- **Semantic Similarity**: Target 0.82 avg
- **Edit Distance**: Target <0.4 for 82.5%
- **BLEU Score**: Measures n-gram overlap
- **Structural Fidelity**: Code structure preservation

## Next Steps

- Review [Training Pipeline](training-pipeline.md) for training process
- Check [Evaluation](evaluation.md) for metrics details
- See [Usage Guide](usage.md) for inference examples
- Explore [Troubleshooting](troubleshooting.md) for common issues
