# Training Pipeline

This document provides comprehensive guidance on training the Smart Contract Decompilation model.

## Overview

The training pipeline consists of three main phases:

1. **Data Collection**: Gathering verified contracts and creating TAC-Solidity pairs
2. **Model Training**: Fine-tuning Llama 3.2 3B with LoRA
3. **Evaluation**: Assessing model performance with multiple metrics

## Phase 1: Data Collection

### Target Dataset

Following the research paper specifications:

- **Total Pairs**: 238,446 TAC-to-Solidity function pairs
- **Training Split**: 85% (~202,679 pairs)
- **Validation Split**: 10% (~23,845 pairs)
- **Test Split**: 5% (~9,731 pairs)

### Collection Process

```python
from src.dataset_pipeline import DatasetBuilder

# Initialize builder
builder = DatasetBuilder(
    etherscan_api_key="your_etherscan_api_key",
    output_dir="data/raw"
)

# Load contract addresses
with open('verified_contracts.txt', 'r') as f:
    contract_addresses = [line.strip() for line in f]

print(f"Collecting {len(contract_addresses)} contracts...")

# Collect contracts (with parallel processing)
collected = builder.collect_contracts(
    contract_addresses,
    max_workers=10,          # Parallel threads
    rate_limit=5.0,          # Calls per second
    batch_size=100           # Contracts per batch
)

print(f"Successfully collected: {len(collected)} contracts")
```

### Creating Function Pairs

```python
# Process contracts to create TAC-Solidity pairs
print("Creating TAC-to-Solidity pairs...")
pairs = builder.process_contracts_to_function_pairs(
    batch_size=100,
    include_metadata=True
)

print(f"Generated {len(pairs)} function pairs")
```

### Quality Filtering

```python
# Apply quality filters per paper specifications
print("Filtering dataset...")
filtered = builder.filter_and_clean_dataset(
    min_length=50,           # Minimum tokens
    max_length=20000,        # Maximum tokens (paper spec)
    remove_duplicates=True,
    validate_syntax=True
)

print(f"Filtered dataset size: {len(filtered)} pairs")
```

### Export Dataset

```python
# Export in JSONL format for training
dataset_path = builder.export_dataset(
    output_format="jsonl",
    output_file="data/processed/smart_contract_dataset.jsonl"
)

print(f"Dataset exported to: {dataset_path}")

# View statistics
stats = builder.get_dataset_statistics()
print(f"\nDataset Statistics:")
print(f"  Total function pairs: {stats['total_function_pairs']}")
print(f"  Average length: {stats['avg_length']:.1f} tokens")
print(f"  Visibility distribution: {stats['visibility_distribution']}")
print(f"  Complexity range: {stats['min_complexity']:.1f} - {stats['max_complexity']:.1f}")
```

## Phase 2: Model Training

### Training Configuration

```python
from src.training_pipeline import TrainingConfig, SmartContractTrainingPipeline

# Configure training (matching paper specifications)
config = TrainingConfig(
    # Data configuration
    dataset_path="data/processed/smart_contract_dataset.jsonl",
    train_split=0.85,
    val_split=0.10,
    test_split=0.05,
    
    # Model configuration
    model_name="meta-llama/Llama-3.2-3B",
    lora_rank=16,              # Paper specification
    lora_alpha=32,             # Alpha = 2 * rank
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    
    # Training configuration
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_epochs=3,
    warmup_steps=500,
    max_grad_norm=1.0,
    
    # Optimization
    optimizer="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    
    # System configuration
    output_dir="models/checkpoints",
    logging_dir="output/logs",
    use_4bit_quantization=True,
    use_gradient_checkpointing=True,
    fp16=False,
    bf16=True,  # Use bfloat16 if available
    
    # Evaluation and checkpointing
    eval_steps=500,
    save_steps=1000,
    save_total_limit=3,
    logging_steps=100
)
```

### Run Complete Pipeline

```python
# Initialize pipeline
pipeline = SmartContractTrainingPipeline(config)

# Option 1: Run complete pipeline
print("Starting complete training pipeline...")
results = pipeline.run_complete_pipeline()

print(f"\nTraining Complete!")
print(f"  Final model: {results['model_path']}")
print(f"  Avg semantic similarity: {results['avg_semantic_similarity']:.3f}")
print(f"  Avg edit distance: {results['avg_edit_distance']:.3f}")
```

### Step-by-Step Training

```python
# Option 2: Run step-by-step for more control
pipeline = SmartContractTrainingPipeline(config)

# Step 1: Prepare dataset
print("Preparing dataset...")
train_path, val_path, test_path = pipeline.collect_and_prepare_dataset()
print(f"  Train: {train_path}")
print(f"  Val: {val_path}")
print(f"  Test: {test_path}")

# Step 2: Train model
print("\nTraining model...")
model_path = pipeline.train_model(train_path, val_path)
print(f"  Model saved to: {model_path}")

# Step 3: Evaluate
print("\nEvaluating model...")
evaluation_results = pipeline.evaluate_model(model_path, test_path)

print(f"\nEvaluation Results:")
print(f"  Semantic Similarity: {evaluation_results['avg_semantic_similarity']:.3f}")
print(f"  Edit Distance: {evaluation_results['avg_edit_distance']:.3f}")
print(f"  BLEU Score: {evaluation_results['avg_bleu_score']:.3f}")
print(f"  Functions > 0.8 similarity: {evaluation_results['high_similarity_pct']:.1f}%")
print(f"  Functions < 0.4 edit distance: {evaluation_results['low_edit_distance_pct']:.1f}%")
```

### Custom Training Loop

For advanced users who need more control:

```python
from transformers import Trainer, TrainingArguments
from src.model_setup import setup_model_for_training, prepare_dataset

# Setup model
model, tokenizer = setup_model_for_training(
    model_name="meta-llama/Llama-3.2-3B",
    lora_config=config.lora_config,
    use_4bit=True
)

# Prepare dataset
train_dataset = prepare_dataset(train_path, tokenizer)
val_dataset = prepare_dataset(val_path, tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir=config.output_dir,
    per_device_train_batch_size=config.batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    learning_rate=config.learning_rate,
    num_train_epochs=config.num_epochs,
    warmup_steps=config.warmup_steps,
    logging_steps=config.logging_steps,
    eval_steps=config.eval_steps,
    save_steps=config.save_steps,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    gradient_checkpointing=True,
    report_to="tensorboard"
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Train
print("Starting training...")
trainer.train()

# Save final model
trainer.save_model("models/final/smart_contract_decompiler")
print("Training complete!")
```

## Phase 3: Evaluation

### Comprehensive Evaluation

```python
from src.training_pipeline import SmartContractEvaluator

evaluator = SmartContractEvaluator()

# Load test dataset
test_data = evaluator.load_test_data(test_path)

# Evaluate model
results = evaluator.evaluate_model(
    model_path="models/final/smart_contract_decompiler",
    test_data=test_data,
    batch_size=8
)

# Print detailed results
evaluator.print_detailed_results(results)
```

### Per-Function Evaluation

```python
# Evaluate single function
original_solidity = """
function transfer(address to, uint256 amount) public returns (bool) {
    require(to != address(0), "Invalid address");
    require(_balances[msg.sender] >= amount, "Insufficient balance");
    _balances[msg.sender] -= amount;
    _balances[to] += amount;
    emit Transfer(msg.sender, to, amount);
    return true;
}
"""

decompiled_solidity = decompiler.decompile_tac_to_solidity(tac)

metrics = evaluator.evaluate_function(original_solidity, decompiled_solidity)

print(f"Metrics for function:")
print(f"  Semantic Similarity: {metrics.semantic_similarity:.3f}")
print(f"  Edit Distance: {metrics.edit_distance:.3f}")
print(f"  BLEU Score: {metrics.bleu_score:.3f}")
print(f"  ROUGE-L: {metrics.rouge_l:.3f}")
print(f"  Structural Preservation: {metrics.structural_preservation:.3f}")
```

### Batch Evaluation

```python
# Evaluate multiple functions
results = []
for original, decompiled in zip(original_functions, decompiled_functions):
    metrics = evaluator.evaluate_function(original, decompiled)
    results.append(metrics)

# Aggregate statistics
avg_semantic_sim = sum(m.semantic_similarity for m in results) / len(results)
avg_edit_dist = sum(m.edit_distance for m in results) / len(results)

print(f"\nAggregate Results ({len(results)} functions):")
print(f"  Average Semantic Similarity: {avg_semantic_sim:.3f}")
print(f"  Average Edit Distance: {avg_edit_dist:.3f}")
```

## Training Progress Monitoring

### TensorBoard

Monitor training progress in real-time:

```bash
# Start TensorBoard
tensorboard --logdir=output/logs

# Open browser to http://localhost:6006
```

### Training Metrics

Key metrics to monitor:

- **Training Loss**: Should decrease steadily
- **Validation Loss**: Should decrease without overfitting
- **Learning Rate**: Follows warmup schedule
- **GPU Memory**: Should remain stable

### Checkpointing

The system automatically saves checkpoints:

- **Frequency**: Every 1,000 steps (configurable)
- **Location**: `models/checkpoints/checkpoint-{step}/`
- **Contents**: Model weights, optimizer state, training progress

Resume from checkpoint:

```python
# Resume training from specific checkpoint
config.resume_from_checkpoint = "models/checkpoints/checkpoint-5000"
pipeline = SmartContractTrainingPipeline(config)
results = pipeline.run_complete_pipeline()
```

## Performance Optimization

### Memory Optimization

```python
# Enable memory-saving features
config = TrainingConfig(
    # ... other config ...
    use_4bit_quantization=True,      # Reduces memory by ~75%
    use_gradient_checkpointing=True, # Trades compute for memory
    gradient_accumulation_steps=8,   # Effective larger batch
    per_device_train_batch_size=2    # Smaller per-device batch
)
```

### Speed Optimization

```python
# Enable speed optimizations
config = TrainingConfig(
    # ... other config ...
    bf16=True,                        # Faster than fp32
    dataloader_num_workers=4,         # Parallel data loading
    dataloader_pin_memory=True,       # Faster GPU transfer
    optim="adamw_torch_fused"         # Fused optimizer (PyTorch 2.0+)
)
```

### Multi-GPU Training

```python
# Distributed training on multiple GPUs
training_args = TrainingArguments(
    # ... other args ...
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    local_rank=-1,                    # Auto-detect
    ddp_find_unused_parameters=False
)

# Launch with torchrun
# torchrun --nproc_per_node=4 train.py
```

## Training Best Practices

### Dataset Quality

1. **Diversity**: Include contracts from multiple Solidity versions
2. **Complexity**: Balance simple and complex functions
3. **Deduplication**: Remove duplicate or near-duplicate functions
4. **Validation**: Ensure both TAC and Solidity are syntactically valid

### Hyperparameter Tuning

1. **Learning Rate**: 2e-4 works well, but try 1e-4 to 5e-4
2. **LoRA Rank**: 16 is standard, higher (32) for more capacity
3. **Batch Size**: Larger effective batch (16-32) is better
4. **Warmup**: 500 steps standard, adjust based on dataset size

### Convergence Indicators

- **Training loss** plateaus around 0.3-0.5
- **Validation loss** follows training loss closely
- **Semantic similarity** reaches 0.8+ on validation set
- **Edit distance** drops below 0.4 for most functions

## Troubleshooting Training Issues

### Issue: Out of Memory

```python
# Solution: Reduce memory usage
config.batch_size = 2
config.gradient_accumulation_steps = 8
config.use_4bit_quantization = True
config.use_gradient_checkpointing = True
```

### Issue: Loss Not Decreasing

```python
# Solution: Adjust learning rate or warmup
config.learning_rate = 1e-4  # Lower learning rate
config.warmup_steps = 1000   # More warmup steps
```

### Issue: Overfitting

```python
# Solution: Increase regularization
config.weight_decay = 0.05   # Higher weight decay
config.lora_dropout = 0.1    # Higher dropout
config.num_epochs = 2        # Fewer epochs
```

### Issue: Slow Training

```python
# Solution: Optimize performance
config.bf16 = True
config.dataloader_num_workers = 4
config.gradient_accumulation_steps = 2  # Fewer accumulation steps
```

## Expected Timeline

For full dataset (238,446 pairs) on NVIDIA A100:

| Phase | Duration |
|-------|----------|
| **Data Collection** | 4-8 hours (depends on API rate) |
| **Data Processing** | 2-4 hours |
| **Model Training** | 24-48 hours (3 epochs) |
| **Evaluation** | 2-4 hours |
| **Total** | 32-64 hours |

## Next Steps

- Review [Evaluation Metrics](evaluation.md) for detailed metric explanations
- Check [Model Details](model-details.md) for architecture information
- See [Troubleshooting](troubleshooting.md) for common issues
- Explore [Data Format](data-format.md) for dataset specifications
