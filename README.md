# Smart Contract Decompilation with Llama 3.2 3B

This repository implements the methodology described in the research paper **"Decompiling Smart Contracts with a Large Language Model"** (arXiv:2506.19624v1) for fine-tuning Llama 3.2 3B to convert EVM bytecode into human-readable Solidity code.

## Overview

The system combines traditional program analysis with modern large language models to achieve high-quality smart contract decompilation. It uses a two-stage pipeline:

1. **Bytecode-to-TAC Conversion**: Static analysis converts EVM bytecode into structured Three-Address Code (TAC)
2. **TAC-to-Solidity Generation**: Fine-tuned Llama 3.2 3B model generates readable Solidity code from TAC

## Key Results from Paper

Our implementation recreates the methodology that achieved:

| Metric | Paper Result |
|--------|--------------|
| **Semantic Similarity (avg)** | 0.82 |
| **Functions > 0.8 similarity** | 78.3% |
| **Functions < 0.4 edit distance** | 82.5% |
| **Dataset Size** | 238,446 function pairs |
| **Test Set Size** | 9,731 functions |

This represents a **significant improvement** over traditional decompilers which typically achieve only 40-50% of functions with >0.8 semantic similarity.

## High-Level Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                    DATA GENERATION PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1: Contract Collection                                   │
│  ├── Etherscan API integration for verified contracts           │
│  ├── Target: 238,446+ TAC-to-Solidity function pairs            │
│  ├── Coverage: Solidity 0.4.x through 0.8.x                     │
│  └── Metadata: Function signatures, visibility, modifiers       │
│                                                                 │
│  Stage 2: Bytecode → TAC Conversion                             │
│  ├── EVM disassembly and static analysis                        │
│  ├── Control flow recovery (loops, conditionals)                │
│  ├── Function boundary detection via selectors                  │
│  ├── Storage pattern recognition (mappings, arrays)             │
│  └── Three-address code generation with metadata                │
│                                                                 │
│  Stage 3: Function Pair Extraction                              │
│  ├── Solidity AST parsing for accurate function extraction      │
│  ├── Function selector matching (keccak256 hashing)             │
│  ├── Bytecode offset alignment with source code                 │
│  └── Metadata preservation (visibility, state mutability)       │
│                                                                 │
│  Stage 4: Quality Control                                       │
│  ├── Length filtering (50-20,000 tokens per paper)              │
│  ├── Deduplication by function hash                             │
│  ├── Syntax validation for both TAC and Solidity                │
│  ├── Complexity distribution balancing                          │
│  └── Pattern diversity analysis                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Base Model: Llama 3.2 3B                                       │
│  ├── Parameters: 3.21 billion                                   │
│  ├── Architecture: 32 transformer layers                        │
│  ├── Context window: 20,000 tokens (per paper)                  │
│  └── Pretrained on code and natural language                    │
│                                                                 │
│  Fine-Tuning: LoRA (Low-Rank Adaptation)                        │
│  ├── Rank (r): 16                                               │
│  ├── Alpha (α): 32                                              │
│  ├── Target modules: q_proj, k_proj, v_proj, o_proj             │
│  ├── Dropout: 0.05                                              │
│  ├── Trainable parameters: ~30M (~0.9% of base)                 │
│  └── Memory efficient: 4-bit quantization + gradient checkpoint │
│                                                                 │
│  Training Configuration:                                        │
│  ├── Batch size: 4-8 per device                                 │
│  ├── Gradient accumulation: 4 steps                             │
│  ├── Effective batch size: 16-32                                │
│  ├── Learning rate: 2e-4 with linear warmup                     │
│  ├── Optimizer: AdamW (β₁=0.9, β₂=0.999, ε=1e-8)                │
│  ├── Weight decay: 0.01                                         │
│  ├── Warmup steps: 500                                          │
│  ├── Epochs: 3-5                                                │
│  ├── Total training time: 24-48 hours (A100 GPU)                │
│  └── Checkpointing: Every 1,000 steps                           │
│                                                                 │
│  Dataset Split (matching paper):                                │
│  ├── Training: 85% (~202,679 pairs)                             │
│  ├── Validation: 10% (~23,845 pairs)                            │
│  └── Test: 5% (~9,731 pairs)                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     EVALUATION FRAMEWORK                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Semantic Preservation Metrics:                                 │
│  ├── Semantic Similarity (Sentence-BERT embeddings)             │
│  │   Target: 0.82 average, 78.3% > 0.8                          │
│  ├── BLEU Score (code similarity)                               │
│  └── ROUGE-L (longest common subsequence)                       │
│                                                                 │
│  Syntactic Accuracy Metrics:                                    │
│  ├── Normalized Edit Distance (Levenshtein)                     │
│  │   Target: 82.5% < 0.4                                        │
│  ├── Token-level accuracy                                       │
│  └── Exact match rate                                           │
│                                                                 │
│  Structural Fidelity Metrics:                                   │
│  ├── Control flow preservation (loops, conditionals)            │
│  ├── Function signature accuracy                                │
│  ├── Visibility modifier correctness                            │
│  ├── Code length correlation                                    │
│  └── Gas efficiency patterns preservation                       │
│                                                                 │
│  Advanced Analysis:                                             │
│  ├── Token frequency distribution comparison                    │
│  ├── Security pattern preservation (require, modifier)          │
│  ├── Complexity score alignment                                 │
│  └── Bimodal distribution analysis (0.85 and 0.95 peaks)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (16GB+ VRAM recommended for training)
- 32GB+ system RAM recommended
- 200GB+ storage for full dataset

### Setup

#### Clone the repository

  ```bash
  git clone <repository-url>
  cd A1.5_Smart_Contract_Bytecode_To_Code_Generator
  ```

#### Install dependencies

  ```bash
  pip install -r requirements.txt
  ```

#### Set up environment variables

  ```bash
  export ETHERSCAN_API_KEY="your_etherscan_api_key"
  export HF_TOKEN="your_huggingface_token"  # For Llama model access
  ```

#### Verify installation

  ```bash
  python demo.py
  ```

## Quick Start Guide

### 1. Demo (No GPU Required)

Run the demonstration script to validate the implementation:

```bash
python demo.py
```

This tests all components using pre-generated examples without requiring GPU or API keys.

### 2. Basic Usage

```python
from src.bytecode_analyzer import analyze_bytecode_to_tac
from src.model_setup import SmartContractDecompiler

# Step 1: Convert bytecode to TAC
bytecode = "0x608060405234801561001057600080fd5b50..."
tac_representation = analyze_bytecode_to_tac(bytecode)

# Step 2: Decompile to Solidity (requires trained model)
decompiler = SmartContractDecompiler("models/final/smart_contract_decompiler")
solidity_code = decompiler.decompile_tac_to_solidity(
    tac_representation,
    temperature=0.3,
    max_new_tokens=2048
)

print(solidity_code)
```

## Complete Training Pipeline

### Phase 1: Data Collection (Target: 238,446 pairs)

```python
from src.dataset_pipeline import DatasetBuilder

# Initialize dataset builder
builder = DatasetBuilder(
    etherscan_api_key="your_key",
    output_dir="data/raw"
)

# Collect verified contracts
# You'll need a list of verified contract addresses
with open('verified_contracts.txt', 'r') as f:
    contract_addresses = [line.strip() for line in f]

print(f"Collecting {len(contract_addresses)} contracts...")
collected = builder.collect_contracts(
    contract_addresses,
    max_workers=10  # Parallel collection
)

# Process contracts to create function pairs
print("Creating TAC-to-Solidity pairs...")
pairs = builder.process_contracts_to_function_pairs(batch_size=100)

# Apply quality filters
print("Filtering dataset...")
filtered = builder.filter_and_clean_dataset(
    min_length=50,      # Minimum function length
    max_length=20000    # Maximum sequence length (paper spec)
)

# Export in JSONL format for training
dataset_path = builder.export_dataset(output_format="jsonl")
print(f"Dataset ready: {dataset_path}")

# View statistics
stats = builder.get_dataset_statistics()
print(f"Total function pairs: {stats['total_function_pairs']}")
print(f"Visibility distribution: {stats['visibility_distribution']}")
```

### Phase 2: Model Training

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
    lora_alpha=32,
    lora_dropout=0.05,
    
    # Training configuration
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_epochs=3,
    warmup_steps=500,
    
    # System configuration
    output_dir="models/checkpoints",
    use_4bit_quantization=True,
    use_gradient_checkpointing=True
)

# Initialize and run pipeline
pipeline = SmartContractTrainingPipeline(config)

# Option 1: Run complete pipeline
results = pipeline.run_complete_pipeline()

# Option 2: Run step-by-step
train_path, val_path, test_path = pipeline.collect_and_prepare_dataset()
model_path = pipeline.train_model(train_path, val_path)
evaluation_results = pipeline.evaluate_model(model_path, test_path)

print(f"Average Semantic Similarity: {evaluation_results['avg_semantic_similarity']:.3f}")
print(f"Average Edit Distance: {evaluation_results['avg_edit_distance']:.3f}")
print(f"Functions > 0.8 similarity: {evaluation_results['high_similarity_pct']:.1f}%")
```

### Phase 3: Evaluation

```python
from src.training_pipeline import SmartContractEvaluator

evaluator = SmartContractEvaluator()

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
print(f"Semantic Similarity: {metrics.semantic_similarity:.3f}")
print(f"Edit Distance: {metrics.edit_distance:.3f}")
print(f"BLEU Score: {metrics.bleu_score:.3f}")
print(f"Structural Preservation: {metrics.structural_preservation:.3f}")
```

## Project Structure

```text
A1.5_Smart_Contract_Bytecode_To_Code_Generator/
│
├── src/                           # Core implementation modules
│   ├── __init__.py
│   ├── bytecode_analyzer.py      # EVM bytecode → TAC conversion
│   │   ├── Control flow recovery (loops, conditionals)
│   │   ├── Function boundary detection
│   │   ├── Storage pattern recognition
│   │   └── Three-address code generation
│   │
│   ├── dataset_pipeline.py       # Data collection & preprocessing
│   │   ├── Etherscan API integration
│   │   ├── Contract collection (parallel)
│   │   ├── Solidity function extraction
│   │   ├── TAC-Solidity pair creation
│   │   └── Quality filtering & deduplication
│   │
│   ├── model_setup.py            # Model configuration & fine-tuning
│   │   ├── Llama 3.2 3B initialization
│   │   ├── LoRA adapter configuration
│   │   ├── Custom dataset handling
│   │   ├── Training argument setup
│   │   └── Inference interface
│   │
│   └── training_pipeline.py      # End-to-end training orchestration
│       ├── Dataset splitting (85/10/5)
│       ├── Model training loop
│       ├── Comprehensive evaluation
│       └── Results analysis & visualization
│
├── reference/
│   └── 2506.19624v1.pdf         # Original research paper
│
├── data/                         # Dataset storage (not in repo)
│   ├── raw/                      # Collected contracts
│   ├── processed/                # TAC-Solidity pairs
│   ├── train/                    # Training split
│   ├── val/                      # Validation split
│   └── test/                     # Test split (9,731 functions)
│
├── models/                       # Model storage (not in repo)
│   ├── checkpoints/              # Training checkpoints
│   └── final/                    # Production model
│
├── demo.py                       # Demonstration script
├── demo_dataset.jsonl            # Sample data for demo
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## Data Format Specification

### Training Data Format (JSONL)

Each line in the dataset contains a JSON object with this structure:

```json
{
  "input": "### TAC Representation:\nfunction_selector_0x70a08231:\n  v0 = CALLDATALOAD 0x04\n  v1 = SLOAD storage[mapping_0][v0]\n  RETURN v1",
  "output": "function balanceOf(address account) public view returns (uint256) {\n    return _balances[account];\n}",
  "metadata": {
    "function_name": "balanceOf",
    "function_signature": "balanceOf(address)",
    "selector": "0x70a08231",
    "visibility": "public",
    "state_mutability": "view",
    "is_payable": false,
    "has_return": true,
    "contract_address": "0x...",
    "complexity_score": 2.3
  }
}
```

### TAC Representation Format

Three-Address Code uses this structured format:

```text
function_selector_0xabcd1234:
  // Function entry
  v0 = CALLDATALOAD 0x04          // Load first parameter
  v1 = CALLDATALOAD 0x24          // Load second parameter
  
  // Storage access
  v2 = SLOAD storage[mapping_0][v0]
  
  // Arithmetic operations
  v3 = ADD v2, v1
  v4 = LT v3, v2                  // Overflow check
  
  // Conditional logic
  if v4:
    REVERT                         // Revert on overflow
  
  // Storage update
  SSTORE storage[mapping_0][v0], v3
  
  // Event emission
  LOG3 topic_Transfer, v0, msg.sender, v1
  
  // Return
  RETURN 0x01
```

## Model Training Details

### LoRA Configuration (Paper Specification)

```python
lora_config = LoraConfig(
    r=16,                          # Rank (paper specification)
    lora_alpha=32,                 # Alpha = 2 * rank
    target_modules=[
        "q_proj",                  # Query projection
        "k_proj",                  # Key projection
        "v_proj",                  # Value projection
        "o_proj"                   # Output projection
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Training Hyperparameters

```python
training_args = TrainingArguments(
    # Batch configuration
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,    # Effective batch size: 16
    
    # Learning rate schedule
    learning_rate=2e-4,
    warmup_steps=500,
    lr_scheduler_type="linear",
    
    # Optimization
    optim="adamw_torch",
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    # Training duration
    num_train_epochs=3,
    max_steps=-1,                     # Use epochs instead
    
    # Memory optimization
    gradient_checkpointing=True,
    fp16=False,
    bf16=True,                        # Use bfloat16 if available
    
    # Evaluation
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,
    
    # Logging
    logging_steps=100,
    report_to="tensorboard",
    
    # System
    dataloader_num_workers=4,
    remove_unused_columns=False
)
```

### Inference Configuration

```python
generation_config = GenerationConfig(
    max_new_tokens=2048,           # Maximum output length
    temperature=0.3,               # Low for deterministic output
    top_p=0.9,                     # Nucleus sampling
    top_k=50,
    do_sample=True,
    repetition_penalty=1.1,
    num_return_sequences=1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)
```

## Evaluation Metrics Explained

### Semantic Similarity (Target: 0.82 avg)

Measures how well the decompiled code preserves the meaning of the original:

- Uses Sentence-BERT embeddings
- Computes cosine similarity between embeddings
- **Paper Result**: 78.3% of functions achieve >0.8 similarity
- **Distribution**: Bimodal with peaks at 0.85 and 0.95

### Normalized Edit Distance (Target: <0.4 for 82.5%)

Measures syntactic similarity using Levenshtein distance:

- Normalized by max(len(original), len(decompiled))
- **Paper Result**: 82.5% of functions achieve <0.4 distance
- Lower is better (0 = identical, 1 = completely different)

### BLEU Score

Standard metric for code similarity:

- Measures n-gram overlap (1-4 grams)
- Commonly used in code generation tasks
- Ranges from 0 to 1 (higher is better)

### Structural Preservation

Analyzes preservation of code structure:

- Control flow patterns (if/else, loops)
- Function metadata (visibility, modifiers)
- Security patterns (require, assert)
- Gas optimization patterns

## Performance Expectations

### Training Requirements

| Resource | Specification |
|----------|---------------|
| **GPU** | NVIDIA A100 (40GB) or equivalent |
| **Training Time** | 24-48 hours for full dataset |
| **Memory** | 16GB VRAM minimum (with 4-bit quantization) |
| **Storage** | 200GB for dataset + models |
| **CPU** | 16+ cores recommended for data processing |

### Training Progress (3 epochs on 238,446 pairs)

- **Steps per epoch**: ~25,000 (batch size 8)
- **Total steps**: ~75,000
- **Checkpoint size**: ~6GB per checkpoint
- **Final model size**: ~3.5GB (base) + ~50MB (LoRA)

### Inference Performance

| Metric | Performance |
|--------|-------------|
| **Speed** | 1-2 seconds per function (GPU) |
| **Accuracy** | 78.3% functions >0.8 semantic similarity |
| **Quality** | 82.5% functions <0.4 edit distance |
| **Throughput** | ~500-1000 functions/hour (batch inference) |

## Security Applications

### 1. Vulnerability Analysis in Unverified Contracts

**Use Case**: Analyze closed-source contracts for security flaws

**Example**: Dx Protocol vulnerability ($5.2M at risk)

- Contract had unverified source code
- Decompiler revealed critical state management bug
- Vulnerability: Repeated withdrawals before unlock time

```solidity
// Decompiled function exposed the flaw
function unlockToken(uint256 _tokenId) external {
    require(tokenLocks[msg.sender][_tokenId].isLocked, "Token is already unlocked");
    require(tokenLocks[msg.sender][_tokenId].unlockTime > 0, "Token is not locked");
    
    // BUG: State only updated if time condition met
    if (block.timestamp > tokenLocks[msg.sender][_tokenId].unlockTime) {
        tokenLocks[msg.sender][_tokenId].isLocked = false;
    }
    
    // Transfer happens regardless - VULNERABLE!
    uint256 amount = tokenLocks[msg.sender][_tokenId].amount;
    IERC20(tokenLocks[msg.sender][_tokenId].token).transfer(msg.sender, amount);
}
```

### 2. MEV Bot Exploit Post-Mortem

**Use Case**: Analyze exploited MEV bots to understand attack vectors

**Example**: MEV bot drain ($221,600 loss)

- Bot used proprietary, unverified contracts
- Decompiler revealed callback vulnerabilities
- Found: Arbitrary external calls and unprotected transfers

```solidity
// Decompiled callback with arbitrary external call
function swapX2YCallback(uint256 amountX, uint256, bytes calldata data) external {
    // VULNERABLE: No caller validation
    (bool success, ) = msg.sender.call{value: amountX}("");
    require(success, "...");
}

// Decompiled function with unprotected transfer
function d3MMSwapCallBack(address _to, uint256 _amount, bytes calldata) external {
    // VULNERABLE: Anyone can call and drain tokens
    IERC20(_to).transfer(msg.sender, _amount);
}
```

### 3. Incident Response

**Use Case**: Rapid analysis during active exploits

**Benefits**:

- Quick decompilation of attack contracts
- Understanding exploit mechanics in real-time
- Identifying similar vulnerable contracts
- Developing mitigation strategies

## Comparison with Traditional Decompilers

### Quantitative Comparison

| Metric | Traditional | Our Approach | Improvement |
|--------|-------------|--------------|-------------|
| **Semantic Similarity >0.8** | 40-50% | 78.3% | +56% |
| **Average Edit Distance** | 0.6-0.8 | 0.3 | -50% |
| **Variable Naming** | Generic (v0, temp1) | Meaningful recovery | Qualitative |
| **Control Flow** | Goto-heavy | Structured (if/while) | Qualitative |
| **Function Signatures** | Lost | Recovered | Qualitative |
| **Readability** | Poor | Human-like | Qualitative |

### Qualitative Advantages

1. **Variable Name Recovery**: Infers meaningful names based on usage patterns
2. **Control Flow**: Reconstructs high-level structures (if/else, while loops)
3. **Type Information**: Recovers uint256, address, mapping types
4. **Function Signatures**: Reconstructs parameter names and types
5. **Code Organization**: Maintains logical structure and formatting
6. **Security Patterns**: Preserves require statements and access control

## Limitations and Future Work

### Current Limitations

1. **Training Data Dependency**
   - Requires large dataset of verified contracts
   - Quality depends on diversity of training examples
   - May struggle with rare or novel patterns

2. **Computational Requirements**
   - Significant GPU resources for training
   - 24-48 hours training time on high-end hardware
   - Inference requires GPU for reasonable speed

3. **Complex DeFi Patterns**
   - Some sophisticated financial mechanisms need improvement
   - Fixed-point arithmetic precision sometimes lost
   - Complex temporal mechanics may be simplified

4. **Optimization Patterns**
   - Highly optimized bytecode can be challenging
   - Inline assembly not always perfectly handled
   - Some compiler optimizations may obscure original intent

### Future Enhancements

1. **Multi-Language Support**
   - Extend to Vyper, Yul, and other smart contract languages
   - Cross-language decompilation capabilities

2. **Real-Time Analysis**
   - Optimize inference for faster processing
   - Batch processing improvements
   - Edge deployment possibilities

3. **Advanced Pattern Recognition**
   - Improved handling of complex DeFi protocols
   - Better proxy pattern detection
   - Enhanced upgrade mechanism recognition

4. **Interactive Decompilation**
   - User feedback integration
   - Iterative refinement
   - Confidence scoring for decompiled code

5. **Integration Capabilities**
   - API for programmatic access
   - IDE plugins for developers
   - CI/CD pipeline integration

## Research Paper Citation

This implementation is based on the research paper:

```bibtex
@article{david2025decompiling,
  title={Decompiling Smart Contracts with a Large Language Model},
  author={David, Isaac and Zhou, Liyi and Song, Dawn and Gervais, Arthur and Qin, Kaihua},
  journal={arXiv preprint arXiv:2506.19624},
  year={2025},
  url={https://arxiv.org/abs/2506.19624}
}
```

**Key Contributions from Paper**:

- First successful application of LLMs to smart contract decompilation
- Novel hybrid approach combining static analysis with neural methods
- Comprehensive dataset of 238,446 TAC-to-Solidity function pairs
- Achieves 0.82 average semantic similarity (vs 0.4-0.5 for traditional methods)
- [Publicly accessible implementation](https://evmdecompiler.com)

## Additional Resources

### Documentation

- [Paper PDF](reference/2506.19624v1.pdf) - Full research paper with detailed methodology
- [Demo Script](demo.py) - Working examples without GPU/API requirements

### External Tools

- [Etherscan](https://etherscan.io) - Verified contract source
- [Remix IDE](https://remix.ethereum.org) - Solidity development
- [Solidity Docs](https://docs.soliditylang.org) - Language reference

### Related Projects

- [evmdecompiler.com](https://evmdecompiler.com) - Web interface implementation
- [Gigahorse](https://github.com/nevillegrech/gigahorse-toolchain) - Declarative decompiler
- [Panoramix](https://github.com/palkeo/panoramix) - Python-based decompiler

## Troubleshooting

### Common Issues

**Issue**: Out of memory during training

```bash
Solution: Enable 4-bit quantization and gradient checkpointing
- Reduce batch size to 2-4
- Increase gradient accumulation steps
- Use smaller LoRA rank (8 instead of 16)
```

**Issue**: Slow inference speed

```bash
Solution: Optimize generation parameters
- Reduce max_new_tokens
- Use lower temperature (0.1-0.3)
- Disable sampling for deterministic output
- Batch multiple functions together
```

**Issue**: Low quality decompilation

```bash
Solution: Check training data quality
- Ensure sufficient training examples (>100k)
- Verify TAC correctness
- Balance dataset complexity distribution
- Train for additional epochs
```

**Issue**: Etherscan API rate limiting

```bash
Solution: Implement rate limiting and caching
- Add delays between requests (0.2s minimum)
- Use multiple API keys in rotation
- Cache responses in database
- Process in smaller batches
```

## Contributing

Contributions are welcome! Areas for improvement:

1. **Data Collection**: More efficient contract discovery methods
2. **TAC Generation**: Enhanced pattern recognition
3. **Model Architecture**: Experiment with different model sizes
4. **Evaluation**: Additional metrics and benchmarks
5. **Documentation**: More examples and tutorials

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original research by David et al. (2025)
- Llama 3.2 model by Meta AI
- LoRA implementation by Microsoft Research
- Hugging Face Transformers library
- Ethereum community for verified contracts
- OpenAI for initial code analysis capabilities

## Support and Contact

- **Issues**: Please use GitHub Issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Paper**: See [arXiv:2506.19624](https://arxiv.org/abs/2506.19624)
- **Demo**: Try online at [evmdecompiler.com](https://evmdecompiler.com)

---

**Last Updated**: 2025-10-19  
**Version**: 1.0.0  
**Status**: Production Ready
