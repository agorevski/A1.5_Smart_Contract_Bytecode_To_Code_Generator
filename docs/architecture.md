# System Architecture

This document provides a detailed overview of the Smart Contract Decompilation system architecture.

## High-Level Architecture

The system uses a two-stage pipeline:

1. **Bytecode-to-TAC Conversion**: Static analysis converts EVM bytecode into structured Three-Address Code (TAC)
2. **TAC-to-Solidity Generation**: Fine-tuned Llama 3.2 3B model generates readable Solidity code from TAC

## Data Generation Pipeline

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
```

### Stage 1: Contract Collection

The system collects verified smart contracts from Etherscan:

- **API Integration**: Uses Etherscan API to fetch contract source code and metadata
- **Parallel Processing**: Supports concurrent collection with configurable worker threads
- **Version Coverage**: Targets contracts from Solidity 0.4.x through 0.8.x
- **Metadata Extraction**: Captures function signatures, visibility modifiers, and state mutability

### Stage 2: Bytecode → TAC Conversion

Static analysis transforms EVM bytecode into Three-Address Code:

- **Disassembly**: Converts bytecode into individual EVM opcodes
- **Control Flow Recovery**: Identifies loops, conditionals, and function boundaries
- **Function Detection**: Uses function selectors (first 4 bytes of keccak256 hash)
- **Storage Pattern Recognition**: Detects mappings, arrays, and storage layouts
- **TAC Generation**: Creates structured intermediate representation

### Stage 3: Function Pair Extraction

Matches Solidity functions with their corresponding TAC representations:

- **AST Parsing**: Analyzes Solidity source to extract function definitions
- **Selector Matching**: Links functions via keccak256 hash matching
- **Offset Alignment**: Ensures bytecode sections match source code functions
- **Metadata Preservation**: Maintains visibility, mutability, and modifier information

### Stage 4: Quality Control

Ensures high-quality training data:

- **Length Filtering**: Removes functions outside 50-20,000 token range
- **Deduplication**: Eliminates duplicate functions using content hashing
- **Syntax Validation**: Verifies both TAC and Solidity parse correctly
- **Complexity Balancing**: Maintains diverse complexity distribution
- **Pattern Analysis**: Ensures variety in contract patterns

## Model Training Pipeline

```text
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
```

### Base Model: Llama 3.2 3B

- **Parameters**: 3.21 billion parameters
- **Architecture**: 32 transformer layers with multi-head attention
- **Context Window**: 20,000 tokens (sufficient for large functions)
- **Pretraining**: Code and natural language corpus

### Fine-Tuning with LoRA

Low-Rank Adaptation enables efficient fine-tuning:

- **Parameter Efficiency**: Only ~30M trainable parameters (0.9% of base model)
- **Memory Efficiency**: 4-bit quantization reduces memory footprint
- **Target Modules**: Attention layers (q_proj, k_proj, v_proj, o_proj)
- **Rank**: 16 (balances capacity and efficiency)

### Training Configuration

- **Batch Strategy**: Small per-device batch with gradient accumulation
- **Learning Rate**: 2e-4 with 500-step linear warmup
- **Optimizer**: AdamW with weight decay for regularization
- **Duration**: 3-5 epochs over full dataset
- **Hardware**: NVIDIA A100 (40GB) recommended

## Evaluation Framework

```text
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

### Semantic Preservation

- **Semantic Similarity**: Measures meaning preservation using Sentence-BERT
- **BLEU Score**: N-gram overlap metric for code similarity
- **ROUGE-L**: Longest common subsequence metric

### Syntactic Accuracy

- **Edit Distance**: Normalized Levenshtein distance
- **Token Accuracy**: Token-level correctness measurement
- **Exact Match**: Percentage of perfectly decompiled functions

### Structural Fidelity

- **Control Flow**: Preservation of if/else, loops, and conditionals
- **Function Signatures**: Accuracy of parameter types and names
- **Modifiers**: Correctness of visibility and state mutability
- **Gas Patterns**: Preservation of optimization patterns

## Component Interactions

### Data Flow

1. **Contract Collection** → Raw Solidity + bytecode
2. **TAC Conversion** → Intermediate representation
3. **Pair Extraction** → Training dataset (JSONL)
4. **Model Training** → Fine-tuned Llama model
5. **Inference** → Decompiled Solidity code
6. **Evaluation** → Quality metrics

### Module Dependencies

- `bytecode_analyzer.py` → EVM disassembly and TAC generation
- `dataset_pipeline.py` → Data collection and preprocessing
- `model_setup.py` → Model configuration and fine-tuning
- `training_pipeline.py` → End-to-end orchestration

## Scalability Considerations

### Data Collection

- Parallel processing with configurable workers
- Rate limiting for API compliance
- Incremental dataset building
- Resumable collection from checkpoints

### Training

- Gradient accumulation for effective batch size
- Gradient checkpointing for memory efficiency
- Distributed training support (multi-GPU)
- Regular checkpointing for fault tolerance

### Inference

- Batch processing for throughput
- GPU optimization for speed
- Caching for repeated queries
- Configurable generation parameters
