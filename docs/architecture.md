# Architecture

## Pipeline Overview

```
EVM Bytecode → BytecodeAnalyzer → TAC → Fine-tuned LLM → Solidity
```

### Stage 1: Bytecode → TAC (`src/bytecode_analyzer.py`)

1. **Disassembly** — EVM bytecode → individual opcodes (via `evmdasm`)
2. **Control flow analysis** — basic blocks, jump targets, dominators, loops
3. **Function identification** — 4-byte selector matching from dispatcher pattern
4. **TAC generation** — structured three-address code per function

### Stage 2: TAC → Solidity (`src/model_setup.py`)

- **Base model**: Llama 3.2 3B with LoRA fine-tuning
- **Input**: TAC with control flow annotations
- **Output**: Human-readable Solidity function

## Module Responsibilities

| Module | Purpose |
|--------|---------|
| `bytecode_analyzer.py` | EVM disassembly, CFG construction, TAC generation |
| `dataset_pipeline.py` | Etherscan data collection, Solidity parsing, function pair extraction |
| `local_compiler.py` | Local solc compilation via py-solc-x for data augmentation |
| `model_setup.py` | Model loading, LoRA configuration, training, inference |
| `training_pipeline.py` | Evaluation metrics (semantic similarity, edit distance) |

## Data Flow

```
download_hf_contracts.py          train.py
        │                              │
        ▼                              ▼
  HuggingFace dataset ──→ contracts.db ──→ JSONL ──→ train/val/test split
        │                                                    │
        ▼                                                    ▼
  solc compilation ──→ TAC+Solidity pairs           LoRA fine-tuning
                                                             │
                                                             ▼
                                                    Trained model
```

## Key Design Decisions

- **Two-stage pipeline** rather than end-to-end: static analysis provides reliable structure; LLM handles semantic recovery
- **LoRA fine-tuning**: trains only ~0.9% of parameters, enabling training on consumer GPUs
- **Multi-layer deduplication** in `download_hf_contracts.py`: source hash → pair hash → normalized pair hash → body frequency cap
- **Selector-based matching**: deterministically links Solidity functions to bytecode functions via keccak256 selectors