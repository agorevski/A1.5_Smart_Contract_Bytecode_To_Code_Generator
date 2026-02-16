# Architecture

## Pipeline Overview

```
EVM Bytecode → BytecodeAnalyzer → TAC → Fine-tuned LLM → Solidity
     │                                                       │
     ├─→ OpcodeFeatureExtractor → MaliciousContractClassifier│
     │                                                       │
     ├─→ VulnerabilityDetector (CFG + pattern matching)      │
     │                                                       ▼
     └──────────────────────────────────────────→ AuditReportGenerator
                                                        │
                                                        ▼
                                               PipelineOrchestrator
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

### Stage 3: Vulnerability Detection (`src/vulnerability_detector.py`)

- **6 vulnerability types**: reentrancy, timestamp dependence, integer overflow, delegatecall, access control, selfdestruct
- **5 severity levels**: critical, high, medium, low, info
- **Methods**: CFG pattern matching on bytecode + regex scanning on decompiled source
- **Output**: `VulnerabilityReport` with findings, risk score, and mitigation recommendations

### Stage 4: Malicious Classification (`src/malicious_classifier.py`)

- **Feature extraction**: Opcode frequency counting, TF-IDF, entropy-based supervised binning (via `OpcodeFeatureExtractor`)
- **Classification**: LightGBM gradient boosting (fallback to threshold-based heuristic)
- **Explainability**: LIME-based feature importance for predictions
- **Output**: `ClassificationResult` with confidence score and explanation

### Stage 5: Audit Report (`src/audit_report.py`)

- **Aggregation**: Combines classification, vulnerability scan, and decompilation results
- **Risk scoring**: 0–1 risk score mapped to levels (critical/high/medium/low/minimal)
- **Output**: `AuditReport` with findings, recommendations, and decompiled source preview

## Module Responsibilities

| Module | Purpose |
|--------|---------|
| `bytecode_analyzer.py` | EVM disassembly, CFG construction, TAC generation |
| `dataset_pipeline.py` | Etherscan data collection, Solidity parsing, function pair extraction |
| `local_compiler.py` | Local solc compilation via py-solc-x for data augmentation |
| `model_setup.py` | Model loading, LoRA configuration, training, inference |
| `training_pipeline.py` | Evaluation metrics (semantic similarity, BLEU, ROUGE, edit distance) |
| `opcode_features.py` | Opcode frequency, TF-IDF, entropy-based feature extraction |
| `vulnerability_detector.py` | CFG + pattern-based vulnerability scanning (6 types) |
| `malicious_classifier.py` | ML-based malicious/legitimate classification with LIME |
| `audit_report.py` | Security audit report generation with risk scoring |
| `pipeline_orchestrator.py` | Coordinates all stages: classify → decompile → detect → report |
| `selector_resolver.py` | 4-byte selector → function signature (4-tier lookup) |

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
                                                             │
                               ┌─────────────────────────────┤
                               ▼                             ▼
                        web/app.py                  PipelineOrchestrator
                    (Flask UI + API)           (classify → decompile → scan → report)
```

## Key Design Decisions

- **Two-stage pipeline** rather than end-to-end: static analysis provides reliable structure; LLM handles semantic recovery
- **LoRA fine-tuning**: trains only ~0.9% of parameters, enabling training on consumer GPUs
- **Multi-layer deduplication** in `download_hf_contracts.py`: source hash → pair hash → normalized pair hash → body frequency cap
- **Selector-based matching**: deterministically links Solidity functions to bytecode functions via keccak256 selectors
- **Modular security analysis**: vulnerability detection, classification, and audit reporting are independent modules coordinated by `PipelineOrchestrator`, allowing individual use or combined pipeline execution
- **4-tier selector resolution**: built-in database (97% confidence) → SQLite registry → local JSON cache → 4byte.directory API, with caching for API results