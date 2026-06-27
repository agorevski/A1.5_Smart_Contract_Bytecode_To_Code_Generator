# Smart Contract Bytecode-to-Solidity Decompiler & Security Analyzer

A comprehensive smart contract analysis system combining LLM-powered bytecode decompilation with vulnerability detection, malicious contract classification, and automated security audit report generation.

**Paper:** [Decompiling Smart Contracts with a Large Language Model](reference/2506.19624v1.pdf) (arXiv:2506.19624v1)

## Architecture

```
EVM Bytecode → Static Analysis → TAC → Fine-tuned LLM → Solidity
     │                                                       │
     ├─→ Opcode Feature Extraction → Malicious Classification │
     │                                                       │
     ├─→ CFG Pattern Matching → Vulnerability Detection       │
     │                                                       ▼
     └──────────────────────────────────────────→ Audit Report
```

The system implements four analysis pipelines coordinated by `PipelineOrchestrator`:

1. **Bytecode → TAC → Solidity** — `BytecodeAnalyzer` disassembles EVM bytecode, constructs control-flow graphs, and emits TAC; a LoRA fine-tuned LLM translates TAC into readable Solidity
2. **Vulnerability Detection** — `VulnerabilityDetector` scans bytecode and source for reentrancy, timestamp dependence, overflow, delegatecall, access control, and selfdestruct vulnerabilities
3. **Malicious Classification** — `MaliciousContractClassifier` uses opcode frequency features + LightGBM with LIME explainability to classify contracts as malicious or legitimate
4. **Audit Report Generation** — `AuditReportGenerator` aggregates all findings into a comprehensive security audit with risk scoring

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

> **Requirements:** Python ≥ 3.10, CUDA-compatible GPU with ≥ 4 GB VRAM (inference) / ≥ 16 GB (training).

### 2. Download Training Data

Download verified Solidity contracts from HuggingFace (`andstor/smart_contracts`), compile each with compatible solc versions, generate TAC, and export training pairs:

```bash
# Download 20 contracts (quick test)
uv run python download_hf_contracts.py --limit 20

# Download 100 contracts with max 3 compiler versions each
uv run python download_hf_contracts.py --limit 100 --max-compiler-versions 3

# Full dataset (all available contracts)
uv run python download_hf_contracts.py
```

**Phases** (can be run independently):

```bash
uv run python download_hf_contracts.py --download-only     # Phase 1: Download only
uv run python download_hf_contracts.py --compile-only       # Phase 2: Compile & generate TAC
uv run python download_hf_contracts.py --export-only        # Phase 3: Export JSONL
```

**Output:** `data/hf_training_dataset.jsonl` — each line is `{"input": "<TAC>", "output": "<Solidity>", "metadata": {...}}`

### 3. Train the Model

```bash
# Train on the downloaded dataset (Llama 3.2 3B with LoRA)
uv run python train.py --skip-collection --dataset data/hf_training_dataset.jsonl

# Quick test (1 epoch, small batch)
uv run python train.py --skip-collection --dataset data/hf_training_dataset.jsonl --small

# Fast E2E test with a tiny model (no GPU needed)
uv run python train.py --skip-collection --dataset data/hf_training_dataset.jsonl --tiny

# Full training with custom parameters
uv run python train.py --skip-collection --dataset data/hf_training_dataset.jsonl \
    --epochs 5 --batch-size 4 --lr 2e-4 --max-seq-length 4096
```

**Output:** Trained model saved to `models/`

### 4. Evaluate

Evaluation runs automatically after training unless `--skip-eval` is passed. Results are saved to `results/`.

```bash
# Train and evaluate
uv run python train.py --skip-collection --dataset data/hf_training_dataset.jsonl

# Train without evaluation
uv run python train.py --skip-collection --dataset data/hf_training_dataset.jsonl --skip-eval
```

## End-to-End Example

```bash
# Step 1: Download and prepare 20 contracts
uv run python download_hf_contracts.py --limit 20

# Step 2: Train the model on the downloaded data
uv run python train.py --skip-collection --dataset data/hf_training_dataset.jsonl --small

# Step 3: Check results
ls results/
```

For the full operator guide, including current-data sufficiency checks, data regeneration, training, evaluation, and model usage, see [docs/runbook.md](docs/runbook.md).

## Running Tests

The pytest configuration disables Web3's unrelated `pytest_ethereum` entry
point so repository tests collect consistently in supported `uv` environments.

```bash
# Run all tests
uv run pytest

# Verbose output
uv run pytest -v

# Run specific module tests (examples)
uv run pytest tests/test_bytecode_analyzer.py -v
uv run pytest tests/test_dataset_pipeline.py -v
uv run pytest tests/test_docs_consistency.py -v
uv run pytest tests/test_vulnerability_detector.py -v
uv run pytest tests/test_malicious_classifier.py -v
uv run pytest tests/test_audit_report.py -v
uv run pytest tests/test_pipeline_orchestrator.py -v
uv run pytest tests/test_opcode_features.py -v
uv run pytest tests/test_web_app.py -v
uv run pytest tests/test_e2e.py -v

# Run with coverage
uv run pytest --cov=src tests/
```

## Project Structure

```
├── download_hf_contracts.py   # CLI: Download HuggingFace data → compile → export
├── train.py                   # CLI: Train & evaluate the decompilation model
├── pyproject.toml             # Project metadata, uv dependencies, and tool config
├── uv.lock                    # Locked Python dependencies generated by uv
│
├── src/                       # Core library
│   ├── __init__.py            # Package exports (v2.0.0)
│   ├── bytecode_analyzer.py   # EVM bytecode → TAC conversion, CFG construction
│   ├── dataset_pipeline.py    # Etherscan data collection, Solidity parsing, DB
│   ├── local_compiler.py      # Local solc compilation (py-solc-x)
│   ├── model_setup.py         # Model config, LoRA fine-tuning, inference
│   ├── training_pipeline.py   # Evaluation metrics (similarity, edit distance, replication precision/recall)
│   ├── opcode_features.py     # Opcode frequency, TF-IDF, entropy-based features
│   ├── vulnerability_detector.py  # Vulnerability scanning (6 types, 5 severities)
│   ├── malicious_classifier.py    # Malicious contract classification + LIME explainability
│   ├── audit_report.py        # Security audit report generation + risk scoring
│   ├── pipeline_orchestrator.py   # Coordinates all analysis stages
│   ├── selector_resolver.py   # 4-byte selector → function signature (4-tier lookup)
│   └── settings.yaml          # API keys config
│
├── tests/                     # Unit, integration, docs, and web tests (pytest)
│   ├── test_bytecode_analyzer.py   # Bytecode parsing, CFG, TAC, stack sim
│   ├── test_dataset_pipeline.py    # Data collection and function pairing
│   ├── test_dataset_quality_issues.py  # Dataset quality regression coverage
│   ├── test_docs_consistency.py    # README/runbook workflow drift checks
│   ├── test_opcode_features.py     # Feature extraction, TF-IDF, entropy
│   ├── test_prompt_metadata.py     # Prompt compiler metadata coverage
│   ├── test_vulnerability_detector.py  # Vulnerability scanning
│   ├── test_malicious_classifier.py    # Classification and explainability
│   ├── test_audit_report.py        # Audit reports and risk scoring
│   ├── test_pipeline_orchestrator.py   # Pipeline orchestration
│   ├── test_replication_metrics.py # Research replication metrics
│   ├── test_web_app.py             # Web API auth and UI integration
│   └── test_e2e.py                 # End-to-end integration
│
├── web/                       # Flask web application
│   ├── app.py                 # Server: decompilation, vuln scan, classify, audit
│   ├── templates/index.html   # Decompiler UI
│   └── static/                # Frontend assets (app.js, style.css)
│
├── scripts/                   # Debug & utility scripts
│   ├── demo.py
│   ├── check_bytecode_format.py
│   ├── inspect_bytecode.py
│   └── debug_*.py             # Ad-hoc debugging scripts
│
├── train_common.sh            # Shared multi-GPU training config
├── run_train_torchrun.sh      # Multi-GPU DDP training (torchrun)
├── run_train_deepspeed.sh     # DeepSpeed training
├── ds_config.json             # DeepSpeed configuration
│
├── data/                      # Generated datasets (gitignored)
├── models/                    # Trained models (gitignored)
├── results/                   # Evaluation results (gitignored)
├── test_data/                 # Test fixtures
├── docs/                      # Documentation
│   ├── architecture.md        # System design & data flow
│   ├── model-details.md       # Model config, LoRA, quantization
│   ├── data-format.md         # JSONL schema, TAC format, DB schema
│   ├── training-recommendations.md  # Model selection & multi-GPU training
│   ├── runbook.md             # E2E training, evaluation, and inference guide
│   └── contributing.md        # Development setup & guidelines
├── demo_dataset.jsonl         # Sample training data (3 examples)
└── reference/                 # Research paper PDF, enhancement plans
```

## CLI Reference

### `download_hf_contracts.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--limit N` | `0` (all) | Max contracts to download |
| `--max-compiler-versions N` | `5` | Max solc versions per contract; each compiles optimizer on/off |
| `--workers N` | auto | Parallel compilation workers |
| `--max-body-dupes N` | `2` | Max copies of same function body |
| `--min-body-length N` | `50` | Min Solidity body length |
| `--cache-dir PATH` | HuggingFace default | Cache directory for Parquet files |
| `--hf-revision REV` | latest | HuggingFace dataset revision/commit to pin downloads |
| `--download-only` | — | Only download, skip compilation |
| `--compile-only` | — | Only compile downloaded contracts |
| `--export-only` | — | Only export existing pairs to JSONL |
| `--output PATH` | `data/hf_training_dataset.jsonl` | Output file |
| `--export-selectors PATH` | — | Export selector registry JSON |
| `--import-selectors PATH` | — | Import selector registry JSON before processing |
| `--db PATH` | `data/contracts.db` | SQLite database path |

### `train.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--skip-collection` | — | Use existing dataset (skip Etherscan) |
| `--dataset PATH` | auto | Path to JSONL dataset |
| `--small` | — | Quick test: 1 epoch, batch=2 |
| `--tiny` | — | Use `facebook/opt-125m` for fast testing |
| `--epochs N` | `3` | Training epochs |
| `--batch-size N` | `4` | Per-device batch size |
| `--lr FLOAT` | `2e-4` | Learning rate |
| `--max-seq-length N` | `2048` | Max token sequence length |
| `--model-name NAME` | `meta-llama/Llama-3.2-3B` | Base model |
| `--skip-eval` | — | Skip post-training evaluation |
| `--dataset-only` | — | Only build dataset, skip training |
| `--no-compiler-metadata` | — | Omit compiler/optimizer metadata from prompts |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | For Llama access | HuggingFace token (gated model) |
| `ETHERSCAN_API_KEY` | For Etherscan collection | Only needed with `train.py` without `--skip-collection` |
| `WEB_API_KEY` | Shared/remote web deployments | Protects API endpoints and GPU telemetry; the browser UI has an in-memory API key field |
| `WEB_HOST` | Optional | Web bind address, default `127.0.0.1`; use with `WEB_API_KEY` for non-loopback hosts |
| `WEB_CORS_ORIGINS` | Optional | Comma-separated allowed API origins, default `http://127.0.0.1:5000,http://localhost:5000` |

Alternatively, set these in `src/settings.yaml`.

## Model Details

- **Base model:** Llama 3.2 3B (`meta-llama/Llama-3.2-3B`)
- **Fine-tuning:** LoRA (r=16, α=32, dropout=0.1)
- **Quantization:** 4-bit NF4 via bitsandbytes by default (`load_in_4bit=True`)
- **Target metrics:** Semantic similarity > 0.8, Edit distance < 0.4

See [docs/model-details.md](docs/model-details.md) for more.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce `--batch-size` or use `--tiny` |
| `HF_TOKEN` required | Set `HF_TOKEN` env var or add to `src/settings.yaml` |
| No training pairs generated | Check solc installation: `uv run python -c "from solcx import get_installed_solc_versions; print(get_installed_solc_versions())"` |
| Download hangs | HuggingFace data is cached after first download; check `~/.cache/huggingface/hub/` |


## License

MIT — see [LICENSE](LICENSE)