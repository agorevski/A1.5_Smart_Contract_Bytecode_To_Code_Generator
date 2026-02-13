# Smart Contract Bytecode-to-Solidity Decompiler

A two-stage pipeline for decompiling EVM smart contract bytecode into human-readable Solidity code, powered by a fine-tuned Llama 3.2 3B model.

**Paper:** [Decompiling Smart Contracts with a Large Language Model](reference/2506.19624v1.pdf) (arXiv:2506.19624v1)

## Architecture

```
EVM Bytecode → Static Analysis → TAC (Three-Address Code) → Fine-tuned LLM → Solidity
```

1. **Bytecode → TAC** — `src/bytecode_analyzer.py` disassembles EVM bytecode, constructs control-flow graphs, and emits Three-Address Code  
2. **TAC → Solidity** — A LoRA fine-tuned Llama 3.2 3B translates TAC into readable Solidity  

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Requirements:** Python ≥ 3.10, CUDA-compatible GPU with ≥ 4 GB VRAM (inference) / ≥ 16 GB (training).

### 2. Download Training Data

Download verified Solidity contracts from HuggingFace (`andstor/smart_contracts`), compile each with compatible solc versions, generate TAC, and export training pairs:

```bash
# Download 20 contracts (quick test)
python download_hf_contracts.py --limit 20

# Download 100 contracts with max 3 compiler versions each
python download_hf_contracts.py --limit 100 --max-compiler-versions 3

# Full dataset (all available contracts)
python download_hf_contracts.py
```

**Phases** (can be run independently):

```bash
python download_hf_contracts.py --download-only     # Phase 1: Download only
python download_hf_contracts.py --compile-only       # Phase 2: Compile & generate TAC
python download_hf_contracts.py --export-only        # Phase 3: Export JSONL
```

**Output:** `data/hf_training_dataset.jsonl` — each line is `{"input": "<TAC>", "output": "<Solidity>", "metadata": {...}}`

### 3. Train the Model

```bash
# Train on the downloaded dataset (Llama 3.2 3B with LoRA)
python train.py --skip-collection --dataset data/hf_training_dataset.jsonl

# Quick test (1 epoch, small batch)
python train.py --skip-collection --dataset data/hf_training_dataset.jsonl --small

# Fast E2E test with a tiny model (no GPU needed)
python train.py --skip-collection --dataset data/hf_training_dataset.jsonl --tiny

# Full training with custom parameters
python train.py --skip-collection --dataset data/hf_training_dataset.jsonl \
    --epochs 5 --batch-size 4 --lr 2e-4 --max-seq-length 4096
```

**Output:** Trained model saved to `models/`

### 4. Evaluate

Evaluation runs automatically after training unless `--skip-eval` is passed. Results are saved to `results/`.

```bash
# Train and evaluate
python train.py --skip-collection --dataset data/hf_training_dataset.jsonl

# Train without evaluation
python train.py --skip-collection --dataset data/hf_training_dataset.jsonl --skip-eval
```

## End-to-End Example

```bash
# Step 1: Download and prepare 20 contracts
python download_hf_contracts.py --limit 20

# Step 2: Train the model on the downloaded data
python train.py --skip-collection --dataset data/hf_training_dataset.jsonl --small

# Step 3: Check results
ls results/
```

## Running Tests

```bash
# Run all tests
python -m pytest

# Verbose output
python -m pytest -v

# Run only bytecode analyzer tests
python -m pytest tests/test_bytecode_analyzer.py -v

# Run only dataset pipeline tests
python -m pytest tests/test_dataset_pipeline.py -v
```

## Project Structure

```
├── download_hf_contracts.py   # CLI: Download HuggingFace data → compile → export
├── train.py                   # CLI: Train & evaluate the decompilation model
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project config (pytest, black, mypy)
│
├── src/                       # Core library
│   ├── __init__.py
│   ├── bytecode_analyzer.py   # EVM bytecode → TAC conversion
│   ├── dataset_pipeline.py    # Etherscan data collection, parsing, DB
│   ├── local_compiler.py      # Local solc compilation (py-solc-x)
│   ├── model_setup.py         # Model config, LoRA fine-tuning, inference
│   ├── training_pipeline.py   # Evaluation metrics (semantic similarity, edit distance)
│   └── settings.yaml          # API keys config
│
├── tests/                     # Unit tests (pytest)
│   ├── test_bytecode_analyzer.py   # 130 tests for bytecode analysis
│   └── test_dataset_pipeline.py    # 65 tests for dataset pipeline
│
├── scripts/                   # Debug & utility scripts
│   ├── demo.py
│   ├── check_bytecode_format.py
│   ├── inspect_bytecode.py
│   └── debug_*.py             # Ad-hoc debugging scripts
│
├── data/                      # Generated datasets (gitignored)
├── models/                    # Trained models (gitignored)
├── results/                   # Evaluation results (gitignored)
├── docs/                      # Documentation
│   ├── architecture.md        # System design & data flow
│   ├── model-details.md       # Model config, LoRA, quantization
│   ├── data-format.md         # JSONL schema, TAC format, DB schema
│   └── contributing.md        # Development setup & guidelines
├── demo_dataset.jsonl         # Sample training data (3 examples)
└── reference/                 # Research paper PDF
```

## CLI Reference

### `download_hf_contracts.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--limit N` | `0` (all) | Max contracts to download |
| `--max-compiler-versions N` | `0` (all) | Max solc versions per contract |
| `--workers N` | auto | Parallel compilation workers |
| `--max-body-dupes N` | `5` | Max copies of same function body |
| `--min-body-length N` | `50` | Min Solidity body length |
| `--download-only` | — | Only download, skip compilation |
| `--compile-only` | — | Only compile downloaded contracts |
| `--export-only` | — | Only export existing pairs to JSONL |
| `--output PATH` | `data/hf_training_dataset.jsonl` | Output file |
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

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | For Llama access | HuggingFace token (gated model) |
| `ETHERSCAN_API_KEY` | For Etherscan collection | Only needed with `train.py` without `--skip-collection` |

Alternatively, set these in `src/settings.yaml`.

## Model Details

- **Base model:** Llama 3.2 3B (`meta-llama/Llama-3.2-3B`)
- **Fine-tuning:** LoRA (r=16, α=32, dropout=0.1)
- **Quantization:** 8-bit via bitsandbytes
- **Target metrics:** Semantic similarity > 0.8, Edit distance < 0.4

See [docs/model-details.md](docs/model-details.md) for more.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce `--batch-size` or use `--tiny` |
| `HF_TOKEN` required | Set `HF_TOKEN` env var or add to `src/settings.yaml` |
| No training pairs generated | Check solc installation: `python -c "from solcx import get_installed_solc_versions; print(get_installed_solc_versions())"` |
| Download hangs | HuggingFace data is cached after first download; check `~/.cache/huggingface/hub/` |


## License

MIT — see [LICENSE](LICENSE)