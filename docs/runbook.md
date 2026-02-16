# Runbook

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.11 |
| GPU (training) | 16 GB VRAM, CUDA-compatible | RTX 8000 / A100 |
| GPU (inference) | 4 GB VRAM | 8 GB+ |
| RAM | 16 GB | 32 GB+ |
| Disk | 10 GB | 50 GB (models + datasets) |

CPU-only is supported for inference and `--tiny` training but is significantly slower.

## 1. Installation

```bash
git clone https://github.com/agorevski/A1.5_Smart_Contract_Bytecode_To_Code_Generator.git
cd A1.5_Smart_Contract_Bytecode_To_Code_Generator

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### Environment Variables

Set these before training or data collection:

```bash
export HF_TOKEN="hf_..."           # HuggingFace token (required for Llama access)
export ETHERSCAN_API_KEY="..."     # Only if collecting from Etherscan directly
```

Alternatively, add them to `src/settings.yaml`.

---

## 2. Prepare Training Data

Download verified Solidity contracts from HuggingFace, compile with compatible solc versions, generate TAC, and export training pairs:

```bash
# Quick test (20 contracts)
python download_hf_contracts.py --limit 20

# Production run (all available contracts)
python download_hf_contracts.py
```

The pipeline runs three phases automatically. Run them independently if needed:

```bash
python download_hf_contracts.py --download-only      # Phase 1: fetch from HuggingFace
python download_hf_contracts.py --compile-only        # Phase 2: compile + generate TAC pairs
python download_hf_contracts.py --export-only         # Phase 3: deduplicate + export JSONL
```

**Key flags:**

| Flag | Default | Purpose |
|------|---------|---------|
| `--limit N` | all | Max contracts to download |
| `--max-compiler-versions N` | all | Solc versions per contract |
| `--workers N` | auto | Parallel compilation workers |
| `--max-body-dupes N` | 5 | Duplicate function body cap |
| `--min-body-length N` | 50 | Min Solidity body length (chars) |
| `--output PATH` | `data/hf_training_dataset.jsonl` | Output JSONL path |

**Output:** `data/hf_training_dataset.jsonl` — each line: `{"input": "<TAC>", "output": "<Solidity>", "metadata": {...}}`

---

## 3. Train the Model

### Single GPU

```bash
# Standard training (Llama 3.2 3B, LoRA, 3 epochs)
python train.py --skip-collection --dataset data/hf_training_dataset.jsonl

# Quick test (1 epoch, small batch)
python train.py --skip-collection --dataset data/hf_training_dataset.jsonl --small

# Fast E2E smoke test (facebook/opt-125m, no GPU needed)
python train.py --skip-collection --dataset data/hf_training_dataset.jsonl --tiny
```

### Multi-GPU (DDP via torchrun)

```bash
./run_train_torchrun.sh                           # 4 GPUs (default)
NGPUS=2 ./run_train_torchrun.sh                   # 2 GPUs
DATASET=./data/custom.jsonl ./run_train_torchrun.sh
```

### Multi-GPU (DeepSpeed)

```bash
./run_train_deepspeed.sh                          # ZeRO Stage 0 + BF16
NGPUS=2 DS_CONFIG=ds_config_z3.json ./run_train_deepspeed.sh  # ZeRO Stage 3
```

> **Note:** For Llama 3.2 3B + LoRA, torchrun DDP is ~40-60% faster than DeepSpeed. Use DeepSpeed for 7B+ models or when GPU memory is constrained.

### Training flags

| Flag | Default | Notes |
|------|---------|-------|
| `--epochs N` | 3 | Overridden to 1 by `--small`/`--tiny` |
| `--batch-size N` | 4 | Overridden to 2 by `--small`/`--tiny` |
| `--lr FLOAT` | 2e-4 | Learning rate (cosine schedule) |
| `--max-seq-length N` | 2048 | Shell scripts auto-detect from dataset |
| `--model-name NAME` | `meta-llama/Llama-3.2-3B` | Any HF model ID |
| `--skip-eval` | — | Skip post-training evaluation |
| `--deepspeed PATH` | — | Enable DeepSpeed with config file |
| `--resume PATH` | — | Resume from checkpoint |

### Using alternative models

```bash
# Qwen 2.5 Coder 32B (recommended upgrade — see docs/training-recommendations.md)
python train.py --skip-collection --dataset data/hf_training_dataset.jsonl \
    --model-name Qwen/Qwen2.5-Coder-32B --batch-size 2 --max-seq-length 4096

# StarCoder2 15B (fast iteration)
python train.py --skip-collection --dataset data/hf_training_dataset.jsonl \
    --model-name bigcode/starcoder2-15b --batch-size 4 --max-seq-length 4096
```

**Output:** Trained model saved to `models/final_model/`

---

## 4. Evaluate

Evaluation runs automatically after training unless `--skip-eval` is passed. Metrics are written to `results/`.

| Metric | Target |
|--------|--------|
| Semantic similarity (avg) | > 0.82 |
| Functions > 0.8 similarity | > 78% |
| Normalized edit distance < 0.4 | > 82% |

---

## 5. Run Inference

### Python API

```python
from src.model_setup import SmartContractDecompiler

decompiler = SmartContractDecompiler("models/final_model")

# Single function
solidity = decompiler.decompile_tac_to_solidity(
    tac_input="function func_0xa9059cbb:\n  block_0x0080:\n    ...",
    max_new_tokens=1024,
    temperature=0.1,
)

# Batch
results = decompiler.decompile_batch(
    tac_inputs=["func1_tac", "func2_tac"],
    max_new_tokens=1024,
)
```

### Full Security Analysis Pipeline

```python
from src.pipeline_orchestrator import PipelineOrchestrator, PipelineConfig, PipelineStage

config = PipelineConfig(
    stages=[
        PipelineStage.CLASSIFY,
        PipelineStage.DECOMPILE,
        PipelineStage.DETECT_VULNERABILITIES,
        PipelineStage.AUDIT_REPORT,
    ]
)
orchestrator = PipelineOrchestrator(config)

result = orchestrator.analyze(bytecode="0x60806040...", contract_address="0x1234...")

print(result.classification_result)   # malicious/legitimate + confidence
print(result.vulnerability_report)    # vulnerabilities + risk score
print(result.decompiled_source)       # TAC output
```

### Web UI

```bash
python web/app.py
# Opens at http://localhost:5000
```

The web app requires a trained model at `models/final_model/`. If absent, decompilation returns TAC only (no Solidity generation).

**API endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/decompile` | POST | SSE-streaming decompilation |
| `/api/vulnerability-scan` | POST | Vulnerability scan |
| `/api/classify` | POST | Malicious/legitimate classification |
| `/api/audit-report` | POST | Full security audit |
| `/api/gpu-stats` | GET | GPU utilization and memory |
| `/api/health` | GET | Server status |

All POST endpoints accept `{"bytecode": "0x..."}`.

---

## 6. Run Tests

```bash
python -m pytest                    # all ~380 tests
python -m pytest -v                 # verbose
python -m pytest -x                 # stop on first failure
python -m pytest tests/test_e2e.py  # end-to-end only
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce `--batch-size` or `--max-seq-length`; use `--tiny` for smoke tests |
| `HF_TOKEN` required | Set env var or add to `src/settings.yaml` |
| No training pairs generated | Check solc: `python -c "from solcx import get_installed_solc_versions; print(get_installed_solc_versions())"` |
| Multi-GPU quantization error | Use `torchrun` or `deepspeed` — not bare `python` with `DataParallel` |
| Web app shows no Solidity | Train a model first; the app needs `models/final_model/` |
| Slow DeepSpeed training | Switch to `run_train_torchrun.sh` for small models (3B + LoRA) |
