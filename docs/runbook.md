# E2E Training and Inference Runbook

This runbook explains how to train the TAC-to-Solidity model with the codebase as it exists today, how to decide whether the available data is enough, how to generate more valid data, and how to use the trained model afterward.

## Quick answer: do I have enough data today?

Yes, if `data/hf_training_dataset.jsonl` is present in your worktree. The prepared worktree inspected for this runbook contained:

| File | Rows | Purpose |
|------|-----:|---------|
| `data/hf_training_dataset.jsonl` | 11,412 | Source dataset for training and re-splitting |
| `data/train_dataset.jsonl` | 8,678 | Current train split |
| `data/val_dataset.jsonl` | 1,022 | Current validation split |
| `data/test_dataset.jsonl` | 1,712 | Current test split |
| `demo_dataset.jsonl` | 3 | Smoke-test/demo only |

That is enough to kick off a real training run and validate the end-to-end pipeline. It is not enough to claim paper-scale coverage or production-quality decompilation. The paper-scale target referenced in the codebase is 238,446 TAC-to-Solidity function pairs, so the current 11,412-pair dataset is roughly 5% of that scale.

Generated data, models, and results are gitignored. A clean clone will not include `data/*.jsonl`, `data/*.db*`, `models/`, or `results/`; regenerate them with the commands below.

## Recommended path

| Goal | Data | Command pattern |
|------|------|-----------------|
| Verify the pipeline quickly | Existing JSONL or small generated set | `uv run python train.py --skip-collection --dataset data/hf_training_dataset.jsonl --tiny --skip-eval` |
| Train a baseline with current data | `data/hf_training_dataset.jsonl` | `uv run python train.py --skip-collection --dataset data/hf_training_dataset.jsonl` |
| Train faster on multiple GPUs | `data/hf_training_dataset.jsonl` | `NGPUS=4 DATASET=./data/hf_training_dataset.jsonl ./run_train_torchrun.sh` |
| Build more data first | Hugging Face verified contracts | `uv run python download_hf_contracts.py --limit 1000 --max-compiler-versions 3` |
| Reproduce paper-scale intent | Large generated dataset | Scale `download_hf_contracts.py` until pair counts approach the target, then train/evaluate |

## 1. Install and configure

```bash
uv sync
```

Requirements:

| Resource | Minimum | Notes |
|----------|---------|-------|
| Python | 3.10+ | Project metadata targets Python 3.10 |
| GPU for training | CUDA GPU, 16 GB+ VRAM recommended | `--tiny` can run CPU-only but is only a smoke test |
| GPU for inference | CUDA GPU, 4 GB+ VRAM recommended | CPU works but is slow |
| Disk | 10 GB+ | More if generating large datasets/checkpoints |

Environment variables:

```bash
export HF_TOKEN="hf_..."        # Required for gated base models such as Llama
export ETHERSCAN_API_KEY="..."  # Only needed for train.py's Etherscan collection path
```

You can also copy `src/settings.yaml.example` to `src/settings.yaml`, but environment variables are easier to keep out of source control.

## 2. Check what data you have

Run this before training:

```bash
wc -l data/*.jsonl demo_dataset.jsonl 2>/dev/null || true
du -h data/*.jsonl demo_dataset.jsonl 2>/dev/null || true
```

Validate that a JSONL file contains usable training pairs:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

path = Path("data/hf_training_dataset.jsonl")
if not path.exists():
    raise SystemExit(f"Missing dataset: {path}. Generate it with download_hf_contracts.py first.")
bad_json = missing_io = rows = 0
for line in path.open():
    if not line.strip():
        continue
    rows += 1
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        bad_json += 1
        continue
    if not item.get("input") or not item.get("output"):
        missing_io += 1

print({"rows": rows, "bad_json": bad_json, "missing_input_or_output": missing_io})
PY
```

A valid training row looks like:

```json
{"input": "<TAC representation>", "output": "<Solidity code>", "metadata": {"function_name": "transfer"}}
```

Use `demo_dataset.jsonl` only for smoke tests. Three examples are not enough for meaningful fine-tuning.

## 3. Generate or refresh training data

The preferred generator is `download_hf_contracts.py`. It reads verified Solidity contracts from Hugging Face `andstor/smart_contracts`, compiles them with compatible `solc` versions, emits TAC with `BytecodeAnalyzer`, deduplicates and filters pairs, and exports JSONL.

```bash
# Quick data-generation test
uv run python download_hf_contracts.py --limit 20

# Larger run with bounded compiler-version expansion
uv run python download_hf_contracts.py --limit 1000 --max-compiler-versions 3

# Full Hugging Face-backed generation
uv run python download_hf_contracts.py
```

Run phases independently when debugging or resuming:

```bash
uv run python download_hf_contracts.py --download-only
uv run python download_hf_contracts.py --compile-only
uv run python download_hf_contracts.py --export-only
```

Useful flags:

| Flag | Default | Use |
|------|---------|-----|
| `--limit N` | all | Cap downloaded contracts for test runs |
| `--max-compiler-versions N` | 5 | Limit compiler-version expansion per contract; optimizer on/off doubles compile jobs |
| `--workers N` | CPU count | Parallel compile workers |
| `--max-body-dupes N` | 2 | Cap repeated normalized Solidity bodies |
| `--min-body-length N` | 50 | Filter short/trivial bodies |
| `--cache-dir PATH` | HuggingFace default | Cache Parquet downloads in a custom directory |
| `--hf-revision REV` | latest | Pin the Hugging Face dataset revision/commit |
| `--download-only` | — | Download contracts without compiling |
| `--compile-only` | — | Compile already-downloaded contracts |
| `--export-only` | — | Export existing pairs to JSONL |
| `--output PATH` | `data/hf_training_dataset.jsonl` | Export JSONL target |
| `--export-selectors PATH` | — | Export selector registry JSON |
| `--import-selectors PATH` | — | Import selector registry JSON before processing |
| `--db PATH` | `data/contracts.db` | SQLite state/cache path |

Expected outputs:

- `data/contracts.db`: downloaded contracts, generated function pairs, and selector registry.
- `data/hf_training_dataset.jsonl`: training-ready TAC-to-Solidity pairs.

If you want to use the older Etherscan path in `train.py`, provide `ETHERSCAN_API_KEY` and a `data/contract_addresses.txt` file, then run without `--skip-collection`. For most runs, prefer the Hugging Face generator followed by `train.py --skip-collection --dataset ...`.

## 4. Kick off training

### Smoke test

Use this to verify the pipeline without spending GPU time on the gated Llama model:

```bash
uv run python train.py \
  --skip-collection \
  --dataset data/hf_training_dataset.jsonl \
  --tiny \
  --skip-eval
```

`--tiny` switches to `facebook/opt-125m`, one epoch, and batch size 2. It is for plumbing validation only.

### Small current-data run

```bash
uv run python train.py \
  --skip-collection \
  --dataset data/hf_training_dataset.jsonl \
  --small
```

`--small` keeps the normal base model but uses one epoch and batch size 2.

### Baseline current-data run

```bash
uv run python train.py \
  --skip-collection \
  --dataset data/hf_training_dataset.jsonl \
  --epochs 3 \
  --batch-size 4 \
  --lr 2e-4 \
  --max-seq-length 2048
```

This re-splits the source JSONL into `data/train_dataset.jsonl`,
`data/val_dataset.jsonl`, and `data/test_dataset.jsonl`, trains the LoRA
adapter, saves to `models/final_model/`, and runs evaluation unless
`--skip-eval` is set. Compiler version and optimizer metadata are included in
prompts by default when dataset rows contain them; use
`--no-compiler-metadata` only for an ablation run.

### Recommended <=8B code model: Qwen2.5-Coder-7B-Instruct

For a stronger code-specialized base model that still fits comfortably on the available RTX 8000 GPUs, use `Qwen/Qwen2.5-Coder-7B-Instruct`. A local full-fp16 LoRA smoke run completed successfully on this repository with a tiny TAC/Solidity subset, batch size 1, one epoch, and `max_seq_length=1024`.

The current CLI has no explicit `--no-quantization` flag; full fp16 requires calling the Python API and setting `use_quantization=False`:

```python
from train import setup_logging, split_dataset, train_model

setup_logging("train_qwen.log")
train_path, val_path, _ = split_dataset(
    "data/hf_training_dataset.jsonl",
    "data",
)

model_path = train_model(
    train_path=train_path,
    val_path=val_path,
    output_dir="models",
    batch_size=1,
    learning_rate=2e-4,
    num_epochs=1,
    max_seq_length=1024,
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    use_quantization=False,
)
print(model_path)
```

For longer runs on this hardware, start with batch size 1-2, sequence length 2048-4096, and compare against the Llama 3.2 3B baseline using the same train/validation/test split.

### Multi-GPU with torchrun

```bash
NGPUS=4 \
DATASET=./data/hf_training_dataset.jsonl \
MODEL=meta-llama/Llama-3.2-3B \
EPOCHS=3 \
LR=2e-4 \
./run_train_torchrun.sh
```

`train_common.sh` auto-detects `MAX_SEQ_LEN` from the dataset and computes `BATCH_SIZE` unless you override them:

```bash
NGPUS=2 MAX_SEQ_LEN=4096 BATCH_SIZE=2 ./run_train_torchrun.sh
```

The wrapper currently passes `--skip-eval`; run evaluation afterward with the eval-only command below.

### DeepSpeed

DeepSpeed is optional and is not installed by default. Sync it only when you need it for larger models or memory-constrained distributed runs.

```bash
uv sync --extra deepspeed
NGPUS=4 DATASET=./data/hf_training_dataset.jsonl ./run_train_deepspeed.sh
```

For Llama 3.2 3B + LoRA, torchrun DDP is usually simpler and faster. Use DeepSpeed primarily for larger models or when ZeRO sharding is needed.

### Resume from a checkpoint

```bash
uv run python train.py \
  --skip-collection \
  --dataset data/hf_training_dataset.jsonl \
  --resume models/checkpoints/checkpoint-1000
```

## 5. Evaluate

Training evaluates automatically unless `--skip-eval` is passed. Results are written to `results/eval_<timestamp>.json`.

Evaluate an existing model:

```bash
uv run python train.py \
  --eval-only \
  --model-path models/final_model \
  --test-dataset data/test_dataset.jsonl
```

Every evaluation writes two artifacts:

- `results/eval_<timestamp>.json` — full machine-readable summary and details.
- `latest_results.txt` — check-in friendly human-readable quality report with
  run metadata, model/config size, training/eval parameters, target checks, and
  replication precision/recall/F1.

For a quick report-generation smoke demo against the latest local model, limit
the eval size:

```bash
uv run python train.py \
  --eval-only \
  --model-path models/final_model_378 \
  --test-dataset data/test_dataset.jsonl \
  --eval-limit 3
```

If you trained with `run_train_torchrun.sh`, evaluate afterward because the wrapper skips evaluation:

```bash
uv run torchrun --nproc_per_node=4 train.py \
  --eval-only \
  --model-path models/final_model \
  --test-dataset data/test_dataset.jsonl
```

Inspect result summaries:

```bash
uv run python - <<'PY'
import glob, json
for path in sorted(glob.glob("results/eval_*.json")):
    with open(path) as f:
        data = json.load(f)
    print(path, json.dumps(data.get("summary", data), indent=2))
PY
```

Key metrics:

| Metric | Good target |
|--------|-------------|
| `semantic_similarity_mean` | > 0.82 |
| `pct_above_0.8_similarity` | > 78% |
| `pct_below_0.4_edit_dist` | > 82% |
| `edit_distance_mean` | lower is better |
| `replication_precision_micro` | higher is better; measures recovered facts that are correct |
| `replication_recall_micro` | higher is better; measures ground-truth facts recovered |
| `replication_f1_micro` | higher is better; balanced structured replication score |

The replication metrics compare structured Solidity facts extracted from the
ground-truth function and generated function: ABI/function facts, visibility,
mutability, modifiers, guards, events, calls, state writes, returns, and control
flow. The evaluation JSON also includes `replication_by_category_micro` so you
can see whether failures are concentrated in ABI recovery, state writes, guards,
calls, or other categories.

## 6. Use the trained model

Fresh training saves to `models/final_model/`.

### Python API: TAC to Solidity

```python
from src.model_setup import SmartContractDecompiler

decompiler = SmartContractDecompiler("models/final_model")

solidity = decompiler.decompile_tac_to_solidity(
    tac_input="function transfer(address to, uint256 amount):\n  // ... TAC ...",
    metadata={
        "compiler_version": "0.8.20",
        "optimizer_enabled": True,
        "optimizer_runs": 200,
    },
    max_new_tokens=1024,
)
print(solidity)
```

Passing compiler metadata is recommended when you know it. Solidity language
semantics and bytecode shape vary by compiler version and optimizer settings,
so the training and inference prompts include `compiler_version`,
`optimizer_enabled`, `optimizer_runs`, and `evm_version` when present.

### Python API: bytecode to contract-level output

```python
from src.model_setup import SmartContractDecompiler

decompiler = SmartContractDecompiler("models/final_model")
result = decompiler.decompile_contract(
    "0x60806040...",
    compiler_version="0.8.20",
    optimizer_enabled=True,
    optimizer_runs=200,
)

print(result["solidity"])
print(result["analysis"])
```

### Existing nonstandard artifact paths

The inspected worktree contained `models/final_model_378/`. You can load it explicitly:

```python
from src.model_setup import SmartContractDecompiler

decompiler = SmartContractDecompiler("models/final_model_378")
```

The web app first checks `WEB_MODEL_PATH`, then `models/final_model/`, then auto-discovers the newest `models/final_model*/` directory containing `model_config.json`. You can also pass a path explicitly:

```bash
WEB_MODEL_PATH=models/final_model_378 uv run python web/app.py
uv run python web/app.py --model-path models/final_model_378
```

### Web app

```bash
uv run python web/app.py
```

Open `http://localhost:5000`. If no model artifact can be resolved, the web app can still analyze bytecode and emit TAC, but model-backed Solidity generation will not be available.

For a shared host, set an API key and restrict CORS to trusted browser
origins. The UI includes an API key field that sends an in-memory
`Authorization: Bearer ...` header to protected endpoints; it does not persist
the key in browser storage.

```bash
WEB_API_KEY='choose-a-long-random-value' \
WEB_HOST=0.0.0.0 \
WEB_CORS_ORIGINS=https://decompiler.example.com \
uv run python web/app.py
```

Without `WEB_API_KEY`, protected API routes are loopback-only. With
`WEB_API_KEY`, `/api/decompile`, security-analysis APIs, and `/api/gpu-stats`
require `Authorization: Bearer <key>` or `X-API-Key: <key>`. GPU telemetry
includes host hardware details, memory, thermals, power, fan, and clock data,
so expose it only to trusted users.

## 7. Preflight checklist

```bash
# CLI help should load
uv run python train.py --help
uv run python download_hf_contracts.py --help

# Repository tests should collect without third-party pytest plugin side effects
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest --collect-only -q

# Dataset should exist and have rows
test -s data/hf_training_dataset.jsonl && wc -l data/hf_training_dataset.jsonl

# GPU visibility
uv run python - <<'PY'
import torch
print({"cuda": torch.cuda.is_available(), "gpus": torch.cuda.device_count()})
PY

# solc versions installed through py-solc-x
uv run python - <<'PY'
from solcx import get_installed_solc_versions
print(get_installed_solc_versions())
PY
```

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| `No dataset found` | `--skip-collection` was used without a JSONL path | Pass `--dataset data/hf_training_dataset.jsonl` or generate data first |
| `HF_TOKEN` or model access error | Base model is gated | Set `HF_TOKEN`, request model access, or use `--tiny` for smoke tests |
| No training pairs generated | Contracts failed parsing/compilation or all pairs were filtered | Check `download_hf_contracts.log`, reduce filters, verify solc install, run with a small `--limit` first |
| CUDA out of memory | Batch/sequence/model too large | Lower `--batch-size`, lower `--max-seq-length`, use `--small`, or choose a smaller model |
| Multi-GPU quantization issues | Bare single-process `uv run python train.py` with quantized model across GPUs | Use `run_train_torchrun.sh` or restrict `CUDA_VISIBLE_DEVICES=0` |
| Web app shows TAC only | `models/final_model/` is absent | Train a model or symlink/copy the trained artifact to `models/final_model/` |
| Evaluation cannot find test data | Split files are absent | Re-run training once with `--skip-collection --dataset ... --skip-eval`, or pass `--test-dataset` explicitly |
