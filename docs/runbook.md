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
| Train faster on multiple GPUs | `data/hf_training_dataset.jsonl` | `uv run python train.py --skip-collection --dataset data/hf_training_dataset.jsonl` or `NGPUS=4 DATASET=./data/hf_training_dataset.jsonl ./run_train_torchrun.sh` |
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

Validate that a JSONL file contains usable training pairs. `train.py` now runs
this schema and token-length preflight automatically before training/eval-only;
the snippet below is only a quick manual check:

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

A production-aligned training row keeps prompt inputs bytecode-derived. If a
selector database supplies a name/signature guess, mark it inferred:

```json
{"input": "<TAC/CFG from bytecode>", "output": "<Solidity code>", "metadata": {"function_selector": "0xa9059cbb", "signature_guess": "transfer(address,uint256)", "signature_guess_inferred": true}}
```

Use `demo_dataset.jsonl` only for smoke tests. Three examples are not enough for meaningful fine-tuning.
Automatic demo fallback is disabled by default; collection runs that produce zero
real pairs fail fast unless `--allow-demo-fallback` is set.

## 3. Generate or refresh training data

The preferred generator is `download_hf_contracts.py`. It reads verified Solidity contracts from Hugging Face `andstor/smart_contracts`, compiles them with compatible `solc` versions, emits TAC with `BytecodeAnalyzer`, deduplicates and filters pairs, validates normalized-body duplicate caps, and exports JSONL plus lineage manifests. For production prompt design, treat compiler version and optimizer fields/comments in generated data as oracle-only and exclude or sanitize them before training.

```bash
# Quick data-generation test
uv run python download_hf_contracts.py --limit 20

# Larger run with bounded compiler-version expansion
uv run python download_hf_contracts.py --limit 1000 --max-compiler-versions 3

# Full Hugging Face-backed generation
uv run python download_hf_contracts.py

# Keep all phase manifests together
uv run python download_hf_contracts.py --limit 1000 --manifest-dir data/manifests
```

Run phases independently when debugging or resuming:

```bash
uv run python download_hf_contracts.py --download-only
uv run python download_hf_contracts.py --compile-only
uv run python download_hf_contracts.py --export-only

# Validate an existing export without downloading or compiling
uv run python download_hf_contracts.py \
  --validate-jsonl data/hf_training_dataset.jsonl \
  --max-body-dupes 2
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
| `--manifest-dir PATH` | next to artifacts | Write download, compile, and export manifests to one directory |
| `--validate-jsonl PATH` | — | Recompute normalized JSONL target-body duplicates and fail if any body exceeds `--max-body-dupes` |
| `--duplicate-sample-limit N` | 5 | Number of duplicate violation/error samples to include in validation output |
| `--db PATH` | `data/contracts.db` | SQLite state/cache path |

Expected outputs:

- `data/contracts.db`: downloaded contracts, generated function pairs, and selector registry.
- `data/hf_training_dataset.jsonl`: training-ready TAC-to-Solidity pairs.
- Manifests: `data/hf_download_manifest.json`, `data/hf_compile_manifest.json`, and `data/hf_training_dataset.jsonl.manifest.json` (or files under `--manifest-dir`). These capture source/revision lineage, command args, git state, artifact hashes, row counts, typed status/drop counts, duplicate stats, timings, and compile failure diagnostics.

If `--validate-jsonl` or export-time validation fails, inspect the reported top normalized-body duplicate samples. Lower duplicate counts by re-exporting with a stricter cap or by fixing missing/stale body hashes in `function_pairs`; the validator always recomputes from JSONL `output` fields, so it catches stale database metadata.

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
`data/val_dataset.jsonl`, and `data/test_dataset.jsonl`, writes
`data/split_manifest.json`, trains the LoRA adapter, saves to
`models/final_model/`, and runs evaluation unless `--skip-eval` is set. The
splitter preserves leakage-connected groups (source hash, contract address,
contract+selector/signature, exact input hash, exact output hash), validates no
overlap across train/val/test, and reports holdout coverage by compiler version,
optimizer, visibility, source, length bucket, and function family. Production
inference only has Etherscan **Contract > Bytecode**, so training prompts
always ignore true compiler/optimizer metadata and sanitize legacy TAC inputs:

```bash
uv run python train.py \
  --skip-collection \
  --dataset data/hf_training_dataset.jsonl
```

`--no-compiler-metadata` is retained as a deprecated no-op for old scripts.
If an older JSONL embeds `// Compiler:` or source/ABI annotations inside the
TAC `input`, the training prompt builder strips those lines before tokenization;
new exports also sanitize TAC prompt inputs.

Before training or eval-only, `train.py` validates JSONL schema and tokenizer
lengths. By default it uses a cached tokenizer when available and falls back to
whitespace counts without downloading; add `--preflight-tokenizer-download` to
force loading the actual tokenizer if it is not cached. Use
`--skip-data-preflight` only for legacy smoke runs.

Every `train.py` run writes a structured manifest under
`models/run_manifests/` by default (override with `--manifest-dir` or
`--run-manifest`). The manifest records CLI args, git state, dataset hashes and
split counts, training config, checkpoint resume choice, telemetry artifacts,
final metrics, and evaluation result references. Throughput telemetry is enabled
by default and writes `models/training_throughput.json` plus
`models/training_throughput.csv`; disable it with `--no-throughput-metrics`.

The default training recipe is `Qwen/Qwen2.5-Coder-7B-Instruct` with LoRA on 4
GPUs. When multiple CUDA devices are visible, `train.py` automatically relaunches
itself with `torchrun --nproc_per_node=4` unless `--num-gpus 1` or
`--no-auto-torchrun` is provided. Gradient accumulation is auto-computed from
`--global-batch-size` (default 16), so scaling to 4 GPUs keeps the effective
batch comparable to the single-GPU recipe instead of silently quadrupling it.

### Production bytecode-only prompt metadata

Production inference starts from Etherscan **Contract > Bytecode** only. Do not
design prompts around true `compiler_version`, `solc_version`,
`optimizer_enabled`, `optimizer_runs`, or `evm_version`; those are verified-source
oracle fields, not bytecode-only inputs.

Safe high-value prompt inputs are derived from the bytecode or from selector
databases:

- TAC/CFG text, function selector, and selector/signature guesses marked
  inferred with their source.
- Basic block and branch counts.
- TAC/opcode counts.
- Storage read/write counts.
- External call counts.
- Event/log and revert counts.
- Bytecode length, instruction count, and function count.

For inference, omit compiler flags and inspect the bytecode-derived TAC/analysis:

```bash
uv run python scripts/decompile.py \
  --model-path models/final_model \
  --bytecode 0x60806040... \
  --format json

uv run python scripts/decompile.py --format tac --bytecode 0x60806040...
```

Use `--no-bytecode-metadata` only if you want to disable the compact
bytecode/TAC-derived count line; TAC/CFG remains the core model input and TAC
sanitization still runs. Legacy inference flags such as `--compiler-version`,
`--optimizer-enabled`, `--optimizer-runs`, and `--evm-version` are deprecated
no-ops and are ignored.

### Oracle-only compiler metadata study

`scripts/run_compiler_metadata_ablation.py` is deprecated for production prompt
design. Current prompt code ignores true compiler/optimizer values and sanitizes
legacy TAC inputs, so the script is retained only as a historical research
artifact and no longer creates a real compiler-metadata control. Do not use the
old oracle result to justify compiler metadata in production prompts.

### Recommended <=8B code model: Qwen2.5-Coder-7B-Instruct

`Qwen/Qwen2.5-Coder-7B-Instruct` is now the default base model. A local full-fp16
LoRA smoke run completed successfully on this repository with a tiny TAC/Solidity
subset, batch size 1, one epoch, and `max_seq_length=1024`.

Full-precision LoRA is the default to avoid quantization-related quality changes.
Use `--quantization` only when VRAM pressure requires 4-bit NF4 loading:

```bash
uv run python train.py \
  --skip-collection \
  --dataset data/hf_training_dataset.jsonl \
  --num-gpus 4 \
  --global-batch-size 16

uv run python train.py \
  --skip-collection \
  --dataset data/hf_training_dataset.jsonl \
  --num-gpus 4 \
  --quantization
```

For longer runs on this hardware, start with batch size 1-2, sequence length
2048-4096, and compare against the Llama 3.2 3B paper baseline using the same
train/validation/test split.

### Multi-GPU with torchrun

```bash
NGPUS=4 \
DATASET=./data/hf_training_dataset.jsonl \
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct \
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

For Qwen2.5-Coder-7B + LoRA, torchrun DDP is usually simpler and faster. Use DeepSpeed primarily for larger models or when ZeRO sharding is needed.

### Resume from a checkpoint

```bash
uv run python train.py \
  --skip-collection \
  --dataset data/hf_training_dataset.jsonl \
  --resume models/checkpoints/checkpoint-1000
```

To safely resume the latest Hugging Face `Trainer` checkpoint under
`--output-dir`, use:

```bash
uv run python train.py \
  --skip-collection \
  --dataset data/hf_training_dataset.jsonl \
  --output-dir models \
  --resume auto
```

## 5. Evaluate

Training evaluates automatically unless `--skip-eval` is passed. Results are written to `results/eval_<timestamp>.json`.

Evaluate an existing model:

```bash
uv run python train.py \
  --eval-only \
  --model-path models/final_model \
  --test-dataset data/test_dataset.jsonl \
  --eval-batch-size 4
```

Every evaluation writes two artifacts:

- `results/eval_<timestamp>.json` — full machine-readable summary and details.
- `latest_results.txt` — check-in friendly human-readable quality report with
  run metadata, model/config size, training/eval parameters, target checks, and
  replication precision/recall/F1.

`--eval-batch-size 1` preserves the historical per-example path. Larger values
use `SmartContractDecompiler.decompile_batch` in chunks and fall back to
single-example retries if a batch hits CUDA OOM.

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
| `solidity_valid_mean` | higher is better; generated Solidity passed compiler/AST validation when local solc was available, otherwise the deterministic scaffold check |

The replication metrics compare structured Solidity facts extracted from the
ground-truth function and generated function: ABI/function facts, visibility,
mutability, modifiers, guards, events, calls, state writes, returns, and control
flow. The evaluation JSON also includes `replication_by_category_micro` so you
can see whether failures are concentrated in ABI recovery, state writes, guards,
calls, or other categories.

Evaluation helpers now use true normalized Levenshtein distance for
`edit_distance_mean` instead of a `difflib.SequenceMatcher` ratio. Compiler/AST
validity is best-effort and offline: installed local `solc` versions are used
without attempting downloads; if no matching compiler is available, the report
falls back to an explicit Solidity scaffold/syntax check. Reusable helpers also
produce `metadata_segments` coverage/per-segment metrics plus deterministic
mean confidence intervals and baseline/regression comparisons for future CLI
wiring.

## 6. Use the trained model

Fresh training saves to `models/final_model/`.

### Python API: TAC to Solidity

```python
from src.model_setup import SmartContractDecompiler

decompiler = SmartContractDecompiler("models/final_model")

solidity = decompiler.decompile_tac_to_solidity(
    tac_input="function transfer(address to, uint256 amount):\n  // ... TAC ...",
    metadata={
        "function_name": "transfer",
        "signature_guess_inferred": True,
        "signature_guess_source": "selector_database",
    },
    max_new_tokens=1024,
)
print(solidity)
```

Only pass prompt metadata that is available from bytecode analysis or selector
databases. Treat function names/signatures as guesses unless they came from a
verified selector mapping, and do not pass true compiler/optimizer settings for
production bytecode-only inference.

### CLI: bytecode to TAC or Solidity

Use `scripts/decompile.py` for shell, CI, and batch inference. JSON output is
machine-readable and includes generated Solidity, TAC, analysis timings,
function errors, Solidity validation diagnostics, request metadata, generation
controls, and model config. If `--model-path` is omitted, the CLI uses the same
model resolution order as the web app: `WEB_MODEL_PATH`, `models/final_model/`,
then the newest `models/final_model*/` artifact containing `model_config.json`:

```bash
uv run python scripts/decompile.py \
  --bytecode 0x60806040... \
  --max-new-tokens 1024 \
  --timeout-seconds 900 \
  --max-functions 128 \
  --format json
```

For static-analysis-only debugging, emit TAC without loading a model:

```bash
uv run python scripts/decompile.py --format tac --bytecode 0x60806040...
```

### Python API: bytecode to contract-level output

```python
from src.model_setup import SmartContractDecompiler

decompiler = SmartContractDecompiler("models/final_model")
result = decompiler.decompile_contract(
    "0x60806040...",
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

The web app and CLI first check `WEB_MODEL_PATH`, then `models/final_model/`,
then auto-discover the newest `models/final_model*/` directory containing
`model_config.json`. You can also pass a path explicitly:

```bash
WEB_MODEL_PATH=models/final_model_378 uv run python web/app.py
uv run python web/app.py --model-path models/final_model_378
uv run python scripts/decompile.py --model-path models/final_model_378 --bytecode 0x...
```

### Web app

```bash
uv run python web/app.py
```

Open `http://localhost:5000`. If no model artifact can be resolved, the web app can still analyze bytecode and emit TAC, but model-backed Solidity generation will not be available.
The readiness panel calls `/api/health` before a job and shows whether the
model is loaded, which model path/config is effective, lookup DB availability,
warmup status, request limits, hard-timeout mode, and default generation
controls. The browser validates bytecode/ABI/metadata format before upload,
provides an explicit cancel button for long jobs, and exposes `.sol`, `.tac`,
and structured JSON downloads after results arrive.

The `/api/decompile` endpoint streams SSE progress and a final result. Each
request includes a `request_id`; when `WEB_INFERENCE_TRACE_ENABLED=true`
(default), a JSON trace is written under `results/inference_traces/` with
bytecode hashes, per-function lookup/model/ABI provenance, token/truncation
diagnostics, validation results, generation settings, timings, and errors. Raw
bytecode/TAC samples are omitted unless `WEB_INFERENCE_TRACE_INCLUDE_SAMPLES=true`.

```bash
curl -N http://127.0.0.1:5000/api/decompile \
  -H 'Content-Type: application/json' \
  -d '{
    "bytecode": "0x60806040...",
    "abi": [
      {
        "type": "function",
        "name": "transfer",
        "inputs": [
          {"name": "to", "type": "address"},
          {"name": "amount", "type": "uint256"}
        ],
        "outputs": [],
        "stateMutability": "nonpayable"
      }
    ],
    "metadata": {"contractName": "Token"},
    "generation": {
      "max_new_tokens": 512,
      "temperature": 0.1,
      "do_sample": false,
      "repetition_penalty": 1.15
    }
  }'
```

`/api/decompile` accepts ABI either as top-level `abi` or as
`metadata.abi`. ABI JSON may be an array or an artifact object containing an
`abi` array. ABI-derived function signatures override selector provenance as
`source="abi"` and are included in per-function metadata, traces, and JSON
exports. The final result includes `validation`, `function_validation`,
`trace_path`, `function_results[*].diagnostics`, `contract_metadata`,
`source_summary`, and `effective_generation_config`.

Supported web inference configuration is environment/CLI driven; web requests do
not read `src/settings.yaml`.

| Knob | Default | Notes |
|------|---------|-------|
| `WEB_MODEL_PATH` / `--model-path` | auto | Model path precedence before `models/final_model/` and newest `models/final_model*/` |
| `WEB_HOST` / `--host` | `127.0.0.1` | Bind address |
| `--port` | `5000` | Flask port |
| `--debug` | off | Flask debug mode |
| `WEB_API_KEY` | unset | Required for non-loopback/protected access when set |
| `WEB_CORS_ORIGINS` | loopback origins | Comma-separated origins or `*` |
| `WEB_MAX_BYTECODE_HEX_LENGTH` | `200000` | Request bytecode hex limit |
| `WEB_MAX_CONCURRENT_DECOMPILES` | `1` | Bounded semaphore capacity |
| `WEB_MAX_DECOMPILE_FUNCTIONS` | `128` | Function work cap |
| `WEB_DECOMPILE_TIMEOUT_SECONDS` | `900` | Hard daemon-thread timeout around analyzer/model calls; `0` disables |
| `WEB_MAX_NEW_TOKENS` | `4096` | Maximum request generation cap |
| `WEB_DEFAULT_MAX_NEW_TOKENS` | `1024` | UI/API default generation cap |
| `WEB_DEFAULT_TEMPERATURE` | `0.1` | UI/API default |
| `WEB_DEFAULT_DO_SAMPLE` | `false` | UI/API default |
| `WEB_DEFAULT_REPETITION_PENALTY` | `1.15` | UI/API default |
| `WEB_ENABLE_REMOTE_SELECTOR_LOOKUP` / `--remote-selector-lookup` | `false` | Enables 4byte.directory lookups |
| `WEB_INFERENCE_TRACE_ENABLED` | `true` | Writes trace JSON |
| `WEB_INFERENCE_TRACE_INCLUDE_SAMPLES` | `false` | Include raw samples in traces |
| `WEB_INFERENCE_TRACE_DIR` | `results/inference_traces/` | Trace output directory |
| `WEB_MAX_ABI_JSON_CHARS` | `200000` | ABI JSON size limit |
| `WEB_MAX_CONTRACT_METADATA_JSON_CHARS` | `200000` | Metadata JSON size limit |
| `WEB_MAX_ABI_ENTRIES` | `512` | ABI entry count limit |
| `WEB_MODEL_WARMUP_ENABLED` / `--warmup` / `--no-warmup` | `false` | Optional startup warmup |
| `WEB_MODEL_WARMUP_TIMEOUT_SECONDS` | `30` | Warmup deadline |
| `WEB_MODEL_WARMUP_MAX_NEW_TOKENS` | `8` | Warmup generation cap |

`/api/health` reports these as `limits`, `generation_defaults`, `tracing`,
`warmup`, `model_path`, `model_loaded`, `inference_ready`, and `lookup`.

The UI also exposes the security-analysis APIs using the same bytecode/API-key
inputs:

```bash
curl http://127.0.0.1:5000/api/vulnerability-scan \
  -H 'Content-Type: application/json' \
  -d '{"bytecode":"0x60806040..."}'

curl http://127.0.0.1:5000/api/classify \
  -H 'Content-Type: application/json' \
  -d '{"bytecode":"0x60806040..."}'

curl http://127.0.0.1:5000/api/audit-report \
  -H 'Content-Type: application/json' \
  -d '{"bytecode":"0x60806040..."}'
```

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
