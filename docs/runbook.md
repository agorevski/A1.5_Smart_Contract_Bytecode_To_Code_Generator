# Operations Runbook

This runbook explains day-to-day operational workflows for the repository as it
exists today: environment setup, data generation/export, bytecode decompilation
and security analysis, the local web app, training, evaluation, data quality
checks, tests, cleanup, and troubleshooting.

## Current operational snapshot

Yes, there is enough local data to run real training/evaluation if
`data/hf_training_dataset.jsonl` is present. The prepared worktree inspected for
this runbook contained:

| Artifact | Current state | Purpose |
|----------|---------------|---------|
| `data/hf_training_dataset.jsonl` | 31,577 rows, ~100 MB | Source dataset for training and re-splitting |
| `data/train_dataset.jsonl` | 25,723 rows | Current train split |
| `data/val_dataset.jsonl` | 3,964 rows | Current validation split |
| `data/test_dataset.jsonl` | 1,890 rows | Current test split |
| `demo_dataset.jsonl` | 55 rows | Tracked smoke-test/demo fixture only |
| `data/contracts.db` | 24,898 contracts, 161,090 function pairs, 69,205 selectors | Hugging Face download/compile/export state |
| `data/tac_lookup.db` | Present | Exact TAC lookup database for inference acceleration/benchmark controls |
| `models/final_model_378/` | Present | Local LoRA model artifact; pass this path explicitly or let the app/CLI auto-discover it |
| `models/checkpoints/checkpoint-*` | `378`, `500`, `1000`, `1359` | Resumable Hugging Face Trainer checkpoints |
| `results/eval_*.json`, `latest_results.txt` | Present | Evaluation outputs and human-readable latest report |
| `results/inference_traces/` | Present | Web/API/CLI inference traces when tracing is enabled |
| `*.log`, `results/ablation/*/*.log` | Present | Operational logs from data generation, training, lookup builds, and historical ablation runs |

That is enough to kick off a real training run and validate the end-to-end
pipeline. It is not enough to claim paper-scale coverage or production-quality
decompilation. The paper-scale target referenced in the codebase is 238,446
TAC-to-Solidity function pairs, so the current 31,577-row exported dataset is
roughly 13% of that scale after deduplication, filtering, and overlength
quarantine.

Generated data, models, results, and logs are gitignored. A clean clone will not
include `data/*.jsonl`, `data/*.db*`, `data/manifests/`,
`data/preflight_cache/`, `models/`, `results/`, or `*.log`; regenerate them
with the commands below. `demo_dataset.jsonl` is tracked for smoke tests, but it
is intentionally too small for meaningful fine-tuning.

## Recommended path

| Goal | Data | Command pattern |
|------|------|-----------------|
| Verify the pipeline quickly | Existing JSONL or small generated set | `uv run python train.py --skip-collection --dataset data/hf_training_dataset.jsonl --tiny --skip-eval` |
| Train a baseline with current data | `data/hf_training_dataset.jsonl` | `uv run python train.py --skip-collection --dataset data/hf_training_dataset.jsonl` |
| Train faster on multiple GPUs | `data/hf_training_dataset.jsonl` | `uv run python train.py --skip-collection --dataset data/hf_training_dataset.jsonl` or `NGPUS=4 DATASET=./data/hf_training_dataset.jsonl ./run_train_torchrun.sh` |
| Build more data first | Hugging Face verified contracts | `uv run python download_hf_contracts.py --limit 1000 --max-compiler-versions 3` |
| Rebuild exact-match lookup | `data/contracts.db` | `uv run python scripts/build_lookup_db.py --source-db data/contracts.db --lookup-db data/tac_lookup.db` |
| Evaluate an existing local model | `models/final_model_378/` and `data/test_dataset.jsonl` | `uv run python train.py --eval-only --model-path models/final_model_378 --test-dataset data/test_dataset.jsonl --eval-limit 3` |
| Run the local UI/API | Bytecode and optional model artifact | `WEB_MODEL_PATH=models/final_model_378 uv run python web/app.py` |
| Reproduce paper-scale intent | Large generated dataset | Scale `download_hf_contracts.py` until exported row counts approach the target, then train/evaluate |

## 1. Install and configure

```bash
uv sync

# Include pytest/black/flake8/mypy when running tests or docs checks
uv sync --dev

# Optional only when using the DeepSpeed wrapper
uv sync --extra deepspeed
```

Requirements:

| Resource | Minimum | Notes |
|----------|---------|-------|
| Python | 3.13.x | Project metadata targets `>=3.13,<3.14` for the validated Linux/GPU training stack |
| GPU for training | CUDA GPU, 16 GB+ VRAM recommended | `--tiny` can run CPU-only but is only a smoke test |
| GPU for inference | CUDA GPU, 4 GB+ VRAM recommended | CPU works but is slow |
| Disk | 10 GB+ | Allow much more for full downloads, checkpoints, traces, and caches |

The locked training stack is validated on Linux x86_64 with NVIDIA driver 580
and CUDA 13 PyTorch wheels. On Quadro RTX 8000 / compute capability 7.5,
training uses FP16 and PyTorch SDPA; Flash Attention 2 is treated as unsupported
unless running on compatible Ampere+ hardware with a separately installed
`flash_attn` package.

Environment variables:

```bash
export HF_TOKEN="hf_..."        # Needed for gated/private HF models or to reduce auth/rate-limit failures
export ETHERSCAN_API_KEY="..."  # Only needed for train.py's older Etherscan collection path
export WEB_API_KEY="..."        # Required for protected/shared web API access
```

You can also copy `src/settings.yaml.example` to `src/settings.yaml` for
`ETHERSCAN_API_KEY` and `HF_TOKEN`, but environment variables are easier to keep
out of source control. Web inference is configured from environment variables
and `web/app.py` CLI flags, not `src/settings.yaml`.

## 2. Check what data you have

Run this before training:

```bash
python - <<'PY'
from pathlib import Path
for path in sorted(Path("data").glob("*.jsonl")) + [Path("demo_dataset.jsonl")]:
    if path.exists():
        rows = sum(1 for line in path.read_text(errors="ignore").splitlines() if line.strip())
        print(f"{rows:>8} {path}")
PY
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

Use `demo_dataset.jsonl` only for smoke tests. Its small tracked fixture is not
enough for meaningful fine-tuning.
Automatic demo fallback is disabled by default; collection runs that produce zero
real pairs fail fast unless `--allow-demo-fallback` is set.

## 3. Generate or refresh training data

The preferred generator is `download_hf_contracts.py`. It reads verified Solidity contracts from Hugging Face `andstor/smart_contracts`, compiles them with compatible `solc` versions, emits TAC with `BytecodeAnalyzer`, deduplicates and filters pairs, validates normalized-body duplicate caps, and exports JSONL plus lineage manifests. For production prompt design, treat compiler version and optimizer fields/comments in generated data as oracle-only and exclude or sanitize them before training.

```bash
# Quick data-generation test
uv run python download_hf_contracts.py --limit 20

# Larger run with bounded compiler-version expansion
uv run python download_hf_contracts.py --limit 1000 --max-compiler-versions 3

# Larger run with explicit manifests and rejects quarantine
uv run python download_hf_contracts.py \
  --limit 1000 \
  --max-compiler-versions 3 \
  --manifest-dir data/manifests \
  --rejects-output data/hf_training_dataset.rejects.jsonl

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
| `--parquet-batch-size N` | 4096 | Rows per streamed Parquet batch during download |
| `--hf-revision REV` | latest | Pin the Hugging Face dataset revision/commit |
| `--download-only` | — | Download contracts without compiling |
| `--compile-only` | — | Compile already-downloaded contracts |
| `--export-only` | — | Export existing pairs to JSONL |
| `--output PATH` | `data/hf_training_dataset.jsonl` | Export JSONL target |
| `--max-seq-length N` | 8192 | Export-time prompt/target length budget |
| `--no-filter-overlength` | off | Keep rows over `--max-seq-length` instead of quarantining them |
| `--rejects-output PATH` | `<output>.rejects.jsonl` | JSONL quarantine for overlength, duplicate, quality, or auxiliary-contract rejects |
| `--export-selectors PATH` | — | Export selector registry JSON |
| `--import-selectors PATH` | — | Import selector registry JSON before processing |
| `--manifest-dir PATH` | next to artifacts | Write download, compile, and export manifests to one directory |
| `--validate-jsonl PATH` | — | Recompute normalized JSONL target-body duplicates and fail if any body exceeds `--max-body-dupes` |
| `--duplicate-sample-limit N` | 5 | Number of duplicate violation/error samples to include in validation output |
| `--db PATH` | `data/contracts.db` | SQLite state/cache path |

Expected outputs:

- `data/contracts.db`: downloaded contracts, generated function pairs, and selector registry.
- `data/hf_training_dataset.jsonl`: training-ready TAC-to-Solidity pairs.
- `data/hf_training_dataset.jsonl.rejects.jsonl` (or `--rejects-output`): quarantined rows and reasons.
- Manifests: `data/hf_download_manifest.json`, `data/hf_compile_manifest.json`, and `data/hf_training_dataset.jsonl.manifest.json` by default, or `hf_download_manifest.json`, `hf_compile_manifest.json`, and `hf_export_manifest.json` under `--manifest-dir`. These capture source/revision lineage, command args, git state, artifact hashes, row counts, typed status/drop counts, duplicate stats, timings, and compile failure diagnostics.

`hf_compile_manifest.json` may legitimately be `completed_with_errors` on large
runs because many verified sources do not compile under the attempted local
compiler configurations. Treat that as actionable only if `function_pairs` and
`rows_exported` are unexpectedly low.

If `--validate-jsonl` or export-time validation fails, inspect the reported top
normalized-body duplicate samples or final-row duplicate samples. Lower
duplicate counts by re-exporting with a stricter cap or by fixing missing/stale
body hashes in `function_pairs`; the validator always recomputes from JSONL
`output` fields, so it catches stale database metadata.

If you want to use the older Etherscan path in `train.py`, provide `ETHERSCAN_API_KEY` and a `data/contract_addresses.txt` file, then run without `--skip-collection`. For most runs, prefer the Hugging Face generator followed by `train.py --skip-collection --dataset ...`.

## 4. Data quality gates before training

Run these checks after a generation/export run and before spending GPU time:

```bash
# Duplicate-cap and final-row duplicate validation; no download/compile work
uv run python download_hf_contracts.py \
  --validate-jsonl data/hf_training_dataset.jsonl \
  --max-body-dupes 2

# Rebuild train/val/test splits, validate leakage/coverage, and run schema/token preflight
uv run python train.py \
  --skip-collection \
  --dataset data/hf_training_dataset.jsonl \
  --dataset-only \
  --allow-whitespace-preflight-fallback

# Force fresh splits when the source export changed but split_manifest.json still exists
uv run python train.py \
  --skip-collection \
  --dataset data/hf_training_dataset.jsonl \
  --dataset-only \
  --force-resplit \
  --allow-whitespace-preflight-fallback
```

`train.py` writes split artifacts to `data/train_dataset.jsonl`,
`data/val_dataset.jsonl`, `data/test_dataset.jsonl`, and
`data/split_manifest.json`. It reuses matching splits by default
(`--reuse-splits`) and regenerates them with `--force-resplit`. Preflight reports
are cached under `data/preflight_cache/` unless `--preflight-cache-dir` points
elsewhere; use `--overwrite-preflight-cache` to recompute.

The CI data-quality gate in `.github/workflows/data-quality.yml` runs
`uv sync --dev --frozen`, targeted CPU-only regression tests for data
generation/export/training CLI behavior, and a deterministic quality-gate report
smoke with Hugging Face and Transformers offline.

## 5. Kick off training

### Smoke test

Use this to verify the pipeline without spending GPU time on the default 7B
model:

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
  --batch-size 1 \
  --lr 2e-4 \
  --max-seq-length 8192
```

This re-splits the source JSONL into `data/train_dataset.jsonl`,
`data/val_dataset.jsonl`, and `data/test_dataset.jsonl`, writes
`data/split_manifest.json`, trains the LoRA adapter, saves to
`models/final_model/`, and runs evaluation unless `--skip-eval` is set. The
current inspected source export splits to 25,723 train rows, 3,964 validation
rows, and 1,890 test rows. The inspected worktree also has a nonstandard
previous artifact at `models/final_model_378/`; pass that path explicitly for
evaluation/inference until a fresh default `models/final_model/` exists. The
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
lengths. By default it fails closed if the requested tokenizer cannot be loaded
from the local cache; add `--preflight-tokenizer-download` to allow downloading
the actual tokenizer or `--allow-whitespace-preflight-fallback` for an explicit
approximate-count override. Use `--skip-data-preflight` only for legacy smoke
runs.

Every `train.py` run writes a structured manifest under
`models/run_manifests/` by default (override with `--manifest-dir` or
`--run-manifest`). The manifest records CLI args, git state, dataset hashes and
split counts, training config, checkpoint resume choice, telemetry artifacts,
final metrics, and evaluation result references. Throughput telemetry is enabled
by default and writes `models/training_throughput.json` plus
`models/training_throughput.csv`; disable it with `--no-throughput-metrics`.
Tokenized dataset caching is enabled by default and resolves to
`data/.tokenized_cache/` unless `--tokenization-cache-dir` is set; use
`--no-tokenization-cache` for one-off runs or `--overwrite-tokenization-cache`
after changing prompt construction.

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
  --model-path models/final_model_378 \
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

For longer runs on this hardware, start from the repository default batch size 1
and sequence length 8192. If the run OOMs, enable `--quantization` first, then
lower sequence length intentionally while keeping the same train/validation/test
split for comparisons.

### Multi-GPU with torchrun

```bash
NGPUS=4 \
DATASET=./data/hf_training_dataset.jsonl \
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct \
EPOCHS=3 \
LR=2e-4 \
./run_train_torchrun.sh
```

`train_common.sh` auto-detects `MAX_SEQ_LEN` from tokenizer counts, computes
`BATCH_SIZE`, runs post-training evaluation, and reports to TensorBoard by
default. Set `SKIP_EVAL=true` only for smoke/sweep runs, and set
`REPORT_TO=wandb` or `REPORT_TO=none` to change Trainer telemetry. To force the
4x RTX 8000 throughput sweep recipe, set `THROUGHPUT_SWEEP_DEFAULTS=true`:

```bash
THROUGHPUT_SWEEP_DEFAULTS=true ./run_train_torchrun.sh
MAX_SEQ_LEN=512 BATCH_SIZE=4 GLOBAL_BATCH_SIZE=16 ./run_train_torchrun.sh
```

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

Use `--resume required` when a preempted job must resume and should fail rather
than silently starting fresh. Checkpoints must include `trainer_state.json`,
`optimizer.pt`, `scheduler.pt`, and model/adapter weights.

## 6. Evaluate

Training evaluates automatically unless `--skip-eval` is passed. Results are written to `results/eval_<timestamp>.json`.

Evaluate an existing model:

```bash
uv run python train.py \
  --eval-only \
  --model-path models/final_model_378 \
  --test-dataset data/test_dataset.jsonl \
  --eval-batch-size 4 \
  --eval-max-new-tokens 256
```

Every evaluation writes two artifacts:

- `results/eval_<timestamp>.json` — full machine-readable summary and details.
- `latest_results.txt` — check-in friendly human-readable quality report with
  run metadata, model/config size, training/eval parameters, executive
  diagnostics, strengths, issue severities/priorities, suggested next
  experiments, target checks, and replication precision/recall/F1.

See [Evaluation diagnostics and model-improvement guide](evaluation-diagnostics.md)
for the triage loop that human reviewers and future LLM agents should follow.

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

If you trained with `SKIP_EVAL=true`, run the eval-only command manually.

Use the quality gate for regression-blocking evaluation. With
`--baseline-results`, baseline deltas are included in the gate and
`--max-baseline-regressions` controls how many regressions are allowed:

```bash
uv run python train.py \
  --eval-only \
  --model-path models/final_model_378 \
  --test-dataset data/test_dataset.jsonl \
  --quality-gate \
  --baseline-results results/eval_1771079247.json
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
| `solidity_valid_mean` | higher is better; generated Solidity passed compiler/AST validation when local solc was available |
| `bytecode_semantic_checked_mean` | should be 100%; rows need opcode/runtime/compiler evidence to count as bytecode-grounded |
| `bytecode_deployable_mean` | should be 100%; scaffold-only syntax checks are not considered deployable |

The replication metrics compare structured Solidity facts extracted from the
ground-truth function and generated function: ABI/function facts, visibility,
mutability, modifiers, guards, events, calls, state writes, returns, and control
flow. The evaluation JSON also includes `replication_by_category_micro` so you
can see whether failures are concentrated in ABI recovery, state writes, guards,
calls, or other categories.

Evaluation helpers now use true normalized Levenshtein distance for
`edit_distance_mean` instead of a `difflib.SequenceMatcher` ratio. Compiler/AST
validity is best-effort and offline: installed local `solc` versions are used
without attempting downloads. Scaffold-only syntax checks are reported
separately and do not satisfy deployability/bytecode-grounded quality gates.
Reusable helpers also produce `metadata_segments` coverage/per-segment metrics,
prompt/truncation diagnostics, deterministic confidence intervals, and
baseline/regression comparisons (`--baseline-results`).

The human report's **Executive Evaluation Readout** answers three questions:

1. What is going well?
2. What needs attention, with severity and priority?
3. What concrete experiment should be tried next?

The same section includes a fenced `Machine-readable diagnostics` JSON block.
Prefer the JSON fields when automating follow-up work:

- `overall_status` — `healthy`, `watch`, `needs_attention`, `blocked`, or
  `insufficient_evidence`.
- `issues[]` — each issue has `category`, `severity`, `priority`, `evidence`,
  likely root causes, and suggested experiments.
- `next_experiments[]` — concrete hypotheses/actions with success metrics.
- `caveats[]` — missing metrics, small samples, or incomplete metadata coverage.

## 7. Use the trained model

Fresh training saves to `models/final_model/`. The current inspected worktree
has `models/final_model_378/`; examples below use that existing artifact where
model-backed generation is required.

### Python API: TAC to Solidity

```python
from src.model_setup import SmartContractDecompiler

decompiler = SmartContractDecompiler("models/final_model_378")

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

decompiler = SmartContractDecompiler("models/final_model_378")
result = decompiler.decompile_contract(
    "0x60806040...",
)

print(result["solidity"])
print(result["analysis"])
```

### Python API: full security-analysis pipeline

There is no separate shell CLI for vulnerability/classification/audit analysis.
Use the web endpoints below or call the orchestrator directly:

```python
from src.pipeline_orchestrator import PipelineConfig, PipelineOrchestrator

pipeline = PipelineOrchestrator(
    PipelineConfig(decompiler_model_path="models/final_model_378")
)
result = pipeline.analyze("0x60806040...", contract_address="")
payload = result.to_dict()

print(payload["stages_completed"])
print(payload.get("classification"))
print(payload.get("vulnerability_summary"))
print(payload.get("audit_report", {}).get("risk_level"))
```

If `decompiler_model_path` is omitted, the orchestrator still performs TAC-only
decompilation plus the configured heuristic classifier/vulnerability/audit
stages.

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
WEB_MODEL_PATH=models/final_model_378 uv run python web/app.py
```

Open `http://localhost:5000`. If no model artifact can be resolved, the web app can still analyze bytecode and emit TAC, but model-backed Solidity generation will not be available.
The readiness panel calls redacted `/api/health` before a job and shows model
availability, lookup DB availability, warmup status, request limits,
killable-timeout mode, and default generation controls without exposing local
model paths or full config. Use `/livez` for minimal public liveness and
protected `/readyz` for internal model path/error/config diagnostics. The
browser validates bytecode/ABI/metadata format before upload, sends a
server-side cancellation request for long jobs, and exposes `.sol`, `.tac`, and
structured JSON downloads after results arrive.

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
    },
    "lookup": {"benchmark_mode": false}
  }'
```

`/api/decompile` accepts ABI either as top-level `abi` or as
`metadata.abi`. ABI JSON may be an array or an artifact object containing an
`abi` array. ABI-derived function signatures override selector provenance as
`source="abi"` and are included in per-function metadata, traces, and JSON
exports. The final result includes `validation`, `function_validation`,
`quality`, `trace_path`, `function_results[*].diagnostics`,
`function_results[*].quality`, lookup provenance for TAC cache hits,
`contract_metadata`, `source_summary`, and `effective_generation_config`.
Set `lookup.benchmark_mode=true` (or `lookup.enabled=false`) to disable exact
TAC lookup for model-quality benchmarks; lookup/model/error counts are reported
separately in `source_summary`.

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
| `WEB_DECOMPILE_TIMEOUT_SECONDS` | `900` | Request deadline for analyzer/model calls; `0` disables |
| `WEB_KILLABLE_INFERENCE_WORKERS` | `true` | Run model calls in terminable worker processes for hard timeout/cancel cleanup |
| `WEB_MAX_NEW_TOKENS` | `4096` | Maximum request generation cap |
| `WEB_DEFAULT_MAX_NEW_TOKENS` | `1024` | UI/API default generation cap |
| `WEB_DEFAULT_TEMPERATURE` | `0.1` | UI/API default |
| `WEB_DEFAULT_DO_SAMPLE` | `false` | UI/API default |
| `WEB_DEFAULT_REPETITION_PENALTY` | `1.15` | UI/API default |
| `WEB_ENABLE_REMOTE_SELECTOR_LOOKUP` / `--remote-selector-lookup` | `false` | Enables 4byte.directory lookups |
| `WEB_TAC_LOOKUP_ENABLED` | `true` | Enables TAC exact-match lookup unless request lookup config disables it |
| `WEB_READYZ_PUBLIC` | `false` | Makes `/readyz` public; keep false on shared hosts |
| `WEB_MOCK_MODEL` / `--mockmodel` | `false` | Use a fake model for web E2E tests without GPU/model artifacts |
| `WEB_INFERENCE_TRACE_ENABLED` | `true` | Writes trace JSON |
| `WEB_INFERENCE_TRACE_INCLUDE_SAMPLES` | `false` | Include raw samples in traces |
| `WEB_INFERENCE_TRACE_DIR` | `results/inference_traces/` | Trace output directory |
| `WEB_MAX_ABI_JSON_CHARS` | `200000` | ABI JSON size limit |
| `WEB_MAX_CONTRACT_METADATA_JSON_CHARS` | `200000` | Metadata JSON size limit |
| `WEB_MAX_ABI_ENTRIES` | `512` | ABI entry count limit |
| `WEB_MODEL_WARMUP_ENABLED` / `--warmup` / `--no-warmup` | `false` | Optional startup warmup |
| `WEB_MODEL_WARMUP_TIMEOUT_SECONDS` | `30` | Warmup deadline |
| `WEB_MODEL_WARMUP_MAX_NEW_TOKENS` | `8` | Warmup generation cap |
| `WEB_LOAD_MODEL_ON_STARTUP` | `false` | Use with WSGI factory to load the model during worker startup |

`/api/health` reports public `limits`, `generation_defaults`, `tracing`,
`warmup`, `model_loaded`, `inference_ready`, and redacted `lookup` state.
`/readyz` returns protected diagnostics including `model_path`, `model_error`,
`model_config`, and active running jobs.

For production, run the WSGI application factory under a container/WSGI server
installed by your deployment image instead of Flask's development server. For
example, if that image includes `gunicorn`:

```bash
WEB_LOAD_MODEL_ON_STARTUP=true \
WEB_MODEL_PATH=models/final_model_378 \
gunicorn 'web.app:create_app()' --bind 0.0.0.0:5000 --workers 1 --timeout 0
```

The TAC lookup builder writes provenance into the database and a JSON manifest.
It logs to `build_lookup_db.log`:

```bash
uv run python scripts/build_lookup_db.py \
  --source-db data/contracts.db \
  --lookup-db data/tac_lookup.db \
  --manifest-path data/tac_lookup.db.manifest.json \
  --decontamination-exclusion validation_split
```

Check an existing lookup database without rebuilding:

```bash
uv run python scripts/build_lookup_db.py --lookup-db data/tac_lookup.db --stats
```

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

## 8. Preflight checklist

```bash
# CLI help should load
uv run python train.py --help
uv run python download_hf_contracts.py --help
uv run python scripts/decompile.py --help
uv run python web/app.py --help

# Repository tests should collect without third-party pytest plugin side effects
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest --collect-only -q

# Docs/runbook drift checks
uv run pytest tests/test_docs_consistency.py -q

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

## 9. Tests and CI

Use the smallest targeted test that covers the workflow you changed:

```bash
# Docs and CLI-help consistency
uv run pytest tests/test_docs_consistency.py -q

# Data generation/export/training quality regressions (matches CI intent)
uv run pytest \
  tests/test_dataset_quality_issues.py \
  tests/test_data_generation_export_issues.py \
  tests/test_model_training_issues.py \
  tests/test_train_cli_issues.py

# CLI and web API smoke coverage
uv run pytest tests/test_decompile_cli.py tests/test_web_app.py

# Full suite when targeted tests expose broader risk
uv run pytest
```

The GitHub Actions workflow `.github/workflows/data-quality.yml` runs on pushes
and pull requests, sets offline/model-safe environment variables
(`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `WANDB_DISABLED=true`,
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`), installs with `uv sync --dev --frozen`, and
runs the CPU data-quality regression set plus a deterministic quality-gate
report smoke.

## 10. Artifact cleanup

Generated artifacts can be large. Preserve manifests/results you need for audit
or comparison before deleting anything.

```bash
# Preview ignored generated files under the main artifact trees
git clean -ndX data models results

# Remove ignored generated files under data/, models/, and results/
git clean -fdX data models results

# Remove common generated logs that may not be covered by the tree clean
rm -f build_lookup_db.log download_hf_contracts.log train.log scripts/build_lookup_db.log

# Remove local Python/test caches
rm -rf .pytest_cache .mypy_cache __pycache__ src/__pycache__ scripts/__pycache__ tests/__pycache__

# Remove tokenization/preflight caches if they were created outside ignored paths
rm -rf data/.tokenized_cache data/preflight_cache
```

After cleanup, regenerate in order: `download_hf_contracts.py`,
`train.py --dataset-only`, training/evaluation, then `scripts/build_lookup_db.py`
if the web/CLI lookup cache is needed.

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| `No dataset found` | `--skip-collection` was used without a JSONL path | Pass `--dataset data/hf_training_dataset.jsonl` or generate data first |
| Demo dataset refused | `demo_dataset.jsonl` was auto-detected without explicit opt-in | Provide a real `--dataset` or add `--allow-demo-fallback` for smoke-only runs |
| `HF_TOKEN` or model access error | Base model is gated/private or local cache is missing | Set `HF_TOKEN`, request model access, use `--preflight-tokenizer-download`, or use `--tiny` for smoke tests |
| Preflight tokenizer failure | Offline/local cache cannot load the requested tokenizer | Use `--preflight-tokenizer-download` when network access is allowed, or `--allow-whitespace-preflight-fallback` for an explicit approximate check |
| No training pairs generated | Contracts failed parsing/compilation or all pairs were filtered | Check `download_hf_contracts.log`, `data/manifests/hf_compile_manifest.json`, and rejects JSONL; reduce filters; verify solc install; run with a small `--limit` first |
| `hf_compile_manifest.json` is `completed_with_errors` | Some verified contracts failed attempted local compiler configs | Expected on large runs if exported rows are healthy; inspect diagnostics only when pair/export counts are too low |
| Many rows quarantined | Overlength, duplicate, TAC-quality, or auxiliary-contract filters fired | Inspect `<output>.rejects.jsonl` and `hf_export_manifest.json`; adjust `--max-seq-length`, caps, or source data intentionally |
| CUDA out of memory | Batch/sequence/model too large | Lower `--batch-size`, lower `--max-seq-length`, use `--small`, or choose a smaller model |
| Multi-GPU quantization issues | Bare single-process `uv run python train.py` with quantized model across GPUs | Use `run_train_torchrun.sh` or restrict `CUDA_VISIBLE_DEVICES=0` |
| CLI says no trained model artifact | No auto-discoverable `models/final_model*/model_config.json` and no `--model-path` | Pass `--model-path models/final_model_378`, set `WEB_MODEL_PATH`, train a model, or use `--format tac` |
| Web app shows TAC only | Model artifact is absent or failed to load | Pass `--model-path models/final_model_378`, set `WEB_MODEL_PATH`, or train a fresh `models/final_model/` |
| Web/API returns 403 | Non-loopback request without configured API key, or wrong key | Set `WEB_API_KEY` and send `Authorization: Bearer <key>` or `X-API-Key: <key>` |
| Benchmark numbers look too good | Exact TAC lookup was enabled | Set request `lookup.benchmark_mode=true`, `lookup.enabled=false`, or `WEB_TAC_LOOKUP_ENABLED=false` |
| Evaluation cannot find test data | Split files are absent or cached test lineage is unverified | Re-run dataset splitting with `--skip-collection --dataset ... --dataset-only`, pass `--test-dataset` explicitly, or use `--allow-unverified-test-dataset` only for audits |
| Quality gate failed | Metrics missed thresholds or baseline regression limit | Inspect `latest_results.txt` and `results/eval_<timestamp>.json`; adjust thresholds only when intentionally changing evaluation criteria |
