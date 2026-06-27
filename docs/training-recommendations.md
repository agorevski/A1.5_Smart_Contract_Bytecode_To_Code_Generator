# Training Recommendations: Current Pipeline

## Defaults to know first

`train.py` now defaults to a Qwen 7B LoRA recipe, not the old Llama 3B paper
baseline:

| Setting | Default |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| LoRA | enabled, rank 16, alpha 32, dropout 0.1 |
| Target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| Quantization | disabled (`--quantization` opts into 4-bit NF4 loading) |
| Precision | `fp16` (`--precision auto|bf16|fp16|fp32`) |
| Max sequence length | 8192 |
| Per-device batch | 1 |
| Target global batch | 16, with accumulation auto-derived from world size |
| GPU count | 4 for automatic torchrun relaunch when multiple CUDA GPUs are visible |
| Gradient checkpointing | enabled |
| Split ratios | 85% train, 10% validation, 5% test |
| Split/preflight caches | split reuse and tokenization cache enabled; preflight cache under `data/preflight_cache/` |

Compiler/source metadata is retained in JSONL `metadata` for analysis, but
training and inference prompts only include bytecode/TAC-derived metadata.
`--no-compiler-metadata` is a deprecated no-op.

## Recommended commands

After generating or restoring the local HF dataset, use the current
split/preflight pipeline:

```bash
uv run python train.py --skip-collection --dataset ./data/hf_training_dataset.jsonl
```

On a 4x RTX 8000 host, prefer the wrapper when you want explicit DDP launch and
auto sequence/batch sizing:

```bash
./run_train_torchrun.sh
```

Use the measured throughput recipe only when maximizing short-context throughput:

```bash
THROUGHPUT_SWEEP_DEFAULTS=true ./run_train_torchrun.sh
```

DeepSpeed is supported through the wrapper and `ds_config.json`:

```bash
./run_train_deepspeed.sh
```

Evaluate an existing adapter/model:

```bash
uv run python train.py --eval-only \
  --model-path models/final_model \
  --test-dataset data/test_dataset.jsonl \
  --eval-batch-size 4
```

Refresh only splits/preflight without training:

```bash
uv run python train.py --skip-collection \
  --dataset ./data/hf_training_dataset.jsonl \
  --dataset-only
```

## Wrapper environment variables

`train_common.sh` is sourced by both wrapper scripts. Override these with
environment variables:

| Variable | Default | Notes |
|---|---|---|
| `NGPUS` | `4` | Passed to `torchrun --nproc_per_node` or `deepspeed --num_gpus`. |
| `EPOCHS` | `3` | `--epochs`. |
| `LR` | `2e-4` | `--lr`. |
| `DATASET` | `./data/hf_training_dataset.jsonl` | Source JSONL; wrappers still let `train.py` split/reuse splits. |
| `MODEL` | `Qwen/Qwen2.5-Coder-7B-Instruct` | `--model-name`. |
| `PRECISION` | `fp16` | Use `auto` for BF16 on Ampere+ and FP16 elsewhere. |
| `GRADIENT_CHECKPOINTING` | `true` | Emits `--gradient-checkpointing` or `--no-gradient-checkpointing`. |
| `REPORT_TO` | `tensorboard` | Wrapper default; direct `train.py` default is `none`. |
| `SKIP_EVAL` | `false` | Adds `--skip-eval` to skip post-training evaluation. |
| `MAX_SEQ_LEN` | auto | If unset, tokenizer P99 rounded to a power of two and capped by `MAX_SEQ_LEN_CAP`, cached at `data/preflight_cache/sequence_lengths.json`. |
| `MAX_SEQ_LEN_CAP` | `8192` | Upper bound for wrapper sequence-length detection. |
| `BATCH_SIZE` | auto | If unset, `8192 / MAX_SEQ_LEN`, clamped to 1..32; throughput sweep sets 9. |
| `GLOBAL_BATCH_SIZE` | `BATCH_SIZE * NGPUS` | Controls auto gradient accumulation. |
| `THROUGHPUT_SWEEP_DEFAULTS` | `false` | Sets `MAX_SEQ_LEN=256`, `BATCH_SIZE=9`, and matching global batch. |
| `SEQ_LEN_CACHE` | `./data/preflight_cache/sequence_lengths.json` | Wrapper sequence-length detection cache. |
| `DS_CONFIG` | `ds_config.json` | DeepSpeed wrapper only. |
| `HF_TOKEN` | unset | Used for tokenizer/model access where required. |

## Launch mode behavior

- Direct `train.py` auto-relaunches itself with `torchrun --standalone` when
  `--num-gpus > 1`, more than one CUDA GPU is visible, and it is not already in a
  distributed launch. Add `--no-auto-torchrun` or `--num-gpus 1` for a true
  single-process run.
- `run_train_torchrun.sh` always uses `uv run torchrun --nproc_per_node=$NGPUS`.
- `run_train_deepspeed.sh` uses `uv run --extra deepspeed deepspeed` with
  `DS_CONFIG` (default `ds_config.json`). The checked-in config is ZeRO stage 0
  with auto BF16/FP16 fields, auto batch sizes, `steps_per_print: 50`, and no
  wall-clock breakdown.
- Quantized models launched without torchrun/DeepSpeed are restricted to GPU 0
  unless you set `CUDA_VISIBLE_DEVICES`; use distributed launch for multi-GPU
  QLoRA.

## Precision, quantization, and checkpointing

- `--precision auto` chooses BF16 on Ampere+ GPUs (`sm_80+`) and FP16 on older
  CUDA GPUs. RTX 8000 is Turing, so use FP16.
- `--quantization` enables 4-bit NF4 loading through bitsandbytes. Leave it off
  unless VRAM pressure or model size requires QLoRA.
- Gradient checkpointing is on by default for the 8192-token context. Disable it
  only after verifying the target hardware has enough headroom.
- For fixed 8192-token Qwen 7B on 4x RTX 8000, the measured viable recipe is
  QLoRA + FP16 + gradient checkpointing + batch size 1 per GPU + global batch 4.
  Larger per-GPU 8192-token batches OOMed; scale effective batch with gradient
  accumulation instead.

## Data, splits, and preflight

`train.py --skip-collection --dataset <source.jsonl>` treats the dataset as a
full source dataset and writes/reuses `data/train_dataset.jsonl`,
`data/val_dataset.jsonl`, `data/test_dataset.jsonl`, and
`data/split_manifest.json`. It refuses to re-split existing split artifacts
unless `--allow-split-artifact-source` is set.

Useful flags:

| Flag | Use |
|---|---|
| `--force-resplit` | Regenerate splits even if the manifest matches. |
| `--reuse-splits / --no-reuse-splits` | Enable/disable split manifest cache reuse. |
| `--skip-split-validation` | Skip leakage/coverage/split-quality gates. |
| `--min-holdout-stratum-count N` | Require common strata to appear in validation and test. |
| `--allow-degenerate-splits` | Allow leakage-free but highly imbalanced splits. |
| `--skip-data-preflight` | Skip JSONL schema/token-length preflight; use only for legacy smoke tests. |
| `--preflight-tokenizer-download` | Allow tokenizer downloads during preflight. |
| `--allow-whitespace-preflight-fallback` | Use approximate whitespace token counts if tokenizer load fails. |
| `--allow-legacy-metadata-schema` | Permit rows missing `metadata.schema_version`. |

Tokenized datasets are cached by default under `.tokenized_cache` next to the
split file. Use `--no-tokenization-cache`, `--tokenization-cache-dir`, or
`--overwrite-tokenization-cache` to control it.

## Output paths

| Path | Contents |
|---|---|
| `models/checkpoints/checkpoint-*` | Trainer checkpoints; `--resume auto` selects the latest valid one. |
| `models/final_model/` | Final adapter/model, tokenizer, and `model_config.json`. |
| `models/final_model/training_input_manifest.json` | Dataset hashes, token-length/truncation summaries, context-window checks, and effective Trainer arguments. |
| `models/final_model/training_metrics.json` | Final Trainer metrics. |
| `models/final_model/training_log_history.{json,csv}` | Trainer log history. |
| `models/training_throughput.{json,csv}` | Throughput telemetry unless `--no-throughput-metrics`. |
| `models/run_manifests/*.manifest.json` | Run manifests unless overridden by `--manifest-dir` or `--run-manifest`. |
| `results/eval_*.json` | Evaluation summary plus per-example details. |
| `latest_results.txt` | Human-readable latest evaluation report. |

## Evaluation and quality gates

Post-training evaluation runs by default unless `--skip-eval` is set. Train-time
validation is controlled separately by `--train-eval-strategy auto|steps|epoch|no`;
`auto` uses epoch validation for tiny datasets and step validation otherwise.

Evaluation supports seeded sampling (`--eval-limit`, `--eval-seed`), first-N
debug sampling (`--eval-first-n`), batched generation (`--eval-batch-size`), and
multi-GPU sharding when launched with torchrun. Results include semantic
similarity, normalized edit distance, BLEU, ROUGE-L, token accuracy, structural
preservation, structured replication precision/recall/F1, hallucination buckets,
Solidity scaffold/compiler/AST validity, bytecode semantic/deployability signals,
prompt truncation diagnostics, metadata segment metrics, confidence intervals,
worst samples, benchmark-suite summaries, and optional baseline comparisons.

Regression-blocking evaluation:

```bash
uv run python train.py --eval-only \
  --model-path models/final_model \
  --test-dataset data/test_dataset.jsonl \
  --baseline-results results/prior_eval.json \
  --quality-gate
```

`latest_results.txt` currently records a small eval-only snapshot of
`models/final_model_378` using `meta-llama/Llama-3.2-3B`, 4-bit LoRA, max
sequence length 2048, and `--eval-limit 3`. Its semantic similarity mean is
0.7340 and replication F1 mean is 0.6062, but this is a tiny checked-in smoke
result, not a Qwen 7B benchmark.

## Hardware recommendations

### 4x RTX 8000 (48 GB, Turing)

- Use FP16, not BF16.
- Default safe training: `./run_train_torchrun.sh` with tokenizer P99 sequence
  length detection capped at 8192 and memory-scaled batch sizing.
- Throughput sweep recipe: `THROUGHPUT_SWEEP_DEFAULTS=true ./run_train_torchrun.sh`.
- Long-context 8192 recipe:

```bash
uv run torchrun --standalone --nproc_per_node=4 train.py \
  --skip-collection \
  --dataset ./data/hf_training_dataset.jsonl \
  --batch-size 1 \
  --global-batch-size 4 \
  --max-seq-length 8192 \
  --precision fp16 \
  --quantization \
  --gradient-checkpointing \
  --report-to tensorboard \
  --train-eval-strategy no
```

### Single GPU or CPU smoke runs

Use a bounded run and disable auto multi-GPU launch:

```bash
uv run python train.py --skip-collection \
  --dataset ./data/hf_training_dataset.jsonl \
  --num-gpus 1 --no-auto-torchrun \
  --max-steps 10 --train-eval-strategy no --skip-eval
```

For CI-style functionality checks, `--tiny` switches to `facebook/opt-125m` and
small defaults.

### Larger model exploration

The repository supports changing `--model-name`, LoRA settings, quantization,
precision, and DeepSpeed config. Current checked-in quality data does not prove a
32B/70B gain. Treat these as recommendations to validate, not measured results:

1. Start with the default Qwen 7B recipe and establish a full test-set baseline.
2. If quality is insufficient, try code-specialized 14B-32B models with
   `--quantization`, lower per-GPU batch size, and 4096-8192 sequence lengths.
3. Use DeepSpeed ZeRO-2/3 only after editing `ds_config.json` or supplying a new
   config; the checked-in `ds_config.json` is ZeRO-0.
4. Do not compare models on the checked-in `latest_results.txt` smoke result;
   run the same test split, eval limit, generation settings, and quality gate for
   each candidate.
