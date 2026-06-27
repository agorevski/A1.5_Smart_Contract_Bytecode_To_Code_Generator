# GPU Parameter Sweep Results

## Scope

This document separates measured GPU sweep results from operational
recommendations. The measured sweeps were short synthetic Qwen 7B LoRA trials on
4x NVIDIA RTX 8000 GPUs. They are throughput/memory data, not model-quality
benchmarks. The checked-in `latest_results.txt` is a tiny eval-only Llama 3B
smoke result and is not used as GPU sweep evidence.

## Short-context recommendation

Use the opt-in throughput recipe wired into the wrappers when the goal is maximum
short-context throughput on 4x RTX 8000:

```bash
THROUGHPUT_SWEEP_DEFAULTS=true ./run_train_torchrun.sh
```

Equivalent explicit command:

```bash
uv run torchrun --nproc_per_node=4 train.py \
  --skip-collection \
  --dataset ./data/hf_training_dataset.jsonl \
  --batch-size 9 \
  --global-batch-size 36 \
  --max-seq-length 256 \
  --precision fp16 \
  --report-to tensorboard \
  --no-gradient-checkpointing
```

Recommended defaults from the measured short-context sweep:

| Parameter | Value | Basis |
|---|---:|---|
| GPUs | 4 | Uses all available RTX 8000 GPUs through DDP. |
| Per-GPU batch size | 9 | Highest successful batch at `seq_len=256`; batch 10 OOMed. |
| Global batch size | 36 | Keeps gradient accumulation at 1 on 4 GPUs. |
| Max sequence length | 256 | Best observed memory/utilization point under the short sweep cap. |
| Precision | fp16 | RTX 8000 is Turing; BF16 is not the native fast path. |
| Gradient checkpointing | off | Checkpointing reduced memory use and did not complete within the short sweep window. |
| Quantization | off | Short-context result was measured with full fp16 base loading plus LoRA. |

If context length matters more than short-context throughput, use
`MAX_SEQ_LEN=512 BATCH_SIZE=4 GLOBAL_BATCH_SIZE=16 ./run_train_torchrun.sh`.
That was the largest successful 512-token configuration in the short sweep.

## Locked 8192-token recommendation

For fixed `max_seq_length=8192` on 4x RTX 8000, the measured viable setup is
4-bit QLoRA, fp16 compute, gradient checkpointing, and per-GPU micro-batch size
1. Full fp16 LoRA OOMed at batch size 1, QLoRA without checkpointing OOMed, and
QLoRA batch size 2 also OOMed.

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

Use gradient accumulation, not a larger per-GPU micro-batch, if a larger
effective global batch is required.

## Current repository wiring

- `train.py` data-safe defaults: `batch_size=1`, `global_batch_size=16`,
  `max_seq_length=8192`, `precision=fp16`, `quantization=false`, and gradient
  checkpointing enabled.
- Direct `train.py` auto-relaunches with `torchrun --standalone` when
  `--num-gpus > 1`, more than one CUDA GPU is visible, and it is not already in a
  distributed launch. Use `--num-gpus 1` or `--no-auto-torchrun` to disable that.
- `train_common.sh`: when `THROUGHPUT_SWEEP_DEFAULTS=true`, sets
  `MAX_SEQ_LEN=256`, `BATCH_SIZE=9`, `GLOBAL_BATCH_SIZE=BATCH_SIZE * NGPUS`,
  `PRECISION=fp16`, and `GRADIENT_CHECKPOINTING=false` unless explicitly
  overridden.
- Without throughput defaults, wrappers auto-detect P99 tokenizer sequence length
  and cache it at `data/preflight_cache/sequence_lengths.json`; batch size is
  `8192 / MAX_SEQ_LEN`, clamped to 1..32.
- `run_train_torchrun.sh` passes the resolved batch/global batch, sequence
  length, precision, gradient-checkpointing, reporting, and optional skip-eval
  settings to `train.py`.
- `run_train_deepspeed.sh` uses `DS_CONFIG` (default `ds_config.json`). The
  checked-in config is ZeRO stage 0 with auto BF16/FP16 fields and auto batch
  sizes; it is not a ZeRO-2/3 large-model config.
- `src/model_setup.py` keeps inference prompt context at least 2048 tokens even
  if a short sweep length such as 256 is saved in model config.

## Sweep method

The sweeps intentionally avoided full dataset runs so each trial stayed near or
under one minute.

The default short-context sweep used:

- `torchrun --standalone --nproc_per_node=4`
- `Qwen/Qwen2.5-Coder-7B-Instruct`
- fp16 full-precision base model loading with LoRA enabled
- LoRA rank 16, alpha 32, dropout 0.1, default target modules
- one synthetic forward/backward/optimizer step after model load
- `nvidia-smi` sampling all four GPUs every 0.5 seconds
- no batch sizes below 4

The locked 8192-token follow-up used the same model, LoRA settings, DDP
launcher, and fp16 compute path, but used one synthetic 8192-token batch per rank
and swept quantization/checkpointing because full-precision LoRA did not fit. The
follow-up table reports `torch.cuda` peak allocated/reserved memory instead of
`nvidia-smi` sampling.

Wall time includes process startup and model load. Step time is the measured
optimizer step after model load.

## Measured results

### Locked `seq_len=8192` follow-up

| Batch/GPU | Seq len | Quantization | Grad checkpointing | Status | Wall time (s) | Step time (s) | Global tokens/step | Tokens/s | Peak allocated/GPU (MiB) | Peak reserved/GPU (MiB) |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 8192 | off | off | OOM | 27.63 | - | - | - | - | - |
| 1 | 8192 | off | on | OOM | 37.54 | - | - | - | - | - |
| 1 | 8192 | on | off | OOM | 25.83 | - | - | - | - | - |
| 1 | 8192 | on | on | ok | 49.25 | 33.55 | 32,768 | 977 | 45,137 | 46,296 |
| 2 | 8192 | on | on | OOM | 46.26 | - | - | - | - | - |
| 4 | 8192 | on | on | OOM | 24.61 | - | - | - | - | - |

### Default short-context sweep

| Batch/GPU | Seq len | Grad checkpointing | Status | Wall time (s) | Step time (s) | Peak allocated/GPU (MiB) | Max `nvidia-smi` memory (MiB) | Max GPU util |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 4 | 256 | off | ok | 31.49 | 2.45 | 28,305 | 29,039 | 100% |
| 6 | 256 | off | ok | 56.73 | 2.73 | 35,036 | 35,867 | 100% |
| 8 | 256 | off | ok | 58.49 | 3.48 | 41,768 | 42,689 | 100% |
| 9 | 256 | off | ok | 56.55 | 3.77 | 45,134 | 46,113 | 100% |
| 10 | 256 | off | OOM | 33.91 | - | - | 46,147 | 100% |
| 4 | 512 | off | ok | 59.84 | 3.41 | 43,336 | 44,409 | 100% |
| 5 | 512 | off | OOM | 59.03 | - | - | 48,267 | 100% |
| 4 | 576 | off | timeout | 60.16 | - | - | 48,399 | 100% |
| 4 | 640 | off | OOM | 57.21 | - | - | 47,707 | 100% |
| 4 | 1024 | off | OOM | 44.28 | - | - | 48,127 | 100% |
| 4 | 512 | on | timeout | 55.20 | - | - | 19,879 | 100% |
| 4 | 1024 | on | timeout | 55.17 | - | - | 24,557 | 100% |

## Interpretation

`batch_size=9`, `max_seq_length=256`, and `--no-gradient-checkpointing` is the
best measured short-context recipe for filling 4x RTX 8000 memory without OOM. It
reached 100% observed GPU utilization, used about 46.1 GiB of the 48 GiB
available per GPU, and batch size 10 failed with CUDA OOM.

The `seq_len=512, batch_size=4` alternative also completed and reached 100%
utilization, but it used less memory and processed fewer tokens per measured step
than `seq_len=256, batch_size=9`. Gradient checkpointing should stay off for the
short-context throughput recipe because it reduced memory use and did not finish
inside the short sweep cap.

At 8192 tokens, batch size 1 per GPU is the practical ceiling on this hardware,
and only with QLoRA plus gradient checkpointing. Increase effective batch size
through accumulation, and validate quality separately with the eval/quality-gate
pipeline.
