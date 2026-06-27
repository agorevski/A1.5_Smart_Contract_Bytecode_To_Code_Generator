# GPU Parameter Sweep Results

## Recommendation

Use the opt-in throughput recipe now configured in the repo:

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
  --no-gradient-checkpointing \
  --skip-eval
```

Recommended defaults:

| Parameter | Default | Reason |
|---|---:|---|
| GPUs | 4 | Uses all available RTX 8000 GPUs through `torchrun` DDP. |
| Per-GPU batch size | 9 | Highest successful batch at `seq_len=256`; batch 10 OOMed. |
| Global batch size | 36 | Keeps gradient accumulation at 1 on 4 GPUs. |
| Max sequence length | 256 | Best observed memory/utilization point under the 1-minute sweep cap. |
| Precision | fp16 | RTX 8000 is Turing generation; fp16 is the fast mixed-precision path. |
| Gradient checkpointing | off | Disabling it fills VRAM and speeds steps; checkpointing under-used memory and timed out in short trials. |
| Quantization | off | Full fp16 LoRA used much more GPU memory and throughput than 4-bit loading would. |

If sequence context is more important than maximum throughput, use `MAX_SEQ_LEN=512 BATCH_SIZE=4 GLOBAL_BATCH_SIZE=16 ./run_train_torchrun.sh`. That was the largest successful 512-token configuration, but it was slightly slower and used less memory than the recommended 256-token recipe.

## Locked 8192-token recommendation

For runs that must lock `max_seq_length=8192` on 4x RTX 8000, use 4-bit QLoRA, fp16 compute, gradient checkpointing, and per-GPU micro-batch size 1. This was the only successful 8192-token configuration in the follow-up sweep; batch size 2 OOMed, so larger micro-batches are not viable on this hardware.

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
  --train-eval-strategy no \
  --skip-eval
```

Recommended 8192-token settings:

| Parameter | 8192-token value | Reason |
|---|---:|---|
| GPUs | 4 | Uses all available RTX 8000 GPUs through `torchrun` DDP. |
| Per-GPU batch size | 1 | Only successful 8192-token micro-batch; batch 2 OOMed. |
| Global batch size | 4 | Keeps gradient accumulation at 1 for fastest optimizer cadence. Increase only via gradient accumulation if a larger effective batch is needed. |
| Max sequence length | 8192 | Locked for long-context training. |
| Precision | fp16 | RTX 8000 is Turing generation; fp16 is the supported mixed-precision path. |
| Gradient checkpointing | on | Required at 8192 tokens even with QLoRA. |
| Quantization | on | Full fp16 LoRA OOMed at batch size 1, with and without gradient checkpointing. |

## Sweep method

The sweeps intentionally avoided large dataset runs so each trial stayed near or under one minute.

The default short-context sweep used:

- `torchrun --standalone --nproc_per_node=4`
- `Qwen/Qwen2.5-Coder-7B-Instruct`
- fp16 full-precision base model loading with LoRA enabled
- LoRA rank 16, alpha 32, dropout 0.1, default target modules
- one synthetic forward/backward/optimizer step after model load
- `nvidia-smi` sampling all four GPUs every 0.5 seconds
- no batch sizes below 4

The locked 8192-token follow-up used the same model, LoRA settings, DDP launcher, and fp16 compute path, but used one synthetic 8192-token batch per rank and swept quantization/checkpointing because full-precision LoRA could not fit. The follow-up table reports `torch.cuda` peak allocated/reserved memory instead of `nvidia-smi` sampling.

Wall time includes process startup and model load. Step time is the measured optimizer step after the model is loaded.

## Results

### Locked `seq_len=8192` follow-up

The 8192-token follow-up used one synthetic fixed-length training batch per rank so every trial exercised the requested context length while staying under the one-minute sweep cap. The model, LoRA rank/alpha/dropout, DDP launcher, and fp16 compute path matched the default Qwen 7B training recipe.

| Batch/GPU | Seq len | Quantization | Grad checkpointing | Status | Wall time (s) | Step time (s) | Global tokens/step | Tokens/s | Peak allocated/GPU (MiB) | Peak reserved/GPU (MiB) |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 8192 | off | off | OOM | 27.63 | - | - | - | - | - |
| 1 | 8192 | off | on | OOM | 37.54 | - | - | - | - | - |
| 1 | 8192 | on | off | OOM | 25.83 | - | - | - | - | - |
| 1 | 8192 | on | on | ok | 49.25 | 33.55 | 32,768 | 977 | 45,137 | 46,296 |
| 2 | 8192 | on | on | OOM | 46.26 | - | - | - | - | - |
| 4 | 8192 | on | on | OOM | 24.61 | - | - | - | - | - |

Interpretation: at 8192 tokens, batch size 1 per GPU is the practical ceiling and therefore the most efficient viable micro-batch. Use gradient accumulation, not a larger per-GPU batch, if the training run needs an effective global batch above 4.

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

`batch_size=9`, `max_seq_length=256`, and `--no-gradient-checkpointing` is the best default for the stated goal: keep all four GPUs busy and close to full memory without OOM. It reached 100% observed GPU utilization, used about 46.1 GiB of the 48 GiB available per GPU, and batch size 10 failed with CUDA OOM, so batch 9 is the practical ceiling for this sequence length.

The `seq_len=512, batch_size=4` alternative also completed and reached 100% utilization, but it used about 44.4 GiB and processed fewer tokens per measured optimizer step than `seq_len=256, batch_size=9`. Gradient checkpointing should stay off for the default recipe because it reduced memory use substantially and did not complete within the short sweep window.

## Throughput recipe wiring

The repository keeps data-safe `train.py` defaults, but exposes this sweep as an
opt-in wrapper recipe:

- `train.py`: data-safe defaults remain `batch_size=4`, `global_batch_size=16`, and `max_seq_length=2048`; precision defaults to `fp16`, and gradient checkpointing defaults to disabled.
- `src/model_setup.py`: `ModelConfig` defaults to `max_sequence_length=2048`, `precision="fp16"`, and `gradient_checkpointing=False`; inference keeps at least a 2048-token prompt context even when a short training sweep length is saved in model config.
- `train_common.sh`: `THROUGHPUT_SWEEP_DEFAULTS=true` sets `BATCH_SIZE=9`, `MAX_SEQ_LEN=256`, `GLOBAL_BATCH_SIZE=BATCH_SIZE * NGPUS`, `PRECISION=fp16`, and `GRADIENT_CHECKPOINTING=false`.
- `run_train_torchrun.sh` and `run_train_deepspeed.sh`: pass the recommended precision, global batch size, and gradient-checkpointing setting through to `train.py`
