# Model Details

## Base model and task

| Item | Current default |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Package config | `src/model_setup.py::ModelConfig` |
| Training task | Function-sized bytecode-derived TAC → Solidity function body |
| Training sequence length | 8,192 tokens by default |
| Inference wrapper | `src/model_setup.py::SmartContractDecompiler` |

The base Qwen model advertises a larger native context window, and this project now configures training/inference around `max_sequence_length=8192` unless overridden. Model setup records and validates the requested context window against the base model configuration.

## Prompt contract

Training rows use JSONL objects with `input` (TAC), `output` (Solidity), and optional `metadata`. The default template is an Alpaca-style prompt:

```text
### Instruction:
Convert the following Three-Address Code (TAC) representation to readable Solidity code.

### Input:
Bytecode metadata: selector=..., tac_blocks=..., tac_ops=..., ...
<TAC>

### Response:
<Solidity target>
```

Prompt metadata is compact and bytecode-derived only. It can include selector, TAC block/op counts, branch/storage/call/log/revert counts, bytecode length/instruction count, and function count. Compiler version, optimizer settings, source names, source storage layout, ABI oracle signatures, and other source-only data are accepted in artifacts but sanitized out of prompt text.

Long TAC is token-budgeted before generation: comment-only lines may be stripped, dead-code blocks may be dropped, and remaining TAC may be hard-truncated with a marker when it cannot fit the prompt budget. During training, prompt tokens are masked with `-100`, supervised target tokens are preserved when the TAC prefix exceeds context, and numeric tokenizers receive an EOS token on the target span so the model learns a stop condition.

## LoRA fine-tuning

| Parameter | Default |
|---|---|
| `use_lora` | `true` |
| Rank (`r`) | 16 |
| Alpha | 32 |
| Dropout | 0.1 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

Full fine-tuning is possible with `--no-lora`, but quantized full fine-tuning is rejected. With LoRA enabled, `models/final_model/model_config.json` records the adapter configuration used for loading.

## Precision, quantization, and DeepSpeed

| Setting | Current behavior |
|---|---|
| Precision | `--precision {auto,bf16,fp16,fp32}`; default is `fp16`. `auto` chooses BF16 on newer CUDA devices, FP16 on older CUDA, and FP32 on CPU. |
| Quantization | `--quantization` enables 4-bit NF4 loading via `bitsandbytes`; default is disabled. 8-bit training is not exposed as a CLI mode. |
| Gradient checkpointing | Enabled by default for 8192-token training; `--no-gradient-checkpointing` disables it. |
| DeepSpeed | Optional through `--deepspeed <config.json>`; precision flags are patched into the DeepSpeed config. |
| Flash attention | Used only when the runtime/model stack supports it; otherwise the model falls back to standard attention. |

## Training defaults

| Parameter | Default |
|---|---|
| CLI | `train.py` |
| GPUs | Auto `torchrun` relaunch for 4 GPUs by default; use `--num-gpus 1` or `--no-auto-torchrun` for single-process runs. |
| Per-device batch size | 1 |
| Target effective global batch | 16 |
| Gradient accumulation | Auto-computed from world size, per-device batch, and global batch unless explicitly set. |
| Learning rate | `2e-4` |
| Optimizer | `adamw_torch_fused` on CUDA, `adamw_torch` otherwise |
| Scheduler | `cosine_with_restarts` |
| Epochs | 3 |
| Seed | 42 |
| Train-time validation | `--train-eval-strategy auto` chooses steps/epoch based on split size; can be disabled with `no`. |
| Tokenization cache | Enabled by default in `train.py`, manifest-validated, and disableable with `--no-tokenization-cache`. |

`train.py` writes run manifests under `models/run_manifests/`, checkpoints under `models/checkpoints/`, and the final loadable artifact under `models/final_model/`. Each completed training run also writes `models/final_model/training_input_manifest.json` with dataset hashes, token-length/truncation summaries, context-window validation, effective Trainer arguments, and train-time eval sample indices.

## Inference behavior

| Parameter | Default |
|---|---|
| `max_new_tokens` | 1,024 in shared inference/web/CLI defaults; evaluation defaults to 256 unless overridden. |
| Temperature | 0.1 |
| Sampling | Greedy (`do_sample=false`) by default |
| Repetition penalty | 1.15 |

The shared inference path first analyzes bytecode into per-function TAC, optionally resolves exact TAC lookup hits from `data/tac_lookup.db`, then runs the model for misses. Results include per-function sources (`exact_match`, `model_inference`, or `error`), validation, reconstruction metadata, prompt diagnostics, lookup provenance, and trace data.

## DPO status

`src/model_setup.py` includes `DPOTrainingConfig` and `DPODatasetBuilder` for preference-pair construction, and `trl` is listed as a dependency. There is currently no active DPO training loop in `train.py`; standard supervised fine-tuning is the implemented training path.

## Evaluation and quality gates

Evaluation uses `SmartContractEvaluator` plus report helpers to compute semantic similarity, normalized edit distance, BLEU, ROUGE-L, token accuracy, structural preservation, replication precision/recall/F1, Solidity validity, bytecode semantic checks, prompt diagnostics, and worst-sample summaries.

Default quality-gate thresholds in `train.py` are targets, not claimed achieved results:

| Metric | Gate |
|---|---|
| Mean semantic similarity | `>= 0.82` |
| Fraction above 0.8 semantic similarity | `>= 0.78` |
| Fraction below 0.4 normalized edit distance | `>= 0.82` |
| Failure rate | `<= 0.0` |
| Mean replication F1 | `>= 0.75` |
| Solidity valid / compiler checked / AST valid | `>= 1.0` each |
| Bytecode semantic score | `>= 0.5` |
| Bytecode checked / deployable | `>= 1.0` each |

## Practical limitations

- Output is Solidity-like reconstruction, not guaranteed original source or semantic equivalence.
- Prompt safety deliberately removes source/compiler oracle data, so names, inheritance, modifiers, and storage labels may be approximate.
- Large functions can lose TAC detail through truncation.
- Exact TAC lookup improves known-function latency and fidelity but depends on a separately built lookup database.
