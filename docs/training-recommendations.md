# Training Recommendations: Model Selection & Configuration

> **Hardware assumed**: 4× NVIDIA RTX 8000 (48 GB VRAM each, ~180 GB total)
>
> **Paper baseline**: Llama 3.2 3B with LoRA fine-tuning on 238,446 TAC-to-Solidity function pairs
>
> **Reference**: "Decompiling Smart Contracts with a Large Language Model" (arXiv:2506.19624v1)

---

## Hardware Budget Analysis

The paper used **Llama 3.2 3B** — a deliberately small model chosen for practical deployment constraints. With 4× RTX 8000 (48 GB each, ~180 GB usable), dramatically larger models can be leveraged that should meaningfully improve accuracy, especially on the failure cases the paper documents (complex DeFi logic, fixed-point arithmetic, nested storage patterns).

| Config | VRAM Required (approx) | Fits on 4× RTX 8000? |
|---|---|---|
| 3B (paper baseline, 4-bit) | ~2–4 GB | ✅ Trivially |
| 8B (4-bit LoRA) | ~6–8 GB | ✅ Trivially |
| 14B (4-bit LoRA) | ~10–12 GB | ✅ Easily |
| 32B (4-bit LoRA) | ~20–24 GB | ✅ Single GPU |
| 70B (4-bit LoRA) | ~40–48 GB | ✅ 1–2 GPUs |
| 70B (full bf16, LoRA) | ~140 GB | ✅ Across 4 GPUs |
| 72B (4-bit LoRA, training) | ~80–100 GB | ✅ Across 2–3 GPUs |
| 405B (4-bit) | ~200+ GB | ❌ Too tight |

---

## Recommended Models (Ranked)

### 🥇 Top Pick: Qwen 2.5 Coder 32B

**Model ID**: `Qwen/Qwen2.5-Coder-32B`

- **Why**: Purpose-built for code generation/understanding. 32B parameters is the sweet spot for this hardware — large enough for significant quality gains, small enough to train comfortably with LoRA on a single RTX 8000 (4-bit) or across 2 GPUs (bf16).
- **Context length**: 128K tokens natively (massive improvement over Llama 3.2's 128K theoretical / 2K practical in the current setup).
- **Code benchmarks**: Outperforms many 70B general models on code tasks.
- **Training fit**: 4-bit LoRA training uses ~24 GB — fits on one GPU, leaving 3 GPUs free for data loading / larger batches.

### 🥈 Strong Alternative: DeepSeek-Coder-V2 33B / DeepSeek-V3 (MoE, 37B active)

- Excellent code understanding, particularly strong on structured/formal languages.
- MoE (Mixture of Experts) architecture means fewer active parameters → faster training.

### 🥉 Maximum Quality: Llama 3.1 70B / Qwen 2.5 72B

- With 4-bit quantization + LoRA, training fits across 2 GPUs (~48 GB each for 4-bit 70B + optimizer states).
- Would require `accelerate` with DeepSpeed ZeRO Stage 2/3 for distributed training.
- Significant quality leap but 2–3× slower training iteration.

### Honorable Mentions

| Model | Size | Strengths |
|---|---|---|
| **CodeLlama 34B** | 34B | Strong code model, Llama architecture (minimal code changes) |
| **StarCoder2 15B** | 15B | Trained on The Stack v2, excellent code understanding, very efficient |
| **Llama 3.1 8B** | 8B | 2.5× bigger than current, minimal infrastructure changes, fast iteration |
| **Phi-3-medium 14B** | 14B | Microsoft's efficient model, surprisingly strong on code |

---

## Why Qwen 2.5 Coder 32B Is the Clear Winner

1. **Code-specialized pre-training**: Unlike general-purpose Llama 3.2 3B, Qwen 2.5 Coder was trained on massive code corpora. It already understands Solidity syntax, EVM patterns, and structured code transformations before fine-tuning.

2. **10× more parameters**: 32B vs 3B. The paper's Case Study 2 (staking rewards, 0.52 similarity) failed precisely because the 3B model lacked capacity to represent complex DeFi patterns. A 32B model has far more capacity for these long-tail patterns.

3. **Comfortable fit**: 4-bit LoRA fine-tuning of 32B uses ~24 GB VRAM. With 4 × 48 GB available, training can run on 1 GPU with the others used for larger batch sizes or parallel experiments.

4. **128K context window**: The paper mentions a max of 20,000 tokens and the current code uses 2,048. With 32B + 128K context, training at 4,096–8,192 tokens is realistic, capturing far more complex functions without truncation.

---

## Token Length Recommendations

The paper states a maximum of **20,000 tokens** for sequence management, but the current code caps at **2,048**. Here is what is reasonable for each model size:

| Model | Practical Training Seq Length | Reasoning |
|---|---|---|
| 3B (current) | 2,048–4,096 | Memory-constrained, limited attention capacity |
| 8B | 4,096 | Good balance, 2× current |
| 14–15B | 4,096–8,192 | Can handle most complex functions |
| **32B (recommended)** | **4,096–8,192** | **Sweet spot: captures 95%+ of functions without truncation** |
| 70B+ | 8,192–16,384 | Diminishing returns beyond 8K for this task |

### Why 4,096 Tokens for the 32B Model

- The dataset's TAC inputs + Solidity outputs need to fit in one sequence.
- The paper found 67.64% of functions have length differences within ±50 characters, with most functions in the 200–300 character range.
- At ~4 characters per token, most functions are 50–150 tokens of TAC + 50–150 tokens of Solidity.
- 4,096 gives 10–20× headroom for complex functions.
- Going to 8,192 captures even the longest DeFi functions but doubles memory per sample.

### VRAM Estimate for 32B 4-bit LoRA Training at 4,096 Seq Length

| Component | VRAM |
|---|---|
| Model weights (4-bit) | ~18 GB |
| LoRA adapters + optimizer | ~4 GB |
| Activations (batch_size=4, seq_len=4096) | ~8–12 GB |
| **Total** | **~30–34 GB** → fits on a single RTX 8000 |

---

## Implementation Changes Required

To switch from Llama 3.2 3B to Qwen 2.5 Coder 32B, the following changes are needed:

1. **Change `model_name`** in `ModelConfig` and `train.py`: set to `"Qwen/Qwen2.5-Coder-32B"`.
2. **Update `max_sequence_length`**: from 2048 → 4096.
3. **Adjust batch size**: likely 2 per GPU (instead of 4) with gradient accumulation of 8–16.
4. **Keep LoRA config identical**: r=16, alpha=32 works well for 32B models too.
5. **Multi-GPU setup**: Add `accelerate` config for FSDP or DeepSpeed if training across multiple GPUs (optional — single GPU works).
6. **Tokenizer handling**: Qwen uses its own tokenizer, but the HuggingFace `AutoTokenizer` abstraction handles this transparently.

The code as written already supports this — pass the following to `train.py`:

```bash
python train.py --model-name Qwen/Qwen2.5-Coder-32B --max-seq-length 4096 --batch-size 2
```

---

## Summary Comparison

| Aspect | Current (Paper) | Recommended |
|---|---|---|
| **Model** | Llama 3.2 3B | Qwen 2.5 Coder 32B |
| **Parameters** | 3B | 32B (10.7×) |
| **Token limit** | 2,048 | 4,096 (2×) |
| **Quantization** | 4-bit NF4 | 4-bit NF4 (same) |
| **LoRA rank** | 16 | 16 (same) |
| **Batch size** | 4 | 2 per device × 4 GPUs |
| **VRAM used** | ~4 GB | ~30–34 GB per GPU |
| **Expected quality gain** | Baseline (0.82 avg sim) | Estimated 0.88–0.92 avg sim |

---

## Multi-GPU Training & DeepSpeed

Three training modes are available, from simplest to fastest:

### 1. Single GPU (`python`)

```bash
python train.py --skip-collection --dataset ./data/hf_training_dataset.jsonl \
    --batch-size 4 --epochs 3 --lr 2e-4 --max-seq-length 2048 --skip-eval
```

Automatically restricts to GPU 0 when using quantized models.

### 2. Multi-GPU DDP (`torchrun`)

```bash
./run_train.sh              # defaults: 4 GPUs, batch_size=4
NGPUS=2 ./run_train.sh      # override GPU count
```

Standard PyTorch Distributed Data Parallel — each GPU holds a full copy of optimizer states and gradients.

### 3. DeepSpeed (`deepspeed`)

```bash
./run_train_deepspeed.sh                    # defaults: 4 GPUs, batch_size=4
NGPUS=2 ./run_train_deepspeed.sh            # override GPU count
```

Uses `ds_config.json` (ZeRO Stage 0 + BF16 by default).

> **Performance note for Llama 3.2 3B + LoRA**: For this small model (3B params,
> 24M trainable LoRA parameters) with 4-bit quantization, **torchrun DDP is
> ~40-60% faster** than DeepSpeed due to DeepSpeed's per-step engine overhead.
> The quantized model is only ~2 GB per GPU, so ZeRO memory sharding provides
> no benefit and its communication adds overhead.
>
> **Use DeepSpeed when**:
> - Training larger models (7B+) where ZeRO memory savings matter
> - Doing full fine-tuning (not LoRA) where optimizer states are large
> - GPU memory is constrained and you need ZeRO stage 2/3 to fit the model
> - Training 32B+ models across multiple GPUs

For larger models, switch to ZeRO Stage 2 in `ds_config.json`:

```json
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 2e8,
    "allgather_bucket_size": 2e8
  }
}
```

| Feature | DDP (torchrun) | DeepSpeed ZeRO-0 | DeepSpeed ZeRO-2 |
|---|---|---|---|
| Optimizer state memory | Full per GPU | Full per GPU | Sharded (1/N) |
| Gradient memory | Full per GPU | Full per GPU | Sharded (1/N) |
| Per-step overhead | Lowest | ~40-60% higher | ~60-100% higher |
| Best for | Small models + LoRA | N/A (use torchrun) | Large models, full FT |

To use DeepSpeed manually (without the wrapper script):

```bash
deepspeed --num_gpus=4 train.py \
    --skip-collection --dataset ./data/hf_training_dataset.jsonl \
    --batch-size 4 --epochs 3 --deepspeed ds_config.json --skip-eval
```

### DeepSpeed ZeRO Stage 3 (for 70B+ models)

For models that don't fit on a single GPU even in 4-bit (e.g., full bf16 70B), use ZeRO Stage 3 which additionally shards model parameters. Create a `ds_config_z3.json`:

```json
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e7,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e5
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

Then: `DS_CONFIG=ds_config_z3.json ./run_train_deepspeed.sh`

---

## Detailed Model Profiles & Training Parameters

The following sections provide complete training configurations for each recommended model, including exact commands and VRAM estimates.

---

### DeepSeek-Coder-V2 33B

**Model ID**: `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` (33B dense variant)

**Why it's interesting for decompilation:**

- Exceptional at formal/structured languages — compiler-adjacent tasks are a sweet spot
- Stronger than Qwen on reconstructing nested storage layouts and fixed-point math
- Better handling of complex dispatcher patterns in DeFi contracts (AMMs, staking, rebasing)

**Where it may beat Qwen 2.5 Coder 32B:**

- Complex bitwise logic and fixed-point arithmetic (common in DeFi)
- Multi-level storage slot resolution
- Contracts with non-standard dispatcher patterns

**Tradeoffs:**

- More finicky training — requires careful hyperparameter selection
- Less forgiving tokenization for bytecode-adjacent text
- Fewer community LoRA examples → less proven fine-tuning recipe

| Parameter | Value |
|---|---|
| **LoRA rank (r)** | 16 |
| **LoRA alpha** | 32 |
| **LoRA dropout** | 0.1 |
| **Target modules** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| **Quantization** | 4-bit NF4 (double quant) |
| **Batch size** | 2 per GPU |
| **Gradient accumulation** | 8 |
| **Sequence length** | 4,096 |
| **Learning rate** | 1.5e-4 (slightly lower than Qwen — DeepSeek is more sensitive) |
| **LR scheduler** | Cosine with warmup ratio 0.05 |
| **Epochs** | 3–5 |
| **VRAM per GPU (4-bit LoRA)** | ~28–34 GB → fits on 1× RTX 8000 |

```bash
python train.py --skip-collection --dataset ./data/hf_training_dataset.jsonl \
    --model-name deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --batch-size 2 --epochs 3 --lr 1.5e-4 --max-seq-length 4096 --skip-eval
```

---

### DeepSeek-V3 (MoE, ~37B active / 671B total)

**Model ID**: `deepseek-ai/DeepSeek-V3` (Mixture of Experts)

**Why it's interesting:**

- MoE architecture: ~37B active parameters out of 671B total — higher effective capacity without full 70B training cost
- State-of-the-art on code benchmarks as of early 2026
- Multi-head latent attention for improved long-range reasoning

**MoE-specific considerations:**

- **Determinism**: MoE routing introduces non-determinism across runs. Set `torch.use_deterministic_algorithms(True)` and fixed seeds for reproducibility, but expect small variations.
- **Load balancing loss**: MoE models include an auxiliary load-balancing loss. The default coefficient is usually fine, but monitor expert utilization during training.
- **Memory**: Despite 671B total parameters, only ~37B are active per token. However, all expert weights must be loaded into memory. 4-bit quantization brings this to ~80–100 GB across 2–3 GPUs.

| Parameter | Value |
|---|---|
| **LoRA rank (r)** | 16 |
| **LoRA alpha** | 32 |
| **LoRA dropout** | 0.1 |
| **Target modules** | All linear layers (use `"all-linear"`) |
| **Quantization** | 4-bit NF4 |
| **Batch size** | 1 per GPU |
| **Gradient accumulation** | 16 |
| **Sequence length** | 4,096 |
| **Learning rate** | 1e-4 |
| **LR scheduler** | Cosine with warmup ratio 0.03 |
| **Epochs** | 3 |
| **VRAM** | ~80–100 GB → requires 2–3× RTX 8000 with DeepSpeed ZeRO-2 |
| **Multi-GPU** | Required — use `run_train_deepspeed.sh` |

```bash
NGPUS=3 deepspeed --num_gpus=3 train.py \
    --skip-collection --dataset ./data/hf_training_dataset.jsonl \
    --model-name deepseek-ai/DeepSeek-V3 \
    --batch-size 1 --epochs 3 --lr 1e-4 --max-seq-length 4096 \
    --deepspeed ds_config.json --skip-eval
```

> ⚠️ **Important**: MoE models may require custom LoRA target module selection. If training fails with the default targets, use `"all-linear"` or inspect the model architecture with `model.named_modules()` to find the correct MoE expert layer names.

---

### Codestral (Mistral)

**Model ID**: `mistralai/Codestral-22B-v0.1`

**Why it's interesting:**

- Very strong at structured generation — produces syntactically correct code consistently
- Smaller parameter count (22B) but high per-parameter efficiency
- Excellent instruction following for code transformation tasks
- 32K context window

**Where it excels:**

- Syntactically perfect Solidity output (fewer parse errors in generated code)
- Clean function boundary reconstruction
- Efficient inference — faster generation than 32B models

**Limitations:**

- Weaker on ultra-long context reasoning compared to Qwen/DeepSeek
- Less capacity for memorizing rare DeFi patterns
- 22B may underperform 32B on the hardest decomposition cases

| Parameter | Value |
|---|---|
| **LoRA rank (r)** | 16 |
| **LoRA alpha** | 32 |
| **LoRA dropout** | 0.1 |
| **Target modules** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| **Quantization** | 4-bit NF4 |
| **Batch size** | 4 per GPU |
| **Gradient accumulation** | 4 |
| **Sequence length** | 4,096 |
| **Learning rate** | 2e-4 |
| **LR scheduler** | Cosine |
| **Epochs** | 3–5 |
| **VRAM per GPU (4-bit LoRA)** | ~16–20 GB → fits easily on 1× RTX 8000 |

```bash
python train.py --skip-collection --dataset ./data/hf_training_dataset.jsonl \
    --model-name mistralai/Codestral-22B-v0.1 \
    --batch-size 4 --epochs 3 --lr 2e-4 --max-seq-length 4096 --skip-eval
```

---

### StarCoder2 15B / 33B

**Model ID**: `bigcode/starcoder2-15b` or `bigcode/starcoder2-33b` (BigCode open-weight)

**Why it's interesting:**

- Trained on The Stack v2 — one of the highest-quality code training corpora
- Lower hallucination rate than general-purpose models
- 15B version is extremely efficient for iteration and experimentation
- Fill-in-the-middle (FIM) capability could help with partial function reconstruction

**Where it excels:**

- Clean, non-hallucinated output — when it doesn't know something, it generates less rather than inventing
- Fast iteration cycles (especially 15B variant)
- Good baseline for ablation studies

**Limitations:**

- Weaker global reasoning than Qwen/DeepSeek for cross-function dependencies
- 15B may lack capacity for the hardest cases
- Less instruction-tuned — may need more careful prompt engineering

| Parameter | 15B | 33B |
|---|---|---|
| **LoRA rank (r)** | 16 | 16 |
| **LoRA alpha** | 32 | 32 |
| **LoRA dropout** | 0.1 | 0.1 |
| **Target modules** | `c_attn, c_proj, c_fc` | `c_attn, c_proj, c_fc` |
| **Quantization** | 4-bit NF4 | 4-bit NF4 |
| **Batch size** | 4 per GPU | 2 per GPU |
| **Gradient accumulation** | 4 | 8 |
| **Sequence length** | 4,096 | 4,096 |
| **Learning rate** | 2e-4 | 2e-4 |
| **Epochs** | 3–5 | 3–5 |
| **VRAM (4-bit LoRA)** | ~12–14 GB | ~24–28 GB |

```bash
# StarCoder2 15B — fast iteration
python train.py --skip-collection --dataset ./data/hf_training_dataset.jsonl \
    --model-name bigcode/starcoder2-15b \
    --batch-size 4 --epochs 5 --lr 2e-4 --max-seq-length 4096 --skip-eval

# StarCoder2 33B — higher quality
python train.py --skip-collection --dataset ./data/hf_training_dataset.jsonl \
    --model-name bigcode/starcoder2-33b \
    --batch-size 2 --epochs 3 --lr 2e-4 --max-seq-length 4096 --skip-eval
```

> **Note**: StarCoder2 uses GPT-style attention layer names (`c_attn`, `c_proj`, `c_fc`). If the auto-detection in `model_setup.py` selects the wrong modules, the code will fall back to `"all-linear"` automatically.

---

### Phi-3 Medium 14B

**Model ID**: `microsoft/Phi-3-medium-14b-4k-instruct`

**Why it's interesting:**

- Surprisingly strong code performance for its size (14B)
- Excellent for auxiliary tasks: pattern classification, IR cleanup, TAC normalization
- Very fast iteration — ideal for prototyping and hyperparameter sweeps
- Strong instruction following

**Best use cases:**

- **Auxiliary model**: Use as a pre-processor to clean/classify TAC before feeding to the primary 32B model
- **Rapid experimentation**: Test new training strategies cheaply before committing to 32B runs
- **Ensemble component**: Generate multiple candidates, rank with a separate model

**Not recommended as primary decompiler** — 14B lacks capacity for complex DeFi patterns.

| Parameter | Value |
|---|---|
| **LoRA rank (r)** | 16 |
| **LoRA alpha** | 32 |
| **LoRA dropout** | 0.1 |
| **Target modules** | `qkv_proj, o_proj, gate_up_proj, down_proj` |
| **Quantization** | 4-bit NF4 |
| **Batch size** | 4 per GPU |
| **Gradient accumulation** | 4 |
| **Sequence length** | 4,096 (native 4K context) |
| **Learning rate** | 2e-4 |
| **Epochs** | 3–5 |
| **VRAM (4-bit LoRA)** | ~10–12 GB → fits trivially on any RTX 8000 |

```bash
python train.py --skip-collection --dataset ./data/hf_training_dataset.jsonl \
    --model-name microsoft/Phi-3-medium-14b-4k-instruct \
    --batch-size 4 --epochs 5 --lr 2e-4 --max-seq-length 4096 --skip-eval
```

---

## Model Comparison Matrix

| Model | Params | VRAM (4-bit LoRA) | Best For | Risk Level | Context |
|---|---|---|---|---|---|
| **Qwen 2.5 Coder 32B** 🥇 | 32B | ~30 GB | General decompilation | Low | 128K |
| **DeepSeek-Coder-V2 33B** 🥈 | 33B | ~32 GB | DeFi/complex math | Medium | 128K |
| **DeepSeek-V3 (MoE)** | 37B active | ~90 GB | Maximum capacity | High | 128K |
| **Codestral 22B** | 22B | ~18 GB | Syntactic correctness | Low | 32K |
| **StarCoder2 15B** | 15B | ~13 GB | Fast iteration / low hallucination | Low | 16K |
| **StarCoder2 33B** | 33B | ~26 GB | Quality + low hallucination | Low | 16K |
| **Phi-3 Medium** | 14B | ~11 GB | Auxiliary tasks / prototyping | Low | 4K |
| **Qwen 2.5 72B** | 72B | ~90 GB | Maximum quality (diminishing returns) | Medium | 128K |
| **Llama 3.1 70B** | 70B | ~45 GB | NL explanation, not reconstruction | Medium | 128K |

## Model Selection Decision Tree

```
START: What is your priority?
│
├─ Fast iteration / prototyping?
│  └─ Phi-3 Medium 14B or StarCoder2 15B
│
├─ Best quality/cost ratio? (RECOMMENDED)
│  └─ Qwen 2.5 Coder 32B
│     ├─ Struggling with DeFi contracts?
│     │  └─ Try DeepSeek-Coder-V2 33B head-to-head
│     └─ Struggling with syntax correctness?
│        └─ Try Codestral 22B head-to-head
│
├─ Absolute maximum quality?
│  └─ Have you exhausted 32B improvements?
│     ├─ No → Stay with 32B, improve dataset first
│     └─ Yes → Qwen 2.5 72B with ZeRO-3
│
└─ Lowest hallucination rate?
   └─ StarCoder2 33B
```

---

## Why NOT "The Biggest Llama"?

Even at 70B, Llama 3.x models are suboptimal for bytecode decompilation:

- **Trained primarily for natural language** — tokenizer is not optimized for bytecode-like text (hex, opcodes, IR syntax)
- **Hallucination tendency** — tends to invent missing logic rather than indicating "unknown"
- **Worse deterministic reconstruction** — bytecode decompilation requires exact pattern matching, not creative writing
- **Tokenizer inefficiency** — hex sequences and opcode mnemonics encode into more tokens than code-specialized tokenizers

Llama is excellent for *explaining* contracts in natural language, but not for *reconstructing* them from bytecode.

---

## Scaling Strategy

A recommended iterative approach:

1. **Phase 1** — Train with **Qwen 2.5 Coder 32B** at 4,096 tokens (primary recommendation). Evaluate against the paper's benchmarks.
2. **Phase 2** — If results are strong, experiment with 8,192 token sequences to capture the long-tail complex functions.
3. **Phase 2b** — Run a head-to-head comparison with **DeepSeek-Coder-V2 33B** on your hardest DeFi samples. If DeepSeek wins on complex math/storage patterns, consider an ensemble or model selection strategy.
4. **Phase 3** — Optionally scale to **70B** (Qwen 2.5 72B preferred over Llama 3.1 70B) using DeepSpeed ZeRO-3 across all 4 GPUs for maximum quality.
5. **Phase 4** — Compare 32B vs 70B results. The 70B may show diminishing returns relative to training cost for this domain-specific task.

> **Critical insight**: Dataset quality improvements (see `docs/dataset-generation.md`) will almost always yield larger gains than scaling from 32B → 70B. Prioritize data quality before model size.
