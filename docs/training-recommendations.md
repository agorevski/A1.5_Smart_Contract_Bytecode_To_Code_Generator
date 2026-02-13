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

## Scaling Strategy

A recommended iterative approach:

1. **Phase 1** — Train with **Qwen 2.5 Coder 32B** at 4,096 tokens (primary recommendation). Evaluate against the paper's benchmarks.
2. **Phase 2** — If results are strong, experiment with 8,192 token sequences to capture the long-tail complex functions.
3. **Phase 3** — Optionally scale to **70B** (Llama 3.1 70B or Qwen 2.5 72B) using DeepSpeed ZeRO-3 across all 4 GPUs for maximum quality.
4. **Phase 4** — Compare 32B vs 70B results. The 70B may show diminishing returns relative to training cost for this domain-specific task.