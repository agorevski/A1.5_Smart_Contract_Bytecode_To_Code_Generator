# Troubleshooting Guide

Common issues and solutions for the Smart Contract Decompilation system.

## Installation Issues

### Issue: CUDA Out of Memory

**Symptoms**:

```text
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions**:

**Enable 4-bit quantization** (already default):

```python
config.use_4bit_quantization = True
```

**Reduce batch size**:

```python
config.batch_size = 2  # Lower from 4
config.gradient_accumulation_steps = 8  # Increase to maintain effective batch
```

**Enable gradient checkpointing**:

```python
config.use_gradient_checkpointing = True
```

**Clear GPU cache**:

```python
import torch
torch.cuda.empty_cache()
```

### Issue: PyTorch Installation Fails

**Symptoms**:

```text
ERROR: Could not find a version that satisfies the requirement torch
```

**Solutions**:

```bash
# Install specific version for your CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or for CPU only
pip install torch torchvision torchaudio
```

### Issue: Transformers Library Error

**Symptoms**:

```text
ImportError: cannot import name 'AutoModelForCausalLM'
```

**Solution**:

```bash
pip install --upgrade transformers accelerate
```

### Issue: Permission Denied for Model Download

**Symptoms**:

```text
HTTPError: 401 Unauthorized
```

**Solutions**:

**Login with Hugging Face CLI**:

```bash
huggingface-cli login
# Enter your token when prompted
```

**Set environment variable**:

```bash
export HF_TOKEN="your_token_here"
```

**Accept model license**: Visit [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) and accept terms

## Training Issues

### Issue: Loss Not Decreasing

**Symptoms**:

- Training loss plateaus immediately
- Validation loss doesn't improve

**Solutions**:

**Lower learning rate**:

```python
config.learning_rate = 1e-4  # Down from 2e-4
```

**Increase warmup**:

```python
config.warmup_steps = 1000  # Up from 500
```

**Check data quality**:

```python
# Verify dataset
for item in dataset[:10]:
    print(f"Input length: {len(item['input'])}")
    print(f"Output length: {len(item['output'])}")
```

### Issue: Overfitting

**Symptoms**:

- Training loss much lower than validation loss
- Large gap between train/val metrics

**Solutions**:

**Increase regularization**:

```python
config.weight_decay = 0.05  # Up from 0.01
config.lora_dropout = 0.1   # Up from 0.05
```

**Reduce epochs**:

```python
config.num_epochs = 2  # Down from 3
```

**Add more training data** or **augment existing data**

### Issue: Slow Training Speed

**Symptoms**:

- Training takes much longer than expected
- Low GPU utilization

**Solutions**:

**Enable bfloat16**:

```python
config.bf16 = True
config.fp16 = False
```

**Increase dataloader workers**:

```python
config.dataloader_num_workers = 4
config.dataloader_pin_memory = True
```

**Check GPU utilization**:

```bash
watch -n 1 nvidia-smi
```

## Data Collection Issues

### Issue: Etherscan API Rate Limiting

**Symptoms**:

```text
Error: Max rate limit reached
```

**Solutions**:

**Add delays between requests**:

```python
builder = DatasetBuilder(
    etherscan_api_key="your_key",
    rate_limit=4.0  # Reduced from 5.0
)
```

**Use multiple API keys**:

```python
api_keys = ["key1", "key2", "key3"]
builder = DatasetBuilder(api_keys=api_keys)
```

**Implement exponential backoff**:

```python
import time
from requests.exceptions import HTTPError

def collect_with_retry(address, max_retries=3):
    for i in range(max_retries):
        try:
            return builder.collect_contract(address)
        except HTTPError as e:
            if '429' in str(e):  # Rate limit
                wait_time = 2 ** i
                time.sleep(wait_time)
            else:
                raise
```

### Issue: Contract Verification Failed

**Symptoms**:

```text
Error: Contract source code not verified
```

**Solutions**:

**Skip unverified contracts**:

```python
verified_addresses = [
    addr for addr in addresses 
    if is_verified(addr)
]
```

**Use bytecode-only mode** (if applicable)

**Request contract owner to verify** on Etherscan

## Inference Issues

### Issue: Slow Inference Speed

**Symptoms**:

- Decompilation takes too long
- Low tokens/second throughput

**Solutions**:

**Reduce max_new_tokens**:

```python
generation_config.max_new_tokens = 1024  # Down from 2048
```

**Use greedy decoding**:

```python
generation_config.do_sample = False
generation_config.num_beams = 1
```

**Merge LoRA weights**:

```python
model = model.merge_and_unload()
```

**Use lower temperature**:

```python
generation_config.temperature = 0.1
```

### Issue: Low Quality Decompilation

**Symptoms**:

- Generated code doesn't compile
- Semantic similarity < 0.5
- Missing critical logic

**Solutions**:

**Adjust temperature**:

```python
# Try different temperatures
for temp in [0.1, 0.3, 0.5]:
    result = decompiler.decompile(tac, temperature=temp)
    evaluate(result)
```

**Check TAC quality**:

```python
# Verify TAC is well-formed
print(tac)
assert 'function_selector' in tac
```

**Retrain with more data** or **fine-tune further**

**Use ensemble approach**:

```python
# Generate multiple candidates
results = []
for i in range(3):
    result = decompiler.decompile(tac, temperature=0.3)
    results.append(result)

# Select best based on metrics
best = max(results, key=lambda x: evaluate(x))
```

## System Issues

### Issue: Disk Space Full

**Symptoms**:

```text
OSError: [Errno 28] No space left on device
```

**Solutions**:

**Clean old checkpoints**:

```bash
# Keep only last 3 checkpoints
ls -t models/checkpoints/checkpoint-* | tail -n +4 | xargs rm -rf
```

**Compress datasets**:

```bash
gzip data/processed/*.jsonl
```

**Use external storage** for models/data

### Issue: Out of System RAM

**Symptoms**:

```text
MemoryError: Unable to allocate array
```

**Solutions**:

**Use streaming datasets**:

```python
from datasets import load_dataset

dataset = load_dataset(
    'json',
    data_files='data/train.jsonl',
    streaming=True  # Don't load all into memory
)
```

**Reduce dataloader workers**:

```python
config.dataloader_num_workers = 2
```

**Process in smaller batches** -

### Issue: Module Import Errors

**Symptoms**:

```text
ModuleNotFoundError: No module named 'src'
```

**Solutions**:

**Add to PYTHONPATH**:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Use absolute imports**:

```python
from A1.5_Smart_Contract_Bytecode_To_Code_Generator.src import module
```

**Install as package**:

```bash
pip install -e .
```

## Model Quality Issues

### Issue: High Edit Distance

**Symptoms**:

- Most functions have edit distance > 0.4
- Syntactic errors in output

**Solutions**:

**Train for more epochs**
**Increase LoRA rank**

```python
config.lora_rank = 32  # Up from 16
```

**Check training data quality**
**Verify TAC generation accuracy**

### Issue: Low Semantic Similarity

**Symptoms**:

- Semantic similarity < 0.7 on average
- Meaning not preserved

**Solutions**:

**Ensure diverse training data**
**Increase model capacity**:

```python
# Use larger base model
config.model_name = "meta-llama/Llama-3.1-8B"
```

**Train for more epochs**
**Check that TAC preserves semantics**

## Debugging Tips

### Enable Verbose Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Check GPU Status

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Validate Environment

```python
# Run system check
python -c "
import torch
import transformers
import peft
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

### Test Components Individually

```python
# Test bytecode analyzer
from src.bytecode_analyzer import analyze_bytecode_to_tac
tac = analyze_bytecode_to_tac("0x6080...")
assert tac is not None

# Test model loading
from src.model_setup import SmartContractDecompiler
decompiler = SmartContractDecompiler("models/final/smart_contract_decompiler")
assert decompiler.model is not None
```

## Getting Help

If issues persist:

1. **Check GitHub Issues**: Search for similar problems
2. **Enable Debug Mode**: Collect detailed error logs
3. **Minimal Reproducible Example**: Create simple test case
4. **System Information**: Include OS, Python version, GPU details
5. **Create Issue**: Provide all above information

## Next Steps

- Review [Installation](installation.md) for setup details
- Check [Training Pipeline](training-pipeline.md) for training guidance
- See [Usage Guide](usage.md) for examples
- Explore [Model Details](model-details.md) for configuration options
