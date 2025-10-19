# Usage Guide

This guide covers basic to advanced usage of the Smart Contract Decompilation system.

## Quick Start

### Running the Demo

The simplest way to validate the system is to run the demo:

```bash
python demo.py
```

This runs through all components without requiring GPU or API access.

## Basic Usage

### 1. Analyzing Bytecode

Convert EVM bytecode to Three-Address Code (TAC):

```python
from src.bytecode_analyzer import analyze_bytecode_to_tac

# Example EVM bytecode
bytecode = "0x608060405234801561001057600080fd5b50..."

# Convert to TAC
tac_representation = analyze_bytecode_to_tac(bytecode)

print(tac_representation)
```

**Output Example:**
```
function_selector_0x70a08231:
  v0 = CALLDATALOAD 0x04
  v1 = SLOAD storage[mapping_0][v0]
  RETURN v1
```

### 2. Decompiling to Solidity

Generate Solidity code from TAC (requires trained model):

```python
from src.model_setup import SmartContractDecompiler

# Initialize decompiler with trained model
decompiler = SmartContractDecompiler(
    model_path="models/final/smart_contract_decompiler"
)

# Decompile TAC to Solidity
solidity_code = decompiler.decompile_tac_to_solidity(
    tac_representation,
    temperature=0.3,      # Low for deterministic output
    max_new_tokens=2048   # Maximum output length
)

print(solidity_code)
```

**Output Example:**
```solidity
function balanceOf(address account) public view returns (uint256) {
    return _balances[account];
}
```

### 3. End-to-End Decompilation

Complete pipeline from bytecode to Solidity:

```python
from src.bytecode_analyzer import analyze_bytecode_to_tac
from src.model_setup import SmartContractDecompiler

# Step 1: Bytecode to TAC
bytecode = "0x608060405234801561001057600080fd5b50..."
tac = analyze_bytecode_to_tac(bytecode)

# Step 2: TAC to Solidity
decompiler = SmartContractDecompiler("models/final/smart_contract_decompiler")
solidity = decompiler.decompile_tac_to_solidity(tac)

print(f"Original bytecode length: {len(bytecode)}")
print(f"Decompiled Solidity:\n{solidity}")
```

## Advanced Usage

### Custom Generation Parameters

Fine-tune the generation process:

```python
solidity_code = decompiler.decompile_tac_to_solidity(
    tac_representation,
    temperature=0.1,           # Very deterministic (0.0-1.0)
    top_p=0.95,               # Nucleus sampling
    top_k=50,                 # Top-k sampling
    max_new_tokens=4096,      # Longer output
    do_sample=True,           # Enable sampling
    repetition_penalty=1.2    # Discourage repetition
)
```

**Parameter Guidelines:**

- **temperature**: Lower (0.1-0.3) for consistent, deterministic output; higher (0.7-1.0) for creative variations
- **top_p**: 0.9-0.95 for balanced quality
- **top_k**: 40-60 for good diversity
- **max_new_tokens**: 2048 for typical functions, 4096+ for complex contracts
- **repetition_penalty**: 1.1-1.3 to avoid repetitive code

### Batch Processing

Process multiple functions efficiently:

```python
from src.model_setup import SmartContractDecompiler

decompiler = SmartContractDecompiler("models/final/smart_contract_decompiler")

# List of TAC representations
tac_list = [tac1, tac2, tac3, ...]

# Batch decompilation
results = []
for tac in tac_list:
    solidity = decompiler.decompile_tac_to_solidity(tac, temperature=0.3)
    results.append(solidity)

# Process results
for i, solidity in enumerate(results):
    print(f"\n=== Function {i+1} ===")
    print(solidity)
```

### Working with Contract Files

Decompile an entire contract from a file:

```python
import json
from src.bytecode_analyzer import analyze_bytecode_to_tac
from src.model_setup import SmartContractDecompiler

# Read bytecode from file
with open('contract_bytecode.txt', 'r') as f:
    bytecode = f.read().strip()

# Analyze and decompile
tac = analyze_bytecode_to_tac(bytecode)
decompiler = SmartContractDecompiler("models/final/smart_contract_decompiler")
solidity = decompiler.decompile_tac_to_solidity(tac)

# Save result
with open('decompiled_contract.sol', 'w') as f:
    f.write(solidity)

print("Decompilation complete! Saved to decompiled_contract.sol")
```

### Analyzing Multiple Contracts

Process a directory of bytecode files:

```python
import os
from pathlib import Path
from src.bytecode_analyzer import analyze_bytecode_to_tac
from src.model_setup import SmartContractDecompiler

# Initialize decompiler
decompiler = SmartContractDecompiler("models/final/smart_contract_decompiler")

# Process directory
bytecode_dir = Path("contracts/bytecode")
output_dir = Path("contracts/decompiled")
output_dir.mkdir(exist_ok=True)

for bytecode_file in bytecode_dir.glob("*.txt"):
    print(f"Processing {bytecode_file.name}...")
    
    # Read bytecode
    with open(bytecode_file, 'r') as f:
        bytecode = f.read().strip()
    
    # Decompile
    try:
        tac = analyze_bytecode_to_tac(bytecode)
        solidity = decompiler.decompile_tac_to_solidity(tac)
        
        # Save result
        output_file = output_dir / bytecode_file.with_suffix('.sol').name
        with open(output_file, 'w') as f:
            f.write(solidity)
        
        print(f"  ✓ Saved to {output_file}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("Batch processing complete!")
```

## API Usage

### Bytecode Analyzer API

```python
from src.bytecode_analyzer import (
    analyze_bytecode_to_tac,
    extract_function_selectors,
    recover_control_flow
)

# Full TAC generation
tac = analyze_bytecode_to_tac(bytecode)

# Extract function selectors only
selectors = extract_function_selectors(bytecode)
print(f"Found selectors: {selectors}")

# Recover control flow structures
cfg = recover_control_flow(bytecode)
print(f"Control flow: {cfg}")
```

### Model Setup API

```python
from src.model_setup import (
    SmartContractDecompiler,
    load_base_model,
    apply_lora_adapter
)

# Load base model without LoRA
base_model, tokenizer = load_base_model(
    model_name="meta-llama/Llama-3.2-3B",
    use_4bit=True
)

# Apply LoRA adapter
model = apply_lora_adapter(
    base_model,
    adapter_path="models/checkpoints/checkpoint-1000"
)

# Create decompiler instance
decompiler = SmartContractDecompiler(
    model=model,
    tokenizer=tokenizer
)
```

## Integration Examples

### Command-Line Tool

Create a simple CLI tool:

```python
#!/usr/bin/env python3
import sys
import argparse
from src.bytecode_analyzer import analyze_bytecode_to_tac
from src.model_setup import SmartContractDecompiler

def main():
    parser = argparse.ArgumentParser(description='Decompile EVM bytecode')
    parser.add_argument('bytecode', help='Bytecode hex string or file path')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    parser.add_argument('-m', '--model', default='models/final/smart_contract_decompiler',
                       help='Model path')
    parser.add_argument('-t', '--temperature', type=float, default=0.3,
                       help='Generation temperature')
    
    args = parser.parse_args()
    
    # Read bytecode
    if args.bytecode.startswith('0x'):
        bytecode = args.bytecode
    else:
        with open(args.bytecode, 'r') as f:
            bytecode = f.read().strip()
    
    # Decompile
    tac = analyze_bytecode_to_tac(bytecode)
    decompiler = SmartContractDecompiler(args.model)
    solidity = decompiler.decompile_tac_to_solidity(
        tac,
        temperature=args.temperature
    )
    
    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(solidity)
        print(f"Saved to {args.output}")
    else:
        print(solidity)

if __name__ == '__main__':
    main()
```

Usage:
```bash
# From hex string
python decompile.py "0x608060405234801561001057..."

# From file
python decompile.py contract.bin -o output.sol

# With custom temperature
python decompile.py contract.bin -t 0.1
```

### Web API Service

Create a Flask API endpoint:

```python
from flask import Flask, request, jsonify
from src.bytecode_analyzer import analyze_bytecode_to_tac
from src.model_setup import SmartContractDecompiler

app = Flask(__name__)

# Initialize decompiler once at startup
decompiler = SmartContractDecompiler("models/final/smart_contract_decompiler")

@app.route('/decompile', methods=['POST'])
def decompile():
    data = request.get_json()
    
    if 'bytecode' not in data:
        return jsonify({'error': 'bytecode field required'}), 400
    
    try:
        # Decompile
        bytecode = data['bytecode']
        temperature = data.get('temperature', 0.3)
        
        tac = analyze_bytecode_to_tac(bytecode)
        solidity = decompiler.decompile_tac_to_solidity(tac, temperature=temperature)
        
        return jsonify({
            'success': True,
            'solidity': solidity,
            'tac': tac
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Usage:
```bash
curl -X POST http://localhost:5000/decompile \
  -H "Content-Type: application/json" \
  -d '{"bytecode": "0x608060405234801561001057..."}'
```

## Performance Optimization

### GPU Memory Management

```python
import torch
from src.model_setup import SmartContractDecompiler

# Enable memory-efficient mode
decompiler = SmartContractDecompiler(
    model_path="models/final/smart_contract_decompiler",
    device_map="auto",
    low_cpu_mem_usage=True
)

# Clear cache between operations
torch.cuda.empty_cache()
```

### Caching Results

```python
from functools import lru_cache
from src.bytecode_analyzer import analyze_bytecode_to_tac

# Cache TAC generation results
@lru_cache(maxsize=1000)
def cached_analyze_bytecode(bytecode):
    return analyze_bytecode_to_tac(bytecode)

# Use cached version
tac = cached_analyze_bytecode(bytecode)
```

## Error Handling

### Robust Decompilation

```python
from src.bytecode_analyzer import analyze_bytecode_to_tac
from src.model_setup import SmartContractDecompiler

def safe_decompile(bytecode, model_path):
    """Safely decompile with error handling"""
    try:
        # Validate bytecode
        if not bytecode.startswith('0x'):
            raise ValueError("Bytecode must start with 0x")
        
        # Convert to TAC
        tac = analyze_bytecode_to_tac(bytecode)
        if not tac:
            raise ValueError("Failed to generate TAC")
        
        # Decompile
        decompiler = SmartContractDecompiler(model_path)
        solidity = decompiler.decompile_tac_to_solidity(tac)
        
        return {
            'success': True,
            'solidity': solidity,
            'tac': tac
        }
    
    except ValueError as e:
        return {'success': False, 'error': f'Validation error: {e}'}
    except FileNotFoundError as e:
        return {'success': False, 'error': f'Model not found: {e}'}
    except Exception as e:
        return {'success': False, 'error': f'Unexpected error: {e}'}

# Use safe version
result = safe_decompile(bytecode, "models/final/smart_contract_decompiler")
if result['success']:
    print(result['solidity'])
else:
    print(f"Error: {result['error']}")
```

## Next Steps

- Learn about [Training Pipeline](training-pipeline.md) for model training
- Explore [Data Format](data-format.md) for dataset specifications
- Check [Model Details](model-details.md) for configuration options
- Review [Troubleshooting](troubleshooting.md) for common issues
