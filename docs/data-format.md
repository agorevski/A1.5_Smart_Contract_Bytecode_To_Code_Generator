# Data Format Specification

This document describes the data formats used throughout the Smart Contract Decompilation system.

## Training Data Format (JSONL)

The training dataset uses JSON Lines format, with one JSON object per line.

### Schema

```json
{
  "input": "string - TAC representation",
  "output": "string - Solidity code",
  "metadata": {
    "function_name": "string",
    "function_signature": "string",
    "selector": "string",
    "visibility": "string",
    "state_mutability": "string",
    "is_payable": "boolean",
    "has_return": "boolean",
    "contract_address": "string",
    "complexity_score": "number"
  }
}
```

### Example Entry

```json
{
  "input": "### TAC Representation:\nfunction_selector_0x70a08231:\n  v0 = CALLDATALOAD 0x04\n  v1 = SLOAD storage[mapping_0][v0]\n  RETURN v1",
  "output": "function balanceOf(address account) public view returns (uint256) {\n    return _balances[account];\n}",
  "metadata": {
    "function_name": "balanceOf",
    "function_signature": "balanceOf(address)",
    "selector": "0x70a08231",
    "visibility": "public",
    "state_mutability": "view",
    "is_payable": false,
    "has_return": true,
    "contract_address": "0x1234567890abcdef1234567890abcdef12345678",
    "complexity_score": 2.3
  }
}
```

## TAC Representation Format

Three-Address Code (TAC) is a structured intermediate representation of EVM bytecode.

### Basic Structure

```text
function_selector_0x<selector>:
  <instruction>
  <instruction>
  ...
```

### Instructions

#### Memory Operations

```text
v0 = CALLDATALOAD 0x04          # Load calldata at offset 0x04
v1 = MLOAD 0x40                 # Load memory at offset 0x40
MSTORE 0x80, v2                 # Store v2 to memory at 0x80
```

#### Storage Operations

```text
v0 = SLOAD storage[mapping_0][v1]           # Load from mapping
SSTORE storage[mapping_0][v1], v2           # Store to mapping
v3 = SLOAD storage[array_0][index]          # Load from array
SSTORE storage[variable_0], v4              # Store to simple variable
```

#### Arithmetic Operations

```text
v0 = ADD v1, v2                 # Addition
v1 = SUB v3, v4                 # Subtraction
v2 = MUL v5, v6                 # Multiplication
v3 = DIV v7, v8                 # Division
v4 = MOD v9, v10                # Modulo
```

#### Comparison Operations

```text
v0 = LT v1, v2                  # Less than
v1 = GT v3, v4                  # Greater than
v2 = EQ v5, v6                  # Equal
v3 = ISZERO v7                  # Is zero
```

#### Control Flow

```text
if v0:
  <instructions>
else:
  <instructions>

while v1:
  <instructions>

REVERT                          # Revert transaction
RETURN v2                       # Return value
```

#### Special Operations

```text
LOG3 topic_Transfer, v0, v1, v2  # Emit event
CALL contract_addr, function, args  # External call
```

### Complete Example

```text
function_selector_0xa9059cbb:
  # Function: transfer(address,uint256)
  
  # Load parameters
  v0 = CALLDATALOAD 0x04          # to address
  v1 = CALLDATALOAD 0x24          # amount
  
  # Validation
  v2 = ISZERO v0
  if v2:
    REVERT                         # require(to != address(0))
  
  # Load sender balance
  v3 = SLOAD storage[mapping_0][msg.sender]
  
  # Check balance
  v4 = LT v3, v1
  if v4:
    REVERT                         # require(balance >= amount)
  
  # Update sender balance
  v5 = SUB v3, v1
  SSTORE storage[mapping_0][msg.sender], v5
  
  # Load receiver balance
  v6 = SLOAD storage[mapping_0][v0]
  
  # Update receiver balance
  v7 = ADD v6, v1
  SSTORE storage[mapping_0][v0], v7
  
  # Emit event
  LOG3 topic_Transfer, msg.sender, v0, v1
  
  # Return success
  RETURN 0x01
```

## Solidity Output Format

The expected Solidity output follows standard Solidity syntax.

### Function Structure

```solidity
function <name>(<parameters>) <visibility> <mutability> returns (<return_type>) {
    <body>
}
```

### Visibility Modifiers

- `public` - Accessible externally and internally
- `external` - Only accessible externally
- `internal` - Only accessible internally
- `private` - Only accessible within contract

### State Mutability

- `pure` - Does not read or modify state
- `view` - Reads but does not modify state
- `payable` - Can receive Ether
- (none) - Can modify state

### Example Functions

#### Simple View Function

```solidity
function balanceOf(address account) public view returns (uint256) {
    return _balances[account];
}
```

#### State-Modifying Function

```solidity
function transfer(address to, uint256 amount) public returns (bool) {
    require(to != address(0), "Invalid address");
    require(_balances[msg.sender] >= amount, "Insufficient balance");
    
    _balances[msg.sender] -= amount;
    _balances[to] += amount;
    
    emit Transfer(msg.sender, to, amount);
    return true;
}
```

#### Payable Function

```solidity
function deposit() public payable {
    require(msg.value > 0, "Must send ETH");
    _balances[msg.sender] += msg.value;
    emit Deposit(msg.sender, msg.value);
}
```

## Metadata Fields

### Required Metadata

| Field | Type | Description |
|-------|------|-------------|
| `function_name` | string | Name of the function |
| `function_signature` | string | Full function signature |
| `selector` | string | 4-byte function selector (hex) |
| `visibility` | string | public/external/internal/private |
| `state_mutability` | string | pure/view/payable/none |

### Optional Metadata

| Field | Type | Description |
|-------|------|-------------|
| `is_payable` | boolean | Whether function accepts Ether |
| `has_return` | boolean | Whether function returns value |
| `contract_address` | string | Source contract address |
| `complexity_score` | number | Cyclomatic complexity |
| `solidity_version` | string | Compiler version |
| `optimization_enabled` | boolean | Whether optimizer was used |

## Dataset Statistics

### Size Distribution

Following the paper specifications:

- **Minimum length**: 50 tokens
- **Maximum length**: 20,000 tokens
- **Average length**: ~150 tokens
- **Median length**: ~100 tokens

### Visibility Distribution

Typical distribution in the dataset:

- **public**: 45%
- **external**: 30%
- **internal**: 20%
- **private**: 5%

### State Mutability Distribution

- **view/pure**: 40%
- **state-modifying**: 55%
- **payable**: 5%

### Complexity Distribution

- **Simple** (score 1-3): 40%
- **Moderate** (score 4-7): 35%
- **Complex** (score 8-15): 20%
- **Very Complex** (score 15+): 5%

## File Formats

### Raw Contract Data

**Format**: JSON
**Location**: `data/raw/`

```json
{
  "address": "0x...",
  "source_code": "contract MyToken { ... }",
  "bytecode": "0x608060405234801561001057600080fd5b50...",
  "abi": [...],
  "compiler_version": "v0.8.19+commit.7dd6d404",
  "optimization_enabled": true,
  "runs": 200
}
```

### Processed Function Pairs

**Format**: JSONL
**Location**: `data/processed/`

One function pair per line (see Training Data Format above).

### Split Datasets

**Format**: JSONL
**Locations**:
- Training: `data/train/train.jsonl`
- Validation: `data/val/val.jsonl`
- Test: `data/test/test.jsonl`

## Data Loading

### Python API

```python
import json

# Load JSONL dataset
def load_dataset(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Load specific fields only
def load_inputs_outputs(path):
    inputs, outputs = [], []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            inputs.append(item['input'])
            outputs.append(item['output'])
    return inputs, outputs
```

### Batch Loading

```python
# Load dataset in batches
def load_dataset_batched(path, batch_size=1000):
    batch = []
    with open(path, 'r') as f:
        for line in f:
            batch.append(json.loads(line))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

# Use batched loading
for batch in load_dataset_batched('data/train/train.jsonl'):
    process_batch(batch)
```

## Data Validation

### Validate TAC Format

```python
import re

def validate_tac(tac):
    # Check function selector format
    if not re.match(r'function_selector_0x[0-9a-f]{8}:', tac):
        return False
    
    # Check for valid instructions
    valid_ops = ['CALLDATALOAD', 'SLOAD', 'SSTORE', 'ADD', 'SUB', 
                 'RETURN', 'REVERT', 'if', 'while', 'LOG3']
    
    for op in valid_ops:
        if op in tac:
            return True
    
    return False
```

### Validate Solidity Format

```python
def validate_solidity(code):
    # Check basic structure
    if not code.strip().startswith('function'):
        return False
    
    # Check for matching braces
    if code.count('{') != code.count('}'):
        return False
    
    # Check for valid visibility
    visibilities = ['public', 'external', 'internal', 'private']
    has_visibility = any(v in code for v in visibilities)
    
    return has_visibility
```

## Data Quality Metrics

### Quality Criteria

1. **Syntactic Validity**: Both TAC and Solidity must parse correctly
2. **Length Bounds**: Within 50-20,000 token range
3. **Semantic Consistency**: TAC and Solidity represent same logic
4. **Completeness**: All required metadata present
5. **Uniqueness**: No duplicate function pairs

### Quality Filtering

```python
def filter_high_quality(dataset):
    filtered = []
    
    for item in dataset:
        # Check length
        input_len = len(item['input'].split())
        output_len = len(item['output'].split())
        if not (50 <= input_len <= 20000 and 10 <= output_len <= 20000):
            continue
        
        # Validate formats
        if not validate_tac(item['input']):
            continue
        if not validate_solidity(item['output']):
            continue
        
        # Check metadata completeness
        required_fields = ['function_name', 'selector', 'visibility']
        if not all(f in item['metadata'] for f in required_fields):
            continue
        
        filtered.append(item)
    
    return filtered
```

## Next Steps

- Review [Training Pipeline](training-pipeline.md) for data usage
- Check [Architecture](architecture.md) for data flow
- See [Usage Guide](usage.md) for working with data
