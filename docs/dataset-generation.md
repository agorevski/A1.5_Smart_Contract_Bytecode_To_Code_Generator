# Dataset Generation: Quality Analysis & Improvement Recommendations

> **Context**: This document analyzes the current dataset generation pipeline
> (`download_hf_contracts.py`, `src/dataset_pipeline.py`, `src/bytecode_analyzer.py`,
> `src/local_compiler.py`) and identifies issues that limit training quality.
>
> **Key insight**: Dataset quality improvements will almost always yield larger
> gains than scaling from 32B → 70B model parameters.

---

## Current Pipeline Summary

The dataset generation pipeline works in three phases:

1. **Download** — Contracts are fetched from HuggingFace (`andstor/smart_contracts`) or Etherscan, with source-code-level deduplication via SHA-256 hashing.
2. **Compile & Extract** — Each contract is compiled with multiple solc versions (optimizer on/off), bytecode is analyzed into TAC via `BytecodeAnalyzer`, and Solidity functions are matched to TAC functions by 4-byte selector.
3. **Export** — Pairs are exported as JSONL with multi-layer deduplication (exact hash, normalized pair hash, body hash frequency cap).

The deduplication strategy is already strong. The issues below focus on **data quality, TAC fidelity, and training signal optimization**.

---

## Issue 1: No Compiler Version Normalization in Training Signal

### Problem

The TAC representation varies significantly by compiler version. Different solc versions produce different bytecode for identical source code due to:

- Optimizer behavior changes between versions
- ABI encoder V1 vs V2 (solc ≥0.8.0)
- Stack layout optimizations
- Different overflow check patterns (solc ≥0.8.0 adds implicit checks)

The metadata stores `compiler_version`, but it is **not included in the training prompt** (`_format_prompt` in `model_setup.py`). This means the model must implicitly learn to handle all compiler variations without any signal about which compiler produced the bytecode.

### Impact

- The model sees different TAC for the same Solidity output, creating conflicting training signals
- Optimizer-enabled bytecode looks structurally different from non-optimized bytecode (fewer blocks, merged operations, removed dead code)
- Pre-0.8.0 bytecode lacks overflow checks; post-0.8.0 includes `REVERT` paths for arithmetic overflow — the model must learn both patterns without knowing which to expect

### Recommended Fix

**Option A (Low effort)**: Add compiler version range to the TAC header comment:

```
function func_0xa9059cbb:
  // Compiler: solc 0.8.x (with optimizer)
  // Selector: 0xa9059cbb
  ...
```

This lets the model learn version-specific patterns without changing the training pipeline.

**Option B (Higher quality)**: Normalize TAC across compiler versions by post-processing:
- Strip compiler-inserted overflow checks and tag them as `// overflow_check`
- Normalize stack ordering differences
- Canonicalize equivalent block structures

**Where to change**: `_extract_tac_for_function()` in `dataset_pipeline.py` and `_extract_tac()` in `download_hf_contracts.py`.

---

## Issue 2: No Storage Slot Labeling

### Problem

The TAC uses raw storage operations with hex slot keys:

```
temp_42 = storage[0x0000000000000000000000000000000000000000000000000000000000000000]
storage[0x0000...0001] = temp_43
```

The model must infer that slot `0x00` is likely `owner` and slot `0x01` is likely `totalSupply` without any help. This is one of the hardest inference tasks in decompilation.

### Impact

- The model has no explicit signal linking storage slots to Solidity state variables
- For complex contracts with computed storage slots (mappings, dynamic arrays), the slot calculation is opaque hex arithmetic
- This is a primary source of incorrect variable naming in decompiled output

### Recommended Fix

**Option A (Use ABI)**: The ABI is already stored in the database but never used. For contracts with verified ABI, extract state variable declarations and annotate storage slots:

```
temp_42 = storage[slot_0]  // likely: address owner
```

**Option B (Use source analysis)**: Parse the Solidity source for state variable declarations and compute their expected storage slots based on declaration order:

```python
# Solidity storage layout: variables are assigned sequential slots
# address owner;    → slot 0
# uint256 supply;   → slot 1
# mapping(...) balances;  → slot 2 (but actual data at keccak256(key . 2))
```

**Option C (Hybrid)**: Add a `// Storage layout:` header to each TAC function listing known slot → variable mappings derived from the source code.

**Where to change**: Add a storage slot resolver to `bytecode_analyzer.py` or as a post-processing step in `dataset_pipeline.py`.

---

## Issue 3: Missing Revert/Panic Code Decoding

### Problem

The TAC represents revert operations as raw memory reads:

```
revert memory[temp_15:temp_16]
```

This discards critical information:
- Custom error messages (`require(condition, "message")`)
- Panic codes (`assert` failures produce `Panic(uint256)` with specific error codes)
- Custom errors (`error InsufficientBalance(uint256 required, uint256 available)`)

### Impact

- The model cannot learn the mapping between revert patterns and Solidity `require`/`assert`/`revert` statements
- Panic codes (e.g., 0x01 = assert, 0x11 = arithmetic overflow, 0x12 = division by zero) are not decoded
- Custom error selectors are not resolved

### Recommended Fix

Decode revert data in the TAC converter:

```
// Instead of:
revert memory[temp_15:temp_16]

// Generate:
revert "Insufficient balance"           // for Error(string) reverts
revert Panic(0x11)                      // for arithmetic overflow  
revert InsufficientBalance(required, available)  // for custom errors
```

Implementation approach:
1. Detect the `REVERT` pattern: preceding `MSTORE` writes the error selector + ABI-encoded data
2. Match error selector `0x08c379a0` → `Error(string)`, `0x4e487b71` → `Panic(uint256)`
3. For custom errors, use the ABI (when available) to resolve the selector

**Where to change**: `_convert_instruction_to_tac()` in `bytecode_analyzer.py`, specifically the REVERT handler. Add a new `_decode_revert_data()` method.

---

## Issue 4: No Failed/Partial Decompilation Examples

### Problem

The pipeline only stores successful selector-matched function pairs. When matching fails:
- A whole-contract fallback pair is created (`_create_fallback_pair`)
- The fallback pairs use the entire contract source as output — a fundamentally different format

There are **no training examples** showing:
- Partially reconstructed functions (e.g., control flow recovered but variable types unknown)
- "Unknown" patterns where the model should output uncertainty markers
- Functions where the TAC is valid but the corresponding Solidity is too complex to fully reconstruct

### Impact

- The model learns to **always produce complete Solidity**, even when the input TAC is insufficient or ambiguous
- At inference time, the model hallucinate plausible-looking but incorrect code rather than admitting uncertainty
- No calibration signal for model confidence

### Recommended Fix

**Option A (Partial ground truth)**: For functions where selector matching fails, still generate TAC and pair it with a partial template:

```solidity
// Partial decompilation — selector: 0xa9059cbb
// Control flow: 3 blocks, 1 conditional branch
function unknown_0xa9059cbb(/* params unknown */) public {
    // Storage read from slot 0
    // Conditional check on msg.sender
    // Storage write to slot 0
    // TODO: Full logic not reconstructed
}
```

**Option B (Negative examples)**: Add a small percentage (5–10%) of training examples where the output is explicitly uncertain:

```solidity
// Unable to decompile: complex proxy pattern
// TAC contains delegatecall to unknown implementation
```

**Option C (Confidence tags)**: Add confidence markers to the output that the model can learn to generate:

```solidity
function transfer(address to, uint256 amount) public /* confidence: high */ {
    // ... reconstructed logic
}
```

**Where to change**: `_create_function_pairs()` in `dataset_pipeline.py` — modify the fallback logic and add partial pair generation.

---

## Issue 5: Trivial Function Filtering Is Too Lenient

### Problem

The current `TRIVIAL_PATTERNS` in `download_hf_contracts.py` only catch three patterns:

```python
TRIVIAL_PATTERNS = [
    re.compile(r"^\s*\{\s*return\s+\w+\s*;\s*\}\s*$", re.DOTALL),     # { return x; }
    re.compile(r"^\s*\{\s*return\s+\d+\s*;\s*\}\s*$", re.DOTALL),     # { return 42; }
    re.compile(r"^\s*\{\s*\}\s*$", re.DOTALL),                          # { }
]
```

Many other trivial patterns still pass through:

- Single `SLOAD` + return (getter functions): `{ return owner; }`
- Simple setters: `{ myVar = newValue; }`
- Event-only functions: `{ emit Transfer(from, to, amount); }`
- Bool-return wrappers: `{ return true; }`
- Type casting: `{ return uint256(x); }`

### Impact

- Trivial functions dilute the training signal — the model spends capacity learning simple patterns that any rule-based system could handle
- With 238K+ function pairs, trivial functions can comprise 30–50% of the dataset
- Training time is wasted on patterns that don't need an LLM

### Recommended Fix

Expand the trivial patterns or implement a token-count based filter:

```python
# Add these patterns
TRIVIAL_PATTERNS += [
    re.compile(r"^\s*\{\s*return\s+\w+(\.\w+)*\s*;\s*\}\s*$", re.DOTALL),  # return obj.prop;
    re.compile(r"^\s*\{\s*\w+\s*=\s*\w+\s*;\s*\}\s*$", re.DOTALL),         # x = y;
    re.compile(r"^\s*\{\s*emit\s+\w+\([^)]*\)\s*;\s*\}\s*$", re.DOTALL),   # emit Event(...);
    re.compile(r"^\s*\{\s*return\s+(true|false)\s*;\s*\}\s*$", re.DOTALL),  # return true/false;
]

# Or: filter by Solidity body token count
MIN_MEANINGFUL_TOKENS = 20  # Skip functions with fewer than 20 tokens in the body
```

Additionally, consider **downsampling** rather than removing — keep 1–2 examples of each trivial pattern so the model still learns them, but don't let them dominate.

**Where to change**: `is_trivial_function()` in `download_hf_contracts.py` and the `min_body_length` parameter (currently default 50 chars, could be increased to 80–100).

---

## Issue 6: Missing Constructor / Receive / Fallback Handling

### Problem

The dispatcher pattern detection in `identify_functions()` only looks for `PUSH4 ... EQ ... JUMPI` sequences to find standard function selectors. This misses:

- **Constructor logic** — Executed during deployment, not present in runtime bytecode (but the training data uses runtime bytecode, so this is less critical)
- **`receive()` function** — Called for plain ETH transfers (no calldata). Typically at the fallback path after the dispatcher (no selector match + `CALLDATASIZE == 0`)
- **`fallback()` function** — Called when no selector matches. Often at the end of the dispatcher after all selector comparisons fail
- **Internal functions** — Called via `JUMP` (not through the dispatcher), so they have no selector

### Impact

- `receive()` and `fallback()` are common in real contracts but never matched to their Solidity source
- Internal functions (often the most complex logic) are excluded from training
- The `identify_functions()` fallback creates a single "fallback" function containing ALL unmatchable blocks, conflating distinct functions

### Recommended Fix

1. **Detect `receive()`**: After the dispatcher, if there's a `CALLDATASIZE ISZERO ... JUMPI` pattern leading to a code path, that's the `receive()` function
2. **Detect `fallback()`**: The default path after all selector comparisons fail is the `fallback()` function
3. **Detect internal functions**: Look for `JUMPDEST` targets that are called via `JUMP` from within identified functions but are not themselves dispatcher targets
4. **Split the catch-all**: Instead of one "fallback" function with all unmatched blocks, separate it into distinct reachable subgraphs

**Where to change**: `identify_functions()` in `bytecode_analyzer.py`.

---

## Issue 7: Shared Utility Blocks Inflate TAC per Function

### Problem

The block collection via graph traversal (`_collect_function_blocks` / `_collect_blocks`) follows all successors from a function's entry block. This means shared utility code — revert helpers, SafeMath-style checks, common validation — is included in **every function's TAC that references it**.

### Impact

- TAC for multiple functions contains identical blocks (e.g., the overflow-check revert block appears in every arithmetic function)
- This bloats the TAC size, potentially causing truncation of the unique logic
- The model sees the same blocks repeated hundreds of times across different functions, wasting context window space

### Recommended Fix

**Option A (Reference shared blocks)**: Instead of inlining shared blocks, reference them:

```
function func_0xa9059cbb:
  block_0x0080:
    ...
    if temp_5 goto shared_revert_0x0200    // reference, not inline
  
// Shared blocks (appended once at end of TAC):
shared_revert_0x0200:
    revert "SafeMath: overflow"
```

**Option B (Dedup at generation time)**: Track which blocks have been emitted already and skip them with a reference comment:

```
block_0x0200:
    // (see shared block in func_0x23b872dd)
```

**Option C (Prune utility blocks)**: Detect common patterns (overflow check, zero-address check, reentrancy guard) and replace them with semantic annotations:

```
    // OVERFLOW_CHECK(temp_5, temp_6)
    // ZERO_ADDRESS_CHECK(temp_2)
```

**Where to change**: `_collect_function_blocks()` in `dataset_pipeline.py` and `generate_function_tac()` in `bytecode_analyzer.py`.

---

## Issue 8: ABI Data Is Stored but Never Used

### Problem

The ABI (Application Binary Interface) is fetched from Etherscan and stored in the `contracts` table, but is never used during TAC generation or function matching. The ABI contains:

- Function names, parameter types, and return types
- Event definitions with indexed parameters
- Error definitions (for custom errors)
- State variable getter signatures (for public state variables)

### Impact

- Function parameter types are not annotated in the TAC — the model must infer `uint256` vs `address` vs `bytes32` from usage patterns alone
- Return types are lost — the TAC only shows `return memory[offset:size]`
- Event signatures could help match `LOG` instructions to specific events

### Recommended Fix

Parse the ABI and enrich the TAC with type information:

```
function transfer(address to, uint256 amount):   // from ABI
  // Returns: bool
  // Selector: 0xa9059cbb
  block_0x0080:
    temp_1 = calldata[0x04]   // param: address to
    temp_2 = calldata[0x24]   // param: uint256 amount
    ...
    log2(...)   // event: Transfer(address indexed from, address indexed to, uint256 value)
```

This dramatically improves the model's ability to produce correct function signatures and event emissions.

**Where to change**: Add ABI parsing to `_extract_tac_for_function()` in `dataset_pipeline.py`. Use `json.loads(abi_string)` to parse, then index by selector for lookup during TAC generation.

---

## Issue 9: Fragile Selector Computation for Complex Types

### Problem

The `_add_selectors()` function computes function selectors by splitting parameters on commas and taking the first whitespace-delimited token:

```python
param_types = [p.strip().split()[0] for p in params_str.split(",") if p.strip()]
```

This breaks for:
- **Tuple types**: `(uint256,address)[]` — the comma inside the tuple is split
- **Multi-word types**: `uint256[] memory` — only `uint256[]` is taken (this one works, but is fragile)
- **Nested tuples**: `(uint256,(address,bool))` — comma splitting destroys the structure
- **Fixed-size arrays with spaces**: `uint256 [3]` — only `uint256` is taken (incorrect)

### Impact

- Selector mismatches cause valid functions to be unmatched, reducing dataset size
- Mismatched selectors create incorrect training pairs (wrong TAC paired with wrong Solidity)

### Recommended Fix

Use a proper Solidity type parser or the ABI to compute selectors:

```python
# Option A: Use ABI for selector (most reliable)
import json
abi = json.loads(contract_data.abi)
for entry in abi:
    if entry["type"] == "function":
        # ABI already has canonical types
        param_types = [inp["type"] for inp in entry["inputs"]]
        canonical = f"{entry['name']}({','.join(param_types)})"

# Option B: Use a proper regex for Solidity types
# Handle tuples by matching balanced parentheses
```

**Where to change**: `_add_selectors_to_solidity_functions()` in `dataset_pipeline.py` and `_add_selectors()` in `download_hf_contracts.py`.

---

## Issue 10: Inconsistent Solidity Output Format

### Problem

The training data contains two fundamentally different output formats:

1. **Function-level pairs** (majority): Output is a single function with signature and body:
   ```solidity
   function transfer(address to, uint256 amount) public returns (bool) {
       require(balances[msg.sender] >= amount, "Insufficient balance");
       balances[msg.sender] -= amount;
       balances[to] += amount;
       return true;
   }
   ```

2. **Whole-contract fallback pairs**: Output is the entire contract source (potentially thousands of lines):
   ```solidity
   // SPDX-License-Identifier: MIT
   pragma solidity ^0.8.0;
   
   contract Token {
       // ... entire contract ...
   }
   ```

### Impact

- The model sees grossly different output lengths and formats for the same task
- Fallback pairs have a very different TAC-to-Solidity ratio than function-level pairs
- At inference time, the model may unpredictably switch between producing a single function or attempting to generate an entire contract

### Recommended Fix

1. **Remove whole-contract fallback pairs entirely** — they hurt more than they help
2. **Or normalize them**: If a fallback pair is used, extract and pair individual functions from the contract body even without selector matching (use position-based heuristics)
3. **Add format consistency check at export time**: Validate that all output values follow the same structural pattern

**Where to change**: `_create_fallback_pair()` in `dataset_pipeline.py` — either remove it or restructure to produce function-level pairs.

---

## Issue 11: No Variable Name Augmentation

### Problem

All training pairs use the original variable names from verified source code (`balances`, `totalSupply`, `owner`, `_msgSender`). The model learns these common naming conventions as part of the pattern.

At inference time, the model has no source names — only TAC temporaries (`temp_1`, `temp_2`). The model must infer semantically meaningful names purely from usage patterns. However, because it was only trained on "correctly named" examples, it may:

- Over-rely on memorized patterns (e.g., always naming the first state variable `owner`)
- Fail to produce meaningful names for unusual patterns it hasn't memorized
- Not distinguish between "I'm confident this is `balances`" and "I'm guessing this is `balances`"

### Impact

- Training signal conflates two tasks: (1) reconstructing logic and (2) naming variables
- The model is never trained on examples where names are arbitrary, so it can't separate naming from logic

### Recommended Fix

**Data augmentation**: For 20–30% of training examples, rename all variables to generic names before creating the pair:

```python
import re

def augment_variable_names(solidity_code: str) -> str:
    """Replace variable names with generic placeholders."""
    # Replace local variables with var_1, var_2, ...
    # Replace state variables with state_1, state_2, ...
    # Keep type names and Solidity keywords unchanged
    counter = 0
    def replace_var(match):
        nonlocal counter
        counter += 1
        return f"var_{counter}"
    # ... (careful regex to avoid replacing types/keywords)
```

This teaches the model that variable naming is a secondary task — the primary task is logic reconstruction.

**Where to change**: Add augmentation to `SmartContractDataset.__getitem__()` in `model_setup.py` (applied randomly during training, not in the stored dataset).

---

## Issue 12: Export-Time Dedup Cap May Be Too Generous

### Problem

The `max_body_dupes` parameter defaults to 5, allowing up to 5 copies of semantically identical function pairs in the training set. For very common patterns (ERC20 `transfer`, `approve`, `balanceOf`), even 5 copies result in those patterns dominating the training distribution.

### Impact

- Common ERC20/ERC721 patterns are overrepresented relative to rare DeFi functions
- The model becomes very good at standard token functions but underperforms on novel logic
- Training time is spent re-learning patterns the model has already mastered

### Recommended Fix

1. **Reduce default to 2–3**: Most patterns are learned within 1–2 exposures; additional copies have diminishing returns
2. **Implement frequency-aware sampling**: Instead of hard caps, use inverse-frequency weighting during training:

```python
# In SmartContractDataset.__getitem__():
# Weight rare functions higher, common functions lower
sample_weight = 1.0 / max(1, body_frequency_count)
```

3. **Stratified split**: Ensure train/val/test splits contain the same distribution of common vs rare patterns (the current random split may put all rare patterns in the test set)

**Where to change**: `export_training_data()` in `download_hf_contracts.py` (reduce `max_body_dupes` default) and `split_dataset()` in `train.py` (add stratification).

---

## Priority-Ordered Improvement Roadmap

Based on expected impact on model quality:

| Priority | Issue | Expected Impact | Effort |
|---|---|---|---|
| 🔴 **P0** | #2 — Storage slot labeling | Very high — directly addresses worst failure mode | Medium |
| 🔴 **P0** | #8 — Use ABI data for type annotations | Very high — free data already in the DB | Low |
| 🟠 **P1** | #3 — Decode revert/panic codes | High — teaches require/assert patterns | Medium |
| 🟠 **P1** | #5 — Better trivial function filtering | High — removes 20–30% noise from dataset | Low |
| 🟠 **P1** | #10 — Remove whole-contract fallback pairs | High — removes format inconsistency | Low |
| 🟡 **P2** | #1 — Compiler version annotation | Medium — reduces conflicting signals | Low |
| 🟡 **P2** | #7 — Shared block deduplication | Medium — reduces TAC bloat/truncation | Medium |
| 🟡 **P2** | #9 — Fix selector computation | Medium — increases matched pair count | Low |
| 🟡 **P2** | #12 — Reduce body dupe cap | Medium — improves distribution balance | Low |
| 🟢 **P3** | #4 — Partial decompilation examples | Medium — reduces hallucination | High |
| 🟢 **P3** | #6 — Constructor/receive/fallback functions | Medium — adds coverage for special functions | Medium |
| 🟢 **P3** | #11 — Variable name augmentation | Medium — better name generalization | Medium |

---

## Quick Wins (Implementable in < 1 Hour Each)

1. **Use ABI for type annotations** (#8): Parse `json.loads(abi)`, index by selector, annotate TAC parameters
2. **Remove fallback pairs** (#10): Delete or skip `_create_fallback_pair()` calls
3. **Reduce `max_body_dupes`** (#12): Change default from 5 → 2 in `download_hf_contracts.py`
4. **Add compiler annotation** (#1): Prepend `// Compiler: solc X.Y.Z` to TAC header
5. **Expand trivial patterns** (#5): Add 4–5 more regex patterns to `TRIVIAL_PATTERNS`

---

## Dataset Statistics to Monitor

After implementing improvements, track these metrics to verify quality gains:

```bash
# Count unique vs total pairs
sqlite3 data/contracts.db "SELECT COUNT(*), COUNT(DISTINCT body_hash), COUNT(DISTINCT pair_norm_hash) FROM function_pairs"

# Distribution of function complexity (by body length)
sqlite3 data/contracts.db "SELECT 
  CASE 
    WHEN LENGTH(solidity_code) < 100 THEN 'trivial (<100 chars)'
    WHEN LENGTH(solidity_code) < 500 THEN 'simple (100-500)'
    WHEN LENGTH(solidity_code) < 2000 THEN 'medium (500-2000)'
    ELSE 'complex (>2000)'
  END as complexity,
  COUNT(*) as count
FROM function_pairs GROUP BY complexity"

# Top 10 most duplicated function bodies
sqlite3 data/contracts.db "SELECT body_hash, COUNT(*) as cnt, MIN(function_name) as example_name FROM function_pairs WHERE body_hash IS NOT NULL GROUP BY body_hash ORDER BY cnt DESC LIMIT 10"

# Visibility distribution  
sqlite3 data/contracts.db "SELECT visibility, COUNT(*) FROM function_pairs GROUP BY visibility"

# Compiler version distribution
sqlite3 data/contracts.db "SELECT compiler_version, COUNT(*) FROM contracts GROUP BY compiler_version ORDER BY COUNT(*) DESC LIMIT 15"