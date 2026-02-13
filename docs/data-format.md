# Data Format

## Training Data (JSONL)

Each line in the training JSONL file:

```json
{
  "input": "<TAC representation>",
  "output": "<Solidity function body>",
  "metadata": {
    "function_name": "transfer",
    "function_signature": "transfer(address,uint256)",
    "selector": "0xa9059cbb",
    "visibility": "public",
    "is_payable": false,
    "is_view": false,
    "contract_address": "0x...",
    "compiler_version": "0.8.19",
    "optimizer_enabled": true
  }
}
```

## TAC Format

Three-Address Code is an intermediate representation emitted by `BytecodeAnalyzer`:

```
function func_0xa9059cbb:
  // Selector: 0xa9059cbb
  // Entry block: block_0x0080
  block_0x0080:
    // Successors: block_0x0095, block_0x00a0
    temp_1 = CALLDATALOAD 0x04
    temp_2 = CALLDATALOAD 0x24
    temp_3 = ISZERO temp_1
    COND_JUMP temp_3 -> block_0x00a0
  block_0x0095:
    temp_4 = SLOAD storage[temp_1]
    temp_5 = SUB temp_4, temp_2
    SSTORE storage[temp_1], temp_5
    RETURN 0x01
  block_0x00a0:
    REVERT
```

### Instruction types

| Category | Examples |
|----------|---------|
| Arithmetic | `ADD`, `SUB`, `MUL`, `DIV`, `MOD`, `EXP` |
| Comparison | `LT`, `GT`, `EQ`, `ISZERO` |
| Bitwise | `AND`, `OR`, `XOR`, `SHL`, `SHR` |
| Memory | `MLOAD`, `MSTORE`, `CALLDATALOAD` |
| Storage | `SLOAD`, `SSTORE` |
| Control flow | `JUMP`, `COND_JUMP`, `RETURN`, `REVERT`, `STOP` |
| Calls | `CALL`, `STATICCALL`, `DELEGATECALL`, `CREATE` |
| Logging | `LOG0`–`LOG4` |
| Environment | `CALLER`, `CALLVALUE`, `ADDRESS`, `TIMESTAMP`, `GAS` |

## Database Schema (`contracts.db`)

**`contracts`** — one row per downloaded contract

| Column | Type | Notes |
|--------|------|-------|
| `address` | TEXT PK | Contract address |
| `source_code` | TEXT | Verified Solidity source |
| `bytecode` | TEXT | Runtime bytecode |
| `compiler_version` | TEXT | Original compiler version |
| `processed` | BOOL | Whether TAC pairs have been generated |
| `source_hash` | TEXT | SHA-256 of normalized source (dedup) |

**`function_pairs`** — one row per TAC↔Solidity training pair

| Column | Type | Notes |
|--------|------|-------|
| `id` | INT PK | Auto-increment |
| `contract_address` | TEXT FK | Links to contracts |
| `function_name` | TEXT | Solidity function name |
| `tac_representation` | TEXT | TAC input |
| `solidity_code` | TEXT | Solidity output |
| `hash` | TEXT UNIQUE | MD5(TAC + body) — exact dedup |
| `pair_norm_hash` | TEXT UNIQUE | MD5(normalized TAC + normalized body) — semantic dedup |
| `body_hash` | TEXT | MD5(normalized body) — frequency cap |