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

Schema version 1 rows include `metadata.schema_version: 1`. Required fields are
non-empty string `input` and `output`; `metadata` is optional only for explicit
legacy compatibility and must be an object when present. Decontamination-critical
metadata values are validated by the shared schema helpers in
`src.dataset_pipeline`:

- `contract_address`: 20-byte `0x`-prefixed hex address
- `selector` / `function_selector`: 4-byte `0x`-prefixed hex selector
- `source_hash`, `source_code_hash`, `body_hash`, `input_hash`, `output_hash`:
  32- or 64-byte hex digests
- `optimizer_enabled`, `is_payable`, `is_view`: booleans
- `compiler_version`: normalized Solidity semver such as `0.8.20`

`train.py` validates JSON parse errors, missing/empty fields, metadata type, and
tokenizer-aware target/context length before training or eval-only runs. Use
`--skip-data-preflight` only for legacy smoke tests.

Recommended decontamination metadata keys, when available:

| Key | Purpose |
|-----|---------|
| `source_hash` / `source_code_hash` | Keep all functions from the same normalized source in one split |
| `contract_address` | Prevent address-level train/holdout leakage |
| `selector` / `function_signature` | Detect contract+function leakage |
| `body_hash` | Preserve exported normalized-body dedup groups |
| `input_hash` / `output_hash` | Optional exact TAC/Solidity hashes; `train.py` also computes exact hashes |

`train.py` writes `split_manifest.json` with seed, group-key precedence, source
file SHA-256, row/group counts, leakage validation for source hash, contract
address, contract+selector/signature, exact input hash, exact output hash, and
holdout coverage counts by compiler version, optimizer, visibility, source,
length bucket, and function family.

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
| `compile_status` | TEXT | Typed compile outcome such as `pending`, `processed`, `no_pairs`, `compile_failed`, or `analysis_failed` |
| `attempt_count` | INT | Number of compile attempts recorded for the contract |
| `last_error` | TEXT | First/most recent diagnostic message for failed or no-output generation |

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

**`compile_diagnostics`** — per-run compile/analysis diagnostics

| Column | Type | Notes |
|--------|------|-------|
| `run_id` | TEXT | Links diagnostics to a compile manifest |
| `contract_address` | TEXT | Contract that produced the diagnostic |
| `compiler_version` | TEXT | Solc version attempted, when available |
| `optimizer_enabled` | BOOL | Optimizer setting attempted, when available |
| `status` | TEXT | Typed status, e.g. `compile_failed`, `analysis_failed`, `prepare_no_jobs` |
| `error` | TEXT | Diagnostic text used in manifest summaries |

## Data-generation manifests

`download_hf_contracts.py` writes JSON manifests for download, compile, and export phases. By default they are placed next to the database/output (`data/hf_download_manifest.json`, `data/hf_compile_manifest.json`, and `<output>.manifest.json`); `--manifest-dir` writes all phase manifests to one directory.

Each manifest includes:

- source lineage (`andstor/smart_contracts`, config/split, requested and resolved revision when available);
- command arguments, current working directory, and git commit/dirty state;
- artifact paths, SHA-256 hashes, byte sizes, and JSONL row counts where applicable;
- row counts, typed status/drop counts, duplicate stats, and timing;
- compile failure summaries with top status/error groups and sample contract addresses.

Download manifests include Parquet streaming throughput and bounded
`parquet_batch_size` so large HuggingFace files are ingested without full
DataFrame materialization. Export manifests include
`validation.body_duplicate_cap`, `validation.token_length_filter`, and
`training_row_schema_version`. Rows exceeding the configured export
`--max-seq-length` are omitted from the main JSONL and written to
`<output>.rejects.jsonl` with source row, contract/function metadata, token
lengths, and reject reasons.

The Etherscan `DatasetBuilder` path writes
`smart_contract_dataset.jsonl.manifest.json` next to its export. That manifest
includes the input address-list hash/count when available, typed
`generation_diagnostics`, `dataset_filter_drops`, output artifact hashes, row
counts, and the training row schema version.

Partial decompilation placeholders (`metadata.partial == true`,
`Partial decompilation`, `TODO: Full logic not reconstructed`, or
`unknown_<selector>` targets) are excluded from the default supervised dataset.
Call `DatasetBuilder.export_dataset(..., include_partial=True)` only when you
want a separate `smart_contract_dataset.partial.jsonl` quarantine artifact.
