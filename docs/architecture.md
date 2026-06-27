# Architecture

This project is a bytecode-first smart contract decompilation and security-analysis system. The main boundary is:

```
source/metadata collection ──► runtime bytecode ──► bytecode-only TAC ──► Solidity-like reconstruction
                                      │                    │
                                      │                    ├─► exact TAC lookup
                                      │                    └─► LoRA-tuned code model
                                      ├─► opcode features ─────► malicious classifier
                                      └─► CFG fragments ───────► vulnerability scanner ──► audit report
```

Compiler/source/ABI information may be stored in datasets, manifests, traces, and selector provenance, but prompt text is intentionally restricted to bytecode-derived TAC plus compact bytecode/TAC statistics.

## Major entry points

| Surface | Entry point | Role |
|---|---|---|
| HuggingFace dataset generation | `download_hf_contracts.py` | Streams `andstor/smart_contracts`, compiles compatible contracts, writes `data/contracts.db`, `data/hf_training_dataset.jsonl`, reject files, and manifests. |
| Etherscan dataset generation | `src/dataset_pipeline.py` via `train.py` collection mode | Fetches verified source/metadata, compiles locally, extracts function pairs, and exports JSONL/CSV/Parquet plus manifests. |
| Training/evaluation | `train.py` | Builds or reuses leakage-aware splits, runs preflight checks, trains the decompiler, and writes evaluation reports. |
| CLI inference | `scripts/decompile.py` | Runs the shared bytecode inference path from the terminal. |
| Web inference/API | `web/app.py` | Flask UI/API with SSE decompilation, health/readiness, cancellation, GPU stats, vulnerability scan, classification, and audit-report endpoints. |
| Programmatic orchestration | `src/pipeline_orchestrator.py` | Coordinates classify → decompile/TAC-only fallback → detect vulnerabilities → report. |

## Dataset generation and export

Two data-generation paths feed the same training-row contract:

1. **HuggingFace exporter (`download_hf_contracts.py`)**
   - Streams the configured HuggingFace contract dataset instead of relying on a node RPC path.
   - Parses Solidity source, selects compatible `solc` versions, compiles runtime bytecode locally, and extracts selector-aligned TAC/Solidity function pairs.
   - Stores intermediate contract records in `data/contracts.db` and exports `data/hf_training_dataset.jsonl` with rejects and manifests.
   - Uses layered deduplication and quality filters before final JSONL rows are emitted.

2. **Etherscan builder (`src/dataset_pipeline.py`)**
   - Fetches verified contract metadata and source using Web3/Etherscan configuration.
   - Handles single-file and multi-file Etherscan source formats, local compilation, selector matching, split-safe metadata, and export to JSONL/CSV/Parquet.
   - `train.py --dataset-only` or normal training can invoke this path, then create `train_dataset.jsonl`, `val_dataset.jsonl`, `test_dataset.jsonl`, and `data/split_manifest.json`.

Shared export behavior lives in `src/dataset_export_primitives.py`: prompt sanitization, normalized hashing, schema validation, TAC quality rejection, final-row deduplication, length reports, and metadata normalization.

## Bytecode analysis and enrichment

| Module | Responsibility |
|---|---|
| `src/bytecode_analyzer.py` | Disassembles EVM bytecode with `evmdasm`, builds basic blocks/CFG, identifies dispatcher functions/selectors, emits per-function TAC, and extracts vulnerability-indicative CFG fragments. |
| `src/local_compiler.py` | Uses `py-solc-x` to install/select compatible solc versions and compile single- or multi-file Solidity into creation/runtime bytecode and ABI. |
| `src/selector_resolver.py` | Resolves 4-byte selectors through built-ins, SQLite registry/cache, local JSON cache, and optional 4byte.directory lookup. |
| `src/abi_enrichment.py` | Optionally parses ABI and source-derived storage-layout hints for TAC annotations and selector/event/error provenance. These annotations are not a prompt oracle. |
| `src/tac_lookup.py` + `scripts/build_lookup_db.py` | Build/query a normalized SQLite `data/tac_lookup.db` for exact TAC→Solidity body hits during inference. |
| `src/contract_reconstruction.py` | Builds semantic reconstruction plans, assembles contract scaffolds, infers quality/source summaries, and keeps per-function LLM calls at semantic boundaries. |

## Reconstruction and inference

`src/inference.py::run_bytecode_inference` is the shared schema used by the CLI, web app, and orchestrator:

1. Normalize generation and lookup configuration.
2. Analyze bytecode into an analyzer, per-function TAC, and combined TAC.
3. Resolve selectors and optionally query `TACLookup` for exact function-body matches.
4. Build a deterministic reconstruction plan and bytecode-derived per-function metadata.
5. For lookup misses, call `SmartContractDecompiler.decompile_tac_to_solidity()` from `src/model_setup.py` when a model is loaded; otherwise return TAC-only or model-not-loaded errors.
6. Assemble the full contract, validate generated Solidity where possible, and return functions, TAC, validation, lookup provenance, source summaries, quality, and trace data.

Default generation settings are greedy: `max_new_tokens=1024`, `temperature=0.1`, `do_sample=false`, `repetition_penalty=1.15`. The web server can cap or override these through `WEB_DEFAULT_*` and request-level generation settings.

## Model training boundary

`train.py` owns the end-to-end training/evaluation workflow. It creates or reuses grouped train/validation/test splits, runs dataset/preflight validation, and calls `src/model_setup.py::SmartContractModelTrainer`.

Training artifacts are written under `models/` by default:

- `models/checkpoints/` for Hugging Face Trainer checkpoints.
- `models/final_model/` for the saved adapter/model/tokenizer plus `model_config.json`.
- `models/final_model/training_metrics.json` and log-history files.
- `models/run_manifests/*.manifest.json` for run provenance.

Evaluation artifacts are written to `results/eval_*.json` and `latest_results.txt`; web inference traces default to `results/inference_traces/`.

## Security-analysis modules

| Module | Current behavior |
|---|---|
| `src/opcode_features.py` | Extracts opcode frequency, normalized frequency, TF-IDF/binary transforms, entropy-based supervised splits, and bytecode validation helpers. |
| `src/malicious_classifier.py` | Loads or trains a LightGBM classifier artifact (`malicious_classifier.pkl`) when available; otherwise falls back to opcode-threshold heuristics. LIME explanations require a fitted model and training matrix. |
| `src/vulnerability_detector.py` | Static CFG-fragment and source-pattern scanner for reentrancy, timestamp dependency, integer overflow/underflow, delegatecall, access-control, and selfdestruct findings. `use_llm` is configuration metadata; a vulnerability LLM is not wired into the scan path. |
| `src/audit_report.py` | Aggregates malicious classification, vulnerability scan results, optional decompilation/source findings, risk score, and recommendations into an audit report. |
| `src/pipeline_orchestrator.py` | Runs configured stages, preserves partial failures, and uses TAC-only decompilation when no model path is configured. |

Security outputs are static-analysis findings and classifier predictions, not formal proofs of exploitability.

## Web and CLI surfaces

- `scripts/decompile.py` validates bytecode input, optionally loads a trained model, and calls the shared inference path.
- `web/app.py` loads `WEB_MODEL_PATH`, `models/final_model`, or the newest `models/final_model*` artifact; `WEB_MOCK_MODEL` enables mock mode for tests/demos.
- `/api/decompile` streams progress and final results over Server-Sent Events.
- `/api/vulnerability-scan`, `/api/classify`, and `/api/audit-report` expose the security modules.
- `/api/health`, `/livez`, `/readyz`, `/api/gpu-stats`, and `/api/decompile/<request_id>/cancel` provide operational endpoints.

## Evaluation and reporting

`src/training_pipeline.py::SmartContractEvaluator` computes semantic similarity, normalized edit distance, BLEU, ROUGE-L, token accuracy, structural/complexity preservation, replication metrics, Solidity validation, and bytecode semantic/deployability checks. `src/replication_metrics.py` extracts structured Solidity facts, `src/evaluation_report.py` writes the human-readable latest report, and `train.py --eval-only` can compare against a baseline and enforce quality gates.

## E2E optimization harness readiness

There is no implemented autonomous experiment optimizer yet, but the current
phase boundaries are designed to be driven by one. A future harness should read
machine artifacts, preserve bytecode-only prompt invariants, propose one primary
change at a time, and link each re-run back to prior manifests/results.

| Phase | Implemented artifact boundary | Harness use |
|---|---|---|
| Data generation/export | HF/Etherscan dataset manifests, dataset JSONL, rejects JSONL, `data/contracts.db` | Decide whether data quality, deduplication, TAC extraction, or token budget should be changed before training. |
| Split/preflight | `data/split_manifest.json`, split JSONL files, preflight report/cache | Keep leakage-free test identity stable and fail closed on schema/token issues. |
| Training | `models/run_manifests/*.manifest.json`, checkpoints, `models/final_model/`, metrics/log history, throughput telemetry | Reproduce the run and compare hyperparameters, seeds, hardware, and runtime failures. |
| Evaluation | `results/eval_*.json`, `latest_results.txt`, run-manifest evaluation references | Compare against baselines, inspect quality gates, failure categories, worst samples, and prompt diagnostics. |
| Proposal/re-run | Planned proposal/observation JSON contracts | Record hypothesis, inputs, accepted changes, invariants, and next-run linkage. |

See `docs/e2e-harness-design.md` for the detailed artifact contracts and
proposal feedback-loop design.

## Known limitations

- Decompiled Solidity is best-effort and depends on the trained adapter, TAC coverage, prompt budget, and exact-lookup coverage.
- The system does not guarantee semantic equivalence to original source or recreate comments, naming, inheritance, modifiers, or compiler-specific source layout.
- Source/compiler/ABI metadata is intentionally excluded from model prompts to prevent oracle leakage; it may still appear in artifacts for auditability.
- DPO preference-pair helpers exist, but `train.py` does not run a DPO optimization loop.
- Ensemble/MoE vulnerability models from the research plan are not implemented as active model-training or inference paths.
- The future E2E optimization harness is documented as a design contract, not as
  an implemented scheduler or automatic proposal generator.
