# Comprehensive Research Enhancement Plan

> **Status: partially implemented in v2.0.0.** The repository now includes bytecode analysis, dataset export, supervised LoRA decompilation, exact TAC lookup, static vulnerability scanning, malicious classification, audit reports, and web/CLI integration. Research items that require trained vulnerability LLMs, DPO optimization, ensembles, or mixture-of-experts routing remain future work.
> E2E experiment-harness artifact contracts are now documented in `docs/e2e-harness-design.md`, but an autonomous optimizer/scheduler is not implemented.

## Research papers considered

| Paper | Idea used in this project | Current implementation status |
|---|---|---|
| SmartBugBert (`2504.05002v2`) | CFG/opcode features for bytecode-level security signals | CFG fragments and opcode features are implemented; no BERT vulnerability model is trained or served. |
| Smart-LLaMA-DPO (`2506.18245v1`) | Preference optimization and explainable vulnerability analysis | DPO config/preference-pair helpers exist; `train.py` still performs supervised fine-tuning only. |
| SAEL (`2507.22371v1`) | Combining expert signals | Reflected as modular components; adaptive MoE routing is not implemented. |
| LLMBugScanner (`2512.02069v1`) | Ensemble/consensus scanning | Not implemented; single decompiler plus static scanners are used. |
| Explainable AI malicious detection (`2512.08782v1`) | Opcode-feature classifier with explanations | Implemented as LightGBM train/load support with heuristic fallback and LIME only for fitted artifacts. |

## Current system summary

The implemented system is bytecode-first:

1. Generate/export TAC-to-Solidity training rows from HuggingFace contracts or Etherscan-verified contracts.
2. Compile source locally with compatible solc versions and extract runtime bytecode.
3. Analyze bytecode into per-function TAC, selectors, CFG blocks, and vulnerability fragments.
4. Train a supervised LoRA adapter on TAC-to-Solidity function pairs.
5. At inference time, use exact TAC lookup when available, otherwise call the decompiler model per function and assemble a deterministic contract scaffold.
6. Run optional malicious classification, static vulnerability detection, and audit-report aggregation.
7. Expose the workflow through `scripts/decompile.py`, `web/app.py`, and `PipelineOrchestrator`.

## E2E harness readiness

The current repository is ready for a human- or LLM-driven optimization loop at
the artifact-contract level:

- Data generation produces JSONL datasets, rejects, databases, and HF/Etherscan
  manifests with lineage, validation, row counts, and drop counts.
- `train.py` creates leakage-aware split manifests, data preflight reports,
  training run manifests, model artifacts, metrics, telemetry, and evaluation
  references.
- Evaluation writes machine-readable `results/eval_*.json` files with summary,
  detail rows, failure categories, prompt diagnostics, optional baseline
  comparison, and quality-gate results.

Planned harness work should add proposal and observation artifacts that link
these existing outputs into a repeated hypothesis -> run -> evaluation ->
decision loop. The design contract is in `docs/e2e-harness-design.md`; no current
CLI command automatically chooses or schedules the next experiment.

## Implemented capabilities

### Dataset and decompilation pipeline

- `download_hf_contracts.py` streams the HuggingFace dataset, compiles compatible contracts, persists `data/contracts.db`, and exports `data/hf_training_dataset.jsonl` with manifests and rejects.
- `src/dataset_pipeline.py` supports Etherscan collection, multi-file parsing, local compilation, function-pair extraction, and JSONL/CSV/Parquet export.
- `src/dataset_export_primitives.py` centralizes prompt sanitization, schema checks, deduplication hashes, TAC quality filters, and export metadata.
- `src/bytecode_analyzer.py` implements disassembly, CFG construction, selector/function discovery, per-function TAC, and vulnerability-fragment extraction.
- `src/tac_lookup.py` and `scripts/build_lookup_db.py` provide exact-match TAC lookup for inference acceleration and fidelity.
- `src/inference.py` provides the shared result schema used by CLI, web, and orchestration.

### Training, inference, and evaluation

- `train.py` implements dataset splitting, preflight validation, run manifests, supervised fine-tuning, checkpoint/final-model artifacts, eval-only mode, baseline comparison, and quality gates.
- `src/model_setup.py` implements Qwen2.5-Coder setup, LoRA, optional 4-bit NF4 loading, precision selection, optional DeepSpeed config, tokenization cache support, and TAC prompt truncation.
- `src/training_pipeline.py`, `src/replication_metrics.py`, and `src/evaluation_report.py` implement semantic/text metrics, structured replication metrics, Solidity validation, bytecode semantic/deployability checks, prompt diagnostics, and latest-results reporting.

### Security and reporting modules

- `src/opcode_features.py` implements opcode feature extraction, bytecode validation, TF-IDF/binary transforms, and entropy-based supervised splits.
- `src/vulnerability_detector.py` implements static CFG/source-pattern scanning for six vulnerability categories. It does not currently invoke a fine-tuned vulnerability LLM.
- `src/malicious_classifier.py` can fit/load a LightGBM classifier artifact and otherwise falls back to opcode-threshold heuristics; LIME explanations require a fitted model.
- `src/audit_report.py` aggregates classifier, vulnerability, and decompilation/source findings into risk-scored reports.
- `src/pipeline_orchestrator.py` coordinates the stages and preserves partial results.
- `web/app.py` exposes SSE decompilation, vulnerability scan, classification, audit report, health/readiness, cancellation, and GPU-stat endpoints.

## Not yet implemented or still research work

1. **DPO optimization loop** — Preference-pair helpers exist, but no `train.py` mode runs TRL DPO training or evaluates DPO checkpoints.
2. **Dedicated vulnerability LLM** — The detector has prompt templates and configuration fields, but the active scanner is static/heuristic.
3. **BERT/CodeT5/SAEL models** — No separate SmartBugBert, CodeT5/T5 prompt-tuned, or adaptive MoE model is trained or served.
4. **LLM ensemble consensus** — The production inference path uses one decompiler model plus exact lookup; no multi-model voting is implemented.
5. **Classifier calibration dataset** — The malicious classifier supports fitting/loading but the repository does not ship a calibrated production artifact.
6. **Formal equivalence** — Evaluation includes semantic and bytecode checks, but generated Solidity is not formally proven equivalent to original bytecode.

## Updated implementation roadmap

### Near term

- Keep dataset export and prompt sanitization aligned across HuggingFace and Etherscan paths.
- Expand exact TAC lookup provenance and decontamination checks for benchmark use.
- Train/evaluate current supervised LoRA adapters with reproducible manifests and baseline comparisons.
- Improve validation summaries for common failure categories: syntax/scaffold, prompt truncation, bytecode grounding, and deployability.
- Use the E2E harness artifact contracts to record experiment proposals,
  invariants, observations, and next-run linkage without claiming automated
  optimization exists yet.

### Medium term

- Add a real DPO training command that consumes `DPODatasetBuilder` pairs and records DPO-specific metrics.
- Train and calibrate the malicious classifier on labeled opcode datasets, then publish artifact metadata and evaluation results.
- Add vulnerability datasets and model-backed vulnerability explanations if the static scanner is insufficient.
- Evaluate selector/ABI enrichment without leaking source/compiler oracle information into prompts.

### Longer term

- Prototype ensemble or MoE routing only after single-model baselines and static scanners are measured.
- Add cross-contract and proxy-aware analyses where bytecode alone is ambiguous.
- Investigate stronger equivalence checks with differential execution or symbolic summaries.

## Design principles retained

1. **Bytecode-only prompts**: source/compiler metadata is used for lineage and evaluation, not as generation input.
2. **Static analysis before generation**: deterministic TAC, selectors, CFG, and lookup constrain the LLM task.
3. **Composable modules**: dataset generation, decompilation, classification, vulnerability detection, and reporting can run independently.
4. **Manifested artifacts**: dataset splits, training runs, lookup databases, and evaluation results should be reproducible and auditable.
5. **No overstated security claims**: scanner findings are heuristic/static unless backed by separately trained and evaluated models.
6. **Artifact-driven iteration**: future optimization should read manifests and
   evaluation JSON, preserve split/prompt invariants, and document proposals
   before re-running experiments.
