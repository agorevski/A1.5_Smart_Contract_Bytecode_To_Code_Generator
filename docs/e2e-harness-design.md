# E2E Optimization Harness Design

## Status and intent

This is a design/readiness contract for a future automated training/evaluation
harness. The repository does **not** currently ship a single autonomous
optimizer, experiment scheduler, or proposal generator. Existing scripts already
emit enough structured data for a human or LLM agent to drive the loop:

```text
data generation -> split/preflight -> training -> evaluation -> proposal -> re-run
```

Agents should make decisions from JSON manifests and evaluation artifacts, not
from ad hoc logs or unrecorded benchmark claims.

## Read these inputs first

Before proposing an experiment, read:

- `docs/architecture.md` for implemented pipeline boundaries.
- `docs/data-format.md` for training-row, manifest, split, and reject formats.
- `docs/training-recommendations.md` for current training/evaluation commands and
  hardware guidance.
- `docs/runbook.md` for operational commands and troubleshooting.
- `reference/new-research-plan.md` for implemented versus research-only ideas.
- Latest artifacts, when present:
  - `data/manifests/hf_*_manifest.json` or
    `data/smart_contract_dataset.*.manifest.json`
  - `data/split_manifest.json`
  - `models/run_manifests/*.manifest.json`
  - `models/final_model/model_config.json`
  - `models/final_model/training_metrics.json`
  - `models/final_model/training_log_history.{json,csv}`
  - `models/training_throughput.{json,csv}`
  - `results/eval_*.json`
  - `latest_results.txt`

Treat `latest_results.txt` as a human-readable report. Use `results/eval_*.json`
for machine decisions.

## Invariants for every loop

1. Keep prompts bytecode-only: source, compiler, optimizer, and ABI facts may be
   used for lineage or evaluation segmentation, but must not become generation
   prompt oracle data.
2. Compare models on the same test split hash, evaluation limit/seed, generation
   settings, and baseline tolerance unless the proposal explicitly changes the
   evaluation protocol.
3. Record artifact `path`, `sha256`, `row_count` or `size_bytes`, command args,
   git state, seed, status, and timestamps at each phase.
4. Preserve rejects and failure diagnostics. A low-quality export is a data issue,
   not a reason to tune model hyperparameters first.
5. Do not promote results from smoke runs or tiny eval samples as benchmark
   evidence.

## Phase boundaries

| Phase | Current inputs | Current outputs | Handoff checks |
|---|---|---|---|
| Data generation/export | HF or Etherscan source config, compiler settings, export filters | Dataset JSONL, rejects JSONL, `data/contracts.db`, HF/Etherscan export manifests | Export status completed, rows exported > 0, duplicate/TAC/token filters understood |
| Split/preflight | Source JSONL, split seed/ratios, tokenizer/max sequence length | `train_dataset.jsonl`, `val_dataset.jsonl`, `test_dataset.jsonl`, `data/split_manifest.json`, preflight report in run manifest/cache | Leakage, coverage, split quality, and preflight status passed or explicitly waived |
| Training | Split artifacts, model/hyperparameter proposal, seed, hardware settings | `models/checkpoints/`, `models/final_model/`, training metrics/log history, throughput telemetry, run manifest | Run manifest status completed; model artifact, metrics, and telemetry are linked |
| Evaluation | Model artifact, test split, eval settings, optional baseline | `results/eval_*.json`, `latest_results.txt`, run-manifest evaluation references | Results JSON exists; quality gate/baseline comparison interpreted |
| Proposal | Prior proposal, run manifest, eval summary/details, rejects, telemetry | Planned `experiment_proposal` JSON | One primary hypothesis, explicit invariant fields, expected metric movement |
| Re-run | Accepted proposal and current artifacts | Next run manifest and eval result linked to proposal ID | Artifacts link predecessor paths/hashes and outcome status |

## Implemented artifact contracts

### Training row JSONL

Each non-empty line is a supervised example:

```json
{
  "input": "sanitized bytecode-only TAC",
  "output": "target Solidity function body",
  "metadata": {
    "schema_version": 1,
    "selector": "0x00000000",
    "function_name": "optional",
    "contract_address": "optional",
    "body_hash": "optional",
    "input_hash": "optional",
    "output_hash": "optional"
  }
}
```

Required: non-empty string `input`, non-empty string `output`, object
`metadata`, and `metadata.schema_version: 1` unless legacy mode is explicitly
enabled.

### Data export manifests

HF export manifests use `manifest_kind: hf_export` and
`manifest_schema_version: 1`. Etherscan export manifests use
`manifest_kind: etherscan_dataset_export` and `manifest_schema_version: 1`.
Harness readers should require at least:

```json
{
  "manifest_kind": "hf_export or etherscan_dataset_export",
  "manifest_schema_version": 1,
  "status": "completed or failed",
  "generated_at": "ISO-8601 UTC timestamp",
  "command": {"argv": []},
  "git": {"commit": "optional", "dirty": true},
  "parameters": {},
  "artifacts": {
    "dataset_or_jsonl": {"path": "...", "sha256": "...", "row_count": 0},
    "rejects_dataset_or_jsonl": {"path": "...", "row_count": 0}
  },
  "row_counts": {},
  "drop_counts": {},
  "validation": {},
  "training_row_schema_version": 1
}
```

Current field names differ slightly by exporter (`jsonl` versus `dataset`,
`rejects_jsonl` versus `rejects_dataset`), so readers should normalize by
artifact role instead of relying only on exact key names.

### Split manifest

`train.py` writes `manifest_kind: dataset_split` with `schema_version: 2`:

```json
{
  "manifest_kind": "dataset_split",
  "schema_version": 2,
  "created_at": "ISO-8601 UTC timestamp",
  "source_dataset": {"path": "...", "sha256": "...", "row_count": 0},
  "input_sha256": "...",
  "parameters": {},
  "row_counts": {"source": 0, "train": 0, "val": 0, "test": 0},
  "group_counts": {},
  "outputs": {
    "train": {"path": "...", "sha256": "...", "row_count": 0},
    "val": {"path": "...", "sha256": "...", "row_count": 0},
    "test": {"path": "...", "sha256": "...", "row_count": 0}
  },
  "leakage_validation": {"status": "passed"},
  "coverage": {"status": "passed"},
  "split_quality": {"status": "passed"},
  "cache": {"reused": false}
}
```

### Training run manifest

Every `train.py` mode writes `manifest_kind: training_run` with
`schema_version: 1`:

```json
{
  "manifest_kind": "training_run",
  "schema_version": 1,
  "run_id": "...",
  "status": "running, completed, or failed",
  "started_at": "ISO-8601 UTC timestamp",
  "finished_at": "optional ISO-8601 UTC timestamp",
  "command": {"argv": [], "args": {}},
  "environment": {},
  "runtime": {"global_seed": 42},
  "git": {},
  "datasets": {},
  "training": {},
  "evaluation": {},
  "telemetry": {},
  "artifacts": {},
  "timing": {},
  "error": "optional failure object"
}
```

### Evaluation result JSON

`results/eval_*.json` is the primary optimization input:

```json
{
  "summary": {
    "num_evaluated": 0,
    "num_succeeded": 0,
    "num_failed": 0,
    "failure_rate": 0.0,
    "model_path": "...",
    "test_dataset": "...",
    "results_path": "...",
    "baseline_results_path": "optional",
    "baseline_comparison": "optional object",
    "quality_gate": "optional object",
    "quality_issue_summary": "optional object",
    "worst_samples": []
  },
  "details": [
    {
      "dataset_index": 0,
      "success": true,
      "input_hash": "...",
      "output_hash": "...",
      "generation_config": {},
      "metrics": {},
      "prompt_diagnostics": "optional object"
    }
  ]
}
```

## Planned harness-only contracts

These files are design targets for future automation and are not enforced by the
current code.

### Experiment proposal

Recommended path: `experiments/<experiment_id>/experiment_proposal.json`.

```json
{
  "artifact_kind": "experiment_proposal",
  "schema_version": 1,
  "proposal_id": "YYYYMMDDTHHMMSSZ-short-name",
  "created_at": "ISO-8601 UTC timestamp",
  "based_on": {
    "run_manifest": {"path": "...", "sha256": "..."},
    "evaluation_results": {"path": "...", "sha256": "..."},
    "split_manifest": {"path": "...", "sha256": "..."}
  },
  "hypothesis": {
    "problem": "single observed bottleneck",
    "expected_effect": "metric or failure category expected to improve",
    "rationale": "evidence from artifacts"
  },
  "change_set": [
    {
      "phase": "data, split, training, evaluation, or inference",
      "field": "flag or config path",
      "from": "previous value",
      "to": "new value"
    }
  ],
  "invariants": {
    "test_dataset_sha256": "...",
    "eval_seed": 123,
    "eval_limit": null,
    "eval_max_new_tokens": 256,
    "bytecode_only_prompts": true
  },
  "execution": {
    "command": ["uv", "run", "python", "train.py"],
    "expected_artifacts": ["models/run_manifests/*.manifest.json", "results/eval_*.json"]
  },
  "acceptance_criteria": [
    {"metric": "failure_rate", "op": "<=", "value": 0.0},
    {"metric": "baseline_regressions", "op": "<=", "value": 0}
  ],
  "risk_controls": ["one primary change", "preserve split hash", "record dirty git state"]
}
```

### Experiment observation

Recommended path: `experiments/<experiment_id>/experiment_observation.json`.

```json
{
  "artifact_kind": "experiment_observation",
  "schema_version": 1,
  "proposal_id": "...",
  "status": "completed, failed, or inconclusive",
  "observed_artifacts": {
    "run_manifest": {"path": "...", "sha256": "..."},
    "evaluation_results": {"path": "...", "sha256": "..."}
  },
  "metric_deltas": {
    "semantic_similarity_mean": {"before": 0.0, "after": 0.0, "delta": 0.0}
  },
  "quality_gate": {"status": "passed or failed"},
  "failure_analysis": {
    "dominant_categories": [],
    "regressions": [],
    "notes": ""
  },
  "next_recommendation": {
    "decision": "promote, retry, revert, or propose_next",
    "reason": ""
  }
}
```

## Choosing the next experiment

Use this priority order:

1. **Artifact validity first**: if export, split, or preflight failed, fix data
   generation/filtering before training changes.
2. **Runtime failures next**: if training or evaluation failed from OOM or missing
   artifacts, adjust batch size, sequence length, quantization, checkpoint resume,
   or eval batch size before judging quality.
3. **Prompt budget issues**: if `prompt_diagnostics` or rejects show truncation or
   overlength rows, test sequence-length and filtering changes while preserving
   the same held-out split for model comparisons.
4. **Syntax/scaffold failures**: inspect `quality_issue_summary` and worst samples;
   then try prompt-shape, target cleaning, or model/hyperparameter changes.
5. **Bytecode grounding failures**: prioritize TAC quality, bytecode-derived
   metadata, exact lookup coverage, and evaluation diagnostics.
6. **Clean quality plateau**: vary one model/training factor at a time (base
   model, LoRA rank/alpha/dropout, learning rate, epochs, context length) and use
   baseline comparison plus quality gates.

Each proposal should identify exactly one primary bottleneck and one primary
change. Bundle unrelated changes only when the current artifacts show they are
causally coupled.

## Readiness audit

Implemented foundations:

- Structured data-export, split, training, and evaluation artifacts.
- Hashes, row counts, rejects, split leakage checks, preflight checks, baseline
  comparisons, quality gates, worst samples, and failure categories.
- Re-runnable CLI entry points for dataset-only, train, and eval-only modes.

Remaining harness work:

- No global experiment ledger or accepted proposal schema is enforced.
- No scheduler links proposal IDs to subsequent `train.py` manifests.
- No automatic proposal generator summarizes deltas and writes observations.
- No promotion artifact defines when a model becomes the new baseline.

