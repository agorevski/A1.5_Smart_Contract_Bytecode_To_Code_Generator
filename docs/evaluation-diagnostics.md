# Evaluation diagnostics and model-improvement guide

Use this guide after every `train.py --eval-only` or post-training evaluation.
It explains how to turn evaluation artifacts into concrete model, data, and
training changes without inventing results.

## Artifacts to inspect

- `results/eval_<timestamp>.json`: authoritative machine-readable output.
  Inspect `summary`, `aggregate_statistics`, `metadata_segments`,
  `benchmark_suites`, `quality_issue_summary`, and per-row `details`.
- `latest_results.txt`: non-developer readout. It includes an executive
  section, strengths, issue severities/priorities, next experiments, and a
  fenced `Machine-readable diagnostics` JSON block generated from the summary.

Future harnesses can also call:

```python
from src.evaluation_report import build_evaluation_diagnostics

diagnostics = build_evaluation_diagnostics(eval_payload["summary"])
```

## How to triage

1. Start with `overall_status`:
   - `healthy`: no metric-driven issues were detected.
   - `watch`: issues exist, but none are high severity.
   - `needs_attention`: high-severity model/data/eval issues exist.
   - `blocked`: critical issues exist.
   - `insufficient_evidence`: key metrics are missing or sample size is too
     small.
2. Fix the highest `priority` issue first (`P0`, then `P1`, `P2`, `P3`).
3. Use each issue's `likely_root_causes` to decide whether the next change
   belongs to data coverage, training configuration, prompt/context handling,
   decoding/model behavior, or evaluation metadata.
4. Run the listed `next_experiments` on the same split before changing broad
   thresholds or retraining on a larger dataset.

## Common issue categories

| Category | What it means | First action |
| --- | --- | --- |
| `inference_reliability` | Rows failed before metrics were computed. | Rerun failed rows with batch size 1 and lower generation length. |
| `model_behavior` | Text or structured behavior does not match references. | Inspect worst samples and weakest replication categories before retraining. |
| `syntax_and_deployability` | Generated Solidity fails syntax/compiler/deployability checks. | Cluster compiler/scaffold errors and repair the largest pattern. |
| `evaluation_data` | Metrics lack bytecode/compiler/opcode evidence or taxonomy clusters. | Backfill metadata or correct evaluation rows before trusting gates. |
| `prompting_and_context` | TAC/context was truncated. | Increase context budget or reduce nonessential prompt metadata. |
| `regression_risk` | Current metrics regressed against a baseline. | Ablate recent data/training/decoding changes on a fixed split. |

## Model-change loop

1. Select one issue and one suggested experiment.
2. Write the hypothesis in terms of a metric, e.g.
   `replication_by_category_micro.state_write.recall` should improve.
3. Modify only the relevant area: data slice, prompt budget, training config, or
   decoding config.
4. Re-run evaluation with the same test dataset and baseline comparison.
5. Keep the change only if the target metric improves without unacceptable
   regressions in semantic similarity, replication F1, grounded hallucination
   rate, Solidity validity, or bytecode semantic score.

