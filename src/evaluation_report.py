"""Human-readable evaluation report generation."""

from __future__ import annotations

import difflib
import json
import math
import platform
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


DEFAULT_LATEST_RESULTS_PATH = "latest_results.txt"

_SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
_PRIORITY_BY_SEVERITY = {
    "critical": "P0",
    "high": "P1",
    "medium": "P2",
    "low": "P3",
    "info": "P3",
}


def write_latest_results_report(
    *,
    summary: Mapping[str, Any],
    model_path: str,
    test_dataset_path: str,
    results_json_path: str,
    latest_results_path: str = DEFAULT_LATEST_RESULTS_PATH,
    started_at: Optional[float] = None,
    finished_at: Optional[float] = None,
    argv: Optional[Sequence[str]] = None,
    eval_limit: Optional[int] = None,
    world_size: int = 1,
) -> str:
    """Write the repo-root latest model-quality report and return its path."""
    output_path = Path(latest_results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text = format_latest_results_report(
        summary=summary,
        model_path=model_path,
        test_dataset_path=test_dataset_path,
        results_json_path=results_json_path,
        started_at=started_at,
        finished_at=finished_at,
        argv=argv,
        eval_limit=eval_limit,
        world_size=world_size,
    )
    output_path.write_text(text, encoding="utf-8")
    return str(output_path)


def build_evaluation_diagnostics(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Build an actionable, machine-readable model improvement plan."""
    metrics_used: Dict[str, Any] = {}
    strengths: list[Dict[str, Any]] = []
    issues: list[Dict[str, Any]] = []
    caveats: list[str] = []

    def metric(name: str, *aliases: str) -> Optional[float]:
        value = _first_numeric_summary_value(summary, name, *aliases)
        if value is not None:
            metrics_used[name] = _json_safe_number(value)
        return value

    num_evaluated = metric("num_evaluated")
    if num_evaluated is not None and num_evaluated < 30:
        caveats.append(
            "Fewer than 30 examples were evaluated; treat this as a smoke test before changing model policy."
        )

    failure_rate = metric("failure_rate")
    if failure_rate is not None:
        if failure_rate > 0:
            severity = "high" if failure_rate >= 0.10 else "medium"
            issues.append(
                _diagnostic_issue(
                    issue_id="generation_failures",
                    category="inference_reliability",
                    severity=severity,
                    title="Some evaluation rows failed before quality metrics could be computed.",
                    evidence=[_diagnostic_metric_text("failure_rate", failure_rate, percent=True)],
                    likely_root_causes=[
                        "Inference runtime errors, CUDA memory pressure, or malformed dataset rows.",
                        "Generation configuration may be too aggressive for the available model/context size.",
                    ],
                    suggested_experiments=[
                        _experiment(
                            "retry-failed-eval-rows",
                            "Rerun only failed rows with batch size 1 and lower max_new_tokens to separate data errors from runtime errors.",
                            "evaluation",
                            "failure_rate returns to 0.00%",
                        )
                    ],
                )
            )
        else:
            strengths.append(
                _diagnostic_strength(
                    "no_generation_failures",
                    "Evaluation completed without generation failures.",
                    [_diagnostic_metric_text("failure_rate", failure_rate, percent=True)],
                )
            )

    semantic_mean = metric("semantic_similarity_mean", "semantic_similarity")
    pct_high_similarity = metric("pct_above_0.8_similarity")
    if _below_any((semantic_mean, 0.82), (pct_high_similarity, 0.78)):
        severity = (
            "high" if _below_any((semantic_mean, 0.70), (pct_high_similarity, 0.50)) else "medium"
        )
        evidence = _available_evidence(
            _target_evidence("semantic_similarity_mean", semantic_mean, ">=", 0.82),
            _target_evidence(
                "pct_above_0.8_similarity", pct_high_similarity, ">=", 0.78, percent=True
            ),
        )
        issues.append(
            _diagnostic_issue(
                issue_id="semantic_fidelity_gap",
                category="model_behavior",
                severity=severity,
                title="Generated Solidity is not consistently semantically close to the reference.",
                evidence=evidence,
                likely_root_causes=[
                    "Training data may under-represent the failing opcode/control-flow slices.",
                    "The model may be learning surface syntax while missing bytecode-grounded behavior.",
                    "Learning rate, LoRA rank, or epoch count may not provide enough capacity for semantic recovery.",
                ],
                suggested_experiments=[
                    _experiment(
                        "segment-targeted-finetune",
                        "Fine-tune on the weakest metadata/opcode segments and compare semantic_similarity_mean by segment.",
                        "data/training",
                        "semantic_similarity_mean improves without baseline regressions",
                    ),
                    _experiment(
                        "generation-config-sweep",
                        "Sweep temperature/top_p/max_new_tokens on the same eval split to separate decoding noise from model capacity.",
                        "model",
                        "pct_above_0.8_similarity increases and edit_distance_mean does not regress",
                    ),
                ],
            )
        )
    elif semantic_mean is not None or pct_high_similarity is not None:
        strengths.append(
            _diagnostic_strength(
                "semantic_similarity_on_target",
                "Semantic similarity is on target for the available evaluation summary.",
                _available_evidence(
                    _target_evidence("semantic_similarity_mean", semantic_mean, ">=", 0.82),
                    _target_evidence(
                        "pct_above_0.8_similarity", pct_high_similarity, ">=", 0.78, percent=True
                    ),
                ),
            )
        )

    edit_mean = metric(
        "edit_distance_mean", "normalized_edit_distance_mean", "normalized_edit_distance"
    )
    pct_low_edit = metric("pct_below_0.4_edit_dist")
    if _above_any((edit_mean, 0.40)) or _below_any((pct_low_edit, 0.82)):
        severity = (
            "high"
            if _above_any((edit_mean, 0.55)) or _below_any((pct_low_edit, 0.60))
            else "medium"
        )
        issues.append(
            _diagnostic_issue(
                issue_id="textual_distance_gap",
                category="model_behavior",
                severity=severity,
                title="Generated Solidity differs substantially from reference source shape.",
                evidence=_available_evidence(
                    _target_evidence("edit_distance_mean", edit_mean, "<=", 0.40),
                    _target_evidence(
                        "pct_below_0.4_edit_dist", pct_low_edit, ">=", 0.82, percent=True
                    ),
                ),
                likely_root_causes=[
                    "The model may be producing generic scaffolds or omitting reference-specific statements.",
                    "Training targets may mix formatting styles or include noisy boilerplate.",
                ],
                suggested_experiments=[
                    _experiment(
                        "target-normalization-audit",
                        "Audit low-edit-distance failures for boilerplate/noisy target patterns before adding more training data.",
                        "data",
                        "edit_distance_mean decreases while replication_f1_micro stays stable or improves",
                    )
                ],
            )
        )

    replication_f1 = metric(
        "replication_f1_micro",
        "replication_metrics.micro.f1",
        "aggregate_statistics.replication_metrics.micro.f1",
    )
    replication_recall = metric(
        "replication_recall_micro",
        "replication_metrics.micro.recall",
        "aggregate_statistics.replication_metrics.micro.recall",
    )
    replication_precision = metric(
        "replication_precision_micro",
        "replication_metrics.micro.precision",
        "aggregate_statistics.replication_metrics.micro.precision",
    )
    category_gaps = _replication_category_gaps(summary)
    replication_issue = _below_any((replication_f1, 0.75), (replication_recall, 0.75)) or bool(
        category_gaps
    )
    if replication_issue:
        severity = (
            "high" if _below_any((replication_f1, 0.50), (replication_recall, 0.55)) else "medium"
        )
        top_categories = ", ".join(
            f"{gap['category']} f1={_format_number(gap.get('f1'))}" for gap in category_gaps[:3]
        )
        evidence = _available_evidence(
            _target_evidence("replication_f1_micro", replication_f1, ">=", 0.75),
            _target_evidence("replication_recall_micro", replication_recall, ">=", 0.75),
            _target_evidence("replication_precision_micro", replication_precision, ">=", 0.85),
            f"weakest categories: {top_categories}" if top_categories else None,
        )
        issues.append(
            _diagnostic_issue(
                issue_id="structured_replication_gap",
                category="model_behavior",
                severity=severity,
                title="Structured Solidity facts are missing or incorrect.",
                evidence=evidence,
                likely_root_causes=[
                    "Dataflow-heavy examples such as state writes, guards, events, or ABI details are under-learned.",
                    "Prompt evidence may not expose enough selector/opcode context for the failing fact categories.",
                    "Training may need harder negative examples that penalize plausible but unsupported facts.",
                ],
                suggested_experiments=[
                    _experiment(
                        "category-balanced-replay",
                        "Oversample rows from the weakest replication categories and rerun the same holdout split.",
                        "data/training",
                        "replication_by_category_micro improves for the targeted categories",
                    ),
                    _experiment(
                        "fact-loss-review",
                        "Manually inspect worst_samples missing/extra facts and propose target-specific data repairs.",
                        "model/data",
                        "replication_f1_micro improves and hallucination_rate does not increase",
                    ),
                ],
                extra={"category_gaps": category_gaps[:5]},
            )
        )
    elif replication_f1 is not None:
        strengths.append(
            _diagnostic_strength(
                "structured_replication_on_target",
                "Structured replication is on target.",
                [_target_evidence("replication_f1_micro", replication_f1, ">=", 0.75)],
            )
        )

    hallucination_rate = _hallucination_rate(summary)
    if hallucination_rate is not None:
        metrics_used["replication_hallucination_rate"] = _json_safe_number(hallucination_rate)
    groundedness = metric(
        "replication_groundedness_score_mean",
        "replication_metrics.groundedness_score_mean",
        "aggregate_statistics.replication_metrics.groundedness_score_mean",
    )
    hallucination_buckets = _top_hallucination_buckets(summary)
    if (hallucination_rate is not None and hallucination_rate > 0.05) or (
        groundedness is not None and groundedness < 0.95
    ):
        severity = (
            "high"
            if (hallucination_rate is not None and hallucination_rate > 0.15)
            or (groundedness is not None and groundedness < 0.85)
            else "medium"
        )
        bucket_text = ", ".join(f"{name}={count}" for name, count in hallucination_buckets[:3])
        issues.append(
            _diagnostic_issue(
                issue_id="grounded_hallucinations",
                category="model_behavior",
                severity=severity,
                title="The model is adding facts not grounded in the reference or bytecode evidence.",
                evidence=_available_evidence(
                    _target_evidence(
                        "replication_hallucination_rate",
                        hallucination_rate,
                        "<=",
                        0.05,
                        percent=True,
                    ),
                    _target_evidence(
                        "replication_groundedness_score_mean", groundedness, ">=", 0.95
                    ),
                    f"top hallucination buckets: {bucket_text}" if bucket_text else None,
                ),
                likely_root_causes=[
                    "Training examples may reward common Solidity boilerplate even when bytecode evidence is absent.",
                    "Decoding may be too permissive, causing over-generation of calls, guards, events, or returns.",
                ],
                suggested_experiments=[
                    _experiment(
                        "hallucination-negative-set",
                        "Create a small negative eval slice from the top hallucination buckets and tune prompts/decoding against it.",
                        "data/model",
                        "replication_hallucination_rate drops without reducing replication_recall_micro",
                    )
                ],
            )
        )
    elif hallucination_rate is not None or groundedness is not None:
        strengths.append(
            _diagnostic_strength(
                "grounded_generation",
                "Generated facts appear well grounded in the available replication metrics.",
                _available_evidence(
                    _target_evidence(
                        "replication_hallucination_rate",
                        hallucination_rate,
                        "<=",
                        0.05,
                        percent=True,
                    ),
                    _target_evidence(
                        "replication_groundedness_score_mean", groundedness, ">=", 0.95
                    ),
                ),
            )
        )

    solidity_valid = metric("solidity_valid_mean")
    if solidity_valid is not None and solidity_valid < 0.95:
        issues.append(
            _diagnostic_issue(
                issue_id="solidity_validity_gap",
                category="syntax_and_deployability",
                severity="high" if solidity_valid < 0.75 else "medium",
                title="Generated outputs often fail Solidity syntax or validation checks.",
                evidence=[
                    _target_evidence(
                        "solidity_valid_mean", solidity_valid, ">=", 0.95, percent=True
                    )
                ],
                likely_root_causes=[
                    "Targets may contain inconsistent Solidity versions or formatting.",
                    "Generation may stop early or exceed the model's learned function boundary patterns.",
                    "Prompt truncation can remove declarations needed for syntactically valid output.",
                ],
                suggested_experiments=[
                    _experiment(
                        "syntax-error-cluster",
                        "Cluster scaffold/compiler errors and patch the largest target/prompt class first.",
                        "data/model",
                        "solidity_valid_mean reaches at least 95%",
                    )
                ],
            )
        )

    bytecode_checked = metric("bytecode_semantic_checked_mean")
    bytecode_score = metric("bytecode_semantic_score_mean")
    bytecode_deployable = metric("bytecode_deployable_mean")
    runtime_checked = metric("bytecode_runtime_checked_mean")
    runtime_match = metric("bytecode_runtime_match_mean")
    if bytecode_checked is not None and bytecode_checked < 0.95:
        issues.append(
            _diagnostic_issue(
                issue_id="bytecode_grounding_coverage_gap",
                category="evaluation_data",
                severity="medium",
                title="Not enough rows have bytecode-grounded semantic checks.",
                evidence=[
                    _target_evidence(
                        "bytecode_semantic_checked_mean", bytecode_checked, ">=", 0.95, percent=True
                    )
                ],
                likely_root_causes=[
                    "Evaluation rows may be missing runtime bytecode, opcode analysis, compiler version, or differential-call evidence.",
                    "Dataset export may not be carrying bytecode metadata into the evaluation artifact.",
                ],
                suggested_experiments=[
                    _experiment(
                        "eval-metadata-backfill",
                        "Backfill bytecode/opcode/runtime metadata for the holdout set before using bytecode metrics as model gates.",
                        "data/evaluation",
                        "bytecode_semantic_checked_mean reaches 95%+",
                    )
                ],
            )
        )
    if bytecode_score is not None and bytecode_score < 0.80:
        issues.append(
            _diagnostic_issue(
                issue_id="bytecode_semantic_gap",
                category="model_behavior",
                severity="high" if bytecode_score < 0.60 else "medium",
                title="Bytecode-grounded behavior does not match reference behavior often enough.",
                evidence=[
                    _target_evidence("bytecode_semantic_score_mean", bytecode_score, ">=", 0.80)
                ],
                likely_root_causes=[
                    "The model may recover text that looks plausible but changes guards, returns, calls, or state effects.",
                    "Training may need examples that emphasize bytecode behavior over lexical similarity.",
                ],
                suggested_experiments=[
                    _experiment(
                        "bytecode-mismatch-slice",
                        "Fine-tune or evaluate separately on rows with bytecode mismatch buckets before broad retraining.",
                        "model/training",
                        "bytecode_semantic_score_mean improves and replication_f1_micro does not regress",
                    )
                ],
            )
        )
    if bytecode_deployable is not None and bytecode_deployable < 0.95:
        issues.append(
            _diagnostic_issue(
                issue_id="deployability_gap",
                category="syntax_and_deployability",
                severity="medium",
                title="Generated Solidity is not consistently deployable in bytecode-aware validation.",
                evidence=[
                    _target_evidence(
                        "bytecode_deployable_mean", bytecode_deployable, ">=", 0.95, percent=True
                    )
                ],
                likely_root_causes=[
                    "Compiler version or constructor/runtime context may be missing.",
                    "Generated source may pass scaffold syntax but fail real compiler/deployment checks.",
                ],
                suggested_experiments=[
                    _experiment(
                        "compiler-matrix-eval",
                        "Run local solc validation for the dominant compiler-version segments and inspect deploy failures.",
                        "evaluation/data",
                        "bytecode_deployable_mean reaches 95%+ on compiler-checked rows",
                    )
                ],
            )
        )
    if (
        runtime_checked is not None
        and runtime_checked > 0
        and runtime_match is not None
        and runtime_match < 0.95
    ):
        issues.append(
            _diagnostic_issue(
                issue_id="runtime_bytecode_mismatch",
                category="model_behavior",
                severity="high" if runtime_match < 0.50 else "medium",
                title="Generated runtime bytecode does not match the reference runtime when checked.",
                evidence=[
                    _target_evidence(
                        "bytecode_runtime_match_mean", runtime_match, ">=", 0.95, percent=True
                    ),
                    _diagnostic_metric_text(
                        "bytecode_runtime_checked_mean", runtime_checked, percent=True
                    ),
                ],
                likely_root_causes=[
                    "The generated function may have the right shape but wrong low-level behavior.",
                    "Compiler settings or metadata may be inconsistent between target and generated source.",
                ],
                suggested_experiments=[
                    _experiment(
                        "runtime-diff-triage",
                        "Compare normalized runtime diffs for the lowest bytecode_semantic_score samples.",
                        "model/evaluation",
                        "bytecode_runtime_match_mean improves on checked rows",
                    )
                ],
            )
        )

    prompt_diagnostics = summary.get("prompt_diagnostics")
    if isinstance(prompt_diagnostics, Mapping):
        truncated_rate = _coerce_float(prompt_diagnostics.get("truncated_rate"))
        if truncated_rate is not None:
            metrics_used["prompt_truncated_rate"] = _json_safe_number(truncated_rate)
            if truncated_rate > 0:
                severity = "high" if truncated_rate >= 0.20 else "medium"
                issues.append(
                    _diagnostic_issue(
                        issue_id="prompt_truncation",
                        category="prompting_and_context",
                        severity=severity,
                        title="Some TAC prompts are truncated before evaluation generation.",
                        evidence=[
                            _diagnostic_metric_text(
                                "prompt_truncated_rate", truncated_rate, percent=True
                            )
                        ],
                        likely_root_causes=[
                            "TAC or metadata is too long for the current max sequence length and prompt budget.",
                            "Important control-flow or storage evidence may be removed before the model sees it.",
                        ],
                        suggested_experiments=[
                            _experiment(
                                "context-budget-sweep",
                                "Increase max sequence length or reduce nonessential metadata and compare truncated_low_quality samples.",
                                "prompt/training",
                                "prompt_truncated_rate decreases and low-quality truncated worst samples disappear",
                            )
                        ],
                    )
                )

    baseline = summary.get("baseline_comparison") or summary.get("regression_comparison")
    if isinstance(baseline, Mapping):
        regressions = _coerce_float(baseline.get("num_regressions"))
        if regressions is not None and regressions > 0:
            regressed_metrics = _regressed_metric_names(baseline)
            issues.append(
                _diagnostic_issue(
                    issue_id="baseline_regressions",
                    category="regression_risk",
                    severity="high" if regressions >= 3 else "medium",
                    title="Current evaluation regressed against the configured baseline.",
                    evidence=[
                        f"num_regressions={int(regressions)}",
                        (
                            f"regressed metrics: {', '.join(regressed_metrics[:5])}"
                            if regressed_metrics
                            else "regressed metrics were not listed"
                        ),
                    ],
                    likely_root_causes=[
                        "Recent data, training, or decoding changes may have improved one metric while hurting another.",
                        "Quality-gate thresholds may not cover the regressed metric directly.",
                    ],
                    suggested_experiments=[
                        _experiment(
                            "ablation-against-baseline",
                            "Rerun the current and baseline configs on the same fixed eval sample and ablate one change at a time.",
                            "training/evaluation",
                            "num_regressions returns to 0 or accepted trade-offs are documented",
                        )
                    ],
                )
            )

    quality_counts = _quality_issue_counts(summary)
    if quality_counts:
        top_counts = ", ".join(f"{key}={value}" for key, value in list(quality_counts.items())[:5])
        issues.append(
            _diagnostic_issue(
                issue_id="quality_taxonomy_findings",
                category="evaluation_data",
                severity="medium",
                title="The detailed evaluation taxonomy found recurring quality issue categories.",
                evidence=[f"top categories: {top_counts}"],
                likely_root_causes=[
                    "Failures are clustered rather than random; use the largest category to choose the next data/model fix.",
                ],
                suggested_experiments=[
                    _experiment(
                        "taxonomy-first-triage",
                        "Address the largest quality_issue_summary category and rerun the same evaluation split.",
                        "evaluation/data/model",
                        "largest quality issue category count decreases",
                    )
                ],
                extra={"category_counts": dict(quality_counts)},
            )
        )

    caveats.extend(_metadata_coverage_caveats(summary))
    _add_missing_metric_caveats(summary, caveats)

    issues = _dedupe_issues(issues)
    issues.sort(
        key=lambda item: (
            -_SEVERITY_RANK.get(str(item.get("severity")), 0),
            str(item.get("priority", "P9")),
            str(item.get("id", "")),
        )
    )
    strengths = strengths[:6]
    next_experiments = _collect_next_experiments(issues)
    issue_counts = _issue_counts_by_severity(issues)
    overall_status = _overall_diagnostic_status(issues, caveats)

    return {
        "schema_version": 1,
        "overall_status": overall_status,
        "issue_counts_by_severity": issue_counts,
        "metrics_used": metrics_used,
        "strengths": strengths,
        "issues": issues,
        "next_experiments": next_experiments,
        "caveats": list(dict.fromkeys(caveats)),
    }


def format_latest_results_report(
    *,
    summary: Mapping[str, Any],
    model_path: str,
    test_dataset_path: str,
    results_json_path: str,
    started_at: Optional[float] = None,
    finished_at: Optional[float] = None,
    argv: Optional[Sequence[str]] = None,
    eval_limit: Optional[int] = None,
    world_size: int = 1,
) -> str:
    """Build a concise, check-in friendly quality report."""
    summary = _summary_with_derived_metrics(summary, results_json_path)
    model = _collect_model_info(model_path)
    dataset = _collect_dataset_info(test_dataset_path)
    git = _collect_git_info()
    duration = _duration_seconds(started_at, finished_at)

    lines = [
        "Smart Contract Decompiler - Latest Evaluation Results",
        "=" * 58,
        "",
        "Run",
        "---",
        f"Generated at: {_utc_timestamp(finished_at)}",
        f"Command: {_format_command(argv)}",
        f"Git commit: {git.get('commit', 'unknown')}",
        f"Git dirty: {git.get('dirty', 'unknown')}",
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        f"Evaluation duration: {_format_duration(duration)}",
        f"Distributed world size: {world_size}",
        f"Eval limit: {eval_limit if eval_limit is not None else 'none'}",
        "",
        "Model",
        "-----",
        f"Model path: {model_path}",
        f"Base model: {model.get('base_model', 'unknown')}",
        f"Model artifact size: {_format_bytes(model.get('artifact_bytes'))}",
        f"Adapter size: {_format_bytes(model.get('adapter_bytes'))}",
        f"Tokenizer size: {_format_bytes(model.get('tokenizer_bytes'))}",
        f"Max sequence length: {model.get('max_sequence_length', 'unknown')}",
        f"LoRA rank: {model.get('lora_rank', 'unknown')}",
        f"LoRA alpha: {model.get('lora_alpha', 'unknown')}",
        f"LoRA dropout: {model.get('lora_dropout', 'unknown')}",
        f"Target modules: {_format_list(model.get('target_modules'))}",
        f"Quantization: {_format_quantization(model)}",
        f"Compiler metadata prompts: {model.get('include_compiler_metadata', 'unknown')}",
    ]

    training_args = model.get("training_args")
    if training_args:
        lines.extend(["", "Training Parameters", "-------------------"])
        for key, value in training_args.items():
            lines.append(f"{key}: {value}")

    lines.extend(
        [
            "",
            "Evaluation Data",
            "---------------",
            f"Test dataset: {test_dataset_path}",
            f"Dataset rows: {dataset.get('rows', 'unknown')}",
            f"Dataset size: {_format_bytes(dataset.get('bytes'))}",
            f"Detailed JSON: {results_json_path}",
        ]
    )

    diagnostics = summary.get("evaluation_diagnostics")
    if not isinstance(diagnostics, Mapping):
        diagnostics = build_evaluation_diagnostics(summary)
    _append_evaluation_diagnostics_report(lines, diagnostics)

    lines.extend(
        [
            "",
            "Quality Summary",
            "---------------",
            f"Examples evaluated: {_metric(summary, 'num_evaluated', integer=True)}",
            f"Examples attempted: {_metric(summary, 'num_attempted', integer=True)}",
            f"Examples succeeded: {_metric(summary, 'num_succeeded', integer=True)}",
            f"Examples failed: {_metric(summary, 'num_failed', integer=True)}",
            f"Failure rate: {_percent_metric(summary, 'failure_rate')}",
            f"Semantic similarity mean: {_metric(summary, 'semantic_similarity_mean')}",
            f"Semantic similarity std: {_metric(summary, 'semantic_similarity_std')}",
            f"Edit distance mean: {_metric(summary, 'edit_distance_mean')}",
            f"Edit distance std: {_metric(summary, 'edit_distance_std')}",
            f"Functions > 0.8 semantic similarity: {_percent_metric(summary, 'pct_above_0.8_similarity')}",
            f"Functions < 0.4 edit distance: {_percent_metric(summary, 'pct_below_0.4_edit_dist')}",
            f"Replication precision mean: {_metric(summary, 'replication_precision_mean')}",
            f"Replication recall mean: {_metric(summary, 'replication_recall_mean')}",
            f"Replication F1 mean: {_metric(summary, 'replication_f1_mean')}",
            f"Replication precision micro: {_metric(summary, 'replication_precision_micro')}",
            f"Replication recall micro: {_metric(summary, 'replication_recall_micro')}",
            f"Replication F1 micro: {_metric(summary, 'replication_f1_micro')}",
            f"Functions > 0.8 replication F1: {_percent_metric(summary, 'pct_above_0.8_replication_f1')}",
        ]
    )

    if any(
        key in summary
        for key in (
            "solidity_valid_mean",
            "solidity_compiler_checked_mean",
            "solidity_ast_valid_mean",
        )
    ):
        lines.extend(
            [
                f"Solidity valid outputs: {_percent_metric(summary, 'solidity_valid_mean')}",
                f"Solidity compiler-checked outputs: {_percent_metric(summary, 'solidity_compiler_checked_mean')}",
                f"Solidity AST-valid outputs: {_percent_metric(summary, 'solidity_ast_valid_mean')}",
            ]
        )

    if any(
        key in summary
        for key in (
            "bytecode_semantic_score_mean",
            "bytecode_semantic_checked_mean",
            "bytecode_deployable_mean",
            "bytecode_runtime_checked_mean",
            "bytecode_runtime_match_mean",
        )
    ):
        lines.extend(
            [
                f"Bytecode semantic score mean: {_metric(summary, 'bytecode_semantic_score_mean')}",
                f"Bytecode semantic checked outputs: {_percent_metric(summary, 'bytecode_semantic_checked_mean')}",
                f"Bytecode deployable outputs: {_percent_metric(summary, 'bytecode_deployable_mean')}",
                f"Runtime bytecode checked outputs: {_percent_metric(summary, 'bytecode_runtime_checked_mean')}",
                f"Runtime bytecode matches: {_percent_metric(summary, 'bytecode_runtime_match_mean')}",
            ]
        )

    lines.extend(
        [
            "",
            "Target Checks",
            "-------------",
            _target_line(
                "semantic_similarity_mean", summary.get("semantic_similarity_mean"), ">=", 0.82
            ),
            _target_line(
                "pct_above_0.8_similarity", summary.get("pct_above_0.8_similarity"), ">=", 0.78
            ),
            _target_line(
                "pct_below_0.4_edit_dist", summary.get("pct_below_0.4_edit_dist"), ">=", 0.82
            ),
        ]
    )

    category_scores = summary.get("replication_by_category_micro")
    if isinstance(category_scores, Mapping) and category_scores:
        lines.extend(["", "Replication by Category (micro)", "-------------------------------"])
        lines.append("category | precision | recall | f1 | TP | FP | FN")
        lines.append("--- | ---: | ---: | ---: | ---: | ---: | ---:")
        for category, score in sorted(category_scores.items()):
            if not isinstance(score, Mapping):
                continue
            lines.append(
                " | ".join(
                    [
                        str(category),
                        _format_number(score.get("precision")),
                        _format_number(score.get("recall")),
                        _format_number(score.get("f1")),
                        str(score.get("true_positives", 0)),
                        str(score.get("false_positives", 0)),
                        str(score.get("false_negatives", 0)),
                    ]
                )
            )

    _append_hallucination_bucket_report(lines, summary)
    _append_quality_issue_summary(lines, summary.get("quality_issue_summary"))
    _append_prompt_diagnostics_report(lines, summary.get("prompt_diagnostics"))

    confidence_intervals = _summary_confidence_intervals(summary)
    if confidence_intervals:
        lines.extend(["", "Confidence Intervals", "--------------------"])
        lines.append("metric | mean | 95% CI | n")
        lines.append("--- | ---: | ---: | ---:")
        for metric, interval in sorted(confidence_intervals.items()):
            lines.append(
                " | ".join(
                    [
                        metric,
                        _format_number(interval.get("mean")),
                        _format_ci(interval),
                        str(interval.get("n", "n/a")),
                    ]
                )
            )

    metadata_segments = summary.get("metadata_segments")
    if isinstance(metadata_segments, Mapping):
        _append_metadata_segment_report(lines, metadata_segments)

    benchmark_suites = summary.get("benchmark_suites")
    if isinstance(benchmark_suites, Mapping):
        _append_benchmark_suite_report(lines, benchmark_suites)

    baseline_comparison = summary.get("baseline_comparison") or summary.get("regression_comparison")
    if isinstance(baseline_comparison, Mapping):
        _append_baseline_comparison(lines, baseline_comparison)

    _append_worst_samples_report(lines, summary.get("worst_samples"), results_json_path)

    if summary.get("error"):
        lines.extend(["", "Evaluation Error", "----------------", str(summary["error"])])

    lines.append("")
    return "\n".join(lines)


def _summary_with_derived_metrics(
    summary: Mapping[str, Any],
    results_json_path: str,
) -> Dict[str, Any]:
    enriched = dict(summary)
    details = _load_results_details(results_json_path)
    if not details:
        return enriched

    for metric in (
        "solidity_valid",
        "solidity_compiler_checked",
        "solidity_ast_valid",
        "bytecode_semantic_score",
        "bytecode_semantic_checked",
        "bytecode_deployable",
        "bytecode_runtime_checked",
        "bytecode_runtime_match",
    ):
        mean_key = f"{metric}_mean"
        if mean_key not in enriched:
            values = _detail_metric_values(details, metric)
            if values:
                enriched[mean_key] = sum(values) / len(values)

    hallucination_summary = _detail_hallucination_summary(details)
    if hallucination_summary:
        for source_key, target_key in {
            "hallucination_buckets": "replication_hallucination_buckets",
            "hallucination_rate_by_bucket": "replication_hallucination_rate_by_bucket",
            "hallucination_total": "replication_hallucination_total",
            "candidate_fact_total": "replication_candidate_fact_total",
            "groundedness_score_mean": "replication_groundedness_score_mean",
        }.items():
            if target_key not in enriched and source_key in hallucination_summary:
                enriched[target_key] = hallucination_summary[source_key]

    if "metadata_segments" not in enriched:
        try:
            from .training_pipeline import compute_metadata_segment_metrics

            enriched["metadata_segments"] = compute_metadata_segment_metrics(details)
        except Exception:
            pass

    if "benchmark_suites" not in enriched:
        try:
            from .training_pipeline import compute_benchmark_suite_metrics

            enriched["benchmark_suites"] = compute_benchmark_suite_metrics(details)
        except Exception:
            pass

    if "prompt_diagnostics" not in enriched:
        try:
            from .training_pipeline import aggregate_prompt_diagnostics

            prompt_diagnostics = aggregate_prompt_diagnostics(details)
            if prompt_diagnostics:
                enriched["prompt_diagnostics"] = prompt_diagnostics
        except Exception:
            pass

    return enriched


def _load_results_details(results_json_path: str) -> Sequence[Mapping[str, Any]]:
    try:
        path = Path(results_json_path)
        if not path.exists():
            return []
        payload = _load_json(path)
    except Exception:
        return []
    details = payload.get("details") or payload.get("detailed_results")
    if not isinstance(details, Sequence) or isinstance(details, (str, bytes)):
        return []
    return [item for item in details if isinstance(item, Mapping)]


def _detail_metric_values(details: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for detail in details:
        metrics = detail.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        value = metrics.get(metric)
        if isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, (int, float)) and not math.isnan(float(value)):
            values.append(float(value))
    return values


def _detail_hallucination_summary(details: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[str, int] = {}
    candidate_fact_total = 0
    groundedness_scores: list[float] = []
    for detail in details:
        replication = _sample_replication_payload(detail)
        if replication is None:
            continue
        candidate_fact_count = replication.get("candidate_fact_count")
        if isinstance(candidate_fact_count, (int, float)):
            candidate_fact_total += int(candidate_fact_count)
        groundedness = replication.get("groundedness_score")
        if isinstance(groundedness, (int, float)) and not math.isnan(float(groundedness)):
            groundedness_scores.append(float(groundedness))
        for bucket, values in replication.get("hallucination_buckets", {}).items():
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                buckets[str(bucket)] = buckets.get(str(bucket), 0) + len(values)
            elif isinstance(values, (int, float)):
                buckets[str(bucket)] = buckets.get(str(bucket), 0) + int(values)

    if not buckets and not groundedness_scores:
        return {}

    total = sum(buckets.values())
    summary: Dict[str, Any] = {
        "hallucination_buckets": dict(sorted(buckets.items())),
        "hallucination_total": total,
        "candidate_fact_total": candidate_fact_total,
        "hallucination_rate_by_bucket": {
            bucket: count / candidate_fact_total if candidate_fact_total else 0.0
            for bucket, count in sorted(buckets.items())
        },
    }
    if groundedness_scores:
        summary["groundedness_score_mean"] = sum(groundedness_scores) / len(groundedness_scores)
    return summary


def _collect_model_info(model_path: str) -> Dict[str, Any]:
    path = Path(model_path)
    info: Dict[str, Any] = {
        "artifact_bytes": _directory_size(path),
        "adapter_bytes": _file_size(path / "adapter_model.safetensors"),
        "tokenizer_bytes": _file_size(path / "tokenizer.json"),
    }

    config = _load_json(path / "model_config.json")
    adapter_config = _load_json(path / "adapter_config.json")
    info.update(config)
    if adapter_config:
        info["base_model"] = adapter_config.get(
            "base_model_name_or_path",
            config.get("model_name"),
        )
        info.setdefault("lora_rank", adapter_config.get("r"))
        info.setdefault("lora_alpha", adapter_config.get("lora_alpha"))
        info.setdefault("lora_dropout", adapter_config.get("lora_dropout"))
        info.setdefault("target_modules", adapter_config.get("target_modules"))
    else:
        info["base_model"] = config.get("model_name")

    training_args = _load_training_args(path / "training_args.bin")
    if training_args:
        info["training_args"] = training_args
    return info


def _load_training_args(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import torch

        args = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"load_error": exc.__class__.__name__}

    wanted = [
        "num_train_epochs",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "lr_scheduler_type",
        "optim",
        "bf16",
        "fp16",
        "seed",
    ]
    values: Dict[str, Any] = {}
    for key in wanted:
        if hasattr(args, key):
            value = getattr(args, key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                values[key] = value
            else:
                values[key] = str(value)
    return values


def _collect_dataset_info(dataset_path: str) -> Dict[str, Any]:
    path = Path(dataset_path)
    return {"rows": _count_nonempty_lines(path), "bytes": _file_size(path)}


def _collect_git_info() -> Dict[str, str]:
    commit = _run_git(["rev-parse", "--short", "HEAD"]) or "unknown"
    status = _run_git(["status", "--short"])
    return {"commit": commit, "dirty": "yes" if status else "no"}


def _run_git(args: Sequence[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ""
    return proc.stdout.strip()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _directory_size(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _file_size(path: Path) -> Optional[int]:
    return path.stat().st_size if path.exists() else None


def _count_nonempty_lines(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _utc_timestamp(timestamp: Optional[float]) -> str:
    dt = (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        if timestamp is not None
        else datetime.now(timezone.utc)
    )
    return dt.isoformat(timespec="seconds")


def _duration_seconds(started_at: Optional[float], finished_at: Optional[float]) -> Optional[float]:
    if started_at is None or finished_at is None:
        return None
    return max(0.0, finished_at - started_at)


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining:.1f}s"


def _format_command(argv: Optional[Sequence[str]]) -> str:
    if not argv:
        return "unknown"
    return " ".join(shlex.quote(str(arg)) for arg in argv)


def _format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def _format_quantization(model: Mapping[str, Any]) -> str:
    if model.get("use_quantization"):
        return "4-bit" if model.get("load_in_4bit") else "enabled"
    return "disabled"


def _format_list(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    return str(value) if value is not None else "unknown"


def _metric(summary: Mapping[str, Any], key: str, integer: bool = False) -> str:
    value = summary.get(key)
    if integer and isinstance(value, (int, float)):
        return str(int(value))
    return _format_number(value)


def _percent_metric(summary: Mapping[str, Any], key: str) -> str:
    value = summary.get(key)
    if not isinstance(value, (int, float)) or math.isnan(float(value)):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _format_number(value: Any) -> str:
    if not isinstance(value, (int, float)) or math.isnan(float(value)):
        return "n/a"
    return f"{float(value):.4f}"


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return None if math.isnan(numeric) else numeric


def _json_safe_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def _first_numeric_summary_value(summary: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = _numeric_summary_value(summary, key)
        if value is not None:
            return value
    return None


def _numeric_summary_value(summary: Mapping[str, Any], key: str) -> Optional[float]:
    if "." in key:
        value = _mapping_path_value(summary, key.split("."))
        return _coerce_float(value)

    direct = summary.get(key)
    value = _mean_or_numeric(direct)
    if value is not None:
        return value

    if key.endswith("_mean"):
        base_key = key[: -len("_mean")]
        value = _mean_or_numeric(summary.get(base_key))
        if value is not None:
            return value
        aggregate = summary.get("aggregate_statistics")
        if isinstance(aggregate, Mapping):
            value = _mean_or_numeric(aggregate.get(base_key))
            if value is not None:
                return value

    aggregate = summary.get("aggregate_statistics")
    if isinstance(aggregate, Mapping):
        value = _mean_or_numeric(aggregate.get(key))
        if value is not None:
            return value
    return None


def _mean_or_numeric(value: Any) -> Optional[float]:
    numeric = _coerce_float(value)
    if numeric is not None:
        return numeric
    if isinstance(value, Mapping):
        return _coerce_float(value.get("mean"))
    return None


def _mapping_path_value(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _below_any(*checks: tuple[Optional[float], float]) -> bool:
    return any(value is not None and value < threshold for value, threshold in checks)


def _above_any(*checks: tuple[Optional[float], float]) -> bool:
    return any(value is not None and value > threshold for value, threshold in checks)


def _diagnostic_issue(
    *,
    issue_id: str,
    category: str,
    severity: str,
    title: str,
    evidence: Sequence[str],
    likely_root_causes: Sequence[str],
    suggested_experiments: Sequence[Mapping[str, Any]],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": issue_id,
        "category": category,
        "severity": severity,
        "priority": _PRIORITY_BY_SEVERITY.get(severity, "P3"),
        "title": title,
        "evidence": [item for item in evidence if item],
        "likely_root_causes": list(likely_root_causes),
        "suggested_experiments": [dict(experiment) for experiment in suggested_experiments],
    }
    if extra:
        payload.update(extra)
    return payload


def _diagnostic_strength(
    strength_id: str,
    title: str,
    evidence: Sequence[str],
) -> Dict[str, Any]:
    return {
        "id": strength_id,
        "title": title,
        "evidence": [item for item in evidence if item],
    }


def _experiment(
    experiment_id: str,
    action: str,
    area: str,
    success_metric: str,
) -> Dict[str, Any]:
    return {
        "id": experiment_id,
        "area": area,
        "action": action,
        "success_metric": success_metric,
    }


def _target_evidence(
    name: str,
    value: Optional[float],
    operator: str,
    target: float,
    *,
    percent: bool = False,
) -> Optional[str]:
    if value is None:
        return None
    actual = _diagnostic_metric_text(name, value, percent=percent)
    target_text = f"{target * 100:.2f}%" if percent else _format_number(target)
    return f"{actual} (target {operator} {target_text})"


def _diagnostic_metric_text(name: str, value: float, *, percent: bool = False) -> str:
    if percent:
        return f"{name}={value * 100:.2f}%"
    return f"{name}={_format_number(value)}"


def _available_evidence(*items: Optional[str]) -> list[str]:
    return [item for item in items if item]


def _replication_category_gaps(summary: Mapping[str, Any]) -> list[Dict[str, Any]]:
    category_scores = summary.get("replication_by_category_micro")
    if not isinstance(category_scores, Mapping):
        category_scores = _mapping_path_value(summary, ("replication_metrics", "by_category_micro"))
    if not isinstance(category_scores, Mapping):
        category_scores = _mapping_path_value(
            summary,
            ("aggregate_statistics", "replication_metrics", "by_category_micro"),
        )
    if not isinstance(category_scores, Mapping):
        return []

    gaps: list[Dict[str, Any]] = []
    for category, score in category_scores.items():
        if not isinstance(score, Mapping):
            continue
        precision = _coerce_float(score.get("precision"))
        recall = _coerce_float(score.get("recall"))
        f1 = _coerce_float(score.get("f1"))
        false_negatives = int(score.get("false_negatives", 0) or 0)
        false_positives = int(score.get("false_positives", 0) or 0)
        if not (
            (f1 is not None and f1 < 0.75)
            or (recall is not None and recall < 0.75)
            or false_negatives
            or false_positives
        ):
            continue
        gaps.append(
            {
                "category": str(category),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_negatives": false_negatives,
                "false_positives": false_positives,
            }
        )

    gaps.sort(
        key=lambda item: (
            item["f1"] if isinstance(item.get("f1"), (int, float)) else 1.0,
            -(int(item.get("false_negatives", 0)) + int(item.get("false_positives", 0))),
            str(item.get("category", "")),
        )
    )
    return gaps


def _hallucination_rate(summary: Mapping[str, Any]) -> Optional[float]:
    direct = _first_numeric_summary_value(
        summary,
        "replication_hallucination_rate",
        "replication_metrics.hallucination_rate",
        "aggregate_statistics.replication_metrics.hallucination_rate",
    )
    if direct is not None:
        return direct
    total = _first_numeric_summary_value(
        summary,
        "replication_hallucination_total",
        "replication_metrics.hallucination_total",
        "aggregate_statistics.replication_metrics.hallucination_total",
    )
    candidate_total = _first_numeric_summary_value(
        summary,
        "replication_candidate_fact_total",
        "replication_metrics.candidate_fact_total",
        "aggregate_statistics.replication_metrics.candidate_fact_total",
    )
    if total is None or candidate_total in (None, 0):
        return None
    return total / candidate_total


def _top_hallucination_buckets(summary: Mapping[str, Any]) -> list[tuple[str, int]]:
    buckets = summary.get("replication_hallucination_buckets")
    if not isinstance(buckets, Mapping):
        buckets = _mapping_path_value(summary, ("replication_metrics", "hallucination_buckets"))
    if not isinstance(buckets, Mapping):
        buckets = _mapping_path_value(
            summary,
            ("aggregate_statistics", "replication_metrics", "hallucination_buckets"),
        )
    if not isinstance(buckets, Mapping):
        return []
    counts = [
        (str(bucket), _hallucination_bucket_count(value)) for bucket, value in buckets.items()
    ]
    counts.sort(key=lambda item: (-item[1], item[0]))
    return [(bucket, count) for bucket, count in counts if count > 0]


def _quality_issue_counts(summary: Mapping[str, Any]) -> Dict[str, int]:
    issue_summary = summary.get("quality_issue_summary")
    if not isinstance(issue_summary, Mapping):
        return {}
    counts = issue_summary.get("category_counts")
    if not isinstance(counts, Mapping):
        return {}
    numeric_counts = {
        str(category): int(count)
        for category, count in counts.items()
        if isinstance(count, (int, float)) and count > 0
    }
    return dict(sorted(numeric_counts.items(), key=lambda item: (-item[1], item[0])))


def _regressed_metric_names(comparison: Mapping[str, Any]) -> list[str]:
    comparisons = comparison.get("comparisons")
    if not isinstance(comparisons, Mapping):
        return []
    rows: list[tuple[str, float]] = []
    for metric, item in comparisons.items():
        if not isinstance(item, Mapping) or item.get("status") != "regressed":
            continue
        relative_delta = _coerce_float(item.get("relative_delta")) or 0.0
        rows.append((str(metric), abs(relative_delta)))
    rows.sort(key=lambda item: (-item[1], item[0]))
    return [metric for metric, _ in rows]


def _metadata_coverage_caveats(summary: Mapping[str, Any]) -> list[str]:
    metadata_segments = summary.get("metadata_segments")
    if not isinstance(metadata_segments, Mapping):
        return []
    coverage = metadata_segments.get("coverage")
    if not isinstance(coverage, Mapping):
        return []

    caveats: list[str] = []
    for field_name in ("compiler_version", "optimizer_enabled", "opcode_group", "control_flow"):
        field = coverage.get(field_name)
        if not isinstance(field, Mapping):
            continue
        known = _coerce_float(field.get("known")) or 0.0
        unknown = _coerce_float(field.get("unknown")) or 0.0
        total = known + unknown
        if total and unknown / total > 0.20:
            caveats.append(
                f"Metadata coverage for {field_name} is incomplete ({unknown / total:.0%} unknown); segment conclusions may be biased."
            )
    return caveats


def _add_missing_metric_caveats(summary: Mapping[str, Any], caveats: list[str]) -> None:
    checks = [
        (
            (
                "replication_f1_micro",
                "replication_metrics.micro.f1",
                "aggregate_statistics.replication_metrics.micro.f1",
            ),
            "Replication metrics are missing; run structured fact evaluation before planning data/model changes.",
        ),
        (
            ("solidity_valid_mean",),
            "Solidity validity metrics are missing; syntax/deployability conclusions are incomplete.",
        ),
        (
            ("bytecode_semantic_checked_mean",),
            "Bytecode-grounded metrics are missing; semantic conclusions rely mainly on text similarity.",
        ),
    ]
    for keys, caveat in checks:
        if _first_numeric_summary_value(summary, *keys) is None:
            caveats.append(caveat)


def _dedupe_issues(issues: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    for issue in issues:
        issue_id = str(issue.get("id", "unknown"))
        existing = deduped.get(issue_id)
        if existing is None or _SEVERITY_RANK.get(
            str(issue.get("severity")), 0
        ) > _SEVERITY_RANK.get(str(existing.get("severity")), 0):
            deduped[issue_id] = dict(issue)
    return list(deduped.values())


def _collect_next_experiments(
    issues: Sequence[Mapping[str, Any]], *, limit: int = 8
) -> list[Dict[str, Any]]:
    experiments: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for issue in issues:
        issue_experiments = issue.get("suggested_experiments")
        if not isinstance(issue_experiments, Sequence) or isinstance(
            issue_experiments, (str, bytes)
        ):
            continue
        for experiment in issue_experiments:
            if not isinstance(experiment, Mapping):
                continue
            experiment_id = str(experiment.get("id", ""))
            if not experiment_id or experiment_id in seen:
                continue
            payload = dict(experiment)
            payload.setdefault("priority", issue.get("priority", "P3"))
            payload.setdefault("triggered_by", issue.get("id"))
            experiments.append(payload)
            seen.add(experiment_id)
            if len(experiments) >= limit:
                return experiments
    return experiments


def _issue_counts_by_severity(issues: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = {severity: 0 for severity in ("critical", "high", "medium", "low")}
    for issue in issues:
        severity = str(issue.get("severity", "low"))
        counts[severity] = counts.get(severity, 0) + 1
    return {severity: count for severity, count in counts.items() if count}


def _overall_diagnostic_status(issues: Sequence[Mapping[str, Any]], caveats: Sequence[str]) -> str:
    if any(issue.get("severity") == "critical" for issue in issues):
        return "blocked"
    if any(issue.get("severity") == "high" for issue in issues):
        return "needs_attention"
    if issues:
        return "watch"
    if caveats:
        return "insufficient_evidence"
    return "healthy"


def _append_evaluation_diagnostics_report(
    lines: list[str],
    diagnostics: Mapping[str, Any],
) -> None:
    if not diagnostics:
        return
    lines.extend(["", "Executive Evaluation Readout", "----------------------------"])
    status = diagnostics.get("overall_status", "unknown")
    lines.append(f"Overall status: {status}")

    strengths = diagnostics.get("strengths")
    lines.extend(["", "What is going well:"])
    if isinstance(strengths, Sequence) and not isinstance(strengths, (str, bytes)) and strengths:
        for strength in strengths[:4]:
            if not isinstance(strength, Mapping):
                continue
            evidence = strength.get("evidence")
            evidence_text = (
                f" ({'; '.join(str(item) for item in evidence[:2])})"
                if isinstance(evidence, Sequence)
                and not isinstance(evidence, (str, bytes))
                and evidence
                else ""
            )
            lines.append(f"- {strength.get('title', 'Strength detected')}{evidence_text}")
    else:
        lines.append("- No clear strengths were detected from the available metrics.")

    issues = diagnostics.get("issues")
    lines.extend(["", "What needs attention:"])
    if isinstance(issues, Sequence) and not isinstance(issues, (str, bytes)) and issues:
        for issue in issues[:5]:
            if not isinstance(issue, Mapping):
                continue
            evidence = issue.get("evidence")
            evidence_text = (
                f" Evidence: {'; '.join(str(item) for item in evidence[:2])}"
                if isinstance(evidence, Sequence)
                and not isinstance(evidence, (str, bytes))
                and evidence
                else ""
            )
            lines.append(
                f"- [{issue.get('priority', 'P3')}/{issue.get('severity', 'low')}] "
                f"{issue.get('title', issue.get('id', 'issue'))}{evidence_text}"
            )
            root_causes = issue.get("likely_root_causes")
            if (
                isinstance(root_causes, Sequence)
                and not isinstance(root_causes, (str, bytes))
                and root_causes
            ):
                lines.append(f"  Likely cause: {root_causes[0]}")
    else:
        lines.append("- No metric-driven issues were detected.")

    experiments = diagnostics.get("next_experiments")
    if (
        isinstance(experiments, Sequence)
        and not isinstance(experiments, (str, bytes))
        and experiments
    ):
        lines.extend(["", "Suggested next experiments:"])
        for experiment in experiments[:5]:
            if not isinstance(experiment, Mapping):
                continue
            lines.append(
                f"- [{experiment.get('priority', 'P3')}] {experiment.get('action')} "
                f"Success metric: {experiment.get('success_metric')}"
            )

    caveats = diagnostics.get("caveats")
    if isinstance(caveats, Sequence) and not isinstance(caveats, (str, bytes)) and caveats:
        lines.extend(["", "Caveats:"])
        for caveat in caveats[:5]:
            lines.append(f"- {caveat}")

    lines.extend(["", "Machine-readable diagnostics:", "```json"])
    lines.append(json.dumps(diagnostics, indent=2, sort_keys=True))
    lines.append("```")


def _target_line(metric: str, value: Any, operator: str, target: float) -> str:
    passed = isinstance(value, (int, float)) and value >= target
    status = "PASS" if passed else "FAIL"
    return f"{metric}: {status} ({_format_number(value)} {operator} {_format_number(target)})"


def _summary_confidence_intervals(summary: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    intervals: Dict[str, Mapping[str, Any]] = {}
    direct = summary.get("confidence_intervals")
    if isinstance(direct, Mapping):
        for metric, interval in direct.items():
            if isinstance(interval, Mapping):
                intervals[str(metric)] = interval

    for metric, value in summary.items():
        if not isinstance(value, Mapping):
            continue
        interval = value.get("confidence_interval_95")
        if isinstance(interval, Mapping):
            intervals[f"{metric}_mean"] = interval
    return intervals


def _format_ci(interval: Mapping[str, Any]) -> str:
    low = interval.get("low")
    high = interval.get("high")
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return "n/a"
    return f"[{_format_number(low)}, {_format_number(high)}]"


def _append_hallucination_bucket_report(lines: list[str], summary: Mapping[str, Any]) -> None:
    buckets = summary.get("replication_hallucination_buckets")
    rates = summary.get("replication_hallucination_rate_by_bucket")
    total = summary.get("replication_hallucination_total")
    groundedness = summary.get("replication_groundedness_score_mean")
    replication_metrics = summary.get("replication_metrics")
    if not isinstance(buckets, Mapping):
        if isinstance(replication_metrics, Mapping):
            buckets = replication_metrics.get("hallucination_buckets")
            rates = replication_metrics.get("hallucination_rate_by_bucket")
            total = replication_metrics.get("hallucination_total")
            groundedness = replication_metrics.get("groundedness_score_mean")
    elif isinstance(replication_metrics, Mapping):
        if not isinstance(rates, Mapping):
            rates = replication_metrics.get("hallucination_rate_by_bucket")
        if not isinstance(total, (int, float)):
            total = replication_metrics.get("hallucination_total")
        if not isinstance(groundedness, (int, float)):
            groundedness = replication_metrics.get("groundedness_score_mean")
    if not isinstance(buckets, Mapping) or not buckets:
        return

    bucket_counts = {
        str(bucket): _hallucination_bucket_count(count) for bucket, count in buckets.items()
    }
    computed_total = sum(bucket_counts.values())
    if not isinstance(total, (int, float)):
        total = computed_total
    lines.extend(["", "Grounded Hallucination Buckets", "------------------------------"])
    lines.append(f"Total hallucinated facts: {int(total)}")
    if isinstance(groundedness, (int, float)):
        lines.append(f"Groundedness score mean: {_format_number(groundedness)}")
    lines.append("bucket | count | rate")
    lines.append("--- | ---: | ---:")
    rate_map = rates if isinstance(rates, Mapping) else {}
    for bucket, numeric_count in sorted(bucket_counts.items()):
        rate = rate_map.get(bucket)
        if not isinstance(rate, (int, float)):
            rate = numeric_count / computed_total if computed_total else 0.0
        lines.append(f"{bucket} | {numeric_count} | {rate * 100:.2f}%")


def _hallucination_bucket_count(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return 0


def _append_quality_issue_summary(lines: list[str], summary: Any) -> None:
    if not isinstance(summary, Mapping) or not summary:
        return
    counts = summary.get("category_counts")
    if not isinstance(counts, Mapping) or not counts:
        return
    lines.extend(["", "Quality Issue Taxonomy", "----------------------"])
    lines.append("category | count")
    lines.append("--- | ---:")
    for category, count in sorted(counts.items()):
        lines.append(f"{category} | {count}")

    hints = summary.get("remediation_hints")
    if isinstance(hints, Mapping) and hints:
        lines.extend(["", "Remediation hints:"])
        for category, hint in sorted(hints.items()):
            lines.append(f"- {category}: {hint}")

    examples = summary.get("example_failures")
    if isinstance(examples, Mapping) and examples:
        lines.extend(["", "Example failures:"])
        for category, rows in sorted(examples.items()):
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
                continue
            formatted = []
            for row in list(rows)[:3]:
                if isinstance(row, Mapping):
                    formatted.append(
                        f"idx={row.get('dataset_index')} input={row.get('input_hash', 'n/a')}"
                    )
            if formatted:
                lines.append(f"- {category}: " + "; ".join(formatted))


def _append_prompt_diagnostics_report(lines: list[str], diagnostics: Any) -> None:
    if not isinstance(diagnostics, Mapping) or not diagnostics:
        return
    total = diagnostics.get("num_details")
    truncated = diagnostics.get("truncated_count")
    rate = diagnostics.get("truncated_rate")
    lines.extend(["", "Prompt/Truncation Diagnostics", "-----------------------------"])
    if isinstance(total, int) and isinstance(truncated, int):
        rate_text = f"{float(rate) * 100:.2f}%" if isinstance(rate, (int, float)) else "n/a"
        lines.append(f"TAC truncated: {truncated}/{total} ({rate_text})")
    strategy_counts = diagnostics.get("strategy_counts")
    if isinstance(strategy_counts, Mapping) and strategy_counts:
        strategies = ", ".join(f"{key}={value}" for key, value in sorted(strategy_counts.items()))
        lines.append(f"Strategies: {strategies}")

    lines.append("metric | mean | p50 | p90 | p95 | max")
    lines.append("--- | ---: | ---: | ---: | ---: | ---:")
    for key in (
        "context_window",
        "prompt_budget",
        "max_new_tokens",
        "tac_tokens_before",
        "tac_tokens_after",
        "prompt_tokens",
        "generated_tokens",
    ):
        stats = diagnostics.get(key)
        if not isinstance(stats, Mapping):
            continue
        percentiles = (
            stats.get("percentiles") if isinstance(stats.get("percentiles"), Mapping) else {}
        )
        lines.append(
            " | ".join(
                [
                    key,
                    _format_number(stats.get("mean")),
                    _format_number(percentiles.get("50th")),
                    _format_number(percentiles.get("90th")),
                    _format_number(percentiles.get("95th")),
                    _format_number(stats.get("max")),
                ]
            )
        )


def _append_benchmark_suite_report(
    lines: list[str],
    benchmark_suites: Mapping[str, Any],
) -> None:
    if not benchmark_suites:
        return
    lines.extend(["", "Benchmark Suites", "----------------"])
    lines.append(
        "suite | count | semantic | edit distance | replication F1 | bytecode semantic | Solidity valid"
    )
    lines.append("--- | ---: | ---: | ---: | ---: | ---: | ---:")
    for suite, suite_summary in sorted(benchmark_suites.items()):
        if not isinstance(suite_summary, Mapping):
            continue
        metrics = suite_summary.get("metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        lines.append(
            " | ".join(
                [
                    str(suite),
                    str(suite_summary.get("count", 0)),
                    _segment_metric(metrics, "semantic_similarity"),
                    _segment_metric(metrics, "normalized_edit_distance"),
                    _segment_metric(metrics, "replication_f1"),
                    _segment_metric(metrics, "bytecode_semantic_score"),
                    _segment_percent_metric(metrics, "solidity_valid"),
                ]
            )
        )


def _append_worst_samples_report(
    lines: list[str],
    worst_samples: Any,
    results_json_path: str,
    *,
    limit: int = 5,
) -> None:
    if isinstance(worst_samples, Mapping):
        samples = []
        ordered_reasons = [
            reason for reason in ("truncated_low_quality",) if reason in worst_samples
        ] + [reason for reason in worst_samples if reason != "truncated_low_quality"]
        for reason in ordered_reasons:
            reason_samples = worst_samples.get(reason)
            if not isinstance(reason_samples, Sequence) or isinstance(reason_samples, (str, bytes)):
                continue
            for sample in reason_samples:
                if not isinstance(sample, Mapping):
                    continue
                sample_payload = dict(sample)
                sample_payload.setdefault("reason", str(reason))
                samples.append(sample_payload)
    elif isinstance(worst_samples, Sequence) and not isinstance(worst_samples, (str, bytes)):
        samples = [sample for sample in worst_samples if isinstance(sample, Mapping)]
    else:
        return
    if not samples:
        return

    lines.extend(["", "Worst Samples", "-------------"])
    lines.append(f"Detailed JSON: {results_json_path}")
    for position, sample in enumerate(samples[:limit], start=1):
        metrics = sample.get("metrics") if isinstance(sample.get("metrics"), Mapping) else {}
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), Mapping) else {}
        dataset_index = (
            sample.get("dataset_index")
            or sample.get("index")
            or sample.get("sample_index")
            or metadata.get("dataset_index")
            or "n/a"
        )
        function = (
            sample.get("function_signature")
            or sample.get("signature")
            or sample.get("function_name")
            or metadata.get("function_signature")
            or metadata.get("function_name")
            or "unknown"
        )
        reason = (
            sample.get("reason") or sample.get("failure_reason") or sample.get("category") or "n/a"
        )
        identifier = (
            sample.get("hash") or sample.get("output_hash") or sample.get("input_hash") or "n/a"
        )
        metric_text = ", ".join(
            _sample_metric(metrics, key)
            for key in (
                "semantic_similarity",
                "normalized_edit_distance",
                "replication_f1",
                "bytecode_semantic_score",
            )
            if key in metrics
        )
        lines.extend(
            [
                "",
                f"{position}. dataset_index={dataset_index} function={function}",
                f"   reason={reason} hash={identifier}",
                f"   metrics={metric_text or 'n/a'}",
            ]
        )
        prompt_diagnostics = sample.get("prompt_diagnostics")
        if isinstance(prompt_diagnostics, Mapping) and prompt_diagnostics:
            diagnostic_parts = []
            for key in (
                "tac_truncated",
                "strategy",
                "marker",
                "context_window",
                "prompt_budget",
                "max_new_tokens",
                "tac_tokens_before",
                "tac_tokens_after",
                "prompt_tokens",
                "generated_tokens",
            ):
                value = prompt_diagnostics.get(key)
                if value not in (None, ""):
                    diagnostic_parts.append(f"{key}={value}")
            if diagnostic_parts:
                lines.append(f"   prompt_diagnostics={', '.join(diagnostic_parts)}")
        replication = _sample_replication_payload(sample)
        if replication:
            missing = _format_fact_diff(replication.get("missing_facts"))
            extra = _format_fact_diff(replication.get("extra_facts"))
            if missing:
                lines.append(f"   missing={missing}")
            if extra:
                lines.append(f"   extra={extra}")

        original = sample.get("original") or sample.get("reference") or sample.get("expected") or ""
        generated = (
            sample.get("decompiled") or sample.get("generated") or sample.get("candidate") or ""
        )
        if original:
            lines.append(f"   original: {_snippet(original)}")
        if generated:
            lines.append(f"   generated: {_snippet(generated)}")
        diff = sample.get("diff")
        if not diff and original and generated:
            diff = "\n".join(
                difflib.unified_diff(
                    str(original).splitlines(),
                    str(generated).splitlines(),
                    fromfile="original",
                    tofile="generated",
                    lineterm="",
                    n=1,
                )
            )
        if diff:
            lines.append("   diff: " + _snippet(diff, limit=320))


def _append_metadata_segment_report(lines: list[str], metadata_segments: Mapping[str, Any]) -> None:
    _append_opcode_control_flow_coverage(lines, metadata_segments)

    coverage = metadata_segments.get("coverage")
    if isinstance(coverage, Mapping) and coverage:
        lines.extend(["", "Metadata Segment Coverage", "-------------------------"])
        lines.append("field | known | unknown | top values")
        lines.append("--- | ---: | ---: | ---")
        for field_name, field_coverage in sorted(coverage.items()):
            if not isinstance(field_coverage, Mapping):
                continue
            values = field_coverage.get("values")
            top_values = _format_top_segment_values(values if isinstance(values, Mapping) else {})
            lines.append(
                " | ".join(
                    [
                        str(field_name),
                        str(field_coverage.get("known", 0)),
                        str(field_coverage.get("unknown", 0)),
                        top_values,
                    ]
                )
            )

    segments = metadata_segments.get("segments")
    if not isinstance(segments, Mapping) or not segments:
        return

    lines.extend(["", "Per-Segment Metrics", "-------------------"])
    lines.append("segment | count | semantic | edit distance | replication F1 | Solidity valid")
    lines.append("--- | ---: | ---: | ---: | ---: | ---:")
    for field_name, field_segments in sorted(segments.items()):
        if not isinstance(field_segments, Mapping):
            continue
        for value, segment in sorted(field_segments.items()):
            if not isinstance(segment, Mapping):
                continue
            metrics = segment.get("metrics")
            metrics = metrics if isinstance(metrics, Mapping) else {}
            lines.append(
                " | ".join(
                    [
                        f"{field_name}={value}",
                        str(segment.get("count", 0)),
                        _segment_metric(metrics, "semantic_similarity"),
                        _segment_metric(metrics, "normalized_edit_distance"),
                        _segment_metric(metrics, "replication_f1"),
                        _segment_percent_metric(metrics, "solidity_valid"),
                    ]
                )
            )
    _append_segment_hallucination_report(lines, segments)


def _append_opcode_control_flow_coverage(
    lines: list[str],
    metadata_segments: Mapping[str, Any],
) -> None:
    coverage = metadata_segments.get("opcode_control_flow_coverage")
    if not isinstance(coverage, Mapping):
        return
    opcode_groups = coverage.get("opcode_groups")
    control_flow = coverage.get("control_flow")
    if not isinstance(opcode_groups, Mapping) and not isinstance(control_flow, Mapping):
        return
    lines.extend(["", "Opcode and Control-Flow Coverage", "--------------------------------"])
    lines.append(f"Examples with opcode groups: {coverage.get('examples_with_opcode_groups', 0)}")
    if isinstance(opcode_groups, Mapping) and opcode_groups:
        lines.append(f"Opcode groups: {_format_top_segment_values(opcode_groups, limit=8)}")
    if isinstance(control_flow, Mapping) and control_flow:
        lines.append(f"Control-flow buckets: {_format_top_segment_values(control_flow, limit=8)}")


def _append_segment_hallucination_report(lines: list[str], segments: Mapping[str, Any]) -> None:
    rows: list[tuple[str, str, int, float]] = []
    for field_name, field_segments in sorted(segments.items()):
        if not isinstance(field_segments, Mapping):
            continue
        for value, segment in sorted(field_segments.items()):
            if not isinstance(segment, Mapping):
                continue
            replication = segment.get("replication_metrics")
            if not isinstance(replication, Mapping):
                continue
            buckets = replication.get("hallucination_buckets")
            if not isinstance(buckets, Mapping):
                continue
            rates = replication.get("hallucination_rate_by_bucket")
            rates = rates if isinstance(rates, Mapping) else {}
            for bucket, count in sorted(buckets.items()):
                if not isinstance(count, int):
                    continue
                rate = rates.get(bucket)
                rows.append(
                    (
                        f"{field_name}={value}",
                        str(bucket),
                        count,
                        float(rate) if isinstance(rate, (int, float)) else 0.0,
                    )
                )
    if not rows:
        return
    lines.extend(["", "Per-Segment Hallucination Buckets", "--------------------------------"])
    lines.append("segment | bucket | count | rate")
    lines.append("--- | --- | ---: | ---:")
    for segment, bucket, count, rate in rows[:20]:
        lines.append(f"{segment} | {bucket} | {count} | {rate * 100:.2f}%")


def _sample_metric(metrics: Mapping[str, Any], key: str) -> str:
    return f"{key}={_format_number(metrics.get(key))}"


def _sample_replication_payload(sample: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    replication = sample.get("replication")
    if isinstance(replication, Mapping):
        return replication
    metadata = sample.get("metadata")
    if isinstance(metadata, Mapping):
        replication = metadata.get("replication")
        if isinstance(replication, Mapping):
            return replication
    metrics = sample.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    metadata = metrics.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    replication = metadata.get("replication")
    return replication if isinstance(replication, Mapping) else None


def _format_fact_diff(value: Any, *, limit: int = 4) -> str:
    if not isinstance(value, Mapping):
        return ""
    parts: list[str] = []
    for category, facts in sorted(value.items()):
        if isinstance(facts, Sequence) and not isinstance(facts, (str, bytes)):
            shown = ", ".join(str(fact) for fact in list(facts)[:limit])
            parts.append(f"{category}=[{shown}]")
    return "; ".join(parts)


def _snippet(value: Any, *, limit: int = 240) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 13].rstrip() + "...<truncated>"


def _format_top_segment_values(values: Mapping[str, Any], *, limit: int = 4) -> str:
    pairs = sorted(
        ((str(value), count) for value, count in values.items()),
        key=lambda item: (-int(item[1]), item[0]) if isinstance(item[1], int) else (0, item[0]),
    )
    return ", ".join(f"{value} ({count})" for value, count in pairs[:limit]) or "n/a"


def _segment_metric(metrics: Mapping[str, Any], metric: str) -> str:
    value = metrics.get(metric)
    if isinstance(value, Mapping):
        return _format_number(value.get("mean"))
    return "n/a"


def _segment_percent_metric(metrics: Mapping[str, Any], metric: str) -> str:
    value = metrics.get(metric)
    if isinstance(value, Mapping):
        mean = value.get("mean")
        if isinstance(mean, (int, float)):
            return f"{mean * 100:.2f}%"
    return "n/a"


def _append_baseline_comparison(lines: list[str], comparison: Mapping[str, Any]) -> None:
    lines.extend(["", "Baseline / Regression Comparison", "--------------------------------"])
    lines.append(f"Metrics compared: {comparison.get('num_metrics_compared', 0)}")
    lines.append(f"Improvements: {comparison.get('num_improvements', 0)}")
    lines.append(f"Regressions: {comparison.get('num_regressions', 0)}")

    comparisons = comparison.get("comparisons")
    if not isinstance(comparisons, Mapping) or not comparisons:
        return

    lines.append("")
    lines.append("metric | current | baseline | delta | relative delta | status")
    lines.append("--- | ---: | ---: | ---: | ---: | ---")
    for metric, item in sorted(comparisons.items()):
        if not isinstance(item, Mapping):
            continue
        lines.append(
            " | ".join(
                [
                    str(metric),
                    _format_number(item.get("current")),
                    _format_number(item.get("baseline")),
                    _format_number(item.get("delta")),
                    _format_number(item.get("relative_delta")),
                    str(item.get("status", "unknown")),
                ]
            )
        )
