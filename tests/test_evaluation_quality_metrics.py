"""Focused tests for evaluation quality metrics and reporting helpers."""

import difflib

import pytest

from src.training_pipeline import (
    SmartContractTrainingPipeline,
    compare_evaluation_to_baseline,
    compute_metadata_segment_metrics,
    mean_confidence_interval,
    normalized_levenshtein_distance,
    validate_generated_solidity,
)


def test_normalized_edit_distance_uses_true_levenshtein_not_sequence_matcher():
    original = "abc"
    generated = "yabd"

    levenshtein_distance = normalized_levenshtein_distance(original, generated)
    sequence_matcher_distance = 1.0 - difflib.SequenceMatcher(None, original, generated).ratio()

    assert levenshtein_distance == pytest.approx(0.5)
    assert levenshtein_distance != pytest.approx(sequence_matcher_distance)


def test_solidity_validity_fallback_accepts_balanced_function_without_solc():
    result = validate_generated_solidity(
        "function transfer(address to, uint256 amount) public { require(to != address(0)); }",
        allow_compiler=False,
    )

    assert result.valid is True
    assert result.method == "scaffold"
    assert result.compiler_checked is False
    assert result.scaffold_errors == []


def test_solidity_validity_fallback_rejects_malformed_function():
    result = validate_generated_solidity(
        "function transfer(address to public { require(to != address(0));",
        allow_compiler=False,
    )

    assert result.valid is False
    assert result.method == "scaffold"
    assert result.scaffold_errors


def test_metadata_segment_metrics_report_coverage_and_per_segment_means():
    results = [
        {
            "metrics": {
                "semantic_similarity": 0.9,
                "normalized_edit_distance": 0.1,
                "replication_f1": 0.8,
                "solidity_valid": True,
            },
            "metadata": {"compiler_version": "0.8.20", "optimizer_enabled": True},
        },
        {
            "metrics": {
                "semantic_similarity": 0.7,
                "normalized_edit_distance": 0.3,
                "replication_f1": 0.6,
                "solidity_valid": False,
            },
            "metadata": {"compiler_version": "0.8.20", "optimizer_enabled": False},
        },
        {
            "metrics": {
                "semantic_similarity": 0.5,
                "normalized_edit_distance": 0.5,
                "replication_f1": 0.4,
                "solidity_valid": True,
            },
            "metadata": {"optimizer_enabled": False},
        },
    ]

    summary = compute_metadata_segment_metrics(
        results,
        segment_fields=("compiler_version", "optimizer_enabled"),
    )

    assert summary["coverage"]["compiler_version"]["known"] == 2
    assert summary["coverage"]["compiler_version"]["unknown"] == 1
    assert summary["coverage"]["optimizer_enabled"]["values"] == {"False": 2, "True": 1}
    compiler_segment = summary["segments"]["compiler_version"]["0.8.20"]
    assert compiler_segment["count"] == 2
    assert compiler_segment["metrics"]["semantic_similarity"]["mean"] == pytest.approx(0.8)
    assert compiler_segment["metrics"]["solidity_valid"]["mean"] == pytest.approx(0.5)


def test_confidence_intervals_and_baseline_comparison_are_deterministic():
    interval = mean_confidence_interval([0.6, 0.8, 1.0])

    assert interval["n"] == 3
    assert interval["low"] < interval["mean"] < interval["high"]
    assert mean_confidence_interval([0.75])["low"] == pytest.approx(0.75)

    comparison = compare_evaluation_to_baseline(
        {
            "semantic_similarity": {"mean": 0.83},
            "normalized_edit_distance": {"mean": 0.32},
            "replication_metrics": {"f1_mean": 0.70},
        },
        {
            "semantic_similarity_mean": 0.80,
            "normalized_edit_distance_mean": 0.25,
            "replication_f1_mean": 0.70,
        },
    )

    assert comparison["comparisons"]["semantic_similarity_mean"]["status"] == "improved"
    assert comparison["comparisons"]["normalized_edit_distance_mean"]["status"] == "regressed"
    assert comparison["comparisons"]["replication_f1_mean"]["status"] == "unchanged"
    assert comparison["num_regressions"] == 1


def test_training_pipeline_aggregate_stats_include_ci_segments_and_baseline():
    pipeline = SmartContractTrainingPipeline.__new__(SmartContractTrainingPipeline)
    results = [
        {
            "metrics": {
                "semantic_similarity": 0.9,
                "normalized_edit_distance": 0.1,
                "solidity_valid": True,
                "metadata": {},
            },
            "metadata": {"compiler_version": "0.8.20"},
        },
        {
            "metrics": {
                "semantic_similarity": 0.7,
                "normalized_edit_distance": 0.3,
                "solidity_valid": False,
                "metadata": {},
            },
            "metadata": {"compiler_version": "0.8.10"},
        },
    ]

    stats = pipeline._compute_aggregate_statistics(
        results,
        baseline_summary={
            "semantic_similarity_mean": 0.75,
            "normalized_edit_distance_mean": 0.15,
        },
    )

    assert stats["semantic_similarity"]["confidence_interval_95"]["n"] == 2
    assert stats["solidity_valid"]["mean"] == pytest.approx(0.5)
    assert stats["metadata_segments"]["coverage"]["compiler_version"]["known"] == 2
    assert stats["baseline_comparison"]["num_metrics_compared"] >= 2
