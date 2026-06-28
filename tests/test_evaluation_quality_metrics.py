"""Focused tests for evaluation quality metrics and reporting helpers."""

import difflib

import pytest

from src.training_pipeline import (
    SmartContractTrainingPipeline,
    compare_evaluation_to_baseline,
    compute_benchmark_suite_metrics,
    compute_metadata_segment_metrics,
    extract_opcode_control_flow_slices,
    load_curated_evaluation_benchmarks,
    mean_confidence_interval,
    normalized_levenshtein_distance,
    solidity_function_signature_matches,
    validate_generated_solidity,
)
from src.replication_metrics import evaluate_replication


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


def test_solidity_validity_keeps_context_limited_fragments_syntax_valid(monkeypatch):
    import src.training_pipeline as training_pipeline

    def fake_solc_validation(source_code, metadata):
        return {
            "compiler_version": "0.8.20",
            "compiler_errors": [
                "Evaluation.sol:2:42: DeclarationError: Undeclared identifier.\n"
                "function guarded() public onlyOwner { balances[msg.sender] = 1; }\n"
                "                          ^-------^"
            ],
            "ast_valid": False,
        }

    monkeypatch.setattr(training_pipeline, "_try_local_solc_ast_validation", fake_solc_validation)

    result = training_pipeline.validate_generated_solidity(
        "function guarded() public onlyOwner { balances[msg.sender] = 1; }"
    )

    assert result.valid is True
    assert result.method == "compiler_ast_context_limited"
    assert result.scaffold_valid is True
    assert result.compiler_checked is True
    assert result.ast_valid is False
    assert result.deployable is False


def test_solidity_validity_rejects_non_context_compiler_errors(monkeypatch):
    import src.training_pipeline as training_pipeline

    def fake_solc_validation(source_code, metadata):
        return {
            "compiler_version": "0.8.20",
            "compiler_errors": ["Evaluation.sol:1:1: ParserError: Expected pragma or contract."],
            "ast_valid": False,
        }

    monkeypatch.setattr(training_pipeline, "_try_local_solc_ast_validation", fake_solc_validation)

    result = training_pipeline.validate_generated_solidity(
        "function guarded() public onlyOwner { balances[msg.sender] = 1; }"
    )

    assert result.valid is False
    assert result.method == "compiler_ast"
    assert result.scaffold_valid is True


def test_function_signature_match_compares_name_params_and_returns():
    reference = """
    function transfer(address to, uint256 amount) public returns (bool) {
        return true;
    }
    """
    wrong_name = """
    function approve(address to, uint256 amount) public returns (bool) {
        return true;
    }
    """
    wrong_params = """
    function transfer(address to) public returns (bool) {
        return true;
    }
    """
    same_signature = """
    function transfer(address recipient, uint256 value) external returns (bool) {
        return true;
    }
    """

    assert solidity_function_signature_matches(reference, same_signature) is True
    assert solidity_function_signature_matches(reference, wrong_name) is False
    assert solidity_function_signature_matches(reference, wrong_params) is False


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


def test_metadata_segment_metrics_honor_precomputed_opcode_slices():
    result = {
        "metrics": {
            "semantic_similarity": 0.9,
            "normalized_edit_distance": 0.1,
            "replication_f1": 0.8,
            "solidity_valid": True,
        },
        "metadata": {
            "opcode_groups": ["storage", "revert"],
            "control_flow": ["branching"],
        },
    }

    slices = extract_opcode_control_flow_slices(result)
    summary = compute_metadata_segment_metrics([result])

    assert slices["opcode_groups"] == ["revert", "storage"]
    assert slices["control_flow"] == ["branching"]
    assert summary["opcode_control_flow_coverage"]["opcode_groups"] == {
        "revert": 1,
        "storage": 1,
    }
    assert summary["opcode_control_flow_coverage"]["control_flow"] == {"branching": 1}


def test_opcode_segments_include_common_context_and_calldata_groups():
    result = {
        "input": """
        temp_1 = calldatasize
        temp_2 = caller
        temp_3 = callvalue
        temp_4 = timestamp
        temp_5 = extcodesize
        """,
        "metadata": {},
    }

    slices = extract_opcode_control_flow_slices(result)

    assert {
        "block_context",
        "calldata",
        "caller_context",
        "call_value",
        "code_introspection",
    }.issubset(slices["opcode_groups"])


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


def test_curated_evaluation_benchmarks_include_expected_facts_and_failures():
    suites = load_curated_evaluation_benchmarks("test_data/evaluation")

    assert set(suites) >= {"golden", "robustness"}
    assert len(suites["golden"]) >= 5
    assert len(suites["robustness"]) >= 4
    assert {
        "golden_create2_factory",
        "golden_delegatecall_proxy",
        "golden_constructor_receive",
    }.issubset({case["case_id"] for case in suites["golden"]})

    for case in suites["golden"]:
        assert case.get("version") == 1
        assert case.get("expected_facts")
        assert case.get("source")
        assert case.get("metadata", {}).get("bytecode")
        assert case.get("metadata", {}).get("tac")

    for case in suites["robustness"]:
        assert case.get("version") == 1
        assert case.get("expected_failure") is True
        assert case.get("expected_behavior")
        assert case.get("metadata", {}).get("bytecode")


def test_benchmark_suite_metrics_separate_curated_and_broad_holdout():
    results = [
        {
            "metrics": {
                "semantic_similarity": 0.95,
                "normalized_edit_distance": 0.05,
                "replication_f1": 0.9,
                "solidity_valid": True,
            },
            "metadata": {"benchmark_suite": "golden", "bytecode": "0x5ff5"},
        },
        {
            "metrics": {
                "semantic_similarity": 0.4,
                "normalized_edit_distance": 0.8,
                "replication_f1": 0.2,
                "solidity_valid": False,
            },
            "metadata": {"benchmark_suite": "robustness", "bytecode": "0xfe"},
        },
        {
            "metrics": {
                "semantic_similarity": 0.7,
                "normalized_edit_distance": 0.25,
                "replication_f1": 0.6,
                "solidity_valid": True,
            },
            "metadata": {"bytecode": "0x6001600055"},
        },
    ]

    summary = compute_benchmark_suite_metrics(results)

    assert summary["golden"]["count"] == 1
    assert summary["robustness"]["count"] == 1
    assert summary["broad_holdout"]["count"] == 1
    assert summary["golden"]["metrics"]["semantic_similarity"]["mean"] == pytest.approx(0.95)
    assert summary["robustness"]["metrics"]["solidity_valid"]["mean"] == pytest.approx(0.0)


def test_grounded_hallucination_buckets_are_distinct_from_missing_facts_by_segment():
    reference = """
    function transfer(address to, uint256 amount) public returns (bool) {
        balances[msg.sender] -= amount;
        balances[to] += amount;
        emit Transfer(msg.sender, to, amount);
        return true;
    }
    """
    candidate = """
    function transfer(address to, uint256 amount) public onlyOwner returns (uint256) {
        require(msg.sender == owner, "owner");
        balances[msg.sender] -= amount;
        feeVault += amount;
        emit FeeTaken(msg.sender, amount);
        _afterTransfer(to, amount);
        return amount;
    }
    """

    evaluation = evaluate_replication(reference, candidate)
    buckets = evaluation.hallucination_buckets

    assert set(buckets) >= {
        "unsupported_abi_elements",
        "unsupported_calls",
        "invented_guards",
        "invented_state_writes",
        "invented_events",
        "unsupported_return_expressions",
    }
    assert "balances[param_0]" in evaluation.missing_facts["state_write"]
    assert all(
        "balances[param_0]" not in fact
        for bucket_facts in buckets.values()
        for fact in bucket_facts
    )

    row = {
        "metrics": {
            "replication_precision": evaluation.overall.precision,
            "replication_recall": evaluation.overall.recall,
            "replication_f1": evaluation.overall.f1,
            "metadata": {"replication": evaluation.to_dict()},
        },
        "metadata": {
            "compiler_version": "0.8.20",
            "bytecode": "0x5ff45557fd5b5b5b",
        },
    }
    segmented = compute_metadata_segment_metrics(
        [row],
        segment_fields=("compiler_version", "opcode_group", "bytecode_length_bucket"),
    )

    compiler_metrics = segmented["segments"]["compiler_version"]["0.8.20"][
        "replication_metrics"
    ]
    assert compiler_metrics["hallucination_buckets"]["unsupported_calls"] == 1
    assert compiler_metrics["hallucination_rate_by_bucket"]["invented_guards"] > 0
    assert segmented["segments"]["opcode_group"]["delegatecall"]["replication_metrics"][
        "hallucination_buckets"
    ]["unsupported_calls"] == 1
    assert segmented["segments"]["bytecode_length_bucket"]["tiny"]["replication_metrics"][
        "hallucination_buckets"
    ]["invented_state_writes"] == 1


def test_grounding_facts_prevent_supported_extra_facts_from_being_hallucinations():
    reference = """
    function approve(address spender, uint256 amount) public returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }
    """
    candidate = """
    function approve(address spender, uint256 amount) public returns (bool) {
        _approve(msg.sender, spender, amount);
        _afterApprove(spender, amount);
        return true;
    }
    """

    evaluation = evaluate_replication(
        reference,
        candidate,
        grounding_facts={"call": ["_afterApprove"]},
    )

    assert "_afterapprove" in evaluation.extra_facts["call"]
    assert "unsupported_calls" not in evaluation.hallucination_buckets
