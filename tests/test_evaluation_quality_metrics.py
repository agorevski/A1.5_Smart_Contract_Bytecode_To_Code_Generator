"""Focused tests for evaluation quality metrics and reporting helpers."""

import difflib
import logging
import subprocess
import sys

import pytest

from src.training_pipeline import (
    SolidityValidityResult,
    SmartContractTrainingPipeline,
    compare_evaluation_to_baseline,
    compute_benchmark_suite_metrics,
    compute_metadata_segment_metrics,
    evaluate_bytecode_semantics,
    extract_opcode_control_flow_slices,
    load_curated_evaluation_benchmarks,
    mean_confidence_interval,
    normalized_levenshtein_distance,
    solidity_function_signature_matches,
    validate_generated_solidity,
)
from src.replication_metrics import evaluate_replication


def test_solidity_validation_does_not_require_evaluation_nlp_extras():
    code = r'''
import importlib.abc
import importlib.util
import sys
blocked = {"nltk", "rouge_score", "sentence_transformers", "sklearn", "scipy"}
find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *a, **kw: (
    None if name.split(".")[0] in blocked else find_spec(name, *a, **kw)
)
class MissingEvaluationExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ModuleNotFoundError("evaluation-only dependency: " + fullname)
sys.meta_path.insert(0, MissingEvaluationExtras())
from src.training_pipeline import validate_generated_solidity
result = validate_generated_solidity("function f() public { return; }", allow_compiler=False)
assert result.valid and result.method == "scaffold"
'''
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_malformed_output_preserves_reference_false_negatives():
    from src.training_pipeline import SmartContractEvaluator
    evaluator = SmartContractEvaluator.__new__(SmartContractEvaluator)
    result = evaluator.evaluate_function("function f() public { count = 1; }", "")
    replication = result.metadata["replication"]
    assert replication["overall"]["false_negatives"] == replication["reference_fact_count"] > 0
    assert replication["overall"]["true_positives"] == 0


def test_sload_evidence_does_not_require_state_write():
    from src.training_pipeline import evaluate_bytecode_semantics
    source = "function f() public view returns (uint) { return count; }"
    result = evaluate_bytecode_semantics(
        source, source, {"input": "v1 = SLOAD 0"},
        solidity_validity=validate_generated_solidity(source, allow_compiler=False),
    )
    assert "storage_write_mismatch" not in result.mismatch_buckets


def test_runtime_match_aggregate_is_conditional_on_checked_rows():
    pipeline = SmartContractTrainingPipeline.__new__(SmartContractTrainingPipeline)
    results = [{"metrics": {"bytecode_runtime_checked": index == 0,
                            "bytecode_runtime_match": index == 0}} for index in range(10)]
    stats = pipeline._compute_aggregate_statistics(results)
    assert stats["bytecode_runtime_checked"]["mean"] == pytest.approx(0.1)
    assert stats["bytecode_runtime_match_checked"]["mean"] == 1.0
    assert stats["bytecode_runtime_match_checked"]["count"] == 1


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


def test_bytecode_score_is_unchecked_without_reference_bytecode_or_opcode_evidence():
    code = "function foo() public { return; }"
    scaffold = validate_generated_solidity(code, allow_compiler=False)
    compiled_candidate = SolidityValidityResult(
        valid=True,
        method="compiler_ast",
        scaffold_valid=True,
        compiler_checked=True,
        ast_valid=True,
        bytecode_checked=True,
        deployable=True,
        compiled_runtime_bytecode="0x6000",
    )

    for validity in (scaffold, compiled_candidate):
        result = evaluate_bytecode_semantics(code, code, {}, solidity_validity=validity)
        assert result.checked is False
        assert result.score == 0.0
        assert result.runtime_bytecode_checked is False
        assert result.to_dict()["checked_kind"] == "reference_bytecode_or_opcode_evidence"
        assert result.to_dict()["score_kind"] == "source_fact_overlap_proxy"


@pytest.fixture
def local_reference_contract():
    solcx = pytest.importorskip("solcx")
    version = "0.8.20"
    if version not in {str(installed) for installed in solcx.get_installed_solc_versions()}:
        pytest.skip("solc 0.8.20 is not installed locally")

    source = (
        "pragma solidity ^0.8.20;\n"
        "contract Decoy { function value() public pure returns (uint256) { return 99; } }\n"
        "contract Counter { function value() public pure returns (uint256) { return 1; } }\n"
    )

    def compile_runtime(code, *, optimized=False):
        output = solcx.compile_standard(
            {
                "language": "Solidity",
                "sources": {"contract.sol": {"content": code}},
                "settings": {
                    "optimizer": {"enabled": optimized, "runs": 200},
                    "outputSelection": {"*": {"*": ["evm.deployedBytecode.object"]}},
                },
            },
            solc_version=version,
        )
        return output["contracts"]["contract.sol"]["Counter"]["evm"]["deployedBytecode"]["object"]

    runtime = compile_runtime(source)
    metadata = {
        "bytecode": "0x" + runtime,
        "compiler_version": version,
        "optimizer_enabled": False,
        "optimizer_runs": 200,
        "runtime_comparison": {
            "contract_name": "Counter",
            "compiler_version": version,
            "optimizer_enabled": False,
            "optimizer_runs": 200,
        },
    }
    return source, metadata, compile_runtime


def test_runtime_comparison_checks_exact_named_full_contract(local_reference_contract):
    source, metadata, _ = local_reference_contract
    equal = evaluate_bytecode_semantics(source, source, metadata)

    assert equal.runtime_bytecode_checked is True
    assert equal.runtime_bytecode_match is True
    assert equal.to_dict()["runtime_comparison_kind"] == "exact_full_contract_runtime_bytecode"
    assert equal.to_dict()["score_kind"] == "source_fact_overlap_proxy"
    assert equal.to_dict()["runtime_bytecode_skip_reason"] is None
    assert (
        evaluate_bytecode_semantics(
            source, source, {**metadata, "runtime_bytecode": metadata["bytecode"]}
        ).runtime_bytecode_match
        is True
    )
    assert (
        evaluate_bytecode_semantics(
            source,
            source,
            {
                **metadata,
                "bytecode": "",
                "evm": {"deployedBytecode": {"object": metadata["bytecode"]}},
            },
        ).runtime_bytecode_match
        is True
    )

    changed = source.replace("return 1;", "return 2;")
    wrong = evaluate_bytecode_semantics(
        source, changed, {**metadata, "candidate_runtime_bytecode": metadata["bytecode"]}
    )
    assert wrong.runtime_bytecode_checked is True
    assert wrong.runtime_bytecode_match is False
    assert wrong.mismatch_buckets["runtime_bytecode_mismatch"] == ["compiled_runtime_differs"]


def test_execution_fixtures_require_verified_full_contract_runtime(
    local_reference_contract, monkeypatch,
):
    import src.training_pipeline as training_pipeline

    source, metadata, _ = local_reference_contract
    calls = []

    def executed(reference_runtime, candidate_runtime, fixtures):
        calls.append((reference_runtime, candidate_runtime, fixtures))
        return {"checked": True, "match": True, "case_count": len(fixtures)}

    monkeypatch.setattr(training_pipeline, "execute_equivalence_subset", executed)
    with_fixtures = {**metadata, "execution_test_calldata": [""]}
    checked = evaluate_bytecode_semantics(source, source, with_fixtures)
    assert checked.runtime_bytecode_checked is True
    assert checked.executed_equivalence == {
        "checked": True, "match": True, "case_count": 1,
    }
    assert calls == [(metadata["bytecode"][2:], metadata["bytecode"][2:], [""])]

    for unchecked_metadata in (
        {**with_fixtures, "bytecode": "0x6000"},
        {**with_fixtures, "runtime_comparison": None},
        {key: value for key, value in with_fixtures.items() if key != "runtime_comparison"},
    ):
        unchecked = evaluate_bytecode_semantics(source, source, unchecked_metadata)
        assert unchecked.executed_equivalence["checked"] is False
    assert len(calls) == 1


def test_runtime_comparison_rejects_unverified_reference_or_compiler_settings(
    local_reference_contract,
):
    source, metadata, compile_runtime = local_reference_contract
    for reference, skip_reason in (
        ({**metadata, "bytecode": "0x6000"}, "reference_runtime_mismatch"),
        ({**metadata, "runtime_bytecode": "0x6000"}, "reference_runtime_mismatch"),
        (
            {**metadata, "bytecode": "0x" + compile_runtime(source, optimized=True)},
            "reference_runtime_mismatch",
        ),
        (
            {**metadata, "bytecode": "", "creation_bytecode": metadata["bytecode"]},
            "missing_or_invalid_reference_runtime",
        ),
        (
            {**metadata, "bytecode": "", "evm": {"bytecode": {"object": metadata["bytecode"]}}},
            "missing_or_invalid_reference_runtime",
        ),
        ({**metadata, "optimizer_enabled": True}, "conflicting_compiler_settings"),
        ({**metadata, "compiler_version": "0.8.19"}, "conflicting_compiler_settings"),
        ({**metadata, "runtime_comparison": True}, "invalid_compiler_settings"),
        (
            {
                **metadata,
                "runtime_comparison": {
                    **metadata["runtime_comparison"],
                    "compiler_version": "9.9.9",
                },
                "compiler_version": "9.9.9",
            },
            "reference_solc_version_not_installed",
        ),
        (
            {
                **metadata,
                "runtime_comparison": {
                    **metadata["runtime_comparison"],
                    "contract_name": "Missing",
                },
            },
            "reference_target_contract_missing",
        ),
        (
            {
                **metadata,
                "runtime_comparison": {**metadata["runtime_comparison"], "contract_name": "Decoy"},
            },
            "reference_runtime_mismatch",
        ),
    ):
        result = evaluate_bytecode_semantics(source, source, reference)
        assert result.runtime_bytecode_checked is False
        assert result.runtime_bytecode_match is None
        assert result.to_dict()["runtime_bytecode_skip_reason"] == skip_reason


def test_runtime_comparison_does_not_treat_compilable_fragments_as_full_contracts(
    local_reference_contract,
):
    _, metadata, _ = local_reference_contract
    fragment = "function value() public pure returns (uint256) { return 1; }"
    validity = validate_generated_solidity(fragment, metadata)
    assert validity.ast_valid is True
    assert validity.compiled_runtime_bytecode

    result = evaluate_bytecode_semantics(
        fragment,
        fragment,
        {**metadata, "candidate_runtime_bytecode": metadata["bytecode"]},
        solidity_validity=validity,
    )
    assert result.runtime_bytecode_checked is False
    assert result.runtime_bytecode_match is None
    assert result.to_dict()["runtime_bytecode_skip_reason"] == "reference_fragment"


@pytest.mark.parametrize(
    "source",
    [
        "contract Counter { uint256 public x; constructor() { x = 1; } }",
        "contract Counter { uint256 public x = 1; }",
        "contract Parent {}\n"
        "contract Counter is Parent { function value() public pure returns (uint256) { return 1; } }",
    ],
)
def test_runtime_comparison_excludes_deployment_or_inherited_context(
    source,
    local_reference_contract,
):
    _, metadata, compile_runtime = local_reference_contract
    metadata = {**metadata, "bytecode": "0x" + compile_runtime(source)}
    result = evaluate_bytecode_semantics(source, source, metadata)
    assert result.runtime_bytecode_checked is False
    assert result.runtime_bytecode_match is None
    assert result.to_dict()["runtime_bytecode_skip_reason"] in {
        "reference_abstract_or_inherited_contract",
        "reference_constructor_or_state_initializer",
    }


def test_runtime_compiler_rejection_exposes_skip_reason_and_log(
    local_reference_contract,
    monkeypatch,
    caplog,
):
    import solcx
    from solcx.exceptions import SolcError

    source, metadata, _ = local_reference_contract
    validity = validate_generated_solidity(source, allow_compiler=False)
    compiler = solcx.compile_standard

    def reject_candidate(*args, **kwargs):
        if "return 2;" in args[0]["sources"]["contract.sol"]["content"]:
            raise SolcError("Candidate compilation failed")
        return compiler(*args, **kwargs)

    monkeypatch.setattr(solcx, "compile_standard", reject_candidate)
    with caplog.at_level(logging.INFO, logger="src.training_pipeline"):
        result = evaluate_bytecode_semantics(
            source,
            source.replace("return 1;", "return 2;"),
            metadata,
            solidity_validity=validity,
        )
    assert result.runtime_bytecode_checked is False
    assert result.runtime_bytecode_match is None
    assert result.to_dict()["runtime_bytecode_skip_reason"] == "candidate_compiler_rejected_source"
    assert "candidate_compiler_rejected_source" in caplog.text


def test_runtime_compiler_io_failure_is_unchecked_and_logged(
    local_reference_contract,
    monkeypatch,
    caplog,
):
    import solcx

    source, metadata, _ = local_reference_contract
    validity = validate_generated_solidity(source, allow_compiler=False)

    def compiler_io_failure(*args, **kwargs):
        raise OSError("solc binary inaccessible")

    monkeypatch.setattr(solcx, "compile_standard", compiler_io_failure)
    with caplog.at_level(logging.WARNING, logger="src.training_pipeline"):
        result = evaluate_bytecode_semantics(source, source, metadata, solidity_validity=validity)
    assert result.runtime_bytecode_checked is False
    assert result.to_dict()["runtime_bytecode_skip_reason"] == "reference_compiler_io_error"
    assert "solc binary inaccessible" in caplog.text


def test_runtime_missing_local_solcx_is_unchecked_and_reported(
    local_reference_contract,
    monkeypatch,
    caplog,
):
    source, metadata, _ = local_reference_contract
    validity = validate_generated_solidity(source, allow_compiler=False)
    monkeypatch.setitem(sys.modules, "solcx", None)

    with caplog.at_level(logging.INFO, logger="src.training_pipeline"):
        result = evaluate_bytecode_semantics(source, source, metadata, solidity_validity=validity)
    assert result.runtime_bytecode_checked is False
    assert result.to_dict()["runtime_bytecode_skip_reason"] == "reference_solcx_unavailable"
    assert "reference_solcx_unavailable" in caplog.text


def test_runtime_unexpected_compiler_error_is_not_silenced(local_reference_contract, monkeypatch):
    import solcx

    source, metadata, _ = local_reference_contract
    validity = validate_generated_solidity(source, allow_compiler=False)

    def unexpected_error(*args, **kwargs):
        raise RuntimeError("unexpected compiler bridge bug")

    monkeypatch.setattr(solcx, "compile_standard", unexpected_error)
    with pytest.raises(RuntimeError, match="unexpected compiler bridge bug"):
        evaluate_bytecode_semantics(source, source, metadata, solidity_validity=validity)


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


def test_runtime_aggregate_reports_checked_count_and_equality_among_checked_only():
    pipeline = SmartContractTrainingPipeline.__new__(SmartContractTrainingPipeline)
    results = [
        {"metrics": {"bytecode_runtime_checked": True, "bytecode_runtime_match": True}},
        {"metrics": {"bytecode_runtime_checked": True, "bytecode_runtime_match": False}},
        {
            "metrics": {
                "bytecode_runtime_checked": False,
                "bytecode_runtime_match": False,
                "metadata": {
                    "bytecode_semantics": {"runtime_bytecode_skip_reason": "not_opted_in"}
                },
            }
        },
    ]
    comparison = pipeline._compute_aggregate_statistics(results)["runtime_bytecode_comparison"]
    assert comparison == {
        "checked_n": 2,
        "equal_n": 1,
        "total_n": 3,
        "equality_rate_checked": 0.5,
        "skip_reasons": {"not_opted_in": 1},
        "scope": "exact_full_contract_runtime_bytecode_not_deployment_or_behavior",
    }
    no_checks = pipeline._compute_aggregate_statistics(results[-1:])["runtime_bytecode_comparison"]
    assert no_checks["checked_n"] == 0
    assert no_checks["equality_rate_checked"] is None
    assert no_checks["skip_reasons"] == {"not_opted_in": 1}


def test_execution_aggregate_counts_only_executed_stateless_fixtures():
    pipeline = SmartContractTrainingPipeline.__new__(SmartContractTrainingPipeline)
    results = [
        {"metrics": {"metadata": {"bytecode_semantics": {
            "executed_equivalence": {"checked": True, "match": True},
        }}}},
        {"metrics": {"metadata": {"bytecode_semantics": {
            "executed_equivalence": {"checked": True, "match": False},
        }}}},
        {"metrics": {"metadata": {"bytecode_semantics": {
            "executed_equivalence": {"checked": False, "reason": "unsupported_opcode"},
        }}}},
    ]
    execution = pipeline._compute_aggregate_statistics(results)["executed_equivalence"]
    assert execution == {
        "checked_n": 2, "matched_n": 1, "total_n": 3,
        "scope": "bounded_stateless_explicit_calldata_fixtures_not_general_equivalence",
    }


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
    assert compiler_metrics["hallucination_buckets"]["unsupported_calls"] == 2
    assert compiler_metrics["hallucination_rate_by_bucket"]["invented_guards"] > 0
    assert segmented["segments"]["opcode_group"]["delegatecall"]["replication_metrics"][
        "hallucination_buckets"
    ]["unsupported_calls"] == 2
    assert segmented["segments"]["bytecode_length_bucket"]["tiny"]["replication_metrics"][
        "hallucination_buckets"
    ]["invented_state_writes"] == 2


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
        grounding_facts={"call": ["_afterApprove", "_afterApprove(param_0,param_1)"]},
    )

    assert "_afterapprove" in evaluation.extra_facts["call"]
    assert "unsupported_calls" not in evaluation.hallucination_buckets
