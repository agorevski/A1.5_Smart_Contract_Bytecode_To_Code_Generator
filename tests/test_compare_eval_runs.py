import hashlib
import json
import copy

import pytest

from scripts.compare_eval_runs import (
    PAIRED_METRICS, SUMMARY_GATE_METRICS, _validate_pair,
    compare_eval_runs, format_markdown_report, load_eval,
)
from src.evaluation_identity import bind_evaluation_payload, build_training_provenance


SETTINGS = {
    "eval_batch_size": 1,
    "eval_max_new_tokens": 512,
    "eval_repetition_penalty": 1.05,
    "include_selector_signature_metadata": True,
    "selector_signature_prompt_policy": "bundled_only_v1",
    "prompt_truncation_count": 0,
    "eval_sampling_strategy": "all",
    "eval_sample_indices": None,
}


def _write_json(path, payload):
    for key, value in SETTINGS.items():
        payload["summary"].setdefault(key, value)
    payload["summary"].setdefault(
        "replication_behavior_only_f1_micro",
        payload["summary"]["replication_f1_micro"],
    )
    payload["summary"].setdefault("num_failed", 0)
    payload["summary"].setdefault("num_succeeded", len(payload["details"]))
    payload["summary"].setdefault("failure_rate", 0.0)
    payload["summary"].setdefault(
        "aggregate_statistics", {"evaluator_error_count": 0}
    )
    payload["summary"].setdefault(
        "prompt_diagnostics",
        {"num_details": len(payload["details"]), "truncated_count": 0},
    )
    dataset = path.parent / "cohort.jsonl"
    dataset.write_text("\n".join(json.dumps({
        "input": f"TAC {i}", "output": f"function f{i}() external {{ return {i}; }}",
    }) for i in range(len(payload["details"]))), encoding="utf-8")
    training = path.parent / "train.jsonl"
    training.write_text(json.dumps({"input": "TAC", "output": "function trainOnly() public { return 10000; }"}))
    model = path.parent / "model"
    model.mkdir(exist_ok=True)
    (model / "training_input_manifest.json").write_text(json.dumps({
        "provenance": build_training_provenance(training),
    }))
    payload = bind_evaluation_payload(payload, dataset, {"max_new_tokens": 512}, model)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _detail(index, replication_f1, bytecode_score, semantic, valid=True, buckets=None):
    source = f"TAC {index}"
    original = f"function f{index}() external {{ return {index}; }}"
    true_positives = round(replication_f1 * 10000)
    false_positives = false_negatives = 10000 - true_positives
    return {
        "dataset_index": index,
        "success": True,
        "input": source,
        "input_hash": hashlib.sha256(source.encode()).hexdigest(),
        "output_hash": hashlib.sha256(original.encode()).hexdigest(),
        "original": original,
        "prompt_diagnostics": {"tac_truncated": False},
        "metadata": {"function_signature": f"function f{index}()"},
        "metrics": {
            "replication_f1": replication_f1,
            "bytecode_semantic_score": bytecode_score,
            "semantic_similarity": semantic,
            "solidity_valid": valid,
            "metadata": {
                "bytecode_semantics": {"mismatch_buckets": buckets or {}},
                "replication": {
                    "overall": {"true_positives": true_positives,
                                "false_positives": false_positives,
                                "false_negatives": false_negatives},
                    "by_category": {"call": {
                        "true_positives": true_positives,
                        "false_positives": false_positives,
                        "false_negatives": false_negatives,
                    }},
                    "hallucination_buckets": {},
                    "missing_facts": {},
                },
            },
        },
    }


def test_compare_eval_runs_rejects_gate_regression(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    details = [_detail(i, 0.8, 0.7, 0.9) for i in range(30)]
    _write_json(
        baseline_path,
        {
            "summary": {
                "num_evaluated": 30,
                "replication_f1_micro": 0.8,
                "bytecode_semantic_score_mean": 0.7,
                "semantic_similarity_mean": 0.9,
                "solidity_valid_mean": 1.0,
            },
            "details": details,
        },
    )
    _write_json(
        candidate_path,
        {
            "summary": {
                "num_evaluated": 30,
                "replication_f1_micro": 0.79,
                "bytecode_semantic_score_mean": 0.71,
                "semantic_similarity_mean": 0.91,
                "solidity_valid_mean": 1.0,
            },
            "details": [_detail(i, 0.79, 0.71, 0.91) for i in range(30)],
        },
    )

    comparison = compare_eval_runs(baseline_path, candidate_path)
    report = format_markdown_report(comparison)

    assert comparison["decision"] == "reject"
    assert "replication_f1_micro" in comparison["decision_reason"]
    assert comparison["summary_deltas"]["replication_f1_micro"]["delta"] < 0
    assert "Decision: **reject**" in report


def test_compare_eval_runs_marks_small_runs_smoke_only(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write_json(
        baseline_path,
        {
            "summary": {
                "num_evaluated": 2,
                "replication_f1_micro": 0.4,
                "bytecode_semantic_score_mean": 0.2,
                "semantic_similarity_mean": 0.5,
                "solidity_valid_mean": 1.0,
            },
            "details": [_detail(0, 0.4, 0.2, 0.5), _detail(1, 0.4, 0.2, 0.5)],
        },
    )
    _write_json(
        candidate_path,
        {
            "summary": {
                "num_evaluated": 2,
                "replication_f1_micro": 0.5,
                "bytecode_semantic_score_mean": 0.3,
                "semantic_similarity_mean": 0.6,
                "solidity_valid_mean": 1.0,
            },
            "details": [_detail(0, 0.5, 0.3, 0.6), _detail(1, 0.5, 0.3, 0.6)],
        },
    )

    comparison = compare_eval_runs(baseline_path, candidate_path)

    assert comparison["decision"] == "smoke_only"
    assert comparison["paired_metric_deltas"]["replication_f1"]["improved_count"] == 2


def test_comparison_rejects_behavior_regression_hidden_by_all_fact_gain(tmp_path):
    baseline, candidate, baseline_data, candidate_data = _pair(tmp_path)
    for payload, behavior, abi in (
        (baseline_data, (5000, 5000, 5000), (5000, 5000, 5000)),
        (candidate_data, (4000, 6000, 6000), (8000, 2000, 2000)),
    ):
        for detail in payload["details"]:
            replication = detail["metrics"]["metadata"]["replication"]
            replication["by_category"] = {
                category: dict(zip(
                    ("true_positives", "false_positives", "false_negatives"), counts
                ))
                for category, counts in (("call", behavior), ("abi", abi))
            }
            replication["overall"] = dict(zip(
                ("true_positives", "false_positives", "false_negatives"),
                [left + right for left, right in zip(behavior, abi)],
            ))
    baseline_data["summary"]["replication_behavior_only_f1_micro"] = 0.5
    candidate_data["summary"]["replication_behavior_only_f1_micro"] = 0.4
    _write_json(baseline, baseline_data)
    _write_json(candidate, candidate_data)

    comparison = compare_eval_runs(baseline, candidate)

    assert comparison["decision"] == "reject"
    assert "replication_behavior_only_f1_micro" in comparison["decision_reason"]


def _pair(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"

    def payload(f1):
        return {
            "summary": {
                "num_evaluated": 30,
                "replication_f1_micro": f1,
                "bytecode_semantic_score_mean": f1,
                "semantic_similarity_mean": f1,
                "solidity_valid_mean": 1.0,
            },
            "details": [_detail(i, f1, f1, f1) for i in range(30)],
        }

    return baseline, candidate, payload(0.5), payload(0.6)


@pytest.mark.parametrize(
    "tamper,expected",
    [
        (lambda data: data["details"][1].update({"dataset_index": 0}), "duplicate"),
        (
            lambda data: data["details"][0].update({"dataset_index": 99}),
            "dataset_index sets differ",
        ),
        (
            lambda data: data["details"][0]["metadata"].update({"function_signature": "other()"}),
            "different source row",
        ),
        (lambda data: data["details"][0].update({"input_hash": "0" * 64}), "input_hash does not match"),
        (
            lambda data: (
                data["details"][0].pop("input_hash"),
                data["details"][0].update(input="changed TAC"),
            ),
            "different source row",
        ),
        (lambda data: data["summary"].update({"num_evaluated": 31}), "num_evaluated"),
        (lambda data: data["summary"].pop("solidity_valid_mean"), "solidity_valid_mean"),
        (lambda data: data["summary"].update({"eval_max_new_tokens": 256}), "eval_max_new_tokens"),
        (lambda data: data["summary"].update({"prompt_truncation_count": 1}), "prompt truncation"),
        (
            lambda data: data["details"][0].update(prompt_diagnostics={"tac_truncated": True}),
            "prompt truncation observed",
        ),
        (
            lambda data: data["details"][0].pop("prompt_diagnostics"),
            "missing prompt truncation measurement",
        ),
        (
            lambda data: data["summary"]["prompt_diagnostics"].update(num_details=29),
            "incomplete prompt truncation diagnostic coverage",
        ),
        (
            lambda data: data["details"][0]["metrics"].pop("replication_f1"),
            "incomplete mandatory detail metric: replication_f1",
        ),
        (
            lambda data: data["details"][0]["metrics"].update({"replication_f1": float("nan")}),
            "incomplete mandatory detail metric: replication_f1",
        ),
    ],
)
def test_comparison_rejects_unpaired_or_unverifiable_improvements(tmp_path, tamper, expected):
    baseline, candidate, baseline_data, candidate_data = _pair(tmp_path)
    _write_json(baseline, baseline_data)
    _write_json(candidate, candidate_data)
    candidate_data = json.loads(candidate.read_text(encoding="utf-8"))
    tamper(candidate_data)
    candidate.write_text(json.dumps(candidate_data), encoding="utf-8")

    comparison = compare_eval_runs(baseline, candidate)

    assert comparison["decision"] == "inconclusive"
    assert expected in comparison["decision_reason"] or any(
        expected in error for error in comparison["comparability_errors"]
    )


def test_comparison_rejects_baseline_without_selector_provenance(tmp_path):
    baseline, candidate, baseline_data, candidate_data = _pair(tmp_path)
    _write_json(baseline, baseline_data)
    _write_json(candidate, candidate_data)
    historical = json.loads(baseline.read_text())
    del historical["summary"]["selector_signature_prompt_policy"]
    baseline.write_text(json.dumps(historical))

    comparison = compare_eval_runs(baseline, candidate)

    assert comparison["decision"] == "inconclusive"
    assert any(
        "selector_signature_prompt_policy" in error for error in comparison["comparability_errors"]
    )


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (
            lambda run: run["summary"].pop("selector_signature_prompt_policy"),
            "selector_signature_prompt_policy",
        ),
        (
            lambda run: run["summary"].update(prompt_truncation_count=1),
            "prompt truncation",
        ),
        (
            lambda run: run["details"][0].update(input_hash="0" * 64),
            "input_hash",
        ),
    ],
)
def test_preflight_pair_validator_rejects_noncomparable_baseline(tmp_path, mutation, reason):
    baseline, _ = _valid_pair(tmp_path)
    run = load_eval(baseline)
    mutation(run)
    with pytest.raises(ValueError, match=reason):
        _validate_pair(run, run, SUMMARY_GATE_METRICS, PAIRED_METRICS)


def test_comparison_rejects_swapped_rows_even_when_indices_and_scores_collide(tmp_path):
    baseline, candidate, baseline_data, candidate_data = _pair(tmp_path)
    _write_json(baseline, baseline_data)
    _write_json(candidate, candidate_data)
    candidate_data = json.loads(candidate.read_text(encoding="utf-8"))
    candidate_data["details"][0]["input_hash"], candidate_data["details"][1]["input_hash"] = (
        candidate_data["details"][1]["input_hash"],
        candidate_data["details"][0]["input_hash"],
    )
    candidate.write_text(json.dumps(candidate_data), encoding="utf-8")

    comparison = compare_eval_runs(baseline, candidate)

    assert comparison["decision"] == "inconclusive"
    assert comparison["paired_rows"] == 28


def _valid_pair(tmp_path, n=30):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    for path, score in ((baseline, 0.5), (candidate, 0.6)):
        _write_json(path, {"summary": {
            "num_evaluated": n, "replication_f1_micro": score,
            "bytecode_semantic_score_mean": score, "semantic_similarity_mean": score,
            "solidity_valid_mean": 1.0,
        }, "details": [_detail(i, score, score, score) for i in range(n)]})
    return baseline, candidate


@pytest.mark.parametrize("mutation", [
    lambda p: p["summary"].pop("replication_f1_micro"),
    lambda p: p["summary"].update(replication_f1_micro=float("nan")),
    lambda p: p["summary"].update(replication_f1_micro=float("inf")),
    lambda p: p["summary"].update(replication_f1_micro=True),
    lambda p: p["summary"].update(num_evaluated=999),
    lambda p: p["details"].append(copy.deepcopy(p["details"][0])),
    lambda p: p["details"][0].update(dataset_index=999),
    lambda p: p["details"][0]["metrics"].pop("replication_f1"),
    lambda p: p["details"][0]["metrics"].update(replication_f1=True),
    lambda p: p["details"][0]["metrics"].update(replication_f1=float("nan")),
    lambda p: p["details"][0]["metrics"]["metadata"].update(error="evaluator failed"),
    lambda p: p["details"][0].update(success=False),
    lambda p: p["summary"].update(num_failed=1),
    lambda p: p["summary"].pop("num_failed"),
    lambda p: p["summary"]["aggregate_statistics"].pop("evaluator_error_count"),
    lambda p: p["details"][0].update(row_content_sha256="changed"),
    lambda p: p.update(evaluator_version="historical"),
    lambda p: p.update(evaluation_config={"max_new_tokens": 1}),
    lambda p: p.pop("model_provenance"),
    lambda p: p["model_provenance"]["overlap_audit"].update(overlap_checked=False),
    lambda p: p["details"].__setitem__(0, "malformed"),
])
def test_acceptance_fails_closed_on_invalid_evidence(tmp_path, mutation):
    baseline, candidate = _valid_pair(tmp_path)
    payload = json.loads(candidate.read_text())
    mutation(payload)
    candidate.write_text(json.dumps(payload))
    assert compare_eval_runs(baseline, candidate)["decision"] == "inconclusive"


def test_independent_units_not_row_count_control_acceptance(tmp_path):
    baseline, candidate = _valid_pair(tmp_path)
    for path in (baseline, candidate):
        payload = json.loads(path.read_text())
        for detail in payload["details"]:
            detail["independent_unit"] = "contract:one"
        path.write_text(json.dumps(payload))
    result = compare_eval_runs(baseline, candidate, min_rows=1)
    assert result["decision"] == "smoke_only"
    assert result["independent_units"] == 1


def test_disjoint_cohorts_cannot_pass(tmp_path):
    baseline, candidate = _valid_pair(tmp_path)
    payload = json.loads(candidate.read_text())
    for row in payload["details"]:
        row["dataset_index"] += 100
    candidate.write_text(json.dumps(payload))
    assert compare_eval_runs(baseline, candidate)["decision"] == "inconclusive"


def test_finite_complete_comparable_improvement_passes(tmp_path):
    baseline, candidate = _valid_pair(tmp_path)
    assert compare_eval_runs(baseline, candidate)["decision"] == "keep_candidate"
