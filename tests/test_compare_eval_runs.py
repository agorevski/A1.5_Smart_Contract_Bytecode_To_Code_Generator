import hashlib
import json

import pytest

from scripts.compare_eval_runs import compare_eval_runs, format_markdown_report


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
    path.write_text(json.dumps(payload), encoding="utf-8")


def _detail(index, replication_f1, bytecode_score, semantic, valid=True, buckets=None):
    source = f"tac {index}"
    original = f"function f{index}() external {{}}"
    return {
        "dataset_index": index,
        "input_hash": hashlib.sha256(source.encode()).hexdigest(),
        "output_hash": hashlib.sha256(original.encode()).hexdigest(),
        "original": original,
        "metadata": {"function_signature": f"function f{index}()"},
        "metrics": {
            "replication_f1": replication_f1,
            "bytecode_semantic_score": bytecode_score,
            "semantic_similarity": semantic,
            "solidity_valid": valid,
            "metadata": {
                "bytecode_semantics": {"mismatch_buckets": buckets or {}},
                "replication": {
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
        (lambda data: data["details"][1].update({"dataset_index": 0}), "duplicate dataset_index"),
        (
            lambda data: data["details"][0].update({"dataset_index": 99}),
            "dataset_index sets differ",
        ),
        (
            lambda data: data["details"][0]["metadata"].update({"function_signature": "other()"}),
            "different source row",
        ),
        (lambda data: data["details"][0].update({"input_hash": "0" * 64}), "different source row"),
        (lambda data: data["details"][0].pop("input_hash"), "missing input_hash"),
        (lambda data: data["summary"].update({"num_evaluated": 31}), "num_evaluated"),
        (lambda data: data["summary"].pop("solidity_valid_mean"), "solidity_valid_mean"),
        (lambda data: data["summary"].update({"eval_max_new_tokens": 256}), "eval_max_new_tokens"),
        (lambda data: data["summary"].update({"prompt_truncation_count": 1}), "prompt truncation"),
        (
            lambda data: data["details"][0]["metrics"].pop("replication_f1"),
            "missing or non-finite replication_f1",
        ),
        (
            lambda data: data["details"][0]["metrics"].update({"replication_f1": float("nan")}),
            "missing or non-finite replication_f1",
        ),
    ],
)
def test_comparison_rejects_unpaired_or_unverifiable_improvements(tmp_path, tamper, expected):
    baseline, candidate, baseline_data, candidate_data = _pair(tmp_path)
    _write_json(baseline, baseline_data)
    tamper(candidate_data)
    _write_json(candidate, candidate_data)

    comparison = compare_eval_runs(baseline, candidate)

    assert comparison["decision"] == "reject"
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

    assert comparison["decision"] == "reject"
    assert any(
        "selector_signature_prompt_policy" in error for error in comparison["comparability_errors"]
    )


def test_comparison_rejects_swapped_rows_even_when_indices_and_scores_collide(tmp_path):
    baseline, candidate, baseline_data, candidate_data = _pair(tmp_path)
    candidate_data["details"][0]["input_hash"], candidate_data["details"][1]["input_hash"] = (
        candidate_data["details"][1]["input_hash"],
        candidate_data["details"][0]["input_hash"],
    )
    _write_json(baseline, baseline_data)
    _write_json(candidate, candidate_data)

    comparison = compare_eval_runs(baseline, candidate)

    assert comparison["decision"] == "reject"
    assert comparison["paired_rows"] == 28
