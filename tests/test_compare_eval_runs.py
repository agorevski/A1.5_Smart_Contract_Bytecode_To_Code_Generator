import json
import copy

import pytest

from scripts.compare_eval_runs import compare_eval_runs, format_markdown_report
from src.evaluation_identity import bind_evaluation_payload, build_training_provenance


def _write_json(path, payload):
    dataset = path.parent / "cohort.jsonl"
    dataset.write_text("\n".join(json.dumps({
        "input": f"TAC {i}", "output": f"function f{i}() public {{ return {i}; }}",
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
    return {
        "dataset_index": index,
        "metadata": {"function_signature": f"function f{index}()"},
        "metrics": {
            "replication_f1": replication_f1,
            "bytecode_semantic_score": bytecode_score,
            "semantic_similarity": semantic,
            "solidity_valid": valid,
            "metadata": {
                "bytecode_semantics": {"mismatch_buckets": buckets or {}},
                "replication": {
                    "overall": {"true_positives": round(replication_f1 * 10000),
                                "false_positives": round((1 - replication_f1) * 10000),
                                "false_negatives": round((1 - replication_f1) * 10000)},
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
