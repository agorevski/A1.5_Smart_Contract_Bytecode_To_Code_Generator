import json

from scripts.compare_eval_runs import compare_eval_runs, format_markdown_report


def _write_json(path, payload):
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
            },
            "details": [_detail(0, 0.5, 0.3, 0.6), _detail(1, 0.5, 0.3, 0.6)],
        },
    )

    comparison = compare_eval_runs(baseline_path, candidate_path)

    assert comparison["decision"] == "smoke_only"
    assert comparison["paired_metric_deltas"]["replication_f1"]["improved_count"] == 2
