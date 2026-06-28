import json

from scripts.eval_gate_suite import evaluate_gate_suite, format_markdown_report


def _write_eval(path, num_rows, f1, bytecode, semantic=0.8, valid=1.0):
    details = [
        {
            "dataset_index": index,
            "metrics": {
                "replication_f1": f1,
                "bytecode_semantic_score": bytecode,
                "semantic_similarity": semantic,
                "solidity_valid": valid,
                "metadata": {
                    "bytecode_semantics": {"mismatch_buckets": {}},
                    "replication": {
                        "hallucination_buckets": {},
                        "missing_facts": {},
                    },
                },
            },
        }
        for index in range(num_rows)
    ]
    path.write_text(
        json.dumps(
            {
                "summary": {
                    "num_evaluated": num_rows,
                    "replication_f1_micro": f1,
                    "bytecode_semantic_score_mean": bytecode,
                    "semantic_similarity_mean": semantic,
                    "solidity_valid_mean": valid,
                },
                "details": details,
            }
        ),
        encoding="utf-8",
    )


def test_eval_gate_suite_rejects_if_any_required_pair_regresses(tmp_path):
    broad_base = tmp_path / "broad_base.json"
    broad_candidate = tmp_path / "broad_candidate.json"
    holdout_base = tmp_path / "holdout_base.json"
    holdout_candidate = tmp_path / "holdout_candidate.json"
    _write_eval(broad_base, 30, 0.6, 0.4)
    _write_eval(broad_candidate, 30, 0.61, 0.41)
    _write_eval(holdout_base, 64, 0.8, 0.7)
    _write_eval(holdout_candidate, 64, 0.79, 0.71)

    suite = evaluate_gate_suite(
        [
            {
                "name": "broad",
                "baseline": broad_base,
                "candidate": broad_candidate,
                "min_rows": 30,
            },
            {
                "name": "holdout",
                "baseline": holdout_base,
                "candidate": holdout_candidate,
                "min_rows": 30,
            },
        ]
    )
    report = format_markdown_report(suite)

    assert suite["decision"] == "reject"
    assert [comparison["decision"] for comparison in suite["comparisons"]] == [
        "keep_candidate",
        "reject",
    ]
    assert "Decision: **reject**" in report


def test_eval_gate_suite_marks_any_small_pair_smoke_only(tmp_path):
    base = tmp_path / "base.json"
    candidate = tmp_path / "candidate.json"
    _write_eval(base, 2, 0.5, 0.4)
    _write_eval(candidate, 2, 0.6, 0.5)

    suite = evaluate_gate_suite(
        [{"name": "tiny", "baseline": base, "candidate": candidate, "min_rows": 30}]
    )

    assert suite["decision"] == "smoke_only"
