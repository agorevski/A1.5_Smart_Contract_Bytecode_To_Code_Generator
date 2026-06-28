"""Tests for eval failure slice analysis tooling."""

import json

from scripts.analyze_eval_failures import (
    analyze_eval,
    build_iteration_plan,
    format_markdown_report,
    write_slice_datasets,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle)
            handle.write("\n")


def test_analyze_eval_ranks_failure_slices_and_buckets(tmp_path):
    eval_path = tmp_path / "eval.json"
    _write_json(
        eval_path,
        {
            "summary": {
                "semantic_similarity_mean": 0.6,
                "replication_f1_micro": 0.4,
                "solidity_valid_mean": 0.5,
            },
            "details": [
                {
                    "dataset_index": 0,
                    "metadata": {
                        "compiler_version": "0.5.17",
                        "opcode_groups": ["calldata", "revert"],
                        "control_flow": ["revert_path"],
                    },
                    "metrics": {
                        "semantic_similarity": 0.4,
                        "replication_f1": 0.2,
                        "function_signature_match": False,
                        "solidity_valid": False,
                        "bytecode_deployable": False,
                        "bytecode_semantic_score": 0.1,
                        "metadata": {
                            "bytecode_semantics": {
                                "mismatch_buckets": {
                                    "call_mismatch": ["missing_call"],
                                    "guard_mismatch": ["missing_guard"],
                                }
                            },
                            "replication": {
                                "hallucination_buckets": {"unsupported_calls": ["foo"]},
                                "missing_facts": {"call": ["bar"], "guard": ["baz"]},
                            },
                            "solidity_validity": {"method": "compiler_ast"},
                        },
                    },
                },
                {
                    "dataset_index": 1,
                    "metadata": {"compiler_version": "0.8.20"},
                    "metrics": {
                        "semantic_similarity": 0.9,
                        "replication_f1": 0.8,
                        "function_signature_match": True,
                        "solidity_valid": True,
                        "bytecode_deployable": False,
                        "bytecode_semantic_score": 0.6,
                        "metadata": {
                            "bytecode_semantics": {
                                "mismatch_buckets": {"storage_write_mismatch": ["slot"]}
                            },
                            "replication": {
                                "hallucination_buckets": {},
                                "missing_facts": {"state_write": ["slot"]},
                            },
                            "solidity_validity": {"method": "compiler_ast_context_limited"},
                        },
                    },
                },
            ],
        },
    )

    analysis = analyze_eval(eval_path, top_n=20)
    slice_names = {item["name"] for item in analysis["ranked_slices"]}

    assert analysis["bytecode_mismatch_buckets"]["call_mismatch"] == 1
    assert analysis["hallucination_buckets"]["unsupported_calls"] == 1
    assert analysis["missing_fact_categories"]["call"] == 1
    assert analysis["compiler_validation_methods"]["compiler_ast"] == 1
    assert {"calls", "guards", "state_writes", "syntax_or_version"}.issubset(slice_names)
    assert analysis["summary_metrics"]["replication_f1_micro"] == 0.4


def test_write_slice_datasets_and_iteration_plan(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {"input": "tac0", "output": "sol0"},
            {"input": "tac1", "output": "sol1"},
            {"input": "tac2", "output": "sol2"},
        ],
    )
    analysis = {
        "eval_path": "results/eval_test.json",
        "num_details": 3,
        "ranked_slices": [
            {
                "name": "calls",
                "count": 2,
                "indices": [0, 2],
                "semantic_similarity_mean": 0.5,
                "replication_f1_mean": 0.2,
                "bytecode_semantic_score_mean": 0.1,
                "solidity_valid_mean": 1.0,
            }
        ],
        "bytecode_mismatch_buckets": {"call_mismatch": 2},
        "hallucination_buckets": {"unsupported_calls": 1},
        "missing_fact_categories": {"call": 2},
    }

    manifest = write_slice_datasets(
        analysis,
        dataset_path=dataset_path,
        output_dir=tmp_path / "slices",
        max_slices=1,
    )
    slice_path = tmp_path / "slices" / "calls.jsonl"
    plan = build_iteration_plan(analysis, count=3)
    report = format_markdown_report(analysis)

    assert manifest["slices"][0]["row_count"] == 2
    assert len(slice_path.read_text(encoding="utf-8").splitlines()) == 2
    assert [item["iteration"] for item in plan] == [1, 2, 3]
    assert plan[0]["focus_bucket"] == "call_mismatch"
    assert "| calls | 2 | 0.5000 | 0.2000 | 0.1000 | 100.00% |" in report
