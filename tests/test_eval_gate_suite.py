import hashlib
import json

from scripts.eval_gate_suite import evaluate_gate_suite, format_markdown_report


def _write_eval(path, num_rows, f1, bytecode, semantic=0.8, valid=1.0):
    details = [
        {
            "dataset_index": index,
            "input_hash": hashlib.sha256(f"tac {index}".encode()).hexdigest(),
            "output_hash": hashlib.sha256(f"sol {index}".encode()).hexdigest(),
            "metadata": {"function_signature": f"f{index}()"},
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
                    "replication_behavior_only_f1_micro": f1,
                    "bytecode_semantic_score_mean": bytecode,
                    "semantic_similarity_mean": semantic,
                    "solidity_valid_mean": valid,
                    "eval_batch_size": 1,
                    "eval_max_new_tokens": 512,
                    "eval_repetition_penalty": 1.05,
                    "include_selector_signature_metadata": True,
                    "selector_signature_prompt_policy": "bundled_only_v1",
                    "prompt_truncation_count": 0,
                    "eval_sampling_strategy": "all",
                    "eval_sample_indices": None,
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


def test_eval_gate_suite_rejects_noncomparable_improvement(tmp_path):
    base = tmp_path / "base.json"
    candidate = tmp_path / "candidate.json"
    _write_eval(base, 30, 0.5, 0.4)
    _write_eval(candidate, 30, 0.6, 0.5)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["details"][0]["input_hash"] = "0" * 64
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    suite = evaluate_gate_suite(
        [{"name": "broad", "baseline": base, "candidate": candidate, "min_rows": 30}]
    )

    assert suite["decision"] == "reject"
    assert "incomparable evaluations" in suite["comparisons"][0]["decision_reason"]


def test_small_slice_improvement_cannot_alone_promote_candidate(tmp_path):
    small_base = tmp_path / "small_base.json"
    small_candidate = tmp_path / "small_candidate.json"
    large_base = tmp_path / "large_base.json"
    large_candidate = tmp_path / "large_candidate.json"
    _write_eval(small_base, 17, 0.5, 0.4)
    _write_eval(small_candidate, 17, 0.6, 0.5)
    _write_eval(large_base, 64, 0.8, 0.7)
    _write_eval(large_candidate, 64, 0.8, 0.7)
    small = {"name": "state17", "baseline": small_base, "candidate": small_candidate, "min_rows": 1}
    large = {
        "name": "holdout64",
        "baseline": large_base,
        "candidate": large_candidate,
        "min_rows": 30,
    }

    assert evaluate_gate_suite([small])["decision"] == "smoke_only"
    suite = evaluate_gate_suite([small, large])
    assert suite["decision"] == "no_change"
    assert suite["comparisons"][0]["decision"] == "keep_candidate"
