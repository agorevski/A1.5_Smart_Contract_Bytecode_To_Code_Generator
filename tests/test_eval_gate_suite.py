import json
import subprocess
import sys
from pathlib import Path

from scripts.eval_gate_suite import evaluate_gate_suite, format_markdown_report
from src.evaluation_identity import bind_evaluation_payload, build_training_provenance


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
                        "overall": {"true_positives": round(f1 * 10000),
                                    "false_positives": round((1 - f1) * 10000),
                                    "false_negatives": round((1 - f1) * 10000)},
                        "hallucination_buckets": {},
                        "missing_facts": {},
                    },
                },
            },
        }
        for index in range(num_rows)
    ]
    dataset = path.parent / f"cohort{num_rows}.jsonl"
    dataset.write_text("\n".join(json.dumps({
        "input": f"TAC {i}", "output": f"function f{i}() public {{ return {i}; }}",
    }) for i in range(num_rows)), encoding="utf-8")
    training = path.parent / "train.jsonl"
    training.write_text(json.dumps({"input": "TAC", "output": "function trainOnly() public { return 10000; }"}))
    model = path.parent / "model"
    model.mkdir(exist_ok=True)
    (model / "training_input_manifest.json").write_text(json.dumps({
        "provenance": build_training_provenance(training),
    }))
    path.write_text(
        json.dumps(
            bind_evaluation_payload({
                "summary": {
                    "num_evaluated": num_rows,
                    "replication_f1_micro": f1,
                    "bytecode_semantic_score_mean": bytecode,
                    "semantic_similarity_mean": semantic,
                    "solidity_valid_mean": valid,
                },
                "details": details,
            }, dataset, {"max_new_tokens": 512}, model)
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


def test_suite_empty_is_inconclusive():
    assert evaluate_gate_suite([])["decision"] == "inconclusive"


def test_rejected_suite_cli_exits_nonzero(tmp_path):
    base, candidate = tmp_path / "base.json", tmp_path / "candidate.json"
    _write_eval(base, 30, 0.6, 0.5)
    _write_eval(candidate, 30, 0.5, 0.5)
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.eval_gate_suite", "--pair", "holdout",
         str(base), str(candidate), "30"], capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "reject" in completed.stdout


def test_tiny_diagnostics_cannot_supply_acceptance_evidence(tmp_path):
    base, candidate = tmp_path / "base.json", tmp_path / "candidate.json"
    _write_eval(base, 2, 0.5, 0.4)
    _write_eval(candidate, 2, 0.6, 0.5)
    result = evaluate_gate_suite([{"name": "tiny", "baseline": base, "candidate": candidate,
                                  "min_rows": 1, "diagnostic_only": True}])
    assert result["decision"] == "inconclusive"


def test_gate_runner_preflights_before_inference_and_uses_explicit_outputs():
    runner = (Path(__file__).resolve().parents[1] / "run_eval_gate_suite_for_model.sh").read_text()
    assert runner.index("scripts/evaluation_preflight.py") < runner.index("uv run --extra evaluation torchrun")
    assert "--eval-output-json" in runner
    assert "--skip-data-preflight" not in runner
    assert "newest_eval_json" not in runner
    assert "--diagnostic calls23 --diagnostic state17" in runner
