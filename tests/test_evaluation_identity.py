import json
import subprocess
import sys

import pytest

from src.evaluation_identity import (
    bind_evaluation_payload, build_training_provenance, check_model_dataset_overlap,
)


def _dataset(path, value, contract=None):
    row = {"input": f"TAC {value}", "output": f"function f() public {{ return {value}; }}",
           "metadata": {"body_hash": "untrusted_claim"}}
    if contract:
        row["metadata"]["contract_address"] = contract
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def _model(path, provenance):
    path.mkdir(exist_ok=True)
    (path / "training_input_manifest.json").write_text(json.dumps({"provenance": provenance}))
    return path


def test_selected_model_ancestor_training_and_selection_overlap_rejected(tmp_path):
    first = _dataset(tmp_path / "first.jsonl", 1)
    selection = _dataset(tmp_path / "selection.jsonl", 2)
    current = _dataset(tmp_path / "current.jsonl", 3)
    parent = {"provenance": build_training_provenance(first, selection)}
    model = _model(tmp_path / "model", build_training_provenance(current, ancestor_manifest=parent, continuation=True))
    for contaminated in (first, selection, current):
        with pytest.raises(ValueError, match="overlap"):
            check_model_dataset_overlap(model, contaminated)
    clean = _dataset(tmp_path / "clean.jsonl", 4)
    assert check_model_dataset_overlap(model, clean)["overlap_checked"] is True


def test_unknown_continuation_lineage_fails_closed(tmp_path):
    train = _dataset(tmp_path / "train.jsonl", 1)
    model = _model(tmp_path / "model", build_training_provenance(train, continuation=True))
    with pytest.raises(ValueError, match="incomplete"):
        check_model_dataset_overlap(model, _dataset(tmp_path / "test.jsonl", 2))


def test_contract_overlap_catches_different_function_bodies(tmp_path):
    train = _dataset(tmp_path / "train.jsonl", 1, "0xAB")
    model = _model(tmp_path / "model", build_training_provenance(train))
    test = _dataset(tmp_path / "test.jsonl", 2, "0xab")
    with pytest.raises(ValueError, match="overlap"):
        check_model_dataset_overlap(model, test)


def test_binding_rejects_duplicate_indices_and_hashes_actual_content(tmp_path):
    dataset = _dataset(tmp_path / "test.jsonl", 1)
    with pytest.raises(ValueError, match="duplicate"):
        bind_evaluation_payload({"details": [{"dataset_index": 0}, {"dataset_index": 0}]}, dataset, {"decode": "greedy"})
    before = bind_evaluation_payload({"details": [{"dataset_index": 0}]}, dataset, {"decode": "greedy"})
    _dataset(dataset, 2)
    after = bind_evaluation_payload({"details": [{"dataset_index": 0}]}, dataset, {"decode": "greedy"})
    assert before["dataset_content_sha256"] != after["dataset_content_sha256"]
    assert before["details"][0]["body_content_sha256"] != after["details"][0]["body_content_sha256"]


def test_preflight_cli_proves_selected_model_and_rejects_overlap(tmp_path):
    from tests.test_compare_eval_runs import _valid_pair
    baseline, _ = _valid_pair(tmp_path)
    model = tmp_path / "model"
    dataset = tmp_path / "cohort.jsonl"
    output = tmp_path / "audit.json"
    args = [sys.executable, "-m", "scripts.evaluation_preflight", "--model", str(model),
            "--pair", str(dataset), str(baseline), "--output", str(output)]
    passed = subprocess.run(args, capture_output=True, text=True)
    assert passed.returncode == 0, passed.stderr
    assert json.loads(output.read_text())["audits"][0]["overlap_checked"] is True
    _model(model, build_training_provenance(dataset))
    rejected = subprocess.run(args, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "overlaps selected model" in rejected.stderr


def test_binding_rejects_wrong_reference_output(tmp_path):
    dataset = _dataset(tmp_path / "test.jsonl", 1)
    with pytest.raises(ValueError, match="reference"):
        bind_evaluation_payload({"details": [{"dataset_index": 0, "original": "different"}]},
                                dataset, {"decode": "greedy"})
