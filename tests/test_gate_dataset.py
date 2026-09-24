import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.gate_dataset import (
    body_identity,
    canonicalize_body_hash,
    exclude_eval_rows,
    file_sha256,
    load_jsonl,
    row_keys,
    sample_dataset,
    verify_baseline_artifact,
    verify_eval_artifact,
    verify_model_gate_exclusion,
)

ROOT = Path(__file__).resolve().parents[1]
GATE_VARS = (
    "BROAD_DATASET",
    "CALLS_DATASET",
    "STATE_DATASET",
    "HOLDOUT64_DATASET",
    "PURE_NEGATIVE_DATASET",
    "LARGE192_DATASET",
)
BASELINE_VARS = (
    "BROAD_BASELINE",
    "CALLS_BASELINE",
    "STATE_BASELINE",
    "HOLDOUT64_BASELINE",
    "PURE_NEGATIVE_BASELINE",
    "LARGE192_BASELINE",
)


def _row(name, *, output=None, **metadata):
    return {
        "input": f"tac {name}",
        "output": output or f"function {name}() external {{ return; }}",
        "metadata": metadata,
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_gate_exclusion_uses_recomputed_body_with_missing_and_stale_hashes():
    gate = _row("shared", body_hash="declared-gate-hash")
    duplicate = _row(
        "different",
        output="FUNCTION shared() external { /*comment*/ return; }",
    )
    stale = _row("shared", body_hash="different-stale-hash")
    variant = _row("variant", body_hash="declared-gate-hash")
    clean = _row("clean")

    selected, count = exclude_eval_rows([duplicate, stale, variant, clean], [gate])

    assert selected == [clean]
    assert count == 3
    assert body_identity(duplicate) == body_identity(gate)
    assert body_identity(variant) != body_identity(gate)


def test_gate_exclusion_keeps_colliding_metadata_namespaces_separate():
    gate = _row("gate", contract_address="a:b", selector="c", body_hash="same")
    unrelated = _row("unrelated", contract_address="a", selector="b:c")
    # A colon-delimited "address:selector" string would incorrectly match.
    selected, count = exclude_eval_rows([unrelated], [gate])

    assert selected == [unrelated]
    assert count == 0
    assert row_keys(unrelated).isdisjoint(row_keys(gate))


def test_gate_exclusion_matches_contract_and_exact_input_without_body_hash():
    gate = _row("gate", contract_address="0xabc")
    same_address = _row("other", contract_address="0xAbC")
    same_input = _row("gate", output="function other() external {}")
    selected, count = exclude_eval_rows([same_address, same_input, _row("clean")], [gate])
    assert selected == [_row("clean")]
    assert count == 2


def test_canonicalized_rows_keep_missing_hash_duplicates_together_in_split():
    import train

    duplicates = [
        _row("first", output="function shared() external { return; }"),
        _row(
            "other",
            output="FUNCTION shared() external { /* comment */ return; }",
            body_hash="stale",
        ),
    ]
    others = [_row(f"other{i}") for i in range(20)]
    selected = [canonicalize_body_hash(row) for row in duplicates + others]

    groups = train._grouped_split(selected, 0.6, 0.2, seed=0)
    locations = [
        next(index for index, group in enumerate(groups) if row in group) for row in selected[:2]
    ]
    assert locations[0] == locations[1]


def test_sample_cache_requires_source_gate_and_output_hashes(tmp_path):
    source = tmp_path / "source.jsonl"
    gate = tmp_path / "gate.jsonl"
    output = tmp_path / "sample.jsonl"
    _write_jsonl(source, [_row("gate"), _row("keep"), _row("extra")])
    _write_jsonl(gate, [_row("gate")])

    manifest = sample_dataset(source, output, 2, 42, [gate])
    assert manifest["excluded_source_rows"] == 1
    assert sample_dataset(source, output, 2, 42, [gate]) == manifest
    assert all(row["metadata"]["body_hash"] == body_identity(row) for row in load_jsonl(output))
    assert {row["input"] for row in load_jsonl(output)} == {"tac keep", "tac extra"}

    _write_jsonl(gate, [_row("keep")])
    with pytest.raises(ValueError, match="unverified or stale"):
        sample_dataset(source, output, 2, 42, [gate])
    _write_jsonl(source, [_row("gate"), _row("keep"), _row("extra"), _row("another")])
    with pytest.raises(ValueError, match="unverified or stale"):
        sample_dataset(source, output, 2, 42, [gate])


def test_gate_sources_fail_closed_on_missing_target_or_empty_gate(tmp_path):
    source = tmp_path / "source.jsonl"
    gate = tmp_path / "gate.jsonl"
    output = tmp_path / "sample.jsonl"
    _write_jsonl(source, [{"input": "tac", "metadata": {"body_hash": "stale"}}])
    _write_jsonl(gate, [_row("gate")])
    with pytest.raises(ValueError, match="no usable output"):
        sample_dataset(source, output, 1, 42, [gate])
    _write_jsonl(source, [_row("clean")])
    gate.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="has no rows"):
        sample_dataset(source, output, 1, 42, [gate])


def _write_model_lineage(tmp_path, splits):
    model = tmp_path / "model" / "final_model"
    model.mkdir(parents=True)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    paths = {}
    for name, rows in splits.items():
        path = split_dir / f"{name}_dataset.jsonl"
        _write_jsonl(path, rows)
        paths[name] = path
    (split_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "manifest_kind": "dataset_split",
                "leakage_validation": {"status": "passed"},
                "split_quality": {"status": "passed"},
                "outputs": {
                    name: {"path": str(path), "sha256": file_sha256(path)}
                    for name, path in paths.items()
                },
            }
        ),
        encoding="utf-8",
    )
    (model / "training_input_manifest.json").write_text(
        json.dumps(
            {
                "manifest_kind": "training_inputs",
                "status": "completed",
                "datasets": {
                    "train": {
                        "artifact": {
                            "path": str(paths["train"]),
                            "sha256": file_sha256(paths["train"]),
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return model, paths


def test_model_gate_check_rejects_eval_overlap_in_any_split(tmp_path):
    gate = tmp_path / "gate.jsonl"
    _write_jsonl(gate, [_row("heldout", body_hash="expected")])
    for leaking_split in ("train", "val", "test"):
        run_dir = tmp_path / leaking_split
        run_dir.mkdir()
        splits = {name: [_row(name)] for name in ("train", "val", "test")}
        splits[leaking_split] = [
            _row(
                "heldout",
                output="FUNCTION heldout() external { /* same */ return; }",
                body_hash="missing-or-stale",
            )
        ]
        model, _ = _write_model_lineage(run_dir, splits)
        with pytest.raises(ValueError, match=f"{leaking_split} split overlaps fixed"):
            verify_model_gate_exclusion(model, [gate])


def test_model_gate_check_rejects_unverifiable_and_cross_split_lineage(tmp_path):
    gate = tmp_path / "gate.jsonl"
    _write_jsonl(gate, [_row("heldout")])
    model, paths = _write_model_lineage(
        tmp_path,
        {
            "train": [_row("first")],
            "val": [_row("second")],
            "test": [_row("third")],
        },
    )
    assert verify_model_gate_exclusion(model, [gate]) == paths["train"]
    checked = subprocess.run(
        [sys.executable, "-m", "scripts.gate_dataset", "verify-model", str(model), str(gate)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert checked.returncode == 0, checked.stderr
    assert checked.stdout.strip() == str(paths["train"])

    _write_jsonl(paths["val"], [_row("first", output="function other() external {}")])
    with pytest.raises(ValueError, match="changed"):
        verify_model_gate_exclusion(model, [gate])
    split_manifest_path = paths["val"].parent / "split_manifest.json"
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["val"]["sha256"] = file_sha256(paths["val"])
    split_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps train split"):
        verify_model_gate_exclusion(model, [gate])


def test_model_gate_check_verifies_train_time_eval_artifact(tmp_path):
    gate = tmp_path / "gate.jsonl"
    _write_jsonl(gate, [_row("heldout")])
    model, paths = _write_model_lineage(
        tmp_path,
        {"train": [_row("first")], "val": [_row("second")], "test": [_row("third")]},
    )
    manifest_path = model / "training_input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["datasets"]["eval"] = {"artifact": {"path": str(gate), "sha256": file_sha256(gate)}}
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="eval dataset differs"):
        verify_model_gate_exclusion(model, [gate])

    manifest["datasets"]["eval"]["artifact"]["path"] = str(paths["val"])
    with pytest.raises(ValueError, match="eval artifact hash differs"):
        manifest_path.write_text(json.dumps(manifest))
        verify_model_gate_exclusion(model, [gate])

    manifest["datasets"]["eval"]["artifact"]["sha256"] = file_sha256(paths["val"])
    manifest_path.write_text(json.dumps(manifest))
    assert verify_model_gate_exclusion(model, [gate]) == paths["train"]


def test_eval_artifact_cannot_be_misattributed_to_a_different_model_or_gate(tmp_path):
    eval_path = tmp_path / "eval.json"
    model = tmp_path / "model"
    dataset = tmp_path / "gate.jsonl"
    eval_path.write_text(
        json.dumps(
            {
                "summary": {
                    "model_path": str(model),
                    "test_dataset": str(dataset),
                    "eval_max_new_tokens": 512,
                    "eval_repetition_penalty": 1.05,
                }
            }
        ),
        encoding="utf-8",
    )

    verify_eval_artifact(eval_path, model, dataset, 512, 1.05)
    with pytest.raises(ValueError, match="does not match requested"):
        verify_eval_artifact(eval_path, tmp_path / "other-model", dataset, 512, 1.05)
    with pytest.raises(ValueError, match="does not match requested"):
        verify_eval_artifact(eval_path, model, tmp_path / "other-gate", 512, 1.05)
    with pytest.raises(ValueError, match="does not match requested"):
        verify_eval_artifact(eval_path, model, dataset, 1024, 1.05)


def test_baseline_preflight_rejects_old_prompt_policy_and_missing_behavior_metric(tmp_path):
    gate = tmp_path / "gate.jsonl"
    baseline = tmp_path / "baseline.json"
    row = _row("gate")
    _write_jsonl(gate, [row])
    detail = {
        "dataset_index": 0,
        "input_hash": hashlib.sha256(row["input"].encode()).hexdigest(),
        "output_hash": hashlib.sha256(row["output"].encode()).hexdigest(),
        "metadata": row["metadata"],
    }
    summary = {
        "model_path": str(tmp_path / "model"),
        "test_dataset": str(gate),
        "eval_max_new_tokens": 512,
        "eval_repetition_penalty": 1.05,
        "num_evaluated": 1,
        "selector_signature_prompt_policy": "bundled_only_v1",
        "prompt_truncation_count": 0,
        "replication_f1_micro": 0.5,
        "replication_behavior_only_f1_micro": 0.4,
        "bytecode_semantic_score_mean": 0.3,
        "semantic_similarity_mean": 0.6,
        "solidity_valid_mean": 1.0,
    }
    baseline.write_text(json.dumps({"summary": summary, "details": [detail]}))
    verify_baseline_artifact(baseline, gate, 512, 1.05)

    summary["selector_signature_prompt_policy"] = "source_db"
    baseline.write_text(json.dumps({"summary": summary, "details": [detail]}))
    with pytest.raises(ValueError, match="bundled_only_v1"):
        verify_baseline_artifact(baseline, gate, 512, 1.05)

    summary["selector_signature_prompt_policy"] = "bundled_only_v1"
    del summary["replication_behavior_only_f1_micro"]
    baseline.write_text(json.dumps({"summary": summary, "details": [detail]}))
    with pytest.raises(ValueError, match="replication_behavior_only_f1_micro"):
        verify_baseline_artifact(baseline, gate, 512, 1.05)

    summary["replication_behavior_only_f1_micro"] = 0.4
    summary["num_evaluated"] = 2
    baseline.write_text(json.dumps({"summary": summary, "details": [detail]}))
    with pytest.raises(ValueError, match="every fixed-gate row"):
        verify_baseline_artifact(baseline, gate, 512, 1.05)

    summary["num_evaluated"] = 1
    summary["prompt_truncation_count"] = 1
    baseline.write_text(json.dumps({"summary": summary, "details": [detail]}))
    with pytest.raises(ValueError, match="prompt truncation"):
        verify_baseline_artifact(baseline, gate, 512, 1.05)

    summary["prompt_truncation_count"] = 0
    baseline.write_text(json.dumps({"summary": summary, "details": [detail]}))
    _write_jsonl(gate, [_row("changed")])
    with pytest.raises(ValueError, match="differs from current gate dataset"):
        verify_baseline_artifact(baseline, gate, 512, 1.05)


def test_full_runner_rejects_missing_baseline_before_materializing_training_data(tmp_path):
    source = tmp_path / "source.jsonl"
    gate = tmp_path / "gate.jsonl"
    _write_jsonl(source, [_row("train"), _row("other")])
    _write_jsonl(gate, [_row("gate")])
    data_dir = tmp_path / "training"
    output_dir = tmp_path / "model"
    env = {
        **os.environ,
        "SOURCE_DATASET": str(source),
        "DATA_DIR": str(data_dir),
        "OUTPUT_DIR": str(output_dir),
        "RUN_GATES": "1",
        "DRY_RUN": "0",
        "BROAD_BASELINE": str(tmp_path / "missing-baseline.json"),
        **{key: str(gate) for key in GATE_VARS},
    }
    result = subprocess.run(
        ["bash", str(ROOT / "run_train_qwen_qlora_full_body_balanced.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Required broad30 baseline eval not found" in result.stderr
    assert not data_dir.exists()
    assert not output_dir.exists()


def test_standalone_gate_runner_rejects_model_without_provenance_before_eval(tmp_path):
    gate = tmp_path / "gate.jsonl"
    model = tmp_path / "model"
    model.mkdir()
    _write_jsonl(gate, [_row("gate")])
    env = {
        **os.environ,
        **{name: str(gate) for name in GATE_VARS + BASELINE_VARS},
        "MODEL_PATH": str(model),
        "GATE_DIR": str(tmp_path / "gates"),
    }
    result = subprocess.run(
        ["bash", str(ROOT / "run_eval_gate_suite_for_model.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "Cannot verify eval-clean model lineage" in result.stderr
    assert not (tmp_path / "gates").exists()


@pytest.mark.parametrize(
    "runner,output_name",
    [
        ("run_train_qwen_qlora_500.sh", "qwen_qlora_2_sample.jsonl"),
        ("run_train_qwen_qlora_full_body_balanced.sh", "body_balanced_dataset.jsonl"),
    ],
)
def test_runner_dry_run_excludes_fixed_gates_and_rejects_stale_cache(tmp_path, runner, output_name):
    source = tmp_path / "source.jsonl"
    gate = tmp_path / "gate.jsonl"
    data_dir = tmp_path / "run"
    _write_jsonl(source, [_row("gate"), _row("clean1"), _row("clean2")])
    _write_jsonl(gate, [_row("gate")])
    env = {
        **os.environ,
        "SOURCE_DATASET": str(source),
        "DATA_DIR": str(data_dir),
        "OUTPUT_DIR": str(tmp_path / "model"),
        "SAMPLE_COUNT": "2",
        "DRY_RUN": "1",
        **{key: str(gate) for key in GATE_VARS},
    }
    command = ["bash", str(ROOT / runner)]

    built = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    assert built.returncode == 0, built.stdout + built.stderr
    rows = load_jsonl(data_dir / output_name)
    assert len(rows) == 2
    assert {row["input"] for row in rows} == {"tac clean1", "tac clean2"}

    reused = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    assert reused.returncode == 0, reused.stdout + reused.stderr
    _write_jsonl(gate, [_row("clean1")])
    stale = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    assert stale.returncode != 0
    assert "unverified or stale" in stale.stderr


def test_body_balanced_runner_caps_duplicate_targets_without_body_hash(tmp_path):
    source = tmp_path / "source.jsonl"
    gate = tmp_path / "gate.jsonl"
    data_dir = tmp_path / "run"
    duplicate = _row(
        "one",
        output="FUNCTION shared() external { /* same target */ return; }",
        body_hash="stale",
    )
    _write_jsonl(
        source,
        [
            _row("first", output="function shared() external { return; }"),
            duplicate,
            _row("distinct"),
            _row("also_distinct"),
        ],
    )
    _write_jsonl(gate, [_row("heldout")])
    env = {
        **os.environ,
        "SOURCE_DATASET": str(source),
        "DATA_DIR": str(data_dir),
        "OUTPUT_DIR": str(tmp_path / "model"),
        "DRY_RUN": "1",
        **{key: str(gate) for key in GATE_VARS},
    }

    result = subprocess.run(
        ["bash", str(ROOT / "run_train_qwen_qlora_full_body_balanced.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    selected = load_jsonl(data_dir / "body_balanced_dataset.jsonl")
    assert len(selected) == 3
    assert len({body_identity(row) for row in selected}) == 3
    assert all(row["metadata"]["body_hash"] == body_identity(row) for row in selected)
    manifest = json.loads((data_dir / "body_balanced_manifest.json").read_text())
    assert manifest["source_body_duplicate_rows"] == 1
    assert manifest["missing_body_hash_source_rows"] == 3


def test_sample_runner_repairs_duplicate_body_hashes_before_grouped_split(tmp_path):
    source = tmp_path / "source.jsonl"
    gate = tmp_path / "gate.jsonl"
    data_dir = tmp_path / "run"
    _write_jsonl(
        source,
        [
            _row("first", output="function shared() external { return; }"),
            _row(
                "other",
                output="FUNCTION shared() external { /* duplicate */ return; }",
                body_hash="stale",
            ),
            _row("distinct"),
        ],
    )
    _write_jsonl(gate, [_row("heldout")])
    env = {
        **os.environ,
        "SOURCE_DATASET": str(source),
        "DATA_DIR": str(data_dir),
        "OUTPUT_DIR": str(tmp_path / "model"),
        "SAMPLE_COUNT": "3",
        "DRY_RUN": "1",
        **{key: str(gate) for key in GATE_VARS},
    }

    result = subprocess.run(
        ["bash", str(ROOT / "run_train_qwen_qlora_500.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    selected = load_jsonl(data_dir / "qwen_qlora_3_sample.jsonl")
    assert len(selected) == 3
    assert len({row["metadata"]["body_hash"] for row in selected}) == 2
    assert all(row["metadata"]["body_hash"] == body_identity(row) for row in selected)
