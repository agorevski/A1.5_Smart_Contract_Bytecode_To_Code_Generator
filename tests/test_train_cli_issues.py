import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

import train


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class TinyTokenizer:
    name_or_path = "tiny-test-tokenizer"

    def __call__(self, text, **_kwargs):
        return {"input_ids": str(text).split()}

    def encode(self, text, add_special_tokens=False):
        return str(text).split()


def test_collect_dataset_does_not_pass_unsupported_max_workers(tmp_path, monkeypatch):
    import src.dataset_pipeline as dataset_pipeline

    captured = {}

    class FakeBuilder:
        def __init__(self, api_key, output_dir):
            captured["api_key"] = api_key
            captured["output_dir"] = output_dir

        def collect_and_compile_contracts(self, addresses, max_compiler_configs=2):
            captured["addresses"] = addresses
            captured["max_compiler_configs"] = max_compiler_configs
            return 1

        def filter_and_clean_dataset(self, min_length, max_length):
            captured["filter"] = (min_length, max_length)
            return 1

        def export_dataset(self, fmt):
            dataset_path = tmp_path / f"dataset.{fmt}"
            _write_jsonl(dataset_path, [{"input": "tac", "output": "sol"}])
            return str(dataset_path)

        def get_dataset_statistics(self):
            return {"rows": 1}

    monkeypatch.setattr(dataset_pipeline, "DatasetBuilder", FakeBuilder)
    monkeypatch.setattr(train, "load_contract_addresses", lambda _path: ["0x1", "0x2"])

    dataset_path = train.collect_dataset(
        "api-key",
        "addresses.txt",
        output_dir=str(tmp_path),
        max_compiler_configs=4,
        max_workers=9,
    )

    assert Path(dataset_path).exists()
    assert captured["addresses"] == ["0x1", "0x2"]
    assert captured["max_compiler_configs"] == 4


def test_collect_dataset_passes_configurable_max_workers_when_supported(tmp_path, monkeypatch):
    import src.dataset_pipeline as dataset_pipeline

    captured = {}

    class FakeBuilder:
        def __init__(self, _api_key, output_dir):
            pass

        def collect_and_compile_contracts(
            self, addresses, max_workers=3, max_compiler_configs=2
        ):
            captured["addresses"] = addresses
            captured["max_workers"] = max_workers
            captured["max_compiler_configs"] = max_compiler_configs
            return 1

        def filter_and_clean_dataset(self, min_length, max_length):
            return 1

        def export_dataset(self, fmt):
            dataset_path = tmp_path / f"dataset.{fmt}"
            _write_jsonl(dataset_path, [{"input": "tac", "output": "sol"}])
            return str(dataset_path)

        def get_dataset_statistics(self):
            return {"rows": 1}

    monkeypatch.setattr(dataset_pipeline, "DatasetBuilder", FakeBuilder)
    monkeypatch.setattr(train, "load_contract_addresses", lambda _path: ["0xabc"])

    train.collect_dataset(
        "api-key",
        "addresses.txt",
        output_dir=str(tmp_path),
        max_compiler_configs=5,
        max_workers=7,
    )

    assert captured == {
        "addresses": ["0xabc"],
        "max_workers": 7,
        "max_compiler_configs": 5,
    }


def test_collect_dataset_fails_fast_without_explicit_demo_fallback(tmp_path, monkeypatch):
    import src.dataset_pipeline as dataset_pipeline

    class FakeBuilder:
        def __init__(self, _api_key, output_dir=None):
            pass

        def collect_and_compile_contracts(self, _addresses, **_kwargs):
            return 0

    monkeypatch.setattr(dataset_pipeline, "DatasetBuilder", FakeBuilder)
    monkeypatch.setattr(train, "load_contract_addresses", lambda _path: ["0xabc"])

    with pytest.raises(RuntimeError, match="allow-demo-fallback"):
        train.collect_dataset("api-key", "addresses.txt", output_dir=str(tmp_path))


def test_collect_dataset_demo_fallback_requires_flag_and_writes_manifest(
    tmp_path, monkeypatch
):
    import src.dataset_pipeline as dataset_pipeline

    class FakeBuilder:
        def __init__(self, _api_key, output_dir=None):
            pass

        def collect_and_compile_contracts(self, _addresses, **_kwargs):
            return 0

    monkeypatch.setattr(dataset_pipeline, "DatasetBuilder", FakeBuilder)
    monkeypatch.setattr(train, "load_contract_addresses", lambda _path: ["0xabc"])

    dataset_path = train.collect_dataset(
        "api-key",
        "addresses.txt",
        output_dir=str(tmp_path),
        allow_demo_fallback=True,
    )

    manifest = json.loads(Path(f"{dataset_path}.manifest.json").read_text())
    assert Path(dataset_path).exists()
    assert manifest["demo_fallback"] is True
    assert manifest["reason"] == "zero_function_pairs"


def test_split_manifest_gates_leakage_and_reports_holdout_coverage(tmp_path):
    rows = [
        {
            "input": "tac shared compiler 0.8.20",
            "output": "function shared() public { return; }",
            "metadata": {
                "id": "shared-a",
                "contract_address": "0xShared",
                "function_signature": "shared()",
                "selector": "0x11111111",
                "compiler_version": "0.8.20",
                "optimizer_enabled": False,
                "visibility": "public",
                "source": "unit",
            },
        },
        {
            "input": "tac shared compiler 0.8.21",
            "output": "function shared() public { return; }",
            "metadata": {
                "id": "shared-b",
                "contract_address": "0xShared",
                "function_signature": "shared()",
                "selector": "0x11111111",
                "compiler_version": "0.8.21",
                "optimizer_enabled": True,
                "visibility": "public",
                "source": "unit",
            },
        },
        {
            "input": "tac duplicate one",
            "output": "function duplicateBody() external { return; }",
            "metadata": {
                "id": "duplicate-a",
                "contract_address": "0xDupA",
                "function_signature": "duplicateBody()",
                "compiler_version": "0.8.20",
                "optimizer_enabled": False,
                "visibility": "external",
                "source": "unit",
            },
        },
        {
            "input": "tac duplicate two",
            "output": "function duplicateBody() external { return; }",
            "metadata": {
                "id": "duplicate-b",
                "contract_address": "0xDupB",
                "function_signature": "duplicateBody()",
                "compiler_version": "0.8.20",
                "optimizer_enabled": True,
                "visibility": "external",
                "source": "unit",
            },
        },
    ]
    for i in range(8):
        rows.append(
            {
                "input": f"tac unique {i}",
                "output": f"function unique{i}() public {{ return; }}",
                "metadata": {
                    "id": f"unique-{i}",
                    "contract_address": f"0x{i:040x}",
                    "function_signature": f"unique{i}()",
                    "compiler_version": "0.8.19" if i % 2 else "0.8.20",
                    "optimizer_enabled": bool(i % 2),
                    "visibility": "public" if i % 2 else "external",
                    "source": "unit",
                },
            }
        )

    dataset_path = tmp_path / "dataset.jsonl"
    split_dir = tmp_path / "splits"
    _write_jsonl(dataset_path, rows)

    train_path, val_path, test_path = train.split_dataset(
        str(dataset_path),
        str(split_dir),
        train_ratio=0.6,
        val_ratio=0.2,
        seed=123,
    )

    locations = {}
    duplicate_locations = []
    for split_name, path in {
        "train": train_path,
        "val": val_path,
        "test": test_path,
    }.items():
        for row in [json.loads(line) for line in Path(path).read_text().splitlines() if line]:
            row_id = row["metadata"]["id"]
            locations[row_id] = split_name
            if row["output"] == "function duplicateBody() external { return; }":
                duplicate_locations.append(split_name)

    assert locations["shared-a"] == locations["shared-b"]
    assert len(set(duplicate_locations)) == 1

    manifest = json.loads((split_dir / "split_manifest.json").read_text())
    assert manifest["parameters"]["seed"] == 123
    assert manifest["parameters"]["group_key_precedence"][0] == "source_hash"
    assert manifest["row_counts"]["source"] == len(rows)
    assert manifest["leakage_validation"]["status"] == "passed"
    assert manifest["coverage"]["fields"] == list(train.COVERAGE_FIELDS)
    assert "0.8.20" in manifest["coverage"]["total"]["fields"]["compiler_version"]


def test_validate_split_leakage_detects_manual_overlap():
    left = {
        "input": "same tac",
        "output": "left",
        "metadata": {"contract_address": "0xabc", "function_signature": "foo()"},
    }
    right = {
        "input": "same tac",
        "output": "right",
        "metadata": {"contract_address": "0xAbC", "function_signature": "foo()"},
    }

    result = train.validate_split_leakage({"train": [left], "val": [right], "test": []})

    assert result["status"] == "failed"
    assert result["overlap_counts"]["contract_address"] == 1
    assert result["overlap_counts"]["contract_function"] >= 1
    assert result["overlap_counts"]["input_hash"] == 1


def test_jsonl_preflight_reports_schema_and_token_length_errors(tmp_path):
    dataset_path = tmp_path / "bad.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"input": "ok", "output": "ok", "metadata": {}}),
                "{not valid json}",
                json.dumps({"output": "missing input", "metadata": {}}),
                json.dumps({"input": "has bad metadata", "output": "ok", "metadata": []}),
                json.dumps({"input": "has empty output", "output": "   ", "metadata": {}}),
                json.dumps(
                    {
                        "input": " ".join(["context"] * 40),
                        "output": "ok",
                        "metadata": {},
                    }
                ),
                json.dumps(
                    {
                        "input": "ok",
                        "output": " ".join(["target"] * 40),
                        "metadata": {},
                    }
                ),
            ]
        )
        + "\n"
    )

    report = train.validate_jsonl_schema_and_lengths(
        dataset_path,
        tokenizer=TinyTokenizer(),
        max_seq_length=32,
    )

    assert report["status"] == "failed"
    assert report["row_count"] == 7
    assert report["error_counts"]["json_parse_error"] == 1
    assert report["error_counts"]["missing_input"] == 1
    assert report["error_counts"]["metadata_type_error"] == 1
    assert report["error_counts"]["empty_output"] == 1
    assert report["error_counts"]["context_overlength"] >= 1
    assert report["error_counts"]["target_overlength"] == 1


@dataclass
class FakeMetrics:
    semantic_similarity: float = 1.0
    normalized_edit_distance: float = 0.0


class FakeEvaluator:
    def evaluate_function(self, _expected, _actual, _metadata):
        return FakeMetrics()


def _patch_evaluation_dependencies(monkeypatch, decompiler_cls):
    import src.evaluation_report as evaluation_report
    import src.model_setup as model_setup
    import src.training_pipeline as training_pipeline

    monkeypatch.setattr(model_setup, "SmartContractDecompiler", decompiler_cls)
    monkeypatch.setattr(training_pipeline, "SmartContractEvaluator", FakeEvaluator)

    def fake_latest_report(**kwargs):
        latest_path = Path(kwargs["latest_results_path"])
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text("latest")
        return str(latest_path)

    monkeypatch.setattr(evaluation_report, "write_latest_results_report", fake_latest_report)


def test_evaluate_model_uses_decompile_batch_chunks(tmp_path, monkeypatch):
    class FakeDecompiler:
        batch_calls = []
        single_calls = []

        def __init__(self, model_path):
            self.model_path = model_path

        def decompile_batch(self, tac_inputs, metadatas=None, max_new_tokens=1024):
            self.batch_calls.append((list(tac_inputs), list(metadatas), max_new_tokens))
            return [f"sol:{tac}" for tac in tac_inputs]

        def decompile_tac_to_solidity(self, tac, metadata=None, max_new_tokens=1024):
            self.single_calls.append((tac, metadata, max_new_tokens))
            return f"sol:{tac}"

    _patch_evaluation_dependencies(monkeypatch, FakeDecompiler)
    dataset_path = tmp_path / "test.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {"input": "tac0", "output": "sol0", "metadata": {"i": 0}},
            {"input": "tac1", "output": "sol1", "metadata": {"i": 1}},
            {"input": "tac2", "output": "sol2", "metadata": {"i": 2}},
        ],
    )

    summary = train.evaluate_model(
        "fake-model",
        str(dataset_path),
        results_dir=str(tmp_path / "results"),
        latest_results_path=str(tmp_path / "latest.txt"),
        eval_batch_size=2,
    )

    assert [call[0] for call in FakeDecompiler.batch_calls] == [["tac0", "tac1"], ["tac2"]]
    assert FakeDecompiler.single_calls == []
    assert summary["eval_batch_size"] == 2
    assert summary["num_evaluated"] == 3
    assert Path(summary["results_path"]).exists()


def test_evaluate_model_batch_oom_falls_back_to_single_examples(tmp_path, monkeypatch):
    class OOMDecompiler:
        batch_calls = []
        single_calls = []

        def __init__(self, model_path):
            self.model_path = model_path

        def decompile_batch(self, tac_inputs, metadatas=None, max_new_tokens=1024):
            self.batch_calls.append(list(tac_inputs))
            raise RuntimeError("CUDA out of memory")

        def decompile_tac_to_solidity(self, tac, metadata=None, max_new_tokens=1024):
            self.single_calls.append((tac, metadata, max_new_tokens))
            return f"single:{tac}"

    _patch_evaluation_dependencies(monkeypatch, OOMDecompiler)
    dataset_path = tmp_path / "test.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {"input": "tac0", "output": "sol0", "metadata": {"i": 0}},
            {"input": "tac1", "output": "sol1", "metadata": {"i": 1}},
        ],
    )

    summary = train.evaluate_model(
        "fake-model",
        str(dataset_path),
        results_dir=str(tmp_path / "results"),
        latest_results_path=str(tmp_path / "latest.txt"),
        eval_batch_size=2,
    )

    assert OOMDecompiler.batch_calls == [["tac0", "tac1"]]
    assert [call[0] for call in OOMDecompiler.single_calls] == ["tac0", "tac1"]
    assert summary["num_evaluated"] == 2


def test_evaluate_model_records_failed_rows_and_traceable_details(tmp_path, monkeypatch):
    class PartiallyFailingDecompiler:
        def __init__(self, model_path):
            self.model_path = model_path

        def decompile_tac_to_solidity(self, tac, metadata=None, max_new_tokens=1024):
            if tac == "bad tac":
                raise RuntimeError("generation failed")
            return f"sol:{tac}"

    _patch_evaluation_dependencies(monkeypatch, PartiallyFailingDecompiler)
    dataset_path = tmp_path / "test.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "input": "good tac",
                "output": "sol good",
                "metadata": {"function_name": "good"},
            },
            {
                "input": "bad tac",
                "output": "sol bad",
                "metadata": {"function_name": "bad"},
            },
        ],
    )

    summary = train.evaluate_model(
        "fake-model",
        str(dataset_path),
        results_dir=str(tmp_path / "results"),
        latest_results_path=str(tmp_path / "latest.txt"),
    )
    payload = json.loads(Path(summary["results_path"]).read_text())

    assert summary["num_attempted"] == 2
    assert summary["num_succeeded"] == 1
    assert summary["num_failed"] == 1
    assert summary["failure_rate"] == 0.5
    assert len(payload["details"]) == 2

    failed = [row for row in payload["details"] if not row["success"]][0]
    assert failed["dataset_index"] == 1
    assert failed["input_hash"]
    assert failed["output_hash"]
    assert failed["metadata"]["function_name"] == "bad"
    assert failed["metrics"]["semantic_similarity"] == 0.0
    assert failed["metrics"]["normalized_edit_distance"] == 1.0
    assert failed["error"]["type"] == "RuntimeError"
    assert summary["worst_samples"]["failed"][0]["dataset_index"] == 1


def test_resolve_resume_checkpoint_auto_selects_latest_numeric_checkpoint(tmp_path):
    output_dir = tmp_path / "models"
    (output_dir / "checkpoint-3").mkdir(parents=True)
    (output_dir / "checkpoint-12").mkdir()
    (output_dir / "not-a-checkpoint").mkdir()
    (output_dir / "checkpoint-3" / "trainer_state.json").write_text("{}")
    (output_dir / "checkpoint-12" / "trainer_state.json").write_text("{}")

    assert train.resolve_resume_checkpoint("auto", str(output_dir)) == str(
        output_dir / "checkpoint-12"
    )
    assert train.resolve_resume_checkpoint("checkpoint-custom", str(output_dir)) == "checkpoint-custom"
    assert train.resolve_resume_checkpoint("auto", str(tmp_path / "empty")) is None


def test_main_persists_run_manifest_with_resume_dataset_and_training_refs(
    tmp_path, monkeypatch
):
    rows = [
        {
            "input": f"tac {i}",
            "output": f"function f{i}() public {{}}",
            "metadata": {"contract_address": f"0x{i}", "function_signature": f"f{i}()"},
        }
        for i in range(6)
    ]
    dataset_path = tmp_path / "source.jsonl"
    _write_jsonl(dataset_path, rows)

    output_dir = tmp_path / "models"
    checkpoint = output_dir / "checkpoint-20"
    older_checkpoint = output_dir / "checkpoint-7"
    checkpoint.mkdir(parents=True)
    older_checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text("{}")
    (older_checkpoint / "trainer_state.json").write_text("{}")
    manifest_path = tmp_path / "run.manifest.json"
    captured_train_kwargs = {}

    def fake_train_model(**kwargs):
        captured_train_kwargs.update(kwargs)
        final_model = Path(kwargs["output_dir"]) / "final_model"
        final_model.mkdir(parents=True, exist_ok=True)
        (final_model / "training_metrics.json").write_text(
            json.dumps({"train_loss": 0.25, "train_runtime": 1.5})
        )
        (checkpoint / "trainer_state.json").write_text(
            json.dumps(
                {
                    "global_step": 20,
                    "best_metric": 0.25,
                    "log_history": [{"loss": 0.25, "step": 20}],
                }
            )
        )
        return str(final_model)

    monkeypatch.setattr(train, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(train, "train_model", fake_train_model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--skip-collection",
            "--dataset",
            str(dataset_path),
            "--data-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(output_dir),
            "--resume",
            "auto",
            "--skip-eval",
            "--run-manifest",
            str(manifest_path),
            "--epochs",
            "1",
            "--batch-size",
            "1",
        ],
    )

    train.main()

    manifest = json.loads(manifest_path.read_text())
    assert captured_train_kwargs["resume_from"] == str(checkpoint)
    assert manifest["status"] == "completed"
    assert manifest["training"]["config"]["resume"] == "auto"
    assert manifest["training"]["config"]["resume_from_checkpoint"] == str(checkpoint)
    assert manifest["datasets"]["source"]["row_count"] == len(rows)
    assert manifest["datasets"]["splits"]["train"]["sha256"]
    assert manifest["training"]["final_metrics"]["train_loss"] == 0.25
    assert manifest["training"]["log_history_references"][0]["log_history_count"] == 1
    assert manifest["evaluation"] == {"skipped": True}
