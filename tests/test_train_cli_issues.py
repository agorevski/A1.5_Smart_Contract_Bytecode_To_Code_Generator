import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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

        def collect_and_compile_contracts(self, addresses, max_workers=3, max_compiler_configs=2):
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


def test_collect_dataset_demo_fallback_requires_flag_and_writes_manifest(tmp_path, monkeypatch):
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
                json.dumps({"input": "ok", "output": "ok", "metadata": {"schema_version": 1}}),
                "{not valid json}",
                json.dumps({"output": "missing input", "metadata": {"schema_version": 1}}),
                json.dumps({"input": "has bad metadata", "output": "ok", "metadata": []}),
                json.dumps(
                    {
                        "input": "has empty output",
                        "output": "   ",
                        "metadata": {"schema_version": 1},
                    }
                ),
                json.dumps(
                    {
                        "input": " ".join(["context"] * 40),
                        "output": "ok",
                        "metadata": {"schema_version": 1},
                    }
                ),
                json.dumps(
                    {
                        "input": "ok",
                        "output": " ".join(["target"] * 40),
                        "metadata": {"schema_version": 1},
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


def test_jsonl_preflight_rejects_malformed_versioned_metadata(tmp_path):
    dataset_path = tmp_path / "bad_metadata.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "input": "tac",
                "output": "sol",
                "metadata": {
                    "schema_version": 1,
                    "contract_address": "0xnot-an-address",
                    "selector": "0x123",
                    "source_hash": "not-hex",
                    "optimizer_enabled": "true",
                    "compiler_version": "solc-nightly",
                    "optimizer_runs": True,
                },
            }
        ],
    )

    report = train.validate_jsonl_schema_and_lengths(
        dataset_path,
        tokenizer=TinyTokenizer(),
        max_seq_length=64,
    )

    assert report["status"] == "failed"
    assert report["metadata_schema"]["allow_legacy"] is False
    assert report["error_counts"]["contract_address_format"] == 1
    assert report["error_counts"]["selector_format"] == 1
    assert report["error_counts"]["hash_format"] == 1
    assert report["error_counts"]["boolean_type_error"] == 1
    assert report["error_counts"]["compiler_version_format"] == 1
    assert report["error_counts"]["optimizer_runs_type_error"] == 1


def test_jsonl_preflight_legacy_schema_requires_explicit_compatibility(tmp_path):
    dataset_path = tmp_path / "legacy.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "input": "tac",
                "output": "sol",
                "metadata": {
                    "contract_address": "0x0000000000000000000000000000000000000001",
                    "selector": "0x12345678",
                    "optimizer_enabled": True,
                    "compiler_version": "0.8.20",
                },
            }
        ],
    )

    strict = train.validate_jsonl_schema_and_lengths(
        dataset_path,
        tokenizer=TinyTokenizer(),
        max_seq_length=64,
    )
    legacy = train.validate_jsonl_schema_and_lengths(
        dataset_path,
        tokenizer=TinyTokenizer(),
        max_seq_length=64,
        allow_legacy_metadata_schema=True,
    )

    assert strict["status"] == "failed"
    assert strict["error_counts"]["schema_version_missing"] == 1
    assert legacy["status"] == "passed"
    assert legacy["metadata_schema"]["allow_legacy"] is True


@dataclass
class FakeMetrics:
    semantic_similarity: float = 1.0
    normalized_edit_distance: float = 0.0
    replication_precision: float = 1.0
    replication_recall: float = 1.0
    replication_f1: float = 1.0
    solidity_valid: bool = True
    solidity_compiler_checked: bool = True
    solidity_ast_valid: bool = True


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

        def decompile_batch(
            self,
            tac_inputs,
            metadatas=None,
            max_new_tokens=1024,
            repetition_penalty=1.15,
        ):
            self.batch_calls.append(
                (list(tac_inputs), list(metadatas), max_new_tokens, repetition_penalty)
            )
            return [f"sol:{tac}" for tac in tac_inputs]

        def decompile_tac_to_solidity(
            self,
            tac,
            metadata=None,
            max_new_tokens=1024,
            repetition_penalty=1.15,
        ):
            self.single_calls.append((tac, metadata, max_new_tokens, repetition_penalty))
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
        eval_max_new_tokens=77,
        eval_repetition_penalty=1.05,
    )

    assert [call[0] for call in FakeDecompiler.batch_calls] == [["tac0", "tac1"], ["tac2"]]
    assert [call[2] for call in FakeDecompiler.batch_calls] == [77, 77]
    assert [call[3] for call in FakeDecompiler.batch_calls] == [1.05, 1.05]
    assert FakeDecompiler.single_calls == []
    assert summary["eval_batch_size"] == 2
    assert summary["eval_max_new_tokens"] == 77
    assert summary["eval_repetition_penalty"] == 1.05
    assert summary["num_evaluated"] == 3
    assert Path(summary["results_path"]).exists()


def test_evaluate_model_batch_oom_falls_back_to_single_examples(tmp_path, monkeypatch):
    class OOMDecompiler:
        batch_calls = []
        single_calls = []

        def __init__(self, model_path):
            self.model_path = model_path

        def decompile_batch(
            self,
            tac_inputs,
            metadatas=None,
            max_new_tokens=1024,
            repetition_penalty=1.15,
        ):
            self.batch_calls.append(list(tac_inputs))
            raise RuntimeError("CUDA out of memory")

        def decompile_tac_to_solidity(
            self,
            tac,
            metadata=None,
            max_new_tokens=1024,
            repetition_penalty=1.15,
        ):
            self.single_calls.append((tac, metadata, max_new_tokens, repetition_penalty))
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

        def decompile_tac_to_solidity(
            self,
            tac,
            metadata=None,
            max_new_tokens=1024,
            repetition_penalty=1.15,
        ):
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


def test_evaluate_model_persists_prompt_truncation_diagnostics(tmp_path, monkeypatch):
    class DiagnosticDecompiler:
        def __init__(self, model_path):
            self.model_path = model_path

        def decompile_tac_to_solidity(
            self,
            tac,
            metadata=None,
            max_new_tokens=1024,
            repetition_penalty=1.15,
        ):
            return "generated solidity"

        def prompt_diagnostics(
            self,
            tac,
            metadata=None,
            max_new_tokens=1024,
            generated_text=None,
        ):
            return {
                "context_window": 32,
                "prompt_budget": 16,
                "max_new_tokens": max_new_tokens,
                "tac_token_budget": 8,
                "tac_tokens_before": len(str(tac).split()),
                "tac_tokens_after": 6,
                "prompt_tokens": 14,
                "generated_tokens": (
                    len(str(generated_text).split()) if generated_text is not None else None
                ),
                "tac_truncated": True,
                "strategy": "hard_truncate",
                "marker": "// ... truncated",
            }

    _patch_evaluation_dependencies(monkeypatch, DiagnosticDecompiler)
    dataset_path = tmp_path / "test.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "input": " ".join(f"OP_{i}" for i in range(50)),
                "output": "sol",
                "metadata": {"schema_version": 1},
            }
        ],
    )

    summary = train.evaluate_model(
        "fake-model",
        str(dataset_path),
        results_dir=str(tmp_path / "results"),
        latest_results_path=str(tmp_path / "latest.txt"),
    )
    payload = json.loads(Path(summary["results_path"]).read_text())
    detail = payload["details"][0]

    assert detail["prompt_diagnostics"]["tac_truncated"] is True
    assert detail["prompt_diagnostics"]["strategy"] == "hard_truncate"
    assert detail["prompt_diagnostics"]["generated_tokens"] == 2
    assert summary["prompt_truncation_count"] == 1
    assert summary["prompt_truncation_rate"] == 1.0
    assert summary["prompt_diagnostics"]["tac_tokens_before"]["max"] == 50
    assert (
        summary["worst_samples"]["truncated_low_quality"][0]["prompt_diagnostics"]["tac_truncated"]
        is True
    )


def test_evaluate_model_uses_seeded_limit_and_full_aggregation(tmp_path, monkeypatch):
    class FakeDecompiler:
        def __init__(self, model_path):
            self.model_path = model_path

        def decompile_tac_to_solidity(
            self,
            tac,
            metadata=None,
            max_new_tokens=1024,
            repetition_penalty=1.15,
        ):
            return f"sol:{tac}"

    _patch_evaluation_dependencies(monkeypatch, FakeDecompiler)
    dataset_path = tmp_path / "test.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "input": f"tac{i}",
                "output": f"sol{i}",
                "metadata": {
                    "compiler_version": "0.8.20" if i % 2 else "0.8.19",
                    "optimizer_enabled": bool(i % 2),
                },
            }
            for i in range(20)
        ],
    )

    summary_a = train.evaluate_model(
        "fake-model",
        str(dataset_path),
        results_dir=str(tmp_path / "results-a"),
        latest_results_path=str(tmp_path / "latest-a.txt"),
        eval_limit=5,
        eval_seed=123,
    )
    summary_b = train.evaluate_model(
        "fake-model",
        str(dataset_path),
        results_dir=str(tmp_path / "results-b"),
        latest_results_path=str(tmp_path / "latest-b.txt"),
        eval_limit=5,
        eval_seed=123,
    )
    summary_c = train.evaluate_model(
        "fake-model",
        str(dataset_path),
        results_dir=str(tmp_path / "results-c"),
        latest_results_path=str(tmp_path / "latest-c.txt"),
        eval_limit=5,
        eval_seed=124,
    )

    assert summary_a["eval_sampling_strategy"] == "seeded_sample"
    assert summary_a["eval_sample_indices"] == summary_b["eval_sample_indices"]
    assert summary_a["eval_sample_indices"] != summary_c["eval_sample_indices"]
    assert summary_a["solidity_valid_mean"] == 1.0
    assert summary_a["solidity_compiler_checked_mean"] == 1.0
    assert summary_a["solidity_ast_valid_mean"] == 1.0
    assert "semantic_similarity_mean" in summary_a["confidence_intervals"]
    assert "metadata_segments" in summary_a
    payload = json.loads(Path(summary_a["results_path"]).read_text())
    assert payload["summary"]["eval_seed"] == 123
    assert [row["dataset_index"] for row in payload["details"]] == summary_a["eval_sample_indices"]


def test_evaluate_model_baseline_comparison_and_quality_gate(tmp_path, monkeypatch):
    class FakeDecompiler:
        def __init__(self, model_path):
            self.model_path = model_path

        def decompile_tac_to_solidity(
            self,
            tac,
            metadata=None,
            max_new_tokens=1024,
            repetition_penalty=1.15,
        ):
            return f"sol:{tac}"

    _patch_evaluation_dependencies(monkeypatch, FakeDecompiler)
    dataset_path = tmp_path / "test.jsonl"
    _write_jsonl(
        dataset_path,
        [{"input": "tac", "output": "sol", "metadata": {"compiler_version": "0.8.20"}}],
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "summary": {
                    "semantic_similarity_mean": 0.5,
                    "edit_distance_mean": 0.5,
                    "failure_rate": 0.0,
                }
            }
        )
    )

    summary = train.evaluate_model(
        "fake-model",
        str(dataset_path),
        results_dir=str(tmp_path / "results"),
        latest_results_path=str(tmp_path / "latest.txt"),
        baseline_results_path=str(baseline_path),
        quality_gate_config={
            "enabled": True,
            "thresholds": {
                "semantic_similarity_mean": {"op": ">=", "value": 1.1, "required": True},
                "failure_rate": {"op": "<=", "value": 0.0, "required": True},
            },
            "max_baseline_regressions": 0,
        },
    )

    comparison = summary["baseline_comparison"]["comparisons"]
    assert comparison["semantic_similarity_mean"]["status"] == "improved"
    assert comparison["edit_distance_mean"]["status"] == "improved"
    assert summary["quality_gate"]["status"] == "failed"
    assert summary["quality_gate"]["failures"][0]["metric"] == "semantic_similarity_mean"


def test_resolve_resume_checkpoint_auto_selects_latest_numeric_checkpoint(tmp_path):
    output_dir = tmp_path / "models"
    (output_dir / "checkpoint-3").mkdir(parents=True)
    (output_dir / "checkpoint-12").mkdir()
    (output_dir / "checkpoints" / "checkpoint-20").mkdir(parents=True)
    (output_dir / "not-a-checkpoint").mkdir()
    custom = output_dir / "checkpoint-custom"
    custom.mkdir()
    for checkpoint in [
        output_dir / "checkpoint-3",
        output_dir / "checkpoint-12",
        output_dir / "checkpoints" / "checkpoint-20",
        custom,
    ]:
        (checkpoint / "trainer_state.json").write_text("{}")
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
        (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
        (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")

    assert train.resolve_resume_checkpoint("auto", str(output_dir)) == str(
        output_dir / "checkpoints" / "checkpoint-20"
    )
    assert train.resolve_resume_checkpoint.last_result["searched_roots"] == [
        str(output_dir),
        str(output_dir / "checkpoints"),
    ]
    assert train.resolve_resume_checkpoint(str(custom), str(output_dir)) == str(custom)
    assert train.resolve_resume_checkpoint("auto", str(tmp_path / "empty")) is None


def test_resolve_resume_checkpoint_required_rejects_partial_checkpoints(tmp_path):
    output_dir = tmp_path / "models"
    partial = output_dir / "checkpoint-5"
    partial.mkdir(parents=True)
    (partial / "trainer_state.json").write_text("{}")

    with pytest.raises(FileNotFoundError, match="missing_required_file"):
        train.resolve_resume_checkpoint("required", str(output_dir))
    assert train.resolve_resume_checkpoint.last_result["invalid_checkpoints"][0]["status"] == "invalid"

    with pytest.raises(ValueError, match="invalid candidates"):
        train.resolve_resume_checkpoint("auto", str(output_dir))

    with pytest.raises(ValueError, match="Invalid resume checkpoint"):
        train.resolve_resume_checkpoint(str(partial), str(output_dir))


def _write_deepspeed_checkpoint(path: Path, step: int) -> None:
    path.mkdir(parents=True)
    (path / "trainer_state.json").write_text("{}")
    (path / "latest").write_text(f"global_step{step}")
    tag_dir = path / f"global_step{step}"
    tag_dir.mkdir()
    (tag_dir / "mp_rank_00_model_states.pt").write_bytes(b"model")
    (tag_dir / "zero_pp_rank_0_mp_rank_00_optim_states.pt").write_bytes(b"optim")


def test_resolve_resume_checkpoint_accepts_deepspeed_layout_when_enabled(tmp_path):
    output_dir = tmp_path / "models"
    checkpoint = output_dir / "checkpoint-8"
    _write_deepspeed_checkpoint(checkpoint, 8)

    assert (
        train.resolve_resume_checkpoint("required", str(output_dir), deepspeed=True)
        == str(checkpoint)
    )
    validation = train.resolve_resume_checkpoint.last_result["invalid_checkpoints"]
    assert validation == []

    assert (
        train.resolve_resume_checkpoint(str(checkpoint), str(output_dir), deepspeed=True)
        == str(checkpoint)
    )
    explicit = train.resolve_resume_checkpoint.last_result["explicit_checkpoint_validation"]
    assert explicit["uses_deepspeed_layout"] is True
    assert explicit["missing_required_files"] == []


def test_resolve_resume_checkpoint_detects_deepspeed_layout_without_flag(tmp_path):
    output_dir = tmp_path / "models"
    checkpoint = output_dir / "checkpoint-14"
    _write_deepspeed_checkpoint(checkpoint, 14)

    assert train.resolve_resume_checkpoint("auto", str(output_dir)) == str(checkpoint)
    report = train._checkpoint_validation_report(checkpoint)
    assert report["uses_deepspeed_layout"] is True
    assert report["deepspeed_layout"]["model_state_shards"] == [
        "global_step14/mp_rank_00_model_states.pt"
    ]


def test_resolve_resume_checkpoint_deepspeed_flag_still_rejects_partial_checkpoints(tmp_path):
    output_dir = tmp_path / "models"
    partial = output_dir / "checkpoint-6"
    partial.mkdir(parents=True)
    (partial / "trainer_state.json").write_text("{}")

    with pytest.raises(FileNotFoundError, match="missing_required_file:optimizer.pt"):
        train.resolve_resume_checkpoint("required", str(output_dir), deepspeed=True)


def test_training_config_file_keys_flatten_to_cli_destinations():
    flattened = train._flatten_cli_config(
        {
            "model": {
                "name": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "max_sequence_length": 4096,
                "quantization": True,
                "lora": {
                    "enabled": True,
                    "rank": 8,
                    "alpha": 16,
                    "dropout": 0.05,
                    "target_modules": ["q_proj", "v_proj"],
                },
            },
            "training": {
                "epochs": 2,
                "batch_size": 2,
                "global_batch_size": 16,
                "gpu_count": 4,
                "learning_rate": 1e-4,
            },
            "dataset": {
                "path": "data/hf_training_dataset.jsonl",
                "skip_collection": True,
            },
            "evaluation": {
                "batch_size": 4,
            },
        }
    )

    assert flattened["model_name"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert flattened["max_seq_length"] == 4096
    assert flattened["use_quantization"] is True
    assert flattened["use_lora"] is True
    assert flattened["lora_rank"] == 8
    assert flattened["lora_alpha"] == 16
    assert flattened["lora_dropout"] == 0.05
    assert flattened["lora_target_modules"] == ["q_proj", "v_proj"]
    assert flattened["epochs"] == 2
    assert flattened["batch_size"] == 2
    assert flattened["global_batch_size"] == 16
    assert flattened["num_gpus"] == 4
    assert flattened["lr"] == 1e-4
    assert flattened["dataset"] == "data/hf_training_dataset.jsonl"
    assert flattened["skip_collection"] is True
    assert flattened["eval_batch_size"] == 4


def test_auto_torchrun_relaunch_uses_default_four_gpus(monkeypatch):
    calls = {}

    def fake_execv(executable, args):
        calls["executable"] = executable
        calls["args"] = args
        raise SystemExit("execv called")

    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(train, "_cuda_device_count", lambda: 8)
    monkeypatch.setattr(train.os, "execv", fake_execv)
    monkeypatch.setattr(sys, "argv", ["train.py", "--skip-collection"])

    with pytest.raises(SystemExit, match="execv called"):
        train._maybe_relaunch_with_torchrun(
            SimpleNamespace(
                dataset_only=False,
                no_auto_torchrun=False,
                num_gpus=train.DEFAULT_NUM_GPUS,
            )
        )

    assert calls["executable"] == sys.executable
    assert "--nproc_per_node=4" in calls["args"]
    assert calls["args"][-2:] == ["train.py", "--skip-collection"]


def test_main_persists_run_manifest_with_resume_dataset_and_training_refs(tmp_path, monkeypatch):
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
    for candidate in (checkpoint, older_checkpoint):
        (candidate / "trainer_state.json").write_text("{}")
        (candidate / "optimizer.pt").write_bytes(b"optimizer")
        (candidate / "scheduler.pt").write_bytes(b"scheduler")
        (candidate / "adapter_model.safetensors").write_bytes(b"adapter")
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
            "--skip-data-preflight",
            "--no-auto-torchrun",
            "--precision",
            "fp16",
            "--no-quantization",
            "--train-eval-strategy",
            "no",
            "--seed",
            "123",
            "--report-to",
            "tensorboard",
            "--dataloader-num-workers",
            "0",
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
    assert captured_train_kwargs["precision"] == "fp16"
    assert captured_train_kwargs["use_quantization"] is False
    assert captured_train_kwargs["train_eval_strategy"] == "no"
    assert captured_train_kwargs["seed"] == 123
    assert captured_train_kwargs["report_to"] == "tensorboard"
    assert captured_train_kwargs["dataloader_num_workers"] == 0
    assert manifest["status"] == "completed"
    assert manifest["training"]["config"]["resume"] == "auto"
    assert manifest["training"]["config"]["resume_from_checkpoint"] == str(checkpoint)
    assert manifest["training"]["config"]["precision"] == "fp16"
    assert manifest["training"]["config"]["seed"] == 123
    assert manifest["training"]["config"]["report_to"] == "tensorboard"
    assert manifest["runtime"]["global_seed"] == 123
    assert manifest["training"]["config"]["resume_resolution"]["searched_roots"] == [
        str(output_dir),
        str(output_dir / "checkpoints"),
    ]
    assert manifest["training"]["config"]["dataloader"]["num_workers"] == 0
    assert manifest["datasets"]["source"]["row_count"] == len(rows)
    assert manifest["datasets"]["splits"]["train"]["sha256"]
    assert manifest["training"]["final_metrics"]["train_loss"] == 0.25
    assert manifest["training"]["log_history_references"][0]["log_history_count"] == 1
    assert manifest["evaluation"] == {"skipped": True}


def test_split_dataset_reuses_matching_manifest_without_resplitting(tmp_path, monkeypatch):
    rows = [
        {
            "input": f"tac {i}",
            "output": f"function f{i}() public {{}}",
            "metadata": {"contract_address": f"0x{i}", "function_signature": f"f{i}()"},
        }
        for i in range(12)
    ]
    dataset_path = tmp_path / "dataset.jsonl"
    split_dir = tmp_path / "splits"
    _write_jsonl(dataset_path, rows)

    first = train.split_dataset(
        str(dataset_path),
        str(split_dir),
        train_ratio=0.6,
        val_ratio=0.2,
        seed=99,
        reuse_existing=True,
    )
    assert train.split_dataset.last_status["reused"] is False

    def fail_if_resplit(*_args, **_kwargs):
        raise AssertionError("split cache miss unexpectedly regenerated splits")

    monkeypatch.setattr(train, "_stratified_component_split", fail_if_resplit)
    second = train.split_dataset(
        str(dataset_path),
        str(split_dir),
        train_ratio=0.6,
        val_ratio=0.2,
        seed=99,
        reuse_existing=True,
    )

    assert second == first
    assert train.split_dataset.last_status["reused"] is True


def test_split_dataset_creates_nested_output_directory(tmp_path):
    rows = [
        {
            "input": f"tac {i}",
            "output": f"function f{i}() public {{}}",
            "metadata": {"contract_address": f"0x{i}", "function_signature": f"f{i}()"},
        }
        for i in range(12)
    ]
    dataset_path = tmp_path / "dataset.jsonl"
    nested_split_dir = tmp_path / "runs" / "curriculum" / "splits"
    _write_jsonl(dataset_path, rows)

    train_path, val_path, test_path = train.split_dataset(
        str(dataset_path),
        str(nested_split_dir),
        train_ratio=0.6,
        val_ratio=0.2,
        seed=99,
    )

    assert Path(train_path).exists()
    assert Path(val_path).exists()
    assert Path(test_path).exists()
    assert (nested_split_dir / "split_manifest.json").exists()


def test_split_quality_gate_fails_oversized_component_by_default(tmp_path):
    rows = [
        {
            "input": f"shared tac {i}",
            "output": "function shared() public { return; }",
            "metadata": {"contract_address": f"0xshared{i}", "function_signature": "shared()"},
        }
        for i in range(110)
    ]
    rows.extend(
        {
            "input": f"unique tac {i}",
            "output": f"function unique{i}() public {{ return; }}",
            "metadata": {"contract_address": f"0xunique{i}", "function_signature": f"u{i}()"},
        }
        for i in range(10)
    )
    dataset_path = tmp_path / "degenerate.jsonl"
    _write_jsonl(dataset_path, rows)

    with pytest.raises(ValueError, match="Split quality validation failed"):
        train.split_dataset(str(dataset_path), str(tmp_path / "blocked"))

    train.split_dataset(
        str(dataset_path),
        str(tmp_path / "allowed"),
        allow_degenerate_splits=True,
    )
    manifest = json.loads((tmp_path / "allowed" / "split_manifest.json").read_text())
    assert manifest["split_quality"]["status"] == "passed"
    assert manifest["split_quality"]["largest_component_rows"] == 110


def test_preflight_tokenizer_failure_fails_closed_and_override_is_explicit(tmp_path, monkeypatch):
    import types

    dataset_path = tmp_path / "dataset.jsonl"
    _write_jsonl(
        dataset_path,
        [{"input": "tac", "output": "sol", "metadata": {"schema_version": 1}}],
    )

    class FailingAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise OSError("missing tokenizer")

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FailingAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    failed = train.run_data_preflight(
        {"train": str(dataset_path)},
        tokenizer_source="missing-model",
        max_seq_length=32,
    )
    assert failed["status"] == "failed"
    assert failed["error"] == "tokenizer_unavailable"
    assert failed["tokenizer"]["mode"] == "unavailable"

    allowed = train.run_data_preflight(
        {"train": str(dataset_path)},
        tokenizer_source="missing-model",
        max_seq_length=32,
        allow_whitespace_fallback=True,
    )
    assert allowed["status"] == "passed"
    assert allowed["tokenizer"]["mode"] == "whitespace_fallback"
    assert allowed["tokenizer"]["fallback_allowed"] is True


def test_eval_only_autodetected_test_dataset_requires_split_manifest(tmp_path):
    test_path = tmp_path / "test_dataset.jsonl"
    _write_jsonl(test_path, [{"input": "tac", "output": "sol", "metadata": {}}])

    with pytest.raises(ValueError, match="not verified"):
        train.verify_autodetected_test_dataset(test_path, tmp_path / "split_manifest.json")

    info = train.verify_autodetected_test_dataset(
        test_path,
        tmp_path / "split_manifest.json",
        allow_unverified=True,
    )
    assert info["status"] == "unverified_allowed"
    assert "split_manifest_missing" in info["problems"]


def test_skip_collection_does_not_autodetect_cached_split_artifacts(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    train_split = data_dir / "train_dataset.jsonl"
    _write_jsonl(train_split, [{"input": "tac", "output": "sol", "metadata": {}}])
    manifest_path = tmp_path / "run.manifest.json"
    monkeypatch.setattr(train, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--skip-collection",
            "--data-dir",
            str(data_dir),
            "--run-manifest",
            str(manifest_path),
            "--no-auto-torchrun",
        ],
    )

    with pytest.raises(SystemExit, match="No dataset found"):
        train.main()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "failed"


def test_split_artifact_source_requires_explicit_override_and_is_manifested(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    train_split = data_dir / "train_dataset.jsonl"
    _write_jsonl(train_split, [{"input": "tac", "output": "sol", "metadata": {}}])
    monkeypatch.setattr(train, "setup_logging", lambda *args, **kwargs: None)
    blocked_manifest = tmp_path / "blocked.manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--skip-collection",
            "--dataset",
            str(train_split),
            "--data-dir",
            str(data_dir),
            "--dataset-only",
            "--skip-data-preflight",
            "--run-manifest",
            str(blocked_manifest),
            "--no-auto-torchrun",
        ],
    )

    with pytest.raises(SystemExit, match="cached split artifact"):
        train.main()

    allowed_manifest = tmp_path / "allowed.manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--skip-collection",
            "--dataset",
            str(train_split),
            "--allow-split-artifact-source",
            "--data-dir",
            str(data_dir),
            "--dataset-only",
            "--skip-data-preflight",
            "--run-manifest",
            str(allowed_manifest),
            "--no-auto-torchrun",
        ],
    )

    train.main()

    manifest = json.loads(allowed_manifest.read_text())
    split_manifest = json.loads((data_dir / "split_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["datasets"]["source"]["dataset_artifact_type"] == "derived_split_artifact"
    assert split_manifest["source_dataset"]["dataset_artifact_type"] == "derived_split_artifact"


def test_main_finalizes_manifest_for_early_eval_only_failure(tmp_path, monkeypatch):
    manifest_path = tmp_path / "failed.manifest.json"
    monkeypatch.setattr(train, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--eval-only",
            "--model-path",
            str(tmp_path / "missing-model"),
            "--run-manifest",
            str(manifest_path),
            "--no-auto-torchrun",
        ],
    )

    with pytest.raises(SystemExit):
        train.main()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["message"].startswith("Model path does not exist")
