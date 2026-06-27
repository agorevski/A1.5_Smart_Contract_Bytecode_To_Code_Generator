"""Tests for human-readable evaluation report generation."""

import json

from src.evaluation_report import format_latest_results_report, write_latest_results_report


def test_format_latest_results_report_includes_quality_and_model_metadata(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model_config.json").write_text(
        json.dumps(
            {
                "model_name": "test/base-model",
                "max_sequence_length": 2048,
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"],
                "use_quantization": True,
                "load_in_4bit": True,
                "include_compiler_metadata": True,
            }
        )
    )
    (model_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    dataset = tmp_path / "test_dataset.jsonl"
    dataset.write_text('{"input": "a", "output": "b"}\n')

    summary = {
        "num_evaluated": 1,
        "semantic_similarity_mean": 0.9,
        "semantic_similarity_std": 0.0,
        "edit_distance_mean": 0.1,
        "edit_distance_std": 0.0,
        "pct_above_0.8_similarity": 1.0,
        "pct_below_0.4_edit_dist": 1.0,
        "replication_precision_micro": 0.8,
        "replication_recall_micro": 0.75,
        "replication_f1_micro": 0.7742,
        "replication_by_category_micro": {
            "abi": {
                "precision": 1.0,
                "recall": 0.5,
                "f1": 0.6667,
                "true_positives": 1,
                "false_positives": 0,
                "false_negatives": 1,
            }
        },
    }

    report = format_latest_results_report(
        summary=summary,
        model_path=str(model_dir),
        test_dataset_path=str(dataset),
        results_json_path="results/eval_test.json",
        started_at=100.0,
        finished_at=104.5,
        argv=["train.py", "--eval-only"],
        world_size=1,
    )

    assert "Smart Contract Decompiler - Latest Evaluation Results" in report
    assert "Base model: test/base-model" in report
    assert "LoRA rank: 16" in report
    assert "Examples evaluated: 1" in report
    assert "Semantic similarity mean: 0.9000" in report
    assert "Replication F1 micro: 0.7742" in report
    assert "abi | 1.0000 | 0.5000 | 0.6667 | 1 | 0 | 1" in report


def test_write_latest_results_report_creates_file(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    dataset = tmp_path / "test_dataset.jsonl"
    dataset.write_text('{"input": "a", "output": "b"}\n')
    output = tmp_path / "latest_results.txt"

    path = write_latest_results_report(
        summary={"num_evaluated": 0, "error": "No successful evaluations"},
        model_path=str(model_dir),
        test_dataset_path=str(dataset),
        results_json_path="results/eval_test.json",
        latest_results_path=str(output),
        argv=["train.py"],
    )

    assert path == str(output)
    assert output.exists()
    assert "No successful evaluations" in output.read_text()
