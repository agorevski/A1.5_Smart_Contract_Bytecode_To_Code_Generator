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
        "solidity_valid_mean": 1.0,
        "solidity_compiler_checked_mean": 0.0,
        "solidity_ast_valid_mean": 0.0,
        "bytecode_semantic_score_mean": 0.875,
        "bytecode_semantic_checked_mean": 1.0,
        "bytecode_deployable_mean": 1.0,
        "bytecode_runtime_checked_mean": 1.0,
        "bytecode_runtime_match_mean": 0.0,
        "confidence_intervals": {
            "semantic_similarity_mean": {
                "mean": 0.9,
                "low": 0.85,
                "high": 0.95,
                "n": 4,
            }
        },
        "metadata_segments": {
            "coverage": {
                "compiler_version": {
                    "known": 1,
                    "unknown": 0,
                    "values": {"0.8.20": 1},
                }
            },
            "segments": {
                "compiler_version": {
                    "0.8.20": {
                        "count": 1,
                        "metrics": {
                            "semantic_similarity": {"mean": 0.9},
                            "normalized_edit_distance": {"mean": 0.1},
                            "replication_f1": {"mean": 0.7742},
                            "solidity_valid": {"mean": 1.0},
                        },
                    }
                }
            },
            "opcode_control_flow_coverage": {
                "total_examples": 1,
                "examples_with_opcode_groups": 1,
                "examples_without_opcode_groups": 0,
                "opcode_groups": {"delegatecall": 1, "push0": 1},
                "examples_with_control_flow": 1,
                "examples_without_control_flow": 0,
                "control_flow": {"branching": 1},
            },
        },
        "replication_hallucination_buckets": {
            "unsupported_calls": 2,
            "invented_state_writes": 1,
        },
        "baseline_comparison": {
            "num_metrics_compared": 2,
            "num_improvements": 1,
            "num_regressions": 1,
            "comparisons": {
                "semantic_similarity_mean": {
                    "current": 0.9,
                    "baseline": 0.8,
                    "delta": 0.1,
                    "relative_delta": 0.125,
                    "status": "improved",
                },
                "edit_distance_mean": {
                    "current": 0.1,
                    "baseline": 0.08,
                    "delta": 0.02,
                    "relative_delta": 0.25,
                    "status": "regressed",
                },
            },
        },
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
    assert "Solidity valid outputs: 100.00%" in report
    assert "Bytecode semantic score mean: 0.8750" in report
    assert "Runtime bytecode matches: 0.00%" in report
    assert "abi | 1.0000 | 0.5000 | 0.6667 | 1 | 0 | 1" in report
    assert "unsupported_calls | 2 | 66.67%" in report
    assert "Opcode and Control-Flow Coverage" in report
    assert "delegatecall (1)" in report
    assert "semantic_similarity_mean | 0.9000 | [0.8500, 0.9500] | 4" in report
    assert "compiler_version | 1 | 0 | 0.8.20 (1)" in report
    assert "compiler_version=0.8.20 | 1 | 0.9000 | 0.1000 | 0.7742 | 100.00%" in report
    assert "Metrics compared: 2" in report
    assert "edit_distance_mean | 0.1000 | 0.0800 | 0.0200 | 0.2500 | regressed" in report


def test_format_latest_results_report_derives_hallucination_rates_from_details(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    dataset = tmp_path / "test_dataset.jsonl"
    dataset.write_text('{"input": "a", "output": "b"}\n')
    results = tmp_path / "eval_results.json"
    results.write_text(
        json.dumps(
            {
                "details": [
                    {
                        "metrics": {
                            "metadata": {
                                "replication": {
                                    "candidate_fact_count": 10,
                                    "groundedness_score": 0.7,
                                    "hallucination_buckets": {
                                        "unsupported_calls": ["call:_afterapprove"],
                                        "invented_guards": [
                                            "guard:require:msg.sender==owner",
                                            "guard:revert:unauthorized",
                                        ],
                                    },
                                }
                            }
                        }
                    }
                ]
            }
        )
    )

    report = format_latest_results_report(
        summary={"num_evaluated": 1},
        model_path=str(model_dir),
        test_dataset_path=str(dataset),
        results_json_path=str(results),
    )

    assert "Total hallucinated facts: 3" in report
    assert "Groundedness score mean: 0.7000" in report
    assert "unsupported_calls | 1 | 10.00%" in report
    assert "invented_guards | 2 | 20.00%" in report


def test_format_latest_results_report_renders_worst_samples_and_benchmark_suites(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    dataset = tmp_path / "test_dataset.jsonl"
    dataset.write_text('{"input": "a", "output": "b"}\n')
    long_original = (
        "function transfer(address to, uint256 amount) public { "
        + " ".join(["balances[msg.sender] -= amount;"] * 20)
        + " }"
    )

    summary = {
        "num_evaluated": 2,
        "semantic_similarity_mean": 0.5,
        "edit_distance_mean": 0.6,
        "worst_samples": [
            {
                "dataset_index": 7,
                "function_signature": "transfer(address,uint256)",
                "reason": "lowest_replication_f1",
                "output_hash": "abc123",
                "original": long_original,
                "decompiled": "function transfer(address to, uint256 amount) public { return; }",
                "metrics": {
                    "semantic_similarity": 0.81,
                    "normalized_edit_distance": 0.62,
                    "replication_f1": 0.1,
                    "bytecode_semantic_score": 0.2,
                    "metadata": {
                        "replication": {
                            "missing_facts": {"state_write": ["balances[param_0]"]},
                            "extra_facts": {"return": ["return"]},
                        }
                    },
                },
            }
        ],
        "benchmark_suites": {
            "golden": {
                "count": 1,
                "metrics": {
                    "semantic_similarity": {"mean": 0.9},
                    "normalized_edit_distance": {"mean": 0.1},
                    "replication_f1": {"mean": 0.8},
                    "bytecode_semantic_score": {"mean": 0.75},
                    "solidity_valid": {"mean": 1.0},
                },
            },
            "robustness": {
                "count": 1,
                "metrics": {
                    "semantic_similarity": {"mean": 0.3},
                    "normalized_edit_distance": {"mean": 0.7},
                    "replication_f1": {"mean": 0.2},
                    "bytecode_semantic_score": {"mean": 0.4},
                    "solidity_valid": {"mean": 0.0},
                },
            },
        },
    }

    report = format_latest_results_report(
        summary=summary,
        model_path=str(model_dir),
        test_dataset_path=str(dataset),
        results_json_path="results/eval_test.json",
    )

    assert "Benchmark Suites" in report
    assert "golden | 1 | 0.9000 | 0.1000 | 0.8000 | 0.7500 | 100.00%" in report
    assert "Worst Samples" in report
    assert "dataset_index=7 function=transfer(address,uint256)" in report
    assert "abc123" in report
    assert "state_write=[balances[param_0]]" in report
    assert "...<truncated>" in report


def test_format_latest_results_report_renders_prompt_diagnostics_and_truncated_worst(
    tmp_path,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    dataset = tmp_path / "test_dataset.jsonl"
    dataset.write_text('{"input": "a", "output": "b"}\n')

    summary = {
        "num_evaluated": 2,
        "prompt_diagnostics": {
            "num_details": 2,
            "truncated_count": 1,
            "truncated_rate": 0.5,
            "strategy_counts": {"hard_truncate": 1, "none": 1},
            "tac_tokens_before": {
                "mean": 125.0,
                "max": 200.0,
                "percentiles": {"50th": 125.0, "90th": 185.0, "95th": 192.5},
            },
            "prompt_tokens": {
                "mean": 80.0,
                "max": 100.0,
                "percentiles": {"50th": 80.0, "90th": 96.0, "95th": 98.0},
            },
        },
        "worst_samples": {
            "truncated_low_quality": [
                {
                    "dataset_index": 3,
                    "function_signature": "big()",
                    "metrics": {
                        "semantic_similarity": 0.2,
                        "normalized_edit_distance": 0.9,
                    },
                    "prompt_diagnostics": {
                        "context_window": 128,
                        "prompt_budget": 96,
                        "max_new_tokens": 32,
                        "tac_tokens_before": 200,
                        "tac_tokens_after": 40,
                        "prompt_tokens": 90,
                        "generated_tokens": 5,
                        "tac_truncated": True,
                        "strategy": "hard_truncate",
                        "marker": "// ... truncated",
                    },
                    "original": "function big() public { return; }",
                    "decompiled": "function big() public {}",
                }
            ]
        },
    }

    report = format_latest_results_report(
        summary=summary,
        model_path=str(model_dir),
        test_dataset_path=str(dataset),
        results_json_path="results/eval_test.json",
    )

    assert "Prompt/Truncation Diagnostics" in report
    assert "TAC truncated: 1/2 (50.00%)" in report
    assert "Strategies: hard_truncate=1, none=1" in report
    assert "tac_tokens_before | 125.0000 | 125.0000 | 185.0000 | 192.5000 | 200.0000" in report
    assert "reason=truncated_low_quality" in report
    assert "prompt_diagnostics=tac_truncated=True, strategy=hard_truncate" in report


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
