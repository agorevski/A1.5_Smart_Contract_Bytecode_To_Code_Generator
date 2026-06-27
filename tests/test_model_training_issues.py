import json
from pathlib import Path

import train
from src import model_setup
from src.model_setup import (
    ModelConfig,
    SmartContractDataset,
    SmartContractDecompiler,
    detect_max_sequence_length,
)
from src.training_pipeline import (
    SmartContractTrainingPipeline,
    TrainingConfig,
    sample_evaluation_data,
)


class WhitespaceTokenizer:
    eos_token_id = 0

    def __call__(self, text, **_kwargs):
        return {"input_ids": text.split()}

    def encode(self, text, add_special_tokens=False):
        return text.split()


class CharacterTokenizer:
    def __call__(self, text, **_kwargs):
        return {"input_ids": list(text)}

    def encode(self, text, add_special_tokens=False):
        return list(text)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_jsonl(path: str):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _dataset_for_item(item, max_length=16):
    dataset = SmartContractDataset.__new__(SmartContractDataset)
    dataset.tokenizer = WhitespaceTokenizer()
    dataset.max_length = max_length
    dataset.template_format = "alpaca"
    dataset.augment_names = False
    dataset.include_compiler_metadata = True
    dataset.AUGMENT_RATE = 0.3
    dataset.data = [item]
    return dataset


def test_grouped_split_keeps_compiler_variants_in_one_split(tmp_path):
    rows = [
        {
            "input": "tac a opt off",
            "output": "function foo() public { return; }",
            "metadata": {
                "contract_address": "0xAaA",
                "function_signature": "foo()",
                "compiler_version": "0.8.20",
                "optimizer_enabled": False,
            },
        },
        {
            "input": "tac a opt on",
            "output": "function foo() public { return; }",
            "metadata": {
                "contract_address": "0xAaA",
                "function_signature": "foo()",
                "compiler_version": "0.8.21",
                "optimizer_enabled": True,
            },
        },
    ]
    for i in range(4):
        rows.append(
            {
                "input": f"tac {i}",
                "output": f"function f{i}() public {{}}",
                "metadata": {
                    "contract_address": f"0x{i}",
                    "function_signature": f"f{i}()",
                },
            }
        )

    dataset_path = tmp_path / "dataset.jsonl"
    _write_jsonl(dataset_path, rows)

    train_path, val_path, test_path = train.split_dataset(
        str(dataset_path),
        str(tmp_path / "splits"),
        train_ratio=0.6,
        val_ratio=0.2,
    )

    assert Path(val_path).name == "val_dataset.jsonl"
    variant_locations = []
    for split_name, path in {
        "train": train_path,
        "val": val_path,
        "test": test_path,
    }.items():
        for row in _read_jsonl(path):
            meta = row["metadata"]
            if meta["contract_address"].lower() == "0xaaa" and meta["function_signature"] == "foo()":
                variant_locations.append(split_name)

    assert len(variant_locations) == 2
    assert len(set(variant_locations)) == 1


def test_tiny_split_does_not_duplicate_rows_across_eval_splits(tmp_path):
    rows = [
        {"input": "tac a", "output": "sol a", "metadata": {"id": "a"}},
        {"input": "tac b", "output": "sol b", "metadata": {"id": "b"}},
    ]
    dataset_path = tmp_path / "tiny.jsonl"
    _write_jsonl(dataset_path, rows)

    train_path, val_path, test_path = train.split_dataset(str(dataset_path), str(tmp_path / "out"))

    assert [row["metadata"]["id"] for row in _read_jsonl(train_path)] == ["a", "b"]
    assert _read_jsonl(val_path) == []
    assert _read_jsonl(test_path) == []


def test_dataset_truncation_preserves_target_when_prefix_exceeds_context():
    dataset = _dataset_for_item(
        {
            "input": " ".join(f"TAC_{i}" for i in range(100)),
            "output": "TARGET_UNIQUE",
            "metadata": {"function_name": "foo"},
        },
        max_length=12,
    )

    item = dataset[0]
    supervised_labels = [label for label in item["labels"] if label != -100]

    assert len(item["input_ids"]) <= 12
    assert "TARGET_UNIQUE" in supervised_labels


def test_dataset_truncation_preserves_full_short_target_span():
    dataset = _dataset_for_item(
        {
            "input": " ".join(f"TAC_{i}" for i in range(200)),
            "output": "TARGET_A TARGET_B TARGET_C",
            "metadata": {},
        },
        max_length=32,
    )

    item = dataset[0]
    supervised_labels = [label for label in item["labels"] if label != -100]

    assert supervised_labels == ["TARGET_A", "TARGET_B", "TARGET_C"]


def test_deepspeed_precision_falls_back_to_fp16_on_pre_ampere(tmp_path, monkeypatch):
    ds_config = tmp_path / "ds_config.json"
    ds_config.write_text(json.dumps({"bf16": {"enabled": True}, "zero_optimization": {"stage": 0}}))

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(model_setup, "TrainingArguments", FakeTrainingArguments)
    monkeypatch.setattr(model_setup.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(model_setup.torch.cuda, "get_device_capability", lambda: (7, 5))

    trainer = model_setup.SmartContractModelTrainer(
        ModelConfig(use_quantization=False),
        output_dir=str(tmp_path / "models"),
    )
    args = trainer.create_training_arguments(
        train_dataset_size=16,
        deepspeed_config=str(ds_config),
    )

    assert args.kwargs["bf16"] is False
    assert args.kwargs["fp16"] is True
    assert args.kwargs["deepspeed"]["bf16"]["enabled"] is False
    assert args.kwargs["deepspeed"]["fp16"]["enabled"] is True


def test_deepspeed_precision_uses_bf16_on_ampere(tmp_path, monkeypatch):
    ds_config = tmp_path / "ds_config.json"
    ds_config.write_text(json.dumps({"bf16": {"enabled": "auto"}, "fp16": {"enabled": "auto"}}))

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(model_setup, "TrainingArguments", FakeTrainingArguments)
    monkeypatch.setattr(model_setup.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(model_setup.torch.cuda, "get_device_capability", lambda: (8, 0))

    trainer = model_setup.SmartContractModelTrainer(
        ModelConfig(use_quantization=False),
        output_dir=str(tmp_path / "models"),
    )
    args = trainer.create_training_arguments(
        train_dataset_size=16,
        deepspeed_config=str(ds_config),
    )

    assert args.kwargs["bf16"] is True
    assert args.kwargs["fp16"] is False
    assert args.kwargs["deepspeed"]["bf16"]["enabled"] is True
    assert args.kwargs["deepspeed"]["fp16"]["enabled"] is False


def test_inference_tac_budget_scales_with_configured_context_length():
    decompiler = SmartContractDecompiler.__new__(SmartContractDecompiler)
    decompiler.tokenizer = WhitespaceTokenizer()
    decompiler.config = ModelConfig(max_sequence_length=2048)

    small_budget = decompiler._tac_token_budget(max_new_tokens=512)
    decompiler.config = ModelConfig(max_sequence_length=4096)
    large_budget = decompiler._tac_token_budget(max_new_tokens=512)

    long_tac = "\n".join(f"OP_{i}" for i in range(2500))
    prompt = decompiler._build_prompt(long_tac, max_new_tokens=512)
    retained_tac_tokens = sum(1 for token in prompt.split() if token.startswith("OP_"))

    assert large_budget - small_budget == 2048
    assert retained_tac_tokens > 1380
    assert len(prompt.split()) <= 4096 - 512


def test_seeded_evaluation_sampling_is_reproducible():
    rows = [{"id": i} for i in range(30)]

    sample_a, indices_a = sample_evaluation_data(rows, sample_size=7, seed=123)
    sample_b, indices_b = sample_evaluation_data(rows, sample_size=7, seed=123)

    assert indices_a == indices_b
    assert sample_a == sample_b
    assert any(
        sample_evaluation_data(rows, sample_size=7, seed=seed)[1] != indices_a
        for seed in range(124, 130)
    )


def test_tokenizer_based_max_seq_detection_changes_with_tokenizer(tmp_path):
    row = {
        "input": " ".join(["longtoken"] * 220),
        "output": " ".join(["target"] * 40),
        "metadata": {"function_name": "foo"},
    }
    dataset_path = tmp_path / "lengths.jsonl"
    _write_jsonl(dataset_path, [row])

    word_length = detect_max_sequence_length(str(dataset_path), WhitespaceTokenizer())
    character_length = detect_max_sequence_length(str(dataset_path), CharacterTokenizer())

    assert word_length >= 128
    assert character_length > word_length


def test_train_common_wrapper_uses_tokenizer_detection():
    wrapper = Path("train_common.sh").read_text()

    assert "AutoTokenizer.from_pretrained" in wrapper
    assert "detect_max_sequence_length" in wrapper
    assert "/ 3.5" not in wrapper


def test_training_pipeline_writes_standard_val_dataset_filename(tmp_path):
    rows = [
        {
            "input": "tac shared off",
            "output": "function shared() public {}",
            "metadata": {
                "contract_address": "0xShared",
                "function_signature": "shared()",
                "optimizer_enabled": False,
            },
        },
        {
            "input": "tac shared on",
            "output": "function shared() public {}",
            "metadata": {
                "contract_address": "0xShared",
                "function_signature": "shared()",
                "optimizer_enabled": True,
            },
        },
    ]
    for i in range(5):
        rows.append(
            {
                "input": f"tac {i}",
                "output": f"function f{i}() public {{}}",
                "metadata": {
                    "contract_address": f"0x{i}",
                    "function_signature": f"f{i}()",
                },
            }
        )
    dataset_path = tmp_path / "dataset.jsonl"
    _write_jsonl(dataset_path, rows)

    data_dir = tmp_path / "pipeline-data"
    data_dir.mkdir()
    pipeline = SmartContractTrainingPipeline.__new__(SmartContractTrainingPipeline)
    pipeline.config = TrainingConfig(
        etherscan_api_key="test",
        data_dir=str(data_dir),
        train_test_split=0.6,
        validation_split=0.2,
    )

    train_path, val_path, test_path = pipeline._split_dataset(str(dataset_path))

    assert Path(val_path).name == "val_dataset.jsonl"
    assert Path(val_path).exists()
    assert not (data_dir / "validation_dataset.jsonl").exists()

    shared_locations = []
    for split_name, path in {
        "train": train_path,
        "val": val_path,
        "test": test_path,
    }.items():
        for row in _read_jsonl(path):
            metadata = row["metadata"]
            if metadata["contract_address"].lower() == "0xshared":
                shared_locations.append(split_name)

    assert len(shared_locations) == 2
    assert len(set(shared_locations)) == 1
