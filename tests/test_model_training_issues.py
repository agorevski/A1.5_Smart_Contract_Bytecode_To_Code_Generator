import json
from pathlib import Path
from types import SimpleNamespace

import train
from src import model_setup
from src.model_setup import (
    ModelConfig,
    SmartContractDataset,
    SmartContractDecompiler,
    TokenizationCacheConfig,
    TrainingInstrumentationCallback,
    TrainingInstrumentationConfig,
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


class CountingTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "</s>"
    eos_token_id = 1
    name_or_path = "counting-tokenizer"
    model_max_length = 2048

    def __init__(self):
        self.calls = 0
        self._vocab = {"<pad>": 0, "</s>": 1, "static": 2}

    def __len__(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def __call__(self, text, **_kwargs):
        self.calls += 1
        return {"input_ids": text.split()}

    def encode(self, text, add_special_tokens=False):
        self.calls += 1
        return text.split()


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


def test_tokenized_dataset_cache_reuses_examples_and_max_length_invalidates(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    rows = [
        {"input": "tac one", "output": "sol one", "metadata": {"function_name": "a"}},
        {"input": "tac two", "output": "sol two", "metadata": {"function_name": "b"}},
    ]
    _write_jsonl(dataset_path, rows)
    cache_config = TokenizationCacheConfig(
        enabled=True,
        cache_dir=str(tmp_path / "token-cache"),
    )

    tokenizer = CountingTokenizer()
    dataset = SmartContractDataset(
        str(dataset_path),
        tokenizer,
        max_length=64,
        tokenization_cache=cache_config,
    )
    build_calls = tokenizer.calls
    first_item = dataset[0]

    assert build_calls > 0
    assert tokenizer.calls == build_calls

    second_tokenizer = CountingTokenizer()
    cached_dataset = SmartContractDataset(
        str(dataset_path),
        second_tokenizer,
        max_length=64,
        tokenization_cache=cache_config,
    )

    assert second_tokenizer.calls == 0
    assert cached_dataset[0] == first_item
    assert second_tokenizer.calls == 0

    changed_length_tokenizer = CountingTokenizer()
    SmartContractDataset(
        str(dataset_path),
        changed_length_tokenizer,
        max_length=32,
        tokenization_cache=cache_config,
    )

    changed_vocab_tokenizer = CountingTokenizer()
    changed_vocab_tokenizer._vocab["new-token"] = 3
    SmartContractDataset(
        str(dataset_path),
        changed_vocab_tokenizer,
        max_length=64,
        tokenization_cache=cache_config,
    )

    assert changed_length_tokenizer.calls > 0
    assert changed_vocab_tokenizer.calls > 0


def test_tokenized_dataset_cache_invalidates_prompt_flags_and_dataset_fingerprint(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    rows = [
        {
            "input": "tac",
            "output": "uint256 amount = 1; return amount;",
            "metadata": {"compiler_version": "0.8.20", "optimizer_enabled": True},
        }
    ]
    _write_jsonl(dataset_path, rows)
    cache_dir = tmp_path / "token-cache"
    cache_config = TokenizationCacheConfig(enabled=True, cache_dir=str(cache_dir))

    SmartContractDataset(
        str(dataset_path),
        CountingTokenizer(),
        max_length=64,
        include_compiler_metadata=True,
        augment_names=False,
        tokenization_cache=cache_config,
    )
    initial_cache_files = set(cache_dir.glob("*.jsonl"))

    compiler_flag_tokenizer = CountingTokenizer()
    SmartContractDataset(
        str(dataset_path),
        compiler_flag_tokenizer,
        max_length=64,
        include_compiler_metadata=False,
        augment_names=False,
        tokenization_cache=cache_config,
    )
    after_compiler_flag_files = set(cache_dir.glob("*.jsonl"))

    augment_flag_tokenizer = CountingTokenizer()
    SmartContractDataset(
        str(dataset_path),
        augment_flag_tokenizer,
        max_length=64,
        include_compiler_metadata=True,
        augment_names=True,
        tokenization_cache=cache_config,
    )
    after_augment_flag_files = set(cache_dir.glob("*.jsonl"))

    _write_jsonl(
        dataset_path,
        rows + [{"input": "new tac", "output": "new sol", "metadata": {}}],
    )
    changed_data_tokenizer = CountingTokenizer()
    SmartContractDataset(
        str(dataset_path),
        changed_data_tokenizer,
        max_length=64,
        include_compiler_metadata=True,
        augment_names=False,
        tokenization_cache=cache_config,
    )
    after_dataset_change_files = set(cache_dir.glob("*.jsonl"))

    assert compiler_flag_tokenizer.calls > 0
    assert augment_flag_tokenizer.calls > 0
    assert changed_data_tokenizer.calls > 0
    assert len(after_compiler_flag_files) == len(initial_cache_files) + 1
    assert len(after_augment_flag_files) == len(after_compiler_flag_files) + 1
    assert len(after_dataset_change_files) == len(after_augment_flag_files) + 1


def test_non_quantized_ddp_setup_loads_without_device_map_and_moves_to_local_rank(
    tmp_path, monkeypatch
):
    captured_load_kwargs = {}
    set_device_calls = []

    class FakeTokenizer:
        pad_token = "<pad>"
        pad_token_id = 0
        eos_token = "</s>"
        eos_token_id = 1

        def __len__(self):
            return 8

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(vocab_size=128)
            self.device = None

        def named_modules(self):
            return [("layers.0.q_proj", object())]

        def to(self, device):
            self.device = device
            return self

        def print_trainable_parameters(self):
            pass

    def fake_from_pretrained(_model_name, **kwargs):
        captured_load_kwargs.update(kwargs)
        return FakeModel()

    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setattr(model_setup.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(model_setup.torch.cuda, "get_device_capability", lambda: (8, 0))
    monkeypatch.setattr(
        model_setup.torch.cuda,
        "set_device",
        lambda rank: set_device_calls.append(rank),
    )
    monkeypatch.setattr(
        model_setup.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        model_setup.AutoModelForCausalLM,
        "from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(model_setup, "get_peft_model", lambda model, _config: model)

    trainer = model_setup.SmartContractModelTrainer(
        ModelConfig(use_quantization=False),
        output_dir=str(tmp_path / "models"),
    )
    _tokenizer, model = trainer.setup_model()

    assert "device_map" not in captured_load_kwargs
    assert str(model.device) == "cuda:2"
    assert set_device_calls == [2]


def test_training_instrumentation_writes_throughput_summary_and_csv(tmp_path):
    summary_path = tmp_path / "throughput.json"
    csv_path = tmp_path / "throughput.csv"
    callback = TrainingInstrumentationCallback(
        TrainingInstrumentationConfig(
            enable_throughput_metrics=True,
            throughput_summary_path=str(summary_path),
            throughput_csv_path=str(csv_path),
            max_throughput_records=2,
        ),
        output_dir=tmp_path,
        train_dataset_size=4,
    )
    args = SimpleNamespace(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        world_size=1,
    )
    state = SimpleNamespace(global_step=0, max_steps=2, num_input_tokens_seen=0)

    callback.on_train_begin(args, state, None)
    state.global_step = 1
    state.num_input_tokens_seen = 10
    callback.on_step_end(args, state, None)
    state.global_step = 2
    state.num_input_tokens_seen = 20
    callback.on_step_end(args, state, None)
    callback.on_train_end(args, state, None)

    summary = json.loads(summary_path.read_text())
    csv_lines = csv_path.read_text().splitlines()

    assert summary["steps"] == 2
    assert summary["estimated_samples"] == 4
    assert summary["input_tokens_seen"] == 20
    assert len(summary["records"]) == 2
    assert csv_lines[0].startswith("step,elapsed_seconds,steps_per_second")
    assert len(csv_lines) == 3


def test_training_instrumentation_starts_bounded_torch_profiler(
    tmp_path, monkeypatch
):
    events = []
    captured_profile_kwargs = {}
    captured_schedule_kwargs = {}

    class FakeProfiler:
        def start(self):
            events.append("start")

        def step(self):
            events.append("step")

        def stop(self):
            events.append("stop")

    def fake_schedule(**kwargs):
        captured_schedule_kwargs.update(kwargs)
        return ("schedule", kwargs)

    def fake_profile(**kwargs):
        captured_profile_kwargs.update(kwargs)
        return FakeProfiler()

    trace_dir = tmp_path / "profiler-trace"
    monkeypatch.setattr(model_setup.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(model_setup.torch.profiler, "schedule", fake_schedule)
    monkeypatch.setattr(
        model_setup.torch.profiler,
        "tensorboard_trace_handler",
        lambda path: ("trace-handler", path),
    )
    monkeypatch.setattr(model_setup.torch.profiler, "profile", fake_profile)

    callback = TrainingInstrumentationCallback(
        TrainingInstrumentationConfig(
            enable_torch_profiler=True,
            profiler_trace_dir=str(trace_dir),
            profiler_wait_steps=0,
            profiler_warmup_steps=0,
            profiler_active_steps=2,
            profiler_repeat=1,
        ),
        output_dir=tmp_path,
    )

    args = SimpleNamespace()
    state = SimpleNamespace(global_step=1)
    callback.on_train_begin(args, state, None)
    callback.on_step_end(args, state, None)
    callback.on_train_end(args, state, None)

    assert events == ["start", "step", "stop"]
    assert captured_schedule_kwargs == {"wait": 0, "warmup": 0, "active": 2, "repeat": 1}
    assert captured_profile_kwargs["on_trace_ready"] == (
        "trace-handler",
        str(trace_dir),
    )
    assert trace_dir.exists()
