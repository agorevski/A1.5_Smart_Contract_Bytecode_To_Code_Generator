from src.model_setup import (
    ModelConfig,
    SmartContractDataset,
    SmartContractDecompiler,
    format_prompt_metadata,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_format_prompt_metadata_includes_compiler_and_optimizer():
    metadata = {
        "function_name": "transfer",
        "visibility": "public",
        "is_payable": False,
        "is_view": False,
        "compiler_version": "v0.8.20",
        "optimizer_enabled": True,
        "optimizer_runs": 200,
    }

    line = format_prompt_metadata(metadata)

    assert "Function: transfer" in line
    assert "Visibility: public" in line
    assert "Solidity compiler: solc 0.8.20" in line
    assert "Optimizer: enabled (200 runs)" in line

    ablated_line = format_prompt_metadata(metadata, include_compiler_metadata=False)
    assert "Solidity compiler" not in ablated_line
    assert "Optimizer" not in ablated_line


def test_dataset_prompt_includes_compiler_metadata():
    dataset = SmartContractDataset.__new__(SmartContractDataset)
    dataset.template_format = "alpaca"
    dataset.include_compiler_metadata = True

    prefix, target, suffix = dataset._format_prompt_parts(
        "function foo():\n  block_0:",
        "function foo() public {}",
        {
            "function_name": "foo",
            "compiler_version": "0.6.12",
            "optimizer_enabled": False,
        },
    )

    assert "Solidity compiler: solc 0.6.12" in prefix
    assert "Optimizer: disabled" in prefix
    assert "function foo() public {}" == target
    assert suffix == ""


def test_decompiler_prompt_includes_compiler_metadata():
    decompiler = SmartContractDecompiler.__new__(SmartContractDecompiler)
    decompiler.tokenizer = FakeTokenizer()
    decompiler.config = ModelConfig(include_compiler_metadata=True)

    prompt = decompiler._build_prompt(
        "function foo():\n  block_0:",
        {
            "function_name": "foo",
            "compiler_version": "0.8.26",
            "optimizer_enabled": True,
            "optimizer_runs": "200",
        },
    )

    assert "Solidity compiler: solc 0.8.26" in prompt
    assert "Optimizer: enabled (200 runs)" in prompt
    assert "### Response:" in prompt
