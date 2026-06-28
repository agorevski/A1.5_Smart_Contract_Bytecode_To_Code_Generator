from src.model_setup import (
    ModelConfig,
    SmartContractDataset,
    SmartContractDecompiler,
    format_prompt_metadata,
    sanitize_tac_for_prompt,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


SOURCE_METADATA_MARKERS = (
    "Solidity compiler",
    "Compiler:",
    "Optimizer",
    "Function: transfer",
    "Visibility:",
    "Payable",
    "View",
    "param[",
    "Returns:",
    "Storage layout",
)


def assert_source_metadata_excluded(text):
    for marker in SOURCE_METADATA_MARKERS:
        assert marker not in text


def test_format_prompt_metadata_uses_only_bytecode_context():
    metadata = {
        "function_name": "transfer",
        "visibility": "public",
        "is_payable": True,
        "is_view": True,
        "compiler_version": "v0.8.20",
        "optimizer_enabled": True,
        "optimizer_runs": 200,
        "params": ["address to", "uint256 amount"],
        "returns": ["bool"],
        "storage_layout": {"balances": {"slot": 0}},
        "selector": "0xa9059cbb",
        "bytecode_length": 512,
        "bytecode_instruction_count": 123,
        "bytecode_function_count": 4,
    }
    tac = """// Compiler: solc 0.8.20
function transfer(address to, uint256 amount):
block_entry:
  v0 = storage[0]
  storage[1] = v0
  if v0 goto block_success
block_success:
  v1 = call(gas, to, amount)
  log1(topic, v1)
  revert
"""

    line = format_prompt_metadata(
        metadata,
        include_bytecode_metadata=True,
        include_compiler_metadata=True,
        tac_input=tac,
    )

    assert line == (
        "Bytecode metadata: selector=0xa9059cbb, "
        "selector_signature=transfer(address,uint256), tac_blocks=2, tac_ops=6, "
        "branches=1, storage_reads=1, storage_writes=1, external_calls=1, "
        "logs=1, reverts=1, bytecode_len=512, bytecode_instructions=123, "
        "functions=4"
    )
    assert_source_metadata_excluded(line)


def test_dataset_prompt_excludes_source_metadata_with_deprecated_flag_enabled():
    tac = """// Compiler: solc 0.6.12
// Visibility: public
// Payable: false
// param[0] at 0x04: address to
// Returns: bool
function transfer(address to):
// selector 0xa9059cbb
block_entry:
  v0 = CALLDATALOAD 0x04
"""
    metadata = {
        "function_name": "transfer",
        "visibility": "public",
        "is_payable": False,
        "is_view": False,
        "compiler_version": "0.6.12",
        "optimizer_enabled": False,
        "selector": "0xa9059cbb",
    }

    dataset = SmartContractDataset.__new__(SmartContractDataset)
    dataset.template_format = "alpaca"
    dataset.include_bytecode_metadata = True
    dataset.include_compiler_metadata = True

    prefix, target, suffix = dataset._format_prompt_parts(
        tac,
        "function transfer(address to) public returns (bool) {}",
        metadata,
    )

    assert "Bytecode metadata: selector=0xa9059cbb" in prefix
    assert "selector_signature=transfer(address,uint256)" in prefix
    assert "function function_0xa9059cbb:" in prefix
    assert "// selector 0xa9059cbb" in prefix
    assert "block_entry:" in prefix
    assert "v0 = CALLDATALOAD 0x04" in prefix
    assert_source_metadata_excluded(prefix)
    assert "function transfer(address to) public returns (bool) {}" == target
    assert suffix == ""


def test_decompiler_prompt_excludes_source_metadata_with_deprecated_config_flag():
    decompiler = SmartContractDecompiler.__new__(SmartContractDecompiler)
    decompiler.tokenizer = FakeTokenizer()
    decompiler.config = ModelConfig(include_compiler_metadata=True)

    prompt = decompiler._build_prompt(
        """// Compiler: solc 0.8.26
// Optimizer: enabled (200 runs)
// Returns: uint256
function balanceOf(address owner):
block_entry:
  v0 = storage[0]
""",
        {
            "function_name": "balanceOf",
            "visibility": "external",
            "is_view": True,
            "compiler_version": "0.8.26",
            "optimizer_enabled": True,
            "optimizer_runs": "200",
            "selector": "0x70a08231",
            "bytecode_length": 256,
        },
    )

    assert decompiler.config.include_compiler_metadata is False
    assert "Bytecode metadata: selector=0x70a08231" in prompt
    assert "selector_signature=balanceOf(address)" in prompt
    assert "bytecode_len=256" in prompt
    assert "function function_0x70a08231:" in prompt
    assert_source_metadata_excluded(prompt)
    assert "function balanceOf" not in prompt
    assert "### Response:" in prompt


def test_decompiler_inference_context_keeps_prompt_budget_above_training_sweep_length():
    decompiler = SmartContractDecompiler.__new__(SmartContractDecompiler)
    decompiler.tokenizer = FakeTokenizer()
    decompiler.config = ModelConfig(max_sequence_length=256)

    assert decompiler._context_window() >= 2048
    assert decompiler._prompt_token_budget(max_new_tokens=1024) >= 1024


def test_sanitize_tac_removes_oracle_annotations_but_keeps_bytecode_tac():
    tac = """// Compiler: solc 0.8.20
// param[0] at 0x04: address to
// Returns: bool
// Storage layout:
// slot 0: mapping(address => uint256) balances
// [slot 1] uint256 totalSupply
function transfer(address to, uint256 amount):
// Function selector: 0xa9059cbb
// selector 0xa9059cbb
// CFG: block_entry -> block_success
block_entry:
  v0 = CALLDATALOAD 0x04
  if v0 goto block_success
block_success:
  return v0
"""

    sanitized = sanitize_tac_for_prompt(tac, {"selector": "0xa9059cbb"})

    assert_source_metadata_excluded(sanitized)
    assert "transfer(address" not in sanitized
    assert "function function_0xa9059cbb:" in sanitized
    assert "// Function selector: 0xa9059cbb" in sanitized
    assert "// selector 0xa9059cbb" in sanitized
    assert "// CFG: block_entry -> block_success" in sanitized
    assert "block_entry:" in sanitized
    assert "v0 = CALLDATALOAD 0x04" in sanitized
    assert "if v0 goto block_success" in sanitized
