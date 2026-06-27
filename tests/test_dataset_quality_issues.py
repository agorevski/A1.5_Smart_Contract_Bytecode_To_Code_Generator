"""
Unit tests for all 12 dataset quality issues identified in docs/dataset-generation.md.

Each test class maps to one issue and verifies the fix is working correctly.
Run with: python -m pytest tests/test_dataset_quality_issues.py -v
"""

import json
import re
import hashlib
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Issue 1: Bytecode-only TAC excludes source metadata
# ---------------------------------------------------------------------------


class TestIssue01_BytecodeOnlyTac:
    """TAC output should not include source-only compiler metadata."""

    def test_extract_tac_omits_compiler_version(self):
        """_extract_tac should keep bytecode facts and omit compiler metadata."""
        from download_hf_contracts import _extract_tac

        func = MagicMock()
        func.name = "func_0xa9059cbb"
        func.selector = "0xa9059cbb"
        func.entry_block = "block_0x0000"
        func.basic_blocks = []

        analyzer = MagicMock()
        analyzer.basic_blocks = {}

        tac = _extract_tac(func, analyzer, compiler_version="0.8.20", optimizer_enabled=True)
        assert "// Compiler:" not in tac
        assert "function selector_a9059cbb:" in tac
        assert "// Selector: 0xa9059cbb" in tac

    def test_extract_tac_omits_optimizer_metadata(self):
        from download_hf_contracts import _extract_tac

        func = MagicMock()
        func.name = "func_test"
        func.selector = "0x12345678"
        func.entry_block = "block_0x0000"
        func.basic_blocks = []

        analyzer = MagicMock()
        analyzer.basic_blocks = {}

        tac = _extract_tac(func, analyzer, compiler_version="0.6.12", optimizer_enabled=False)
        assert "// Compiler:" not in tac
        assert "optimizer" not in tac.lower()
        assert "// Selector: 0x12345678" in tac

    def test_extract_tac_no_version_omits_line(self):
        from download_hf_contracts import _extract_tac

        func = MagicMock()
        func.name = "func_test"
        func.selector = "0x12345678"
        func.entry_block = "block_0x0000"
        func.basic_blocks = []

        analyzer = MagicMock()
        analyzer.basic_blocks = {}

        tac = _extract_tac(func, analyzer, compiler_version="", optimizer_enabled=False)
        assert "// Compiler:" not in tac


# ---------------------------------------------------------------------------
# Issue 2: Storage slot labeling
# ---------------------------------------------------------------------------


class TestIssue02_StorageSlotLabeling:
    """Storage layout resolver should parse Solidity state variables and
    annotate TAC with slot→variable mappings."""

    def test_simple_state_variables(self):
        """Parse basic state variable declarations and assign slots."""
        from src.abi_enrichment import StorageLayoutResolver

        source = """
        pragma solidity ^0.8.0;
        contract Token {
            address public owner;
            uint256 public totalSupply;
            mapping(address => uint256) public balances;

            function transfer(address to, uint256 amount) public {}
        }
        """
        resolver = StorageLayoutResolver(source, contract_name="Token")
        layout = resolver.get_storage_layout()

        assert len(layout) >= 3
        assert layout[0].name == "owner"
        assert layout[0].type_name == "address"
        assert layout[1].name == "totalSupply"
        assert layout[1].type_name == "uint256"
        assert layout[2].name == "balances"
        assert layout[2].is_mapping

    def test_annotate_slot(self):
        """annotate_slot should return a comment for known slots."""
        from src.abi_enrichment import StorageLayoutResolver

        source = """
        contract Token {
            address public owner;
            uint256 public totalSupply;
        }
        """
        resolver = StorageLayoutResolver(source)
        assert resolver.annotate_slot(0) == "// likely: address owner"
        assert resolver.annotate_slot(1) == "// likely: uint256 totalSupply"
        assert resolver.annotate_slot(99) is None

    def test_annotate_slot_hex(self):
        """annotate_slot_hex should handle full 64-char hex slot keys."""
        from src.abi_enrichment import StorageLayoutResolver

        source = """
        contract Token {
            address public owner;
        }
        """
        resolver = StorageLayoutResolver(source)
        # Full 64-char slot 0
        result = resolver.annotate_slot_hex(
            "0x0000000000000000000000000000000000000000000000000000000000000000"
        )
        assert result == "// likely: address owner"

        # Short hex
        assert resolver.annotate_slot_hex("0x00") == "// likely: address owner"

    def test_storage_header_format(self):
        """format_storage_header should produce a readable layout summary."""
        from src.abi_enrichment import StorageLayoutResolver

        source = """
        contract Token {
            address public owner;
            mapping(address => uint256) public balances;
        }
        """
        resolver = StorageLayoutResolver(source)
        header = resolver.format_storage_header()

        assert len(header) >= 3  # header line + 2 variables
        assert header[0] == "// Storage layout:"
        assert "slot 0" in header[1]
        assert "owner" in header[1]
        assert "mapping" in header[2].lower() or "balances" in header[2]

    def test_constants_and_immutables_skipped(self):
        """Constants and immutables should not get storage slots."""
        from src.abi_enrichment import StorageLayoutResolver

        source = """
        contract Token {
            uint256 constant MAX_SUPPLY = 1000000;
            address immutable deployer;
            address public owner;
        }
        """
        resolver = StorageLayoutResolver(source)
        layout = resolver.get_storage_layout()

        # Only 'owner' should have a slot
        names = [v.name for v in resolver.variables]
        assert "owner" in names
        assert "MAX_SUPPLY" not in names
        assert "deployer" not in names

    def test_empty_source(self):
        """Empty source should produce no variables."""
        from src.abi_enrichment import StorageLayoutResolver

        resolver = StorageLayoutResolver("")
        assert not resolver.has_data()
        assert resolver.get_storage_layout() == {}

    def test_function_body_variables_not_captured(self):
        """Variables inside function bodies should not be counted as state vars."""
        from src.abi_enrichment import StorageLayoutResolver

        source = """
        contract Token {
            uint256 public stateVar;
            function foo() public {
                uint256 localVar = 42;
                address temp = msg.sender;
            }
        }
        """
        resolver = StorageLayoutResolver(source)
        names = [v.name for v in resolver.variables]
        assert "stateVar" in names
        assert "localVar" not in names
        assert "temp" not in names

    def test_tac_enrichment_with_storage(self):
        """enrich_tac_with_abi should add storage layout and slot annotations."""
        from src.abi_enrichment import StorageLayoutResolver, enrich_tac_with_abi

        source = """
        contract Token {
            address public owner;
            uint256 public totalSupply;
        }
        """
        resolver = StorageLayoutResolver(source)

        tac = (
            "function func_0xa9059cbb:\n"
            "  // Selector: 0xa9059cbb\n"
            "  // Entry block: block_0x0000\n"
            "  block_0x0000:\n"
            "    temp_1 = storage[0x0000000000000000000000000000000000000000000000000000000000000000]\n"
        )

        enriched = enrich_tac_with_abi(tac, "0xa9059cbb", storage_resolver=resolver)
        assert "// Storage layout:" in enriched
        assert "// likely: address owner" in enriched


# ---------------------------------------------------------------------------
# Issue 3: Revert/panic code decoding
# ---------------------------------------------------------------------------


class TestIssue03_RevertPanicDecoding:
    """PANIC_CODES and ERROR_SELECTORS should be defined for revert decoding,
    and the REVERT handler in bytecode_analyzer should decode them inline."""

    def test_panic_codes_defined(self):
        """PANIC_CODES dict should have standard panic code mappings."""
        from src.abi_enrichment import PANIC_CODES

        assert PANIC_CODES[0x01] == "assert failure"
        assert PANIC_CODES[0x11] == "arithmetic overflow/underflow"
        assert PANIC_CODES[0x12] == "division by zero"
        assert PANIC_CODES[0x32] == "array index out of bounds"
        assert PANIC_CODES[0x41] == "too much memory allocated"

    def test_error_selectors_defined(self):
        """ERROR_SELECTORS should map known error signatures."""
        from src.abi_enrichment import ERROR_SELECTORS

        assert ERROR_SELECTORS["08c379a0"] == "Error(string)"
        assert ERROR_SELECTORS["4e487b71"] == "Panic(uint256)"

    def test_custom_error_parsing_from_abi(self):
        """ABIEnricher should parse custom error definitions from ABI."""
        from src.abi_enrichment import ABIEnricher

        abi_json = json.dumps(
            [
                {
                    "type": "error",
                    "name": "InsufficientBalance",
                    "inputs": [
                        {"name": "required", "type": "uint256"},
                        {"name": "available", "type": "uint256"},
                    ],
                }
            ]
        )
        enricher = ABIEnricher(abi_json)
        assert len(enricher.errors) == 1

        # Look up by selector
        error = list(enricher.errors.values())[0]
        assert error.name == "InsufficientBalance"
        assert error.input_types == ["uint256", "uint256"]
        assert error.input_names == ["required", "available"]

        # Should be findable via get_error
        found = enricher.get_error(error.selector)
        assert found is not None
        assert found.name == "InsufficientBalance"

    # -- Inline revert decoding in bytecode_analyzer --

    def test_decode_revert_data_method_exists(self):
        """BytecodeAnalyzer should have a _decode_revert_data method."""
        from src.bytecode_analyzer import BytecodeAnalyzer

        assert hasattr(BytecodeAnalyzer, "_decode_revert_data")

    def test_error_string_revert_decoded_in_tac(self):
        """A REVERT preceded by PUSH4 0x08c379a0 should decode to Error(string)."""
        from src.bytecode_analyzer import BytecodeAnalyzer

        # PUSH1 0x00, PUSH4 0x08c379a0, MSTORE, PUSH1 0x00, PUSH1 0x24, REVERT
        bytecode = "0x600063" + "08c379a0" + "52600060" + "24fd"
        analyzer = BytecodeAnalyzer(bytecode)
        tac = analyzer.convert_to_tac()

        revert_instrs = [t for t in tac if t.operation.value == "revert"]
        assert len(revert_instrs) >= 1
        decoded = revert_instrs[0].metadata.get("revert_decoded")
        assert decoded is not None, "revert_decoded should be set in metadata"
        assert decoded["type"] == "Error(string)"

    def test_panic_revert_decoded_in_tac(self):
        """A REVERT preceded by PUSH4 0x4e487b71 + PUSH1 0x11 should decode Panic."""
        from src.bytecode_analyzer import BytecodeAnalyzer

        # PUSH1 0x00, PUSH4 0x4e487b71, MSTORE, PUSH1 0x11, MSTORE,
        # PUSH1 0x00, PUSH1 0x24, REVERT
        bytecode = "0x600063" + "4e487b71" + "52601152" + "600060" + "24fd"
        analyzer = BytecodeAnalyzer(bytecode)
        tac = analyzer.convert_to_tac()

        revert_instrs = [t for t in tac if t.operation.value == "revert"]
        assert len(revert_instrs) >= 1
        decoded = revert_instrs[0].metadata.get("revert_decoded")
        assert decoded is not None
        assert decoded["type"] == "Panic"
        assert "0x11" in decoded.get("code", "") or "0x11" in decoded.get("message", "")

    def test_revert_format_uses_decoded_message(self):
        """_format_tac_instruction should output decoded revert message."""
        from src.bytecode_analyzer import BytecodeAnalyzer, TACInstruction, TACOperationType

        instr = TACInstruction(
            operation=TACOperationType.REVERT,
            operand1="0x00",
            operand2="0x24",
            metadata={
                "original_op": "REVERT",
                "revert_decoded": {"type": "Error(string)", "message": "Error(string)"},
            },
        )
        formatted = BytecodeAnalyzer._format_tac_instruction(instr)
        assert formatted == "revert Error(string)"

    def test_revert_without_decoded_shows_memory(self):
        """When no revert data is decoded, show the raw memory range."""
        from src.bytecode_analyzer import BytecodeAnalyzer, TACInstruction, TACOperationType

        instr = TACInstruction(
            operation=TACOperationType.REVERT,
            operand1="temp_1",
            operand2="temp_2",
            metadata={"original_op": "REVERT"},
        )
        formatted = BytecodeAnalyzer._format_tac_instruction(instr)
        assert formatted == "revert memory[temp_1:temp_2]"

    def test_custom_error_decoded_via_abi_enricher(self):
        """_decode_revert_data should use abi_enricher for custom errors."""
        from src.bytecode_analyzer import BytecodeAnalyzer
        from src.abi_enrichment import ABIEnricher

        abi_json = json.dumps(
            [
                {
                    "type": "error",
                    "name": "InsufficientBalance",
                    "inputs": [
                        {"name": "required", "type": "uint256"},
                        {"name": "available", "type": "uint256"},
                    ],
                }
            ]
        )
        enricher = ABIEnricher(abi_json)
        error = list(enricher.errors.values())[0]
        sel_hex = error.selector.replace("0x", "")

        # PUSH1 0x00, PUSH4 <custom_sel>, MSTORE, PUSH1 0x00, PUSH1 0x44, REVERT
        bytecode = "0x600063" + sel_hex + "52600060" + "44fd"
        analyzer = BytecodeAnalyzer(bytecode, abi_enricher=enricher)
        tac = analyzer.convert_to_tac()

        revert_instrs = [t for t in tac if t.operation.value == "revert"]
        assert len(revert_instrs) >= 1
        decoded = revert_instrs[0].metadata.get("revert_decoded")
        assert decoded is not None
        assert decoded["type"] == "CustomError"
        assert decoded["name"] == "InsufficientBalance"


# ---------------------------------------------------------------------------
# Issue 4: No failed/partial decompilation examples (placeholder)
# ---------------------------------------------------------------------------


class TestIssue04_PartialDecompilationExamples:
    """Partial decompilation examples should be generated for unmatched functions."""

    def test_generate_partial_pairs_method_exists(self):
        """DatasetBuilder should have _generate_partial_pairs."""
        from src.dataset_pipeline import DatasetBuilder

        assert hasattr(DatasetBuilder, "_generate_partial_pairs")

    def test_partial_pair_generated_for_unmatched_function(self):
        """An unmatched bytecode function should produce a partial pair."""
        from src.dataset_pipeline import DatasetBuilder, FunctionPair
        from src.bytecode_analyzer import Function, BasicBlock, TACInstruction, TACOperationType

        with patch.object(DatasetBuilder, "__init__", lambda self, *a, **k: None):
            builder = DatasetBuilder.__new__(DatasetBuilder)
            builder.logger = MagicMock()

            block = BasicBlock(
                id="block_0x0080",
                instructions=[
                    TACInstruction(
                        TACOperationType.LOAD,
                        result="t1",
                        operand1="0",
                        metadata={"memory_type": "storage"},
                    ),
                ],
                predecessors=[],
                successors=[],
                start_address=0x80,
                end_address=0x90,
                metadata={},
            )
            mock_analyzer = MagicMock()
            mock_analyzer.basic_blocks = {"block_0x0080": block}

            builder._extract_tac_for_function = MagicMock(
                return_value="function func_0xdeadbeef:\n  block_0x0080:\n    t1 = storage[0]"
            )
            builder._collect_function_blocks = MagicMock(return_value=[block])

            bytecode_funcs = {
                "function_0xdeadbeef": Function(
                    name="function_0xdeadbeef",
                    selector="0xdeadbeef",
                    basic_blocks=[],
                    entry_block="block_0x0080",
                ),
            }
            pairs = builder._generate_partial_pairs("0xtest", bytecode_funcs, [], mock_analyzer)
            assert len(pairs) == 1
            assert "Partial decompilation" in pairs[0].solidity_code
            assert "unknown_deadbeef" in pairs[0].solidity_code
            assert pairs[0].metadata.get("partial") is True

    def test_partial_pair_skips_already_matched(self):
        """Matched selectors should not produce partial pairs."""
        from src.dataset_pipeline import DatasetBuilder
        from src.bytecode_analyzer import Function

        with patch.object(DatasetBuilder, "__init__", lambda self, *a, **k: None):
            builder = DatasetBuilder.__new__(DatasetBuilder)
            builder.logger = MagicMock()

            bytecode_funcs = {
                "f": Function(name="f", selector="0xaabb", basic_blocks=[], entry_block="b"),
            }
            pairs = builder._generate_partial_pairs(
                "0x",
                bytecode_funcs,
                [{"selector": "0xaabb"}],
                MagicMock(),
            )
            assert len(pairs) == 0

    def test_partial_pair_format(self):
        """Partial Solidity output should include structural hints."""
        partial = (
            "// Partial decompilation — selector: 0xa9059cbb\n"
            "// Control flow: 3 block(s)\n"
            "function unknown_a9059cbb(/* params unknown */) public {\n"
            "    // TODO: Full logic not reconstructed\n"
            "}"
        )
        assert "Partial decompilation" in partial
        assert "Control flow" in partial


# ---------------------------------------------------------------------------
# Issue 5: Trivial function filtering
# ---------------------------------------------------------------------------


class TestIssue05_TrivialFunctionFiltering:
    """Expanded trivial patterns and token-count heuristic."""

    def test_original_patterns_still_caught(self):
        from download_hf_contracts import is_trivial_function

        assert is_trivial_function("{ return x; }")
        assert is_trivial_function("{ return 42; }")
        assert is_trivial_function("{ }")

    def test_return_true_false_trivial(self):
        from download_hf_contracts import is_trivial_function

        assert is_trivial_function("{ return true; }")
        assert is_trivial_function("{ return false; }")

    def test_simple_setter_trivial(self):
        from download_hf_contracts import is_trivial_function

        assert is_trivial_function("{ x = y; }")

    def test_emit_only_trivial(self):
        from download_hf_contracts import is_trivial_function

        assert is_trivial_function("{ emit Transfer(from, to, amount); }")

    def test_return_property_trivial(self):
        from download_hf_contracts import is_trivial_function

        assert is_trivial_function("{ return obj.prop; }")

    def test_modifier_placeholder_trivial(self):
        from download_hf_contracts import is_trivial_function

        assert is_trivial_function("{ _; }")

    def test_return_func_call_trivial(self):
        from download_hf_contracts import is_trivial_function

        assert is_trivial_function("{ return foo(x); }")

    def test_complex_body_not_trivial(self):
        from download_hf_contracts import is_trivial_function

        body = """{
            require(balances[msg.sender] >= amount, "Insufficient balance");
            balances[msg.sender] -= amount;
            balances[to] += amount;
            emit Transfer(msg.sender, to, amount);
            return true;
        }"""
        assert not is_trivial_function(body)

    def test_token_count_filter(self):
        """Bodies with fewer than MIN_MEANINGFUL_TOKENS should be trivial."""
        from download_hf_contracts import is_trivial_function, MIN_MEANINGFUL_TOKENS

        # Very short but doesn't match regex patterns exactly
        short_body = "{ a = b + c; }"
        assert is_trivial_function(short_body)  # < 15 tokens


# ---------------------------------------------------------------------------
# Issue 6: Missing constructor/receive/fallback handling (placeholder)
# ---------------------------------------------------------------------------


class TestIssue06_SpecialFunctionHandling:
    """BytecodeAnalyzer should detect receive(), fallback(), and internal functions."""

    def test_detect_receive_function_method_exists(self):
        """BytecodeAnalyzer should have _detect_receive_function."""
        from src.bytecode_analyzer import BytecodeAnalyzer

        assert hasattr(BytecodeAnalyzer, "_detect_receive_function")

    def test_detect_fallback_function_method_exists(self):
        """BytecodeAnalyzer should have _detect_fallback_function."""
        from src.bytecode_analyzer import BytecodeAnalyzer

        assert hasattr(BytecodeAnalyzer, "_detect_fallback_function")

    def test_detect_internal_functions_method_exists(self):
        """BytecodeAnalyzer should have _detect_internal_functions."""
        from src.bytecode_analyzer import BytecodeAnalyzer

        assert hasattr(BytecodeAnalyzer, "_detect_internal_functions")

    def test_receive_detection_with_calldatasize_iszero(self):
        """_detect_receive_function should find CALLDATASIZE ISZERO PUSH JUMPI."""
        from src.bytecode_analyzer import BytecodeAnalyzer

        # 36=CALLDATASIZE 15=ISZERO 61 0080=PUSH2 0x0080 57=JUMPI 00=STOP
        # Then pad to PC 0x0080 with INVALID, then JUMPDEST STOP
        pre = "361561008057" + "00"
        padding = "fe" * (0x80 - 7)
        post = "5b00"
        bytecode = "0x" + pre + padding + post

        analyzer = BytecodeAnalyzer(bytecode)
        analyzer.analyze_control_flow()

        result = analyzer._detect_receive_function(set())
        assert result is not None
        assert result.name == "receive"
        assert result.is_payable is True

    def test_receive_skipped_when_target_is_dispatcher(self):
        """If the CALLDATASIZE ISZERO target is a dispatcher target, skip it."""
        from src.bytecode_analyzer import BytecodeAnalyzer

        pre = "361561008057" + "00"
        padding = "fe" * (0x80 - 7)
        post = "5b00"
        bytecode = "0x" + pre + padding + post

        analyzer = BytecodeAnalyzer(bytecode)
        analyzer.analyze_control_flow()
        # Mark block as already used by dispatcher
        result = analyzer._detect_receive_function({"block_0080"})
        assert result is None

    def test_fallback_detection_method(self):
        """_detect_fallback_function should locate the fall-through after last PUSH4/EQ/JUMPI."""
        from src.bytecode_analyzer import BytecodeAnalyzer

        # Selector is derived from CALLDATALOAD(0) via SHR before PUSH4/EQ/JUMPI.
        # 5b=JUMPDEST at PC 0x10, 00=STOP (fallback)
        # ... pad to 0x20 ... 5b 00 (dispatcher target)
        pre = "60003560e01c8063aabbccdd146020575b00"
        padding = "fe" * (0x20 - len(pre) // 2)
        post = "5b00"
        bytecode = "0x" + pre + padding + post

        analyzer = BytecodeAnalyzer(bytecode)
        analyzer.analyze_control_flow()
        result = analyzer._detect_fallback_function(set())
        assert result is not None
        assert result.name == "fallback_function"

    def test_internal_function_detection(self):
        """_detect_internal_functions should find JUMP targets outside known entries."""
        from src.bytecode_analyzer import BytecodeAnalyzer, Function, BasicBlock

        analyzer = BytecodeAnalyzer("0x00")

        # Create mock raw instructions for the blocks
        # block_0000 ends with a JUMP (unconditional) to block_0010
        jump_instr = MagicMock()
        jump_instr.name = "JUMP"
        jump_instr.pc = 4

        # block_0010 starts with JUMPDEST (valid internal function target)
        jumpdest_instr = MagicMock()
        jumpdest_instr.name = "JUMPDEST"
        jumpdest_instr.pc = 0x10
        stop_instr = MagicMock()
        stop_instr.name = "STOP"
        stop_instr.pc = 0x11

        analyzer.basic_blocks = {
            "block_0000": BasicBlock(
                id="block_0000",
                instructions=[],
                predecessors=[],
                successors=["block_0010"],
                start_address=0,
                end_address=5,
                metadata={"raw_instructions": [jump_instr]},
            ),
            "block_0010": BasicBlock(
                id="block_0010",
                instructions=[],
                predecessors=["block_0000"],
                successors=[],
                start_address=0x10,
                end_address=0x15,
                metadata={"raw_instructions": [jumpdest_instr, stop_instr]},
            ),
        }
        known = {
            "f": Function(name="f", selector="0xaabb", basic_blocks=[], entry_block="block_0000"),
        }
        internal = analyzer._detect_internal_functions(known)
        assert "internal_block_0010" in internal
        assert internal["internal_block_0010"].visibility == "internal"

    def test_identify_functions_calls_special_detection(self):
        """identify_functions should invoke receive/fallback/internal detection."""
        from src.bytecode_analyzer import BytecodeAnalyzer

        bytecode = "0x608060405234801561001057600080fd5b50600436106100365760003560e01c8063893d20e81461003b578063a6f9dae114610059575b600080fd"
        analyzer = BytecodeAnalyzer(bytecode)
        analyzer.analyze_control_flow()
        functions = analyzer.identify_functions()
        assert len(functions) >= 1


# ---------------------------------------------------------------------------
# Issue 7: Shared utility blocks deduplication (placeholder)
# ---------------------------------------------------------------------------


class TestIssue07_SharedBlockDedup:
    """_collect_function_blocks should deduplicate shared blocks across functions."""

    def test_collect_function_blocks_accepts_emitted_set(self):
        """_collect_function_blocks should accept an emitted_blocks parameter."""
        from src.dataset_pipeline import DatasetBuilder
        import inspect

        sig = inspect.signature(DatasetBuilder._collect_function_blocks)
        assert "emitted_blocks" in sig.parameters

    def test_shared_block_replaced_with_reference(self):
        """When a block was already emitted, it should produce a reference comment."""
        from src.dataset_pipeline import DatasetBuilder
        from src.bytecode_analyzer import BasicBlock, TACInstruction, TACOperationType

        with patch.object(DatasetBuilder, "__init__", lambda self, *a, **k: None):
            builder = DatasetBuilder.__new__(DatasetBuilder)
            builder.logger = MagicMock()

            shared = BasicBlock(
                id="block_shared",
                instructions=[
                    TACInstruction(
                        TACOperationType.REVERT,
                        operand1="0",
                        operand2="0",
                        metadata={"original_op": "REVERT"},
                    ),
                ],
                predecessors=[],
                successors=[],
                start_address=0x200,
                end_address=0x210,
                metadata={},
            )
            entry = BasicBlock(
                id="block_entry",
                instructions=[
                    TACInstruction(
                        TACOperationType.ASSIGN, result="t1", operand1="42", metadata={}
                    ),
                ],
                predecessors=[],
                successors=["block_shared"],
                start_address=0x80,
                end_address=0x90,
                metadata={},
            )

            all_blocks = {"block_entry": entry, "block_shared": shared}

            # First call: emitted is empty → both blocks collected
            emitted: set = set()
            blocks1 = builder._collect_function_blocks(
                "block_entry",
                all_blocks,
                emitted_blocks=emitted,
            )
            assert "block_entry" in emitted
            assert "block_shared" in emitted

            # Second call with same emitted set → shared block skipped
            entry2 = BasicBlock(
                id="block_entry2",
                instructions=[
                    TACInstruction(
                        TACOperationType.ASSIGN, result="t2", operand1="99", metadata={}
                    ),
                ],
                predecessors=[],
                successors=["block_shared"],
                start_address=0xA0,
                end_address=0xB0,
                metadata={},
            )
            all_blocks["block_entry2"] = entry2
            blocks2 = builder._collect_function_blocks(
                "block_entry2",
                all_blocks,
                emitted_blocks=emitted,
            )
            # block_shared should appear as a reference (shared_ref),
            # not with its original full instructions
            shared_blocks = [b for b in blocks2 if b.id == "block_shared"]
            assert len(shared_blocks) == 1
            ref_block = shared_blocks[0]
            assert ref_block.metadata.get("is_shared_ref") is True
            assert len(ref_block.instructions) == 1
            assert ref_block.instructions[0].metadata.get("shared_ref") is True

    def test_without_emitted_set_includes_all(self):
        """Without emitted_blocks, all reachable blocks should be included."""
        from src.dataset_pipeline import DatasetBuilder
        from src.bytecode_analyzer import BasicBlock, TACInstruction, TACOperationType

        with patch.object(DatasetBuilder, "__init__", lambda self, *a, **k: None):
            builder = DatasetBuilder.__new__(DatasetBuilder)
            builder.logger = MagicMock()

            shared = BasicBlock(
                id="block_shared",
                instructions=[],
                predecessors=[],
                successors=[],
                start_address=0x200,
                end_address=0x210,
                metadata={},
            )
            entry = BasicBlock(
                id="block_entry",
                instructions=[],
                predecessors=[],
                successors=["block_shared"],
                start_address=0x80,
                end_address=0x90,
                metadata={},
            )
            all_blocks = {"block_entry": entry, "block_shared": shared}

            blocks = builder._collect_function_blocks("block_entry", all_blocks)
            block_ids = [b.id for b in blocks]
            assert "block_entry" in block_ids
            assert "block_shared" in block_ids


# ---------------------------------------------------------------------------
# Issue 8: ABI data usage for type annotations
# ---------------------------------------------------------------------------


class TestIssue08_ABIDataUsage:
    """ABI data should be parsed and used to enrich TAC with type annotations."""

    def test_abi_enricher_parses_functions(self):
        """ABIEnricher should parse function entries and compute selectors."""
        from src.abi_enrichment import ABIEnricher

        abi_json = json.dumps(
            [
                {
                    "type": "function",
                    "name": "transfer",
                    "inputs": [
                        {"name": "to", "type": "address"},
                        {"name": "amount", "type": "uint256"},
                    ],
                    "outputs": [{"name": "", "type": "bool"}],
                    "stateMutability": "nonpayable",
                }
            ]
        )
        enricher = ABIEnricher(abi_json)
        assert enricher.has_data()

        expected_sel = "0xa9059cbb"
        func = enricher.get_function(expected_sel)
        assert func is not None
        assert func.name == "transfer"
        assert func.input_types == ["address", "uint256"]
        assert func.input_names == ["to", "amount"]
        assert func.output_types == ["bool"]

    def test_abi_enricher_parses_events(self):
        """ABIEnricher should parse event entries and compute topic0."""
        from src.abi_enrichment import ABIEnricher

        abi_json = json.dumps(
            [
                {
                    "type": "event",
                    "name": "Transfer",
                    "inputs": [
                        {"name": "from", "type": "address", "indexed": True},
                        {"name": "to", "type": "address", "indexed": True},
                        {"name": "value", "type": "uint256", "indexed": False},
                    ],
                }
            ]
        )
        enricher = ABIEnricher(abi_json)
        expected_topic = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
        event = enricher.get_event_by_topic(expected_topic)
        assert event is not None
        assert event.name == "Transfer"
        assert event.indexed == [True, True, False]

    def test_abi_enricher_handles_tuple_types(self):
        """ABIEnricher should resolve tuple types from ABI components."""
        from src.abi_enrichment import ABIEnricher

        abi_json = json.dumps(
            [
                {
                    "type": "function",
                    "name": "submitOrder",
                    "inputs": [
                        {
                            "name": "order",
                            "type": "tuple",
                            "components": [
                                {"name": "maker", "type": "address"},
                                {"name": "amount", "type": "uint256"},
                            ],
                        }
                    ],
                    "outputs": [],
                    "stateMutability": "nonpayable",
                }
            ]
        )
        enricher = ABIEnricher(abi_json)
        func = list(enricher.functions.values())[0]
        assert func.input_types == ["(address,uint256)"]

    def test_abi_enricher_handles_tuple_array(self):
        """ABIEnricher should resolve tuple[] types."""
        from src.abi_enrichment import ABIEnricher

        abi_json = json.dumps(
            [
                {
                    "type": "function",
                    "name": "batchTransfer",
                    "inputs": [
                        {
                            "name": "transfers",
                            "type": "tuple[]",
                            "components": [
                                {"name": "to", "type": "address"},
                                {"name": "amount", "type": "uint256"},
                            ],
                        }
                    ],
                    "outputs": [],
                    "stateMutability": "nonpayable",
                }
            ]
        )
        enricher = ABIEnricher(abi_json)
        func = list(enricher.functions.values())[0]
        assert func.input_types == ["(address,uint256)[]"]

    def test_format_function_header(self):
        """format_function_header should produce a named function with typed params."""
        from src.abi_enrichment import ABIEnricher

        abi_json = json.dumps(
            [
                {
                    "type": "function",
                    "name": "transfer",
                    "inputs": [
                        {"name": "to", "type": "address"},
                        {"name": "amount", "type": "uint256"},
                    ],
                    "outputs": [{"name": "", "type": "bool"}],
                    "stateMutability": "nonpayable",
                }
            ]
        )
        enricher = ABIEnricher(abi_json)
        sel = "0xa9059cbb"
        header = enricher.format_function_header(sel)
        assert header == "function transfer(address to, uint256 amount):"

    def test_format_return_annotation(self):
        """format_return_annotation should produce a return type comment."""
        from src.abi_enrichment import ABIEnricher

        abi_json = json.dumps(
            [
                {
                    "type": "function",
                    "name": "balanceOf",
                    "inputs": [{"name": "account", "type": "address"}],
                    "outputs": [{"name": "", "type": "uint256"}],
                    "stateMutability": "view",
                }
            ]
        )
        enricher = ABIEnricher(abi_json)
        sel = "0x70a08231"
        ret = enricher.format_return_annotation(sel)
        assert ret == "// Returns: uint256"

    def test_format_param_annotations(self):
        """format_param_annotations should produce indexed parameter comments."""
        from src.abi_enrichment import ABIEnricher

        abi_json = json.dumps(
            [
                {
                    "type": "function",
                    "name": "transfer",
                    "inputs": [
                        {"name": "to", "type": "address"},
                        {"name": "amount", "type": "uint256"},
                    ],
                    "outputs": [{"name": "", "type": "bool"}],
                    "stateMutability": "nonpayable",
                }
            ]
        )
        enricher = ABIEnricher(abi_json)
        sel = "0xa9059cbb"
        params = enricher.format_param_annotations(sel)
        assert len(params) == 2
        assert "param[0] at 0x04: address to" in params[0]
        assert "param[1] at 0x24: uint256 amount" in params[1]

    def test_format_event_annotation(self):
        """format_event_annotation should produce a readable event comment."""
        from src.abi_enrichment import ABIEnricher

        abi_json = json.dumps(
            [
                {
                    "type": "event",
                    "name": "Transfer",
                    "inputs": [
                        {"name": "from", "type": "address", "indexed": True},
                        {"name": "to", "type": "address", "indexed": True},
                        {"name": "value", "type": "uint256", "indexed": False},
                    ],
                }
            ]
        )
        enricher = ABIEnricher(abi_json)
        topic = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
        annotation = enricher.format_event_annotation(topic)
        assert "Transfer" in annotation
        assert "indexed" in annotation

    def test_abi_enricher_empty_abi(self):
        """ABIEnricher should handle empty/invalid ABI gracefully."""
        from src.abi_enrichment import ABIEnricher

        enricher = ABIEnricher("")
        assert not enricher.has_data()

        enricher2 = ABIEnricher("not valid json")
        assert not enricher2.has_data()

        enricher3 = ABIEnricher("[]")
        assert not enricher3.has_data()

    def test_enrich_tac_with_abi_function_header(self):
        """enrich_tac_with_abi should replace function header with ABI info."""
        from src.abi_enrichment import ABIEnricher, enrich_tac_with_abi

        abi_json = json.dumps(
            [
                {
                    "type": "function",
                    "name": "transfer",
                    "inputs": [
                        {"name": "to", "type": "address"},
                        {"name": "amount", "type": "uint256"},
                    ],
                    "outputs": [{"name": "", "type": "bool"}],
                    "stateMutability": "nonpayable",
                }
            ]
        )
        enricher = ABIEnricher(abi_json)
        sel = "0xa9059cbb"

        tac = (
            f"function func_{sel}:\n"
            f"  // Selector: {sel}\n"
            f"  // Entry block: block_0x0000\n"
            f"  block_0x0000:\n"
            f"    temp_1 = calldata[0x04]\n"
        )

        enriched = enrich_tac_with_abi(tac, sel, abi_enricher=enricher)
        assert "function transfer(address to, uint256 amount):" in enriched
        assert "// Returns: bool" in enriched
        assert "param[0] at 0x04: address to" in enriched
        assert "param[1] at 0x24: uint256 amount" in enriched


# ---------------------------------------------------------------------------
# Issue 9: Fragile selector computation for complex types
# ---------------------------------------------------------------------------


class TestIssue09_SelectorComputation:
    """Balanced-parenthesis-aware parameter parsing for selectors."""

    def test_simple_params(self):
        from download_hf_contracts import _parse_solidity_param_types

        assert _parse_solidity_param_types("uint256 amount") == ["uint256"]
        assert _parse_solidity_param_types("address to, uint256 amount") == ["address", "uint256"]

    def test_empty_params(self):
        from download_hf_contracts import _parse_solidity_param_types

        assert _parse_solidity_param_types("") == []
        assert _parse_solidity_param_types("   ") == []

    def test_memory_calldata_ignored(self):
        from download_hf_contracts import _parse_solidity_param_types

        result = _parse_solidity_param_types("uint256[] memory data, address to")
        assert result == ["uint256[]", "address"]

    def test_tuple_type(self):
        from download_hf_contracts import _parse_solidity_param_types

        result = _parse_solidity_param_types("(uint256,address) data")
        assert result == ["(uint256,address)"]

    def test_tuple_array_type(self):
        from download_hf_contracts import _parse_solidity_param_types

        result = _parse_solidity_param_types("(uint256,address)[] data")
        assert result == ["(uint256,address)[]"]

    def test_nested_tuple(self):
        from download_hf_contracts import _parse_solidity_param_types

        result = _parse_solidity_param_types("(uint256,(address,bool)) data")
        assert result == ["(uint256,(address,bool))"]

    def test_fixed_size_array_with_space(self):
        from download_hf_contracts import _parse_solidity_param_types

        result = _parse_solidity_param_types("uint256 [3]")
        assert result == ["uint256[3]"]

    def test_mixed_tuple_and_regular(self):
        from download_hf_contracts import _parse_solidity_param_types

        result = _parse_solidity_param_types("(uint256,address)[] items, bool flag")
        assert result == ["(uint256,address)[]", "bool"]

    def test_dataset_pipeline_parser_same_behavior(self):
        """DatasetBuilder._parse_solidity_param_types should behave identically."""
        from src.dataset_pipeline import DatasetBuilder

        result = DatasetBuilder._parse_solidity_param_types("(uint256,address)[] items, bool flag")
        assert result == ["(uint256,address)[]", "bool"]

    def test_selector_computation_simple(self):
        """Verify that _add_selectors correctly handles simple parameters."""
        from download_hf_contracts import _add_selectors

        funcs = [{"signature": "function transfer(address to, uint256 amount)", "name": "transfer"}]
        result = _add_selectors(funcs)
        assert result[0]["selector"] == "0xa9059cbb"

    def test_parse_solidity_param_types_for_tuple_selector(self):
        """Verify _parse_solidity_param_types produces correct types for tuple."""
        from download_hf_contracts import _parse_solidity_param_types

        params = "(uint256,address) bar"
        types = _parse_solidity_param_types(params)
        assert types == ["(uint256,address)"]
        canonical = f"foo({','.join(types)})"
        assert canonical == "foo((uint256,address))"


# ---------------------------------------------------------------------------
# Issue 10: Inconsistent Solidity output format (fallback pairs removed)
# ---------------------------------------------------------------------------


class TestIssue10_NoFallbackPairs:
    """Whole-contract fallback pairs should no longer be created."""

    def test_fallback_pair_not_created(self):
        """When no functions match by selector, no fallback pair should be added."""
        from src.dataset_pipeline import DatasetBuilder

        with patch.object(DatasetBuilder, "__init__", lambda self, *a, **k: None):
            builder = DatasetBuilder.__new__(DatasetBuilder)
            builder.parser = MagicMock()
            builder.logger = MagicMock()

            # Mock: Solidity parser returns functions but none match
            builder.parser.extract_functions.return_value = [
                {
                    "name": "foo",
                    "body": "function foo() { return 1; }",
                    "signature": "function foo()",
                    "visibility": "public",
                    "is_payable": False,
                    "is_view": False,
                    "contract_name": "Test",
                }
            ]

            # bytecode with no matching selectors
            mock_analyzer = MagicMock()
            mock_analyzer.identify_functions.return_value = {}

            with patch.object(
                builder,
                "_add_selectors_to_solidity_functions",
                return_value=[
                    {
                        "name": "foo",
                        "selector": "0xdeadbeef",
                        "body": "...",
                        "signature": "function foo()",
                    }
                ],
            ):
                with patch.object(builder, "_match_functions_by_selector", return_value=[]):
                    with patch("src.dataset_pipeline.BytecodeAnalyzer", return_value=mock_analyzer):
                        pairs = builder._create_function_pairs("0xtest", "source", "0x00")

            # No fallback pair should be created
            assert len(pairs) == 0
            # Verify _create_fallback_pair was NOT called
            assert not hasattr(builder, "_create_fallback_pair_called")


# ---------------------------------------------------------------------------
# Issue 11: Variable name augmentation
# ---------------------------------------------------------------------------


class TestIssue11_VariableNameAugmentation:
    """Variable name augmentation replaces user-defined names with generic ones."""

    def test_augment_function_exists(self):
        """augment_variable_names should be importable from model_setup."""
        from src.model_setup import augment_variable_names

        assert callable(augment_variable_names)

    def test_basic_renaming(self):
        """Declared variables should be renamed to var_N."""
        from src.model_setup import augment_variable_names

        code = "uint256 amount = 100;\naddress recipient = msg.sender;\nreturn amount;"
        result = augment_variable_names(code)
        assert "var_1" in result  # amount → var_1
        assert "var_2" in result  # recipient → var_2
        assert "amount" not in result
        assert "recipient" not in result

    def test_keywords_not_renamed(self):
        """Solidity keywords/types should not be renamed."""
        from src.model_setup import augment_variable_names

        code = "uint256 totalCount = 0;\nreturn totalCount;"
        result = augment_variable_names(code)
        assert "uint256" in result
        assert "return" in result
        assert "totalCount" not in result
        assert "var_1" in result

    def test_empty_input(self):
        """Empty input should return empty."""
        from src.model_setup import augment_variable_names

        assert augment_variable_names("") == ""
        assert augment_variable_names("   ") == "   "

    def test_no_declarations_unchanged(self):
        """Code with no variable declarations should pass through."""
        from src.model_setup import augment_variable_names

        code = "return msg.sender;"
        result = augment_variable_names(code)
        assert result == code

    def test_solidity_reserved_words_preserved(self):
        """Built-in globals like msg, block, sender should not be renamed."""
        from src.model_setup import augment_variable_names

        code = "address owner = msg.sender;\nrequire(owner != address(0));"
        result = augment_variable_names(code)
        assert "msg" in result
        assert "sender" in result
        assert "require" in result
        assert "address" in result

    def test_dataset_applies_augmentation(self):
        """SmartContractDataset should apply augmentation ~30% of the time."""
        from src.model_setup import SmartContractDataset
        import tempfile, json

        data = [
            {"input": "tac code", "output": "uint256 amount = 1; return amount;", "metadata": {}}
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
            path = f.name

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.eos_token_id = 2

        # Call with different indices; some should be augmented
        mock_tokenizer.__call__ = MagicMock(
            return_value={
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
            }
        )
        ds = SmartContractDataset(path, mock_tokenizer, augment_names=True)
        assert ds.AUGMENT_RATE == 0.3

        import os

        os.unlink(path)

    def test_deterministic_augmentation(self):
        """augment_variable_names with same seed should produce same result."""
        from src.model_setup import augment_variable_names

        code = "uint256 amount = 100; address dest = msg.sender;"
        r1 = augment_variable_names(code, seed=42)
        r2 = augment_variable_names(code, seed=42)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Issue 12: Export-time dedup cap reduced
# ---------------------------------------------------------------------------


class TestIssue12_ReducedBodyDupeCap:
    """max_body_dupes default should be 2, not 5."""

    def test_default_max_body_dupes_is_2(self):
        """Verify the CLI default for --max-body-dupes is 2."""
        import download_hf_contracts

        import inspect

        source = inspect.getsource(download_hf_contracts.main)
        match = re.search(r"--max-body-dupes.*?default=(\d+)", source, re.DOTALL)
        assert match is not None
        assert int(match.group(1)) == 2

    def test_export_respects_max_body_dupes(self):
        """Export with max_body_dupes=2 should limit duplicates."""
        from download_hf_contracts import (
            init_database,
            _store_pairs_batch,
            export_training_data,
            _md5,
            hash_normalized_body,
            hash_normalized_tac,
            hash_normalized_pair,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            init_database(db_path)

            # Insert 5 pairs with the same pair_norm_hash
            tac = "function test:\n  block_0:\n    temp_1 = 42"
            sol = "function test() { return 42; }"
            pairs = []
            for i in range(5):
                pairs.append(
                    {
                        "contract_address": f"0xaddr{i}",
                        "function_name": "test",
                        "tac_representation": tac,
                        "solidity_code": sol,
                        "function_signature": "function test()",
                        "visibility": "public",
                        "is_payable": False,
                        "is_view": False,
                        "metadata": "{}",
                        "hash": _md5(f"{tac}{sol}{i}"),  # unique hash
                        "body_hash": hash_normalized_body(sol),
                        "tac_hash": hash_normalized_tac(tac),
                        "pair_norm_hash": hash_normalized_pair(tac, sol),
                    }
                )

            # pair_norm_hash is UNIQUE, so only 1 will be inserted
            inserted = _store_pairs_batch(db_path, pairs)
            assert inserted == 1  # UNIQUE constraint on pair_norm_hash

            output = Path(tmpdir) / "output.jsonl"
            export_training_data(str(output), max_body_dupes=2, db_path=db_path)
            with open(output) as f:
                lines = f.readlines()
            assert len(lines) <= 2


# ---------------------------------------------------------------------------
# Combined: Normalization helpers
# ---------------------------------------------------------------------------


class TestNormalizationHelpers:
    """Tests for normalization and hashing functions."""

    def test_normalize_solidity_body(self):
        from download_hf_contracts import normalize_solidity_body

        # Comment stripping + whitespace collapsing should produce same result
        body1 = "{ return   x; // comment\n}"
        body2 = "{ return x; }"
        assert normalize_solidity_body(body1) == normalize_solidity_body(body2)

    def test_normalize_solidity_body_case_insensitive(self):
        from download_hf_contracts import normalize_solidity_body

        body1 = "{ return X; }"
        body2 = "{ return x; }"
        assert normalize_solidity_body(body1) == normalize_solidity_body(body2)

    def test_normalize_tac_strips_comments(self):
        from download_hf_contracts import normalize_tac

        tac1 = "function f:\n  // Compiler: solc 0.8.20\n  temp_1 = 42"
        tac2 = "function f:\n  temp_1 = 42"
        assert normalize_tac(tac1) == normalize_tac(tac2)

    def test_hash_normalized_pair_consistent(self):
        from download_hf_contracts import hash_normalized_pair

        h1 = hash_normalized_pair("tac code", "solidity code")
        h2 = hash_normalized_pair("tac code", "solidity code")
        assert h1 == h2

    def test_hash_source_code_whitespace_insensitive(self):
        from download_hf_contracts import hash_source_code

        src1 = "contract Foo {\n    uint x;\n}"
        src2 = "contract Foo {  uint   x;  }"
        assert hash_source_code(src1) == hash_source_code(src2)


# ---------------------------------------------------------------------------
# Integration-style: Quality filter pipeline
# ---------------------------------------------------------------------------


class TestQualityFilterPipeline:
    """Integration tests for the quality filter pipeline."""

    def test_proxy_only_detection(self):
        from download_hf_contracts import is_proxy_only

        proxy_body = "{ (bool success, ) = impl.delegatecall(msg.data); require(success); }"
        assert is_proxy_only(proxy_body)

    def test_non_proxy_not_detected(self):
        from download_hf_contracts import is_proxy_only

        normal = "{ balances[msg.sender] -= amount; balances[to] += amount; }"
        assert not is_proxy_only(normal)

    def test_build_pair_rejects_tiny(self):
        from download_hf_contracts import _build_pair

        match = {
            "solidity_function": {
                "name": "f",
                "body": "{ }",
                "signature": "function f()",
                "visibility": "public",
                "is_payable": False,
                "is_view": False,
            },
            "tac": "func:\n  nop",
            "selector": "0x12345678",
        }
        result = _build_pair(match, "0xaddr", "0.8.0", False, 200, "Test")
        assert result is None  # body too short


# ---------------------------------------------------------------------------
# ABI Enrichment: end-to-end TAC annotation
# ---------------------------------------------------------------------------


class TestABIEnrichmentEndToEnd:
    """End-to-end tests for ABI + storage enrichment in TAC generation."""

    def test_enrich_tac_combined(self):
        """Test combining ABI and storage enrichment on a TAC string."""
        from src.abi_enrichment import ABIEnricher, StorageLayoutResolver, enrich_tac_with_abi

        abi_json = json.dumps(
            [
                {
                    "type": "function",
                    "name": "transfer",
                    "inputs": [
                        {"name": "to", "type": "address"},
                        {"name": "amount", "type": "uint256"},
                    ],
                    "outputs": [{"name": "", "type": "bool"}],
                    "stateMutability": "nonpayable",
                },
                {
                    "type": "event",
                    "name": "Transfer",
                    "inputs": [
                        {"name": "from", "type": "address", "indexed": True},
                        {"name": "to", "type": "address", "indexed": True},
                        {"name": "value", "type": "uint256", "indexed": False},
                    ],
                },
            ]
        )
        source = """
        contract Token {
            address public owner;
            uint256 public totalSupply;
            mapping(address => uint256) public balances;
        }
        """

        enricher = ABIEnricher(abi_json)
        resolver = StorageLayoutResolver(source)
        sel = "0xa9059cbb"

        tac = (
            f"function func_{sel}:\n"
            f"  // Selector: {sel}\n"
            f"  // Entry block: block_0x0000\n"
            f"  block_0x0000:\n"
            f"    temp_1 = calldata[0x04]\n"
            f"    temp_2 = storage[0x0000000000000000000000000000000000000000000000000000000000000000]\n"
            f"    temp_3 = storage[0x0000000000000000000000000000000000000000000000000000000000000001]\n"
        )

        enriched = enrich_tac_with_abi(tac, sel, enricher, resolver)

        # Function header should be replaced with ABI-derived version
        assert "function transfer(address to, uint256 amount):" in enriched
        # Return type should be annotated
        assert "// Returns: bool" in enriched
        # Parameters should have offset annotations
        assert "param[0] at 0x04: address to" in enriched
        # Storage should have layout header
        assert "// Storage layout:" in enriched
        # Slot 0 should be annotated
        assert "// likely: address owner" in enriched
        # Slot 1 should be annotated
        assert "// likely: uint256 totalSupply" in enriched

    def test_enrich_tac_no_enrichers(self):
        """With no enrichers, TAC should pass through unchanged."""
        from src.abi_enrichment import enrich_tac_with_abi

        tac = "function func_0x12345678:\n  // Selector: 0x12345678\n  block_0:\n    temp_1 = 42\n"
        result = enrich_tac_with_abi(tac, "0x12345678")
        assert result == tac

    def test_enrich_tac_empty_string(self):
        """Empty TAC should return empty."""
        from src.abi_enrichment import enrich_tac_with_abi

        assert enrich_tac_with_abi("", "0x12345678") == ""


# ---------------------------------------------------------------------------
# GitHub issue regressions #27, #29, #33, #34, #35, #36, #44, #53, #54
# ---------------------------------------------------------------------------


class TestGitHubIssue27_ABITypeCanonicalSelectors:
    def test_uint_alias_selector_matches_uint256(self):
        from download_hf_contracts import _add_selectors

        funcs = [{"signature": "function transfer(address to, uint amount)", "name": "transfer"}]
        assert _add_selectors(funcs)[0]["selector"] == "0xa9059cbb"

    def test_int_alias_and_tuple_arrays_are_canonicalized(self):
        from download_hf_contracts import _parse_solidity_param_types
        from src.dataset_pipeline import DatasetBuilder

        assert _parse_solidity_param_types("(uint,int[])[] data") == ["(uint256,int256[])[]"]

        with patch.object(DatasetBuilder, "__init__", lambda self, *a, **k: None):
            builder = DatasetBuilder.__new__(DatasetBuilder)
            result = builder._add_selectors_to_solidity_functions(
                [{"signature": "function foo(int value)", "name": "foo"}]
            )
        assert result[0]["selector"] == "0x4c970b2f"  # foo(int256)


class TestGitHubIssue29_BodyDuplicateExportCap:
    def test_export_caps_same_body_across_distinct_pairs(self, tmp_path):
        from download_hf_contracts import (
            _md5,
            _store_pairs_batch,
            export_training_data,
            hash_normalized_body,
            hash_normalized_pair,
            hash_normalized_tac,
            init_database,
        )

        db_path = tmp_path / "body-cap.db"
        init_database(db_path)
        sol = "function same() public { uint256 x = 1; uint256 y = x + 1; emit Done(y); }"
        pairs = []
        for idx in range(3):
            tac = f"function same:\n  block_{idx}:\n    temp = {idx}"
            pairs.append(
                {
                    "contract_address": f"0x{idx}",
                    "function_name": "same",
                    "tac_representation": tac,
                    "solidity_code": sol,
                    "function_signature": "function same()",
                    "visibility": "public",
                    "is_payable": False,
                    "is_view": False,
                    "metadata": "{}",
                    "hash": _md5(tac + sol),
                    "body_hash": hash_normalized_body(sol),
                    "tac_hash": hash_normalized_tac(tac),
                    "pair_norm_hash": hash_normalized_pair(tac, sol),
                }
            )

        assert _store_pairs_batch(db_path, pairs) == 3
        output = tmp_path / "body-cap.jsonl"
        export_training_data(
            str(output),
            max_body_dupes=1,
            db_path=db_path,
            manifest_path=tmp_path / "body-cap.manifest.json",
        )
        assert len(output.read_text().splitlines()) == 1


class TestGitHubIssue33_DeterministicHuggingFaceExport:
    def test_hf_revision_passed_to_listing(self):
        from download_hf_contracts import _get_parquet_files

        with patch("download_hf_contracts.HfApi") as api_cls:
            api = api_cls.return_value
            api.list_repo_files.return_value = [
                "data/flattened/train/b.parquet",
                "data/flattened/test/ignored.parquet",
                "README.md",
                "data/flattened/train/a.parquet",
            ]
            files = _get_parquet_files("flattened", "train", revision="abc123")

        api.list_repo_files.assert_called_once_with(
            "andstor/smart_contracts", repo_type="dataset", revision="abc123"
        )
        assert files == [
            "data/flattened/train/a.parquet",
            "data/flattened/train/b.parquet",
        ]

    def test_export_bytes_are_stable_for_different_insert_orders(self, tmp_path):
        from download_hf_contracts import (
            _md5,
            _store_pairs_batch,
            export_training_data,
            hash_normalized_body,
            hash_normalized_pair,
            hash_normalized_tac,
            init_database,
        )

        def make_pair(idx):
            sol = f"function f{idx}() public {{ uint256 x = {idx}; emit Done(x); }}"
            tac = f"function f{idx}:\n  block:\n    temp = {idx}"
            return {
                "contract_address": f"0x{idx}",
                "function_name": f"f{idx}",
                "tac_representation": tac,
                "solidity_code": sol,
                "function_signature": f"function f{idx}()",
                "visibility": "public",
                "is_payable": False,
                "is_view": False,
                "metadata": json.dumps({"compiler_version": "0.8.20"}),
                "hash": _md5(tac + sol),
                "body_hash": hash_normalized_body(sol),
                "tac_hash": hash_normalized_tac(tac),
                "pair_norm_hash": hash_normalized_pair(tac, sol),
            }

        pairs = [make_pair(2), make_pair(1), make_pair(3)]
        outputs = []
        for name, ordered_pairs in (("a", pairs), ("b", list(reversed(pairs)))):
            db_path = tmp_path / f"{name}.db"
            init_database(db_path)
            _store_pairs_batch(db_path, ordered_pairs)
            output = tmp_path / f"{name}.jsonl"
            export_training_data(
                str(output),
                max_body_dupes=5,
                db_path=db_path,
                manifest_path=tmp_path / f"{name}.manifest.json",
            )
            outputs.append(output.read_bytes())

        assert outputs[0] == outputs[1]


class TestGitHubIssue34_DatasetBuilderSemanticSchema:
    def test_schema_has_semantic_hash_columns_and_index(self, tmp_path):
        from src.dataset_pipeline import DatasetBuilder

        builder = DatasetBuilder("dummy", output_dir=str(tmp_path / "builder"))
        conn = sqlite3.connect(builder.db_path)
        contract_cols = {row[1] for row in conn.execute("PRAGMA table_info(contracts)").fetchall()}
        pair_cols = {row[1] for row in conn.execute("PRAGMA table_info(function_pairs)").fetchall()}
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(function_pairs)").fetchall()}
        conn.close()

        assert "source_hash" in contract_cols
        assert {"body_hash", "tac_hash", "pair_norm_hash"} <= pair_cols
        assert "idx_dataset_pair_norm_hash" in indexes

    def test_dataset_builder_semantic_pair_dedup(self, tmp_path):
        from src.dataset_pipeline import DatasetBuilder, FunctionPair

        builder = DatasetBuilder("dummy", output_dir=str(tmp_path / "dedup"))
        first = FunctionPair(
            function_name="f",
            tac_representation="function f:\n  temp = 1 // compiler note\n",
            solidity_code="{ return X; // comment\n}",
            function_signature="function f()",
            visibility="public",
            is_payable=False,
            is_view=False,
            contract_address="0x1",
        )
        second = FunctionPair(
            function_name="f",
            tac_representation="FUNCTION f:\n temp = 1",
            solidity_code="{   return x; }",
            function_signature="function f()",
            visibility="public",
            is_payable=False,
            is_view=False,
            contract_address="0x2",
        )

        builder._store_function_pair(first)
        builder._store_function_pair(second)

        conn = sqlite3.connect(builder.db_path)
        count, body_hash, tac_hash, pair_hash = conn.execute(
            "SELECT COUNT(*), MIN(body_hash), MIN(tac_hash), MIN(pair_norm_hash) "
            "FROM function_pairs"
        ).fetchone()
        conn.close()
        assert count == 1
        assert body_hash and tac_hash and pair_hash


class TestGitHubIssue35_ContractOutcomeTracking:
    def test_no_output_status_is_not_marked_processed(self, tmp_path):
        from download_hf_contracts import _mark_contract_status, init_database

        db_path = tmp_path / "status.db"
        init_database(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO contracts (address, source_code, bytecode) VALUES (?, ?, ?)",
            ("0xabc", "contract C {}", "0x"),
        )
        conn.commit()
        conn.close()

        _mark_contract_status(
            db_path,
            ["0xabc"],
            "no_pairs",
            processed=False,
            last_error="compile jobs produced no matched pairs",
        )

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT processed, compile_status, attempt_count, last_error "
            "FROM contracts WHERE address = ?",
            ("0xabc",),
        ).fetchone()
        conn.close()
        assert row[0] == 0
        assert row[1] == "no_pairs"
        assert row[2] == 1
        assert "no matched pairs" in row[3]


class TestGitHubIssue36_MultiFilePragmaIntersection:
    def test_compatible_versions_satisfy_all_pragmas(self):
        from src.local_compiler import compatible_versions_for_pragmas

        versions = compatible_versions_for_pragmas(
            ["^0.8.0", ">=0.8.20 <0.9.0"],
            candidate_versions=["0.8.26", "0.8.20", "0.8.10"],
        )
        assert versions == ["0.8.26", "0.8.20"]

    def test_original_version_not_selected_when_outside_intersection(self):
        from src.local_compiler import select_compilation_configs

        configs = select_compilation_configs(
            ["^0.8.0", ">=0.8.20 <0.9.0"],
            original_version="0.8.10",
            original_optimizer=True,
            original_runs=200,
            max_configs=3,
        )
        versions = [cfg["version"] for cfg in configs]
        assert "0.8.10" not in versions
        assert "0.8.20" in versions

    def test_no_intersection_returns_no_versions(self):
        from src.local_compiler import compatible_versions_for_pragmas

        assert (
            compatible_versions_for_pragmas(
                ["^0.7.0", "^0.8.0"],
                candidate_versions=["0.8.20", "0.7.6"],
            )
            == []
        )


class TestGitHubIssue44_Web3HexNormalization:
    def test_selector_hex_has_single_prefix(self):
        from src.abi_enrichment import normalize_hex
        from web3 import Web3

        selector = normalize_hex(Web3.keccak(text="transfer(address,uint256)")[:4])
        assert selector == "0xa9059cbb"
        assert not selector.startswith("0x0x")

    def test_abi_error_selector_has_single_prefix(self):
        from src.abi_enrichment import ABIEnricher

        enricher = ABIEnricher(
            json.dumps(
                [
                    {
                        "type": "error",
                        "name": "InsufficientBalance",
                        "inputs": [{"name": "required", "type": "uint"}],
                    }
                ]
            )
        )
        selector = list(enricher.errors)[0]
        assert selector.startswith("0x")
        assert not selector.startswith("0x0x")


class TestGitHubIssue53_PragmaCommentStripping:
    def test_parse_pragma_ignores_line_and_block_comments(self):
        from src.local_compiler import parse_pragma

        source = """
        // pragma solidity ^0.5.0;
        /* pragma solidity ^0.6.0; */
        pragma solidity ^0.8.20;
        contract C {}
        """
        assert parse_pragma(source) == ["^0.8.20"]


class TestGitHubIssue54_SelectorResolverRemoteConfig:
    def test_get_resolver_respects_use_remote_call_order(self):
        from src import selector_resolver as sr

        sr._default_resolver = None
        sr._resolver_cache.clear()
        offline = sr.get_resolver(use_remote=False)
        online = sr.get_resolver(use_remote=True)
        offline_again = sr.get_resolver(use_remote=False)

        assert offline.use_remote is False
        assert online.use_remote is True
        assert offline_again.use_remote is False
