"""
Comprehensive tests for src/bytecode_analyzer.py

Covers:
  - Bytecode parsing
  - Control flow analysis (basic blocks, edges, loops, dominance, reachability)
  - Jump target detection and filtering
  - Function identification from dispatcher patterns
  - TAC conversion for all supported opcode categories
  - Stack simulation accuracy
  - Fallback / error-recovery paths
  - Formatted output generation
"""

import pytest
import logging
from dataclasses import fields
from src.bytecode_analyzer import (
    BytecodeAnalyzer,
    TACOperationType,
    TACInstruction,
    BasicBlock,
    Function,
    analyze_bytecode_to_tac,
    _EVM_STACK_EFFECTS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal valid bytecode: PUSH1 0x00 STOP
MINIMAL_BYTECODE = "0x600000"

# Simple contract with a dispatcher (two functions):
#   PUSH1 0x80, PUSH1 0x40, MSTORE, PUSH1 0x04, CALLDATASIZE, LT,
#   PUSH1 <fallback>, JUMPI, ...
SAMPLE_OWNER_BYTECODE = (
    "0x608060405234801561001057600080fd5b50600436106100365760003560e01c"
    "8063893d20e81461003b578063a6f9dae114610059575b600080fd5b610043610075565b"
    "6040516100509190610166565b60405180910390f35b610073600480360381019061006e"
    "91906101b2565b61009e565b005b60008060009054906101000a900473ffffffffffffff"
    "ffffffffffffffffffffffffff16905090565b8073ffffffffffffffffffffffffffffffff"
    "ffffffffff163373ffffffffffffffffffffffffffffffffffffffff16036100d35780600080"
    "6101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffff"
    "ffffffffffffffffffffffffffffffff1602179055505b50565b600073ffffffffffffffffffffff"
    "ffffffffffffffffff82169050919050565b6000610101826100d6565b9050919050565b610111"
    "816100f6565b82525050565b600060208201905061012c6000830184610108565b92915050565b"
    "600080fd5b610140816100f6565b811461014b57600080fd5b50565b60008135905061015d8161"
    "0137565b92915050565b60006020828403121561017957610178610132565b5b600061018784828"
    "50161014e565b91505092915050565b7f4e487b710000000000000000000000000000000000000000"
    "0000000000000000600052602260045260246000fd5b600060028204905060018216806101d857607f"
    "821691505b6020821081036101eb576101ea610190565b5b5091905056fea264697066735822122"
    "09d84a3c5d1d6c4c5f9c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c564736f6c"
    "634300080a0033"
)


def _make_dict_instructions(names):
    """Build a list of dict-format instructions with auto-incremented PCs."""
    instrs = []
    pc = 0
    for name in names:
        entry = {"name": name, "pc": pc}
        if name.startswith("PUSH"):
            # Give a dummy operand
            entry["operand"] = "0x00"
        instrs.append(entry)
        pc += 1
    return instrs


# ---------------------------------------------------------------------------
# 1. Parsing Tests
# ---------------------------------------------------------------------------

class TestParsing:
    def test_parse_minimal_bytecode(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        assert len(analyzer.instructions) > 0

    def test_parse_with_0x_prefix(self):
        a = BytecodeAnalyzer("0x6000")
        b = BytecodeAnalyzer("6000")
        assert len(a.instructions) == len(b.instructions)

    def test_parse_empty_bytecode(self):
        analyzer = BytecodeAnalyzer("")
        assert analyzer.instructions == []

    def test_parse_invalid_hex(self):
        analyzer = BytecodeAnalyzer("ZZZZ")
        # Should not crash; instructions may be empty or partial
        assert isinstance(analyzer.instructions, list)

    def test_pc_to_index_populated(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        assert len(analyzer._pc_to_index) == len(analyzer.instructions)


# ---------------------------------------------------------------------------
# 2. Dataclass / Enum Tests
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_function_defaults_are_lists(self):
        """parameters and return_types should default to fresh lists, not None."""
        f = Function(name="test", selector=None, basic_blocks=[], entry_block="b0")
        assert f.parameters == []
        assert f.return_types == []
        # Verify they are independent instances
        f2 = Function(name="test2", selector=None, basic_blocks=[], entry_block="b0")
        f.parameters.append("x")
        assert f2.parameters == []

    def test_basic_block_metadata_default(self):
        b = BasicBlock(id="b", instructions=[], predecessors=[], successors=[],
                       start_address=0, end_address=0)
        assert isinstance(b.metadata, dict)

    def test_tac_operation_types(self):
        """All expected operation types should exist."""
        expected = {'ASSIGN', 'BINARY_OP', 'UNARY_OP', 'LOAD', 'STORE',
                    'CALL', 'JUMP', 'CONDITIONAL_JUMP', 'RETURN', 'REVERT',
                    'HALT', 'LOG', 'NOP'}
        actual = {t.name for t in TACOperationType}
        assert expected.issubset(actual)


# ---------------------------------------------------------------------------
# 3. Jump Target Detection Tests
# ---------------------------------------------------------------------------

class TestJumpTargetDetection:
    def test_entry_point_always_included(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        targets = analyzer._detect_jump_targets()
        first_pc = analyzer._get_pc(analyzer.instructions[0], 0)
        assert first_pc in targets

    def test_jumpdest_included(self):
        # PUSH1 0x04, JUMP, JUMPDEST, STOP  →  bytecode: 6004565b00
        analyzer = BytecodeAnalyzer("0x6004565b00")
        targets = analyzer._detect_jump_targets()
        # JUMPDEST is at pc=3 (PUSH1=0, 0x04=operand byte at 1, JUMP=2, JUMPDEST=3)
        jumpdest_pcs = {
            analyzer._get_pc(i, idx)
            for idx, i in enumerate(analyzer.instructions)
            if analyzer._get_instruction_name(i) == 'JUMPDEST'
        }
        assert jumpdest_pcs.issubset(targets)

    def test_filter_removes_non_jumpdest_targets(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        # Manually inject a bad target
        raw_targets = {0, 9999}
        filtered = analyzer._filter_jump_targets(raw_targets)
        assert 9999 not in filtered

    def test_jumpi_adds_fall_through(self):
        # PUSH1 0x05, PUSH1 0x01, JUMPI, STOP, JUMPDEST, STOP
        # bytecode: 6005600157005b00
        analyzer = BytecodeAnalyzer("0x6005600157005b00")
        targets = analyzer._detect_jump_targets()
        # After JUMPI (pc depends on instruction widths) there should be a fall-through
        # At minimum, entry + JUMPDEST should be present
        assert len(targets) >= 2


# ---------------------------------------------------------------------------
# 4. Operand Parsing Tests
# ---------------------------------------------------------------------------

class TestOperandParsing:
    def test_int_operand(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        assert analyzer._parse_operand_as_int(42) == 42

    def test_hex_string_with_prefix(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        assert analyzer._parse_operand_as_int("0xff") == 255

    def test_hex_string_without_prefix(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        assert analyzer._parse_operand_as_int("ff") == 255

    def test_invalid_string(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        assert analyzer._parse_operand_as_int("not_a_number") is None

    def test_none_operand(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        assert analyzer._parse_operand_as_int(None) is None


# ---------------------------------------------------------------------------
# 5. Control Flow Analysis Tests
# ---------------------------------------------------------------------------

class TestControlFlowAnalysis:
    def test_basic_blocks_created(self):
        analyzer = BytecodeAnalyzer(SAMPLE_OWNER_BYTECODE)
        blocks = analyzer.analyze_control_flow()
        assert len(blocks) > 0

    def test_blocks_have_metadata(self):
        analyzer = BytecodeAnalyzer(SAMPLE_OWNER_BYTECODE)
        blocks = analyzer.analyze_control_flow()
        for block in blocks.values():
            assert 'raw_instructions' in block.metadata

    def test_edges_are_consistent(self):
        """Every successor relationship has a matching predecessor."""
        analyzer = BytecodeAnalyzer(SAMPLE_OWNER_BYTECODE)
        blocks = analyzer.analyze_control_flow()
        for bid, block in blocks.items():
            for succ in block.successors:
                if succ in blocks:
                    assert bid in blocks[succ].predecessors, \
                        f"{bid} -> {succ} edge missing reverse predecessor"

    def test_reachability_marks(self):
        analyzer = BytecodeAnalyzer(SAMPLE_OWNER_BYTECODE)
        blocks = analyzer.analyze_control_flow()
        for block in blocks.values():
            assert 'is_reachable' in block.metadata
            assert 'is_dead_code' in block.metadata
            assert block.metadata['is_reachable'] != block.metadata['is_dead_code']

    def test_unreachable_predecessorless_block_is_dead(self):
        analyzer = BytecodeAnalyzer("0x005b00")  # STOP; JUMPDEST; STOP
        blocks = analyzer.analyze_control_flow()
        assert blocks["block_0000"].metadata["is_reachable"] is True
        assert blocks["block_0001"].metadata["is_dead_code"] is True

    def test_dominance_computed(self):
        analyzer = BytecodeAnalyzer(SAMPLE_OWNER_BYTECODE)
        blocks = analyzer.analyze_control_flow()
        for block in blocks.values():
            assert 'dominators' in block.metadata
            # Every block dominates itself
            assert block.id in block.metadata['dominators']

    def test_loop_detection_metadata(self):
        analyzer = BytecodeAnalyzer(SAMPLE_OWNER_BYTECODE)
        blocks = analyzer.analyze_control_flow()
        for block in blocks.values():
            assert 'is_loop_header' in block.metadata
            assert isinstance(block.metadata['is_loop_header'], bool)

    def test_fallback_analysis_on_empty(self):
        analyzer = BytecodeAnalyzer("")
        blocks = analyzer._fallback_control_flow_analysis()
        assert blocks == {}


# ---------------------------------------------------------------------------
# 6. Function Identification Tests
# ---------------------------------------------------------------------------

class TestFunctionIdentification:
    def test_identifies_functions_from_dispatcher(self):
        analyzer = BytecodeAnalyzer(SAMPLE_OWNER_BYTECODE)
        analyzer.analyze_control_flow()
        functions = analyzer.identify_functions()
        assert len(functions) >= 2

    def test_function_selectors_are_valid(self):
        analyzer = BytecodeAnalyzer(SAMPLE_OWNER_BYTECODE)
        analyzer.analyze_control_flow()
        functions = analyzer.identify_functions()
        for func in functions.values():
            if func.selector:
                assert func.selector.startswith("0x")
                assert len(func.selector) == 10  # 0x + 8 hex digits

    def test_fallback_function_when_no_dispatcher(self):
        # Minimal bytecode with no PUSH4 patterns
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        analyzer.analyze_control_flow()
        functions = analyzer.identify_functions()
        assert "fallback" in functions

    def test_function_parameters_are_lists(self):
        analyzer = BytecodeAnalyzer(SAMPLE_OWNER_BYTECODE)
        analyzer.analyze_control_flow()
        functions = analyzer.identify_functions()
        for func in functions.values():
            assert isinstance(func.parameters, list)
            assert isinstance(func.return_types, list)

    def test_constant_compare_does_not_create_public_selector(self):
        analyzer = BytecodeAnalyzer("0x63deadbeef63deadbeef14600f57005b00")
        analyzer.analyze_control_flow()
        functions = analyzer.identify_functions()
        assert all(func.selector != "0xdeadbeef" for func in functions.values())

    def test_shr_dispatcher_identifies_selector(self):
        analyzer = BytecodeAnalyzer("0x60003560e01c806312345678146010575b00")
        analyzer.analyze_control_flow()
        functions = analyzer.identify_functions()
        assert "function_0x12345678" in functions

    def test_legacy_div_dispatcher_identifies_selector(self):
        divisor = "01" + ("00" * 28)
        bytecode = "0x6000357c" + divisor + "0480631234567814602c575b00"
        analyzer = BytecodeAnalyzer(bytecode)
        analyzer.analyze_control_flow()
        functions = analyzer.identify_functions()
        assert "function_0x12345678" in functions


# ---------------------------------------------------------------------------
# 7. TAC Conversion Tests – Individual Opcodes
# ---------------------------------------------------------------------------

class TestTACConversion:
    """Test _convert_instruction_to_tac for specific opcode categories."""

    def _convert_single(self, name, stack=None, operand=None):
        """Helper: convert a single dict-instruction and return (tac_result, stack)."""
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        if stack is None:
            stack = []
        instr = {"name": name, "pc": 0}
        if operand is not None:
            instr["operand"] = operand
        result = analyzer._convert_instruction_to_tac(instr, stack)
        return result, stack

    # --- PUSH ---
    def test_push(self):
        tac, stack = self._convert_single("PUSH1", operand="0x42")
        assert tac.operation == TACOperationType.ASSIGN
        assert tac.operand1 == "0x42"
        assert len(stack) == 1

    def test_push0_emits_zero_constant(self):
        analyzer = BytecodeAnalyzer("0x5f00")
        tac = analyzer.convert_to_tac()
        assert tac[0].operation == TACOperationType.ASSIGN
        assert tac[0].operand1 == "0"
        assert tac[0].metadata["original_op"] == "PUSH0"
        assert all(t.metadata.get("original_op") != "UNKNOWN_0x5f" for t in tac if t.metadata)

    # --- POP ---
    def test_pop(self):
        tac, stack = self._convert_single("POP", stack=["a"])
        assert tac is None
        assert stack == []

    # --- JUMPDEST (no-op) ---
    def test_jumpdest_is_noop(self):
        tac, stack = self._convert_single("JUMPDEST")
        assert tac is None
        assert stack == []

    # --- DUP1..DUP16 ---
    def test_dup1(self):
        tac, stack = self._convert_single("DUP1", stack=["a"])
        assert tac.operation == TACOperationType.ASSIGN
        assert tac.operand1 == "a"
        assert len(stack) == 2

    def test_dup3(self):
        tac, stack = self._convert_single("DUP3", stack=["a", "b", "c"])
        assert tac.operand1 == "a"  # 3rd from top
        assert len(stack) == 4

    def test_dup_underflow(self):
        tac, stack = self._convert_single("DUP5", stack=["a"])
        assert tac.operand1 == "stack_underflow"

    # --- SWAP1..SWAP16 ---
    def test_swap1(self):
        tac, stack = self._convert_single("SWAP1", stack=["a", "b"])
        assert tac is None
        assert stack == ["b", "a"]

    def test_swap3(self):
        tac, stack = self._convert_single("SWAP3", stack=["w", "x", "y", "z"])
        assert tac is None
        assert stack[-1] == "w"
        assert stack[0] == "z"

    # --- Binary ops: arithmetic ---
    def test_add(self):
        tac, stack = self._convert_single("ADD", stack=["a", "b"])
        assert tac.operation == TACOperationType.BINARY_OP
        assert tac.operator == "+"
        assert len(stack) == 1

    def test_sub(self):
        tac, _ = self._convert_single("SUB", stack=["a", "b"])
        assert tac.operator == "-"
        assert tac.operand1 == "a"
        assert tac.operand2 == "b"

    def test_mul(self):
        tac, _ = self._convert_single("MUL", stack=["a", "b"])
        assert tac.operator == "*"

    def test_div(self):
        tac, _ = self._convert_single("DIV", stack=["a", "b"])
        assert tac.operator == "/"

    def test_sdiv(self):
        tac, _ = self._convert_single("SDIV", stack=["a", "b"])
        assert tac.operator == "s/"

    def test_mod(self):
        tac, _ = self._convert_single("MOD", stack=["a", "b"])
        assert tac.operator == "%"

    def test_exp(self):
        tac, _ = self._convert_single("EXP", stack=["a", "b"])
        assert tac.operator == "**"

    # --- Binary ops: comparison ---
    def test_lt(self):
        tac, _ = self._convert_single("LT", stack=["a", "b"])
        assert tac.operator == "<"

    def test_gt(self):
        tac, _ = self._convert_single("GT", stack=["a", "b"])
        assert tac.operator == ">"

    def test_eq(self):
        tac, _ = self._convert_single("EQ", stack=["a", "b"])
        assert tac.operator == "=="

    def test_slt(self):
        tac, _ = self._convert_single("SLT", stack=["a", "b"])
        assert tac.operator == "s<"

    def test_sgt(self):
        tac, _ = self._convert_single("SGT", stack=["a", "b"])
        assert tac.operator == "s>"

    # --- Binary ops: bitwise ---
    def test_and(self):
        tac, _ = self._convert_single("AND", stack=["a", "b"])
        assert tac.operator == "&"

    def test_or(self):
        tac, _ = self._convert_single("OR", stack=["a", "b"])
        assert tac.operator == "|"

    def test_xor(self):
        tac, _ = self._convert_single("XOR", stack=["a", "b"])
        assert tac.operator == "^"

    def test_shl(self):
        tac, _ = self._convert_single("SHL", stack=["a", "b"])
        assert tac.operator == "<<"

    def test_shr(self):
        tac, _ = self._convert_single("SHR", stack=["a", "b"])
        assert tac.operator == ">>"

    def test_sar(self):
        tac, _ = self._convert_single("SAR", stack=["a", "b"])
        assert tac.operator == "sar"

    # --- Binary underflow guard ---
    def test_binary_op_underflow(self):
        tac, stack = self._convert_single("ADD", stack=["a"])
        assert tac.operand1 == "stack_underflow" or tac.operand2 == "stack_underflow"
        assert len(stack) == 1

    # --- Unary ops ---
    def test_iszero(self):
        tac, stack = self._convert_single("ISZERO", stack=["a"])
        assert tac.operation == TACOperationType.UNARY_OP
        assert tac.operator == "!"
        assert len(stack) == 1

    def test_not(self):
        tac, _ = self._convert_single("NOT", stack=["a"])
        assert tac.operator == "~"

    # --- Ternary ops ---
    def test_addmod(self):
        tac, stack = self._convert_single("ADDMOD", stack=["a", "b", "c"])
        assert tac.operation == TACOperationType.BINARY_OP
        assert "addmod" in tac.operand1
        assert tac.operand1 == "addmod(a, b, c)"
        assert len(stack) == 1  # 3 popped, 1 pushed

    def test_mulmod(self):
        tac, stack = self._convert_single("MULMOD", stack=["a", "b", "c"])
        assert "mulmod" in tac.operand1
        assert tac.operand1 == "mulmod(a, b, c)"
        assert len(stack) == 1

    @pytest.mark.parametrize(
        "opcode, operator",
        [
            ("03", "-"),
            ("04", "/"),
            ("10", "<"),
            ("11", ">"),
            ("1b", "<<"),
            ("1c", ">>"),
        ],
    )
    def test_non_commutative_bytecode_operand_order(self, opcode, operator):
        analyzer = BytecodeAnalyzer("0x60056002" + opcode + "00")
        tac = analyzer.convert_to_tac()
        formatted = [analyzer._format_tac_instruction(t) for t in tac]
        assert f"temp_3 = temp_1 {operator} temp_2" in formatted

    # --- Memory ops ---
    def test_mload(self):
        tac, stack = self._convert_single("MLOAD", stack=["addr"])
        assert tac.operation == TACOperationType.LOAD
        assert tac.metadata['memory_type'] == 'memory'
        assert len(stack) == 1

    def test_mstore(self):
        tac, stack = self._convert_single("MSTORE", stack=["addr", "val"])
        assert tac.operation == TACOperationType.STORE
        assert tac.metadata['memory_type'] == 'memory'
        assert len(stack) == 0

    def test_mstore8(self):
        tac, _ = self._convert_single("MSTORE8", stack=["addr", "val"])
        assert tac.metadata['memory_type'] == 'memory8'

    # --- Storage ops ---
    def test_sload(self):
        tac, stack = self._convert_single("SLOAD", stack=["key"])
        assert tac.operation == TACOperationType.LOAD
        assert tac.metadata['memory_type'] == 'storage'
        assert len(stack) == 1

    def test_sstore(self):
        tac, stack = self._convert_single("SSTORE", stack=["key", "val"])
        assert tac.operation == TACOperationType.STORE
        assert tac.metadata['memory_type'] == 'storage'
        assert len(stack) == 0

    def test_transient_storage_opcodes(self):
        load_tac = BytecodeAnalyzer("0x60015c00").convert_to_tac()
        assert load_tac[1].operation == TACOperationType.LOAD
        assert load_tac[1].metadata["original_op"] == "TLOAD"
        assert load_tac[1].metadata["memory_type"] == "transient_storage"

        store_tac = BytecodeAnalyzer("0x600260015d00").convert_to_tac()
        assert store_tac[2].operation == TACOperationType.STORE
        assert store_tac[2].metadata["original_op"] == "TSTORE"
        assert "stack_underflow" not in {
            store_tac[2].operand1,
            store_tac[2].operand2,
        }

    # --- SHA3 / KECCAK256 ---
    def test_sha3(self):
        tac, stack = self._convert_single("SHA3", stack=["off", "sz"])
        assert tac.operation == TACOperationType.UNARY_OP
        assert "keccak256" in tac.operand1
        assert len(stack) == 1

    def test_keccak256(self):
        tac, _ = self._convert_single("KECCAK256", stack=["off", "sz"])
        assert "keccak256" in tac.operand1

    # --- CALLDATALOAD ---
    def test_calldataload(self):
        tac, stack = self._convert_single("CALLDATALOAD", stack=["off"])
        assert tac.operation == TACOperationType.LOAD
        assert tac.metadata['memory_type'] == 'calldata'
        assert len(stack) == 1

    # --- Copy ops ---
    def test_calldatacopy(self):
        tac, stack = self._convert_single("CALLDATACOPY", stack=["d", "s", "l"])
        assert tac.operation == TACOperationType.STORE
        assert len(stack) == 0

    def test_codecopy(self):
        tac, _ = self._convert_single("CODECOPY", stack=["d", "s", "l"])
        assert tac.operation == TACOperationType.STORE

    def test_returndatacopy(self):
        tac, _ = self._convert_single("RETURNDATACOPY", stack=["d", "s", "l"])
        assert tac.operation == TACOperationType.STORE

    def test_mcopy_opcode(self):
        tac = BytecodeAnalyzer("0x6003600260015e00").convert_to_tac()
        assert tac[3].operation == TACOperationType.STORE
        assert tac[3].metadata["original_op"] == "MCOPY"
        assert "stack_underflow" not in tac[3].operand2

    def test_extcodecopy(self):
        tac, stack = self._convert_single("EXTCODECOPY", stack=["a", "d", "s", "l"])
        assert tac.operation == TACOperationType.STORE
        assert len(stack) == 0

    # --- Environmental (0-pop, 1-push) ---
    def test_caller(self):
        tac, stack = self._convert_single("CALLER")
        assert tac.operation == TACOperationType.ASSIGN
        assert tac.operand1 == "caller"
        assert len(stack) == 1

    def test_callvalue(self):
        tac, _ = self._convert_single("CALLVALUE")
        assert tac.operand1 == "callvalue"

    def test_address(self):
        tac, _ = self._convert_single("ADDRESS")
        assert tac.operand1 == "address"

    def test_timestamp(self):
        tac, _ = self._convert_single("TIMESTAMP")
        assert tac.operand1 == "timestamp"

    def test_gas(self):
        tac, _ = self._convert_single("GAS")
        assert tac.operand1 == "gas"

    def test_blob_opcodes(self):
        blobhash_tac = BytecodeAnalyzer("0x60014900").convert_to_tac()
        assert blobhash_tac[1].operation == TACOperationType.UNARY_OP
        assert blobhash_tac[1].metadata["original_op"] == "BLOBHASH"

        blobbasefee_tac = BytecodeAnalyzer("0x4a00").convert_to_tac()
        assert blobbasefee_tac[0].operation == TACOperationType.ASSIGN
        assert blobbasefee_tac[0].operand1 == "blobbasefee"
        assert blobbasefee_tac[0].metadata["original_op"] == "BLOBBASEFEE"

    # --- 1-pop, 1-push environment ---
    def test_balance(self):
        tac, stack = self._convert_single("BALANCE", stack=["addr"])
        assert tac.operation == TACOperationType.UNARY_OP
        assert tac.operator == "balance"
        assert len(stack) == 1

    def test_extcodesize(self):
        tac, _ = self._convert_single("EXTCODESIZE", stack=["addr"])
        assert tac.operator == "extcodesize"

    def test_blockhash(self):
        tac, _ = self._convert_single("BLOCKHASH", stack=["num"])
        assert tac.operator == "blockhash"

    # --- Control flow ---
    def test_jump(self):
        tac, stack = self._convert_single("JUMP", stack=["target"])
        assert tac.operation == TACOperationType.JUMP
        assert tac.target == "target"
        assert len(stack) == 0

    def test_jumpi(self):
        # EVM JUMPI: stack is [..., dest, cond] — cond on top, dest below
        tac, stack = self._convert_single("JUMPI", stack=["cond", "target"])
        assert tac.operation == TACOperationType.CONDITIONAL_JUMP
        assert tac.target == "target"
        assert tac.operand1 == "cond"
        assert len(stack) == 0

    # --- RETURN / REVERT ---
    def test_return(self):
        tac, stack = self._convert_single("RETURN", stack=["off", "sz"])
        assert tac.operation == TACOperationType.RETURN
        assert len(stack) == 0

    def test_revert(self):
        tac, stack = self._convert_single("REVERT", stack=["off", "sz"])
        assert tac.operation == TACOperationType.REVERT
        assert len(stack) == 0

    @pytest.mark.parametrize(
        "operand",
        [
            "cf479181",
            "0xcf479181",
            0xCF479181,
            "00000000000000000000000000000000000000000000000000000000cf479181",
            "cf47918100000000000000000000000000000000000000000000000000000000",
        ],
    )
    def test_custom_error_decode_operand_variants(self, operand):
        class ErrorInfo:
            name = "InsufficientBalance"
            selector = "0xcf479181"
            input_types = ["uint256", "uint256"]
            input_names = ["required", "available"]

        class Enricher:
            errors = {"0xcf479181": ErrorInfo()}

            def get_error(self, selector):
                return self.errors.get(selector)

        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE, abi_enricher=Enricher())
        analyzer.instructions = [
            {"name": "PUSH4", "operand": operand, "pc": 0},
            {"name": "MSTORE", "pc": 1},
            {"name": "REVERT", "pc": 2},
        ]
        analyzer._pc_to_index = {0: 0, 1: 1, 2: 2}
        decoded = analyzer._decode_revert_data(analyzer.instructions[-1], [])
        assert decoded["type"] == "CustomError"
        assert decoded["name"] == "InsufficientBalance"

    # --- STOP / SELFDESTRUCT / INVALID ---
    def test_stop(self):
        tac, stack = self._convert_single("STOP")
        assert tac.operation == TACOperationType.HALT
        assert len(stack) == 0

    def test_selfdestruct(self):
        tac, stack = self._convert_single("SELFDESTRUCT", stack=["ben"])
        assert tac.operation == TACOperationType.HALT
        assert tac.operand1 == "ben"
        assert len(stack) == 0

    def test_invalid(self):
        tac, _ = self._convert_single("INVALID")
        assert tac.operation == TACOperationType.HALT

    # --- CALL variants ---
    def test_call(self):
        s = ["gas", "addr", "val", "ao", "al", "ro", "rl"]
        tac, stack = self._convert_single("CALL", stack=s)
        assert tac.operation == TACOperationType.CALL
        assert len(stack) == 1  # 7 popped, 1 pushed

    def test_staticcall(self):
        s = ["gas", "addr", "ao", "al", "ro", "rl"]
        tac, stack = self._convert_single("STATICCALL", stack=s)
        assert tac.operation == TACOperationType.CALL
        assert len(stack) == 1  # 6 popped, 1 pushed

    def test_delegatecall(self):
        s = ["gas", "addr", "ao", "al", "ro", "rl"]
        tac, stack = self._convert_single("DELEGATECALL", stack=s)
        assert len(stack) == 1

    # --- CREATE / CREATE2 ---
    def test_create(self):
        tac, stack = self._convert_single("CREATE", stack=["v", "o", "l"])
        assert tac.operation == TACOperationType.CALL
        assert len(stack) == 1

    def test_create2(self):
        tac, stack = self._convert_single("CREATE2", stack=["v", "o", "l", "s"])
        assert tac.operation == TACOperationType.CALL
        assert len(stack) == 1

    # --- LOG0..LOG4 ---
    def test_log0(self):
        tac, stack = self._convert_single("LOG0", stack=["off", "sz"])
        assert tac.operation == TACOperationType.LOG
        assert tac.metadata['topic_count'] == 0
        assert len(stack) == 0

    def test_log2(self):
        tac, stack = self._convert_single("LOG2", stack=["off", "sz", "t1", "t2"])
        assert tac.metadata['topic_count'] == 2
        assert len(tac.metadata['topics']) == 2
        assert len(stack) == 0

    def test_log4(self):
        tac, stack = self._convert_single("LOG4",
                                          stack=["off", "sz", "t1", "t2", "t3", "t4"])
        assert tac.metadata['topic_count'] == 4
        assert len(stack) == 0

    # --- Fallback / unknown opcode ---
    def test_unknown_opcode_preserves_stack(self):
        """Completely unknown opcodes should not invent a stack push."""
        tac, stack = self._convert_single("SOME_FUTURE_OP")
        assert tac is not None
        assert tac.metadata.get('unhandled') is True
        assert tac.operation == TACOperationType.NOP
        assert len(stack) == 0


# ---------------------------------------------------------------------------
# 8. Stack Simulator Tests
# ---------------------------------------------------------------------------

class TestStackSimulator:
    def _make_sim(self):
        return BytecodeAnalyzer._StackSimulator()

    def test_push_value(self):
        sim = self._make_sim()
        instr = type('I', (), {'name': 'PUSH1', 'operand': '0x0a'})()
        sim.process_instruction(instr, 0)
        assert sim.get_stack_top_value() == 10

    def test_pop(self):
        sim = self._make_sim()
        instr_push = type('I', (), {'name': 'PUSH1', 'operand': '0x01'})()
        instr_pop = type('I', (), {'name': 'POP'})()
        sim.process_instruction(instr_push, 0)
        sim.process_instruction(instr_pop, 1)
        assert sim.get_stack_top_value() is None

    def test_dup(self):
        sim = self._make_sim()
        instr = type('I', (), {'name': 'PUSH1', 'operand': '0x05'})()
        sim.process_instruction(instr, 0)
        dup = type('I', (), {'name': 'DUP1'})()
        sim.process_instruction(dup, 1)
        assert len(sim.stack) == 2
        assert sim.stack[-1] == 5

    def test_swap(self):
        sim = self._make_sim()
        p1 = type('I', (), {'name': 'PUSH1', 'operand': '0x01'})()
        p2 = type('I', (), {'name': 'PUSH1', 'operand': '0x02'})()
        sw = type('I', (), {'name': 'SWAP1'})()
        sim.process_instruction(p1, 0)
        sim.process_instruction(p2, 1)
        sim.process_instruction(sw, 2)
        assert sim.stack[-1] == 1
        assert sim.stack[-2] == 2

    def test_addmod_pops_three(self):
        sim = self._make_sim()
        for i in range(3):
            p = type('I', (), {'name': 'PUSH1', 'operand': f'0x0{i}'})()
            sim.process_instruction(p, i)
        assert len(sim.stack) == 3
        am = type('I', (), {'name': 'ADDMOD'})()
        sim.process_instruction(am, 3)
        assert len(sim.stack) == 1  # 3 popped, 1 pushed

    def test_mulmod_pops_three(self):
        sim = self._make_sim()
        for i in range(3):
            p = type('I', (), {'name': 'PUSH1', 'operand': f'0x0{i}'})()
            sim.process_instruction(p, i)
        mm = type('I', (), {'name': 'MULMOD'})()
        sim.process_instruction(mm, 3)
        assert len(sim.stack) == 1

    def test_binary_op(self):
        sim = self._make_sim()
        p1 = type('I', (), {'name': 'PUSH1', 'operand': '0x01'})()
        p2 = type('I', (), {'name': 'PUSH1', 'operand': '0x02'})()
        add = type('I', (), {'name': 'ADD'})()
        sim.process_instruction(p1, 0)
        sim.process_instruction(p2, 1)
        sim.process_instruction(add, 2)
        assert len(sim.stack) == 1

    def test_stack_effects_table_coverage(self):
        """All entries in _EVM_STACK_EFFECTS should be handled by the simulator."""
        sim = self._make_sim()
        for opname, (pops, pushes) in _EVM_STACK_EFFECTS.items():
            # Skip opcodes that have explicit handlers in the simulator
            if opname in ('POP',):
                continue
            sim.stack = [None] * max(pops, 1)
            instr = type('I', (), {'name': opname})()
            sim.process_instruction(instr, 0)
            # Should not crash


# ---------------------------------------------------------------------------
# 9. Formatted Output Tests
# ---------------------------------------------------------------------------

class TestFormattedOutput:
    def test_format_tac_assign(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        instr = TACInstruction(operation=TACOperationType.ASSIGN, result="t1", operand1="0x42")
        out = analyzer._format_tac_instruction(instr)
        assert "t1" in out and "0x42" in out

    def test_format_tac_binary(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        instr = TACInstruction(operation=TACOperationType.BINARY_OP,
                               result="t1", operand1="a", operand2="b", operator="+")
        out = analyzer._format_tac_instruction(instr)
        assert "t1 = a + b" == out

    def test_format_tac_unary(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        instr = TACInstruction(operation=TACOperationType.UNARY_OP,
                               result="t1", operand1="x", operator="!")
        out = analyzer._format_tac_instruction(instr)
        assert "t1 = !(x)" == out

    def test_format_tac_load(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        instr = TACInstruction(operation=TACOperationType.LOAD,
                               result="t1", operand1="addr",
                               metadata={'memory_type': 'storage'})
        out = analyzer._format_tac_instruction(instr)
        assert "storage[addr]" in out

    def test_format_tac_store(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        instr = TACInstruction(operation=TACOperationType.STORE,
                               operand1="k", operand2="v",
                               metadata={'memory_type': 'storage'})
        out = analyzer._format_tac_instruction(instr)
        assert "storage[k] = v" == out

    def test_format_tac_jump(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        instr = TACInstruction(operation=TACOperationType.JUMP, target="0x10")
        assert "goto 0x10" == analyzer._format_tac_instruction(instr)

    def test_format_tac_cond_jump(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        instr = TACInstruction(operation=TACOperationType.CONDITIONAL_JUMP,
                               target="0x10", operand1="cond")
        assert "if cond goto 0x10" == analyzer._format_tac_instruction(instr)

    def test_format_tac_return(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        instr = TACInstruction(operation=TACOperationType.RETURN,
                               operand1="off", operand2="sz")
        assert "return memory[off:sz]" == analyzer._format_tac_instruction(instr)

    def test_format_tac_revert(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        instr = TACInstruction(operation=TACOperationType.REVERT,
                               operand1="off", operand2="sz")
        assert "revert memory[off:sz]" == analyzer._format_tac_instruction(instr)

    def test_format_tac_halt(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        instr = TACInstruction(operation=TACOperationType.HALT,
                               metadata={'original_op': 'STOP'})
        assert "stop()" == analyzer._format_tac_instruction(instr)

    def test_format_tac_log(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        instr = TACInstruction(operation=TACOperationType.LOG,
                               operand1="off", operand2="sz",
                               metadata={'topic_count': 2})
        out = analyzer._format_tac_instruction(instr)
        assert "log2" in out

    def test_format_tac_call(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        instr = TACInstruction(operation=TACOperationType.CALL,
                               result="t1", operand1="addr",
                               metadata={'original_op': 'STATICCALL'})
        out = analyzer._format_tac_instruction(instr)
        assert "staticcall" in out


# ---------------------------------------------------------------------------
# 10. End-to-End / Integration Tests
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_analyze_bytecode_to_tac_convenience(self):
        output = analyze_bytecode_to_tac(MINIMAL_BYTECODE)
        assert isinstance(output, str)
        assert "Three-Address Code" in output

    def test_full_pipeline_sample_contract(self):
        output = analyze_bytecode_to_tac(SAMPLE_OWNER_BYTECODE)
        assert "Analysis Summary" in output
        assert "Basic blocks:" in output
        assert "Functions identified:" in output
        # Should have identified the 2 known selectors
        assert "0x893d20e8" in output
        assert "0xa6f9dae1" in output

    def test_generate_tac_representation(self):
        analyzer = BytecodeAnalyzer(SAMPLE_OWNER_BYTECODE)
        output = analyzer.generate_tac_representation()
        assert len(output) > 100  # Non-trivial output
        # Should contain block and function info
        assert "block_" in output
        assert "function" in output.lower()

    def test_convert_to_tac_list(self):
        analyzer = BytecodeAnalyzer(SAMPLE_OWNER_BYTECODE)
        tac_list = analyzer.convert_to_tac()
        assert len(tac_list) > 0
        assert all(isinstance(t, TACInstruction) for t in tac_list)

    def test_fallback_tac_on_empty(self):
        analyzer = BytecodeAnalyzer("")
        output = analyzer.generate_tac_representation()
        assert isinstance(output, str)

    def test_block_tac_integration(self):
        """After full pipeline, each non-empty block should have TAC instructions."""
        analyzer = BytecodeAnalyzer(SAMPLE_OWNER_BYTECODE)
        analyzer.generate_tac_representation()
        blocks_with_tac = [b for b in analyzer.basic_blocks.values() if b.instructions]
        assert len(blocks_with_tac) > 0

    def test_block_metadata_after_integration(self):
        """After integration, blocks should have block_type metadata."""
        analyzer = BytecodeAnalyzer(SAMPLE_OWNER_BYTECODE)
        analyzer.generate_tac_representation()
        for block in analyzer.basic_blocks.values():
            assert 'block_type' in block.metadata
            assert block.metadata['block_type'] in ('exit', 'sequential', 'conditional', 'complex')

    def test_stack_propagates_across_jump_blocks(self):
        analyzer = BytecodeAnalyzer("0x60056005565b60020300")
        output = analyzer.generate_tac_representation()
        assert "stack_underflow - stack_underflow" not in output
        assert "temp_4 = temp_1 - temp_3" in output

    def test_stack_merge_uses_phi_for_conflicting_predecessors(self):
        bytecode = "0x600a6001600e57506002600e56005b60010100"
        analyzer = BytecodeAnalyzer(bytecode)
        output = analyzer.generate_tac_representation()
        assert "stack_underflow" not in output
        assert "phi_block_000e_0" in output


# ---------------------------------------------------------------------------
# 11. Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_get_next_instruction_pc_at_end(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        if analyzer.instructions:
            last_pc = analyzer._get_pc(analyzer.instructions[-1], 999)
            result = analyzer._get_next_instruction_pc(last_pc)
            assert result is None

    def test_get_next_instruction_pc_invalid(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        result = analyzer._get_next_instruction_pc(99999)
        assert result is None

    def test_add_edge_ignores_missing_blocks(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        blocks = {}
        # Should not raise
        analyzer._add_edge("nonexistent_a", "nonexistent_b", blocks)

    def test_temp_var_uniqueness(self):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        vars_generated = {analyzer._generate_temp_var() for _ in range(100)}
        assert len(vars_generated) == 100

    def test_dict_instruction_handling(self):
        """Analyzer helpers should handle dict-format instructions."""
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        d = {"name": "ADD", "pc": 5, "operand": "0x10"}
        assert analyzer._get_instruction_name(d) == "ADD"
        assert analyzer._get_pc(d, 0) == 5
        assert analyzer._get_operand(d) == "0x10"

    def test_no_imports_of_removed_modules(self):
        """Ensure json, to_hex, Web3 are not imported."""
        import src.bytecode_analyzer as module
        source = open(module.__file__, 'r').read()
        # Should not have these unused imports
        assert "import json" not in source
        assert "from eth_utils import to_hex" not in source
        assert "from web3 import Web3" not in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])