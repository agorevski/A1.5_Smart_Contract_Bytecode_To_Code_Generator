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
    _BINARY_OPS,
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

# solc 0.8.20, optimizer enabled (200 runs): public stored getter + update helper.
OPTIMIZED_GETTER_BYTECODE = (
    "0x6080604052348015600e575f80fd5b50600436106030575f3560e01c806382ab890a"
    "146034578063e582dd31146045575b5f80fd5b6043603f366004607c565b605e565b"
    "005b604c5f5481565b60405190815260200160405180910390f35b5f606682606c565b"
    "5f555050565b5f60768260016092565b92915050565b5f60208284031215608b575f80"
    "fd5b5035919050565b80820180821115607657634e487b7160e01b5f526011600452"
    "60245ffdfea2646970667358221220716cbfad7bed540550d4fcbeb66cd0fc5110f7e"
    "9b20a9d062f6a203614de258c64736f6c63430008140033"
)

# solc 0.8.20, optimizer 200, Shanghai, metadata disabled: read3(uint256)
# returns stored + x + 3; write3(uint256) stores x + 3.
PARAMETERIZED_GETTER_BYTECODE = (
    "0x6080604052348015600e575f80fd5b50600436106026575f3560e01c80630342c79d"
    "14602a575b5f80fd5b603960353660046066565b604b565b604051908152602001604051"
    "80910390f35b5f815f5460579190607c565b6060906003607c565b92915050565b5f60"
    "2082840312156075575f80fd5b5035919050565b80820180821115606057634e487b71"
    "60e01b5f52601160045260245ffd"
)
PARAMETERIZED_SETTER_BYTECODE = (
    "0x6080604052348015600e575f80fd5b50600436106026575f3560e01c80631e9e15ab"
    "14602a575b5f80fd5b603960353660046049565b603b565b005b6044816003605f565b"
    "5f5550565b5f602082840312156058575f80fd5b5035919050565b8082018082111560"
    "7d57634e487b7160e01b5f52601160045260245ffd5b9291505056"
)
SHARED_PARAMETER_DECODER_BYTECODE = (
    "0x6080604052348015600e575f80fd5b50600436106030575f3560e01c80633f81a2c0"
    "1460345780634d0392a8146045575b5f80fd5b6043603f3660046086565b6066565b"
    "005b605460503660046086565b6074565b60405190815260200160405180910390f35b60"
    "6f816001609c565b5f5550565b5f815f5460809190609c565b92915050565b5f602082"
    "840312156095575f80fd5b5035919050565b80820180821115608057634e487b7160e0"
    "1b5f52601160045260245ffd"
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

    def test_large_dispatcher_identifies_selectors_beyond_256_instructions(self):
        prefix = "60003560e01c"
        entries = "".join(f"8063{selector:08x}1461029a57" for selector in range(1, 61))
        analyzer = BytecodeAnalyzer("0x" + prefix + entries + "5b00")
        analyzer.analyze_control_flow()
        functions = analyzer.identify_functions()

        assert len(analyzer.instructions) > 256
        assert sum(func.selector is not None for func in functions.values()) == 60
        assert "function_0x0000003c" in functions
        assert "block_029a:" in analyzer.generate_per_function_tac()["function_0x0000003c"]

    def test_large_dispatcher_target_after_jumpdest_is_not_a_function(self):
        prefix = "60003560e01c"
        # The original 60-selector fixture pointed at STOP (0x29b), not JUMPDEST (0x29a).
        entries = "".join(f"8063{selector:08x}1461029b57" for selector in range(1, 61))
        tac = BytecodeAnalyzer("0x" + prefix + entries + "5b00").generate_per_function_tac()

        assert set(tac) == {"fallback_function"}
        assert "block_029a:" in tac["fallback_function"]

    @pytest.mark.parametrize("target", ["11", "20", "99"])
    def test_invalid_dispatcher_target_does_not_create_function(self, target):
        # The fallback JUMPDEST is at 0x10; 0x11 is STOP, and 0x20/0x99 are absent.
        analyzer = BytecodeAnalyzer("0x60003560e01c8063123456781460" + target + "575b00")
        tac = analyzer.generate_per_function_tac()

        assert set(analyzer.basic_blocks) == {"block_0000", "block_0010"}
        assert "function_0x12345678" not in tac
        assert analyzer.rejected_dispatcher_targets == {"0x12345678": int(target, 16)}
        assert set(tac) == {"fallback_function"}
        assert "block_0010:" in tac["fallback_function"]
        assert "block_0000:" not in tac["fallback_function"]
        fake = Function(
            name="function_0x12345678", selector="0x12345678",
            basic_blocks=list(analyzer.basic_blocks.values()), entry_block=f"block_00{target}",
        )
        assert analyzer.generate_function_tac(fake) == ""
        assert analyzer._blocks_for_function(fake, fallback_to_all=True) == []

    def test_valid_dispatcher_target_includes_only_its_reachable_block(self):
        analyzer = BytecodeAnalyzer("0x60003560e01c806312345678146010575b00")
        tac = analyzer.generate_per_function_tac()["function_0x12345678"]

        assert analyzer.functions["function_0x12345678"].entry_block == "block_0010"
        assert [b.id for b in analyzer._blocks_for_function(
            analyzer.functions["function_0x12345678"], fallback_to_all=False
        )] == ["block_0010"]
        assert "block_0010:" in tac
        assert "block_0000:" not in tac

    def test_invalid_selector_does_not_hide_valid_receive_and_fallback(self):
        # receive() at 0x17, fallback at 0x15; selector points outside code.
        bytecode = "0x361560175760003560e01c8063123456781460ff575b005b005b00"
        tac = BytecodeAnalyzer(bytecode).generate_per_function_tac()

        assert "function_0x12345678" not in tac
        assert "block_0017:" in tac["receive"]
        assert "block_0015:" in tac["fallback_function"]

    def test_invalid_selector_without_fallthrough_has_no_function_tac(self):
        analyzer = BytecodeAnalyzer("0x60003560e01c80631234567814602057")
        assert analyzer.generate_per_function_tac() == {}

    @pytest.mark.parametrize("target", ["05", "99"])
    def test_invalid_receive_target_does_not_create_receive_function(self, target):
        invalid = BytecodeAnalyzer("0x361560" + target + "57005b00")
        assert "receive" not in invalid.generate_per_function_tac()
        valid = BytecodeAnalyzer("0x3615600657005b00")
        assert "block_0006:" in valid.generate_per_function_tac()["receive"]

    def test_legacy_div_dispatcher_identifies_selector(self):
        divisor = "01" + ("00" * 28)
        bytecode = "0x7c" + divisor + "6000350480631234567814602c575b00"
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
        assert tac.operand1 == "b"
        assert tac.operand2 == "a"

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
        assert tac.operand1 == "addmod(c, b, a)"
        assert len(stack) == 1  # 3 popped, 1 pushed

    def test_mulmod(self):
        tac, stack = self._convert_single("MULMOD", stack=["a", "b", "c"])
        assert "mulmod" in tac.operand1
        assert tac.operand1 == "mulmod(c, b, a)"
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
        left, right = ("temp_1", "temp_2") if opcode in ("1b", "1c") else ("temp_2", "temp_1")
        assert f"temp_3 = {left} {operator} {right}" in formatted

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

    def test_optimized_getter_tracks_jump_target_below_consumed_push0(self):
        analyzer = BytecodeAnalyzer(OPTIMIZED_GETTER_BYTECODE)
        functions = analyzer.generate_per_function_tac()

        getter = functions["function_0xe582dd31"]
        assert analyzer.basic_blocks["block_0045"].successors == ["block_004c"]
        assert "block_004c:" in getter
        assert "return memory[" in getter
        assert "stack_underflow" not in getter
        assert "block_004c:" not in functions["function_0x82ab890a"]
        assert not any(line.strip().startswith("storage[") for line in getter.splitlines())
        assert any(
            line.strip().startswith("storage[")
            for line in functions["function_0x82ab890a"].splitlines()
        )
        assert "stack_underflow" not in functions["function_0x82ab890a"]

    @pytest.mark.parametrize(
        "bytecode,selector,handoff,witness,rendered",
        [
            (PARAMETERIZED_GETTER_BYTECODE, "0x0342c79d", "block_0075",
             "block_0039", "return memory["),
            (PARAMETERIZED_SETTER_BYTECODE, "0x1e9e15ab", "block_0058",
             "block_0044", "storage["),
        ],
    )
    def test_compiler_parameter_decoder_returns_to_function(
        self, bytecode, selector, handoff, witness, rendered
    ):
        analyzer = BytecodeAnalyzer(bytecode)
        tac = analyzer.generate_per_function_tac()["function_" + selector]

        assert analyzer.basic_blocks[handoff].successors == ["block_0035"]
        assert witness + ":" in tac
        assert rendered in tac
        assert "stack_underflow" not in tac
        for block in analyzer.basic_blocks.values():
            raw = block.metadata["raw_instructions"]
            if analyzer._get_instruction_name(raw[-1]) == "JUMP":
                assert all(
                    analyzer._get_instruction_name(
                        analyzer.basic_blocks[successor].metadata["raw_instructions"][0]
                    ) == "JUMPDEST"
                    for successor in block.successors
                )

    def test_shared_parameter_decoder_keeps_selector_paths_separate(self):
        analyzer = BytecodeAnalyzer(SHARED_PARAMETER_DECODER_BYTECODE)
        tac = analyzer.generate_per_function_tac()
        setter = tac["function_0x3f81a2c0"]
        getter = tac["function_0x4d0392a8"]

        assert set(analyzer.basic_blocks["block_0095"].successors) == {
            "block_003f", "block_0050",
        }
        assert "block_003f:" in setter and "block_0050:" not in setter
        assert "block_0050:" in getter and "block_003f:" not in getter
        assert "  // Successors: block_003f" in setter
        assert "  // Successors: block_0050" in getter
        assert any(line.strip().startswith("storage[") for line in setter.splitlines())
        assert not any(line.strip().startswith("storage[") for line in getter.splitlines())
        assert "return memory[" in getter

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
        subtraction = next(t for b in analyzer.basic_blocks.values()
                           for t in b.instructions if t.operator == "-")
        assert subtraction.operand1 != subtraction.operand2
        assert subtraction.operand2 == "temp_1"

    def test_stack_merge_uses_phi_for_conflicting_predecessors(self):
        bytecode = "0x600a6001600e57506002600e56005b60010100"
        analyzer = BytecodeAnalyzer(bytecode)
        output = analyzer.generate_tac_representation()
        assert "stack_underflow" not in output
        assert "phi_block_000e_0" in output
        phi = next(t for t in analyzer.basic_blocks["block_000e"].instructions
                   if t.operation == TACOperationType.PHI)
        assert set(phi.metadata["incoming"]) == {"block_0000", "block_0007"}
        assert len(set(phi.metadata["incoming"].values())) == 2
        definitions = {t.result for b in analyzer.basic_blocks.values()
                       for t in b.instructions if t.result}
        assert set(phi.metadata["incoming"].values()) <= definitions


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


class TestFaithfulEVMSemantics:
    # solc 0.8.20, optimizer enabled (200 runs), including original CBOR metadata.
    @pytest.mark.parametrize("contract,bytecode,expected_functions", [
        ("Base",
         "6080604052348015600e575f80fd5b50600436106026575f3560e01c806326121ff014602a575b5f80fd"
         "5b600160405190815260200160405180910390f3fea2646970667358221220bb33626db79f71555b404882"
         "13b2791252bdbc90761389bf6f590b8e8c150c4a64736f6c63430008140033",
         [("f", "0x26121ff0", "Base", "function f() public pure virtual returns (uint) { return 1; }")]),
        ("Derived",
         "6080604052348015600e575f80fd5b50600436106030575f3560e01c806326121ff0146034578063e2179b8e"
         "146049575b5f80fd5b60015b60405190815260200160405180910390f35b6003603756fea264697066735822"
         "122013c372b4a92ee7478c8923b107cde3cffcdac64f38605135558f432434d5d64b64736f6c63430008140033",
         [("f", "0x26121ff0", "Base", "function f() public pure virtual returns (uint) { return 1; }"),
          ("g", "0xe2179b8e", "Derived", "function g() public pure returns (uint) { return 3; }")]),
        ("Other",
         "6080604052348015600e575f80fd5b50600436106026575f3560e01c806326121ff014602a575b5f80fd"
         "5b600260405190815260200160405180910390f3fea2646970667358221220f396cc60c9f133d3a10f2c4aad"
         "87d60608775be3a61a82085a232ed72b04a8b564736f6c63430008140033",
         [("f", "0x26121ff0", "Other", "function f() public pure returns (uint) { return 2; }")]),
    ])
    def test_compiled_inheritance_runtime_survives_strict_matching(self, contract, bytecode, expected_functions):
        from src.dataset_export_primitives import match_functions_by_selector

        analyzer = BytecodeAnalyzer(bytecode)
        analyzer.analyze_control_flow()
        functions = analyzer.identify_functions()
        labels = [{"name": name, "selector": selector, "contract_name": declaring, "body": body}
                  for name, selector, declaring, body in expected_functions]
        matches = match_functions_by_selector(labels, functions, analyzer)
        assert analyzer.analysis_status == {"status": "ok", "issues": [], "schema_version": 2}, contract
        assert len(matches) == len(expected_functions) >= 1
        assert {m["selector"]: m["solidity_function"]["body"] for m in matches} == {
            selector: body for _, selector, _, body in expected_functions
        }
        assert all(m["tac"] for m in matches)
        dead = [b for b in analyzer.basic_blocks.values() if b.metadata["is_dead_code"]]
        assert dead
        assert all(not b.instructions for b in dead)

    def test_terminal_join_keeps_common_top_suffix_not_unused_prefix(self):
        analyzer = BytecodeAnalyzer("6001600c5760aa60bb6012565b60cc6012565b5100")
        analyzer.generate_tac_representation()
        assert analyzer.analysis_status["status"] == "ok"
        terminal = analyzer.basic_blocks["block_0012"]
        phi = next(t for t in terminal.instructions if t.operation == TACOperationType.PHI)
        values = {t.result: t.operand1 for b in analyzer.basic_blocks.values()
                  for t in b.instructions if t.operation == TACOperationType.ASSIGN}
        assert {int(values[v], 16) for v in phi.metadata["incoming"].values()} == {0xBB, 0xCC}
        load = next(t for t in terminal.instructions if t.operation == TACOperationType.LOAD)
        assert load.operand1 == phi.result

    @pytest.mark.parametrize("bytecode", ["00a201fe", "fea201"])
    def test_unreachable_metadata_is_not_executed(self, bytecode):
        analyzer = BytecodeAnalyzer(bytecode)
        analyzer.generate_tac_representation()
        assert analyzer.analysis_status["status"] == "ok"
        assert all(not b.instructions for b in analyzer.basic_blocks.values()
                   if b.metadata["is_dead_code"])

    def test_reachable_unknown_opcode_remains_degraded(self):
        analyzer = BytecodeAnalyzer("0c00")
        analyzer.generate_tac_representation()
        assert analyzer.analysis_status["status"] == "degraded"
        assert "unsupported_opcode" in {i["code"] for i in analyzer.analysis_status["issues"]}

    def test_terminal_join_with_missing_used_suffix_is_degraded(self):
        # One predecessor supplies MLOAD's address, the other does not.
        analyzer = BytecodeAnalyzer("6001600a5760bb600e565b600e565b5100")
        analyzer.generate_tac_representation()
        assert analyzer.analysis_status["status"] == "degraded"
        assert {"inconsistent_stack_height", "unresolved_edge"} & {
            i["code"] for i in analyzer.analysis_status["issues"]
        }

    @pytest.mark.parametrize("opcode", list(_BINARY_OPS) + ["ADDMOD", "MULMOD"])
    def test_stack_simulator_matches_py_evm(self, opcode):
        arithmetic = pytest.importorskip("eth.vm.logic.arithmetic")
        from eth.vm.logic import comparison
        from eth.vm.stack import Stack
        from types import SimpleNamespace
        from random import Random

        aliases = {"AND": "and_op", "OR": "or_op", "BYTE": "byte_op"}
        reference = getattr(arithmetic, opcode.lower(), None)
        if reference is None:
            reference = getattr(comparison, aliases.get(opcode, opcode.lower()))
        random = Random(73)
        values = [0, 1, 31, 32, 255, 256, 1 << 255, (1 << 256) - 1]
        values.extend(random.getrandbits(256) for _ in range(8))
        for first in values:
            for second in values:
                inputs = [second, first]
                if opcode in ("ADDMOD", "MULMOD"):
                    inputs.insert(0, values[(first + second) % len(values)])
                reference_stack = Stack()
                for value in inputs:
                    reference_stack.push_int(value)
                computation = SimpleNamespace(
                    stack_pop_ints=reference_stack.pop_ints,
                    stack_push_int=reference_stack.push_int,
                    consume_gas=lambda *args, **kwargs: None,
                )
                if opcode == "EXP":
                    reference(computation, gas_per_byte=50)
                else:
                    reference(computation)
                simulator = BytecodeAnalyzer._StackSimulator()
                simulator.stack = inputs
                simulator.process_instruction({"name": opcode}, 0)
                assert simulator.stack == [reference_stack.pop1_int()], (opcode, first, second)

    @pytest.mark.parametrize("opcode,first,second,expected", [
        ("SUB", 2, 5, (1 << 256) - 3),
        ("DIV", 8, 3, 2),
        ("DIV", 8, 0, 0),
        ("MOD", 8, 3, 2),
        ("SDIV", (1 << 256) - 8, 3, (1 << 256) - 2),
        ("SDIV", 1 << 255, (1 << 256) - 1, 1 << 255),
        ("SMOD", (1 << 256) - 8, 3, (1 << 256) - 2),
        ("EXP", 3, 2, 9),
        ("LT", 2, 5, 1),
        ("GT", 2, 5, 0),
        ("SLT", (1 << 256) - 1, 0, 1),
        ("SGT", (1 << 256) - 1, 0, 0),
        ("SHL", 2, 5, 20),
        ("SHL", 256, 5, 0),
        ("SHR", 2, 20, 5),
        ("SAR", 2, (1 << 256) - 8, (1 << 256) - 2),
        ("SAR", 256, 1 << 255, (1 << 256) - 1),
        ("BYTE", 31, 0x1234, 0x34),
        ("BYTE", 32, 0x1234, 0),
        ("SIGNEXTEND", 0, 0x80, (1 << 256) - 128),
        ("SIGNEXTEND", 32, 0x80, 0x80),
        ("ADD", (1 << 256) - 1, 1, 0),
    ])
    def test_evm_reference_vectors(self, opcode, first, second, expected):
        simulator = BytecodeAnalyzer._StackSimulator()
        simulator.stack = [second, first]
        simulator.process_instruction({"name": opcode}, 0)
        assert simulator.stack == [expected]
        analyzer = BytecodeAnalyzer("")
        tac = analyzer._convert_instruction_to_tac({"name": opcode}, ["second", "first"])
        expected_operands = ("second", "first") if opcode in ("SHL", "SHR", "SAR") else ("first", "second")
        assert (tac.operand1, tac.operand2) == expected_operands
        assert tac.metadata["word_bits"] == 256

    @pytest.mark.parametrize("opcode,stack,expected", [
        ("ADDMOD", [7, 5, 6], 4),
        ("MULMOD", [7, 5, 6], 2),
        ("ADDMOD", [0, 5, 6], 0),
        ("MULMOD", [7, (1 << 256) - 1, (1 << 256) - 1],
         (((1 << 256) - 1) ** 2) % 7),
    ])
    def test_modular_arithmetic_uses_third_pop_modulus(self, opcode, stack, expected):
        simulator = BytecodeAnalyzer._StackSimulator()
        simulator.stack = list(stack)
        simulator.process_instruction({"name": opcode}, 0)
        assert simulator.stack == [expected]

    @pytest.mark.parametrize("opcode", ["CALL", "CALLCODE", "STATICCALL", "DELEGATECALL"])
    def test_call_formatter_preserves_all_arguments(self, opcode):
        analyzer = BytecodeAnalyzer("")
        stack = ["return_length", "return_offset", "input_length", "input_offset"]
        if opcode in ("CALL", "CALLCODE"):
            stack.append("amount")
        stack.extend(["callee", "gas_limit"])
        tac = analyzer._convert_instruction_to_tac({"name": opcode}, stack)
        text = analyzer._format_tac_instruction(tac)
        assert f"{opcode.lower()}(" in text
        for expected in ["gas=gas_limit", "address=callee", "args_offset=input_offset",
                         "args_length=input_length", "ret_offset=return_offset",
                         "ret_length=return_length"]:
            assert expected in text
        assert ("value=amount" in text) == (opcode in ("CALL", "CALLCODE"))
        assert stack == [tac.result]

    def test_loop_phi_contains_forward_and_backedge_definitions(self):
        # Counter at loop header pc=5 is updated and carried through the backedge.
        analyzer = BytecodeAnalyzer("0x60016005565b6001018060055700")
        analyzer.generate_tac_representation()
        header = analyzer.basic_blocks["block_0005"]
        phi = next(t for t in header.instructions if t.operation == TACOperationType.PHI)
        assert set(phi.metadata["incoming"]) == {"block_0000", "block_0005"}
        definitions = {t.result for b in analyzer.basic_blocks.values() for t in b.instructions if t.result}
        assert set(phi.metadata["incoming"].values()) <= definitions
        assert analyzer.analysis_status["status"] == "ok"
        first = analyzer.generate_tac_representation()
        assert first == analyzer.generate_tac_representation()

    def test_reverse_address_predecessor_is_propagated(self):
        # Execution: block 0 -> block 8 -> block 3; not address order.
        analyzer = BytecodeAnalyzer("0x6008565b600103005b6005600356")
        output = analyzer.generate_tac_representation()
        block = analyzer.basic_blocks["block_0003"]
        assert "unresolved_" not in output
        assert block.metadata["entry_stack"] == analyzer.basic_blocks["block_0008"].metadata["exit_stack"]
        assert analyzer.analysis_status["status"] == "ok"

    @pytest.mark.parametrize("bytecode,code", [
        ("0x50", "stack_underflow"),
        ("0x90", "stack_underflow"),
        ("0x60003556", "unresolved_jump"),
        ("0x6000600056", "unresolved_jump"),
        ("ZZZZ", "parse_error"),
    ])
    def test_degradation_is_structured(self, bytecode, code):
        analyzer = BytecodeAnalyzer(bytecode)
        analyzer.generate_tac_representation()
        status = analyzer.get_analysis_status()
        assert status["status"] in ("degraded", "failed")
        assert code in {issue["code"] for issue in status["issues"]}
        assert status["schema_version"] == analyzer.tac_schema_version == 2

    def test_inconsistent_loop_height_is_reported(self):
        analyzer = BytecodeAnalyzer("0x6003565b6001600356")
        analyzer.generate_tac_representation()
        assert "inconsistent_stack_height" in {i["code"] for i in analyzer.analysis_status["issues"]}

    def test_fallback_is_not_success(self, monkeypatch):
        analyzer = BytecodeAnalyzer(MINIMAL_BYTECODE)
        def fail():
            raise RuntimeError("analysis unavailable")
        monkeypatch.setattr(analyzer, "_convert_and_integrate_tac", fail)
        assert "Fallback" in analyzer.generate_tac_representation()
        assert analyzer.analysis_status["status"] == "degraded"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])