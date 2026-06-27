"""
EVM Bytecode to Three-Address Code (TAC) Converter

This module implements static analysis techniques to convert EVM bytecode
into a structured three-address code representation, as described in the paper
"Decompiling Smart Contracts with a Large Language Model" (arXiv:2506.19624v1).

Pipeline:
    1. Parse raw EVM bytecode into disassembled instructions (via evmdasm)
    2. Detect jump targets and construct basic blocks
    3. Build the control-flow graph (predecessors / successors)
    4. Run advanced analyses (dominance, loop detection, reachability)
    5. Identify function boundaries from the dispatcher pattern
    6. Convert each basic block's instructions to Three-Address Code
    7. Emit a human-readable TAC string suitable for LLM input
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

from .abi_enrichment import PANIC_CODES, ERROR_SELECTORS

# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------


class TACOperationType(Enum):
    """Types of operations in Three-Address Code representation."""

    ASSIGN = "assign"
    BINARY_OP = "binary_op"
    UNARY_OP = "unary_op"
    LOAD = "load"
    STORE = "store"
    CALL = "call"
    JUMP = "jump"
    CONDITIONAL_JUMP = "conditional_jump"
    RETURN = "return"
    REVERT = "revert"
    HALT = "halt"
    LOG = "log"
    NOP = "nop"


@dataclass
class TACInstruction:
    """Represents a single Three-Address Code instruction."""

    operation: TACOperationType
    result: Optional[str] = None
    operand1: Optional[str] = None
    operand2: Optional[str] = None
    operator: Optional[str] = None
    target: Optional[str] = None  # For jumps
    metadata: Optional[Dict] = None


@dataclass
class BasicBlock:
    """Represents a basic block in control flow analysis."""

    id: str
    instructions: List[TACInstruction]
    predecessors: List[str]
    successors: List[str]
    start_address: int
    end_address: int
    metadata: Dict = field(default_factory=dict)


@dataclass
class Function:
    """Represents a function in the smart contract."""

    name: str
    selector: Optional[str]  # 4-byte function selector
    basic_blocks: List[BasicBlock]
    entry_block: str
    visibility: str = "public"
    is_payable: bool = False
    is_view: bool = False
    parameters: List[str] = field(default_factory=list)
    return_types: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Module-Level Constants
# ---------------------------------------------------------------------------

_RAW_OPCODE_NAMES: Dict[int, str] = {
    0x49: "BLOBHASH",
    0x4A: "BLOBBASEFEE",
    0x5C: "TLOAD",
    0x5D: "TSTORE",
    0x5E: "MCOPY",
    0x5F: "PUSH0",
}

# EVM opcode stack effects: (pops, pushes) for opcodes handled by the
# generic fallback in the stack simulator and TAC converter.
_EVM_STACK_EFFECTS: Dict[str, Tuple[int, int]] = {
    # Environmental / context opcodes  (0 pops, 1 push)
    "ADDRESS": (0, 1), "ORIGIN": (0, 1), "CALLER": (0, 1),
    "CALLVALUE": (0, 1), "CALLDATASIZE": (0, 1), "CODESIZE": (0, 1),
    "GASPRICE": (0, 1), "RETURNDATASIZE": (0, 1), "COINBASE": (0, 1),
    "TIMESTAMP": (0, 1), "NUMBER": (0, 1), "DIFFICULTY": (0, 1),
    "PREVRANDAO": (0, 1), "GASLIMIT": (0, 1), "CHAINID": (0, 1),
    "SELFBALANCE": (0, 1), "BASEFEE": (0, 1), "BLOBBASEFEE": (0, 1), "GAS": (0, 1),
    "PC": (0, 1), "MSIZE": (0, 1),
    # 1-pop, 1-push
    "BALANCE": (1, 1), "EXTCODESIZE": (1, 1), "EXTCODEHASH": (1, 1),
    "BLOCKHASH": (1, 1), "CALLDATALOAD": (1, 1), "BLOBHASH": (1, 1),
    "TLOAD": (1, 1),
    # 1-pop, 0-push
    "POP": (1, 0),
    # 2-pop, 0-push
    "MSTORE8": (2, 0), "TSTORE": (2, 0),
    # 3-pop, 0-push
    "CALLDATACOPY": (3, 0), "CODECOPY": (3, 0), "RETURNDATACOPY": (3, 0),
    "MCOPY": (3, 0),
    # 4-pop, 0-push
    "EXTCODECOPY": (4, 0),
    # CALL variants
    "CALL": (7, 1), "CALLCODE": (7, 1),
    "DELEGATECALL": (6, 1), "STATICCALL": (6, 1),
    # CREATE variants
    "CREATE": (3, 1), "CREATE2": (4, 1),
    # LOG variants
    "LOG0": (2, 0), "LOG1": (3, 0), "LOG2": (4, 0),
    "LOG3": (5, 0), "LOG4": (6, 0),
    # Halting
    "STOP": (0, 0), "SELFDESTRUCT": (1, 0), "INVALID": (0, 0),
    # SHA3 / KECCAK256
    "SHA3": (2, 1), "KECCAK256": (2, 1),
}

# Binary arithmetic / comparison / bitwise operators and their TAC symbols.
_BINARY_OPS: Dict[str, str] = {
    "ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/",
    "SDIV": "s/", "MOD": "%", "SMOD": "s%", "EXP": "**",
    "SIGNEXTEND": "signextend",
    "LT": "<", "GT": ">", "SLT": "s<", "SGT": "s>", "EQ": "==",
    "AND": "&", "OR": "|", "XOR": "^",
    "BYTE": "byte", "SHL": "<<", "SHR": ">>", "SAR": "sar",
}

# Environmental opcodes that push a named value (0 pops, 1 push).
_ENV_OPS: frozenset = frozenset({
    "ADDRESS", "ORIGIN", "CALLER", "CALLVALUE", "CALLDATASIZE",
    "CODESIZE", "GASPRICE", "RETURNDATASIZE", "COINBASE",
    "TIMESTAMP", "NUMBER", "DIFFICULTY", "PREVRANDAO",
    "GASLIMIT", "CHAINID", "SELFBALANCE", "BASEFEE", "BLOBBASEFEE",
    "GAS", "PC", "MSIZE",
})

# Opcodes that terminate a basic block.
_TERMINATING_OPS: frozenset = frozenset({
    "JUMP", "JUMPI", "RETURN", "REVERT", "STOP", "SELFDESTRUCT", "INVALID",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stack_pop(stack: List[str]) -> str:
    """Pop from *stack*, returning ``"stack_underflow"`` if empty."""
    return stack.pop() if stack else "stack_underflow"


def _opcode_as_int(instr: Any) -> Optional[int]:
    """Return a raw opcode byte for dict or evmdasm instructions when available."""
    raw = instr.get("opcode") if isinstance(instr, dict) else getattr(instr, "opcode", None)
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, bytes):
        return int.from_bytes(raw, byteorder="big")
    try:
        return int(str(raw), 0)
    except (TypeError, ValueError):
        return None


def _normalize_opcode_name(name: Any, instr: Any = None) -> str:
    """Normalize disassembler-specific names for opcodes added after older forks."""
    text = str(name or "UNKNOWN").upper()
    opcode = _opcode_as_int(instr) if instr is not None else None
    if opcode in _RAW_OPCODE_NAMES:
        return _RAW_OPCODE_NAMES[opcode]
    if text.startswith("UNKNOWN_0X"):
        try:
            raw = int(text.rsplit("0X", 1)[1], 16)
        except ValueError:
            return text
        return _RAW_OPCODE_NAMES.get(raw, text)
    return text


def _normalize_hex_operand(operand: Any) -> Optional[str]:
    """Return lowercase hexadecimal text without ``0x`` prefixes."""
    if operand is None:
        return None
    if isinstance(operand, int):
        return format(operand, "x")
    text = str(operand).strip().lower()
    while text.startswith("0x"):
        text = text[2:]
    return "".join(ch for ch in text if ch in "0123456789abcdef")


# ---------------------------------------------------------------------------
# Main Analyzer
# ---------------------------------------------------------------------------


class BytecodeAnalyzer:
    """Analyze EVM bytecode and convert it to Three-Address Code.

    Based on the static analysis approach described in the paper,
    this class performs control flow analysis, function boundary
    identification, and TAC generation.
    """

    def __init__(self, bytecode: str, abi_enricher: Any = None) -> None:
        self.bytecode: str = bytecode
        self.instructions: List = []
        self.basic_blocks: Dict[str, BasicBlock] = {}
        self.functions: Dict[str, Function] = {}
        self.variable_counter: int = 0
        self.logger = logging.getLogger(__name__)
        self.abi_enricher = abi_enricher  # Optional ABIEnricher for custom error decoding

        # Pre-built lookup: PC → instruction index (populated after parsing)
        self._pc_to_index: Dict[int, int] = {}

        self._parse_bytecode()

    # ------------------------------------------------------------------ #
    #  Bytecode Parsing
    # ------------------------------------------------------------------ #

    def _parse_bytecode(self) -> None:
        """Parse EVM bytecode into instruction objects using *evmdasm*."""
        try:
            try:
                from evmdasm import EvmBytecode
            except ImportError:
                raise ImportError(
                    "evmdasm is required for bytecode disassembly. "
                    "Run: uv sync"
                )
            clean = self.bytecode[2:] if self.bytecode.startswith("0x") else self.bytecode
            evm = EvmBytecode(clean)
            self.instructions = list(evm.disassemble())

            for i, instr in enumerate(self.instructions):
                self._pc_to_index[self._get_pc(instr, i)] = i

            logger.info("Parsed %d instructions from bytecode", len(self.instructions))
        except Exception as e:
            logger.error("Failed to parse bytecode: %s", e)
            self.instructions = []

    # ------------------------------------------------------------------ #
    #  Instruction Accessors (support dict & evmdasm objects)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_instruction_name(instr: Any) -> str:
        """Return the mnemonic name of *instr*."""
        if isinstance(instr, dict):
            return _normalize_opcode_name(instr.get("name", "UNKNOWN"), instr)
        if hasattr(instr, "name"):
            return _normalize_opcode_name(instr.name, instr)
        if hasattr(instr, "opcode"):
            raw = instr.opcode.name if hasattr(instr.opcode, "name") else str(instr.opcode)
            return _normalize_opcode_name(raw, instr)
        return _normalize_opcode_name(str(instr), instr)

    @staticmethod
    def _get_pc(instr: Any, fallback: int) -> int:
        """Return the program counter of *instr*."""
        if isinstance(instr, dict):
            return instr.get("pc", instr.get("address", fallback))
        pc = getattr(instr, "pc", None)
        if pc is not None:
            return pc
        addr = getattr(instr, "address", None)
        if addr is not None:
            return addr
        return fallback

    @staticmethod
    def _get_operand(instr: Any) -> Optional[Any]:
        """Return the operand of *instr*, or ``None``."""
        if isinstance(instr, dict):
            return instr.get("operand", None)
        return getattr(instr, "operand", None)

    @staticmethod
    def _parse_operand_as_int(operand: Any) -> Optional[int]:
        """Parse an operand value as an integer (hex or decimal)."""
        try:
            if isinstance(operand, int):
                return operand
            if isinstance(operand, str):
                if operand.startswith(("0x", "0X")):
                    return int(operand, 16)
                try:
                    return int(operand, 16)
                except ValueError:
                    return int(operand, 10)
            return int(operand)
        except (ValueError, TypeError):
            return None

    def _generate_temp_var(self) -> str:
        """Generate a unique temporary variable name."""
        self.variable_counter += 1
        return f"temp_{self.variable_counter}"

    # ------------------------------------------------------------------ #
    #  Control-Flow Analysis — Public Entry Point
    # ------------------------------------------------------------------ #

    def analyze_control_flow(self) -> Dict[str, BasicBlock]:
        """Perform comprehensive control-flow analysis.

        Returns:
            Dictionary mapping block IDs to :class:`BasicBlock` objects.
        """
        logger.info("Starting comprehensive control flow analysis")
        try:
            jump_targets = self._detect_jump_targets()
            logger.info("Detected %d jump targets", len(jump_targets))

            blocks = self._construct_basic_blocks(jump_targets)
            logger.info("Constructed %d basic blocks", len(blocks))

            self._analyze_block_relationships(blocks)
            logger.info("Completed block relationship analysis")

            self._perform_advanced_analysis(blocks)
            logger.info("Completed advanced control flow analysis")

            self.basic_blocks = blocks
            return blocks
        except Exception as e:
            logger.error("Control flow analysis failed: %s", e)
            return self._fallback_control_flow_analysis()

    # ------------------------------------------------------------------ #
    #  Jump-Target Detection
    # ------------------------------------------------------------------ #

    def _detect_jump_targets(self) -> set:
        """Detect valid jump targets via JUMPDEST scanning and pattern matching."""
        targets: set = set()

        if not self.instructions:
            return targets

        # Entry point
        targets.add(self._get_pc(self.instructions[0], 0))

        # All JUMPDEST locations
        for i, instr in enumerate(self.instructions):
            if self._get_instruction_name(instr) == "JUMPDEST":
                targets.add(self._get_pc(instr, i))

        # Analyse each instruction for jump / dispatcher patterns
        for i, instr in enumerate(self.instructions):
            name = self._get_instruction_name(instr)
            pc = self._get_pc(instr, i)

            if name == "PUSH4" and i + 1 < len(self.instructions):
                if self._get_instruction_name(self.instructions[i + 1]) == "EQ":
                    targets.update(self._analyze_function_dispatch_pattern(i))
                self._collect_push_jump_targets(i, instr, targets)

            elif name.startswith("PUSH"):
                self._collect_push_jump_targets(i, instr, targets)

            elif name == "JUMP":
                targets.update(self._resolve_jump_targets(i, None))

            elif name == "JUMPI":
                targets.update(self._resolve_jump_targets(i, None))
                if i + 1 < len(self.instructions):
                    targets.add(self._get_pc(self.instructions[i + 1], i + 1))

            elif name in ("REVERT", "INVALID", "SELFDESTRUCT", "RETURN", "STOP"):
                if i + 1 < len(self.instructions):
                    targets.add(self._get_pc(self.instructions[i + 1], i + 1))

        return self._filter_jump_targets(targets)

    def _collect_push_jump_targets(self, push_idx: int, instr: Any, targets: set) -> None:
        """If *instr* (a PUSH) feeds a JUMP/JUMPI within a few instructions, add it."""
        try:
            operand = self._get_operand(instr)
            if operand is None:
                return
            target = self._parse_operand_as_int(operand)
            if target is None or target < 0 or target > 100_000:
                return
            passthrough = {"DUP1", "DUP2", "DUP3", "DUP4", "SWAP1", "SWAP2", "SWAP3", "SWAP4"}
            for j in range(push_idx + 1, min(push_idx + 5, len(self.instructions))):
                nxt = self._get_instruction_name(self.instructions[j])
                if nxt in ("JUMP", "JUMPI"):
                    targets.add(target)
                    break
                if nxt not in passthrough:
                    break
        except (ValueError, AttributeError):
            pass

    def _filter_jump_targets(self, targets: set) -> set:
        """Keep valid jump destinations and basic-block fall-through starts."""
        if not self.instructions:
            return set()
        first_pc = self._get_pc(self.instructions[0], 0)
        valid = {
            self._get_pc(instr, i)
            for i, instr in enumerate(self.instructions)
            if self._get_instruction_name(instr) == "JUMPDEST"
        }
        block_starts = {first_pc}
        for i, instr in enumerate(self.instructions[:-1]):
            if self._get_instruction_name(instr) in _TERMINATING_OPS:
                block_starts.add(self._get_pc(self.instructions[i + 1], i + 1))
        return {t for t in targets if t in valid or t in block_starts}

    def _resolve_jump_targets(self, jump_idx: int, stack_sim: Any) -> set:
        """Resolve targets for a JUMP/JUMPI by inspecting the preceding PUSH."""
        targets: set = set()
        if stack_sim is not None and hasattr(stack_sim, "get_stack_top_value"):
            val = stack_sim.get_stack_top_value()
            if isinstance(val, int):
                targets.add(val)

        for i in range(jump_idx - 1, max(0, jump_idx - 10), -1):
            instr = self.instructions[i]
            if self._get_instruction_name(instr).startswith("PUSH"):
                operand = self._get_operand(instr)
                if operand is not None:
                    val = self._parse_operand_as_int(operand)
                    if val is not None:
                        targets.add(val)
                break
        return targets

    def _analyze_function_dispatch_pattern(self, push4_idx: int) -> set:
        """Detect ``PUSH4 <sel> EQ PUSH<n> <target> JUMPI`` dispatcher pattern."""
        targets: set = set()
        try:
            if push4_idx + 3 >= len(self.instructions):
                return targets
            eq = self.instructions[push4_idx + 1]
            push = self.instructions[push4_idx + 2]
            jumpi = self.instructions[push4_idx + 3]
            if (self._get_instruction_name(eq) == "EQ"
                    and self._get_instruction_name(push).startswith("PUSH")
                    and self._get_instruction_name(jumpi) == "JUMPI"):
                val = self._parse_operand_as_int(self._get_operand(push))
                if val is not None:
                    targets.add(val)
        except (ValueError, AttributeError, IndexError):
            pass
        return targets

    # ------------------------------------------------------------------ #
    #  Basic-Block Construction
    # ------------------------------------------------------------------ #

    def _construct_basic_blocks(self, jump_targets: set) -> Dict[str, BasicBlock]:
        """Build basic blocks from sorted *jump_targets*."""
        blocks: Dict[str, BasicBlock] = {}
        sorted_targets = sorted(jump_targets)

        pc_to_instr: Dict[int, Any] = {}
        pc_to_idx: Dict[int, int] = {}
        for i, instr in enumerate(self.instructions):
            pc = self._get_pc(instr, i)
            pc_to_instr[pc] = instr
            pc_to_idx[pc] = i

        for ti, start_pc in enumerate(sorted_targets):
            end_pc = self._find_block_end(start_pc, sorted_targets, ti)
            raw: List = []
            cur = start_pc
            while cur <= end_pc and cur in pc_to_instr:
                instr = pc_to_instr[cur]
                raw.append(instr)
                if self._get_instruction_name(instr) in _TERMINATING_OPS:
                    break
                idx = pc_to_idx.get(cur, -1)
                if idx + 1 < len(self.instructions):
                    cur = self._get_pc(self.instructions[idx + 1], cur + 1)
                else:
                    break

            if raw:
                bid = f"block_{start_pc:04x}"
                blocks[bid] = BasicBlock(
                    id=bid,
                    instructions=[],
                    predecessors=[],
                    successors=[],
                    start_address=start_pc,
                    end_address=self._get_pc(raw[-1], end_pc),
                    metadata={"raw_instructions": raw},
                )
        return blocks

    def _find_block_end(self, start_pc: int, sorted_targets: List[int], idx: int) -> int:
        """Return the last PC belonging to the block starting at *start_pc*."""
        if idx + 1 < len(sorted_targets):
            return sorted_targets[idx + 1] - 1
        if self.instructions:
            return self._get_pc(self.instructions[-1], len(self.instructions) - 1)
        return start_pc

    # ------------------------------------------------------------------ #
    #  Block-Relationship (CFG Edge) Analysis
    # ------------------------------------------------------------------ #

    def _analyze_block_relationships(self, blocks: Dict[str, BasicBlock]) -> None:
        """Build predecessor / successor edges for every block."""
        pc_to_block: Dict[int, str] = {b.start_address: b.id for b in blocks.values()}
        for block in blocks.values():
            self._analyze_single_block_relationships(block, blocks, pc_to_block)

    def _analyze_single_block_relationships(
        self,
        block: BasicBlock,
        all_blocks: Dict[str, BasicBlock],
        pc_to_block: Dict[int, str],
    ) -> None:
        raw: List = block.metadata.get("raw_instructions", [])
        if not raw:
            return

        last = raw[-1]
        last_name = self._get_instruction_name(last)
        last_pc = self._get_pc(last, block.end_address)

        if last_name == "JUMP":
            for t in self._get_jump_targets_from_block(raw):
                if t in pc_to_block:
                    self._add_edge(block.id, pc_to_block[t], all_blocks)

        elif last_name == "JUMPI":
            for t in self._get_jump_targets_from_block(raw):
                if t in pc_to_block:
                    self._add_edge(block.id, pc_to_block[t], all_blocks)
            ft = self._get_next_instruction_pc(last_pc)
            if ft is not None and ft in pc_to_block:
                self._add_edge(block.id, pc_to_block[ft], all_blocks)

        elif last_name not in ("RETURN", "REVERT", "STOP", "SELFDESTRUCT", "INVALID"):
            ft = self._get_next_instruction_pc(last_pc)
            if ft is not None and ft in pc_to_block:
                self._add_edge(block.id, pc_to_block[ft], all_blocks)

    def _get_jump_targets_from_block(self, instructions: List) -> set:
        """Return the nearest PUSH operand preceding the terminating jump."""
        targets: set = set()
        for i in range(len(instructions) - 2, -1, -1):
            instr = instructions[i]
            if self._get_instruction_name(instr).startswith("PUSH"):
                val = self._parse_operand_as_int(self._get_operand(instr))
                if val is not None:
                    targets.add(val)
                break
        return targets

    def _get_next_instruction_pc(self, current_pc: int) -> Optional[int]:
        """Return the PC of the instruction after *current_pc* (O(1) lookup)."""
        idx = self._pc_to_index.get(current_pc)
        if idx is not None and idx + 1 < len(self.instructions):
            return self._get_pc(self.instructions[idx + 1], idx + 1)
        return None

    @staticmethod
    def _add_edge(src: str, dst: str, blocks: Dict[str, BasicBlock]) -> None:
        """Add a directed edge *src* → *dst* in *blocks*."""
        if src in blocks and dst in blocks:
            if dst not in blocks[src].successors:
                blocks[src].successors.append(dst)
            if src not in blocks[dst].predecessors:
                blocks[dst].predecessors.append(src)

    # ------------------------------------------------------------------ #
    #  Advanced Analysis (loops, dominance, reachability)
    # ------------------------------------------------------------------ #

    def _perform_advanced_analysis(self, blocks: Dict[str, BasicBlock]) -> None:
        self._detect_loops(blocks)
        self._compute_dominance(blocks)
        self._analyze_reachability(blocks)

    # -- loops --

    def _detect_loops(self, blocks: Dict[str, BasicBlock]) -> None:
        """Identify back-edges with DFS and mark loop headers."""
        if not blocks:
            return
        visited: set = set()
        on_stack: set = set()
        back_edges: List[Tuple[str, str]] = []

        def dfs(bid: str) -> None:
            visited.add(bid)
            on_stack.add(bid)
            for succ in blocks.get(bid, BasicBlock("", [], [], [], 0, 0)).successors:
                if succ not in visited:
                    dfs(succ)
                elif succ in on_stack:
                    back_edges.append((bid, succ))
            on_stack.discard(bid)

        entries = self._entry_blocks(blocks)
        for e in entries:
            if e not in visited:
                dfs(e)

        loop_headers = {edge[1] for edge in back_edges}
        for b in blocks.values():
            b.metadata["back_edges"] = back_edges
            b.metadata["is_loop_header"] = b.id in loop_headers

    # -- dominance --

    def _compute_dominance(self, blocks: Dict[str, BasicBlock]) -> None:
        """Iterative dominator computation."""
        if not blocks:
            return
        ids = list(blocks.keys())
        dom: Dict[str, set] = {bid: set(ids) for bid in ids}

        entries = self._entry_blocks(blocks)
        for e in entries:
            dom[e] = {e}

        changed = True
        while changed:
            changed = False
            for bid in ids:
                if bid in entries:
                    continue
                preds = [p for p in blocks[bid].predecessors if p in dom]
                new = set.intersection(*(dom[p] for p in preds)) if preds else set()
                new.add(bid)
                if new != dom[bid]:
                    dom[bid] = new
                    changed = True

        for bid, b in blocks.items():
            b.metadata["dominators"] = dom[bid]

    # -- reachability --

    def _analyze_reachability(self, blocks: Dict[str, BasicBlock]) -> None:
        """Mark each block as reachable or dead code."""
        if not blocks:
            return
        reachable: set = set()

        def walk(bid: str) -> None:
            if bid in reachable or bid not in blocks:
                return
            reachable.add(bid)
            for s in blocks[bid].successors:
                walk(s)

        for e in self._entry_blocks(blocks):
            walk(e)

        for bid, b in blocks.items():
            b.metadata["is_reachable"] = bid in reachable
            b.metadata["is_dead_code"] = bid not in reachable

    # -- helpers --

    @staticmethod
    def _entry_blocks(blocks: Dict[str, BasicBlock]) -> List[str]:
        """Return the bytecode entry block only."""
        if not blocks:
            return []
        entry = min(blocks.values(), key=lambda b: b.start_address)
        return [entry.id]

    # ------------------------------------------------------------------ #
    #  Fallback (error recovery)
    # ------------------------------------------------------------------ #

    def _fallback_control_flow_analysis(self) -> Dict[str, BasicBlock]:
        """Create a single basic block containing all instructions."""
        logger.warning("Using fallback control flow analysis")
        if not self.instructions:
            return {}
        first_pc = self._get_pc(self.instructions[0], 0)
        last_pc = self._get_pc(self.instructions[-1], len(self.instructions) - 1)
        bid = f"block_{first_pc:04x}"
        return {
            bid: BasicBlock(
                id=bid, instructions=[], predecessors=[], successors=[],
                start_address=first_pc, end_address=last_pc,
                metadata={"raw_instructions": list(self.instructions)},
            )
        }

    # ------------------------------------------------------------------ #
    #  Function Identification
    # ------------------------------------------------------------------ #

    def identify_functions(self) -> Dict[str, Function]:
        """Identify functions by scanning the dispatcher for ``PUSH4 … EQ … JUMPI`` patterns."""
        functions: Dict[str, Function] = {}
        logger.info("Identifying functions from bytecode dispatcher")

        for pattern in self._find_dispatcher_patterns():
            selector = pattern["selector"]
            jump_target = pattern["target"]

            fname = f"function_{selector}"
            entry = self._block_at_address(jump_target) or f"block_{jump_target:04x}"
            functions[fname] = Function(
                name=fname, selector=selector, basic_blocks=[], entry_block=entry,
            )
            logger.debug("Identified function %s → %04x", selector, jump_target)

        logger.info("Identified %d functions from dispatcher", len(functions))

        # Issue #6: Detect receive(), fallback(), and internal functions
        dispatcher_targets = {f.entry_block for f in functions.values()}
        receive = self._detect_receive_function(dispatcher_targets)
        if receive:
            functions["receive"] = receive
            self.logger.info("Detected receive() function at %s", receive.entry_block)

        fallback_func = self._detect_fallback_function(dispatcher_targets)
        if fallback_func:
            functions["fallback_function"] = fallback_func
            self.logger.info("Detected fallback() function at %s", fallback_func.entry_block)

        internal = self._detect_internal_functions(functions)
        for iname, ifunc in internal.items():
            functions[iname] = ifunc
        if internal:
            self.logger.info("Detected %d internal function(s)", len(internal))

        if not functions and self.basic_blocks:
            entry = next(iter(self.basic_blocks))
            functions["fallback"] = Function(
                name="fallback", selector=None,
                basic_blocks=list(self.basic_blocks.values()), entry_block=entry,
            )
            logger.info("No functions found – created fallback function")

        self.functions = functions
        return functions

    # ------------------------------------------------------------------ #
    #  Special Function Detection (Issue #6)
    # ------------------------------------------------------------------ #

    def _detect_receive_function(self, dispatcher_targets: set) -> Optional[Function]:
        """Detect receive() function: ``CALLDATASIZE ISZERO ... JUMPI`` pattern.

        The receive function is called for plain ETH transfers with no calldata.
        In the dispatcher, this shows up as a check for CALLDATASIZE == 0
        followed by a jump to the receive handler.
        """
        for i, instr in enumerate(self.instructions):
            if self._get_instruction_name(instr) != "CALLDATASIZE":
                continue
            # Look for ISZERO ... JUMPI within the next few instructions
            for j in range(i + 1, min(i + 5, len(self.instructions))):
                if self._get_instruction_name(self.instructions[j]) == "ISZERO":
                    # Look for a subsequent PUSH + JUMPI
                    target = None
                    for k in range(j + 1, min(j + 5, len(self.instructions))):
                        kname = self._get_instruction_name(self.instructions[k])
                        if kname.startswith("PUSH"):
                            target = self._parse_operand_as_int(
                                self._get_operand(self.instructions[k])
                            )
                        elif kname == "JUMPI" and target is not None:
                            entry = self._block_at_address(target) or f"block_{target:04x}"
                            if entry not in dispatcher_targets:
                                return Function(
                                    name="receive",
                                    selector=None,
                                    basic_blocks=[],
                                    entry_block=entry,
                                    visibility="external",
                                    is_payable=True,
                                )
                            break
                    break
        return None

    def _detect_fallback_function(self, dispatcher_targets: set) -> Optional[Function]:
        """Detect fallback() function: the default path after all selector comparisons fail.

        After the last PUSH4/EQ/JUMPI dispatcher sequence, the fall-through
        path is the fallback function. We find the last dispatcher entry and
        take the fall-through block after its JUMPI.
        """
        patterns = self._find_dispatcher_patterns()
        if not patterns:
            return None

        last_jumpi_idx = patterns[-1]["jumpi_idx"]
        last_jumpi_pc = self._get_pc(self.instructions[last_jumpi_idx], last_jumpi_idx)

        # The fall-through from the last dispatcher JUMPI is the fallback
        ft = self._get_next_instruction_pc(last_jumpi_pc)
        if ft is None:
            return None

        entry = self._block_at_address(ft) or f"block_{ft:04x}"
        if entry in dispatcher_targets or entry not in self.basic_blocks:
            return None

        return Function(
            name="fallback_function",
            selector=None,
            basic_blocks=[],
            entry_block=entry,
            visibility="external",
        )

    def _detect_internal_functions(self, known_functions: Dict[str, Function]) -> Dict[str, Function]:
        """Detect internal functions: JUMPDEST targets called via JUMP from
        within identified functions but not themselves dispatcher targets.

        Internal functions are blocks that:
        1. Are JUMP targets (not JUMPI conditional destinations)
        2. Are NOT entry blocks of already-identified functions
        3. Are reached from within identified function blocks
        """
        known_entries = {f.entry_block for f in known_functions.values()}

        # Collect all blocks reachable from known functions
        known_block_ids: set = set()
        for func in known_functions.values():
            visited: set = set()
            def _walk(bid: str) -> None:
                if bid in visited or bid not in self.basic_blocks:
                    return
                visited.add(bid)
                for s in self.basic_blocks[bid].successors:
                    _walk(s)
            _walk(func.entry_block)
            known_block_ids.update(visited)

        internal: Dict[str, Function] = {}

        # Look for JUMP targets within known function blocks that point to
        # blocks outside the known entry set
        for bid in list(known_block_ids):
            block = self.basic_blocks.get(bid)
            if not block:
                continue
            raw = block.metadata.get("raw_instructions", [])
            if not raw:
                continue
            last = raw[-1]
            last_name = self._get_instruction_name(last)
            # Only unconditional JUMPs suggest internal function calls
            if last_name != "JUMP":
                continue

            for target_bid in block.successors:
                if target_bid in known_entries:
                    continue
                if target_bid in internal:
                    continue
                if target_bid not in self.basic_blocks:
                    continue
                # This looks like an internal function call
                target_block = self.basic_blocks[target_bid]
                # Check it has raw instructions starting with JUMPDEST
                target_raw = target_block.metadata.get("raw_instructions", [])
                if target_raw and self._get_instruction_name(target_raw[0]) == "JUMPDEST":
                    fname = f"internal_{target_bid}"
                    internal[fname] = Function(
                        name=fname,
                        selector=None,
                        basic_blocks=[],
                        entry_block=target_bid,
                        visibility="internal",
                    )

        return internal

    # -- helpers --

    @staticmethod
    def _normalize_selector(raw: Any) -> Optional[str]:
        """Return a ``0x``-prefixed 8-hex-digit selector string, or ``None``."""
        if isinstance(raw, str):
            sel = raw if raw.startswith("0x") else "0x" + raw
        else:
            sel = "0x" + format(int(raw), "08x")
        return sel if len(sel) == 10 else None

    def _find_dispatcher_patterns(self) -> List[Dict[str, Any]]:
        """Find selector comparisons fed by calldata selector extraction."""
        patterns: List[Dict[str, Any]] = []
        stack: List[Any] = []
        max_scan = min(len(self.instructions), 256)

        def pop() -> Any:
            return stack.pop() if stack else ("unknown",)

        def push_unknown() -> None:
            stack.append(("unknown",))

        def const_value(sym: Any) -> Optional[int]:
            if isinstance(sym, tuple) and len(sym) >= 2 and sym[0] == "const":
                return sym[1]
            return None

        def is_selector(sym: Any) -> bool:
            return sym == ("selector",)

        def selector_comparison(left: Any, right: Any) -> Optional[Tuple[str, int]]:
            for selector_sym, const_sym in ((left, right), (right, left)):
                value = const_value(const_sym)
                if is_selector(selector_sym) and value is not None and 0 <= value <= 0xFFFFFFFF:
                    selector = self._normalize_selector(value)
                    if selector is not None:
                        push_idx = const_sym[2] if len(const_sym) > 2 else -1
                        return selector, push_idx
            return None

        def is_selector_divisor(value: Optional[int]) -> bool:
            return value == (1 << 224)

        for i in range(max_scan):
            instr = self.instructions[i]
            name = self._get_instruction_name(instr)

            if patterns and name == "JUMPDEST":
                break

            if name == "JUMPDEST":
                continue

            if name == "PUSH0":
                stack.append(("const", 0, i))
                continue

            if name.startswith("PUSH"):
                value = self._parse_operand_as_int(self._get_operand(instr))
                stack.append(("const", value, i) if value is not None else ("unknown",))
                continue

            if name == "POP":
                pop()
                continue

            if name.startswith("DUP"):
                try:
                    depth = int(name[3:]) if len(name) > 3 else 1
                except ValueError:
                    depth = 1
                stack.append(stack[-depth] if len(stack) >= depth else ("unknown",))
                continue

            if name.startswith("SWAP"):
                try:
                    depth = int(name[4:]) if len(name) > 4 else 1
                except ValueError:
                    depth = 1
                if len(stack) > depth:
                    stack[-1], stack[-1 - depth] = stack[-1 - depth], stack[-1]
                continue

            if name == "CALLDATASIZE":
                stack.append(("calldatasize",))
                continue

            if name == "CALLDATALOAD":
                offset = const_value(pop())
                stack.append(("calldata_word_0",) if offset == 0 else ("unknown",))
                continue

            if name == "SHR":
                shift = const_value(pop())
                value = pop()
                stack.append(("selector",) if value == ("calldata_word_0",) and shift == 224 else ("unknown",))
                continue

            if name == "DIV":
                divisor = const_value(pop())
                value = pop()
                stack.append(("selector",) if value == ("calldata_word_0",) and is_selector_divisor(divisor) else ("unknown",))
                continue

            if name == "AND":
                right = pop()
                left = pop()
                right_const = const_value(right)
                left_const = const_value(left)
                if (is_selector(left) and right_const == 0xFFFFFFFF) or (
                    is_selector(right) and left_const == 0xFFFFFFFF
                ):
                    stack.append(("selector",))
                else:
                    push_unknown()
                continue

            if name == "EQ":
                right = pop()
                left = pop()
                comparison = selector_comparison(left, right)
                stack.append(("selector_eq", comparison[0], comparison[1]) if comparison else ("unknown",))
                continue

            if name == "JUMPI":
                target = const_value(pop())
                cond = pop()
                if (
                    target is not None
                    and isinstance(cond, tuple)
                    and len(cond) >= 3
                    and cond[0] == "selector_eq"
                ):
                    patterns.append(
                        {
                            "selector": cond[1],
                            "target": target,
                            "push4_idx": cond[2],
                            "jumpi_idx": i,
                        }
                    )
                continue

            if name == "JUMP":
                pop()
                if patterns:
                    break
                continue

            if name in ("STOP", "RETURN", "REVERT", "INVALID", "SELFDESTRUCT"):
                if patterns:
                    break
                effects = _EVM_STACK_EFFECTS.get(name, (0, 0))
            elif name in _BINARY_OPS:
                effects = (2, 1)
            elif name in ("ADDMOD", "MULMOD"):
                effects = (3, 1)
            else:
                effects = _EVM_STACK_EFFECTS.get(name)

            if effects is None:
                push_unknown()
                continue
            pops, pushes = effects
            for _ in range(pops):
                pop()
            for _ in range(pushes):
                push_unknown()

        return patterns

    def _find_dispatch_target(self, push4_idx: int) -> Optional[int]:
        """Walk ahead from *push4_idx* looking for ``EQ … PUSH<n> <target> JUMPI``."""
        for j in range(push4_idx + 1, min(push4_idx + 10, len(self.instructions))):
            if self._get_instruction_name(self.instructions[j]) != "EQ":
                continue
            target: Optional[int] = None
            for k in range(j + 1, min(j + 5, len(self.instructions))):
                kname = self._get_instruction_name(self.instructions[k])
                if kname.startswith("PUSH"):
                    target = self._parse_operand_as_int(self._get_operand(self.instructions[k]))
                elif kname == "JUMPI" and target is not None:
                    return target
            break
        return None

    def _block_at_address(self, addr: int) -> Optional[str]:
        """Return the block ID whose start address equals *addr*, or ``None``."""
        for bid, b in self.basic_blocks.items():
            if b.start_address == addr:
                return bid
        return None

    # ------------------------------------------------------------------ #
    #  TAC Conversion — Public API
    # ------------------------------------------------------------------ #

    def convert_to_tac(self) -> List[TACInstruction]:
        """Convert all instructions to TAC (without control-flow context)."""
        tac: List[TACInstruction] = []
        stack: List[str] = []
        for instr in self.instructions:
            result = self._convert_instruction_to_tac(instr, stack)
            if result:
                if isinstance(result, list):
                    tac.extend(result)
                else:
                    tac.append(result)
        return tac

    # ------------------------------------------------------------------ #
    #  TAC Conversion — Per-Instruction
    # ------------------------------------------------------------------ #

    def _convert_instruction_to_tac(
        self, instr: Any, stack: List[str],
    ) -> Union[TACInstruction, List[TACInstruction], None]:
        """Convert a single EVM instruction to TAC, updating *stack* in place."""
        name = self._get_instruction_name(instr)

        # -- No-ops --
        if name == "JUMPDEST":
            return None

        # -- PUSH --
        if name.startswith("PUSH"):
            tmp = self._generate_temp_var()
            stack.append(tmp)
            operand = "0" if name == "PUSH0" else self._get_operand(instr)
            return TACInstruction(
                TACOperationType.ASSIGN, result=tmp,
                operand1=str(operand) if operand is not None else "unknown",
                metadata={"original_op": name},
            )

        # -- POP --
        if name == "POP":
            _stack_pop(stack)
            return None

        # -- DUP1..DUP16 --
        if name.startswith("DUP"):
            depth = int(name[3:]) if len(name) > 3 else 1
            src = stack[-depth] if len(stack) >= depth else "stack_underflow"
            tmp = self._generate_temp_var()
            stack.append(tmp)
            return TACInstruction(
                TACOperationType.ASSIGN, result=tmp, operand1=src,
                metadata={"original_op": name},
            )

        # -- SWAP1..SWAP16 --
        if name.startswith("SWAP"):
            depth = int(name[4:]) if len(name) > 4 else 1
            if len(stack) > depth:
                stack[-1], stack[-1 - depth] = stack[-1 - depth], stack[-1]
            return None

        # -- Binary ops --
        if name in _BINARY_OPS:
            right = _stack_pop(stack)
            left = _stack_pop(stack)
            tmp = self._generate_temp_var()
            stack.append(tmp)
            return TACInstruction(
                TACOperationType.BINARY_OP, result=tmp,
                operand1=left, operand2=right, operator=_BINARY_OPS[name],
                metadata={"original_op": name},
            )

        # -- Ternary: ADDMOD / MULMOD --
        if name in ("ADDMOD", "MULMOD"):
            sym = "addmod" if name == "ADDMOD" else "mulmod"
            modulus = _stack_pop(stack)
            right = _stack_pop(stack)
            left = _stack_pop(stack)
            tmp = self._generate_temp_var()
            stack.append(tmp)
            return TACInstruction(
                TACOperationType.BINARY_OP, result=tmp,
                operand1=f"{sym}({left}, {right}, {modulus})",
                metadata={"original_op": name},
            )

        # -- Unary: ISZERO / NOT --
        if name == "ISZERO":
            return self._tac_unary(stack, "!", name)
        if name == "NOT":
            return self._tac_unary(stack, "~", name)

        # -- Memory: MLOAD --
        if name == "MLOAD":
            addr = _stack_pop(stack)
            tmp = self._generate_temp_var()
            stack.append(tmp)
            return TACInstruction(
                TACOperationType.LOAD, result=tmp, operand1=addr,
                metadata={"memory_type": "memory", "original_op": name},
            )

        # -- Memory: MSTORE / MSTORE8 --
        if name in ("MSTORE", "MSTORE8"):
            addr, val = _stack_pop(stack), _stack_pop(stack)
            mtype = "memory8" if name == "MSTORE8" else "memory"
            return TACInstruction(
                TACOperationType.STORE, operand1=addr, operand2=val,
                metadata={"memory_type": mtype, "original_op": name},
            )

        # -- Storage: SLOAD / SSTORE --
        if name == "SLOAD":
            key = _stack_pop(stack)
            tmp = self._generate_temp_var()
            stack.append(tmp)
            return TACInstruction(
                TACOperationType.LOAD, result=tmp, operand1=key,
                metadata={"memory_type": "storage", "original_op": name},
            )
        if name == "SSTORE":
            key, val = _stack_pop(stack), _stack_pop(stack)
            return TACInstruction(
                TACOperationType.STORE, operand1=key, operand2=val,
                metadata={"memory_type": "storage", "original_op": name},
            )

        # -- Transient storage: TLOAD / TSTORE --
        if name == "TLOAD":
            key = _stack_pop(stack)
            tmp = self._generate_temp_var()
            stack.append(tmp)
            return TACInstruction(
                TACOperationType.LOAD, result=tmp, operand1=key,
                metadata={"memory_type": "transient_storage", "original_op": name},
            )
        if name == "TSTORE":
            key, val = _stack_pop(stack), _stack_pop(stack)
            return TACInstruction(
                TACOperationType.STORE, operand1=key, operand2=val,
                metadata={"memory_type": "transient_storage", "original_op": name},
            )

        # -- SHA3 / KECCAK256 --
        if name in ("SHA3", "KECCAK256"):
            off, sz = _stack_pop(stack), _stack_pop(stack)
            tmp = self._generate_temp_var()
            stack.append(tmp)
            return TACInstruction(
                TACOperationType.UNARY_OP, result=tmp,
                operand1=f"keccak256(memory[{off}:{sz}])",
                metadata={"original_op": name},
            )

        # -- CALLDATALOAD --
        if name == "CALLDATALOAD":
            off = _stack_pop(stack)
            tmp = self._generate_temp_var()
            stack.append(tmp)
            return TACInstruction(
                TACOperationType.LOAD, result=tmp, operand1=off,
                metadata={"memory_type": "calldata", "original_op": name},
            )

        # -- Copy ops (3-pop, 0-push) --
        if name in ("CALLDATACOPY", "CODECOPY", "RETURNDATACOPY"):
            dst, src, length = _stack_pop(stack), _stack_pop(stack), _stack_pop(stack)
            ctype = name.replace("COPY", "").lower()
            return TACInstruction(
                TACOperationType.STORE, operand1=dst,
                operand2=f"{ctype}[{src}:{length}]",
                metadata={"memory_type": "copy", "original_op": name},
            )

        # -- Memory-to-memory copy (EIP-5656) --
        if name == "MCOPY":
            dst, src, length = _stack_pop(stack), _stack_pop(stack), _stack_pop(stack)
            return TACInstruction(
                TACOperationType.STORE, operand1=dst,
                operand2=f"memory[{src}:{length}]",
                metadata={"memory_type": "memory", "original_op": name},
            )

        # -- EXTCODECOPY (4-pop, 0-push) --
        if name == "EXTCODECOPY":
            addr, dst, src, length = (
                _stack_pop(stack), _stack_pop(stack),
                _stack_pop(stack), _stack_pop(stack),
            )
            return TACInstruction(
                TACOperationType.STORE, operand1=dst,
                operand2=f"extcode({addr})[{src}:{length}]",
                metadata={"memory_type": "copy", "original_op": name},
            )

        # -- Environmental info (0-pop, 1-push) --
        if name in _ENV_OPS:
            tmp = self._generate_temp_var()
            stack.append(tmp)
            return TACInstruction(
                TACOperationType.ASSIGN, result=tmp, operand1=name.lower(),
                metadata={"original_op": name},
            )

        # -- 1-pop, 1-push environment --
        if name in ("BALANCE", "EXTCODESIZE", "EXTCODEHASH", "BLOCKHASH", "BLOBHASH"):
            return self._tac_unary(stack, name.lower(), name)

        # -- JUMP --
        if name == "JUMP":
            return TACInstruction(
                TACOperationType.JUMP, target=_stack_pop(stack),
                metadata={"original_op": name},
            )

        # -- JUMPI --
        if name == "JUMPI":
            tgt, cond = _stack_pop(stack), _stack_pop(stack)
            return TACInstruction(
                TACOperationType.CONDITIONAL_JUMP, target=tgt, operand1=cond,
                metadata={"original_op": name},
            )

        # -- RETURN --
        if name == "RETURN":
            off, sz = _stack_pop(stack), _stack_pop(stack)
            return TACInstruction(
                TACOperationType.RETURN, operand1=off, operand2=sz,
                metadata={"original_op": name},
            )

        # -- REVERT --
        if name == "REVERT":
            off, sz = _stack_pop(stack), _stack_pop(stack)
            decoded = self._decode_revert_data(instr, stack)
            meta: Dict[str, Any] = {"original_op": name}
            if decoded:
                meta["revert_decoded"] = decoded
            return TACInstruction(
                TACOperationType.REVERT, operand1=off, operand2=sz,
                metadata=meta,
            )

        # -- STOP --
        if name == "STOP":
            return TACInstruction(TACOperationType.HALT, metadata={"original_op": name})

        # -- SELFDESTRUCT --
        if name == "SELFDESTRUCT":
            return TACInstruction(
                TACOperationType.HALT, operand1=_stack_pop(stack),
                metadata={"original_op": name},
            )

        # -- INVALID --
        if name == "INVALID":
            return TACInstruction(
                TACOperationType.HALT,
                metadata={"original_op": name, "reason": "invalid opcode"},
            )

        # -- CALL / CALLCODE (7-pop, 1-push) --
        if name in ("CALL", "CALLCODE"):
            gas, addr, val = _stack_pop(stack), _stack_pop(stack), _stack_pop(stack)
            ao, al, ro, rl = (
                _stack_pop(stack), _stack_pop(stack),
                _stack_pop(stack), _stack_pop(stack),
            )
            tmp = self._generate_temp_var()
            stack.append(tmp)
            return TACInstruction(
                TACOperationType.CALL, result=tmp, operand1=addr, operand2=val,
                metadata={
                    "original_op": name, "gas": gas,
                    "args_offset": ao, "args_length": al,
                    "ret_offset": ro, "ret_length": rl,
                },
            )

        # -- DELEGATECALL / STATICCALL (6-pop, 1-push) --
        if name in ("DELEGATECALL", "STATICCALL"):
            gas, addr = _stack_pop(stack), _stack_pop(stack)
            ao, al, ro, rl = (
                _stack_pop(stack), _stack_pop(stack),
                _stack_pop(stack), _stack_pop(stack),
            )
            tmp = self._generate_temp_var()
            stack.append(tmp)
            return TACInstruction(
                TACOperationType.CALL, result=tmp, operand1=addr,
                metadata={
                    "original_op": name, "gas": gas,
                    "args_offset": ao, "args_length": al,
                    "ret_offset": ro, "ret_length": rl,
                },
            )

        # -- CREATE --
        if name == "CREATE":
            val, off, length = _stack_pop(stack), _stack_pop(stack), _stack_pop(stack)
            tmp = self._generate_temp_var()
            stack.append(tmp)
            return TACInstruction(
                TACOperationType.CALL, result=tmp,
                operand1=f"create({val}, memory[{off}:{length}])",
                metadata={"original_op": name},
            )

        # -- CREATE2 --
        if name == "CREATE2":
            val, off, length, salt = (
                _stack_pop(stack), _stack_pop(stack),
                _stack_pop(stack), _stack_pop(stack),
            )
            tmp = self._generate_temp_var()
            stack.append(tmp)
            return TACInstruction(
                TACOperationType.CALL, result=tmp,
                operand1=f"create2({val}, memory[{off}:{length}], {salt})",
                metadata={"original_op": name},
            )

        # -- LOG0..LOG4 --
        if name.startswith("LOG"):
            try:
                tc = int(name[3:])
            except ValueError:
                tc = 0
            off, sz = _stack_pop(stack), _stack_pop(stack)
            topics = [_stack_pop(stack) for _ in range(tc)]
            return TACInstruction(
                TACOperationType.LOG, operand1=off, operand2=sz,
                metadata={"original_op": name, "topics": topics, "topic_count": tc},
            )

        # -- Generic fallback --
        return self._tac_fallback(name, instr, stack)

    # ------------------------------------------------------------------ #
    #  Revert Data Decoder (Issue #3)
    # ------------------------------------------------------------------ #

    def _decode_revert_data(self, revert_instr: Any, stack: List[str]) -> Optional[Dict[str, str]]:
        """Attempt to decode revert error data from preceding MSTORE instructions.

        Looks backward from the REVERT instruction to find a selector written
        via MSTORE, then matches it against well-known error selectors and
        custom errors from the ABI enricher.

        Returns:
            A dict with keys ``type`` and ``message`` if decoded, else ``None``.
        """
        revert_pc = self._get_pc(revert_instr, -1)
        revert_idx = self._pc_to_index.get(revert_pc)
        if revert_idx is None:
            return None

        # Scan backward up to 20 instructions looking for a PUSH4/PUSH32 that
        # contains a known error selector
        for i in range(revert_idx - 1, max(0, revert_idx - 20) - 1, -1):
            instr = self.instructions[i]
            name = self._get_instruction_name(instr)
            if not name.startswith("PUSH"):
                continue
            operand = self._get_operand(instr)
            if operand is None:
                continue
            # Normalize operand to lowercase hex (no 0x prefix).
            # evmdasm may return int, hex-string, bare hex, or prefixed forms.
            operand_str = _normalize_hex_operand(operand)
            if not operand_str:
                continue

            # Check for Error(string) selector: 08c379a0
            if "08c379a0" in operand_str:
                return {"type": "Error(string)", "message": "Error(string)"}

            # Check for Panic(uint256) selector: 4e487b71
            if "4e487b71" in operand_str:
                # Try to find the panic code from a nearby PUSH
                panic_code = self._find_panic_code(i, revert_idx)
                if panic_code is not None and panic_code in PANIC_CODES:
                    desc = PANIC_CODES[panic_code]
                    return {
                        "type": "Panic",
                        "code": hex(panic_code),
                        "message": f"Panic({hex(panic_code)})  // {desc}",
                    }
                return {"type": "Panic", "message": "Panic(uint256)"}

            # Check for custom error selectors (4 bytes = 8 hex chars)
            if self.abi_enricher is not None:
                error_info = self._lookup_custom_error(operand_str)
                if error_info is not None:
                    params = ", ".join(
                        f"{t} {n}"
                        for t, n in zip(error_info.input_types, error_info.input_names)
                    )
                    return {
                        "type": "CustomError",
                        "name": error_info.name,
                        "message": f"{error_info.name}({params})",
                    }

        return None

    def _lookup_custom_error(self, operand_hex: str) -> Optional[Any]:
        """Find an ABI custom error whose selector appears in *operand_hex*."""
        candidates = self._selector_candidates_from_operand(operand_hex)
        if not candidates or self.abi_enricher is None:
            return None

        get_error = getattr(self.abi_enricher, "get_error", None)
        if callable(get_error):
            for selector in candidates:
                for key in (f"0x{selector}", f"0x0x{selector}"):
                    error_info = get_error(key)
                    if error_info is not None:
                        return error_info

        for error_info in getattr(self.abi_enricher, "errors", {}).values():
            selector = _normalize_hex_operand(getattr(error_info, "selector", None))
            if selector and selector[-8:] in candidates:
                return error_info

        return None

    @staticmethod
    def _selector_candidates_from_operand(operand_hex: str) -> List[str]:
        """Return likely 4-byte selector candidates from normalized operand hex."""
        text = operand_hex.lower()
        if len(text) < 8:
            text = text.zfill(8)

        candidates: List[str] = []
        for candidate in (text, text[:8], text[-8:]):
            if len(candidate) == 8 and candidate not in candidates:
                candidates.append(candidate)

        if len(text) > 8:
            for i in range(0, len(text) - 7, 2):
                candidate = text[i:i + 8]
                if candidate not in candidates:
                    candidates.append(candidate)
        return candidates

    def _find_panic_code(self, selector_idx: int, revert_idx: int) -> Optional[int]:
        """Search between *selector_idx* and *revert_idx* for a small PUSH value
        that represents the panic code."""
        for i in range(selector_idx + 1, revert_idx):
            instr = self.instructions[i]
            name = self._get_instruction_name(instr)
            if name.startswith("PUSH"):
                val = self._parse_operand_as_int(self._get_operand(instr))
                if val is not None and val <= 0xFF:
                    return val
        return None

    # -- small helpers used by the converter --

    def _tac_unary(self, stack: List[str], operator: str, name: str) -> TACInstruction:
        """Emit a unary-op TAC instruction (1-pop, 1-push)."""
        arg = _stack_pop(stack)
        tmp = self._generate_temp_var()
        stack.append(tmp)
        return TACInstruction(
            TACOperationType.UNARY_OP, result=tmp, operand1=arg, operator=operator,
            metadata={"original_op": name},
        )

    def _tac_fallback(self, name: str, instr: Any, stack: List[str]) -> TACInstruction:
        """Emit a best-effort ASSIGN for an unhandled opcode."""
        effects = _EVM_STACK_EFFECTS.get(name)
        if effects is not None:
            pops, pushes = effects
            for _ in range(pops):
                _stack_pop(stack)
            result_var: Optional[str] = None
            for _ in range(pushes):
                result_var = self._generate_temp_var()
                stack.append(result_var)
            return TACInstruction(
                TACOperationType.ASSIGN, result=result_var,
                metadata={"original_op": name, "unhandled": True},
            )
        # Totally unknown — preserve the stack because the effect is unknown.
        return TACInstruction(
            TACOperationType.NOP,
            metadata={"original_op": name, "unhandled": True,
                       "unknown_stack_effect": True,
                       "operand": getattr(instr, "operand", None)},
        )

    # ------------------------------------------------------------------ #
    #  Stack Simulator (used during control-flow analysis)
    # ------------------------------------------------------------------ #

    class _StackSimulator:
        """Lightweight EVM stack simulator for jump-target resolution."""

        def __init__(self) -> None:
            self.stack: List = []
            self.stack_values: Dict[int, int] = {}

        def process_instruction(self, instr: Any, index: int) -> None:
            name = self._get_name(instr)

            if name == "PUSH0":
                self.stack.append(0)
                self.stack_values[index] = 0

            elif name.startswith("PUSH"):
                try:
                    operand = instr.get("operand") if isinstance(instr, dict) else getattr(instr, "operand", None)
                    if operand is not None:
                        val = int(operand, 16) if isinstance(operand, str) else int(operand)
                        self.stack.append(val)
                        self.stack_values[index] = val
                    else:
                        self.stack.append(None)
                except (ValueError, AttributeError):
                    self.stack.append(None)

            elif name == "POP":
                if self.stack:
                    self.stack.pop()

            elif name.startswith("DUP"):
                d = int(name[3:]) if len(name) > 3 else 1
                self.stack.append(self.stack[-d] if len(self.stack) >= d else None)

            elif name.startswith("SWAP"):
                d = int(name[4:]) if len(name) > 4 else 1
                if len(self.stack) > d:
                    self.stack[-1], self.stack[-1 - d] = self.stack[-1 - d], self.stack[-1]

            elif name in ("ADDMOD", "MULMOD"):
                for _ in range(min(3, len(self.stack))):
                    self.stack.pop()
                self.stack.append(None)

            elif name in ("ISZERO", "NOT"):
                if self.stack:
                    self.stack[-1] = None

            elif name in _BINARY_OPS:
                if len(self.stack) >= 2:
                    self.stack.pop()
                    self.stack[-1] = None

            else:
                effects = _EVM_STACK_EFFECTS.get(name)
                if effects:
                    pops, pushes = effects
                    for _ in range(min(pops, len(self.stack))):
                        self.stack.pop()
                    self.stack.extend([None] * pushes)

            if len(self.stack) > 1024:
                self.stack = self.stack[-1024:]

        def get_stack_top_value(self) -> Optional[int]:
            if self.stack and isinstance(self.stack[-1], int):
                return self.stack[-1]
            return None

        @staticmethod
        def _get_name(instr: Any) -> str:
            if isinstance(instr, dict):
                return _normalize_opcode_name(instr.get("name", "UNKNOWN"), instr)
            if hasattr(instr, "name"):
                return _normalize_opcode_name(instr.name, instr)
            if hasattr(instr, "opcode"):
                raw = instr.opcode.name if hasattr(instr.opcode, "name") else str(instr.opcode)
                return _normalize_opcode_name(raw, instr)
            return _normalize_opcode_name(str(instr), instr)

    # ------------------------------------------------------------------ #
    #  Per-Function TAC Generation
    # ------------------------------------------------------------------ #

    def generate_function_tac(self, func: Function) -> str:
        """Generate a compact TAC string for a single function.

        Only includes the basic blocks reachable from the function's
        entry block, keeping the output small enough for the LLM
        context window.

        Args:
            func: A :class:`Function` previously returned by
                :meth:`identify_functions`.

        Returns:
            A human-readable TAC string for *func*.
        """
        if self.basic_blocks and not any(b.instructions for b in self.basic_blocks.values()):
            self._convert_and_integrate_tac()

        blocks = self._blocks_for_function(func, fallback_to_all=True)

        lines: List[str] = []
        lines.append(f"function {func.name}:")
        if func.selector:
            lines.append(f"  // Function selector: {func.selector}")
        lines.append(f"  // Entry block: {func.entry_block}")
        lines.append("")

        for b in blocks:
            self._format_block_tac(lines, b, indent="  ")

        return "\n".join(lines)

    def _blocks_for_function(
        self, func: Function, fallback_to_all: bool = False
    ) -> List[BasicBlock]:
        """Return blocks reachable from a function entry without global duplication."""
        reachable_ids: set = set()

        def _walk(bid: str) -> None:
            if bid in reachable_ids or bid not in self.basic_blocks:
                return
            reachable_ids.add(bid)
            for s in self.basic_blocks[bid].successors:
                _walk(s)

        entry = func.entry_block
        _walk(entry)

        # If walk found nothing (entry not in basic_blocks), use func.basic_blocks
        if not reachable_ids and func.basic_blocks:
            reachable_ids = {b.id for b in func.basic_blocks}

        if fallback_to_all and not reachable_ids:
            reachable_ids = set(self.basic_blocks.keys())

        blocks = [
            self.basic_blocks[bid]
            for bid in reachable_ids
            if bid in self.basic_blocks
        ]
        blocks.sort(key=lambda b: b.start_address)
        return blocks

    def generate_per_function_tac(self) -> Dict[str, str]:
        """Run the full pipeline and return a dict mapping function name → TAC string.

        This is the preferred entry point for the inference pipeline.
        Each TAC string is small enough to fit within the model's
        context window.
        """
        logger.info("Starting per-function TAC generation")
        self.analyze_control_flow()
        self.identify_functions()
        self._convert_and_integrate_tac()

        result: Dict[str, str] = {}
        for fname, func in self.functions.items():
            result[fname] = self.generate_function_tac(func)
        return result

    # ------------------------------------------------------------------ #
    #  Full TAC Generation Pipeline
    # ------------------------------------------------------------------ #

    def generate_tac_representation(self) -> str:
        """Run the full pipeline and return a formatted TAC string."""
        try:
            logger.info("Starting comprehensive TAC generation")
            self.analyze_control_flow()
            self.identify_functions()
            self._convert_and_integrate_tac()
            return self._format_integrated_tac_output()
        except Exception as e:
            logger.error("TAC generation failed: %s", e)
            return self._generate_fallback_tac()

    def _convert_and_integrate_tac(self) -> None:
        """Convert raw instructions in each block to TAC and attach metadata."""
        entry_heights = self._compute_block_entry_stack_heights()
        exit_stacks: Dict[str, List[str]] = {}
        actual_entries = set(self._entry_blocks(self.basic_blocks))

        for bid, block in sorted(self.basic_blocks.items(), key=lambda item: item[1].start_address):
            tac_list: List[TACInstruction] = []
            stack = self._entry_stack_for_block(bid, block, exit_stacks, entry_heights, actual_entries)

            for instr in block.metadata.get("raw_instructions", []):
                result = self._convert_instruction_to_tac(instr, stack)
                if result:
                    if isinstance(result, list):
                        tac_list.extend(result)
                    else:
                        tac_list.append(result)

            for t in tac_list:
                if not t.metadata:
                    t.metadata = {}
                t.metadata.update(block_id=bid, block_start=block.start_address, block_end=block.end_address)

            block.instructions = tac_list
            self._add_control_flow_metadata_to_block(block)
            block.metadata["is_entry_block"] = bid in actual_entries
            exit_stacks[bid] = list(stack)

    def _compute_block_entry_stack_heights(self) -> Dict[str, int]:
        """Compute conservative incoming stack heights for each reachable block."""
        if not self.basic_blocks:
            return {}
        heights: Dict[str, int] = {entry: 0 for entry in self._entry_blocks(self.basic_blocks)}
        worklist: List[str] = sorted(heights, key=lambda bid: self.basic_blocks[bid].start_address)
        iterations = 0
        max_iterations = max(1, len(self.basic_blocks) * 128)

        while worklist and iterations < max_iterations:
            iterations += 1
            bid = worklist.pop(0)
            block = self.basic_blocks.get(bid)
            if block is None:
                continue
            exit_height = self._simulate_block_stack_height(
                block.metadata.get("raw_instructions", []),
                heights.get(bid, 0),
            )
            for succ in block.successors:
                if succ not in self.basic_blocks:
                    continue
                merged = max(heights.get(succ, 0), exit_height)
                if succ not in heights or merged != heights[succ]:
                    heights[succ] = min(merged, 1024)
                    if succ not in worklist:
                        worklist.append(succ)
                        worklist.sort(key=lambda item: self.basic_blocks[item].start_address)

        return heights

    def _simulate_block_stack_height(self, raw_instructions: List[Any], entry_height: int) -> int:
        """Apply EVM stack effects to a height without creating TAC temps."""
        stack: List[str] = ["_"] * min(max(entry_height, 0), 1024)

        def pop() -> None:
            if stack:
                stack.pop()

        def pop_many(count: int) -> None:
            for _ in range(count):
                pop()

        for instr in raw_instructions:
            name = self._get_instruction_name(instr)
            if name == "JUMPDEST":
                continue
            if name.startswith("PUSH"):
                stack.append("_")
            elif name == "POP":
                pop()
            elif name.startswith("DUP"):
                stack.append("_")
            elif name.startswith("SWAP"):
                continue
            elif name in _BINARY_OPS:
                pop_many(2)
                stack.append("_")
            elif name in ("ADDMOD", "MULMOD"):
                pop_many(3)
                stack.append("_")
            elif name in ("ISZERO", "NOT"):
                pop()
                stack.append("_")
            elif name in ("MLOAD", "SLOAD", "TLOAD", "CALLDATALOAD"):
                pop()
                stack.append("_")
            elif name in ("MSTORE", "MSTORE8", "SSTORE", "TSTORE"):
                pop_many(2)
            elif name in ("SHA3", "KECCAK256"):
                pop_many(2)
                stack.append("_")
            elif name in ("CALLDATACOPY", "CODECOPY", "RETURNDATACOPY", "MCOPY"):
                pop_many(3)
            elif name == "EXTCODECOPY":
                pop_many(4)
            elif name in _ENV_OPS:
                stack.append("_")
            elif name in ("BALANCE", "EXTCODESIZE", "EXTCODEHASH", "BLOCKHASH", "BLOBHASH"):
                pop()
                stack.append("_")
            elif name == "JUMP":
                pop()
            elif name == "JUMPI":
                pop_many(2)
            elif name in ("RETURN", "REVERT", "LOG0"):
                pop_many(2)
            elif name.startswith("LOG"):
                try:
                    pop_many(2 + int(name[3:]))
                except ValueError:
                    pop_many(2)
            elif name in ("CALL", "CALLCODE"):
                pop_many(7)
                stack.append("_")
            elif name in ("DELEGATECALL", "STATICCALL"):
                pop_many(6)
                stack.append("_")
            elif name == "CREATE":
                pop_many(3)
                stack.append("_")
            elif name == "CREATE2":
                pop_many(4)
                stack.append("_")
            elif name == "SELFDESTRUCT":
                pop()
            else:
                effects = _EVM_STACK_EFFECTS.get(name)
                if effects is not None:
                    pops, pushes = effects
                    pop_many(pops)
                    stack.extend(["_"] * pushes)
            if len(stack) > 1024:
                stack = stack[-1024:]

        return len(stack)

    def _entry_stack_for_block(
        self,
        bid: str,
        block: BasicBlock,
        exit_stacks: Dict[str, List[str]],
        entry_heights: Dict[str, int],
        actual_entries: set,
    ) -> List[str]:
        """Merge predecessor exit stacks into a deterministic block entry stack."""
        if bid in actual_entries:
            return []

        available_predecessors = [
            (pred, exit_stacks[pred])
            for pred in block.predecessors
            if pred in exit_stacks
        ]
        predecessor_stacks = [stack for _, stack in available_predecessors]
        expected_height = entry_heights.get(bid, 0)

        if not predecessor_stacks:
            return [f"phi_{bid}_{i}" for i in range(expected_height)]

        height = max([expected_height] + [len(s) for s in predecessor_stacks])
        merged: List[str] = []
        for i in range(height):
            values = [
                s[i] if i < len(s) else f"missing_{pred}_{i}"
                for pred, s in available_predecessors
            ]
            if values and all(v == values[0] for v in values) and len(predecessor_stacks) == len(block.predecessors):
                merged.append(values[0])
            elif len(predecessor_stacks) == 1 and i < len(predecessor_stacks[0]) and len(block.predecessors) == 1:
                merged.append(predecessor_stacks[0][i])
            else:
                merged.append(f"phi_{bid}_{i}")
        return merged

    @staticmethod
    def _add_control_flow_metadata_to_block(block: BasicBlock) -> None:
        """Annotate *block* with structural metadata."""
        m = block.metadata
        m["num_predecessors"] = len(block.predecessors)
        m["num_successors"] = len(block.successors)
        m["is_entry_block"] = len(block.predecessors) == 0
        m["is_exit_block"] = len(block.successors) == 0

        n_succ = len(block.successors)
        if n_succ == 0:
            m["block_type"] = "exit"
        elif n_succ == 1:
            m["block_type"] = "sequential"
        elif n_succ == 2:
            m["block_type"] = "conditional"
        else:
            m["block_type"] = "complex"

        m.setdefault("is_loop_header", False)
        m.setdefault("is_reachable", True)

    # ------------------------------------------------------------------ #
    #  Output Formatting
    # ------------------------------------------------------------------ #

    def _format_integrated_tac_output(self) -> str:
        lines: List[str] = [
            "// Three-Address Code Representation with Control Flow Analysis",
            "// Generated from comprehensive EVM bytecode analysis",
            "",
            "// Analysis Summary:",
            f"//   Total instructions: {len(self.instructions)}",
            f"//   Basic blocks: {len(self.basic_blocks)}",
            f"//   Functions identified: {len(self.functions)}",
            "",
        ]
        if self.functions:
            for fname, func in self.functions.items():
                self._format_function_tac(lines, fname, func)
        else:
            self._format_all_blocks_tac(lines)
        return "\n".join(lines)

    def _format_function_tac(self, lines: List[str], fname: str, func: Function) -> None:
        lines.append(f"function {fname}:")
        if func.selector:
            lines.append(f"  // Function selector: {func.selector}")
        lines.append(f"  // Entry block: {func.entry_block}")
        lines.append(f"  // Visibility: {func.visibility}")
        if func.is_payable:
            lines.append("  // Payable: true")
        if func.is_view:
            lines.append("  // View: true")
        lines.append("")

        blocks = self._blocks_for_function(func, fallback_to_all=False)
        if not blocks:
            lines.append("  // No reachable basic blocks found for this function")
            lines.append("")
            return
        for b in sorted(blocks, key=lambda x: x.start_address):
            self._format_block_tac(lines, b, indent="  ")

    def _format_all_blocks_tac(self, lines: List[str]) -> None:
        lines.append("main:")
        lines.append("  // No specific functions identified - showing all basic blocks")
        lines.append("")
        for b in sorted(self.basic_blocks.values(), key=lambda x: x.start_address):
            self._format_block_tac(lines, b, indent="  ")

    def _format_block_tac(self, lines: List[str], block: BasicBlock, indent: str = "") -> None:
        lines.append(f"{indent}{block.id}:")
        lines.append(f"{indent}  // Address range: {block.start_address:04x} - {block.end_address:04x}")
        if block.predecessors:
            lines.append(f"{indent}  // Predecessors: {', '.join(block.predecessors)}")
        if block.successors:
            lines.append(f"{indent}  // Successors: {', '.join(block.successors)}")
        if "block_type" in block.metadata:
            lines.append(f"{indent}  // Block type: {block.metadata['block_type']}")
        if block.metadata.get("is_loop_header"):
            lines.append(f"{indent}  // Loop header")
        if block.metadata.get("is_dead_code"):
            lines.append(f"{indent}  // Dead code (unreachable)")
        if block.instructions:
            for t in block.instructions:
                lines.append(f"{indent}    {self._format_tac_instruction(t)}")
        else:
            lines.append(f"{indent}    // No TAC instructions")
        lines.append("")

    def _generate_fallback_tac(self) -> str:
        logger.warning("Using fallback TAC generation")
        lines: List[str] = [
            "// Three-Address Code Representation (Fallback Mode)",
            "// Basic analysis due to errors in comprehensive mode",
            "",
        ]
        try:
            stack: List[str] = []
            lines.append("main:")
            for instr in self.instructions:
                result = self._convert_instruction_to_tac(instr, stack)
                if result:
                    items = result if isinstance(result, list) else [result]
                    for t in items:
                        lines.append(f"  {self._format_tac_instruction(t)}")
        except Exception as e:
            logger.error("Fallback TAC generation also failed: %s", e)
            lines.append("// Error: Unable to generate TAC representation")
            lines.append(f"// {e}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  TAC Instruction Formatter
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_tac_instruction(instr: TACInstruction) -> str:
        """Return a human-readable string for *instr*."""
        op = instr.operation
        meta = instr.metadata or {}

        if op == TACOperationType.ASSIGN:
            return f"{instr.result} = {instr.operand1 or '<unknown>'}"

        if op == TACOperationType.BINARY_OP:
            if instr.operand2:
                return f"{instr.result} = {instr.operand1} {instr.operator} {instr.operand2}"
            return f"{instr.result} = {instr.operand1}"

        if op == TACOperationType.UNARY_OP:
            return f"{instr.result} = {instr.operator}({instr.operand1})"

        if op == TACOperationType.LOAD:
            mtype = meta.get("memory_type", "memory")
            return f"{instr.result} = {mtype}[{instr.operand1}]"

        if op == TACOperationType.STORE:
            mtype = meta.get("memory_type", "memory")
            return f"{mtype}[{instr.operand1}] = {instr.operand2}"

        if op == TACOperationType.CALL:
            op_name = meta.get("original_op", "call").lower()
            return f"{instr.result} = {op_name}({instr.operand1})"

        if op == TACOperationType.JUMP:
            return f"goto {instr.target}"

        if op == TACOperationType.CONDITIONAL_JUMP:
            return f"if {instr.operand1} goto {instr.target}"

        if op == TACOperationType.RETURN:
            return f"return memory[{instr.operand1}:{instr.operand2}]"

        if op == TACOperationType.REVERT:
            decoded = meta.get("revert_decoded")
            if decoded:
                return f"revert {decoded['message']}"
            return f"revert memory[{instr.operand1}:{instr.operand2}]"

        if op == TACOperationType.HALT:
            op_name = meta.get("original_op", "halt").lower()
            return f"{op_name}({instr.operand1})" if instr.operand1 else f"{op_name}()"

        if op == TACOperationType.LOG:
            tc = meta.get("topic_count", 0)
            topics = meta.get("topics") or []
            topic_args = [
                f"topic{i}={topic}" for i, topic in enumerate(topics)
            ]
            args = [f"memory[{instr.operand1}:{instr.operand2}]"] + topic_args
            return f"log{tc}({', '.join(args)})"

        if op == TACOperationType.NOP:
            if meta.get("shared_ref") and meta.get("comment"):
                return f"// {meta['comment']}"
            return "// nop"

        return f"// {op.value}: {meta}"

    # ------------------------------------------------------------------ #
    #  Vulnerability-Indicative CFG Fragment Extraction (SmartBugBert)
    # ------------------------------------------------------------------ #

    def extract_vulnerability_fragments(self) -> Dict[str, List[Dict]]:
        """
        Extract CFG fragments that indicate potential vulnerabilities.

        Based on SmartBugBert (2504.05002v2) methodology of identifying
        vulnerability patterns in control flow graph basic blocks.

        Returns dict mapping vulnerability type to list of fragment dicts,
        each containing block_id, opcodes, and pattern description.
        """
        if not self.basic_blocks:
            self.analyze_control_flow()

        fragments: Dict[str, List[Dict]] = {
            "reentrancy": [],
            "selfdestruct": [],
            "timestamp_dependency": [],
            "arithmetic_overflow": [],
            "delegatecall": [],
            "access_control": [],
        }

        block_opcodes = {}
        for bid, block in self.basic_blocks.items():
            opcodes = []
            for instr in block.metadata.get("raw_instructions", []):
                name = self._get_instruction_name(instr)
                opcodes.append(name)
            block_opcodes[bid] = opcodes

        for bid, opcodes in block_opcodes.items():
            block = self.basic_blocks[bid]

            # Reentrancy: CALL/CALLCODE followed by SSTORE in same or successor blocks
            call_ops = {"CALL", "CALLCODE"}
            if any(op in call_ops for op in opcodes):
                has_sstore_after = "SSTORE" in opcodes
                if not has_sstore_after:
                    for succ_id in block.successors:
                        if succ_id in block_opcodes and "SSTORE" in block_opcodes[succ_id]:
                            has_sstore_after = True
                            break
                if has_sstore_after:
                    fragments["reentrancy"].append({
                        "block_id": bid,
                        "opcodes": opcodes,
                        "pattern": "External call followed by state modification (SSTORE)",
                        "severity": "high",
                    })

            # Self-destruct vulnerability
            if "SELFDESTRUCT" in opcodes:
                fragments["selfdestruct"].append({
                    "block_id": bid,
                    "opcodes": opcodes,
                    "pattern": "SELFDESTRUCT opcode present",
                    "severity": "critical",
                })

            # Timestamp dependency
            if "TIMESTAMP" in opcodes:
                fragments["timestamp_dependency"].append({
                    "block_id": bid,
                    "opcodes": opcodes,
                    "pattern": "Block timestamp used in execution logic",
                    "severity": "medium",
                })

            # Arithmetic without overflow checks (ADD/SUB/MUL without LT/GT/EQ + JUMPI)
            arith_ops = {"ADD", "SUB", "MUL", "DIV"}
            check_ops = {"LT", "GT", "EQ", "ISZERO"}
            if any(op in arith_ops for op in opcodes):
                has_check = any(op in check_ops for op in opcodes) and "JUMPI" in opcodes
                if not has_check:
                    fragments["arithmetic_overflow"].append({
                        "block_id": bid,
                        "opcodes": opcodes,
                        "pattern": "Arithmetic operation without overflow/underflow check",
                        "severity": "medium",
                    })

            # Delegatecall vulnerability
            if "DELEGATECALL" in opcodes:
                fragments["delegatecall"].append({
                    "block_id": bid,
                    "opcodes": opcodes,
                    "pattern": "DELEGATECALL used - potential storage manipulation",
                    "severity": "high",
                })

            # Access control: CALLER check missing before sensitive operations
            sensitive_ops = {"SELFDESTRUCT", "DELEGATECALL", "SSTORE"}
            if any(op in sensitive_ops for op in opcodes):
                has_caller_check = False
                for pred_id in block.predecessors:
                    if pred_id in block_opcodes:
                        pred_ops = block_opcodes[pred_id]
                        if "CALLER" in pred_ops and "EQ" in pred_ops:
                            has_caller_check = True
                            break
                if not has_caller_check and "CALLER" not in opcodes:
                    fragments["access_control"].append({
                        "block_id": bid,
                        "opcodes": opcodes,
                        "pattern": "Sensitive operation without caller verification",
                        "severity": "high",
                    })

        # Remove empty categories
        return {k: v for k, v in fragments.items() if v}


# ---------------------------------------------------------------------------
# Convenience Function
# ---------------------------------------------------------------------------


def analyze_bytecode_to_tac(bytecode: str) -> str:
    """Create a :class:`BytecodeAnalyzer` and return the TAC string."""
    return BytecodeAnalyzer(bytecode).generate_tac_representation()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "0x608060405234801561001057600080fd5b50600436106100365760003560e01c"
        "8063893d20e81461003b578063a6f9dae114610059575b600080fd5b610043610075565b"
        "6040516100509190610166565b60405180910390f35b610073600480360381019061006e"
        "91906101b2565b61009e565b005b60008060009054906101000a900473ffffffffffff"
        "ffffffffffffffffffffffffffff16905090565b8073ffffffffffffffffffffffffff"
        "ffffffffffffff163373ffffffffffffffffffffffffffffffffffffffff16036100d3"
        "57806000806101000a81548173ffffffffffffffffffffffffffffffffffffffff0219"
        "16908373ffffffffffffffffffffffffffffffffffffffff1602179055505b50565b60"
        "0073ffffffffffffffffffffffffffffffffffffffff82169050919050565b60006101"
        "01826100d6565b9050919050565b610111816100f6565b82525050565b600060208201"
        "905061012c6000830184610108565b92915050565b600080fd5b610140816100f6565b"
        "811461014b57600080fd5b50565b60008135905061015d81610137565b92915050565b"
        "60006020828403121561017957610178610132565b5b60006101878482850161014e56"
        "5b91505092915050565b7f4e487b7100000000000000000000000000000000000000000"
        "000000000000000600052602260045260246000fd5b600060028204905060018216806101"
        "d857607f821691505b6020821081036101eb576101ea610190565b5b5091905056fea264"
        "6970667358221220"
        "9d84a3c5d1d6c4c5f9c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5"
        "64736f6c634300080a0033"
    )
    try:
        print("TAC Representation:")
        print(analyze_bytecode_to_tac(sample))
    except Exception as exc:
        print(f"Error: {exc}")