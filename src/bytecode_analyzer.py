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

from evmdasm import EvmBytecode

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

# EVM opcode stack effects: (pops, pushes) for opcodes handled by the
# generic fallback in the stack simulator and TAC converter.
_EVM_STACK_EFFECTS: Dict[str, Tuple[int, int]] = {
    # Environmental / context opcodes  (0 pops, 1 push)
    "ADDRESS": (0, 1), "ORIGIN": (0, 1), "CALLER": (0, 1),
    "CALLVALUE": (0, 1), "CALLDATASIZE": (0, 1), "CODESIZE": (0, 1),
    "GASPRICE": (0, 1), "RETURNDATASIZE": (0, 1), "COINBASE": (0, 1),
    "TIMESTAMP": (0, 1), "NUMBER": (0, 1), "DIFFICULTY": (0, 1),
    "PREVRANDAO": (0, 1), "GASLIMIT": (0, 1), "CHAINID": (0, 1),
    "SELFBALANCE": (0, 1), "BASEFEE": (0, 1), "GAS": (0, 1),
    "PC": (0, 1), "MSIZE": (0, 1),
    # 1-pop, 1-push
    "BALANCE": (1, 1), "EXTCODESIZE": (1, 1), "EXTCODEHASH": (1, 1),
    "BLOCKHASH": (1, 1), "CALLDATALOAD": (1, 1),
    # 1-pop, 0-push
    "POP": (1, 0),
    # 2-pop, 0-push
    "MSTORE8": (2, 0),
    # 3-pop, 0-push
    "CALLDATACOPY": (3, 0), "CODECOPY": (3, 0), "RETURNDATACOPY": (3, 0),
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
    "GASLIMIT", "CHAINID", "SELFBALANCE", "BASEFEE",
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


# ---------------------------------------------------------------------------
# Main Analyzer
# ---------------------------------------------------------------------------


class BytecodeAnalyzer:
    """Analyze EVM bytecode and convert it to Three-Address Code.

    Based on the static analysis approach described in the paper,
    this class performs control flow analysis, function boundary
    identification, and TAC generation.
    """

    def __init__(self, bytecode: str) -> None:
        self.bytecode: str = bytecode
        self.instructions: List = []
        self.basic_blocks: Dict[str, BasicBlock] = {}
        self.functions: Dict[str, Function] = {}
        self.variable_counter: int = 0
        self.logger = logging.getLogger(__name__)

        # Pre-built lookup: PC → instruction index (populated after parsing)
        self._pc_to_index: Dict[int, int] = {}

        self._parse_bytecode()

    # ------------------------------------------------------------------ #
    #  Bytecode Parsing
    # ------------------------------------------------------------------ #

    def _parse_bytecode(self) -> None:
        """Parse EVM bytecode into instruction objects using *evmdasm*."""
        try:
            clean = self.bytecode[2:] if self.bytecode.startswith("0x") else self.bytecode
            evm = EvmBytecode(clean)
            self.instructions = list(evm.disassemble())

            for i, instr in enumerate(self.instructions):
                self._pc_to_index[self._get_pc(instr, i)] = i

            self.logger.info("Parsed %d instructions from bytecode", len(self.instructions))
        except Exception as e:
            self.logger.error("Failed to parse bytecode: %s", e)
            self.instructions = []

    # ------------------------------------------------------------------ #
    #  Instruction Accessors (support dict & evmdasm objects)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_instruction_name(instr: Any) -> str:
        """Return the mnemonic name of *instr*."""
        if isinstance(instr, dict):
            return instr.get("name", "UNKNOWN")
        if hasattr(instr, "name"):
            return instr.name
        if hasattr(instr, "opcode"):
            return instr.opcode.name if hasattr(instr.opcode, "name") else str(instr.opcode)
        return str(instr)

    @staticmethod
    def _get_pc(instr: Any, fallback: int) -> int:
        """Return the program counter of *instr*."""
        if isinstance(instr, dict):
            return instr.get("pc", fallback)
        return getattr(instr, "pc", fallback)

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
        self.logger.info("Starting comprehensive control flow analysis")
        try:
            jump_targets = self._detect_jump_targets()
            self.logger.info("Detected %d jump targets", len(jump_targets))

            blocks = self._construct_basic_blocks(jump_targets)
            self.logger.info("Constructed %d basic blocks", len(blocks))

            self._analyze_block_relationships(blocks)
            self.logger.info("Completed block relationship analysis")

            self._perform_advanced_analysis(blocks)
            self.logger.info("Completed advanced control flow analysis")

            self.basic_blocks = blocks
            return blocks
        except Exception as e:
            self.logger.error("Control flow analysis failed: %s", e)
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
        """Keep only the entry point and valid JUMPDEST locations."""
        if not self.instructions:
            return set()
        first_pc = self._get_pc(self.instructions[0], 0)
        valid = {
            self._get_pc(instr, i)
            for i, instr in enumerate(self.instructions)
            if self._get_instruction_name(instr) == "JUMPDEST"
        }
        return {t for t in targets if t == first_pc or t in valid}

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
        """Return entry-block IDs (no predecessors), with a fallback."""
        entries = [bid for bid, b in blocks.items() if not b.predecessors]
        if not entries and blocks:
            entries = [next(iter(blocks))]
        return entries

    # ------------------------------------------------------------------ #
    #  Fallback (error recovery)
    # ------------------------------------------------------------------ #

    def _fallback_control_flow_analysis(self) -> Dict[str, BasicBlock]:
        """Create a single basic block containing all instructions."""
        self.logger.warning("Using fallback control flow analysis")
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
        self.logger.info("Identifying functions from bytecode dispatcher")

        for i, instr in enumerate(self.instructions):
            if self._get_instruction_name(instr) != "PUSH4":
                continue

            selector_val = self._get_operand(instr)
            if not selector_val:
                continue

            selector = self._normalize_selector(selector_val)
            if selector is None:
                continue

            jump_target = self._find_dispatch_target(i)
            if jump_target is None:
                continue

            fname = f"function_{selector}"
            entry = self._block_at_address(jump_target) or f"block_{jump_target:04x}"
            functions[fname] = Function(
                name=fname, selector=selector, basic_blocks=[], entry_block=entry,
            )
            self.logger.debug("Identified function %s → %04x", selector, jump_target)

        self.logger.info("Identified %d functions from dispatcher", len(functions))

        if not functions and self.basic_blocks:
            entry = next(iter(self.basic_blocks))
            functions["fallback"] = Function(
                name="fallback", selector=None,
                basic_blocks=list(self.basic_blocks.values()), entry_block=entry,
            )
            self.logger.info("No functions found – created fallback function")

        self.functions = functions
        return functions

    # -- helpers --

    @staticmethod
    def _normalize_selector(raw: Any) -> Optional[str]:
        """Return a ``0x``-prefixed 8-hex-digit selector string, or ``None``."""
        if isinstance(raw, str):
            sel = raw if raw.startswith("0x") else "0x" + raw
        else:
            sel = "0x" + format(int(raw), "08x")
        return sel if len(sel) == 10 else None

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
            operand = self._get_operand(instr)
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
            if len(stack) >= 2:
                a, b = stack.pop(), stack.pop()
            else:
                a, b = "stack_underflow", "stack_underflow"
            tmp = self._generate_temp_var()
            stack.append(tmp)
            return TACInstruction(
                TACOperationType.BINARY_OP, result=tmp,
                operand1=a, operand2=b, operator=_BINARY_OPS[name],
                metadata={"original_op": name},
            )

        # -- Ternary: ADDMOD / MULMOD --
        if name in ("ADDMOD", "MULMOD"):
            sym = "addmod" if name == "ADDMOD" else "mulmod"
            a, b, n = _stack_pop(stack), _stack_pop(stack), _stack_pop(stack)
            tmp = self._generate_temp_var()
            stack.append(tmp)
            return TACInstruction(
                TACOperationType.BINARY_OP, result=tmp,
                operand1=f"{sym}({a}, {b}, {n})",
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
        if name in ("BALANCE", "EXTCODESIZE", "EXTCODEHASH", "BLOCKHASH"):
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
            return TACInstruction(
                TACOperationType.REVERT, operand1=off, operand2=sz,
                metadata={"original_op": name},
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
        # Totally unknown — push one temp
        result_var = self._generate_temp_var()
        stack.append(result_var)
        return TACInstruction(
            TACOperationType.ASSIGN, result=result_var,
            metadata={"original_op": name, "unhandled": True,
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

            if name.startswith("PUSH"):
                try:
                    operand = getattr(instr, "operand", None)
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
            if hasattr(instr, "name"):
                return instr.name
            if hasattr(instr, "opcode"):
                return instr.opcode.name if hasattr(instr.opcode, "name") else str(instr.opcode)
            return str(instr)

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
        # Collect blocks reachable from the entry block
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

        # Fall back to all blocks if still empty
        if not reachable_ids:
            reachable_ids = set(self.basic_blocks.keys())

        blocks = [
            self.basic_blocks[bid]
            for bid in reachable_ids
            if bid in self.basic_blocks
        ]
        blocks.sort(key=lambda b: b.start_address)

        lines: List[str] = []
        lines.append(f"function {func.name}:")
        if func.selector:
            lines.append(f"  // Function selector: {func.selector}")
        lines.append(f"  // Entry block: {func.entry_block}")
        lines.append("")

        for b in blocks:
            self._format_block_tac(lines, b, indent="  ")

        return "\n".join(lines)

    def generate_per_function_tac(self) -> Dict[str, str]:
        """Run the full pipeline and return a dict mapping function name → TAC string.

        This is the preferred entry point for the inference pipeline.
        Each TAC string is small enough to fit within the model's
        context window.
        """
        self.logger.info("Starting per-function TAC generation")
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
            self.logger.info("Starting comprehensive TAC generation")
            self.analyze_control_flow()
            self.identify_functions()
            self._convert_and_integrate_tac()
            return self._format_integrated_tac_output()
        except Exception as e:
            self.logger.error("TAC generation failed: %s", e)
            return self._generate_fallback_tac()

    def _convert_and_integrate_tac(self) -> None:
        """Convert raw instructions in each block to TAC and attach metadata."""
        for bid, block in self.basic_blocks.items():
            tac_list: List[TACInstruction] = []
            stack: List[str] = []

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

        blocks = func.basic_blocks or list(self.basic_blocks.values())
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
        self.logger.warning("Using fallback TAC generation")
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
            self.logger.error("Fallback TAC generation also failed: %s", e)
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
            return f"revert memory[{instr.operand1}:{instr.operand2}]"

        if op == TACOperationType.HALT:
            op_name = meta.get("original_op", "halt").lower()
            return f"{op_name}({instr.operand1})" if instr.operand1 else f"{op_name}()"

        if op == TACOperationType.LOG:
            tc = meta.get("topic_count", 0)
            return f"log{tc}(memory[{instr.operand1}:{instr.operand2}])"

        if op == TACOperationType.NOP:
            return "// nop"

        return f"// {op.value}: {meta}"


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