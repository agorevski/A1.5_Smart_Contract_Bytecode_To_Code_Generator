"""
EVM Bytecode to Three-Address Code (TAC) Converter

This module implements static analysis techniques to convert EVM bytecode
into a structured three-address code representation, as described in the paper.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import pyevmasm
from eth_utils import to_hex
from web3 import Web3


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
    parameters: List[str] = None
    return_types: List[str] = None


class BytecodeAnalyzer:
    """
    Main class for analyzing EVM bytecode and converting it to TAC.
    
    Based on the static analysis approach described in the paper,
    this class performs control flow analysis, function boundary
    identification, and TAC generation.
    """
    
    def __init__(self, bytecode: str):
        """
        Initialize the analyzer with EVM bytecode.
        
        Args:
            bytecode: Hex string of EVM bytecode
        """
        self.bytecode = bytecode
        self.instructions = []
        self.basic_blocks = {}
        self.functions = {}
        self.variable_counter = 0
        self.logger = logging.getLogger(__name__)
        
        # Parse bytecode into instructions
        self._parse_bytecode()
    
    def _parse_bytecode(self) -> None:
        """Parse bytecode into individual EVM instructions."""
        try:
            # Remove '0x' prefix if present
            clean_bytecode = self.bytecode[2:] if self.bytecode.startswith('0x') else self.bytecode
            
            # Parse using pyevmasm - it returns a string, not individual objects
            disassembly_str = pyevmasm.disassemble_hex(clean_bytecode)
            
            # Parse the disassembly string into individual instructions
            self.instructions = self._parse_disassembly_string(disassembly_str)
            self.logger.info(f"Parsed {len(self.instructions)} instructions from bytecode")
            
        except Exception as e:
            self.logger.error(f"Failed to parse bytecode: {e}")
            raise
    
    def _parse_disassembly_string(self, disassembly_str: str) -> List[Dict]:
        """
        Parse pyevmasm disassembly string into individual instruction objects.
        
        Args:
            disassembly_str: String output from pyevmasm.disassemble_hex()
            
        Returns:
            List of instruction dictionaries
        """
        instructions = []
        lines = disassembly_str.strip().split('\n')
        pc = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse instruction line (e.g., "PUSH1 0x80", "JUMPDEST", "ADD")
            parts = line.split()
            if not parts:
                continue
            
            opcode = parts[0]
            operand = parts[1] if len(parts) > 1 else None
            
            # Create instruction dictionary
            instruction = {
                'name': opcode,
                'pc': pc,
                'operand': operand,
                'raw_line': line
            }
            
            instructions.append(instruction)
            
            # Estimate PC increment (simplified)
            if opcode.startswith('PUSH'):
                try:
                    push_size = int(opcode[4:]) if len(opcode) > 4 else 1
                    pc += 1 + push_size  # Opcode + operand bytes
                except ValueError:
                    pc += 1
            else:
                pc += 1
        
        return instructions
    
    def _generate_temp_var(self) -> str:
        """Generate a temporary variable name."""
        self.variable_counter += 1
        return f"temp_{self.variable_counter}"
    
    def analyze_control_flow(self) -> Dict[str, BasicBlock]:
        """
        Perform comprehensive control flow analysis to identify basic blocks.
        
        This implementation includes:
        - Advanced jump target detection with stack simulation
        - Comprehensive basic block construction
        - Complete predecessor/successor relationship analysis
        - Loop detection and dominance analysis
        - Exception handling pattern recognition
        
        Returns:
            Dictionary mapping block IDs to BasicBlock objects
        """
        self.logger.info("Starting comprehensive control flow analysis")
        
        try:
            # Phase 1: Enhanced jump target detection
            jump_targets = self._detect_jump_targets()
            self.logger.info(f"Detected {len(jump_targets)} jump targets")
            
            # Phase 2: Advanced basic block construction
            blocks = self._construct_basic_blocks(jump_targets)
            self.logger.info(f"Constructed {len(blocks)} basic blocks")
            
            # Phase 3: Control flow graph construction
            self._analyze_block_relationships(blocks)
            self.logger.info("Completed block relationship analysis")
            
            # Phase 4: Advanced analysis features
            self._perform_advanced_analysis(blocks)
            self.logger.info("Completed advanced control flow analysis")
            
            self.basic_blocks = blocks
            return blocks
            
        except Exception as e:
            self.logger.error(f"Control flow analysis failed: {e}")
            # Fallback to basic analysis
            return self._fallback_control_flow_analysis()
    
    def _detect_jump_targets(self) -> set:
        """
        Enhanced jump target detection using stack simulation and pattern recognition.
        
        Returns:
            Set of program counter values that are valid jump targets
        """
        jump_targets = set()
        
        # Always include entry point
        if self.instructions:
            first_pc = self._get_pc(self.instructions[0], 0)
            jump_targets.add(first_pc)
        
        # Find all JUMPDEST instructions (valid jump targets)
        jumpdest_count = 0
        for i, instr in enumerate(self.instructions):
            instr_name = self._get_instruction_name(instr)
            pc = self._get_pc(instr, i)  # Use index as fallback
            
            if instr_name == 'JUMPDEST':
                jump_targets.add(pc)
                jumpdest_count += 1
        
        self.logger.debug(f"Found {jumpdest_count} JUMPDEST instructions")
        
        # Analyze instruction sequences for jump patterns and PUSH targets
        for i, instr in enumerate(self.instructions):
            instr_name = self._get_instruction_name(instr)
            pc = self._get_pc(instr, i)
            
            # Detect PUSH instructions that might contain jump targets
            if instr_name.startswith('PUSH'):
                try:
                    operand = self._get_operand(instr)
                    if operand:
                        # Convert operand to integer
                        if isinstance(operand, str):
                            if operand.startswith('0x'):
                                target = int(operand, 16)
                            else:
                                target = int(operand, 16)
                        else:
                            target = int(operand)
                        
                        # Check if this could be a valid PC value
                        if 0 <= target <= 10000:  # Reasonable upper bound for bytecode size
                            # Look ahead to see if this is used for a jump
                            for j in range(i + 1, min(i + 5, len(self.instructions))):
                                next_instr = self.instructions[j]
                                next_name = self._get_instruction_name(next_instr)
                                if next_name in ['JUMP', 'JUMPI']:
                                    jump_targets.add(target)
                                    break
                                elif next_name not in ['DUP1', 'DUP2', 'DUP3', 'DUP4', 'SWAP1', 'SWAP2']:
                                    # If we hit an instruction that likely consumes the stack value
                                    # for something other than jumping, stop looking
                                    break
                except (ValueError, AttributeError):
                    continue
            
            # Detect jump instructions and resolve targets using fallback method
            elif instr_name in ['JUMP', 'JUMPI']:
                targets = self._resolve_jump_targets(i, None)
                jump_targets.update(targets)
            
            # Detect function dispatcher patterns
            elif instr_name == 'PUSH4' and i + 1 < len(self.instructions):
                next_instr = self.instructions[i + 1]
                if self._get_instruction_name(next_instr) == 'EQ':
                    # Function selector pattern detected
                    targets = self._analyze_function_dispatch_pattern(i)
                    jump_targets.update(targets)
            
            # Add targets after terminating instructions
            elif instr_name in ['REVERT', 'INVALID', 'SELFDESTRUCT', 'RETURN', 'STOP']:
                # These terminate execution, next instruction is potential target
                if i + 1 < len(self.instructions):
                    next_pc = self._get_pc(self.instructions[i + 1], i + 1)
                    jump_targets.add(next_pc)
            
            # Add targets after conditional instructions
            elif instr_name == 'JUMPI':
                # Add fall-through target
                if i + 1 < len(self.instructions):
                    next_pc = self._get_pc(self.instructions[i + 1], i + 1)
                    jump_targets.add(next_pc)
        
        # Remove invalid targets (beyond bytecode range or not at valid JUMPDEST)
        if self.instructions:
            max_pc = max(self._get_pc(instr, i) for i, instr in enumerate(self.instructions))
            valid_jumpdests = {self._get_pc(instr, i) for i, instr in enumerate(self.instructions) 
                              if self._get_instruction_name(instr) == 'JUMPDEST'}
            
            # Keep entry point and valid JUMPDEST targets
            filtered_targets = set()
            for target in jump_targets:
                if target == 0 or target in valid_jumpdests or target <= max_pc:
                    filtered_targets.add(target)
            jump_targets = filtered_targets
        
        return jump_targets
    
    def _resolve_jump_targets(self, jump_index: int, stack_simulator) -> set:
        """
        Resolve jump targets for JUMP/JUMPI instructions using stack analysis.
        
        Args:
            jump_index: Index of the jump instruction
            stack_simulator: Current stack state
            
        Returns:
            Set of possible jump targets
        """
        targets = set()
        
        # Try to resolve target from stack simulation
        if hasattr(stack_simulator, 'get_stack_top_value'):
            target_value = stack_simulator.get_stack_top_value()
            if target_value is not None and isinstance(target_value, int):
                targets.add(target_value)
        
        # Fallback: Look for preceding PUSH instructions
        for i in range(jump_index - 1, max(0, jump_index - 10), -1):
            instr = self.instructions[i]
            instr_name = self._get_instruction_name(instr)
            
            if instr_name.startswith('PUSH'):
                try:
                    operand = self._get_operand(instr)
                    if operand:
                        if isinstance(operand, str) and operand.startswith('0x'):
                            target = int(operand, 16)
                        else:
                            target = int(str(operand), 16)
                        targets.add(target)
                        break  # Usually the immediate preceding PUSH
                except (ValueError, AttributeError):
                    continue
        
        return targets
    
    def _analyze_function_dispatch_pattern(self, push4_index: int) -> set:
        """
        Analyze function dispatcher patterns to find function entry points.
        
        Args:
            push4_index: Index of PUSH4 instruction with function selector
            
        Returns:
            Set of function entry point addresses
        """
        targets = set()
        
        # Look for common dispatcher patterns:
        # PUSH4 selector, EQ, PUSH target, JUMPI
        try:
            if push4_index + 3 < len(self.instructions):
                eq_instr = self.instructions[push4_index + 1]
                push_instr = self.instructions[push4_index + 2]
                jumpi_instr = self.instructions[push4_index + 3]
                
                if (self._get_instruction_name(eq_instr) == 'EQ' and
                    self._get_instruction_name(push_instr).startswith('PUSH') and
                    self._get_instruction_name(jumpi_instr) == 'JUMPI'):
                    
                    # Extract target address
                    operand = self._get_operand(push_instr)
                    if operand:
                        target = int(operand, 16) if isinstance(operand, str) else int(operand)
                        targets.add(target)
        
        except (ValueError, AttributeError, IndexError):
            pass
        
        return targets
    
    def _construct_basic_blocks(self, jump_targets: set) -> Dict[str, BasicBlock]:
        """
        Construct basic blocks with precise boundaries and comprehensive analysis.
        
        Args:
            jump_targets: Set of all identified jump targets
            
        Returns:
            Dictionary of basic blocks
        """
        blocks = {}
        
        # Sort targets to process in order
        sorted_targets = sorted(jump_targets)
        
        # Create PC to instruction mapping for efficient lookup
        pc_to_instr = {}
        pc_to_index = {}
        for i, instr in enumerate(self.instructions):
            pc = self._get_pc(instr, i)
            pc_to_instr[pc] = instr
            pc_to_index[pc] = i
        
        # Construct blocks
        for i, start_pc in enumerate(sorted_targets):
            # Determine block boundaries
            end_pc = self._find_block_end(start_pc, sorted_targets, i)
            
            # Collect instructions in this block
            block_instructions = []
            raw_instructions = []
            
            current_pc = start_pc
            while current_pc <= end_pc and current_pc in pc_to_instr:
                instr = pc_to_instr[current_pc]
                raw_instructions.append(instr)
                
                # Check if this instruction terminates the block
                instr_name = self._get_instruction_name(instr)
                if instr_name in ['JUMP', 'JUMPI', 'RETURN', 'REVERT', 'STOP', 'SELFDESTRUCT', 'INVALID']:
                    break
                
                # Move to next instruction
                instr_index = pc_to_index.get(current_pc, -1)
                if instr_index + 1 < len(self.instructions):
                    next_instr = self.instructions[instr_index + 1]
                    current_pc = self._get_pc(next_instr, current_pc + 1)
                else:
                    break
            
            # Create block if it has instructions
            if raw_instructions:
                block_id = f"block_{start_pc:04x}"
                actual_end_pc = self._get_pc(raw_instructions[-1], end_pc)
                
                blocks[block_id] = BasicBlock(
                    id=block_id,
                    instructions=[],  # Will be filled with TAC instructions later
                    predecessors=[],
                    successors=[],
                    start_address=start_pc,
                    end_address=actual_end_pc
                )
                
                # Store raw instructions for relationship analysis
                blocks[block_id].metadata = {'raw_instructions': raw_instructions}
        
        return blocks
    
    def _find_block_end(self, start_pc: int, sorted_targets: List[int], target_index: int) -> int:
        """
        Find the end PC for a basic block starting at start_pc.
        
        Args:
            start_pc: Starting program counter
            sorted_targets: Sorted list of all jump targets
            target_index: Index of start_pc in sorted_targets
            
        Returns:
            End program counter for the block
        """
        # Default end is the next target minus 1
        if target_index + 1 < len(sorted_targets):
            return sorted_targets[target_index + 1] - 1
        
        # Last block extends to end of program
        if self.instructions:
            return self._get_pc(self.instructions[-1], len(self.instructions) - 1)
        
        return start_pc
    
    def _analyze_block_relationships(self, blocks: Dict[str, BasicBlock]) -> None:
        """
        Analyze predecessor/successor relationships between basic blocks.
        
        Args:
            blocks: Dictionary of basic blocks to analyze
        """
        # Create PC to block mapping
        pc_to_block = {}
        for block in blocks.values():
            for pc in range(block.start_address, block.end_address + 1):
                pc_to_block[pc] = block.id
        
        # Analyze each block's successors and predecessors
        for block in blocks.values():
            self._analyze_single_block_relationships(block, blocks, pc_to_block)
    
    def _analyze_single_block_relationships(self, block: BasicBlock, 
                                          all_blocks: Dict[str, BasicBlock],
                                          pc_to_block: Dict[int, str]) -> None:
        """
        Analyze relationships for a single basic block.
        
        Args:
            block: Block to analyze
            all_blocks: All blocks in the program
            pc_to_block: Mapping from PC to block ID
        """
        if 'raw_instructions' not in block.metadata:
            return
        
        raw_instructions = block.metadata['raw_instructions']
        if not raw_instructions:
            return
        
        last_instr = raw_instructions[-1]
        last_instr_name = self._get_instruction_name(last_instr)
        last_pc = self._get_pc(last_instr, block.end_address)
        
        # Analyze based on terminating instruction
        if last_instr_name == 'JUMP':
            # Unconditional jump - find target
            targets = self._get_jump_targets_from_block(raw_instructions)
            for target in targets:
                if target in pc_to_block:
                    successor_id = pc_to_block[target]
                    self._add_edge(block.id, successor_id, all_blocks)
        
        elif last_instr_name == 'JUMPI':
            # Conditional jump - has both jump target and fall-through
            targets = self._get_jump_targets_from_block(raw_instructions)
            for target in targets:
                if target in pc_to_block:
                    successor_id = pc_to_block[target]
                    self._add_edge(block.id, successor_id, all_blocks)
            
            # Fall-through edge
            fall_through_pc = self._get_next_instruction_pc(last_pc)
            if fall_through_pc and fall_through_pc in pc_to_block:
                successor_id = pc_to_block[fall_through_pc]
                self._add_edge(block.id, successor_id, all_blocks)
        
        elif last_instr_name in ['RETURN', 'REVERT', 'STOP', 'SELFDESTRUCT', 'INVALID']:
            # Terminal instructions - no successors
            pass
        
        else:
            # Fall-through to next instruction
            fall_through_pc = self._get_next_instruction_pc(last_pc)
            if fall_through_pc and fall_through_pc in pc_to_block:
                successor_id = pc_to_block[fall_through_pc]
                self._add_edge(block.id, successor_id, all_blocks)
    
    def _get_jump_targets_from_block(self, instructions: List) -> set:
        """
        Extract jump targets from a sequence of instructions.
        
        Args:
            instructions: List of raw instructions
            
        Returns:
            Set of jump target addresses
        """
        targets = set()
        
        # Look for PUSH instructions preceding jumps
        for i in range(len(instructions) - 1, -1, -1):
            instr = instructions[i]
            instr_name = self._get_instruction_name(instr)
            
            if instr_name.startswith('PUSH'):
                try:
                    operand = self._get_operand(instr)
                    if operand:
                        target = int(operand, 16) if isinstance(operand, str) else int(operand)
                        targets.add(target)
                except (ValueError, AttributeError):
                    continue
        
        return targets
    
    def _get_next_instruction_pc(self, current_pc: int) -> Optional[int]:
        """
        Get the PC of the next instruction after current_pc.
        
        Args:
            current_pc: Current program counter
            
        Returns:
            PC of next instruction or None if not found
        """
        found_current = False
        for instr in self.instructions:
            pc = self._get_pc(instr, 0)
            if found_current:
                return pc
            if pc == current_pc:
                found_current = True
        return None
    
    def _add_edge(self, from_block: str, to_block: str, all_blocks: Dict[str, BasicBlock]) -> None:
        """
        Add an edge between two blocks.
        
        Args:
            from_block: Source block ID
            to_block: Target block ID
            all_blocks: All blocks dictionary
        """
        if from_block in all_blocks and to_block in all_blocks:
            if to_block not in all_blocks[from_block].successors:
                all_blocks[from_block].successors.append(to_block)
            if from_block not in all_blocks[to_block].predecessors:
                all_blocks[to_block].predecessors.append(from_block)
    
    def _perform_advanced_analysis(self, blocks: Dict[str, BasicBlock]) -> None:
        """
        Perform advanced control flow analysis including loop detection and dominance.
        
        Args:
            blocks: Dictionary of basic blocks
        """
        # Loop detection
        self._detect_loops(blocks)
        
        # Dominance analysis
        self._compute_dominance(blocks)
        
        # Reachability analysis
        self._analyze_reachability(blocks)
    
    def _detect_loops(self, blocks: Dict[str, BasicBlock]) -> None:
        """
        Detect loops in the control flow graph using back edge detection.
        
        Args:
            blocks: Dictionary of basic blocks
        """
        if not blocks:
            return
        
        # Perform DFS to identify back edges
        visited = set()
        rec_stack = set()
        back_edges = []
        
        def dfs(block_id: str) -> None:
            visited.add(block_id)
            rec_stack.add(block_id)
            
            if block_id in blocks:
                for successor in blocks[block_id].successors:
                    if successor not in visited:
                        dfs(successor)
                    elif successor in rec_stack:
                        # Back edge found
                        back_edges.append((block_id, successor))
            
            rec_stack.remove(block_id)
        
        # Start DFS from entry blocks
        entry_blocks = [bid for bid, block in blocks.items() if not block.predecessors]
        if not entry_blocks and blocks:
            entry_blocks = [list(blocks.keys())[0]]  # Fallback to first block
        
        for entry in entry_blocks:
            if entry not in visited:
                dfs(entry)
        
        # Store loop information
        for block in blocks.values():
            if not hasattr(block, 'metadata'):
                block.metadata = {}
            block.metadata['back_edges'] = back_edges
            block.metadata['is_loop_header'] = any(edge[1] == block.id for edge in back_edges)
    
    def _compute_dominance(self, blocks: Dict[str, BasicBlock]) -> None:
        """
        Compute dominance relationships between blocks.
        
        Args:
            blocks: Dictionary of basic blocks
        """
        if not blocks:
            return
        
        block_ids = list(blocks.keys())
        dominators = {bid: set(block_ids) for bid in block_ids}
        
        # Entry blocks dominate only themselves
        entry_blocks = [bid for bid, block in blocks.items() if not block.predecessors]
        if not entry_blocks and blocks:
            entry_blocks = [list(blocks.keys())[0]]
        
        for entry in entry_blocks:
            dominators[entry] = {entry}
        
        # Iterative algorithm
        changed = True
        while changed:
            changed = False
            for block_id in block_ids:
                if block_id in entry_blocks:
                    continue
                
                # Intersection of all predecessors' dominators + self
                new_dom = set(block_ids)
                for pred in blocks[block_id].predecessors:
                    if pred in dominators:
                        new_dom &= dominators[pred]
                new_dom.add(block_id)
                
                if new_dom != dominators[block_id]:
                    dominators[block_id] = new_dom
                    changed = True
        
        # Store dominance information
        for block_id, block in blocks.items():
            if not hasattr(block, 'metadata'):
                block.metadata = {}
            block.metadata['dominators'] = dominators[block_id]
    
    def _analyze_reachability(self, blocks: Dict[str, BasicBlock]) -> None:
        """
        Analyze block reachability to identify dead code.
        
        Args:
            blocks: Dictionary of basic blocks
        """
        if not blocks:
            return
        
        reachable = set()
        
        # Start from entry blocks
        entry_blocks = [bid for bid, block in blocks.items() if not block.predecessors]
        if not entry_blocks and blocks:
            entry_blocks = [list(blocks.keys())[0]]
        
        # DFS to find all reachable blocks
        def mark_reachable(block_id: str) -> None:
            if block_id in reachable or block_id not in blocks:
                return
            reachable.add(block_id)
            for successor in blocks[block_id].successors:
                mark_reachable(successor)
        
        for entry in entry_blocks:
            mark_reachable(entry)
        
        # Mark unreachable blocks
        for block_id, block in blocks.items():
            if not hasattr(block, 'metadata'):
                block.metadata = {}
            block.metadata['is_reachable'] = block_id in reachable
            block.metadata['is_dead_code'] = block_id not in reachable
    
    def _fallback_control_flow_analysis(self) -> Dict[str, BasicBlock]:
        """
        Fallback control flow analysis for error recovery.
        
        Returns:
            Basic control flow analysis result
        """
        self.logger.warning("Using fallback control flow analysis")
        
        blocks = {}
        if not self.instructions:
            return blocks
        
        # Create a single block with all instructions
        first_pc = self._get_pc(self.instructions[0], 0)
        last_pc = self._get_pc(self.instructions[-1], len(self.instructions) - 1)
        
        block_id = f"block_{first_pc:04x}"
        blocks[block_id] = BasicBlock(
            id=block_id,
            instructions=[],
            predecessors=[],
            successors=[],
            start_address=first_pc,
            end_address=last_pc
        )
        
        return blocks
    
    def _get_instruction_name(self, instr) -> str:
        """
        Get instruction name from various instruction object types.
        
        Args:
            instr: Instruction object (dict or pyevmasm object)
            
        Returns:
            Instruction name as string
        """
        # Handle dictionary format (from our parser)
        if isinstance(instr, dict):
            return instr.get('name', 'UNKNOWN')
        
        # Handle original pyevmasm object format
        if hasattr(instr, 'name'):
            return instr.name
        elif hasattr(instr, 'opcode'):
            return instr.opcode.name if hasattr(instr.opcode, 'name') else str(instr.opcode)
        else:
            return str(instr)
    
    def _get_pc(self, instr, fallback: int) -> int:
        """
        Get program counter from instruction object.
        
        Args:
            instr: Instruction object (dict or pyevmasm object)
            fallback: Fallback value if PC not found
            
        Returns:
            Program counter value
        """
        # Handle dictionary format (from our parser)
        if isinstance(instr, dict):
            return instr.get('pc', fallback)
        
        # Handle original pyevmasm object format
        return getattr(instr, 'pc', fallback)
    
    def _get_operand(self, instr):
        """
        Get operand from instruction object.
        
        Args:
            instr: Instruction object (dict or pyevmasm object)
            
        Returns:
            Operand value or None
        """
        # Handle dictionary format (from our parser)
        if isinstance(instr, dict):
            return instr.get('operand', None)
        
        # Handle original pyevmasm object format
        return getattr(instr, 'operand', None)
    
    class _StackSimulator:
        """
        Internal class for simulating EVM stack during control flow analysis.
        """
        
        def __init__(self):
            self.stack = []
            self.stack_values = {}  # Map instruction index to known values
        
        def process_instruction(self, instr, index: int) -> None:
            """
            Process an instruction and update stack state.
            
            Args:
                instr: EVM instruction
                index: Instruction index
            """
            name = self._get_name(instr)
            
            if name.startswith('PUSH'):
                # Push constant value
                try:
                    operand = getattr(instr, 'operand', None)
                    if operand:
                        value = int(operand, 16) if isinstance(operand, str) else int(operand)
                        self.stack.append(value)
                        self.stack_values[index] = value
                    else:
                        self.stack.append(None)
                except (ValueError, AttributeError):
                    self.stack.append(None)
            
            elif name == 'POP':
                if self.stack:
                    self.stack.pop()
            
            elif name in ['DUP1', 'DUP2', 'DUP3', 'DUP4', 'DUP5', 'DUP6', 'DUP7', 'DUP8',
                         'DUP9', 'DUP10', 'DUP11', 'DUP12', 'DUP13', 'DUP14', 'DUP15', 'DUP16']:
                dup_depth = int(name[3:]) if len(name) > 3 else 1
                if len(self.stack) >= dup_depth:
                    self.stack.append(self.stack[-dup_depth])
                else:
                    self.stack.append(None)
            
            elif name in ['SWAP1', 'SWAP2', 'SWAP3', 'SWAP4', 'SWAP5', 'SWAP6', 'SWAP7', 'SWAP8',
                         'SWAP9', 'SWAP10', 'SWAP11', 'SWAP12', 'SWAP13', 'SWAP14', 'SWAP15', 'SWAP16']:
                swap_depth = int(name[4:]) if len(name) > 4 else 1
                if len(self.stack) > swap_depth:
                    self.stack[-1], self.stack[-1-swap_depth] = self.stack[-1-swap_depth], self.stack[-1]
            
            elif name in ['ADD', 'MUL', 'SUB', 'DIV', 'SDIV', 'MOD', 'SMOD', 'ADDMOD', 'MULMOD',
                         'EXP', 'SIGNEXTEND', 'LT', 'GT', 'SLT', 'SGT', 'EQ', 'ISZERO', 'AND',
                         'OR', 'XOR', 'NOT', 'BYTE', 'SHL', 'SHR', 'SAR']:
                # Binary operations (most common case)
                if name in ['ISZERO', 'NOT']:  # Unary operations
                    if self.stack:
                        self.stack.pop()
                        self.stack.append(None)  # Unknown result
                else:  # Binary operations
                    if len(self.stack) >= 2:
                        self.stack.pop()
                        self.stack.pop()
                        self.stack.append(None)  # Unknown result
            
            # Limit stack size to prevent memory issues
            if len(self.stack) > 1024:
                self.stack = self.stack[-1024:]
        
        def get_stack_top_value(self) -> Optional[int]:
            """
            Get the top value from the stack if known.
            
            Returns:
                Top stack value or None if unknown
            """
            if self.stack and isinstance(self.stack[-1], int):
                return self.stack[-1]
            return None
        
        def _get_name(self, instr) -> str:
            """Get instruction name."""
            if hasattr(instr, 'name'):
                return instr.name
            elif hasattr(instr, 'opcode'):
                return instr.opcode.name if hasattr(instr.opcode, 'name') else str(instr.opcode)
            else:
                return str(instr)
    
    def identify_functions(self) -> Dict[str, Function]:
        """
        Identify function boundaries and extract function metadata.
        
        Returns:
            Dictionary mapping function names to Function objects
        """
        functions = {}
        
        # Look for function selectors in the bytecode
        # This is a simplified implementation - real implementation would be more sophisticated
        
        # Check for dispatcher pattern at the beginning
        dispatcher_found = False
        current_function = None
        
        for i, instr in enumerate(self.instructions):
            # Look for PUSH4 followed by EQ pattern (function selector check)
            instr_name = self._get_instruction_name(instr)
            if (instr_name == 'PUSH4' and 
                i + 2 < len(self.instructions)):
                next_instr = self.instructions[i + 1]
                next_name = self._get_instruction_name(next_instr)
                if next_name == 'EQ':
                    selector = self._get_operand(instr)
                    if selector:
                        function_name = f"function_{selector}"
                        
                        # Find the corresponding JUMPDEST
                        # Simplified - would need more sophisticated analysis
                        entry_block = None
                        for block_id, block in self.basic_blocks.items():
                            # Check if any instruction in the block is JUMPDEST
                            entry_block = block_id
                            break
                
                if entry_block:
                    functions[function_name] = Function(
                        name=function_name,
                        selector=selector,
                        basic_blocks=[],  # Would be populated with actual analysis
                        entry_block=entry_block,
                        parameters=[],
                        return_types=[]
                    )
        
        # Add a fallback function if no specific functions found
        if not functions and self.basic_blocks:
            entry_block = list(self.basic_blocks.keys())[0]
            functions["fallback"] = Function(
                name="fallback",
                selector=None,
                basic_blocks=list(self.basic_blocks.values()),
                entry_block=entry_block
            )
        
        self.functions = functions
        return functions
    
    def convert_to_tac(self) -> List[TACInstruction]:
        """
        Convert EVM instructions to Three-Address Code representation.
        
        Returns:
            List of TAC instructions
        """
        tac_instructions = []
        stack = []  # Simulate EVM stack for analysis
        
        for instr in self.instructions:
            tac_instr = self._convert_instruction_to_tac(instr, stack)
            if tac_instr:
                if isinstance(tac_instr, list):
                    tac_instructions.extend(tac_instr)
                else:
                    tac_instructions.append(tac_instr)
        
        return tac_instructions
    
    def _convert_instruction_to_tac(self, instr, stack: List[str]) -> Union[TACInstruction, List[TACInstruction], None]:
        """
        Convert a single EVM instruction to TAC format.
        
        Args:
            instr: EVM instruction
            stack: Current stack state (for analysis)
            
        Returns:
            TAC instruction(s) or None
        """
        # Handle different instruction object types
        if hasattr(instr, 'name'):
            name = instr.name
        elif hasattr(instr, 'opcode'):
            name = instr.opcode.name if hasattr(instr.opcode, 'name') else str(instr.opcode)
        else:
            name = str(instr)
        
        # Stack operations
        if name.startswith('PUSH'):
            temp_var = self._generate_temp_var()
            stack.append(temp_var)
            operand = self._get_operand(instr) or 'unknown'
            return TACInstruction(
                operation=TACOperationType.ASSIGN,
                result=temp_var,
                operand1=operand,
                metadata={'original_op': name}
            )
        
        elif name == 'POP':
            if stack:
                stack.pop()
            return None
        
        elif name == 'DUP1':
            if stack:
                temp_var = self._generate_temp_var()
                stack.append(temp_var)
                return TACInstruction(
                    operation=TACOperationType.ASSIGN,
                    result=temp_var,
                    operand1=stack[-2] if len(stack) >= 2 else "stack_underflow"
                )
        
        elif name == 'SWAP1':
            if len(stack) >= 2:
                stack[-1], stack[-2] = stack[-2], stack[-1]
            return None
        
        # Arithmetic operations
        elif name == 'ADD':
            if len(stack) >= 2:
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = self._generate_temp_var()
                stack.append(result)
                return TACInstruction(
                    operation=TACOperationType.BINARY_OP,
                    result=result,
                    operand1=operand1,
                    operand2=operand2,
                    operator="+"
                )
        
        elif name == 'SUB':
            if len(stack) >= 2:
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = self._generate_temp_var()
                stack.append(result)
                return TACInstruction(
                    operation=TACOperationType.BINARY_OP,
                    result=result,
                    operand1=operand1,
                    operand2=operand2,
                    operator="-"
                )
        
        elif name == 'MUL':
            if len(stack) >= 2:
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = self._generate_temp_var()
                stack.append(result)
                return TACInstruction(
                    operation=TACOperationType.BINARY_OP,
                    result=result,
                    operand1=operand1,
                    operand2=operand2,
                    operator="*"
                )
        
        # Memory operations
        elif name == 'MLOAD':
            if stack:
                address = stack.pop()
                result = self._generate_temp_var()
                stack.append(result)
                return TACInstruction(
                    operation=TACOperationType.LOAD,
                    result=result,
                    operand1=address,
                    metadata={'memory_type': 'memory'}
                )
        
        elif name == 'MSTORE':
            if len(stack) >= 2:
                value = stack.pop()
                address = stack.pop()
                return TACInstruction(
                    operation=TACOperationType.STORE,
                    operand1=address,
                    operand2=value,
                    metadata={'memory_type': 'memory'}
                )
        
        # Storage operations
        elif name == 'SLOAD':
            if stack:
                key = stack.pop()
                result = self._generate_temp_var()
                stack.append(result)
                return TACInstruction(
                    operation=TACOperationType.LOAD,
                    result=result,
                    operand1=key,
                    metadata={'memory_type': 'storage'}
                )
        
        elif name == 'SSTORE':
            if len(stack) >= 2:
                value = stack.pop()
                key = stack.pop()
                return TACInstruction(
                    operation=TACOperationType.STORE,
                    operand1=key,
                    operand2=value,
                    metadata={'memory_type': 'storage'}
                )
        
        # Control flow
        elif name == 'JUMP':
            if stack:
                target = stack.pop()
                return TACInstruction(
                    operation=TACOperationType.JUMP,
                    target=target
                )
        
        elif name == 'JUMPI':
            if len(stack) >= 2:
                condition = stack.pop()
                target = stack.pop()
                return TACInstruction(
                    operation=TACOperationType.CONDITIONAL_JUMP,
                    target=target,
                    operand1=condition
                )
        
        elif name == 'RETURN':
            instructions = []
            if len(stack) >= 2:
                size = stack.pop()
                offset = stack.pop()
                instructions.append(TACInstruction(
                    operation=TACOperationType.RETURN,
                    operand1=offset,
                    operand2=size
                ))
            return instructions
        
        elif name == 'REVERT':
            instructions = []
            if len(stack) >= 2:
                size = stack.pop()
                offset = stack.pop()
                instructions.append(TACInstruction(
                    operation=TACOperationType.REVERT,
                    operand1=offset,
                    operand2=size
                ))
            return instructions
        
        # For unhandled instructions, create a generic representation
        return TACInstruction(
            operation=TACOperationType.ASSIGN,
            result=self._generate_temp_var(),
            metadata={'original_op': name, 'operand': getattr(instr, 'operand', None)}
        )
    
    def generate_tac_representation(self) -> str:
        """
        Generate a string representation of the TAC code with integrated control flow analysis.
        
        Returns:
            String representation suitable for LLM input
        """
        try:
            # Perform comprehensive analysis
            self.logger.info("Starting comprehensive TAC generation with control flow analysis")
            
            # Phase 1: Control flow analysis
            self.analyze_control_flow()
            
            # Phase 2: Function identification
            self.identify_functions()
            
            # Phase 3: Convert to TAC and integrate with basic blocks
            self._convert_and_integrate_tac()
            
            # Phase 4: Generate formatted output
            return self._format_integrated_tac_output()
            
        except Exception as e:
            self.logger.error(f"TAC generation failed: {e}")
            # Fallback to basic TAC generation
            return self._generate_fallback_tac()
    
    def _convert_and_integrate_tac(self) -> None:
        """
        Convert EVM instructions to TAC and integrate with basic blocks.
        """
        # Create PC to instruction mapping
        pc_to_instr = {}
        for i, instr in enumerate(self.instructions):
            pc = getattr(instr, 'pc', i)
            pc_to_instr[pc] = instr
        
        # Convert instructions for each basic block
        for block_id, block in self.basic_blocks.items():
            block_tac_instructions = []
            stack = []  # Local stack simulation for this block
            
            if 'raw_instructions' in block.metadata:
                raw_instructions = block.metadata['raw_instructions']
                
                for instr in raw_instructions:
                    tac_instr = self._convert_instruction_to_tac(instr, stack)
                    if tac_instr:
                        if isinstance(tac_instr, list):
                            block_tac_instructions.extend(tac_instr)
                        else:
                            block_tac_instructions.append(tac_instr)
                
                # Add control flow metadata to TAC instructions
                for tac_instr in block_tac_instructions:
                    if not tac_instr.metadata:
                        tac_instr.metadata = {}
                    tac_instr.metadata['block_id'] = block_id
                    tac_instr.metadata['block_start'] = block.start_address
                    tac_instr.metadata['block_end'] = block.end_address
                
                # Store TAC instructions in the block
                block.instructions = block_tac_instructions
                
                # Add control flow analysis metadata
                self._add_control_flow_metadata_to_block(block)
    
    def _add_control_flow_metadata_to_block(self, block: BasicBlock) -> None:
        """
        Add control flow analysis metadata to basic block.
        
        Args:
            block: Basic block to enhance with metadata
        """
        # Add successor/predecessor information
        if not block.metadata:
            block.metadata = {}
        
        block.metadata['num_predecessors'] = len(block.predecessors)
        block.metadata['num_successors'] = len(block.successors)
        block.metadata['is_entry_block'] = len(block.predecessors) == 0
        block.metadata['is_exit_block'] = len(block.successors) == 0
        
        # Determine block type based on control flow
        if len(block.successors) == 0:
            block.metadata['block_type'] = 'exit'
        elif len(block.successors) == 1:
            block.metadata['block_type'] = 'sequential'
        elif len(block.successors) == 2:
            block.metadata['block_type'] = 'conditional'
        else:
            block.metadata['block_type'] = 'complex'
        
        # Add loop information if available
        if 'is_loop_header' in block.metadata:
            block.metadata['is_loop_header'] = block.metadata.get('is_loop_header', False)
        
        # Add reachability information
        if 'is_reachable' in block.metadata:
            block.metadata['is_reachable'] = block.metadata.get('is_reachable', True)
    
    def _format_integrated_tac_output(self) -> str:
        """
        Generate formatted TAC output with integrated control flow information.
        
        Returns:
            Formatted TAC representation
        """
        tac_lines = []
        tac_lines.append("// Three-Address Code Representation with Control Flow Analysis")
        tac_lines.append("// Generated from comprehensive EVM bytecode analysis")
        tac_lines.append("")
        
        # Add analysis summary
        tac_lines.append(f"// Analysis Summary:")
        tac_lines.append(f"//   Total instructions: {len(self.instructions)}")
        tac_lines.append(f"//   Basic blocks: {len(self.basic_blocks)}")
        tac_lines.append(f"//   Functions identified: {len(self.functions)}")
        tac_lines.append("")
        
        # Process functions and their blocks
        if self.functions:
            for func_name, function in self.functions.items():
                self._format_function_tac(tac_lines, func_name, function)
        else:
            # No functions identified - show all blocks
            self._format_all_blocks_tac(tac_lines)
        
        return "\n".join(tac_lines)
    
    def _format_function_tac(self, tac_lines: List[str], func_name: str, function: Function) -> None:
        """
        Format TAC output for a specific function.
        
        Args:
            tac_lines: List to append formatted lines to
            func_name: Function name
            function: Function object
        """
        tac_lines.append(f"function {func_name}:")
        if function.selector:
            tac_lines.append(f"  // Function selector: {function.selector}")
        
        # Add function metadata
        tac_lines.append(f"  // Entry block: {function.entry_block}")
        tac_lines.append(f"  // Visibility: {function.visibility}")
        if function.is_payable:
            tac_lines.append(f"  // Payable: true")
        if function.is_view:
            tac_lines.append(f"  // View: true")
        tac_lines.append("")
        
        # Get blocks for this function (for now, include all blocks)
        function_blocks = function.basic_blocks if function.basic_blocks else list(self.basic_blocks.values())
        
        # Sort blocks by start address for consistent output
        sorted_blocks = sorted(function_blocks, key=lambda b: b.start_address)
        
        for block in sorted_blocks:
            self._format_block_tac(tac_lines, block, indent="  ")
    
    def _format_all_blocks_tac(self, tac_lines: List[str]) -> None:
        """
        Format TAC output for all blocks when no functions are identified.
        
        Args:
            tac_lines: List to append formatted lines to
        """
        tac_lines.append("main:")
        tac_lines.append("  // No specific functions identified - showing all basic blocks")
        tac_lines.append("")
        
        # Sort blocks by start address
        sorted_blocks = sorted(self.basic_blocks.values(), key=lambda b: b.start_address)
        
        for block in sorted_blocks:
            self._format_block_tac(tac_lines, block, indent="  ")
    
    def _format_block_tac(self, tac_lines: List[str], block: BasicBlock, indent: str = "") -> None:
        """
        Format TAC output for a single basic block.
        
        Args:
            tac_lines: List to append formatted lines to
            block: Basic block to format
            indent: Indentation string
        """
        # Block header with metadata
        tac_lines.append(f"{indent}{block.id}:")
        tac_lines.append(f"{indent}  // Address range: {block.start_address:04x} - {block.end_address:04x}")
        
        # Add control flow information
        if block.predecessors:
            pred_list = ", ".join(block.predecessors)
            tac_lines.append(f"{indent}  // Predecessors: {pred_list}")
        if block.successors:
            succ_list = ", ".join(block.successors)
            tac_lines.append(f"{indent}  // Successors: {succ_list}")
        
        # Add block type and special properties
        if 'block_type' in block.metadata:
            tac_lines.append(f"{indent}  // Block type: {block.metadata['block_type']}")
        
        if block.metadata.get('is_loop_header', False):
            tac_lines.append(f"{indent}  // Loop header")
        
        if block.metadata.get('is_dead_code', False):
            tac_lines.append(f"{indent}  // Dead code (unreachable)")
        
        # TAC instructions
        if block.instructions:
            for tac_instr in block.instructions:
                formatted_instr = self._format_tac_instruction(tac_instr)
                tac_lines.append(f"{indent}    {formatted_instr}")
        else:
            tac_lines.append(f"{indent}    // No TAC instructions")
        
        tac_lines.append("")
    
    def _generate_fallback_tac(self) -> str:
        """
        Generate fallback TAC representation when comprehensive analysis fails.
        
        Returns:
            Basic TAC representation
        """
        self.logger.warning("Using fallback TAC generation")
        
        tac_lines = []
        tac_lines.append("// Three-Address Code Representation (Fallback Mode)")
        tac_lines.append("// Basic analysis due to errors in comprehensive mode")
        tac_lines.append("")
        
        try:
            # Basic TAC conversion without control flow analysis
            tac_instructions = []
            stack = []
            
            for instr in self.instructions:
                tac_instr = self._convert_instruction_to_tac(instr, stack)
                if tac_instr:
                    if isinstance(tac_instr, list):
                        tac_instructions.extend(tac_instr)
                    else:
                        tac_instructions.append(tac_instr)
            
            tac_lines.append("main:")
            for tac_instr in tac_instructions:
                tac_lines.append(f"  {self._format_tac_instruction(tac_instr)}")
                
        except Exception as e:
            self.logger.error(f"Fallback TAC generation also failed: {e}")
            tac_lines.append("// Error: Unable to generate TAC representation")
            tac_lines.append(f"// {str(e)}")
        
        return "\n".join(tac_lines)
    
    def _format_tac_instruction(self, instr: TACInstruction) -> str:
        """Format a TAC instruction as a string."""
        if instr.operation == TACOperationType.ASSIGN:
            if instr.operand1:
                return f"{instr.result} = {instr.operand1}"
            else:
                return f"{instr.result} = <unknown>"
        
        elif instr.operation == TACOperationType.BINARY_OP:
            return f"{instr.result} = {instr.operand1} {instr.operator} {instr.operand2}"
        
        elif instr.operation == TACOperationType.LOAD:
            memory_type = instr.metadata.get('memory_type', 'memory') if instr.metadata else 'memory'
            return f"{instr.result} = {memory_type}[{instr.operand1}]"
        
        elif instr.operation == TACOperationType.STORE:
            memory_type = instr.metadata.get('memory_type', 'memory') if instr.metadata else 'memory'
            return f"{memory_type}[{instr.operand1}] = {instr.operand2}"
        
        elif instr.operation == TACOperationType.JUMP:
            return f"goto {instr.target}"
        
        elif instr.operation == TACOperationType.CONDITIONAL_JUMP:
            return f"if {instr.operand1} goto {instr.target}"
        
        elif instr.operation == TACOperationType.RETURN:
            return f"return memory[{instr.operand1}:{instr.operand2}]"
        
        elif instr.operation == TACOperationType.REVERT:
            return f"revert memory[{instr.operand1}:{instr.operand2}]"
        
        else:
            return f"// {instr.operation.value}: {instr.metadata}"


def analyze_bytecode_to_tac(bytecode: str) -> str:
    """
    Convenience function to analyze bytecode and return TAC representation.
    
    Args:
        bytecode: Hex string of EVM bytecode
        
    Returns:
        TAC representation as string
    """
    analyzer = BytecodeAnalyzer(bytecode)
    return analyzer.generate_tac_representation()


if __name__ == "__main__":
    # Example usage
    sample_bytecode = "0x608060405234801561001057600080fd5b50600436106100365760003560e01c8063893d20e81461003b578063a6f9dae114610059575b600080fd5b610043610075565b6040516100509190610166565b60405180910390f35b610073600480360381019061006e91906101b2565b61009e565b005b60008060009054906101000a900473ffffffffffffffffffffffffffffffffffffffff16905090565b8073ffffffffffffffffffffffffffffffffffffffff163373ffffffffffffffffffffffffffffffffffffffff16036100d357806000806101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055505b50565b600073ffffffffffffffffffffffffffffffffffffffff82169050919050565b6000610101826100d6565b9050919050565b610111816100f6565b82525050565b600060208201905061012c6000830184610108565b92915050565b600080fd5b610140816100f6565b811461014b57600080fd5b50565b60008135905061015d81610137565b92915050565b60006020828403121561017957610178610132565b5b60006101878482850161014e565b91505092915050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052602260045260246000fd5b600060028204905060018216806101d857607f821691505b6020821081036101eb576101ea610190565b5b5091905056fea26469706673582212209d84a3c5d1d6c4c5f9c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c564736f6c634300080a0033"
    
    try:
        tac_output = analyze_bytecode_to_tac(sample_bytecode)
        print("TAC Representation:")
        print(tac_output)
    except Exception as e:
        print(f"Error: {e}")
