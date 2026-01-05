"""
Dataset Collection and Preprocessing Pipeline

This module implements the data collection, processing, and filtering pipeline
to create high-quality training examples for the smart contract decompilation model,
as described in the paper (238,446 TAC-to-Solidity function pairs).
"""

import json
import re
import os
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from web3 import Web3
from eth_utils import to_hex, to_checksum_address
import requests
from tqdm import tqdm

from .bytecode_analyzer import BytecodeAnalyzer, analyze_bytecode_to_tac
import yaml
import traceback

@dataclass
class ContractData:
    """Represents collected smart contract data."""
    address: str
    source_code: str
    bytecode: str
    compiler_version: str
    optimization_enabled: bool
    optimization_runs: int
    creation_block: Optional[int] = None
    creation_timestamp: Optional[int] = None
    abi: Optional[str] = None

@dataclass
class FunctionPair:
    """Represents a TAC-to-Solidity function pair for training."""
    function_name: str
    tac_representation: str
    solidity_code: str
    function_signature: str
    visibility: str
    is_payable: bool
    is_view: bool
    contract_address: str
    metadata: Dict = None

class EtherscanAPI:
    """Interface for collecting verified contracts from Etherscan."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.etherscan.io/v2/api"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
    
    def get_contract_source(self, address: str) -> Optional[ContractData]:
        """
        Get verified contract source code from Etherscan.
        
        Args:
            address: Contract address
            
        Returns:
            ContractData object or None if not available
        """
        try:
            # Get source code
            params = {
                'chainid': '1',
                'module': 'contract',
                'action': 'getsourcecode',
                'address': address,
                'apikey': self.api_key
            }
            
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') != '1' or not data.get('result'):
                return None
            
            result = data['result'][0]
            
            # Check if contract is verified
            if not result.get('SourceCode'):
                return None
            
            # Get runtime bytecode
            bytecode_params = {
                'chainid': '1',
                'module': 'proxy',
                'action': 'eth_getCode',
                'address': address,
                'tag': 'latest',
                'apikey': self.api_key
            }
            
            bytecode_response = self.session.get(self.base_url, params=bytecode_params, timeout=30)
            bytecode_data = bytecode_response.json()
            
            if bytecode_data.get('result') == '0x':
                return None
            
            return ContractData(
                address=to_checksum_address(address),
                source_code=result.get('SourceCode', ''),
                bytecode=bytecode_data.get('result', ''),
                compiler_version=result.get('CompilerVersion', ''),
                optimization_enabled=result.get('OptimizationUsed') == '1',
                optimization_runs=int(result.get('Runs', '0')) if result.get('Runs') else 0,
                abi=result.get('ABI')
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get contract {address}: {e}")
            return None
    
    def get_verified_contracts_batch(self, start_block: int, end_block: int, limit: int = 1000) -> List[str]:
        """
        Get a batch of verified contract addresses from a block range.
        
        Args:
            start_block: Starting block number
            end_block: Ending block number
            limit: Maximum number of contracts to return
            
        Returns:
            List of contract addresses
        """
        try:
            # This is a simplified implementation
            # In practice, you would need to scan blocks or use other methods
            # to find verified contracts efficiently
            
            # For demonstration, we'll use a different approach
            # Get list of verified contracts (this API endpoint varies by implementation)
            contracts = []
            
            # Note: Real implementation would require more sophisticated discovery
            # This is a placeholder for the concept
            
            return contracts[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get contracts batch: {e}")
            return []

class SolidityParser:
    """Parser for extracting functions from Solidity source code."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_functions(self, source_code: str, contract_name: Optional[str] = None) -> List[Dict]:
        """
        Extract individual functions from Solidity source code.
        
        Args:
            source_code: Complete Solidity source code
            contract_name: Optional specific contract name to extract from
            
        Returns:
            List of function dictionaries
        """
        functions = []
        
        try:
            # Clean and normalize source code
            cleaned_source = self._clean_source_code(source_code)
            
            # Extract contract blocks
            contracts = self._extract_contracts(cleaned_source)
            
            for contract in contracts:
                if contract_name and contract['name'] != contract_name:
                    continue
                
                # Extract functions from this contract
                contract_functions = self._extract_functions_from_contract(contract['body'])
                
                for func in contract_functions:
                    func['contract_name'] = contract['name']
                    functions.append(func)
        
        except Exception as e:
            self.logger.error(f"Failed to parse Solidity code: {e}")
        
        return functions
    
    def _clean_source_code(self, source_code: str) -> str:
        """Clean and normalize Solidity source code."""
        # Handle JSON-encoded source (Etherscan format)
        if source_code.startswith('{'):
            try:
                # Try to parse as JSON
                source_json = json.loads(source_code)
                
                if 'sources' in source_json:
                    # Multiple file format
                    combined_source = ""
                    for file_path, file_data in source_json['sources'].items():
                        if 'content' in file_data:
                            combined_source += f"// File: {file_path}\n"
                            combined_source += file_data['content'] + "\n\n"
                    source_code = combined_source
                elif 'content' in source_json:
                    source_code = source_json['content']
                    
            except json.JSONDecodeError:
                # Handle double-brace format {{...}}
                if source_code.startswith('{{'):
                    try:
                        # Remove outer braces and parse
                        inner_json = source_code[1:-1]
                        source_json = json.loads(inner_json)
                        
                        if 'sources' in source_json:
                            combined_source = ""
                            for file_path, file_data in source_json['sources'].items():
                                if 'content' in file_data:
                                    combined_source += f"// File: {file_path}\n"
                                    combined_source += file_data['content'] + "\n\n"
                            source_code = combined_source
                    except (json.JSONDecodeError, IndexError):
                        self.logger.warning("Failed to parse JSON-encoded source code")
        
        # Don't remove comments at this stage - they might contain important info
        # We'll handle them during function extraction if needed
        
        return source_code
    
    def _extract_contracts(self, source_code: str) -> List[Dict]:
        """Extract contract definitions from source code using brace counting."""
        contracts = []
        
        # Find contract keywords with proper brace matching
        contract_starts = []
        for match in re.finditer(r'\b(contract|interface|library)\s+(\w+)', source_code):
            contract_starts.append({
                'type': match.group(1),
                'name': match.group(2),
                'start': match.end()
            })
        
        for contract_info in contract_starts:
            # Find the opening brace
            start_pos = contract_info['start']
            brace_pos = source_code.find('{', start_pos)
            
            if brace_pos == -1:
                continue
            
            # Use brace counting to find matching closing brace
            body_start = brace_pos + 1
            body_end = self._find_matching_brace(source_code, brace_pos)
            
            if body_end != -1:
                contracts.append({
                    'name': contract_info['name'],
                    'type': contract_info['type'],
                    'body': source_code[body_start:body_end]
                })
                self.logger.debug(f"Extracted {contract_info['type']} {contract_info['name']}")
        
        return contracts
    
    def _extract_functions_from_contract(self, contract_body: str) -> List[Dict]:
        """Extract function definitions from contract body using brace counting."""
        functions = []
        
        # Find all function declarations
        function_starts = []
        for match in re.finditer(r'\bfunction\s+(\w+)\s*\(', contract_body):
            function_starts.append({
                'name': match.group(1),
                'start': match.start(),
                'params_start': match.end()
            })
        
        for func_info in function_starts:
            try:
                # Find the end of parameters
                params_end = contract_body.find(')', func_info['params_start'])
                if params_end == -1:
                    continue
                
                # Extract the part between function declaration and opening brace
                # This contains modifiers, visibility, etc.
                modifier_start = params_end + 1
                brace_pos = contract_body.find('{', modifier_start)
                
                if brace_pos == -1:
                    # Function might be abstract/interface (no body)
                    semicolon = contract_body.find(';', modifier_start)
                    if semicolon != -1 and (brace_pos == -1 or semicolon < brace_pos):
                        # Abstract function
                        full_function = contract_body[func_info['start']:semicolon + 1]
                        self._add_function_to_list(functions, func_info['name'], full_function, "")
                    continue
                
                # Find matching closing brace
                body_end = self._find_matching_brace(contract_body, brace_pos)
                
                if body_end != -1:
                    # Extract full function including body
                    full_function = contract_body[func_info['start']:body_end + 1]
                    function_body = contract_body[brace_pos + 1:body_end]
                    
                    self._add_function_to_list(functions, func_info['name'], full_function, function_body)
                    
            except Exception as e:
                self.logger.warning(f"Failed to extract function {func_info['name']}: {e}")
                continue
        
        return functions
    
    def _find_matching_brace(self, text: str, start_pos: int) -> int:
        """
        Find the matching closing brace for an opening brace at start_pos.
        
        Args:
            text: Source text
            start_pos: Position of opening brace
            
        Returns:
            Position of matching closing brace, or -1 if not found
        """
        if start_pos >= len(text) or text[start_pos] != '{':
            return -1
        
        brace_count = 1
        pos = start_pos + 1
        in_string = False
        in_char = False
        in_comment = False
        in_line_comment = False
        
        while pos < len(text) and brace_count > 0:
            char = text[pos]
            
            # Handle line comments
            if not in_string and not in_char and pos < len(text) - 1:
                if text[pos:pos+2] == '//':
                    in_line_comment = True
                    pos += 2
                    continue
                elif text[pos:pos+2] == '/*':
                    in_comment = True
                    pos += 2
                    continue
                elif text[pos:pos+2] == '*/' and in_comment:
                    in_comment = False
                    pos += 2
                    continue
            
            if in_line_comment:
                if char == '\n':
                    in_line_comment = False
                pos += 1
                continue
            
            if in_comment:
                pos += 1
                continue
            
            # Handle strings
            if char == '"' and not in_char:
                if pos > 0 and text[pos-1] != '\\':
                    in_string = not in_string
            elif char == "'" and not in_string:
                if pos > 0 and text[pos-1] != '\\':
                    in_char = not in_char
            
            # Count braces only outside strings and comments
            if not in_string and not in_char:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
            
            pos += 1
        
        return pos - 1 if brace_count == 0 else -1
    
    def _add_function_to_list(self, functions: List[Dict], name: str, 
                             full_function: str, body: str):
        """Helper to add a parsed function to the list."""
        # Extract function metadata
        visibility = self._extract_visibility(full_function)
        is_payable = 'payable' in full_function
        is_view = 'view' in full_function or 'pure' in full_function
        
        # Extract function signature
        signature_match = re.search(r'function\s+\w+\s*\([^)]*\)', full_function)
        signature = signature_match.group(0) if signature_match else f"function {name}()"
        
        functions.append({
            'name': name,
            'body': full_function,
            'signature': signature,
            'visibility': visibility,
            'is_payable': is_payable,
            'is_view': is_view
        })
        
        self.logger.debug(f"Extracted function {name} ({visibility})")
    
    def _extract_visibility(self, function_code: str) -> str:
        """Extract function visibility."""
        # Check in order of specificity
        if re.search(r'\bprivate\b', function_code):
            return 'private'
        elif re.search(r'\binternal\b', function_code):
            return 'internal'
        elif re.search(r'\bexternal\b', function_code):
            return 'external'
        elif re.search(r'\bpublic\b', function_code):
            return 'public'
        else:
            # Default visibility in Solidity
            return 'public'

class DatasetBuilder:
    """Main class for building the training dataset."""
    
    def __init__(self, etherscan_api_key: str, output_dir: str = "data"):
        self.etherscan = EtherscanAPI(etherscan_api_key)
        self.parser = SolidityParser()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Initialize database for caching
        self.db_path = self.output_dir / "contracts.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for caching contract data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS contracts (
                address TEXT PRIMARY KEY,
                source_code TEXT,
                bytecode TEXT,
                compiler_version TEXT,
                optimization_enabled BOOLEAN,
                optimization_runs INTEGER,
                processed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS function_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_address TEXT,
                function_name TEXT,
                tac_representation TEXT,
                solidity_code TEXT,
                function_signature TEXT,
                visibility TEXT,
                is_payable BOOLEAN,
                is_view BOOLEAN,
                metadata TEXT,
                hash TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (contract_address) REFERENCES contracts (address)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def collect_contracts(self, contract_addresses: List[str], max_workers: int = 10) -> int:
        """
        Collect contract data from Etherscan.
        
        Args:
            contract_addresses: List of contract addresses to collect
            max_workers: Number of parallel workers
            
        Returns:
            Number of successfully collected contracts
        """
        collected = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_address = {
                executor.submit(self.etherscan.get_contract_source, addr): addr
                for addr in contract_addresses
            }
            
            # Process results
            for future in tqdm(as_completed(future_to_address), total=len(contract_addresses), desc="Collecting contracts"):
                address = future_to_address[future]
                try:
                    contract_data = future.result()
                    if contract_data:
                        self._store_contract(contract_data)
                        collected += 1
                except Exception as e:
                    self.logger.error(f"Failed to process contract {address}: {e}")
        
        self.logger.info(f"Collected {collected} contracts out of {len(contract_addresses)} addresses")
        return collected
    
    def _store_contract(self, contract_data: ContractData):
        """Store contract data in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO contracts 
            (address, source_code, bytecode, compiler_version, optimization_enabled, optimization_runs)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            contract_data.address,
            contract_data.source_code,
            contract_data.bytecode,
            contract_data.compiler_version,
            contract_data.optimization_enabled,
            contract_data.optimization_runs
        ))
        
        conn.commit()
        conn.close()
    
    def process_contracts_to_function_pairs(self, batch_size: int = 1) -> int:
        """
        Process stored contracts to create TAC-to-Solidity function pairs.
        
        Args:
            batch_size: Number of contracts to process in each batch
            
        Returns:
            Number of function pairs created
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get unprocessed contracts
        # cursor.execute('SELECT address, source_code, bytecode FROM contracts WHERE processed = FALSE')
        cursor.execute('SELECT address, source_code, bytecode FROM contracts')
        unprocessed_contracts = cursor.fetchall()
        total_pairs = 0
        
        for i in tqdm(range(0, len(unprocessed_contracts), batch_size), desc="Processing contracts"):
            batch = unprocessed_contracts[i:i+batch_size]
            
            for address, source_code, bytecode in batch:
                try:
                    pairs = self._create_function_pairs(address, source_code, bytecode)
                    
                    # Store function pairs
                    for pair in pairs:
                        self._store_function_pair(pair)
                        total_pairs += 1
                    
                    # Mark contract as processed
                    cursor.execute('UPDATE contracts SET processed = TRUE WHERE address = ?', (address,))
                    
                except Exception as e:
                    self.logger.error(f"Failed to process contract {address}: {e}")
            
            conn.commit()
        
        conn.close()
        self.logger.info(f"Created {total_pairs} function pairs")
        return total_pairs
    
    def _create_function_pairs(self, address: str, source_code: str, bytecode: str) -> List[FunctionPair]:
        """Create TAC-to-Solidity function pairs from contract data."""
        pairs = []
        
        try:
            # Step 1: Analyze bytecode to get structured data
            self.logger.info(f"Analyzing bytecode for {address}")
            analyzer = BytecodeAnalyzer(bytecode)
            analyzer.analyze_control_flow()
            bytecode_functions = analyzer.identify_functions()
            
            # Step 2: Parse Solidity to get functions
            self.logger.info(f"Parsing Solidity source for {address}")
            solidity_functions = self.parser.extract_functions(source_code)
            
            if not solidity_functions:
                self.logger.warning(f"No Solidity functions found for {address}")
                return pairs
            
            # Step 3: Calculate function selectors for Solidity functions
            solidity_with_selectors = self._add_selectors_to_solidity_functions(solidity_functions)
            
            # Step 4: Match functions by selector
            matched_pairs = self._match_functions_by_selector(
                solidity_with_selectors, 
                bytecode_functions, 
                analyzer
            )
            
            self.logger.info(f"Matched {len(matched_pairs)} functions for {address}")
            
            # Step 5: Create training pairs from matches
            for match in matched_pairs:
                try:
                    pair = self._build_training_pair(match, address)
                    if pair:
                        pairs.append(pair)
                except Exception as e:
                    self.logger.error(f"Failed to build training pair: {e}")
                    continue
            
            # Fallback: If no pairs created and we have functions, create whole-contract pair
            if not pairs and solidity_functions:
                self.logger.info(f"No matched functions, creating whole-contract pair for {address}")
                fallback_pair = self._create_fallback_pair(address, source_code, analyzer)
                if fallback_pair:
                    pairs.append(fallback_pair)
        
        except Exception as e:
            self.logger.error(f"Failed to create function pairs for {address}: {e}")
            self.logger.error(f"Error traceback: {traceback.format_exc()}")
        
        return pairs
    
    def _add_selectors_to_solidity_functions(self, functions: List[Dict]) -> List[Dict]:
        """
        Calculate function selectors for Solidity functions.
        
        Args:
            functions: List of function dictionaries
            
        Returns:
            Functions with added 'selector' field
        """
        for func in functions:
            try:
                # Extract parameter types from signature
                signature = func['signature']
                # Remove 'function' keyword and normalize
                sig_normalized = signature.replace('function ', '').strip()
                
                # Calculate selector (first 4 bytes of keccak256 hash)
                selector_hash = Web3.keccak(text=sig_normalized)[:4]
                func['selector'] = '0x' + selector_hash.hex()
                
                self.logger.debug(f"Calculated selector {func['selector']} for {sig_normalized}")
            except Exception as e:
                self.logger.warning(f"Failed to calculate selector for {func['name']}: {e}")
                func['selector'] = None
        
        return functions
    
    def _match_functions_by_selector(self, solidity_functions: List[Dict], 
                                     bytecode_functions: Dict, 
                                     analyzer: BytecodeAnalyzer) -> List[Dict]:
        """
        Match Solidity functions with bytecode functions by selector.
        
        Args:
            solidity_functions: Solidity functions with selectors
            bytecode_functions: Bytecode functions from analyzer
            analyzer: BytecodeAnalyzer instance
            
        Returns:
            List of matched function pairs
        """
        matches = []
        
        # Create selector mappings
        solidity_by_selector = {f['selector']: f for f in solidity_functions if f.get('selector')}
        bytecode_by_selector = {f.selector: f for f in bytecode_functions.values() if f.selector}
        
        # Match functions
        for selector, sol_func in solidity_by_selector.items():
            if selector in bytecode_by_selector:
                bytecode_func = bytecode_by_selector[selector]
                
                # Extract TAC for this function
                tac = self._extract_tac_for_function(bytecode_func, analyzer)
                
                matches.append({
                    'solidity_function': sol_func,
                    'bytecode_function': bytecode_func,
                    'tac': tac,
                    'selector': selector
                })
                
                self.logger.debug(f"Matched function {sol_func['name']} with selector {selector}")
            else:
                self.logger.debug(f"No bytecode match for {sol_func['name']} (selector: {selector})")
        
        return matches
    
    def _extract_tac_for_function(self, bytecode_function, analyzer: BytecodeAnalyzer) -> str:
        """
        Extract TAC representation for a specific function using structured data.
        
        Args:
            bytecode_function: Function object from BytecodeAnalyzer
            analyzer: BytecodeAnalyzer instance with analyzed blocks
            
        Returns:
            Formatted TAC string for the function
        """
        tac_lines = []
        
        try:
            # Add function header
            func_name = bytecode_function.name
            tac_lines.append(f"function {func_name}:")
            
            if bytecode_function.selector:
                tac_lines.append(f"  // Selector: {bytecode_function.selector}")
            
            tac_lines.append(f"  // Entry block: {bytecode_function.entry_block}")
            
            # Get blocks for this function
            function_blocks = bytecode_function.basic_blocks if bytecode_function.basic_blocks else []
            
            # If no specific blocks assigned, try to extract from entry block
            if not function_blocks and bytecode_function.entry_block in analyzer.basic_blocks:
                # Use basic blocks starting from entry point
                function_blocks = self._collect_function_blocks(
                    bytecode_function.entry_block, 
                    analyzer.basic_blocks
                )
            
            # Format TAC instructions for each block
            for block in function_blocks:
                tac_lines.append(f"  {block.id}:")
                
                # Add block metadata
                if block.predecessors:
                    tac_lines.append(f"    // Predecessors: {', '.join(block.predecessors)}")
                if block.successors:
                    tac_lines.append(f"    // Successors: {', '.join(block.successors)}")
                
                # Add TAC instructions
                for instr in block.instructions:
                    formatted = analyzer._format_tac_instruction(instr)
                    tac_lines.append(f"    {formatted}")
                
                tac_lines.append("")
            
        except Exception as e:
            self.logger.error(f"Failed to extract TAC for function: {e}")
            tac_lines.append(f"  // Error extracting TAC: {e}")
        
        return '\n'.join(tac_lines)
    
    def _collect_function_blocks(self, entry_block_id: str, 
                                 all_blocks: Dict) -> List:
        """
        Collect all basic blocks belonging to a function via graph traversal.
        
        Args:
            entry_block_id: Entry block ID for the function
            all_blocks: Dictionary of all basic blocks
            
        Returns:
            List of basic blocks in the function
        """
        if entry_block_id not in all_blocks:
            return []
        
        visited = set()
        blocks = []
        
        def traverse(block_id: str):
            if block_id in visited or block_id not in all_blocks:
                return
            
            visited.add(block_id)
            block = all_blocks[block_id]
            blocks.append(block)
            
            # Traverse successors
            for successor in block.successors:
                traverse(successor)
        
        traverse(entry_block_id)
        return blocks
    
    def _build_training_pair(self, match: Dict, address: str) -> Optional[FunctionPair]:
        """
        Build a training pair from a matched function.
        
        Args:
            match: Matched function dictionary
            address: Contract address
            
        Returns:
            FunctionPair object or None
        """
        sol_func = match['solidity_function']
        tac = match['tac']
        
        # Filter very short functions
        if len(sol_func['body'].strip()) < 10:
            return None
        
        # Filter empty TAC
        if not tac or len(tac.strip()) < 10:
            return None
        
        return FunctionPair(
            function_name=sol_func['name'],
            tac_representation=tac,
            solidity_code=sol_func['body'],
            function_signature=sol_func['signature'],
            visibility=sol_func['visibility'],
            is_payable=sol_func['is_payable'],
            is_view=sol_func['is_view'],
            contract_address=address,
            metadata={
                'contract_name': sol_func.get('contract_name'),
                'selector': match['selector'],
                'matched_by_selector': True
            }
        )
    
    def _create_fallback_pair(self, address: str, source_code: str, 
                            analyzer: BytecodeAnalyzer) -> Optional[FunctionPair]:
        """
        Create a whole-contract training pair as fallback.
        
        Args:
            address: Contract address
            source_code: Full Solidity source
            analyzer: BytecodeAnalyzer instance
            
        Returns:
            FunctionPair for entire contract or None
        """
        try:
            # Generate full TAC representation
            tac = analyzer.generate_tac_representation()
            
            # Clean and prepare source code
            cleaned_source = self.parser._clean_source_code(source_code)
            
            return FunctionPair(
                function_name="contract",
                tac_representation=tac,
                solidity_code=cleaned_source,
                function_signature="contract",
                visibility="public",
                is_payable=False,
                is_view=False,
                contract_address=address,
                metadata={
                    'whole_contract': True,
                    'fallback_pair': True
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to create fallback pair: {e}")
            return None
    
    def _store_function_pair(self, pair: FunctionPair):
        """Store function pair in database."""
        # Create hash for deduplication
        content = f"{pair.tac_representation}{pair.solidity_code}"
        hash_value = hashlib.md5(content.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO function_pairs 
                (contract_address, function_name, tac_representation, solidity_code, 
                 function_signature, visibility, is_payable, is_view, metadata, hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pair.contract_address,
                pair.function_name,
                pair.tac_representation,
                pair.solidity_code,
                pair.function_signature,
                pair.visibility,
                pair.is_payable,
                pair.is_view,
                json.dumps(pair.metadata) if pair.metadata else None,
                hash_value
            ))
            conn.commit()
        except sqlite3.IntegrityError:
            # Duplicate hash - skip
            pass
        
        conn.close()
    
    def filter_and_clean_dataset(self, min_length: int = 50, max_length: int = 20000) -> int:
        """
        Filter and clean the dataset according to paper specifications.
        
        Args:
            min_length: Minimum function length
            max_length: Maximum sequence length (20,000 tokens as per paper)
            
        Returns:
            Number of function pairs after filtering
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Remove functions that are too short or too long
        cursor.execute('''
            DELETE FROM function_pairs 
            WHERE LENGTH(solidity_code) < ? OR LENGTH(tac_representation) > ?
        ''', (min_length, max_length))
        
        # Remove duplicate functions (same function signature and similar code)
        cursor.execute('''
            DELETE FROM function_pairs 
            WHERE id NOT IN (
                SELECT MIN(id) 
                FROM function_pairs 
                GROUP BY function_signature, SUBSTR(solidity_code, 1, 100)
            )
        ''')
        
        conn.commit()
        
        # Get final count
        cursor.execute('SELECT COUNT(*) FROM function_pairs')
        final_count = cursor.fetchone()[0]
        
        conn.close()
        
        self.logger.info(f"Dataset filtered to {final_count} function pairs")
        return final_count
    
    def export_dataset(self, output_format: str = "jsonl") -> str:
        """
        Export the dataset in specified format.
        
        Args:
            output_format: Export format ('jsonl', 'csv', 'parquet')
            
        Returns:
            Path to exported file
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get all function pairs
        query = '''
            SELECT function_name, tac_representation, solidity_code, function_signature,
                   visibility, is_payable, is_view, contract_address, metadata
            FROM function_pairs
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Create export filename
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"smart_contract_dataset_{timestamp}.{output_format}"
        filepath = self.output_dir / filename
        
        if output_format == "jsonl":
            with open(filepath, 'w') as f:
                for _, row in df.iterrows():
                    record = {
                        'input': row['tac_representation'],
                        'output': row['solidity_code'],
                        'metadata': {
                            'function_name': row['function_name'],
                            'function_signature': row['function_signature'],
                            'visibility': row['visibility'],
                            'is_payable': bool(row['is_payable']),
                            'is_view': bool(row['is_view']),
                            'contract_address': row['contract_address']
                        }
                    }
                    f.write(json.dumps(record) + '\n')
        
        elif output_format == "csv":
            df.to_csv(filepath, index=False)
        
        elif output_format == "parquet":
            df.to_parquet(filepath, index=False)
        
        self.logger.info(f"Dataset exported to {filepath}")
        return str(filepath)
    
    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the collected dataset."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total contracts and function pairs
        cursor.execute('SELECT COUNT(*) FROM contracts')
        stats['total_contracts'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM function_pairs')
        stats['total_function_pairs'] = cursor.fetchone()[0]
        
        # Function visibility distribution
        cursor.execute('''
            SELECT visibility, COUNT(*) as count 
            FROM function_pairs 
            GROUP BY visibility
        ''')
        stats['visibility_distribution'] = dict(cursor.fetchall())
        
        # Function length statistics
        cursor.execute('''
            SELECT 
                AVG(LENGTH(solidity_code)) as avg_solidity_length,
                AVG(LENGTH(tac_representation)) as avg_tac_length,
                MIN(LENGTH(solidity_code)) as min_solidity_length,
                MAX(LENGTH(solidity_code)) as max_solidity_length
            FROM function_pairs
        ''')
        length_stats = cursor.fetchone()
        stats['length_statistics'] = {
            'avg_solidity_length': length_stats[0],
            'avg_tac_length': length_stats[1],
            'min_solidity_length': length_stats[2],
            'max_solidity_length': length_stats[3]
        }
        
        conn.close()
        return stats

def main():
    """Example usage of the dataset pipeline."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dataset_pipeline.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Initialize dataset builder
    api_key = os.getenv('ETHERSCAN_API_KEY')
    # Try to load API key from settings.yaml first
    settings_path = Path(__file__).parent / "settings.yaml"
    if settings_path.exists():
        with open(settings_path, 'r') as f:
            settings = yaml.safe_load(f)
            api_key = settings.get('ETHERSCAN_API_KEY', api_key)
    
    if not api_key:
        logger.error("Please set ETHERSCAN_API_KEY environment variable or add it to settings.yaml")
        return
    
    builder = DatasetBuilder(api_key)
    
    # Example contract addresses (you would have a larger list)
    sample_addresses = [
        "0x183c1c01832b3fC9547f7036ebe7cA78fF57D783",  # Example addresses
        "0xF735732D923E758b59e11445E491403a0290f791",
        "0x155227B89B27d809bac144448b255cbd0CEa3AFd"
        # Add more verified contract addresses
    ]
    
    # Collect contracts
    logger.info("Starting contract collection...")
    collected = builder.collect_contracts(sample_addresses)
    logger.info(f"Collected {collected} contracts")
    
    if collected == 0:
        logger.warning("No contracts collected. Check API key and contract addresses.")
        return
    
    # Process to function pairs
    logger.info("Creating function pairs...")
    pairs = builder.process_contracts_to_function_pairs()
    logger.info(f"Created {pairs} function pairs")
    
    if pairs == 0:
        logger.warning("No function pairs created. Check logs for details.")
        return
    
    # Filter dataset
    logger.info("Filtering dataset...")
    filtered = builder.filter_and_clean_dataset()
    logger.info(f"Filtered to {filtered} function pairs")
    
    # Export dataset
    logger.info("Exporting dataset...")
    output_file = builder.export_dataset("jsonl")
    logger.info(f"Dataset exported to {output_file}")
    
    # Print statistics
    stats = builder.get_dataset_statistics()
    logger.info("\nDataset Statistics:")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")

if __name__ == "__main__":
    main()
