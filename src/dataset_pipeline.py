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
    
    def __init__(self, api_key: str, base_url: str = "https://api.etherscan.io/api"):
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
                'module': 'contract',
                'action': 'getsourcecode',
                'address': address,
                'apikey': self.api_key
            }
            
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] != '1' or not data['result']:
                return None
            
            result = data['result'][0]
            
            # Check if contract is verified
            if not result['SourceCode']:
                return None
            
            # Get runtime bytecode
            bytecode_params = {
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
                source_code=result['SourceCode'],
                bytecode=bytecode_data.get('result', ''),
                compiler_version=result['CompilerVersion'],
                optimization_enabled=result['OptimizationUsed'] == '1',
                optimization_runs=int(result['Runs']) if result['Runs'] else 0,
                abi=result['ABI']
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
                pass
        
        # Remove comments (simplified - real parser would be more sophisticated)
        source_code = re.sub(r'//.*?\n', '\n', source_code)
        source_code = re.sub(r'/\*.*?\*/', '', source_code, flags=re.DOTALL)
        
        return source_code
    
    def _extract_contracts(self, source_code: str) -> List[Dict]:
        """Extract contract definitions from source code."""
        contracts = []
        
        # Find contract definitions
        contract_pattern = r'contract\s+(\w+)(?:\s+is\s+[^{]*)?{([^{}]*(?:{[^{}]*}[^{}]*)*)}'
        
        for match in re.finditer(contract_pattern, source_code, re.DOTALL):
            contracts.append({
                'name': match.group(1),
                'body': match.group(2)
            })
        
        return contracts
    
    def _extract_functions_from_contract(self, contract_body: str) -> List[Dict]:
        """Extract function definitions from contract body."""
        functions = []
        
        # Function pattern (simplified)
        function_pattern = r'function\s+(\w+)\s*\([^)]*\)(?:\s+[^{]*)?{([^{}]*(?:{[^{}]*}[^{}]*)*)}'
        
        for match in re.finditer(function_pattern, contract_body, re.DOTALL):
            function_name = match.group(1)
            function_body = match.group(2)
            full_function = match.group(0)
            
            # Extract function metadata
            visibility = self._extract_visibility(full_function)
            is_payable = 'payable' in full_function
            is_view = 'view' in full_function or 'pure' in full_function
            
            # Extract function signature
            signature_match = re.search(r'function\s+\w+\s*\([^)]*\)', full_function)
            signature = signature_match.group(0) if signature_match else f"function {function_name}()"
            
            functions.append({
                'name': function_name,
                'body': full_function,
                'signature': signature,
                'visibility': visibility,
                'is_payable': is_payable,
                'is_view': is_view
            })
        
        return functions
    
    def _extract_visibility(self, function_code: str) -> str:
        """Extract function visibility."""
        if 'private' in function_code:
            return 'private'
        elif 'internal' in function_code:
            return 'internal'
        elif 'external' in function_code:
            return 'external'
        else:
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
    
    def process_contracts_to_function_pairs(self, batch_size: int = 100) -> int:
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
        cursor.execute('SELECT address, source_code, bytecode FROM contracts WHERE processed = FALSE')
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
            # Extract functions from Solidity source
            solidity_functions = self.parser.extract_functions(source_code)
            
            # Generate TAC representation from bytecode
            analyzer = BytecodeAnalyzer(bytecode)
            tac_representation = analyzer.generate_tac_representation()
            
            # For each Solidity function, create a training pair
            # Note: This is simplified - real implementation would need more sophisticated
            # matching between bytecode functions and Solidity functions
            
            for func in solidity_functions:
                # Create function-specific TAC (simplified approach)
                function_tac = self._extract_function_tac(tac_representation, func['name'])
                
                if function_tac and len(func['body'].strip()) > 10:  # Filter very short functions
                    pair = FunctionPair(
                        function_name=func['name'],
                        tac_representation=function_tac,
                        solidity_code=func['body'],
                        function_signature=func['signature'],
                        visibility=func['visibility'],
                        is_payable=func['is_payable'],
                        is_view=func['is_view'],
                        contract_address=address,
                        metadata={'contract_name': func.get('contract_name')}
                    )
                    pairs.append(pair)
        
        except Exception as e:
            self.logger.error(f"Failed to create function pairs for {address}: {e}")
        
        return pairs
    
    def _extract_function_tac(self, full_tac: str, function_name: str) -> Optional[str]:
        """
        Extract TAC representation for a specific function.
        
        This is a simplified implementation. Real implementation would need
        more sophisticated function boundary detection.
        """
        lines = full_tac.split('\n')
        function_lines = []
        in_function = False
        
        for line in lines:
            if f'function {function_name}' in line or f'function_' in line:
                in_function = True
                function_lines.append(line)
            elif in_function:
                if line.strip() and not line.startswith('  ') and not line.startswith('\t'):
                    # End of function
                    break
                function_lines.append(line)
        
        return '\n'.join(function_lines) if function_lines else full_tac[:1000]  # Fallback
    
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
    # Initialize dataset builder
    api_key = os.getenv('ETHERSCAN_API_KEY')
    if not api_key:
        print("Please set ETHERSCAN_API_KEY environment variable")
        return
    
    builder = DatasetBuilder(api_key)
    
    # Example contract addresses (you would have a larger list)
    sample_addresses = [
        "0xA0b86a33E6411a3b4E4c3c4C4e4b5b5b5b5b5b5b",  # Example addresses
        "0xB0b86a33E6411a3b4E4c3c4C4e4b5b5b5b5b5b5b",
        # Add more verified contract addresses
    ]
    
    # Collect contracts
    print("Collecting contracts...")
    collected = builder.collect_contracts(sample_addresses)
    print(f"Collected {collected} contracts")
    
    # Process to function pairs
    print("Creating function pairs...")
    pairs = builder.process_contracts_to_function_pairs()
    print(f"Created {pairs} function pairs")
    
    # Filter dataset
    print("Filtering dataset...")
    filtered = builder.filter_and_clean_dataset()
    print(f"Filtered to {filtered} function pairs")
    
    # Export dataset
    print("Exporting dataset...")
    output_file = builder.export_dataset("jsonl")
    print(f"Dataset exported to {output_file}")
    
    # Print statistics
    stats = builder.get_dataset_statistics()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
