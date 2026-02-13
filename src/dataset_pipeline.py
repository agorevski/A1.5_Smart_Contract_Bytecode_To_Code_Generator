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
import traceback
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from web3 import Web3
from eth_utils import to_checksum_address
import requests
from tqdm import tqdm

from .bytecode_analyzer import BytecodeAnalyzer, analyze_bytecode_to_tac
from .local_compiler import (
    compile_source,
    compile_multi_file,
    parse_etherscan_source,
    parse_pragma,
    select_compilation_configs,
    install_solc_version,
    _normalize_version,
)
import yaml


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
    metadata: Optional[Dict] = field(default=None)


class EtherscanAPI:
    """Interface for collecting verified contracts from Etherscan."""

    def __init__(self, api_key: str, base_url: str = "https://api.etherscan.io/v2/api"):
        """Initialize the Etherscan API client.

        Args:
            api_key: Etherscan API key for authentication.
            base_url: Base URL for Etherscan API. Defaults to v2 API.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)

    def get_contract_source(self, address: str) -> Optional[ContractData]:
        """Get verified contract source code from Etherscan.

        Args:
            address: Contract address.

        Returns:
            ContractData object or None if not available.
        """
        try:
            params = {
                "chainid": "1",
                "module": "contract",
                "action": "getsourcecode",
                "address": address,
                "apikey": self.api_key,
            }

            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "1" or not data.get("result"):
                return None

            result = data["result"][0]

            if not result.get("SourceCode"):
                return None

            bytecode_params = {
                "chainid": "1",
                "module": "proxy",
                "action": "eth_getCode",
                "address": address,
                "tag": "latest",
                "apikey": self.api_key,
            }

            bytecode_response = self.session.get(
                self.base_url, params=bytecode_params, timeout=30
            )
            bytecode_data = bytecode_response.json()

            if bytecode_data.get("result") == "0x":
                return None

            return ContractData(
                address=to_checksum_address(address),
                source_code=result.get("SourceCode", ""),
                bytecode=bytecode_data.get("result", ""),
                compiler_version=result.get("CompilerVersion", ""),
                optimization_enabled=result.get("OptimizationUsed") == "1",
                optimization_runs=(
                    int(result.get("Runs", "0")) if result.get("Runs") else 0
                ),
                abi=result.get("ABI"),
            )

        except Exception as e:
            self.logger.error(f"Failed to get contract {address}: {e}")
            return None

    def get_verified_contracts_batch(
        self, start_block: int, end_block: int, limit: int = 1000
    ) -> List[str]:
        """Get a batch of verified contract addresses from a block range.

        This is a simplified placeholder. A production implementation would
        scan blocks or use dedicated discovery endpoints.

        Args:
            start_block: Starting block number.
            end_block: Ending block number.
            limit: Maximum number of contracts to return.

        Returns:
            List of contract addresses.
        """
        try:
            contracts: List[str] = []
            return contracts[:limit]
        except Exception as e:
            self.logger.error(f"Failed to get contracts batch: {e}")
            return []


class SolidityParser:
    """Parser for extracting functions from Solidity source code."""

    def __init__(self):
        """Initialize the Solidity parser."""
        self.logger = logging.getLogger(__name__)

    def extract_functions(
        self, source_code: str, contract_name: Optional[str] = None
    ) -> List[Dict]:
        """Extract individual functions from Solidity source code.

        Args:
            source_code: Complete Solidity source code.
            contract_name: Optional specific contract name to extract from.

        Returns:
            List of function dictionaries with keys: name, body, signature,
            visibility, is_payable, is_view, contract_name.
        """
        functions: List[Dict] = []

        try:
            cleaned_source = self._clean_source_code(source_code)
            contracts = self._extract_contracts(cleaned_source)

            for contract in contracts:
                if contract_name and contract["name"] != contract_name:
                    continue

                contract_functions = self._extract_functions_from_contract(
                    contract["body"]
                )

                for func in contract_functions:
                    func["contract_name"] = contract["name"]
                    functions.append(func)

        except Exception as e:
            self.logger.error(f"Failed to parse Solidity code: {e}")

        return functions

    def _clean_source_code(self, source_code: str) -> str:
        """Clean and normalize Solidity source code.

        Handles JSON-encoded source (Etherscan multi-file format) and
        double-brace wrapped JSON (``{{...}}``).

        Args:
            source_code: Raw Solidity source code, possibly JSON-encoded.

        Returns:
            Cleaned and normalized source code string.
        """
        if source_code.startswith("{"):
            try:
                source_json = json.loads(source_code)

                if "sources" in source_json:
                    combined_source = ""
                    for file_path, file_data in source_json["sources"].items():
                        if "content" in file_data:
                            combined_source += f"// File: {file_path}\n"
                            combined_source += file_data["content"] + "\n\n"
                    source_code = combined_source
                elif "content" in source_json:
                    source_code = source_json["content"]

            except json.JSONDecodeError:
                if source_code.startswith("{{"):
                    try:
                        inner_json = source_code[1:-1]
                        source_json = json.loads(inner_json)

                        if "sources" in source_json:
                            combined_source = ""
                            for file_path, file_data in source_json[
                                "sources"
                            ].items():
                                if "content" in file_data:
                                    combined_source += f"// File: {file_path}\n"
                                    combined_source += file_data["content"] + "\n\n"
                            source_code = combined_source
                    except (json.JSONDecodeError, IndexError):
                        self.logger.warning(
                            "Failed to parse JSON-encoded source code"
                        )

        return source_code

    def _extract_contracts(self, source_code: str) -> List[Dict]:
        """Extract contract definitions from source code using brace counting.

        Args:
            source_code: Cleaned Solidity source code.

        Returns:
            List of dicts with keys: name, type, body.
        """
        contracts: List[Dict] = []

        contract_starts = []
        for match in re.finditer(
            r"\b(contract|interface|library)\s+(\w+)", source_code
        ):
            contract_starts.append(
                {"type": match.group(1), "name": match.group(2), "start": match.end()}
            )

        for contract_info in contract_starts:
            start_pos = contract_info["start"]
            brace_pos = source_code.find("{", start_pos)

            if brace_pos == -1:
                continue

            body_start = brace_pos + 1
            body_end = self._find_matching_brace(source_code, brace_pos)

            if body_end != -1:
                contracts.append(
                    {
                        "name": contract_info["name"],
                        "type": contract_info["type"],
                        "body": source_code[body_start:body_end],
                    }
                )
                self.logger.debug(
                    f"Extracted {contract_info['type']} {contract_info['name']}"
                )

        return contracts

    def _extract_functions_from_contract(self, contract_body: str) -> List[Dict]:
        """Extract function definitions from a contract body.

        Handles both concrete functions (with ``{ body }``) and abstract /
        interface functions (terminated by ``;``).

        Args:
            contract_body: The body content of a Solidity contract.

        Returns:
            List of function dicts with keys: name, body, signature,
            visibility, is_payable, is_view.
        """
        functions: List[Dict] = []

        function_starts = []
        for match in re.finditer(r"\bfunction\s+(\w+)\s*\(", contract_body):
            function_starts.append(
                {
                    "name": match.group(1),
                    "start": match.start(),
                    "params_start": match.end(),
                }
            )

        for func_info in function_starts:
            try:
                params_end = contract_body.find(")", func_info["params_start"])
                if params_end == -1:
                    continue

                modifier_start = params_end + 1
                brace_pos = contract_body.find("{", modifier_start)
                semicolon_pos = contract_body.find(";", modifier_start)

                # Determine if this is an abstract function (no body).
                # An abstract function has a semicolon before any opening
                # brace, or has no opening brace at all.
                is_abstract = False
                if brace_pos == -1:
                    is_abstract = True
                elif semicolon_pos != -1 and semicolon_pos < brace_pos:
                    is_abstract = True

                if is_abstract:
                    if semicolon_pos != -1:
                        full_function = contract_body[
                            func_info["start"] : semicolon_pos + 1
                        ]
                        self._add_function_to_list(
                            functions, func_info["name"], full_function, ""
                        )
                    continue

                # Concrete function with a body.
                body_end = self._find_matching_brace(contract_body, brace_pos)

                if body_end != -1:
                    full_function = contract_body[func_info["start"] : body_end + 1]
                    function_body = contract_body[brace_pos + 1 : body_end]
                    self._add_function_to_list(
                        functions, func_info["name"], full_function, function_body
                    )

            except Exception as e:
                self.logger.warning(
                    f"Failed to extract function {func_info['name']}: {e}"
                )
                continue

        return functions

    def _find_matching_brace(self, text: str, start_pos: int) -> int:
        """Find the matching closing brace for an opening brace.

        Correctly handles single-line comments (``//``), block comments
        (``/* ... */``), string literals, and character literals.

        Args:
            text: Source text.
            start_pos: Position of the opening ``{`` in *text*.

        Returns:
            Position of matching ``}``, or ``-1`` if not found.
        """
        if start_pos >= len(text) or text[start_pos] != "{":
            return -1

        brace_count = 1
        pos = start_pos + 1
        in_string = False
        in_char = False
        in_comment = False
        in_line_comment = False

        while pos < len(text) and brace_count > 0:
            char = text[pos]

            # Detect comment starts (only outside strings / chars).
            if not in_string and not in_char and not in_comment and not in_line_comment:
                if pos < len(text) - 1:
                    two = text[pos : pos + 2]
                    if two == "//":
                        in_line_comment = True
                        pos += 2
                        continue
                    if two == "/*":
                        in_comment = True
                        pos += 2
                        continue

            # End of block comment.
            if in_comment:
                if pos < len(text) - 1 and text[pos : pos + 2] == "*/":
                    in_comment = False
                    pos += 2
                    continue
                pos += 1
                continue

            # End of line comment.
            if in_line_comment:
                if char == "\n":
                    in_line_comment = False
                pos += 1
                continue

            # Toggle string literal.
            if char == '"' and not in_char:
                if pos == 0 or text[pos - 1] != "\\":
                    in_string = not in_string

            # Toggle char literal.
            elif char == "'" and not in_string:
                if pos == 0 or text[pos - 1] != "\\":
                    in_char = not in_char

            # Count braces only outside strings, chars, and comments.
            if not in_string and not in_char:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1

            pos += 1

        return pos - 1 if brace_count == 0 else -1

    def _add_function_to_list(
        self,
        functions: List[Dict],
        name: str,
        full_function: str,
        body: str,
    ) -> None:
        """Add a parsed function to *functions* with extracted metadata.

        Args:
            functions: List to append the function dict to.
            name: Function name.
            full_function: Complete function source including signature and body.
            body: Function body content only (empty string for abstract functions).
        """
        visibility = self._extract_visibility(full_function)
        is_payable = "payable" in full_function
        is_view = "view" in full_function or "pure" in full_function

        signature_match = re.search(r"function\s+\w+\s*\([^)]*\)", full_function)
        signature = (
            signature_match.group(0) if signature_match else f"function {name}()"
        )

        functions.append(
            {
                "name": name,
                "body": full_function,
                "signature": signature,
                "visibility": visibility,
                "is_payable": is_payable,
                "is_view": is_view,
            }
        )

        self.logger.debug(f"Extracted function {name} ({visibility})")

    def _extract_visibility(self, function_code: str) -> str:
        """Extract function visibility from the function signature.

        Only examines the text up to the first ``{`` or ``;`` so that
        keywords inside the function body do not cause false matches.

        Args:
            function_code: Complete function source code.

        Returns:
            One of ``'private'``, ``'internal'``, ``'external'``, or ``'public'``.
        """
        # Restrict search to the signature portion (before { or ;).
        sig_end = len(function_code)
        brace = function_code.find("{")
        semi = function_code.find(";")
        if brace != -1:
            sig_end = min(sig_end, brace)
        if semi != -1:
            sig_end = min(sig_end, semi)
        signature_part = function_code[:sig_end]

        if re.search(r"\bprivate\b", signature_part):
            return "private"
        if re.search(r"\binternal\b", signature_part):
            return "internal"
        if re.search(r"\bexternal\b", signature_part):
            return "external"
        if re.search(r"\bpublic\b", signature_part):
            return "public"
        # Default visibility in Solidity.
        return "public"


class DatasetBuilder:
    """Main class for building the training dataset."""

    def __init__(self, etherscan_api_key: str, output_dir: str = "data"):
        """Initialize the dataset builder.

        Args:
            etherscan_api_key: API key for Etherscan.
            output_dir: Directory for output files and database. Defaults to ``"data"``.
        """
        self.etherscan = EtherscanAPI(etherscan_api_key)
        self.parser = SolidityParser()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

        self.db_path = self.output_dir / "contracts.db"
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database for caching contract data.

        Creates ``contracts`` and ``function_pairs`` tables if they don't exist.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
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
        """
        )

        cursor.execute(
            """
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
        """
        )

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------ #
    #  Contract Collection
    # ------------------------------------------------------------------ #

    def collect_contracts(
        self, contract_addresses: List[str], max_workers: int = 10
    ) -> int:
        """Collect contract data from Etherscan in parallel.

        Args:
            contract_addresses: List of contract addresses to collect.
            max_workers: Number of parallel workers.

        Returns:
            Number of successfully collected contracts.
        """
        collected = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_address = {
                executor.submit(self.etherscan.get_contract_source, addr): addr
                for addr in contract_addresses
            }

            for future in tqdm(
                as_completed(future_to_address),
                total=len(contract_addresses),
                desc="Collecting contracts",
            ):
                address = future_to_address[future]
                try:
                    contract_data = future.result()
                    if contract_data:
                        self._store_contract(contract_data)
                        collected += 1
                except Exception as e:
                    self.logger.error(f"Failed to process contract {address}: {e}")

        self.logger.info(
            f"Collected {collected} contracts out of {len(contract_addresses)} addresses"
        )
        return collected

    def _store_contract(self, contract_data: ContractData) -> None:
        """Store contract data in database.

        Args:
            contract_data: ContractData object containing contract information.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO contracts
            (address, source_code, bytecode, compiler_version,
             optimization_enabled, optimization_runs)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                contract_data.address,
                contract_data.source_code,
                contract_data.bytecode,
                contract_data.compiler_version,
                contract_data.optimization_enabled,
                contract_data.optimization_runs,
            ),
        )

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------ #
    #  Compile-and-collect pipeline
    # ------------------------------------------------------------------ #

    def collect_and_compile_contracts(
        self,
        contract_addresses: List[str],
        max_compiler_configs: int = 2,
    ) -> int:
        """Download source from Etherscan and compile locally with multiple solc versions.

        This avoids fetching runtime bytecode from a live node.  Instead it:

        1. Downloads verified source + compiler metadata from Etherscan.
        2. Parses the pragma to find compatible compiler versions.
        3. Compiles locally with up to *max_compiler_configs* settings.
        4. Creates TAC→Solidity training pairs for each compilation.

        Args:
            contract_addresses: List of contract addresses.
            max_compiler_configs: Max compilations per contract for augmentation.

        Returns:
            Total number of function pairs created.
        """
        total_pairs = 0

        for addr in tqdm(contract_addresses, desc="Contracts"):
            try:
                contract_data = self.etherscan.get_contract_source(addr)
                if contract_data is None:
                    self.logger.warning(
                        f"Skipping {addr}: not verified or unavailable"
                    )
                    continue

                raw_source = contract_data.source_code
                original_version = contract_data.compiler_version
                original_opt = contract_data.optimization_enabled
                original_runs = contract_data.optimization_runs

                source_files = parse_etherscan_source(raw_source)
                if not source_files:
                    self.logger.warning(f"Skipping {addr}: empty source")
                    continue

                combined_source = "\n\n".join(source_files.values())
                pragmas = parse_pragma(combined_source)
                pragma = pragmas[0] if pragmas else ">=0.4.0"

                configs = select_compilation_configs(
                    pragma,
                    original_version=original_version,
                    original_optimizer=original_opt,
                    original_runs=original_runs,
                    max_configs=max_compiler_configs,
                )

                if not configs:
                    self.logger.warning(
                        f"Skipping {addr}: no compatible compiler found"
                    )
                    continue

                solidity_functions = self.parser.extract_functions(combined_source)
                if not solidity_functions:
                    self.logger.warning(
                        f"Skipping {addr}: no functions extracted from source"
                    )
                    continue

                solidity_with_selectors = self._add_selectors_to_solidity_functions(
                    solidity_functions
                )

                for cfg in configs:
                    ver = cfg["version"]
                    opt = cfg["optimizer_enabled"]
                    runs = cfg["optimizer_runs"]

                    self.logger.info(
                        f"  Compiling {addr} with solc {ver} "
                        f"(opt={'on' if opt else 'off'}, runs={runs})"
                    )

                    if len(source_files) > 1:
                        comp = compile_multi_file(source_files, ver, opt, runs)
                    else:
                        first_source = next(iter(source_files.values()))
                        comp = compile_source(first_source, ver, opt, runs)

                    if not comp.success:
                        self.logger.warning(
                            f"  Compilation failed for {addr} with solc {ver}: "
                            + "; ".join(comp.errors[:2])
                        )
                        continue

                    for cname, compiled in comp.contracts.items():
                        bytecode_hex = "0x" + compiled.runtime_bytecode
                        try:
                            analyzer = BytecodeAnalyzer(bytecode_hex)
                            analyzer.analyze_control_flow()
                            bytecode_functions = analyzer.identify_functions()
                        except Exception as e:
                            self.logger.warning(
                                f"  TAC analysis failed for {cname}: {e}"
                            )
                            continue

                        contract_sol_funcs = [
                            f
                            for f in solidity_with_selectors
                            if f.get("contract_name", "") == cname
                        ] or solidity_with_selectors

                        matched = self._match_functions_by_selector(
                            contract_sol_funcs, bytecode_functions, analyzer
                        )

                        for m in matched:
                            pair = self._build_training_pair(m, addr)
                            if pair:
                                pair.metadata = pair.metadata or {}
                                pair.metadata["compiler_version"] = ver
                                pair.metadata["optimizer_enabled"] = opt
                                pair.metadata["optimizer_runs"] = runs
                                pair.metadata["compiled_contract"] = cname
                                self._store_function_pair(pair)
                                total_pairs += 1

                    self.logger.info(
                        f"  {addr} solc {ver}: contributed pairs so far = {total_pairs}"
                    )

                self._store_contract(contract_data)

            except Exception as e:
                self.logger.error(f"Failed to process {addr}: {e}")
                self.logger.debug(traceback.format_exc())

        self.logger.info(f"Total function pairs created: {total_pairs}")
        return total_pairs

    # ------------------------------------------------------------------ #
    #  Process stored contracts
    # ------------------------------------------------------------------ #

    def process_contracts_to_function_pairs(self, batch_size: int = 1) -> int:
        """Process stored contracts to create TAC-to-Solidity function pairs.

        Only contracts that have not yet been processed are considered.

        Args:
            batch_size: Number of contracts to process in each batch.

        Returns:
            Number of function pairs created.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT address, source_code, bytecode FROM contracts WHERE processed = FALSE"
        )
        unprocessed_contracts = cursor.fetchall()
        total_pairs = 0

        for i in tqdm(
            range(0, len(unprocessed_contracts), batch_size),
            desc="Processing contracts",
        ):
            batch = unprocessed_contracts[i : i + batch_size]

            for address, source_code, bytecode in batch:
                try:
                    pairs = self._create_function_pairs(address, source_code, bytecode)

                    for pair in pairs:
                        self._store_function_pair(pair)
                        total_pairs += 1

                    cursor.execute(
                        "UPDATE contracts SET processed = TRUE WHERE address = ?",
                        (address,),
                    )

                except Exception as e:
                    self.logger.error(f"Failed to process contract {address}: {e}")

            conn.commit()

        conn.close()
        self.logger.info(f"Created {total_pairs} function pairs")
        return total_pairs

    def _create_function_pairs(
        self, address: str, source_code: str, bytecode: str
    ) -> List[FunctionPair]:
        """Create TAC-to-Solidity function pairs from contract data.

        Args:
            address: Contract address.
            source_code: Verified Solidity source code.
            bytecode: Contract runtime bytecode.

        Returns:
            List of FunctionPair objects for training.
        """
        pairs: List[FunctionPair] = []

        try:
            self.logger.info(f"Analyzing bytecode for {address}")
            analyzer = BytecodeAnalyzer(bytecode)
            analyzer.analyze_control_flow()
            bytecode_functions = analyzer.identify_functions()

            self.logger.info(f"Parsing Solidity source for {address}")
            solidity_functions = self.parser.extract_functions(source_code)

            if not solidity_functions:
                self.logger.warning(f"No Solidity functions found for {address}")
                return pairs

            solidity_with_selectors = self._add_selectors_to_solidity_functions(
                solidity_functions
            )

            matched_pairs = self._match_functions_by_selector(
                solidity_with_selectors, bytecode_functions, analyzer
            )

            self.logger.info(f"Matched {len(matched_pairs)} functions for {address}")

            for match in matched_pairs:
                try:
                    pair = self._build_training_pair(match, address)
                    if pair:
                        pairs.append(pair)
                except Exception as e:
                    self.logger.error(f"Failed to build training pair: {e}")
                    continue

            if not pairs and solidity_functions:
                self.logger.info(
                    f"No matched functions, creating whole-contract pair for {address}"
                )
                fallback_pair = self._create_fallback_pair(
                    address, source_code, analyzer
                )
                if fallback_pair:
                    pairs.append(fallback_pair)

        except Exception as e:
            self.logger.error(f"Failed to create function pairs for {address}: {e}")
            self.logger.error(f"Error traceback: {traceback.format_exc()}")

        return pairs

    # ------------------------------------------------------------------ #
    #  Selector helpers
    # ------------------------------------------------------------------ #

    def _add_selectors_to_solidity_functions(
        self, functions: List[Dict]
    ) -> List[Dict]:
        """Calculate function selectors for Solidity functions.

        Mutates each dict in *functions* by adding a ``'selector'`` key.

        Args:
            functions: List of function dictionaries.

        Returns:
            The same list, with ``'selector'`` added to each element.
        """
        for func in functions:
            try:
                signature = func["signature"]
                match = re.match(r"function\s+(\w+)\s*\(([^)]*)\)", signature)
                if match:
                    func_name = match.group(1)
                    params_str = match.group(2).strip()

                    if params_str:
                        param_types = []
                        for param in params_str.split(","):
                            param = param.strip()
                            if param:
                                parts = param.split()
                                param_type = parts[0]
                                param_types.append(param_type)
                        canonical = f"{func_name}({','.join(param_types)})"
                    else:
                        canonical = f"{func_name}()"
                else:
                    canonical = signature.replace("function ", "").strip()

                selector_hash = Web3.keccak(text=canonical)[:4]
                func["selector"] = "0x" + selector_hash.hex()

                self.logger.debug(
                    f"Calculated selector {func['selector']} for {canonical}"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to calculate selector for {func['name']}: {e}"
                )
                func["selector"] = None

        return functions

    def _match_functions_by_selector(
        self,
        solidity_functions: List[Dict],
        bytecode_functions: Dict,
        analyzer: BytecodeAnalyzer,
    ) -> List[Dict]:
        """Match Solidity functions with bytecode functions by selector.

        Args:
            solidity_functions: Solidity functions with selectors.
            bytecode_functions: Bytecode functions from analyzer (name → Function).
            analyzer: BytecodeAnalyzer instance.

        Returns:
            List of matched function dicts.
        """
        matches: List[Dict] = []

        solidity_by_selector = {
            f["selector"]: f for f in solidity_functions if f.get("selector")
        }
        bytecode_by_selector = {
            f.selector: f for f in bytecode_functions.values() if f.selector
        }

        for selector, sol_func in solidity_by_selector.items():
            if selector in bytecode_by_selector:
                bytecode_func = bytecode_by_selector[selector]
                tac = self._extract_tac_for_function(bytecode_func, analyzer)
                matches.append(
                    {
                        "solidity_function": sol_func,
                        "bytecode_function": bytecode_func,
                        "tac": tac,
                        "selector": selector,
                    }
                )
                self.logger.debug(
                    f"Matched function {sol_func['name']} with selector {selector}"
                )
            else:
                self.logger.debug(
                    f"No bytecode match for {sol_func['name']} (selector: {selector})"
                )

        return matches

    # ------------------------------------------------------------------ #
    #  TAC extraction
    # ------------------------------------------------------------------ #

    def _extract_tac_for_function(
        self, bytecode_function, analyzer: BytecodeAnalyzer
    ) -> str:
        """Extract TAC representation for a specific function.

        Args:
            bytecode_function: Function object from BytecodeAnalyzer.
            analyzer: BytecodeAnalyzer instance with analyzed blocks.

        Returns:
            Formatted TAC string for the function.
        """
        tac_lines: List[str] = []

        try:
            func_name = bytecode_function.name
            tac_lines.append(f"function {func_name}:")

            if bytecode_function.selector:
                tac_lines.append(f"  // Selector: {bytecode_function.selector}")

            tac_lines.append(f"  // Entry block: {bytecode_function.entry_block}")

            function_blocks = (
                bytecode_function.basic_blocks
                if bytecode_function.basic_blocks
                else []
            )

            if (
                not function_blocks
                and bytecode_function.entry_block in analyzer.basic_blocks
            ):
                function_blocks = self._collect_function_blocks(
                    bytecode_function.entry_block, analyzer.basic_blocks
                )

            for block in function_blocks:
                tac_lines.append(f"  {block.id}:")

                if block.predecessors:
                    tac_lines.append(
                        f"    // Predecessors: {', '.join(block.predecessors)}"
                    )
                if block.successors:
                    tac_lines.append(
                        f"    // Successors: {', '.join(block.successors)}"
                    )

                for instr in block.instructions:
                    formatted = analyzer._format_tac_instruction(instr)
                    tac_lines.append(f"    {formatted}")

                tac_lines.append("")

        except Exception as e:
            self.logger.error(f"Failed to extract TAC for function: {e}")
            tac_lines.append(f"  // Error extracting TAC: {e}")

        return "\n".join(tac_lines)

    def _collect_function_blocks(
        self, entry_block_id: str, all_blocks: Dict
    ) -> List:
        """Collect all basic blocks belonging to a function via graph traversal.

        Args:
            entry_block_id: Entry block ID for the function.
            all_blocks: Dictionary of all basic blocks.

        Returns:
            List of BasicBlock objects in the function.
        """
        if entry_block_id not in all_blocks:
            return []

        visited: set = set()
        blocks: list = []

        def traverse(block_id: str) -> None:
            if block_id in visited or block_id not in all_blocks:
                return
            visited.add(block_id)
            block = all_blocks[block_id]
            blocks.append(block)
            for successor in block.successors:
                traverse(successor)

        traverse(entry_block_id)
        return blocks

    # ------------------------------------------------------------------ #
    #  Training pair construction
    # ------------------------------------------------------------------ #

    def _build_training_pair(
        self, match: Dict, address: str
    ) -> Optional[FunctionPair]:
        """Build a training pair from a matched function.

        Args:
            match: Matched function dictionary.
            address: Contract address.

        Returns:
            FunctionPair object or ``None`` if the function is too small.
        """
        sol_func = match["solidity_function"]
        tac = match["tac"]

        if len(sol_func["body"].strip()) < 10:
            return None

        if not tac or len(tac.strip()) < 10:
            return None

        return FunctionPair(
            function_name=sol_func["name"],
            tac_representation=tac,
            solidity_code=sol_func["body"],
            function_signature=sol_func["signature"],
            visibility=sol_func["visibility"],
            is_payable=sol_func["is_payable"],
            is_view=sol_func["is_view"],
            contract_address=address,
            metadata={
                "contract_name": sol_func.get("contract_name"),
                "selector": match["selector"],
                "matched_by_selector": True,
            },
        )

    def _create_fallback_pair(
        self,
        address: str,
        source_code: str,
        analyzer: BytecodeAnalyzer,
    ) -> Optional[FunctionPair]:
        """Create a whole-contract training pair as fallback.

        Args:
            address: Contract address.
            source_code: Full Solidity source.
            analyzer: BytecodeAnalyzer instance.

        Returns:
            FunctionPair for entire contract, or ``None`` on error.
        """
        try:
            tac = analyzer.generate_tac_representation()
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
                metadata={"whole_contract": True, "fallback_pair": True},
            )
        except Exception as e:
            self.logger.error(f"Failed to create fallback pair: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Database helpers
    # ------------------------------------------------------------------ #

    def _store_function_pair(self, pair: FunctionPair) -> None:
        """Store function pair in database. Duplicates are skipped via hash.

        Args:
            pair: FunctionPair object to store.
        """
        content = f"{pair.tac_representation}{pair.solidity_code}"
        hash_value = hashlib.md5(content.encode()).hexdigest()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO function_pairs
                (contract_address, function_name, tac_representation, solidity_code,
                 function_signature, visibility, is_payable, is_view, metadata, hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    pair.contract_address,
                    pair.function_name,
                    pair.tac_representation,
                    pair.solidity_code,
                    pair.function_signature,
                    pair.visibility,
                    pair.is_payable,
                    pair.is_view,
                    json.dumps(pair.metadata) if pair.metadata else None,
                    hash_value,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            pass  # Duplicate hash — skip.

        conn.close()

    # ------------------------------------------------------------------ #
    #  Filtering & export
    # ------------------------------------------------------------------ #

    def filter_and_clean_dataset(
        self, min_length: int = 50, max_length: int = 20000
    ) -> int:
        """Filter and clean the dataset according to paper specifications.

        Args:
            min_length: Minimum Solidity function length (chars).
            max_length: Maximum TAC representation length (chars, 20 000 as per paper).

        Returns:
            Number of function pairs after filtering.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM function_pairs
            WHERE LENGTH(solidity_code) < ? OR LENGTH(tac_representation) > ?
        """,
            (min_length, max_length),
        )

        cursor.execute(
            """
            DELETE FROM function_pairs
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM function_pairs
                GROUP BY function_signature, SUBSTR(solidity_code, 1, 100)
            )
        """
        )

        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM function_pairs")
        final_count = cursor.fetchone()[0]

        conn.close()

        self.logger.info(f"Dataset filtered to {final_count} function pairs")
        return final_count

    def export_dataset(self, output_format: str = "jsonl") -> str:
        """Export the dataset in the specified format.

        Args:
            output_format: Export format (``'jsonl'``, ``'csv'``, or ``'parquet'``).

        Returns:
            Path to exported file.
        """
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT function_name, tac_representation, solidity_code, function_signature,
                   visibility, is_payable, is_view, contract_address, metadata
            FROM function_pairs
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"smart_contract_dataset_{timestamp}.{output_format}"
        filepath = self.output_dir / filename

        if output_format == "jsonl":
            with open(filepath, "w") as f:
                for _, row in df.iterrows():
                    record = {
                        "input": row["tac_representation"],
                        "output": row["solidity_code"],
                        "metadata": {
                            "function_name": row["function_name"],
                            "function_signature": row["function_signature"],
                            "visibility": row["visibility"],
                            "is_payable": bool(row["is_payable"]),
                            "is_view": bool(row["is_view"]),
                            "contract_address": row["contract_address"],
                        },
                    }
                    f.write(json.dumps(record) + "\n")

        elif output_format == "csv":
            df.to_csv(filepath, index=False)

        elif output_format == "parquet":
            df.to_parquet(filepath, index=False)

        self.logger.info(f"Dataset exported to {filepath}")
        return str(filepath)

    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the collected dataset.

        Returns:
            Dict with total_contracts, total_function_pairs,
            visibility_distribution, and length_statistics.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats: Dict = {}

        cursor.execute("SELECT COUNT(*) FROM contracts")
        stats["total_contracts"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM function_pairs")
        stats["total_function_pairs"] = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT visibility, COUNT(*) as count
            FROM function_pairs
            GROUP BY visibility
        """
        )
        stats["visibility_distribution"] = dict(cursor.fetchall())

        cursor.execute(
            """
            SELECT
                AVG(LENGTH(solidity_code)) as avg_solidity_length,
                AVG(LENGTH(tac_representation)) as avg_tac_length,
                MIN(LENGTH(solidity_code)) as min_solidity_length,
                MAX(LENGTH(solidity_code)) as max_solidity_length
            FROM function_pairs
        """
        )
        length_stats = cursor.fetchone()
        stats["length_statistics"] = {
            "avg_solidity_length": length_stats[0],
            "avg_tac_length": length_stats[1],
            "min_solidity_length": length_stats[2],
            "max_solidity_length": length_stats[3],
        }

        conn.close()
        return stats


# ---------------------------------------------------------------------- #
#  CLI entry point
# ---------------------------------------------------------------------- #


def main() -> None:
    """Run example usage of the dataset pipeline.

    Demonstrates the complete workflow: collecting contracts from Etherscan,
    processing them into TAC-Solidity function pairs, filtering the dataset,
    and exporting to JSONL format.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("dataset_pipeline.log"), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)

    api_key = os.getenv("ETHERSCAN_API_KEY")
    settings_path = Path(__file__).parent / "settings.yaml"
    if settings_path.exists():
        with open(settings_path, "r") as f:
            settings = yaml.safe_load(f)
            api_key = settings.get("ETHERSCAN_API_KEY", api_key)

    if not api_key:
        logger.error(
            "Please set ETHERSCAN_API_KEY environment variable or add it to settings.yaml"
        )
        return

    builder = DatasetBuilder(api_key)

    sample_addresses = [
        "0x183c1c01832b3fC9547f7036ebe7cA78fF57D783",
        "0xF735732D923E758b59e11445E491403a0290f791",
        "0x155227B89B27d809bac144448b255cbd0CEa3AFd",
    ]

    logger.info("Starting contract collection...")
    collected = builder.collect_contracts(sample_addresses)
    logger.info(f"Collected {collected} contracts")

    if collected == 0:
        logger.warning("No contracts collected. Check API key and contract addresses.")
        return

    logger.info("Creating function pairs...")
    pairs = builder.process_contracts_to_function_pairs()
    logger.info(f"Created {pairs} function pairs")

    if pairs == 0:
        logger.warning("No function pairs created. Check logs for details.")
        return

    logger.info("Filtering dataset...")
    filtered = builder.filter_and_clean_dataset()
    logger.info(f"Filtered to {filtered} function pairs")

    logger.info("Exporting dataset...")
    output_file = builder.export_dataset("jsonl")
    logger.info(f"Dataset exported to {output_file}")

    stats = builder.get_dataset_statistics()
    logger.info("\nDataset Statistics:")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")


if __name__ == "__main__":
    main()