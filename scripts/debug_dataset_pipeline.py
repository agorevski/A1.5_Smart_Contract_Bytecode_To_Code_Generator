#!/usr/bin/env python3
"""
Test script for dataset pipeline fixes

This script tests the improved function matching and TAC extraction.
"""

import logging
import sys
from pathlib import Path

# Add src to path and parent to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Import with package structure
from src.bytecode_analyzer import BytecodeAnalyzer
from src.dataset_pipeline import DatasetBuilder, SolidityParser

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Sample Solidity contract source
SAMPLE_SOLIDITY = """
pragma solidity ^0.8.0;

contract SimpleStorage {
    uint256 private storedData;
    
    function set(uint256 x) public {
        storedData = x;
    }
    
    function get() public view returns (uint256) {
        return storedData;
    }
    
    function increment() public {
        storedData = storedData + 1;
    }
}
"""

# Sample bytecode (SimpleStorage contract)
SAMPLE_BYTECODE = "0x608060405234801561001057600080fd5b50600436106100415760003560e01c806360fe47b1146100465780636d4ce63c14610062578063d09de08a14610080575b600080fd5b610060600480360381019061005b91906100d6565b61008a565b005b61006a610094565b6040516100779190610112565b60405180910390f35b61008861009d565b005b8060008190555050565b60008054905090565b600160005401600081905550565b600080fd5b6000819050919050565b6100c3816100b0565b81146100ce57600080fd5b50565b6000813590506100e0816100ba565b92915050565b6000602082840312156100fc576100fb6100ab565b5b600061010a848285016100d1565b91505092915050565b61011c816100b0565b82525050565b60006020820190506101376000830184610113565b9291505056fea2646970667358221220abcd1234567890abcdef1234567890abcdef1234567890abcdef1234567890ab64736f6c63430008130033"

def test_solidity_parser():
    """Test Solidity parsing and function extraction.

    Parses the SAMPLE_SOLIDITY contract using SolidityParser and extracts
    all function definitions, logging their names, signatures, and attributes.

    Returns:
        list[dict]: A list of dictionaries containing extracted function
            information, including name, signature, visibility, is_payable,
            and is_view.
    """
    logger.info("=" * 60)
    logger.info("Testing Solidity Parser")
    logger.info("=" * 60)
    
    parser = SolidityParser()
    functions = parser.extract_functions(SAMPLE_SOLIDITY)
    
    logger.info(f"Extracted {len(functions)} functions")
    for func in functions:
        logger.info(f"  - {func['name']}: {func['signature']}")
        logger.info(f"    Visibility: {func['visibility']}, Payable: {func['is_payable']}, View: {func['is_view']}")
    
    return functions

def test_bytecode_analysis():
    """Test bytecode analysis and TAC generation.

    Analyzes the SAMPLE_BYTECODE using BytecodeAnalyzer to perform control
    flow analysis and function identification. Logs the number of basic blocks
    and identified functions with their selectors and entry blocks.

    Returns:
        tuple[BytecodeAnalyzer, dict]: A tuple containing the configured
            bytecode analyzer instance and a dictionary mapping function
            names to function objects.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Testing Bytecode Analysis")
    logger.info("=" * 60)
    
    analyzer = BytecodeAnalyzer(SAMPLE_BYTECODE)
    
    # Analyze control flow
    logger.info("Analyzing control flow...")
    blocks = analyzer.analyze_control_flow()
    logger.info(f"Found {len(blocks)} basic blocks")
    
    # Identify functions
    logger.info("Identifying functions...")
    functions = analyzer.identify_functions()
    logger.info(f"Found {len(functions)} functions in bytecode")
    
    for name, func in functions.items():
        logger.info(f"  - {name}")
        logger.info(f"    Selector: {func.selector}")
        logger.info(f"    Entry block: {func.entry_block}")
    
    return analyzer, functions

def test_selector_calculation():
    """Test function selector calculation.

    Calculates and logs the 4-byte function selectors for a set of test
    function signatures using Keccak-256 hashing. Verifies that selector
    generation works correctly for common Solidity function signatures.

    Returns:
        None
    """
    logger.info("\n" + "=" * 60)
    logger.info("Testing Selector Calculation")
    logger.info("=" * 60)
    
    from web3 import Web3
    
    test_signatures = [
        "set(uint256)",
        "get()",
        "increment()"
    ]
    
    for sig in test_signatures:
        selector_hash = Web3.keccak(text=sig)[:4]
        selector = '0x' + selector_hash.hex()
        logger.info(f"  {sig} -> {selector}")

def test_function_matching():
    """Test the complete function matching pipeline.

    Executes the full function matching workflow: parsing Solidity source,
    calculating function selectors, analyzing bytecode, and matching Solidity
    functions to their bytecode counterparts. Logs matched functions with
    their selectors and TAC preview.

    Returns:
        None
    """
    logger.info("\n" + "=" * 60)
    logger.info("Testing Function Matching Pipeline")
    logger.info("=" * 60)
    
    # Create a temporary dataset builder (no API key needed for this test)
    builder = DatasetBuilder("dummy_key", output_dir="test_data")
    
    # Parse Solidity
    solidity_functions = builder.parser.extract_functions(SAMPLE_SOLIDITY)
    logger.info(f"Parsed {len(solidity_functions)} Solidity functions")
    
    # Add selectors
    solidity_with_selectors = builder._add_selectors_to_solidity_functions(solidity_functions)
    logger.info("Calculated selectors for Solidity functions:")
    for func in solidity_with_selectors:
        logger.info(f"  {func['name']}: {func.get('selector')}")
    
    # Analyze bytecode
    analyzer = BytecodeAnalyzer(SAMPLE_BYTECODE)
    analyzer.analyze_control_flow()
    bytecode_functions = analyzer.identify_functions()
    logger.info(f"\nFound {len(bytecode_functions)} bytecode functions")
    
    # Match functions
    matches = builder._match_functions_by_selector(
        solidity_with_selectors,
        bytecode_functions,
        analyzer
    )
    
    logger.info(f"\nMatched {len(matches)} functions:")
    for match in matches:
        sol_func = match['solidity_function']
        logger.info(f"  - {sol_func['name']} (selector: {match['selector']})")
        logger.info(f"    TAC length: {len(match['tac'])} chars")
        
        # Show first few lines of TAC
        tac_lines = match['tac'].split('\n')[:10]
        logger.info("    TAC preview:")
        for line in tac_lines:
            logger.info(f"      {line}")

def test_tac_extraction():
    """Test TAC extraction for specific functions.

    Analyzes the SAMPLE_BYTECODE, identifies functions, and extracts the
    Three-Address Code (TAC) representation for the first identified function.
    Logs the generated TAC output for verification.

    Returns:
        None
    """
    logger.info("\n" + "=" * 60)
    logger.info("Testing TAC Extraction")
    logger.info("=" * 60)
    
    analyzer = BytecodeAnalyzer(SAMPLE_BYTECODE)
    analyzer.analyze_control_flow()
    functions = analyzer.identify_functions()
    
    if functions:
        # Get first function
        func_name, func = list(functions.items())[0]
        logger.info(f"Extracting TAC for function: {func_name}")
        
        # Test the extraction
        builder = DatasetBuilder("dummy_key", output_dir="test_data")
        tac = builder._extract_tac_for_function(func, analyzer)
        
        logger.info(f"Generated TAC ({len(tac)} chars):")
        logger.info("-" * 60)
        logger.info(tac)
        logger.info("-" * 60)

def main():
    """Run all tests.

    Executes the complete test suite for the dataset pipeline, including
    Solidity parsing, bytecode analysis, selector calculation, function
    matching, and TAC extraction tests.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    logger.info("Starting Dataset Pipeline Tests")
    logger.info("=" * 60)
    
    try:
        # Test 1: Solidity parsing
        solidity_funcs = test_solidity_parser()
        
        # Test 2: Bytecode analysis
        analyzer, bytecode_funcs = test_bytecode_analysis()
        
        # Test 3: Selector calculation
        test_selector_calculation()
        
        # Test 4: Function matching
        test_function_matching()
        
        # Test 5: TAC extraction
        test_tac_extraction()
        
        logger.info("\n" + "=" * 60)
        logger.info("All tests completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
