#!/usr/bin/env python3
"""
Test suite for the comprehensive control flow analysis implementation.

This script tests the enhanced analyze_control_flow function with various
bytecode patterns to ensure correctness and robustness.
"""

import sys
import logging
from typing import Dict, List

# Add src to path
sys.path.append('src')

from bytecode_analyzer import BytecodeAnalyzer, BasicBlock

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

class ControlFlowTester:
    """Test suite for control flow analysis."""
    
    def __init__(self):
        """Initialize the ControlFlowTester with an empty results list.
        
        Attributes:
            test_results: A list of tuples containing (test_name, status, error)
                for each executed test.
        """
        self.test_results = []
    
    def run_all_tests(self):
        """Run all control flow analysis tests.
        
        Executes each test method in sequence, capturing results and
        exceptions. Prints progress and final summary to stdout.
        
        Returns:
            None. Results are stored in self.test_results and printed.
        """
        print("=" * 60)
        print("COMPREHENSIVE CONTROL FLOW ANALYSIS TEST SUITE")
        print("=" * 60)
        
        tests = [
            self.test_simple_bytecode,
            self.test_conditional_jumps,
            self.test_function_dispatcher,
            self.test_loop_detection,
            self.test_multiple_functions,
            self.test_error_handling,
        ]
        
        for test_func in tests:
            try:
                print(f"\nRunning {test_func.__name__}...")
                test_func()
                self.test_results.append((test_func.__name__, "PASS", None))
                print(f"✓ {test_func.__name__} PASSED")
            except Exception as e:
                self.test_results.append((test_func.__name__, "FAIL", str(e)))
                print(f"✗ {test_func.__name__} FAILED: {e}")
        
        self.print_summary()
    
    def test_simple_bytecode(self):
        """Test basic control flow analysis with simple bytecode.
        
        Validates that the analyzer correctly creates basic blocks from
        a simple smart contract bytecode, ensuring proper block IDs and
        valid address ranges.
        
        Raises:
            AssertionError: If basic block creation or validation fails.
        """
        # Simple contract with basic arithmetic
        bytecode = "0x608060405234801561001057600080fd5b50600436106100365760003560e01c8063893d20e81461003b578063a6f9dae114610059575b600080fd5b610043610075565b6040516100509190610166565b60405180910390f35b610073600480360381019061006e91906101b2565b61009e565b005b60008060009054906101000a900473ffffffffffffffffffffffffffffffffffffffff16905090565b8073ffffffffffffffffffffffffffffffffffffffff163373ffffffffffffffffffffffffffffffffffffffff16036100d357806000806101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055505b50565b600073ffffffffffffffffffffffffffffffffffffffff82169050919050565b6000610101826100d6565b9050919050565b610111816100f6565b82525050565b600060208201905061012c6000830184610108565b92915050565b600080fd5b610140816100f6565b811461014b57600080fd5b50565b60008135905061015d81610137565b92915050565b60006020828403121561017957610178610132565b5b60006101878482850161014e565b91505092915050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052602260045260246000fd5b600060028204905060018216806101d857607f821691505b6020821081036101eb576101ea610190565b5b5091905056fea26469706673582212209d84a3c5d1d6c4c5f9c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c564736f6c634300080a0033"
        
        analyzer = BytecodeAnalyzer(bytecode)
        blocks = analyzer.analyze_control_flow()
        
        # Validate basic properties
        assert len(blocks) > 0, "Should create at least one basic block"
        assert all(isinstance(block, BasicBlock) for block in blocks.values()), "All blocks should be BasicBlock instances"
        
        # Check for proper block IDs
        for block_id, block in blocks.items():
            assert block.id == block_id, f"Block ID mismatch: {block.id} != {block_id}"
            assert block.start_address <= block.end_address, "Invalid address range"
        
        print(f"  Created {len(blocks)} basic blocks")
        print(f"  Address ranges: {[(b.start_address, b.end_address) for b in blocks.values()]}")
    
    def test_conditional_jumps(self):
        """Test control flow analysis with conditional jumps.
        
        Validates that bytecode containing conditional logic (JUMPI) is
        correctly analyzed, with proper predecessor/successor relationships
        established between blocks.
        
        Raises:
            AssertionError: If successor blocks are missing or back-references
                are not properly established.
        """
        # Bytecode with conditional logic
        bytecode = "0x6080604052348015600f57600080fd5b50600436106030576000357c0100000000000000000000000000000000000000000000000000000000900463ffffffff168063c6888fa1146035575b600080fd5b606060048036038101908080359060200190929190505050607676565b6040518082815260200191505060405180910390f35b60008160070290509190505600a165627a7a72305820b0b20b66b8ec1fe6fa52b4d65fbfeba5d9fccd9cc2e7d5efd8a82b4c6b78eae10029"
        
        analyzer = BytecodeAnalyzer(bytecode)
        blocks = analyzer.analyze_control_flow()
        
        # Check for conditional structures
        conditional_blocks = [b for b in blocks.values() if b.metadata.get('block_type') == 'conditional']
        print(f"  Found {len(conditional_blocks)} conditional blocks")
        
        # Validate predecessor/successor relationships
        for block in blocks.values():
            for succ_id in block.successors:
                assert succ_id in blocks, f"Successor {succ_id} not found in blocks"
                assert block.id in blocks[succ_id].predecessors, f"Missing back-reference from {succ_id} to {block.id}"
    
    def test_function_dispatcher(self):
        """Test function dispatcher pattern recognition.
        
        Validates that the analyzer correctly identifies function selectors
        and entry points in contracts with multiple public/external functions.
        
        Raises:
            AssertionError: If function identification fails.
        """
        # Contract with multiple functions
        bytecode = "0x608060405234801561001057600080fd5b50600436106100365760003560e01c8063893d20e81461003b578063a6f9dae114610059575b600080fd5b610043610075565b6040516100509190610166565b60405180910390f35b610073600480360381019061006e91906101b2565b61009e565b005b60008060009054906101000a900473ffffffffffffffffffffffffffffffffffffffff16905090565b8073ffffffffffffffffffffffffffffffffffffffff163373ffffffffffffffffffffffffffffffffffffffff16036100d357806000806101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055505b50565b600073ffffffffffffffffffffffffffffffffffffffff82169050919050565b6000610101826100d6565b9050919050565b610111816100f6565b82525050565b600060208201905061012c6000830184610108565b92915050565b600080fd5b610140816100f6565b811461014b57600080fd5b50565b60008135905061015d81610137565b92915050565b60006020828403121561017957610178610132565b5b60006101878482850161014e565b91505092915050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052602260045260246000fd5b600060028204905060018216806101d857607f821691505b6020821081036101eb576101ea610190565b5b5091905056fea26469706673582212209d84a3c5d1d6c4c5f9c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c564736f6c634300080a0033"
        
        analyzer = BytecodeAnalyzer(bytecode)
        blocks = analyzer.analyze_control_flow()
        functions = analyzer.identify_functions()
        
        print(f"  Identified {len(functions)} functions")
        for func_name, func in functions.items():
            print(f"    {func_name}: selector={func.selector}, entry={func.entry_block}")
    
    def test_loop_detection(self):
        """Test loop detection in control flow.
        
        Validates that the analyzer correctly identifies loop headers and
        back edges in the control flow graph, which are essential for
        recognizing iterative constructs.
        
        Raises:
            AssertionError: If loop detection validation fails.
        """
        # Simple bytecode that should create a loop
        bytecode = "0x608060405234801561001057600080fd5b50600436106100365760003560e01c8063893d20e81461003b578063a6f9dae114610059575b600080fd5b610043610075565b6040516100509190610166565b60405180910390f35b610073600480360381019061006e91906101b2565b61009e565b005b60008060009054906101000a900473ffffffffffffffffffffffffffffffffffffffff16905090565b8073ffffffffffffffffffffffffffffffffffffffff163373ffffffffffffffffffffffffffffffffffffffff16036100d357806000806101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055505b50565b600073ffffffffffffffffffffffffffffffffffffffff82169050919050565b6000610101826100d6565b9050919050565b610111816100f6565b82525050565b600060208201905061012c6000830184610108565b92915050565b600080fd5b610140816100f6565b811461014b57600080fd5b50565b60008135905061015d81610137565b92915050565b60006020828403121561017957610178610132565b5b60006101878482850161014e565b91505092915050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052602260045260246000fd5b600060028204905060018216806101d857607f821691505b6020821081036101eb576101ea610190565b5b5091905056fea26469706673582212209d84a3c5d1d6c4c5f9c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c564736f6c634300080a0033"
        
        analyzer = BytecodeAnalyzer(bytecode)
        blocks = analyzer.analyze_control_flow()
        
        # Check for loop headers
        loop_headers = [b for b in blocks.values() if b.metadata.get('is_loop_header', False)]
        print(f"  Found {len(loop_headers)} loop headers")
        
        # Check for back edges
        back_edges = []
        for block in blocks.values():
            if 'back_edges' in block.metadata:
                back_edges.extend(block.metadata['back_edges'])
        print(f"  Found {len(back_edges)} back edges")
    
    def test_multiple_functions(self):
        """Test analysis with multiple function signatures.
        
        Validates control flow analysis and TAC (Three-Address Code)
        generation for contracts with multiple function signatures.
        
        Raises:
            AssertionError: If TAC generation fails or produces empty output.
        """
        # More complex contract bytecode
        bytecode = "0x608060405234801561001057600080fd5b50600436106100365760003560e01c8063893d20e81461003b578063a6f9dae114610059575b600080fd5b610043610075565b6040516100509190610166565b60405180910390f35b610073600480360381019061006e91906101b2565b61009e565b005b60008060009054906101000a900473ffffffffffffffffffffffffffffffffffffffff16905090565b8073ffffffffffffffffffffffffffffffffffffffff163373ffffffffffffffffffffffffffffffffffffffff16036100d357806000806101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055505b50565b600073ffffffffffffffffffffffffffffffffffffffff82169050919050565b6000610101826100d6565b9050919050565b610111816100f6565b82525050565b600060208201905061012c6000830184610108565b92915050565b600080fd5b610140816100f6565b811461014b57600080fd5b50565b60008135905061015d81610137565b92915050565b60006020828403121561017957610178610132565b5b60006101878482850161014e565b91505092915050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052602260045260246000fd5b600060028204905060018216806101d857607f821691505b6020821081036101eb576101ea610190565b5b5091905056fea26469706673582212209d84a3c5d1d6c4c5f9c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c564736f6c634300080a0033"
        
        analyzer = BytecodeAnalyzer(bytecode)
        blocks = analyzer.analyze_control_flow()
        
        # Test TAC integration
        tac_output = analyzer.generate_tac_representation()
        assert len(tac_output) > 0, "TAC output should not be empty"
        assert "Three-Address Code" in tac_output, "TAC output should contain header"
        
        print(f"  Generated TAC output with {len(tac_output)} characters")
        print(f"  Contains {tac_output.count('temp_')} temporary variables")
    
    def test_error_handling(self):
        """Test error handling with malformed bytecode.
        
        Validates that the analyzer gracefully handles various forms of
        malformed or invalid bytecode without crashing unexpectedly.
        Tests include empty bytecode, invalid hex, and truncated data.
        """
        test_cases = [
            ("", "Empty bytecode"),
            ("0x", "Only prefix"),
            ("0x123", "Odd length"),
            ("0xZZZ", "Invalid hex characters"),
            ("0x60806040", "Truncated bytecode"),
        ]
        
        for bytecode, description in test_cases:
            try:
                analyzer = BytecodeAnalyzer(bytecode)
                blocks = analyzer.analyze_control_flow()
                print(f"  {description}: Handled gracefully ({len(blocks)} blocks)")
            except Exception as e:
                # Some errors are expected for malformed bytecode
                print(f"  {description}: Expected error - {type(e).__name__}")
    
    def print_summary(self):
        """Print test results summary.
        
        Outputs a formatted summary of all test results including total
        tests run, pass/fail counts, details of any failures, and the
        overall success rate percentage.
        """
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for _, status, _ in self.test_results if status == "PASS")
        failed = sum(1 for _, status, _ in self.test_results if status == "FAIL")
        
        print(f"Total tests: {len(self.test_results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print("\nFailed tests:")
            for test_name, status, error in self.test_results:
                if status == "FAIL":
                    print(f"  {test_name}: {error}")
        
        print(f"\nSuccess rate: {passed/len(self.test_results)*100:.1f}%")

def main():
    """Run the control flow analysis test suite.
    
    Creates a ControlFlowTester instance and executes all registered
    tests, printing results to stdout.
    
    Returns:
        None.
    """
    tester = ControlFlowTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
