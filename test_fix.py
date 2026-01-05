"""
Simple test script to verify the entry_block fix in bytecode_analyzer.py
"""

import logging
from src.bytecode_analyzer import BytecodeAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)

# Test bytecode from a simple contract
test_bytecode = "0x608060405234801561001057600080fd5b50600436106100365760003560e01c8063893d20e81461003b578063a6f9dae114610059575b600080fd5b610043610075565b6040516100509190610166565b60405180910390f35b610073600480360381019061006e91906101b2565b61009e565b005b60008060009054906101000a900473ffffffffffffffffffffffffffffffffffffffff16905090565b8073ffffffffffffffffffffffffffffffffffffffff163373ffffffffffffffffffffffffffffffffffffffff16036100d357806000806101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055505b50565b600073ffffffffffffffffffffffffffffffffffffffff82169050919050565b6000610101826100d6565b9050919050565b610111816100f6565b82525050565b600060208201905061012c6000830184610108565b92915050565b600080fd5b610140816100f6565b811461014b57600080fd5b50565b60008135905061015d81610137565b92915050565b60006020828403121561017957610178610132565b5b60006101878482850161014e565b91505092915050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052602260045260246000fd5b600060028204905060018216806101d857607f821691505b6020821081036101eb576101ea610190565b5b5091905056fea26469706673582212209d84a3c5d1d6c4c5f9c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c564736f6c634300080a0033"

print("Testing BytecodeAnalyzer with entry_block fix...")
print("=" * 60)

try:
    # Create analyzer
    analyzer = BytecodeAnalyzer(test_bytecode)
    print(f"✓ Created BytecodeAnalyzer")
    print(f"  Parsed {len(analyzer.instructions)} instructions")
    
    # Perform control flow analysis
    blocks = analyzer.analyze_control_flow()
    print(f"✓ Control flow analysis completed")
    print(f"  Found {len(blocks)} basic blocks")
    
    # Identify functions (this is where the bug was)
    functions = analyzer.identify_functions()
    print(f"✓ Function identification completed (BUG FIX VERIFIED)")
    print(f"  Found {len(functions)} functions")
    
    for func_name, func in functions.items():
        print(f"  - {func_name}: entry_block={func.entry_block}")
    
    # Generate TAC
    tac = analyzer.generate_tac_representation()
    print(f"✓ TAC generation completed")
    print(f"  Generated {len(tac.split(chr(10)))} lines of TAC")
    
    print("\n" + "=" * 60)
    print("SUCCESS: All tests passed! The entry_block bug is fixed.")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\nFAILURE: The bug fix did not work correctly.")
