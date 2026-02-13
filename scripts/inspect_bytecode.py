import sys
sys.path.append('src')

from src.bytecode_analyzer import BytecodeAnalyzer
import sqlite3

# Get bytecode from database
conn = sqlite3.connect('data/contracts.db')
cursor = conn.cursor()
cursor.execute('SELECT bytecode FROM contracts')
bytecode = cursor.fetchone()[0]
conn.close()

print(f"Bytecode length: {len(bytecode)}")
print(f"First 500 chars: {bytecode[:500]}\n")

# Analyze with detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

analyzer = BytecodeAnalyzer(bytecode)
print(f"\nTotal instructions: {len(analyzer.instructions)}")

# Show first 50 instructions
print("\nFirst 50 instructions:")
for i, instr in enumerate(analyzer.instructions[:50]):
    name = analyzer._get_instruction_name(instr)
    pc = analyzer._get_pc(instr, i)
    operand = analyzer._get_operand(instr)
    
    if operand:
        print(f"{i:4d}: PC={pc:5d} {name:15s} {operand}")
    else:
        print(f"{i:4d}: PC={pc:5d} {name:15s}")

# Look specifically for PUSH4 instructions
print("\n\nPUSH4 instructions (potential function selectors):")
for i, instr in enumerate(analyzer.instructions):
    name = analyzer._get_instruction_name(instr)
    if name == 'PUSH4':
        operand = analyzer._get_operand(instr)
        pc = analyzer._get_pc(instr, i)
        print(f"  Index {i}: PC={pc}, Operand={operand}")
        
        # Show next few instructions
        print(f"    Following instructions:")
        for j in range(i+1, min(i+6, len(analyzer.instructions))):
            next_instr = analyzer.instructions[j]
            next_name = analyzer._get_instruction_name(next_instr)
            next_operand = analyzer._get_operand(next_instr)
            if next_operand:
                print(f"      {next_name} {next_operand}")
            else:
                print(f"      {next_name}")

# Count JUMPDEST instructions
jumpdest_count = sum(1 for instr in analyzer.instructions 
                     if analyzer._get_instruction_name(instr) == 'JUMPDEST')
print(f"\n\nTotal JUMPDEST instructions: {jumpdest_count}")
