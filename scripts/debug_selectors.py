import sys
import logging
sys.path.append('src')

from src.dataset_pipeline import SolidityParser, DatasetBuilder
from src.bytecode_analyzer import BytecodeAnalyzer
import sqlite3
import yaml
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)

# Get data from database
conn = sqlite3.connect('data/contracts.db')
cursor = conn.cursor()
cursor.execute('SELECT address, source_code, bytecode FROM contracts')
address, source, bytecode = cursor.fetchone()
conn.close()

print(f"Contract: {address}")
print(f"Source length: {len(source)}, Bytecode length: {len(bytecode)}\n")

# Parse Solidity functions
parser = SolidityParser()
funcs = parser.extract_functions(source)
print(f"Solidity functions found: {len(funcs)}")

# Get API key and create builder to calculate selectors
settings_path = Path('src/settings.yaml')
with open(settings_path, 'r') as f:
    settings = yaml.safe_load(f)
    api_key = settings.get('ETHERSCAN_API_KEY')

builder = DatasetBuilder(api_key)
funcs_with_selectors = builder._add_selectors_to_solidity_functions(funcs)

print("\nSolidity functions with selectors:")
for f in funcs_with_selectors:
    print(f"  {f['name']:20} - Selector: {f.get('selector', 'NONE'):10} - Sig: {f['signature']}")

# Analyze bytecode
print("\nAnalyzing bytecode...")
analyzer = BytecodeAnalyzer(bytecode)
analyzer.analyze_control_flow()
bytecode_funcs = analyzer.identify_functions()

print(f"\nBytecode functions found: {len(bytecode_funcs)}")
for name, func in list(bytecode_funcs.items())[:10]:
    print(f"  {name:20} - Selector: {func.selector if func.selector else 'NONE'}")

# Check for matches
print("\nMatching functions:")
solidity_selectors = {f['selector'] for f in funcs_with_selectors if f.get('selector')}
bytecode_selectors = {f.selector for f in bytecode_funcs.values() if f.selector}

print(f"Solidity selectors: {solidity_selectors}")
print(f"Bytecode selectors: {bytecode_selectors}")
print(f"Matching selectors: {solidity_selectors & bytecode_selectors}")
