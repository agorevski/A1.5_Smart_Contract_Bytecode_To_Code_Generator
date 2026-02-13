import sys
import logging
sys.path.append('src')

from src.dataset_pipeline import SolidityParser
import sqlite3

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Get source code from database
conn = sqlite3.connect('data/contracts.db')
cursor = conn.cursor()
cursor.execute('SELECT source_code FROM contracts')
source = cursor.fetchone()[0]
conn.close()

print(f"Source code length: {len(source)}")
print(f"First 500 chars:\n{source[:500]}\n")

# Parse functions
parser = SolidityParser()
funcs = parser.extract_functions(source)

print(f"\nFound {len(funcs)} functions:")
for i, f in enumerate(funcs[:10], 1):
    print(f"{i}. {f['name']} ({f['visibility']}) - {f['signature']}")
    print(f"   Body length: {len(f['body'])} chars")
