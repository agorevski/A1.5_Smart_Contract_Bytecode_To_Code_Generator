import sqlite3

# Get bytecode from database
conn = sqlite3.connect('data/contracts.db')
cursor = conn.cursor()
cursor.execute('SELECT bytecode FROM contracts')
bytecode = cursor.fetchone()[0]
conn.close()

print(f"Raw bytecode (first 100 chars): {bytecode[:100]}")
print(f"Starts with 0x: {bytecode.startswith('0x')}")
print(f"Total length: {len(bytecode)}")

# Try to parse manually
if bytecode.startswith('0x'):
    clean_bytecode = bytecode[2:]
else:
    clean_bytecode = bytecode

print(f"\nCleaned bytecode (first 100 chars): {clean_bytecode[:100]}")

# Check first few bytes
first_bytes = bytes.fromhex(clean_bytecode[:20])
print(f"\nFirst 10 bytes as hex: {first_bytes.hex()}")
print(f"First 10 bytes as integers: {[b for b in first_bytes]}")

# Common EVM opcodes
print("\nDecoding first few opcodes:")
print(f"  Byte 0: 0x{first_bytes[0]:02x} = {first_bytes[0]:3d} (should be PUSH1=0x60 or similar)")
print(f"  Byte 1: 0x{first_bytes[1]:02x} = {first_bytes[1]:3d}")
print(f"  Byte 2: 0x{first_bytes[2]:02x} = {first_bytes[2]:3d}")
print(f"  Byte 3: 0x{first_bytes[3]:02x} = {first_bytes[3]:3d}")

# Standard contract preamble is usually: 60 80 60 40 (PUSH1 0x80, PUSH1 0x40)
print("\nExpected start: 60 80 60 40 52 34... (PUSH1 0x80, PUSH1 0x40, MSTORE, CALLVALUE...)")
