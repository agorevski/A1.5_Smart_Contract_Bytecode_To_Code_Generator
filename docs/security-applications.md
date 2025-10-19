# Security Applications

This document describes real-world security applications of smart contract decompilation.

## Overview

Decompilation enables security analysis of unverified contracts, vulnerability discovery, and post-mortem analysis of exploits.

## Use Case 1: Vulnerability Analysis

### Dx Protocol Vulnerability ($5.2M at Risk)

**Context**: Unverified contract with critical state management bug

**Discovery Process**:

1. Contract bytecode obtained from blockchain
2. Decompiled to readable Solidity
3. Identified flawed unlock mechanism
4. Confirmed vulnerability allowing repeated withdrawals

**Vulnerable Code** (decompiled):

```solidity
function unlockToken(uint256 _tokenId) external {
    require(tokenLocks[msg.sender][_tokenId].isLocked, "Token is already unlocked");
    require(tokenLocks[msg.sender][_tokenId].unlockTime > 0, "Token is not locked");
    
    // BUG: State only updated if time condition met
    if (block.timestamp > tokenLocks[msg.sender][_tokenId].unlockTime) {
        tokenLocks[msg.sender][_tokenId].isLocked = false;
    }
    
    // Transfer happens regardless - VULNERABLE!
    uint256 amount = tokenLocks[msg.sender][_tokenId].amount;
    IERC20(tokenLocks[msg.sender][_tokenId].token).transfer(msg.sender, amount);
}
```

**Impact**: Users could withdraw locked tokens before unlock time

## Use Case 2: MEV Bot Exploit Analysis

### MEV Bot Drain ($221,600 Loss)

**Context**: Proprietary MEV bot with unverified contracts exploited

**Analysis**:

1. Bot contract bytecode decompiled
2. Identified vulnerable callback functions
3. Found missing access controls
4. Discovered arbitrary external call vulnerability

**Vulnerable Callbacks** (decompiled):

```solidity
// Callback 1: Arbitrary external call
function swapX2YCallback(uint256 amountX, uint256, bytes calldata data) external {
    // VULNERABLE: No caller validation
    (bool success, ) = msg.sender.call{value: amountX}("");
    require(success, "...");
}

// Callback 2: Unprotected transfer
function d3MMSwapCallBack(address _to, uint256 _amount, bytes calldata) external {
    // VULNERABLE: Anyone can call and drain tokens
    IERC20(_to).transfer(msg.sender, _amount);
}
```

**Exploit Mechanism**:

1. Attacker deployed malicious contract
2. Called bot's callback functions directly
3. Drained tokens via unprotected transfers

## Use Case 3: Incident Response

### Real-Time Attack Analysis

**Scenario**: Active exploit detected on blockchain

**Response Workflow**:

```python
# 1. Detect suspicious transaction
tx_hash = "0x..."
attack_contract = get_contract_from_tx(tx_hash)

# 2. Quick decompilation
from src.bytecode_analyzer import analyze_bytecode_to_tac
from src.model_setup import SmartContractDecompiler

bytecode = get_bytecode(attack_contract)
tac = analyze_bytecode_to_tac(bytecode)

decompiler = SmartContractDecompiler("models/final/smart_contract_decompiler")
solidity = decompiler.decompile_tac_to_solidity(tac)

# 3. Analyze attack pattern
vulnerability_type = analyze_attack_pattern(solidity)

# 4. Identify similar vulnerable contracts
similar_contracts = find_similar_patterns(vulnerability_type)

# 5. Alert affected projects
alert_vulnerable_contracts(similar_contracts)
```

**Timeline**:

- **Detection**: T+0 minutes
- **Decompilation**: T+2 minutes
- **Analysis**: T+10 minutes
- **Alerts**: T+15 minutes

## Common Vulnerability Patterns

### 1. Reentrancy Attacks

**Vulnerable Pattern**:

```solidity
function withdraw(uint256 amount) external {
    require(balances[msg.sender] >= amount);
    
    // VULNERABLE: External call before state update
    (bool success, ) = msg.sender.call{value: amount}("");
    require(success);
    
    balances[msg.sender] -= amount;  // Too late!
}
```

**Detection via Decompilation**:

- Identify external calls
- Check state update ordering
- Verify reentrancy guards

### 2. Access Control Issues

**Vulnerable Pattern**:

```solidity
function setOwner(address newOwner) external {
    // VULNERABLE: Missing access control
    owner = newOwner;
}

function withdrawAll() external {
    // VULNERABLE: Anyone can call
    payable(owner).transfer(address(this).balance);
}
```

**Detection**:

- Look for missing `onlyOwner` modifiers
- Check `msg.sender` validations
- Verify role-based access control

### 3. Integer Overflow/Underflow

**Vulnerable Pattern** (pre-Solidity 0.8):

```solidity
function transfer(address to, uint256 amount) external {
    // VULNERABLE: No overflow check
    balances[msg.sender] -= amount;
    balances[to] += amount;
}
```

**Detection**:

- Check for SafeMath usage
- Verify compiler version (< 0.8.0)
- Look for unchecked arithmetic

### 4. Unprotected Selfdestruct

**Vulnerable Pattern**:

```solidity
function kill() external {
    // VULNERABLE: Anyone can destroy contract
    selfdestruct(payable(msg.sender));
}
```

**Detection**:

- Identify `selfdestruct` calls
- Check access controls
- Verify authorization logic

## Security Analysis Workflow

### Automated Analysis

```python
from src.security_analyzer import SecurityAnalyzer

analyzer = SecurityAnalyzer()

# Decompile contract
solidity = decompiler.decompile_tac_to_solidity(tac)

# Run security checks
report = analyzer.analyze(solidity)

print(f"Vulnerabilities found: {len(report.vulnerabilities)}")
for vuln in report.vulnerabilities:
    print(f"  [{vuln.severity}] {vuln.type}: {vuln.description}")
```

**Output Example**:

```text
Vulnerabilities found: 3
  [HIGH] Reentrancy: External call before state update in withdraw()
  [MEDIUM] Access Control: Missing owner check in setAdmin()
  [LOW] Unchecked Return: Call return value not checked
```

### Manual Review Checklist

- [ ] Access control on critical functions
- [ ] Reentrancy protection on external calls
- [ ] Input validation and bounds checking
- [ ] Integer overflow protection
- [ ] Proper use of `require`/`assert`
- [ ] Safe external contract interactions
- [ ] Event emission for state changes
- [ ] Upgradability considerations

## Case Studies

### Flash Loan Attack Analysis

**Contract**: DeFi protocol with price oracle manipulation

**Decompilation revealed**:

```solidity
function liquidate(address user) external {
    uint256 price = oracle.getPrice();  // VULNERABLE: Manipulable
    
    if (collateral[user] * price < debt[user]) {
        // Liquidation logic
    }
}
```

**Vulnerability**: Oracle could be manipulated via flash loans

**Fix**:

- Use time-weighted average prices (TWAP)
- Implement price deviation checks
- Add liquidation delays

### Proxy Pattern Exploitation

**Contract**: Upgradeable contract with storage collision

**Decompilation revealed**:

```solidity
// Proxy storage
address implementation;  // slot 0
address admin;          // slot 1

// Implementation storage
address token;          // slot 0 - COLLISION!
uint256 balance;        // slot 1 - COLLISION!
```

**Vulnerability**: Storage layout mismatch between proxy and implementation

## Prevention and Mitigation

### Best Practices

1. **Verify Contracts**: Always verify source code on Etherscan
2. **Security Audits**: Professional audits before mainnet deployment
3. **Testing**: Comprehensive test coverage including edge cases
4. **Access Controls**: Proper role-based permissions
5. **Monitoring**: Real-time transaction monitoring
6. **Upgradability**: Consider upgrade mechanisms carefully

### Decompilation in Security Workflow

```text
┌─────────────────────────────────────────────────┐
│         Smart Contract Security Workflow        │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. Contract Deployed (Unverified)              │
│     ↓                                           │
│  2. Bytecode Retrieval                          │
│     ↓                                           │
│  3. Decompilation to Solidity                   │
│     ↓                                           │
│  4. Automated Vulnerability Scanning            │
│     ↓                                           │
│  5. Manual Security Review                      │
│     ↓                                           │
│  6. Risk Assessment Report                      │
│     ↓                                           │
│  7. Mitigation Recommendations                  │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Tools Integration

### With Static Analyzers

```python
# Decompile and analyze
solidity = decompiler.decompile(bytecode)

# Run Slither
os.system(f"slither --solc-disable-warnings {solidity_file}")

# Run Mythril
os.system(f"myth analyze {solidity_file}")
```

### With Monitoring Systems

```python
# Monitor new contracts
def monitor_new_contracts():
    for tx in get_new_transactions():
        if is_contract_creation(tx):
            bytecode = get_contract_bytecode(tx.contract_address)
            solidity = decompile(bytecode)
            
            # Check for known vulnerability patterns
            if has_vulnerabilities(solidity):
                alert_security_team(tx.contract_address, solidity)
```

## Resources

- [Smart Contract Security Best Practices](https://consensys.github.io/smart-contract-best-practices/)
- [SWC Registry](https://swcregistry.io/) - Smart Contract Weakness Classification
- [Rekt News](https://rekt.news/) - DeFi hack post-mortems

## Next Steps

- Review [Evaluation](evaluation.md) for quality metrics
- Check [Comparisons](comparisons.md) for decompiler benchmarks
- See [Usage Guide](usage.md) for practical examples
