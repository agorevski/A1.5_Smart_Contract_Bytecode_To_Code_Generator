"""
Local Solidity Compiler Integration

Uses py-solc-x to compile Solidity source code locally, producing runtime
bytecode for TAC generation. The training-data path should compile each
verified source with a single source-aligned compiler configuration instead of
expanding one target body across synthetic compiler-version variants.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import solcx
from solcx.exceptions import SolcError

logger = logging.getLogger(__name__)


@dataclass
class CompilationResult:
    """Result of compiling a Solidity source file."""

    compiler_version: str
    optimizer_enabled: bool
    optimizer_runs: int
    contracts: Dict[str, "CompiledContract"] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    success: bool = False


@dataclass
class CompiledContract:
    """A single compiled contract from a compilation unit."""

    name: str
    runtime_bytecode: str  # hex string, no 0x prefix
    creation_bytecode: str
    abi: list
    source_file: str = ""
    effective_functions: list = field(default_factory=list)
    label_resolution_error: str = ""


def resolve_effective_functions(output: dict, sources: Dict[str, str],
                                source_file: str, contract_name: str) -> list:
    """Resolve public implementations using solc identities and C3 linearization.

    Source ranges are byte offsets, not Python character offsets. Missing AST
    identity is an error: guessing from a selector can label unrelated code.
    """
    contracts = {}
    source_ids = {}
    for path, info in output.get("sources", {}).items():
        source_ids[info.get("id")] = path
        for node in info.get("ast", {}).get("nodes", []):
            if node.get("nodeType") == "ContractDefinition":
                contracts[node["id"]] = (path, node)
    targets = [node for path, node in contracts.values()
               if path == source_file and node.get("name") == contract_name]
    if len(targets) != 1:
        raise ValueError("missing or ambiguous solc contract AST identity")
    lineage = targets[0].get("linearizedBaseContracts")
    if not lineage or lineage[0] != targets[0]["id"]:
        raise ValueError("missing solc linearized inheritance")
    methods = output.get("contracts", {}).get(source_file, {}).get(
        contract_name, {}).get("evm", {}).get("methodIdentifiers", {})

    def source_slice(node):
        start, length, file_id = map(int, node["src"].split(":"))
        path = source_ids.get(file_id)
        if path not in sources or start < 0 or length < 0:
            raise ValueError("unresolved solc source range")
        encoded = sources[path].encode("utf-8")
        if start + length > len(encoded):
            raise ValueError("solc source range exceeds source")
        return encoded[start:start + length].decode("utf-8"), path

    effective = {}
    for contract_id in lineage:
        if contract_id not in contracts:
            raise ValueError("unresolved solc base contract identity")
        _, contract = contracts[contract_id]
        local_selectors = set()
        for node in contract.get("nodes", []):
            if node.get("nodeType") not in ("FunctionDefinition", "VariableDeclaration"):
                continue
            if node.get("visibility") not in ("public", "external"):
                continue
            selector = node.get("functionSelector")
            if not selector and node.get("nodeType") == "FunctionDefinition":
                types = []
                for param in node.get("parameters", {}).get("parameters", []):
                    value = param.get("typeDescriptions", {}).get("typeString", "")
                    value = re.sub(r" (memory|storage|calldata)( ref| pointer)?$", "", value)
                    value = value.replace("address payable", "address")
                    if value.startswith("contract "):
                        value = "address"
                    types.append(value)
                selector = methods.get(f"{node.get('name')}({','.join(types)})")
            if not selector:
                # Constructors/fallbacks have no selector. Unknown callable
                # types must not allow a base implementation to win instead.
                if (node.get("kind") in ("constructor", "fallback", "receive") or
                        node.get("isConstructor") or not node.get("name")):
                    continue
                raise ValueError("unresolved public AST selector")
            selector = "0x" + selector.removeprefix("0x").lower()
            if selector in local_selectors:
                raise ValueError("ambiguous selector in contract AST")
            local_selectors.add(selector)
            if selector in effective:
                continue
            effective[selector] = None
            if node.get("nodeType") != "FunctionDefinition" or not node.get("body"):
                continue
            text, path = source_slice(node)
            source_slice(node["body"])
            # Signature ends at the body byte offset, not the first '{'
            # (which may occur in comments or strings).
            start = int(node["src"].split(":")[0])
            body_start = int(node["body"]["src"].split(":")[0])
            signature = text.encode("utf-8")[:body_start - start].decode("utf-8").strip()
            effective[selector] = {
                "name": node["name"], "selector": selector, "body": text,
                "signature": signature, "visibility": node["visibility"],
                "is_payable": node.get("stateMutability") == "payable",
                "is_view": node.get("stateMutability") in ("view", "pure"),
                "contract_name": contract["name"], "source_file": path,
                "ast_id": node["id"], "declaring_contract_id": contract_id,
                "compiled_contract_id": targets[0]["id"],
            }
    return [function for function in effective.values() if function is not None]


def _compiled_contracts(output: dict, sources: Dict[str, str]) -> Dict[str, CompiledContract]:
    result = {}
    names = {}
    for file_contracts in output.get("contracts", {}).values():
        for name in file_contracts:
            names[name] = names.get(name, 0) + 1
    for path, file_contracts in output.get("contracts", {}).items():
        for name, data in file_contracts.items():
            evm = data.get("evm", {})
            runtime = evm.get("deployedBytecode", {}).get("object", "")
            if not runtime:
                continue
            compiled = CompiledContract(
                name, runtime, evm.get("bytecode", {}).get("object", ""),
                data.get("abi", []), path,
            )
            try:
                compiled.effective_functions = resolve_effective_functions(output, sources, path, name)
            except (ValueError, KeyError, TypeError, UnicodeError) as exc:
                compiled.label_resolution_error = str(exc)
            result[name if names[name] == 1 else f"{path}:{name}"] = compiled
    return result


def get_installed_versions() -> List[str]:
    """Return list of installed solc version strings."""
    return [str(v) for v in solcx.get_installed_solc_versions()]


def install_solc_version(version: str) -> bool:
    """Install a specific solc version if not already installed.

    Args:
        version: Version string like '0.8.20' (no 'v' prefix).

    Returns:
        True if version is available (installed or already present).
    """
    # Normalize: strip leading 'v' and any commit suffix
    version = _normalize_version(version)
    if not version:
        return False

    installed = get_installed_versions()
    if version in installed:
        return True

    try:
        logger.info(f"Installing solc {version}...")
        solcx.install_solc(version)
        logger.info(f"Installed solc {version}")
        return True
    except Exception as e:
        logger.warning(f"Failed to install solc {version}: {e}")
        return False


def strip_solidity_comments(source_code: str) -> str:
    """Remove Solidity comments while preserving string literals."""
    result: List[str] = []
    i = 0
    in_string: Optional[str] = None
    escaped = False

    while i < len(source_code):
        ch = source_code[i]
        nxt = source_code[i + 1] if i + 1 < len(source_code) else ""

        if in_string:
            result.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_string:
                in_string = None
            i += 1
            continue

        if ch in ("'", '"'):
            in_string = ch
            result.append(ch)
            i += 1
            continue

        if ch == "/" and nxt == "/":
            i += 2
            while i < len(source_code) and source_code[i] != "\n":
                i += 1
            if i < len(source_code):
                result.append("\n")
                i += 1
            continue

        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < len(source_code) and not (
                source_code[i] == "*" and source_code[i + 1] == "/"
            ):
                result.append("\n" if source_code[i] == "\n" else " ")
                i += 1
            i = i + 2 if i + 1 < len(source_code) else len(source_code)
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def parse_pragma(source_code: str) -> List[str]:
    """Extract pragma solidity version constraints from source code.

    Args:
        source_code: Solidity source code.

    Returns:
        List of version constraint strings, e.g. ['^0.8.0', '>=0.7.0 <0.9.0'].
    """
    source_code = strip_solidity_comments(source_code)
    pragmas = []
    for match in re.finditer(
        r"^\s*pragma\s+solidity\s+([^;]+);", source_code, re.MULTILINE
    ):
        constraint = match.group(1).strip()
        pragmas.append(constraint)
    return pragmas


def version_satisfies_all_pragmas(version: str, pragma_constraints: List[str]) -> bool:
    """Return True when *version* satisfies every pragma constraint."""
    constraints = pragma_constraints or [">=0.4.0"]
    return all(_version_matches_pragma(version, pragma) for pragma in constraints)


def compatible_versions_for_pragmas(
    pragma_constraints: List[str],
    candidate_versions: Optional[List[str]] = None,
) -> List[str]:
    """Determine solc versions satisfying the intersection of all pragmas."""
    constraints = pragma_constraints or [">=0.4.0"]
    compatible = compatible_versions_for_pragma(constraints[0], candidate_versions)
    if len(constraints) == 1:
        return compatible
    return [
        version
        for version in compatible
        if version_satisfies_all_pragmas(version, constraints[1:])
    ]


def compatible_versions_for_pragma(
    pragma_constraint: str,
    candidate_versions: Optional[List[str]] = None,
) -> List[str]:
    """Determine which solc versions satisfy a pragma constraint.

    Supports ^, >=, <=, >, <, = operators and ranges.

    Args:
        pragma_constraint: e.g. '^0.8.0', '>=0.7.0 <0.9.0'
        candidate_versions: Versions to check. Defaults to a curated set.

    Returns:
        List of compatible version strings, sorted descending.
    """
    if candidate_versions is None:
        # Curated set: the 5 most commonly deployed solc versions on Ethereum.
        # These span the major codegen eras so trained models generalise to
        # the vast majority of on-chain bytecode.
        candidate_versions = [
            # 0.8.26 – latest widely-adopted release (2024+); modern
            #           optimiser, latest ABI encoding, PUSH0 opcode.
            "0.8.26",
            # 0.8.20 – most popular single version by deployment count
            #           (2023-2024); first version with PUSH0 support,
            #           mature Yul/IR pipeline.
            "0.8.20",
            # 0.8.10 – early 0.8.x workhorse; built-in overflow checks,
            #           widely used by major DeFi protocols (Uniswap V3,
            #           Aave V3, etc.).
            "0.8.10",
            # 0.6.12 – dominant pre-0.8 version; last stable 0.6.x,
            #           huge legacy footprint (SafeMath era, pre-built-in
            #           overflow). Different ABI encoder defaults.
            "0.6.12",
            # 0.5.17 – last 0.5.x release; significant legacy contracts
            #           (early DeFi, MakerDAO). No receive/fallback split,
            #           pre-ABIEncoderV2 default, different codegen.
            "0.5.17",
        ]

    compatible = []
    for ver in candidate_versions:
        if _version_matches_pragma(ver, pragma_constraint):
            compatible.append(ver)

    return compatible


def _version_matches_pragma(version: str, pragma: str) -> bool:
    """Check if a version satisfies a pragma constraint.

    Args:
        version: Version string like '0.8.20'.
        pragma: Pragma constraint like '^0.8.0' or '>=0.7.0 <0.9.0'.

    Returns:
        True if version satisfies the constraint.
    """
    parts = _parse_version(version)
    if not parts:
        return False

    # Solidity pragmas can combine AND constraints within a clause and OR
    # disjunctions across clauses, e.g. ">=0.5.0 <0.7.0 || ^0.8.0".
    for clause in re.split(r"\s*\|\|\s*", pragma):
        clause = clause.strip()
        if not clause:
            continue

        constraints = re.findall(r"([><=^~!]*\s*\d+\.\d+\.\d+)", clause)
        if not constraints:
            constraints = [clause]

        if all(_single_constraint_matches(parts, c.strip()) for c in constraints):
            return True

    return False


def _single_constraint_matches(
    version_parts: Tuple[int, int, int], constraint: str
) -> bool:
    """Check a single constraint like '^0.8.0' or '>=0.7.0'."""
    # Extract operator and version from constraint
    match = re.match(r"([><=^~!]*)(\d+\.\d+\.\d+)", constraint)
    if not match:
        return False

    op = match.group(1).strip()
    target = _parse_version(match.group(2))
    if not target:
        return False

    major, minor, patch = version_parts
    t_major, t_minor, t_patch = target

    if op in ("", "=", "=="):
        return version_parts == target
    elif op == "^":
        # ^0.8.0 means >=0.8.0 and <0.9.0 (for 0.x, <0.(x+1).0)
        if t_major == 0:
            return (
                major == t_major
                and minor == t_minor
                and patch >= t_patch
            )
        else:
            return major == t_major and (minor, patch) >= (t_minor, t_patch)
    elif op == "~":
        # ~0.8.0 means >=0.8.0 and <0.8+1.0
        return (
            major == t_major
            and minor == t_minor
            and patch >= t_patch
        )
    elif op == ">=":
        return version_parts >= target
    elif op == ">":
        return version_parts > target
    elif op == "<=":
        return version_parts <= target
    elif op == "<":
        return version_parts < target
    elif op == "!=":
        return version_parts != target
    else:
        return False


def _parse_version(version_str: str) -> Optional[Tuple[int, int, int]]:
    """Parse '0.8.20' into (0, 8, 20)."""
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)
    if match:
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None


def _normalize_version(version_str: str) -> Optional[str]:
    """Normalize version string: 'v0.8.20+commit.abc' -> '0.8.20'."""
    if not version_str:
        return None
    # Strip leading 'v'
    version_str = version_str.lstrip("v")
    # Strip commit suffix
    match = re.match(r"(\d+\.\d+\.\d+)", version_str)
    return match.group(1) if match else None


def compile_source(
    source_code: str,
    solc_version: str,
    optimizer_enabled: bool = True,
    optimizer_runs: int = 200,
    allow_paths: Optional[List[str]] = None,
    remappings: Optional[List[str]] = None,
    source_filename: str = "contract.sol",
) -> CompilationResult:
    """Compile Solidity source code with a specific compiler version.

    Args:
        source_code: Solidity source code (single file or combined).
        solc_version: Compiler version to use, e.g. '0.8.20'.
        optimizer_enabled: Whether to enable the optimizer.
        optimizer_runs: Number of optimization runs.
        allow_paths: Additional allowed paths for imports.
        remappings: Import remappings.
        source_filename: Original source identity in solc output and AST ranges.

    Returns:
        CompilationResult with compiled contracts.
    """
    solc_version = _normalize_version(solc_version)
    if not solc_version:
        return CompilationResult(
            compiler_version="unknown",
            optimizer_enabled=optimizer_enabled,
            optimizer_runs=optimizer_runs,
            errors=["Invalid compiler version"],
        )

    # Ensure version is installed
    if not install_solc_version(solc_version):
        return CompilationResult(
            compiler_version=solc_version,
            optimizer_enabled=optimizer_enabled,
            optimizer_runs=optimizer_runs,
            errors=[f"Could not install solc {solc_version}"],
        )

    result = CompilationResult(
        compiler_version=solc_version,
        optimizer_enabled=optimizer_enabled,
        optimizer_runs=optimizer_runs,
    )

    try:
        # Prepare input JSON for solc standard JSON input
        input_json = {
            "language": "Solidity",
            "sources": {source_filename: {"content": source_code}},
            "settings": {
                "optimizer": {
                    "enabled": optimizer_enabled,
                    "runs": optimizer_runs,
                },
                "outputSelection": {
                    "*": {
                        "*": [
                            "abi",
                            "evm.bytecode.object",
                            "evm.deployedBytecode.object",
                            "evm.methodIdentifiers",
                        ],
                        "": ["ast"],
                    }
                },
            },
        }

        if remappings:
            input_json["settings"]["remappings"] = remappings

        # Compile using standard JSON
        output = solcx.compile_standard(
            input_json,
            solc_version=solc_version,
            allow_paths=allow_paths or ["."],
        )

        # Check for errors
        if "errors" in output:
            for err in output["errors"]:
                if err.get("severity") == "error":
                    result.errors.append(err.get("formattedMessage", str(err)))

        if result.errors:
            return result

        result.contracts = _compiled_contracts(output, {source_filename: source_code})

        result.success = bool(result.contracts)

    except SolcError as e:
        result.errors.append(f"Compilation error: {e}")
    except Exception as e:
        result.errors.append(f"Unexpected error: {e}")

    return result


def compile_multi_file(
    sources: Dict[str, str],
    solc_version: str,
    optimizer_enabled: bool = True,
    optimizer_runs: int = 200,
    remappings: Optional[List[str]] = None,
) -> CompilationResult:
    """Compile a multi-file Solidity project.

    Args:
        sources: Dict mapping file paths to source content.
                 e.g. {"contracts/Token.sol": "pragma solidity...", "interfaces/IERC20.sol": "..."}
        solc_version: Compiler version.
        optimizer_enabled: Whether to enable optimizer.
        optimizer_runs: Optimization runs.
        remappings: Import remappings.

    Returns:
        CompilationResult.
    """
    solc_version = _normalize_version(solc_version)
    if not solc_version:
        return CompilationResult(
            compiler_version="unknown",
            optimizer_enabled=optimizer_enabled,
            optimizer_runs=optimizer_runs,
            errors=["Invalid compiler version"],
        )

    if not install_solc_version(solc_version):
        return CompilationResult(
            compiler_version=solc_version,
            optimizer_enabled=optimizer_enabled,
            optimizer_runs=optimizer_runs,
            errors=[f"Could not install solc {solc_version}"],
        )

    result = CompilationResult(
        compiler_version=solc_version,
        optimizer_enabled=optimizer_enabled,
        optimizer_runs=optimizer_runs,
    )

    try:
        input_json = {
            "language": "Solidity",
            "sources": {
                path: {"content": content} for path, content in sources.items()
            },
            "settings": {
                "optimizer": {
                    "enabled": optimizer_enabled,
                    "runs": optimizer_runs,
                },
                "outputSelection": {
                    "*": {
                        "*": [
                            "abi",
                            "evm.bytecode.object",
                            "evm.deployedBytecode.object",
                            "evm.methodIdentifiers",
                        ],
                        "": ["ast"],
                    }
                },
            },
        }

        if remappings:
            input_json["settings"]["remappings"] = remappings

        output = solcx.compile_standard(
            input_json,
            solc_version=solc_version,
            allow_paths=["."],
        )

        if "errors" in output:
            for err in output["errors"]:
                if err.get("severity") == "error":
                    result.errors.append(err.get("formattedMessage", str(err)))

        if result.errors:
            return result

        result.contracts = _compiled_contracts(output, sources)

        result.success = bool(result.contracts)

    except SolcError as e:
        result.errors.append(f"Compilation error: {e}")
    except Exception as e:
        result.errors.append(f"Unexpected error: {e}")

    return result


def parse_etherscan_source(raw_source: str) -> Dict[str, str]:
    """Parse Etherscan's source code format into a dict of files.

    Etherscan returns source code in several formats:
    1. Plain Solidity (single file)
    2. JSON with 'sources' key (multi-file)
    3. Double-brace wrapped JSON {{...}} (multi-file)

    Args:
        raw_source: Raw source code string from Etherscan API.

    Returns:
        Dict mapping file paths to source content.
        Single-file contracts use key 'contract.sol'.
    """
    # Try JSON formats first
    if raw_source.startswith("{"):
        try:
            parsed = json.loads(raw_source)
            if isinstance(parsed, dict) and "sources" in parsed:
                return {
                    path: data["content"]
                    for path, data in parsed["sources"].items()
                    if "content" in data
                }
            if isinstance(parsed, dict) and "content" in parsed:
                return {"contract.sol": parsed["content"]}
        except json.JSONDecodeError:
            pass

        # Double-brace format
        if raw_source.startswith("{{"):
            try:
                inner = raw_source[1:-1]
                parsed = json.loads(inner)
                if isinstance(parsed, dict) and "sources" in parsed:
                    return {
                        path: data["content"]
                        for path, data in parsed["sources"].items()
                        if "content" in data
                    }
            except (json.JSONDecodeError, IndexError):
                pass

    # Plain Solidity
    return {"contract.sol": raw_source}


def select_compilation_configs(
    pragma_constraint: Union[str, List[str]],
    original_version: Optional[str] = None,
    original_optimizer: Optional[bool] = None,
    original_runs: Optional[int] = None,
    max_configs: int = 3,
) -> List[Dict]:
    """Select the single source-aligned compilation configuration.

    The old generator used this hook for multi-version data augmentation. That
    over-weights duplicate Solidity bodies and does not match the verified
    on-chain compiler setting, so generation now returns at most one config:
    the verified compiler/optimizer settings when they satisfy all pragmas, or
    one compatible fallback when verified compiler metadata is unavailable.

    Args:
        pragma_constraint: Pragma version constraint(s) from source.
        original_version: Original compiler version used (from Etherscan).
        original_optimizer: Original optimizer setting.
        original_runs: Original optimizer runs.
        max_configs: Retained for API compatibility; values above one no longer
            request synthetic compiler expansion.

    Returns:
        A zero- or one-element list of config dicts with keys: version,
        optimizer_enabled, optimizer_runs.
    """
    pragma_constraints = (
        list(pragma_constraint)
        if isinstance(pragma_constraint, (list, tuple))
        else [pragma_constraint]
    )
    if not pragma_constraints:
        pragma_constraints = [">=0.4.0"]

    def build_config(version: str) -> Dict:
        return {
            "version": version,
            "optimizer_enabled": (
                bool(original_optimizer) if original_optimizer is not None else True
            ),
            "optimizer_runs": original_runs if original_runs is not None else 200,
        }

    if original_version:
        norm_ver = _normalize_version(original_version)
        if norm_ver and version_satisfies_all_pragmas(norm_ver, pragma_constraints):
            return [build_config(norm_ver)]

    compatible = compatible_versions_for_pragmas(pragma_constraints)
    if not compatible:
        return []

    return [build_config(compatible[0])]
