"""
Local Solidity Compiler Integration

Uses py-solc-x to compile Solidity source code locally with multiple
compiler versions, producing runtime bytecode for TAC generation.
This eliminates the need to fetch bytecode from a live Ethereum node
and enables data augmentation through multi-version compilation.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def parse_pragma(source_code: str) -> List[str]:
    """Extract pragma solidity version constraints from source code.

    Args:
        source_code: Solidity source code.

    Returns:
        List of version constraint strings, e.g. ['^0.8.0', '>=0.7.0 <0.9.0'].
    """
    pragmas = []
    for match in re.finditer(
        r"pragma\s+solidity\s+([^;]+);", source_code, re.MULTILINE
    ):
        constraint = match.group(1).strip()
        pragmas.append(constraint)
    return pragmas


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
        # Curated set of common versions to try
        candidate_versions = [
            "0.8.28", "0.8.26", "0.8.24", "0.8.22", "0.8.20",
            "0.8.19", "0.8.17", "0.8.13", "0.8.10", "0.8.7", "0.8.4", "0.8.0",
            "0.7.6", "0.7.5", "0.7.0",
            "0.6.12", "0.6.6", "0.6.0",
            "0.5.17", "0.5.16", "0.5.0",
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

    # Split pragma into individual constraints
    # Handle compound constraints like '>=0.7.0 <0.9.0'
    constraints = re.findall(r"([><=^~!]*\s*\d+\.\d+\.\d+)", pragma)
    if not constraints:
        # Try simple version match
        constraints = [pragma.strip()]

    for constraint in constraints:
        constraint = constraint.strip()
        if not _single_constraint_matches(parts, constraint):
            return False

    return True


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
) -> CompilationResult:
    """Compile Solidity source code with a specific compiler version.

    Args:
        source_code: Solidity source code (single file or combined).
        solc_version: Compiler version to use, e.g. '0.8.20'.
        optimizer_enabled: Whether to enable the optimizer.
        optimizer_runs: Number of optimization runs.
        allow_paths: Additional allowed paths for imports.
        remappings: Import remappings.

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
            "sources": {"contract.sol": {"content": source_code}},
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
                        ]
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

        # Extract compiled contracts
        contracts_output = output.get("contracts", {})
        for source_file, file_contracts in contracts_output.items():
            for contract_name, contract_data in file_contracts.items():
                evm = contract_data.get("evm", {})
                runtime_bc = evm.get("deployedBytecode", {}).get("object", "")
                creation_bc = evm.get("bytecode", {}).get("object", "")
                abi = contract_data.get("abi", [])

                if runtime_bc:  # Only include contracts with bytecode
                    result.contracts[contract_name] = CompiledContract(
                        name=contract_name,
                        runtime_bytecode=runtime_bc,
                        creation_bytecode=creation_bc,
                        abi=abi,
                        source_file=source_file,
                    )

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
                        ]
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

        contracts_output = output.get("contracts", {})
        for source_file, file_contracts in contracts_output.items():
            for contract_name, contract_data in file_contracts.items():
                evm = contract_data.get("evm", {})
                runtime_bc = evm.get("deployedBytecode", {}).get("object", "")
                creation_bc = evm.get("bytecode", {}).get("object", "")
                abi = contract_data.get("abi", [])

                if runtime_bc:
                    result.contracts[contract_name] = CompiledContract(
                        name=contract_name,
                        runtime_bytecode=runtime_bc,
                        creation_bytecode=creation_bc,
                        abi=abi,
                        source_file=source_file,
                    )

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
    pragma_constraint: str,
    original_version: Optional[str] = None,
    original_optimizer: Optional[bool] = None,
    original_runs: Optional[int] = None,
    max_configs: int = 3,
) -> List[Dict]:
    """Select compilation configurations for data augmentation.

    Picks diverse compiler settings that are compatible with the pragma.
    Always includes the original settings if provided, plus variants.

    Args:
        pragma_constraint: Pragma version constraint from source.
        original_version: Original compiler version used (from Etherscan).
        original_optimizer: Original optimizer setting.
        original_runs: Original optimizer runs.
        max_configs: Maximum number of configs to return.

    Returns:
        List of config dicts with keys: version, optimizer_enabled, optimizer_runs.
    """
    configs = []

    # 1. Always include the original config if available
    if original_version:
        norm_ver = _normalize_version(original_version)
        if norm_ver:
            configs.append(
                {
                    "version": norm_ver,
                    "optimizer_enabled": original_optimizer if original_optimizer is not None else True,
                    "optimizer_runs": original_runs if original_runs is not None else 200,
                }
            )

    # 2. Find compatible versions for augmentation
    compatible = compatible_versions_for_pragma(pragma_constraint)

    # Deduplicate against original
    existing_versions = {c["version"] for c in configs}

    for ver in compatible:
        if len(configs) >= max_configs:
            break
        if ver in existing_versions:
            continue

        # Alternate optimizer settings for diversity
        opt_enabled = len(configs) % 2 == 0  # alternate on/off
        configs.append(
            {
                "version": ver,
                "optimizer_enabled": opt_enabled,
                "optimizer_runs": 200,
            }
        )
        existing_versions.add(ver)

    return configs[:max_configs]