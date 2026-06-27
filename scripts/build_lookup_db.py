#!/usr/bin/env python3
"""
Build Exhaustive TAC Lookup Database

Compiles every contract in the training database with EVERY compatible
solc version (not just the curated 5) and stores all TAC→Solidity
mappings in a normalized lookup database for exact-match inference.

This database is NOT used for training — it exists purely to accelerate
inference by skipping the LLM for functions whose TAC has been seen before.

Usage:
    python scripts/build_lookup_db.py                        # full build
    python scripts/build_lookup_db.py --workers 8            # parallelism
    python scripts/build_lookup_db.py --max-versions 0       # 0 = unlimited
    python scripts/build_lookup_db.py --source-db data/contracts.db
    python scripts/build_lookup_db.py --lookup-db data/tac_lookup.db
    python scripts/build_lookup_db.py --stats                # print stats only
"""

import argparse
import logging
import os
import re
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.bytecode_analyzer import BytecodeAnalyzer
from src.dataset_pipeline import SolidityParser
from src.local_compiler import (
    compile_source,
    compile_multi_file,
    install_solc_version,
    parse_etherscan_source,
    parse_pragma,
    _normalize_version,
)
from src.tac_lookup import TACLookupBuilder, hash_normalized_tac
from web3 import Web3

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FILE = "build_lookup_db.log"

NOISY_LOGGERS = (
    "evmdasm",
    "evmdasm.disassembler",
    "src.bytecode_analyzer",
    "httpx",
    "httpcore",
)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(),
        ],
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Get ALL installable solc versions
# ---------------------------------------------------------------------------

def get_all_solc_versions() -> List[str]:
    """Return all installable solc versions, sorted descending."""
    import solcx
    versions = solcx.get_installable_solc_versions()
    return [str(v) for v in sorted(versions, reverse=True)]

def versions_for_pragma(pragma: str, all_versions: List[str]) -> List[str]:
    """Filter installable versions by pragma constraint."""
    from src.local_compiler import _version_matches_pragma
    return [v for v in all_versions if _version_matches_pragma(v, pragma)]

# ---------------------------------------------------------------------------
# Selector computation
# ---------------------------------------------------------------------------

def _parse_solidity_param_types(params_str: str) -> List[str]:
    """Parse Solidity parameter types with balanced-parenthesis awareness."""
    if not params_str or not params_str.strip():
        return []

    params: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in params_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            params.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        params.append(''.join(current).strip())

    types: List[str] = []
    for param in params:
        param = param.strip()
        if not param:
            continue
        if param.startswith('('):
            d = 0
            end = 0
            for i, c in enumerate(param):
                if c == '(':
                    d += 1
                elif c == ')':
                    d -= 1
                    if d == 0:
                        end = i + 1
                        break
            while end < len(param) and param[end] in '[]0123456789':
                end += 1
            types.append(param[:end])
        else:
            parts = param.split()
            type_token = parts[0]
            if len(parts) > 1 and parts[1].startswith('['):
                type_token += parts[1]
            types.append(type_token)

    return types

def _extract_function_signature_parts(signature: str) -> Optional[Tuple[str, str]]:
    """Return (function_name, params_text) from a Solidity function signature."""
    match = re.search(r"\bfunction\s+(\w+)\s*\(", signature)
    if not match:
        return None

    open_pos = signature.find("(", match.start())
    if open_pos == -1:
        return None

    depth = 1
    for pos in range(open_pos + 1, len(signature)):
        ch = signature[pos]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return match.group(1), signature[open_pos + 1 : pos].strip()

    return None

def add_selectors(functions: List[Dict]) -> List[Dict]:
    """Compute 4-byte function selectors for each Solidity function."""
    for func in functions:
        try:
            signature = func["signature"]
            parsed = _extract_function_signature_parts(signature)
            if parsed:
                func_name, params_str = parsed
                if params_str:
                    param_types = _parse_solidity_param_types(params_str)
                    canonical = f"{func_name}({','.join(param_types)})"
                else:
                    canonical = f"{func_name}()"
            else:
                canonical = signature.replace("function ", "").strip()

            func["selector"] = "0x" + Web3.keccak(text=canonical)[:4].hex()
        except Exception:
            func["selector"] = None
    return functions

# ---------------------------------------------------------------------------
# Worker function — compile one (contract, version, optimizer) combo
# ---------------------------------------------------------------------------

def _compile_one(
    address: str,
    source_files: Dict[str, str],
    solidity_functions: List[Dict],
    solc_version: str,
    opt_enabled: bool,
    runs: int,
) -> List[Dict]:
    """Compile and extract TAC→Solidity pairs. Runs in a worker process."""
    logging.disable(logging.CRITICAL)
    try:
        if not install_solc_version(solc_version):
            return []

        try:
            if len(source_files) > 1:
                comp = compile_multi_file(source_files, solc_version, opt_enabled, runs)
            else:
                first_src = next(iter(source_files.values()))
                comp = compile_source(first_src, solc_version, opt_enabled, runs)
        except Exception:
            return []

        if not comp.success:
            return []

        pairs: List[Dict] = []
        for cname, compiled in comp.contracts.items():
            bytecode_hex = "0x" + compiled.runtime_bytecode
            if len(bytecode_hex) < 10:
                continue

            try:
                analyzer = BytecodeAnalyzer(bytecode_hex)
                analyzer.analyze_control_flow()
                bytecode_functions = analyzer.identify_functions()
                _ensure_tac_integrated(analyzer)
            except Exception:
                continue

            # Match by selector
            sol_by_sel = {
                f["selector"]: f for f in solidity_functions
                if f.get("selector") and f.get("contract_name", "") == cname
            }
            if not sol_by_sel:
                sol_by_sel = {
                    f["selector"]: f for f in solidity_functions
                    if f.get("selector")
                }

            bc_by_sel = {
                f.selector: f for f in bytecode_functions.values()
                if f.selector
            }

            for selector, sol_func in sol_by_sel.items():
                if selector not in bc_by_sel:
                    continue

                bc_func = bc_by_sel[selector]

                # Extract TAC
                try:
                    tac_lines: List[str] = []
                    tac_lines.append(f"function {bc_func.name}:")
                    if bc_func.selector:
                        tac_lines.append(f"  // Selector: {bc_func.selector}")
                    tac_lines.append(f"  // Entry block: {bc_func.entry_block}")

                    blocks = bc_func.basic_blocks or []
                    if not blocks and bc_func.entry_block in analyzer.basic_blocks:
                        visited: Set[str] = set()
                        block_list: list = []

                        def traverse(bid: str):
                            if bid in visited or bid not in analyzer.basic_blocks:
                                return
                            visited.add(bid)
                            block_list.append(analyzer.basic_blocks[bid])
                            for s in analyzer.basic_blocks[bid].successors:
                                traverse(s)

                        traverse(bc_func.entry_block)
                        blocks = block_list

                    for block in blocks:
                        tac_lines.append(f"  {block.id}:")
                        if block.predecessors:
                            tac_lines.append(
                                f"    // Predecessors: {', '.join(block.predecessors)}"
                            )
                        if block.successors:
                            tac_lines.append(
                                f"    // Successors: {', '.join(block.successors)}"
                            )
                        for instr in block.instructions:
                            tac_lines.append(
                                f"    {analyzer._format_tac_instruction(instr)}"
                            )
                        tac_lines.append("")

                    tac = "\n".join(tac_lines)
                except Exception:
                    continue

                if not tac or len(tac.strip()) < 10:
                    continue

                sol_body = sol_func["body"]
                if len(sol_body.strip()) < 5:
                    continue

                pairs.append({
                    "tac": tac,
                    "solidity_code": sol_body,
                    "function_name": sol_func["name"],
                    "selector": selector,
                    "function_signature": sol_func.get("signature", ""),
                    "visibility": sol_func.get("visibility", "public"),
                    "is_payable": sol_func.get("is_payable", False),
                    "is_view": sol_func.get("is_view", False),
                    "compiler_version": solc_version,
                    "optimizer_enabled": opt_enabled,
                    "optimizer_runs": runs,
                })

        return pairs
    except Exception:
        return []

def _ensure_tac_integrated(analyzer) -> None:
    """Populate analyzer basic blocks with TAC instructions once."""
    blocks = getattr(analyzer, "basic_blocks", {}) or {}
    if not blocks:
        return
    if any(getattr(block, "instructions", None) for block in blocks.values()):
        return
    if not any(
        (getattr(block, "metadata", {}) or {}).get("raw_instructions")
        for block in blocks.values()
    ):
        return
    converter = getattr(analyzer, "_convert_and_integrate_tac", None)
    if callable(converter):
        converter()

# ---------------------------------------------------------------------------
# Main build pipeline
# ---------------------------------------------------------------------------

def build_lookup_db(
    source_db: str = "data/contracts.db",
    lookup_db: str = "data/tac_lookup.db",
    max_versions: int = 0,
    workers: int = 0,
):
    """Build the exhaustive TAC lookup database.

    Args:
        source_db: Path to the training contracts database.
        lookup_db: Path to the output lookup database.
        max_versions: Max solc versions per contract (0 = unlimited).
        workers: Number of parallel workers (0 = auto-detect).
    """
    if workers <= 0:
        workers = os.cpu_count() or 4

    logger.info("=" * 70)
    logger.info("Building Exhaustive TAC Lookup Database")
    logger.info("=" * 70)
    logger.info(f"  Source DB:       {source_db}")
    logger.info(f"  Lookup DB:       {lookup_db}")
    logger.info(f"  Max versions:    {'unlimited' if max_versions == 0 else max_versions}")
    logger.info(f"  Workers:         {workers}")

    # Get all installable solc versions
    all_versions = get_all_solc_versions()
    logger.info(f"  Available solc:  {len(all_versions)} versions")
    logger.info(f"  Range:           {all_versions[-1]} → {all_versions[0]}")

    # Initialize lookup builder
    builder = TACLookupBuilder(lookup_db)

    # Load set of already-completed jobs for re-runnability
    completed_jobs = builder.get_completed_jobs()
    logger.info(f"  Already done:    {len(completed_jobs)} compile jobs")

    # Load contracts from source DB
    conn = sqlite3.connect(source_db)
    # Try both table names for compatibility
    cursor = conn.cursor()
    tables = [r[0] for r in cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    if "hf_contracts" in tables:
        table = "hf_contracts"
    elif "contracts" in tables:
        table = "contracts"
    else:
        conn.close()
        raise RuntimeError(
            f"No 'contracts' or 'hf_contracts' table found in {source_db}. "
            f"Available tables: {tables}"
        )

    rows = conn.execute(
        f"SELECT address, source_code, compiler_version, "
        f"optimization_runs FROM {table}"
    ).fetchall()
    conn.close()

    total_contracts = len(rows)
    logger.info(f"  Contracts:       {total_contracts}")

    # Phase 1: Prepare contracts (parse source, find compatible versions)
    logger.info("\n--- Phase 1: Preparing contracts ---")
    prepared: Dict[str, Dict] = {}
    parser = SolidityParser()

    for addr, source, orig_ver, runs in tqdm(rows, desc="Parsing source", unit="contract"):
        try:
            source_files = parse_etherscan_source(source)
            if not source_files:
                continue

            combined = "\n\n".join(source_files.values())
            pragmas = parse_pragma(combined)
            pragma = pragmas[0] if pragmas else ">=0.4.0"

            compatible = versions_for_pragma(pragma, all_versions)
            if not compatible:
                continue

            # Add original version if not in list
            if orig_ver:
                norm = _normalize_version(orig_ver)
                if norm and norm not in compatible:
                    compatible.insert(0, norm)

            if max_versions > 0:
                compatible = compatible[:max_versions]

            solidity_functions = parser.extract_functions(combined)
            if not solidity_functions:
                continue

            solidity_functions = add_selectors(solidity_functions)

            prepared[addr] = {
                "source_files": source_files,
                "compatible_versions": compatible,
                "solidity_functions": solidity_functions,
                "runs": runs or 200,
            }
        except Exception:
            continue

    logger.info(f"Prepared {len(prepared)} contracts with source + compatible versions")

    # Phase 2: Build compile jobs (skip already-completed ones)
    compile_jobs = []
    skipped_jobs = 0
    for addr, prep in prepared.items():
        for ver in prep["compatible_versions"]:
            for opt in (True, False):
                if (addr, ver, int(opt)) in completed_jobs:
                    skipped_jobs += 1
                    continue
                compile_jobs.append(
                    (addr, prep["source_files"], prep["solidity_functions"],
                     ver, opt, prep["runs"])
                )

    total_jobs = len(compile_jobs)
    logger.info(f"\n--- Phase 2: Compiling {total_jobs} jobs ({len(prepared)} contracts), skipped {skipped_jobs} already done ---")

    if total_jobs == 0:
        logger.warning("No compile jobs to run!")
        return

    # Phase 3: Compile in parallel and store results
    total_new_bodies = 0
    total_new_tacs = 0
    total_pairs_seen = 0
    errors = 0
    pair_buffer: List[Dict] = []
    done_buffer: List[tuple] = []  # (address, version, opt_enabled, status, pairs)
    FLUSH_SIZE = 500
    SUBMIT_BATCH = 5000

    def _flush_buffers():
        nonlocal total_new_bodies, total_new_tacs, pair_buffer, done_buffer
        if pair_buffer:
            nb, nt = builder.bulk_insert_pairs(pair_buffer)
            total_new_bodies += nb
            total_new_tacs += nt
            pair_buffer = []
        if done_buffer:
            builder.mark_jobs_done_batch(done_buffer)
            done_buffer = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        with tqdm(total=total_jobs, desc="Compiling", unit="job") as pbar:
            for batch_start in range(0, total_jobs, SUBMIT_BATCH):
                batch_jobs = compile_jobs[batch_start:batch_start + SUBMIT_BATCH]
                futures = {
                    executor.submit(_compile_one, *job): (job[0], job[3], job[4])
                    for job in batch_jobs
                }  # value = (address, version, opt_enabled)

                for fut in as_completed(futures):
                    addr, ver, opt = futures[fut]
                    try:
                        pairs = fut.result()
                        if pairs:
                            pair_buffer.extend(pairs)
                            total_pairs_seen += len(pairs)
                        done_buffer.append(
                            (addr, ver, int(opt),
                             "ok" if pairs else "no_pairs",
                             len(pairs) if pairs else 0)
                        )
                    except Exception:
                        errors += 1
                        done_buffer.append(
                            (addr, ver, int(opt), "error", 0)
                        )

                    if len(pair_buffer) >= FLUSH_SIZE or len(done_buffer) >= FLUSH_SIZE:
                        _flush_buffers()

                    pbar.set_postfix(
                        pairs=total_pairs_seen,
                        bodies=total_new_bodies,
                        tacs=total_new_tacs,
                        errors=errors,
                        refresh=False,
                    )
                    pbar.update(1)

    # Flush remaining
    _flush_buffers()

    # Print final stats
    stats = builder.stats()
    logger.info("")
    logger.info("=" * 70)
    logger.info("Build complete!")
    logger.info(f"  Contracts compiled:    {len(prepared)}")
    logger.info(f"  Compile jobs:          {total_jobs}")
    logger.info(f"  Pairs seen:            {total_pairs_seen}")
    logger.info(f"  New unique bodies:     {total_new_bodies}")
    logger.info(f"  New TAC hashes:        {total_new_tacs}")
    logger.info(f"  Total bodies in DB:    {stats['unique_bodies']}")
    logger.info(f"  Total TAC hashes:      {stats['tac_hashes']}")
    logger.info(f"  Avg TAC per body:      {stats['avg_tac_per_body']}")
    logger.info(f"  Errors:                {errors}")
    logger.info(f"  Lookup DB:             {lookup_db}")
    logger.info("=" * 70)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build exhaustive TAC lookup database for inference-time exact matching.",
    )
    default_source = os.path.join(PROJECT_ROOT, "data", "contracts.db")
    default_lookup = os.path.join(PROJECT_ROOT, "data", "tac_lookup.db")
    ap.add_argument("--source-db", default=default_source,
                    help=f"Training contracts database (default: {default_source})")
    ap.add_argument("--lookup-db", default=default_lookup,
                    help=f"Output lookup database (default: {default_lookup})")
    ap.add_argument("--max-versions", type=int, default=0,
                    help="Max solc versions per contract (0 = unlimited, default: 0)")
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel workers (0 = auto-detect)")
    ap.add_argument("--stats", action="store_true",
                    help="Print stats of existing lookup DB and exit")

    args = ap.parse_args()

    setup_logging()

    if args.stats:
        if not Path(args.lookup_db).exists():
            logger.error("Lookup DB not found: %s", args.lookup_db)
            return
        b = TACLookupBuilder(args.lookup_db)
        stats = b.stats()
        logger.info("Lookup DB stats:")
        for k, v in stats.items():
            logger.info(f"  {k}: {v}")
        return

    build_lookup_db(
        source_db=args.source_db,
        lookup_db=args.lookup_db,
        max_versions=args.max_versions,
        workers=args.workers,
    )

if __name__ == "__main__":
    main()