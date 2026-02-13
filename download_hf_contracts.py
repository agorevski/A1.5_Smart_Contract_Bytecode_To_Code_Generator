#!/usr/bin/env python3
"""
Download Smart Contracts from HuggingFace & Generate Training Data

Downloads verified Solidity contracts from the andstor/smart_contracts
dataset on HuggingFace, compiles each with every compatible solc version
(optimizer on + off), generates TAC via BytecodeAnalyzer, and exports
training pairs in {"input": "<TAC>", "output": "<Solidity>"} format.

Deduplication strategy (multi-layer):
  1. Contract-level: address PRIMARY KEY + source_hash dedup
  2. Pair-level exact: hash(TAC + body) UNIQUE in function_pairs
  3. Pair-level semantic: pair_norm_hash(normalized_TAC + normalized_body) UNIQUE
     -- catches identical bytecode from different compilers
  4. Body-level: normalized Solidity body hash with frequency cap
  5. TAC-level: normalized TAC hash for input-side frequency awareness
  6. Quality filters: min body length, skip trivial getters/proxies
  7. Export-time: ROW_NUMBER partitioned by pair_norm_hash

Usage:
    python download_hf_contracts.py                          # full pipeline
    python download_hf_contracts.py --download-only          # download only
    python download_hf_contracts.py --compile-only           # compile only
    python download_hf_contracts.py --limit 100              # limit downloads
    python download_hf_contracts.py --workers 8              # set parallelism
    python download_hf_contracts.py --max-compiler-versions 3
    python download_hf_contracts.py --max-body-dupes 5       # cap duplicates
    python download_hf_contracts.py --min-body-length 50     # quality filter
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
from web3 import Web3

from src.bytecode_analyzer import BytecodeAnalyzer
from src.dataset_pipeline import SolidityParser
from src.local_compiler import (
    compile_source,
    compile_multi_file,
    compatible_versions_for_pragma,
    install_solc_version,
    parse_etherscan_source,
    parse_pragma,
    _normalize_version,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_FILE = "download_hf_contracts.log"
DB_PATH = Path("data/contracts.db")
FLUSH_SIZE = 200
BATCH_INSERT_SIZE = 500

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

NOISY_LOGGERS = (
    "evmdasm",
    "evmdasm.disassembler",
    "src.bytecode_analyzer",
    "httpx",
    "httpcore",
)


def setup_logging():
    """Configure root logger with file + console handlers."""
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


# ---------------------------------------------------------------------------
# Normalization & quality helpers
# ---------------------------------------------------------------------------

BOILERPLATE_PATTERNS = [
    re.compile(rf"\b{name}\b")
    for name in (
        "SafeMath", "Ownable", "Context", "IERC20", "ERC20",
        "ERC721", "ERC1155", "ReentrancyGuard", "Pausable", "AccessControl",
    )
]

TRIVIAL_PATTERNS = [
    re.compile(r"^\s*\{\s*return\s+\w+\s*;\s*\}\s*$", re.DOTALL),
    re.compile(r"^\s*\{\s*return\s+\d+\s*;\s*\}\s*$", re.DOTALL),
    re.compile(r"^\s*\{\s*\}\s*$", re.DOTALL),
]

PROXY_PATTERN = re.compile(r"\bdelegatecall\b")


def _strip_comments(text: str) -> str:
    """Remove single-line and multi-line comments."""
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_solidity_body(body: str) -> str:
    """Normalize a Solidity function body for dedup hashing."""
    return _collapse_whitespace(_strip_comments(body)).lower()


def normalize_tac(tac: str) -> str:
    """Normalize TAC for dedup hashing.

    Strips comments (which may contain compiler version info),
    collapses whitespace, and lowercases so that identical bytecode
    compiled by different solc versions produces the same hash.
    """
    text = re.sub(r"//[^\n]*", "", tac)
    return _collapse_whitespace(text).lower()


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def hash_normalized_body(body: str) -> str:
    return _md5(normalize_solidity_body(body))


def hash_normalized_tac(tac: str) -> str:
    return _md5(normalize_tac(tac))


def hash_normalized_pair(tac: str, body: str) -> str:
    """Hash normalized (TAC + body) -- the ultimate semantic dedup key."""
    return _md5(normalize_tac(tac) + "|" + normalize_solidity_body(body))


def hash_source_code(source: str) -> str:
    """Hash source code for contract-level dedup (normalize whitespace)."""
    normalized = _collapse_whitespace(source)
    return hashlib.sha256(normalized.encode()).hexdigest()


def is_trivial_function(body: str) -> bool:
    """Check if a function body is too trivial to be useful training data."""
    return any(pat.match(body) for pat in TRIVIAL_PATTERNS)


def is_proxy_only(body: str) -> bool:
    """Check if a function is purely a proxy delegatecall forwarder."""
    if not PROXY_PATTERN.search(body):
        return False
    statements = [s.strip() for s in normalize_solidity_body(body).split(";") if s.strip()]
    return len(statements) <= 3


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

@contextmanager
def _db_connection(db_path: Path = DB_PATH):
    """Context manager for SQLite connections."""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def _add_columns(cur: sqlite3.Cursor, table: str, columns: List[tuple]):
    """Safely add columns to a table (ignores if already present)."""
    for col_name, col_type in columns:
        try:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass


def _ensure_indexes(cur: sqlite3.Cursor, indexes: List[tuple]):
    """Create indexes if they don't exist. Each tuple: (name, unique, sql)."""
    for name, unique, definition in indexes:
        keyword = "UNIQUE INDEX" if unique else "INDEX"
        try:
            cur.execute(f"CREATE {keyword} IF NOT EXISTS {name} ON {definition}")
        except sqlite3.OperationalError:
            pass


def init_database(db_path: Path = DB_PATH):
    """Create or migrate the contracts database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with _db_connection(db_path) as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS contracts (
                address TEXT PRIMARY KEY,
                source_code TEXT,
                bytecode TEXT,
                compiler_version TEXT,
                optimization_enabled BOOLEAN,
                optimization_runs INTEGER,
                processed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS function_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_address TEXT,
                function_name TEXT,
                tac_representation TEXT,
                solidity_code TEXT,
                function_signature TEXT,
                visibility TEXT,
                is_payable BOOLEAN,
                is_view BOOLEAN,
                metadata TEXT,
                hash TEXT UNIQUE,
                body_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (contract_address) REFERENCES contracts (address)
            )
        """)

        # Selector registry — maps 4-byte selectors to signatures
        cur.execute("""
            CREATE TABLE IF NOT EXISTS selector_registry (
                selector    TEXT NOT NULL,
                signature   TEXT NOT NULL,
                source      TEXT DEFAULT 'compiled',
                occurrences INTEGER DEFAULT 1,
                first_seen  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (selector, signature)
            )
        """)

        # Migration: add columns that may not exist yet
        _add_columns(cur, "contracts", [
            ("abi", "TEXT"),
            ("contract_name", "TEXT"),
            ("source", "TEXT DEFAULT 'etherscan'"),
            ("source_hash", "TEXT"),
        ])
        _add_columns(cur, "function_pairs", [
            ("body_hash", "TEXT"),
            ("tac_hash", "TEXT"),
            ("pair_norm_hash", "TEXT"),
        ])

        # Migration: ensure indexes
        _ensure_indexes(cur, [
            ("idx_body_hash",      False, "function_pairs(body_hash)"),
            ("idx_tac_hash",       False, "function_pairs(tac_hash)"),
            ("idx_pair_norm_hash", True,  "function_pairs(pair_norm_hash)"),
            ("idx_source_hash",    False, "contracts(source_hash)"),
            ("idx_sel_reg_sel",    False, "selector_registry(selector)"),
        ])

        conn.commit()

    # Seed the registry with built-in selectors
    _seed_builtin_selectors(db_path)


def _seed_builtin_selectors(db_path: Path = DB_PATH):
    """Insert the curated built-in selectors into selector_registry."""
    from src.selector_resolver import _BUILTIN_SELECTORS

    with _db_connection(db_path) as conn:
        cur = conn.cursor()
        for sel, sig in _BUILTIN_SELECTORS.items():
            cur.execute("""
                INSERT INTO selector_registry (selector, signature, source, occurrences)
                VALUES (?, ?, 'builtin', 1000)
                ON CONFLICT(selector, signature) DO UPDATE SET
                    source = CASE WHEN selector_registry.source = 'builtin'
                                  THEN 'builtin' ELSE selector_registry.source END
            """, (sel.lower(), sig))
        conn.commit()
        logger.info("Seeded %d built-in selectors into registry", len(_BUILTIN_SELECTORS))


def store_selectors_batch(
    selectors: List[tuple],
    source: str = "compiled",
    db_path: Path = DB_PATH,
):
    """Bulk-upsert selector→signature mappings into the registry.

    Each tuple: (selector_hex, canonical_signature)
    """
    if not selectors:
        return
    with _db_connection(db_path) as conn:
        cur = conn.cursor()
        cur.executemany("""
            INSERT INTO selector_registry (selector, signature, source, occurrences)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(selector, signature) DO UPDATE SET
                occurrences = selector_registry.occurrences + 1,
                last_seen = CURRENT_TIMESTAMP
        """, [(s.lower(), sig, source) for s, sig in selectors])
        conn.commit()


def export_selector_registry(
    output_path: str = "data/selectors.json",
    db_path: Path = DB_PATH,
) -> int:
    """Export the full selector registry as a JSON file."""
    with _db_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT selector, signature, source, occurrences "
            "FROM selector_registry ORDER BY occurrences DESC"
        ).fetchall()

    registry = {}
    for sel, sig, source, occ in rows:
        if sel not in registry:
            registry[sel] = []
        registry[sel].append({
            "signature": sig,
            "source": source,
            "occurrences": occ,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    logger.info("Exported %d unique selectors (%d mappings) to %s",
                len(registry), len(rows), output_path)
    return len(registry)


def import_selector_registry(
    input_path: str = "data/selectors.json",
    db_path: Path = DB_PATH,
) -> int:
    """Import selectors from a JSON file into the registry."""
    if not os.path.exists(input_path):
        logger.warning("Selector file not found: %s", input_path)
        return 0

    with open(input_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    pairs = []
    for sel, entries in registry.items():
        for entry in entries:
            pairs.append((sel, entry["signature"], entry.get("source", "imported"),
                          entry.get("occurrences", 1)))

    with _db_connection(db_path) as conn:
        cur = conn.cursor()
        cur.executemany("""
            INSERT INTO selector_registry (selector, signature, source, occurrences)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(selector, signature) DO UPDATE SET
                occurrences = MAX(selector_registry.occurrences, excluded.occurrences),
                last_seen = CURRENT_TIMESTAMP
        """, pairs)
        conn.commit()

    logger.info("Imported %d selector mappings from %s", len(pairs), input_path)
    return len(pairs)


def count_selectors(db_path: Path = DB_PATH) -> int:
    """Count unique selectors in the registry."""
    return _count_query(db_path, "SELECT COUNT(DISTINCT selector) FROM selector_registry")


def _count_query(db_path: Path, sql: str, params: tuple = ()) -> int:
    with _db_connection(db_path) as conn:
        return conn.execute(sql, params).fetchone()[0]


def count_contracts(db_path: Path = DB_PATH, source: str = None) -> int:
    if source:
        return _count_query(db_path, "SELECT COUNT(*) FROM contracts WHERE source = ?", (source,))
    return _count_query(db_path, "SELECT COUNT(*) FROM contracts")


def count_unprocessed(db_path: Path = DB_PATH) -> int:
    return _count_query(db_path, "SELECT COUNT(*) FROM contracts WHERE processed = FALSE")


def count_function_pairs(db_path: Path = DB_PATH) -> int:
    return _count_query(db_path, "SELECT COUNT(*) FROM function_pairs")


def get_existing_source_hashes(db_path: Path = DB_PATH) -> Set[str]:
    """Load all existing source_hash values for fast dedup lookups."""
    with _db_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT source_hash FROM contracts WHERE source_hash IS NOT NULL"
        ).fetchall()
    return {r[0] for r in rows}


def get_body_hash_counts(db_path: Path = DB_PATH) -> Dict[str, int]:
    """Get count of each body_hash for frequency capping."""
    with _db_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT body_hash, COUNT(*) FROM function_pairs "
            "WHERE body_hash IS NOT NULL GROUP BY body_hash"
        ).fetchall()
    return {r[0]: r[1] for r in rows}


# ---------------------------------------------------------------------------
# Phase 1 -- Download from HuggingFace
# ---------------------------------------------------------------------------

def _get_parquet_files(config: str = "flattened", split: str = "train") -> List[str]:
    """List Parquet files in the HuggingFace dataset repo."""
    api = HfApi()
    all_files = api.list_repo_files("andstor/smart_contracts", repo_type="dataset")
    prefix = f"data/{config}/{split}/"
    return sorted(f for f in all_files if f.startswith(prefix) and f.endswith(".parquet"))


def _flush_batch(cur: sqlite3.Cursor, batch: list):
    """Bulk-insert a batch of contract rows (duplicates ignored)."""
    cur.executemany("""
        INSERT OR IGNORE INTO contracts
            (address, source_code, bytecode, compiler_version,
             optimization_enabled, optimization_runs, abi, contract_name,
             source, source_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, batch)


def _parse_opt_used(value) -> bool:
    """Coerce optimization_used to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


def download_contracts(
    limit: int = 0,
    db_path: Path = DB_PATH,
    cache_dir: Optional[str] = None,
) -> int:
    """Download contracts from HuggingFace with source-code-level dedup.

    Parquet files are cached locally (default: ~/.cache/huggingface/hub/).
    On rerun, cached files are reused without re-downloading.
    """
    logger.info("Listing Parquet files from andstor/smart_contracts (flattened/train)...")
    parquet_files = _get_parquet_files("flattened", "train")
    if not parquet_files:
        logger.error("No Parquet files found!")
        return 0
    logger.info(f"Found {len(parquet_files)} Parquet file(s)")

    existing_hashes = get_existing_source_hashes(db_path)
    logger.info(f"Loaded {len(existing_hashes)} existing source hashes for dedup")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    inserted = 0
    skipped = 0
    deduped = 0
    total_seen = 0

    try:
        for pq_idx, pq_file in enumerate(parquet_files):
            logger.info(f"Downloading Parquet file {pq_idx + 1}/{len(parquet_files)}: {pq_file}")

            try:
                local_path = hf_hub_download(
                    repo_id="andstor/smart_contracts",
                    filename=pq_file,
                    repo_type="dataset",
                    cache_dir=cache_dir,
                )
                logger.info(f"  Cached at: {local_path}")
                df = pd.read_parquet(local_path)
            except Exception as e:
                logger.warning(f"Failed to download/read {pq_file}: {e}")
                continue

            logger.info(f"  Loaded {len(df)} rows from {pq_file}")

            batch: list = []
            for _, row in df.iterrows():
                total_seen += 1
                if limit and total_seen > limit:
                    break

                if str(row.get("language", "")).strip().lower() != "solidity":
                    skipped += 1
                    continue

                src = str(row.get("source_code", "") or "")
                if len(src.strip()) < 20:
                    skipped += 1
                    continue

                src_hash = hash_source_code(src)
                if src_hash in existing_hashes:
                    deduped += 1
                    continue
                existing_hashes.add(src_hash)

                address = str(row.get("contract_address", "") or "").strip()
                if not address:
                    address = f"hf_{hashlib.md5(src[:200].encode()).hexdigest()}"

                compiler_version = str(row.get("compiler_version", "") or "")
                opt_used = _parse_opt_used(row.get("optimization_used", False))
                runs = int(row.get("runs", 200) or 200) if not pd.isna(row.get("runs", 200)) else 200

                batch.append((
                    address, src, "",
                    compiler_version, opt_used, runs,
                    str(row.get("abi", "") or ""),
                    str(row.get("contract_name", "") or ""),
                    "huggingface",
                    src_hash,
                ))

                if len(batch) >= BATCH_INSERT_SIZE:
                    _flush_batch(cur, batch)
                    inserted += len(batch)
                    batch = []
                    conn.commit()

            if batch:
                _flush_batch(cur, batch)
                inserted += len(batch)
                batch = []
                conn.commit()

            logger.info(
                f"  Running: {inserted} inserted, {deduped} deduped, "
                f"{skipped} skipped, {total_seen} seen"
            )
            if limit and total_seen > limit:
                logger.info(f"Reached download limit of {limit}")
                break

    except KeyboardInterrupt:
        logger.warning("Download interrupted by user")

    conn.close()
    logger.info(
        f"Download complete: {inserted} stored, {deduped} source-deduped, "
        f"{skipped} skipped, {total_seen} total"
    )
    return inserted


# ---------------------------------------------------------------------------
# Phase 2 -- Compile & generate TAC pairs
# ---------------------------------------------------------------------------

def _prepare_contract(
    address: str,
    source_code: str,
    orig_version: str,
    orig_runs: int,
    max_compiler_versions: int,
) -> Optional[Dict]:
    """Parse a contract and return a compile job spec (runs in worker)."""
    logging.disable(logging.CRITICAL)
    try:
        source_files = parse_etherscan_source(source_code)
        if not source_files:
            return None

        combined_source = "\n\n".join(source_files.values())
        pragmas = parse_pragma(combined_source)
        pragma = pragmas[0] if pragmas else ">=0.4.0"

        compatible = compatible_versions_for_pragma(pragma)
        if not compatible:
            return None

        if orig_version:
            norm_orig = _normalize_version(orig_version)
            if norm_orig and norm_orig not in compatible:
                compatible.insert(0, norm_orig)

        if max_compiler_versions > 0:
            compatible = compatible[:max_compiler_versions]

        parser = SolidityParser()
        solidity_functions = parser.extract_functions(combined_source)
        if not solidity_functions:
            return None

        return {
            "address": address,
            "source_files": source_files,
            "compatible_versions": compatible,
            "solidity_functions": _add_selectors(solidity_functions),
            "runs": orig_runs or 200,
        }
    except Exception:
        return None


def _compile_one_job(
    address: str,
    source_files: Dict[str, str],
    solidity_functions: List[Dict],
    solc_version: str,
    opt_enabled: bool,
    runs: int,
    min_body_length: int,
) -> List[Dict]:
    """Compile one (contract, version, optimizer) combo and return pairs.

    Quality filters applied: min body length, trivial getters, proxy forwarders.
    """
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
            except Exception:
                continue

            contract_sol_funcs = [
                f for f in solidity_functions if f.get("contract_name", "") == cname
            ] or solidity_functions

            for m in _match_functions(contract_sol_funcs, bytecode_functions, analyzer):
                sol_body = m["solidity_function"]["body"]
                if len(sol_body.strip()) < min_body_length:
                    continue
                if is_trivial_function(sol_body) or is_proxy_only(sol_body):
                    continue

                pair = _build_pair(m, address, solc_version, opt_enabled, runs, cname)
                if pair:
                    pairs.append(pair)

        return pairs
    except Exception:
        return []


def compile_and_generate(
    max_compiler_versions: int = 0,
    workers: int = 0,
    max_body_dupes: int = 5,
    min_body_length: int = 50,
    db_path: Path = DB_PATH,
) -> int:
    """Compile unprocessed contracts and generate TAC->Solidity pairs.

    Dedup is handled deterministically by DB unique indexes and
    ROW_NUMBER() at export time (not in-memory frequency caps).
    """
    with _db_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT address, source_code, compiler_version, "
            "optimization_enabled, optimization_runs "
            "FROM contracts WHERE processed = FALSE"
        ).fetchall()

    total_contracts = len(rows)
    logger.info(f"Found {total_contracts} unprocessed contracts to compile")
    if total_contracts == 0:
        return 0

    if workers <= 0:
        workers = os.cpu_count() or 4
    logger.info(f"Using {workers} worker processes (of {os.cpu_count()} CPUs)")
    logger.info(f"Quality: min_body_length={min_body_length}, max_body_dupes={max_body_dupes}")

    # Phase 2a: prepare contracts in parallel (batched to avoid memory/IPC flood)
    logger.info("Phase 2a: Preparing contracts...")
    prepared: Dict[str, Dict] = {}
    no_work_addresses: List[str] = []
    SUBMIT_BATCH = 10000

    with ProcessPoolExecutor(max_workers=workers) as executor:
        with tqdm(total=total_contracts, desc="Preparing contracts", unit="contract") as pbar:
            for batch_start in range(0, total_contracts, SUBMIT_BATCH):
                batch_rows = rows[batch_start : batch_start + SUBMIT_BATCH]
                futures = {
                    executor.submit(
                        _prepare_contract, addr, src, ver, runs, max_compiler_versions
                    ): addr
                    for addr, src, ver, _opt, runs in batch_rows
                }
                for fut in as_completed(futures):
                    addr = futures[fut]
                    result = fut.result()
                    if result:
                        prepared[addr] = result
                    else:
                        no_work_addresses.append(addr)
                    pbar.update(1)

    # Mark contracts with no work as processed
    if no_work_addresses:
        with _db_connection(db_path) as conn:
            cur = conn.cursor()
            for addr in no_work_addresses:
                cur.execute("UPDATE contracts SET processed = TRUE WHERE address = ?", (addr,))
            conn.commit()

    # Build compile jobs: each (contract x version x optimizer_flag)
    compile_jobs = [
        (addr, prep["source_files"], prep["solidity_functions"],
         ver, opt, prep["runs"], min_body_length)
        for addr, prep in prepared.items()
        for ver in prep["compatible_versions"]
        for opt in (True, False)
    ]

    total_jobs = len(compile_jobs)
    logger.info(
        f"Phase 2b: {len(prepared)} contracts -> {total_jobs} compile jobs "
        f"({total_jobs // max(len(prepared), 1)} avg jobs/contract)"
    )
    if total_jobs == 0:
        return 0

    # Phase 2b: compile in parallel (batched to avoid memory/IPC flood)
    total_pairs = 0
    total_db_deduped = 0
    errors_count = 0
    pair_buffer: List[Dict] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        with tqdm(total=total_jobs, desc="Compiling variants", unit="job") as pbar:
            for batch_start in range(0, total_jobs, SUBMIT_BATCH):
                batch_jobs = compile_jobs[batch_start : batch_start + SUBMIT_BATCH]
                futures = {executor.submit(_compile_one_job, *job): job[0] for job in batch_jobs}

                for fut in as_completed(futures):
                    try:
                        pairs = fut.result()
                        if pairs:
                            pair_buffer.extend(pairs)
                            total_pairs += len(pairs)
                    except Exception:
                        errors_count += 1

                    if len(pair_buffer) >= FLUSH_SIZE:
                        inserted = _store_pairs_batch(db_path, pair_buffer)
                        total_db_deduped += len(pair_buffer) - inserted
                        pair_buffer = []

                    pbar.set_postfix(pairs=total_pairs, errors=errors_count, refresh=False)
                    pbar.update(1)

    if pair_buffer:
        inserted = _store_pairs_batch(db_path, pair_buffer)
        total_db_deduped += len(pair_buffer) - inserted

    # Mark prepared contracts as processed
    with _db_connection(db_path) as conn:
        cur = conn.cursor()
        for addr in prepared:
            cur.execute("UPDATE contracts SET processed = TRUE WHERE address = ?", (addr,))
        conn.commit()

    # Harvest selectors from all prepared contracts into the registry
    all_selectors: List[tuple] = []
    for prep in prepared.values():
        for func in prep.get("solidity_functions", []):
            sel = func.get("selector")
            if not sel:
                continue
            sig = func.get("signature", "")
            match = re.match(r"function\s+(\w+)\s*\(([^)]*)\)", sig)
            if match:
                func_name = match.group(1)
                params_str = match.group(2).strip()
                if params_str:
                    param_types = [p.strip().split()[0] for p in params_str.split(",") if p.strip()]
                    canonical = f"{func_name}({','.join(param_types)})"
                else:
                    canonical = f"{func_name}()"
                all_selectors.append((sel, canonical))

    if all_selectors:
        store_selectors_batch(all_selectors, source="compiled", db_path=db_path)
        unique_sels = len(set(s[0] for s in all_selectors))
        logger.info(
            f"Selector registry: harvested {len(all_selectors)} mappings "
            f"({unique_sels} unique selectors) from {len(prepared)} contracts"
        )

    logger.info(
        f"Compilation complete: {len(prepared)} contracts, {total_jobs} jobs, "
        f"{total_pairs} pairs generated, {total_db_deduped} deduped at DB insert, "
        f"{errors_count} errors"
    )
    return total_pairs


# ---------------------------------------------------------------------------
# Pair building, matching & storage
# ---------------------------------------------------------------------------

def _add_selectors(functions: List[Dict]) -> List[Dict]:
    """Compute 4-byte function selectors for each Solidity function."""
    for func in functions:
        try:
            signature = func["signature"]
            match = re.match(r"function\s+(\w+)\s*\(([^)]*)\)", signature)
            if match:
                func_name = match.group(1)
                params_str = match.group(2).strip()
                if params_str:
                    param_types = [p.strip().split()[0] for p in params_str.split(",") if p.strip()]
                    canonical = f"{func_name}({','.join(param_types)})"
                else:
                    canonical = f"{func_name}()"
            else:
                canonical = signature.replace("function ", "").strip()

            func["selector"] = "0x" + Web3.keccak(text=canonical)[:4].hex()
        except Exception:
            func["selector"] = None
    return functions


def _match_functions(
    solidity_functions: List[Dict],
    bytecode_functions: Dict,
    analyzer,
) -> List[Dict]:
    """Match Solidity functions to bytecode functions by selector."""
    sol_by_sel = {f["selector"]: f for f in solidity_functions if f.get("selector")}
    bc_by_sel = {f.selector: f for f in bytecode_functions.values() if f.selector}

    matches = []
    for selector, sol_func in sol_by_sel.items():
        if selector in bc_by_sel:
            matches.append({
                "solidity_function": sol_func,
                "bytecode_function": bc_by_sel[selector],
                "tac": _extract_tac(bc_by_sel[selector], analyzer),
                "selector": selector,
            })
    return matches


def _extract_tac(bytecode_function, analyzer) -> str:
    """Generate TAC text for a single bytecode function."""
    lines: List[str] = []
    try:
        lines.append(f"function {bytecode_function.name}:")
        if bytecode_function.selector:
            lines.append(f"  // Selector: {bytecode_function.selector}")
        lines.append(f"  // Entry block: {bytecode_function.entry_block}")

        blocks = bytecode_function.basic_blocks or []
        if not blocks and bytecode_function.entry_block in analyzer.basic_blocks:
            blocks = _collect_blocks(bytecode_function.entry_block, analyzer.basic_blocks)

        for block in blocks:
            lines.append(f"  {block.id}:")
            if block.predecessors:
                lines.append(f"    // Predecessors: {', '.join(block.predecessors)}")
            if block.successors:
                lines.append(f"    // Successors: {', '.join(block.successors)}")
            for instr in block.instructions:
                lines.append(f"    {analyzer._format_tac_instruction(instr)}")
            lines.append("")
    except Exception as e:
        lines.append(f"  // Error extracting TAC: {e}")
    return "\n".join(lines)


def _collect_blocks(entry_block_id: str, all_blocks: Dict) -> list:
    """Walk the CFG from an entry block and collect reachable blocks."""
    if entry_block_id not in all_blocks:
        return []

    visited: Set[str] = set()
    result: list = []

    def traverse(bid: str):
        if bid in visited or bid not in all_blocks:
            return
        visited.add(bid)
        block = all_blocks[bid]
        result.append(block)
        for s in block.successors:
            traverse(s)

    traverse(entry_block_id)
    return result


def _build_pair(
    match: Dict,
    address: str,
    compiler_version: str,
    optimizer_enabled: bool,
    optimizer_runs: int,
    compiled_contract: str,
) -> Optional[Dict]:
    """Build a single training pair dict from a function match."""
    sol_func = match["solidity_function"]
    tac = match["tac"]

    if len(sol_func["body"].strip()) < 10 or not tac or len(tac.strip()) < 10:
        return None

    return {
        "contract_address": address,
        "function_name": sol_func["name"],
        "tac_representation": tac,
        "solidity_code": sol_func["body"],
        "function_signature": sol_func["signature"],
        "visibility": sol_func["visibility"],
        "is_payable": sol_func["is_payable"],
        "is_view": sol_func["is_view"],
        "metadata": json.dumps({
            "contract_name": sol_func.get("contract_name"),
            "selector": match["selector"],
            "compiler_version": compiler_version,
            "optimizer_enabled": optimizer_enabled,
            "optimizer_runs": optimizer_runs,
            "compiled_contract": compiled_contract,
            "source": "huggingface",
        }),
        "hash": _md5(tac + sol_func["body"]),
        "body_hash": hash_normalized_body(sol_func["body"]),
        "tac_hash": hash_normalized_tac(tac),
        "pair_norm_hash": hash_normalized_pair(tac, sol_func["body"]),
    }


def _store_pairs_batch(db_path: Path, pairs: List[Dict]) -> int:
    """Insert pairs into the DB, ignoring duplicates. Returns rows inserted."""
    if not pairs:
        return 0

    values = [
        (
            p["contract_address"], p["function_name"], p["tac_representation"],
            p["solidity_code"], p["function_signature"], p["visibility"],
            p["is_payable"], p["is_view"], p["metadata"], p["hash"],
            p["body_hash"], p.get("tac_hash"), p.get("pair_norm_hash"),
        )
        for p in pairs
    ]

    with _db_connection(db_path) as conn:
        cur = conn.cursor()
        before = cur.execute("SELECT COUNT(*) FROM function_pairs").fetchone()[0]
        cur.executemany("""
            INSERT OR IGNORE INTO function_pairs
                (contract_address, function_name, tac_representation,
                 solidity_code, function_signature, visibility,
                 is_payable, is_view, metadata, hash, body_hash,
                 tac_hash, pair_norm_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, values)
        conn.commit()
        after = cur.execute("SELECT COUNT(*) FROM function_pairs").fetchone()[0]

    return after - before


# ---------------------------------------------------------------------------
# Phase 3 -- Export training data
# ---------------------------------------------------------------------------

def export_training_data(
    output_path: str = "data/hf_training_dataset.jsonl",
    max_body_dupes: int = 5,
    db_path: Path = DB_PATH,
) -> str:
    """Export function pairs as JSONL, deduped at export time.

    Uses ROW_NUMBER() partitioned by pair_norm_hash to keep at most
    `max_body_dupes` rows per semantic pair.
    """
    with _db_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT function_name, tac_representation, solidity_code,
                   function_signature, visibility, is_payable, is_view,
                   contract_address, metadata, body_hash
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY COALESCE(pair_norm_hash, body_hash, hash)
                           ORDER BY created_at
                       ) AS rn
                FROM function_pairs
            )
            WHERE rn <= ?
        """, (max_body_dupes,)).fetchall()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for (func_name, tac, solidity, sig, vis,
             is_pay, is_view, addr, meta_str, body_hash) in rows:
            record = {
                "input": tac,
                "output": solidity,
                "metadata": {
                    "function_name": func_name,
                    "function_signature": sig,
                    "visibility": vis,
                    "is_payable": bool(is_pay),
                    "is_view": bool(is_view),
                    "contract_address": addr,
                },
            }
            if meta_str:
                try:
                    record["metadata"].update(json.loads(meta_str))
                except json.JSONDecodeError:
                    pass
            f.write(json.dumps(record) + "\n")
            count += 1

    logger.info(f"Exported {count} training pairs to {output_path}")

    # Report dedup stats
    with _db_connection(db_path) as conn:
        total_in_db = conn.execute("SELECT COUNT(*) FROM function_pairs").fetchone()[0]
        unique_pairs = conn.execute(
            "SELECT COUNT(DISTINCT COALESCE(pair_norm_hash, body_hash, hash)) FROM function_pairs"
        ).fetchone()[0]
        unique_bodies = conn.execute(
            "SELECT COUNT(DISTINCT COALESCE(body_hash, hash)) FROM function_pairs"
        ).fetchone()[0]
        unique_tacs = conn.execute(
            "SELECT COUNT(DISTINCT tac_hash) FROM function_pairs WHERE tac_hash IS NOT NULL"
        ).fetchone()[0]

    logger.info(
        f"Dedup stats: {total_in_db} total in DB, {unique_pairs} unique pairs, "
        f"{unique_bodies} unique bodies, {unique_tacs} unique TACs, "
        f"{count} exported (max {max_body_dupes} per semantic pair)"
    )
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download smart contracts from HuggingFace, compile with every "
            "compatible solc version, and generate TAC->Solidity training data. "
            "Includes multi-layer deduplication for clean training data."
        ),
    )
    parser.add_argument("--download-only", action="store_true",
                        help="Only download contracts, skip compilation.")
    parser.add_argument("--compile-only", action="store_true",
                        help="Only compile already-downloaded contracts.")
    parser.add_argument("--export-only", action="store_true",
                        help="Only export existing pairs to JSONL.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max contracts to download (0 = all).")
    parser.add_argument("--max-compiler-versions", type=int, default=0,
                        help="Max solc versions per contract (0 = all compatible).")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0 = auto-detect CPU count).")
    parser.add_argument("--max-body-dupes", type=int, default=5,
                        help="Max copies of any normalized function body (default: 5).")
    parser.add_argument("--min-body-length", type=int, default=50,
                        help="Min Solidity body length to keep (default: 50 chars).")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HuggingFace cache dir for Parquet files.")
    parser.add_argument("--output", type=str, default="data/hf_training_dataset.jsonl",
                        help="Output JSONL file path.")
    parser.add_argument("--export-selectors", type=str, default=None,
                        help="Export selector registry to JSON file (e.g. data/selectors.json).")
    parser.add_argument("--import-selectors", type=str, default=None,
                        help="Import selector registry from JSON file before processing.")
    parser.add_argument("--db", type=str, default="data/contracts.db",
                        help="SQLite database path.")

    args = parser.parse_args()
    db_path = Path(args.db)

    setup_logging()

    logger.info("=" * 70)
    logger.info("Smart Contract HuggingFace Dataset -> Training Data Pipeline")
    logger.info("=" * 70)

    init_database(db_path)

    # Import selectors if requested
    if args.import_selectors:
        import_selector_registry(args.import_selectors, db_path)

    # Phase 1: Download
    if not args.compile_only and not args.export_only:
        logger.info("\n--- Phase 1: Download from HuggingFace ---")
        before = count_contracts(db_path)
        download_contracts(limit=args.limit, db_path=db_path, cache_dir=args.cache_dir)
        after = count_contracts(db_path)
        logger.info(f"Database now has {after} contracts (was {before}, added {after - before})")
        if args.download_only:
            logger.info("Download-only mode. Stopping.")
            return

    # Phase 2: Compile & generate pairs
    if not args.download_only and not args.export_only:
        logger.info("\n--- Phase 2: Compile & Generate TAC Pairs ---")
        unprocessed = count_unprocessed(db_path)
        logger.info(f"Unprocessed contracts: {unprocessed}")

        if unprocessed > 0:
            pairs = compile_and_generate(
                max_compiler_versions=args.max_compiler_versions,
                workers=args.workers,
                max_body_dupes=args.max_body_dupes,
                min_body_length=args.min_body_length,
                db_path=db_path,
            )
            logger.info(f"New training pairs created: {pairs}")
        else:
            logger.info("All contracts already processed.")

        if args.compile_only:
            logger.info(f"Total function pairs in database: {count_function_pairs(db_path)}")
            return

    # Phase 3: Export
    logger.info("\n--- Phase 3: Export Training Data ---")
    total = count_function_pairs(db_path)
    logger.info(f"Total function pairs in DB: {total}")

    if total > 0:
        out = export_training_data(
            output_path=args.output,
            max_body_dupes=args.max_body_dupes,
            db_path=db_path,
        )
        logger.info(f"Training data written to: {out}")
    else:
        logger.warning("No function pairs to export!")

    # Export selectors if requested
    if args.export_selectors:
        export_selector_registry(args.export_selectors, db_path)

    sel_count = count_selectors(db_path)
    logger.info("")
    logger.info("=" * 70)
    logger.info("Pipeline complete!")
    logger.info(f"  Contracts in DB:    {count_contracts(db_path)}")
    logger.info(f"  Function pairs:     {total}")
    logger.info(f"  Unique selectors:   {sel_count}")
    logger.info(f"  Output file:        {args.output}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()