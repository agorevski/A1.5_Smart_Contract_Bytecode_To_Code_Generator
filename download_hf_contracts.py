#!/usr/bin/env python3
"""
Download Smart Contracts from HuggingFace & Generate Training Data

Downloads verified Solidity contracts from the andstor/smart_contracts
dataset on HuggingFace, compiles each with every compatible solc version
(optimizer on + off), generates TAC via BytecodeAnalyzer, and exports
JSONL training pairs with bytecode-only TAC input, Solidity output, and
metadata retained only for analysis/filtering/manifests.

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
    python download_hf_contracts.py --validate-jsonl data/hf_training_dataset.jsonl
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
from web3 import Web3

from src.bytecode_analyzer import BytecodeAnalyzer
from src.dataset_pipeline import (
    TRAINING_ROW_SCHEMA_VERSION,
    SolidityParser,
    normalize_training_metadata,
    sanitize_tac_prompt_input,
)
from src.abi_enrichment import (
    canonicalize_abi_type,
    normalize_hex,
)
from src.local_compiler import (
    compile_source,
    compile_multi_file,
    compatible_versions_for_pragmas,
    install_solc_version,
    parse_etherscan_source,
    parse_pragma,
    version_satisfies_all_pragmas,
    _normalize_version,
)

try:
    import resource
except ImportError:  # pragma: no cover - Windows compatibility
    resource = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_FILE = "download_hf_contracts.log"
DB_PATH = Path("data/contracts.db")
FLUSH_SIZE = 200
BATCH_INSERT_SIZE = 500
PARQUET_READ_BATCH_SIZE = 4096
DEFAULT_EXPORT_MAX_SEQ_LENGTH = 2048
HF_DATASET_REPO = "andstor/smart_contracts"
MANIFEST_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parent

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
        "SafeMath",
        "Ownable",
        "Context",
        "IERC20",
        "ERC20",
        "ERC721",
        "ERC1155",
        "ReentrancyGuard",
        "Pausable",
        "AccessControl",
    )
]

TRIVIAL_PATTERNS = [
    re.compile(r"^\s*\{\s*return\s+\w+\s*;\s*\}\s*$", re.DOTALL),
    re.compile(r"^\s*\{\s*return\s+\d+\s*;\s*\}\s*$", re.DOTALL),
    re.compile(r"^\s*\{\s*\}\s*$", re.DOTALL),
    # Issue 5: expanded trivial patterns
    re.compile(r"^\s*\{\s*return\s+\w+(\.\w+)*\s*;\s*\}\s*$", re.DOTALL),  # return obj.prop;
    re.compile(r"^\s*\{\s*\w+\s*=\s*\w+\s*;\s*\}\s*$", re.DOTALL),  # x = y;
    re.compile(r"^\s*\{\s*emit\s+\w+\([^)]*\)\s*;\s*\}\s*$", re.DOTALL),  # emit Event(...);
    re.compile(r"^\s*\{\s*return\s+(true|false)\s*;\s*\}\s*$", re.DOTALL),  # return true/false;
    re.compile(r"^\s*\{\s*return\s+\w+\([^)]*\)\s*;\s*\}\s*$", re.DOTALL),  # return func(x);
    re.compile(r"^\s*\{\s*_\s*;\s*\}\s*$", re.DOTALL),  # modifier { _; }
]

# Minimum meaningful non-whitespace token count for a function body
MIN_MEANINGFUL_TOKENS = 15

PROXY_PATTERN = re.compile(r"\bdelegatecall\b")
ETH_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


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

    Strips comments (including any legacy compiler/source annotations),
    collapses whitespace, and lowercases so identical bytecode compiled by
    different solc versions produces the same hash.
    """
    text = re.sub(r"//[^\n]*", "", tac)
    return _collapse_whitespace(text).lower()


def _selector_safe_name(selector: Optional[str]) -> Optional[str]:
    """Return a selector-based TAC function name, or ``None`` without a selector."""
    if not selector:
        return None
    selector_text = str(selector).strip().lower()
    if selector_text.startswith("0x"):
        selector_text = selector_text[2:]
    selector_text = re.sub(r"[^0-9a-f]", "", selector_text)
    if len(selector_text) != 8:
        return None
    return f"selector_{selector_text}"


def _safe_tac_function_name(bytecode_function) -> str:
    """Name TAC headers from bytecode-visible facts, preferring selectors."""
    selector_name = _selector_safe_name(getattr(bytecode_function, "selector", None))
    if selector_name:
        return selector_name

    raw_name = str(getattr(bytecode_function, "name", "") or "").strip()
    if raw_name in {"fallback", "fallback_function", "receive"}:
        return re.sub(r"[^A-Za-z0-9_]", "_", raw_name).strip("_") or "bytecode_function"

    if raw_name.startswith("internal_"):
        return re.sub(r"[^A-Za-z0-9_]", "_", raw_name).strip("_") or "bytecode_function"

    entry_block = str(getattr(bytecode_function, "entry_block", "") or "").strip()
    if entry_block:
        safe_entry = re.sub(r"[^A-Za-z0-9_]", "_", entry_block).strip("_")
        if safe_entry:
            return f"bytecode_{safe_entry}"

    return "bytecode_function"


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


def _now_utc_iso() -> str:
    """Return a UTC timestamp suitable for manifests."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _make_run_id(prefix: str) -> str:
    """Generate a stable-enough run identifier for linking manifests/diagnostics."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{os.getpid()}"


def _write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    """Write a deterministic JSON manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _run_git(args: List[str]) -> Optional[str]:
    """Run a small git command for manifest provenance."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _get_git_state() -> Dict[str, Any]:
    """Capture best-effort git state for dataset lineage manifests."""
    commit = _run_git(["rev-parse", "HEAD"])
    if not commit:
        return {"available": False}
    status_short = _run_git(["status", "--short"]) or ""
    return {
        "available": True,
        "repository_root": _run_git(["rev-parse", "--show-toplevel"]) or str(REPO_ROOT),
        "commit": commit,
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status_short),
        "status_short": status_short.splitlines()[:100],
    }


def _command_context(command_args: Optional[List[str]] = None) -> Dict[str, Any]:
    """Build command provenance; callers may pass args for programmatic runs."""
    args = list(sys.argv[1:] if command_args is None else command_args)
    executable = Path(sys.argv[0]).name if sys.argv else "python"
    return {
        "cwd": str(Path.cwd()),
        "executable": executable,
        "args": args,
        "argv": [executable, *args],
    }


def _base_manifest(
    kind: str,
    *,
    status: str,
    started_at: str,
    duration_seconds: float,
    command_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Common manifest envelope for download/compile/export stages."""
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_kind": kind,
        "status": status,
        "generated_at": _now_utc_iso(),
        "started_at": started_at,
        "duration_seconds": round(duration_seconds, 3),
        "command": _command_context(command_args),
        "git": _get_git_state(),
    }


def _sha256_file(path: Path) -> Optional[str]:
    """Hash a file artifact if it exists."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_row_count(path: Path) -> int:
    """Count non-empty JSONL rows."""
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _artifact_summary(path: Path, row_count: Optional[int] = None) -> Dict[str, Any]:
    """Return path/size/hash metadata for an emitted artifact."""
    path = Path(path)
    summary: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists() and path.is_file():
        summary.update(
            {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if row_count is not None:
        summary["row_count"] = row_count
    return summary


def _resolve_hf_revision(hf_revision: Optional[str]) -> Optional[str]:
    """Resolve a Hugging Face dataset revision to a commit SHA when available."""
    try:
        info = HfApi().repo_info(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            revision=hf_revision,
        )
    except Exception:
        return None
    return getattr(info, "sha", None)


def _hf_lineage(
    hf_revision: Optional[str], resolved_revision: Optional[str] = None
) -> Dict[str, Any]:
    """Source lineage block for the Hugging Face verified-contracts dataset."""
    return {
        "type": "huggingface_dataset",
        "repo": HF_DATASET_REPO,
        "config": "flattened",
        "split": "train",
        "requested_revision": hf_revision or "default",
        "resolved_revision": resolved_revision,
    }


def is_trivial_function(body: str) -> bool:
    """Check if a function body is too trivial to be useful training data.

    A function is trivial if it matches any TRIVIAL_PATTERNS regex, or if
    the body has fewer than MIN_MEANINGFUL_TOKENS non-whitespace tokens.
    """
    if any(pat.match(body) for pat in TRIVIAL_PATTERNS):
        return True
    # Token-count heuristic: split on whitespace and punctuation-ish boundaries
    tokens = re.findall(r"[A-Za-z_]\w*|[0-9]+|[^\s\w]", body)
    if len(tokens) < MIN_MEANINGFUL_TOKENS:
        return True
    return False


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

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS contracts (
                address TEXT PRIMARY KEY,
                source_code TEXT,
                bytecode TEXT,
                compiler_version TEXT,
                optimization_enabled BOOLEAN,
                optimization_runs INTEGER,
                processed BOOLEAN DEFAULT FALSE,
                compile_status TEXT DEFAULT 'pending',
                attempt_count INTEGER DEFAULT 0,
                last_error TEXT,
                processed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cur.execute(
            """
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
        """
        )

        # Selector registry — maps 4-byte selectors to signatures
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS selector_registry (
                selector    TEXT NOT NULL,
                signature   TEXT NOT NULL,
                source      TEXT DEFAULT 'compiled',
                occurrences INTEGER DEFAULT 1,
                first_seen  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (selector, signature)
            )
        """
        )

        # Per-run compile failures/no-output diagnostics for reproducibility.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS compile_diagnostics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                contract_address TEXT,
                compiler_version TEXT,
                optimizer_enabled BOOLEAN,
                optimization_runs INTEGER,
                status TEXT NOT NULL,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (contract_address) REFERENCES contracts (address)
            )
        """
        )

        # Migration: add columns that may not exist yet
        _add_columns(
            cur,
            "contracts",
            [
                ("abi", "TEXT"),
                ("contract_name", "TEXT"),
                ("source", "TEXT DEFAULT 'etherscan'"),
                ("source_hash", "TEXT"),
                ("compile_status", "TEXT DEFAULT 'pending'"),
                ("attempt_count", "INTEGER DEFAULT 0"),
                ("last_error", "TEXT"),
                ("processed_at", "TIMESTAMP"),
            ],
        )
        _add_columns(
            cur,
            "function_pairs",
            [
                ("body_hash", "TEXT"),
                ("tac_hash", "TEXT"),
                ("pair_norm_hash", "TEXT"),
            ],
        )

        # Migration: ensure indexes
        _ensure_indexes(
            cur,
            [
                ("idx_body_hash", False, "function_pairs(body_hash)"),
                ("idx_tac_hash", False, "function_pairs(tac_hash)"),
                ("idx_pair_norm_hash", True, "function_pairs(pair_norm_hash)"),
                ("idx_source_hash", False, "contracts(source_hash)"),
                ("idx_sel_reg_sel", False, "selector_registry(selector)"),
                ("idx_compile_diag_run", False, "compile_diagnostics(run_id)"),
                ("idx_compile_diag_status", False, "compile_diagnostics(status)"),
            ],
        )

        conn.commit()

    # Seed the registry with built-in selectors
    _seed_builtin_selectors(db_path)


def _seed_builtin_selectors(db_path: Path = DB_PATH):
    """Insert the curated built-in selectors into selector_registry."""
    from src.selector_resolver import _BUILTIN_SELECTORS

    with _db_connection(db_path) as conn:
        cur = conn.cursor()
        for sel, sig in _BUILTIN_SELECTORS.items():
            cur.execute(
                """
                INSERT INTO selector_registry (selector, signature, source, occurrences)
                VALUES (?, ?, 'builtin', 1000)
                ON CONFLICT(selector, signature) DO UPDATE SET
                    source = CASE WHEN selector_registry.source = 'builtin'
                                  THEN 'builtin' ELSE selector_registry.source END
            """,
                (sel.lower(), sig),
            )
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
        cur.executemany(
            """
            INSERT INTO selector_registry (selector, signature, source, occurrences)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(selector, signature) DO UPDATE SET
                occurrences = selector_registry.occurrences + 1,
                last_seen = CURRENT_TIMESTAMP
        """,
            [(s.lower(), sig, source) for s, sig in selectors],
        )
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
        registry[sel].append(
            {
                "signature": sig,
                "source": source,
                "occurrences": occ,
            }
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    logger.info(
        "Exported %d unique selectors (%d mappings) to %s", len(registry), len(rows), output_path
    )
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
            pairs.append(
                (
                    sel,
                    entry["signature"],
                    entry.get("source", "imported"),
                    entry.get("occurrences", 1),
                )
            )

    with _db_connection(db_path) as conn:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO selector_registry (selector, signature, source, occurrences)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(selector, signature) DO UPDATE SET
                occurrences = MAX(selector_registry.occurrences, excluded.occurrences),
                last_seen = CURRENT_TIMESTAMP
        """,
            pairs,
        )
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


def _db_table_counts(db_path: Path = DB_PATH) -> Dict[str, int]:
    """Return row counts for known data-generation tables."""
    counts: Dict[str, int] = {}
    with _db_connection(db_path) as conn:
        for table in ("contracts", "function_pairs", "selector_registry", "compile_diagnostics"):
            try:
                counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.OperationalError:
                counts[table] = 0
    return counts


def _contract_source_counts(db_path: Path = DB_PATH) -> Dict[str, int]:
    with _db_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT COALESCE(source, 'unknown'), COUNT(*) FROM contracts GROUP BY COALESCE(source, 'unknown')"
        ).fetchall()
    return {source: count for source, count in rows}


def _contract_status_counts(db_path: Path = DB_PATH) -> Dict[str, int]:
    with _db_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT COALESCE(compile_status, 'pending'), COUNT(*) "
            "FROM contracts GROUP BY COALESCE(compile_status, 'pending')"
        ).fetchall()
    return {status: count for status, count in rows}


def _db_duplicate_stats(
    db_path: Path = DB_PATH,
    max_body_dupes: int = 2,
    sample_limit: int = 5,
) -> Dict[str, Any]:
    """Compute duplicate stats from the SQLite pair table."""
    with _db_connection(db_path) as conn:
        total_pairs = conn.execute("SELECT COUNT(*) FROM function_pairs").fetchone()[0]
        unique_pairs = conn.execute(
            "SELECT COUNT(DISTINCT COALESCE(pair_norm_hash, hash)) FROM function_pairs"
        ).fetchone()[0]
        unique_bodies = conn.execute(
            "SELECT COUNT(DISTINCT COALESCE(body_hash, hash)) FROM function_pairs"
        ).fetchone()[0]
        unique_tacs = conn.execute(
            "SELECT COUNT(DISTINCT tac_hash) FROM function_pairs WHERE tac_hash IS NOT NULL"
        ).fetchone()[0]
        over_cap_count, rows_over_cap = conn.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(cnt - ?), 0)
            FROM (
                SELECT COUNT(*) AS cnt
                FROM function_pairs
                GROUP BY COALESCE(body_hash, hash)
            )
            WHERE cnt > ?
            """,
            (max_body_dupes, max_body_dupes),
        ).fetchone()
        top_rows = conn.execute(
            """
            SELECT COALESCE(body_hash, hash) AS body_key, COUNT(*) AS cnt
            FROM function_pairs
            GROUP BY COALESCE(body_hash, hash)
            HAVING COUNT(*) > 1
            ORDER BY cnt DESC, body_key
            LIMIT ?
            """,
            (sample_limit,),
        ).fetchall()
        top_bodies = []
        for body_key, count in top_rows:
            samples = conn.execute(
                """
                SELECT function_name, function_signature, contract_address
                FROM function_pairs
                WHERE COALESCE(body_hash, hash) = ?
                ORDER BY COALESCE(contract_address, ''),
                         COALESCE(function_signature, ''),
                         COALESCE(function_name, '')
                LIMIT ?
                """,
                (body_key, sample_limit),
            ).fetchall()
            top_bodies.append(
                {
                    "body_hash": body_key,
                    "count": count,
                    "over_cap_by": max(count - max_body_dupes, 0),
                    "samples": [
                        {
                            "function_name": name,
                            "function_signature": sig,
                            "contract_address": addr,
                        }
                        for name, sig, addr in samples
                    ],
                }
            )

    return {
        "total_pairs": total_pairs,
        "unique_pairs": unique_pairs,
        "duplicate_pair_rows": max(total_pairs - unique_pairs, 0),
        "unique_bodies": unique_bodies,
        "duplicate_body_rows": max(total_pairs - unique_bodies, 0),
        "unique_tacs": unique_tacs,
        "body_cap": max_body_dupes,
        "body_hashes_over_cap": over_cap_count,
        "rows_over_body_cap": rows_over_cap,
        "top_duplicate_bodies": top_bodies,
    }


def _export_selection_stats(db_path: Path = DB_PATH, max_body_dupes: int = 2) -> Dict[str, int]:
    """Mirror export SQL to count rows dropped by pair dedup and body caps."""
    with _db_connection(db_path) as conn:
        total, pair_kept, exported, pair_dropped, body_dropped = conn.execute(
            """
            WITH pair_ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY COALESCE(pair_norm_hash, hash)
                           ORDER BY COALESCE(body_hash, ''),
                                    COALESCE(tac_hash, ''),
                                    COALESCE(contract_address, ''),
                                    COALESCE(function_signature, ''),
                                    COALESCE(function_name, ''),
                                    id
                       ) AS pair_rn
                FROM function_pairs
            ),
            body_ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY COALESCE(body_hash, hash)
                           ORDER BY COALESCE(pair_norm_hash, ''),
                                    COALESCE(tac_hash, ''),
                                    COALESCE(contract_address, ''),
                                    COALESCE(function_signature, ''),
                                    COALESCE(function_name, ''),
                                    id
                       ) AS body_rn
                FROM pair_ranked
                WHERE pair_rn = 1
            )
            SELECT
                (SELECT COUNT(*) FROM function_pairs),
                (SELECT COUNT(*) FROM pair_ranked WHERE pair_rn = 1),
                (SELECT COUNT(*) FROM body_ranked WHERE body_rn <= ?),
                (SELECT COUNT(*) FROM pair_ranked WHERE pair_rn > 1),
                (SELECT COUNT(*) FROM body_ranked WHERE body_rn > ?)
            """,
            (max_body_dupes, max_body_dupes),
        ).fetchone()
    return {
        "total_pairs_in_db": total,
        "rows_after_pair_dedup": pair_kept,
        "rows_exported": exported,
        "pair_duplicate_rows_dropped": pair_dropped,
        "body_cap_rows_dropped": body_dropped,
    }


def _store_compile_diagnostics(db_path: Path, diagnostics: List[Dict[str, Any]]) -> int:
    """Persist compile/analysis failure diagnostics for a generation run."""
    if not diagnostics:
        return 0
    values = [
        (
            d.get("run_id"),
            d.get("contract_address"),
            d.get("compiler_version"),
            d.get("optimizer_enabled"),
            d.get("optimization_runs"),
            d.get("status") or "unknown",
            d.get("error") or "",
        )
        for d in diagnostics
    ]
    with _db_connection(db_path) as conn:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO compile_diagnostics
                (run_id, contract_address, compiler_version, optimizer_enabled,
                 optimization_runs, status, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        conn.commit()
    return len(values)


def _summarize_compile_diagnostics(
    db_path: Path = DB_PATH,
    run_id: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """Summarize stored compile diagnostics by status/error with sample contracts."""
    where = ""
    params: List[Any] = []
    if run_id:
        where = "WHERE run_id = ?"
        params.append(run_id)

    with _db_connection(db_path) as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM compile_diagnostics {where}",
            params,
        ).fetchone()[0]
        status_rows = conn.execute(
            f"SELECT status, COUNT(*) FROM compile_diagnostics {where} GROUP BY status",
            params,
        ).fetchall()
        top_rows = conn.execute(
            f"""
            SELECT status, COALESCE(error, ''), COUNT(*), GROUP_CONCAT(contract_address)
            FROM compile_diagnostics
            {where}
            GROUP BY status, COALESCE(error, '')
            ORDER BY COUNT(*) DESC, status, COALESCE(error, '')
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()

    top_errors = []
    for status, error, count, addresses in top_rows:
        sample_addresses = [a for a in (addresses or "").split(",") if a][:5]
        top_errors.append(
            {
                "status": status,
                "error": error,
                "count": count,
                "sample_contract_addresses": sample_addresses,
            }
        )
    return {
        "run_id": run_id,
        "total_diagnostics": total,
        "status_counts": {status: count for status, count in status_rows},
        "top_errors": top_errors,
    }


# ---------------------------------------------------------------------------
# Phase 1 -- Download from HuggingFace
# ---------------------------------------------------------------------------


def _get_parquet_files(
    config: str = "flattened",
    split: str = "train",
    revision: Optional[str] = None,
) -> List[str]:
    """List Parquet files in the HuggingFace dataset repo."""
    api = HfApi()
    kwargs: Dict[str, Any] = {"repo_type": "dataset"}
    if revision:
        kwargs["revision"] = revision
    all_files = api.list_repo_files(HF_DATASET_REPO, **kwargs)
    prefix = f"data/{config}/{split}/"
    return sorted(f for f in all_files if f.startswith(prefix) and f.endswith(".parquet"))


def _flush_batch(cur: sqlite3.Cursor, batch: list) -> int:
    """Bulk-insert a batch of contract rows and return actual insert count."""
    cur.executemany(
        """
        INSERT OR IGNORE INTO contracts
            (address, source_code, bytecode, compiler_version,
             optimization_enabled, optimization_runs, abi, contract_name,
             source, source_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        batch,
    )
    return cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0


def _mark_contract_status(
    db_path: Path,
    addresses: List[str],
    status: str,
    *,
    processed: bool,
    last_error: str = "",
) -> None:
    """Record compile outcome without conflating failures with success."""
    if not addresses:
        return
    with _db_connection(db_path) as conn:
        cur = conn.cursor()
        cur.executemany(
            """
            UPDATE contracts
            SET processed = ?,
                compile_status = ?,
                attempt_count = COALESCE(attempt_count, 0) + 1,
                last_error = ?,
                processed_at = CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE processed_at END
            WHERE address = ?
            """,
            [(processed, status, last_error or None, processed, addr) for addr in addresses],
        )
        conn.commit()


def _record_contract_outcomes(
    db_path: Path,
    outcomes: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    """Persist per-contract outcomes and return status counts."""
    status_to_addresses: Dict[Tuple[str, bool, str], List[str]] = {}
    for addr, outcome in outcomes.items():
        pair_count = int(outcome.get("pairs", 0) or 0)
        failures = outcome.get("failures", []) or []
        if pair_count > 0:
            key = ("processed", True, "")
        elif failures:
            first_failure = failures[0]
            key = (
                first_failure.get("status", "compile_failed"),
                False,
                first_failure.get("error", "") or "compile job failed",
            )
        else:
            key = ("no_pairs", False, "compile jobs produced no matched pairs")
        status_to_addresses.setdefault(key, []).append(addr)

    counts: Dict[str, int] = {}
    for (status, processed, error), addresses in status_to_addresses.items():
        _mark_contract_status(
            db_path,
            addresses,
            status,
            processed=processed,
            last_error=error,
        )
        counts[status] = counts.get(status, 0) + len(addresses)
    return counts


def _write_compile_manifest(
    db_path: Path,
    manifest_path: Path,
    *,
    run_id: str,
    status: str,
    started_at: str,
    duration_seconds: float,
    parameters: Dict[str, Any],
    summary: Dict[str, Any],
    status_counts: Dict[str, Any],
    drop_counts: Dict[str, Any],
    command_args: Optional[List[str]] = None,
    hf_revision: Optional[str] = None,
) -> Dict[str, Any]:
    """Write a compile-stage manifest and return its payload for tests."""
    manifest = _base_manifest(
        "hf_compile",
        status=status,
        started_at=started_at,
        duration_seconds=duration_seconds,
        command_args=command_args,
    )
    manifest.update(
        {
            "run_id": run_id,
            "lineage": {
                "source": _hf_lineage(hf_revision),
                "database": str(db_path),
            },
            "parameters": parameters,
            "artifacts": {"database": _artifact_summary(db_path)},
            "row_counts": _db_table_counts(db_path),
            "status_counts": {
                **status_counts,
                "contracts": _contract_status_counts(db_path),
            },
            "drop_counts": drop_counts,
            "summary": summary,
            "failure_diagnostics": _summarize_compile_diagnostics(db_path, run_id=run_id),
        }
    )
    _write_manifest(manifest_path, manifest)
    return manifest


def _parse_opt_used(value) -> bool:
    """Coerce optimization_used to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(value != value)
    except Exception:
        return False


def _row_get(row: Dict[str, Any], key: str, default: Any = "") -> Any:
    value = row.get(key, default)
    return default if _is_missing_value(value) else value


def _parse_runs(value: Any, default: int = 200) -> int:
    if _is_missing_value(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _open_parquet_file(local_path: str):
    """Open a Parquet file lazily so rows can be consumed in bounded batches."""
    import pyarrow.parquet as pq

    return pq.ParquetFile(local_path)


def _max_rss_mb() -> float:
    """Return best-effort process max RSS in MiB for ingestion manifests."""
    if resource is None:
        return 0.0
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports ru_maxrss in KiB; macOS reports bytes. This runner is Linux,
    # but handle both scales defensively for local developer runs.
    value = float(getattr(usage, "ru_maxrss", 0) or 0)
    if value > 1024 * 1024 * 10:
        return round(value / (1024 * 1024), 2)
    return round(value / 1024, 2)


def _token_count_estimate(text: Any) -> int:
    text = str(text or "")
    whitespace_tokens = len(text.split())
    bpeish_tokens = (len(text) + 3) // 4
    return max(whitespace_tokens, bpeish_tokens)


def _export_prompt_parts(record: Dict[str, Any]) -> Tuple[str, str, str]:
    try:
        import train

        return train._preflight_prompt_parts(
            record,
            include_bytecode_metadata=True,
            template_format="alpaca",
        )
    except Exception:
        return record.get("input", ""), record.get("output", ""), ""


def _export_length_report(record: Dict[str, Any], max_seq_length: int) -> Dict[str, Any]:
    prefix, target, suffix = _export_prompt_parts(record)
    context_tokens = _token_count_estimate(prefix)
    target_tokens = _token_count_estimate(f"{target}{suffix}")
    total_tokens = context_tokens + target_tokens
    reasons: List[str] = []
    if target_tokens >= max_seq_length:
        reasons.append("target_overlength")
    if total_tokens > max_seq_length:
        reasons.append("context_overlength")
    return {
        "context_tokens": context_tokens,
        "target_tokens": target_tokens,
        "total_tokens": total_tokens,
        "max_seq_length": max_seq_length,
        "reasons": reasons,
    }


def download_contracts(
    limit: int = 0,
    db_path: Path = DB_PATH,
    cache_dir: Optional[str] = None,
    hf_revision: Optional[str] = None,
    manifest_path: Optional[Path] = None,
    command_args: Optional[List[str]] = None,
    parquet_batch_size: int = PARQUET_READ_BATCH_SIZE,
) -> int:
    """Download contracts from HuggingFace with source-code-level dedup.

    Parquet files are cached locally (default: ~/.cache/huggingface/hub/).
    On rerun, cached files are reused without re-downloading.
    """
    started_at = _now_utc_iso()
    start_time = time.perf_counter()
    logger.info(
        "Listing Parquet files from %s (flattened/train, revision=%s)...",
        HF_DATASET_REPO,
        hf_revision or "default",
    )
    resolved_revision = _resolve_hf_revision(hf_revision)
    parquet_files = _get_parquet_files("flattened", "train", revision=hf_revision)
    if not parquet_files:
        logger.error("No Parquet files found!")
        manifest = _base_manifest(
            "hf_download",
            status="no_source_files",
            started_at=started_at,
            duration_seconds=time.perf_counter() - start_time,
            command_args=command_args,
        )
        manifest.update(
            {
                "lineage": {"source": _hf_lineage(hf_revision, resolved_revision)},
                "parameters": {
                    "limit": limit,
                    "cache_dir": cache_dir,
                    "parquet_batch_size": parquet_batch_size,
                },
                "artifacts": {"database": _artifact_summary(db_path)},
                "row_counts": _db_table_counts(db_path),
                "status_counts": {},
                "drop_counts": {"missing_parquet_files": 1},
            }
        )
        _write_manifest(
            manifest_path or Path(db_path).with_name("hf_download_manifest.json"), manifest
        )
        return 0
    logger.info(f"Found {len(parquet_files)} Parquet file(s)")

    existing_hashes = get_existing_source_hashes(db_path)
    logger.info(f"Loaded {len(existing_hashes)} existing source hashes for dedup")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    inserted = 0
    ignored = 0
    skipped = 0
    skipped_non_solidity = 0
    skipped_short_source = 0
    deduped = 0
    total_seen = 0
    download_errors: List[Dict[str, str]] = []
    parquet_streams: List[Dict[str, Any]] = []
    limit_reached = False

    try:
        for pq_idx, pq_file in enumerate(parquet_files):
            logger.info(f"Downloading Parquet file {pq_idx + 1}/{len(parquet_files)}: {pq_file}")

            try:
                local_path = hf_hub_download(
                    repo_id=HF_DATASET_REPO,
                    filename=pq_file,
                    repo_type="dataset",
                    cache_dir=cache_dir,
                    revision=hf_revision,
                )
                logger.info(f"  Cached at: {local_path}")
                parquet = _open_parquet_file(local_path)
            except Exception as e:
                logger.warning(f"Failed to download/read {pq_file}: {e}")
                download_errors.append({"parquet_file": pq_file, "error": str(e)})
                continue

            expected_rows = getattr(getattr(parquet, "metadata", None), "num_rows", None)
            logger.info(
                "  Streaming %s rows from %s in Parquet batches of %s",
                expected_rows if expected_rows is not None else "unknown",
                pq_file,
                parquet_batch_size,
            )

            batch: list = []
            file_started = time.perf_counter()
            file_seen_before = total_seen
            file_batches = 0
            try:
                for record_batch in parquet.iter_batches(batch_size=parquet_batch_size):
                    file_batches += 1
                    for row in record_batch.to_pylist():
                        total_seen += 1
                        if limit and total_seen > limit:
                            limit_reached = True
                            break

                        if str(_row_get(row, "language", "")).strip().lower() != "solidity":
                            skipped += 1
                            skipped_non_solidity += 1
                            continue

                        src = str(_row_get(row, "source_code", "") or "")
                        if len(src.strip()) < 20:
                            skipped += 1
                            skipped_short_source += 1
                            continue

                        src_hash = hash_source_code(src)
                        if src_hash in existing_hashes:
                            deduped += 1
                            continue
                        existing_hashes.add(src_hash)

                        address = str(_row_get(row, "contract_address", "") or "").strip()
                        if not address:
                            address = f"hf_{src_hash}"

                        compiler_version = str(_row_get(row, "compiler_version", "") or "")
                        opt_used = _parse_opt_used(_row_get(row, "optimization_used", False))
                        runs = _parse_runs(_row_get(row, "runs", 200))

                        batch.append(
                            (
                                address,
                                src,
                                "",
                                compiler_version,
                                opt_used,
                                runs,
                                str(_row_get(row, "abi", "") or ""),
                                str(_row_get(row, "contract_name", "") or ""),
                                "huggingface",
                                src_hash,
                            )
                        )

                        if len(batch) >= BATCH_INSERT_SIZE:
                            inserted_now = _flush_batch(cur, batch)
                            inserted += inserted_now
                            ignored += len(batch) - inserted_now
                            batch = []
                            conn.commit()

                    if limit_reached:
                        break
            except Exception as e:
                logger.warning(f"Failed while streaming {pq_file}: {e}")
                download_errors.append({"parquet_file": pq_file, "error": str(e)})

            if batch:
                inserted_now = _flush_batch(cur, batch)
                inserted += inserted_now
                ignored += len(batch) - inserted_now
                batch = []
                conn.commit()

            file_duration = time.perf_counter() - file_started
            file_rows_seen = total_seen - file_seen_before
            rows_per_second = file_rows_seen / file_duration if file_duration > 0 else 0.0
            max_rss_mb = _max_rss_mb()
            parquet_streams.append(
                {
                    "parquet_file": pq_file,
                    "expected_rows": expected_rows,
                    "rows_seen": file_rows_seen,
                    "record_batches": file_batches,
                    "duration_seconds": round(file_duration, 3),
                    "rows_per_second": round(rows_per_second, 2),
                    "batch_size": parquet_batch_size,
                    "max_rss_mb": max_rss_mb,
                }
            )
            logger.info(
                "  Streamed %s rows in %.2fs (%.1f rows/s, %s record batches, max RSS %.1f MiB)",
                file_rows_seen,
                file_duration,
                rows_per_second,
                file_batches,
                max_rss_mb,
            )
            logger.info(
                f"  Running: {inserted} inserted, {ignored} ignored, "
                f"{deduped} deduped, {skipped} skipped, {total_seen} seen"
            )
            if limit_reached:
                logger.info(f"Reached download limit of {limit}")
                break

    except KeyboardInterrupt:
        logger.warning("Download interrupted by user")

    conn.close()
    logger.info(
        f"Download complete: {inserted} stored, {ignored} ignored, "
        f"{deduped} source-deduped, {skipped} skipped, {total_seen} total"
    )

    duration_seconds = time.perf_counter() - start_time
    status = "completed" if inserted or total_seen else "completed_no_rows"
    manifest = _base_manifest(
        "hf_download",
        status=status,
        started_at=started_at,
        duration_seconds=duration_seconds,
        command_args=command_args,
    )
    manifest.update(
        {
            "lineage": {
                "source": _hf_lineage(hf_revision, resolved_revision),
                "parquet_files": parquet_files,
            },
            "parameters": {
                "limit": limit,
                "cache_dir": cache_dir,
                "parquet_batch_size": parquet_batch_size,
            },
            "artifacts": {"database": _artifact_summary(db_path)},
            "row_counts": {
                **_db_table_counts(db_path),
                "source_rows_seen": total_seen,
                "contracts_inserted": inserted,
            },
            "status_counts": {
                "inserted": inserted,
                "insert_ignored": ignored,
                "source_deduped": deduped,
                "skipped": skipped,
                "download_errors": len(download_errors),
            },
            "drop_counts": {
                "non_solidity": skipped_non_solidity,
                "short_or_empty_source": skipped_short_source,
                "source_deduped": deduped,
                "insert_ignored": ignored,
                "download_read_errors": len(download_errors),
            },
            "diagnostics": {"download_errors": download_errors[:20]},
            "performance": {
                "parquet_batch_size": parquet_batch_size,
                "rows_per_second": (
                    round(total_seen / duration_seconds, 2) if duration_seconds > 0 else 0.0
                ),
                "max_rss_mb": _max_rss_mb(),
                "parquet_streams": parquet_streams,
            },
        }
    )
    _write_manifest(manifest_path or Path(db_path).with_name("hf_download_manifest.json"), manifest)
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
        pragma_constraints = pragmas or [">=0.4.0"]

        compatible = compatible_versions_for_pragmas(pragma_constraints)
        if not compatible:
            return None

        if orig_version:
            norm_orig = _normalize_version(orig_version)
            if (
                norm_orig
                and norm_orig not in compatible
                and version_satisfies_all_pragmas(norm_orig, pragma_constraints)
            ):
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
) -> Dict[str, Any]:
    """Compile one (contract, version, optimizer) combo and return outcome.

    Quality filters applied: min body length, trivial getters, proxy forwarders.
    Compiler/source/ABI facts may be stored in metadata, but TAC prompt text is
    bytecode-only and excludes source signatures, return types, and storage
    layout names.
    """
    logging.disable(logging.CRITICAL)
    try:
        if not install_solc_version(solc_version):
            return {
                "pairs": [],
                "status": "compile_failed",
                "error": f"failed to install solc {solc_version}",
            }

        try:
            if len(source_files) > 1:
                comp = compile_multi_file(source_files, solc_version, opt_enabled, runs)
            else:
                first_src = next(iter(source_files.values()))
                comp = compile_source(first_src, solc_version, opt_enabled, runs)
        except Exception as e:
            return {"pairs": [], "status": "compile_failed", "error": str(e)}

        if not comp.success:
            return {
                "pairs": [],
                "status": "compile_failed",
                "error": "; ".join(comp.errors) if comp.errors else "solc compilation failed",
            }

        pairs: List[Dict] = []
        analysis_errors: List[str] = []
        for cname, compiled in comp.contracts.items():
            bytecode_hex = "0x" + compiled.runtime_bytecode
            if len(bytecode_hex) < 10:
                continue

            try:
                analyzer = BytecodeAnalyzer(bytecode_hex)
                analyzer.analyze_control_flow()
                bytecode_functions = analyzer.identify_functions()
            except Exception as e:
                analysis_errors.append(str(e))
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

        if pairs:
            return {"pairs": pairs, "status": "processed", "error": ""}
        if analysis_errors:
            return {
                "pairs": [],
                "status": "analysis_failed",
                "error": "; ".join(analysis_errors[:3]),
            }
        return {"pairs": [], "status": "no_pairs", "error": "no matched functions"}
    except Exception as e:
        return {"pairs": [], "status": "compile_failed", "error": str(e)}


def compile_and_generate(
    max_compiler_versions: int = 0,
    workers: int = 0,
    max_body_dupes: int = 5,
    min_body_length: int = 50,
    db_path: Path = DB_PATH,
    manifest_path: Optional[Path] = None,
    command_args: Optional[List[str]] = None,
    hf_revision: Optional[str] = None,
    run_id: Optional[str] = None,
) -> int:
    """Compile unprocessed contracts and generate TAC->Solidity pairs.

    Dedup is handled deterministically by DB unique indexes and
    ROW_NUMBER() at export time (not in-memory frequency caps).
    """
    started_at = _now_utc_iso()
    start_time = time.perf_counter()
    run_id = run_id or _make_run_id("compile")
    with _db_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT address, source_code, compiler_version, "
            "optimization_enabled, optimization_runs "
            "FROM contracts WHERE processed = FALSE"
        ).fetchall()

    total_contracts = len(rows)
    logger.info(f"Found {total_contracts} unprocessed contracts to compile")
    if total_contracts == 0:
        _write_compile_manifest(
            db_path,
            manifest_path or Path(db_path).with_name("hf_compile_manifest.json"),
            run_id=run_id,
            status="skipped_no_unprocessed_contracts",
            started_at=started_at,
            duration_seconds=time.perf_counter() - start_time,
            parameters={
                "max_compiler_versions": max_compiler_versions,
                "workers": workers,
                "max_body_dupes": max_body_dupes,
                "min_body_length": min_body_length,
            },
            summary={
                "contracts_selected": 0,
                "contracts_prepared": 0,
                "compile_jobs": 0,
                "pairs_generated": 0,
                "pairs_inserted": 0,
                "db_pair_duplicates": 0,
            },
            status_counts={},
            drop_counts={},
            command_args=command_args,
            hf_revision=hf_revision,
        )
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
                    try:
                        result = fut.result()
                    except Exception:
                        result = None
                    if result:
                        prepared[addr] = result
                    else:
                        no_work_addresses.append(addr)
                    pbar.update(1)

    # Track contracts with no compilable work without marking them successful.
    if no_work_addresses:
        _mark_contract_status(
            db_path,
            no_work_addresses,
            "no_pairs",
            processed=False,
            last_error="prepare produced no compile jobs",
        )
        _store_compile_diagnostics(
            db_path,
            [
                {
                    "run_id": run_id,
                    "contract_address": addr,
                    "status": "prepare_no_jobs",
                    "error": "prepare produced no compile jobs",
                }
                for addr in no_work_addresses
            ],
        )

    # Build compile jobs: each (contract x version x optimizer_flag).
    # Verified ABI/source details stay out of TAC prompt inputs.
    compile_jobs = [
        (
            addr,
            prep["source_files"],
            prep["solidity_functions"],
            ver,
            opt,
            prep["runs"],
            min_body_length,
        )
        for addr, prep in sorted(prepared.items())
        for ver in prep["compatible_versions"]
        for opt in (True, False)
    ]

    total_jobs = len(compile_jobs)
    logger.info(
        f"Phase 2b: {len(prepared)} contracts -> {total_jobs} compile jobs "
        f"({total_jobs // max(len(prepared), 1)} avg jobs/contract)"
    )
    if total_jobs == 0:
        outcome_counts = _record_contract_outcomes(
            db_path,
            {addr: {"pairs": 0, "failures": []} for addr in prepared},
        )
        _write_compile_manifest(
            db_path,
            manifest_path or Path(db_path).with_name("hf_compile_manifest.json"),
            run_id=run_id,
            status="completed_no_compile_jobs",
            started_at=started_at,
            duration_seconds=time.perf_counter() - start_time,
            parameters={
                "max_compiler_versions": max_compiler_versions,
                "workers": workers,
                "max_body_dupes": max_body_dupes,
                "min_body_length": min_body_length,
            },
            summary={
                "contracts_selected": total_contracts,
                "contracts_prepared": len(prepared),
                "prepare_no_jobs": len(no_work_addresses),
                "compile_jobs": 0,
                "pairs_generated": 0,
                "pairs_inserted": 0,
                "db_pair_duplicates": 0,
            },
            status_counts={"contract_outcomes": outcome_counts, "compile_jobs": {}},
            drop_counts={
                "prepare_no_jobs": len(no_work_addresses),
                "compile_or_analysis_errors": 0,
                "db_pair_duplicates": 0,
            },
            command_args=command_args,
            hf_revision=hf_revision,
        )
        return 0

    # Phase 2b: compile in parallel (batched to avoid memory/IPC flood)
    total_pairs = 0
    total_inserted = 0
    total_db_deduped = 0
    errors_count = 0
    pair_buffer: List[Dict] = []
    diagnostics_buffer: List[Dict[str, Any]] = []
    job_status_counts: Dict[str, int] = {}
    contract_outcomes: Dict[str, Dict[str, Any]] = {
        addr: {"pairs": 0, "failures": []} for addr in prepared
    }

    with ProcessPoolExecutor(max_workers=workers) as executor:
        with tqdm(total=total_jobs, desc="Compiling variants", unit="job") as pbar:
            for batch_start in range(0, total_jobs, SUBMIT_BATCH):
                batch_jobs = compile_jobs[batch_start : batch_start + SUBMIT_BATCH]
                futures = {
                    executor.submit(_compile_one_job, *job): (batch_start + idx, job)
                    for idx, job in enumerate(batch_jobs)
                }
                batch_results: Dict[int, Tuple[tuple, Dict[str, Any]]] = {}

                for fut in as_completed(futures):
                    job_idx, job = futures[fut]
                    try:
                        result = fut.result()
                        if isinstance(result, list):
                            result = {
                                "pairs": result,
                                "status": "processed" if result else "no_pairs",
                                "error": "",
                            }
                        batch_results[job_idx] = (job, result)
                    except Exception as e:
                        batch_results[job_idx] = (
                            job,
                            {"pairs": [], "status": "compile_failed", "error": str(e)},
                        )

                    pbar.set_postfix(pairs=total_pairs, errors=errors_count, refresh=False)
                    pbar.update(1)

                for job_idx in sorted(batch_results):
                    job, result = batch_results[job_idx]
                    addr = job[0]
                    pairs = result.get("pairs", []) or []
                    status = result.get("status", "no_pairs")
                    job_status_counts[status] = job_status_counts.get(status, 0) + 1
                    if pairs:
                        pair_buffer.extend(pairs)
                        total_pairs += len(pairs)
                        contract_outcomes[addr]["pairs"] += len(pairs)
                    elif status not in ("no_pairs", "processed"):
                        errors_count += 1
                        contract_outcomes[addr]["failures"].append(
                            {"status": status, "error": result.get("error", "")}
                        )
                        diagnostics_buffer.append(
                            {
                                "run_id": run_id,
                                "contract_address": addr,
                                "compiler_version": job[3],
                                "optimizer_enabled": job[4],
                                "optimization_runs": job[5],
                                "status": status,
                                "error": result.get("error", ""),
                            }
                        )

                    if len(pair_buffer) >= FLUSH_SIZE:
                        inserted = _store_pairs_batch(db_path, pair_buffer)
                        total_inserted += inserted
                        total_db_deduped += len(pair_buffer) - inserted
                        pair_buffer = []
                    if len(diagnostics_buffer) >= FLUSH_SIZE:
                        _store_compile_diagnostics(db_path, diagnostics_buffer)
                        diagnostics_buffer = []

    if pair_buffer:
        inserted = _store_pairs_batch(db_path, pair_buffer)
        total_inserted += inserted
        total_db_deduped += len(pair_buffer) - inserted
    if diagnostics_buffer:
        _store_compile_diagnostics(db_path, diagnostics_buffer)

    outcome_counts = _record_contract_outcomes(db_path, contract_outcomes)

    # Harvest selectors from all prepared contracts into the registry
    all_selectors: List[tuple] = []
    for addr in sorted(prepared):
        prep = prepared[addr]
        for func in prep.get("solidity_functions", []):
            sel = func.get("selector")
            if not sel:
                continue
            sig = func.get("signature", "")
            parsed = _extract_function_signature_parts(sig)
            if parsed:
                func_name, params_str = parsed
                if params_str:
                    param_types = _parse_solidity_param_types(params_str)
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
    logger.info("Contract outcomes: %s", outcome_counts)
    _write_compile_manifest(
        db_path,
        manifest_path or Path(db_path).with_name("hf_compile_manifest.json"),
        run_id=run_id,
        status="completed_with_errors" if errors_count else "completed",
        started_at=started_at,
        duration_seconds=time.perf_counter() - start_time,
        parameters={
            "max_compiler_versions": max_compiler_versions,
            "workers": workers,
            "max_body_dupes": max_body_dupes,
            "min_body_length": min_body_length,
        },
        summary={
            "contracts_selected": total_contracts,
            "contracts_prepared": len(prepared),
            "prepare_no_jobs": len(no_work_addresses),
            "compile_jobs": total_jobs,
            "pairs_generated": total_pairs,
            "pairs_inserted": total_inserted,
            "db_pair_duplicates": total_db_deduped,
            "selectors_harvested": len(all_selectors),
        },
        status_counts={
            "compile_jobs": job_status_counts,
            "contract_outcomes": outcome_counts,
        },
        drop_counts={
            "prepare_no_jobs": len(no_work_addresses),
            "no_pair_jobs": job_status_counts.get("no_pairs", 0),
            "compile_or_analysis_errors": errors_count,
            "db_pair_duplicates": total_db_deduped,
        },
        command_args=command_args,
        hf_revision=hf_revision,
    )
    return total_pairs


# ---------------------------------------------------------------------------
# Pair building, matching & storage
# ---------------------------------------------------------------------------


def _parse_solidity_param_types(params_str: str) -> List[str]:
    """Parse Solidity parameter types with balanced-parenthesis awareness.

    Correctly handles tuple types like ``(uint256,address)[]``,
    nested tuples ``(uint256,(address,bool))``, and fixed-size arrays
    ``uint256[3]``. Only splits on commas that are at depth 0 (not
    inside parentheses).
    """
    if not params_str or not params_str.strip():
        return []

    # Split on commas at depth 0 only
    params: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in params_str:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            params.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        params.append("".join(current).strip())

    # Extract the type from each parameter (first whitespace-delimited token,
    # but keep tuple parentheses and array brackets together)
    types: List[str] = []
    for param in params:
        param = param.strip()
        if not param:
            continue
        if param.startswith("("):
            # Tuple type — find the end of the tuple (including trailing [] etc.)
            d = 0
            end = 0
            for i, c in enumerate(param):
                if c == "(":
                    d += 1
                elif c == ")":
                    d -= 1
                    if d == 0:
                        end = i + 1
                        break
            # Include trailing array brackets like [] or [3]
            while end < len(param) and param[end] in "[]0123456789":
                end += 1
            types.append(canonicalize_abi_type(param[:end]))
        else:
            # Regular type — first whitespace token, but include [] brackets
            parts = param.split()
            type_token = parts[0]
            # Handle "uint256 [3]" (space before bracket) — merge next part
            if len(parts) > 1 and parts[1].startswith("["):
                type_token += parts[1]
            types.append(canonicalize_abi_type(type_token))

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


def _add_selectors(functions: List[Dict]) -> List[Dict]:
    """Compute 4-byte function selectors for each Solidity function.

    Uses balanced-parenthesis-aware parameter parsing (Issue 9) to correctly
    handle tuple types and nested tuples.
    """
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

            func["selector"] = normalize_hex(Web3.keccak(text=canonical)[:4])
        except Exception:
            func["selector"] = None
    return functions


def _match_functions(
    solidity_functions: List[Dict],
    bytecode_functions: Dict,
    analyzer,
    compiler_version: str = "",
    optimizer_enabled: bool = False,
    abi_enricher: Optional[Any] = None,
    storage_resolver: Optional[Any] = None,
) -> List[Dict]:
    """Match Solidity labels to bytecode functions by selector.

    Source/ABI/compiler arguments are accepted for backwards compatibility but
    are intentionally ignored when constructing TAC prompt text.
    """
    _ensure_tac_integrated(analyzer)
    sol_by_sel = {f["selector"]: f for f in solidity_functions if f.get("selector")}
    bc_by_sel = {f.selector: f for f in bytecode_functions.values() if f.selector}

    matches = []
    for selector, sol_func in sol_by_sel.items():
        if selector in bc_by_sel:
            tac = _extract_tac(bc_by_sel[selector], analyzer)
            matches.append(
                {
                    "solidity_function": sol_func,
                    "bytecode_function": bc_by_sel[selector],
                    "tac": tac,
                    "selector": selector,
                }
            )
    return matches


def _ensure_tac_integrated(analyzer) -> None:
    """Populate analyzer basic blocks with TAC instructions once."""
    blocks = getattr(analyzer, "basic_blocks", {}) or {}
    if not blocks:
        return
    if any(getattr(block, "instructions", None) for block in blocks.values()):
        return
    if not any(
        (getattr(block, "metadata", {}) or {}).get("raw_instructions") for block in blocks.values()
    ):
        return
    converter = getattr(analyzer, "_convert_and_integrate_tac", None)
    if callable(converter):
        converter()


def _extract_tac(
    bytecode_function, analyzer, compiler_version: str = "", optimizer_enabled: bool = False
) -> str:
    """Generate bytecode-only TAC text for a single bytecode function.

    Compiler/optimizer arguments are accepted for compatibility, but they are
    not emitted because production inference only has bytecode.
    """
    lines: List[str] = []
    try:
        _ensure_tac_integrated(analyzer)
        lines.append(f"function {_safe_tac_function_name(bytecode_function)}:")
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
    """Build one training pair with bytecode-only TAC and analysis metadata."""
    sol_func = match["solidity_function"]
    tac = sanitize_tac_prompt_input(match["tac"])

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
        "metadata": json.dumps(
            {
                "contract_name": sol_func.get("contract_name"),
                "selector": match["selector"],
                "compiler_version": compiler_version,
                "optimizer_enabled": optimizer_enabled,
                "optimizer_runs": optimizer_runs,
                "compiled_contract": compiled_contract,
                "source": "huggingface",
            }
        ),
        "hash": _md5(tac + sol_func["body"]),
        "body_hash": hash_normalized_body(sol_func["body"]),
        "tac_hash": hash_normalized_tac(tac),
        "pair_norm_hash": hash_normalized_pair(tac, sol_func["body"]),
    }


def _store_pairs_batch(db_path: Path, pairs: List[Dict]) -> int:
    """Insert pairs into the DB, ignoring duplicates. Returns rows inserted.

    TAC prompt text is sanitized before storage; compiler/source-only facts stay
    in JSON metadata for analysis.
    """
    if not pairs:
        return 0

    values = []
    for p in pairs:
        tac = sanitize_tac_prompt_input(p["tac_representation"])
        solidity = p["solidity_code"]
        values.append(
            (
                p["contract_address"],
                p["function_name"],
                tac,
                solidity,
                p["function_signature"],
                p["visibility"],
                p["is_payable"],
                p["is_view"],
                p["metadata"],
                _md5(tac + solidity),
                hash_normalized_body(solidity),
                hash_normalized_tac(tac),
                hash_normalized_pair(tac, solidity),
            )
        )

    with _db_connection(db_path) as conn:
        cur = conn.cursor()
        before = cur.execute("SELECT COUNT(*) FROM function_pairs").fetchone()[0]
        cur.executemany(
            """
            INSERT OR IGNORE INTO function_pairs
                (contract_address, function_name, tac_representation,
                 solidity_code, function_signature, visibility,
                 is_payable, is_view, metadata, hash, body_hash,
                 tac_hash, pair_norm_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            values,
        )
        conn.commit()
        after = cur.execute("SELECT COUNT(*) FROM function_pairs").fetchone()[0]

    return after - before


def validate_jsonl_body_duplicate_cap(
    jsonl_path: Path,
    max_body_dupes: int = 2,
    sample_limit: int = 5,
) -> Dict[str, Any]:
    """Recompute normalized target-body duplicates directly from exported JSONL."""
    path = Path(jsonl_path)
    counts: Counter[str] = Counter()
    samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    invalid_records: List[Dict[str, Any]] = []
    missing_targets: List[Dict[str, Any]] = []
    row_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_records.append({"line_number": line_number, "error": str(exc)})
                continue
            target = record.get("output", record.get("solidity_code"))
            if not isinstance(target, str):
                missing_targets.append({"line_number": line_number})
                continue
            normalized = normalize_solidity_body(target)
            body_hash = _md5(normalized)
            counts[body_hash] += 1
            if len(samples[body_hash]) < sample_limit:
                metadata = record.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                samples[body_hash].append(
                    {
                        "line_number": line_number,
                        "function_name": metadata.get("function_name"),
                        "function_signature": metadata.get("function_signature"),
                        "contract_address": metadata.get("contract_address"),
                        "output_preview": _collapse_whitespace(target)[:200],
                    }
                )

    violations = []
    for body_hash, count in counts.items():
        if count > max_body_dupes:
            violations.append(
                {
                    "body_hash": body_hash,
                    "count": count,
                    "over_cap_by": count - max_body_dupes,
                    "samples": samples[body_hash],
                }
            )
    violations.sort(key=lambda item: (-item["count"], item["body_hash"]))

    status = "passed"
    if invalid_records or missing_targets:
        status = "invalid"
    elif violations:
        status = "failed"

    return {
        "path": str(path),
        "status": status,
        "max_body_dupes": max_body_dupes,
        "rows_checked": row_count,
        "unique_normalized_bodies": len(counts),
        "duplicate_body_rows": sum(max(count - 1, 0) for count in counts.values()),
        "body_hashes_over_cap": len(violations),
        "rows_over_cap": sum(item["over_cap_by"] for item in violations),
        "violations": violations[:sample_limit],
        "invalid_records": invalid_records[:sample_limit],
        "missing_targets": missing_targets[:sample_limit],
    }


def format_duplicate_cap_error(validation: Dict[str, Any]) -> str:
    """Format duplicate-cap validation failures with actionable samples."""
    path = validation.get("path", "<jsonl>")
    cap = validation.get("max_body_dupes")
    lines = [
        f"{path}: normalized body duplicate-cap validation {validation.get('status')} "
        f"(cap={cap})"
    ]
    if validation.get("invalid_records"):
        lines.append("Invalid JSONL records:")
        for item in validation["invalid_records"]:
            lines.append(f"  line {item['line_number']}: {item['error']}")
    if validation.get("missing_targets"):
        lines.append("Rows without an output/solidity_code target:")
        for item in validation["missing_targets"]:
            lines.append(f"  line {item['line_number']}")
    for idx, item in enumerate(validation.get("violations", []), start=1):
        lines.append(
            f"  {idx}. body_hash={item['body_hash']} count={item['count']} "
            f"over_cap_by={item['over_cap_by']}"
        )
        for sample in item.get("samples", []):
            label = sample.get("function_signature") or sample.get("function_name") or "<unknown>"
            lines.append(
                f"     line {sample['line_number']}: {label} "
                f"{sample.get('output_preview', '')[:120]}"
            )
    return "\n".join(lines)


def enforce_jsonl_body_duplicate_cap(
    jsonl_path: Path,
    max_body_dupes: int = 2,
    sample_limit: int = 5,
) -> Dict[str, Any]:
    """Validate an exported JSONL body cap and raise with top samples on failure."""
    validation = validate_jsonl_body_duplicate_cap(jsonl_path, max_body_dupes, sample_limit)
    if validation["status"] != "passed":
        raise ValueError(format_duplicate_cap_error(validation))
    return validation


# ---------------------------------------------------------------------------
# Phase 3 -- Export training data
# ---------------------------------------------------------------------------


def export_training_data(
    output_path: str = "data/hf_training_dataset.jsonl",
    max_body_dupes: int = 5,
    db_path: Path = DB_PATH,
    hf_revision: Optional[str] = None,
    manifest_path: Optional[Path] = None,
    command_args: Optional[List[str]] = None,
    validate_body_dupes: bool = True,
    max_seq_length: int = DEFAULT_EXPORT_MAX_SEQ_LENGTH,
    filter_overlength: bool = True,
    rejects_path: Optional[Path] = None,
) -> str:
    """Export deterministic JSONL with bytecode-only inputs and metadata.

    Compiler/source facts may remain in ``metadata`` for analysis, but ``input``
    is sanitized TAC built only from runtime bytecode facts.
    """
    started_at = _now_utc_iso()
    start_time = time.perf_counter()
    selection_stats = _export_selection_stats(db_path, max_body_dupes=max_body_dupes)
    with _db_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT function_name, tac_representation, solidity_code,
                   function_signature, visibility, is_payable, is_view,
                   contract_address, metadata, body_hash
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY COALESCE(body_hash, hash)
                           ORDER BY COALESCE(pair_norm_hash, ''),
                                    COALESCE(tac_hash, ''),
                                    COALESCE(contract_address, ''),
                                    COALESCE(function_signature, ''),
                                    COALESCE(function_name, ''),
                                    id
                       ) AS body_rn
                FROM (
                    SELECT *,
                          ROW_NUMBER() OVER (
                              PARTITION BY COALESCE(pair_norm_hash, hash)
                              ORDER BY COALESCE(body_hash, ''),
                                       COALESCE(tac_hash, ''),
                                       COALESCE(contract_address, ''),
                                       COALESCE(function_signature, ''),
                                       COALESCE(function_name, ''),
                                       id
                          ) AS pair_rn
                    FROM function_pairs
                )
                WHERE pair_rn = 1
            )
            WHERE body_rn <= ?
            ORDER BY COALESCE(body_hash, ''),
                     COALESCE(pair_norm_hash, ''),
                     COALESCE(tac_hash, ''),
                     COALESCE(contract_address, ''),
                     COALESCE(function_signature, ''),
                     COALESCE(function_name, '')
        """,
            (max_body_dupes,),
        ).fetchall()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    reject_artifact_path = (
        Path(rejects_path) if rejects_path else Path(f"{output_path}.rejects.jsonl")
    )
    reject_artifact_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    rejected_count = 0
    overlength_counts: Counter = Counter()
    solc_versions: Set[str] = set()
    reject_samples: List[Dict[str, Any]] = []
    with (
        open(output_path, "w", encoding="utf-8") as f,
        open(reject_artifact_path, "w", encoding="utf-8") as reject_f,
    ):
        for source_index, (
            func_name,
            tac,
            solidity,
            sig,
            vis,
            is_pay,
            is_view,
            addr,
            meta_str,
            body_hash,
        ) in enumerate(rows, start=1):
            prompt_input = sanitize_tac_prompt_input(tac)
            metadata = {
                "function_name": func_name,
                "function_signature": sig,
                "visibility": vis,
                "is_payable": bool(is_pay),
                "is_view": bool(is_view),
                "body_hash": body_hash,
            }
            if isinstance(addr, str) and ETH_ADDRESS_RE.match(addr):
                metadata["contract_address"] = addr
            else:
                metadata["contract_id"] = addr
            if meta_str:
                try:
                    parsed_meta = json.loads(meta_str)
                    if isinstance(parsed_meta, dict):
                        metadata.update(parsed_meta)
                        compiler_version = parsed_meta.get("compiler_version")
                        if compiler_version:
                            solc_versions.add(compiler_version)
                except json.JSONDecodeError:
                    pass
            record = {
                "input": prompt_input,
                "output": solidity,
                "metadata": normalize_training_metadata(metadata),
            }

            length_report = _export_length_report(record, max_seq_length)
            if filter_overlength and length_report["reasons"]:
                rejected_count += 1
                for reason in length_report["reasons"]:
                    overlength_counts[reason] += 1
                reject_record = {
                    "source_row_number": source_index,
                    "contract_address": addr,
                    "function_name": func_name,
                    "function_signature": sig,
                    "body_hash": body_hash,
                    "reasons": length_report["reasons"],
                    "lengths": {
                        key: value for key, value in length_report.items() if key != "reasons"
                    },
                }
                reject_f.write(json.dumps(reject_record, sort_keys=True) + "\n")
                if len(reject_samples) < 10:
                    reject_samples.append(reject_record)
                continue

            f.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1

    logger.info(
        "Exported %s training pairs to %s (%s overlength rows quarantined at %s)",
        count,
        output_path,
        rejected_count,
        reject_artifact_path,
    )

    # Report dedup stats
    duplicate_stats = _db_duplicate_stats(db_path, max_body_dupes=max_body_dupes)
    total_in_db = duplicate_stats["total_pairs"]
    unique_pairs = duplicate_stats["unique_pairs"]
    unique_bodies = duplicate_stats["unique_bodies"]
    unique_tacs = duplicate_stats["unique_tacs"]

    logger.info(
        f"Dedup stats: {total_in_db} total in DB, {unique_pairs} unique pairs, "
        f"{unique_bodies} unique bodies, {unique_tacs} unique TACs, "
        f"{count} exported (max {max_body_dupes} per normalized body)"
    )
    validation_stats = (
        validate_jsonl_body_duplicate_cap(output_path, max_body_dupes=max_body_dupes)
        if validate_body_dupes
        else {"status": "skipped", "max_body_dupes": max_body_dupes}
    )
    manifest_status = (
        "completed" if validation_stats["status"] in ("passed", "skipped") else "failed"
    )
    manifest = _base_manifest(
        "hf_export",
        status=manifest_status,
        started_at=started_at,
        duration_seconds=time.perf_counter() - start_time,
        command_args=command_args,
    )
    manifest.update(
        {
            "lineage": {
                "source": _hf_lineage(hf_revision),
                "database": str(db_path),
                "source_counts": _contract_source_counts(db_path),
            },
            "parameters": {
                "output_path": output_path,
                "max_body_dupes": max_body_dupes,
                "validate_body_dupes": validate_body_dupes,
                "max_seq_length": max_seq_length,
                "filter_overlength": filter_overlength,
            },
            "artifacts": {
                "jsonl": _artifact_summary(output_path, row_count=count),
                "rejects_jsonl": _artifact_summary(reject_artifact_path, row_count=rejected_count),
                "database": _artifact_summary(db_path),
            },
            "row_counts": {
                **_db_table_counts(db_path),
                "rows_exported": count,
                "rows_rejected_overlength": rejected_count,
            },
            "status_counts": {
                "contracts": _contract_status_counts(db_path),
                "validation": {validation_stats["status"]: 1},
            },
            "drop_counts": {
                "pair_duplicate_rows": selection_stats["pair_duplicate_rows_dropped"],
                "body_cap_rows": selection_stats["body_cap_rows_dropped"],
                "overlength_rows": rejected_count,
                **{f"overlength_{reason}": count for reason, count in overlength_counts.items()},
            },
            "duplicate_stats": duplicate_stats,
            "export_selection": selection_stats,
            "validation": {
                "body_duplicate_cap": validation_stats,
                "token_length_filter": {
                    "status": "filtered" if rejected_count else "passed",
                    "max_seq_length": max_seq_length,
                    "filter_overlength": filter_overlength,
                    "reject_count": rejected_count,
                    "reason_counts": dict(sorted(overlength_counts.items())),
                    "reject_samples": reject_samples,
                },
            },
            "training_row_schema_version": TRAINING_ROW_SCHEMA_VERSION,
            "solc_versions": sorted(solc_versions),
        }
    )
    _write_manifest(manifest_path or Path(f"{output_path}.manifest.json"), manifest)
    if validation_stats["status"] not in ("passed", "skipped"):
        raise ValueError(format_duplicate_cap_error(validation_stats))
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
    parser.add_argument(
        "--download-only", action="store_true", help="Only download contracts, skip compilation."
    )
    parser.add_argument(
        "--compile-only", action="store_true", help="Only compile already-downloaded contracts."
    )
    parser.add_argument(
        "--export-only", action="store_true", help="Only export existing pairs to JSONL."
    )
    parser.add_argument("--limit", type=int, default=0, help="Max contracts to download (0 = all).")
    parser.add_argument(
        "--max-compiler-versions",
        type=int,
        default=5,
        help="Max solc versions per contract (default: 5). "
        "Each version is compiled with optimizer on+off, "
        "so 5 versions = up to 10 compile jobs per contract.",
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Parallel workers (0 = auto-detect CPU count)."
    )
    parser.add_argument(
        "--max-body-dupes",
        type=int,
        default=2,
        help="Max copies of any normalized function body (default: 2).",
    )
    parser.add_argument(
        "--min-body-length",
        type=int,
        default=50,
        help="Min Solidity body length to keep (default: 50 chars).",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None, help="HuggingFace cache dir for Parquet files."
    )
    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=PARQUET_READ_BATCH_SIZE,
        help="Rows per PyArrow RecordBatch while streaming Parquet downloads.",
    )
    parser.add_argument(
        "--hf-revision",
        type=str,
        default=None,
        help="HuggingFace dataset revision/commit to pin downloads.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/hf_training_dataset.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_EXPORT_MAX_SEQ_LENGTH,
        help="Default training context budget used for export-time filtering.",
    )
    parser.add_argument(
        "--no-filter-overlength",
        action="store_true",
        help="Do not quarantine rows that exceed --max-seq-length.",
    )
    parser.add_argument(
        "--rejects-output",
        type=str,
        default=None,
        help="Optional JSONL path for export-time rejected rows.",
    )
    parser.add_argument(
        "--export-selectors",
        type=str,
        default=None,
        help="Export selector registry to JSON file (e.g. data/selectors.json).",
    )
    parser.add_argument(
        "--import-selectors",
        type=str,
        default=None,
        help="Import selector registry from JSON file before processing.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default=None,
        help="Directory for phase manifests (download/compile/export).",
    )
    parser.add_argument(
        "--validate-jsonl",
        type=str,
        default=None,
        help="Validate an exported JSONL's normalized body duplicate cap and exit.",
    )
    parser.add_argument(
        "--duplicate-sample-limit",
        type=int,
        default=5,
        help="Number of duplicate-cap violation samples to report.",
    )
    parser.add_argument("--db", type=str, default="data/contracts.db", help="SQLite database path.")

    args = parser.parse_args()
    db_path = Path(args.db)
    command_args = sys.argv[1:]
    if args.parquet_batch_size < 1:
        raise SystemExit("--parquet-batch-size must be at least 1")
    if args.max_seq_length < 1:
        raise SystemExit("--max-seq-length must be at least 1")

    setup_logging()

    if args.validate_jsonl:
        validation = validate_jsonl_body_duplicate_cap(
            args.validate_jsonl,
            max_body_dupes=args.max_body_dupes,
            sample_limit=args.duplicate_sample_limit,
        )
        if validation["status"] != "passed":
            logger.error(format_duplicate_cap_error(validation))
            raise SystemExit(1)
        logger.info(
            "Duplicate-cap validation passed: %s rows, %s unique normalized bodies, cap=%s",
            validation["rows_checked"],
            validation["unique_normalized_bodies"],
            validation["max_body_dupes"],
        )
        return

    logger.info("=" * 70)
    logger.info("Smart Contract HuggingFace Dataset -> Training Data Pipeline")
    logger.info("=" * 70)

    init_database(db_path)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else None
    if manifest_dir:
        manifest_dir.mkdir(parents=True, exist_ok=True)
    download_manifest = manifest_dir / "hf_download_manifest.json" if manifest_dir else None
    compile_manifest = manifest_dir / "hf_compile_manifest.json" if manifest_dir else None
    export_manifest = manifest_dir / "hf_export_manifest.json" if manifest_dir else None

    # Import selectors if requested
    if args.import_selectors:
        import_selector_registry(args.import_selectors, db_path)

    # Phase 1: Download
    if not args.compile_only and not args.export_only:
        logger.info("\n--- Phase 1: Download from HuggingFace ---")
        before = count_contracts(db_path)
        download_contracts(
            limit=args.limit,
            db_path=db_path,
            cache_dir=args.cache_dir,
            hf_revision=args.hf_revision,
            manifest_path=download_manifest,
            command_args=command_args,
            parquet_batch_size=args.parquet_batch_size,
        )
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
                manifest_path=compile_manifest,
                command_args=command_args,
                hf_revision=args.hf_revision,
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
            hf_revision=args.hf_revision,
            manifest_path=export_manifest,
            command_args=command_args,
            max_seq_length=args.max_seq_length,
            filter_overlength=not args.no_filter_overlength,
            rejects_path=Path(args.rejects_output) if args.rejects_output else None,
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
