"""
TAC Hash Lookup Database

Provides a normalized SQLite database for exact-match TAC→Solidity lookups
at inference time.  This is separate from the training database and is
designed for maximum coverage (every compiler version, no quality filters).

Schema:
  solidity_bodies  — unique function bodies (stored once)
  tac_to_body      — many TAC hashes map to one body (many-to-one)

Usage at inference time:
    lookup = TACLookup("data/tac_lookup.db")
    result = lookup.query(tac_text)
    if result:
        print(result["solidity"])   # exact match, skip inference
"""

import hashlib
import logging
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalization (must match download_hf_contracts.py logic exactly)
# ---------------------------------------------------------------------------

def _strip_comments(text: str) -> str:
    """Remove single-line and multi-line comments."""
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_tac(tac: str) -> str:
    """Normalize TAC for dedup hashing.

    Strips comments (which may contain compiler version info),
    collapses whitespace, and lowercases so that identical bytecode
    compiled by different solc versions produces the same hash.
    """
    text = re.sub(r"//[^\n]*", "", tac)
    return _collapse_whitespace(text).lower()


def normalize_solidity_body(body: str) -> str:
    """Normalize a Solidity function body for dedup hashing."""
    return _collapse_whitespace(_strip_comments(body)).lower()


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def hash_normalized_tac(tac: str) -> str:
    return _md5(normalize_tac(tac))


def hash_normalized_body(body: str) -> str:
    return _md5(normalize_solidity_body(body))


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

@contextmanager
def _db_connection(db_path: Path):
    """Context manager for SQLite connections with WAL mode."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# TACLookup — query interface (used by web server)
# ---------------------------------------------------------------------------

class TACLookup:
    """Fast TAC hash → Solidity lookup backed by a SQLite database.

    Designed to be instantiated once at server startup and reused for
    every inference request.
    """

    def __init__(self, db_path: str = "data/tac_lookup.db"):
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

        if not self.db_path.exists():
            logger.warning("TAC lookup DB not found at %s — lookups disabled", db_path)
            return

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row

        count = self._conn.execute("SELECT COUNT(*) FROM tac_to_body").fetchone()[0]
        body_count = self._conn.execute("SELECT COUNT(*) FROM solidity_bodies").fetchone()[0]
        logger.info(
            "TAC lookup DB loaded: %d TAC hashes → %d unique bodies (%s)",
            count, body_count, db_path,
        )

    @property
    def available(self) -> bool:
        """Whether the lookup database is loaded and ready."""
        return self._conn is not None

    def query(self, tac_text: str) -> Optional[Dict]:
        """Look up a TAC string and return the matching Solidity body.

        Args:
            tac_text: Raw TAC representation of a single function.

        Returns:
            Dict with keys: solidity, function_name, selector,
            function_signature, visibility, is_payable, is_view,
            compiler_version, occurrences, source.
            Returns None if no match.
        """
        if not self._conn:
            return None

        tac_hash = hash_normalized_tac(tac_text)

        row = self._conn.execute("""
            SELECT sb.solidity_code, sb.function_name, sb.selector,
                   sb.function_signature, sb.visibility,
                   sb.is_payable, sb.is_view,
                   tb.compiler_version, tb.occurrences
            FROM tac_to_body tb
            JOIN solidity_bodies sb ON tb.body_id = sb.body_id
            WHERE tb.tac_hash = ?
        """, (tac_hash,)).fetchone()

        if not row:
            return None

        return {
            "solidity": row["solidity_code"],
            "function_name": row["function_name"],
            "selector": row["selector"],
            "function_signature": row["function_signature"],
            "visibility": row["visibility"],
            "is_payable": bool(row["is_payable"]),
            "is_view": bool(row["is_view"]),
            "compiler_version": row["compiler_version"],
            "occurrences": row["occurrences"],
            "source": "exact_match",
        }

    def query_batch(self, tac_texts: List[str]) -> Dict[int, Dict]:
        """Look up multiple TAC strings at once.

        Args:
            tac_texts: List of TAC strings.

        Returns:
            Dict mapping index → result dict (only for matches).
        """
        results: Dict[int, Dict] = {}
        for i, tac in enumerate(tac_texts):
            result = self.query(tac)
            if result:
                results[i] = result
        return results

    def stats(self) -> Dict:
        """Return summary statistics about the lookup database."""
        if not self._conn:
            return {"available": False}

        tac_count = self._conn.execute("SELECT COUNT(*) FROM tac_to_body").fetchone()[0]
        body_count = self._conn.execute("SELECT COUNT(*) FROM solidity_bodies").fetchone()[0]
        total_occ = self._conn.execute(
            "SELECT SUM(occurrences) FROM tac_to_body"
        ).fetchone()[0] or 0
        avg_tac_per_body = round(tac_count / max(body_count, 1), 1)

        return {
            "available": True,
            "tac_hashes": tac_count,
            "unique_bodies": body_count,
            "total_occurrences": total_occ,
            "avg_tac_per_body": avg_tac_per_body,
            "db_path": str(self.db_path),
        }

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# TACLookupBuilder — build/populate the lookup database
# ---------------------------------------------------------------------------

class TACLookupBuilder:
    """Build and populate the exhaustive TAC lookup database.

    Used by ``scripts/build_lookup_db.py`` to compile contracts with
    every compatible solc version and store all TAC→Solidity mappings.
    """

    def __init__(self, db_path: str = "data/tac_lookup.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Create the lookup database schema."""
        with _db_connection(self.db_path) as conn:
            cur = conn.cursor()

            cur.execute("""
                CREATE TABLE IF NOT EXISTS solidity_bodies (
                    body_id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    body_hash         TEXT UNIQUE NOT NULL,
                    solidity_code     TEXT NOT NULL,
                    function_name     TEXT,
                    selector          TEXT,
                    function_signature TEXT,
                    visibility        TEXT,
                    is_payable        BOOLEAN DEFAULT FALSE,
                    is_view           BOOLEAN DEFAULT FALSE,
                    first_seen        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS tac_to_body (
                    tac_hash          TEXT PRIMARY KEY,
                    body_id           INTEGER NOT NULL,
                    compiler_version  TEXT,
                    optimizer_enabled BOOLEAN,
                    optimizer_runs    INTEGER,
                    occurrences       INTEGER DEFAULT 1,
                    FOREIGN KEY (body_id) REFERENCES solidity_bodies(body_id)
                )
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tac_body_id
                ON tac_to_body(body_id)
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_bodies_selector
                ON solidity_bodies(selector)
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS compiled_jobs (
                    address          TEXT NOT NULL,
                    solc_version     TEXT NOT NULL,
                    optimizer_enabled INTEGER NOT NULL,
                    status           TEXT NOT NULL DEFAULT 'ok',
                    pairs_found      INTEGER DEFAULT 0,
                    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (address, solc_version, optimizer_enabled)
                )
            """)

            conn.commit()

        logger.info("Lookup DB initialized at %s", self.db_path)

    def upsert_body(
        self,
        solidity_code: str,
        function_name: str = "",
        selector: str = "",
        function_signature: str = "",
        visibility: str = "public",
        is_payable: bool = False,
        is_view: bool = False,
    ) -> int:
        """Insert or retrieve a Solidity body. Returns body_id.

        If the body already exists (by hash), updates the function_name
        if the new one is longer/more descriptive.
        """
        body_hash = hash_normalized_body(solidity_code)

        with _db_connection(self.db_path) as conn:
            cur = conn.cursor()

            # Try to find existing
            row = cur.execute(
                "SELECT body_id, function_name FROM solidity_bodies WHERE body_hash = ?",
                (body_hash,),
            ).fetchone()

            if row:
                body_id = row[0]
                existing_name = row[1] or ""
                # Keep the longer/more descriptive name
                if len(function_name) > len(existing_name):
                    cur.execute(
                        "UPDATE solidity_bodies SET function_name = ? WHERE body_id = ?",
                        (function_name, body_id),
                    )
                    conn.commit()
                return body_id

            # Insert new body
            cur.execute("""
                INSERT INTO solidity_bodies
                    (body_hash, solidity_code, function_name, selector,
                     function_signature, visibility, is_payable, is_view)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                body_hash, solidity_code, function_name, selector,
                function_signature, visibility, is_payable, is_view,
            ))
            conn.commit()
            return cur.lastrowid

    def upsert_tac_mapping(
        self,
        tac_text: str,
        body_id: int,
        compiler_version: str = "",
        optimizer_enabled: bool = True,
        optimizer_runs: int = 200,
    ) -> bool:
        """Map a TAC hash to a body_id. Returns True if newly inserted.

        If the tac_hash already exists, increments occurrences.
        """
        tac_hash = hash_normalized_tac(tac_text)

        with _db_connection(self.db_path) as conn:
            cur = conn.cursor()

            try:
                cur.execute("""
                    INSERT INTO tac_to_body
                        (tac_hash, body_id, compiler_version,
                         optimizer_enabled, optimizer_runs)
                    VALUES (?, ?, ?, ?, ?)
                """, (tac_hash, body_id, compiler_version,
                      optimizer_enabled, optimizer_runs))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                # Already exists — increment occurrences
                cur.execute(
                    "UPDATE tac_to_body SET occurrences = occurrences + 1 WHERE tac_hash = ?",
                    (tac_hash,),
                )
                conn.commit()
                return False

    def insert_pair(
        self,
        tac_text: str,
        solidity_code: str,
        function_name: str = "",
        selector: str = "",
        function_signature: str = "",
        visibility: str = "public",
        is_payable: bool = False,
        is_view: bool = False,
        compiler_version: str = "",
        optimizer_enabled: bool = True,
        optimizer_runs: int = 200,
    ) -> Tuple[int, bool]:
        """Convenience method: insert both body and TAC mapping.

        Returns:
            (body_id, is_new_tac) — body_id of the Solidity body,
            and whether the TAC hash was newly inserted.
        """
        body_id = self.upsert_body(
            solidity_code=solidity_code,
            function_name=function_name,
            selector=selector,
            function_signature=function_signature,
            visibility=visibility,
            is_payable=is_payable,
            is_view=is_view,
        )

        is_new = self.upsert_tac_mapping(
            tac_text=tac_text,
            body_id=body_id,
            compiler_version=compiler_version,
            optimizer_enabled=optimizer_enabled,
            optimizer_runs=optimizer_runs,
        )

        return body_id, is_new

    def bulk_insert_pairs(self, pairs: List[Dict]) -> Tuple[int, int]:
        """Insert multiple pairs efficiently.

        Each dict must have keys: tac, solidity_code, and optionally
        function_name, selector, function_signature, visibility,
        is_payable, is_view, compiler_version, optimizer_enabled,
        optimizer_runs.

        Returns:
            (new_bodies, new_tac_hashes)
        """
        new_bodies = 0
        new_tacs = 0

        with _db_connection(self.db_path) as conn:
            cur = conn.cursor()

            for p in pairs:
                body_hash = hash_normalized_body(p["solidity_code"])
                tac_hash = hash_normalized_tac(p["tac"])

                # Upsert body
                row = cur.execute(
                    "SELECT body_id FROM solidity_bodies WHERE body_hash = ?",
                    (body_hash,),
                ).fetchone()

                if row:
                    body_id = row[0]
                else:
                    cur.execute("""
                        INSERT INTO solidity_bodies
                            (body_hash, solidity_code, function_name, selector,
                             function_signature, visibility, is_payable, is_view)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        body_hash, p["solidity_code"],
                        p.get("function_name", ""),
                        p.get("selector", ""),
                        p.get("function_signature", ""),
                        p.get("visibility", "public"),
                        p.get("is_payable", False),
                        p.get("is_view", False),
                    ))
                    body_id = cur.lastrowid
                    new_bodies += 1

                # Upsert TAC mapping
                try:
                    cur.execute("""
                        INSERT INTO tac_to_body
                            (tac_hash, body_id, compiler_version,
                             optimizer_enabled, optimizer_runs)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        tac_hash, body_id,
                        p.get("compiler_version", ""),
                        p.get("optimizer_enabled", True),
                        p.get("optimizer_runs", 200),
                    ))
                    new_tacs += 1
                except sqlite3.IntegrityError:
                    cur.execute(
                        "UPDATE tac_to_body SET occurrences = occurrences + 1 "
                        "WHERE tac_hash = ?",
                        (tac_hash,),
                    )

            conn.commit()

        return new_bodies, new_tacs

    def stats(self) -> Dict:
        """Return summary statistics."""
        with _db_connection(self.db_path) as conn:
            tac_count = conn.execute("SELECT COUNT(*) FROM tac_to_body").fetchone()[0]
            body_count = conn.execute("SELECT COUNT(*) FROM solidity_bodies").fetchone()[0]
            total_occ = conn.execute(
                "SELECT SUM(occurrences) FROM tac_to_body"
            ).fetchone()[0] or 0

        return {
            "tac_hashes": tac_count,
            "unique_bodies": body_count,
            "total_occurrences": total_occ,
            "avg_tac_per_body": round(tac_count / max(body_count, 1), 1),
            "db_path": str(self.db_path),
        }

    # ── job tracking for re-runnability ───────────────────────────────────

    def get_completed_jobs(self) -> set:
        """Return set of (address, solc_version, optimizer_enabled) already done."""
        with _db_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT address, solc_version, optimizer_enabled FROM compiled_jobs"
            ).fetchall()
        return {(r[0], r[1], r[2]) for r in rows}

    def mark_jobs_done_batch(self, jobs: List[tuple]) -> None:
        """Batch-mark jobs as done.

        Each tuple: (address, version, opt_enabled, status, pairs_found).
        """
        with _db_connection(self.db_path) as conn:
            conn.executemany(
                """INSERT OR IGNORE INTO compiled_jobs
                   (address, solc_version, optimizer_enabled, status, pairs_found)
                   VALUES (?, ?, ?, ?, ?)""",
                jobs,
            )
            conn.commit()
