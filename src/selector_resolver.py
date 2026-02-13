"""
Function Selector Resolver — maps 4-byte EVM selectors to human-readable signatures.

Provides confidence-scored lookups using a tiered strategy:
  1. A curated built-in database of common selectors (ERC-20, ERC-721, Ownable, etc.)
  2. A persistent SQLite ``selector_registry`` table (populated by the data pipeline)
  3. A local JSON file (``data/selectors.json``) for portable sharing
  4. The 4byte.directory public API as a remote fallback
  
Results from tiers 3–4 are written back to the local cache so subsequent
lookups are instant.
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ResolvedSelector:
    """A single candidate match for a function selector."""

    selector: str  # e.g. "0x893d20e8"
    signature: str  # e.g. "getOwner()"
    confidence: float  # 0.0 – 1.0
    source: str  # "builtin" | "db" | "json" | "4byte" | "unknown"

@dataclass
class SelectorResult:
    """All resolution results for one selector."""

    selector: str
    candidates: List[ResolvedSelector] = field(default_factory=list)
    best_match: Optional[ResolvedSelector] = None

    def to_dict(self) -> dict:
        best = None
        if self.best_match:
            best = {
                "signature": self.best_match.signature,
                "confidence": round(self.best_match.confidence * 100, 1),
                "source": self.best_match.source,
            }
        return {
            "selector": self.selector,
            "candidates": [
                {
                    "signature": c.signature,
                    "confidence": round(c.confidence * 100, 1),
                    "source": c.source,
                }
                for c in self.candidates
            ],
            "best_match": best,
        }

# ---------------------------------------------------------------------------
# Built-in selector database (curated, high-confidence)
# ---------------------------------------------------------------------------

# Mapping: selector hex (lowercase, 0x-prefixed) → function signature string
_BUILTIN_SELECTORS: Dict[str, str] = {
    # ---- ERC-20 ----
    "0x06fdde03": "name()",
    "0x95d89b41": "symbol()",
    "0x313ce567": "decimals()",
    "0x18160ddd": "totalSupply()",
    "0x70a08231": "balanceOf(address)",
    "0xdd62ed3e": "allowance(address,address)",
    "0xa9059cbb": "transfer(address,uint256)",
    "0x23b872dd": "transferFrom(address,address,uint256)",
    "0x095ea7b3": "approve(address,uint256)",
    # ---- ERC-721 ----
    "0x6352211e": "ownerOf(uint256)",
    "0x42842e0e": "safeTransferFrom(address,address,uint256)",
    "0xb88d4fde": "safeTransferFrom(address,address,uint256,bytes)",
    "0xe985e9c5": "isApprovedForAll(address,address)",
    "0xa22cb465": "setApprovalForAll(address,bool)",
    "0x081812fc": "getApproved(uint256)",
    "0x01ffc9a7": "supportsInterface(bytes4)",
    "0xc87b56dd": "tokenURI(uint256)",
    "0x4f6ccce7": "tokenByIndex(uint256)",
    "0x2f745c59": "tokenOfOwnerByIndex(address,uint256)",
    # ---- ERC-1155 ----
    "0x00fdd58e": "balanceOf(address,uint256)",
    "0x4e1273f4": "balanceOfBatch(address[],uint256[])",
    "0xf242432a": "safeTransferFrom(address,address,uint256,uint256,bytes)",
    "0x2eb2c2d6": "safeBatchTransferFrom(address,address,uint256[],uint256[],bytes)",
    "0x0e89341c": "uri(uint256)",
    # ---- Ownable / Access Control ----
    "0x893d20e8": "getOwner()",
    "0x8da5cb5b": "owner()",
    "0xf2fde38b": "transferOwnership(address)",
    "0x715018a6": "renounceOwnership()",
    "0xa6f9dae1": "changeOwner(address)",
    "0x13af4035": "setOwner(address)",
    # ---- Proxy / Upgradeable ----
    "0x5c60da1b": "implementation()",
    "0x3659cfe6": "upgradeTo(address)",
    "0x4f1ef286": "upgradeToAndCall(address,bytes)",
    "0xf851a440": "admin()",
    "0x8f283970": "changeAdmin(address)",
    # ---- Common Utility ----
    "0x12065fe0": "getBalance()",
    "0x3ccfd60b": "withdraw()",
    "0xd0e30db0": "deposit()",
    "0x2e1a7d4d": "withdraw(uint256)",
    "0xe2d9d468": "withdrawAll()",
    "0x8129fc1c": "initialize()",
    "0xc4d66de8": "initialize(address)",
    "0xfe4b84df": "initialize(uint256)",
    "0x1249c58b": "mint()",
    "0xa0712d68": "mint(uint256)",
    "0x40c10f19": "mint(address,uint256)",
    "0x42966c68": "burn(uint256)",
    "0x79cc6790": "burnFrom(address,uint256)",
    "0x9dc29fac": "burn(address,uint256)",
    "0x5c975abb": "paused()",
    "0x8456cb59": "pause()",
    "0x3f4ba83a": "unpause()",
    # ---- ERC-2612 Permit ----
    "0xd505accf": "permit(address,address,uint256,uint256,uint8,bytes32,bytes32)",
    "0x3644e515": "DOMAIN_SEPARATOR()",
    "0x7ecebe00": "nonces(address)",
    # ---- Multicall ----
    "0xac9650d8": "multicall(bytes[])",
    "0x5ae401dc": "multicall(uint256,bytes[])",
    # ---- Uniswap / DEX ----
    "0x38ed1739": "swapExactTokensForTokens(uint256,uint256,address[],address,uint256)",
    "0x7ff36ab5": "swapExactETHForTokens(uint256,address[],address,uint256)",
    "0x18cbafe5": "swapExactTokensForETH(uint256,uint256,address[],address,uint256)",
    "0xe8e33700": "addLiquidity(address,address,uint256,uint256,uint256,uint256,address,uint256)",
    "0xf305d719": "addLiquidityETH(address,uint256,uint256,uint256,address,uint256)",
    "0xbaa2abde": "removeLiquidity(address,address,uint256,uint256,uint256,address,uint256)",
    "0x02751cec": "removeLiquidityETH(address,uint256,uint256,uint256,address,uint256)",
    "0x0902f1ac": "getReserves()",
    "0xd21220a7": "token1()",
    "0x0dfe1681": "token0()",
    "0x022c0d9f": "swap(uint256,uint256,address,bytes)",
    "0x6a627842": "mint(address)",
    "0x89afcb44": "burn(address)",
    "0xfff6cae9": "sync()",
    # ---- Aave ----
    "0xe8eda9df": "deposit(address,uint256,address,uint16)",
    "0x69328dec": "withdraw(address,uint256,address)",
    "0xa415bcad": "borrow(address,uint256,uint256,uint16,address)",
    "0x573ade81": "repay(address,uint256,uint256,address)",
    # ---- Compound ----
    "0xdb006a75": "redeem(uint256)",
    "0x852a12e3": "redeemUnderlying(uint256)",
    "0xc5ebeaec": "borrow(uint256)",
    "0x0e752702": "repayBorrow(uint256)",
    # ---- Governance ----
    "0xda95691a": "propose(address[],uint256[],string[],bytes[],string)",
    "0x56781388": "castVote(uint256,uint8)",
    "0x2656227d": "execute(uint256)",
    "0xe23a9a52": "getReceipt(uint256,address)",
    "0x3e4f49e6": "state(uint256)",
    # ---- Miscellaneous ----
    "0x8b7afe2e": "getAccountLiquidity(address)",
    "0xc45a0155": "factory()",
    "0xad5c4648": "WETH()",
    "0x252dba42": "aggregate((address,bytes)[])",
    "0x82ad56cb": "aggregate3((address,bool,bytes)[])",
}

# ---------------------------------------------------------------------------
# Confidence helpers
# ---------------------------------------------------------------------------

def _db_confidence(occurrences: int) -> float:
    """Map occurrence count to a confidence score for DB-sourced selectors."""
    if occurrences >= 10:
        return 0.95
    if occurrences >= 5:
        return 0.90
    if occurrences >= 2:
        return 0.85
    return 0.80

# ---------------------------------------------------------------------------
# Resolver class
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = Path("data/contracts.db")
_DEFAULT_JSON_PATH = Path("data/selectors.json")

class SelectorResolver:
    """Resolve EVM function selectors to human-readable signatures.

    Lookup order:
      1. In-memory builtin dict (97% confidence)
      2. SQLite ``selector_registry`` table (80-95% based on occurrences)
      3. Local JSON file ``data/selectors.json`` (85% confidence)
      4. 4byte.directory API (60-88% confidence)

    Results from tiers 3-4 are cached in memory and optionally written
    back to the DB for future lookups.
    """

    _FOUR_BYTE_API = "https://www.4byte.directory/api/v1/signatures/"

    def __init__(
        self,
        *,
        use_remote: bool = True,
        timeout: float = 3.0,
        db_path: Optional[Path] = None,
        json_path: Optional[Path] = None,
    ) -> None:
        self.use_remote = use_remote
        self.timeout = timeout
        self.db_path = db_path or _DEFAULT_DB_PATH
        self.json_path = json_path or _DEFAULT_JSON_PATH

        # Runtime cache for all lookups
        self._cache: Dict[str, SelectorResult] = {}

        # Load JSON file into memory if available
        self._json_selectors: Dict[str, list] = {}
        self._load_json_file()

    def _load_json_file(self):
        """Load the local selectors JSON file into memory."""
        try:
            if self.json_path.exists():
                with open(self.json_path, "r", encoding="utf-8") as f:
                    self._json_selectors = json.load(f)
                logger.debug("Loaded %d selectors from %s",
                             len(self._json_selectors), self.json_path)
        except Exception as e:
            logger.debug("Could not load selectors JSON: %s", e)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def resolve(self, selector: str) -> SelectorResult:
        """Resolve a single selector (e.g. ``"0x893d20e8"``)."""
        selector = self._normalise(selector)

        # Check cache
        if selector in self._cache:
            return self._cache[selector]

        result = SelectorResult(selector=selector)

        # 1. Built-in DB (highest confidence)
        if selector in _BUILTIN_SELECTORS:
            sig = _BUILTIN_SELECTORS[selector]
            candidate = ResolvedSelector(
                selector=selector,
                signature=sig,
                confidence=0.97,
                source="builtin",
            )
            result.candidates.append(candidate)
            result.best_match = candidate

        # 2. SQLite selector_registry lookup
        if not result.best_match:
            db_results = self._query_db(selector)
            if db_results:
                result.candidates.extend(db_results)
                result.best_match = db_results[0]

        # 3. Local JSON file lookup
        if not result.best_match:
            json_results = self._query_json(selector)
            if json_results:
                result.candidates.extend(json_results)
                result.best_match = json_results[0]

        # 4. Remote 4byte.directory lookup
        if self.use_remote and not result.best_match:
            remote = self._query_4byte(selector)
            if remote:
                result.candidates.extend(remote)
                if not result.best_match and remote:
                    result.best_match = remote[0]
                # Write back to DB for future lookups
                self._writeback_to_db(selector, remote)

        # 5. If still nothing, mark unknown
        if not result.best_match:
            result.best_match = ResolvedSelector(
                selector=selector,
                signature=f"unknown_{selector}",
                confidence=0.0,
                source="unknown",
            )

        self._cache[selector] = result
        return result

    def resolve_many(self, selectors: List[str]) -> Dict[str, SelectorResult]:
        """Resolve multiple selectors. Returns dict keyed by normalised selector."""
        return {s: self.resolve(s) for s in selectors}

    def resolve_function_names(
        self, function_names: List[str]
    ) -> Dict[str, SelectorResult]:
        """Resolve from function names like ``function_0x893d20e8``.

        Returns dict keyed by the original function name.
        """
        results: Dict[str, SelectorResult] = {}
        for fname in function_names:
            selector = self._extract_selector(fname)
            if selector:
                results[fname] = self.resolve(selector)
            else:
                results[fname] = SelectorResult(
                    selector="",
                    best_match=ResolvedSelector(
                        selector="",
                        signature=fname,
                        confidence=0.0,
                        source="unknown",
                    ),
                )
        return results

    def registry_stats(self) -> dict:
        """Return statistics about the selector registry."""
        stats = {
            "builtin_count": len(_BUILTIN_SELECTORS),
            "json_count": len(self._json_selectors),
            "db_count": 0,
            "cache_count": len(self._cache),
        }
        try:
            if self.db_path.exists():
                with self._db_conn() as conn:
                    stats["db_count"] = conn.execute(
                        "SELECT COUNT(DISTINCT selector) FROM selector_registry"
                    ).fetchone()[0]
        except Exception:
            pass
        return stats

    # ------------------------------------------------------------------ #
    #  DB helpers
    # ------------------------------------------------------------------ #

    @contextmanager
    def _db_conn(self):
        """Context manager for SQLite connection to contracts.db."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _query_db(self, selector: str) -> List[ResolvedSelector]:
        """Query the selector_registry table."""
        try:
            if not self.db_path.exists():
                return []
            with self._db_conn() as conn:
                rows = conn.execute(
                    "SELECT signature, source, occurrences "
                    "FROM selector_registry WHERE selector = ? "
                    "ORDER BY occurrences DESC",
                    (selector,),
                ).fetchall()

            candidates = []
            for sig, source, occ in rows:
                candidates.append(ResolvedSelector(
                    selector=selector,
                    signature=sig,
                    confidence=_db_confidence(occ),
                    source="db",
                ))
            return candidates
        except Exception as e:
            logger.debug("DB lookup for %s failed: %s", selector, e)
            return []

    def _query_json(self, selector: str) -> List[ResolvedSelector]:
        """Look up selector in the loaded JSON data."""
        entries = self._json_selectors.get(selector, [])
        if not entries:
            return []

        candidates = []
        for entry in entries:
            sig = entry.get("signature", "")
            occ = entry.get("occurrences", 1)
            if sig:
                candidates.append(ResolvedSelector(
                    selector=selector,
                    signature=sig,
                    confidence=min(_db_confidence(occ), 0.92),
                    source="json",
                ))
        return candidates

    def _writeback_to_db(self, selector: str, results: List[ResolvedSelector]):
        """Write remote lookup results back to the DB for future use."""
        try:
            if not self.db_path.exists():
                return
            with self._db_conn() as conn:
                cur = conn.cursor()
                # Ensure table exists
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
                for r in results:
                    cur.execute("""
                        INSERT INTO selector_registry (selector, signature, source, occurrences)
                        VALUES (?, ?, ?, 1)
                        ON CONFLICT(selector, signature) DO UPDATE SET
                            occurrences = selector_registry.occurrences + 1,
                            last_seen = CURRENT_TIMESTAMP
                    """, (selector, r.signature, r.source))
                conn.commit()
        except Exception as e:
            logger.debug("Failed to write back selector %s: %s", selector, e)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise(selector: str) -> str:
        """Ensure ``0x``-prefixed, lowercase, 10-char hex."""
        s = selector.strip().lower()
        if not s.startswith("0x"):
            s = "0x" + s
        return s

    @staticmethod
    def _extract_selector(function_name: str) -> Optional[str]:
        """Extract selector hex from names like ``function_0x893d20e8``."""
        if function_name.startswith("function_0x") and len(function_name) >= 19:
            return "0x" + function_name[11:19].lower()
        if function_name.startswith("function_") and len(function_name) >= 17:
            raw = function_name[9:17]
            try:
                int(raw, 16)
                return "0x" + raw.lower()
            except ValueError:
                pass
        return None

    def _query_4byte(self, selector: str) -> List[ResolvedSelector]:
        """Query 4byte.directory API for selector matches."""
        try:
            resp = requests.get(
                self._FOUR_BYTE_API,
                params={"hex_signature": selector, "ordering": "created_at"},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            results_list = data.get("results", [])
            if not results_list:
                return []

            candidates: List[ResolvedSelector] = []
            total = len(results_list)
            for i, entry in enumerate(results_list):
                sig = entry.get("text_signature", "")
                if not sig:
                    continue
                # Confidence heuristic:
                #  - single match → 0.88
                #  - first of multiple → 0.80, decaying
                if total == 1:
                    conf = 0.88
                else:
                    conf = max(0.40, 0.80 - i * 0.10)
                candidates.append(
                    ResolvedSelector(
                        selector=selector,
                        signature=sig,
                        confidence=conf,
                        source="4byte",
                    )
                )
            return candidates

        except Exception as e:
            logger.debug("4byte.directory lookup for %s failed: %s", selector, e)
            return []

# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

# Shared default instance
_default_resolver: Optional[SelectorResolver] = None

def get_resolver(*, use_remote: bool = True) -> SelectorResolver:
    """Return or create the default shared resolver instance."""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = SelectorResolver(use_remote=use_remote)
    return _default_resolver