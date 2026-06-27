"""
ABI Enrichment & Storage Layout Resolution for TAC Generation

This module provides two key capabilities for improving TAC quality:

1. **ABIEnricher** (Issue #8): Parses contract ABI JSON to annotate TAC with
   function names, parameter types, return types, and event signatures.

2. **StorageLayoutResolver** (Issue #2): Parses Solidity source code to infer
   storage slot assignments for state variables, enabling TAC annotations
   like ``storage[slot_0]  // likely: address owner``.

Both classes are designed to be used as optional enrichment — if ABI or source
data is unavailable, TAC generation proceeds without annotations.
"""

import json
import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from web3 import Web3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Issue #8: ABI Enrichment
# ---------------------------------------------------------------------------


def normalize_hex(value) -> str:
    """Return lowercase hex with exactly one ``0x`` prefix."""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (bytes, bytearray)):
        text = value.hex()
    elif hasattr(value, "hex"):
        text = value.hex()
    else:
        text = str(value)

    text = text.strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    return "0x" + text


def _split_top_level_commas(value: str) -> List[str]:
    """Split a Solidity tuple component list on commas outside nested tuples."""
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    for ch in value:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def _find_matching_paren(value: str) -> int:
    depth = 0
    for idx, ch in enumerate(value):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _extract_type_token(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    parts = value.split()
    token = parts[0]
    if len(parts) > 1 and parts[1].startswith("["):
        token += parts[1]
    return token


def canonicalize_abi_type(type_name: str) -> str:
    """Canonicalize a Solidity ABI type for selector/topic hashing.

    Handles aliases (``uint``/``int``/``fixed``/``ufixed``), tuple components
    recursively, and array suffixes.
    """
    typ = (type_name or "").strip()
    if not typ:
        return typ

    if typ.startswith("("):
        end = _find_matching_paren(typ)
        if end != -1:
            inner = typ[1:end]
            rest = typ[end + 1 :].strip()
            suffix_match = re.match(r"((?:\[[0-9]*\])*)", rest)
            suffix = suffix_match.group(1) if suffix_match else ""
            components = [
                canonicalize_abi_type(part)
                for part in _split_top_level_commas(inner)
                if part
            ]
            return f"({','.join(components)}){suffix}"

    token = _extract_type_token(typ)
    match = re.match(r"^(.+?)(\[[0-9]*\])*$", token.replace(" ", ""))
    if match:
        base = match.group(1)
        suffix = token.replace(" ", "")[len(base):]
    else:
        base = token.replace(" ", "")
        suffix = ""

    aliases = {
        "uint": "uint256",
        "int": "int256",
        "fixed": "fixed128x18",
        "ufixed": "ufixed128x18",
        "byte": "bytes1",
    }
    return aliases.get(base, base) + suffix

# Well-known error selectors
ERROR_SELECTORS = {
    "08c379a0": "Error(string)",
    "4e487b71": "Panic(uint256)",
}

# Well-known panic codes (Issue #3)
PANIC_CODES = {
    0x00: "generic",
    0x01: "assert failure",
    0x11: "arithmetic overflow/underflow",
    0x12: "division by zero",
    0x21: "enum conversion out of bounds",
    0x22: "storage encoding error",
    0x31: "pop on empty array",
    0x32: "array index out of bounds",
    0x41: "too much memory allocated",
    0x51: "zero-initialized function pointer",
}


@dataclass
class ABIFunctionInfo:
    """Parsed ABI function information."""
    name: str
    selector: str
    input_types: List[str]
    input_names: List[str]
    output_types: List[str]
    state_mutability: str = ""


@dataclass
class ABIEventInfo:
    """Parsed ABI event information."""
    name: str
    topic0: str  # keccak256 of the event signature
    input_types: List[str]
    input_names: List[str]
    indexed: List[bool]


@dataclass
class ABIErrorInfo:
    """Parsed ABI error information."""
    name: str
    selector: str  # first 4 bytes of keccak256
    input_types: List[str]
    input_names: List[str]


class ABIEnricher:
    """Parse contract ABI and provide type annotations for TAC enrichment.

    Usage::

        enricher = ABIEnricher(abi_json_string)
        func_info = enricher.get_function("0xa9059cbb")
        # func_info.name == "transfer"
        # func_info.input_types == ["address", "uint256"]

        event_info = enricher.get_event_by_topic(topic0_hex)
        error_info = enricher.get_error("0x08c379a0")
    """

    def __init__(self, abi_json: str = "") -> None:
        self.functions: Dict[str, ABIFunctionInfo] = {}
        self.events: Dict[str, ABIEventInfo] = {}  # keyed by topic0
        self.errors: Dict[str, ABIErrorInfo] = {}  # keyed by 4-byte selector
        self._parse(abi_json)

    def _parse(self, abi_json: str) -> None:
        """Parse ABI JSON string into structured lookups."""
        if not abi_json or not abi_json.strip():
            return

        try:
            abi = json.loads(abi_json)
        except (json.JSONDecodeError, TypeError):
            return

        if not isinstance(abi, list):
            return

        for entry in abi:
            try:
                entry_type = entry.get("type", "")
                if entry_type == "function":
                    self._parse_function(entry)
                elif entry_type == "event":
                    self._parse_event(entry)
                elif entry_type == "error":
                    self._parse_error(entry)
            except Exception:
                continue

    def _parse_function(self, entry: dict) -> None:
        """Parse a function ABI entry."""
        name = entry.get("name", "")
        if not name:
            return

        inputs = entry.get("inputs", [])
        outputs = entry.get("outputs", [])

        input_types = [self._resolve_type(inp) for inp in inputs]
        input_names = [inp.get("name", f"param{i}") for i, inp in enumerate(inputs)]
        output_types = [self._resolve_type(out) for out in outputs]

        canonical = f"{name}({','.join(input_types)})"
        selector = normalize_hex(Web3.keccak(text=canonical)[:4])

        self.functions[selector] = ABIFunctionInfo(
            name=name,
            selector=selector,
            input_types=input_types,
            input_names=input_names,
            output_types=output_types,
            state_mutability=entry.get("stateMutability", ""),
        )

    def _parse_event(self, entry: dict) -> None:
        """Parse an event ABI entry."""
        name = entry.get("name", "")
        if not name:
            return

        inputs = entry.get("inputs", [])
        input_types = [self._resolve_type(inp) for inp in inputs]
        input_names = [inp.get("name", f"param{i}") for i, inp in enumerate(inputs)]
        indexed = [inp.get("indexed", False) for inp in inputs]

        canonical = f"{name}({','.join(input_types)})"
        topic0 = normalize_hex(Web3.keccak(text=canonical))

        self.events[topic0] = ABIEventInfo(
            name=name,
            topic0=topic0,
            input_types=input_types,
            input_names=input_names,
            indexed=indexed,
        )

    def _parse_error(self, entry: dict) -> None:
        """Parse a custom error ABI entry."""
        name = entry.get("name", "")
        if not name:
            return

        inputs = entry.get("inputs", [])
        input_types = [self._resolve_type(inp) for inp in inputs]
        input_names = [inp.get("name", f"param{i}") for i, inp in enumerate(inputs)]

        canonical = f"{name}({','.join(input_types)})"
        selector = normalize_hex(Web3.keccak(text=canonical)[:4])

        self.errors[selector] = ABIErrorInfo(
            name=name,
            selector=selector,
            input_types=input_types,
            input_names=input_names,
        )

    @staticmethod
    def _resolve_type(param: dict) -> str:
        """Resolve an ABI parameter to its canonical type string.

        Handles tuple types by recursively resolving component types.
        """
        typ = param.get("type", "")
        if typ == "tuple" or typ.startswith("tuple"):
            components = param.get("components", [])
            inner = ",".join(
                ABIEnricher._resolve_type(c) for c in components
            )
            # Preserve array suffix: tuple[] → (...)[]
            suffix = typ[5:]  # everything after "tuple"
            return canonicalize_abi_type(f"({inner}){suffix}")
        return canonicalize_abi_type(typ)

    def get_function(self, selector: str) -> Optional[ABIFunctionInfo]:
        """Look up function info by 4-byte selector."""
        return self.functions.get(selector.lower()) or self.functions.get(selector)

    def get_event_by_topic(self, topic0: str) -> Optional[ABIEventInfo]:
        """Look up event info by topic0 hash."""
        return self.events.get(topic0.lower()) or self.events.get(topic0)

    def get_error(self, selector: str) -> Optional[ABIErrorInfo]:
        """Look up custom error info by 4-byte selector."""
        key = selector.lower() if selector.startswith("0x") else "0x" + selector.lower()
        return self.errors.get(key)

    def has_data(self) -> bool:
        """Return True if any ABI data was successfully parsed."""
        return bool(self.functions or self.events or self.errors)

    def format_function_header(self, selector: str) -> Optional[str]:
        """Generate a TAC function header line from ABI info.

        Returns a string like:
            ``function transfer(address to, uint256 amount):``
        or ``None`` if no ABI info is available for this selector.
        """
        func = self.get_function(selector)
        if not func:
            return None

        params = ", ".join(
            f"{t} {n}" for t, n in zip(func.input_types, func.input_names)
        )
        return f"function {func.name}({params}):"

    def format_return_annotation(self, selector: str) -> Optional[str]:
        """Generate a return type annotation comment.

        Returns a string like ``// Returns: bool`` or ``None``.
        """
        func = self.get_function(selector)
        if not func or not func.output_types:
            return None
        return f"// Returns: {', '.join(func.output_types)}"

    def format_param_annotations(self, selector: str) -> List[str]:
        """Generate calldata parameter annotations.

        Returns lines like:
            ``// param[0] at 0x04: address to``
            ``// param[1] at 0x24: uint256 amount``
        """
        func = self.get_function(selector)
        if not func or not func.input_types:
            return []

        lines = []
        offset = 0x04  # first param starts at byte 4 (after selector)
        for i, (typ, name) in enumerate(zip(func.input_types, func.input_names)):
            lines.append(f"// param[{i}] at 0x{offset:02x}: {typ} {name}")
            offset += 0x20  # each ABI-encoded param is 32 bytes
        return lines

    def format_event_annotation(self, topic0: str) -> Optional[str]:
        """Generate an event annotation from topic0.

        Returns a string like:
            ``// event: Transfer(address indexed from, address indexed to, uint256 value)``
        """
        event = self.get_event_by_topic(topic0)
        if not event:
            return None

        params = []
        for t, n, idx in zip(event.input_types, event.input_names, event.indexed):
            prefix = "indexed " if idx else ""
            params.append(f"{t} {prefix}{n}")
        return f"// event: {event.name}({', '.join(params)})"


# ---------------------------------------------------------------------------
# Issue #2: Storage Layout Resolution
# ---------------------------------------------------------------------------

@dataclass
class StateVariable:
    """Represents a Solidity state variable with its storage slot."""
    name: str
    type_name: str
    slot: int
    contract_name: str = ""
    is_mapping: bool = False
    is_array: bool = False


# Solidity types and their storage sizes in slots
_SLOT_SIZES = {
    # Types that fit in a single 32-byte slot
    "uint256": 32, "int256": 32, "bytes32": 32, "address": 20,
    "bool": 1, "uint8": 1, "int8": 1, "uint16": 2, "int16": 2,
    "uint32": 4, "int32": 4, "uint64": 8, "int64": 8,
    "uint128": 16, "int128": 16, "uint160": 20, "int160": 20,
    "bytes1": 1, "bytes2": 2, "bytes4": 4, "bytes8": 8,
    "bytes16": 16, "bytes20": 20,
}

# Regex to match state variable declarations
_STATE_VAR_PATTERN = re.compile(
    r"""
    ^\s*                                    # leading whitespace
    (?:(?:public|private|internal)\s+)?     # optional visibility
    (mapping\s*\([^)]+\)                    # mapping type
    |[a-zA-Z_]\w*(?:\[\])?)                # or regular type (possibly array)
    \s+                                     # whitespace
    (?:(?:public|private|internal|constant|immutable)\s+)*  # modifiers
    (\w+)                                   # variable name
    \s*(?:=[^;]*)?;                         # optional initializer + semicolon
    """,
    re.VERBOSE | re.MULTILINE,
)

# More specific pattern for catching common types
_SIMPLE_VAR_PATTERN = re.compile(
    r"^\s*"
    r"((?:u?int(?:8|16|32|64|128|160|256)?|address|bool|bytes(?:\d+)?|string"
    r"|mapping\s*\([^;]+\)|[A-Z]\w*(?:\[\])?)\s*)"  # type
    r"(?:(?:public|private|internal|constant|immutable)\s+)*"
    r"(\w+)"       # name
    r"\s*(?:=[^;]*)?;",  # optional init
    re.MULTILINE,
)


class StorageLayoutResolver:
    """Infer Solidity storage slot assignments from source code.

    Solidity assigns storage slots sequentially to state variables declared
    at the contract level. This resolver parses the source to build a
    slot → variable mapping.

    Usage::

        resolver = StorageLayoutResolver(solidity_source)
        layout = resolver.get_storage_layout()
        # layout == {0: StateVariable(name="owner", type_name="address", slot=0), ...}

        annotation = resolver.annotate_slot(0)
        # annotation == "// likely: address owner"
    """

    def __init__(self, source_code: str = "", contract_name: str = "") -> None:
        self.source_code = source_code
        self.contract_name = contract_name
        self.variables: List[StateVariable] = []
        self.slot_map: Dict[int, StateVariable] = {}
        if source_code:
            self._parse()

    def _parse(self) -> None:
        """Parse state variables from the Solidity source and assign slots."""
        # Extract contract body (handle inheritance by looking for all contracts)
        contracts = self._extract_contract_bodies()
        if not contracts:
            return

        # Use the target contract or the last one
        if self.contract_name and self.contract_name in contracts:
            body = contracts[self.contract_name]
        else:
            body = list(contracts.values())[-1]

        self._extract_state_variables(body)
        self._assign_slots()

    def _extract_contract_bodies(self) -> Dict[str, str]:
        """Extract contract name → body text from source."""
        results: Dict[str, str] = {}
        # Match contract/library/interface declarations
        pattern = re.compile(
            r"(?:contract|library|abstract\s+contract)\s+(\w+)"
            r"(?:\s+is\s+[^{]+)?\s*\{",
            re.MULTILINE,
        )

        for match in pattern.finditer(self.source_code):
            name = match.group(1)
            start = match.end()
            # Find matching closing brace
            depth = 1
            pos = start
            while pos < len(self.source_code) and depth > 0:
                ch = self.source_code[pos]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                pos += 1
            if depth == 0:
                results[name] = self.source_code[start:pos - 1]

        return results

    def _extract_state_variables(self, contract_body: str) -> None:
        """Extract state variable declarations from a contract body.

        Only considers lines at the top level of the contract (not inside
        function bodies).
        """
        # Remove function/modifier/event/constructor bodies to avoid false matches
        cleaned = self._strip_function_bodies(contract_body)

        for match in _SIMPLE_VAR_PATTERN.finditer(cleaned):
            type_str = match.group(1).strip()
            var_name = match.group(2).strip()

            # Skip constants and immutables (they don't use storage)
            line = match.group(0)
            if "constant " in line or "immutable " in line:
                continue

            # Skip event/error declarations
            if type_str.startswith("event ") or type_str.startswith("error "):
                continue

            is_mapping = type_str.startswith("mapping")
            is_array = type_str.endswith("[]")

            self.variables.append(StateVariable(
                name=var_name,
                type_name=type_str,
                slot=-1,  # assigned later
                contract_name=self.contract_name,
                is_mapping=is_mapping,
                is_array=is_array,
            ))

    @staticmethod
    def _strip_function_bodies(text: str) -> str:
        """Remove function/modifier/constructor bodies to isolate state vars."""
        # Replace function bodies with empty blocks
        result: List[str] = []
        i = 0
        depth = 0
        in_func = False

        while i < len(text):
            ch = text[i]
            if not in_func:
                # Check if we're entering a function/modifier/constructor
                remaining = text[i:]
                func_match = re.match(
                    r"(?:function|modifier|constructor|receive|fallback)\s",
                    remaining,
                )
                if func_match:
                    in_func = True
                    # Skip to the opening brace
                    while i < len(text) and text[i] != '{':
                        i += 1
                    if i < len(text):
                        depth = 1
                        i += 1  # skip the {
                    continue
                result.append(ch)
            else:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        in_func = False
            i += 1

        return ''.join(result)

    def _assign_slots(self) -> None:
        """Assign sequential storage slots to variables.

        Simple types occupy 1 slot each. Mappings and dynamic arrays
        also occupy 1 slot (the data is at keccak256-derived locations).
        Packing of small types is not modeled (conservative: 1 slot each).
        """
        slot = 0
        for var in self.variables:
            var.slot = slot
            self.slot_map[slot] = var
            slot += 1

    def get_storage_layout(self) -> Dict[int, StateVariable]:
        """Return the slot → variable mapping."""
        return dict(self.slot_map)

    def annotate_slot(self, slot: int) -> Optional[str]:
        """Return a comment annotation for a storage slot.

        Args:
            slot: The storage slot number.

        Returns:
            A string like ``// likely: address owner`` or ``None``.
        """
        var = self.slot_map.get(slot)
        if var is None:
            return None
        return f"// likely: {var.type_name} {var.name}"

    def annotate_slot_hex(self, slot_hex: str) -> Optional[str]:
        """Return a comment annotation for a hex storage slot string.

        Handles full 64-char hex slot keys (e.g. ``0x000...0000``)
        and short hex (e.g. ``0x00``, ``0x01``).
        """
        try:
            slot_clean = slot_hex.replace("0x", "").replace("0X", "").lstrip("0") or "0"
            slot_int = int(slot_clean, 16)
            return self.annotate_slot(slot_int)
        except (ValueError, TypeError):
            return None

    def format_storage_header(self) -> List[str]:
        """Generate a ``// Storage layout:`` header block for TAC."""
        if not self.variables:
            return []

        lines = ["// Storage layout:"]
        for var in self.variables:
            extra = ""
            if var.is_mapping:
                extra = " (mapping — data at keccak256(key . slot))"
            elif var.is_array:
                extra = " (dynamic array — data at keccak256(slot))"
            lines.append(f"//   slot {var.slot}: {var.type_name} {var.name}{extra}")
        return lines

    def has_data(self) -> bool:
        """Return True if any state variables were found."""
        return bool(self.variables)


# ---------------------------------------------------------------------------
# Convenience: enrich a TAC string with ABI + storage annotations
# ---------------------------------------------------------------------------

def enrich_tac_with_abi(
    tac: str,
    selector: str,
    abi_enricher: Optional[ABIEnricher] = None,
    storage_resolver: Optional[StorageLayoutResolver] = None,
) -> str:
    """Post-process a TAC string to add ABI and storage annotations.

    This is the main integration point. It:
    1. Replaces the function header with ABI-derived name + params
    2. Adds return type annotation
    3. Adds parameter offset annotations
    4. Annotates storage slot accesses
    5. Annotates LOG instructions with event names

    Args:
        tac: The raw TAC string.
        selector: The 4-byte function selector (e.g. ``0xa9059cbb``).
        abi_enricher: Optional ABIEnricher instance.
        storage_resolver: Optional StorageLayoutResolver instance.

    Returns:
        Enriched TAC string with annotations.
    """
    if not tac:
        return tac

    lines = tac.split("\n")
    enriched: List[str] = []

    # Track if we've inserted ABI header annotations
    abi_header_done = False

    for line in lines:
        stripped = line.strip()

        # Replace function header with ABI-derived version
        if (not abi_header_done and abi_enricher and
                stripped.startswith("function ") and stripped.endswith(":")):
            abi_header = abi_enricher.format_function_header(selector)
            if abi_header:
                # Preserve indentation
                indent = line[:len(line) - len(line.lstrip())]
                enriched.append(f"{indent}{abi_header}")
            else:
                enriched.append(line)
            abi_header_done = True
            continue

        # After the selector comment, add return type + param annotations
        if abi_enricher and "// Selector:" in stripped:
            enriched.append(line)
            indent = line[:len(line) - len(line.lstrip())]

            ret = abi_enricher.format_return_annotation(selector)
            if ret:
                enriched.append(f"{indent}{ret}")

            params = abi_enricher.format_param_annotations(selector)
            for p in params:
                enriched.append(f"{indent}{p}")
            continue

        # After entry block comment, add storage layout header
        if storage_resolver and "// Entry block:" in stripped:
            enriched.append(line)
            indent = line[:len(line) - len(line.lstrip())]
            for sl in storage_resolver.format_storage_header():
                enriched.append(f"{indent}{sl}")
            continue

        # Annotate storage accesses: storage[0x00...] → add comment
        if storage_resolver and "storage[" in stripped:
            slot_match = re.search(r"storage\[(\w+)\]", stripped)
            if slot_match:
                slot_val = slot_match.group(1)
                annotation = storage_resolver.annotate_slot_hex(slot_val)
                if annotation:
                    enriched.append(f"{line}  {annotation}")
                    continue

        # Annotate LOG instructions with event names
        if abi_enricher and stripped.startswith("log") and "memory[" in stripped:
            topic_match = re.search(r"topic0=(0x[a-fA-F0-9]{64})", stripped)
            if topic_match:
                annotation = abi_enricher.format_event_annotation(topic_match.group(1))
                if annotation:
                    indent = line[:len(line) - len(line.lstrip())]
                    enriched.append(line)
                    enriched.append(f"{indent}{annotation}")
                    continue
            enriched.append(line)
            continue

        enriched.append(line)

    return "\n".join(enriched)