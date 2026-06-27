"""Structured replication metrics for Solidity decompilation outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set


FactMap = Dict[str, Set[str]]

_VISIBILITIES = {"public", "private", "internal", "external"}
_MUTABILITY = {"view", "pure"}
_SIGNATURE_QUALIFIERS = {
    "calldata",
    "external",
    "internal",
    "memory",
    "override",
    "payable",
    "private",
    "public",
    "pure",
    "returns",
    "storage",
    "view",
    "virtual",
}
_PARAM_QUALIFIERS = {"memory", "storage", "calldata", "indexed"}
_CONTROL_KEYWORDS = {"if", "for", "while"}
_CALL_EXCLUSIONS = {
    "assert",
    "catch",
    "emit",
    "for",
    "function",
    "if",
    "modifier",
    "require",
    "return",
    "returns",
    "revert",
    "while",
}
_RESERVED_IDENTIFIERS = {
    "address",
    "bool",
    "bytes",
    "calldata",
    "false",
    "memory",
    "msg",
    "payable",
    "return",
    "returns",
    "storage",
    "string",
    "true",
    "uint",
    "uint256",
}


@dataclass(frozen=True)
class PrecisionRecallF1:
    """Precision/recall/F1 counts for a fact set comparison."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    @classmethod
    def from_counts(cls, true_positives: int, false_positives: int, false_negatives: int):
        precision_denominator = true_positives + false_positives
        recall_denominator = true_positives + false_negatives
        precision = true_positives / precision_denominator if precision_denominator else 1.0
        recall = true_positives / recall_denominator if recall_denominator else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return cls(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1=f1,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True)
class ReplicationEvaluation:
    """Structured comparison between reference and generated Solidity."""

    overall: PrecisionRecallF1
    by_category: Dict[str, PrecisionRecallF1]
    reference_fact_count: int
    candidate_fact_count: int
    missing_facts: Dict[str, List[str]]
    extra_facts: Dict[str, List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall.to_dict(),
            "by_category": {
                category: score.to_dict() for category, score in sorted(self.by_category.items())
            },
            "reference_fact_count": self.reference_fact_count,
            "candidate_fact_count": self.candidate_fact_count,
            "missing_facts": self.missing_facts,
            "extra_facts": self.extra_facts,
        }


def extract_solidity_facts(solidity_code: str) -> FactMap:
    """Extract comparable semantic facts from a Solidity function or contract."""
    facts: FactMap = {}
    code = _strip_comments(solidity_code or "")
    masked_code = _mask_string_literals(code)

    signature = _extract_function_signature(masked_code)
    body = _extract_function_body(masked_code, signature["match_end"])
    params = signature["params"]
    local_names = _extract_local_variable_names(body)
    non_state_names = set(params["names"]) | local_names
    aliases = {
        _normalize_identifier(name): f"param_{index}" for index, name in enumerate(params["names"])
    }

    _add_signature_facts(facts, signature)
    _add_event_facts(facts, body)
    _add_call_facts(facts, body, signature.get("function_name"))
    _add_guard_facts(facts, body, aliases)
    _add_state_write_facts(facts, body, non_state_names, aliases)
    _add_return_facts(facts, body, aliases)
    _add_control_flow_facts(facts, body)

    return facts


def evaluate_replication(reference_code: str, candidate_code: str) -> ReplicationEvaluation:
    """Compare reference and candidate Solidity using structured fact overlap."""
    reference = extract_solidity_facts(reference_code)
    candidate = extract_solidity_facts(candidate_code)
    categories = sorted(set(reference) | set(candidate))

    by_category: Dict[str, PrecisionRecallF1] = {}
    missing: Dict[str, List[str]] = {}
    extra: Dict[str, List[str]] = {}
    total_tp = total_fp = total_fn = 0

    for category in categories:
        reference_values = reference.get(category, set())
        candidate_values = candidate.get(category, set())
        true_positives = len(reference_values & candidate_values)
        false_positives = len(candidate_values - reference_values)
        false_negatives = len(reference_values - candidate_values)

        by_category[category] = PrecisionRecallF1.from_counts(
            true_positives, false_positives, false_negatives
        )
        total_tp += true_positives
        total_fp += false_positives
        total_fn += false_negatives

        category_missing = sorted(reference_values - candidate_values)
        category_extra = sorted(candidate_values - reference_values)
        if category_missing:
            missing[category] = category_missing
        if category_extra:
            extra[category] = category_extra

    reference_fact_count = sum(len(values) for values in reference.values())
    candidate_fact_count = sum(len(values) for values in candidate.values())

    return ReplicationEvaluation(
        overall=PrecisionRecallF1.from_counts(total_tp, total_fp, total_fn),
        by_category=by_category,
        reference_fact_count=reference_fact_count,
        candidate_fact_count=candidate_fact_count,
        missing_facts=missing,
        extra_facts=extra,
    )


def aggregate_replication_scores(metrics: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-example replication metrics into mean and micro scores."""
    metric_rows = list(metrics)
    if not metric_rows:
        return {}

    precisions = _numeric_metric_values(metric_rows, "replication_precision")
    recalls = _numeric_metric_values(metric_rows, "replication_recall")
    f1s = _numeric_metric_values(metric_rows, "replication_f1")

    summary: Dict[str, Any] = {}
    if precisions:
        summary["precision_mean"] = sum(precisions) / len(precisions)
    if recalls:
        summary["recall_mean"] = sum(recalls) / len(recalls)
    if f1s:
        summary["f1_mean"] = sum(f1s) / len(f1s)
        summary["pct_above_0_8_f1"] = sum(1 for score in f1s if score > 0.8) / len(f1s)

    overall_counts = {"tp": 0, "fp": 0, "fn": 0}
    category_counts: Dict[str, Dict[str, int]] = {}

    for row in metric_rows:
        replication = _replication_payload(row)
        if not replication:
            continue

        overall = replication.get("overall", {})
        overall_counts["tp"] += int(overall.get("true_positives", 0))
        overall_counts["fp"] += int(overall.get("false_positives", 0))
        overall_counts["fn"] += int(overall.get("false_negatives", 0))

        for category, score in replication.get("by_category", {}).items():
            counts = category_counts.setdefault(category, {"tp": 0, "fp": 0, "fn": 0})
            counts["tp"] += int(score.get("true_positives", 0))
            counts["fp"] += int(score.get("false_positives", 0))
            counts["fn"] += int(score.get("false_negatives", 0))

    if overall_counts["tp"] or overall_counts["fp"] or overall_counts["fn"]:
        summary["micro"] = PrecisionRecallF1.from_counts(
            overall_counts["tp"],
            overall_counts["fp"],
            overall_counts["fn"],
        ).to_dict()
        summary["by_category_micro"] = {
            category: PrecisionRecallF1.from_counts(
                counts["tp"],
                counts["fp"],
                counts["fn"],
            ).to_dict()
            for category, counts in sorted(category_counts.items())
        }

    return summary


def _numeric_metric_values(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _replication_payload(row: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    replication = metadata.get("replication")
    if not isinstance(replication, Mapping):
        return None
    return replication


def _add_fact(facts: FactMap, category: str, value: str):
    facts.setdefault(category, set()).add(value)


def _strip_comments(code: str) -> str:
    code = re.sub(r"//.*", "", code)
    return re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)


def _mask_string_literals(code: str) -> str:
    return re.sub(r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')', '""', code)


def _extract_function_signature(code: str) -> Dict[str, Any]:
    match = re.search(
        r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)?\s*" r"\((?P<params>.*?)\)\s*(?P<tail>[^{;]*)",
        code,
        flags=re.DOTALL,
    )
    if not match:
        return {
            "function_name": None,
            "params": {"types": [], "names": []},
            "returns": [],
            "visibility": None,
            "mutability": None,
            "is_payable": False,
            "modifiers": [],
            "match_end": 0,
        }

    tail = match.group("tail") or ""
    returns_match = re.search(r"\breturns\s*\((?P<returns>.*?)\)", tail, re.DOTALL)
    returns = (
        _parse_parameter_list(returns_match.group("returns"))
        if returns_match
        else {
            "types": [],
            "names": [],
        }
    )
    tail_without_returns = (
        tail[: returns_match.start()] + tail[returns_match.end() :] if returns_match else tail
    )
    tail_tokens = [
        token.lower() if token.lower() in _SIGNATURE_QUALIFIERS else token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", tail_without_returns)
    ]

    visibility = next((token for token in tail_tokens if token in _VISIBILITIES), None)
    mutability = next((token for token in tail_tokens if token in _MUTABILITY), None)
    modifiers = [
        token
        for token in tail_tokens
        if token not in _SIGNATURE_QUALIFIERS and token not in _VISIBILITIES
    ]

    return {
        "function_name": match.group(1),
        "params": _parse_parameter_list(match.group("params")),
        "returns": returns["types"],
        "visibility": visibility,
        "mutability": mutability,
        "is_payable": "payable" in tail_tokens,
        "modifiers": sorted(set(modifiers)),
        "match_end": match.end(),
    }


def _add_signature_facts(facts: FactMap, signature: Mapping[str, Any]):
    function_name = signature.get("function_name")
    if function_name:
        _add_fact(facts, "abi", f"function_name:{_normalize_identifier(function_name)}")

    param_types = signature.get("params", {}).get("types", [])
    _add_fact(facts, "abi", f"param_count:{len(param_types)}")
    for index, param_type in enumerate(param_types):
        _add_fact(facts, "abi", f"param_type:{index}:{param_type}")

    returns = signature.get("returns", [])
    _add_fact(facts, "abi", f"return_count:{len(returns)}")
    for index, return_type in enumerate(returns):
        _add_fact(facts, "abi", f"return_type:{index}:{return_type}")

    visibility = signature.get("visibility")
    if visibility:
        _add_fact(facts, "visibility", visibility)

    mutability = signature.get("mutability")
    if mutability:
        _add_fact(facts, "mutability", mutability)

    if signature.get("is_payable"):
        _add_fact(facts, "mutability", "payable")

    for modifier in signature.get("modifiers", []):
        _add_fact(facts, "modifier", _normalize_identifier(modifier))


def _extract_function_body(code: str, match_end: int) -> str:
    start = code.find("{", max(0, match_end - 1))
    if start < 0:
        return code
    end = _find_matching_brace(code, start)
    return code[start + 1 : end] if end is not None else code[start + 1 :]


def _find_matching_brace(text: str, start_index: int) -> Optional[int]:
    depth = 0
    for index in range(start_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _parse_parameter_list(parameter_text: str) -> Dict[str, List[str]]:
    types: List[str] = []
    names: List[str] = []

    for parameter in _split_top_level_commas(parameter_text or ""):
        parameter = parameter.strip()
        if not parameter:
            continue
        types.append(_canonical_type(parameter))
        name = _parameter_name(parameter)
        if name:
            names.append(name)

    return {"types": types, "names": names}


def _split_top_level_commas(text: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    for char in text:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth = max(0, depth - 1)

        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current))
    return parts


def _canonical_type(parameter: str) -> str:
    parameter = parameter.split("=")[0].strip()
    parameter = re.sub(
        rf"\b({'|'.join(sorted(_PARAM_QUALIFIERS))})\b",
        " ",
        parameter,
    )
    name = _parameter_name(parameter)
    if name:
        parameter = re.sub(rf"\s+{re.escape(name)}\s*$", "", parameter)
    parameter = re.sub(r"\buint\b", "uint256", parameter)
    parameter = re.sub(r"\bint\b", "int256", parameter)
    parameter = re.sub(r"\s+", " ", parameter.strip())
    parameter = re.sub(r"\s*=>\s*", "=>", parameter)
    parameter = re.sub(r"\s*([()[\],])\s*", r"\1", parameter)
    return parameter.lower()


def _parameter_name(parameter: str) -> Optional[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", parameter)
    if len(tokens) < 2:
        return None
    candidate = tokens[-1]
    if candidate.lower() in _PARAM_QUALIFIERS:
        return None
    return candidate


def _extract_local_variable_names(body: str) -> Set[str]:
    names: Set[str] = set()
    declaration_re = re.compile(
        r"\b(?:u?int(?:\d+)?|address|bool|string|bytes(?:\d+)?|"
        r"mapping\s*\([^;=]+\)|[A-Z][A-Za-z0-9_]*)"
        r"\s+(?:memory|storage|calldata\s+)?([A-Za-z_][A-Za-z0-9_]*)\b"
    )
    for match in declaration_re.finditer(body):
        names.add(match.group(1))
    return names


def _add_event_facts(facts: FactMap, body: str):
    for event_name in re.findall(r"\bemit\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", body):
        _add_fact(facts, "event", _normalize_identifier(event_name))


def _add_call_facts(facts: FactMap, body: str, function_name: Optional[str]):
    for method in re.findall(r"\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", body):
        _add_fact(facts, "member_call", _normalize_identifier(method))

    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", body):
        call_name = match.group(1)
        normalized = _normalize_identifier(call_name)
        if normalized in _CALL_EXCLUSIONS:
            continue
        if normalized in _RESERVED_IDENTIFIERS:
            continue
        if function_name and normalized == _normalize_identifier(function_name):
            continue
        _add_fact(facts, "call", normalized)


def _add_guard_facts(facts: FactMap, body: str, aliases: Mapping[str, str]):
    for keyword in ("require", "assert"):
        for args in _extract_call_arguments(body, keyword):
            condition = _split_top_level_commas(args)[0].strip() if args.strip() else ""
            if condition:
                _add_fact(
                    facts,
                    "guard",
                    f"{keyword}:{_normalize_expression(condition, aliases)}",
                )

    for args in _extract_call_arguments(body, "revert"):
        value = _normalize_expression(args, aliases) if args.strip() else "revert"
        _add_fact(facts, "guard", f"revert:{value}")


def _extract_call_arguments(body: str, call_name: str) -> List[str]:
    args: List[str] = []
    pattern = re.compile(rf"\b{re.escape(call_name)}\s*\(")
    for match in pattern.finditer(body):
        open_paren = body.find("(", match.start())
        close_paren = _find_matching_paren(body, open_paren)
        if close_paren is not None:
            args.append(body[open_paren + 1 : close_paren])
    return args


def _find_matching_paren(text: str, start_index: int) -> Optional[int]:
    depth = 0
    for index in range(start_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _add_state_write_facts(
    facts: FactMap,
    body: str,
    non_state_names: Set[str],
    aliases: Mapping[str, str],
):
    assignment_re = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*(?:\s*\[[^\]]+\])?"
        r"(?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*)?)\s*"
        r"(\+\+|--|\+=|-=|\*=|/=|%=|=(?!=))"
    )
    for match in assignment_re.finditer(body):
        lhs = match.group(1)
        root = re.match(r"[A-Za-z_][A-Za-z0-9_]*", lhs)
        if not root:
            continue
        root_name = root.group(0)
        normalized_root = _normalize_identifier(root_name)
        if root_name in non_state_names or normalized_root in _RESERVED_IDENTIFIERS:
            continue
        _add_fact(facts, "state_write", _normalize_state_reference(lhs, aliases))


def _add_return_facts(facts: FactMap, body: str, aliases: Mapping[str, str]):
    for match in re.finditer(r"\breturn\b\s*([^;]*);", body, flags=re.DOTALL):
        expression = match.group(1).strip()
        if expression:
            _add_fact(facts, "return", _normalize_expression(expression, aliases))


def _add_control_flow_facts(facts: FactMap, body: str):
    for keyword in _CONTROL_KEYWORDS:
        count = len(re.findall(rf"\b{keyword}\s*\(", body))
        if count:
            _add_fact(facts, "control_flow", f"{keyword}_count:{count}")


def _normalize_identifier(identifier: str) -> str:
    return re.sub(r"\s+", "", identifier).lower()


def _normalize_state_reference(reference: str, aliases: Mapping[str, str]) -> str:
    return _normalize_expression(reference, aliases)


def _normalize_expression(
    expression: str,
    aliases: Optional[Mapping[str, str]] = None,
) -> str:
    expression = re.sub(r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')', '""', expression)
    expression = re.sub(r"\buint\b", "uint256", expression)
    expression = re.sub(r"\bint\b", "int256", expression)
    if aliases:
        expression = re.sub(
            r"\b[A-Za-z_][A-Za-z0-9_]*\b",
            lambda match: aliases.get(
                _normalize_identifier(match.group(0)),
                match.group(0).lower(),
            ),
            expression,
        )
    expression = re.sub(r"\s+", "", expression)
    return expression.lower()
