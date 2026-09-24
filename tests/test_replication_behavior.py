"""Dependency-light mutation coverage for behavior facts and runtime fixtures."""

import pytest

from src.replication_metrics import aggregate_replication_scores, evaluate_replication
from src.runtime_equivalence import execute_equivalence_subset


@pytest.mark.parametrize("reference,candidate,category", [
    ("balance += amount;", "balance -= amount;", "state_write"),
    ("balance = amount;", "balance = 0;", "state_write"),
    ("balance++;", "balance--;", "state_write"),
    ("balance &= amount;", "balance |= amount;", "state_write"),
    ("++balance;", "--balance;", "state_write"),
    ("delete balances[to];", "delete balances[msg.sender];", "state_write"),
    ("token.transfer(to, amount);", "other.transfer(to, amount);", "call"),
    ("token.transfer(to, amount);", "token.transfer(msg.sender, amount);", "call"),
    ("token.transfer(to, amount);", "token.transfer(to, 0);", "call"),
    ("IERC20(token).transfer(to, amount);", "IERC20(other).transfer(to, amount);", "call"),
    ("to.call{value: amount}(data);", "to.call{value: 0}(data);", "call"),
    ("emit Transfer(to, amount);", "emit Transfer(msg.sender, amount);", "event"),
    ('emit Message("yes");', 'emit Message("no");', "event"),
    ("if (amount > 0) balance++;", "if (amount == 0) balance++;", "control_flow"),
])
def test_behavior_mutations_cannot_score_perfectly(reference, candidate, category):
    wrap = lambda body: f"function send(address to, uint amount) public {{ {body} }}"
    result = evaluate_replication(wrap(reference), wrap(candidate))
    assert result.overall.f1 < 1
    assert result.by_category[category].f1 < 1
    assert result.by_category[category].false_negatives > 0
    assert result.by_category[category].false_positives > 0


def test_behavior_facts_preserve_parameter_rename_and_equivalent_update():
    reference = "function f(address to,uint amount) public { balance += amount; token.transfer(to,amount); emit Paid(to,amount); }"
    candidate = "function f(address recipient,uint value) public { balance = balance + value; token.transfer(recipient,value); emit Paid(recipient,value); }"
    assert evaluate_replication(reference, candidate).overall.f1 == 1


def test_missing_generation_counts_every_reference_fact_as_false_negative():
    result = evaluate_replication("function f() public { balance = 1; }", "")
    assert result.overall.true_positives == 0
    assert result.overall.false_negatives == result.reference_fact_count > 0
    assert result.overall.f1 == 0


def test_missing_replication_payload_never_produces_complete_micro():
    summary = aggregate_replication_scores([{"replication_f1": 0, "metadata": {
        "error": "failed", "error_kind": "evaluator",
    }}])
    assert "micro" not in summary
    assert summary["micro_incomplete"] is True
    assert summary["evaluator_error_count"] == 1
    assert summary["replication_payload_coverage"] == 0


def test_method_name_grounding_does_not_prove_call_arguments():
    result = evaluate_replication(
        "function f(address to) public {}",
        "function f(address to) public { token.transfer(to,100); }",
        grounding_facts={"call": ["transfer"], "member_call": ["transfer"]},
    )
    assert result.hallucination_buckets["unsupported_calls"] == ["call:token.transfer(param_0,100)"]


def test_runtime_subset_refuses_stateful_or_missing_fixtures():
    assert execute_equivalence_subset("60005400", "60005400", [""])["checked"] is False
    assert execute_equivalence_subset("00", "00", [])["checked"] is False


def test_runtime_subset_independently_detects_return_mutation():
    pytest.importorskip("eth")
    one = "600160005260206000f3"
    two = "600260005260206000f3"
    same = execute_equivalence_subset(one, one, [""])
    changed = execute_equivalence_subset(one, two, [""])
    assert same["checked"] is True and same["match"] is True
    assert changed["checked"] is True and changed["match"] is False
