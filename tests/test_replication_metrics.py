"""
Tests for structured Solidity replication precision/recall metrics.
"""

from src.replication_metrics import (
    aggregate_replication_scores,
    evaluate_replication,
    extract_solidity_facts,
)


class TestSolidityFactExtraction:
    def test_extracts_structural_facts(self):
        code = """
        function transfer(address to, uint amount) public returns (bool) {
            require(to != address(0), "bad recipient");
            balances[msg.sender] -= amount;
            balances[to] += amount;
            emit Transfer(msg.sender, to, amount);
            return true;
        }
        """

        facts = extract_solidity_facts(code)

        assert "function_name:transfer" in facts["abi"]
        assert "param_type:0:address" in facts["abi"]
        assert "param_type:1:uint256" in facts["abi"]
        assert "return_type:0:bool" in facts["abi"]
        assert facts["visibility"] == {"public"}
        assert facts["event"] == {"transfer"}
        assert "require:param_0!=address(0)" in facts["guard"]
        assert facts["state_write"] == {
            "balances[msg.sender]",
            "balances[param_0]",
        }

    def test_parameter_renames_do_not_penalize_guard_matching(self):
        reference = """
        function setOwner(address newOwner) external onlyOwner {
            require(newOwner != address(0), "zero");
            owner = newOwner;
        }
        """
        candidate = """
        function setOwner(address var_1) external onlyOwner {
            require(var_1 != address(0), "zero");
            owner = var_1;
        }
        """

        evaluation = evaluate_replication(reference, candidate)

        assert evaluation.by_category["guard"].recall == 1.0
        assert evaluation.by_category["modifier"].f1 == 1.0
        assert evaluation.overall.f1 == 1.0


class TestReplicationEvaluation:
    def test_missing_state_write_reduces_recall(self):
        reference = """
        function transfer(address to, uint256 amount) public returns (bool) {
            balances[msg.sender] -= amount;
            balances[to] += amount;
            emit Transfer(msg.sender, to, amount);
            return true;
        }
        """
        candidate = """
        function transfer(address to, uint256 amount) public returns (bool) {
            balances[msg.sender] -= amount;
            emit Transfer(msg.sender, to, amount);
            return true;
        }
        """

        evaluation = evaluate_replication(reference, candidate)
        state_write_score = evaluation.by_category["state_write"]

        assert state_write_score.true_positives == 1
        assert state_write_score.false_negatives == 1
        assert state_write_score.recall == 0.5
        assert "balances[param_0]" in evaluation.missing_facts["state_write"]

    def test_extra_call_reduces_precision(self):
        reference = """
        function approve(address spender, uint256 amount) public returns (bool) {
            _approve(msg.sender, spender, amount);
            return true;
        }
        """
        candidate = """
        function approve(address spender, uint256 amount) public returns (bool) {
            _approve(msg.sender, spender, amount);
            _afterApprove(spender, amount);
            return true;
        }
        """

        evaluation = evaluate_replication(reference, candidate)
        call_score = evaluation.by_category["call"]

        assert call_score.true_positives == 1
        assert call_score.false_positives == 1
        assert call_score.precision == 0.5
        assert "_afterapprove" in evaluation.extra_facts["call"]


class TestAggregateReplicationScores:
    def test_aggregates_mean_and_micro_scores(self):
        first = evaluate_replication(
            "function a() public { count = 1; }",
            "function a() public { count = 1; }",
        ).to_dict()
        second = evaluate_replication(
            "function b() public { count = 1; owner = msg.sender; }",
            "function b() public { count = 1; }",
        ).to_dict()

        summary = aggregate_replication_scores(
            [
                {
                    "replication_precision": first["overall"]["precision"],
                    "replication_recall": first["overall"]["recall"],
                    "replication_f1": first["overall"]["f1"],
                    "metadata": {"replication": first},
                },
                {
                    "replication_precision": second["overall"]["precision"],
                    "replication_recall": second["overall"]["recall"],
                    "replication_f1": second["overall"]["f1"],
                    "metadata": {"replication": second},
                },
            ]
        )

        assert summary["precision_mean"] == 1.0
        assert 0 < summary["recall_mean"] < 1.0
        assert 0 < summary["f1_mean"] < 1.0
        assert summary["micro"]["false_negatives"] == 1
        assert summary["by_category_micro"]["state_write"]["false_negatives"] == 1
