"""
Tests for structured Solidity replication precision/recall metrics.
"""

from src.replication_metrics import (
    aggregate_replication_scores,
    behavior_only_replication_micro,
    evaluate_replication,
    extract_solidity_facts,
)
from src.training_pipeline import (
    compute_metadata_segment_metrics,
    evaluate_bytecode_semantics,
    load_curated_evaluation_benchmarks,
    validate_generated_solidity,
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

    def test_extracts_nested_mapping_state_writes(self):
        code = """
        function approve(address spender, uint256 amount) public returns (bool) {
            allowed[msg.sender][spender] = amount;
            freemintAddresses[addresses[i]] = true;
            return true;
        }
        """

        facts = extract_solidity_facts(code)

        assert facts["state_write"] == {
            "allowed[msg.sender][param_0]",
            "freemintaddresses[addresses[i]]",
        }

    def test_nested_mapping_parameter_type_changes_reduce_abi_f1(self):
        reference = (
            "function f(mapping(address => uint256) storage ledger) internal "
            "{ ledger[msg.sender] = 1; }"
        )
        candidate = (
            "function f(mapping(address => bool) storage ledger) internal "
            "{ ledger[msg.sender] = true; }"
        )

        assert "param_type:0:mapping(address=>uint256)" in extract_solidity_facts(reference)["abi"]
        assert "param_type:0:mapping(address=>bool)" in extract_solidity_facts(candidate)["abi"]
        result = evaluate_replication(reference, candidate)
        assert result.by_category["abi"].f1 < 1.0
        assert "modifier" not in result.by_category

    def test_nested_function_type_signature_keeps_outer_parameter_and_return(self):
        code = (
            "function f(function(uint256) external returns (bool) callback) public "
            "returns (uint256) { return callback(1) ? 1 : 0; }"
        )
        facts = extract_solidity_facts(code)

        assert "param_count:1" in facts["abi"]
        assert "param_type:0:function(uint256)external returns(bool)" in facts["abi"]
        assert "return_type:0:uint256" in facts["abi"]
        assert facts["visibility"] == {"public"}

    def test_integer_and_fixed_bytes_casts_are_not_calls(self):
        code = (
            "function f(uint8 value) external pure returns (bytes32) { "
            "_record(bytes32(uint256(value))); return bytes32(uint8(value)); }"
        )

        assert extract_solidity_facts(code)["call"] == {"_record"}

    def test_bodyless_declaration_does_not_borrow_next_function_body(self):
        code = (
            "abstract contract C { function first() public virtual; "
            "function second() public { _unrelatedCall(); } }"
        )

        facts = extract_solidity_facts(code)

        assert "function_name:first" in facts["abi"]
        assert "call" not in facts

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

    def test_extra_generated_facts_are_bucketed_as_grounded_hallucinations(self):
        reference = """
        function approve(address spender, uint256 amount) public returns (bool) {
            _approve(msg.sender, spender, amount);
            return true;
        }
        """
        candidate = """
        function approve(address spender, uint256 amount) public returns (bool) {
            require(msg.sender == owner, "owner");
            _approve(msg.sender, spender, amount);
            _afterApprove(spender, amount);
            owner = msg.sender;
            emit Approval(msg.sender, spender, amount);
            return false;
        }
        """

        evaluation = evaluate_replication(reference, candidate)

        buckets = evaluation.hallucination_buckets
        assert "call:_afterapprove" in buckets["unsupported_calls"]
        assert any(fact.startswith("guard:require:") for fact in buckets["invented_guards"])
        assert "state_write:owner" in buckets["invented_state_writes"]
        assert "event:approval" in buckets["invented_events"]
        assert "return:false" in buckets["unsupported_return_expressions"]
        assert evaluation.groundedness_score < 1.0


class TestAggregateReplicationScores:
    def test_behavior_only_micro_excludes_abi_and_requires_behavior_facts(self):
        categories = {
            "abi": {"true_positives": 9000, "false_positives": 0, "false_negatives": 0},
            "modifier": {"true_positives": 100, "false_positives": 0, "false_negatives": 0},
            "call": {"true_positives": 0, "false_positives": 3, "false_negatives": 2},
            "state_write": {"true_positives": 1, "false_positives": 0, "false_negatives": 1},
        }

        assert behavior_only_replication_micro({"abi": categories["abi"]}) is None
        behavior = behavior_only_replication_micro(categories)

        assert behavior["true_positives"] == 1
        assert behavior["false_positives"] == 3
        assert behavior["false_negatives"] == 3
        assert behavior["f1"] == 0.25

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
        assert summary["fact_error_totals"] == {"matched": 10, "extra": 0, "missing": 1}
        assert summary["by_category_micro"]["state_write"]["false_negatives"] == 1
        assert summary["behavior_only_micro"]["f1"] == 0.8
        assert summary["behavior_only_micro"]["true_positives"] == 2
        assert summary["micro"]["true_positives"] == 10
        assert summary["category_gap_summary"][0]["category"] == "state_write"
        assert summary["category_gap_summary"][0]["primary_error"] == "recall"

    def test_aggregates_hallucination_bucket_counts(self):
        row = evaluate_replication(
            "function a() public { count = 1; }",
            "function a() public { count = 1; owner = msg.sender; emit OwnerSet(owner); }",
        ).to_dict()

        summary = aggregate_replication_scores(
            [
                {
                    "replication_precision": row["overall"]["precision"],
                    "replication_recall": row["overall"]["recall"],
                    "replication_f1": row["overall"]["f1"],
                    "metadata": {"replication": row},
                }
            ]
        )

        assert summary["hallucination_buckets"]["invented_state_writes"] == 1
        assert summary["hallucination_buckets"]["invented_events"] == 1
        assert summary["hallucination_rate"] > 0


class TestBytecodeGroundedEvaluation:
    def test_bytecode_semantic_checks_catch_high_text_behavior_mismatch(self):
        reference = """
        function allowed(address user) public view returns (bool) {
            return user == owner;
        }
        """
        candidate = """
        function allowed(address user) public view returns (bool) {
            return true;
        }
        """
        solidity_validity = validate_generated_solidity(candidate, allow_compiler=False)

        result = evaluate_bytecode_semantics(
            reference,
            candidate,
            {
                "bytecode": "0x60015414600057fd5b6001f3",
                "function_selector": "0x12345678",
            },
            solidity_validity=solidity_validity,
        )

        assert result.checked is True
        assert result.score < 1.0
        assert "return_mismatch" in result.mismatch_buckets
        assert "guard_mismatch" in result.mismatch_buckets

    def test_opcode_and_control_flow_segments_cover_rare_slices(self):
        summary = compute_metadata_segment_metrics(
            [
                {
                    "metrics": {
                        "semantic_similarity": 0.75,
                        "normalized_edit_distance": 0.25,
                        "replication_f1": 0.5,
                        "solidity_valid": True,
                    },
                    "metadata": {"bytecode": "0x5ff45f57fd"},
                }
            ],
            segment_fields=("opcode_group", "control_flow"),
        )

        assert summary["coverage"]["opcode_group"]["values"]["delegatecall"] == 1
        assert summary["coverage"]["opcode_group"]["values"]["push0"] == 1
        assert summary["coverage"]["control_flow"]["values"]["branching"] == 1
        assert summary["opcode_control_flow_coverage"]["opcode_groups"]["delegatecall"] == 1
        assert summary["segments"]["opcode_group"]["delegatecall"]["count"] == 1

    def test_curated_golden_and_robustness_benchmarks_are_loadable(self):
        suites = load_curated_evaluation_benchmarks("test_data/evaluation")

        assert "golden" in suites
        assert "robustness" in suites
        assert any(case["case_id"] == "golden_create2_factory" for case in suites["golden"])
        assert any(case.get("expected_failure") for case in suites["robustness"])
