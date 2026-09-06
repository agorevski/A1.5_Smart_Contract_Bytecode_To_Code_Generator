from types import SimpleNamespace

from src.contract_reconstruction import (
    assemble_reconstructed_contract,
    build_contract_quality,
    build_function_quality,
    build_reconstruction_plan,
)


def test_reconstruction_plan_handles_contract_fallback_chunk():
    analyzer = SimpleNamespace(
        instructions=[SimpleNamespace(name="SLOAD"), SimpleNamespace(name="SSTORE")],
        basic_blocks={
            "block_0000": SimpleNamespace(
                id="block_0000",
                successors=[],
                metadata={"raw_instructions": [{"name": "SLOAD"}, {"name": "SSTORE"}]},
            )
        },
        functions={},
    )
    plan = build_reconstruction_plan(
        "0x6000",
        analyzer,
        {"contract": "contract:\n  storage[temp_1] = temp_2\n  return memory[0:32]"},
    )

    assert plan["strategy"] == "semantic_function_chunks"
    assert plan["chunk_count"] == 1
    assert plan["semantic_chunks"][0]["basic_blocks"] == ["block_0000"]
    assert plan["semantic_chunks"][0]["storage_writes"] == ["temp_1"]
    assert plan["contract_facts"]["storage_writes"] == ["temp_1"]


def test_assembler_strips_nested_contract_wrapper():
    analyzer = SimpleNamespace(instructions=[], basic_blocks={}, functions={})
    plan = build_reconstruction_plan("0x6000", analyzer, {"contract": "main:\n  stop()"})

    source = assemble_reconstructed_contract(
        {"contract": "contract Inner {\n    function recovered() public {}\n}"},
        analyzer,
        reconstruction_plan=plan,
        contract_metadata={"metadata": {"contractName": "RecoveredToken"}},
    )

    assert "contract RecoveredToken {" in source
    assert "contract Inner" not in source
    assert "function recovered() public {}" in source


def test_quality_labels_scaffold_only_as_non_deployable():
    function_quality = build_function_quality(
        validation={"valid": True, "method": "scaffold", "compiler_checked": False},
        diagnostics={"tac_truncated": True},
        source="model_inference",
    )
    contract_quality = build_contract_quality(
        {"valid": True, "method": "scaffold", "compiler_checked": False},
        [
            {
                "name": "func_00000000",
                "status": "ok",
                "source": "model_inference",
                "diagnostics": {"tac_truncated": True},
            }
        ],
        {"model_inference": 1, "exact_match": 0, "error": 0, "unknown": 0},
        {"chunk_count": 1, "contract_facts": {"proxy": {"is_proxy_like": False}}},
    )

    assert function_quality["severity"] == "warning"
    assert function_quality["deployable"] is False
    assert contract_quality["scaffold_only"] is True
    assert contract_quality["deployable"] is False
    assert contract_quality["truncated_functions"] == ["func_00000000"]


def test_reconciles_shared_state_and_identical_helpers():
    plan = {}
    source = assemble_reconstructed_contract(
        {
            "a": "contract A { uint256 private balance; function helper() internal {} function a() public { balance = 1; helper(); } }",
            "b": "contract B { uint256 private balance; function helper() internal {} function b() public { balance = 2; helper(); } }",
        },
        SimpleNamespace(), reconstruction_plan=plan,
    )
    assert source.count("uint256 private balance;") == 1
    assert source.count("function helper()") == 1
    assert "function a()" in source and "function b()" in source
    assert len(plan["reconciliation"]["duplicate_declarations_merged"]) == 2
    assert plan["reconciliation"]["storage_consistency"] == "unverified"


def test_preserves_conflicting_state_and_function_bodies():
    plan = {}
    source = assemble_reconstructed_contract(
        {
            "a": "uint256 value; function helper() internal { value = 1; }",
            "b": "address value; function helper() internal { revert(); }",
        },
        SimpleNamespace(), reconstruction_plan=plan,
    )
    assert "uint256 value;" in source and "address value;" in source
    assert source.count("function helper() internal") == 2
    assert len(plan["reconciliation"]["conflicts"]) == 2
    assert plan["reconciliation"]["storage_consistency"] == "conflict"
    quality = build_contract_quality({"valid": True, "compiler_checked": True}, reconstruction_plan=plan)
    assert quality["deployable"] is False


def test_assembly_strings_overloads_and_unresolved_helpers():
    plan = {}
    source = assemble_reconstructed_contract(
        {
            "a": 'contract A { function f(uint x) public { assembly { mstore(0, x) } missing(x); string memory s = "}"; } }',
            "b": "function f(address x) public { assembly { sstore(0, x) } }",
        },
        SimpleNamespace(), reconstruction_plan=plan,
    )
    assert "sstore(0, x)" in source and "mstore(0, x)" in source
    assert 'string memory s = "}"' in source
    assert plan["reconciliation"]["conflicts"] == []
    assert plan["reconciliation"]["unresolved_helpers"] == ["missing"]


def test_conflicting_storage_order_is_not_silently_accepted():
    plan = {}
    assemble_reconstructed_contract(
        {"a": "uint a; uint b;", "b": "uint b; uint a;"},
        SimpleNamespace(), reconstruction_plan=plan,
    )
    assert plan["reconciliation"]["storage_consistency"] == "conflict"
    assert plan["reconciliation"]["conflicts"][0]["kind"] == "storage_layout"


def test_contract_text_inside_comments_does_not_replace_real_body():
    source = assemble_reconstructed_contract(
        {"a": "// contract Decoy { }\ncontract Real { function f() public {} }"},
        SimpleNamespace(),
    )
    assert "function f()" in source
