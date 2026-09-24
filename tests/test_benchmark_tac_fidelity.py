"""Local-solc fidelity benchmark checks; no model or network access."""

from collections import Counter

import pytest

from scripts import benchmark_tac_fidelity as benchmark


@pytest.fixture(scope="module")
def report():
    if benchmark.SOLC_VERSION not in {
        str(version) for version in benchmark.solcx.get_installed_solc_versions()
    }:
        pytest.skip(f"locally installed solc {benchmark.SOLC_VERSION} required")
    return benchmark.run_benchmark()


def test_fixture_has_unique_functions_in_each_stratum():
    specs, source = benchmark._fixtures()
    assert len(specs) == 36
    assert Counter(spec[0] for spec in specs) == {name: 6 for name in benchmark.OPCODES}
    assert len({(contract, name) for _, contract, name, _ in specs}) == 36
    assert source == benchmark._fixtures()[1]
    assert "pragma solidity ^0.8.20;" in source


def test_source_map_uses_opcode_pcs_not_push_data_or_dispatcher():
    # 0xf3 inside PUSH1 is not a RETURN; the actual RETURN is at PC 2.
    assert benchmark._source_mapped_opcode_pcs("60f3f3", "0:3:0;0:3:0", 0, (0, 3), "RETURN") == [2]
    assert benchmark._source_mapped_opcode_pcs("60f3f3", "0:3:0;4:3:0", 0, (0, 3), "RETURN") == []
    with pytest.raises(ValueError, match="source-map/instruction mismatch"):
        benchmark._source_mapped_opcode_pcs("60f3f3", "0:3:0", 0, (0, 3), "RETURN")


def test_missing_local_compiler_refuses_download(monkeypatch):
    monkeypatch.setattr(benchmark.solcx, "get_installed_solc_versions", lambda: [])

    def no_compilation(*_args, **_kwargs):
        pytest.fail("must not invoke compilation or download for a missing compiler")

    monkeypatch.setattr(benchmark.solcx, "compile_standard", no_compilation)
    with pytest.raises(RuntimeError, match="must already be installed locally"):
        benchmark.run_benchmark()


def test_report_compares_abi_selectors_with_source_mapped_tac(report):
    summary = report["summary"]
    assert report["schema_version"] == 1
    assert report["lineage"]["compiler_version"].startswith("0.8.20+commit.")
    assert report["lineage"]["compiler_settings"]["evmVersion"] == "shanghai"
    assert report["lineage"]["compiler_settings"]["optimizer"] == {"enabled": True, "runs": 200}
    assert len(report["lineage"]["source_sha256"]) == 64
    assert len(report["lineage"]["compiler_binary_sha256"]) == 64
    assert summary["contracts"] == summary["cases"] == 36
    assert summary["unique_selectors"] == len({case["selector"] for case in report["cases"]}) == 36
    assert summary["selector_coverage"] == {"found": 36, "total": 36, "rate": 1}
    assert summary["source_mapped_opcode_reachability"] == {
        "found": 36,
        "total": 36,
        "rate": 1,
    }
    assert summary["reachable_tac_opcode_coverage"] == {
        "found": 36,
        "total": 36,
        "rate": 1,
    }
    assert summary["end_to_end_tac_opcode_coverage"] == {
        "found": 36,
        "total": 36,
        "rate": 1,
    }
    assert summary["failing_cases"] == 0
    assert report["sample_failures"] == []
    assert set(report["strata"]) == set(benchmark.OPCODES)
    for name, stratum in report["strata"].items():
        assert stratum["cases"] == stratum["selector_coverage"]["total"] == 6
        assert stratum["expected_opcode"] == benchmark.OPCODES[name]
        assert stratum["source_mapped_opcode_reachability"]["total"] == 6
        assert stratum["end_to_end_tac_opcode_coverage"]["found"] == 6
    assert all(len(case["source_mapped_opcode_pcs"]) > 0 for case in report["cases"])
    assert all(len(case["runtime_sha256"]) == 64 for case in report["cases"])
    assert sum(s["failing_cases"] for s in report["strata"].values()) == summary["failing_cases"]
    assert "does not establish" in report["interpretation"]


def test_source_mapped_witnesses_are_not_hidden_by_unrelated_tac_opcodes(report):
    cases = {case["signature"]: case for case in report["cases"]}
    simple_getter = cases["read0()"]
    assert (
        simple_getter["source_mapped_opcode_pcs"]
        == (simple_getter["reachable_source_opcode_pcs"])
        == simple_getter["tac_witness_pcs"]
    )
    assert not simple_getter["failure_reasons"]

    for signature in ("read3(uint256)", "guard3(uint256)"):
        case = cases[signature]
        assert case["source_mapped_opcode_pcs"]
        assert case["source_opcode_blocks"]
        assert set(case["tac_witness_pcs"]).issubset(case["reachable_source_opcode_pcs"])
        if case["reachable_source_opcode_pcs"] == case["source_mapped_opcode_pcs"]:
            assert case["tac_witness_pcs"] == case["source_mapped_opcode_pcs"]
            assert not case["failure_reasons"]
        else:
            assert "compiler source-mapped opcode missing from function CFG reachability" in (
                case["failure_reasons"]
            )

    guard = cases["guard3(uint256)"]
    if not guard["reachable_source_opcode_pcs"]:
        assert guard["rendered_tac_opcode"]  # ABI decoder REVERT is not the source guard.
        assert not guard["tac_witness_pcs"]
