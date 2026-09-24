"""Local-solc regression controls for the opt-in runtime identity comparator."""

from collections import Counter

import pytest

from scripts.benchmark_runtime_controls import (
    build_contracts,
    check_outcomes,
    run_benchmark,
    select_solc_version,
)


def test_deterministic_local_compiler_selection_without_downloads():
    assert select_solc_version(["0.8.33", "0.8.20", "0.8.21"]) == "0.8.20"
    assert select_solc_version(["0.7.6", "0.8.25", "0.8.22"]) == "0.8.22"
    with pytest.raises(RuntimeError, match="no downloads attempted"):
        select_solc_version(["0.8.19", "0.7.6"])


def test_contract_sources_are_distinct_full_contracts_with_mutations():
    cases = build_contracts()
    assert len(cases) == len({case.name for case in cases}) == 30
    assert len({case.source for case in cases}) == 30
    assert Counter(case.changed_control for case in cases) == {
        "return_changed": 15,
        "storage_changed": 15,
    }
    for case in cases:
        assert f"contract {case.name} " in case.source
        assert "pragma solidity" in case.source
        assert case.source != case.changed_source
        assert "import " not in case.source


@pytest.fixture(scope="module")
def local_report():
    solcx = pytest.importorskip("solcx")
    try:
        select_solc_version(solcx.get_installed_solc_versions())
    except RuntimeError as exc:
        pytest.skip(str(exc))
    return run_benchmark()


def test_real_solc_controls_are_grounded_and_labels_match(local_report):
    assert local_report["failures"] == []
    assert local_report["distinct_complete_contracts"] == 30
    assert local_report["independently_compiled_reference_runtimes"] == 30
    assert local_report["distinct_executable_prefixes"] == 30
    assert local_report["control_counts"] == {
        "comment_only": 1,
        "fragment": 30,
        "identical_source": 30,
        "return_changed": 15,
        "storage_changed": 15,
        "unsupported_import": 30,
    }
    assert local_report["coverage"] == {
        "total": 121,
        "checked": 61,
        "equal": 30,
        "unequal": 31,
        "unchecked": 60,
        "skip_reasons": {"candidate_fragment": 30, "candidate_imports_unsupported": 30},
    }
    for row in local_report["rows"]:
        assert row["score_kind"] == "source_fact_overlap_proxy"
        assert row["comparison_kind"] == (
            "exact_full_contract_runtime_bytecode" if row["checked"] else None
        )
    assert check_outcomes(local_report["rows"]) == []


def test_label_mismatches_fail_with_contract_and_skip_reason(local_report):
    identical = next(row for row in local_report["rows"] if row["control"] == "identical_source")
    fragment = next(row for row in local_report["rows"] if row["control"] == "fragment")
    comment_only = next(row for row in local_report["rows"] if row["control"] == "comment_only")
    failures = check_outcomes(
        [
            {**identical, "equal": False},
            {**fragment, "checked": True, "equal": False, "skip_reason": None},
            {**comment_only, "equal": True},
        ]
    )
    assert [failure["control"] for failure in failures] == [
        "identical_source",
        "fragment",
        "comment_only",
    ]
    assert failures[1]["expected"]["skip_reason"] == "candidate_fragment"
    assert all(failure["contract"] for failure in failures)
