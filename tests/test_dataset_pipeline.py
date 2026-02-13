"""
Unit tests for src/dataset_pipeline.py

Covers: SolidityParser, EtherscanAPI, DatasetBuilder, ContractData, FunctionPair.
"""

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.dataset_pipeline import (
    ContractData,
    DatasetBuilder,
    EtherscanAPI,
    FunctionPair,
    SolidityParser,
)


# ====================================================================== #
#  Fixtures
# ====================================================================== #

@pytest.fixture
def parser():
    """Return a fresh SolidityParser instance."""
    return SolidityParser()


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Return a temporary directory path string for DatasetBuilder."""
    return str(tmp_path / "test_output")


@pytest.fixture
def builder(tmp_output_dir):
    """Return a DatasetBuilder with a dummy API key and temp output dir."""
    return DatasetBuilder("dummy_api_key", output_dir=tmp_output_dir)


# ====================================================================== #
#  Dataclass construction tests
# ====================================================================== #

class TestContractData:
    def test_required_fields(self):
        cd = ContractData(
            address="0xabc",
            source_code="pragma solidity ^0.8.0;",
            bytecode="0x6080",
            compiler_version="v0.8.20",
            optimization_enabled=True,
            optimization_runs=200,
        )
        assert cd.address == "0xabc"
        assert cd.creation_block is None
        assert cd.abi is None

    def test_optional_fields(self):
        cd = ContractData(
            address="0x1",
            source_code="",
            bytecode="0x",
            compiler_version="",
            optimization_enabled=False,
            optimization_runs=0,
            creation_block=100,
            creation_timestamp=9999,
            abi='[{"type":"function"}]',
        )
        assert cd.creation_block == 100
        assert cd.creation_timestamp == 9999


class TestFunctionPair:
    def test_default_metadata_is_none(self):
        fp = FunctionPair(
            function_name="foo",
            tac_representation="tac",
            solidity_code="code",
            function_signature="function foo()",
            visibility="public",
            is_payable=False,
            is_view=False,
            contract_address="0xabc",
        )
        assert fp.metadata is None

    def test_metadata_independent_instances(self):
        """Each instance should have its own metadata."""
        fp1 = FunctionPair(
            function_name="a", tac_representation="", solidity_code="",
            function_signature="", visibility="", is_payable=False,
            is_view=False, contract_address="",
            metadata={"key": "val1"},
        )
        fp2 = FunctionPair(
            function_name="b", tac_representation="", solidity_code="",
            function_signature="", visibility="", is_payable=False,
            is_view=False, contract_address="",
            metadata={"key": "val2"},
        )
        assert fp1.metadata != fp2.metadata


# ====================================================================== #
#  EtherscanAPI tests
# ====================================================================== #

class TestEtherscanAPI:
    def test_constructor(self):
        api = EtherscanAPI("my_key")
        assert api.api_key == "my_key"
        assert "etherscan" in api.base_url

    def test_custom_base_url(self):
        api = EtherscanAPI("key", base_url="https://custom.api/v1")
        assert api.base_url == "https://custom.api/v1"

    def test_get_verified_contracts_batch_returns_empty(self):
        api = EtherscanAPI("key")
        result = api.get_verified_contracts_batch(0, 100)
        assert result == []


# ====================================================================== #
#  SolidityParser tests
# ====================================================================== #

class TestCleanSourceCode:
    def test_plain_solidity_passthrough(self, parser):
        src = "pragma solidity ^0.8.0;\ncontract A {}"
        assert parser._clean_source_code(src) == src

    def test_json_single_file_content_key(self, parser):
        payload = json.dumps({"content": "pragma solidity ^0.8.0;"})
        result = parser._clean_source_code(payload)
        assert "pragma solidity" in result

    def test_json_multi_file(self, parser):
        payload = json.dumps({
            "sources": {
                "A.sol": {"content": "contract A {}"},
                "B.sol": {"content": "contract B {}"},
            }
        })
        result = parser._clean_source_code(payload)
        assert "contract A" in result
        assert "contract B" in result
        assert "// File: A.sol" in result

    def test_double_brace_format(self, parser):
        inner = json.dumps({
            "sources": {
                "Token.sol": {"content": "contract Token {}"},
            }
        })
        double_brace = "{" + inner + "}"
        result = parser._clean_source_code(double_brace)
        assert "contract Token" in result

    def test_invalid_json_passthrough(self, parser):
        src = "{not valid json at all"
        result = parser._clean_source_code(src)
        # Should return the original string unchanged.
        assert result == src


class TestExtractContracts:
    def test_single_contract(self, parser):
        src = "pragma solidity ^0.8.0;\ncontract Foo { uint x; }"
        contracts = parser._extract_contracts(src)
        assert len(contracts) == 1
        assert contracts[0]["name"] == "Foo"
        assert contracts[0]["type"] == "contract"

    def test_interface_and_library(self, parser):
        src = """
        interface IFoo { function bar() external; }
        library SafeMath { function add(uint a, uint b) internal pure returns (uint) { return a + b; } }
        """
        contracts = parser._extract_contracts(src)
        names = {c["name"] for c in contracts}
        types = {c["type"] for c in contracts}
        assert "IFoo" in names
        assert "SafeMath" in names
        assert "interface" in types
        assert "library" in types

    def test_no_contracts(self, parser):
        src = "pragma solidity ^0.8.0;"
        assert parser._extract_contracts(src) == []

    def test_nested_braces(self, parser):
        src = """
        contract A {
            struct S { uint x; }
            function foo() public { if (true) { uint y = 1; } }
        }
        """
        contracts = parser._extract_contracts(src)
        assert len(contracts) == 1
        body = contracts[0]["body"]
        assert "struct S" in body
        assert "function foo" in body


class TestFindMatchingBrace:
    def test_simple_pair(self, parser):
        text = "{ hello }"
        assert parser._find_matching_brace(text, 0) == 8

    def test_nested(self, parser):
        text = "{ { inner } }"
        assert parser._find_matching_brace(text, 0) == 12

    def test_not_opening_brace(self, parser):
        assert parser._find_matching_brace("hello", 0) == -1

    def test_out_of_range(self, parser):
        assert parser._find_matching_brace("", 0) == -1

    def test_unmatched(self, parser):
        assert parser._find_matching_brace("{ unclosed", 0) == -1

    def test_brace_in_string_ignored(self, parser):
        text = '{ "}" }'
        assert parser._find_matching_brace(text, 0) == 6

    def test_brace_in_line_comment_ignored(self, parser):
        text = "{\n// }\n}"
        assert parser._find_matching_brace(text, 0) == 7

    def test_brace_in_block_comment_ignored(self, parser):
        text = "{ /* } */ }"
        assert parser._find_matching_brace(text, 0) == 10


class TestExtractFunctionsFromContract:
    def test_simple_function(self, parser):
        body = """
        function foo() public { return; }
        """
        funcs = parser._extract_functions_from_contract(body)
        assert len(funcs) == 1
        assert funcs[0]["name"] == "foo"

    def test_multiple_functions(self, parser):
        body = """
        function a() public { }
        function b(uint x) external returns (uint) { return x; }
        """
        funcs = parser._extract_functions_from_contract(body)
        assert len(funcs) == 2
        names = {f["name"] for f in funcs}
        assert names == {"a", "b"}

    def test_abstract_function_with_semicolon(self, parser):
        """Abstract functions ending with ; should be extracted separately."""
        body = """
        function abstractFunc() external;
        function concreteFunc() public { uint x = 1; }
        """
        funcs = parser._extract_functions_from_contract(body)
        assert len(funcs) == 2
        abstract = next(f for f in funcs if f["name"] == "abstractFunc")
        concrete = next(f for f in funcs if f["name"] == "concreteFunc")
        # Abstract function body text should end with ;, not contain concrete's body.
        assert "{" not in abstract["body"]
        assert "uint x = 1" not in abstract["body"]
        # Concrete function should have its own body.
        assert "uint x = 1" in concrete["body"]

    def test_abstract_function_no_brace(self, parser):
        body = "function onlyAbstract() external returns (uint);"
        funcs = parser._extract_functions_from_contract(body)
        assert len(funcs) == 1
        assert funcs[0]["name"] == "onlyAbstract"

    def test_no_functions(self, parser):
        body = "uint public value;"
        assert parser._extract_functions_from_contract(body) == []


class TestExtractVisibility:
    def test_public(self, parser):
        assert parser._extract_visibility("function foo() public { }") == "public"

    def test_private(self, parser):
        assert parser._extract_visibility("function foo() private { }") == "private"

    def test_internal(self, parser):
        assert parser._extract_visibility("function foo() internal { }") == "internal"

    def test_external(self, parser):
        assert parser._extract_visibility("function foo() external;") == "external"

    def test_default_public(self, parser):
        assert parser._extract_visibility("function foo() { }") == "public"

    def test_visibility_in_body_ignored(self, parser):
        """Keywords in the body should not affect the extracted visibility."""
        code = 'function foo() public { string memory s = "private"; }'
        assert parser._extract_visibility(code) == "public"

    def test_visibility_in_body_ignored_no_sig_visibility(self, parser):
        """If signature has no visibility keyword but body mentions 'internal'."""
        code = 'function foo() { require(msg.sender == internal_var); }'
        # No visibility keyword before {, so should default to public.
        assert parser._extract_visibility(code) == "public"


class TestExtractFunctions:
    def test_full_extraction(self, parser):
        src = """
        pragma solidity ^0.8.0;
        contract MyContract {
            function transfer(address to, uint256 amount) public {
                // do transfer
            }
            function balanceOf(address owner) public view returns (uint256) {
                return 0;
            }
        }
        """
        funcs = parser.extract_functions(src)
        assert len(funcs) == 2
        names = {f["name"] for f in funcs}
        assert names == {"transfer", "balanceOf"}
        for f in funcs:
            assert f["contract_name"] == "MyContract"

    def test_filter_by_contract_name(self, parser):
        src = """
        contract A { function fa() public {} }
        contract B { function fb() public {} }
        """
        funcs = parser.extract_functions(src, contract_name="B")
        assert len(funcs) == 1
        assert funcs[0]["name"] == "fb"

    def test_empty_source(self, parser):
        assert parser.extract_functions("") == []

    def test_payable_detection(self, parser):
        src = "contract C { function pay() public payable { } }"
        funcs = parser.extract_functions(src)
        assert len(funcs) == 1
        assert funcs[0]["is_payable"] is True

    def test_view_detection(self, parser):
        src = "contract C { function peek() public view returns (uint) { return 0; } }"
        funcs = parser.extract_functions(src)
        assert len(funcs) == 1
        assert funcs[0]["is_view"] is True

    def test_pure_counts_as_view(self, parser):
        src = "contract C { function add(uint a, uint b) public pure returns (uint) { return a+b; } }"
        funcs = parser.extract_functions(src)
        assert funcs[0]["is_view"] is True

    def test_interface_abstract_functions(self, parser):
        src = """
        interface IERC20 {
            function totalSupply() external view returns (uint256);
            function balanceOf(address account) external view returns (uint256);
        }
        """
        funcs = parser.extract_functions(src)
        assert len(funcs) == 2
        for f in funcs:
            assert f["visibility"] == "external"
            assert f["contract_name"] == "IERC20"


# ====================================================================== #
#  DatasetBuilder tests
# ====================================================================== #

class TestDatasetBuilderInit:
    def test_database_created(self, builder, tmp_output_dir):
        db_path = Path(tmp_output_dir) / "contracts.db"
        assert db_path.exists()

    def test_tables_exist(self, builder, tmp_output_dir):
        db_path = Path(tmp_output_dir) / "contracts.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "contracts" in tables
        assert "function_pairs" in tables


class TestStoreContract:
    def test_insert_and_retrieve(self, builder, tmp_output_dir):
        cd = ContractData(
            address="0xABCDEF",
            source_code="contract X {}",
            bytecode="0x6080",
            compiler_version="v0.8.20",
            optimization_enabled=True,
            optimization_runs=200,
        )
        builder._store_contract(cd)

        conn = sqlite3.connect(Path(tmp_output_dir) / "contracts.db")
        cursor = conn.cursor()
        cursor.execute("SELECT address, source_code FROM contracts")
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == "0xABCDEF"
        assert rows[0][1] == "contract X {}"

    def test_replace_on_duplicate(self, builder, tmp_output_dir):
        cd1 = ContractData(
            address="0x1", source_code="v1", bytecode="0x",
            compiler_version="", optimization_enabled=False, optimization_runs=0,
        )
        cd2 = ContractData(
            address="0x1", source_code="v2", bytecode="0x",
            compiler_version="", optimization_enabled=False, optimization_runs=0,
        )
        builder._store_contract(cd1)
        builder._store_contract(cd2)

        conn = sqlite3.connect(Path(tmp_output_dir) / "contracts.db")
        cursor = conn.cursor()
        cursor.execute("SELECT source_code FROM contracts WHERE address='0x1'")
        assert cursor.fetchone()[0] == "v2"
        conn.close()


class TestStoreFunctionPair:
    def test_insert(self, builder, tmp_output_dir):
        fp = FunctionPair(
            function_name="foo",
            tac_representation="tac_data",
            solidity_code="function foo() {}",
            function_signature="function foo()",
            visibility="public",
            is_payable=False,
            is_view=False,
            contract_address="0x1",
            metadata={"key": "val"},
        )
        builder._store_function_pair(fp)

        conn = sqlite3.connect(Path(tmp_output_dir) / "contracts.db")
        cursor = conn.cursor()
        cursor.execute("SELECT function_name, metadata FROM function_pairs")
        row = cursor.fetchone()
        conn.close()

        assert row[0] == "foo"
        meta = json.loads(row[1])
        assert meta["key"] == "val"

    def test_duplicate_skipped(self, builder, tmp_output_dir):
        fp = FunctionPair(
            function_name="bar",
            tac_representation="same_tac",
            solidity_code="same_code",
            function_signature="function bar()",
            visibility="public",
            is_payable=False,
            is_view=False,
            contract_address="0x2",
        )
        builder._store_function_pair(fp)
        builder._store_function_pair(fp)  # duplicate

        conn = sqlite3.connect(Path(tmp_output_dir) / "contracts.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM function_pairs")
        assert cursor.fetchone()[0] == 1
        conn.close()


class TestAddSelectorsToSolidityFunctions:
    def test_selector_calculation(self, builder):
        funcs = [
            {"name": "transfer", "signature": "function transfer(address to, uint256 amount)"},
            {"name": "balanceOf", "signature": "function balanceOf(address owner)"},
        ]
        result = builder._add_selectors_to_solidity_functions(funcs)
        assert len(result) == 2

        # Known selectors.
        assert result[0]["selector"] == "0xa9059cbb"  # transfer(address,uint256)
        assert result[1]["selector"] == "0x70a08231"  # balanceOf(address)

    def test_no_params(self, builder):
        funcs = [{"name": "totalSupply", "signature": "function totalSupply()"}]
        result = builder._add_selectors_to_solidity_functions(funcs)
        assert result[0]["selector"] is not None

    def test_invalid_signature(self, builder):
        funcs = [{"name": "bad", "signature": ""}]
        result = builder._add_selectors_to_solidity_functions(funcs)
        # Should not crash; selector may be None or computed from empty.
        assert "selector" in result[0]


class TestBuildTrainingPair:
    def test_valid_pair(self, builder):
        match = {
            "solidity_function": {
                "name": "foo",
                "body": "function foo() public { uint x = 1; }",
                "signature": "function foo()",
                "visibility": "public",
                "is_payable": False,
                "is_view": False,
                "contract_name": "Test",
            },
            "tac": "function foo:\n  temp_1 = 1\n  return",
            "selector": "0x12345678",
        }
        pair = builder._build_training_pair(match, "0xaddr")
        assert pair is not None
        assert pair.function_name == "foo"
        assert pair.contract_address == "0xaddr"
        assert pair.metadata["selector"] == "0x12345678"

    def test_too_short_solidity(self, builder):
        match = {
            "solidity_function": {
                "name": "x", "body": "short",
                "signature": "function x()", "visibility": "public",
                "is_payable": False, "is_view": False,
            },
            "tac": "function x:\n  temp_1 = 1\n  return",
            "selector": "0x1",
        }
        assert builder._build_training_pair(match, "0x") is None

    def test_empty_tac(self, builder):
        match = {
            "solidity_function": {
                "name": "y", "body": "function y() public { lots of code here }",
                "signature": "function y()", "visibility": "public",
                "is_payable": False, "is_view": False,
            },
            "tac": "",
            "selector": "0x2",
        }
        assert builder._build_training_pair(match, "0x") is None


class TestFilterAndCleanDataset:
    def test_filtering_removes_short(self, builder, tmp_output_dir):
        # Insert a short and a long function pair.
        short = FunctionPair(
            function_name="s", tac_representation="t" * 100,
            solidity_code="short",  # < 50 chars
            function_signature="function s()", visibility="public",
            is_payable=False, is_view=False, contract_address="0x1",
        )
        long_enough = FunctionPair(
            function_name="l", tac_representation="t" * 100,
            solidity_code="x" * 100,
            function_signature="function l()", visibility="public",
            is_payable=False, is_view=False, contract_address="0x2",
        )
        builder._store_function_pair(short)
        builder._store_function_pair(long_enough)

        count = builder.filter_and_clean_dataset(min_length=50, max_length=20000)
        assert count == 1

    def test_filtering_removes_long_tac(self, builder):
        fp = FunctionPair(
            function_name="big", tac_representation="t" * 30000,
            solidity_code="x" * 100,
            function_signature="function big()", visibility="public",
            is_payable=False, is_view=False, contract_address="0x3",
        )
        builder._store_function_pair(fp)
        count = builder.filter_and_clean_dataset(min_length=50, max_length=20000)
        assert count == 0


class TestExportDataset:
    def test_jsonl_export(self, builder, tmp_output_dir):
        fp = FunctionPair(
            function_name="exp", tac_representation="tac_content",
            solidity_code="function exp() public {}",
            function_signature="function exp()", visibility="public",
            is_payable=False, is_view=True, contract_address="0xE",
        )
        builder._store_function_pair(fp)

        path = builder.export_dataset("jsonl")
        assert path.endswith(".jsonl")
        assert os.path.exists(path)

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["input"] == "tac_content"
        assert record["output"] == "function exp() public {}"
        assert record["metadata"]["function_name"] == "exp"

    def test_csv_export(self, builder, tmp_output_dir):
        fp = FunctionPair(
            function_name="csv_fn", tac_representation="tac",
            solidity_code="code", function_signature="function csv_fn()",
            visibility="public", is_payable=False, is_view=False,
            contract_address="0xC",
        )
        builder._store_function_pair(fp)
        path = builder.export_dataset("csv")
        assert path.endswith(".csv")
        assert os.path.exists(path)


class TestGetDatasetStatistics:
    def test_empty_stats(self, builder):
        stats = builder.get_dataset_statistics()
        assert stats["total_contracts"] == 0
        assert stats["total_function_pairs"] == 0
        assert stats["visibility_distribution"] == {}

    def test_stats_with_data(self, builder):
        cd = ContractData(
            address="0xS", source_code="", bytecode="0x",
            compiler_version="", optimization_enabled=False, optimization_runs=0,
        )
        builder._store_contract(cd)

        fp = FunctionPair(
            function_name="stat", tac_representation="tac123",
            solidity_code="code456",
            function_signature="function stat()", visibility="external",
            is_payable=False, is_view=False, contract_address="0xS",
        )
        builder._store_function_pair(fp)

        stats = builder.get_dataset_statistics()
        assert stats["total_contracts"] == 1
        assert stats["total_function_pairs"] == 1
        assert stats["visibility_distribution"]["external"] == 1
        assert stats["length_statistics"]["avg_solidity_length"] is not None


class TestProcessContractsToFunctionPairs:
    def test_only_unprocessed_contracts(self, builder, tmp_output_dir):
        """Contracts already marked processed should be skipped."""
        db_path = Path(tmp_output_dir) / "contracts.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Insert a contract already marked as processed.
        cursor.execute(
            """
            INSERT INTO contracts (address, source_code, bytecode,
                compiler_version, optimization_enabled, optimization_runs, processed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("0xPROCESSED", "contract A { function f() public {} }",
             "0x6080", "v0.8.20", True, 200, True),
        )
        conn.commit()
        conn.close()

        # process_contracts_to_function_pairs should return 0 because
        # the only contract is already processed.
        pairs = builder.process_contracts_to_function_pairs()
        assert pairs == 0


class TestMatchFunctionsBySelector:
    def test_matching(self, builder):
        sol_funcs = [
            {"name": "transfer", "selector": "0xa9059cbb"},
            {"name": "approve", "selector": "0x095ea7b3"},
        ]

        # Create mock bytecode functions.
        mock_func = MagicMock()
        mock_func.selector = "0xa9059cbb"
        mock_func.name = "function_0xa9059cbb"
        mock_func.entry_block = "block_0000"
        mock_func.basic_blocks = []

        bc_funcs = {"function_0xa9059cbb": mock_func}

        mock_analyzer = MagicMock()
        mock_analyzer.basic_blocks = {}

        matches = builder._match_functions_by_selector(
            sol_funcs, bc_funcs, mock_analyzer
        )
        assert len(matches) == 1
        assert matches[0]["selector"] == "0xa9059cbb"
        assert matches[0]["solidity_function"]["name"] == "transfer"

    def test_no_matches(self, builder):
        sol_funcs = [{"name": "foo", "selector": "0x11111111"}]
        mock_func = MagicMock()
        mock_func.selector = "0x22222222"
        bc_funcs = {"f": mock_func}

        mock_analyzer = MagicMock()
        matches = builder._match_functions_by_selector(
            sol_funcs, bc_funcs, mock_analyzer
        )
        assert matches == []


class TestCollectFunctionBlocks:
    def test_traversal(self, builder):
        block_a = MagicMock()
        block_a.successors = ["b"]
        block_b = MagicMock()
        block_b.successors = ["c"]
        block_c = MagicMock()
        block_c.successors = []

        all_blocks = {"a": block_a, "b": block_b, "c": block_c}
        result = builder._collect_function_blocks("a", all_blocks)
        assert len(result) == 3

    def test_missing_entry(self, builder):
        assert builder._collect_function_blocks("missing", {}) == []

    def test_cycle_handling(self, builder):
        block_a = MagicMock()
        block_a.successors = ["b"]
        block_b = MagicMock()
        block_b.successors = ["a"]  # cycle

        all_blocks = {"a": block_a, "b": block_b}
        result = builder._collect_function_blocks("a", all_blocks)
        assert len(result) == 2