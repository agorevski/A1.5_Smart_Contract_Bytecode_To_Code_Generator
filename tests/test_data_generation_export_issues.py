import json
import sqlite3
import sys
from pathlib import Path

import pytest


class _CharacterTokenizer:
    name_or_path = "character-test-tokenizer"

    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(len(text)))}


def _make_pair(download_hf_contracts, idx, body):
    tac = f"function same:\n  block_{idx}:\n    temp = {idx}"
    return {
        "contract_address": f"0x{idx:040x}",
        "function_name": "same",
        "tac_representation": tac,
        "solidity_code": body,
        "function_signature": "function same()",
        "visibility": "public",
        "is_payable": False,
        "is_view": False,
        "metadata": json.dumps({"compiler_version": "0.8.20"}),
        "hash": download_hf_contracts._md5(tac + body),
        "body_hash": download_hf_contracts.hash_normalized_body(body),
        "tac_hash": download_hf_contracts.hash_normalized_tac(tac),
        "pair_norm_hash": download_hf_contracts.hash_normalized_pair(tac, body),
    }


def test_export_manifest_records_lineage_artifacts_and_duplicate_validation(tmp_path):
    import download_hf_contracts

    db_path = tmp_path / "contracts.db"
    output_path = tmp_path / "dataset.jsonl"
    manifest_path = tmp_path / "dataset.manifest.json"
    body = "function same() public { uint256 x = 1; uint256 y = x + 1; emit Done(y); }"

    download_hf_contracts.init_database(db_path)
    pairs = [_make_pair(download_hf_contracts, idx, body) for idx in range(3)]
    assert download_hf_contracts._store_pairs_batch(db_path, pairs) == 3

    download_hf_contracts.export_training_data(
        str(output_path),
        max_body_dupes=2,
        db_path=db_path,
        manifest_path=manifest_path,
        command_args=["--export-only", "--max-body-dupes", "2"],
    )

    assert len(output_path.read_text().splitlines()) == 2
    manifest = json.loads(manifest_path.read_text())
    assert manifest["manifest_kind"] == "hf_export"
    assert manifest["status"] == "completed"
    assert manifest["lineage"]["source"]["repo"] == "andstor/smart_contracts"
    assert manifest["command"]["args"] == ["--export-only", "--max-body-dupes", "2"]
    assert manifest["artifacts"]["jsonl"]["row_count"] == 2
    assert manifest["artifacts"]["jsonl"]["sha256"]
    assert manifest["drop_counts"]["body_cap_rows"] == 1
    assert manifest["validation"]["body_duplicate_cap"]["status"] == "passed"
    assert manifest["training_row_schema_version"] == 1
    first_row = json.loads(output_path.read_text().splitlines()[0])
    assert first_row["metadata"]["schema_version"] == 1


def test_jsonl_duplicate_cap_validation_reports_top_samples(tmp_path):
    from download_hf_contracts import (
        enforce_jsonl_body_duplicate_cap,
        validate_jsonl_body_duplicate_cap,
    )

    path = tmp_path / "dupes.jsonl"
    rows = [
        {
            "input": f"tac {idx}",
            "output": "function same() public { return VALUE; // comment\n}",
            "metadata": {"function_signature": f"same{idx}()"},
        }
        for idx in range(3)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    validation = validate_jsonl_body_duplicate_cap(path, max_body_dupes=2, sample_limit=2)
    assert validation["status"] == "failed"
    assert validation["body_hashes_over_cap"] == 1
    assert validation["rows_over_cap"] == 1
    assert validation["violations"][0]["samples"][0]["line_number"] == 1

    with pytest.raises(ValueError, match="body_hash=.*count=3"):
        enforce_jsonl_body_duplicate_cap(path, max_body_dupes=2, sample_limit=2)


def test_compile_manifest_summarizes_persisted_failure_diagnostics(tmp_path):
    import download_hf_contracts

    db_path = tmp_path / "compile.db"
    manifest_path = tmp_path / "compile.manifest.json"
    run_id = "compile-test-run"
    download_hf_contracts.init_database(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO contracts (address, source_code, bytecode) VALUES (?, ?, ?)",
            [
                ("0xabc", "contract A {}", "0x"),
                ("0xdef", "contract B {}", "0x"),
            ],
        )
    download_hf_contracts._mark_contract_status(
        db_path,
        ["0xabc"],
        "compile_failed",
        processed=False,
        last_error="solc install failed",
    )
    download_hf_contracts._store_compile_diagnostics(
        db_path,
        [
            {
                "run_id": run_id,
                "contract_address": "0xabc",
                "compiler_version": "0.8.20",
                "optimizer_enabled": True,
                "optimization_runs": 200,
                "status": "compile_failed",
                "error": "solc install failed",
            }
        ],
    )

    payload = download_hf_contracts._write_compile_manifest(
        db_path,
        manifest_path,
        run_id=run_id,
        status="completed_with_errors",
        started_at="2026-01-01T00:00:00Z",
        duration_seconds=0.25,
        parameters={"workers": 1},
        summary={"compile_jobs": 1, "pairs_generated": 0},
        status_counts={"compile_jobs": {"compile_failed": 1}},
        drop_counts={"compile_or_analysis_errors": 1},
        command_args=["--compile-only"],
    )

    assert payload["failure_diagnostics"]["total_diagnostics"] == 1
    assert payload["failure_diagnostics"]["top_errors"][0]["status"] == "compile_failed"
    assert payload["failure_diagnostics"]["top_errors"][0]["sample_contract_addresses"] == ["0xabc"]
    assert payload["status_counts"]["contracts"]["compile_failed"] == 1
    assert json.loads(manifest_path.read_text())["run_id"] == run_id


def test_download_contracts_streams_parquet_batches_without_full_dataframe(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    import inspect
    import download_hf_contracts

    parquet_path = tmp_path / "contracts.parquet"
    source_a = "pragma solidity ^0.8.0; contract A { function a() public {} }"
    source_b = "pragma solidity ^0.8.0; contract B { function b() public {} }"
    table = pa.table(
        {
            "language": ["Solidity", "Vyper", "Solidity"],
            "source_code": [source_a, "contract ignored", source_b],
            "contract_address": [
                "0x0000000000000000000000000000000000000001",
                "0x0000000000000000000000000000000000000002",
                "0x0000000000000000000000000000000000000003",
            ],
            "compiler_version": ["v0.8.20", "", "v0.8.19"],
            "optimization_used": [True, False, False],
            "runs": [200, None, 300],
            "abi": ["[]", "", "[]"],
            "contract_name": ["A", "", "B"],
        }
    )
    pq.write_table(table, parquet_path)

    monkeypatch.setattr(
        download_hf_contracts, "_get_parquet_files", lambda *a, **k: ["data/train/0.parquet"]
    )
    monkeypatch.setattr(download_hf_contracts, "_resolve_hf_revision", lambda _revision: "resolved")
    monkeypatch.setattr(
        download_hf_contracts, "hf_hub_download", lambda **_kwargs: str(parquet_path)
    )

    db_path = tmp_path / "contracts.db"
    manifest_path = tmp_path / "download.manifest.json"
    download_hf_contracts.init_database(db_path)

    inserted = download_hf_contracts.download_contracts(
        db_path=db_path,
        manifest_path=manifest_path,
        parquet_batch_size=2,
    )

    assert inserted == 2
    assert "read_parquet" not in inspect.getsource(download_hf_contracts.download_contracts)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["parameters"]["parquet_batch_size"] == 2
    assert manifest["performance"]["max_rss_mb"] > 0
    assert manifest["performance"]["parquet_streams"][0]["batch_size"] == 2
    assert manifest["drop_counts"]["non_solidity"] == 1


def test_download_preserves_distinct_targets_and_verified_compiler_configs(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    import download_hf_contracts

    source = (
        "pragma solidity ^0.8.0; "
        "contract A { function a() public { emit Done(1); } } "
        "contract B { function b() public { emit Done(2); } }"
    )
    rows = [
        ("A", "v0.8.20", True, 200),
        ("B", "v0.8.20", True, 200),
        ("A", "v0.8.19", True, 200),
        ("A", "v0.8.20", False, 200),
        ("A", "v0.8.20+commit.a1b79de6", True, 200),
        ("B", "v0.8.19", True, 200),
        ("B", "v0.8.19", False, 200),
    ]
    table = pa.table(
        {
            "language": ["Solidity"] * len(rows),
            "source_code": [source] * len(rows),
            "contract_address": [
                f"0x{idx:040x}" if idx <= 5 else ""
                for idx in range(1, len(rows) + 1)
            ],
            "contract_name": [row[0] for row in rows],
            "compiler_version": [row[1] for row in rows],
            "optimization_used": [row[2] for row in rows],
            "runs": [row[3] for row in rows],
        }
    )
    parquet_path = tmp_path / "same-source.parquet"
    pq.write_table(table, parquet_path)
    monkeypatch.setattr(
        download_hf_contracts, "_get_parquet_files", lambda *a, **k: ["data/train/0.parquet"]
    )
    monkeypatch.setattr(download_hf_contracts, "_resolve_hf_revision", lambda _: "resolved")
    monkeypatch.setattr(
        download_hf_contracts, "hf_hub_download", lambda **_kwargs: str(parquet_path)
    )

    db_path = tmp_path / "contracts.db"
    manifest_path = tmp_path / "download.manifest.json"
    download_hf_contracts.init_database(db_path)
    assert download_hf_contracts.download_contracts(
        db_path=db_path, manifest_path=manifest_path
    ) == 6
    with sqlite3.connect(db_path) as conn:
        stored = conn.execute(
            "SELECT address, contract_name, compiler_version, optimization_enabled "
            "FROM contracts ORDER BY address"
        ).fetchall()
    named = [row[1:] for row in stored if row[0].startswith("0x")]
    synthetic = [row for row in stored if row[0].startswith("hf_")]
    assert named == [
        ("A", "v0.8.20", 1),
        ("B", "v0.8.20", 1),
        ("A", "v0.8.19", 1),
        ("A", "v0.8.20", 0),
    ]
    assert len({row[0] for row in synthetic}) == 2
    assert {row[1:] for row in synthetic} == {
        ("B", "v0.8.19", 1), ("B", "v0.8.19", 0)
    }
    assert download_hf_contracts.download_contracts(
        db_path=db_path, manifest_path=manifest_path
    ) == 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["drop_counts"]["source_deduped"] == 7
    assert manifest["parameters"]["dedup_key"] == [
        "source_hash", "contract_name", "compiler_version",
        "optimization_enabled", "optimization_runs",
    ]


def test_prepare_contract_uses_single_source_aligned_compiler_config():
    import download_hf_contracts

    source = """
    pragma solidity ^0.8.0;
    contract Token {
        function transfer(address to, uint256 amount) public {
            require(to != address(0));
            emit Transfer(to, amount);
        }
        event Transfer(address indexed to, uint256 amount);
    }
    """

    prepared = download_hf_contracts._prepare_contract(
        "0x0000000000000000000000000000000000000001",
        source,
        "v0.8.20+commit.a1b79de6",
        False,
        777,
        "Token",
    )

    assert prepared is not None
    assert prepared["compile_configs"] == [
        {
            "version": "0.8.20",
            "optimizer_enabled": False,
            "optimizer_runs": 777,
        }
    ]
    assert "compatible_versions" not in prepared


def test_hf_compile_quarantines_ambiguous_selector_alignment(monkeypatch):
    from types import SimpleNamespace
    import download_hf_contracts

    selector = "0x23b872dd"
    source_functions = [
        {
            "name": "transferFrom", "selector": selector, "contract_name": "ERC20",
            "body": "function transferFrom(address from, address to, uint256 amount) public returns (bool) { return true; }",
        },
        {
            "name": "transferFrom", "selector": selector, "contract_name": "ERC721",
            "body": "function transferFrom(address from, address to, uint256 tokenId) public { ownerOf[tokenId] = to; }",
        },
    ]
    bytecode_function = SimpleNamespace(
        selector=selector, entry_block="block_0", basic_blocks=[]
    )

    class FakeAnalyzer:
        basic_blocks = {}

        def __init__(self, _bytecode):
            pass

        def analyze_control_flow(self):
            pass

        def identify_functions(self):
            return {"transferFrom": bytecode_function}

    monkeypatch.setattr(download_hf_contracts, "install_solc_version", lambda _: True)
    monkeypatch.setattr(download_hf_contracts, "BytecodeAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(
        download_hf_contracts, "compile_source",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=True,
            contracts={
                "Derived": SimpleNamespace(
                    runtime_bytecode="6000600055",
                    abi=[{"name": "transferFrom", "type": "function"}],
                )
            },
        ),
    )
    outcome = download_hf_contracts._compile_one_job(
        "0x" + "1" * 40,
        {
            "contract.sol": (
                "pragma solidity ^0.8.0; contract ERC20 { "
                "function transferFrom(address from, address to, uint256 value) "
                "public returns (bool) { return true; } } "
                "contract Derived is ERC20 {}"
            )
        },
        source_functions,
        "0.8.20",
        True,
        200,
        min_body_length=0,
        target_contract_name="Derived",
    )
    assert outcome["pairs"] == []
    assert outcome["status"] == "ambiguous_selector_alignment"
    assert outcome["drop_counts"]["ambiguous_selector_alignment"] == 1
    assert selector in outcome["error"]


def test_etherscan_compile_records_ambiguous_selector_diagnostic(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace
    from src import dataset_pipeline

    source = """
    pragma solidity ^0.8.0;
    contract ERC20 {
        function transferFrom(address from, address to, uint256 value)
            public returns (bool) { return true; }
    }
    contract ERC721 {
        function transferFrom(address from, address to, uint256 tokenId)
            public { ownerOf[tokenId] = to; }
    }
    contract Derived is ERC20 {}
    """

    class FakeAnalyzer:
        basic_blocks = {}

        def __init__(self, _bytecode):
            pass

        def analyze_control_flow(self):
            pass

        def identify_functions(self):
            return {
                "transferFrom": SimpleNamespace(
                    selector="0x23b872dd", entry_block="block_0", basic_blocks=[]
                )
            }

    builder = dataset_pipeline.DatasetBuilder("dummy", output_dir=str(tmp_path / "builder"))
    builder.etherscan.get_contract_source = lambda address: dataset_pipeline.ContractData(
        address=address, source_code=source, bytecode="0x6000600055",
        compiler_version="v0.8.20", optimization_enabled=True, optimization_runs=200,
    )
    monkeypatch.setattr(dataset_pipeline, "install_solc_version", lambda _: True)
    monkeypatch.setattr(dataset_pipeline, "BytecodeAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(
        dataset_pipeline, "compile_source",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=True,
            contracts={
                "Derived": SimpleNamespace(runtime_bytecode="6000600055", abi=[])
            },
        ),
    )
    outcome = builder._collect_compile_address(
        "0x" + "1" * 40, max_compiler_configs=1, compiler_install_lock=threading.Lock()
    )
    assert outcome.pairs == []
    assert outcome.status_update["status"] == "ambiguous_selector_alignment"
    assert any(
        diag["status"] == "ambiguous_selector_alignment"
        and "0x23b872dd" in diag["error"]
        for diag in outcome.diagnostics
    )


def test_export_training_data_quarantines_overlength_rows(tmp_path):
    import download_hf_contracts

    db_path = tmp_path / "contracts.db"
    output_path = tmp_path / "dataset.jsonl"
    manifest_path = tmp_path / "dataset.manifest.json"
    download_hf_contracts.init_database(db_path)

    good_body = "function good() public { uint256 x = 1; uint256 y = x + 1; emit Done(y); }"
    target_long = "function huge() public { " + " ".join(["uint256 x = 1;"] * 200) + " }"
    context_long = "function huge:\n  " + " ".join(["temp = ADD temp 1"] * 200)
    pairs = [
        _make_pair(download_hf_contracts, 1, good_body),
        _make_pair(download_hf_contracts, 2, target_long),
        {
            **_make_pair(download_hf_contracts, 3, good_body.replace("good", "contextHeavy")),
            "tac_representation": context_long,
        },
    ]
    assert download_hf_contracts._store_pairs_batch(db_path, pairs) == 3

    download_hf_contracts.export_training_data(
        str(output_path),
        max_body_dupes=5,
        db_path=db_path,
        manifest_path=manifest_path,
        max_seq_length=128,
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    rejects = [
        json.loads(line) for line in Path(f"{output_path}.rejects.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert len(rejects) == 2
    assert {reason for row in rejects for reason in row["reasons"]} >= {
        "target_overlength",
        "context_overlength",
    }
    manifest = json.loads(manifest_path.read_text())
    assert manifest["drop_counts"]["overlength_rows"] == 2
    assert manifest["artifacts"]["rejects_jsonl"]["row_count"] == 2


def test_export_length_report_uses_training_prompt_and_exact_tokenizer():
    import train
    from src.dataset_export_primitives import export_length_report, export_prompt_parts

    record = {
        "input": "function selector_12345678:\n  block_0:\n    temp_1 = ADD 1 2",
        "output": "function add() public pure returns (uint256) { return 1 + 2; }",
        "metadata": {"selector": "0x12345678"},
    }
    expected_parts = train._preflight_prompt_parts(
        record,
        include_bytecode_metadata=True,
        include_selector_signature_metadata=True,
        template_format="alpaca",
    )
    assert export_prompt_parts(record) == expected_parts
    prefix, target, suffix = expected_parts
    assert prefix.startswith("### Instruction:")
    assert "Bytecode metadata:" in prefix
    tokenizer = _CharacterTokenizer()
    exact = export_length_report(record, len(prefix) + len(target + suffix) - 1, tokenizer)
    assert exact["context_tokens"] == len(prefix)
    assert exact["target_tokens"] == len(target + suffix)
    assert exact["total_tokens"] == len(prefix) + len(target + suffix)
    assert exact["token_count_method"] == "tokenizer"
    assert exact["reasons"] == ["context_overlength"]
    estimated = export_length_report(record, exact["max_seq_length"])
    assert estimated["token_count_method"] == "estimate"
    assert estimated["total_tokens"] < exact["total_tokens"]


def test_export_exact_length_includes_training_eos():
    from src.dataset_export_primitives import export_length_report

    class EosCharacterTokenizer(_CharacterTokenizer):
        eos_token_id = 999999

    record = {"input": "function example:\n  return", "output": "function example() {}"}
    without_eos = export_length_report(record, 8192, _CharacterTokenizer())
    with_eos = export_length_report(
        record, without_eos["total_tokens"], EosCharacterTokenizer()
    )

    assert with_eos["target_tokens"] == without_eos["target_tokens"] + 1
    assert with_eos["reasons"] == ["context_overlength"]


def test_both_export_paths_quarantine_exact_tokenizer_overlength(tmp_path):
    import download_hf_contracts
    from src.dataset_export_primitives import build_training_record, export_length_report
    from src.dataset_pipeline import DatasetBuilder, FunctionPair

    tokenizer = _CharacterTokenizer()
    short_body = "function short() public { uint256 x = 1; emit Done(x); }"
    long_body = "function longer() public { uint256 x = 2; emit Done(x); }"
    short_tac = "function same:\n  block_1:\n    temp = 1"
    long_tac = "function same:\n  block_2:\n    temp = " + "ADD 1 2 " * 45
    budget = export_length_report(
        build_training_record(short_tac, short_body), 8192, tokenizer
    )["total_tokens"] + 1
    assert not export_length_report(
        build_training_record(long_tac, long_body), budget
    )["reasons"]
    assert "context_overlength" in export_length_report(
        build_training_record(long_tac, long_body), budget, tokenizer
    )["reasons"]

    db_path = tmp_path / "hf.db"
    output_path = tmp_path / "hf.jsonl"
    manifest_path = tmp_path / "hf.manifest.json"
    download_hf_contracts.init_database(db_path)
    pairs = [
        {**_make_pair(download_hf_contracts, idx, body), "tac_representation": tac}
        for idx, (body, tac) in enumerate(
            [(short_body, short_tac), (long_body, long_tac)], start=1
        )
    ]
    assert download_hf_contracts._store_pairs_batch(db_path, pairs) == 2
    download_hf_contracts.export_training_data(
        str(output_path),
        db_path=db_path,
        manifest_path=manifest_path,
        max_seq_length=budget,
        length_tokenizer=tokenizer,
    )
    assert len(output_path.read_text().splitlines()) == 1
    hf_rejects = [
        json.loads(line) for line in Path(f"{output_path}.rejects.jsonl").read_text().splitlines()
    ]
    assert len(hf_rejects) == 1
    assert hf_rejects[0]["reasons"] == ["context_overlength"]
    assert hf_rejects[0]["lengths"]["token_count_method"] == "tokenizer"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["parameters"]["length_tokenizer"] == tokenizer.name_or_path
    assert manifest["validation"]["token_length_filter"]["token_count_method"] == "tokenizer"

    builder = DatasetBuilder("dummy", output_dir=str(tmp_path / "builder"))
    for idx, (body, tac) in enumerate(
        [(short_body, short_tac), (long_body, long_tac)], start=1
    ):
        builder._store_function_pair(
            FunctionPair(
                function_name="short" if idx == 1 else "longer",
                tac_representation=tac,
                solidity_code=body,
                function_signature="function short()" if idx == 1 else "function longer()",
                visibility="public",
                is_payable=False,
                is_view=False,
                contract_address=f"0x{idx:040x}",
            )
        )
    builder_output = Path(
        builder.export_dataset("jsonl", max_seq_length=budget, length_tokenizer=tokenizer)
    )
    assert len(builder_output.read_text().splitlines()) == 1
    builder_rejects = [
        json.loads(line)
        for line in Path(f"{builder_output}.rejects.jsonl").read_text().splitlines()
    ]
    assert len(builder_rejects) == 1
    assert builder_rejects[0]["reasons"] == ["context_overlength"]
    builder_manifest = json.loads(Path(f"{builder_output}.manifest.json").read_text())
    assert builder_manifest["parameters"]["length_tokenizer"] == tokenizer.name_or_path


def test_export_cli_loads_exact_tokenizer_offline(tmp_path, monkeypatch):
    import download_hf_contracts
    from transformers import AutoTokenizer

    db_path = tmp_path / "contracts.db"
    output_path = tmp_path / "dataset.jsonl"
    download_hf_contracts.init_database(db_path)
    body = "function short() public { uint256 x = 1; emit Done(x); }"
    download_hf_contracts._store_pairs_batch(
        db_path, [_make_pair(download_hf_contracts, 1, body)]
    )
    loaded = []

    def load_tokenizer(name, **kwargs):
        loaded.append((name, kwargs))
        return _CharacterTokenizer()

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(
        sys, "argv",
        [
            "download_hf_contracts.py", "--export-only", "--db", str(db_path),
            "--output", str(output_path), "--length-tokenizer", "locally-cached-tokenizer",
            "--max-seq-length", "128",
        ],
    )
    download_hf_contracts.main()
    assert loaded == [("locally-cached-tokenizer", {"local_files_only": True})]
    assert output_path.read_text() == ""
    rejects = [
        json.loads(line) for line in Path(f"{output_path}.rejects.jsonl").read_text().splitlines()
    ]
    assert len(rejects) == 1
    assert "context_overlength" in rejects[0]["reasons"]


def test_shared_primitives_produce_same_final_rows_across_export_paths(tmp_path):
    import download_hf_contracts
    from src import dataset_pipeline
    from src.dataset_export_primitives import final_row_hash
    from src.dataset_pipeline import DatasetBuilder, FunctionPair

    assert download_hf_contracts.hash_normalized_pair is dataset_pipeline.hash_normalized_pair
    assert download_hf_contracts.sanitize_tac_prompt_input is dataset_pipeline.sanitize_tac_prompt_input

    tac = (
        "function transfer(address to):\n"
        "  // Selector: 0xa9059cbb\n"
        "  // Compiler: solc 0.8.20\n"
        "  block:\n"
        "    temp = 1\n"
    )
    body = "function transfer(address to) public { uint256 x = 1; emit Done(to, x); }"

    hf_db = tmp_path / "hf.db"
    hf_output = tmp_path / "hf.jsonl"
    download_hf_contracts.init_database(hf_db)
    assert download_hf_contracts._store_pairs_batch(
        hf_db,
        [_make_pair(download_hf_contracts, 1, body) | {"tac_representation": tac}],
    ) == 1
    download_hf_contracts.export_training_data(
        str(hf_output),
        max_body_dupes=5,
        db_path=hf_db,
        manifest_path=tmp_path / "hf.manifest.json",
    )

    builder = DatasetBuilder("dummy", output_dir=str(tmp_path / "builder"))
    builder._store_function_pair(
        FunctionPair(
            function_name="transfer",
            tac_representation=tac,
            solidity_code=body,
            function_signature="function transfer(address)",
            visibility="public",
            is_payable=False,
            is_view=False,
            contract_address="0x0000000000000000000000000000000000000001",
            metadata={"selector": "0xa9059cbb"},
        )
    )
    builder_output = Path(builder.export_dataset("jsonl"))

    hf_row = json.loads(hf_output.read_text().splitlines()[0])
    builder_row = json.loads(builder_output.read_text().splitlines()[0])
    assert hf_row["input"] == builder_row["input"]
    assert hf_row["output"] == builder_row["output"]
    assert final_row_hash(hf_row["input"], hf_row["output"]) == final_row_hash(
        builder_row["input"], builder_row["output"]
    )


def test_export_training_data_deduplicates_after_final_sanitization(tmp_path):
    import download_hf_contracts

    db_path = tmp_path / "contracts.db"
    output_path = tmp_path / "dataset.jsonl"
    manifest_path = tmp_path / "dataset.manifest.json"
    download_hf_contracts.init_database(db_path)

    body = "function transfer(address to) public { uint256 x = 1; emit Done(to, x); }"
    tac_a = (
        "function transfer(address to):\n"
        "  // Selector: 0xa9059cbb\n"
        "  block:\n"
        "    temp = 1"
    )
    tac_b = (
        "function selector_a9059cbb:\n"
        "  // Selector: 0xa9059cbb\n"
        "  block:\n"
        "    temp = 1"
    )
    with sqlite3.connect(db_path) as conn:
        for idx, tac in enumerate([tac_a, tac_b], start=1):
            conn.execute(
                """
                INSERT INTO function_pairs (
                    contract_address, function_name, tac_representation, solidity_code,
                    function_signature, visibility, is_payable, is_view, metadata, hash,
                    body_hash, tac_hash, pair_norm_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"0x{idx:040x}",
                    "transfer",
                    tac,
                    body,
                    "function transfer(address)",
                    "public",
                    False,
                    False,
                    json.dumps({"selector": "0xa9059cbb"}),
                    download_hf_contracts._md5(f"{tac}{body}{idx}"),
                    download_hf_contracts.hash_normalized_body(body),
                    download_hf_contracts.hash_normalized_tac(tac),
                    download_hf_contracts._md5(f"distinct-{idx}"),
                ),
            )

    download_hf_contracts.export_training_data(
        str(output_path),
        max_body_dupes=5,
        db_path=db_path,
        manifest_path=manifest_path,
    )

    rows = output_path.read_text().splitlines()
    rejects = [json.loads(line) for line in Path(f"{output_path}.rejects.jsonl").read_text().splitlines()]
    manifest = json.loads(manifest_path.read_text())
    assert len(rows) == 1
    assert len(rejects) == 1
    assert rejects[0]["reasons"] == ["final_row_duplicate"]
    assert manifest["drop_counts"]["final_row_duplicate_rows"] == 1
    assert manifest["validation"]["final_row_duplicates"]["export_reject_count"] == 1


def test_export_training_data_quarantines_tac_errors_and_auxiliary_contracts(tmp_path):
    import download_hf_contracts

    db_path = tmp_path / "contracts.db"
    output_path = tmp_path / "dataset.jsonl"
    manifest_path = tmp_path / "dataset.manifest.json"
    download_hf_contracts.init_database(db_path)
    address = "0x00000000000000000000000000000000000000aa"

    rows = [
        (
            "good",
            "function selector_11111111:\n  // Selector: 0x11111111\n  block:\n    temp = 1",
            "function good() public { uint256 x = 1; emit Done(x); }",
            "Target",
        ),
        (
            "badTac",
            "function selector_22222222:\n  // Selector: 0x22222222\n  block:\n    goto stack_underflow",
            "function badTac() public { uint256 x = 2; emit Done(x); }",
            "Target",
        ),
        (
            "helper",
            "function selector_33333333:\n  // Selector: 0x33333333\n  block:\n    temp = 3",
            "function helper() public { uint256 x = 3; emit Done(x); }",
            "SafeMathIntLib",
        ),
    ]
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO contracts (address, source_code, bytecode, contract_name) VALUES (?, ?, ?, ?)",
            (address, "contract Target {}", "0x00", "Target"),
        )
        for idx, (name, tac, body, compiled_contract) in enumerate(rows, start=1):
            conn.execute(
                """
                INSERT INTO function_pairs (
                    contract_address, function_name, tac_representation, solidity_code,
                    function_signature, visibility, is_payable, is_view, metadata, hash,
                    body_hash, tac_hash, pair_norm_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    address,
                    name,
                    tac,
                    body,
                    f"function {name}()",
                    "public",
                    False,
                    False,
                    json.dumps({"compiled_contract": compiled_contract}),
                    download_hf_contracts._md5(f"{tac}{body}{idx}"),
                    download_hf_contracts.hash_normalized_body(body),
                    download_hf_contracts.hash_normalized_tac(tac),
                    download_hf_contracts.hash_normalized_pair(tac, body),
                ),
            )

    download_hf_contracts.export_training_data(
        str(output_path),
        max_body_dupes=5,
        db_path=db_path,
        manifest_path=manifest_path,
    )

    exported = [json.loads(line) for line in output_path.read_text().splitlines()]
    rejects = [json.loads(line) for line in Path(f"{output_path}.rejects.jsonl").read_text().splitlines()]
    manifest = json.loads(manifest_path.read_text())
    assert [row["metadata"]["function_name"] for row in exported] == ["good"]
    assert {reason for row in rejects for reason in row["reasons"]} >= {
        "tac_stack_underflow",
        "auxiliary_compiled_contract",
    }
    assert manifest["drop_counts"]["tac_error_rows"] == 1
    assert manifest["drop_counts"]["auxiliary_compiled_contract_rows"] == 1


def test_dataset_builder_manifest_filter_drops_and_partial_quarantine(tmp_path):
    from src.dataset_pipeline import DatasetBuilder, FunctionPair

    builder = DatasetBuilder("dummy", output_dir=str(tmp_path / "builder"))
    rich_body = (
        "function rich() public { uint256 a = 1; uint256 b = 2; uint256 c = 3; "
        "uint256 d = 4; uint256 e = a + b + c + d; emit Done(e); }"
    )
    normal = FunctionPair(
        function_name="rich",
        tac_representation="function selector_12345678:\n  // Selector: 0x12345678\n  temp = 1",
        solidity_code=rich_body,
        function_signature="function rich()",
        visibility="public",
        is_payable=False,
        is_view=False,
        contract_address="0x0000000000000000000000000000000000000001",
    )
    partial = FunctionPair(
        function_name="unknown_deadbeef",
        tac_representation="function selector_deadbeef:\n  // Selector: 0xdeadbeef\n  temp = 2",
        solidity_code=(
            "// Partial decompilation — selector: 0xdeadbeef\n"
            "function unknown_deadbeef(/* params unknown */) public {\n"
            "    // TODO: Full logic not reconstructed\n"
            "}"
        ),
        function_signature="function unknown_deadbeef()",
        visibility="public",
        is_payable=False,
        is_view=False,
        contract_address="0x0000000000000000000000000000000000000001",
        metadata={"partial": True, "selector": "0xdeadbeef"},
    )
    builder._store_function_pair(normal)
    builder._store_function_pair(partial)
    builder._record_generation_diagnostic(
        stage="compile",
        contract_address=normal.contract_address,
        compiler_version="0.8.20",
        optimizer_enabled=True,
        optimization_runs=200,
        status="compile_failed",
        error="sample solc error",
    )

    exported = Path(builder.export_dataset("jsonl", include_partial=True))
    main_rows = [json.loads(line) for line in exported.read_text().splitlines()]
    partial_rows = [
        json.loads(line)
        for line in (exported.parent / "smart_contract_dataset.partial.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(main_rows) == 1
    assert "Partial decompilation" not in main_rows[0]["output"]
    assert len(partial_rows) == 1
    assert partial_rows[0]["metadata"]["partial"] is True
    assert partial_rows[0]["metadata"]["partial_split"] == "partial_placeholders"

    builder.filter_and_clean_dataset(min_length=20, max_length=1000)
    filtered_manifest_path = Path(builder.export_dataset("jsonl")).with_suffix(
        ".jsonl.manifest.json"
    )
    manifest = json.loads(filtered_manifest_path.read_text())
    assert manifest["drop_counts"]["partial_placeholder"] == 1
    assert manifest["failure_diagnostics"]["status_counts"]["compile_failed"] == 1


def test_versioned_training_metadata_schema_validator():
    from src.dataset_pipeline import (
        normalize_training_metadata,
        validate_training_record_schema,
    )

    valid = {
        "input": "function selector_a9059cbb:",
        "output": "function transfer() public {}",
        "metadata": normalize_training_metadata(
            {
                "contract_address": "0x0000000000000000000000000000000000000001",
                "selector": "0xa9059cbb",
                "compiler_version": "0.8.20",
                "optimizer_enabled": True,
                "body_hash": "a" * 32,
            }
        ),
    }
    assert validate_training_record_schema(valid)["status"] == "passed"

    malformed = {
        "input": "tac",
        "output": "solidity",
        "metadata": {
            "schema_version": 1,
            "contract_address": "0xabc",
            "selector": "0x1234",
            "optimizer_enabled": "true",
            "compiler_version": "latest",
            "body_hash": "not-a-hash",
        },
    }
    result = validate_training_record_schema(malformed)
    assert result["status"] == "failed"
    assert {
        "contract_address_format",
        "selector_format",
        "boolean_type_error",
        "compiler_version_format",
        "hash_format",
    } <= {error["code"] for error in result["errors"]}

    legacy = {"input": "tac", "output": "solidity", "metadata": {}}
    assert validate_training_record_schema(legacy)["status"] == "failed"
    assert validate_training_record_schema(legacy, allow_legacy=True)["status"] == "passed"


def test_sanitized_training_record_adds_hashes_and_quality_signals():
    from src.dataset_export_primitives import (
        build_training_record,
        final_row_hash,
        prompt_leakage_reject_reasons,
        sanitize_tac_prompt_input,
        validate_training_record_schema,
    )

    tac = """// Solidity compiler: 0.8.20
// Optimizer: enabled
// Function: transfer
// Visibility: public
// Payable: false
// Storage layout:
// [slot 1] uint256 totalSupply
function transfer(address to, uint256 amount):
// selector 0xa9059cbb
block_0:
  temp_1 = CALLDATALOAD 0x04
  storage[0] = temp_1  // likely: mapping(address => uint256) balances
"""
    output = "function transfer(address to, uint256 amount) public { emit Done(to, amount); }"

    sanitized = sanitize_tac_prompt_input(tac)
    assert "function selector_a9059cbb:" in sanitized
    assert "transfer(address" not in sanitized
    assert "Compiler" not in sanitized
    assert "Optimizer" not in sanitized
    assert "Visibility" not in sanitized
    assert "Storage layout" not in sanitized
    assert "likely:" not in sanitized
    assert prompt_leakage_reject_reasons(sanitized) == []
    assert prompt_leakage_reject_reasons("  return memory[temp_1:temp_2]") == []
    assert prompt_leakage_reject_reasons("// Returns: amount") == [
        "prompt_source_metadata_leak"
    ]

    record = build_training_record(
        tac,
        output,
        {
            "selector": "0xa9059cbb",
            "contract_address": "0x0000000000000000000000000000000000000001",
        },
    )
    metadata = record["metadata"]
    assert metadata["input_hash"]
    assert metadata["output_hash"]
    assert metadata["final_row_hash"] == final_row_hash(record["input"], record["output"])
    assert metadata["quality"]["tac_instruction_lines"] == 2
    assert validate_training_record_schema(record)["status"] == "passed"


def test_export_training_data_quarantines_malformed_metadata_with_quality_signals(tmp_path):
    import download_hf_contracts

    db_path = tmp_path / "contracts.db"
    output_path = tmp_path / "dataset.jsonl"
    manifest_path = tmp_path / "dataset.manifest.json"
    download_hf_contracts.init_database(db_path)

    good_body = "function good() public { uint256 x = 1; uint256 y = x + 1; emit Done(y); }"
    bad_body = "function bad() public { uint256 x = 2; uint256 y = x + 2; emit Done(y); }"
    bad_pair = _make_pair(download_hf_contracts, 2, bad_body)
    bad_pair["metadata"] = json.dumps({"compiler_version": "latest"})
    assert (
        download_hf_contracts._store_pairs_batch(
            db_path,
            [_make_pair(download_hf_contracts, 1, good_body), bad_pair],
        )
        == 2
    )

    download_hf_contracts.export_training_data(
        str(output_path),
        max_body_dupes=5,
        db_path=db_path,
        manifest_path=manifest_path,
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    rejects = [json.loads(line) for line in Path(f"{output_path}.rejects.jsonl").read_text().splitlines()]
    manifest = json.loads(manifest_path.read_text())

    assert len(rows) == 1
    assert rows[0]["metadata"]["final_row_hash"]
    assert rows[0]["metadata"]["quality"]["input_token_estimate"] > 0
    assert len(rejects) == 1
    assert rejects[0]["reasons"] == ["metadata_schema_invalid"]
    assert rejects[0]["quality"]["tac_instruction_lines"] > 0
    assert rejects[0]["schema_errors"][0]["code"] == "compiler_version_format"
    assert manifest["validation"]["quality_filter"]["reason_counts"] == {
        "metadata_schema_invalid": 1
    }


def test_record_quality_report_flags_selector_mismatch_and_bad_outputs():
    from src.dataset_export_primitives import (
        build_training_record,
        training_record_quality_report,
    )

    record = build_training_record(
        "function bytecode_function:\n  temp_1 = CALLDATALOAD 0x04",
        "uint256 x = 1;",
        {"selector": "0xdeadbeef"},
    )

    report = training_record_quality_report(record)

    assert report["status"] == "failed"
    assert "selector_input_mismatch" in report["reasons"]
    assert "output_missing_solidity_entrypoint" in report["reasons"]


def test_data_quality_ci_workflow_runs_regression_subset():
    workflow = Path(".github/workflows/data-quality.yml")
    assert workflow.exists()
    text = workflow.read_text()
    assert "tests/test_dataset_quality_issues.py" in text
    assert "tests/test_data_generation_export_issues.py" in text
    assert "pull_request" in text
