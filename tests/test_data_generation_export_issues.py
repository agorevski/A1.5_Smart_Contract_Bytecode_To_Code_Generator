import json
import sqlite3
from pathlib import Path

import pytest


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


def test_data_quality_ci_workflow_runs_regression_subset():
    workflow = Path(".github/workflows/data-quality.yml")
    assert workflow.exists()
    text = workflow.read_text()
    assert "tests/test_dataset_quality_issues.py" in text
    assert "tests/test_data_generation_export_issues.py" in text
    assert "pull_request" in text
