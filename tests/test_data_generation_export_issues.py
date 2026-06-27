import json
import sqlite3

import pytest


def _make_pair(download_hf_contracts, idx, body):
    tac = f"function same:\n  block_{idx}:\n    temp = {idx}"
    return {
        "contract_address": f"0x{idx}",
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
    assert payload["failure_diagnostics"]["top_errors"][0]["sample_contract_addresses"] == [
        "0xabc"
    ]
    assert payload["status_counts"]["contracts"]["compile_failed"] == 1
    assert json.loads(manifest_path.read_text())["run_id"] == run_id
