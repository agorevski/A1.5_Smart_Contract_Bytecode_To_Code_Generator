import json
import sqlite3
from pathlib import Path

import pytest


def _inheritance_ast(order=("Base", "Derived", "Unrelated"), override=False):
    sources = {}
    output = {"sources": {}, "contracts": {}}
    ids = {"Base": 10, "Derived": 20, "Unrelated": 30}
    for file_id, name in enumerate(order):
        function = (
            f"function value() public view returns (uint256) {{ return {ids[name]}; }}"
            if name != "Derived" or override else ""
        )
        source = f"// café\ncontract {name} {{ {function} }}"
        path = f"{name}.sol"
        sources[path] = source
        encoded = source.encode()
        nodes = []
        if function:
            start = encoded.index(b"function")
            body_start = encoded.index(b"{", start)
            body_end = encoded.index(b"}", body_start) + 1
            nodes.append({
                "nodeType": "FunctionDefinition", "id": ids[name] + 1,
                "name": "value", "visibility": "public", "stateMutability": "view",
                "functionSelector": "12345678",
                "src": f"{start}:{len(function.encode())}:{file_id}",
                "body": {"src": f"{body_start}:{body_end - body_start}:{file_id}"},
            })
        output["sources"][path] = {
            "id": file_id,
            "ast": {"nodes": [{
                "nodeType": "ContractDefinition", "id": ids[name], "name": name,
                "linearizedBaseContracts": [20, 10] if name == "Derived" else [ids[name]],
                "nodes": nodes,
            }]},
        }
        output["contracts"][path] = {name: {
            "evm": {"deployedBytecode": {"object": "6001600055"},
                    "methodIdentifiers": {"value()": "12345678"}}
        }}
    return output, sources


@pytest.mark.parametrize("order", [
    ("Base", "Derived", "Unrelated"), ("Unrelated", "Derived", "Base"),
])
@pytest.mark.parametrize("override", [False, True])
def test_ast_effective_implementation_is_inherited_or_overridden_not_unrelated(order, override):
    from src.local_compiler import resolve_effective_functions

    output, sources = _inheritance_ast(order, override)
    functions = resolve_effective_functions(output, sources, "Derived.sol", "Derived")
    assert len(functions) == 1
    expected = "Derived" if override else "Base"
    assert functions[0]["source_file"] == f"{expected}.sol"
    assert functions[0]["contract_name"] == expected
    assert functions[0]["body"].startswith("function value()")
    assert f"return {20 if override else 10};" in functions[0]["body"]


def test_ast_uses_solc_linearization_not_source_declaration_order():
    from src.local_compiler import resolve_effective_functions

    output, sources = _inheritance_ast()
    derived = output["sources"]["Derived.sol"]["ast"]["nodes"][0]
    # Treat the second base as the rightmost base in multiple inheritance.
    derived["linearizedBaseContracts"] = [20, 30, 10]
    assert resolve_effective_functions(
        output, sources, "Derived.sol", "Derived"
    )[0]["declaring_contract_id"] == 30


def test_ast_rejects_missing_identity_and_duplicate_local_selector():
    from src.local_compiler import resolve_effective_functions

    output, sources = _inheritance_ast()
    with pytest.raises(ValueError, match="identity"):
        resolve_effective_functions(output, sources, "missing.sol", "Derived")
    base = output["sources"]["Base.sol"]["ast"]["nodes"][0]
    base["nodes"].append(dict(base["nodes"][0], id=99))
    with pytest.raises(ValueError, match="ambiguous selector"):
        resolve_effective_functions(output, sources, "Derived.sol", "Derived")


def test_compiler_requests_ast_and_quarantines_missing_ast(monkeypatch):
    from src import local_compiler

    output, sources = _inheritance_ast()
    captured = {}

    def compile_standard(value, **kwargs):
        captured.update(value)
        return output

    monkeypatch.setattr(local_compiler, "install_solc_version", lambda _: True)
    monkeypatch.setattr(local_compiler.solcx, "compile_standard", compile_standard)
    result = local_compiler.compile_multi_file(sources, "0.8.20")
    assert result.success
    assert captured["settings"]["outputSelection"]["*"][""] == ["ast"]
    assert result.contracts["Derived"].effective_functions[0]["contract_name"] == "Base"
    output["sources"] = {}
    result = local_compiler.compile_multi_file(sources, "0.8.20")
    assert result.contracts["Derived"].effective_functions == []
    assert "AST identity" in result.contracts["Derived"].label_resolution_error


def test_single_source_compiler_preserves_original_filename(monkeypatch):
    from src import local_compiler

    output, sources = _inheritance_ast(("Base",))
    captured = {}

    def compile_standard(value, **kwargs):
        captured.update(value)
        return output

    monkeypatch.setattr(local_compiler, "install_solc_version", lambda _: True)
    monkeypatch.setattr(local_compiler.solcx, "compile_standard", compile_standard)
    result = local_compiler.compile_source(
        sources["Base.sol"], "0.8.20", source_filename="Base.sol"
    )
    assert set(captured["sources"]) == {"Base.sol"}
    assert result.contracts["Base"].effective_functions[0]["source_file"] == "Base.sol"


def test_ast_same_contract_name_in_other_file_does_not_overwrite_artifact():
    from src.local_compiler import _compiled_contracts

    output, sources = _inheritance_ast(("Base", "Unrelated"))
    source = sources.pop("Unrelated.sol").replace("Unrelated", "Base")
    sources["Other.sol"] = source
    info = output["sources"].pop("Unrelated.sol")
    info["ast"]["nodes"][0]["name"] = "Base"
    # The shorter contract name changes the function source offsets.
    node = info["ast"]["nodes"][0]["nodes"][0]
    for span in (node, node["body"]):
        start, length, file_id = map(int, span["src"].split(":"))
        span["src"] = f"{start - 5}:{length}:{file_id}"
    output["sources"]["Other.sol"] = info
    output["contracts"]["Other.sol"] = {"Base": output["contracts"].pop("Unrelated.sol")["Unrelated"]}
    contracts = _compiled_contracts(output, sources)
    assert set(contracts) == {"Base.sol:Base", "Other.sol:Base"}
    assert contracts["Base.sol:Base"].effective_functions[0]["source_file"] == "Base.sol"
    assert "return 30;" in contracts["Other.sol:Base"].effective_functions[0]["body"]


@pytest.mark.parametrize("generator", ["hf", "lookup"])
def test_generation_uses_compiled_ast_labels_not_source_selector_fallback(monkeypatch, generator):
    from types import SimpleNamespace
    from src.local_compiler import _compiled_contracts
    import download_hf_contracts as hf
    from scripts import build_lookup_db as lookup

    output, sources = _inheritance_ast()
    compiled = _compiled_contracts(output, sources)
    compilation = SimpleNamespace(success=True, contracts=compiled, errors=[])

    class Analyzer:
        def __init__(self, bytecode):
            self.basic_blocks = {}

        def analyze_control_flow(self):
            pass

        def identify_functions(self):
            return {"value": SimpleNamespace(
                selector="0x12345678", entry_block="block_0",
                basic_blocks=[SimpleNamespace(
                    id="block_0", instructions=["return 10"], predecessors=[], successors=[],
                )],
            )}

        def _format_tac_instruction(self, instruction):
            return instruction

    module = hf if generator == "hf" else lookup
    monkeypatch.setattr(module, "install_solc_version", lambda _: True)
    monkeypatch.setattr(module, "compile_multi_file", lambda *args: compilation)
    monkeypatch.setattr(module, "BytecodeAnalyzer", Analyzer)
    poison = [{"selector": "0x12345678", "body": "unrelated wrong label"}]
    if generator == "hf":
        monkeypatch.setattr(hf, "is_trivial_function", lambda _: False)
        result = hf._compile_one_job("0x" + "1" * 40, sources, poison, "0.8.20",
                                     False, 200, 0, "Derived")
        assert result["status"] == "processed"
        assert len(result["pairs"]) == 1
        assert "return 10;" in result["pairs"][0]["solidity_code"]
        assert json.loads(result["pairs"][0]["metadata"])["declaring_contract_id"] == 10
        compiled["Derived"].effective_functions = []
        result = hf._compile_one_job("0x" + "1" * 40, sources, poison, "0.8.20",
                                     False, 200, 0, "Derived")
        assert result["pairs"] == []
        assert result["drop_counts"]["unresolved_ast_implementation"] == 1
        Analyzer.analysis_status = {"status": "degraded", "issues": [{"code": "stack_underflow"}]}
        result = hf._compile_one_job("0x" + "1" * 40, sources, poison, "0.8.20",
                                     False, 200, 0, "Derived")
        assert result["pairs"] == []
        assert result["drop_counts"]["tac_analysis_degraded"] == 1
    else:
        status, pairs = lookup._compile_one("address", sources, poison, "0.8.20", False, 200)
        assert status == "ok"
        assert len(pairs) == 3
        assert all("wrong label" not in pair["solidity_code"] for pair in pairs)
        Analyzer.analysis_status = {"status": "failed", "issues": [{"code": "parse_error"}]}
        status, pairs = lookup._compile_one("address", sources, poison, "0.8.20", False, 200)
        assert status == "no_pairs"
        assert pairs == []


@pytest.mark.parametrize("status", ["degraded", "failed"])
def test_uncertain_analysis_never_becomes_exact_training_label(status):
    from types import SimpleNamespace
    from src.dataset_export_primitives import extract_tac_for_function, match_functions_by_selector

    analyzer = SimpleNamespace(analysis_status={"status": status, "issues": []})
    function = SimpleNamespace(selector="0x12345678")
    source = [{"selector": "0x12345678", "body": "wrong label"}]
    assert extract_tac_for_function(function, analyzer) == ""
    assert match_functions_by_selector(source, {"value": function}, analyzer) == []


def test_analysis_metadata_is_an_independent_snapshot():
    from types import SimpleNamespace
    from src.dataset_export_primitives import analysis_status_snapshot

    analyzer = SimpleNamespace(analysis_status={"status": "ok", "issues": []})
    snapshot = analysis_status_snapshot(analyzer)
    analyzer.analysis_status["issues"].append({"code": "fallback"})
    analyzer.analysis_status["status"] = "degraded"
    assert snapshot == {"status": "ok", "issues": []}


def test_selector_matcher_rejects_ambiguous_candidates_in_either_order():
    from types import SimpleNamespace
    from src.dataset_export_primitives import match_functions_by_selector

    analyzer = SimpleNamespace()
    candidates = [{"selector": "0x12345678", "body": "one"},
                  {"selector": "0x12345678", "body": "two"}]
    bytecode = {"value": SimpleNamespace(selector="0x12345678")}
    assert match_functions_by_selector(candidates, bytecode, analyzer) == []
    assert match_functions_by_selector(list(reversed(candidates)), bytecode, analyzer) == []


@pytest.mark.parametrize("version", [None, 1, 999])
def test_lookup_rejects_unversioned_or_stale_db_without_relabeling(tmp_path, version):
    from src.tac_lookup import TACLookup, TACLookupBuilder

    path = tmp_path / "lookup.db"
    TACLookupBuilder(str(path))
    with sqlite3.connect(path) as conn:
        if version is None:
            conn.execute("DROP TABLE lookup_manifest")
        else:
            conn.execute("UPDATE lookup_manifest SET value_json = ?", (
                json.dumps({"tac_schema_version": version, "label_schema_version": 2}),
            ))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Incompatible TAC lookup schema"):
        TACLookup(str(path))
    with pytest.raises(ValueError, match="Incompatible TAC lookup schema"):
        TACLookupBuilder(str(path))
    assert path.read_bytes() == before


def test_new_lookup_and_legacy_row_metadata_versions(tmp_path):
    from src.tac_lookup import TACLookup, TACLookupBuilder
    from src.tac_schema import TAC_SCHEMA_VERSION
    from src.dataset_export_primitives import normalize_training_metadata

    path = tmp_path / "lookup.db"
    builder = TACLookupBuilder(str(path))
    builder.record_manifest({"build_id": "example"})
    builder.insert_pair("return 1", "function value() public { return 1; }")
    lookup = TACLookup(str(path))
    assert lookup.available
    assert lookup.manifest()["tac_schema_version"] == TAC_SCHEMA_VERSION
    assert lookup.manifest()["label_schema_version"] == 2
    assert lookup.query("return 1")["solidity"] == "function value() public { return 1; }"
    lookup.close()
    legacy = normalize_training_metadata({})
    assert legacy["tac_schema_version"] is None
    assert legacy["label_schema_version"] is None


def _run_balanced_materializer(monkeypatch, source, output, manifest, exclusion, cap="1", seed="42", recreate=""):
    import os
    import sys
    script = Path("run_train_qwen_qlora_full_body_balanced.sh").read_text(encoding="utf-8")
    python = script.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    monkeypatch.setattr(sys, "argv", [
        "-", str(source), str(output), str(manifest), cap, seed, "",
        os.path.relpath(exclusion), recreate,
    ])
    exec(compile(python, "<balanced-materializer>", "exec"), {})


def _balanced_files(tmp_path):
    source, output, manifest, exclusion = [
        tmp_path / name for name in ("source.jsonl", "balanced.jsonl", "manifest.json", "eval.jsonl")
    ]
    source.write_text(json.dumps({
        "input": "return 1", "output": "function value() public { return 1; }",
        "metadata": {"body_hash": "one"},
    }) + "\n", encoding="utf-8")
    exclusion.write_text("", encoding="utf-8")
    return source, output, manifest, exclusion


@pytest.mark.parametrize("change", ["source", "cap", "seed", "exclusion", "manifest"])
def test_balanced_reuse_rejects_changed_inputs(tmp_path, monkeypatch, change):
    source, output, manifest, exclusion = _balanced_files(tmp_path)
    _run_balanced_materializer(monkeypatch, source, output, manifest, exclusion)
    cap, seed = "1", "42"
    if change == "source":
        source.write_text(source.read_text() + "\n", encoding="utf-8")
    elif change == "cap":
        cap = "2"
    elif change == "seed":
        seed = "7"
    elif change == "exclusion":
        exclusion.write_text('{"input": "different", "output": "different"}\n', encoding="utf-8")
    else:
        manifest.unlink()
    with pytest.raises(ValueError, match="Stale balanced dataset"):
        _run_balanced_materializer(monkeypatch, source, output, manifest, exclusion, cap, seed)


def test_balanced_reuse_verifies_output_and_independent_eval_overlap(tmp_path, monkeypatch):
    import hashlib
    source, output, manifest, exclusion = _balanced_files(tmp_path)
    heldout = {"input": "heldout", "output": "function heldout() public { return 2; }"}
    exclusion.write_text(json.dumps(heldout) + "\n", encoding="utf-8")
    _run_balanced_materializer(monkeypatch, source, output, manifest, exclusion)
    with pytest.raises(SystemExit) as exit_info:
        _run_balanced_materializer(monkeypatch, source, output, manifest, exclusion)
    assert exit_info.value.code == 0
    output.write_text(output.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content changed"):
        _run_balanced_materializer(monkeypatch, source, output, manifest, exclusion)
    output.write_text(json.dumps(heldout) + "\n", encoding="utf-8")
    data = json.loads(manifest.read_text())
    data["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap detected"):
        _run_balanced_materializer(monkeypatch, source, output, manifest, exclusion)


@pytest.mark.parametrize("mode", ["success", "preflight_failure", "existing_gate_dir"])
def test_balanced_post_training_uses_shared_gate_and_bound_diagnostic_outputs(tmp_path, mode):
    import shutil
    import subprocess

    bash = shutil.which("bash")
    if not bash:
        pytest.skip("Bash is required to exercise the Bash runner")
    script = Path("run_train_qwen_qlora_full_body_balanced.sh").read_text(encoding="utf-8")
    tail = script[script.index("# Use the same fail-closed"):]
    assert "newest_eval_json" not in tail
    assert "--skip-data-preflight" not in tail
    assert "scripts/eval_gate_suite.py" not in tail
    assert "large192_baseline" not in tail
    forwarded = [
        "MODEL_PATH", "GATE_DIR", "NUM_GPUS", "EVAL_BATCH_SIZE",
        "EVAL_MAX_NEW_TOKENS", "EVAL_REPETITION_PENALTY",
        "BROAD_DATASET", "CALLS_DATASET", "STATE_DATASET", "HOLDOUT64_DATASET",
        "PURE_NEGATIVE_DATASET", "LARGE192_DATASET", "BROAD_BASELINE", "CALLS_BASELINE",
        "STATE_BASELINE", "HOLDOUT64_BASELINE", "PURE_NEGATIVE_BASELINE", "LARGE192_BASELINE",
    ]
    child = """#!/usr/bin/env bash
set -euo pipefail
if [[ "${TEST_GATE_MODE}" == "preflight_failure" ]]; then exit 17; fi
mkdir "${GATE_DIR}"
printf '{}' > "${GATE_DIR}/gate_suite.json"
: > "${GATE_DIR}/eval_paths.tsv"
"""
    child += "\n".join(f'printf \'%s\\n\' "${{{name}}}" >> forwarded.txt' for name in forwarded)
    (tmp_path / "run_eval_gate_suite_for_model.sh").write_bytes(child.encode())
    if mode == "existing_gate_dir":
        (tmp_path / "gates").mkdir()
        (tmp_path / "gates" / "gate_suite.json").write_text("stale")
    prefix = """set -euo pipefail
SCRIPT_DIR="$PWD"
OUTPUT_DIR="$PWD/model"
FINAL_MODEL="${OUTPUT_DIR}/final_model"
DATA_DIR="$PWD/data"
GATE_DIR="$PWD/gates"
NUM_GPUS=3
EVAL_BATCH_SIZE=2
EVAL_MAX_NEW_TOKENS=123
EVAL_REPETITION_PENALTY=1.2
uv() {
    printf '%s\\n' "$@" > diagnostic_args.txt
    while [[ "$#" -gt 0 ]]; do
        if [[ "$1" == "--eval-output-json" ]]; then
            shift
            printf '{}' > "$1"
        fi
        shift
    done
}
"""
    prefix += f"export TEST_GATE_MODE={mode}\n"
    prefix += "\n".join(f"{name}={name}-override" for name in forwarded[6:]) + "\n"
    result = subprocess.run(
        [bash], input=(prefix + tail).encode(), cwd=tmp_path,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
    )
    if mode != "success":
        assert result.returncode != 0
        assert not (tmp_path / "diagnostic_args.txt").exists()
        if mode == "preflight_failure":
            assert result.returncode == 17
        else:
            assert (tmp_path / "gates" / "gate_suite.json").read_text() == "stale"
        return
    assert result.returncode == 0, result.stderr.decode()
    values = (tmp_path / "forwarded.txt").read_text().splitlines()
    assert values[0].endswith("/model/final_model")
    assert values[1].endswith("/gates")
    assert values[2:6] == ["3", "2", "123", "1.2"]
    assert values[6:] == [f"{name}-override" for name in forwarded[6:]]
    args = (tmp_path / "diagnostic_args.txt").read_text().splitlines()
    assert args[:3] == ["run", "--extra", "quantization"]
    assert "--eval-output-json" in args
    assert args[args.index("--eval-limit") + 1] == "30"
    assert "--eval-first-n" in args
    assert (tmp_path / "gates" / "eval_train_first30.json").is_file()
    mapping = (tmp_path / "gates" / "eval_paths.tsv").read_text()
    assert mapping.startswith("train_first30\t")


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
    if download_hf_contracts.resource is None:
        assert manifest["performance"]["max_rss_mb"] == 0
    else:
        assert manifest["performance"]["max_rss_mb"] > 0
    assert manifest["performance"]["parquet_streams"][0]["batch_size"] == 2
    assert manifest["drop_counts"]["non_solidity"] == 1


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


def test_data_quality_ci_workflow_runs_full_regression_suite():
    import yaml

    workflow = Path(".github/workflows/data-quality.yml")
    assert workflow.exists()
    text = workflow.read_text()
    config = yaml.safe_load(text)
    steps = config["jobs"]["data-quality-and-quality-gate"]["steps"]
    regression = next(step for step in steps if step.get("name") == "Run CPU-only regression suite")
    assert regression["run"].strip().endswith("pytest")
    assert "--extra cpu" in regression["run"]
    assert "pull_request" in text
