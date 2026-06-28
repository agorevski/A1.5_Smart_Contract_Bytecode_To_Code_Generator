import json
from pathlib import Path

from scripts.build_curriculum_dataset import (
    excluded_identities,
    focus_categories,
    load_jsonl,
    row_identity,
    select_curriculum_rows,
    write_curriculum,
)


def _row(body_hash, output, input_text="function selector_deadbeef:\n  CALL"):
    return {
        "input": input_text,
        "metadata": {
            "body_hash": body_hash,
            "function_signature": f"function f_{body_hash}()",
        },
        "output": output,
    }


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle)
            handle.write("\n")


def test_select_curriculum_rows_focuses_calls_and_excludes_identities(tmp_path):
    source_rows = [
        _row("excluded", "function excluded() public { token.transfer(msg.sender, 1); }"),
        _row(
            "best",
            "function duplicateLong() public { token.balanceOf(msg.sender); token.transfer(owner, 1); }",
            input_text="function selector_deadbeef:\n  CALL\n" + ("  JUMP\n" * 100),
        ),
        _row(
            "best",
            "function best() public { token.balanceOf(msg.sender); token.transfer(owner, 1); }",
        ),
        _row("state", "function state() public { total = 1; }"),
    ]
    excluded_path = tmp_path / "excluded.jsonl"
    _write_jsonl(excluded_path, [source_rows[0]])

    selected = select_curriculum_rows(
        source_rows,
        categories=focus_categories("calls"),
        exclude=excluded_identities([excluded_path]),
        max_rows=4,
    )

    assert [candidate.identity for candidate in selected] == [row_identity(source_rows[1])]
    assert selected[0].source_index == 2
    assert selected[0].focus_counts == {"call": 2, "member_call": 2}


def test_write_curriculum_outputs_dataset_and_manifest(tmp_path):
    source_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "calls.jsonl"
    manifest_path = tmp_path / "calls.manifest.json"
    source_rows = [
        _row("calls", "function calls() public { token.transfer(owner, 1); }"),
        _row("plain", "function plain() public { }"),
    ]
    _write_jsonl(source_path, source_rows)
    candidates = select_curriculum_rows(
        load_jsonl(source_path),
        categories=focus_categories("calls"),
        max_rows=8,
    )

    manifest = write_curriculum(
        candidates,
        output_path=output_path,
        manifest_path=manifest_path,
        source_dataset=source_path,
        focus="calls",
        categories=focus_categories("calls"),
        exclude_datasets=[],
    )

    assert output_path.exists()
    assert manifest_path.exists()
    assert len(load_jsonl(output_path)) == 1
    assert manifest["selected_rows"] == 1
    assert manifest["rows"][0]["focus_counts"]["member_call"] == 1
