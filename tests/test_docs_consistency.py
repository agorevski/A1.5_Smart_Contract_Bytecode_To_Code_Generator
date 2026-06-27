import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_project_file(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def test_pytest_workflow_disables_unrelated_web3_plugin():
    pyproject = read_project_file("pyproject.toml")
    readme = read_project_file("README.md")
    runbook = read_project_file("docs/runbook.md")

    assert "-p no:pytest_ethereum" in pyproject
    assert "uv run pytest" in readme
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest --collect-only -q" in runbook


def test_downloader_defaults_are_documented_from_cli_help():
    help_result = subprocess.run(
        [sys.executable, "download_hf_contracts.py", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    help_text = " ".join(help_result.stdout.split())
    readme = read_project_file("README.md")
    runbook = read_project_file("docs/runbook.md")

    assert re.search(r"--max-compiler-versions .*default: 5", help_text)
    assert re.search(r"--max-body-dupes .*default: 2", help_text)
    assert "| `--max-compiler-versions N` | `5` |" in readme
    assert "| `--max-body-dupes N` | `2` |" in readme
    assert "| `--max-compiler-versions N` | 5 |" in runbook
    assert "| `--max-body-dupes N` | 2 |" in runbook
    for flag in ("--cache-dir", "--hf-revision", "--export-selectors", "--import-selectors"):
        assert flag in help_text
        assert flag in readme
        assert flag in runbook


def test_test_inventory_docs_do_not_hard_code_stale_counts():
    readme = read_project_file("README.md")
    runbook = read_project_file("docs/runbook.md")

    assert "~380" not in readme
    assert "across 8 files" not in readme
    assert "~380" not in runbook
    assert "across 8 files" not in runbook
