import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_project_file(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


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

    assert re.search(r"--max-compiler-versions .*default: 1", help_text)
    assert re.search(r"--max-body-dupes .*default: 2", help_text)
    assert re.search(r"--max-seq-length .*default: 8192", help_text)
    assert "| `--max-compiler-versions N` | `1` |" in readme
    assert "| `--max-body-dupes N` | `2` |" in readme
    assert "| `--max-seq-length N` | `8192` |" in readme
    assert "| `--max-compiler-versions N` | 1 |" in runbook
    assert "| `--max-body-dupes N` | 2 |" in runbook
    assert "| `--max-seq-length N` | 8192 |" in runbook
    for flag in ("--cache-dir", "--hf-revision", "--export-selectors", "--import-selectors"):
        assert flag in help_text
        assert flag in readme
        assert flag in runbook


def test_training_defaults_are_documented_from_cli_constants():
    import train

    readme = read_project_file("README.md")
    runbook = read_project_file("docs/runbook.md")
    gpu_notes = read_project_file("docs/gpu-parameter-sweep-results.md")

    assert f"| `--batch-size N` | `{train.DEFAULT_BATCH_SIZE}` |" in readme
    assert f"| `--max-seq-length N` | `{train.DEFAULT_MAX_SEQ_LENGTH}` |" in readme
    assert "| `--gradient-checkpointing` / `--no-gradient-checkpointing` | on |" in readme
    assert f"`max_seq_length={train.DEFAULT_MAX_SEQ_LENGTH}`" in gpu_notes
    assert f"`batch_size={train.DEFAULT_BATCH_SIZE}`" in gpu_notes
    assert f"--max-seq-length {train.DEFAULT_MAX_SEQ_LENGTH}" in runbook


def test_test_inventory_docs_do_not_hard_code_stale_counts():
    readme = read_project_file("README.md")
    runbook = read_project_file("docs/runbook.md")

    assert "~380" not in readme
    assert "across 8 files" not in readme
    assert "~380" not in runbook
    assert "across 8 files" not in runbook


def test_supported_python_and_cpu_profiles_are_consistent():
    pyproject = read_project_file("pyproject.toml")
    workflow = read_project_file(".github/workflows/data-quality.yml")
    readme = read_project_file("README.md")
    runbook = read_project_file("docs/runbook.md")

    assert 'requires-python = ">=3.10,<3.13"' in pyproject
    assert 'python-version: ["3.10", "3.12"]' in workflow
    assert "3.10\u20133.12" in readme
    assert "3.10\u20133.12" in runbook
    assert ">=3.13,<3.14" not in runbook
    assert 'extra = "cpu"' in pyproject
    assert 'url = "https://download.pytorch.org/whl/cpu"' in pyproject
    assert 'url = "https://download.pytorch.org/whl/cu130"' in pyproject
    assert 'conflicts = [[{ extra = "cpu" }, { extra = "cuda" }]]' in pyproject
    assert "assert torch.version.cuda is None" in workflow
    assert "--extra training --extra web --extra analysis --extra cpu pytest" in workflow
    core_dependencies = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
    for optional in ("torch", "bitsandbytes", "flask", "lightgbm"):
        assert f'"{optional}' not in core_dependencies
