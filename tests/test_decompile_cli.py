import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "decompile.py"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_rejects_invalid_bytecode():
    result = run_cli("--format", "tac", "--bytecode", "0x123")

    assert result.returncode != 0
    assert "even number" in result.stderr


def test_cli_rejects_missing_model_for_json():
    result = run_cli("--bytecode", "0x6000")

    assert result.returncode != 0
    assert "--model-path is required" in result.stderr


def test_cli_tac_output_without_model():
    result = run_cli("--format", "tac", "--bytecode", "0x600000")

    assert result.returncode == 0
    assert "TAC" in result.stdout or "block" in result.stdout or result.stdout.strip()
