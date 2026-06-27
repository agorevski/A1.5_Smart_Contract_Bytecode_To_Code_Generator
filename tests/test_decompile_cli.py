import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "decompile.py"


def load_decompile_module():
    spec = importlib.util.spec_from_file_location("decompile_cli_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_cli(*args, env=None):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def test_cli_rejects_invalid_bytecode():
    result = run_cli("--format", "tac", "--bytecode", "0x123")

    assert result.returncode != 0
    assert "even number" in result.stderr


def test_cli_reports_missing_autodiscovered_model_from_env():
    env = os.environ.copy()
    env["WEB_MODEL_PATH"] = "models/definitely_missing_for_test"

    result = run_cli("--bytecode", "0x6000", env=env)

    assert result.returncode != 0
    assert "no trained model artifact found" in result.stderr
    assert "WEB_MODEL_PATH" in result.stderr


def test_cli_tac_output_without_model():
    result = run_cli("--format", "tac", "--bytecode", "0x600000")

    assert result.returncode == 0
    assert "TAC" in result.stdout or "block" in result.stdout or result.stdout.strip()


def test_cli_autodiscovers_newest_final_model(monkeypatch):
    decompile = load_decompile_module()
    model_root = ROOT / "results" / "test_cli_models"
    shutil.rmtree(model_root, ignore_errors=True)
    older = model_root / "final_model_001"
    newer = model_root / "final_model_999"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "model_config.json").write_text("{}", encoding="utf-8")
    (newer / "model_config.json").write_text("{}", encoding="utf-8")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    monkeypatch.setattr(decompile, "MODELS_DIR", model_root)
    monkeypatch.setattr(decompile, "DEFAULT_MODEL_PATH", model_root / "final_model")
    monkeypatch.delenv("WEB_MODEL_PATH", raising=False)

    assert decompile._resolve_model_path(None) == newer.resolve()

    shutil.rmtree(model_root, ignore_errors=True)


def test_cli_function_limit_returns_structured_json(monkeypatch, capsys):
    decompile = load_decompile_module()

    analyzer = SimpleNamespace(instructions=[1], basic_blocks={"b": 1}, functions={})
    monkeypatch.setattr(
        decompile,
        "_analyze_tac",
        lambda bytecode: (analyzer, {"f1": "tac1", "f2": "tac2"}, "tac1\n\ntac2"),
    )

    code = decompile.main(["--format", "json", "--bytecode", "0x6000", "--max-functions", "1"])
    data = json.loads(capsys.readouterr().out)

    assert code == 3
    assert data["success"] is False
    assert data["decompilation_status"] == "work_limit_exceeded"


def test_cli_timeout_returns_structured_json(monkeypatch, capsys):
    decompile = load_decompile_module()

    def slow_analyze(bytecode):
        import time

        time.sleep(0.05)
        analyzer = SimpleNamespace(instructions=[1], basic_blocks={"b": 1}, functions={})
        return analyzer, {"f1": "tac1"}, "tac1"

    monkeypatch.setattr(decompile, "_analyze_tac", slow_analyze)

    code = decompile.main(["--format", "json", "--bytecode", "0x6000", "--timeout-seconds", "0.01"])
    data = json.loads(capsys.readouterr().out)

    assert code == 124
    assert data["success"] is False
    assert data["decompilation_status"] == "timeout"


def test_cli_json_includes_solidity_validation(monkeypatch, capsys):
    decompile = load_decompile_module()
    model_dir = ROOT / "results" / "test_cli_fake_model"
    shutil.rmtree(model_dir, ignore_errors=True)
    model_dir.mkdir(parents=True)
    (model_dir / "model_config.json").write_text("{}", encoding="utf-8")

    analyzer = SimpleNamespace(
        instructions=[1],
        basic_blocks={"b": 1},
        functions={
            "func_00000000": SimpleNamespace(
                selector="0x00000000",
                basic_blocks=[],
            )
        },
    )
    monkeypatch.setattr(decompile, "_resolve_model_path", lambda model_path: model_dir)
    monkeypatch.setattr(
        decompile,
        "_analyze_tac",
        lambda bytecode: (
            analyzer,
            {"func_00000000": "function function_0x00000000:\n  RETURN 0"},
            "function function_0x00000000:\n  RETURN 0",
        ),
    )

    class FakeDecompiler:
        def __init__(self, model_path):
            self.model_path = model_path

        def decompile_tac_to_solidity(self, tac, metadata=None, **kwargs):
            return "function foo() public { }"

        def _assemble_contract(self, functions, analyzer):
            return "contract DecompiledContract {\n" + "\n".join(functions.values()) + "\n}"

    import src.model_setup as model_setup

    monkeypatch.setattr(model_setup, "SmartContractDecompiler", FakeDecompiler)

    code = decompile.main(["--format", "json", "--bytecode", "0x6000"])
    data = json.loads(capsys.readouterr().out)

    assert code == 0
    assert data["validation"]["valid"] is True
    assert data["function_validation"]["func_00000000"]["valid"] is True
    assert data["reconstruction"]["strategy"] == "semantic_function_chunks"
    assert data["quality"]["semantic_chunk_count"] == 1
    assert data["function_results"][0]["source"] == "model_inference"
    assert data["lookup"]["enabled"] is False
    assert data["trace"]["status"] in {"success", "partial"}

    shutil.rmtree(model_dir, ignore_errors=True)
