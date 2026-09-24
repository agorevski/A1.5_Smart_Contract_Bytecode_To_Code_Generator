import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_sequence_length_cache_tracks_selector_snapshot_and_flag(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "train_common.sh"
    script = source.read_text(encoding="utf-8").split("<<'PY'", 1)[1].split("\nPY\n", 1)[0]
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"input":"TAC","output":"function f() public {}"}\n', encoding="utf-8")
    cache = tmp_path / "lengths.json"
    monkeypatch.setenv("DATASET", str(dataset))
    monkeypatch.setenv("MODEL", "local-model")
    monkeypatch.setenv("SEQ_LEN_CACHE", str(cache))
    monkeypatch.setenv("MAX_SEQ_LEN_CAP", "8192")
    monkeypatch.setenv("SELECTOR_SIGNATURE_METADATA", "true")
    context = {"digest": "first"}
    tokenizer_revision = {"vocab_digest": "first-tokenizer"}
    calls = []

    def detect(path, tokenizer, **kwargs):
        calls.append(kwargs)
        return 256

    monkeypatch.setitem(
        sys.modules, "src.model_setup",
        SimpleNamespace(
            detect_max_sequence_length=detect, TOKENIZATION_CACHE_VERSION=2,
            tokenizer_cache_identity=lambda tokenizer: dict(tokenizer_revision),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "src.selector_resolver",
        SimpleNamespace(snapshot_local_selector_context=lambda: dict(context)),
    )
    monkeypatch.setitem(
        sys.modules, "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())),
    )

    exec(compile(script, str(source), "exec"), {})
    assert calls[-1]["selector_context"] == {"digest": "first"}
    with pytest.raises(SystemExit) as reused:
        exec(compile(script, str(source), "exec"), {})
    assert reused.value.code == 0
    assert len(calls) == 1

    context["digest"] = "updated"
    exec(compile(script, str(source), "exec"), {})
    assert len(calls) == 2
    assert calls[-1]["selector_context"] == {"digest": "updated"}

    monkeypatch.setenv("SELECTOR_SIGNATURE_METADATA", "false")
    exec(compile(script, str(source), "exec"), {})
    assert calls[-1]["include_selector_signature_metadata"] is False
    assert calls[-1]["selector_context"] is None
    tokenizer_revision["vocab_digest"] = "updated-tokenizer"
    exec(compile(script, str(source), "exec"), {})
    assert len(calls) == 4
    identities = [json.loads(key) for key in json.loads(cache.read_text(encoding="utf-8"))]
    assert len(identities) == 4
    assert all(identity["tac_schema_version"] == 2 for identity in identities)
    assert all(identity["label_schema_version"] == 2 for identity in identities)
