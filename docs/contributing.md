# Contributing

## Setup

```bash
git clone https://github.com/agorevski/A1.5_Smart_Contract_Bytecode_To_Code_Generator.git
cd A1.5_Smart_Contract_Bytecode_To_Code_Generator
uv sync --dev
```

## Development Workflow

1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes
3. Run tests: `uv run pytest -v`
4. Format: `uv run black src/ tests/`
5. Commit with conventional messages (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`)
6. Push and open a pull request

The data-quality GitHub Actions workflow runs targeted CPU regression checks, but contributors should still run the relevant local `uv` commands before opening a pull request.

## Code Standards

- Type hints on all public functions
- Docstrings with `Args`/`Returns` sections
- Logging via `logging.getLogger(__name__)`
- Error handling with try/except and fallback behavior
- Enums for constants, dataclasses for structured data

## Testing

```bash
uv run pytest                              # all tests
uv run pytest tests/test_bytecode_analyzer.py -v  # specific module
uv run pytest tests/test_vulnerability_detector.py -v  # vulnerability detection
uv run pytest tests/test_e2e.py -v                # end-to-end integration
```

## Areas for Contribution

- **Bytecode analysis**: improved pattern recognition, new opcode support, CFG enhancements
- **Data pipeline**: additional data sources, better dedup strategies, dataset quality (see `docs/runbook.md` and `docs/data-format.md`)
- **Model**: alternative architectures, training improvements (see `docs/training-recommendations.md`)
- **Vulnerability detection**: new vulnerability types, improved pattern matching
- **Malicious classification**: enhanced features, model improvements, explainability
- **Audit reports**: richer findings, better risk scoring
- **Pipeline orchestration**: new stages, parallel execution, caching
- **Web application**: UI improvements, new API endpoints
- **Tests**: edge cases, integration tests, new module coverage
- **Docs**: examples, corrections, tutorials

## License

Contributions are licensed under MIT (see [LICENSE](../LICENSE)).