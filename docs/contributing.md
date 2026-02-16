# Contributing

## Setup

```bash
git clone https://github.com/agorevski/A1.5_Smart_Contract_Bytecode_To_Code_Generator.git
cd A1.5_Smart_Contract_Bytecode_To_Code_Generator
python -m venv venv && venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Development Workflow

1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes
3. Run tests: `python -m pytest -v`
4. Format: `black src/ tests/`
5. Commit with conventional messages (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`)
6. Push and open a pull request

## Code Standards

- Type hints on all public functions
- Docstrings with `Args`/`Returns` sections
- Logging via `logging.getLogger(__name__)`
- Error handling with try/except and fallback behavior
- Enums for constants, dataclasses for structured data

## Testing

```bash
python -m pytest                              # all tests (~380 across 8 files)
python -m pytest tests/test_bytecode_analyzer.py -v  # specific module
python -m pytest tests/test_vulnerability_detector.py -v  # vulnerability detection
python -m pytest tests/test_e2e.py -v                # end-to-end integration
python -m pytest --cov=src tests/             # with coverage
```

## Areas for Contribution

- **Bytecode analysis**: improved pattern recognition, new opcode support, CFG enhancements
- **Data pipeline**: additional data sources, better dedup strategies, dataset quality (see `docs/dataset-generation.md`)
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