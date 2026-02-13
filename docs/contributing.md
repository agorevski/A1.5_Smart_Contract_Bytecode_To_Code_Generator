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
python -m pytest                              # all tests
python -m pytest tests/test_bytecode_analyzer.py -v  # specific module
python -m pytest --cov=src tests/             # with coverage
```

## Areas for Contribution

- **Bytecode analysis**: improved pattern recognition, new opcode support
- **Data pipeline**: additional data sources, better dedup strategies
- **Model**: alternative architectures, training improvements
- **Tests**: edge cases, integration tests
- **Docs**: examples, corrections

## License

Contributions are licensed under MIT (see [LICENSE](../LICENSE)).