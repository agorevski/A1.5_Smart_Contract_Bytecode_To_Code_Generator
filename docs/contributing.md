# Contributing Guidelines

We welcome contributions to the Smart Contract Decompilation project!

## Areas for Contribution

### 1. Data Collection

- More efficient contract discovery methods
- Additional data sources beyond Etherscan
- Improved contract filtering and selection

### 2. TAC Generation

- Enhanced pattern recognition
- Better control flow recovery
- Improved storage layout detection

### 3. Model Architecture

- Experiment with different model sizes
- Alternative fine-tuning approaches
- Ensemble methods

### 4. Evaluation

- Additional metrics and benchmarks
- Comparative studies
- Real-world case studies

### 5. Documentation

- More examples and tutorials
- Improved explanations
- Additional use cases

## Getting Started

### 1. Fork the Repository

```bash
# Fork on GitHub, then clone
git clone https://github.com/YOUR_USERNAME/A1.5_Smart_Contract_Bytecode_To_Code_Generator.git
cd A1.5_Smart_Contract_Bytecode_To_Code_Generator
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

## Development Guidelines

### Code Style

We follow PEP 8 style guidelines:

```bash
# Format code
black src/

# Check linting
flake8 src/

# Type checking
mypy src/
```

### Testing

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_bytecode_analyzer.py

# Run with coverage
pytest --cov=src tests/
```

### Documentation

- Add docstrings to all functions and classes
- Update relevant documentation files
- Include examples where appropriate

## Submission Process

### 1. Commit Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

Use conventional commit messages:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring

### 2. Push to Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

- Go to GitHub and create a pull request
- Provide clear description of changes
- Link any related issues
- Ensure all tests pass

## Pull Request Guidelines

### Requirements

- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No merge conflicts

### Review Process

1. Automated checks run
2. Maintainer reviews code
3. Feedback addressed
4. Approved and merged

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Collaborate openly

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or inflammatory comments
- Personal attacks
- Spam or off-topic content

## Questions?

- Check existing [GitHub Issues](../../issues)
- Create new issue for bugs or features
- Join discussions for questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
