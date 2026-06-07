# Testing and CI

To maintain the high quality required for OpenSSF Best Practices, `synth-nmr` enforces strict testing, linting, and type-checking requirements.

## Automated Test Suite

We use `pytest` for all unit and integration tests. Tests are located in the `tests/` directory.

To run the entire test suite:
```bash
pytest
```

To run a specific test file:
```bash
pytest tests/test_relaxation.py
```

### Coverage Requirements
We aim to maintain >95% code coverage. When contributing new features, you must include corresponding tests. To check coverage locally:
```bash
pytest --cov=synth_nmr --cov-report=term-missing
```

## Linting and Formatting

We use `ruff` for extremely fast linting and `black` for deterministic code formatting.

1. **Format Code**:
   ```bash
   black .
   ```

2. **Lint Code**:
   ```bash
   ruff check .
   ```

## Static Type Checking

We use `mypy` to enforce strict static typing across the entire codebase. This prevents an entire class of runtime errors before they happen.

To run type checking:
```bash
mypy .
```

If `mypy` raises an issue, you must correct the type annotations before your pull request can be merged.

## Continuous Integration (CI)

All of the above checks are run automatically on GitHub Actions for every pull request and push to the `master` branch. A PR cannot be merged if any of the CI pipelines fail.
