# Contributing to HIIT-Methylation-Classification

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Report Issues](#how-to-report-issues)
- [Pull Request Process](#pull-request-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## How to Report Issues

### Bug Reports

When reporting bugs, please include:

1. **Environment Information**
   - Python version
   - Operating system
   - Package versions (output of `pip freeze`)

2. **Bug Description**
   - Clear and concise description of the bug
   - Steps to reproduce the behavior
   - Expected behavior
   - Actual behavior

3. **Additional Context**
   - Error messages (full traceback)
   - Screenshots if applicable
   - Sample data (if shareable and relevant)

### Feature Requests

For feature requests, please include:

1. **Problem Statement**: Describe the problem you're trying to solve
2. **Proposed Solution**: Your suggested implementation
3. **Alternatives Considered**: Other approaches you've considered
4. **Additional Context**: Any other relevant information

### Issue Templates

Use the appropriate issue template when available:
- `bug_report.md` - For bug reports
- `feature_request.md` - For feature requests

## Pull Request Process

### Before You Start

1. **Check existing issues**: Look for related issues or PRs
2. **Open an issue first**: For significant changes, discuss in an issue before coding
3. **Fork the repository**: Create your own fork to work on

### Development Workflow

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes**
   - Follow the code style guidelines
   - Write tests for new functionality
   - Update documentation as needed

3. **Commit your changes**
   ```bash
   git commit -m "Brief description of changes"
   ```

   Commit message guidelines:
   - Use present tense ("Add feature" not "Added feature")
   - Use imperative mood ("Move cursor to..." not "Moves cursor to...")
   - Limit first line to 72 characters
   - Reference issues and PRs in the body

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**
   - Fill out the PR template completely
   - Link related issues
   - Request review from maintainers

### PR Review Criteria

PRs will be reviewed for:
- Code quality and style compliance
- Test coverage
- Documentation updates
- Compatibility with existing codebase
- Performance implications

## Code Style Guidelines

### Python Style

We follow PEP 8 with the following specifications:

1. **Formatting**
   - Use [Black](https://github.com/psf/black) for code formatting
   - Line length: 88 characters (Black default)
   - Use 4 spaces for indentation (no tabs)

2. **Type Hints**
   - Use type hints for all function signatures
   - Use `typing` module for complex types

   ```python
   from typing import List, Dict, Optional, Tuple
   import numpy as np
   import pandas as pd

   def process_data(
       data: pd.DataFrame,
       threshold: float = 0.05,
       columns: Optional[List[str]] = None
   ) -> Tuple[pd.DataFrame, Dict[str, float]]:
       """Process methylation data with specified threshold."""
       pass
   ```

3. **Docstrings**
   - Use Google-style docstrings
   - Include type information in docstrings

   ```python
   def select_features(
       data: pd.DataFrame,
       level: int,
       method: str = "variance"
   ) -> pd.DataFrame:
       """Select features using the ten-level framework.

       Args:
           data: Input methylation beta values matrix.
           level: Feature selection level (1-10).
           method: Selection method ("variance", "differential", "combined").

       Returns:
           DataFrame with selected features.

       Raises:
           ValueError: If level is not between 1 and 10.

       Example:
           >>> selected = select_features(data, level=5)
           >>> print(f"Selected {len(selected.columns)} features")
       """
       pass
   ```

4. **Imports**
   - Use `isort` for import sorting
   - Group imports: standard library, third-party, local

   ```python
   # Standard library
   import os
   from pathlib import Path

   # Third-party
   import numpy as np
   import pandas as pd
   from sklearn.model_selection import cross_val_score

   # Local
   from src.features import FeatureSelector
   from src.models import HIITClassifier
   ```

5. **Naming Conventions**
   - Classes: `CamelCase`
   - Functions/variables: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`
   - Private methods: `_single_leading_underscore`

### Running Style Checks

```bash
# Format code
black src/ scripts/ tests/

# Sort imports
isort src/ scripts/ tests/

# Check types
mypy src/

# Lint
flake8 src/ scripts/ tests/
```

## Testing Requirements

### Test Structure

```
tests/
├── unit/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_visualization.py
├── integration/
│   ├── test_pipeline.py
│   └── test_enrichment.py
└── conftest.py
```

### Writing Tests

1. **Use pytest** as the testing framework
2. **Naming**: Test files and functions should start with `test_`
3. **Coverage**: Aim for >80% code coverage for new code

```python
import pytest
import numpy as np
import pandas as pd
from src.features import FeatureSelector

class TestFeatureSelector:
    """Tests for FeatureSelector class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample methylation data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            np.random.rand(100, 50),
            columns=[f"cg{i:08d}" for i in range(50)]
        )

    def test_ten_level_selection(self, sample_data):
        """Test ten-level feature selection."""
        selector = FeatureSelector(method="ten_level", level=5)
        selected = selector.fit_transform(sample_data)

        assert isinstance(selected, pd.DataFrame)
        assert len(selected.columns) <= len(sample_data.columns)

    def test_invalid_level_raises_error(self, sample_data):
        """Test that invalid level raises ValueError."""
        with pytest.raises(ValueError):
            selector = FeatureSelector(method="ten_level", level=15)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_features.py

# Run specific test
pytest tests/unit/test_features.py::TestFeatureSelector::test_ten_level_selection
```

## Documentation

### Updating Documentation

- Update docstrings for any modified functions/classes
- Update relevant markdown files in `docs/`
- Add examples for new features

### Building Documentation

```bash
# If using Sphinx
cd docs
make html
```

## Questions?

If you have questions about contributing, please:
1. Check existing documentation
2. Search closed issues
3. Open a new issue with your question

Thank you for contributing!
