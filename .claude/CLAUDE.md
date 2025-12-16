# Claude Instructions for DeepOBS

## Project Setup

This project uses **UV** as the Python package manager. All Python commands should be run using UV.

### Running Python Commands

Always use UV to run Python commands:

```bash
# Run Python scripts
uv run python script.py

# Run pytest
uv run pytest

# Run any Python command
uv run <command>
```

### Installing Dependencies

```bash
# Install project with dependencies
uv pip install -e ".[pytorch,dev]"

# Install specific extras
uv pip install -e ".[pytorch]"
uv pip install -e ".[all,dev]"
```

### Virtual Environment

The project uses `.venv/` for the virtual environment (already in .gitignore).

```bash
# Create virtual environment (if needed)
uv venv

# UV commands automatically use the virtual environment
```

## Testing

Always use `uv run` for testing:

```bash
# Run smoke test
uv run python smoke_test.py

# Run all tests
uv run pytest tests/

# Run specific tests
uv run pytest tests/test_datasets.py

# Run with coverage
uv run pytest tests/ --cov=deepobs.pytorch
```

## Development Workflow

1. Make changes to code
2. Run tests with `uv run pytest`
3. Format with `uv run black .`
4. Lint with `uv run flake8`

## Documentation Organization

### Planning and Temporary Documentation

All planning documentation, temporary notes, and development documentation should be kept in the `planning/` folder:

- Implementation plans
- Design documents
- Temporary notes and scratchpads
- Development logs
- Migration notes
- Work-in-progress documentation

**Important**: The `planning/` folder is for transient documentation. Once work is complete, relevant information should be moved to:
- `docs/` for permanent documentation
- `README.md` or `docs/README_PYTORCH.md` for user-facing guides
- Code comments for implementation details
