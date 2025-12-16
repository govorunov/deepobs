# Claude Instructions for DeepOBS

## Project Overview

**DeepOBS is now a PyTorch-only project.** All TensorFlow-related code can and should be safely removed. When fixing issues or updating code:
- Always target PyTorch implementations
- Remove any TensorFlow dependencies or legacy code
- Update imports and naming to follow PyTorch conventions
- No backward compatibility with TensorFlow is needed

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

## Important Notes

- **Always use `uv run` prefix** for Python commands
- UV is faster than pip and provides better dependency resolution
- The virtual environment is automatically managed by UV
