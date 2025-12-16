# Claude Instructions for DeepOBS

## IMPORTANT: Read Documentation First

**Before performing any task or answering any question about this project, you MUST:**

1. **Read `README.md`** - Understand the project's purpose, features, and usage
2. **Review relevant documentation in `docs/`**:
   - `docs/QUICK_START_BENCHMARK.md` - For benchmark-related tasks
   - `docs/BENCHMARK_SUITE_README.md` - For comprehensive benchmark documentation
   - `docs/README_PYTORCH.md` - For PyTorch API and usage details
   - `docs/API_REFERENCE.md` - For API documentation
   - `docs/KNOWN_ISSUES.md` - For current limitations and issues
3. **Check example scripts in `examples/`** - For usage patterns and workflows
4. **Review configuration files** (`.yaml` files) - For benchmark configurations

**Never make assumptions about:**
- How to run benchmarks
- What test problems are available
- How to configure optimizers
- How to analyze results
- Project structure and organization

**Always consult the documentation to understand:**
- The correct commands to use
- Available configuration options
- Expected workflows
- Output formats and locations

This ensures accurate, context-aware assistance that aligns with the project's actual capabilities and conventions.

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
