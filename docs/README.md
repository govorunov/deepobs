# DeepOBS Documentation

This directory contains all user-facing documentation for DeepOBS.

## 📚 Documentation Files

### PyTorch Implementation (New)

**Main Guides:**
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Complete guide for migrating from TensorFlow to PyTorch
- **[API_REFERENCE.md](API_REFERENCE.md)** - Full PyTorch API documentation
- **[KNOWN_ISSUES.md](KNOWN_ISSUES.md)** - Known limitations and workarounds
- **[CHANGELOG_PYTORCH.md](CHANGELOG_PYTORCH.md)** - PyTorch version history
- **[CONTRIBUTORS.md](CONTRIBUTORS.md)** - Contributors and acknowledgments

**Quick Links:**
- Main README: See `/README.md`
- PyTorch Guide: See `/README_PYTORCH.md`
- Examples: See `/examples/`
- Planning Docs: See `/planning/`

### Original Documentation (Sphinx)

The following directories contain the original Sphinx-based documentation:
- `user_guide/` - User guide (RST files)
- `api/` - API documentation (RST files)
- `pytorch-migration/` - Migration notes

**Building Sphinx Docs:**
```bash
cd docs
make html
```

## 📖 Quick Start

**For PyTorch Users:**
1. Read `/README_PYTORCH.md` - Quick start guide
2. Check `MIGRATION_GUIDE.md` - If coming from TensorFlow
3. See `API_REFERENCE.md` - For detailed API info
4. Try `/examples/` - Runnable code examples

**For Documentation Developers:**
- User-facing docs go in `docs/`
- Planning/internal docs go in `planning/`
- Keep project root minimal

## 🔗 Related Directories

- `/examples/` - Runnable example scripts
- `/tests/` - Test suite
- `/planning/` - Internal planning and tracking docs
- `/deepobs/pytorch/` - PyTorch implementation source code

---

**Last Updated**: 2025-12-15
