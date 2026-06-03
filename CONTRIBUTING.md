# Contributing to DeFi Sentinel

Thanks for your interest in contributing!

## Development Setup

```bash
# Clone
git clone https://github.com/santuy69/defi-sentinel.git
cd defi-sentinel

# Install in dev mode
pip install -e ".[inference,gpu-benchmark]"

# Run tests
pytest tests/ -v
```

## Adding a New Chain

1. Create `src/defi_sentinel/chains/<chain>.py`
2. Implement `ChainAdapter` base class
3. Add preset to `CHAIN_PRESETS` in `evm.py`
4. Add tests in `tests/test_chains.py`

## Adding a New Protocol

1. Create `src/defi_sentinel/protocols/<protocol>.py`
2. Implement standard parser interface
3. Add tests

## Code Style

- Python 3.11+, type hints required
- ruff for linting, 88 char line limit
- Docstrings for all public classes/functions
