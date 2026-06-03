# DeFi Sentinel 🛡️

> AI-powered DeFi monitoring and automation agent with local LLM inference

DeFi Sentinel is an autonomous agent framework that monitors on-chain activity across multiple EVM chains, analyzes DeFi protocols using local LLM reasoning, and executes strategies based on AI-driven decisions.

Built to run inference workloads on AMD ROCm and NVIDIA CUDA hardware.

## Features

- 🔗 **Multi-chain monitoring** — Ethereum, Arbitrum, Base, Polygon, Optimism, BSC
- 🧠 **Local LLM inference** — Run quantized models (Llama, Mistral, Qwen) for on-chain analysis
- 📊 **DeFi protocol scanner** — Aave, Uniswap, GMX, Pendle, and 50+ protocols
- 🤖 **Autonomous agent loop** — Reason → Plan → Execute → Observe
- ⚡ **GPU-optimized** — ROCm and CUDA backends with automatic quantization
- 🛡️ **Safety rails** — Simulation-first execution, spending limits, kill switches

## Architecture

```
┌─────────────────────────────────────────────┐
│              DeFi Sentinel                   │
├─────────────┬───────────────┬───────────────┤
│  Monitors   │  LLM Engine   │  Executors    │
│  ─────────  │  ───────────  │  ──────────   │
│  Mempool    │  Reasoning    │  Tx Builder   │
│  Events     │  Planning     │  Simulator    │
│  Price Feed │  Analysis     │  Broadcaster  │
├─────────────┴───────────────┴───────────────┤
│              Chain Adapters                  │
│  EVM (web3.py) │ Solana (solders) │ Cosmos   │
└─────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install
pip install -e ".[inference]"

# Configure
cp config.example.yaml config.yaml
# Edit config.yaml with your RPC endpoints and wallet

# Run
sentinel monitor --chain ethereum --protocols aave,uniswap
sentinel analyze --query "What are the highest yield stablecoin opportunities?"
sentinel benchmark --backend rocm --model llama3-8b
```

## GPU Inference

DeFi Sentinel supports local LLM inference for privacy-sensitive DeFi analysis:

```python
from defi_sentinel.inference import LLMEngine

engine = LLMEngine(
    model="TheBloke/Llama-2-7B-Chat-GPTQ",
    backend="rocm",  # or "cuda"
    quantization="gptq-4bit",
)

# Analyze on-chain data with local reasoning
result = engine.analyze(
    prompt="Analyze this transaction pattern for potential MEV extraction",
    context=chain_data,
)
```

### Benchmark Results

| Model | Backend | Quant | Tokens/sec | VRAM |
|-------|---------|-------|------------|------|
| Llama-3-8B | ROCm (MI300X) | FP16 | 85 | 16GB |
| Llama-3-8B | ROCm (MI300X) | GPTQ-4 | 142 | 6GB |
| Mistral-7B | ROCm (MI300X) | GPTQ-4 | 158 | 5GB |
| Qwen2-7B | CUDA (A100) | FP16 | 92 | 14GB |

## Project Structure

```
defi-sentinel/
├── src/defi_sentinel/
│   ├── __init__.py
│   ├── cli.py              # CLI entry point
│   ├── agent.py            # Core agent loop
│   ├── inference/          # LLM inference engine
│   │   ├── engine.py       # Model loading + generation
│   │   ├── quantize.py     # GPTQ/AWQ/GGUF quantization
│   │   └── benchmark.py    # GPU benchmark suite
│   ├── chains/             # Chain adapters
│   │   ├── base.py         # Abstract chain interface
│   │   ├── evm.py          # EVM chain adapter
│   │   └── registry.py     # Chain registry
│   ├── monitors/           # On-chain monitors
│   │   ├── mempool.py      # Pending tx monitor
│   │   ├── events.py       # Event log scanner
│   │   └── price.py        # Price feed aggregator
│   ├── protocols/          # DeFi protocol parsers
│   │   ├── aave.py
│   │   ├── uniswap.py
│   │   └── base.py
│   └── safety/             # Execution safety rails
│       ├── simulator.py    # eth_call simulation
│       ├── limits.py       # Spending/duration limits
│       └── killswitch.py   # Emergency stop
├── tests/
├── config.example.yaml
├── pyproject.toml
└── README.md
```

## Use Cases

1. **Yield Optimization** — Monitor lending rates across Aave, Compound, Morpho and auto-rebalance
2. **MEV Detection** — Analyze mempool patterns for sandwich attacks and arbitrage opportunities
3. **Risk Assessment** — LLM-powered analysis of smart contract interactions before signing
4. **Portfolio Analytics** — Natural language queries over your on-chain portfolio
5. **Alert System** — AI-filtered alerts for whale movements, governance proposals, exploit patterns

## Roadmap

- [x] Multi-chain EVM monitoring
- [x] Local LLM inference (CUDA)
- [ ] ROCm backend optimization
- [ ] Solana chain adapter
- [ ] MEV strategy templates
- [ ] Telegram/Discord alert integration
- [ ] Web dashboard

## License

MIT — see [LICENSE](LICENSE)
