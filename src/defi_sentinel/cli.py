"""CLI entry point for DeFi Sentinel."""

from __future__ import annotations

import click
import logging
import yaml
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
logger = logging.getLogger("defi_sentinel")

BANNER = r"""
[bold cyan]
 ╔══════════════════════════════════════╗
 ║  🛡️  DeFi Sentinel v0.1.0           ║
 ║  AI-Powered DeFi Agent              ║
 ╚══════════════════════════════════════╝
[/bold cyan]
"""


def load_config(path: str = "config.yaml") -> dict:
    """Load config from YAML file."""
    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


@click.group()
@click.option("--config", "-c", default="config.yaml", help="Config file path")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
@click.pass_context
def cli(ctx: click.Context, config: str, verbose: bool) -> None:
    """DeFi Sentinel — AI-powered DeFi monitoring agent."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(name)s] %(message)s")
    console.print(BANNER)


@cli.command()
@click.option("--chain", "-c", multiple=True, default=["ethereum"], help="Chains to monitor")
@click.option("--protocols", "-p", default="aave,uniswap", help="Comma-separated protocols")
@click.pass_context
def monitor(ctx: click.Context, chain: tuple[str, ...], protocols: str) -> None:
    """Start on-chain monitoring."""
    from defi_sentinel.agent import SentinelAgent, AgentConfig

    config = AgentConfig(chains=list(chain))
    agent = SentinelAgent(config)
    agent.initialize()

    console.print(Panel(
        f"[green]Monitoring started[/green]\n"
        f"Chains: {', '.join(chain)}\n"
        f"Protocols: {protocols}",
        title="🛡️ Sentinel Monitor",
    ))

    # In production, this would be an async event loop
    console.print("[dim]Press Ctrl+C to stop...[/dim]")


@cli.command()
@click.argument("query")
@click.option("--chain", "-c", multiple=True, default=["ethereum"])
@click.option("--simulate/--no-simulate", default=True, help="Simulate before executing")
@click.pass_context
def analyze(ctx: click.Context, query: str, chain: tuple[str, ...], simulate: bool) -> None:
    """Run AI analysis on on-chain data."""
    from defi_sentinel.agent import SentinelAgent, AgentConfig

    config = AgentConfig(chains=list(chain), simulate_before_execute=simulate)
    agent = SentinelAgent(config)
    agent.initialize()

    console.print(f"[dim]Analyzing:[/dim] {query}")
    result = agent.run(query)

    console.print(Panel(result["response"], title="🧠 Analysis"))

    if result["actions"]:
        table = Table(title="Actions Taken")
        table.add_column("Action", style="cyan")
        table.add_column("Status", style="green")
        for action in result["actions"]:
            table.add_row(str(action.get("type", "?")), str(action.get("status", "?")))
        console.print(table)


@cli.command()
@click.option("--backend", "-b", type=click.Choice(["cuda", "rocm", "cpu", "auto"]), default="auto")
@click.option("--model", "-m", default="TheBloke/Llama-2-7B-Chat-GPTQ", help="Model name/path")
@click.option("--quant", "-q", type=click.Choice(["fp16", "gptq-4bit", "awq-4bit", "gguf-q4"]), default="gptq-4bit")
@click.option("--iterations", "-n", default=50, help="Benchmark iterations")
@click.pass_context
def benchmark(ctx: click.Context, backend: str, model: str, quant: str, iterations: int) -> None:
    """Run GPU inference benchmark."""
    from defi_sentinel.inference.benchmark import run_benchmark

    console.print(Panel(
        f"Backend: {backend}\nModel: {model}\nQuantization: {quant}\nIterations: {iterations}",
        title="⚡ GPU Benchmark",
    ))

    results = run_benchmark(model=model, backend=backend, quantization=quant, iterations=iterations)

    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for key, value in results.items():
        table.add_row(key, str(value))
    console.print(table)


@cli.command()
@click.pass_context
def chains(ctx: click.Context) -> None:
    """List supported chains."""
    from defi_sentinel.chains.registry import ChainRegistry

    registry = ChainRegistry()

    table = Table(title="Supported Chains")
    table.add_column("Chain", style="cyan")
    table.add_column("Chain ID", style="yellow")
    table.add_column("Status", style="green")

    for name, chain in registry.list_all().items():
        table.add_row(name, str(chain.chain_id), "✅ Active")

    console.print(table)


def main() -> None:
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
