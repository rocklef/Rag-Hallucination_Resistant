#!/usr/bin/env python
"""
CLI entry point for the Multi-Agent RAG system.

Usage:
    python main.py --query "What is RAG?"
    python main.py --query "How does hallucination happen?" --use-graph
    python main.py --interactive
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import print as rprint

from core.config import config

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format="%(levelname)s: %(message)s")
console = Console()


def print_result(result_dict: dict) -> None:
    """Pretty-print the RAG result."""
    # Header
    console.print(Panel.fit(
        f"[bold]Query:[/bold] {result_dict['query']}",
        border_style="blue",
    ))

    # Answer
    console.print(Panel(
        Markdown(result_dict["answer"]),
        title="[bold green]Answer[/bold green]",
        border_style="green",
    ))

    # Metrics table
    table = Table(title="Pipeline Metrics", border_style="cyan", show_header=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    conf = result_dict.get("hallucination_confidence", 0)
    verdict = result_dict.get("hallucination_verdict", "?")
    if conf >= 0.80:
        conf_display = f"[green]{conf:.2f} 🟢 HIGH[/green]"
    elif conf >= 0.55:
        conf_display = f"[yellow]{conf:.2f} 🟡 MEDIUM[/yellow]"
    else:
        conf_display = f"[red]{conf:.2f} 🔴 LOW[/red]"

    table.add_row("Sub-queries", str(len(result_dict.get("sub_queries", []))))
    table.add_row("Chunks retrieved", str(result_dict.get("retrieved_chunks", 0)))
    table.add_row("Chunks after filter", str(result_dict.get("filtered_chunks", 0)))
    table.add_row("Consistency", result_dict.get("consistency", "?"))
    table.add_row("Hallucination verdict", verdict)
    table.add_row("Confidence score", conf_display)
    table.add_row("Retries", str(result_dict.get("retry_count", 0)))
    table.add_row("Time", f"{result_dict.get('elapsed_seconds', 0):.2f}s")
    console.print(table)

    # Sources
    sources = result_dict.get("sources", [])
    if sources:
        console.print("[bold cyan]Sources:[/bold cyan]")
        for s in sources:
            console.print(f"  • {s}")

    # Agent trace
    trace = result_dict.get("agent_trace", [])
    if trace:
        console.print("\n[bold dim]Agent Trace:[/bold dim]")
        for step in trace:
            console.print(f"  {step}")


def run_query(query: str, use_graph: bool = False) -> dict:
    """Execute a single query using either the orchestrator or LangGraph."""
    if use_graph:
        from graph.rag_graph import run_graph
        state = run_graph(query, max_retries=config.MAX_RETRIES)
        return {
            "query": state["query"],
            "answer": state["answer"],
            "sources": state.get("sources", []),
            "sub_queries": state.get("sub_queries", []),
            "retrieved_chunks": len(state.get("retrieved_chunks", [])),
            "filtered_chunks": len(state.get("filtered_chunks", [])),
            "consistency": state.get("cross_ref_report", {}).get("consistency", "?"),
            "hallucination_verdict": state.get("hallucination_verdict", "?"),
            "hallucination_confidence": state.get("hallucination_confidence", 0),
            "retry_count": state.get("attempt", 0),
            "elapsed_seconds": 0,
            "agent_trace": state.get("agent_trace", []),
        }
    else:
        from agents.orchestrator import OrchestratorAgent
        orchestrator = OrchestratorAgent()
        result = orchestrator.run(query)
        return result.to_dict()


def interactive_mode(use_graph: bool = False) -> None:
    """REPL-style interactive querying."""
    console.print(Panel.fit(
        "[bold cyan]🤖 Multi-Agent Hallucination-Resistant RAG[/bold cyan]\n"
        "[dim]Type your question and press Enter. Type 'exit' to quit.[/dim]",
        border_style="cyan",
    ))
    while True:
        try:
            query = console.input("\n[bold yellow]Query:[/bold yellow] ").strip()
            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                console.print("[dim]Goodbye![/dim]")
                break
            result = run_query(query, use_graph=use_graph)
            print_result(result)
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Hallucination-Resistant RAG System"
    )
    parser.add_argument("--query", "-q", type=str, help="Single query to run")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--use-graph", action="store_true", help="Use LangGraph instead of orchestrator")
    args = parser.parse_args()

    if args.query:
        result = run_query(args.query, use_graph=args.use_graph)
        print_result(result)
    elif args.interactive:
        interactive_mode(use_graph=args.use_graph)
    else:
        parser.print_help()
        console.print("\n[yellow]Tip: Run `python ingest.py` first to load documents.[/yellow]")


if __name__ == "__main__":
    main()
