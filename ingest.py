#!/usr/bin/env python
"""
Document ingestion script — indexes documents into ChromaDB vector store.

Usage:
    python ingest.py                          # Ingest sample docs
    python ingest.py --dir /path/to/docs      # Ingest custom directory
    python ingest.py --file /path/to/file.pdf # Ingest single file
    python ingest.py --reset                  # Reset vector store first
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

from core.config import config
from core.document_loader import load_directory, load_file
from core.vector_store import get_vector_store

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG vector store")
    parser.add_argument("--dir", type=str, help="Directory of documents to ingest")
    parser.add_argument("--file", type=str, help="Single file to ingest")
    parser.add_argument("--reset", action="store_true", help="Reset vector store before ingesting")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]🚀 Multi-Agent RAG — Document Ingestion[/bold cyan]\n"
        f"[dim]Vector Store: {config.CHROMA_PERSIST_DIR} | Collection: {config.CHROMA_COLLECTION}[/dim]",
        border_style="cyan",
    ))

    store = get_vector_store()

    if args.reset:
        console.print("[yellow]⚠️  Resetting vector store...[/yellow]")
        store.reset()
        console.print("[green]✓ Vector store reset[/green]")

    # Determine source
    if args.file:
        source = Path(args.file)
        if not source.exists():
            console.print(f"[red]Error: File not found: {source}[/red]")
            sys.exit(1)
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
            task = p.add_task(f"Loading {source.name}...", total=None)
            docs = load_file(source)
            p.update(task, completed=True, description=f"Loaded {len(docs)} chunks")
    elif args.dir:
        source_dir = Path(args.dir)
        if not source_dir.exists():
            console.print(f"[red]Error: Directory not found: {source_dir}[/red]")
            sys.exit(1)
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
            task = p.add_task(f"Loading directory: {source_dir}...", total=None)
            docs = load_directory(source_dir)
            p.update(task, completed=True, description=f"Loaded {len(docs)} chunks")
    else:
        # Default: sample docs
        sample_dir = Path(__file__).parent / "data" / "sample_docs"
        console.print(f"[cyan]No source specified — using sample docs: {sample_dir}[/cyan]")
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
            task = p.add_task("Loading sample documents...", total=None)
            docs = load_directory(sample_dir)
            p.update(task, completed=True, description=f"Loaded {len(docs)} chunks")

    if not docs:
        console.print("[red]No documents loaded — nothing to ingest.[/red]")
        sys.exit(1)

    # Ingest
    console.print(f"\n[bold]Ingesting {len(docs)} chunks into vector store...[/bold]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        task = p.add_task("Embedding and storing...", total=None)
        added = store.add_documents(docs)
        p.update(task, completed=True, description=f"Ingested {added} chunks")

    # Summary table
    table = Table(title="Ingestion Summary", border_style="green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Documents processed", str(len(docs)))
    table.add_row("Chunks stored", str(added))
    table.add_row("Total in store", str(store.count()))
    table.add_row("Embedding model", config.EMBEDDING_MODEL)
    table.add_row("Collection", config.CHROMA_COLLECTION)
    console.print(table)
    console.print("\n[bold green]✅ Ingestion complete! Run `streamlit run ui/app.py` to query.[/bold green]")


if __name__ == "__main__":
    main()
