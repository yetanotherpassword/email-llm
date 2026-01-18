"""Command-line interface for email-llm."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .config import settings
from .embeddings import VectorStore
from .indexer import EmailIndexer
from .llm_client import LLMClient
from .rag_engine import RAGEngine
from .yolo_analyzer import get_available_object_classes

app = typer.Typer(
    name="email-llm",
    help="Natural language search for your Thunderbird emails using local LLMs",
)
console = Console()


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


@app.command()
def index(
    profile_path: Optional[Path] = typer.Option(
        None,
        "--profile",
        "-p",
        help="Path to Thunderbird profile directory",
    ),
    reindex: bool = typer.Option(
        False,
        "--reindex",
        "-r",
        help="Clear and rebuild the entire index",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        "-f",
        help="Re-process all emails (don't skip already indexed ones)",
    ),
    max_emails: Optional[int] = typer.Option(
        None,
        "--max",
        "-m",
        help="Maximum number of emails to index (for testing)",
    ),
    process_images: bool = typer.Option(
        True,
        "--images/--no-images",
        help="Process images with YOLO detection",
    ),
    use_vision: bool = typer.Option(
        False,
        "--vision",
        "-v",
        help="Use vision model for deep image analysis (slower)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging",
    ),
):
    """Index your Thunderbird emails for search.

    By default, only indexes NEW emails (incremental mode).
    Use --reindex to clear and rebuild everything.
    Use --full to re-process all emails without clearing.
    """
    setup_logging(verbose)

    # Determine mode
    if reindex:
        mode = "Full reindex (clearing existing)"
    elif full:
        mode = "Full scan (keeping existing)"
    else:
        mode = "Incremental (new emails only)"

    console.print(Panel.fit(
        "[bold blue]Email-LLM Indexer[/bold blue]\n"
        "Indexing your Thunderbird emails for natural language search",
        border_style="blue",
    ))

    path = profile_path or settings.thunderbird_profile_path

    console.print(f"\n[dim]Profile path:[/dim] {path}")
    console.print(f"[dim]Mode:[/dim] {mode}")
    console.print(f"[dim]Process images:[/dim] {process_images}")
    console.print(f"[dim]Use vision model:[/dim] {use_vision}")

    if reindex:
        console.print("\n[yellow]Warning: This will clear the existing index![/yellow]")
        if not typer.confirm("Continue?"):
            raise typer.Abort()

    try:
        indexer = EmailIndexer(
            profile_path=path,
            process_images=process_images,
            use_vision_model=use_vision,
        )

        console.print("\n[bold]Starting indexing...[/bold]\n")
        # incremental=True by default, unless --full is specified
        stats = indexer.index_all(
            reindex=reindex,
            max_emails=max_emails,
            incremental=not full and not reindex,
        )

        # Display results
        table = Table(title="Indexing Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("New Emails Indexed", str(stats["emails_processed"]))
        table.add_row("Emails Skipped (already indexed)", str(stats.get("emails_skipped", 0)))
        table.add_row("Emails Failed", str(stats["emails_failed"]))
        table.add_row("Attachments Processed", str(stats["attachments_processed"]))
        table.add_row("Images Analyzed", str(stats["images_analyzed"]))
        table.add_row("Search Chunks Created", str(stats["chunks_created"]))

        console.print(table)

        if stats["emails_processed"] == 0 and stats.get("emails_skipped", 0) > 0:
            console.print("\n[green]No new emails to index. Everything is up to date![/green]")
        else:
            console.print("\n[green]Indexing complete![/green]")

    except FileNotFoundError as e:
        console.print(f"\n[red]Error:[/red] {e}")
        console.print(
            "\n[dim]Hint: Check your Thunderbird profile path. "
            "For Snap installations, try:[/dim]\n"
            "  ~/snap/thunderbird/common/.thunderbird/"
        )
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Error during indexing:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results to show"),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        help="Skip LLM answer generation, show only search results",
    ),
    backend: str = typer.Option(
        "lmstudio",
        "--backend",
        "-b",
        help="LLM backend: 'lmstudio' or 'ollama'",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging"),
):
    """Search your emails using natural language."""
    setup_logging(verbose)

    async def do_search():
        llm = LLMClient(backend=backend)
        rag = RAGEngine(llm_client=llm)

        # Check LLM connection if using it
        if not no_llm:
            connected = await llm.check_connection()
            if not connected:
                console.print(
                    f"[yellow]Warning: Could not connect to {backend}. "
                    "Showing search results only.[/yellow]\n"
                )
                no_llm_local = True
            else:
                no_llm_local = False
        else:
            no_llm_local = True

        console.print(f"\n[bold]Searching for:[/bold] {query}\n")

        results = await rag.search(query, limit=limit, use_llm=not no_llm_local)

        # Display answer if available
        if "answer" in results and results["answer"]:
            console.print(Panel(
                Markdown(results["answer"]),
                title="[bold green]Answer[/bold green]",
                border_style="green",
            ))
            console.print()

        # Display search results
        if results["results"]:
            table = Table(title=f"Search Results ({results['total_found']} found)")
            table.add_column("#", style="dim", width=3)
            table.add_column("Score", style="cyan", width=6)
            table.add_column("From", style="green", width=25)
            table.add_column("Subject", style="white", width=40)
            table.add_column("Date", style="dim", width=12)

            for i, result in enumerate(results["results"], 1):
                table.add_row(
                    str(i),
                    f"{result['score']:.2f}",
                    result["from_address"][:25] if result["from_address"] else "-",
                    result["subject"][:40] if result["subject"] else "-",
                    result["date"][:10] if result["date"] else "-",
                )

            console.print(table)

            # Show matched text for top result
            if results["results"]:
                top = results["results"][0]
                console.print(Panel(
                    top["matched_text"][:500] + "..." if len(top["matched_text"]) > 500 else top["matched_text"],
                    title="[bold]Top Match Preview[/bold]",
                    border_style="dim",
                ))
        else:
            console.print("[yellow]No results found.[/yellow]")

    asyncio.run(do_search())


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask about your emails"),
    backend: str = typer.Option(
        "lmstudio",
        "--backend",
        "-b",
        help="LLM backend: 'lmstudio' or 'ollama'",
    ),
):
    """Ask a question about your emails and get a direct answer."""
    async def do_ask():
        llm = LLMClient(backend=backend)
        rag = RAGEngine(llm_client=llm)

        console.print(f"\n[bold]Question:[/bold] {question}\n")

        with console.status("[bold green]Thinking..."):
            answer = await rag.ask(question)

        console.print(Panel(
            Markdown(answer),
            title="[bold green]Answer[/bold green]",
            border_style="green",
        ))

    asyncio.run(do_ask())


@app.command()
def stats():
    """Show index statistics."""
    try:
        vs = VectorStore()
        stats = {
            "total_chunks": vs.count(),
            "collection": vs.collection_name,
            "chroma_path": str(settings.get_chroma_path()),
        }

        table = Table(title="Index Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in stats.items():
            table.add_row(key.replace("_", " ").title(), str(value))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@app.command()
def objects():
    """List objects that YOLO can detect for image filtering."""
    classes = get_available_object_classes()

    table = Table(title="Detectable Objects (YOLO)")
    table.add_column("#", style="dim", width=4)
    table.add_column("Object", style="cyan")

    for i, cls in enumerate(classes, 1):
        table.add_row(str(i), cls)

    console.print(table)
    console.print(f"\n[dim]Total: {len(classes)} object classes[/dim]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start the web UI server."""
    import uvicorn

    console.print(Panel.fit(
        f"[bold blue]Email-LLM Web UI[/bold blue]\n"
        f"Starting server at http://{host}:{port}",
        border_style="blue",
    ))

    uvicorn.run(
        "src.web:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def check():
    """Check system configuration and dependencies."""
    console.print("[bold]Checking system configuration...[/bold]\n")

    checks = []

    # Check Thunderbird profile
    tb_path = settings.thunderbird_profile_path
    if tb_path.exists():
        checks.append(("Thunderbird profile", "Found", "green", str(tb_path)))
    else:
        checks.append(("Thunderbird profile", "Not found", "red", str(tb_path)))

    # Check data directory
    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    checks.append(("Data directory", "OK", "green", str(data_dir)))

    # Check ChromaDB
    try:
        vs = VectorStore()
        checks.append(("ChromaDB", "OK", "green", f"{vs.count()} chunks"))
    except Exception as e:
        checks.append(("ChromaDB", "Error", "red", str(e)))

    # Check LM Studio
    async def check_lm():
        llm = LLMClient(backend="lmstudio")
        return await llm.check_connection()

    try:
        lm_ok = asyncio.run(check_lm())
        if lm_ok:
            checks.append(("LM Studio", "Connected", "green", settings.lmstudio_base_url))
        else:
            checks.append(("LM Studio", "Not running", "yellow", settings.lmstudio_base_url))
    except Exception as e:
        checks.append(("LM Studio", "Error", "red", str(e)))

    # Check Ollama
    async def check_ollama():
        llm = LLMClient(backend="ollama")
        return await llm.check_connection()

    try:
        ollama_ok = asyncio.run(check_ollama())
        if ollama_ok:
            checks.append(("Ollama", "Connected", "green", settings.ollama_base_url))
        else:
            checks.append(("Ollama", "Not running", "yellow", settings.ollama_base_url))
    except Exception as e:
        checks.append(("Ollama", "Error", "red", str(e)))

    # Check Tesseract
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        checks.append(("Tesseract OCR", "Installed", "green", ""))
    except Exception:
        checks.append(("Tesseract OCR", "Not found", "yellow", "Install with: sudo apt install tesseract-ocr"))

    # Display results
    table = Table(title="System Check")
    table.add_column("Component", style="cyan")
    table.add_column("Status")
    table.add_column("Details", style="dim")

    for name, status, color, details in checks:
        table.add_row(name, f"[{color}]{status}[/{color}]", details)

    console.print(table)


if __name__ == "__main__":
    app()
