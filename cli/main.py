import typer
from injestion.file_processor import process_file
from app.Query_Handler import handle_query
import asyncio
app = typer.Typer()

@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to the .txt file"),
    metadata: str | None = typer.Option(
        None,
        "--metadata",
        "-m",
        help="Metadata for the document",
    ),
):
    """
    THis Functions arguments are:
    path: Path to the .txt file
    metadata: Metadata for the document
    """
    try:
        print("Processing file...")
        process_file(path, metadata)
        typer.echo("File processed successfully!")
    except Exception as e:
        typer.echo(f"An error occurred: {e}")



@app.command()
def ask():
    asyncio.run(_ask_loop())

async def _ask_loop():
    typer.echo("Starting the interactive Query Mode..., type '\\q' to exit")

    while True:
        query = typer.prompt(">>")
        if query.lower() == "\\q":
            break

        response = await handle_query(query)
        typer.echo(response['response'])







if __name__ == "__main__":
    app()