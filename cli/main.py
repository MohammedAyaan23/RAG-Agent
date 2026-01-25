import typer
from injestion.file_processor import process_file
from app.Query_Handler import handle_query

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
        process_file(path, metadata)
        typer.echo("File processed successfully!")
    except Exception as e:
        typer.echo(f"An error occurred: {e}")



@app.command()
def ask():
    """Start the interactive CLI Seesion"""
    typer.echo("Starting the interactive Query Mode..., type '\q' to exit")

    while True:
        query = typer.prompt(">>")
        if query.lower() == "\q":
            typer.echo("Exiting Query mode,Goodbye!")
            break
        
        try:
            print("Processing query...")
            response = handle_query(query)
            answer = response.get("response", "")
            typer.echo(f"\n{answer}\n")
            
            
        except Exception as e:
            typer.echo(f"An error occurred: {e}")
        
        








if __name__ == "__main__":
    app()