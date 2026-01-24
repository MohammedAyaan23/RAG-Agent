import typer

app = typer.Typer()

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
            
        except Exception as e:
            typer.echo(f"An error occurred: {e}")
        
        








if __name__ == "__main__":
    app()