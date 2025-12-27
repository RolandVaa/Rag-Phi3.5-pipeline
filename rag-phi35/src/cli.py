from pathlib import Path

import typer
from rich import print

from src.rag.ingest import ingest as ingest_fn
from src.rag.query import retrieve
from src.llm.phi_genai import PhiGenAI

# Typer builds a nice CLI with automatic --help, argument parsing, etc.
# We disable shell auto-completion generation for simplicity.
app = typer.Typer(add_completion=False)


@app.command()
def ingest(
    docs_dir: str = typer.Option("kb", help="Folder with .txt/.md/.pdf documents"),
    out_dir: str = typer.Option("data/index", help="Where to write the vector index"),
):
    """
    Build the RAG vector index from documents in docs_dir.

    This reads supported files (txt/md/pdf), chunks them, embeds each chunk,
    and writes an ANN index + metadata to out_dir.
    """
    ingest_fn(Path(docs_dir), Path(out_dir))
    print("[green]Done.[/green]")


@app.command()
def ask(
    question: str = typer.Argument(...),
    provider: str = typer.Option("cpu", help="cpu or cuda"),
    index_dir: str = typer.Option("data/index", help="Vector index folder"),
    k: int = typer.Option(5, help="Top-k chunks"),
    max_new: int = typer.Option(250, help="Max new tokens"),
    model_dir: str = typer.Option("", help="Optional override (auto if empty)"),
):
    """
    Ask a question using RAG:
      1) Retrieve top-k relevant chunks from the index
      2) Build a grounded prompt that includes the retrieved context
      3) Run Phi-3.5 locally (CPU or CUDA) to generate an answer
      4) Print answer + sources + performance metrics
    """
    # Retrieve the most relevant chunks for this question.
    chunks, dists = retrieve(Path(index_dir), question, k=k)

    # Format retrieved chunks into a single context block.
    # Each chunk is labeled with its source file and chunk id for traceability.
    context = "\n\n".join(
        [f"[Source: {c.source} | chunk {c.chunk_id}]\n{c.text}" for c in chunks]
    )

    # Phi chat prompt format (Phi-3.5 uses special role tokens).
    # We explicitly instruct the model to answer ONLY from the provided context.
    prompt = (
        "<|user|>\n"
        "You are a helpful assistant. Answer using ONLY the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "<|end|>\n"
        "<|assistant|>"
    )

    # Load the model (CPU by default). If model_dir == "", PhiGenAI chooses a default
    # path based on the provider (cpu vs cuda).
    llm = PhiGenAI(model_dir=model_dir, provider=provider)

    # Run generation and collect performance metrics.
    out = llm.generate(prompt, max_new=max_new)

    # Print accelerator + model load latency (one-time init cost).
    print(f"device_type={llm.device_type}  model_load_s={llm.model_load_s:.3f}")

    # Print per-request performance metrics.
    print(
        f"prompt_tokens={out['prompt_tokens']}  new_tokens={out['new_tokens']}  "
        f"ttft_s={out['ttft_s']:.6f}  gen_total_s={out['gen_total_s']:.3f}  tok/s={out['tok_per_s']:.2f}"
    )

    # Model answer text.
    print("\n[bold]Answer:[/bold]\n" + out["text"])

    # Retrieval provenance: list the chunks returned by the retriever and their distances.
    # Smaller distance = closer match (since we use cosine distance in the index).
    print("\n[bold]Sources:[/bold]")
    for c, d in zip(chunks, dists):
        print(f"- {c.source} (chunk {c.chunk_id}, dist={d:.4f})")


if __name__ == "__main__":
    # Entry point: `python -m src.cli ...`
    app()
