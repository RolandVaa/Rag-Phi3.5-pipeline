from __future__ import annotations

from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from .chunking import chunk_text
from .store import HNSWStore, DocChunk

# File types we allow users to place into the knowledge-base folder.
SUPPORTED = {".txt", ".md", ".pdf"}


def read_file(path: Path) -> str:
    """Read a supported file and return extracted text."""
    suf = path.suffix.lower()

    # Plain text / markdown: just read the file contents.
    if suf in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    # PDF: extract text from each page and concatenate.
    # (Some pages may return None; we treat those as empty strings.)
    if suf == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in reader.pages)

    raise ValueError(f"Unsupported file type: {path}")


def ingest(docs_dir: Path, out_dir: Path, embed_model: str = "BAAI/bge-small-en-v1.5"):
    """
    Build a vector index from documents under docs_dir.

    Pipeline:
      1) Find supported files
      2) Read and chunk each document
      3) Embed each chunk with a sentence-transformer model
      4) Store vectors + metadata in an HNSW index on disk (out_dir)
    """
    docs_dir = docs_dir.resolve()
    out_dir = out_dir.resolve()

    # Recursively gather all supported files.
    files = [p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED]
    if not files:
        raise RuntimeError(f"No documents found in {docs_dir} (supported: {sorted(SUPPORTED)})")

    # Embedding model used for retrieval (not the LLM).
    embed = SentenceTransformer(embed_model)
    dim = embed.get_sentence_embedding_dimension()

    # We keep (a) chunk metadata and (b) raw chunk text for embedding.
    chunks: list[DocChunk] = []
    texts: list[str] = []
    cid = 0  # global chunk id across all files

    for fp in files:
        text = read_file(fp)

        # Split the document into overlapping chunks for better retrieval.
        for c in chunk_text(text):
            c = c.strip()
            if not c:
                continue

            chunks.append(DocChunk(chunk_id=cid, source=str(fp), text=c))
            texts.append(c)
            cid += 1

    # Create embeddings (unit-normalized so cosine similarity works well).
    vectors = embed.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)

    # Build the ANN index and persist it to disk.
    store = HNSWStore(dim=dim, space="cosine")
    store.init(max_elements=len(chunks))
    ids = np.arange(len(chunks), dtype=np.int64)
    store.add(ids, vectors, chunks)
    store.save(out_dir)

    print(f"Ingested {len(chunks)} chunks from {len(files)} files into: {out_dir}")
