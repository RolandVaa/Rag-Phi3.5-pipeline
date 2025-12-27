from __future__ import annotations
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from .store import HNSWStore

def load_embedder(name: str = "BAAI/bge-small-en-v1.5") -> SentenceTransformer:
    return SentenceTransformer(name)

def retrieve(index_dir: Path, question: str, k: int = 5, embed_model: str = "BAAI/bge-small-en-v1.5"):
    embed = load_embedder(embed_model)
    dim = embed.get_sentence_embedding_dimension()
    store = HNSWStore.load(index_dir, dim=dim, space="cosine")

    qvec = embed.encode([question], normalize_embeddings=True)
    qvec = np.asarray(qvec, dtype=np.float32)

    ids, dists = store.search(qvec, k=k)
    chunks = [store.meta[i] for i in ids]
    return chunks, dists
