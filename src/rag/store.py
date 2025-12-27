from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import hnswlib
import numpy as np


@dataclass
class DocChunk:
    """
    Metadata for one chunk stored in the vector index.

    chunk_id: Stable integer ID used as the HNSW label.
    source:   File path (or document identifier) the chunk came from.
    text:     The actual chunk text that will be inserted into the RAG prompt.
    """
    chunk_id: int
    source: str
    text: str


class HNSWStore:
    """
    Thin wrapper around hnswlib to store:
      - an HNSW vector index (fast approximate nearest-neighbor search)
      - chunk metadata (text + source) keyed by chunk_id
    """

    def __init__(self, dim: int, space: str = "cosine"):
        # dim: embedding dimension (must match embedding model output)
        # space: distance metric (cosine works well with normalized embeddings)
        self.dim = dim
        self.space = space
        self.index = hnswlib.Index(space=space, dim=dim)

        # Mapping from integer label -> DocChunk metadata.
        self.meta: dict[int, DocChunk] = {}

    def init(self, max_elements: int, ef_construction: int = 200, M: int = 16):
        """
        Initialize the HNSW index.

        max_elements: maximum number of vectors the index will hold.
        ef_construction/M: quality vs build-time tradeoffs (hnswlib parameters).
        """
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M,
        )

        # ef controls recall/speed at query time. Larger = better recall, slower search.
        self.index.set_ef(50)

    def add(self, ids: np.ndarray, vectors: np.ndarray, chunks: list[DocChunk]):
        """
        Add items to the index and store metadata.

        ids: integer labels for each vector (must align with vectors order).
        vectors: embedding matrix (N x dim).
        chunks: DocChunk metadata objects (their chunk_id should match ids).
        """
        self.index.add_items(vectors, ids)

        # Store metadata for later retrieval (used to build the RAG prompt).
        for c in chunks:
            self.meta[c.chunk_id] = c

    def save(self, dirpath: Path):
        """Persist the vector index + metadata to disk."""
        dirpath.mkdir(parents=True, exist_ok=True)

        # Binary HNSW graph + vectors
        self.index.save_index(str(dirpath / "hnsw.bin"))

        # Human-readable metadata (chunk_id -> {source, text})
        with open(dirpath / "meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {str(k): c.__dict__ for k, c in self.meta.items()},
                f,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, dirpath: Path, dim: int, space: str = "cosine") -> "HNSWStore":
        """Load a previously saved index + metadata."""
        store = cls(dim=dim, space=space)

        store.index.load_index(str(dirpath / "hnsw.bin"))

        with open(dirpath / "meta.json", "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Convert JSON keys back to ints and rebuild DocChunk objects.
        store.meta = {int(k): DocChunk(**v) for k, v in raw.items()}

        # Restore query-time search parameter.
        store.index.set_ef(50)
        return store

    def search(self, query_vec: np.ndarray, k: int = 5):
        """
        Return the top-k nearest chunk ids and their distances.

        query_vec: shape (1, dim) or (dim,) depending on caller usage.
        """
        labels, distances = self.index.knn_query(query_vec, k=k)

        # hnswlib returns arrays of shape (num_queries, k). We use a single query.
        return labels[0].tolist(), distances[0].tolist()
