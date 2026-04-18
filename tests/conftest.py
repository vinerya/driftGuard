from __future__ import annotations

import numpy as np
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Deterministic fake embeddings for testing.

    Known texts get predefined vectors. Unknown texts get a
    deterministic hash-based random vector.
    """

    def __init__(self, mapping: dict[str, list[float]] | None = None, dim: int = 8):
        self._mapping = mapping or {}
        self._dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if text in self._mapping:
            return self._mapping[text]
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        return rng.randn(self._dim).tolist()
