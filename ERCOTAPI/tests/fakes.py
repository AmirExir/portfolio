"""Small deterministic test doubles for embedding operations."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence


class FakeEmbedder:
    """Return deterministic local vectors while recording every requested batch."""

    model = "test-embedding-v1"

    def __init__(self, *, dimension: int = 4) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []
        self.fail_on: str | None = None

    def embed_texts(self, texts: Sequence[str]):
        import numpy as np

        batch = list(texts)
        self.calls.append(batch)
        if self.fail_on and any(self.fail_on in text for text in batch):
            raise RuntimeError(f"synthetic embedding failure for {self.fail_on}")

        vectors: list[list[float]] = []
        for text in batch:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append(
                [
                    (int.from_bytes(digest[offset : offset + 4], "big") + 1)
                    / float(2**32)
                    for offset in range(0, self.dimension * 4, 4)
                ]
            )
        return np.asarray(vectors, dtype="float32")

    @property
    def embedded_text_count(self) -> int:
        return sum(len(batch) for batch in self.calls)

