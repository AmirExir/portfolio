"""Embedding providers; OpenAI is imported and contacted only on demand."""

from __future__ import annotations

import os
from typing import Any, Protocol, Sequence


class EmbeddingProvider(Protocol):
    """Small dependency-injection surface used by the ingestion pipeline."""

    model: str

    def embed_texts(self, texts: Sequence[str]) -> Any:
        """Return a two-dimensional numeric array with one row per text."""


class OpenAIEmbedder:
    """Lazy OpenAI embedding provider with bounded request batches."""

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        *,
        api_key: str | None = None,
        batch_size: int = 64,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._batch_size = max(1, batch_size)
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = self._api_key or os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required to embed changed ERCOT documents")
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OpenAI embedding requires the `openai` package") from exc
        # A cold central build can legitimately cross the embeddings TPM
        # window even though the account has quota.  Let the SDK honor the
        # server's retry delay instead of turning a temporary rate limit into
        # a partial generation.
        max_retries = max(2, int(os.getenv("ERCOT_RAG_OPENAI_MAX_RETRIES", "10")))
        self._client = OpenAI(api_key=api_key, max_retries=max_retries)
        return self._client

    def embed_texts(self, texts: Sequence[str]) -> Any:
        if not texts:
            try:
                import numpy as np
            except ImportError as exc:  # pragma: no cover - dependency failure
                raise RuntimeError("Embedding storage requires the `numpy` package") from exc
            return np.empty((0, 0), dtype="float32")

        client = self._get_client()
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), self._batch_size):
            batch = list(texts[offset : offset + self._batch_size])
            response = client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(response.data, key=lambda item: getattr(item, "index", 0))
            if len(ordered) != len(batch):
                raise RuntimeError(
                    f"Embedding service returned {len(ordered)} vectors for {len(batch)} chunks"
                )
            vectors.extend(item.embedding for item in ordered)
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - dependency failure
            raise RuntimeError("Embedding storage requires the `numpy` package") from exc
        array = np.asarray(vectors, dtype="float32")
        if array.ndim != 2 or array.shape[0] != len(texts):
            raise RuntimeError("Embedding provider returned an invalid array shape")
        return array


def provider_model(provider: EmbeddingProvider, fallback: str) -> str:
    value = str(getattr(provider, "model", "") or "").strip()
    return value or fallback
