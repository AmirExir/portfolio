"""Centralized, incremental ERCOT RAG ingestion and retrieval.

Importing this package performs no document parsing, embedding, or network I/O.
Use :class:`IngestionPipeline` for updates and :mod:`.retrieval` from chatbot
applications.
"""

from .config import Collection, IngestionConfig, SourceRoot, default_config
from .pipeline import IngestionPipeline
from .startup import (
    CentralIndexUnavailable,
    ensure_central_generation,
    load_startup_index,
    startup_index_state,
)

__all__ = [
    "Collection",
    "IngestionConfig",
    "IngestionPipeline",
    "CentralIndexUnavailable",
    "ensure_central_generation",
    "load_startup_index",
    "startup_index_state",
    "SourceRoot",
    "default_config",
]
