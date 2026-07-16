"""Centralized, incremental ERCOT RAG ingestion and retrieval.

Importing this package performs no document parsing, embedding, or network I/O.
Use :class:`IngestionPipeline` for updates and :mod:`.retrieval` from chatbot
applications.
"""

from .config import Collection, IngestionConfig, SourceRoot, default_config
from .pipeline import IngestionPipeline

__all__ = [
    "Collection",
    "IngestionConfig",
    "IngestionPipeline",
    "SourceRoot",
    "default_config",
]

