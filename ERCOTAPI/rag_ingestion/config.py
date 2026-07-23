"""Repository-relative configuration for ERCOT ingestion."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable


class Collection(str, Enum):
    """Stable collection names consumed by ERCOT chatbots."""

    GENERAL = "general"
    PLANNING = "planning"
    PROTOCOLS = "protocols"
    OPERATIONS = "operations"
    RESOURCE_INTEGRATION = "resource_integration"
    DWG_SSWG = "dwg_sswg"
    MARKET = "market"
    NEWS = "news"


SUPPORTED_EXTENSIONS = frozenset({".pdf", ".txt", ".html", ".htm", ".docx", ".csv", ".xlsx"})
SIDECAR_SUFFIX = ".metadata.json"


@dataclass(frozen=True)
class SourceRoot:
    """A directory containing documents with a common trust level."""

    name: str
    path: Path
    source_authority: str
    is_generated: bool
    default_source_kind: str
    default_collections: tuple[str, ...]


@dataclass(frozen=True)
class IngestionConfig:
    """All filesystem and embedding settings needed by the pipeline."""

    repo_root: Path
    index_dir: Path
    source_roots: tuple[SourceRoot, ...]
    embedding_model: str = "text-embedding-3-large"
    chunk_size: int = 7_600
    chunk_overlap: int = 400
    max_file_bytes: int = 100 * 1024 * 1024
    generation_retention: int = 10
    legacy_chunks_path: Path | None = None
    legacy_embeddings_path: Path | None = None
    legacy_sources_dir: Path | None = None
    legacy_embedding_model: str = "text-embedding-3-large"
    legacy_chunk_size: int = 7_600
    legacy_chunk_overlap: int = 400
    ignored_names: frozenset[str] = field(
        default_factory=lambda: frozenset({".DS_Store", "Thumbs.db", "requirements.txt"})
    )

    def with_source_roots(self, roots: Iterable[SourceRoot]) -> "IngestionConfig":
        """Return a copy with custom roots, useful for deployments and tests."""

        return IngestionConfig(
            repo_root=self.repo_root,
            index_dir=self.index_dir,
            source_roots=tuple(roots),
            embedding_model=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            max_file_bytes=self.max_file_bytes,
            generation_retention=self.generation_retention,
            legacy_chunks_path=self.legacy_chunks_path,
            legacy_embeddings_path=self.legacy_embeddings_path,
            legacy_sources_dir=self.legacy_sources_dir,
            legacy_embedding_model=self.legacy_embedding_model,
            legacy_chunk_size=self.legacy_chunk_size,
            legacy_chunk_overlap=self.legacy_chunk_overlap,
            ignored_names=self.ignored_names,
        )


def repository_root() -> Path:
    """Return the repository root without relying on the process cwd."""

    return Path(__file__).resolve().parents[2]


def _has_complete_generation(store: Path) -> bool:
    """Return whether a local or packaged store has a readable generation layout."""

    pointer = store / "CURRENT"
    try:
        generation_id = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    if not generation_id or Path(generation_id).name != generation_id:
        return False
    generation = store / "generations" / generation_id
    has_chunks = (generation / "chunks.json").is_file() or (
        generation / "chunks.json.gz"
    ).is_file()
    has_embeddings = (generation / "embeddings.npy").is_file() or (
        generation / "embeddings.npy.gz"
    ).is_file()
    return (generation / "manifest.json").is_file() and has_chunks and has_embeddings


def default_config(
    *,
    repo_root: Path | None = None,
    index_dir: Path | None = None,
) -> IngestionConfig:
    """Build the default configuration, honoring deployment environment overrides."""

    root = (repo_root or repository_root()).resolve()
    configured_index = (
        os.getenv("ERCOT_RAG_STORE", "").strip()
        or os.getenv("ERCOT_RAG_INDEX_DIR", "").strip()
    )
    local_store = root / "ERCOTAPI" / ".rag_store"
    packaged_store = root / "ERCOTAPI" / "deployment_rag_store"
    if index_dir is not None:
        destination = index_dir
    elif configured_index:
        destination = Path(configured_index).expanduser()
    elif _has_complete_generation(local_store):
        destination = local_store
    elif _has_complete_generation(packaged_store):
        destination = packaged_store
    else:
        destination = local_store

    sources = (
        SourceRoot(
            name="authoritative_static",
            path=root / "chatbot_ercot_all_in_one" / "ercot_sources",
            source_authority="ERCOT",
            is_generated=False,
            default_source_kind="ERCOT Reference",
            default_collections=(Collection.GENERAL.value,),
        ),
        SourceRoot(
            name="official_downloads",
            path=root / "ERCOTAPI" / "NEWS" / "official",
            source_authority="ERCOT",
            is_generated=False,
            default_source_kind="Official Document",
            default_collections=(Collection.GENERAL.value,),
        ),
        SourceRoot(
            name="planning_guide_uploads",
            path=root / "ERCOTAPI" / "sources" / "official" / "planning_guides",
            source_authority="ERCOT",
            is_generated=False,
            default_source_kind="Planning Guide",
            default_collections=(Collection.GENERAL.value, Collection.PLANNING.value),
        ),
        SourceRoot(
            name="nodal_protocol_uploads",
            path=root / "ERCOTAPI" / "sources" / "official" / "nodal_protocols",
            source_authority="ERCOT",
            is_generated=False,
            default_source_kind="Protocol",
            default_collections=(Collection.GENERAL.value, Collection.PROTOCOLS.value),
        ),
        SourceRoot(
            name="dwg_sswg_uploads",
            path=root / "ERCOTAPI" / "sources" / "official" / "dwg_sswg",
            source_authority="ERCOT",
            is_generated=False,
            default_source_kind="DWG",
            default_collections=(Collection.GENERAL.value, Collection.DWG_SSWG.value),
        ),
        SourceRoot(
            name="market_document_uploads",
            path=root / "ERCOTAPI" / "sources" / "official" / "market",
            source_authority="ERCOT",
            is_generated=False,
            default_source_kind="ERCOT Report",
            default_collections=(Collection.GENERAL.value, Collection.MARKET.value),
        ),
    )

    return IngestionConfig(
        repo_root=root,
        index_dir=destination.resolve(),
        source_roots=sources,
        embedding_model=os.getenv("ERCOT_RAG_EMBEDDING_MODEL", "text-embedding-3-large").strip()
        or "text-embedding-3-large",
        chunk_size=max(500, int(os.getenv("ERCOT_RAG_CHUNK_SIZE", "7600"))),
        chunk_overlap=max(0, int(os.getenv("ERCOT_RAG_CHUNK_OVERLAP", "400"))),
        max_file_bytes=max(1, int(os.getenv("ERCOT_RAG_MAX_FILE_BYTES", str(100 * 1024 * 1024)))),
        generation_retention=max(2, int(os.getenv("ERCOT_RAG_KEEP_GENERATIONS", "10"))),
        legacy_chunks_path=root / "chatbot_ercot_all_in_one" / "ercot_chunks_cached.json",
        legacy_embeddings_path=root / "chatbot_ercot_all_in_one" / "ercot_embeddings.npy",
        legacy_sources_dir=root / "chatbot_ercot_all_in_one" / "ercot_sources",
    )
