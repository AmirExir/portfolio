"""Incremental ERCOT ingestion orchestration and CLI."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import unicodedata
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .chunking import build_chunks, chunk_id, document_id, normalize_text
from .classify import authority_rank, classify_document, load_sidecar
from .config import (
    SIDECAR_SUFFIX,
    SUPPORTED_EXTENSIONS,
    IngestionConfig,
    SourceRoot,
    default_config,
)
from .embeddings import EmbeddingProvider, OpenAIEmbedder, provider_model
from .loaders import load_document
from .store import (
    SCHEMA_VERSION,
    load_generation,
    prune_generations,
    update_lock,
    write_generation,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _generation_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("ERCOT RAG ingestion requires the `numpy` package") from exc
    return np


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _mtime_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


METADATA_CHANGE_FIELDS = (
    "title",
    "filename",
    "source_path",
    "source_category",
    "source_authority",
    "source_kind",
    "is_generated",
    "collections",
    "original_url",
    "url_aliases",
    "source_page_urls",
    "provenance",
    "downloaded_at",
    "published_date",
    "effective_date",
    "document_number",
    "document_status",
    "revision",
    "document_type",
    "size",
    "mtime_ns",
    "modification_timestamp",
)


def _metadata_changed(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    return any(previous.get(field) != current.get(field) for field in METADATA_CHANGE_FIELDS)


def _date_rank(value: Any) -> int:
    candidate = str(value or "").strip()
    if not candidate:
        return 0
    iso_candidate = candidate[:10]
    for pattern in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            parsed = datetime.strptime(
                iso_candidate if pattern == "%Y-%m-%d" else candidate,
                pattern,
            )
        except ValueError:
            continue
        return int(parsed.strftime("%Y%m%d"))
    return 0


def _lifecycle_rank(record: Mapping[str, Any]) -> tuple[int, int, int, str]:
    status_rank = {
        "effective": 8,
        "withdrawn": 7,
        "rejected": 7,
        "approved": 6,
        "clean": 5,
        "pending": 4,
        "draft": 3,
        "redline": 2,
    }.get(str(record.get("document_status") or "").strip().lower(), 0)
    lifecycle_date = max(
        _date_rank(record.get("effective_date")),
        _date_rank(record.get("published_date")),
    )
    return (
        lifecycle_date,
        status_rank,
        _date_rank(record.get("downloaded_at")),
        str(record.get("source_path") or ""),
    )


def _newest_value(records: Sequence[Mapping[str, Any]], field: str) -> Any:
    populated = [record.get(field) for record in records if record.get(field)]
    return max(populated, key=lambda value: (_date_rank(value), str(value))) if populated else None


def _legacy_comparable_text(text: str) -> str:
    """Normalize harmless extraction differences before validating a legacy cache."""

    normalized = unicodedata.normalize("NFKC", normalize_text(text))
    return " ".join(normalized.split())


def _legacy_chunks_cover_source(
    source_text: str,
    chunks: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether ordered cached chunks continuously cover the current source text."""

    comparable_source = _legacy_comparable_text(source_text)
    if not comparable_source or not chunks:
        return False

    def ordering(item: tuple[int, Mapping[str, Any]]) -> tuple[int, int]:
        ordinal, chunk = item
        try:
            index = int(chunk.get("chunk_index", ordinal))
        except (TypeError, ValueError):
            index = ordinal
        return index, ordinal

    ordered = sorted(enumerate(chunks), key=ordering)
    previous_start = -1
    covered_until = 0
    for sequence_index, (_, chunk) in enumerate(ordered):
        comparable_chunk = _legacy_comparable_text(str(chunk.get("text", "")))
        if not comparable_chunk:
            return False
        search_start = 0 if previous_start < 0 else previous_start + 1
        position = comparable_source.find(comparable_chunk, search_start)
        if position < 0:
            return False
        if sequence_index == 0 and position != 0:
            return False
        if position > covered_until:
            return False
        chunk_end = position + len(comparable_chunk)
        if previous_start >= 0 and chunk_end <= covered_until:
            return False
        previous_start = position
        covered_until = max(covered_until, chunk_end)

    return covered_until == len(comparable_source)


GENERIC_SOURCE_KINDS = frozenset(
    {
        "",
        "BOARD",
        "ERCOT REFERENCE",
        "ERCOT REPORT",
        "OFFICIAL DOCUMENT",
        "RPG",
        "ROS",
        "TAC",
    }
)


def _canonical_record_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Prefer trusted, domain-specific duplicate metadata deterministically."""

    document_number = str(record.get("document_number") or "").strip()
    source_kind = str(record.get("source_kind") or "").strip()
    title = str(record.get("title") or "").strip()
    filename_stem = Path(str(record.get("filename") or "")).stem

    def compact(value: str) -> str:
        return "".join(character for character in value.upper() if character.isalnum())

    normalized_title = compact(title)
    title_is_specific = bool(title) and normalized_title != compact(filename_stem)
    if document_number:
        title_is_specific = title_is_specific or compact(document_number) in normalized_title
    metadata_richness = sum(
        bool(record.get(field))
        for field in (
            "document_number",
            "document_status",
            "effective_date",
            "published_date",
            "revision",
            "original_url",
        )
    )
    return (
        -authority_rank(record),
        -int(bool(document_number)),
        -int(source_kind.upper() not in GENERIC_SOURCE_KINDS),
        -int(title_is_specific),
        -metadata_richness,
        str(record.get("source_path") or ""),
    )


@dataclass
class _BaseIndex:
    generation_id: str | None
    manifest: dict[str, Any]
    chunks: list[dict[str, Any]]
    embeddings: Any
    is_legacy_bootstrap: bool = False


@dataclass
class _ContentPayload:
    chunks: list[dict[str, Any]]
    embeddings: Any


class IngestionPipeline:
    """Discover, parse, embed, and atomically publish ERCOT documents."""

    def __init__(
        self,
        config: IngestionConfig | None = None,
        *,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self.config = config or default_config()
        self._embedder = embedder

    def _get_embedder(self) -> EmbeddingProvider:
        if self._embedder is None:
            self._embedder = OpenAIEmbedder(self.config.embedding_model)
        return self._embedder

    def _relative(self, path: Path) -> str:
        resolved = path.resolve(strict=False)
        try:
            return resolved.relative_to(self.config.repo_root.resolve()).as_posix()
        except ValueError:
            return resolved.as_posix()

    def _source_for(self, path: Path) -> SourceRoot:
        resolved = path.resolve(strict=False)
        matches: list[SourceRoot] = []
        for source in self.config.source_roots:
            try:
                resolved.relative_to(source.path.resolve(strict=False))
            except ValueError:
                continue
            matches.append(source)
        if not matches:
            raise ValueError(f"Path is outside configured ERCOT source roots: {path}")
        return max(matches, key=lambda source: len(source.path.resolve(strict=False).parts))

    def _walk_directory(self, directory: Path, source: SourceRoot) -> list[tuple[Path, SourceRoot]]:
        results: list[tuple[Path, SourceRoot]] = []
        if not directory.exists():
            return results
        for candidate in sorted(directory.rglob("*")):
            # Watched roots are trust boundaries. Never follow a file symlink
            # that could make automatic ingestion read/embed data elsewhere on
            # the host, even when the link itself has a supported extension.
            if candidate.is_symlink():
                continue
            if not candidate.is_file():
                continue
            try:
                candidate.resolve(strict=True).relative_to(source.path.resolve(strict=True))
                relative_parts = candidate.relative_to(source.path).parts
            except ValueError:
                continue
            except OSError:
                continue
            if any(part == "dwg_sswg_chunks" for part in relative_parts):
                continue
            if candidate.name == "placeholder.txt":
                continue
            if candidate.name.startswith("chunk") and candidate.suffix.lower() == ".txt":
                continue
            if any(part.startswith(".") or part == "__pycache__" for part in relative_parts):
                continue
            if candidate.name in self.config.ignored_names or candidate.name.endswith(SIDECAR_SUFFIX):
                continue
            results.append((candidate, source))
        return results

    def _discover(
        self, paths: Sequence[Path] | None
    ) -> tuple[list[tuple[Path, SourceRoot]], set[str], bool]:
        """Return candidates, source names fully scanned, and whether scan was global."""

        if paths is None:
            candidates: list[tuple[Path, SourceRoot]] = []
            scanned_roots: set[str] = set()
            for source in self.config.source_roots:
                if source.path.exists():
                    scanned_roots.add(source.name)
                    candidates.extend(self._walk_directory(source.path, source))
            deduped = {self._relative(path): (path, source) for path, source in candidates}
            return [deduped[key] for key in sorted(deduped)], scanned_roots, True

        candidates = []
        scanned_roots: set[str] = set()
        for supplied in paths:
            candidate = supplied if supplied.is_absolute() else self.config.repo_root / supplied
            source = self._source_for(candidate)
            if candidate.is_dir():
                candidates.extend(self._walk_directory(candidate, source))
                # An explicitly supplied source root is a complete
                # reconciliation scope, not merely a list of changed files.
                # This lets callers such as the official-download monitor
                # tombstone files that were deleted or renamed between runs.
                if candidate.resolve(strict=False) == source.path.resolve(strict=False):
                    scanned_roots.add(source.name)
            else:
                candidates.append((candidate, source))
        deduped = {self._relative(path): (path, source) for path, source in candidates}
        return (
            [deduped[key] for key in sorted(deduped)],
            scanned_roots,
            bool(scanned_roots),
        )

    def _empty_base(self) -> _BaseIndex:
        np = _require_numpy()
        return _BaseIndex(
            generation_id=None,
            manifest={
                "schema_version": SCHEMA_VERSION,
                "documents": {},
                "content": {},
                "embedding_model": self.config.embedding_model,
            },
            chunks=[],
            embeddings=np.empty((0, 0), dtype="float32"),
        )

    def _legacy_source_root(self) -> SourceRoot | None:
        for source in self.config.source_roots:
            if source.name == "authoritative_static":
                return source
        return None

    def _bootstrap_legacy(self) -> _BaseIndex | None:
        """Adapt the checked-in all-in-one JSON/NPY vectors without re-embedding."""

        chunks_path = self.config.legacy_chunks_path
        embeddings_path = self.config.legacy_embeddings_path
        sources_dir = self.config.legacy_sources_dir
        source_root = self._legacy_source_root()
        if not chunks_path or not embeddings_path or not sources_dir or not source_root:
            return None
        if not chunks_path.exists() or not embeddings_path.exists() or not sources_dir.exists():
            return None
        try:
            with chunks_path.open("r", encoding="utf-8") as handle:
                legacy_chunks = json.load(handle)
            np = _require_numpy()
            legacy_embeddings = np.load(embeddings_path, allow_pickle=False)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to bootstrap legacy ERCOT vectors: {exc}") from exc
        if not isinstance(legacy_chunks, list) or legacy_embeddings.ndim != 2:
            raise RuntimeError("Legacy ERCOT vector files have invalid structures")
        if len(legacy_chunks) != int(legacy_embeddings.shape[0]):
            raise RuntimeError(
                f"Legacy ERCOT cache mismatch: {len(legacy_chunks)} chunks vs "
                f"{legacy_embeddings.shape[0]} vectors"
            )

        documents: dict[str, dict[str, Any]] = {}
        converted: list[dict[str, Any]] = []
        converted_rows: list[int] = []
        chunk_ids_by_path: dict[str, list[str]] = defaultdict(list)
        source_info: dict[str, tuple[str, dict[str, Any], str]] = {}
        chunks_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for legacy in legacy_chunks:
            if isinstance(legacy, Mapping):
                chunks_by_source[str(legacy.get("source", ""))].append(legacy)

        for source_name in sorted(chunks_by_source):
            if not source_name:
                continue
            raw_path = sources_dir / Path(source_name).name
            if raw_path.exists():
                try:
                    hash_before_load = _file_hash(raw_path)
                    source_text = load_document(
                        raw_path,
                        max_file_bytes=self.config.max_file_bytes,
                    )
                    content_hash = _file_hash(raw_path)
                except Exception:
                    # A legacy vector must never stand in for a source that the
                    # current loader cannot validate. Normal update processing
                    # will parse it again and record any resulting error.
                    continue
                if content_hash != hash_before_load:
                    continue
                if not _legacy_chunks_cover_source(source_text, chunks_by_source[source_name]):
                    continue
                stat = raw_path.stat()
                metadata = classify_document(raw_path, source_root, self.config.repo_root)
                relative = self._relative(raw_path)
                source_info[source_name] = (content_hash, metadata, relative)
                documents[relative] = {
                    **metadata,
                    "path": relative,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "modification_timestamp": _mtime_iso(stat.st_mtime),
                    "sha256": content_hash,
                    "indexed_sha256": content_hash,
                    "document_id": document_id(content_hash),
                    "document_type": raw_path.suffix.lower().lstrip("."),
                    "ingestion_timestamp": _utc_now(),
                    "status": "ingested",
                    "error": None,
                    "duplicate_of": None,
                    "bootstrap_source": "chatbot_ercot_all_in_one",
                    "chunk_ids": [],
                }

        for row, legacy in enumerate(legacy_chunks):
            if not isinstance(legacy, Mapping):
                continue
            source_name = str(legacy.get("source", ""))
            info = source_info.get(source_name)
            if info is None:
                continue
            content_hash, metadata, relative = info
            text = str(legacy.get("text", ""))
            index = int(legacy.get("chunk_index", len(chunk_ids_by_path[relative])))
            doc_identifier = document_id(content_hash)
            identifier = chunk_id(doc_identifier, index, text)
            record = dict(metadata)
            record.update(
                {
                    "id": identifier,
                    "chunk_id": identifier,
                    "document_id": doc_identifier,
                    "content_hash": content_hash,
                    "chunk_index": index,
                    "text": text,
                    "source": relative,
                    "aliases": [relative],
                }
            )
            converted.append(record)
            converted_rows.append(row)
            chunk_ids_by_path[relative].append(identifier)

        for relative, identifiers in chunk_ids_by_path.items():
            documents[relative]["chunk_ids"] = identifiers

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generation_id": "legacy-bootstrap",
            "created_at": _utc_now(),
            "previous_generation": None,
            "embedding_model": self.config.legacy_embedding_model,
            "embedding_dimension": int(legacy_embeddings.shape[1]),
            "chunk_size": self.config.legacy_chunk_size,
            "chunk_overlap": self.config.legacy_chunk_overlap,
            "documents": documents,
            "content": {},
            "summary": {"bootstrap_chunks": len(converted)},
        }
        return _BaseIndex(
            generation_id="legacy-bootstrap",
            manifest=manifest,
            chunks=converted,
            embeddings=np.asarray(legacy_embeddings[converted_rows], dtype="float32"),
            is_legacy_bootstrap=True,
        )

    def _load_base(self, *, allow_legacy: bool = True) -> _BaseIndex:
        generation = load_generation(self.config.index_dir)
        if generation is not None:
            return _BaseIndex(
                generation_id=generation.generation_id,
                manifest=generation.manifest,
                chunks=generation.chunks,
                embeddings=generation.embeddings,
            )
        if allow_legacy:
            legacy = self._bootstrap_legacy()
            if legacy is not None:
                return legacy
        return self._empty_base()

    def _inspect_record(
        self,
        path: Path,
        source: SourceRoot,
        previous: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        relative = self._relative(path)
        if not path.exists():
            if previous:
                record = copy.deepcopy(dict(previous))
                record.update(
                    {
                        "path": relative,
                        "status": "deleted",
                        "deleted_at": _utc_now(),
                        "error": None,
                        "chunk_ids": [],
                    }
                )
                return record
            attempted_at = _utc_now()
            return {
                "path": relative,
                "source_path": relative,
                "status": "missing",
                "error": "File does not exist",
                "chunk_ids": [],
                "ingestion_timestamp": attempted_at,
                "last_attempted_at": attempted_at,
            }

        try:
            stat = path.stat()
        except OSError as exc:
            record = copy.deepcopy(dict(previous or {}))
            attempted_at = _utc_now()
            record.update(
                {
                    "path": relative,
                    "source_path": relative,
                    "status": "error",
                    "error": f"Unable to stat source file: {exc}",
                    "ingestion_timestamp": record.get("ingestion_timestamp") or attempted_at,
                    "last_attempted_at": attempted_at,
                }
            )
            return record
        extension = path.suffix.lower()
        base = copy.deepcopy(dict(previous or {}))
        base.update(
            {
                "path": relative,
                "source_path": relative,
                "filename": path.name,
                "source_category": source.name,
                "source_authority": source.source_authority,
                "is_generated": source.is_generated,
                "document_type": extension.lstrip(".") or "unknown",
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "modification_timestamp": _mtime_iso(stat.st_mtime),
                "chunk_ids": [],
            }
        )
        base.pop("deleted_at", None)
        base.pop("stale", None)
        try:
            content_hash = _file_hash(path)
        except OSError as exc:
            attempted_at = _utc_now()
            base.update(
                {
                    "status": "error",
                    "error": f"Unable to hash source file: {exc}",
                    "sha256": base.get("sha256"),
                    "indexed_sha256": (previous or {}).get("indexed_sha256")
                    or (previous or {}).get("sha256"),
                    "document_id": (previous or {}).get("document_id"),
                    "chunk_ids": list((previous or {}).get("chunk_ids", [])),
                    "ingestion_timestamp": base.get("ingestion_timestamp") or attempted_at,
                    "last_attempted_at": attempted_at,
                }
            )
            return base
        try:
            metadata = classify_document(
                path,
                source,
                self.config.repo_root,
                sidecar=load_sidecar(path),
            )
        except Exception as exc:
            attempted_at = _utc_now()
            base.update(
                {
                    "status": "error",
                    "error": str(exc),
                    "sha256": content_hash,
                    "indexed_sha256": (previous or {}).get("indexed_sha256")
                    or (previous or {}).get("sha256"),
                    "document_id": document_id(content_hash),
                    "chunk_ids": list((previous or {}).get("chunk_ids", [])),
                    "ingestion_timestamp": base.get("ingestion_timestamp") or attempted_at,
                    "last_attempted_at": attempted_at,
                }
            )
            return base

        prior_indexed = (previous or {}).get("indexed_sha256")
        if not prior_indexed and (previous or {}).get("status") in {"ingested", "duplicate"}:
            prior_indexed = (previous or {}).get("sha256")
        base.update(metadata)
        if extension not in SUPPORTED_EXTENSIONS:
            base.update(
                {
                    "status": "skipped",
                    "error": f"Unsupported document extension: {extension or '(none)'}",
                    "sha256": content_hash,
                    "indexed_sha256": None,
                    "document_id": document_id(content_hash),
                    "duplicate_of": None,
                    "ingestion_timestamp": base.get("ingestion_timestamp") or _utc_now(),
                }
            )
            return base
        base.update(
            {
                "sha256": content_hash,
                "indexed_sha256": prior_indexed,
                "document_id": document_id(content_hash),
                "status": "pending",
                "error": None,
                "duplicate_of": None,
            }
        )
        if previous and previous.get("sha256") == content_hash:
            # Keep unchanged records byte-stable so repeated incremental runs
            # do not publish a generation merely to refresh a timestamp.
            base["ingestion_timestamp"] = previous.get("ingestion_timestamp")
        return base

    def _scan_against(
        self,
        base_documents: Mapping[str, Mapping[str, Any]],
        paths: Sequence[Path] | None,
        *,
        force: bool,
    ) -> dict[str, Any]:
        candidates, scanned_roots, global_scan = self._discover(paths)
        files: list[dict[str, Any]] = []
        seen: set[str] = set()
        missing_by_hash: dict[str, list[str]] = defaultdict(list)

        if global_scan:
            for relative, previous in base_documents.items():
                if previous.get("source_category") in scanned_roots:
                    missing_by_hash[str(previous.get("sha256", ""))].append(relative)

        current_hash_paths: dict[str, list[str]] = defaultdict(list)
        inspected: list[tuple[dict[str, Any], Mapping[str, Any] | None]] = []
        for path, source in candidates:
            relative = self._relative(path)
            seen.add(relative)
            previous = base_documents.get(relative)
            record = self._inspect_record(path, source, previous)
            inspected.append((record, previous))
            if record.get("sha256"):
                current_hash_paths[str(record["sha256"])].append(relative)

        active_hashes = {
            str(document.get("indexed_sha256") or document.get("sha256"))
            for document in base_documents.values()
            if document.get("status") in {"ingested", "duplicate", "error"}
        }
        for record, previous in inspected:
            status = record.get("status")
            content_hash = str(record.get("sha256", ""))
            if status in {"skipped", "error", "missing"}:
                action = status
            elif force:
                action = "reindex"
            elif previous and previous.get("sha256") == content_hash and previous.get("status") in {
                "ingested",
                "duplicate",
            }:
                action = "metadata_changed" if _metadata_changed(previous, record) else "unchanged"
            elif previous and previous.get("sha256") != content_hash:
                action = "modified"
            elif global_scan and any(
                old not in seen for old in missing_by_hash.get(content_hash, [])
            ):
                action = "renamed"
            elif content_hash in active_hashes or len(current_hash_paths.get(content_hash, [])) > 1:
                action = "duplicate"
            else:
                action = "new"
            files.append(
                {
                    "path": record.get("path"),
                    "action": action,
                    "status": status,
                    "sha256": record.get("sha256"),
                    "source_kind": record.get("source_kind"),
                    "collections": record.get("collections", []),
                    "is_generated": record.get("is_generated"),
                    "error": record.get("error"),
                }
            )

        if global_scan:
            for relative, previous in sorted(base_documents.items()):
                if (
                    previous.get("source_category") in scanned_roots
                    and relative not in seen
                    and previous.get("status") not in {"deleted", "missing"}
                ):
                    files.append(
                        {
                            "path": relative,
                            "action": "deleted",
                            "status": "deleted",
                            "sha256": previous.get("sha256"),
                            "source_kind": previous.get("source_kind"),
                            "collections": previous.get("collections", []),
                            "is_generated": previous.get("is_generated"),
                            "error": None,
                        }
                    )
        files.sort(key=lambda item: str(item.get("path", "")))
        actions = Counter(str(item["action"]) for item in files)
        return {
            "files": files,
            "summary": dict(sorted(actions.items())),
            "configured_roots": [
                {
                    "name": root.name,
                    "path": self._relative(root.path),
                    "exists": root.path.exists(),
                    "source_authority": root.source_authority,
                    "is_generated": root.is_generated,
                }
                for root in self.config.source_roots
            ],
        }

    def scan(self, paths: Sequence[Path] | None = None, *, force: bool = False) -> dict[str, Any]:
        """Describe pending work without parsing, embedding, or writing state."""

        generation = load_generation(self.config.index_dir)
        documents = generation.manifest.get("documents", {}) if generation else {}
        result = self._scan_against(documents, paths, force=force)
        result.update(
            {
                "command": "scan",
                "dry_run": True,
                "current_generation": generation.generation_id if generation else None,
                "legacy_bootstrap_available": bool(
                    self.config.legacy_chunks_path
                    and self.config.legacy_embeddings_path
                    and self.config.legacy_chunks_path.exists()
                    and self.config.legacy_embeddings_path.exists()
                ),
            }
        )
        return result

    def _payloads_from_base(self, base: _BaseIndex) -> dict[str, _ContentPayload]:
        np = _require_numpy()
        indices: dict[str, list[int]] = defaultdict(list)
        for index, chunk in enumerate(base.chunks):
            content_hash = str(chunk.get("content_hash", ""))
            if content_hash:
                indices[content_hash].append(index)
        return {
            content_hash: _ContentPayload(
                chunks=[copy.deepcopy(base.chunks[index]) for index in rows],
                embeddings=np.asarray(base.embeddings[rows], dtype="float32"),
            )
            for content_hash, rows in indices.items()
        }

    def _embed(self, texts: Sequence[str]) -> Any:
        provider = self._get_embedder()
        method = getattr(provider, "embed_texts", None)
        if method is None:
            if not callable(provider):
                raise TypeError("Embedding provider must define embed_texts(texts)")
            return provider(texts)  # type: ignore[misc,operator]
        return method(texts)

    def _prepare_documents(
        self,
        base: _BaseIndex,
        paths: Sequence[Path] | None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Path], set[str], bool]:
        documents = copy.deepcopy(dict(base.manifest.get("documents", {})))
        candidates, scanned_roots, global_scan = self._discover(paths)
        candidate_paths: dict[str, Path] = {}
        seen: set[str] = set()
        for path, source in candidates:
            relative = self._relative(path)
            seen.add(relative)
            candidate_paths[relative] = path
            documents[relative] = self._inspect_record(path, source, documents.get(relative))

        if global_scan:
            for relative, record in documents.items():
                if (
                    record.get("source_category") in scanned_roots
                    and relative not in seen
                    and record.get("status") not in {"deleted", "missing"}
                ):
                    record.update(
                        {
                            "status": "deleted",
                            "deleted_at": _utc_now(),
                            "error": None,
                            "chunk_ids": [],
                            "duplicate_of": None,
                        }
                    )
        return documents, candidate_paths, scanned_roots, global_scan

    def _process_pending_content(
        self,
        documents: dict[str, dict[str, Any]],
        candidate_paths: Mapping[str, Path],
        payloads: dict[str, _ContentPayload],
        *,
        force: bool,
        base_dimension: int | None,
        base_model: str | None,
        base_chunk_size: int | None,
        base_chunk_overlap: int | None,
    ) -> tuple[int, int, int | None]:
        np = _require_numpy()
        by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in documents.values():
            if record.get("status") == "pending" and record.get("sha256"):
                by_hash[str(record["sha256"])].append(record)

        reused = 0
        embedded = 0
        expected_dimension = base_dimension
        for content_hash in sorted(by_hash):
            records = by_hash[content_hash]
            existing = payloads.get(content_hash)
            if existing is not None and not force:
                for record in records:
                    record.update(
                        {
                            "status": "ingested",
                            "indexed_sha256": content_hash,
                            "error": None,
                            "ingestion_timestamp": record.get("ingestion_timestamp") or _utc_now(),
                        }
                    )
                reused += len(existing.chunks)
                continue

            target_model = provider_model(self._get_embedder(), self.config.embedding_model)
            if expected_dimension is not None and base_model and target_model != base_model:
                raise RuntimeError(
                    f"Embedding model changed from {base_model!r} to {target_model!r}; "
                    "run a full `rebuild --force` without --path to migrate atomically"
                )
            if expected_dimension is not None and (
                (base_chunk_size is not None and base_chunk_size != self.config.chunk_size)
                or (
                    base_chunk_overlap is not None
                    and base_chunk_overlap != self.config.chunk_overlap
                )
            ):
                raise RuntimeError(
                    "Chunking configuration differs from the active generation; run a full "
                    "`rebuild --force` without --path"
                )

            representative = min(records, key=lambda record: str(record.get("source_path", "")))
            source_path = str(representative.get("source_path", ""))
            path = candidate_paths.get(source_path)
            if path is None:
                error = f"No local source path available for {source_path}"
                attempted_at = _utc_now()
                for record in records:
                    record.update(
                        {
                            "status": "error",
                            "error": error,
                            "ingestion_timestamp": record.get("ingestion_timestamp")
                            or attempted_at,
                            "last_attempted_at": attempted_at,
                        }
                    )
                continue
            try:
                text = load_document(path, max_file_bytes=self.config.max_file_bytes)
                chunks = build_chunks(
                    text,
                    content_hash=content_hash,
                    metadata=representative,
                    chunk_size=self.config.chunk_size,
                    overlap=self.config.chunk_overlap,
                )
                if not chunks:
                    raise ValueError("Document produced no chunks")
                vectors = np.asarray(self._embed([chunk["text"] for chunk in chunks]), dtype="float32")
                if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
                    raise ValueError(
                        f"Embedding provider returned shape {vectors.shape} for {len(chunks)} chunks"
                    )
                dimension = int(vectors.shape[1])
                if expected_dimension is not None and dimension != expected_dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch: new vectors are {dimension}, active index is "
                        f"{expected_dimension}"
                    )
                expected_dimension = dimension
                payloads[content_hash] = _ContentPayload(chunks=chunks, embeddings=vectors)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                attempted_at = _utc_now()
                for record in records:
                    record.update(
                        {
                            "status": "error",
                            "error": error,
                            "ingestion_timestamp": record.get("ingestion_timestamp")
                            or attempted_at,
                            "last_attempted_at": attempted_at,
                        }
                    )
                continue

            attempted_at = _utc_now()
            for record in records:
                record.update(
                    {
                        "status": "ingested",
                        "indexed_sha256": content_hash,
                        "error": None,
                        "ingestion_timestamp": attempted_at,
                        "last_attempted_at": attempted_at,
                    }
                )
            embedded += len(chunks)
        return reused, embedded, expected_dimension

    def _finalize(
        self,
        documents: dict[str, dict[str, Any]],
        payloads: Mapping[str, _ContentPayload],
        dimension: int | None,
    ) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
        np = _require_numpy()
        successful: dict[str, list[dict[str, Any]]] = defaultdict(list)
        retained: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in documents.values():
            content_hash = str(record.get("sha256", ""))
            if record.get("status") in {"ingested", "duplicate"} and content_hash in payloads:
                successful[content_hash].append(record)
                retained[content_hash].append(record)
            elif record.get("status") == "error":
                indexed_hash = str(record.get("indexed_sha256", ""))
                if not indexed_hash and content_hash in payloads:
                    # A rename can lose path-local history before metadata
                    # parsing fails. Exact content identity is sufficient to
                    # retain the known-good payload without clearing or
                    # concealing the new path's ingestion error.
                    indexed_hash = content_hash
                    record["indexed_sha256"] = content_hash
                if indexed_hash in payloads:
                    retained[indexed_hash].append(record)
                    record["stale"] = True

        for content_hash, records in successful.items():
            canonical = min(records, key=_canonical_record_key)
            canonical_path = str(canonical.get("source_path", ""))
            for record in records:
                if record is canonical:
                    record.update({"status": "ingested", "duplicate_of": None})
                else:
                    record.update({"status": "duplicate", "duplicate_of": canonical_path})

        all_chunks: list[dict[str, Any]] = []
        vectors: list[Any] = []
        content_registry: dict[str, Any] = {}
        for indexed_hash in sorted(retained):
            payload = payloads.get(indexed_hash)
            if payload is None:
                continue
            indexed_records = retained[indexed_hash]
            live_records = [
                record
                for record in indexed_records
                if record.get("status") in {"ingested", "duplicate"}
                and str(record.get("sha256", "")) == indexed_hash
            ]
            stale_records = [
                record for record in indexed_records if record.get("status") == "error"
            ]
            citation_records = live_records or stale_records
            stale_only = not live_records and bool(stale_records)
            aliases = sorted(
                {
                    str(record.get("source_path", ""))
                    for record in citation_records
                    if record.get("source_path")
                }
            )
            canonical_record = min(citation_records, key=_canonical_record_key)
            provenance_records = [
                record
                for record in citation_records
                if authority_rank(record) == authority_rank(canonical_record)
                and bool(record.get("is_generated"))
                == bool(canonical_record.get("is_generated"))
            ]
            merged_url_aliases = sorted(
                {
                    str(value)
                    for record in provenance_records
                    for value in [
                        record.get("original_url"),
                        *(record.get("url_aliases") or []),
                    ]
                    if value
                }
            )
            merged_source_page_urls = sorted(
                {
                    str(value)
                    for record in provenance_records
                    for value in (record.get("source_page_urls") or [])
                    if value
                }
            )
            merged_provenance_by_json = {
                json.dumps(value, sort_keys=True, ensure_ascii=False): copy.deepcopy(value)
                for record in provenance_records
                for value in (record.get("provenance") or [])
                if isinstance(value, dict)
            }
            merged_provenance = [
                merged_provenance_by_json[key]
                for key in sorted(merged_provenance_by_json)
            ]
            status_records = [
                record for record in provenance_records if record.get("document_status")
            ]
            lifecycle_record = max(status_records or provenance_records, key=_lifecycle_rank)
            merged_downloaded_at = _newest_value(provenance_records, "downloaded_at")
            merged_published_date = _newest_value(provenance_records, "published_date")
            merged_effective_date = _newest_value(provenance_records, "effective_date")
            if live_records:
                collections = sorted(
                    {
                        str(collection)
                        for record in live_records
                        for collection in record.get("collections", [])
                    }
                )
            else:
                collections = sorted(
                    {
                        str(collection)
                        for chunk in payload.chunks
                        for collection in chunk.get("collections", [])
                    }
                )
            chunk_ids: list[str] = []
            order = sorted(
                range(len(payload.chunks)),
                key=lambda index: (
                    int(payload.chunks[index].get("chunk_index", index)),
                    str(payload.chunks[index].get("chunk_id", "")),
                ),
            )
            for payload_index in order:
                chunk = copy.deepcopy(payload.chunks[payload_index])
                if not stale_only:
                    chunk.update(
                        {
                            "path": canonical_record.get("source_path"),
                            "source": canonical_record.get("source_path"),
                            "source_path": canonical_record.get("source_path"),
                            "filename": canonical_record.get("filename"),
                            "title": canonical_record.get("title"),
                            "source_category": canonical_record.get("source_category"),
                            "source_authority": canonical_record.get("source_authority"),
                            "source_kind": canonical_record.get("source_kind"),
                            "is_generated": canonical_record.get("is_generated"),
                            "original_url": canonical_record.get("original_url"),
                            "url_aliases": merged_url_aliases,
                            "source_page_urls": merged_source_page_urls,
                            "provenance": merged_provenance,
                            "downloaded_at": merged_downloaded_at,
                            "published_date": merged_published_date,
                            "effective_date": merged_effective_date,
                            "document_number": canonical_record.get("document_number"),
                            "document_status": lifecycle_record.get("document_status"),
                            "revision": lifecycle_record.get("revision"),
                        }
                    )
                    chunk.pop("stale", None)
                    chunk.pop("current_content_hash", None)
                    chunk.pop("ingestion_error", None)
                chunk.update(
                    {
                        "collections": collections,
                        "aliases": aliases,
                        "content_hash": indexed_hash,
                    }
                )
                if stale_only:
                    chunk.update(
                        {
                            "stale": True,
                            "current_content_hash": canonical_record.get("sha256"),
                            "ingestion_error": canonical_record.get("error"),
                        }
                    )
                all_chunks.append(chunk)
                vectors.append(payload.embeddings[payload_index])
                chunk_ids.append(str(chunk.get("chunk_id") or chunk.get("id")))
            for record in indexed_records:
                record["chunk_ids"] = chunk_ids
            content_registry[indexed_hash] = {
                "document_id": document_id(indexed_hash),
                "canonical_path": (
                    payload.chunks[0].get("source_path")
                    if stale_only and payload.chunks
                    else canonical_record.get("source_path")
                ),
                "aliases": aliases,
                "collections": collections,
                "chunk_ids": chunk_ids,
                "source_authority": (
                    payload.chunks[0].get("source_authority")
                    if stale_only and payload.chunks
                    else canonical_record.get("source_authority")
                ),
                "is_generated": (
                    payload.chunks[0].get("is_generated")
                    if stale_only and payload.chunks
                    else canonical_record.get("is_generated")
                ),
                "stale": stale_only,
            }

        if vectors:
            matrix = np.asarray(vectors, dtype="float32")
        else:
            matrix = np.empty((0, dimension or 0), dtype="float32")
        return all_chunks, matrix, content_registry

    def update(
        self,
        paths: Sequence[Path] | None = None,
        *,
        dry_run: bool = False,
        force: bool = False,
        _fresh_rebuild: bool = False,
    ) -> dict[str, Any]:
        """Incrementally publish changes, preserving the active generation on failure."""

        if dry_run:
            result = self.scan(paths, force=force)
            result["command"] = "update"
            return result

        with update_lock(self.config.index_dir):
            base = self._load_base(allow_legacy=True)
            original_documents = copy.deepcopy(dict(base.manifest.get("documents", {})))
            documents, candidate_paths, _, _ = self._prepare_documents(base, paths)
            base_payloads = self._payloads_from_base(base)
            payloads = {} if _fresh_rebuild else base_payloads
            if _fresh_rebuild:
                unavailable = sorted(
                    relative
                    for relative, record in documents.items()
                    if record.get("status") in {"ingested", "duplicate", "error"}
                    and record.get("indexed_sha256")
                    and str(record.get("indexed_sha256")) in base_payloads
                    and relative not in candidate_paths
                )
                if unavailable:
                    preview = ", ".join(unavailable[:5])
                    suffix = " ..." if len(unavailable) > 5 else ""
                    raise RuntimeError(
                        "Full rebuild requires every currently indexed source to be available; "
                        f"missing {preview}{suffix}"
                    )
            base_dimension = (
                int(base.embeddings.shape[1])
                if not _fresh_rebuild
                and getattr(base.embeddings, "ndim", 0) == 2
                and int(base.embeddings.shape[0]) > 0
                else None
            )
            reused, embedded, dimension = self._process_pending_content(
                documents,
                candidate_paths,
                payloads,
                force=force,
                base_dimension=base_dimension,
                base_model=(
                    None
                    if _fresh_rebuild
                    else str(base.manifest.get("embedding_model") or "") or None
                ),
                base_chunk_size=(
                    None if _fresh_rebuild else base.manifest.get("chunk_size")
                ),
                base_chunk_overlap=(
                    None if _fresh_rebuild else base.manifest.get("chunk_overlap")
                ),
            )
            if _fresh_rebuild and base.generation_id:
                rebuild_errors = sorted(
                    relative
                    for relative, record in documents.items()
                    if record.get("status") == "error"
                )
                if rebuild_errors:
                    preview = ", ".join(rebuild_errors[:5])
                    suffix = " ..." if len(rebuild_errors) > 5 else ""
                    raise RuntimeError(
                        "Full rebuild was not published because source processing failed for "
                        f"{preview}{suffix}"
                    )
            chunks, embeddings, content_registry = self._finalize(documents, payloads, dimension)

            material_change = (
                documents != original_documents
                or force
                or _fresh_rebuild
                or base.is_legacy_bootstrap
            )
            if not material_change and base.generation_id:
                return {
                    "command": "update",
                    "changed": False,
                    "generation": base.generation_id,
                    "documents": len(documents),
                    "chunks": len(chunks),
                    "embedded_chunks": 0,
                    "reused_chunks": len(chunks),
                    "errors": sum(1 for record in documents.values() if record.get("status") == "error"),
                }

            identifier = _generation_id()
            if embedded or _fresh_rebuild or not base.generation_id:
                embedding_model = provider_model(
                    self._embedder or self._get_embedder(),
                    self.config.embedding_model,
                )
                manifest_chunk_size = self.config.chunk_size
                manifest_chunk_overlap = self.config.chunk_overlap
            else:
                embedding_model = str(
                    base.manifest.get("embedding_model") or self.config.embedding_model
                )
                manifest_chunk_size = int(
                    base.manifest.get("chunk_size") or self.config.chunk_size
                )
                manifest_chunk_overlap = int(
                    base.manifest.get("chunk_overlap")
                    if base.manifest.get("chunk_overlap") is not None
                    else self.config.chunk_overlap
                )
            statuses = Counter(str(record.get("status", "unknown")) for record in documents.values())
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "generation_id": identifier,
                "created_at": _utc_now(),
                "previous_generation": base.generation_id,
                "embedding_model": embedding_model,
                "embedding_dimension": int(embeddings.shape[1]),
                "chunk_size": manifest_chunk_size,
                "chunk_overlap": manifest_chunk_overlap,
                "documents": dict(sorted(documents.items())),
                "content": content_registry,
                "summary": {
                    "documents": len(documents),
                    "chunks": len(chunks),
                    "embedded_chunks": embedded,
                    "reused_chunks": reused,
                    "statuses": dict(sorted(statuses.items())),
                },
            }
            published = write_generation(
                self.config.index_dir,
                identifier,
                manifest,
                chunks,
                embeddings,
            )
            pruned_generations: list[str] = []
            retention_error: str | None = None
            try:
                pruned_generations = prune_generations(
                    self.config.index_dir,
                    keep=self.config.generation_retention,
                )
            except OSError as exc:
                # The new CURRENT is already committed. Retention failure is a
                # maintenance warning, not an ingestion failure.
                retention_error = f"{type(exc).__name__}: {exc}"
            return {
                "command": "update",
                "changed": True,
                "generation": published.generation_id,
                "previous_generation": base.generation_id,
                "documents": len(documents),
                "chunks": len(chunks),
                "embedded_chunks": embedded,
                "reused_chunks": reused,
                "statuses": dict(sorted(statuses.items())),
                "errors": statuses.get("error", 0),
                "pruned_generations": pruned_generations,
                "retention_error": retention_error,
            }

    def rebuild(
        self,
        paths: Sequence[Path] | None = None,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Force re-embedding while retaining the old generation until publish succeeds."""

        result = self.update(
            paths,
            dry_run=dry_run,
            force=True,
            _fresh_rebuild=paths is None and not dry_run,
        )
        result["command"] = "rebuild"
        return result

    def status(self) -> dict[str, Any]:
        """Return machine-readable active generation and manifest health."""

        generation = load_generation(self.config.index_dir)
        if generation is None:
            return {
                "command": "status",
                "ready": False,
                "current_generation": None,
                "index_dir": str(self.config.index_dir),
                "legacy_bootstrap_available": bool(
                    self.config.legacy_chunks_path
                    and self.config.legacy_embeddings_path
                    and self.config.legacy_chunks_path.exists()
                    and self.config.legacy_embeddings_path.exists()
                ),
                "configured_roots": [str(root.path) for root in self.config.source_roots],
            }
        documents = generation.manifest.get("documents", {})
        statuses = Counter(str(record.get("status", "unknown")) for record in documents.values())
        errors = [
            {"path": path, "error": record.get("error"), "stale": bool(record.get("stale"))}
            for path, record in documents.items()
            if record.get("status") == "error"
        ]
        return {
            "command": "status",
            "ready": True,
            "current_generation": generation.generation_id,
            "index_dir": str(self.config.index_dir),
            "generation_dir": str(generation.path),
            "created_at": generation.manifest.get("created_at"),
            "embedding_model": generation.manifest.get("embedding_model"),
            "embedding_dimension": int(generation.embeddings.shape[1]),
            "documents": len(documents),
            "chunks": len(generation.chunks),
            "statuses": dict(sorted(statuses.items())),
            "errors": errors,
        }


def _path_values(values: Iterable[str], repo_root: Path) -> list[Path]:
    return [Path(value) if Path(value).is_absolute() else repo_root / value for value in values]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Incremental ERCOT RAG ingestion")
    parser.add_argument("--repo-root", type=Path, help="Repository root (auto-detected by default)")
    parser.add_argument("--index-dir", type=Path, help="Generation store directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Show new, changed, duplicate, and deleted files")
    scan.add_argument("--dry-run", action="store_true", help="Compatibility flag; scan is always read-only")
    scan.add_argument("--force", action="store_true", help="Show every supported file as a reindex")
    scan.add_argument("--path", action="append", default=[], help="Limit the scan to a source path")

    update = subparsers.add_parser("update", help="Publish an incremental generation")
    update.add_argument("--dry-run", action="store_true", help="Plan without parsing, embedding, or writing")
    update.add_argument("--force", action="store_true", help="Re-embed all documents in scope")
    update.add_argument("--changed-only", action="store_true", help="Compatibility flag; incremental is the default")
    update.add_argument("--path", action="append", default=[], help="Limit the update to a source path")

    rebuild = subparsers.add_parser("rebuild", help="Build a new generation with fresh vectors")
    rebuild.add_argument("--dry-run", action="store_true", help="Plan the rebuild without writing")
    rebuild.add_argument("--force", action="store_true", help="Compatibility flag; rebuild always forces")
    rebuild.add_argument("--path", action="append", default=[], help="Limit the rebuild to a source path")

    subparsers.add_parser("status", help="Show the active generation and ingestion errors")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the JSON-only CLI; diagnostics and failures go to stderr."""

    parser = _parser()
    args = parser.parse_args(argv)
    config = default_config(repo_root=args.repo_root, index_dir=args.index_dir)
    pipeline = IngestionPipeline(config)
    raw_paths = getattr(args, "path", [])
    paths = _path_values(raw_paths, config.repo_root) if raw_paths else None
    try:
        if args.command == "scan":
            result = pipeline.scan(paths, force=args.force)
        elif args.command == "update":
            result = pipeline.update(paths, dry_run=args.dry_run, force=args.force)
        elif args.command == "rebuild":
            result = pipeline.rebuild(paths, dry_run=args.dry_run)
        else:
            result = pipeline.status()
    except Exception as exc:
        print(f"ERCOT RAG {args.command} failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
