"""Atomic, generation-based JSON/NPY vector store."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence


SCHEMA_VERSION = 1
CURRENT_FILE = "CURRENT"
GENERATIONS_DIR = "generations"
MANIFEST_FILE = "manifest.json"
CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"


@dataclass
class Generation:
    generation_id: str
    path: Path
    manifest: dict[str, Any]
    chunks: list[dict[str, Any]]
    embeddings: Any


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("ERCOT RAG indexes require the `numpy` package") from exc
    return np


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: Path, value: Any) -> None:
    """Write deterministic JSON atomically."""

    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def current_generation_id(index_dir: Path) -> str | None:
    pointer = index_dir / CURRENT_FILE
    if not pointer.exists():
        return None
    try:
        value = pointer.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Unable to read ERCOT RAG CURRENT pointer: {exc}") from exc
    if not value or value in {".", ".."} or Path(value).name != value:
        raise RuntimeError("ERCOT RAG CURRENT pointer contains an invalid generation ID")
    # A generated store can be intentionally omitted from a deployment while
    # the small tracked CURRENT file remains in the checkout. Treat that
    # dangling pointer exactly like an absent store so startup can bootstrap a
    # fresh central generation from the checked-in ERCOT documents.
    if not (index_dir / GENERATIONS_DIR / value).is_dir():
        return None
    return value


def generation_state(index_dir: Path) -> tuple[str | None, int]:
    """Return a cheap cache token without loading an index."""

    pointer = index_dir / CURRENT_FILE
    generation_id = current_generation_id(index_dir)
    try:
        modified = pointer.stat().st_mtime_ns
    except OSError:
        modified = 0
    return generation_id, modified


def load_manifest(
    index_dir: Path,
    generation_id: str | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Load only a generation manifest, without materializing vector arrays."""

    selected = generation_id or current_generation_id(index_dir)
    if not selected:
        return None
    try:
        manifest = read_json(index_dir / GENERATIONS_DIR / selected / MANIFEST_FILE)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load ERCOT RAG manifest {selected}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"ERCOT RAG manifest {selected} is not a JSON object")
    if int(manifest.get("schema_version", 0)) != SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported ERCOT RAG manifest schema: {manifest.get('schema_version')!r}"
        )
    return selected, manifest


def load_generation(index_dir: Path, generation_id: str | None = None) -> Generation | None:
    """Load and validate a complete generation, or return ``None`` if absent."""

    selected = generation_id or current_generation_id(index_dir)
    if not selected:
        return None
    path = index_dir / GENERATIONS_DIR / selected
    try:
        manifest = read_json(path / MANIFEST_FILE)
        chunks = read_json(path / CHUNKS_FILE)
        np = _require_numpy()
        embeddings = np.load(path / EMBEDDINGS_FILE, allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load ERCOT RAG generation {selected}: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(chunks, list):
        raise RuntimeError(f"ERCOT RAG generation {selected} has invalid JSON structures")
    if int(manifest.get("schema_version", 0)) != SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported ERCOT RAG manifest schema: {manifest.get('schema_version')!r}"
        )
    if embeddings.ndim != 2:
        raise RuntimeError(f"Generation {selected} embeddings must be a 2-D array")
    if len(chunks) != int(embeddings.shape[0]):
        raise RuntimeError(
            f"Generation {selected} has {len(chunks)} chunks but {embeddings.shape[0]} vectors"
        )
    return Generation(selected, path, manifest, chunks, embeddings)


def _write_npy(path: Path, embeddings: Any) -> None:
    np = _require_numpy()
    array = np.asarray(embeddings, dtype="float32")
    if array.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional array")
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def write_generation(
    index_dir: Path,
    generation_id: str,
    manifest: dict[str, Any],
    chunks: Sequence[dict[str, Any]],
    embeddings: Any,
) -> Generation:
    """Publish a fully written generation and atomically switch ``CURRENT``."""

    generations = index_dir / GENERATIONS_DIR
    generations.mkdir(parents=True, exist_ok=True)
    destination = generations / generation_id
    if destination.exists():
        raise FileExistsError(f"Generation already exists: {generation_id}")
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=str(generations)))
    try:
        write_json(temporary / MANIFEST_FILE, manifest)
        write_json(temporary / CHUNKS_FILE, list(chunks))
        _write_npy(temporary / EMBEDDINGS_FILE, embeddings)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    loaded = load_generation(index_dir, generation_id)
    if loaded is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Published generation could not be loaded: {generation_id}")
    # Validation happens before the only reader-visible mutation. If validation
    # fails, the complete but unreachable generation is safe to inspect/remove
    # while readers continue using the previous CURRENT target.
    _atomic_text(index_dir / CURRENT_FILE, generation_id + "\n")
    return loaded


def prune_generations(index_dir: Path, *, keep: int) -> list[str]:
    """Remove old complete generations while retaining current rollback safety."""

    keep = max(2, keep)
    current = current_generation_id(index_dir)
    if not current:
        return []
    root = index_dir / GENERATIONS_DIR
    if not root.exists():
        return []

    complete = sorted(
        (
            candidate
            for candidate in root.iterdir()
            if candidate.is_dir()
            and not candidate.name.startswith(".")
            and (candidate / MANIFEST_FILE).is_file()
            and (candidate / CHUNKS_FILE).is_file()
            and (candidate / EMBEDDINGS_FILE).is_file()
        ),
        key=lambda candidate: candidate.name,
        reverse=True,
    )
    protected = {current}
    try:
        current_manifest = read_json(root / current / MANIFEST_FILE)
        previous = str(current_manifest.get("previous_generation") or "")
        if previous and (root / previous).is_dir():
            protected.add(previous)
    except (OSError, ValueError, json.JSONDecodeError, AttributeError):
        # CURRENT has already been validated during publication. A damaged
        # manifest must make pruning conservative, never endanger the pointer.
        pass
    for candidate in complete:
        if len(protected) >= keep:
            break
        protected.add(candidate.name)

    removed: list[str] = []
    for candidate in complete:
        if candidate.name in protected:
            continue
        shutil.rmtree(candidate)
        removed.append(candidate.name)
    return removed


@contextmanager
def update_lock(index_dir: Path) -> Iterator[None]:
    """Serialize writers on macOS/Linux without affecting read-only consumers."""

    index_dir.mkdir(parents=True, exist_ok=True)
    lock_path = index_dir / ".update.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Windows fallback
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except ImportError:  # pragma: no cover - Windows fallback
            pass
        handle.close()
