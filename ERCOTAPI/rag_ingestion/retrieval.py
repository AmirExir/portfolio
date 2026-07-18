"""Read-only collection filtering, authority-aware ranking, and citations."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .classify import authority_rank
from .config import Collection, IngestionConfig, default_config
from .store import generation_state, load_generation


LEGACY_SOURCES_BY_COLLECTION: dict[str, frozenset[str]] = {
    Collection.PLANNING.value: frozenset({"ercotaiassistant.txt"}),
    Collection.PROTOCOLS.value: frozenset({"ercotnodals.txt"}),
    Collection.RESOURCE_INTEGRATION.value: frozenset({"ercotRIhandbook.txt"}),
    Collection.DWG_SSWG.value: frozenset({"DWG_SSWG_Manuals.txt"}),
}


@dataclass(frozen=True)
class LoadedIndex:
    """An immutable snapshot suitable for a Streamlit cache or API process."""

    chunks: list[dict[str, Any]]
    embeddings: Any
    embedding_model: str
    generation_id: str | None
    source: str
    collections: tuple[str, ...]
    state_token: tuple[str | None, int]

    @property
    def ready(self) -> bool:
        return bool(self.chunks) and getattr(self.embeddings, "ndim", 0) == 2


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("ERCOT RAG retrieval requires the `numpy` package") from exc
    return np


def _normalize_collections(collections: str | Iterable[str] | None) -> tuple[str, ...]:
    if collections is None:
        return ()
    if isinstance(collections, str):
        values = (collections,)
    else:
        values = tuple(str(value) for value in collections)
    return tuple(sorted({value.strip() for value in values if value.strip()}))


def index_state(
    *,
    config: IngestionConfig | None = None,
    index_dir: Path | None = None,
) -> tuple[str | None, int]:
    """Return a cheap token that changes only after an atomic generation switch."""

    selected = config or default_config(index_dir=index_dir)
    return generation_state(selected.index_dir)


def _filter_rows(
    chunks: Sequence[Mapping[str, Any]],
    collections: tuple[str, ...],
) -> list[int]:
    if not collections:
        return list(range(len(chunks)))
    requested = set(collections)
    return [
        index
        for index, chunk in enumerate(chunks)
        if requested.intersection(str(value) for value in chunk.get("collections", []))
    ]


def _legacy_source_filter(collections: tuple[str, ...]) -> set[str] | None:
    if not collections or Collection.GENERAL.value in collections:
        return None
    selected: set[str] = set()
    mapped = False
    for collection in collections:
        names = LEGACY_SOURCES_BY_COLLECTION.get(collection)
        if names is not None:
            mapped = True
            selected.update(names)
    return selected if mapped else set()


def _load_legacy(
    chunks_path: Path,
    embeddings_path: Path,
    *,
    collections: tuple[str, ...],
    source_names: Iterable[str] | None,
    embedding_model: str,
) -> LoadedIndex:
    try:
        with chunks_path.open("r", encoding="utf-8") as handle:
            chunks = json.load(handle)
        np = _require_numpy()
        embeddings = np.load(embeddings_path, allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load legacy ERCOT retrieval cache: {exc}") from exc
    if not isinstance(chunks, list) or embeddings.ndim != 2:
        raise RuntimeError("Legacy ERCOT retrieval cache has invalid structures")
    if len(chunks) != int(embeddings.shape[0]):
        raise RuntimeError(
            f"Legacy ERCOT cache has {len(chunks)} chunks but {embeddings.shape[0]} vectors"
        )

    allowed = set(source_names) if source_names is not None else _legacy_source_filter(collections)
    if allowed is None:
        rows = list(range(len(chunks)))
    else:
        rows = [
            index
            for index, chunk in enumerate(chunks)
            if Path(str(chunk.get("source") or chunk.get("filename") or "")).name in allowed
        ]
    filtered_chunks: list[dict[str, Any]] = []
    for index in rows:
        chunk = dict(chunks[index])
        source = str(chunk.get("source") or chunk.get("filename") or "legacy")
        chunk.setdefault("source_path", source)
        chunk.setdefault("source", source)
        chunk.setdefault("filename", Path(source).name)
        chunk.setdefault("source_authority", "ERCOT")
        chunk.setdefault("source_kind", "ERCOT Reference")
        chunk.setdefault("is_generated", False)
        chunk.setdefault("collections", list(collections) or [Collection.GENERAL.value])
        chunk.setdefault("chunk_id", f"legacy-{index}")
        chunk.setdefault("chunk_index", index)
        filtered_chunks.append(chunk)
    matrix = np.asarray(embeddings[rows], dtype="float32")
    return LoadedIndex(
        chunks=filtered_chunks,
        embeddings=matrix,
        embedding_model=embedding_model,
        generation_id=None,
        source="legacy",
        collections=collections,
        state_token=(None, max(chunks_path.stat().st_mtime_ns, embeddings_path.stat().st_mtime_ns)),
    )


def load_index(
    collections: str | Iterable[str] | None = None,
    *,
    config: IngestionConfig | None = None,
    index_dir: Path | None = None,
    allow_legacy: bool = False,
    legacy_chunks_path: Path | None = None,
    legacy_embeddings_path: Path | None = None,
    legacy_source_names: Iterable[str] | None = None,
    legacy_embedding_model: str = "text-embedding-3-large",
) -> LoadedIndex:
    """Load a central collection; legacy JSON/NPY loading requires explicit opt-in."""

    selected = config or default_config(index_dir=index_dir)
    requested = _normalize_collections(collections)
    generation = load_generation(selected.index_dir)
    if generation is not None:
        np = _require_numpy()
        rows = _filter_rows(generation.chunks, requested)
        chunks = [dict(generation.chunks[index]) for index in rows]
        matrix = np.asarray(generation.embeddings[rows], dtype="float32")
        return LoadedIndex(
            chunks=chunks,
            embeddings=matrix,
            embedding_model=str(
                generation.manifest.get("embedding_model") or selected.embedding_model
            ),
            generation_id=generation.generation_id,
            source="central",
            collections=requested,
            state_token=generation_state(selected.index_dir),
        )

    if not allow_legacy:
        np = _require_numpy()
        return LoadedIndex(
            chunks=[],
            embeddings=np.empty((0, 0), dtype="float32"),
            embedding_model=selected.embedding_model,
            generation_id=None,
            source="missing",
            collections=requested,
            state_token=(None, 0),
        )

    chunks_path = legacy_chunks_path or selected.legacy_chunks_path
    embeddings_path = legacy_embeddings_path or selected.legacy_embeddings_path
    if not chunks_path or not embeddings_path or not chunks_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            "No active central ERCOT RAG generation and no complete legacy JSON/NPY fallback"
        )
    return _load_legacy(
        chunks_path,
        embeddings_path,
        collections=requested,
        source_names=legacy_source_names,
        embedding_model=legacy_embedding_model,
    )


load_collection = load_index


def _openai_query_vector(question: str, model: str, client: Any | None) -> Any:
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required to embed an ERCOT retrieval query")
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("ERCOT query embedding requires the `openai` package") from exc
        client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=model, input=question)
    return response.data[0].embedding


def _query_vector(
    question: str,
    model: str,
    query_embedder: Callable[[str], Any] | Any | None,
    client: Any | None,
) -> Any:
    np = _require_numpy()
    if query_embedder is None:
        value = _openai_query_vector(question, model, client)
    elif callable(query_embedder):
        value = query_embedder(question)
    elif hasattr(query_embedder, "embed_query"):
        value = query_embedder.embed_query(question)
    elif hasattr(query_embedder, "embed_texts"):
        value = query_embedder.embed_texts([question])
    else:
        raise TypeError("query_embedder must be callable or expose embed_query/embed_texts")
    vector = np.asarray(value, dtype="float32")
    if vector.ndim == 2 and vector.shape[0] == 1:
        vector = vector[0]
    if vector.ndim != 1:
        raise ValueError(f"Query embedder returned invalid shape {vector.shape}")
    return vector


def _cosine_scores(query: Any, embeddings: Any) -> Any:
    np = _require_numpy()
    matrix = np.asarray(embeddings, dtype="float32")
    if matrix.ndim != 2 or query.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: query is {query.shape[0]}, index is "
            f"{matrix.shape[1] if matrix.ndim == 2 else 'invalid'}"
        )
    query_norm = float(np.linalg.norm(query))
    row_norms = np.linalg.norm(matrix, axis=1)
    denominator = row_norms * query_norm
    denominator = np.where(denominator == 0, 1.0, denominator)
    return matrix.dot(query) / denominator


_QUERY_STOP_WORDS = frozenset(
    {
        "about",
        "and",
        "does",
        "ercot",
        "for",
        "from",
        "guide",
        "how",
        "mean",
        "section",
        "that",
        "the",
        "this",
        "what",
        "when",
        "where",
        "why",
        "with",
    }
)


# Each route contributes only a small tie-breaking signal. Vector similarity remains
# the primary score, while domain terms can resolve near-ties between ERCOT manuals.
_DOMAIN_ROUTES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("planning guide", "planning guides"),
        ("planning guide", "planning guides", "ercot planning guide", "pgrr"),
    ),
    (
        ("nodal operating guide", "operating guide", "operating guides"),
        ("nodal operating guide", "operating guide", "operating guides", "nogrr"),
    ),
    (
        ("nodal protocol", "nodal protocols"),
        ("nodal protocol", "nodal protocols", "nprr"),
    ),
    (
        (
            "generator interconnection",
            "generation interconnection",
            "resource interconnection",
            "interconnection process",
        ),
        (
            "generator interconnection",
            "generation interconnection",
            "resource interconnection",
            "resource interconnection handbook",
            "generation interconnection process",
            "ginr",
        ),
    ),
    (
        ("ginr", "generation interconnection or change request"),
        ("ginr", "generation interconnection or change request"),
    ),
    (
        ("fis", "full interconnection study"),
        ("fis", "full interconnection study"),
    ),
)


_CURRENT_UPLOAD_DOMAINS = {
    "planning_guide_uploads": "planning",
    "nodal_protocol_uploads": "protocols",
    "dwg_sswg_uploads": "dwg",
}
_CURRENT_UPLOAD_PATH_DOMAINS = {
    "ERCOTAPI/sources/official/planning_guides/": "planning",
    "ERCOTAPI/sources/official/nodal_protocols/": "protocols",
    "ERCOTAPI/sources/official/dwg_sswg/": "dwg",
}
_HISTORICAL_ROOT_DOMAINS = {
    "planning_guides": "planning",
    "nodal_protocols": "protocols",
}
_HISTORICAL_CANONICAL_DOMAINS = {
    "DWG": "dwg",
    "PLANNING GUIDE": "planning",
    "PROTOCOL": "protocols",
}
_COMBINED_DWG_SSWG_FILENAME = "dwg_sswg_manuals.txt"
_DWG_MANUAL_HEADING_RE = re.compile(
    r"\bdynamics\s+working\s+group\s+procedure\s+manual\b",
    re.IGNORECASE,
)


def _normalize_query(question: str) -> str:
    normalized = question.lower()
    normalized = re.sub(r"\b(?:plannig|planing)\b", "planning", normalized)
    normalized = re.sub(r"[-_/]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _query_terms(question: str) -> set[str]:
    """Return meaningful terms without splitting decimal section identifiers."""

    return {
        token
        for token in re.findall(r"[a-z0-9]+(?:\.[0-9]+)*", _normalize_query(question))
        if len(token) > 2 and token not in _QUERY_STOP_WORDS
    }


def _requests_historical_material(question: str) -> bool:
    normalized = _normalize_query(question)
    years = {int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", normalized)}
    if any(year < datetime.now().year for year in years):
        return True
    return bool(
        re.search(
            r"\b(?:former|formerly|historical|history|old|older|previous|prior|superseded)\b"
            r"|\brevision\s+\d{1,3}[a-z]?\b",
            normalized,
        )
    )


def _combined_manual_key(chunk: Mapping[str, Any]) -> str | None:
    category = str(chunk.get("source_category") or "")
    if category not in {"authoritative_static", "dwg_sswg_manuals"}:
        return None
    source_path = str(chunk.get("source_path") or chunk.get("source") or "")
    filename = source_path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if filename != _COMBINED_DWG_SSWG_FILENAME:
        return None
    return str(chunk.get("content_hash") or source_path)


def _chunk_index(chunk: Mapping[str, Any]) -> int:
    try:
        return int(chunk.get("chunk_index") or 0)
    except (TypeError, ValueError):
        return 0


def _combined_dwg_starts(
    chunks: Sequence[Mapping[str, Any]],
    rows: Sequence[int],
) -> dict[str, int]:
    """Locate the DWG boundary inside the historical combined SSWG/DWG file."""

    starts: dict[str, int] = {}
    for row in rows:
        chunk = chunks[row]
        key = _combined_manual_key(chunk)
        if key is None or not _DWG_MANUAL_HEADING_RE.search(str(chunk.get("text") or "")):
            continue
        index = _chunk_index(chunk)
        starts[key] = min(index, starts.get(key, index))
    return starts


def _chunk_paths(chunk: Mapping[str, Any]) -> tuple[str, ...]:
    values = [
        str(chunk.get("source_path") or chunk.get("source") or ""),
        *(str(value) for value in (chunk.get("aliases") or []) if value),
    ]
    return tuple(dict.fromkeys(value.replace("\\", "/") for value in values if value))


def _current_upload_domain(chunk: Mapping[str, Any]) -> str | None:
    category = str(chunk.get("source_category") or "")
    if category in _CURRENT_UPLOAD_DOMAINS:
        return _CURRENT_UPLOAD_DOMAINS[category]
    for path in _chunk_paths(chunk):
        for marker, domain in _CURRENT_UPLOAD_PATH_DOMAINS.items():
            if marker in path:
                return domain
    return None


def _historical_domain(
    chunk: Mapping[str, Any],
    combined_dwg_starts: Mapping[str, int],
) -> str | None:
    # Content-addressed monitor files can be byte-identical to a checked-in
    # current upload.  Their alias list preserves that current path, so do not
    # demote the canonical archived copy as historical.
    if _current_upload_domain(chunk) is not None:
        return None
    if _is_combined_historical_dwg(chunk, combined_dwg_starts):
        return "dwg"
    category = str(chunk.get("source_category") or "")
    if category in _HISTORICAL_ROOT_DOMAINS:
        return _HISTORICAL_ROOT_DOMAINS[category]
    kind = str(chunk.get("source_kind") or "").upper()
    if category == "dwg_sswg_manuals":
        return "dwg" if kind == "DWG" else None
    # The monitor archive can contain complete historical copies of the
    # Planning Guide, Nodal Protocols, and DWG manual alongside notices and
    # revision requests.  When a checked-in current bundle exists, treat only
    # those complete manual copies like the canonical static history.  Other
    # official downloads (NPRRs, PGRRs, notices, committee files, and so on)
    # remain eligible for ordinary current retrieval.
    if category not in {"authoritative_static", "official_downloads"}:
        return None
    return _HISTORICAL_CANONICAL_DOMAINS.get(kind)


def _is_combined_historical_dwg(
    chunk: Mapping[str, Any],
    combined_dwg_starts: Mapping[str, int],
) -> bool:
    combined_key = _combined_manual_key(chunk)
    if combined_key is not None and combined_key in combined_dwg_starts:
        return _chunk_index(chunk) >= combined_dwg_starts[combined_key]
    return False


def _prefer_current_rows(
    question: str,
    chunks: Sequence[Mapping[str, Any]],
    rows: Sequence[int],
) -> list[int]:
    """Hide superseded static corpora when a current domain bundle is present.

    Historical chunks remain in the central generation and become eligible
    when the question names a prior year or explicitly asks for prior/history
    text. The historical combined SSWG/DWG file is filtered at its internal
    DWG heading so its distinct SSWG section remains available.
    """

    if _requests_historical_material(question):
        return list(rows)
    combined_dwg_starts = _combined_dwg_starts(chunks, rows)
    current_domains = {
        domain for row in rows if (domain := _current_upload_domain(chunks[row])) is not None
    }
    if not current_domains and not combined_dwg_starts:
        return list(rows)
    return [
        row
        for row in rows
        if not _is_combined_historical_dwg(chunks[row], combined_dwg_starts)
        and (
            (domain := _historical_domain(chunks[row], combined_dwg_starts)) is None
            or domain not in current_domains
        )
    ]


def _requested_section_spec(question: str) -> tuple[str, re.Pattern[str]] | None:
    normalized = _normalize_query(question)
    if "planning guide" in normalized:
        domain = "planning"
    elif "nodal protocol" in normalized:
        domain = "protocols"
    else:
        return None
    match = re.search(r"\bsection\s*(\d+)(?:\.\d+)?([a-z]?)\b", normalized)
    if match is None:
        return None
    requested = f"{int(match.group(1))}{match.group(2)}"
    return domain, re.compile(
        rf"^0*{re.escape(requested)}(?:[a-z])?(?:[-_.\s]|$)",
        re.IGNORECASE,
    )


def _matches_requested_section(
    chunk: Mapping[str, Any],
    spec: tuple[str, re.Pattern[str]],
) -> bool:
    domain, prefix = spec
    return _current_upload_domain(chunk) == domain and any(
        prefix.search(Path(path).name) for path in _chunk_paths(chunk)
    )


def _prefer_requested_section_bundle(
    question: str,
    chunks: Sequence[Mapping[str, Any]],
    rows: Sequence[int],
) -> list[int]:
    """Limit a split current manual to the explicitly requested section file.

    Every split Planning Guide and Nodal Protocol file contains a table of
    contents, so vector and lexical matching alone can rank Section 1 above a
    request for Section 9.  Keep non-manual material (for example PGRRs and
    notices), but discard sibling files from the same current manual bundle
    when the requested section file is present.
    """

    spec = _requested_section_spec(question)
    if spec is None:
        return list(rows)
    domain, _ = spec
    matching = {
        row
        for row in rows
        if _matches_requested_section(chunks[row], spec)
    }
    if not matching:
        return list(rows)
    return [
        row
        for row in rows
        if _current_upload_domain(chunks[row]) != domain or row in matching
    ]


def _contains_token(haystack: str, token: str) -> bool:
    return bool(
        re.search(
            rf"(?<![a-z0-9.]){re.escape(token)}(?![a-z0-9.])",
            haystack,
        )
    )


def _contains_phrase(haystack: str, phrase: str) -> bool:
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])",
            haystack,
        )
    )


def _lexical_boost(question: str, chunk: Mapping[str, Any]) -> float:
    normalized_question = _normalize_query(question)
    haystack = _normalize_query(
        " ".join(
            str(chunk.get(key, ""))
            for key in (
                "title",
                "document_number",
                "source_kind",
                "filename",
                "source_path",
                "text",
            )
        )
    ).lower()

    terms = _query_terms(normalized_question)
    term_boost = 0.0
    if terms:
        matched = sum(1 for term in terms if _contains_token(haystack, term))
        term_boost = 0.02 * matched / len(terms)

    route_matches = sum(
        1
        for query_phrases, document_phrases in _DOMAIN_ROUTES
        if any(_contains_phrase(normalized_question, phrase) for phrase in query_phrases)
        and any(_contains_phrase(haystack, phrase) for phrase in document_phrases)
    )
    route_boost = min(0.036, 0.018 * route_matches)

    section_references = set(
        re.findall(r"(?:\bsection\s*|§\s*)(\d+(?:\.\d+)*)", normalized_question)
    )
    section_matches = sum(
        1 for section in section_references if _contains_token(haystack, section)
    )
    section_boost = min(0.036, 0.03 * section_matches)

    # Keep all lexical routing bounded below ordinary vector-score differences.
    return min(0.07, term_boost + route_boost + section_boost)


def _ranking_score(question: str, vector_score: float, chunk: Mapping[str, Any]) -> float:
    trust = authority_rank(chunk)
    authority_boost = 0.04 if trust == 2 else (-0.04 if trust == 0 else 0.0)
    status = str(chunk.get("document_status", "")).lower()
    status_boost = 0.01 if status in {"approved", "effective", "clean"} else 0.0
    if status in {"withdrawn", "rejected"}:
        status_boost -= 0.015
    stale_penalty = -0.05 if chunk.get("stale") else 0.0
    return (
        vector_score
        + authority_boost
        + status_boost
        + stale_penalty
        + _lexical_boost(question, chunk)
    )


def _version_rank(chunk: Mapping[str, Any]) -> tuple[int, int]:
    """Prefer newer effective dates/revisions after relevance and trust tie."""

    date_rank = 0
    for value in (
        chunk.get("effective_date"),
        chunk.get("published_date"),
        chunk.get("downloaded_at"),
    ):
        candidate = str(value or "").strip()
        for pattern in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
            try:
                parsed = datetime.strptime(candidate[:10] if pattern == "%Y-%m-%d" else candidate, pattern)
            except ValueError:
                continue
            date_rank = int(parsed.strftime("%Y%m%d"))
            break
        if date_rank:
            break
    revision_values = re.findall(r"\d+", str(chunk.get("revision") or ""))
    revision_rank = int(revision_values[-1]) if revision_values else 0
    return date_rank, revision_rank


def retrieve_chunks(
    question: str,
    index: LoadedIndex,
    *,
    top_k: int = 8,
    collections: str | Iterable[str] | None = None,
    query_embedder: Callable[[str], Any] | Any | None = None,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    """Retrieve chunks, prioritizing authoritative ERCOT sources over summaries."""

    if not question.strip():
        raise ValueError("Retrieval question cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    requested = _normalize_collections(collections)
    rows = _filter_rows(index.chunks, requested) if requested else list(range(len(index.chunks)))
    rows = _prefer_current_rows(question, index.chunks, rows)
    rows = _prefer_requested_section_bundle(question, index.chunks, rows)
    if not rows:
        return []
    np = _require_numpy()
    matrix = np.asarray(index.embeddings[rows], dtype="float32")
    query = _query_vector(question, index.embedding_model, query_embedder, client)
    scores = _cosine_scores(query, matrix)
    section_spec = _requested_section_spec(question)
    ranked: list[tuple[int, float, float, int, int, int]] = []
    for local_index, source_index in enumerate(rows):
        vector_score = float(scores[local_index])
        date_rank, revision_rank = _version_rank(index.chunks[source_index])
        ranked.append(
            (
                int(
                    section_spec is not None
                    and _matches_requested_section(index.chunks[source_index], section_spec)
                ),
                _ranking_score(question, vector_score, index.chunks[source_index]),
                vector_score,
                date_rank,
                revision_rank,
                source_index,
            )
        )
    ranked.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            -item[2],
            -item[3],
            -item[4],
            str(index.chunks[item[5]].get("chunk_id", "")),
        )
    )
    results: list[dict[str, Any]] = []
    for _, score, vector_score, _, _, source_index in ranked[: min(top_k, len(ranked))]:
        chunk = dict(index.chunks[source_index])
        chunk["retrieval_score"] = score
        chunk["vector_score"] = vector_score
        chunk["citation"] = format_citation(chunk)
        results.append(chunk)
    return results


retrieve = retrieve_chunks


def format_citation(chunk: Mapping[str, Any]) -> str:
    """Format a compact citation with trust, version, path, and original URL."""

    authority = str(chunk.get("source_authority") or "Unknown source")
    kind = str(chunk.get("source_kind") or "Document")
    number = str(chunk.get("document_number") or "").strip()
    title = str(chunk.get("title") or chunk.get("filename") or "Untitled").strip()
    path = str(chunk.get("source_path") or chunk.get("source") or "unknown").strip()
    chunk_number = int(chunk.get("chunk_index", 0)) + 1
    parts = [authority, kind]
    if chunk.get("stale"):
        parts.append("STALE INDEXED COPY")
    if number:
        parts.append(number)
    for label, field in (
        ("Status", "document_status"),
        ("Effective", "effective_date"),
        ("Published", "published_date"),
        ("Revision", "revision"),
    ):
        value = str(chunk.get(field) or "").strip()
        if value:
            parts.append(f"{label} {value}")
    parts.extend((title, path, f"chunk {chunk_number}"))
    original_url = str(chunk.get("original_url") or "").strip()
    if original_url:
        parts.append(original_url)
    return "[" + " | ".join(parts) + "]"


def format_context(
    chunks: Sequence[Mapping[str, Any]],
    *,
    max_words: int | None = None,
) -> str:
    """Build grounded LLM context while retaining a citation per chunk."""

    blocks: list[str] = []
    used_words = 0
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        words = text.split()
        if max_words is not None and used_words + len(words) > max_words:
            remaining = max_words - used_words
            if remaining <= 0:
                break
            text = " ".join(words[:remaining])
            words = words[:remaining]
        blocks.append(f"Citation: {format_citation(chunk)}\n\n{text}")
        used_words += len(words)
    return "\n\n---\n\n".join(blocks)


def format_source_list(
    chunks: Sequence[Mapping[str, Any]],
    *,
    max_sources: int = 8,
) -> str:
    """Return a deterministic, deduplicated Markdown source footer."""

    if max_sources < 1:
        return ""
    lines: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        citation = str(chunk.get("citation") or format_citation(chunk))
        original_url = str(chunk.get("original_url") or "").strip()
        source_urls = tuple(
            dict.fromkeys(
                value
                for value in (
                    original_url,
                    *(str(item).strip() for item in (chunk.get("url_aliases") or [])),
                )
                if re.fullmatch(r"https?://[^\s]+", value)
            )
        )
        identity = str(
            chunk.get("document_id")
            or chunk.get("content_hash")
            or chunk.get("source_path")
            or chunk.get("source")
            or citation
        )
        if identity in seen:
            continue
        seen.add(identity)
        line = f"- {citation}"
        for index, source_url in enumerate(source_urls[:3]):
            label = "open ERCOT source" if index == 0 else f"alternate source {index + 1}"
            line += f" — [{label}](<{source_url}>)"
        lines.append(line)
        if len(lines) >= max_sources:
            break
    return "\n\n**Retrieved sources**\n\n" + "\n".join(lines) if lines else ""
