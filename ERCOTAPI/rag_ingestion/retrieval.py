"""Read-only collection filtering, authority-aware ranking, and citations."""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .classify import authority_rank
from .change_tracking import compare_document_versions, logical_document_key
from .config import Collection, IngestionConfig, default_config
from .requirements import (
    analyze_question,
    annotate_evidence,
    answer_contract,
    diversify_evidence,
    evidence_summary,
    is_notice,
    lifecycle_metadata,
)
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
        ),
        (
            "generator interconnection",
            "generation interconnection",
            "resource interconnection",
            "resource interconnection handbook",
            "ercotrihandbook",
            "generation interconnection process",
            "ginr",
        ),
    ),
    (
        ("interconnection process", "interconnect in ercot"),
        (
            "generator interconnection",
            "resource interconnection handbook",
            "ercotrihandbook",
            "large load interconnection",
            "batch zero",
            "planning guide section 5",
            "planning guide section 9",
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


_GENERATION_INTERCONNECTION_TERMS = (
    "generator",
    "generation",
    "resource interconnection",
    "solar",
    "wind",
    "bess",
    "energy storage resource",
    "esr",
    "ginr",
    "gim",
    "full interconnection study",
    "fis",
)
_LOAD_INTERCONNECTION_TERMS = (
    "large load",
    "load interconnection",
    "data center",
    "ille",
    "batch zero",
)
_INTERCONNECTION_PROCESS_ALIASES = (
    "ginr",
    "gim",
    "full interconnection study",
    "fis",
    "batch zero",
)
_INTERCONNECTION_SPECIFIC_TOPICS = (
    "agreement",
    "breaker",
    "capability",
    "commissioning",
    "contingency",
    "cost",
    "deadline",
    "dynamic",
    "energization",
    "equipment",
    "fault",
    "fee",
    "model",
    "ownership",
    "point of interconnection",
    "poi",
    "protection",
    "pscad",
    "psse",
    "reactive",
    "relay",
    "responsibility",
    "ride through",
    "security screening",
    "sgia",
    "short circuit",
    "site control",
    "stability",
    "steady state",
    "study requirement",
    "synchronization",
    "var",
    "voltage",
)
_GENERATION_INTERCONNECTION_EXPANSION = (
    "Generator Interconnection or Modification GIM Planning Guide Section 5 "
    "applicability initiation RIOO Security Screening Study Full Interconnection "
    "Study FIS scoping steady state short circuit stability facilities SGIA "
    "registration modeling energization synchronization commissioning Resource "
    "Interconnection Handbook"
)
_LOAD_INTERCONNECTION_EXPANSION = (
    "Large Load Interconnection or Modification Planning Guide Section 9 "
    "applicability submission Batch Zero Interconnection Study allocation "
    "refinement transmission plan Load Commissioning Plan initial energization"
)


def _interconnection_facets(question: str) -> tuple[str, ...]:
    """Resolve generator/load tracks for an interconnection-process question."""

    normalized = _normalize_query(question)
    generation = any(
        _contains_phrase(normalized, term) for term in _GENERATION_INTERCONNECTION_TERMS
    )
    load = any(_contains_phrase(normalized, term) for term in _LOAD_INTERCONNECTION_TERMS)
    process_alias = any(
        _contains_phrase(normalized, term) for term in _INTERCONNECTION_PROCESS_ALIASES
    )
    if "interconnect" not in normalized and not process_alias:
        return ()
    # An unqualified ERCOT interconnection question is ambiguous. Retrieve both
    # governing tracks so the answer can distinguish generation from Large Load.
    if not generation and not load:
        return ("generation", "load")
    facets: list[str] = []
    if generation:
        facets.append("generation")
    if load:
        facets.append("load")
    return tuple(facets)


def _expanded_retrieval_query(question: str) -> str:
    """Add official ERCOT process language without changing the user's intent."""

    if not _is_broad_interconnection_process_question(question):
        return question.strip()
    facets = _interconnection_facets(question)
    expansions: list[str] = [question.strip()]
    if "generation" in facets:
        expansions.append(_GENERATION_INTERCONNECTION_EXPANSION)
    if "load" in facets:
        expansions.append(_LOAD_INTERCONNECTION_EXPANSION)
    return "\n".join(expansions)


def _is_broad_interconnection_process_question(question: str) -> bool:
    """Return true only when the user asks for an end-to-end process overview."""

    facets = _interconnection_facets(question)
    if not facets or _requests_historical_material(question):
        return False
    normalized = _normalize_query(question)
    if any(
        _contains_phrase(normalized, term)
        for term in _INTERCONNECTION_SPECIFIC_TOPICS
    ):
        return False
    return bool(
        re.search(
            r"\b(?:process|procedures?|steps?|stages?|timeline|overview|workflow)\b",
            normalized,
        )
        or (
            normalized.startswith(("explain ", "describe "))
            and "interconnect" in normalized
        )
        or re.search(r"\bhow\b.*\binterconnect", normalized)
    )


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
            r"\b(?:change|changed|changes|compare|comparison|difference|former|formerly|"
            r"historical|history|old|older|previous|prior|redline|superseded)\b"
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


@dataclass(frozen=True)
class _LexicalQueryProfile:
    normalized_question: str
    terms: frozenset[str]
    route_document_phrases: tuple[tuple[str, ...], ...]
    section_references: frozenset[str]


def _lexical_query_profile(question: str) -> _LexicalQueryProfile:
    normalized = _normalize_query(question)
    routes = tuple(
        document_phrases
        for query_phrases, document_phrases in _DOMAIN_ROUTES
        if any(_contains_phrase(normalized, phrase) for phrase in query_phrases)
    )
    sections = frozenset(
        re.findall(r"(?:\bsection\s*|§\s*)(\d+(?:\.\d+)*)", normalized)
    )
    return _LexicalQueryProfile(
        normalized_question=normalized,
        terms=frozenset(_query_terms(normalized)),
        route_document_phrases=routes,
        section_references=sections,
    )


def _lexical_boost(
    question: str,
    chunk: Mapping[str, Any],
    *,
    profile: _LexicalQueryProfile | None = None,
) -> float:
    query_profile = profile or _lexical_query_profile(question)
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
    haystack_terms = set(re.findall(r"[a-z0-9]+(?:\.[0-9]+)*", haystack))

    term_boost = 0.0
    if query_profile.terms:
        matched = len(query_profile.terms.intersection(haystack_terms))
        term_boost = 0.02 * matched / len(query_profile.terms)

    route_matches = sum(
        1
        for document_phrases in query_profile.route_document_phrases
        if any(phrase in haystack for phrase in document_phrases)
    )
    route_boost = min(0.036, 0.018 * route_matches)

    section_matches = sum(
        1 for section in query_profile.section_references if section in haystack_terms
    )
    section_boost = min(0.036, 0.03 * section_matches)

    # Keep all lexical routing bounded below ordinary vector-score differences.
    return min(0.07, term_boost + route_boost + section_boost)


def _ranking_score(
    question: str,
    vector_score: float,
    chunk: Mapping[str, Any],
    *,
    as_of: date | datetime | str | None = None,
    question_analysis: Any | None = None,
    lexical_profile: _LexicalQueryProfile | None = None,
) -> float:
    trust = authority_rank(chunk)
    authority_boost = 0.04 if trust == 2 else (-0.04 if trust == 0 else 0.0)
    state = lifecycle_metadata(chunk, as_of=as_of)["effective_state"]
    status_boost = {
        "effective": 0.035,
        "approved_procedure": 0.015,
        "implemented_change_record": 0.005,
        "effectiveness_unknown": -0.005,
        "effective_edition_currentness_unverified": -0.01,
        "approved_effectiveness_unverified": -0.02,
        "approved_not_effective": -0.055,
        "proposed_or_pending": -0.06,
        "not_effective": -0.10,
    }.get(str(state), 0.0)
    analysis = question_analysis or analyze_question(question, as_of=as_of)
    if analysis.asks_for_changes or analysis.asks_for_history or analysis.asks_for_status:
        status_boost = max(status_boost, -0.015)
    stale_penalty = -0.05 if chunk.get("stale") else 0.0
    return (
        vector_score
        + authority_boost
        + status_boost
        + stale_penalty
        + _lexical_boost(question, chunk, profile=lexical_profile)
    )


def _version_rank(chunk: Mapping[str, Any]) -> tuple[int, int, int]:
    """Prefer newer effective dates/revisions after relevance and trust tie."""

    current_rank = int(_current_upload_domain(chunk) is not None)
    date_rank = 0
    lifecycle = lifecycle_metadata(chunk)
    for value in (
        lifecycle.get("resolved_effective_date"),
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
    return current_rank, date_rank, revision_rank


def retrieve_chunks(
    question: str,
    index: LoadedIndex,
    *,
    top_k: int = 8,
    collections: str | Iterable[str] | None = None,
    query_embedder: Callable[[str], Any] | Any | None = None,
    client: Any | None = None,
    as_of: date | datetime | str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve chunks, prioritizing authoritative ERCOT sources over summaries."""

    if not question.strip():
        raise ValueError("Retrieval question cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    requested = _normalize_collections(collections)
    rows = _filter_rows(index.chunks, requested) if requested else list(range(len(index.chunks)))
    rows = [row for row in rows if not is_notice(index.chunks[row])]
    rows = _prefer_current_rows(question, index.chunks, rows)
    rows = _prefer_requested_section_bundle(question, index.chunks, rows)
    if not rows:
        return []
    np = _require_numpy()
    matrix = np.asarray(index.embeddings[rows], dtype="float32")
    query = _query_vector(
        _expanded_retrieval_query(question),
        index.embedding_model,
        query_embedder,
        client,
    )
    scores = _cosine_scores(query, matrix)
    section_spec = _requested_section_spec(question)
    question_analysis = analyze_question(question, as_of=as_of)
    lexical_profile = _lexical_query_profile(question)
    pool_limit = min(len(rows), max(top_k * 20, 300))
    if pool_limit < len(rows):
        local_candidates = set(
            int(value)
            for value in np.argpartition(scores, -pool_limit)[-pool_limit:]
        )
        # Exact identifiers and metadata matches are cheap safeguards against a
        # semantic miss. Full chunk-text lexical scoring is reserved for the
        # bounded candidate pool, which keeps large saved indexes responsive.
        for local_index, source_index in enumerate(rows):
            chunk = index.chunks[source_index]
            document_number = str(chunk.get("document_number") or "").upper()
            if document_number and document_number in question_analysis.requested_documents:
                local_candidates.add(local_index)
                continue
            metadata_text = _normalize_query(
                " ".join(
                    str(chunk.get(field) or "")
                    for field in ("title", "source_kind", "filename", "source_path")
                )
            )
            route_metadata_match = any(
                phrase in metadata_text
                for document_phrases in lexical_profile.route_document_phrases
                for phrase in document_phrases
            )
            if route_metadata_match:
                local_candidates.add(local_index)
                continue
            if lexical_profile.terms and len(
                lexical_profile.terms.intersection(
                    re.findall(r"[a-z0-9]+(?:\.[0-9]+)*", metadata_text)
                )
            ) >= min(2, len(lexical_profile.terms)):
                local_candidates.add(local_index)
        candidate_indices = sorted(local_candidates)
    else:
        candidate_indices = list(range(len(rows)))
    ranked: list[tuple[int, float, float, int, int, int, int]] = []
    for local_index in candidate_indices:
        source_index = rows[local_index]
        vector_score = float(scores[local_index])
        current_rank, date_rank, revision_rank = _version_rank(index.chunks[source_index])
        ranked.append(
            (
                int(
                    section_spec is not None
                    and _matches_requested_section(index.chunks[source_index], section_spec)
                ),
                _ranking_score(
                    question,
                    vector_score,
                    index.chunks[source_index],
                    as_of=as_of,
                    question_analysis=question_analysis,
                    lexical_profile=lexical_profile,
                ),
                vector_score,
                current_rank,
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
            -item[5],
            str(index.chunks[item[6]].get("chunk_id", "")),
        )
    )
    results: list[dict[str, Any]] = []
    for _, score, vector_score, _, _, _, source_index in ranked[: min(top_k, len(ranked))]:
        chunk = annotate_evidence(
            index.chunks[source_index],
            as_of=question_analysis.as_of,
            requested_sections=question_analysis.requested_sections,
        )
        chunk["retrieval_score"] = score
        chunk["vector_score"] = vector_score
        chunk["citation"] = format_citation(chunk)
        results.append(chunk)
    return results


retrieve = retrieve_chunks


def _planning_section_path(chunk: Mapping[str, Any], section: int) -> bool:
    if _current_upload_domain(chunk) != "planning":
        return False
    prefix = re.compile(rf"^0*{section}(?:[-_.\s]|$)", re.IGNORECASE)
    return any(prefix.search(Path(path).name) for path in _chunk_paths(chunk))


def _interconnection_anchor(
    chunk: Mapping[str, Any],
    facets: Sequence[str],
) -> tuple[str, str, str, int] | None:
    """Identify a process-stage anchor in the controlled ERCOT corpus."""

    generation_guide = (
        "generation" in facets and _planning_section_path(chunk, 5)
    )
    load_guide = "load" in facets and _planning_section_path(chunk, 9)
    path_names = {
        Path(path).name.lower() for path in _chunk_paths(chunk)
    }
    resource_handbook = (
        "generation" in facets and "ercotrihandbook.txt" in path_names
    )
    if not generation_guide and not load_guide and not resource_handbook:
        return None

    text = " ".join(str(chunk.get("text") or "").split()).lower()
    if generation_guide:
        if "must initiate a generator interconnection or modification" in text:
            return (
                "generation-initiation",
                "5.2.2",
                "Initiating a Generator Interconnection or Modification",
                text.index("must initiate a generator interconnection or modification"),
            )
        if (
            "the provisions in this section establish the procedures for conducting "
            "the security screening study and full interconnection" in text
        ):
            return (
                "generation-studies",
                "5.3",
                "Interconnection Study Procedures for Large Generators",
                text.index(
                    "the provisions in this section establish the procedures for conducting "
                    "the security screening study and full interconnection"
                ),
            )
        commissioning = (
            "5.5generator commissioning and continuing operations "
            "(1)for each interconnecting"
        )
        if commissioning in text:
            return (
                "generation-commissioning",
                "5.5",
                "Generator Commissioning and Continuing Operations",
                text.index(commissioning),
            )

    if (
        resource_handbook
        and "divided into the following three stages" in text
    ):
        return (
            "generation-handbook",
            "",
            "Resource Interconnection Handbook — three-stage process",
            text.index("divided into the following three stages"),
        )

    if load_guide:
        introduction = "defines the requirements and processes used to facilitate new or modified large load"
        if introduction in text:
            return (
                "load-introduction",
                "9.1",
                "Large Load Interconnection or Modification — Introduction",
                text.index(introduction),
            )
        overview = "9.3.1batch zero process overview and timelines"
        if overview in text:
            return (
                "load-batch-zero",
                "9.3.1",
                "Batch Zero Process Overview and Timelines",
                text.index(overview),
            )
        refinement = "9.5batch zero study refinement and delivery of transmission plan"
        if refinement in text:
            return (
                "load-refinement",
                "9.5",
                "Batch Zero Study Refinement and Delivery of Transmission Plan",
                text.index(refinement),
            )
    return None


def _augment_interconnection_candidates(
    question: str,
    index: LoadedIndex,
    candidates: Sequence[Mapping[str, Any]],
    *,
    collections: str | Iterable[str] | None = None,
    as_of: date | datetime | str | None = None,
) -> list[dict[str, Any]]:
    """Guarantee broad process questions contain each governing process stage."""

    if not _is_broad_interconnection_process_question(question):
        return [dict(chunk) for chunk in candidates]
    facets = _interconnection_facets(question)
    requested = set(_normalize_collections(collections))
    selected_as_of = analyze_question(question, as_of=as_of).as_of
    best_by_anchor: dict[
        str,
        tuple[tuple[int, int, int], int, Mapping[str, Any], str, str],
    ] = {}
    for chunk in index.chunks:
        if is_notice(chunk):
            continue
        if (
            _current_upload_domain(chunk) == "planning"
            and lifecycle_metadata(chunk, as_of=as_of)["effective_state"] != "effective"
        ):
            continue
        if requested and not requested.intersection(
            str(value) for value in chunk.get("collections", [])
        ):
            continue
        match = _interconnection_anchor(chunk, facets)
        if match is None:
            continue
        anchor_id, section_number, section_title, marker_position = match
        if (
            anchor_id == "generation-handbook"
            and selected_as_of != date.today().isoformat()
        ):
            # The saved Handbook has no reliable effective-date metadata. Do
            # not force the current snapshot into a historical/future answer.
            continue
        candidate = (
            _version_rank(chunk),
            -marker_position,
            chunk,
            section_number,
            section_title,
        )
        previous = best_by_anchor.get(anchor_id)
        if previous is None or candidate[:2] > previous[:2]:
            best_by_anchor[anchor_id] = candidate

    augmented = [dict(chunk) for chunk in candidates]
    positions = {
        str(chunk.get("chunk_id") or chunk.get("id") or ""): position
        for position, chunk in enumerate(augmented)
    }
    top_score = max(
        (float(chunk.get("retrieval_score", 0.0)) for chunk in augmented),
        default=0.0,
    )
    anchor_order = (
        "generation-initiation",
        "generation-studies",
        "generation-commissioning",
        "generation-handbook",
        "load-introduction",
        "load-batch-zero",
        "load-refinement",
    )
    for offset, anchor_id in enumerate(anchor_order):
        match = best_by_anchor.get(anchor_id)
        if match is None:
            continue
        _, _, raw_chunk, section_number, section_title = match
        chunk_id = str(raw_chunk.get("chunk_id") or raw_chunk.get("id") or "")
        if chunk_id in positions:
            anchor_chunk = augmented[positions[chunk_id]]
        else:
            anchor_chunk = dict(raw_chunk)
            positions[chunk_id] = len(augmented)
            augmented.append(anchor_chunk)
        # Keep deterministic process anchors inside the relevance window while
        # retaining the original vector score for diagnostics.
        anchor_chunk["retrieval_score"] = max(
            float(anchor_chunk.get("retrieval_score", 0.0)),
            top_score + 0.04 - offset * 0.003,
        )
        if section_number:
            anchor_chunk["section_number"] = section_number
        if section_title:
            anchor_chunk["section_title"] = section_title
        anchor_chunk["citation"] = format_citation(anchor_chunk)
        anchor_chunk["retrieval_anchor"] = anchor_id
        anchor_chunk["retrieval_anchor_as_of"] = str(as_of or "")
    return augmented


def retrieve_requirement_evidence(
    question: str,
    index: LoadedIndex,
    *,
    top_k: int = 12,
    collections: str | Iterable[str] | None = None,
    query_embedder: Callable[[str], Any] | Any | None = None,
    client: Any | None = None,
    as_of: date | datetime | str | None = None,
    candidate_chunks: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Retrieve and organize rules, criteria, and related ERCOT change records."""

    analysis = analyze_question(question, as_of=as_of)
    candidates = (
        list(candidate_chunks)
        if candidate_chunks is not None
        else retrieve_chunks(
            question,
            index,
            top_k=min(len(index.chunks), max(top_k * 5, 40)),
            collections=collections,
            query_embedder=query_embedder,
            client=client,
            as_of=analysis.as_of,
        )
    )
    candidates = _augment_interconnection_candidates(
        question,
        index,
        candidates,
        collections=collections,
        as_of=analysis.as_of,
    )
    selected = diversify_evidence(
        question,
        candidates,
        top_k=top_k,
        as_of=analysis.as_of,
    )
    # Diversification annotates lifecycle and section metadata. Rebuild every
    # citation afterward so inferred effective dates and anchor locations are
    # reflected in the model context and displayed source list.
    for chunk in selected:
        chunk["citation"] = format_citation(chunk)
    change_reports = (
        _build_change_reports(
            index,
            candidates,
            as_of=analysis.as_of,
            requested_sections=analysis.requested_sections,
        )
        if analysis.asks_for_changes
        else []
    )
    return {
        "analysis": analysis.to_dict(),
        "chunks": selected,
        "evidence": evidence_summary(selected),
        "answer_contract": answer_contract(analysis),
        "change_reports": change_reports,
    }


def _join_document_chunks(chunks: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(
        chunks,
        key=lambda chunk: (
            int(chunk.get("chunk_index") or 0),
            str(chunk.get("chunk_id") or ""),
        ),
    )
    combined = ""
    for chunk in ordered:
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        if not combined:
            combined = text
            continue
        overlap = 0
        maximum = min(2_000, len(combined), len(text))
        for size in range(maximum, 39, -1):
            if combined[-size:] == text[:size]:
                overlap = size
                break
        combined += "\n" + text[overlap:].lstrip()
    return combined


def _change_comparable_artifact(chunk: Mapping[str, Any]) -> bool:
    """Limit automatic redlines to documents, not mislabeled crawler pages."""

    values = [
        str(chunk.get("source_path") or chunk.get("source") or ""),
        str(chunk.get("original_url") or ""),
        str(chunk.get("final_url") or ""),
        *(str(value) for value in (chunk.get("aliases") or []) if value),
        *(str(value) for value in (chunk.get("url_aliases") or []) if value),
    ]
    return any(
        re.search(r"\.(?:pdf|docx?)\b", value, re.IGNORECASE)
        for value in values
    )


def _build_change_reports(
    index: LoadedIndex,
    candidates: Sequence[Mapping[str, Any]],
    *,
    as_of: date | datetime | str | None = None,
    requested_sections: Sequence[str] = (),
    max_reports: int = 3,
) -> list[dict[str, Any]]:
    """Compare the two newest retrievable versions of relevant logical documents."""

    candidate_keys = [logical_document_key(chunk) for chunk in candidates]
    ordered_keys = list(
        dict.fromkeys(
            key
            for key in candidate_keys
            if key
            and key != "unknown"
            # An xRR's comments, ballots, and committee reports are related
            # lifecycle artifacts, not sequential versions of one document.
            and not key.startswith("revision-request:")
        )
    )
    target_keys = set(ordered_keys[:12])
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        key: {} for key in ordered_keys[:12]
    }
    for chunk in index.chunks:
        if is_notice(chunk) or not _change_comparable_artifact(chunk):
            continue
        key = logical_document_key(chunk)
        if key not in target_keys:
            continue
        identity = str(
            chunk.get("document_id")
            or chunk.get("content_hash")
            or chunk.get("source_path")
            or ""
        )
        if identity:
            grouped[key].setdefault(identity, []).append(chunk)
    reports: list[dict[str, Any]] = []
    for key in ordered_keys[:12]:
        by_document = grouped.get(key, {})
        if len(by_document) < 2:
            continue
        versions = sorted(
            by_document.items(),
            key=lambda item: (
                _version_rank(item[1][0]),
                item[0],
            ),
            reverse=True,
        )
        (new_id, new_chunks), (old_id, old_chunks) = versions[:2]
        old_metadata = annotate_evidence(old_chunks[0], as_of=as_of)
        new_metadata = annotate_evidence(new_chunks[0], as_of=as_of)
        report = compare_document_versions(
            _join_document_chunks(old_chunks),
            _join_document_chunks(new_chunks),
            old_metadata,
            new_metadata,
        ).to_dict()
        all_changes = list(report.get("changes") or [])
        if requested_sections:
            relevant_changes = [
                change
                for change in all_changes
                if any(
                    str(change.get("section_number") or "") == requested
                    or str(change.get("section_number") or "").startswith(f"{requested}.")
                    for requested in requested_sections
                )
            ]
            if not relevant_changes:
                continue
            report["all_counts"] = report.get("counts") or {}
            report["counts"] = {
                status: sum(change.get("status") == status for change in relevant_changes)
                for status in ("added", "modified", "removed", "unchanged")
            }
        else:
            relevant_changes = all_changes
        compact_changes: list[dict[str, Any]] = []
        for change in relevant_changes:
            if change.get("status") == "unchanged":
                continue
            compact = {
                "section_number": change.get("section_number"),
                "status": change.get("status"),
                "old_citation": change.get("old_citation"),
                "new_citation": change.get("new_citation"),
            }
            for side in ("old_section", "new_section"):
                section = change.get(side)
                if not isinstance(section, dict):
                    compact[side] = None
                    continue
                compact[side] = {
                    "number": section.get("number"),
                    "title": section.get("title"),
                    "page_start": section.get("page_start"),
                    "page_end": section.get("page_end"),
                    "text_excerpt": str(section.get("text") or "")[:800],
                }
            compact_changes.append(compact)
        report["changes"] = compact_changes[:100]
        report.update(
            {
                "old_document_id": old_id,
                "new_document_id": new_id,
                "old_effectiveness": old_metadata.get("effectiveness_label"),
                "new_effectiveness": new_metadata.get("effectiveness_label"),
            }
        )
        reports.append(report)
        if len(reports) >= max_reports:
            break
    return reports


def format_change_reports(reports: Sequence[Mapping[str, Any]]) -> str:
    """Format compact deterministic change evidence for the answer model."""

    blocks: list[str] = []
    for report in reports:
        counts = report.get("counts") or {}
        lines = [
            f"Logical document: {report.get('logical_key', 'unknown')}",
            f"Old effectiveness: {report.get('old_effectiveness') or 'not established'}",
            f"New effectiveness: {report.get('new_effectiveness') or 'not established'}",
            "Counts: "
            + ", ".join(
                f"{status}={int(counts.get(status, 0))}"
                for status in ("added", "modified", "removed", "unchanged")
            ),
        ]
        changed = [
            change
            for change in (report.get("changes") or [])
            if change.get("status") != "unchanged"
        ]
        for change in changed[:40]:
            citations = " -> ".join(
                value
                for value in (
                    str(change.get("old_citation") or ""),
                    str(change.get("new_citation") or ""),
                )
                if value
            )
            lines.append(
                f"- {change.get('status')}: Section {change.get('section_number')}"
                + (f" ({citations})" if citations else "")
            )
            old_section = change.get("old_section") or {}
            new_section = change.get("new_section") or {}
            old_excerpt = str(old_section.get("text_excerpt") or "").strip()
            new_excerpt = str(new_section.get("text_excerpt") or "").strip()
            if old_excerpt:
                lines.append(f"  Prior text excerpt: {old_excerpt}")
            if new_excerpt:
                lines.append(f"  New text excerpt: {new_excerpt}")
        blocks.append("\n".join(lines))
    formatted = "\n\n---\n\n".join(blocks)
    if len(formatted) > 20_000:
        return formatted[:20_000].rstrip() + "\n[change report truncated]"
    return formatted


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
    section_number = str(
        chunk.get("section_number") or chunk.get("section") or ""
    ).strip()
    section_title = str(chunk.get("section_title") or "").strip()
    if section_number:
        location = f"Section {section_number}"
        if section_title:
            location += f" {section_title}"
        parts.append(location)
    page_start = chunk.get("page_start")
    page_end = chunk.get("page_end")
    if page_start not in (None, ""):
        if page_end not in (None, "", page_start, str(page_start)):
            parts.append(f"pages {page_start}-{page_end}")
        else:
            parts.append(f"page {page_start}")
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
    for position, chunk in enumerate(chunks, start=1):
        text = str(chunk.get("text", "")).strip()
        words = text.split()
        if max_words is not None and used_words + len(words) > max_words:
            remaining = max_words - used_words
            if remaining <= 0:
                break
            text = " ".join(words[:remaining])
            words = words[:remaining]
        evidence_id = str(chunk.get("evidence_id") or f"E{position}")
        lifecycle = str(chunk.get("effectiveness_label") or "Effectiveness not established")
        role = str(chunk.get("evidence_role") or "supporting_evidence")
        basis = str(chunk.get("effectiveness_basis") or "")
        blocks.append(
            f"Evidence [{evidence_id}]\n"
            f"Citation: {format_citation(chunk)}\n"
            f"Evidence role: {role}\n"
            f"Effectiveness: {lifecycle}\n"
            f"Effectiveness basis: {basis}\n\n{text}"
        )
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
        evidence_id = str(chunk.get("evidence_id") or "").strip()
        prefix = f"**[{evidence_id}]** " if evidence_id else ""
        line = f"- {prefix}{citation}"
        for index, source_url in enumerate(source_urls[:3]):
            label = "open ERCOT source" if index == 0 else f"alternate source {index + 1}"
            line += f" — [{label}](<{source_url}>)"
        lines.append(line)
        if len(lines) >= max_sources:
            break
    return "\n\n**Retrieved sources**\n\n" + "\n".join(lines) if lines else ""
