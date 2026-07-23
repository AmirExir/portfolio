"""FastAPI endpoint backed by the atomic central ERCOT RAG index."""

from __future__ import annotations

import re
import sys
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from ERCOTAPI.rag_ingestion.config import default_config
    from ERCOTAPI.rag_ingestion.retrieval import (
        LoadedIndex,
        format_context,
        format_change_reports,
        format_source_list,
        index_state,
        retrieve_requirement_evidence,
        retrieve_chunks,
    )
    from ERCOTAPI.rag_ingestion.startup import load_startup_index
    from ERCOTAPI.rag_ingestion.store import load_manifest
except ModuleNotFoundError as exc:  # Supports launching from this app directory.
    if exc.name != "ERCOTAPI":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ERCOTAPI.rag_ingestion.config import default_config
    from ERCOTAPI.rag_ingestion.retrieval import (
        LoadedIndex,
        format_context,
        format_change_reports,
        format_source_list,
        index_state,
        retrieve_requirement_evidence,
        retrieve_chunks,
    )
    from ERCOTAPI.rag_ingestion.startup import load_startup_index
    from ERCOTAPI.rag_ingestion.store import load_manifest


CollectionName = Literal[
    "general",
    "planning",
    "protocols",
    "operations",
    "resource_integration",
    "dwg_sswg",
    "market",
    "news",
]
COLLECTION_NAMES = (
    "general",
    "planning",
    "protocols",
    "operations",
    "resource_integration",
    "dwg_sswg",
    "market",
    "news",
)
REQUIRED_COLLECTION_NAMES = tuple(name for name in COLLECTION_NAMES if name != "news")


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: int = Field(default=8, ge=1, le=30)
    max_context_tokens: int = Field(default=12000, ge=1000, le=100000)
    collection: CollectionName = "general"
    include_generated: bool = True
    prefer_authoritative: bool = True
    as_of: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class SourceRecord(BaseModel):
    chunk_id: str
    citation: str
    title: str | None = None
    source_path: str | None = None
    source_authority: str | None = None
    source_kind: str | None = None
    document_number: str | None = None
    document_status: str | None = None
    effective_date: str | None = None
    published_date: str | None = None
    revision: str | None = None
    authority_class: str | None = None
    effective_state: str | None = None
    effectiveness_label: str | None = None
    effectiveness_basis: str | None = None
    resolved_effective_date: str | None = None
    effective_date_inferred: bool = False
    evidence_role: str | None = None
    is_governing: bool = False
    logical_document_id: str | None = None
    evidence_id: str | None = None
    section_number: str | None = None
    section_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    is_generated: bool = False
    original_url: str | None = None
    url_aliases: List[str] = Field(default_factory=list)
    score: float


class RetrieveResponse(BaseModel):
    question: str
    context: str
    used_chunks: int
    sources: List[SourceRecord] = Field(default_factory=list)
    collection: str = "general"
    generation: str | None = None
    analysis: Dict[str, Any] = Field(default_factory=dict)
    evidence: Dict[str, List[str]] = Field(default_factory=dict)
    answer_contract: str = ""
    source_footer: str = ""
    change_reports: List[Dict[str, Any]] = Field(default_factory=list)


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Warm or bootstrap the central index when an ASGI server starts."""

    global LOAD_ERROR
    try:
        _get_index("general")
    except Exception as exc:  # Health reports a deployment or ingestion failure.
        LOAD_ERROR = str(exc)
    yield


app = FastAPI(title="ERCOT Retrieval API", version="2.0.0", lifespan=_lifespan)

INDEXES: dict[str, LoadedIndex] = {}
DATA_STATE: tuple[object, ...] = ()
LOAD_ERROR = ""
INDEX_LOCK = threading.RLock()


def _file_state() -> tuple[object, ...]:
    return index_state()


def _normalize_question(question: str) -> str:
    question = re.sub(r"\bplannig\b", "planning", question, flags=re.IGNORECASE)
    return re.sub(r"\bplaning\b", "planning", question, flags=re.IGNORECASE)


def _load_collection(collection: str) -> LoadedIndex:
    return load_startup_index(collection)


def _get_index(collection: str) -> LoadedIndex:
    """Hot-reload only after a complete atomic CURRENT switch."""

    global DATA_STATE, LOAD_ERROR, INDEXES
    with INDEX_LOCK:
        try:
            state = _file_state()
            if state == DATA_STATE and collection in INDEXES:
                return INDEXES[collection]
            loaded = _load_collection(collection)
        except Exception as exc:
            LOAD_ERROR = str(exc)
            raise
        state = _file_state()
        if loaded.generation_id != state[0]:
            LOAD_ERROR = (
                "Central ERCOT generation changed while it was loading; "
                "retry the request against the new CURRENT generation"
            )
            raise RuntimeError(LOAD_ERROR)
        if state != DATA_STATE:
            INDEXES = {}
        INDEXES[collection] = loaded
        DATA_STATE = state
        LOAD_ERROR = ""
        return loaded


def _without_generated(index: LoadedIndex) -> LoadedIndex:
    rows = [position for position, chunk in enumerate(index.chunks) if not chunk.get("is_generated")]
    return LoadedIndex(
        chunks=[index.chunks[position] for position in rows],
        embeddings=index.embeddings[rows],
        embedding_model=index.embedding_model,
        generation_id=index.generation_id,
        source=index.source,
        collections=index.collections,
        state_token=index.state_token,
    )


def _source_record(chunk: dict) -> SourceRecord:
    return SourceRecord(
        chunk_id=str(chunk.get("chunk_id") or chunk.get("id") or ""),
        citation=str(chunk.get("citation") or ""),
        title=str(chunk.get("title")) if chunk.get("title") else None,
        source_path=str(chunk.get("source_path") or chunk.get("source"))
        if chunk.get("source_path") or chunk.get("source")
        else None,
        source_authority=str(chunk.get("source_authority"))
        if chunk.get("source_authority")
        else None,
        source_kind=str(chunk.get("source_kind")) if chunk.get("source_kind") else None,
        document_number=str(chunk.get("document_number"))
        if chunk.get("document_number")
        else None,
        document_status=str(chunk.get("document_status"))
        if chunk.get("document_status")
        else None,
        effective_date=str(chunk.get("effective_date")) if chunk.get("effective_date") else None,
        published_date=str(chunk.get("published_date")) if chunk.get("published_date") else None,
        revision=str(chunk.get("revision")) if chunk.get("revision") else None,
        authority_class=str(chunk.get("authority_class"))
        if chunk.get("authority_class")
        else None,
        effective_state=str(chunk.get("effective_state"))
        if chunk.get("effective_state")
        else None,
        effectiveness_label=str(chunk.get("effectiveness_label"))
        if chunk.get("effectiveness_label")
        else None,
        effectiveness_basis=str(chunk.get("effectiveness_basis"))
        if chunk.get("effectiveness_basis")
        else None,
        resolved_effective_date=str(chunk.get("resolved_effective_date"))
        if chunk.get("resolved_effective_date")
        else None,
        effective_date_inferred=bool(chunk.get("effective_date_inferred")),
        evidence_role=str(chunk.get("evidence_role"))
        if chunk.get("evidence_role")
        else None,
        is_governing=bool(chunk.get("is_governing")),
        logical_document_id=str(chunk.get("logical_document_id"))
        if chunk.get("logical_document_id")
        else None,
        evidence_id=str(chunk.get("evidence_id")) if chunk.get("evidence_id") else None,
        section_number=str(chunk.get("section_number"))
        if chunk.get("section_number")
        else None,
        section_title=str(chunk.get("section_title"))
        if chunk.get("section_title")
        else None,
        page_start=int(chunk["page_start"]) if chunk.get("page_start") is not None else None,
        page_end=int(chunk["page_end"]) if chunk.get("page_end") is not None else None,
        is_generated=bool(chunk.get("is_generated")),
        original_url=str(chunk.get("original_url")) if chunk.get("original_url") else None,
        url_aliases=[str(value) for value in (chunk.get("url_aliases") or []) if value],
        score=float(chunk.get("retrieval_score", 0.0)),
    )


def _limit_context(chunks: list[dict], max_words: int) -> list[dict]:
    limited: list[dict] = []
    remaining = max_words
    for original in chunks:
        if remaining <= 0:
            break
        chunk = dict(original)
        words = str(chunk.get("text", "")).split()
        if len(words) > remaining:
            chunk["text"] = " ".join(words[:remaining])
            words = words[:remaining]
        if words:
            limited.append(chunk)
            remaining -= len(words)
    return limited


def _manifest_collection_counts(generation_id: str) -> tuple[str, dict[str, int]]:
    """Read collection readiness from manifest JSON without copying vectors."""

    loaded = load_manifest(default_config().index_dir, generation_id)
    if loaded is None:
        raise RuntimeError("No active central ERCOT RAG manifest")
    generation_id, manifest = loaded
    counts = {name: 0 for name in COLLECTION_NAMES}
    content = manifest.get("content", {})
    if not isinstance(content, dict):
        raise RuntimeError("Active ERCOT RAG manifest has invalid content metadata")
    for record in content.values():
        if not isinstance(record, dict):
            continue
        chunk_count = len(record.get("chunk_ids", []))
        for collection in record.get("collections", []):
            name = str(collection)
            if name in counts:
                counts[name] += chunk_count
    return generation_id, counts


@app.get("/health")
def health() -> dict:
    collection_status: dict[str, dict[str, object]] = {}
    index: LoadedIndex | None = None
    try:
        general = _get_index("general")
        if general.source != "central" or not general.generation_id:
            raise RuntimeError("ERCOT retrieval API requires a central generation")
        generation_id, counts = _manifest_collection_counts(general.generation_id)
        for collection in COLLECTION_NAMES:
            count = counts.get(collection, 0)
            collection_status[collection] = {
                "ready": count > 0,
                "chunks_loaded": count,
                "index_source": "central",
                "generation": generation_id,
                "error": None,
            }
        index = general
    except Exception as exc:
        collection_status = {
            collection: {
                "ready": False,
                "chunks_loaded": 0,
                "index_source": None,
                "generation": None,
                "error": str(exc),
            }
            for collection in COLLECTION_NAMES
        }

    # A central manifest records chunk IDs and collection routing. Reading that
    # small JSON avoids loading/copying eight full embedding matrices merely to
    # answer a health probe.
    general_error = str(collection_status["general"].get("error") or "")
    unavailable = [
        name
        for name in REQUIRED_COLLECTION_NAMES
        if not collection_status[name]["ready"]
    ]
    optional_unavailable = [
        name
        for name in COLLECTION_NAMES
        if name not in REQUIRED_COLLECTION_NAMES and not collection_status[name]["ready"]
    ]
    return {
        # Preserve the original general-corpus health meaning for existing
        # consumers while exposing complete readiness for every advertised
        # collection.
        "ok": index is not None and index.ready and not general_error,
        "degraded": bool(unavailable),
        "all_collections_ready": not unavailable,
        "unavailable_collections": unavailable,
        "optional_unavailable_collections": optional_unavailable,
        "collections": collection_status,
        "chunks_loaded": len(index.chunks) if index else 0,
        "embeddings_loaded": int(index.embeddings.shape[0]) if index else 0,
        "embedding_dim": int(index.embeddings.shape[1]) if index and index.embeddings.ndim == 2 else 0,
        "data_state": DATA_STATE,
        "generation": index.generation_id if index else None,
        "index_source": index.source if index else None,
        "error": general_error,
    }


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(payload: RetrieveRequest) -> RetrieveResponse:
    try:
        index = _get_index(payload.collection)
        if not payload.include_generated:
            index = _without_generated(index)
        if not index.ready:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Collection {payload.collection!r} is not ready; run the central "
                    "ERCOT RAG ingestion update or rebuild command"
                ),
            )

        normalized_question = _normalize_question(payload.question)
        candidate_limit = (
            len(index.chunks)
            if not payload.prefer_authoritative
            else min(len(index.chunks), max(payload.top_k, payload.top_k * 5))
        )
        candidates = retrieve_chunks(
            normalized_question,
            index,
            top_k=candidate_limit,
            as_of=payload.as_of,
        )
        if payload.prefer_authoritative:
            bundle = retrieve_requirement_evidence(
                normalized_question,
                index,
                top_k=payload.top_k,
                as_of=payload.as_of,
                candidate_chunks=candidates,
            )
            candidates = bundle["chunks"]
        else:
            candidates.sort(key=lambda chunk: float(chunk.get("vector_score", 0.0)), reverse=True)
            for chunk in candidates:
                chunk["retrieval_score"] = float(chunk.get("vector_score", 0.0))
            bundle = {
                "analysis": {},
                "evidence": {},
                "answer_contract": "",
                "change_reports": [],
            }
        # The public field retains its old name for workflow compatibility, but
        # the implementation now converts the token budget to a conservative
        # word budget instead of treating tokens as words.
        selected = _limit_context(
            candidates[: payload.top_k],
            max(1, int(payload.max_context_tokens * 0.72)),
        )
        context = format_context(selected)
        change_context = format_change_reports(bundle.get("change_reports") or [])
        if change_context:
            context += "\n\n=== SECTION-LEVEL CHANGE REPORTS ===\n\n" + change_context
        return RetrieveResponse(
            question=payload.question,
            context=context,
            used_chunks=len(selected),
            sources=[_source_record(chunk) for chunk in selected],
            collection=payload.collection,
            generation=index.generation_id,
            analysis=bundle["analysis"],
            evidence=bundle["evidence"],
            answer_contract=bundle["answer_contract"],
            source_footer=format_source_list(selected, max_sources=8),
            change_reports=bundle.get("change_reports") or [],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
