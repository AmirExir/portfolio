import json
import os
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(__file__)
CHUNKS_PATH = os.path.join(BASE_DIR, "ercot_chunks_cached.json")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "ercot_embeddings.npy")


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: int = Field(default=8, ge=1, le=30)
    max_context_tokens: int = Field(default=12000, ge=1000, le=100000)


class RetrieveResponse(BaseModel):
    question: str
    context: str
    used_chunks: int


app = FastAPI(title="ERCOT Retrieval API", version="1.0.0")


def _load_data() -> tuple[list, np.ndarray]:
    if not os.path.exists(CHUNKS_PATH) or not os.path.exists(EMBEDDINGS_PATH):
        raise RuntimeError("Missing ERCOT chunks or embeddings files.")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(EMBEDDINGS_PATH)
    return chunks, embeddings


try:
    CHUNKS, EMBEDDINGS = _load_data()
except Exception as exc:
    CHUNKS, EMBEDDINGS = [], np.array([])
    LOAD_ERROR = str(exc)
else:
    LOAD_ERROR = ""


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set for retrieval embeddings.")
    return OpenAI(api_key=api_key)


def _embed_query(question: str) -> np.ndarray:
    client = _get_client()
    resp = client.embeddings.create(model="text-embedding-3-large", input=question)
    return np.array(resp.data[0].embedding).reshape(1, -1)


def _limit_chunks_by_word_budget(chunks: List[dict], max_context_tokens: int) -> List[dict]:
    total = 0
    selected = []
    for chunk in chunks:
        text = chunk.get("text", "")
        word_count = len(text.split())
        if total + word_count > max_context_tokens:
            break
        selected.append(chunk)
        total += word_count
    return selected


def _build_context(chunks: List[dict]) -> str:
    return "\n\n---\n\n".join(c.get("text", "") for c in chunks)


@app.get("/health")
def health() -> dict:
    return {
        "ok": LOAD_ERROR == "",
        "chunks_loaded": len(CHUNKS),
        "embeddings_loaded": int(EMBEDDINGS.shape[0]) if EMBEDDINGS.size else 0,
        "embedding_dim": int(EMBEDDINGS.shape[1]) if EMBEDDINGS.size else 0,
        "error": LOAD_ERROR,
    }


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(payload: RetrieveRequest) -> RetrieveResponse:
    try:
        if LOAD_ERROR:
            raise RuntimeError(LOAD_ERROR)
        if EMBEDDINGS.size == 0:
            raise RuntimeError("Embeddings are empty.")

        query_vec = _embed_query(payload.question)
        if query_vec.shape[1] != EMBEDDINGS.shape[1]:
            raise RuntimeError(
                f"Embedding dimension mismatch: {query_vec.shape[1]} vs {EMBEDDINGS.shape[1]}"
            )

        scores = cosine_similarity(query_vec, EMBEDDINGS).flatten()
        top_indices = scores.argsort()[-payload.top_k :][::-1]
        top_chunks = [CHUNKS[i] for i in top_indices]

        limited = _limit_chunks_by_word_budget(top_chunks, payload.max_context_tokens)
        context = _build_context(limited)

        return RetrieveResponse(
            question=payload.question,
            context=context,
            used_chunks=len(limited),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
