"""Serve current answer text independently of the saved semantic-search cache."""

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embedding_utils import chunk_texts, chunks_digest
from interview_retrieval import clear_candidate_pool
from sync_manager_content import CONTENT_FILE, SOURCE_NAME, build_chunks, load_json


@dataclass(frozen=True)
class RuntimeCorpus:
    """Current evidence and the saved-vector rows that still represent its text."""

    chunks: list[dict]
    semantic_row_map: dict[int, int]
    source_locations: dict[int, dict[str, Any]]

    @property
    def pending_embedding_count(self) -> int:
        """Count current records without a matching, unchanged saved vector."""
        return len(self.chunks) - len(self.semantic_row_map)

    def remap_scores(self, saved_scores: Mapping[int, float]) -> dict[int, float]:
        """Discard removed or changed evidence before mapping semantic scores."""
        return {
            self.semantic_row_map[row]: score for row, score in saved_scores.items()
            if row in self.semantic_row_map
        }


def build_runtime_corpus(saved_chunks: list[dict], manager_content: list[dict]) -> RuntimeCorpus:
    """Resolve current managed records without attributing old vectors to edits.

    Callers must validate the saved embedding cache against ``saved_chunks``.
    Base records keep their original evidence. Managed records come only from
    current source content; IDs establish identity, and exact embedding-input
    text equality is additionally required before reusing a semantic score.
    """
    if not isinstance(saved_chunks, list) or not all(isinstance(item, dict) for item in saved_chunks):
        raise ValueError("Saved interview chunks must be an array of records")
    if not isinstance(manager_content, list) or not all(isinstance(item, dict) for item in manager_content):
        raise ValueError("Manager content must be an array of records")
    saved_texts = chunk_texts(saved_chunks)
    saved_managed_rows: dict[str, int] = {}
    base_rows = []
    for row, chunk in enumerate(saved_chunks):
        if chunk.get("source") == SOURCE_NAME:
            identifier = chunk.get("source_id")
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError("Saved managed content requires a nonempty source_id")
            if identifier in saved_managed_rows:
                raise ValueError(f"Duplicate saved manager content id: {identifier}")
            saved_managed_rows[identifier] = row
        else:
            base_rows.append(row)

    chunks, base_count, _ = build_chunks(saved_chunks, manager_content)
    current_texts = chunk_texts(chunks)
    row_map = {saved_row: current_row for current_row, saved_row in enumerate(base_rows)}
    locations = {
        current_row: {"file": "chunks_cleaned.json", "index": saved_row}
        for current_row, saved_row in enumerate(base_rows)
    }
    for source_index, chunk in enumerate(chunks[base_count:]):
        current_row = base_count + source_index
        locations[current_row] = {"file": CONTENT_FILE.name, "index": source_index}
        saved_row = saved_managed_rows.get(chunk["source_id"])
        if saved_row is not None and saved_texts[saved_row] == current_texts[current_row]:
            row_map[saved_row] = current_row
    return RuntimeCorpus(chunks, row_map, locations)


def load_runtime_corpus(saved_chunks: list[dict], content_file: Path = CONTENT_FILE) -> RuntimeCorpus:
    """Load current source on every rerun; malformed or missing content is an error."""
    return build_runtime_corpus(saved_chunks, load_json(content_file))


def activate_runtime_content(state: MutableMapping[str, Any], chunks: list[dict]) -> bool:
    """Clear actions tied to older evidence while retaining job settings/history."""
    digest = chunks_digest(chunks)
    if state.get("answer_corpus_digest") == digest:
        return False
    had_prior_content = "answer_corpus_digest" in state or "active_answer" in state
    clear_candidate_pool(state)
    for key in (
        "active_answer", "pending_answer", "pending_search", "answer_audio",
        "answer_error", "search_error", "audio_error", "practice_feedback",
        "practice_feedback_signature", "practice_loaded_source", "practice_source",
        "practice_question", "practice_answer", "practice_audio", "practice_recording",
        "practice_transcript", "practice_error", "evidence_story",
    ):
        state.pop(key, None)
    state["answer_corpus_digest"] = digest
    return had_prior_content
