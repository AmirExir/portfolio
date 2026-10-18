"""Offline evidence evaluation with explicit fixture and review provenance.

Replay recorded answers, or exercise a deliberately lexical local smoke baseline.
No mode calls an embedding service or an answer model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .answer_support import ABSTENTION_TEXT
from .change_tracking import parse_numbered_sections
from .config import repository_root
from .loaders import load_document_with_report
from .requirements import annotate_evidence, validate_answer_citations


def _source_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Evaluation sources must remain inside the repository root")
    return path


def _identity(source: Mapping[str, Any]) -> tuple[str, str]:
    return str(source.get("source_path", "")), str(source.get("section_number", ""))


def validate_cases(dataset: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    """Validate case identities and any source anchors without inventing labels."""

    if not isinstance(dataset, Mapping):
        raise ValueError("Case dataset must be a JSON object")
    status = dataset.get("ground_truth_status")
    if status not in {"synthetic_fixture", "draft_requires_engineer_review", "engineer_reviewed"}:
        raise ValueError("Declare ground_truth_status explicitly")
    if status == "engineer_reviewed" and not (dataset.get("reviewer") and dataset.get("reviewed_at")):
        raise ValueError("Engineer-reviewed labels require reviewer and reviewed_at metadata")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation requires a nonempty cases list")
    if any(not isinstance(case, Mapping) for case in cases):
        raise ValueError("Each evaluation case must be an object")
    identifiers = [str(case.get("id") or "") for case in cases]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Each evaluation case must have a unique nonempty id")
    anchors: list[dict[str, Any]] = []
    loaded: dict[str, str] = {}
    for case in cases:
        if not str(case.get("question") or "").strip():
            raise ValueError(f"Case {case['id']} needs a question")
        sources = case.get("expected_sources", [])
        if not isinstance(sources, list) or any(not isinstance(item, Mapping) for item in sources):
            raise ValueError(f"Expected sources for {case['id']} must be a list of objects")
        if not isinstance(case.get("expected_effective_states", {}), Mapping):
            raise ValueError(f"Expected effective states for {case['id']} must be an object")
        for label in ("expected_abstention", "expected_support_review"):
            if label in case and not isinstance(case[label], bool):
                raise ValueError(f"{label} for {case['id']} must be boolean")
        for expected in case.get("expected_sources", []):
            if not expected.get("anchor_text"):
                continue
            relative = str(expected.get("source_path") or "")
            path = _source_path(root, relative)
            if relative not in loaded:
                loaded[relative] = load_document_with_report(path).text
            sections = parse_numbered_sections(loaded[relative])
            matching = [item for item in sections if item.number == str(expected.get("section_number"))]
            anchor = " ".join(str(expected["anchor_text"]).split()).casefold()
            found = any(anchor in item.normalized_text.casefold() for item in matching)
            if not found:
                raise ValueError(f"Source anchor no longer matches {case['id']}: {relative}")
            anchors.append({
                "case_id": case["id"], "source_path": relative,
                "section_number": expected["section_number"],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "anchor_found": True,
            })
    return anchors


def evaluate_predictions(
    dataset: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]], *, root: Path,
) -> dict[str, Any]:
    """Score replayed evidence and answer screening; semantic truth stays unscored."""

    anchors = validate_cases(dataset, root)
    if not isinstance(predictions, (list, tuple)) or any(not isinstance(item, Mapping) for item in predictions):
        raise ValueError("Predictions must be a JSON list of objects")
    by_id = {str(item.get("id", "")): item for item in predictions}
    expected_ids = {case["id"] for case in dataset["cases"]}
    if len(by_id) != len(predictions) or set(by_id) - expected_ids:
        raise ValueError("Prediction IDs must be unique and belong to the case set")
    metrics: dict[str, list[float]] = {
        "expected_evidence_recall": [], "requested_as_of_match_rate": [],
        "effective_state_accuracy": [], "abstention_accuracy": [],
        "abstention_recall": [], "non_abstention_rate_on_answerable_cases": [],
        "support_review_detection_accuracy": [], "citation_format_pass_rate": [],
    }
    details = []
    for case in dataset["cases"]:
        prediction = by_id.get(case["id"], {})
        retrieved = prediction.get("retrieved", [])
        if not isinstance(retrieved, list) or any(not isinstance(item, Mapping) for item in retrieved):
            raise ValueError(f"Retrieved evidence for {case['id']} must be a list of objects")
        answer = str(prediction.get("answer") or "")
        expected = {_identity(item) for item in case.get("expected_sources", [])}
        actual = {_identity(item) for item in retrieved}
        recall = len(expected & actual) / len(expected) if expected else None
        if recall is not None:
            metrics["expected_evidence_recall"].append(recall)
        if case.get("as_of"):
            metrics["requested_as_of_match_rate"].append(float(prediction.get("as_of") == case["as_of"]))
        expected_states = case.get("expected_effective_states", {})
        for source_path, state in expected_states.items():
            matches = [item for item in retrieved if item.get("source_path") == source_path]
            metrics["effective_state_accuracy"].append(float(bool(matches) and all(item.get("effective_state") == state for item in matches)))
        audit = validate_answer_citations(answer, retrieved)
        if "expected_abstention" in case:
            # A missing answer is a failed result, not an appropriate abstention.
            abstained = audit["abstention_detected"] and not audit["claim_checks"]
            correct = float(bool(answer) and abstained == case["expected_abstention"])
            metrics["abstention_accuracy"].append(correct)
            metrics["abstention_recall" if case["expected_abstention"] else "non_abstention_rate_on_answerable_cases"].append(correct)
        if "expected_support_review" in case:
            metrics["support_review_detection_accuracy"].append(float(bool(answer) and audit["support_review_required"] == case["expected_support_review"]))
        if not case.get("expected_abstention"):
            metrics["citation_format_pass_rate"].append(float(bool(answer) and audit["citation_format_passed"]))
        details.append({
            "id": case["id"], "prediction_present": bool(prediction),
            "expected_evidence_recall": recall, "audit": audit,
        })
    return {
        "dataset": dataset.get("name"),
        "ground_truth_status": dataset["ground_truth_status"],
        "prediction_method": dataset.get("prediction_method", "recorded_predictions"),
        "case_count": len(details),
        "prediction_count": len(predictions),
        "metrics": {
            key: {"value": round(sum(values) / len(values), 4) if values else None, "scored_cases": len(values)}
            for key, values in metrics.items()
        },
        "semantic_support_accuracy": None,
        "source_anchors": anchors,
        "cases": details,
        "limitations": [
            "Draft and synthetic labels do not establish performance on engineer-reviewed questions.",
            "Expected evidence lists may be incomplete; recall is against the listed references only.",
            "Support screening measures observable flags, not entailment or engineering correctness.",
            "A missing prediction counts as a failure for each applicable metric.",
            "Lexical smoke mode echoes the supplied as-of date and emits titles, not model answers; its scores do not measure production RAG.",
        ],
    }


def lexical_smoke_predictions(dataset: Mapping[str, Any], *, root: Path, top_k: int = 5) -> list[dict[str, Any]]:
    """Exercise real local sections with lexical retrieval and extractive answers.

    This deliberately does not simulate production embeddings or generated answers.
    It is useful for testing the harness and source anchors, not model selection.
    """

    if top_k < 1:
        raise ValueError("top_k must be positive")
    corpus = []
    for source in dataset.get("corpus_sources", []):
        loaded = load_document_with_report(_source_path(root, source["source_path"]))
        for section in parse_numbered_sections(loaded.text):
            corpus.append({
                **source, "section_number": section.number,
                "text": section.title + "\n" + section.text,
                "title": section.title, "extraction_report": loaded.extraction_report,
            })
    if not corpus:
        raise ValueError("Lexical smoke mode requires corpus_sources with numbered sections")

    def terms(text: str) -> Counter[str]:
        return Counter(re.findall(r"[a-z0-9]+", text.casefold()))

    def score(question: str, text: str) -> float:
        query, document = terms(question), terms(text)
        denominator = math.sqrt(sum(value * value for value in query.values()) * sum(value * value for value in document.values()))
        return sum(value * document[token] for token, value in query.items()) / denominator if denominator else 0.0

    predictions = []
    for case in dataset["cases"]:
        ranked = sorted(corpus, key=lambda item: score(case["question"], item["text"]), reverse=True)
        retrieved = [
            {**annotate_evidence(item, as_of=case.get("as_of")), "evidence_id": f"E{index}"}
            for index, item in enumerate(ranked[:top_k], start=1)
        ]
        answer = f"{retrieved[0]['title']} [E1]." if retrieved else ABSTENTION_TEXT
        predictions.append({"id": case["id"], "as_of": case.get("as_of"), "retrieved": retrieved, "answer": answer})
    return predictions


def main(argv: Sequence[str] | None = None) -> None:
    """Run evaluation without network access or modifications to the saved index."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predictions", type=Path)
    mode.add_argument("--lexical-smoke", action="store_true")
    mode.add_argument("--check-sources", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        dataset = json.loads(args.cases.read_text(encoding="utf-8"))
        if args.check_sources:
            result = {"ground_truth_status": dataset.get("ground_truth_status"), "source_anchors": validate_cases(dataset, args.repo_root)}
        else:
            if args.lexical_smoke:
                predictions = lexical_smoke_predictions(dataset, root=args.repo_root, top_k=args.top_k)
                dataset["prediction_method"] = "lexical_smoke_with_extractive_titles_not_production_rag"
            else:
                predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
            result = evaluate_predictions(dataset, predictions, root=args.repo_root)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
