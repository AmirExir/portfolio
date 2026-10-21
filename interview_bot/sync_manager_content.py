import argparse
import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CONTENT_FILE = BASE_DIR / "manager_content.json"
CHUNKS_FILE = BASE_DIR / "chunks_cleaned.json"
SOURCE_NAME = "manager_content"
ASSIGNMENT_FIELDS = ("answer_set", "question_key", "scenario_id")


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def require_text(item, field):
    value = item.get(field, "")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{item.get('id', '<missing id>')}: '{field}' must be non-empty text")
    return value.strip()


def serialize_content(item):
    content_id = require_text(item, "id")
    response_type = require_text(item, "response_type").lower()
    question = require_text(item, "question")
    aliases = item.get("aliases", [])
    if not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
        raise ValueError(f"{content_id}: 'aliases' must be a list of strings")

    if response_type == "direct":
        category = require_text(item, "category")
        answer = require_text(item, "answer")
        fields = [
            "Response Type: Direct",
            f"Category: {category}",
            f"Question: {question}",
        ]
        if aliases:
            fields.append(f"Aliases: {'; '.join(alias.strip() for alias in aliases if alias.strip())}")
        fields.append(f"Answer: {answer}")
    elif response_type == "story":
        fields = [
            "Response Type: Story",
            f"Principle: {require_text(item, 'principle')}",
            f"Question: {question}",
            f"Situation: {require_text(item, 'situation')}",
            f"Task: {require_text(item, 'task')}",
            f"Action: {require_text(item, 'action')}",
            f"Result: {require_text(item, 'result')}",
        ]
        if aliases:
            fields.append(f"Aliases: {'; '.join(alias.strip() for alias in aliases if alias.strip())}")
        if "answer" in item:
            fields.append(f"Answer: {require_text(item, 'answer')}")
    else:
        raise ValueError(f"{content_id}: unsupported response_type '{response_type}'")

    assignment = {}
    if "backup_for" in item:
        if any(field in item for field in ("answer_set", "question_key")):
            raise ValueError(f"{content_id}: a backup cannot also be a primary question assignment")
        assignment = {
            "backup_for": require_text(item, "backup_for"),
            "scenario_id": require_text(item, "scenario_id"),
        }
        require_text(item, "answer")
        fields.extend([
            f"Backup For: {assignment['backup_for']}",
            f"Scenario ID: {assignment['scenario_id']}",
        ])
    elif any(field in item for field in ASSIGNMENT_FIELDS):
        assignment = {field: require_text(item, field) for field in ASSIGNMENT_FIELDS}
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", assignment["answer_set"]):
            raise ValueError(f"{content_id}: 'answer_set' must be a lowercase slug")
        if not re.fullmatch(r"[1-9][0-9]? [A-Z][A-Z ]*", assignment["question_key"]):
            raise ValueError(f"{content_id}: 'question_key' must contain a number and uppercase topic")
        require_text(item, "answer")
        fields.extend([
            f"Answer Set: {assignment['answer_set']}",
            f"Question Key: {assignment['question_key']}",
            f"Scenario ID: {assignment['scenario_id']}",
        ])

    if "source_document" in item:
        question_number = item.get("source_question")
        if type(question_number) is not int or question_number < 1:
            raise ValueError(f"{content_id}: 'source_question' must be a positive integer")
        source_hash = require_text(item, "source_sha256")
        if not re.fullmatch(r"[a-f0-9]{64}", source_hash):
            raise ValueError(f"{content_id}: 'source_sha256' must be a SHA-256 hex digest")
        fields.extend([
            f"Source Document: {require_text(item, 'source_document')}",
            f"Source Question: {question_number}",
            f"Source SHA256: {source_hash}",
            f"Source Status: {require_text(item, 'source_status')}",
            "Verification: User-provided interview preparation; personal claims and cited "
            "references have not been independently verified. Reference dates and statements "
            "that a document was checked are supplied by the author, not verified by this bot.",
        ])
    if "answer_revision" in item:
        revision = item["answer_revision"]
        revision_fields = {"source_document", "source_question", "source_sha256", "editorial_note"}
        if not isinstance(revision, dict) or set(revision) != revision_fields:
            raise ValueError(
                f"{content_id}: 'answer_revision' must contain source_document, "
                "source_question, source_sha256, and editorial_note"
            )
        revision = {**revision, "id": content_id}
        revision_question = revision["source_question"]
        if type(revision_question) is not int or revision_question < 1:
            raise ValueError(f"{content_id}: answer revision source_question must be a positive integer")
        revision_hash = require_text(revision, "source_sha256")
        if not re.fullmatch(r"[a-f0-9]{64}", revision_hash):
            raise ValueError(f"{content_id}: answer revision source_sha256 must be a SHA-256 hex digest")
        fields.append(
            f"Answer Revision: Adapted from {require_text(revision, 'source_document')}; "
            f"question {revision_question}; SHA-256 {revision_hash}. "
            f"Editorial note: {require_text(revision, 'editorial_note')}"
        )
    references = item.get("references", [])
    if not isinstance(references, list) or not all(
        isinstance(reference, str) and reference.strip() for reference in references
    ):
        raise ValueError(f"{content_id}: 'references' must be a list of non-empty strings")
    if references:
        fields.append("References: " + "\n".join(references))
    if "preparation_note" in item:
        fields.append(f"Preparation Note: {require_text(item, 'preparation_note')}")

    return {
        "text": " | ".join(fields),
        "source": SOURCE_NAME,
        "source_id": content_id,
        "response_type": response_type,
        **assignment,
    }


def build_chunks(existing_chunks, manager_content):
    if not isinstance(existing_chunks, list):
        raise ValueError("chunks_cleaned.json must contain a JSON array")
    if not isinstance(manager_content, list):
        raise ValueError("manager_content.json must contain a JSON array")

    ids = [require_text(item, "id") for item in manager_content]
    duplicates = sorted({content_id for content_id in ids if ids.count(content_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate manager content ids: {', '.join(duplicates)}")

    base_chunks = [chunk for chunk in existing_chunks if chunk.get("source") != SOURCE_NAME]
    manager_chunks = [serialize_content(item) for item in manager_content]
    by_id = {chunk["source_id"]: chunk for chunk in manager_chunks}
    scenario_ids = {chunk["scenario_id"] for chunk in manager_chunks if "answer_set" in chunk}
    for chunk in manager_chunks:
        if "backup_for" not in chunk:
            continue
        parent = by_id.get(chunk["backup_for"])
        if parent is None or "answer_set" not in parent:
            raise ValueError(f"{chunk['source_id']}: backup_for must reference a primary assigned answer")
        if chunk["scenario_id"] in scenario_ids:
            raise ValueError(f"{chunk['source_id']}: backup scenario_id must be unique")
        scenario_ids.add(chunk["scenario_id"])
    assignments = {}
    for chunk in manager_chunks:
        if "answer_set" not in chunk:
            continue
        members = assignments.setdefault(chunk["answer_set"], [])
        members.append(chunk)
    for answer_set, members in assignments.items():
        for field in ("question_key", "scenario_id"):
            values = [member[field] for member in members]
            if len(values) != len(set(values)):
                raise ValueError(f"{answer_set}: each {field} must have a unique one-to-one assignment")
        if answer_set == "ercot-planning-20":
            numbers = [int(member["question_key"].split()[0]) for member in members]
            if sorted(numbers) != list(range(1, 21)):
                raise ValueError("ercot-planning-20: exactly one assignment is required for questions 1 through 20")
    return base_chunks + manager_chunks, len(base_chunks), len(manager_chunks)


def main():
    parser = argparse.ArgumentParser(
        description="Validate manager interview content and sync it into chunks_cleaned.json."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate and report the resulting chunk count without writing files.",
    )
    args = parser.parse_args()

    chunks = load_json(CHUNKS_FILE)
    manager_content = load_json(CONTENT_FILE)
    updated_chunks, base_count, manager_count = build_chunks(chunks, manager_content)

    print(f"Base chunks: {base_count}")
    print(f"Manager chunks: {manager_count}")
    print(f"Total chunks: {len(updated_chunks)}")

    if args.check:
        print("Validation complete; no files written.")
        return

    temporary_file = CHUNKS_FILE.with_suffix(".json.tmp")
    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(updated_chunks, file, indent=2, ensure_ascii=True)
        file.write("\n")
    temporary_file.replace(CHUNKS_FILE)
    print(f"Updated {CHUNKS_FILE.name}")


if __name__ == "__main__":
    main()
